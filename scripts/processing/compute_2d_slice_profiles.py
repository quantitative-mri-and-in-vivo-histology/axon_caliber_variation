"""
Compute per-instance 2D regionprops for all slices in a labeled volume.

Extracts morphometric measurements for every labeled region in every
cross-sectional slice, saving the raw (unfiltered) data as a flat table
in NPZ format. Downstream scripts apply filtering thresholds as needed.

Input: OME-Zarr volumes (canonical format: Z = fiber axis, slice axis always 0)

Output NPZ contains parallel arrays forming a table with one row per
region-instance per slice:
    slice_index, label, area_pixels, eccentricity, solidity,
    axis_major_length, axis_minor_length, orientation,
    centroid_row, centroid_col, radius_circular_um, radius_minor_um

Plus metadata: voxel_size_um, volume_shape, source_file

Usage:
  # Single Zarr volume (canonical format, slice axis = 0)
  python compute_2d_slice_profiles.py data/processed/LM/sham_25_ipsi_cc_myelin.zarr \\
      data/processed/LM/sham_25_ipsi_cc_slice_profiles.npz

  # Batch mode with glob pattern
  python compute_2d_slice_profiles.py "data/processed/LM/*_myelin.zarr" \\
      data/processed/LM/ --output-suffix _slice_profiles
"""

import argparse
import glob
import logging
import sys
import traceback
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm

# Find repo root
_root = Path(__file__).resolve().parent
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from axonometry.io import load_zarr_volume

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Column definitions: (key, dtype) for the per-instance output table
_COLUMN_DTYPES = (
    ("slice_index", np.int32),
    ("label", np.int32),
    ("area_pixels", np.int32),
    ("eccentricity", np.float32),
    ("solidity", np.float32),
    ("axis_major_length", np.float32),
    ("axis_minor_length", np.float32),
    ("orientation", np.float32),
    ("centroid_row", np.float32),
    ("centroid_col", np.float32),
)
_COLUMN_KEYS = tuple(k for k, _ in _COLUMN_DTYPES)

# Global worker state (shared via fork)
_volume_data = None


def _init_worker(volume: np.ndarray):
    global _volume_data
    _volume_data = volume


def _process_slice(z: int) -> dict:
    """Extract regionprops for all instances in a single 2D slice."""
    slice_2d = _volume_data[z, :, :]

    regions = regionprops(slice_2d.astype(np.int32))

    if len(regions) == 0:
        return {k: np.empty(0, dtype=dt) for k, dt in _COLUMN_DTYPES}

    n = len(regions)
    result = {k: np.empty(n, dtype=dt) for k, dt in _COLUMN_DTYPES}
    result["slice_index"][:] = z

    for i, region in enumerate(regions):
        result["label"][i] = region.label
        result["area_pixels"][i] = region.area
        result["eccentricity"][i] = region.eccentricity
        result["solidity"][i] = region.solidity
        result["axis_major_length"][i] = region.axis_major_length
        result["axis_minor_length"][i] = region.axis_minor_length
        result["orientation"][i] = region.orientation
        result["centroid_row"][i] = region.centroid[0]
        result["centroid_col"][i] = region.centroid[1]

    return result


def compute_2d_slice_profiles(
    input_path: Path,
    output_file: Path,
    n_jobs: int = -1,
) -> Path:
    """
    Compute per-instance regionprops for every slice and save to NPZ.

    Slices along axis 0 (Z), which is the fiber axis by convention
    (guaranteed by the preparation pipeline).

    Args:
        input_path: Path to .zarr directory
        output_file: Output .npz path
        n_jobs: Number of parallel workers (-1 = all CPUs)

    Returns:
        Path to saved NPZ file
    """
    # Load volume
    logger.info(f"Loading Zarr volume: {input_path.name}")
    volume, iso_voxel_size = load_zarr_volume(input_path)
    voxel_size_tuple = (iso_voxel_size,) * 3

    # Ensure volume is a contiguous in-memory array (not lazy/memory-mapped)
    # before forking workers, and guard against int32 label truncation
    volume = np.ascontiguousarray(volume)
    max_label = volume.max()
    if max_label > np.iinfo(np.int32).max:
        raise ValueError(
            f"Max label {max_label} exceeds int32 range; regionprops requires int32 input"
        )

    n_slices = volume.shape[0]
    logger.info(f"Volume shape: {volume.shape}, voxel size: {iso_voxel_size:.4f} um")
    logger.info(f"Slicing along axis 0 (Z), {n_slices} slices")

    if n_slices == 0:
        logger.warning(f"Skipping {input_path.name}: volume has 0 slices along axis 0")
        return None

    # Determine workers
    available = cpu_count() or 1
    if n_jobs == -1:
        n_workers = available
    else:
        n_workers = min(max(1, n_jobs), available)
    logger.info(f"Using {n_workers} parallel workers")

    # Process slices
    slice_indices = list(range(n_slices))

    global _volume_data
    try:
        if n_workers > 1:
            with Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(volume,),
            ) as pool:
                results = list(
                    tqdm(
                        pool.imap(_process_slice, slice_indices),
                        total=n_slices,
                        desc="Processing slices",
                    )
                )
        else:
            _volume_data = volume
            results = [_process_slice(z) for z in tqdm(slice_indices, desc="Processing slices")]
    finally:
        _volume_data = None

    # Concatenate results into flat arrays
    arrays = {k: np.concatenate([r[k] for r in results]) for k in _COLUMN_KEYS}

    n_total = len(arrays["slice_index"])
    logger.info(f"Total instances: {n_total:,} across {n_slices} slices")

    # Compute derived radius measures (convenience columns)
    pixel_area_um2 = iso_voxel_size ** 2
    area_um2 = arrays["area_pixels"].astype(np.float32) * pixel_area_um2
    arrays["radius_circular_um"] = np.sqrt(area_um2 / np.pi).astype(np.float32)
    arrays["radius_minor_um"] = (
        arrays["axis_minor_length"] / 2.0 * iso_voxel_size
    ).astype(np.float32)

    # Compute instances per slice for quick lookups
    n_instances_per_slice = np.zeros(n_slices, dtype=np.int32)
    for r in results:
        if len(r["slice_index"]) > 0:
            z = r["slice_index"][0]
            n_instances_per_slice[z] = len(r["slice_index"])

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_file,
        # Per-instance data (flat table)
        **arrays,
        # Per-slice summary
        n_instances_per_slice=n_instances_per_slice,
        # Metadata
        voxel_size_um=np.float64(iso_voxel_size),
        voxel_size_original=np.array(voxel_size_tuple, dtype=np.float64),
        volume_shape=np.array(volume.shape, dtype=np.int32),
        source_file=str(input_path),
        n_slices=np.int32(n_slices),
        n_instances=np.int32(n_total),
    )

    logger.info(f"Saved {output_file} ({output_file.stat().st_size / 1024**2:.1f} MB)")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-instance 2D regionprops for all slices in a labeled volume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input", type=str, nargs="?",
        default="data/raw/rat/lm/*_myelin.zarr",
        help="Path to .zarr directory or glob pattern",
    )
    parser.add_argument(
        "output", type=Path, nargs="?",
        default=Path("data/processed/rat/lm"),
        help="Output .npz file (single) or output directory (batch mode)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Number of parallel workers (default: -1 = all CPUs)",
    )
    parser.add_argument(
        "--output-suffix", type=str, default="_slice_profiles",
        help='Suffix for output filenames in batch mode (default: "_slice_profiles")',
    )

    args = parser.parse_args()

    # Expand glob
    matched = sorted(glob.glob(args.input, recursive=True))
    if not matched:
        parser.error(f"No files matched: {args.input}")

    if len(matched) == 1:
        # Single file mode
        input_path = Path(matched[0])
        output_path = args.output
        if output_path.suffix != ".npz":
            # Output is a directory — construct filename
            stem = input_path.stem
            if stem.endswith(".zarr"):
                stem = Path(stem).stem
            output_path = output_path / f"{stem}{args.output_suffix}.npz"

        compute_2d_slice_profiles(
            input_path, output_path,
            n_jobs=args.n_jobs,
        )
    else:
        # Batch mode
        logger.info(f"Batch mode: {len(matched)} files")
        output_dir = args.output
        if output_dir.suffix == ".npz":
            parser.error("Output must be a directory in batch mode")

        for i, path_str in enumerate(matched, 1):
            input_path = Path(path_str)
            stem = input_path.stem
            output_path = output_dir / f"{stem}{args.output_suffix}.npz"

            logger.info(f"\n[{i}/{len(matched)}] {input_path.name} -> {output_path.name}")
            try:
                compute_2d_slice_profiles(
                    input_path, output_path,
                    n_jobs=args.n_jobs,
                )
            except Exception as e:
                logger.error(f"Failed: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    main()
