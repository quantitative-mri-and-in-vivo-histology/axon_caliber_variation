"""
Extract all LM ROI sub-volumes as OME-Zarr with multi-resolution pyramids.

Finds all *_roi.json files in roi_dir, matches each to its source .mat file
in source_dir, and extracts cropped, axis-aligned volumes.

Example usage:
    # Use defaults (data/source/rat, data/raw/rat/lm)
    python scripts/preparation/extract_rois.py

    # Custom directories
    python scripts/preparation/extract_rois.py \
        --source-dir data/source/rat \
        --roi-dir data/raw/rat/lm
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
from axonometry.io import write_ome_zarr_pyramid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VOXEL_SIZE_UM = 0.05  # LM data is isotropic


def find_source_mat(roi_json: Path, source_dir: Path) -> Path:
    """
    Find the source .mat file for a given ROI JSON.

    ROI JSON name: sham_25_ipsi_cc_roi.json
    Source .mat:   data/source/rat/Sham_25_ipsi/LM_25_ipsi_myelinated_axons.mat

    Extracts {id}_{hemisphere} (e.g. "25_ipsi") from the ROI name and searches
    for the matching LM_*_myelinated_axons.mat in source_dir.
    """
    stem = roi_json.stem.replace('_roi', '')  # e.g. "sham_25_ipsi_cc"
    # Remove condition prefix and population suffix to get id_hemisphere
    # Pattern: {condition}_{id}_{hemisphere}_{pop} or {condition}_{id}_{hemisphere}
    match = re.match(r'[a-z]+_(\d+_\w+?)(?:_(?:cc|cg))?$', stem)
    if not match:
        raise ValueError(f"Cannot parse ROI JSON name: {roi_json.name}")

    id_hemi = match.group(1)  # e.g. "25_ipsi"

    # Search for matching .mat file
    pattern = f"LM_{id_hemi}_myelinated_axons.mat"
    matches = list(source_dir.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No source .mat found for {pattern} under {source_dir}")
    return matches[0]


def extract_roi(
    mat_file: Path,
    roi_json: Path,
    output_dir: Path,
    num_levels: int = 4,
) -> Dict:
    """
    Extract a single ROI sub-volume.

    Args:
        mat_file: Source .mat file with labeled volume
        roi_json: Path to *_roi.json (fiber_dir_axis, min, max)
        output_dir: Output directory
        num_levels: Number of pyramid levels

    Returns:
        Dictionary with output file paths
    """
    with open(roi_json) as f:
        roi = json.load(f)

    fiber_dir_axis = roi['fiber_dir_axis']
    roi_min = roi['min']
    roi_max = roi['max']

    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive name from ROI JSON filename (e.g. sham_25_ipsi_cc_roi.json -> sham_25_ipsi_cc)
    sample_name = roi_json.stem.replace('_roi', '')

    # Axis permutation: fiber_dir_axis -> axis 0
    if fiber_dir_axis == 0:
        perm = (0, 1, 2)
    elif fiber_dir_axis == 1:
        perm = (1, 0, 2)
    else:
        perm = (2, 0, 1)

    logger.info(f"\n--- Extracting {sample_name} ---")
    logger.info(f"  Source: {mat_file}")
    logger.info(f"  ROI: {roi_min} -> {roi_max}")
    logger.info(f"  fiber_dir_axis: {fiber_dir_axis} -> permutation {perm}")

    slices = tuple(slice(roi_min[i], roi_max[i]) for i in range(3))

    # Check for empty ROI
    extent = [roi_max[i] - roi_min[i] for i in range(3)]
    if any(e <= 0 for e in extent):
        logger.warning(f"  Skipping: empty ROI (extent {extent})")
        return {}

    # Load segmentation (cropped)
    logger.info(f"  Loading segmentation...")
    with h5py.File(str(mat_file), 'r') as f:
        for key in f.keys():
            if not key.startswith('#') and not key.startswith('_'):
                seg_cropped = f[key][slices]
                break
        else:
            raise ValueError(f"No data found in {mat_file}")

    seg_aligned = np.transpose(seg_cropped, perm)
    logger.info(f"  Segmentation: {seg_cropped.shape} -> {seg_aligned.shape}")

    seg_path = output_dir / f"{sample_name}_myelin.zarr"
    write_ome_zarr_pyramid(
        seg_aligned, seg_path,
        voxel_size_um=VOXEL_SIZE_UM,
        num_levels=num_levels,
        downsample_mode='nearest',
    )
    output_paths = {'segmentation': str(seg_path)}

    del seg_cropped, seg_aligned

    # Try companion grayscale .h5
    grayscale_name = mat_file.stem.replace('_myelinated_axons', '') + '.h5'
    grayscale_file = mat_file.parent / grayscale_name
    if grayscale_file.exists():
        logger.info(f"  Loading grayscale: {grayscale_file.name}")
        with h5py.File(str(grayscale_file), 'r') as f:
            for dset_name in ['raw', 'data', 'image']:
                if dset_name in f:
                    break
            else:
                dset_name = list(f.keys())[0]
            gray_cropped = f[dset_name][slices]

        gray_aligned = np.transpose(gray_cropped, perm)

        gray_path = output_dir / f"{sample_name}_grayscale.zarr"
        write_ome_zarr_pyramid(
            gray_aligned, gray_path,
            voxel_size_um=VOXEL_SIZE_UM,
            num_levels=num_levels,
            downsample_mode='mean',
        )
        output_paths['grayscale'] = str(gray_path)
        del gray_cropped, gray_aligned
    else:
        logger.warning(f"  No grayscale file found: {grayscale_file}")

    logger.info(f"  Done: {sample_name}")
    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description='Extract all LM ROI sub-volumes as OME-Zarr.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Finds all *_roi.json files in roi_dir, matches to source .mat files,
and extracts cropped, axis-aligned OME-Zarr volumes.

Outputs per ROI:
    - <sample>_myelin.zarr:     Segmentation (OME-Zarr, multi-resolution)
    - <sample>_grayscale.zarr:  Grayscale image (if companion .h5 exists)
        """
    )
    parser.add_argument(
        '--source-dir', type=Path, default=Path('data/source/rat'),
        help='Root directory containing source .mat files (default: data/source/rat)'
    )
    parser.add_argument(
        '--mask-dir', type=Path, default=Path('data/masks/rat/lm'),
        help='Directory containing *_roi.json files (default: data/masks/rat/lm)'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('data/raw/rat/lm'),
        help='Output directory for Zarr volumes (default: data/raw/rat/lm)'
    )
    parser.add_argument(
        '--num-levels', type=int, default=4,
        help='Number of pyramid levels (default: 4)'
    )

    args = parser.parse_args()

    # Find all ROI JSONs
    roi_files = sorted(args.mask_dir.glob('*_roi.json'))
    if not roi_files:
        logger.error(f"No *_roi.json files found in {args.mask_dir}")
        return

    logger.info(f"Found {len(roi_files)} ROI file(s)")

    for roi_file in roi_files:
        try:
            mat_file = find_source_mat(roi_file, args.source_dir)
            extract_roi(
                mat_file, roi_file,
                output_dir=args.output_dir,
                num_levels=args.num_levels,
            )
        except Exception as e:
            logger.error(f"Failed to process {roi_file}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
