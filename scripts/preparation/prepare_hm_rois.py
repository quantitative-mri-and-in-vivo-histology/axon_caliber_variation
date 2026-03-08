#!/usr/bin/env python3
"""
Identify fiber direction and extract HM volumes as OME-Zarr.

Combines identification (eccentricity-based dominant axis detection) and
extraction (axis alignment + isotropic resampling) into a single step.

HM (high magnification) volumes contain a single white matter population.
The dominant axis is the one where axon cross-sections appear most circular
(lowest eccentricity), indicating the fiber direction is perpendicular to
that slicing plane.

Example usage:
    # Use defaults
    python scripts/preparation/prepare_hm_rois.py

    # Custom directories
    python scripts/preparation/prepare_hm_rois.py \
        --source-dir data/source/rat \
        --output-dir data/raw/rat/hm
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
from skimage.measure import regionprops

from axonometry.io import load_volume_with_metadata, resample_to_isotropic
from axonometry.zarr_io import write_ome_zarr_pyramid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_slice_eccentricity(slice_2d: np.ndarray, min_area: int = 10) -> Tuple[float, int]:
    """
    Compute average eccentricity of axons in a 2D slice.

    Lower eccentricity = more circular cross-sections = perpendicular to fibers.

    Returns:
        Tuple of (mean_eccentricity, n_axons)
    """
    regions = regionprops(slice_2d.astype(np.int32))
    eccentricities = [r.eccentricity for r in regions if r.area >= min_area]
    if not eccentricities:
        return 1.0, 0
    return float(np.mean(eccentricities)), len(eccentricities)


def find_dominant_axis(
    volume: np.ndarray,
    min_area: int = 10,
    n_samples: int = 10,
) -> int:
    """
    Find the dominant axis by comparing eccentricity along each axis.

    Samples multiple evenly-spaced slices per axis (avoiding the edges)
    and uses the median eccentricity for a robust estimate.

    The axis with the lowest eccentricity (most circular cross-sections)
    is the fiber direction.

    Args:
        volume: 3D labeled volume (Z, Y, X), isotropic voxels
        min_area: Minimum area in pixels for valid axons
        n_samples: Number of slices to sample per axis

    Returns:
        Dominant axis index
    """
    axis_names = ['Z', 'Y', 'X']
    results = {}

    for axis in range(3):
        n_slices = volume.shape[axis]

        # Sample evenly-spaced slices from the central 60% of the axis
        margin = int(n_slices * 0.2)
        n_actual = min(n_samples, n_slices - 2 * margin)
        if n_actual < 1:
            n_actual = 1
            margin = n_slices // 2
        indices = np.linspace(margin, n_slices - 1 - margin, n_actual, dtype=int)

        slice_eccs = []
        total_axons = 0
        for idx in indices:
            if axis == 0:
                slice_2d = volume[idx, :, :]
            elif axis == 1:
                slice_2d = volume[:, idx, :]
            else:
                slice_2d = volume[:, :, idx]

            mean_ecc, n_axons = compute_slice_eccentricity(slice_2d, min_area)
            if n_axons > 0:
                slice_eccs.append(mean_ecc)
                total_axons += n_axons

        if slice_eccs:
            median_ecc = float(np.median(slice_eccs))
            iqr = float(np.percentile(slice_eccs, 75) - np.percentile(slice_eccs, 25))
        else:
            median_ecc = 1.0
            iqr = 0.0

        results[axis] = median_ecc
        logger.info(f"  Axis {axis} ({axis_names[axis]}): eccentricity={median_ecc:.3f} "
                     f"(IQR={iqr:.3f}), {len(slice_eccs)} slices, {total_axons} axons")

    dominant = min(results, key=lambda a: results[a])
    logger.info(f"  Dominant axis: {dominant} ({axis_names[dominant]})")

    return dominant


def prepare_hm_volume(
    mat_file: Path,
    output_dir: Path,
    voxel_size: Tuple[float, float, float] = (0.015, 0.015, 0.05),
    min_area: int = 10,
    num_levels: int = 4,
) -> Dict:
    """
    Identify fiber direction and extract an HM volume in one step.

    Args:
        mat_file: Source .mat file with labeled volume
        output_dir: Output directory for Zarr volumes
        voxel_size: Voxel size (X, Y, Z) in micrometers
        min_area: Minimum axon area for eccentricity analysis
        num_levels: Number of pyramid levels

    Returns:
        Dictionary with output file paths
    """
    # Derive sample name: HM_25_ipsi_myelinated_axons -> hm_25_ipsi
    sample_name = mat_file.stem.replace('_myelinated_axons', '').lower()

    logger.info(f"\n{'='*80}")
    logger.info(f"Preparing HM volume: {sample_name}")
    logger.info(f"  Source: {mat_file}")
    logger.info(f"{'='*80}\n")

    # Load volume
    logger.info("Loading volume...")
    volume, voxel_size_tuple, _ = load_volume_with_metadata(
        mat_file, voxel_size_override=voxel_size,
    )
    logger.info(f"  Shape: {volume.shape}, dtype: {volume.dtype}, voxel size: {voxel_size_tuple}")

    # Resample to isotropic for eccentricity analysis
    logger.info("Resampling to isotropic...")
    volume_iso, iso_voxel_size = resample_to_isotropic(volume, voxel_size_tuple)
    logger.info(f"  Isotropic shape: {volume_iso.shape}, voxel size: {iso_voxel_size:.4f} um")

    # Find dominant axis on isotropic volume
    logger.info("Analyzing eccentricity per axis...")
    fiber_dir_axis = find_dominant_axis(volume_iso, min_area)

    del volume  # free original

    # Axis permutation: fiber_dir_axis -> axis 0
    if fiber_dir_axis == 0:
        perm = (0, 1, 2)
    elif fiber_dir_axis == 1:
        perm = (1, 0, 2)
    else:
        perm = (2, 0, 1)

    logger.info(f"  fiber_dir_axis: {fiber_dir_axis} -> permutation {perm}")

    # Permute isotropic volume (already isotropic, so no need to resample again)
    volume_aligned = np.transpose(volume_iso, perm)
    logger.info(f"  After permutation: {volume_aligned.shape}")

    del volume_iso

    # Write segmentation as OME-Zarr
    output_dir.mkdir(parents=True, exist_ok=True)
    seg_path = output_dir / f"{sample_name}_myelin.zarr"
    logger.info(f"Writing segmentation: {seg_path}")
    write_ome_zarr_pyramid(
        volume_aligned, seg_path,
        voxel_size_um=iso_voxel_size,
        num_levels=num_levels,
        downsample_mode='nearest',
    )
    output_paths = {'segmentation': str(seg_path)}

    del volume_aligned

    # Find companion grayscale .h5 file
    grayscale_name = mat_file.stem.replace('_myelinated_axons', '') + '.h5'
    grayscale_file = mat_file.parent / grayscale_name
    if grayscale_file.exists():
        logger.info(f"\nLoading grayscale: {grayscale_file.name}")
        with h5py.File(str(grayscale_file), 'r') as f:
            for dset_name in ['raw', 'data', 'image']:
                if dset_name in f:
                    break
            else:
                dset_name = list(f.keys())[0]
            grayscale = f[dset_name][:]
        logger.info(f"  Grayscale raw shape: {grayscale.shape}, dtype: {grayscale.dtype}")

        # Remove singleton channel dims (some .h5 files are 4D)
        grayscale = grayscale.squeeze()
        logger.info(f"  Grayscale after squeeze: {grayscale.shape}")

        # Align axes with segmentation: .h5 axes are reversed relative to .mat
        grayscale = np.transpose(grayscale, (2, 1, 0))
        logger.info(f"  Grayscale after axis alignment: {grayscale.shape}")

        # Resample to isotropic and permute
        grayscale_iso, _ = resample_to_isotropic(grayscale, voxel_size_tuple)
        del grayscale
        grayscale_aligned = np.transpose(grayscale_iso, perm)
        del grayscale_iso

        gray_path = output_dir / f"{sample_name}_grayscale.zarr"
        logger.info(f"Writing grayscale: {gray_path}")
        write_ome_zarr_pyramid(
            grayscale_aligned, gray_path,
            voxel_size_um=iso_voxel_size,
            num_levels=num_levels,
            downsample_mode='mean',
        )
        output_paths['grayscale'] = str(gray_path)
        del grayscale_aligned
    else:
        logger.warning(f"No grayscale file found: {grayscale_file}")

    logger.info(f"\n{'='*80}")
    logger.info("Extraction complete!")
    for kind, path in output_paths.items():
        logger.info(f"  {kind}: {path}")
    logger.info(f"{'='*80}\n")

    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description='Identify fiber direction and extract all HM volumes as OME-Zarr.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Finds all HM_*_myelinated_axons.mat files under source_dir, determines the
fiber direction via eccentricity analysis, and extracts axis-aligned,
isotropically resampled OME-Zarr volumes.

Outputs per volume:
    - <sample>_myelin.zarr:      Segmentation (OME-Zarr, isotropic, axis-aligned)
    - <sample>_grayscale.zarr:   Grayscale image (if companion .h5 exists)
        """
    )
    parser.add_argument(
        '--source-dir', type=Path, default=Path('data/source/rat'),
        help='Root directory containing source .mat files (default: data/source/rat)'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('data/raw/rat/hm'),
        help='Output directory for Zarr volumes (default: data/raw/rat/hm)'
    )
    parser.add_argument(
        '--voxel-size', type=float, nargs=3, default=[0.015, 0.015, 0.05],
        metavar=('X', 'Y', 'Z'),
        help='Voxel size in um (default: 0.015 0.015 0.05)'
    )
    parser.add_argument(
        '--min-area', type=int, default=10,
        help='Minimum axon area in pixels for eccentricity analysis (default: 10)'
    )
    parser.add_argument(
        '--num-levels', type=int, default=4,
        help='Number of pyramid levels (default: 4)'
    )

    args = parser.parse_args()

    # Find all HM .mat files under source_dir
    input_files = sorted(args.source_dir.rglob('HM_*_myelinated_axons.mat'))

    if not input_files:
        logger.error(f"No HM_*_myelinated_axons.mat files found under {args.source_dir}")
        return

    logger.info(f"Found {len(input_files)} HM volume(s)")

    for mat_path in input_files:
        try:
            prepare_hm_volume(
                mat_path,
                args.output_dir,
                voxel_size=tuple(args.voxel_size),
                min_area=args.min_area,
                num_levels=args.num_levels,
            )
        except Exception as e:
            logger.error(f"Failed to process {mat_path}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
