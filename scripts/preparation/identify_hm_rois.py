#!/usr/bin/env python3
"""
Identify the fiber direction (dominant axis) in HM volumes.

HM (high magnification) volumes contain a single white matter population.
The dominant axis is the one where axon cross-sections appear most circular
(lowest eccentricity), indicating the fiber direction is perpendicular to
that slicing plane.

Outputs a JSON file with the dominant axis and volume metadata, used by
extract_hm_rois.py for axis-aligned extraction.

Example usage:
    # Single file
    python scripts/preparation/identify_hm_rois.py \
        data/raw/Sham_25_ipsi/HM_25_ipsi_myelinated_axons.mat \
        data/raw/rat/HM

    # Batch processing
    python scripts/preparation/identify_hm_rois.py \
        "data/raw/**/HM*_myelinated_axons.mat" \
        data/raw/rat/HM
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from skimage.measure import regionprops

from axonometry.io import load_volume_with_metadata, resample_to_isotropic

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
) -> Tuple[int, Dict]:
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
        Tuple of (dominant_axis_index, per_axis_results)
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

        results[axis] = {
            'eccentricity': median_ecc,
            'eccentricity_iqr': iqr,
            'n_slices_sampled': len(slice_eccs),
            'n_axons_total': total_axons,
            'axis_name': axis_names[axis],
        }
        logger.info(f"  Axis {axis} ({axis_names[axis]}): eccentricity={median_ecc:.3f} "
                     f"(IQR={iqr:.3f}), {len(slice_eccs)} slices, {total_axons} axons")

    dominant = min(results, key=lambda a: results[a]['eccentricity'])
    logger.info(f"  Dominant axis: {dominant} ({axis_names[dominant]})")

    return dominant, results


def identify_hm_roi(
    mat_file: Path,
    output_dir: Path,
    voxel_size: Tuple[float, float, float] = (0.015, 0.015, 0.05),
    min_area: int = 10,
) -> Dict:
    """
    Identify the fiber direction in an HM volume and write ROI JSON.

    Args:
        mat_file: Input .mat file with labeled volume
        output_dir: Output directory for JSON results
        voxel_size: Voxel size (X, Y, Z) in micrometers
        min_area: Minimum axon area for eccentricity analysis

    Returns:
        ROI metadata dictionary
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Identifying HM ROI: {mat_file.name}")
    logger.info(f"{'='*80}\n")

    # Load volume
    # voxel_size is (X, Y, Z) from CLI, which matches h5py axis order
    # (h5py reverses MATLAB axes, so .mat array is in ~(X, Y, Z) order)
    logger.info("Loading volume...")
    volume, voxel_size_tuple, _ = load_volume_with_metadata(
        mat_file, voxel_size_override=voxel_size,
    )
    logger.info(f"  Shape: {volume.shape}, voxel size: {voxel_size_tuple}")

    # Resample to isotropic for eccentricity analysis
    logger.info("Resampling to isotropic...")
    volume_iso, iso_voxel_size = resample_to_isotropic(volume, voxel_size_tuple)
    logger.info(f"  Isotropic shape: {volume_iso.shape}, voxel size: {iso_voxel_size:.4f} um")

    # Find dominant axis
    logger.info("Analyzing eccentricity per axis...")
    dominant_axis, _axis_results = find_dominant_axis(volume_iso, min_area)

    # Derive sample name: HM_25_ipsi_myelinated_axons -> hm_25_ipsi
    sample_name = mat_file.stem.replace('_myelinated_axons', '').lower()

    # Build simplified ROI JSON (full volume, just fiber direction)
    roi_data = {
        "fiber_dir_axis": dominant_axis,
        "min": [0, 0, 0],
        "max": list(volume.shape),
    }

    # Write JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{sample_name}_roi.json"
    with open(json_path, 'w') as f:
        json.dump(roi_data, f, indent=2)

    axis_names = {0: 'Z', 1: 'Y', 2: 'X'}
    logger.info(f"\n{'='*80}")
    logger.info(f"Saved: {json_path}")
    logger.info(f"  fiber_dir_axis: {dominant_axis} ({axis_names[dominant_axis]})")
    logger.info(f"  min: [0, 0, 0], max: {list(volume.shape)}")
    logger.info(f"{'='*80}\n")

    return roi_data


def main():
    parser = argparse.ArgumentParser(
        description='Identify fiber direction in all HM volumes under a source directory.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/preparation/identify_hm_rois.py \\
        data/source/rat \\
        data/raw/rat/HM

Finds all HM_*_myelinated_axons.mat files under source_dir and writes
ROI JSONs to output_dir.
        """
    )
    parser.add_argument(
        'source_dir', type=Path, nargs='?', default=Path('data/source/rat'),
        help='Root directory containing HM source .mat files (default: data/source/rat)'
    )
    parser.add_argument(
        'output_dir', type=Path, nargs='?', default=Path('data/raw/rat/hm'),
        help='Output directory for ROI JSONs (default: data/raw/rat/hm)'
    )
    parser.add_argument(
        '--voxel-size', type=float, nargs=3, default=[0.015, 0.015, 0.05],
        metavar=('X', 'Y', 'Z'),
        help='Voxel size in um (default: 0.015 0.015 0.05)'
    )
    parser.add_argument(
        '--min-area', type=int, default=10,
        help='Minimum axon area in pixels (default: 10)'
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
            identify_hm_roi(
                mat_path,
                args.output_dir,
                voxel_size=tuple(args.voxel_size),
                min_area=args.min_area,
            )
        except Exception as e:
            logger.error(f"Failed to process {mat_path}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
