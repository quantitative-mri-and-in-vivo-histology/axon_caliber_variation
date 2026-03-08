#!/usr/bin/env python3
"""
Extract all HM volumes as OME-Zarr with axis alignment and isotropic resampling.

Finds all *_roi.json files in roi_dir, matches each to its source .mat file
in source_dir, and extracts axis-aligned, isotropically resampled volumes.

Example usage:
    # Use defaults (data/source/rat, data/raw/rat/hm)
    python scripts/preparation/extract_hm_rois.py

    # Custom directories
    python scripts/preparation/extract_hm_rois.py \
        --source-dir data/source/rat \
        --roi-dir data/raw/rat/hm
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
from axonometry.io import load_volume_with_metadata, resample_to_isotropic
from axonometry.zarr_io import write_ome_zarr_pyramid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_source_mat(roi_json: Path, source_dir: Path) -> Path:
    """
    Find the source .mat file for a given ROI JSON.

    ROI JSON name: hm_25_ipsi_roi.json
    Source .mat:   data/source/rat/Sham_25_ipsi/HM_25_ipsi_myelinated_axons.mat

    Extracts {id}_{hemisphere} (e.g. "25_ipsi") from the ROI name and searches
    for the matching HM_*_myelinated_axons.mat in source_dir.
    """
    stem = roi_json.stem.replace('_roi', '')  # e.g. "hm_25_ipsi"
    # Remove "hm_" prefix to get id_hemisphere
    match = re.match(r'hm_(.+)$', stem)
    if not match:
        raise ValueError(f"Cannot parse ROI JSON name: {roi_json.name}")

    id_hemi = match.group(1)  # e.g. "25_ipsi"

    # Search for matching .mat file
    pattern = f"HM_{id_hemi}_myelinated_axons.mat"
    matches = list(source_dir.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No source .mat found for {pattern} under {source_dir}")
    return matches[0]


def extract_hm_volume(
    mat_file: Path,
    roi_json: Path,
    output_dir: Path,
    voxel_size: Tuple[float, float, float] = (0.015, 0.015, 0.05),
    num_levels: int = 4,
) -> Dict:
    """
    Extract an HM volume with axis alignment and isotropic resampling.

    Args:
        mat_file: Source .mat file with labeled volume
        roi_json: Path to *_roi.json (fiber_dir_axis, min, max)
        output_dir: Output directory
        voxel_size: Voxel size (X, Y, Z) matching h5py axis order
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

    # Derive sample name from ROI JSON filename
    sample_name = roi_json.stem.replace('_roi', '')

    # Axis permutation: fiber_dir_axis -> axis 0
    if fiber_dir_axis == 0:
        perm = (0, 1, 2)
    elif fiber_dir_axis == 1:
        perm = (1, 0, 2)
    else:
        perm = (2, 0, 1)

    logger.info(f"\n{'='*80}")
    logger.info(f"Extracting HM volume: {sample_name}")
    logger.info(f"  Source: {mat_file}")
    logger.info(f"  ROI: {roi_min} -> {roi_max}")
    logger.info(f"  fiber_dir_axis: {fiber_dir_axis} -> permutation {perm}")
    logger.info(f"{'='*80}\n")

    # Load segmentation volume
    # voxel_size is (X, Y, Z) from CLI, which matches h5py axis order
    # (h5py reverses MATLAB axes, so .mat array is in ~(X, Y, Z) order)
    logger.info("Loading segmentation volume...")
    volume, voxel_size_tuple, _ = load_volume_with_metadata(
        mat_file, voxel_size_override=voxel_size,
    )
    logger.info(f"  Shape: {volume.shape}, dtype: {volume.dtype}")

    # Crop to ROI
    slices = tuple(slice(roi_min[i], roi_max[i]) for i in range(3))
    volume = volume[slices]
    logger.info(f"  After crop: {volume.shape}")

    # Permute axes so fiber direction is Z
    volume_aligned = np.transpose(volume, perm)
    logger.info(f"  After permutation: {volume_aligned.shape}")

    # Permute voxel size accordingly
    voxel_size_aligned = tuple(voxel_size[p] for p in perm)
    logger.info(f"  Aligned voxel size: {voxel_size_aligned} um")

    # Resample to isotropic voxels
    logger.info("Resampling to isotropic voxels...")
    volume_iso, iso_voxel_size = resample_to_isotropic(volume_aligned, voxel_size_aligned)
    logger.info(f"  Isotropic shape: {volume_iso.shape}, voxel size: {iso_voxel_size:.4f} um")

    del volume, volume_aligned

    # Write segmentation as OME-Zarr
    seg_path = output_dir / f"{sample_name}_myelin.zarr"
    logger.info(f"Writing segmentation: {seg_path}")
    write_ome_zarr_pyramid(
        volume_iso, seg_path,
        voxel_size_um=iso_voxel_size,
        num_levels=num_levels,
        downsample_mode='nearest',
    )
    output_paths = {'segmentation': str(seg_path)}

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

        # Crop to ROI
        grayscale = grayscale[slices]

        # Apply same permutation and resampling as segmentation
        grayscale_aligned = np.transpose(grayscale, perm)
        grayscale_iso, _ = resample_to_isotropic(grayscale_aligned, voxel_size_aligned)
        del grayscale, grayscale_aligned

        gray_path = output_dir / f"{sample_name}_grayscale.zarr"
        logger.info(f"Writing grayscale: {gray_path}")
        write_ome_zarr_pyramid(
            grayscale_iso, gray_path,
            voxel_size_um=iso_voxel_size,
            num_levels=num_levels,
            downsample_mode='mean',
        )
        output_paths['grayscale'] = str(gray_path)
        del grayscale_iso
    else:
        logger.warning(f"No grayscale file found: {grayscale_file}")

    del volume_iso

    logger.info(f"\n{'='*80}")
    logger.info("Extraction complete!")
    for kind, path in output_paths.items():
        logger.info(f"  {kind}: {path}")
    logger.info(f"{'='*80}\n")

    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description='Extract all HM volumes as OME-Zarr with axis alignment and isotropic resampling.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Finds all *_roi.json files in roi_dir, matches to source .mat files,
and extracts axis-aligned, isotropically resampled OME-Zarr volumes.

Outputs per ROI:
    - <sample>_myelin.zarr:      Segmentation (OME-Zarr, isotropic, axis-aligned)
    - <sample>_grayscale.zarr:   Grayscale image (if companion .h5 exists)
        """
    )
    parser.add_argument(
        '--source-dir', type=Path, default=Path('data/source/rat'),
        help='Root directory containing source .mat files (default: data/source/rat)'
    )
    parser.add_argument(
        '--roi-dir', type=Path, default=Path('data/raw/rat/hm'),
        help='Directory containing *_roi.json files and output (default: data/raw/rat/hm)'
    )
    parser.add_argument(
        '--voxel-size', type=float, nargs=3, default=[0.015, 0.015, 0.05],
        metavar=('X', 'Y', 'Z'),
        help='Voxel size in um (default: 0.015 0.015 0.05)'
    )
    parser.add_argument(
        '--num-levels', type=int, default=4,
        help='Number of pyramid levels (default: 4)'
    )

    args = parser.parse_args()

    # Find all ROI JSONs
    roi_files = sorted(args.roi_dir.glob('*_roi.json'))
    if not roi_files:
        logger.error(f"No *_roi.json files found in {args.roi_dir}")
        return

    logger.info(f"Found {len(roi_files)} ROI file(s)")

    for roi_file in roi_files:
        try:
            mat_file = find_source_mat(roi_file, args.source_dir)
            extract_hm_volume(
                mat_file, roi_file,
                output_dir=args.roi_dir,
                voxel_size=tuple(args.voxel_size),
                num_levels=args.num_levels,
            )
        except Exception as e:
            logger.error(f"Failed to process {roi_file}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
