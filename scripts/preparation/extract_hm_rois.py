#!/usr/bin/env python3
"""
Extract HM volumes as OME-Zarr with axis alignment and isotropic resampling.

Reads ROI definitions from *_roi.json (produced by identify_hm_rois.py)
and extracts the full volume with:
1. Axis permutation so the fiber direction (dominant axis) becomes Z
2. Resampling to isotropic voxels (nearest-neighbor for labels)
3. Multi-resolution OME-Zarr pyramid output

Example usage:
    # Single file
    python scripts/preparation/extract_hm_rois.py \
        data/raw/rat/HM/HM_25_ipsi_roi.json

    # Batch processing
    python scripts/preparation/extract_hm_rois.py \
        "data/raw/rat/HM/*_roi.json"

    # Custom output directory
    python scripts/preparation/extract_hm_rois.py \
        data/raw/rat/HM/HM_25_ipsi_roi.json \
        --output-dir data/processed/rat/HM
"""

import argparse
import json
import logging
from glob import glob
from pathlib import Path
from typing import Dict

import numpy as np
from axonometry.io import load_volume_with_metadata, resample_to_isotropic
from axonometry.zarr_io import write_ome_zarr_pyramid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_hm_volume(
    roi_json: Path,
    output_dir: Path = None,
    num_levels: int = 4,
) -> Dict:
    """
    Extract an HM volume with axis alignment and isotropic resampling.

    Args:
        roi_json: Path to *_roi.json from identify_hm_rois.py
        output_dir: Output directory (default: same directory as roi_json)
        num_levels: Number of pyramid levels for OME-Zarr output

    Returns:
        Dictionary with output file paths
    """
    with open(roi_json) as f:
        roi_meta = json.load(f)

    mat_file = Path(roi_meta['source_file'])
    voxel_size_xyz = tuple(roi_meta['voxel_size_xyz_um'])
    volume_shape = tuple(roi_meta['volume_shape_zyx'])
    dominant_axis = roi_meta['dominant_axis']
    dominant_axis_name = roi_meta['dominant_axis_name']

    if output_dir is None:
        output_dir = roi_json.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive sample name from JSON filename
    sample_name = roi_json.stem.replace('_roi', '')

    logger.info(f"\n{'='*80}")
    logger.info(f"Extracting HM volume: {sample_name}")
    logger.info(f"  Source: {mat_file}")
    logger.info(f"  Volume shape: {volume_shape}")
    logger.info(f"  Voxel size (X,Y,Z): {voxel_size_xyz} um")
    logger.info(f"  Dominant axis: {dominant_axis} ({dominant_axis_name})")
    logger.info(f"{'='*80}\n")

    if not mat_file.exists():
        raise FileNotFoundError(f"Source .mat file not found: {mat_file}")

    # Build axis permutation: dominant_axis -> axis 0 (fiber direction along Z)
    if dominant_axis == 0:
        perm = (0, 1, 2)
    elif dominant_axis == 1:
        perm = (1, 0, 2)
    else:
        perm = (2, 0, 1)

    axis_names = {0: 'Z', 1: 'Y', 2: 'X'}
    logger.info(f"Axis permutation: {perm} ({axis_names[dominant_axis]} -> Z)")

    # Load full segmentation volume
    logger.info("\nLoading full segmentation volume...")
    volume, voxel_size_tuple, _ = load_volume_with_metadata(
        mat_file, voxel_size_override=voxel_size_xyz,
    )
    logger.info(f"  Shape: {volume.shape}, dtype: {volume.dtype}")

    # Permute axes so fiber direction is Z
    volume_aligned = np.transpose(volume, perm)
    logger.info(f"  After permutation: {volume_aligned.shape}")

    # Permute voxel size accordingly
    # h5py reverses MATLAB axes, so the loaded array is in ~(X, Y, Z) order
    # and voxel_size_xyz = (vx, vy, vz) already matches h5py axis order
    voxel_size_aligned = tuple(voxel_size_xyz[p] for p in perm)
    logger.info(f"  Aligned voxel size: {voxel_size_aligned} um")

    # Resample to isotropic voxels
    logger.info("\nResampling to isotropic voxels...")
    volume_iso, iso_voxel_size = resample_to_isotropic(volume_aligned, voxel_size_aligned)
    logger.info(f"  Isotropic shape: {volume_iso.shape}, voxel size: {iso_voxel_size:.4f} um")

    del volume, volume_aligned

    # Write segmentation as OME-Zarr
    seg_path = output_dir / f"{sample_name}_myelin.zarr"
    logger.info(f"\nWriting segmentation: {seg_path}")
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
        import h5py
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

    # Write per-volume metadata JSON
    meta_path = output_dir / f"{sample_name}_myelin.json"
    vol_meta = {
        "voxel_size": [iso_voxel_size] * 3,
        "unit": "micrometer",
        "source_file": str(mat_file),
        "roi_json": str(roi_json),
        "original_voxel_size_xyz_um": list(voxel_size_xyz),
        "shape_zyx": list(volume_iso.shape),
        "size_um_zyx": [s * iso_voxel_size for s in volume_iso.shape],
        "dominant_axis": dominant_axis,
        "dominant_axis_name": dominant_axis_name,
        "axis_permutation": list(perm),
        "isotropic_voxel_size_um": iso_voxel_size,
        "num_levels": num_levels,
    }
    with open(meta_path, 'w') as f:
        json.dump(vol_meta, f, indent=2)

    output_paths['metadata'] = str(meta_path)

    del volume_iso

    logger.info(f"\n{'='*80}")
    logger.info("Extraction complete!")
    for kind, path in output_paths.items():
        logger.info(f"  {kind}: {path}")
    logger.info(f"{'='*80}\n")

    return output_paths


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Extract HM volumes as OME-Zarr with axis alignment and isotropic resampling.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single ROI file
    python scripts/preparation/extract_hm_rois.py \\
        data/raw/rat/HM/HM_25_ipsi_roi.json

    # Batch processing
    python scripts/preparation/extract_hm_rois.py \\
        "data/raw/rat/HM/*_roi.json"

    # Custom output directory
    python scripts/preparation/extract_hm_rois.py \\
        data/raw/rat/HM/HM_25_ipsi_roi.json \\
        --output-dir data/processed/rat/HM

Inputs:
    *_roi.json from identify_hm_rois.py

Outputs:
    - <sample>_myelin.zarr:      Segmentation (OME-Zarr, isotropic, axis-aligned)
    - <sample>_grayscale.zarr:   Grayscale image (OME-Zarr, if companion .h5 exists)
    - <sample>_myelin.json:      Per-volume metadata (shape, voxel size, etc.)
        """
    )
    parser.add_argument(
        'input_files', type=str,
        help='*_roi.json file(s) (glob pattern supported)'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=None,
        help='Output directory (default: same as input JSON)'
    )
    parser.add_argument(
        '--num-levels', type=int, default=4,
        help='Number of pyramid levels (default: 4)'
    )

    args = parser.parse_args()

    # Expand glob pattern
    input_pattern = args.input_files
    if '*' in input_pattern:
        input_files = sorted(glob(input_pattern, recursive=True))
    else:
        input_files = [input_pattern]

    if not input_files:
        logger.error(f"No files found matching: {args.input_files}")
        return

    logger.info(f"Found {len(input_files)} input file(s)")

    for json_file in input_files:
        json_path = Path(json_file)
        if not json_path.exists():
            logger.warning(f"File not found: {json_file}")
            continue

        try:
            extract_hm_volume(
                json_path,
                output_dir=args.output_dir,
                num_levels=args.num_levels,
            )
        except Exception as e:
            logger.error(f"Failed to process {json_file}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
