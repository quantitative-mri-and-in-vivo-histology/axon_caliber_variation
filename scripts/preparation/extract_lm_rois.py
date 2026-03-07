#!/usr/bin/env python3
"""
Extract LM ROI sub-volumes as OME-Zarr with multi-resolution pyramids.

Reads ROI definitions from *_population_rois.json (produced by
identify_lm_rois.py) and extracts axis-aligned, cropped sub-volumes
for each population (CC, CG) with companion grayscale if available.

Example usage:
    # Single ROI file
    python scripts/preparation/extract_lm_rois.py \
        data/raw/rat/LM/sham_25_ipsi_population_rois.json

    # Batch processing
    python scripts/preparation/extract_lm_rois.py \
        "data/raw/rat/LM/*_population_rois.json"

    # Custom output directory and pyramid levels
    python scripts/preparation/extract_lm_rois.py \
        data/raw/rat/LM/sham_25_ipsi_population_rois.json \
        --output-dir data/processed/rat/LM \
        --num-levels 5
"""

import argparse
import json
import logging
from glob import glob
from pathlib import Path
from typing import Dict

import numpy as np
from axonometry.io import load_volume_with_metadata
from axonometry.zarr_io import write_ome_zarr_pyramid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_rois(
    roi_json: Path,
    output_dir: Path = None,
    num_levels: int = 4,
) -> Dict:
    """
    Extract ROI sub-volumes from a *_population_rois.json file.

    Args:
        roi_json: Path to *_population_rois.json from identify_lm_rois.py
        output_dir: Output directory (default: same directory as roi_json)
        num_levels: Number of pyramid levels for OME-Zarr output

    Returns:
        Dictionary with output file paths per population
    """
    with open(roi_json) as f:
        roi_meta = json.load(f)

    mat_file = Path(roi_meta['source_file'])
    voxel_size_um = roi_meta['voxel_size_um']
    volume_shape = tuple(roi_meta['volume_shape_zyx'])
    rois = roi_meta['rois']
    separation = roi_meta['separation']

    if output_dir is None:
        output_dir = roi_json.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive sample name from JSON filename
    sample_name = roi_json.stem.replace('_population_rois', '')

    logger.info(f"\n{'='*80}")
    logger.info(f"Extracting ROIs: {sample_name}")
    logger.info(f"  Source: {mat_file}")
    logger.info(f"  Volume shape: {volume_shape}")
    logger.info(f"  Separation: axis {separation['axis_name']} at {separation['position_um']:.1f} um")
    logger.info(f"{'='*80}\n")

    if not mat_file.exists():
        raise FileNotFoundError(f"Source .mat file not found: {mat_file}")

    # Build axis permutation: separation_axis -> axis 0 (fiber direction along z)
    separation_axis = separation['axis']
    axis_names = {0: 'Z', 1: 'Y', 2: 'X'}
    if separation_axis == 0:
        perm = (0, 1, 2)
    elif separation_axis == 1:
        perm = (1, 0, 2)
    else:
        perm = (2, 0, 1)
    logger.info(f"Axis permutation: {perm} ({axis_names[separation_axis]} -> Z)")

    # Load full segmentation volume
    logger.info("\nLoading full segmentation volume...")
    volume_full, _, _ = load_volume_with_metadata(mat_file, voxel_size_override=None)

    # Find companion grayscale .h5 file
    grayscale_name = mat_file.stem.replace('_myelinated_axons', '') + '.h5'
    grayscale_file = mat_file.parent / grayscale_name
    grayscale_full = None
    if grayscale_file.exists():
        import h5py
        logger.info(f"Loading grayscale: {grayscale_file.name}")
        with h5py.File(str(grayscale_file), 'r') as f:
            for dset_name in ['raw', 'data', 'image']:
                if dset_name in f:
                    break
            else:
                dset_name = list(f.keys())[0]
            grayscale_full = f[dset_name][:]
        logger.info(f"  Grayscale shape: {grayscale_full.shape}, dtype: {grayscale_full.dtype}")
    else:
        logger.warning(f"No grayscale file found: {grayscale_file}")

    # Extract each ROI
    output_paths = {}
    for pop_name, roi in rois.items():
        logger.info(f"\n--- Extracting {pop_name.upper()} ---")

        # Convert ROI bounds from um to voxels
        min_vox = [max(0, int(roi['min_um'][i] / voxel_size_um)) for i in range(3)]
        max_vox = [min(volume_shape[i], int(np.ceil(roi['max_um'][i] / voxel_size_um))) for i in range(3)]

        slices = tuple(slice(min_vox[i], max_vox[i]) for i in range(3))

        # Crop and permute segmentation
        seg_cropped = volume_full[slices]
        seg_aligned = np.transpose(seg_cropped, perm)
        logger.info(f"  Segmentation: {volume_full.shape} -> {seg_cropped.shape} -> {seg_aligned.shape}")

        seg_path = output_dir / f"{sample_name}_{pop_name}_myelin.zarr"
        write_ome_zarr_pyramid(
            seg_aligned, seg_path,
            voxel_size_um=voxel_size_um,
            num_levels=num_levels,
            downsample_mode='nearest',
        )
        output_paths[pop_name] = {'segmentation': str(seg_path)}

        # Crop and permute grayscale
        if grayscale_full is not None:
            gray_cropped = grayscale_full[slices]
            gray_aligned = np.transpose(gray_cropped, perm)

            gray_path = output_dir / f"{sample_name}_{pop_name}_grayscale.zarr"
            write_ome_zarr_pyramid(
                gray_aligned, gray_path,
                voxel_size_um=voxel_size_um,
                num_levels=num_levels,
                downsample_mode='mean',
            )
            output_paths[pop_name]['grayscale'] = str(gray_path)

        # Write per-ROI metadata JSON
        roi_shape_aligned = list(seg_aligned.shape)
        roi_size_um = [s * voxel_size_um for s in roi_shape_aligned]

        meta_path = output_dir / f"{sample_name}_{pop_name}_myelin.json"
        pop_meta = {
            "voxel_size": [voxel_size_um] * 3,
            "unit": "micrometer",
            "source_file": str(mat_file),
            "roi_json": str(roi_json),
            "population": pop_name,
            "shape_zyx": roi_shape_aligned,
            "size_um_zyx": roi_size_um,
            "roi_min_um": roi['min_um'],
            "roi_max_um": roi['max_um'],
            "axis_permutation": list(perm),
            "num_levels": num_levels,
        }
        with open(meta_path, 'w') as f:
            json.dump(pop_meta, f, indent=2)

        output_paths[pop_name]['metadata'] = str(meta_path)
        logger.info(f"  Metadata: {meta_path}")

    del volume_full
    if grayscale_full is not None:
        del grayscale_full

    logger.info(f"\n{'='*80}")
    logger.info("Extraction complete!")
    for pop_name, paths in output_paths.items():
        for kind, path in paths.items():
            logger.info(f"  {pop_name.upper()} {kind}: {path}")
    logger.info(f"{'='*80}\n")

    return output_paths


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Extract LM ROI sub-volumes as OME-Zarr with multi-resolution pyramids.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single ROI file
    python scripts/preparation/extract_lm_rois.py \\
        data/raw/rat/LM/sham_25_ipsi_population_rois.json

    # Batch processing
    python scripts/preparation/extract_lm_rois.py \\
        "data/raw/rat/LM/*_population_rois.json"

    # Custom output directory
    python scripts/preparation/extract_lm_rois.py \\
        data/raw/rat/LM/sham_25_ipsi_population_rois.json \\
        --output-dir data/processed/rat/LM

Inputs:
    *_population_rois.json from identify_lm_rois.py

Outputs per population (CC, CG):
    - <sample>_<pop>_myelin.zarr:     Segmentation (OME-Zarr, multi-resolution)
    - <sample>_<pop>_grayscale.zarr:  Grayscale image (OME-Zarr, multi-resolution)
    - <sample>_<pop>_myelin.json:     Per-ROI metadata (shape, voxel size, etc.)
        """
    )
    parser.add_argument(
        'input_files', type=str,
        help='*_population_rois.json file(s) (glob pattern supported)'
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
            extract_rois(
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
