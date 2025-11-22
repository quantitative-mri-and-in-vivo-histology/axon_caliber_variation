#!/usr/bin/env python3
"""
Batch extraction of CC and CG bundles from all .mat files in a directory.

Processes all .mat files and outputs Zarr files with CC/CG bundle volumes.
"""

import logging
import sys
from pathlib import Path

from extract_cc_cg_bundles import extract_cc_cg_bundles

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_sample_name(mat_file: Path) -> str:
    """Extract sample name from .mat filename.

    Example: LM_25_ipsi_myelinated_axons.mat -> LM_25_ipsi
    """
    name = mat_file.stem
    # Remove common suffixes
    for suffix in ['_myelinated_axons', '_myelinated', '_axons']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def batch_extract(input_dir: Path,
                  output_dir: Path,
                  downsample: int = 4,
                  voxel_size_um: float = 0.05,
                  orientation_std_threshold: float = 2.0,
                  spatial_weight: float = 0.5,
                  min_length_um: float = 50.0,
                  min_voxels: int = 10,
                  k_neighbors: int = 10,
                  max_neighbor_distance_um: float = 30.0,
                  n_levels: int = 5):
    """
    Process all .mat files in input directory.

    Args:
        input_dir: Directory containing .mat files
        output_dir: Directory for output Zarr files
        Other args: Passed to extract_cc_cg_bundles()
    """
    # Find all .mat files
    mat_files = sorted(input_dir.glob('*.mat'))

    if not mat_files:
        logger.error(f"No .mat files found in {input_dir}")
        return

    logger.info(f"Found {len(mat_files)} .mat files to process")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    successful = []
    failed = []

    for i, mat_file in enumerate(mat_files, 1):
        sample_name = extract_sample_name(mat_file)
        output_zarr = output_dir / f"{sample_name}_bundles.zarr"
        metadata_file = output_dir / f"{sample_name}_bundles.json"

        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {i}/{len(mat_files)}: {mat_file.name}")
        logger.info(f"Output: {output_zarr.name}")
        logger.info(f"{'='*80}")

        try:
            extract_cc_cg_bundles(
                mat_file=mat_file,
                output_zarr=output_zarr,
                metadata_file=metadata_file,
                downsample=downsample,
                voxel_size_um=voxel_size_um,
                orientation_std_threshold=orientation_std_threshold,
                spatial_weight=spatial_weight,
                min_length_um=min_length_um,
                min_voxels=min_voxels,
                k_neighbors=k_neighbors,
                max_neighbor_distance_um=max_neighbor_distance_um,
                n_levels=n_levels
            )
            successful.append(sample_name)
            logger.info(f"✓ Successfully processed {sample_name}")

        except Exception as e:
            failed.append((sample_name, str(e)))
            logger.error(f"✗ Failed to process {sample_name}: {e}")
            continue

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("BATCH EXTRACTION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Successful: {len(successful)}/{len(mat_files)}")

    if successful:
        logger.info(f"  Processed: {', '.join(successful)}")

    if failed:
        logger.info(f"Failed: {len(failed)}/{len(mat_files)}")
        for name, error in failed:
            logger.info(f"  {name}: {error}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Batch extract CC and CG bundles from all .mat files'
    )
    parser.add_argument('input_dir', type=Path,
                        help='Directory containing .mat files')
    parser.add_argument('output_dir', type=Path,
                        help='Output directory for Zarr files')
    parser.add_argument('--downsample', type=int, default=4,
                        help='Downsampling factor for identification (default: 4)')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                        help='Voxel size in μm (default: 0.05)')
    parser.add_argument('--orientation-std-threshold', type=float, default=2.0,
                        help='Std deviations for outlier rejection (default: 2.0)')
    parser.add_argument('--spatial-weight', type=float, default=0.5,
                        help='Weight for spatial position, 0-1 (default: 0.5)')
    parser.add_argument('--min-length', type=float, default=50.0,
                        help='Minimum axon length in μm (default: 50.0)')
    parser.add_argument('--min-voxels', type=int, default=10,
                        help='Minimum voxels per axon (default: 10)')
    parser.add_argument('--k-neighbors', type=int, default=10,
                        help='K for sparse axon filtering (default: 10)')
    parser.add_argument('--max-neighbor-distance', type=float, default=30.0,
                        help='Max distance to k-th neighbor in μm (default: 30.0)')
    parser.add_argument('--n-levels', type=int, default=5,
                        help='Pyramid levels (default: 5)')

    args = parser.parse_args()

    batch_extract(
        args.input_dir,
        args.output_dir,
        downsample=args.downsample,
        voxel_size_um=args.voxel_size,
        orientation_std_threshold=args.orientation_std_threshold,
        spatial_weight=args.spatial_weight,
        min_length_um=args.min_length,
        min_voxels=args.min_voxels,
        k_neighbors=args.k_neighbors,
        max_neighbor_distance_um=args.max_neighbor_distance,
        n_levels=args.n_levels
    )
