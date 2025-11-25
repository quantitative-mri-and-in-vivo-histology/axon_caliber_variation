#!/usr/bin/env python3
"""
Batch computation of axon radius profiles using Kimimaro skeletonization.

Processes all *_myelinated_axons.mat files and computes radius profiles
using Kimimaro's batch skeletonization and cross-sectional area computation.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

from compute_axon_radius_profiles_kimimaro import compute_radius_profiles, parse_voxel_size

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_sample_name(mat_file: Path) -> str:
    """Extract sample name from MAT filename.

    Example: LM_25_ipsi_myelinated_axons.mat -> LM_25_ipsi
    """
    name = mat_file.stem
    # Remove common suffixes
    for suffix in ['_myelinated_axons', '_axons']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def batch_compute_profiles(input_dir: Path,
                           output_dir: Path,
                           voxel_size_um=0.05,
                           downsample: int = 1,
                           smoothing_window: int = 5,
                           parallel: int = 0,
                           max_axons: int = 0):
    """
    Compute radius profiles for all *_myelinated_axons.mat files in input directory
    using Kimimaro batch skeletonization.

    Args:
        input_dir: Directory containing *_myelinated_axons.mat files
        output_dir: Directory for output .npz files
        voxel_size_um: Voxel size in micrometers. Can be:
                      - float: isotropic voxel size (e.g., 0.05)
                      - tuple: anisotropic (vx, vy, vz) (e.g., (0.015, 0.015, 0.05))
        downsample: Downsampling factor applied before processing (default: 1)
        smoothing_window: Rolling average window for plane normals (default: 5)
        parallel: Number of parallel workers for Kimimaro (0 = all CPUs, default: 0)
        max_axons: Maximum number of axons to process (0 = all)

    Returns:
        Dict mapping sample_name -> summary statistics
    """
    # Find all *_myelinated_axons.mat files
    mat_files = sorted(input_dir.glob('*_myelinated_axons.mat'))

    if not mat_files:
        logger.error(f"No *_myelinated_axons.mat files found in {input_dir}")
        return {}

    logger.info(f"Found {len(mat_files)} MAT files to process")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    all_results = {}
    successful = []
    failed = []

    for i, mat_file in enumerate(mat_files, 1):
        sample_name = extract_sample_name(mat_file)
        output_file = output_dir / f'{sample_name}_radius_profiles_kimimaro.npz'

        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {i}/{len(mat_files)}: {mat_file.name}")
        logger.info(f"Output: {output_file}")
        logger.info(f"{'='*80}")

        try:
            compute_radius_profiles(
                mat_file,
                output_file,
                voxel_size_um=voxel_size_um,
                downsample=downsample,
                smoothing_window=smoothing_window,
                parallel=parallel,
                max_axons=max_axons
            )

            # Load results to get summary stats
            data = np.load(output_file, allow_pickle=True)
            all_radii = data['all_radii_um']
            n_axons = len(data['labels'])
            n_samples = len(all_radii)

            # Compute effective radius
            r2 = np.mean(all_radii ** 2)
            r6 = np.mean(all_radii ** 6)
            r_eff = (r6 / r2) ** 0.25 if r2 > 0 else 0.0

            all_results[sample_name] = {
                'n_axons': int(n_axons),
                'n_samples': int(n_samples),
                'r_eff': float(r_eff),
                'mean_radius': float(np.mean(all_radii)),
                'std_radius': float(np.std(all_radii)),
                'median_radius': float(np.median(all_radii))
            }

            successful.append(sample_name)
            logger.info(f"Successfully processed {sample_name}: {n_axons} axons, r_eff={r_eff:.3f} μm")

        except Exception as e:
            failed.append((sample_name, str(e)))
            logger.error(f"Failed to process {sample_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Successful: {len(successful)}/{len(mat_files)}")

    if successful:
        logger.info(f"  Processed: {', '.join(successful)}")

    if failed:
        logger.info(f"Failed: {len(failed)}/{len(mat_files)}")
        for name, error in failed:
            logger.info(f"  {name}: {error}")

    # Print summary of effective radii
    if all_results:
        logger.info(f"\n{'='*80}")
        logger.info("EFFECTIVE RADIUS SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"{'Sample':<25} {'Axons':<8} {'Samples':<10} {'r_eff (μm)':<12} {'Mean (μm)':<12}")
        logger.info(f"{'-'*67}")

        for sample_name in sorted(all_results.keys()):
            result = all_results[sample_name]
            logger.info(f"{sample_name:<25} {result['n_axons']:<8} {result['n_samples']:<10} "
                       f"{result['r_eff']:<12.3f} {result['mean_radius']:<12.3f}")

        # Save summary to JSON
        summary_file = output_dir / 'radius_profiles_kimimaro_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nSaved summary to {summary_file}")

    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Batch compute axon radius profiles using Kimimaro skeletonization'
    )
    parser.add_argument('input_dir', type=Path,
                        help='Directory containing *_myelinated_axons.mat files')
    parser.add_argument('output_dir', type=Path,
                        help='Output directory for .npz files and summary')
    parser.add_argument('--voxel-size', nargs='+', default=['0.05'],
                        help='Voxel size in micrometers. Can be: '
                             '(1) single float for isotropic (e.g., 0.05) '
                             'or (2) three floats for anisotropic (e.g., 0.015 0.015 0.05)')
    parser.add_argument('--downsample', type=int, default=1,
                        help='Downsampling factor applied before processing (default: 1)')
    parser.add_argument('--smoothing-window', type=int, default=5,
                        help='Rolling average window for plane normals (default: 5)')
    parser.add_argument('--parallel', type=int, default=0,
                        help='Parallel workers for Kimimaro (0 = all CPUs, default: 0)')
    parser.add_argument('--max-axons', type=int, default=0,
                        help='Maximum axons to process per volume (0 = all, default: 0)')

    args = parser.parse_args()

    # Parse voxel size (scalar or tuple)
    voxel_size_um = parse_voxel_size(args.voxel_size)

    batch_compute_profiles(
        args.input_dir,
        args.output_dir,
        voxel_size_um=voxel_size_um,
        downsample=args.downsample,
        smoothing_window=args.smoothing_window,
        parallel=args.parallel,
        max_axons=args.max_axons
    )
