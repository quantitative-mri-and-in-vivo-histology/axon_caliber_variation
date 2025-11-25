#!/usr/bin/env python3
"""
Batch analysis of effective MRI-visible radius from spatial subvolumes (CC/CG).

Processes all *_spatial.zarr files created by batch_segment_spatial.py and
generates effective radius profiles for each population.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from analyze_effective_radius import analyze_population, parse_voxel_size_arg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_sample_name(file_path: Path) -> str:
    """Extract sample name from Zarr or MAT filename.

    Examples:
        LM_25_ipsi_spatial.zarr -> LM_25_ipsi
        LM_25_ipsi_myelinated_axons.mat -> LM_25_ipsi
    """
    name = file_path.stem
    # Remove common suffixes
    for suffix in ['_spatial', '_bundles', '_myelinated_axons', '_axons']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def batch_analyze(input_dir: Path,
                  output_dir: Path,
                  voxel_size_um: Union[float, Tuple[float, float, float]] = 0.2,
                  n_jobs: int = -1,
                  min_axon_fraction: float = 0.75,
                  load_in_memory: bool = False,
                  use_minor_axis: bool = False,
                  max_ellipse_ratio: float = 0.0,
                  slice_axis: int = 2) -> Dict[str, Dict[str, Tuple[float, np.ndarray]]]:
    """
    Analyze effective radius for all *_spatial.zarr and *_myelinated_axons.mat files in input directory.

    Supports both:
    - Zarr spatial subvolumes (CC/CG segmented)
    - MATLAB .mat files (v5.0 and v7.3 HDF5 formats)

    Args:
        input_dir: Directory containing *_spatial.zarr or *_myelinated_axons.mat files
        output_dir: Directory for output plots
        voxel_size_um: Voxel size in micrometers. Can be:
                      - float: isotropic voxel size (e.g., 0.05)
                      - tuple: anisotropic (vx, vy, vz) (e.g., (0.015, 0.015, 0.05))
        n_jobs: Number of parallel jobs (-1 = use all CPUs)
        min_axon_fraction: Minimum fraction of max axon count to include slice
        load_in_memory: If True, load entire volume into memory first
        use_minor_axis: If True, use ellipse minor axis instead of circular-equivalent radius
        max_ellipse_ratio: Max ratio of ellipse area to voxel area (0 = no filter, 2.0 recommended)
        slice_axis: Axis along which to slice for legacy formats (0=Z, 1=Y, 2=X), default=2

    Returns:
        Dict mapping sample_name -> {population_name -> (global_r_eff, r_eff_per_slice)}
    """
    # Find all *_spatial.zarr directories and *_myelinated_axons.mat files
    zarr_files = sorted(input_dir.glob('*_spatial.zarr'))
    mat_files = sorted(input_dir.glob('*_myelinated_axons.mat'))
    all_files = zarr_files + mat_files

    if not all_files:
        logger.error(f"No *_spatial.zarr or *_myelinated_axons.mat files found in {input_dir}")
        return {}

    logger.info(f"Found {len(zarr_files)} spatial Zarr files and {len(mat_files)} MAT files to analyze")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    all_results = {}
    successful = []
    failed = []

    for i, input_file in enumerate(all_files, 1):
        sample_name = extract_sample_name(input_file)
        sample_output_dir = output_dir / sample_name
        file_type = "Zarr" if input_file.suffix == '.zarr' else "MAT"

        logger.info(f"\n{'='*80}")
        logger.info(f"Analyzing {i}/{len(all_files)}: {input_file.name} ({file_type})")
        logger.info(f"Output: {sample_output_dir}")
        logger.info(f"{'='*80}")

        try:
            results = analyze_population(
                input_file,
                sample_output_dir,
                voxel_size_um=voxel_size_um,
                n_jobs=n_jobs,
                min_axon_fraction=min_axon_fraction,
                load_in_memory=load_in_memory,
                use_minor_axis=use_minor_axis,
                max_ellipse_ratio=max_ellipse_ratio,
                slice_axis=slice_axis
            )
            all_results[sample_name] = results
            successful.append(sample_name)
            logger.info(f"Successfully analyzed {sample_name} ({file_type})")

        except Exception as e:
            failed.append((sample_name, str(e)))
            logger.error(f"Failed to analyze {sample_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("BATCH ANALYSIS COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Successful: {len(successful)}/{len(all_files)}")

    if successful:
        logger.info(f"  Analyzed: {', '.join(successful)}")

    if failed:
        logger.info(f"Failed: {len(failed)}/{len(all_files)}")
        for name, error in failed:
            logger.info(f"  {name}: {error}")

    # Print summary of effective radii
    if all_results:
        logger.info(f"\n{'='*80}")
        logger.info("EFFECTIVE RADIUS SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"{'Sample':<20} {'Population':<10} {'r_eff (um)':<12}")
        logger.info(f"{'-'*42}")

        for sample_name in sorted(all_results.keys()):
            results = all_results[sample_name]
            for pop_name in sorted(results.keys()):
                r_eff_global, _ = results[pop_name]
                logger.info(f"{sample_name:<20} {pop_name:<10} {r_eff_global:<12.3f}")

        # Save summary to JSON
        summary_file = output_dir / 'effective_radius_summary.json'
        summary = {}
        for sample_name, results in all_results.items():
            summary[sample_name] = {
                pop_name: {
                    'r_eff_global': float(r_eff_global),
                    'n_slices': len(r_eff_per_slice),
                    'r_eff_mean': float(np.mean(r_eff_per_slice[r_eff_per_slice > 0])) if np.any(r_eff_per_slice > 0) else 0.0,
                    'r_eff_std': float(np.std(r_eff_per_slice[r_eff_per_slice > 0])) if np.any(r_eff_per_slice > 0) else 0.0
                }
                for pop_name, (r_eff_global, r_eff_per_slice) in results.items()
            }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nSaved summary to {summary_file}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch analyze effective radius from spatial subvolumes or MAT files'
    )
    parser.add_argument('input_dir', type=Path,
                        help='Directory containing *_spatial.zarr or *_myelinated_axons.mat files')
    parser.add_argument('output_dir', type=Path,
                        help='Output directory for plots and summary')
    parser.add_argument('--voxel-size', type=parse_voxel_size_arg, default=0.2,
                        help='Voxel size in micrometers: single value for isotropic (e.g., 0.05) '
                             'or vx,vy,vz for anisotropic (e.g., 0.015,0.015,0.05). Default: 0.2')
    parser.add_argument('--slice-axis', type=int, default=2, choices=[0, 1, 2],
                        help='Axis to slice along for legacy formats: 0=Z, 1=Y, 2=X (default: 2). '
                             'Ignored for Zarr files with CC/CG subgroups which use stored metadata.')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (default: -1 = use all CPUs)')
    parser.add_argument('--min-axon-fraction', type=float, default=0.75,
                        help='Minimum fraction of max axon count to include slice (default: 0.75)')
    parser.add_argument('--load-in-memory', action='store_true',
                        help='Load entire volume into memory first (faster but uses more RAM)')
    parser.add_argument('--use-minor-axis', action='store_true',
                        help='Use ellipse minor axis instead of circular-equivalent radius')
    parser.add_argument('--max-ellipse-ratio', type=float, default=0.0,
                        help='Max ratio of ellipse area to voxel area to filter sparse regions (0 = no filter, 2.0 recommended)')

    args = parser.parse_args()

    batch_analyze(
        args.input_dir,
        args.output_dir,
        voxel_size_um=args.voxel_size,
        n_jobs=args.n_jobs,
        min_axon_fraction=args.min_axon_fraction,
        load_in_memory=args.load_in_memory,
        use_minor_axis=args.use_minor_axis,
        max_ellipse_ratio=args.max_ellipse_ratio,
        slice_axis=args.slice_axis
    )
