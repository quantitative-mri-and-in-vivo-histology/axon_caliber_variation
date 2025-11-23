#!/usr/bin/env python3
"""
Batch plotting of largest axon radius profiles from spatial subvolumes (CC/CG).

Processes all *_spatial.zarr files and generates largest axon profile plots
for each population.
"""

import logging
from pathlib import Path

from plot_largest_axon_profiles import analyze_largest_axons

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_sample_name(zarr_file: Path) -> str:
    """Extract sample name from Zarr filename.

    Example: LM_25_ipsi_spatial.zarr -> LM_25_ipsi
    """
    name = zarr_file.stem
    # Remove common suffixes
    for suffix in ['_spatial', '_bundles']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def batch_plot_largest_axons(input_dir: Path,
                              output_dir: Path,
                              n_largest: int = 10,
                              voxel_size_um: float = 0.05,
                              use_minor_axis: bool = False,
                              max_ellipse_ratio: float = 0.0,
                              load_in_memory: bool = False,
                              n_jobs: int = -1):
    """
    Plot largest axon profiles for all *_spatial.zarr files in input directory.

    Args:
        input_dir: Directory containing *_spatial.zarr files
        output_dir: Directory for output plots
        n_largest: Number of largest axons to plot
        voxel_size_um: Voxel size in micrometers
        use_minor_axis: If True, use ellipse minor axis instead of circular-equivalent radius
        max_ellipse_ratio: Max ratio of ellipse area to voxel area (0 = no filter)
        load_in_memory: If True, load entire volume into memory first
        n_jobs: Number of parallel jobs (-1 = use all CPUs)
    """
    # Find all *_spatial.zarr files
    zarr_files = sorted(input_dir.glob('*_spatial.zarr'))

    if not zarr_files:
        logger.error(f"No *_spatial.zarr files found in {input_dir}")
        return

    logger.info(f"Found {len(zarr_files)} spatial Zarr files to process")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    successful = []
    failed = []

    for i, zarr_file in enumerate(zarr_files, 1):
        sample_name = extract_sample_name(zarr_file)
        sample_output_dir = output_dir / sample_name

        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {i}/{len(zarr_files)}: {zarr_file.name}")
        logger.info(f"Output: {sample_output_dir}")
        logger.info(f"{'='*80}")

        try:
            analyze_largest_axons(
                zarr_file,
                sample_output_dir,
                n_largest=n_largest,
                voxel_size_um=voxel_size_um,
                use_minor_axis=use_minor_axis,
                max_ellipse_ratio=max_ellipse_ratio,
                load_in_memory=load_in_memory,
                n_jobs=n_jobs
            )
            successful.append(sample_name)
            logger.info(f"Successfully processed {sample_name}")

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
    logger.info(f"Successful: {len(successful)}/{len(zarr_files)}")

    if successful:
        logger.info(f"  Processed: {', '.join(successful)}")

    if failed:
        logger.info(f"Failed: {len(failed)}/{len(zarr_files)}")
        for name, error in failed:
            logger.info(f"  {name}: {error}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Batch plot largest axon profiles from spatial subvolumes (CC/CG)'
    )
    parser.add_argument('input_dir', type=Path,
                        help='Directory containing *_spatial.zarr files')
    parser.add_argument('output_dir', type=Path,
                        help='Output directory for plots')
    parser.add_argument('--n-largest', type=int, default=10,
                        help='Number of largest axons to plot (default: 10)')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                        help='Voxel size in micrometers (default: 0.05)')
    parser.add_argument('--use-minor-axis', action='store_true',
                        help='Use ellipse minor axis instead of circular-equivalent radius')
    parser.add_argument('--max-ellipse-ratio', type=float, default=0.0,
                        help='Max ratio of ellipse area to voxel area to filter sparse regions (0 = no filter)')
    parser.add_argument('--load-in-memory', action='store_true',
                        help='Load entire volume into memory first (faster but uses more RAM)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (default: -1 = use all CPUs)')

    args = parser.parse_args()

    batch_plot_largest_axons(
        args.input_dir,
        args.output_dir,
        n_largest=args.n_largest,
        voxel_size_um=args.voxel_size,
        use_minor_axis=args.use_minor_axis,
        max_ellipse_ratio=args.max_ellipse_ratio,
        load_in_memory=args.load_in_memory,
        n_jobs=args.n_jobs
    )
