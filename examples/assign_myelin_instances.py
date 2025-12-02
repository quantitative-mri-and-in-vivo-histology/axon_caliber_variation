#!/usr/bin/env python3
"""
Assign binary myelin segmentation to axon instances.

Reads paired axon and myelin files and creates instance-labeled myelin
volumes using watershed algorithm.

Input files:
- Axon segmentation: *_myelinated_axons.mat (instance-labeled)
- Myelin segmentation: *_myelin.mat (binary)

Output:
- Instance-labeled myelin: *_instances.mat

Example single file:
    python assign_myelin_instances.py single \\
        data/raw/TBI_24_ipsi/HM_24_ipsi_myelinated_axons.mat \\
        data/raw/TBI_24_ipsi/HM_24_ipsi_myelin.mat \\
        data/processed/TBI_24_ipsi/HM_24_ipsi_instances.mat

Example batch processing:
    python assign_myelin_instances.py batch \\
        data/raw \\
        data/processed \\
        --pattern "*_myelinated_axons.mat"
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import scipy.io as sio

# Add parent directory to path for axonometry import
sys.path.insert(0, str(Path(__file__).parent.parent))

from axonometry.morphometry import assign_myelin_to_axons
from axonometry.io import load_mat_volume

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_myelin_file(axon_file: Path) -> Path:
    """
    Find corresponding myelin file for an axon file.

    Replaces '_myelinated_axons.mat' with '_myelin.mat' in the filename.

    Args:
        axon_file: Path to axon segmentation file

    Returns:
        Path to myelin segmentation file

    Raises:
        FileNotFoundError: If myelin file doesn't exist
        ValueError: If filename doesn't match expected pattern
    """
    if not axon_file.name.endswith('_myelinated_axons.mat'):
        raise ValueError(
            f"Axon file must end with '_myelinated_axons.mat': {axon_file.name}"
        )

    myelin_file = axon_file.parent / axon_file.name.replace(
        '_myelinated_axons.mat', '_myelin.mat'
    )

    if not myelin_file.exists():
        raise FileNotFoundError(f"Myelin file not found: {myelin_file}")

    return myelin_file


def construct_output_path(axon_file: Path, output_dir: Path) -> Path:
    """
    Construct output path for instance-labeled myelin file.

    Replaces '_myelinated_axons.mat' with '_instances.mat' and
    preserves directory structure under output_dir.

    Args:
        axon_file: Path to axon segmentation file
        output_dir: Output root directory

    Returns:
        Path to output file
    """
    # Preserve relative directory structure
    # e.g., data/raw/TBI_24_ipsi/HM_24_ipsi_myelinated_axons.mat
    # -> data/processed/TBI_24_ipsi/HM_24_ipsi_instances.mat

    parent_name = axon_file.parent.name  # e.g., "TBI_24_ipsi"
    output_subdir = output_dir / parent_name

    output_filename = axon_file.name.replace(
        '_myelinated_axons.mat', '_instances.mat'
    )

    return output_subdir / output_filename


def save_mat_volume(volume: np.ndarray, output_file: Path, variable_name: str = 'myelin_instances'):
    """
    Save labeled volume to .mat file.

    Uses scipy.io.savemat for compatibility with older MATLAB versions.

    Args:
        volume: Labeled volume to save
        output_file: Output .mat file path
        variable_name: Variable name in .mat file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving to {output_file.name}")
    logger.info(f"  Shape: {volume.shape}, dtype: {volume.dtype}")
    logger.info(f"  Variable name: '{variable_name}'")

    sio.savemat(
        str(output_file),
        {variable_name: volume},
        do_compression=True
    )

    file_size_mb = output_file.stat().st_size / (1024 ** 2)
    logger.info(f"  File size: {file_size_mb:.1f} MB")


def process_single_file(
    axon_file: Path,
    myelin_file: Path,
    output_file: Path,
    max_distance: float = None,
    connectivity: int = 1
):
    """
    Process a single axon-myelin pair.

    Args:
        axon_file: Path to axon segmentation file
        myelin_file: Path to myelin segmentation file
        output_file: Path to output file
        max_distance: Optional max distance in voxels for assignment
        connectivity: Connectivity for myelin component labeling
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {axon_file.name}")
    logger.info(f"{'='*80}\n")

    # Load volumes
    logger.info("Loading axon volume...")
    axon_volume = load_mat_volume(axon_file)

    logger.info("Loading myelin volume...")
    myelin_volume = load_mat_volume(myelin_file)

    # Assign myelin to axons
    myelin_instances = assign_myelin_to_axons(
        axon_volume,
        myelin_volume,
        max_distance=max_distance,
        connectivity=connectivity
    )

    # Save output
    save_mat_volume(myelin_instances, output_file)

    logger.info(f"\n{'='*80}")
    logger.info(f"Completed: {output_file.name}")
    logger.info(f"{'='*80}\n")


def process_batch(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*_myelinated_axons.mat",
    max_distance: float = None,
    connectivity: int = 1
):
    """
    Process all axon files in a directory tree.

    Args:
        input_dir: Root input directory
        output_dir: Root output directory
        pattern: Glob pattern for axon files
        max_distance: Optional max distance in voxels for assignment
        connectivity: Connectivity for myelin component labeling
    """
    # Find all axon files
    axon_files = sorted(input_dir.rglob(pattern))

    if not axon_files:
        logger.error(f"No files matching pattern '{pattern}' found in {input_dir}")
        return

    logger.info(f"Found {len(axon_files)} axon file(s) to process")

    # Process each file
    for i, axon_file in enumerate(axon_files, 1):
        logger.info(f"\n[{i}/{len(axon_files)}] Processing {axon_file.relative_to(input_dir)}")

        try:
            # Find corresponding myelin file
            myelin_file = find_myelin_file(axon_file)

            # Construct output path
            output_file = construct_output_path(axon_file, output_dir)

            # Check if output already exists
            if output_file.exists():
                logger.warning(f"Output file already exists, skipping: {output_file}")
                continue

            # Process
            process_single_file(
                axon_file,
                myelin_file,
                output_file,
                max_distance=max_distance,
                connectivity=connectivity
            )

        except Exception as e:
            logger.error(f"Error processing {axon_file.name}: {e}", exc_info=True)
            continue

    logger.info(f"\nBatch processing complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Assign binary myelin segmentation to axon instances',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Create subparsers for single vs batch mode
    subparsers = parser.add_subparsers(dest='mode', help='Processing mode')

    # Single file mode
    single_parser = subparsers.add_parser('single', help='Process a single file')
    single_parser.add_argument(
        'axon_file',
        type=Path,
        help='Input axon file (.mat)'
    )
    single_parser.add_argument(
        'myelin_file',
        type=Path,
        help='Input myelin file (.mat)'
    )
    single_parser.add_argument(
        'output_file',
        type=Path,
        help='Output file (.mat)'
    )

    # Batch mode
    batch_parser = subparsers.add_parser('batch', help='Process multiple files')
    batch_parser.add_argument(
        'input_dir',
        type=Path,
        help='Input directory containing axon files'
    )
    batch_parser.add_argument(
        'output_dir',
        type=Path,
        help='Output directory'
    )
    batch_parser.add_argument(
        '--pattern',
        type=str,
        default='*_myelinated_axons.mat',
        help='Glob pattern for axon files (default: *_myelinated_axons.mat)'
    )

    # Common arguments for both modes
    for p in [single_parser, batch_parser]:
        p.add_argument(
            '--max-distance',
            type=float,
            default=None,
            help='Maximum distance in voxels for myelin assignment (default: no limit)'
        )
        p.add_argument(
            '--connectivity',
            type=int,
            choices=[1, 2],
            default=1,
            help='Connectivity for myelin component labeling (1=faces, 2=faces+edges, default: 1)'
        )

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    # Execute appropriate mode
    if args.mode == 'single':
        if not args.axon_file.is_file():
            parser.error(f"Axon file not found: {args.axon_file}")
        if not args.myelin_file.is_file():
            parser.error(f"Myelin file not found: {args.myelin_file}")

        process_single_file(
            args.axon_file,
            args.myelin_file,
            args.output_file,
            max_distance=args.max_distance,
            connectivity=args.connectivity
        )

    elif args.mode == 'batch':
        if not args.input_dir.is_dir():
            parser.error(f"Input directory not found: {args.input_dir}")

        process_batch(
            args.input_dir,
            args.output_dir,
            pattern=args.pattern,
            max_distance=args.max_distance,
            connectivity=args.connectivity
        )


if __name__ == '__main__':
    main()
