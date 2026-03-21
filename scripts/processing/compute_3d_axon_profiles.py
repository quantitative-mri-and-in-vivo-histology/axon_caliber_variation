#!/usr/bin/env python3
"""
Compute fiber morphometry profiles by sampling perpendicular cross-sections along skeletons.

Uses the DeepACSON CSD approach (Abdollahzadeh et al., 2019) for skeleton
extraction and cross-section sampling.

Two backends available:
- 'fast' (default): optimized version (axonometry.deepacson.fast)
- 'original': verbatim DeepACSON code (axonometry.deepacson.original)

Expects OME-Zarr volumes (canonical format from preparation pipeline)
with isotropic voxels and Z axis = along-axon direction.

Usage:
  # Single Zarr volume
  python compute_3d_axon_profiles.py data/processed/rat/LM/sham_25_ipsi_cc_myelin.zarr \\
      data/processed/rat/LM/sham_25_ipsi_cc_axon_profiles.npz

  # With original DeepACSON backend
  python compute_3d_axon_profiles.py data/debug/sham_25_ipsi_cg_myelin_small.zarr \\
      /tmp/test_profiles.npz --backend original

  # Batch mode with glob pattern
  python compute_3d_axon_profiles.py "data/processed/rat/LM/*_myelin.zarr" \\
      data/processed/rat/LM/ --output-suffix _axon_profiles
"""

import argparse
import glob
import logging
import os
import traceback
from pathlib import Path

# Import from axonometry library
import sys
_root = Path(__file__).resolve().parent
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

from axonometry.axon_profiles import compute_axon_radius_profiles

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def batch_compute(matched_files, output_root, args):
    """Batch process multiple .zarr files matched by glob pattern."""
    if len(matched_files) == 1:
        common_root = matched_files[0].parent
    else:
        common_root = Path(os.path.commonpath([str(f.parent) for f in matched_files]))

    logger.info(f"\n{'='*80}")
    logger.info(f"Batch Processing Mode")
    logger.info(f"{'='*80}")
    logger.info(f"Found {len(matched_files)} files to process")
    logger.info(f"Common root: {common_root}")
    logger.info(f"Output root: {output_root}")
    logger.info(f"{'='*80}\n")

    successful = []
    failed = []

    for i, input_file in enumerate(matched_files, 1):
        stem = input_file.stem
        if input_file.suffix == ".zarr" or (input_file.is_dir() and ".zarr" in input_file.name):
            stem = input_file.with_suffix("").stem if "." in input_file.stem else input_file.stem
        output_file = output_root / f"{stem}{args.output_suffix}.npz"

        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {i}/{len(matched_files)}: {input_file.name}")
        logger.info(f"Output: {output_file.relative_to(output_root) if output_file.is_relative_to(output_root) else output_file}")
        logger.info(f"{'='*80}")

        try:
            compute_axon_radius_profiles(
                input_file,
                output_file,
                max_radius_um=args.max_axon_radius,
                step_size_um=args.step_size,
                max_axons=args.max_axons,
                axon_ids=args.axon_ids,
                backend=args.backend,
                n_jobs=args.n_jobs,
            )
            successful.append(input_file.name)
        except Exception as e:
            error_msg = str(e)
            failed.append((input_file.name, error_msg))
            logger.error(f"Failed to process {input_file.name}: {error_msg}")
            traceback.print_exc()
            continue

    logger.info(f"\n{'='*80}")
    logger.info("Batch Processing Complete")
    logger.info(f"{'='*80}")
    logger.info(f"Successfully processed: {len(successful)}/{len(matched_files)} files")

    if len(failed) > 0:
        logger.info(f"Failed: {len(failed)} files")
        for filename, error in failed:
            logger.info(f"  - {filename}: {error}")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute fiber morphometry profiles using DeepACSON CSD approach'
    )
    parser.add_argument('input', type=str,
                        help="Path to .zarr directory or glob pattern")
    parser.add_argument('output', type=Path,
                        help='Output .npz file (single file) OR output directory (batch mode)')
    parser.add_argument('--backend', type=str, default='fast',
                        choices=['fast', 'original'],
                        help="'fast' (optimized, default) or 'original' (verbatim DeepACSON)")
    parser.add_argument('--max-axon-radius', type=float, default=10.0,
                        help='Maximum expected axon radius in μm (sets cross-section grid size, default: 10.0)')
    parser.add_argument('--step-size', type=float, default=0.05,
                        help='Step size along skeleton in μm (default: 0.05)')
    parser.add_argument('--max-axons', type=int, default=0,
                        help='Maximum axons to process (0 = all)')
    parser.add_argument('--axon-ids', type=int, nargs='+', default=None,
                        help='Only process these specific axon label IDs')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel workers (default: 1, -1 = all cores)')
    parser.add_argument('--output-suffix', type=str, default='_axon_profiles',
                        help='Suffix to append to output filenames in batch mode')

    args = parser.parse_args()

    input_pattern = args.input
    output_path = args.output

    matched_files = sorted(glob.glob(input_pattern, recursive=True))

    if len(matched_files) == 0:
        parser.error(f"No files matched pattern: {input_pattern}")
    elif len(matched_files) == 1:
        input_path = Path(matched_files[0])

        if output_path.suffix != ".npz":
            stem = input_path.stem
            if input_path.suffix == ".zarr" or (input_path.is_dir() and ".zarr" in input_path.name):
                stem = input_path.with_suffix("").stem if "." in input_path.stem else input_path.stem
            output_path = output_path / f"{stem}{args.output_suffix}.npz"

        compute_axon_radius_profiles(
            input_path,
            output_path,
            max_radius_um=args.max_axon_radius,
            step_size_um=args.step_size,
            max_axons=args.max_axons,
            axon_ids=args.axon_ids,
            backend=args.backend,
            n_jobs=args.n_jobs,
        )
    else:
        matched_paths = [Path(f) for f in matched_files]
        if output_path.suffix == ".npz":
            parser.error("Output must be a directory in batch mode")
        batch_compute(matched_paths, output_path, args)
