#!/usr/bin/env python3
"""
Compute fiber morphometry profiles by sampling perpendicular cross-sections along skeletons.

Uses the DeepACSON CSD approach (Abdollahzadeh et al., 2019) for skeleton
extraction and cross-section sampling via axonometry.deepacson.

Two backends available:
- 'fast' (default): optimized version (axonometry.deepacson_fast)
- 'original': verbatim DeepACSON code (axonometry.deepacson)

Supports both OME-Zarr volumes (canonical format from preparation pipeline)
and legacy .mat files.

Usage:
  # Single Zarr volume
  python compute_3d_axon_profiles.py data/processed/rat/LM/sham_25_ipsi_cc_myelin.zarr \
      data/processed/rat/LM/sham_25_ipsi_cc_axon_profiles.npz

  # With original DeepACSON backend
  python compute_3d_axon_profiles.py data/debug/sham_25_ipsi_cg_myelin_small.zarr \
      /tmp/test_profiles.npz --backend original

  # Batch mode with glob pattern
  python compute_3d_axon_profiles.py "data/processed/rat/LM/*_myelin.zarr" \
      data/processed/rat/LM/ --output-suffix _axon_profiles
"""

import argparse
import glob
import logging
import os
import traceback
from pathlib import Path
from typing import Tuple, Union, Optional, List

import numpy as np
from scipy.ndimage import find_objects
from tqdm import tqdm

# Import from axonometry library
import sys
_root = Path(__file__).resolve().parent
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

from axonometry import (
    load_volume_with_metadata,
    resample_to_isotropic,
)
from axonometry.axon_profiles import (
    axon_radius_profile,
    axon_radius_profile_fast,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ===========================================================================
# Volume loading
# ===========================================================================

def load_zarr_volume(zarr_path: Path) -> Tuple[np.ndarray, float]:
    """Load level-0 volume and voxel size from an OME-Zarr store."""
    import zarr

    store = zarr.open_group(str(zarr_path), mode="r")
    volume = np.asarray(store["0"])

    multiscales = store.attrs["multiscales"]
    scale = multiscales[0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    voxel_size_z = scale[0]

    if not np.allclose(scale, scale[0]):
        logger.warning(
            f"Zarr voxel size is not isotropic: {scale}. Using Z voxel size ({voxel_size_z})."
        )

    return volume, float(voxel_size_z)


# ===========================================================================
# Bounding boxes
# ===========================================================================

def compute_bounding_boxes(volume: np.ndarray) -> dict:
    """Compute bounding boxes for all labels via find_objects (single pass)."""
    logger.info("Computing bounding boxes...")
    slices = find_objects(volume)

    bboxes = {}
    for label_idx, bbox_slices in enumerate(slices):
        if bbox_slices is None:
            continue
        lbl = label_idx + 1
        min_coords = np.array([s.start for s in bbox_slices])
        max_coords = np.array([s.stop for s in bbox_slices])
        bboxes[lbl] = (min_coords, max_coords)

    logger.info(f"Computed bounding boxes for {len(bboxes)} labels")
    return bboxes


# ===========================================================================
# Per-axon processing
# ===========================================================================

def process_single_axon(volume, axon_label, bbox, voxel_size_um,
                        g_radius, g_res, step_voxels, backend):
    """
    Process a single axon: crop, extract profile, convert to μm.

    Returns dict with results or None if failed.
    """
    try:
        min_coords, max_coords = bbox
        vol_shape = np.array(volume.shape)

        pad = g_radius + 5
        min_padded = np.maximum(min_coords - pad, 0)
        max_padded = np.minimum(max_coords + pad, vol_shape)

        subvol = volume[min_padded[0]:max_padded[0],
                        min_padded[1]:max_padded[1],
                        min_padded[2]:max_padded[2]]
        binary = (subvol == axon_label).astype(np.float64)

        n_voxels = np.count_nonzero(binary)
        if n_voxels < 100:
            return None

        # Call the appropriate backend
        if backend == 'original':
            result = axon_radius_profile(
                binary, g_radius, g_res=g_res, step_voxels=step_voxels
            )
        else:
            result = axon_radius_profile_fast(
                binary, g_radius, g_res=g_res, step_voxels=step_voxels
            )

        if result is None:
            return None

        # Convert from voxels to μm
        radii_um = result['radii_voxels'] * voxel_size_um
        skel_um = (result['skeleton_points'] + min_padded) * voxel_size_um
        length_um = result['length_voxels'] * voxel_size_um

        return {
            'label': axon_label,
            'radii_um': radii_um,
            'skeleton_um': skel_um,
            'n_points': len(radii_um),
            'n_segments': result['n_segments'],
            'mean_radius_um': np.mean(radii_um),
            'std_radius_um': np.std(radii_um),
            'length_um': length_um,
        }

    except Exception as e:
        logger.debug(f"Axon {axon_label}: failed - {e}")
        return None


# ===========================================================================
# Main orchestration
# ===========================================================================

def compute_fiber_profiles(input_path: Path,
                           output_file: Path,
                           voxel_size_um: Optional[Union[float, Tuple[float, float, float]]] = None,
                           max_radius_um: float = 5.0,
                           step_size_um: float = 0.05,
                           max_axons: int = 0,
                           anisotropy_mode: str = 'simple',
                           backend: str = 'fast'):
    """
    Compute morphometry profiles for all fibers in a labeled volume.

    Args:
        input_path: Path to .zarr directory or .mat file with labeled axons
        output_file: Path to save results (.npz)
        voxel_size_um: Voxel size in micrometers (scalar or (vz, vy, vx) tuple).
                       If None, auto-detected from Zarr metadata or companion JSON.
        max_radius_um: Maximum expected axon radius in micrometers (sets grid size)
        step_size_um: Step size along skeleton in micrometers
        max_axons: Maximum number of axons to process (0 = all)
        anisotropy_mode: 'simple' (resample to isotropic) or 'none' (use geometric mean)
        backend: 'fast' (optimized) or 'original' (verbatim DeepACSON)
    """
    # Load volume
    suffix = input_path.suffix.lower()
    if suffix == ".zarr" or input_path.is_dir():
        logger.info(f"Loading Zarr volume: {input_path.name}")
        volume, iso_voxel_size = load_zarr_volume(input_path)
        voxel_size_tuple = (iso_voxel_size, iso_voxel_size, iso_voxel_size)
    elif suffix == ".mat":
        logger.info(f"Loading .mat volume: {input_path.name}")
        volume, voxel_size_tuple, _ = load_volume_with_metadata(input_path, voxel_size_um)
        if anisotropy_mode == 'simple':
            volume, iso_voxel_size = resample_to_isotropic(volume, voxel_size_tuple)
            voxel_size_tuple = (iso_voxel_size, iso_voxel_size, iso_voxel_size)
        elif anisotropy_mode != 'none':
            raise ValueError(f"Unknown anisotropy_mode: {anisotropy_mode}")
    else:
        raise ValueError(f"Unsupported input format: {suffix}. Use .zarr or .mat")

    vz, vy, vx = voxel_size_tuple
    voxel_size = (vz * vy * vx) ** (1/3)

    # Convert max_radius from μm to voxels for grid sizing
    g_radius = int(np.ceil(max_radius_um / voxel_size))
    g_res = 0.25

    logger.info(f"Volume shape: {volume.shape}, voxel: {voxel_size:.4f} μm")
    logger.info(f"Backend: {backend}")
    logger.info(f"Max axon radius: {max_radius_um:.1f} μm = {g_radius} voxels")

    # Compute bounding boxes
    bboxes = compute_bounding_boxes(volume)
    axon_labels = sorted(bboxes.keys())

    logger.info(f"Found {len(axon_labels)} axons")

    if max_axons > 0:
        axon_labels = axon_labels[:max_axons]
        logger.info(f"Processing first {max_axons} axons")

    step_voxels = step_size_um / voxel_size if step_size_um is not None else None

    logger.info(f"Parameters: g_radius={g_radius} voxels, g_res={g_res}, "
                f"step={step_size_um} μm = {step_voxels:.1f} voxels "
                f"(stride ~{max(1, round(step_voxels / 0.1))})")

    # Process axons sequentially
    results = []
    for axon_label in tqdm(axon_labels, desc=f"Processing axons ({backend})"):
        result = process_single_axon(
            volume, axon_label, bboxes[axon_label],
            voxel_size, g_radius, g_res, step_voxels, backend
        )
        if result is not None:
            results.append(result)

    logger.info(f"Successfully processed {len(results)}/{len(axon_labels)} axons")

    if len(results) == 0:
        logger.error("No axons were successfully processed!")
        return

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)

    labels = np.array([r['label'] for r in results])
    n_points = np.array([r['n_points'] for r in results])
    mean_radii = np.array([r['mean_radius_um'] for r in results])
    std_radii = np.array([r['std_radius_um'] for r in results])
    lengths = np.array([r['length_um'] for r in results])
    radii_profiles = np.array([r['radii_um'] for r in results], dtype=object)
    skeleton_coords = np.array([r['skeleton_um'] for r in results], dtype=object)

    all_radii = np.concatenate([r['radii_um'] for r in results])

    np.savez(
        output_file,
        labels=labels,
        n_points=n_points,
        mean_radii_um=mean_radii,
        std_radii_um=std_radii,
        lengths_um=lengths,
        radii_profiles_um=radii_profiles,
        skeleton_coords_um=skeleton_coords,
        all_radii_um=all_radii,
        voxel_size_um=voxel_size,
        max_radius_um=max_radius_um,
        g_radius_voxels=g_radius,
        step_size_um=step_size_um,
        source_file=str(input_path),
        method=f'deepacson_csd_{backend}',
    )

    logger.info(f"Saved results to {output_file}")

    # Summary statistics
    r_eff = (np.mean(all_radii**6) / np.mean(all_radii**2)) ** 0.25
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total axons processed: {len(results)}")
    logger.info(f"  Total radius samples: {len(all_radii)}")
    logger.info(f"  Mean axon length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f} μm")
    logger.info(f"  Mean points per axon: {np.mean(n_points):.1f}")
    logger.info(f"  r̄: {np.mean(all_radii):.4f} μm")
    logger.info(f"  r_eff: {r_eff:.4f} μm")


def batch_compute_fiber_profiles(
    matched_files: List[Path],
    output_root: Path,
    voxel_size_um: Optional[Union[float, Tuple[float, float, float]]],
    max_radius_um: float,
    step_size_um: float,
    max_axons: int,
    anisotropy_mode: str,
    backend: str,
    output_suffix: str = '_axon_profiles',
):
    """Batch process multiple .zarr/.mat files matched by glob pattern."""
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
        output_file = output_root / f"{stem}{output_suffix}.npz"

        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {i}/{len(matched_files)}: {input_file.name}")
        logger.info(f"Output: {output_file.relative_to(output_root) if output_file.is_relative_to(output_root) else output_file}")
        logger.info(f"{'='*80}")

        try:
            compute_fiber_profiles(
                input_file,
                output_file,
                voxel_size_um=voxel_size_um,
                max_radius_um=max_radius_um,
                step_size_um=step_size_um,
                max_axons=max_axons,
                anisotropy_mode=anisotropy_mode,
                backend=backend,
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


def parse_voxel_size_arg(value: str) -> Union[float, Tuple[float, float, float]]:
    """Parse voxel size CLI argument."""
    if ',' in value:
        parts = value.split(',')
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(f"Expected 1 or 3 values, got {len(parts)}")
        return tuple(float(p.strip()) for p in parts)
    return float(value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute fiber morphometry profiles using DeepACSON CSD approach'
    )
    parser.add_argument('input', type=str,
                        help="Path to .zarr directory, .mat file, or glob pattern")
    parser.add_argument('output', type=Path,
                        help='Output .npz file (single file) OR output directory (batch mode)')
    parser.add_argument('--backend', type=str, default='fast',
                        choices=['fast', 'original'],
                        help="'fast' (optimized, default) or 'original' (verbatim DeepACSON)")
    parser.add_argument('--voxel-size', type=parse_voxel_size_arg, default=None,
                        help='Voxel size in μm: single value (isotropic) or vz,vy,vx')
    parser.add_argument('--max-radius', type=float, default=5.0,
                        help='Maximum expected axon radius in μm (sets cross-section grid size, default: 5.0)')
    parser.add_argument('--step-size', type=float, default=0.05,
                        help='Step size along skeleton in μm (default: 0.05)')
    parser.add_argument('--max-axons', type=int, default=0,
                        help='Maximum axons to process (0 = all)')
    parser.add_argument('--anisotropy-mode', type=str, default='simple',
                        choices=['simple', 'none'],
                        help="'simple' resamples to isotropic, 'none' uses geometric mean")
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

        compute_fiber_profiles(
            input_path,
            output_path,
            voxel_size_um=args.voxel_size,
            max_radius_um=args.max_radius,
            step_size_um=args.step_size,
            max_axons=args.max_axons,
            anisotropy_mode=args.anisotropy_mode,
            backend=args.backend,
        )
    else:
        matched_paths = [Path(f) for f in matched_files]
        if output_path.suffix == ".npz":
            parser.error("Output must be a directory in batch mode")
        batch_compute_fiber_profiles(
            matched_paths,
            output_path,
            voxel_size_um=args.voxel_size,
            max_radius_um=args.max_radius,
            step_size_um=args.step_size,
            max_axons=args.max_axons,
            anisotropy_mode=args.anisotropy_mode,
            backend=args.backend,
            output_suffix=args.output_suffix,
        )
