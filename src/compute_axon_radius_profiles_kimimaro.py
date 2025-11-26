#!/usr/bin/env python3
"""
Compute axon radius profiles using Kimimaro for batch skeletonization.

This script processes the entire labeled volume at once using Kimimaro,
which is more efficient than processing axons individually.

For each axon:
1. Extract skeleton from batch result
2. Compute cross-sectional areas using Kimimaro's optimized implementation
3. Convert areas to equivalent circular radii
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Union

import h5py
import kimimaro
import numpy as np
import scipy.io as sio
import scipy.ndimage as ndi
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_axon_skeleton(label, skel, voxel_size_um):
    """Process a single axon skeleton to extract radius profile from cross-sectional areas.

    Note: Cross-sectional areas from Kimimaro are already in physical units (nm²),
    determined by the anisotropy parameter passed to kimimaro functions.
    We convert nm² → μm² by dividing by 1e6, then compute radius = sqrt(A/π).
    """
    try:
        vertices = skel.vertices
        areas = skel.cross_sectional_area
        contacts = skel.cross_sectional_area_contacts

        if len(vertices) < 3:
            return None

        # Filter out points that contacted image border
        valid_mask = (areas > 0) & (contacts == 0)
        if np.sum(valid_mask) < 2:
            return None

        valid_areas = areas[valid_mask]
        valid_vertices = vertices[valid_mask]

        # Convert areas from nm² (from anisotropy in nm) to μm² and compute equivalent circular radii
        # areas are in nm², convert to μm²: divide by 1,000,000
        areas_um2 = valid_areas / 1_000_000
        radii = np.sqrt(areas_um2 / np.pi)
        # Kimimaro vertices are in physical nm units, convert to μm
        skel_coords = valid_vertices / 1000  # nm → μm

        # Compute total skeleton length using Kimimaro's cable_length method
        # This correctly sums edge lengths in the skeleton graph (not sequential vertex diffs)
        # cable_length() returns length in nm (since vertices are in nm from anisotropy)
        total_length = skel.cable_length() / 1000  # nm → μm

        return {
            'label': label,
            'radii_um': radii,
            'skeleton_um': skel_coords,
            'n_points': len(radii),
            'mean_radius_um': np.mean(radii),
            'std_radius_um': np.std(radii),
            'length_um': total_length
        }

    except Exception as e:
        logger.debug(f"Axon {label}: Processing failed - {e}")
        return None


def compute_radius_profiles(mat_file: Path,
                            output_file: Path,
                            voxel_size_um: Union[float, Tuple[float, float, float]] = 0.05,
                            downsample: int = 1,
                            smoothing_window: int = 5,
                            parallel: int = 0,
                            max_axons: int = 0,
                            teasar_scale: float = None,
                            teasar_const: float = None,
                            tick_threshold: float = 0):
    """
    Compute radius profiles for all axons using batch skeletonization.

    Args:
        mat_file: Path to .mat file with labeled axons
        output_file: Path to save results (.npz)
        voxel_size_um: Original voxel size in micrometers.
                      Can be a scalar (isotropic) or tuple (vz, vy, vx) for anisotropic data.
        downsample: Downsampling factor (1 = no downsampling)
        smoothing_window: Rolling average window for plane normals
        parallel: Number of parallel workers for Kimimaro (0 = all CPUs)
        max_axons: Maximum number of axons to process (0 = all)
        teasar_scale: TEASAR scale parameter (larger = fewer branches). Default: Kimimaro default.
        teasar_const: TEASAR const parameter in nm (larger = fewer branches). Default: Kimimaro default.
        tick_threshold: Remove branches shorter than this (nm). Default: 0 (no removal).
    """
    if not mat_file.exists():
        raise FileNotFoundError(f"Input file not found: {mat_file}")

    # Load labeled volume
    logger.info(f"Loading {mat_file}")
    volume = None

    try:
        with h5py.File(str(mat_file), 'r') as f:
            volume_key = None
            for key in f.keys():
                if not key.startswith('#') and not key.startswith('_'):
                    volume_key = key
                    break
            if volume_key is None:
                raise ValueError(f"No data found in {mat_file}")
            volume = f[volume_key][:]
            logger.info(f"Loaded HDF5 format, volume shape: {volume.shape}, dtype: {volume.dtype}")
    except OSError as e:
        logger.info(f"HDF5 failed ({e}), trying scipy.io...")
        mat_data = sio.loadmat(str(mat_file))
        volume_key = None
        for key in mat_data.keys():
            if not key.startswith('__'):
                volume_key = key
                break
        if volume_key is None:
            raise ValueError(f"No data found in {mat_file}")
        volume = mat_data[volume_key]
        logger.info(f"Loaded scipy.io format, volume shape: {volume.shape}, dtype: {volume.dtype}")

    # Parse voxel size (scalar or tuple)
    # Input is (vz, vy, vx) matching array axes (Z, Y, X)
    if isinstance(voxel_size_um, (list, tuple)):
        vz, vy, vx = voxel_size_um
    else:
        vx = vy = vz = voxel_size_um

    # Apply downsampling if requested
    if downsample > 1:
        logger.info(f"Downsampling by factor {downsample}...")
        original_shape = volume.shape
        # Use nearest-neighbor for labeled data to preserve label values
        volume = volume[::downsample, ::downsample, ::downsample]
        logger.info(f"Downsampled: {original_shape} -> {volume.shape}")
        vx *= downsample
        vy *= downsample
        vz *= downsample
        logger.info(f"Effective voxel size: ({vx:.4f}, {vy:.4f}, {vz:.4f}) μm")
    else:
        logger.info(f"Voxel size: ({vx:.4f}, {vy:.4f}, {vz:.4f}) μm")

    # Store effective voxel size for later use (for isotropic case, use geometric mean)
    if isinstance(voxel_size_um, (list, tuple)):
        effective_voxel_size = (vx * vy * vz) ** (1/3)
    else:
        effective_voxel_size = vx  # Isotropic case

    # Get unique axon labels
    axon_labels = np.unique(volume)
    axon_labels = axon_labels[axon_labels > 0]
    n_total = len(axon_labels)

    logger.info(f"Found {n_total} axons")

    if max_axons > 0:
        axon_labels = axon_labels[:max_axons]
        logger.info(f"Processing first {max_axons} axons")

    # Run batch skeletonization with Kimimaro
    logger.info(f"Running Kimimaro skeletonization (parallel={parallel})...")
    logger.info(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")
    logger.info(f"Labels to process: {len(axon_labels)}")

    # Convert voxel sizes from μm to nm for Kimimaro
    # Kimimaro anisotropy should match array axis order (axis0, axis1, axis2)
    # Our voxel_size is (vz, vy, vx) matching array axes, so anisotropy = (vz, vy, vx) * 1000
    anisotropy_nm = (vz * 1000, vy * 1000, vx * 1000)
    logger.info(f"Anisotropy (nm) for axes (0,1,2): {anisotropy_nm}")

    # Build TEASAR parameters if specified
    # TEASAR invalidation sphere radius = scale * DBF(p) + const
    # Larger values = fewer branches (more aggressive pruning during skeletonization)
    teasar_params = {}
    if teasar_scale is not None:
        teasar_params['scale'] = teasar_scale
        logger.info(f"TEASAR scale: {teasar_scale}")
    if teasar_const is not None:
        teasar_params['const'] = teasar_const
        logger.info(f"TEASAR const: {teasar_const} nm")

    skeletonize_kwargs = {
        'progress': True,
        'parallel': parallel,
        'fix_branching': True,
        'parallel_chunk_size': 50,
        'in_place': False,
        'anisotropy': anisotropy_nm,
    }
    if teasar_params:
        skeletonize_kwargs['teasar_params'] = teasar_params

    skels = kimimaro.skeletonize(volume, **skeletonize_kwargs)

    logger.info(f"Skeletonization complete. Got {len(skels)} skeletons")

    # Postprocess to remove short branches (ticks) if requested
    if tick_threshold > 0:
        logger.info(f"Removing branches shorter than {tick_threshold} nm...")
        for label in list(skels.keys()):
            skels[label] = kimimaro.postprocess(
                skels[label],
                tick_threshold=tick_threshold
            )
        logger.info("Postprocessing complete")

    # Compute cross-sectional areas using Kimimaro's optimized implementation
    logger.info(f"Computing cross-sectional areas (smoothing_window={smoothing_window})...")
    skels = kimimaro.cross_sectional_area(
        volume,
        skels,
        anisotropy=anisotropy_nm,
        smoothing_window=smoothing_window,
        progress=True,
    )

    logger.info("Cross-sectional area computation complete")

    # Process each skeleton to extract radius profiles
    logger.info("Extracting radius profiles...")
    results = []

    for label in tqdm(axon_labels, desc="Processing axons"):
        if label not in skels:
            continue

        result = process_axon_skeleton(label, skels[label], effective_voxel_size)

        if result is not None:
            results.append(result)
            logger.debug(f"Axon {label}: {result['n_points']} points, "
                        f"mean r={result['mean_radius_um']:.3f} μm")

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
        voxel_size_um=voxel_size_um,
        effective_voxel_size_um=effective_voxel_size,
        downsample=downsample,
        smoothing_window=smoothing_window,
        source_file=str(mat_file)
    )

    logger.info(f"Saved results to {output_file}")

    # Print summary
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total axons processed: {len(results)}")
    logger.info(f"  Total radius samples: {len(all_radii)}")
    logger.info(f"  Mean axon length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f} μm")
    logger.info(f"  Mean points per axon: {np.mean(n_points):.1f}")
    logger.info(f"  Mean radius: {np.mean(all_radii):.3f} ± {np.std(all_radii):.3f} μm")
    logger.info(f"  Median radius: {np.median(all_radii):.3f} μm")

    r_eff = (np.mean(all_radii**6) / np.mean(all_radii**2)) ** 0.25
    logger.info(f"  Effective radius (r_eff): {r_eff:.3f} μm")


def parse_voxel_size(values):
    """Parse voxel size from CLI arguments.

    Accepts either:
    - Single float: treated as isotropic (vx = vy = vz)
    - Three floats: treated as anisotropic (vx, vy, vz)
    """
    floats = [float(v) for v in values]
    if len(floats) == 1:
        return floats[0]
    elif len(floats) == 3:
        return tuple(floats)
    else:
        raise ValueError(f"Expected 1 or 3 voxel size values, got {len(floats)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute axon radius profiles using Kimimaro batch skeletonization'
    )
    parser.add_argument('mat_file', type=Path,
                        help='Path to .mat file with labeled axons')
    parser.add_argument('output_file', type=Path,
                        help='Output .npz file for results')
    parser.add_argument('--voxel-size', nargs='+', default=['0.05'],
                        help='Voxel size in micrometers. Can be: '
                             '(1) single float for isotropic (e.g., 0.05) '
                             'or (2) three floats for anisotropic (e.g., 0.015 0.015 0.05)')
    parser.add_argument('--downsample', type=int, default=1,
                        help='Downsampling factor applied before processing (default: 1)')
    parser.add_argument('--smoothing-window', type=int, default=5,
                        help='Rolling average window for plane normals (default: 5)')
    parser.add_argument('--parallel', type=int, default=0,
                        help='Parallel workers for Kimimaro (0 = all CPUs)')
    parser.add_argument('--max-axons', type=int, default=0,
                        help='Maximum axons to process (0 = all)')
    parser.add_argument('--teasar-scale', type=float, default=None,
                        help='TEASAR scale parameter for invalidation sphere (default: Kimimaro default). '
                             'Larger values reduce branching.')
    parser.add_argument('--teasar-const', type=float, default=None,
                        help='TEASAR const parameter in nm (default: Kimimaro default). '
                             'Larger values reduce branching.')
    parser.add_argument('--tick-threshold', type=float, default=0,
                        help='Remove branches shorter than this threshold in nm (default: 0 = no removal). '
                             'Recommended: 1000-3000 nm to remove small spurious branches.')

    args = parser.parse_args()

    # Parse voxel size
    voxel_size_um = parse_voxel_size(args.voxel_size)

    compute_radius_profiles(
        args.mat_file,
        args.output_file,
        voxel_size_um=voxel_size_um,
        downsample=args.downsample,
        smoothing_window=args.smoothing_window,
        parallel=args.parallel,
        max_axons=args.max_axons,
        teasar_scale=args.teasar_scale,
        teasar_const=args.teasar_const,
        tick_threshold=args.tick_threshold
    )
