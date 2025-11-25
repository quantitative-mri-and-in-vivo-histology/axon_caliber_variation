#!/usr/bin/env python3
"""
Compute axon radius profiles by sampling perpendicular cross-sections along skeletons.

For each axon:
1. Extract skeleton using DeepACSON
2. Walk along skeleton points
3. Sample perpendicular cross-section at each point
4. Compute equivalent circular radius from cross-section area

Output format matches cc_morph.mat structure: per-axon arrays of (radius_profile, skeleton_coords)
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path
import multiprocessing as mp
from typing import Tuple, Union

import numpy as np
import numba
from numba import njit, prange
import h5py
import scipy.io as sio
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import kimimaro

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_voxel_size(voxel_size_um: Union[float, Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """
    Parse voxel size to a (vx, vy, vz) tuple.

    Args:
        voxel_size_um: Scalar (isotropic) or (vx, vy, vz) tuple (anisotropic)

    Returns:
        Tuple of (vx, vy, vz) in micrometers
    """
    if isinstance(voxel_size_um, (tuple, list)):
        if len(voxel_size_um) == 3:
            return tuple(float(v) for v in voxel_size_um)
        elif len(voxel_size_um) == 1:
            v = float(voxel_size_um[0])
            return (v, v, v)
        else:
            raise ValueError(f"Expected 1 or 3 voxel size values, got {len(voxel_size_um)}")
    # Scalar - isotropic
    v = float(voxel_size_um)
    return (v, v, v)


def parse_voxel_size_arg(value: str) -> Union[float, Tuple[float, float, float]]:
    """Parse voxel size CLI argument: single float or comma-separated triple."""
    if ',' in value:
        parts = value.split(',')
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"Expected 1 or 3 values, got {len(parts)}: '{value}'"
            )
        return tuple(float(p.strip()) for p in parts)
    return float(value)

# Global variable for volume access in worker processes
_shared_volume = None


def init_worker(volume):
    """Initialize worker process with shared volume reference."""
    global _shared_volume
    _shared_volume = volume


def unit_tangent_vector(curve):
    """
    Compute unit tangent vectors along a curve.

    Args:
        curve: (N, 3) array of points

    Returns:
        (N, 3) array of unit tangent vectors
    """
    d_curve = np.gradient(curve, axis=0)
    ds = np.sqrt(np.sum(d_curve**2, axis=1, keepdims=True))
    ds[ds == 0] = 1e-5
    return d_curve / ds


@njit(cache=True)
def rotation_matrix_3D(vector, theta):
    """
    Create rotation matrix for counterclockwise rotation about a unit vector.
    Uses Euler-Rodrigues formula.
    """
    a = np.cos(theta / 2.0)
    sin_half = np.sin(theta / 2.0)
    b = -vector[0] * sin_half
    c = -vector[1] * sin_half
    d = -vector[2] * sin_half
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

    result = np.empty((3, 3), dtype=np.float64)
    result[0, 0] = aa+bb-cc-dd
    result[0, 1] = 2*(bc+ad)
    result[0, 2] = 2*(bd-ac)
    result[1, 0] = 2*(bc-ad)
    result[1, 1] = aa+cc-bb-dd
    result[1, 2] = 2*(cd+ab)
    result[2, 0] = 2*(bd+ac)
    result[2, 1] = 2*(cd-ab)
    result[2, 2] = aa+dd-bb-cc
    return result


@njit(cache=True)
def unit_normal_vector(vec1, vec2):
    """Compute unit normal vector from cross product."""
    n = np.cross(vec1, vec2)
    norm_sq = n[0]*n[0] + n[1]*n[1] + n[2]*n[2]
    if norm_sq < 1e-10:
        n = vec1.copy()
        norm_sq = n[0]*n[0] + n[1]*n[1] + n[2]*n[2]
    s = max(np.sqrt(norm_sq), 1e-5)
    return n / s


@njit(cache=True)
def angle_between(vec1, vec2):
    """Compute angle between two vectors."""
    norm1 = np.sqrt(vec1[0]*vec1[0] + vec1[1]*vec1[1] + vec1[2]*vec1[2])
    norm2 = np.sqrt(vec2[0]*vec2[0] + vec2[1]*vec2[1] + vec2[2]*vec2[2])
    dot = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]
    cos_angle = dot / (norm1 * norm2 + 1e-10)
    if cos_angle > 1.0:
        cos_angle = 1.0
    elif cos_angle < -1.0:
        cos_angle = -1.0
    return np.arccos(cos_angle)


@njit(cache=True, parallel=True)
def prepare_sample_coordinates(points, tangent_vecs, base_xyz, n_grid):
    """
    Prepare all sampling coordinates for all skeleton points.
    JIT-compiled for speed.
    """
    n_points = len(points)
    all_sample_coords = np.zeros((n_points * n_grid, 3), dtype=np.float32)
    z_axis = np.array([0.0, 0.0, 1.0])

    for i in prange(n_points):
        point = points[i]
        tangent = tangent_vecs[i]

        # Check if tangent is zero
        tangent_norm = np.sqrt(tangent[0]**2 + tangent[1]**2 + tangent[2]**2)
        if tangent_norm < 1e-10:
            # Fill with out-of-bounds coords
            for j in range(n_grid):
                all_sample_coords[i * n_grid + j, 0] = -1.0
                all_sample_coords[i * n_grid + j, 1] = -1.0
                all_sample_coords[i * n_grid + j, 2] = -1.0
            continue

        # Compute rotation
        dot_z = tangent[2]  # dot product with z_axis

        # Check if aligned with z-axis
        if abs(dot_z) > 0.9999:
            # Already aligned or opposite
            if dot_z < 0:
                # Flip z
                for j in range(n_grid):
                    all_sample_coords[i * n_grid + j, 0] = base_xyz[j, 0] + point[0]
                    all_sample_coords[i * n_grid + j, 1] = base_xyz[j, 1] + point[1]
                    all_sample_coords[i * n_grid + j, 2] = -base_xyz[j, 2] + point[2]
            else:
                for j in range(n_grid):
                    all_sample_coords[i * n_grid + j, 0] = base_xyz[j, 0] + point[0]
                    all_sample_coords[i * n_grid + j, 1] = base_xyz[j, 1] + point[1]
                    all_sample_coords[i * n_grid + j, 2] = base_xyz[j, 2] + point[2]
        else:
            # Compute rotation matrix
            rot_axis = unit_normal_vector(z_axis, tangent)
            theta = angle_between(z_axis, tangent)
            rot_mat = rotation_matrix_3D(rot_axis, theta)

            # Rotate and translate each grid point
            for j in range(n_grid):
                x = base_xyz[j, 0]
                y = base_xyz[j, 1]
                z = base_xyz[j, 2]
                # Matrix multiply: rot_mat.T @ [x, y, z]
                rx = rot_mat[0, 0] * x + rot_mat[1, 0] * y + rot_mat[2, 0] * z
                ry = rot_mat[0, 1] * x + rot_mat[1, 1] * y + rot_mat[2, 1] * z
                rz = rot_mat[0, 2] * x + rot_mat[1, 2] * y + rot_mat[2, 2] * z
                all_sample_coords[i * n_grid + j, 0] = rx + point[0]
                all_sample_coords[i * n_grid + j, 1] = ry + point[1]
                all_sample_coords[i * n_grid + j, 2] = rz + point[2]

    return all_sample_coords


@njit(cache=True, parallel=True)
def compute_areas_from_samples(all_values, n_points, n_grid, plane_resolution):
    """
    Count voxels and compute areas for all cross-sections.

    Assumes clean data where the interpolated values represent only the target axon.
    Simply counts voxels above threshold and converts to area.

    Args:
        all_values: Flattened array of all interpolated values (n_points * n_grid,)
        n_points: Number of skeleton points
        n_grid: Number of grid points per cross-section
        plane_resolution: Resolution of sampling plane

    Returns:
        areas: Array of cross-section areas (n_points,)
    """
    areas = np.zeros(n_points, dtype=np.float32)
    res_squared = plane_resolution * plane_resolution

    for i in prange(n_points):
        count = 0
        start_idx = i * n_grid
        for j in range(n_grid):
            if all_values[start_idx + j] >= 0.5:
                count += 1
        areas[i] = count * res_squared

    return areas


def sample_cross_sections_batched(volume, points, tangent_vecs,
                                   plane_radius=5.0, plane_resolution=0.5):
    """
    Sample perpendicular cross-sections for multiple skeleton points at once.

    Args:
        volume: 3D binary array of the axon (float32)
        points: (N, 3) array of skeleton point coordinates
        tangent_vecs: (N, 3) array of unit tangent vectors
        plane_radius: Radius of sampling plane in voxels
        plane_resolution: Resolution of sampling plane

    Returns:
        areas: Array of cross-section areas in voxels^2
    """
    n_points = len(points)
    if n_points == 0:
        return []

    # Create base sampling grid in XY plane (reused for all points)
    coords = np.arange(-plane_radius, plane_radius + plane_resolution, plane_resolution)
    x, y = np.meshgrid(coords, coords)
    grid_shape = x.shape
    n_grid = x.size
    z = np.zeros_like(x)
    base_xyz = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)  # (n_grid, 3)

    # Prepare all sampling coordinates using JIT-compiled function
    points_f32 = np.ascontiguousarray(points.astype(np.float64))
    tangent_f64 = np.ascontiguousarray(tangent_vecs.astype(np.float64))
    base_xyz_f64 = base_xyz.astype(np.float64)

    all_sample_coords = prepare_sample_coordinates(points_f32, tangent_f64, base_xyz_f64, n_grid)

    # Single interpolation call for all points using map_coordinates
    # map_coordinates expects (ndim, npoints), so transpose
    # order=1 for linear interpolation, mode='constant' with cval=0 for out-of-bounds
    all_values = map_coordinates(
        volume,
        all_sample_coords.T,
        order=1,
        mode='constant',
        cval=0.0
    )

    # Compute areas using JIT-compiled function (parallel over cross-sections)
    areas = compute_areas_from_samples(all_values, n_points, n_grid, plane_resolution)

    return areas


def process_single_axon(args):
    """
    Process a single axon to extract radius profile along skeleton.

    Args:
        args: Tuple of (axon_label, voxel_size_um, plane_radius, plane_resolution, step_size)
              voxel_size_um: (vx, vy, vz) tuple in micrometers

    Returns:
        Dict with radius profile and skeleton coords, or None if failed
    """
    axon_label, voxel_size_um, plane_radius, plane_resolution, step_size = args

    # Parse voxel size to ensure we have (vx, vy, vz)
    vx, vy, vz = parse_voxel_size(voxel_size_um)
    # Geometric mean for approximate isotropic conversions
    voxel_geom_mean = (vx * vy * vz) ** (1/3)

    try:
        # Get axon coordinates directly (avoids 20GB boolean array per worker)
        # The temporary boolean from comparison is garbage collected after argwhere
        coords = np.argwhere(_shared_volume == axon_label)
        if len(coords) < 100:  # Skip very small axons
            return None

        # Compute bounding box with padding
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)

        padding = int(plane_radius) + 5
        min_padded = np.maximum(min_coords - padding, 0)
        max_padded = np.minimum(max_coords + padding, np.array(_shared_volume.shape))

        # Create binary mask only for the cropped region (much smaller)
        cropped_region = _shared_volume[
            min_padded[0]:max_padded[0],
            min_padded[1]:max_padded[1],
            min_padded[2]:max_padded[2]
        ]
        # Kimimaro expects a labeled volume (uint8 with label 1 for the axon)
        cropped = np.ascontiguousarray((cropped_region == axon_label).astype(np.uint8))

        # Run skeletonization with Kimimaro
        # Use conservative parameters for axon skeletonization
        # Note: We keep anisotropy=(1,1,1) so vertices are returned in voxel coordinates.
        # We handle anisotropy ourselves when computing physical lengths and radii.
        skels = kimimaro.skeletonize(
            cropped,
            teasar_params={
                "scale": 1.5,
                "const": 10,  # Small const for thin structures like axons
            },
            anisotropy=(1, 1, 1),  # Keep voxel coordinates; scale ourselves later
            dust_threshold=50,  # Remove very small components
            progress=False,
            parallel=1,  # Single thread (already parallelizing at axon level)
        )

        if len(skels) == 0 or 1 not in skels:
            return None

        skel = skels[1]  # Get skeleton for label 1
        vertices = skel.vertices  # (N, 3) array
        edges = skel.edges  # (M, 2) array

        if len(vertices) < 3:
            return None

        # Convert skeleton graph to ordered path by tracing from an endpoint
        # Build adjacency list
        from collections import defaultdict
        adj = defaultdict(list)
        for e in edges:
            adj[e[0]].append(e[1])
            adj[e[1]].append(e[0])

        # Find endpoints (degree 1) or use first vertex
        endpoints = [v for v in adj if len(adj[v]) == 1]
        if len(endpoints) == 0:
            start = 0
        else:
            start = endpoints[0]

        # Trace path using DFS
        visited = set()
        path = []
        stack = [start]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            path.append(vertices[v])
            # Add unvisited neighbors
            for neighbor in adj[v]:
                if neighbor not in visited:
                    stack.append(neighbor)

        main_skel = np.array(path)

        if len(main_skel) < 3:
            return None

        # Subsample skeleton to step_size intervals
        # Compute cumulative arc length in physical units (accounting for anisotropy)
        # Skeleton coords are (z, y, x) in voxel indices
        diffs = np.diff(main_skel, axis=0)
        # Scale by voxel sizes: axis 0 = z, axis 1 = y, axis 2 = x
        diffs_physical = diffs * np.array([vz, vy, vx])
        segment_lengths = np.sqrt(np.sum(diffs_physical**2, axis=1))
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_length[-1]  # Now in physical units (μm)

        if total_length < step_size:
            # Too short - use all points
            sampled_skel = main_skel
        else:
            # Sample at regular intervals
            sample_positions = np.arange(0, total_length, step_size)
            sampled_skel = []
            for pos in sample_positions:
                # Find the segment containing this position
                idx = np.searchsorted(cumulative_length, pos) - 1
                idx = max(0, min(idx, len(main_skel) - 2))

                # Interpolate within segment
                t = (pos - cumulative_length[idx]) / (segment_lengths[idx] + 1e-10)
                t = np.clip(t, 0, 1)
                point = main_skel[idx] + t * (main_skel[idx + 1] - main_skel[idx])
                sampled_skel.append(point)

            sampled_skel = np.array(sampled_skel)

        if len(sampled_skel) < 2:
            return None

        # Compute tangent vectors
        tangent_vecs = unit_tangent_vector(sampled_skel)

        # Convert to float32 for interpolation
        cropped_float = cropped.astype(np.float32)

        # Sample all cross-sections in one batch (major speedup)
        areas = sample_cross_sections_batched(
            cropped_float, sampled_skel, tangent_vecs,
            plane_radius=plane_radius,
            plane_resolution=plane_resolution
        )

        # Convert areas to radii and filter valid points (area > 0)
        valid_mask = areas > 0
        if np.sum(valid_mask) < 2:
            return None

        valid_areas = areas[valid_mask]
        valid_skel = sampled_skel[valid_mask]

        # Convert area to equivalent circular radius: A = π * r², so r = sqrt(A / π)
        # For anisotropic voxels, use geometric mean for the cross-section plane
        # (cross-section could be at any orientation, so we use 3D geometric mean)
        radii = np.sqrt(valid_areas / np.pi) * voxel_geom_mean

        # Convert skeleton points back to original coordinates (per-axis scaling)
        # Skeleton coords are (z, y, x) in voxel indices
        skel_coords = (valid_skel + min_padded) * np.array([vz, vy, vx])

        result = {
            'label': axon_label,
            'radii_um': radii,
            'skeleton_um': skel_coords,
            'n_points': len(radii),
            'mean_radius_um': np.mean(radii),
            'std_radius_um': np.std(radii),
            'length_um': total_length  # Already in physical units (μm)
        }

        logger.info(f"Axon {axon_label}: {len(radii)} points, "
                   f"mean r={np.mean(radii):.3f} μm, length={total_length:.1f} μm")

        return result

    except Exception as e:
        logger.error(f"Axon {axon_label}: Processing failed - {e}\n{traceback.format_exc()}")
        return None


def compute_radius_profiles(mat_file: Path,
                            output_file: Path,
                            voxel_size_um: Union[float, Tuple[float, float, float]] = 0.05,
                            plane_radius: float = 10.0,
                            plane_resolution: float = 0.5,
                            step_size: float = 2.0,
                            n_jobs: int = -1,
                            max_axons: int = 0):
    """
    Compute radius profiles for all axons in a labeled volume.

    Args:
        mat_file: Path to .mat file with labeled axons
        output_file: Path to save results (.npz)
        voxel_size_um: Voxel size in micrometers. Can be:
                      - float: isotropic voxel size (e.g., 0.05)
                      - tuple: anisotropic (vx, vy, vz) (e.g., (0.015, 0.015, 0.05))
        plane_radius: Radius of sampling plane in voxels
        plane_resolution: Resolution of sampling plane
        step_size: Step size along skeleton in physical units (μm)
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        max_axons: Maximum number of axons to process (0 = all)
    """
    # Parse voxel size to (vx, vy, vz) tuple
    voxel_size_tuple = parse_voxel_size(voxel_size_um)
    # Validate input file
    if not mat_file.exists():
        raise FileNotFoundError(f"Input file not found: {mat_file}")

    # Load labeled volume - try HDF5 first, then scipy.io for older formats
    logger.info(f"Loading {mat_file}")
    volume = None

    # Try HDF5 format first (MATLAB v7.3)
    try:
        with h5py.File(str(mat_file), 'r') as f:
            # Find the labeled volume key
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
        # Try scipy.io for older MATLAB formats (v5/v6/v7)
        logger.info(f"HDF5 failed ({e}), trying scipy.io for older MATLAB format...")
        mat_data = sio.loadmat(str(mat_file))

        # Find the labeled volume key (skip MATLAB metadata keys)
        volume_key = None
        for key in mat_data.keys():
            if not key.startswith('__'):
                volume_key = key
                break

        if volume_key is None:
            raise ValueError(f"No data found in {mat_file}")

        volume = mat_data[volume_key]
        logger.info(f"Loaded scipy.io format, volume shape: {volume.shape}, dtype: {volume.dtype}")

    # Get unique axon labels
    axon_labels = np.unique(volume)
    axon_labels = axon_labels[axon_labels > 0]  # Remove background

    logger.info(f"Found {len(axon_labels)} axons")

    if max_axons > 0:
        axon_labels = axon_labels[:max_axons]
        logger.info(f"Processing first {max_axons} axons")

    # Handle n_jobs
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    # Prepare arguments for parallel processing
    args_list = [
        (label, voxel_size_tuple, plane_radius, plane_resolution, step_size)
        for label in axon_labels
    ]

    # Log voxel size info
    vx, vy, vz = voxel_size_tuple
    if vx == vy == vz:
        logger.info(f"Voxel size (isotropic): {vx:.4f} μm")
    else:
        logger.info(f"Voxel size (anisotropic): vx={vx:.4f}, vy={vy:.4f}, vz={vz:.4f} μm")

    logger.info(f"Processing {len(args_list)} axons with {n_jobs} workers")
    logger.info(f"Parameters: plane_radius={plane_radius}, resolution={plane_resolution}, step={step_size} μm")

    results = []
    if n_jobs == 1:
        # Sequential processing
        global _shared_volume
        _shared_volume = volume
        for args in tqdm(args_list, desc="Processing axons"):
            result = process_single_axon(args)
            if result is not None:
                results.append(result)
    else:
        # Parallel processing using fork
        with mp.Pool(n_jobs, initializer=init_worker, initargs=(volume,)) as pool:
            for result in tqdm(pool.imap_unordered(process_single_axon, args_list),
                               total=len(args_list), desc="Processing axons"):
                if result is not None:
                    results.append(result)

    logger.info(f"Successfully processed {len(results)}/{len(args_list)} axons")

    if len(results) == 0:
        logger.error("No axons were successfully processed!")
        return

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to arrays
    labels = np.array([r['label'] for r in results])
    n_points = np.array([r['n_points'] for r in results])
    mean_radii = np.array([r['mean_radius_um'] for r in results])
    std_radii = np.array([r['std_radius_um'] for r in results])
    lengths = np.array([r['length_um'] for r in results])

    # Per-axon profiles (variable length, stored as object arrays)
    radii_profiles = np.array([r['radii_um'] for r in results], dtype=object)
    skeleton_coords = np.array([r['skeleton_um'] for r in results], dtype=object)

    # Flatten all radii for distribution analysis
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
        voxel_size_um=np.array(voxel_size_tuple),  # Store as (vx, vy, vz) array
        plane_radius=plane_radius,
        plane_resolution=plane_resolution,
        step_size=step_size,
        source_file=str(mat_file)
    )

    logger.info(f"Saved results to {output_file}")

    # Print summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total axons processed: {len(results)}")
    logger.info(f"  Total radius samples: {len(all_radii)}")
    logger.info(f"  Mean axon length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f} μm")
    logger.info(f"  Mean points per axon: {np.mean(n_points):.1f}")
    logger.info(f"  Mean radius: {np.mean(all_radii):.3f} ± {np.std(all_radii):.3f} μm")
    logger.info(f"  Median radius: {np.median(all_radii):.3f} μm")

    # Compute effective radius
    r_eff = (np.mean(all_radii**6) / np.mean(all_radii**2)) ** 0.25
    logger.info(f"  Effective radius (r_eff): {r_eff:.3f} μm")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute axon radius profiles by sampling perpendicular cross-sections'
    )
    parser.add_argument('mat_file', type=Path,
                        help='Path to .mat file with labeled axons')
    parser.add_argument('output_file', type=Path,
                        help='Output .npz file for results')
    parser.add_argument('--voxel-size', type=parse_voxel_size_arg, default=0.05,
                        help='Voxel size in micrometers: single value for isotropic (e.g., 0.05) '
                             'or vx,vy,vz for anisotropic (e.g., 0.015,0.015,0.05). Default: 0.05')
    parser.add_argument('--plane-radius', type=float, default=10.0,
                        help='Radius of sampling plane in voxels (default: 10.0)')
    parser.add_argument('--plane-resolution', type=float, default=0.5,
                        help='Resolution of sampling plane (default: 0.5)')
    parser.add_argument('--step-size', type=float, default=2.0,
                        help='Step size along skeleton in physical units μm (default: 2.0)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (default: -1 = all CPUs)')
    parser.add_argument('--max-axons', type=int, default=0,
                        help='Maximum axons to process (0 = all, default: 0)')

    args = parser.parse_args()

    compute_radius_profiles(
        args.mat_file,
        args.output_file,
        voxel_size_um=args.voxel_size,
        plane_radius=args.plane_radius,
        plane_resolution=args.plane_resolution,
        step_size=args.step_size,
        n_jobs=args.n_jobs,
        max_axons=args.max_axons
    )
