#!/usr/bin/env python3
"""
Compute axon radius profiles using Kimimaro for batch skeletonization.

This script processes the entire labeled volume at once using Kimimaro,
which is more efficient than processing axons individually.

For each axon:
1. Extract skeleton from batch result
2. Walk along skeleton points
3. Sample perpendicular cross-section at each point
4. Compute equivalent circular radius from cross-section area
"""

import argparse
import logging
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
from numba import njit, prange
import h5py
import scipy.io as sio
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import kimimaro

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def unit_tangent_vector(curve):
    """Compute unit tangent vectors along a curve."""
    d_curve = np.gradient(curve, axis=0)
    ds = np.sqrt(np.sum(d_curve**2, axis=1, keepdims=True))
    ds[ds == 0] = 1e-5
    return d_curve / ds


@njit(cache=True)
def rotation_matrix_3D(vector, theta):
    """Create rotation matrix for counterclockwise rotation about a unit vector."""
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
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return np.arccos(cos_angle)


@njit(cache=True, parallel=True)
def prepare_sample_coordinates(points, tangent_vecs, base_xyz, n_grid):
    """Prepare all sampling coordinates for all skeleton points."""
    n_points = len(points)
    all_sample_coords = np.zeros((n_points * n_grid, 3), dtype=np.float32)
    z_axis = np.array([0.0, 0.0, 1.0])

    for i in prange(n_points):
        point = points[i]
        tangent = tangent_vecs[i]

        tangent_norm = np.sqrt(tangent[0]**2 + tangent[1]**2 + tangent[2]**2)
        if tangent_norm < 1e-10:
            for j in range(n_grid):
                all_sample_coords[i * n_grid + j, 0] = -1.0
                all_sample_coords[i * n_grid + j, 1] = -1.0
                all_sample_coords[i * n_grid + j, 2] = -1.0
            continue

        dot_z = tangent[2]

        if abs(dot_z) > 0.9999:
            if dot_z < 0:
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
            rot_axis = unit_normal_vector(z_axis, tangent)
            theta = angle_between(z_axis, tangent)
            rot_mat = rotation_matrix_3D(rot_axis, theta)

            for j in range(n_grid):
                x = base_xyz[j, 0]
                y = base_xyz[j, 1]
                z = base_xyz[j, 2]
                rx = rot_mat[0, 0] * x + rot_mat[1, 0] * y + rot_mat[2, 0] * z
                ry = rot_mat[0, 1] * x + rot_mat[1, 1] * y + rot_mat[2, 1] * z
                rz = rot_mat[0, 2] * x + rot_mat[1, 2] * y + rot_mat[2, 2] * z
                all_sample_coords[i * n_grid + j, 0] = rx + point[0]
                all_sample_coords[i * n_grid + j, 1] = ry + point[1]
                all_sample_coords[i * n_grid + j, 2] = rz + point[2]

    return all_sample_coords


@njit(cache=True, parallel=True)
def compute_areas_from_samples(all_values, n_points, n_grid, plane_resolution):
    """Count voxels and compute areas for all cross-sections."""
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


def skeleton_to_path(vertices, edges):
    """Convert skeleton graph to ordered path by tracing from an endpoint."""
    if len(vertices) < 2:
        return vertices

    # Build adjacency list
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
        for neighbor in adj[v]:
            if neighbor not in visited:
                stack.append(neighbor)

    return np.array(path)


def sample_cross_sections_batched(volume, points, tangent_vecs,
                                   plane_radius=5.0, plane_resolution=0.5):
    """Sample perpendicular cross-sections for multiple skeleton points."""
    n_points = len(points)
    if n_points == 0:
        return []

    coords = np.arange(-plane_radius, plane_radius + plane_resolution, plane_resolution)
    x, y = np.meshgrid(coords, coords)
    n_grid = x.size
    z = np.zeros_like(x)
    base_xyz = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)

    points_f32 = np.ascontiguousarray(points.astype(np.float64))
    tangent_f64 = np.ascontiguousarray(tangent_vecs.astype(np.float64))
    base_xyz_f64 = base_xyz.astype(np.float64)

    all_sample_coords = prepare_sample_coordinates(points_f32, tangent_f64, base_xyz_f64, n_grid)

    all_values = map_coordinates(
        volume,
        all_sample_coords.T,
        order=1,
        mode='constant',
        cval=0.0
    )

    areas = compute_areas_from_samples(all_values, n_points, n_grid, plane_resolution)

    return areas


def process_axon_skeleton(label, skel, volume, voxel_size_um, plane_radius,
                          plane_resolution, step_size):
    """Process a single axon skeleton to extract radius profile."""
    try:
        vertices = skel.vertices
        edges = skel.edges

        if len(vertices) < 3:
            return None

        # Convert skeleton to ordered path
        main_skel = skeleton_to_path(vertices, edges)

        if len(main_skel) < 3:
            return None

        # Subsample skeleton to step_size intervals
        diffs = np.diff(main_skel, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_length[-1]

        if total_length < step_size:
            sampled_skel = main_skel
        else:
            sample_positions = np.arange(0, total_length, step_size)
            sampled_skel = []
            for pos in sample_positions:
                idx = np.searchsorted(cumulative_length, pos) - 1
                idx = max(0, min(idx, len(main_skel) - 2))
                t = (pos - cumulative_length[idx]) / (segment_lengths[idx] + 1e-10)
                t = np.clip(t, 0, 1)
                point = main_skel[idx] + t * (main_skel[idx + 1] - main_skel[idx])
                sampled_skel.append(point)
            sampled_skel = np.array(sampled_skel)

        if len(sampled_skel) < 2:
            return None

        # Compute tangent vectors
        tangent_vecs = unit_tangent_vector(sampled_skel)

        # Create binary mask for this axon
        axon_mask = (volume == label).astype(np.float32)

        # Sample all cross-sections
        areas = sample_cross_sections_batched(
            axon_mask, sampled_skel, tangent_vecs,
            plane_radius=plane_radius,
            plane_resolution=plane_resolution
        )

        # Convert areas to radii and filter valid points
        valid_mask = areas > 0
        if np.sum(valid_mask) < 2:
            return None

        valid_areas = areas[valid_mask]
        valid_skel = sampled_skel[valid_mask]

        radii = np.sqrt(valid_areas / np.pi) * voxel_size_um
        skel_coords = valid_skel * voxel_size_um

        return {
            'label': label,
            'radii_um': radii,
            'skeleton_um': skel_coords,
            'n_points': len(radii),
            'mean_radius_um': np.mean(radii),
            'std_radius_um': np.std(radii),
            'length_um': total_length * voxel_size_um
        }

    except Exception as e:
        logger.debug(f"Axon {label}: Processing failed - {e}")
        return None


def compute_radius_profiles(mat_file: Path,
                            output_file: Path,
                            voxel_size_um: float = 0.05,
                            plane_radius: float = 10.0,
                            plane_resolution: float = 0.5,
                            step_size: float = 2.0,
                            parallel: int = 0,
                            max_axons: int = 0):
    """
    Compute radius profiles for all axons using batch skeletonization.

    Args:
        mat_file: Path to .mat file with labeled axons
        output_file: Path to save results (.npz)
        voxel_size_um: Voxel size in micrometers
        plane_radius: Radius of sampling plane in voxels
        plane_resolution: Resolution of sampling plane
        step_size: Step size along skeleton in voxels
        parallel: Number of parallel workers for Kimimaro (0 = all CPUs)
        max_axons: Maximum number of axons to process (0 = all)
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

    skels = kimimaro.skeletonize(
        volume,
        teasar_params={
            "scale": 1.5,
            "const": 10,
        },
        anisotropy=(1, 1, 1),
        dust_threshold=100,
        progress=True,
        parallel=parallel,
        fix_branching=True,
    )

    logger.info(f"Skeletonization complete. Got {len(skels)} skeletons")

    # Process each skeleton to extract radius profiles
    logger.info("Extracting radius profiles...")
    results = []

    for label in tqdm(axon_labels, desc="Processing axons"):
        if label not in skels:
            continue

        result = process_axon_skeleton(
            label, skels[label], volume,
            voxel_size_um, plane_radius, plane_resolution, step_size
        )

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
        plane_radius=plane_radius,
        plane_resolution=plane_resolution,
        step_size=step_size,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute axon radius profiles using Kimimaro batch skeletonization'
    )
    parser.add_argument('mat_file', type=Path,
                        help='Path to .mat file with labeled axons')
    parser.add_argument('output_file', type=Path,
                        help='Output .npz file for results')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                        help='Voxel size in micrometers (default: 0.05)')
    parser.add_argument('--plane-radius', type=float, default=10.0,
                        help='Radius of sampling plane in voxels (default: 10.0)')
    parser.add_argument('--plane-resolution', type=float, default=0.5,
                        help='Resolution of sampling plane (default: 0.5)')
    parser.add_argument('--step-size', type=float, default=2.0,
                        help='Step size along skeleton in voxels (default: 2.0)')
    parser.add_argument('--parallel', type=int, default=0,
                        help='Parallel workers for Kimimaro (0 = all CPUs)')
    parser.add_argument('--max-axons', type=int, default=0,
                        help='Maximum axons to process (0 = all)')

    args = parser.parse_args()

    compute_radius_profiles(
        args.mat_file,
        args.output_file,
        voxel_size_um=args.voxel_size,
        plane_radius=args.plane_radius,
        plane_resolution=args.plane_resolution,
        step_size=args.step_size,
        parallel=args.parallel,
        max_axons=args.max_axons
    )
