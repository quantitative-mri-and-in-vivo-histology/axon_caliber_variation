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

import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator as rgi
from skimage.measure import label, regionprops
from tqdm import tqdm

# Add external DeepACSON to path
DEEPACSON_PATH = Path(__file__).parent.parent / 'external' / 'DeepACSON' / 'CSD'
sys.path.insert(0, str(DEEPACSON_PATH))

from skeleton3D import skeleton

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


def rotation_matrix_3D(vector, theta):
    """
    Create rotation matrix for counterclockwise rotation about a unit vector.
    Uses Euler-Rodrigues formula.
    """
    a = np.cos(theta / 2.0)
    b, c, d = -vector * np.sin(theta / 2.0)
    aa, bb, cc, dd = a**2, b**2, c**2, d**2
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

    return np.array([
        [aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
        [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
        [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]
    ])


def unit_normal_vector(vec1, vec2):
    """Compute unit normal vector from cross product."""
    n = np.cross(vec1, vec2)
    if np.allclose(n, 0):
        n = vec1.copy()
    s = max(np.linalg.norm(n), 1e-5)
    return n / s


def angle_between(vec1, vec2):
    """Compute angle between two vectors."""
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle)


def sample_perpendicular_cross_section(binary_volume, point, tangent_vec,
                                       plane_radius=5.0, plane_resolution=0.5):
    """
    Sample a perpendicular cross-section at a skeleton point.

    Args:
        binary_volume: 3D binary array of the axon
        point: (3,) skeleton point coordinates
        tangent_vec: (3,) unit tangent vector at the point
        plane_radius: Radius of sampling plane in voxels
        plane_resolution: Resolution of sampling plane

    Returns:
        area: Cross-section area in voxels^2, or None if sampling failed
    """
    # Create sampling grid in XY plane
    coords = np.arange(-plane_radius, plane_radius + plane_resolution, plane_resolution)
    x, y = np.meshgrid(coords, coords)
    z = np.zeros_like(x)

    # Flatten for rotation
    xyz = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    # Compute rotation to align Z-axis with tangent vector
    # We want to rotate the XY plane (with normal Z) so its normal aligns with tangent
    z_axis = np.array([0, 0, 1])

    if np.allclose(tangent_vec, z_axis) or np.allclose(tangent_vec, -z_axis):
        # Already aligned or opposite
        if np.dot(tangent_vec, z_axis) < 0:
            # Flip
            rotated_plane = xyz * np.array([1, 1, -1])
        else:
            rotated_plane = xyz
    else:
        # Rotation axis is perpendicular to both z_axis and tangent_vec
        # To rotate z_axis -> tangent_vec, use z_axis x tangent_vec
        rot_axis = unit_normal_vector(z_axis, tangent_vec)
        theta = angle_between(z_axis, tangent_vec)
        rot_mat = rotation_matrix_3D(rot_axis, theta)
        # Apply rotation: for row vectors xyz (N,3), use xyz @ rot_mat.T
        rotated_plane = xyz @ rot_mat.T

    # Translate to skeleton point
    sample_coords = rotated_plane + point

    # Set up interpolator
    # Note: skeleton points from DeepACSON are in array index order (i, j, k)
    # which matches RegularGridInterpolator's expected coordinate order
    sz = binary_volume.shape
    interpolator = rgi(
        (range(sz[0]), range(sz[1]), range(sz[2])),
        binary_volume.astype(float),
        bounds_error=False,
        fill_value=0
    )

    # Sample cross-section
    # sample_coords are in (i, j, k) order after rotation and translation
    cross_section = interpolator(sample_coords)
    bw_cross_section = (cross_section >= 0.5).reshape(x.shape)

    # Label connected components
    labeled, n_components = label(bw_cross_section, connectivity=1, return_num=True)

    if n_components == 0:
        return None

    # Find the component at the center (should be the axon)
    center_idx = bw_cross_section.shape[0] // 2
    center_label = labeled[center_idx, center_idx]

    if center_label == 0:
        # No component at center - find closest
        props = regionprops(labeled)
        if not props:
            return None
        # Use largest component
        areas = [p.area for p in props]
        center_label = props[np.argmax(areas)].label

    # Get area of the main component
    main_component = (labeled == center_label)
    area = np.sum(main_component)

    # Convert from grid squares to actual area
    area_voxels = area * (plane_resolution ** 2)

    return area_voxels


def process_single_axon(args):
    """
    Process a single axon to extract radius profile along skeleton.

    Args:
        args: Tuple of (axon_label, voxel_size_um, plane_radius, plane_resolution, step_size)

    Returns:
        Dict with radius profile and skeleton coords, or None if failed
    """
    axon_label, voxel_size_um, plane_radius, plane_resolution, step_size = args

    try:
        # Extract binary volume from shared memory
        axon_binary = (_shared_volume == axon_label).astype(np.uint8)

        # Get axon coordinates and bounding box
        coords = np.argwhere(axon_binary)
        if len(coords) < 100:  # Skip very small axons
            return None

        # Crop to bounding box with padding
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)

        padding = int(plane_radius) + 5
        min_padded = np.maximum(min_coords - padding, 0)
        max_padded = np.minimum(max_coords + padding, np.array(axon_binary.shape))

        cropped = axon_binary[
            min_padded[0]:max_padded[0],
            min_padded[1]:max_padded[1],
            min_padded[2]:max_padded[2]
        ].copy()

        # Run skeletonization
        skel_segments = skeleton(cropped)

        if len(skel_segments) == 0:
            return None

        # Combine all skeleton segments into one continuous skeleton
        # For simplicity, use the longest segment
        if len(skel_segments) == 1:
            main_skel = skel_segments[0]
        else:
            # Find longest segment
            lengths = [len(seg) for seg in skel_segments]
            main_skel = skel_segments[np.argmax(lengths)]

        if len(main_skel) < 3:
            return None

        # Subsample skeleton to step_size intervals
        # Compute cumulative arc length
        diffs = np.diff(main_skel, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_length[-1]

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

        # Sample cross-sections and compute radii
        radii = []
        valid_skel_points = []

        for i, (point, tangent) in enumerate(zip(sampled_skel, tangent_vecs)):
            # Skip if tangent is zero
            if np.allclose(tangent, 0):
                continue

            area = sample_perpendicular_cross_section(
                cropped, point, tangent,
                plane_radius=plane_radius,
                plane_resolution=plane_resolution
            )

            if area is not None and area > 0:
                # Convert area to equivalent circular radius
                # A = π * r², so r = sqrt(A / π)
                radius_voxels = np.sqrt(area / np.pi)
                radius_um = radius_voxels * voxel_size_um

                radii.append(radius_um)
                # Convert skeleton point back to original coordinates
                original_point = point + min_padded
                valid_skel_points.append(original_point * voxel_size_um)

        if len(radii) < 2:
            return None

        radii = np.array(radii)
        skel_coords = np.array(valid_skel_points)

        result = {
            'label': axon_label,
            'radii_um': radii,
            'skeleton_um': skel_coords,
            'n_points': len(radii),
            'mean_radius_um': np.mean(radii),
            'std_radius_um': np.std(radii),
            'length_um': total_length * voxel_size_um
        }

        logger.info(f"Axon {axon_label}: {len(radii)} points, "
                   f"mean r={np.mean(radii):.3f} μm, length={total_length * voxel_size_um:.1f} μm")

        return result

    except Exception as e:
        logger.error(f"Axon {axon_label}: Processing failed - {e}\n{traceback.format_exc()}")
        return None


def compute_radius_profiles(mat_file: Path,
                            output_file: Path,
                            voxel_size_um: float = 0.05,
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
        voxel_size_um: Voxel size in micrometers
        plane_radius: Radius of sampling plane in voxels
        plane_resolution: Resolution of sampling plane
        step_size: Step size along skeleton in voxels
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        max_axons: Maximum number of axons to process (0 = all)
    """
    # Validate input file
    if not mat_file.exists():
        raise FileNotFoundError(f"Input file not found: {mat_file}")

    # Load labeled volume
    logger.info(f"Loading {mat_file}")
    mat = sio.loadmat(str(mat_file))

    # Find the labeled volume key
    volume_key = None
    for key in mat.keys():
        if not key.startswith('_'):
            volume_key = key
            break

    if volume_key is None:
        raise ValueError(f"No data found in {mat_file}")

    volume = mat[volume_key]
    logger.info(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")

    # Get unique axon labels
    axon_labels = np.unique(volume)
    axon_labels = axon_labels[axon_labels > 0]  # Remove background

    logger.info(f"Found {len(axon_labels)} axons")

    if max_axons > 0:
        axon_labels = axon_labels[:max_axons]
        logger.info(f"Processing first {max_axons} axons")

    # Prepare arguments
    args_list = [
        (label, voxel_size_um, plane_radius, plane_resolution, step_size)
        for label in axon_labels
    ]

    # Process axons
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    logger.info(f"Processing {len(args_list)} axons with {n_jobs} workers")
    logger.info(f"Parameters: plane_radius={plane_radius}, resolution={plane_resolution}, step={step_size}")

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
        voxel_size_um=voxel_size_um,
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
    parser.add_argument('--voxel-size', type=float, default=0.05,
                        help='Voxel size in micrometers (default: 0.05)')
    parser.add_argument('--plane-radius', type=float, default=10.0,
                        help='Radius of sampling plane in voxels (default: 10.0)')
    parser.add_argument('--plane-resolution', type=float, default=0.5,
                        help='Resolution of sampling plane (default: 0.5)')
    parser.add_argument('--step-size', type=float, default=2.0,
                        help='Step size along skeleton in voxels (default: 2.0)')
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
