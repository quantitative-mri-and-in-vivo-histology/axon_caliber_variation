"""
Geometric utilities for 3D curve and cross-section analysis.

Includes:
- Tangent vector computation along curves
- 3D rotation matrices (Euler-Rodrigues formula)
- Perpendicular plane sampling grids
"""

import numpy as np


def unit_tangent_vector(curve):
    """
    Compute unit tangent vectors along a curve.

    Uses numpy.gradient for smooth derivatives at endpoints.

    Args:
        curve: (N, 3) array of points

    Returns:
        (N, 3) array of unit tangent vectors
    """
    d_curve = np.gradient(curve, axis=0)
    ds = np.sqrt(np.sum(d_curve**2, axis=1, keepdims=True))
    ds[ds == 0] = 1e-5
    return d_curve / ds


def rotation_matrix_3d(axis, theta):
    """
    Create rotation matrix for counterclockwise rotation about a unit vector.

    Uses Euler-Rodrigues formula.

    Args:
        axis: (3,) unit vector defining rotation axis
        theta: Rotation angle in radians

    Returns:
        (3, 3) rotation matrix
    """
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a**2, b**2, c**2, d**2
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

    return np.array([
        [aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
        [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
        [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]
    ])


def unit_normal_vector(vec1, vec2):
    """
    Compute unit normal vector from cross product of two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Unit normal vector perpendicular to both inputs
    """
    n = np.cross(vec1, vec2)
    if np.allclose(n, 0):
        n = vec1.copy()
    s = max(np.linalg.norm(n), 1e-5)
    return n / s


def angle_between(vec1, vec2):
    """
    Compute angle between two vectors in radians.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Angle in radians [0, pi]
    """
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle)


def create_perpendicular_plane_grid(tangent_vec, plane_radius, plane_resolution):
    """
    Create a grid of sample points on a plane perpendicular to a tangent vector.

    The grid is centered at the origin and must be translated to the desired
    sampling location.

    Args:
        tangent_vec: (3,) unit tangent vector (plane normal)
        plane_radius: Radius of sampling plane
        plane_resolution: Spacing between grid points

    Returns:
        (N, 3) array of sample points on the perpendicular plane
    """
    # Create sampling grid in XY plane
    coords = np.arange(-plane_radius, plane_radius + plane_resolution, plane_resolution)
    x, y = np.meshgrid(coords, coords)
    z = np.zeros_like(x)

    # Flatten for rotation
    xyz = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    # Compute rotation to align Z-axis with tangent vector
    z_axis = np.array([0, 0, 1])

    if np.allclose(tangent_vec, z_axis) or np.allclose(tangent_vec, -z_axis):
        # Already aligned or opposite
        if np.dot(tangent_vec, z_axis) < 0:
            rotated_plane = xyz * np.array([1, 1, -1])
        else:
            rotated_plane = xyz
    else:
        rot_axis = unit_normal_vector(z_axis, tangent_vec)
        theta = angle_between(z_axis, tangent_vec)
        rot_mat = rotation_matrix_3d(rot_axis, theta)
        rotated_plane = xyz @ rot_mat.T

    return rotated_plane


def compute_arc_length(curve, voxel_size=None):
    """
    Compute cumulative arc length along a curve.

    Args:
        curve: (N, 3) array of points in voxel coordinates
        voxel_size: Optional (vz, vy, vx) tuple for physical units.
                   If None, returns arc length in voxel units.

    Returns:
        (N,) array of cumulative arc lengths starting from 0
    """
    if len(curve) < 2:
        return np.array([0.0])

    diffs = np.diff(curve, axis=0)

    if voxel_size is not None:
        # Scale by voxel sizes: axis 0 = z, axis 1 = y, axis 2 = x
        vz, vy, vx = voxel_size
        diffs_physical = diffs * np.array([vz, vy, vx])
        segment_lengths = np.sqrt(np.sum(diffs_physical**2, axis=1))
    else:
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))

    return np.concatenate([[0], np.cumsum(segment_lengths)])


def resample_curve_by_arc_length(curve, step_size, voxel_size=None):
    """
    Resample a curve at regular arc length intervals.

    Args:
        curve: (N, 3) array of points
        step_size: Desired spacing between samples (in same units as voxel_size)
        voxel_size: Optional (vz, vy, vx) tuple for physical units

    Returns:
        (M, 3) array of resampled points
    """
    cumulative_length = compute_arc_length(curve, voxel_size)
    total_length = cumulative_length[-1]

    if total_length < step_size:
        return curve

    # Sample at regular intervals
    sample_positions = np.arange(0, total_length, step_size)

    # Compute segment lengths for interpolation
    if voxel_size is not None:
        vz, vy, vx = voxel_size
        diffs = np.diff(curve, axis=0) * np.array([vz, vy, vx])
    else:
        diffs = np.diff(curve, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))

    sampled_points = []
    for pos in sample_positions:
        # Find the segment containing this position
        idx = np.searchsorted(cumulative_length, pos) - 1
        idx = max(0, min(idx, len(curve) - 2))

        # Interpolate within segment
        t = (pos - cumulative_length[idx]) / (segment_lengths[idx] + 1e-10)
        t = np.clip(t, 0, 1)
        point = curve[idx] + t * (curve[idx + 1] - curve[idx])
        sampled_points.append(point)

    return np.array(sampled_points)
