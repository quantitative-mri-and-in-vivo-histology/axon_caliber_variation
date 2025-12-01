"""
Cross-section sampling utilities for fiber morphometry.

Includes:
- Numba-accelerated nearest neighbor interpolation
- Perpendicular cross-section extraction
- Area-to-radius conversion
"""

import numpy as np
from numba import njit
from skimage.measure import label, regionprops

from .geometry import create_perpendicular_plane_grid


@njit(fastmath=True)
def nearest_interp_3d(volume, coords):
    """
    Fast nearest neighbor 3D interpolation using Numba.

    ~22x faster than scipy.ndimage.map_coordinates for binary label data.

    Args:
        volume: 3D float64 array (binary mask as float)
        coords: (3, N) float64 array of sample coordinates

    Returns:
        (N,) float64 array of interpolated values
    """
    n_points = coords.shape[1]
    result = np.empty(n_points, dtype=np.float64)

    sz0, sz1, sz2 = volume.shape

    for i in range(n_points):
        # Round to nearest integer
        x = int(coords[0, i] + 0.5)
        y = int(coords[1, i] + 0.5)
        z = int(coords[2, i] + 0.5)

        # Bounds check
        if x < 0 or x >= sz0 or y < 0 or y >= sz1 or z < 0 or z >= sz2:
            result[i] = 0.0
        else:
            result[i] = volume[x, y, z]

    return result


def sample_perpendicular_cross_section(binary_volume, point, tangent_vec,
                                       plane_radius=5.0, plane_resolution=0.5,
                                       volume_float=None):
    """
    Sample a perpendicular cross-section at a skeleton point.

    Args:
        binary_volume: 3D binary array of the object
        point: (3,) skeleton point coordinates
        tangent_vec: (3,) unit tangent vector at the point
        plane_radius: Radius of sampling plane in voxels
        plane_resolution: Resolution of sampling plane
        volume_float: Optional pre-converted float64 volume for interpolation

    Returns:
        area: Cross-section area in voxels^2, or None if sampling failed
    """
    # Create perpendicular plane grid
    rotated_plane = create_perpendicular_plane_grid(
        tangent_vec, plane_radius, plane_resolution
    )

    # Translate to skeleton point
    sample_coords = rotated_plane + point

    # Use pre-converted float volume or convert on the fly
    if volume_float is None:
        volume_float = binary_volume.astype(np.float64)

    # Sample cross-section using Numba nearest neighbor interpolation (~22x faster)
    # For binary label data, nearest neighbor is accurate and much faster
    coords_for_interp = np.ascontiguousarray(sample_coords.T)
    cross_section = nearest_interp_3d(volume_float, coords_for_interp)

    # Reshape to grid
    grid_size = int(2 * plane_radius / plane_resolution) + 1
    bw_cross_section = (cross_section >= 0.5).reshape(grid_size, grid_size)

    # Label connected components
    labeled, n_components = label(bw_cross_section, connectivity=1, return_num=True)

    if n_components == 0:
        return None

    # Find the component at the center (should be the object)
    center_idx = bw_cross_section.shape[0] // 2
    center_label = labeled[center_idx, center_idx]

    if center_label == 0:
        # No component at center - find largest
        props = regionprops(labeled)
        if not props:
            return None
        areas = [p.area for p in props]
        center_label = props[np.argmax(areas)].label

    # Get area of the main component
    main_component = (labeled == center_label)
    area = np.sum(main_component)

    # Convert from grid squares to actual area
    area_voxels = area * (plane_resolution ** 2)

    return area_voxels


def area_to_equivalent_radius(area, voxel_size_um=1.0):
    """
    Convert cross-section area to equivalent circular radius.

    Args:
        area: Cross-section area in voxels^2
        voxel_size_um: Voxel size in micrometers (scalar for isotropic)

    Returns:
        Equivalent circular radius in micrometers
    """
    if area is None or area <= 0:
        return None
    radius_voxels = np.sqrt(area / np.pi)
    return radius_voxels * voxel_size_um


def validate_skeleton_points(skeleton_points, binary_volume):
    """
    Validate that skeleton points are inside the binary volume.

    This catches cases where the path tracing escapes the actual
    object boundaries (e.g., skeleton Z=13μm when object only extends to Z=6μm).

    Args:
        skeleton_points: (N, 3) array of skeleton coordinates in voxel units
        binary_volume: 3D binary array of the object

    Returns:
        Boolean mask of valid points (True = inside the object)
    """
    n_points = len(skeleton_points)
    valid = np.ones(n_points, dtype=bool)
    shape = np.array(binary_volume.shape)

    for i, point in enumerate(skeleton_points):
        # Round to nearest voxel
        voxel = np.round(point).astype(int)

        # Check bounds
        if np.any(voxel < 0) or np.any(voxel >= shape):
            valid[i] = False
            continue

        # Check if this voxel is inside the object
        if binary_volume[voxel[0], voxel[1], voxel[2]] == 0:
            valid[i] = False

    return valid


def find_longest_contiguous_segment(valid_mask):
    """
    Find the longest contiguous segment of True values in a boolean mask.

    Args:
        valid_mask: 1D boolean array

    Returns:
        Tuple of (start_index, end_index) for the longest contiguous segment,
        or None if no valid segment found
    """
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 2:
        return None

    # Find gaps in valid indices
    gaps = np.diff(valid_indices)
    segment_starts = np.concatenate([[0], np.where(gaps > 1)[0] + 1])
    segment_ends = np.concatenate([np.where(gaps > 1)[0] + 1, [len(valid_indices)]])

    # Find longest contiguous segment
    segment_lengths = segment_ends - segment_starts
    longest_seg_idx = np.argmax(segment_lengths)
    start_idx = segment_starts[longest_seg_idx]
    end_idx = segment_ends[longest_seg_idx]

    return valid_indices[start_idx], valid_indices[end_idx - 1] + 1
