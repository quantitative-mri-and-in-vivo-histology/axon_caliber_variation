"""
Optimized 3D skeletonization tools.

This is a runtime-optimized implementation of the skeletonization method
described in:
    Abdollahzadeh A, Belevich I, Jokitalo E, Tohka J, Sierra A.
    "DeepACSON automated segmentation of white matter in 3D electron microscopy."
    Commun Biol. 2021 Feb 18;4(1):179.
    https://pubmed.ncbi.nlm.nih.gov/33568775/

Original implementation: https://github.com/aAbdz/DeepACSON
Copyright (c) 2021 Abdollahzadeh et al.
Licensed under the MIT License.

This runtime-optimized version includes modifications for performance:
- Discrete gradient descent (default): ~2.8x faster, voxel-level accuracy
- Numba JIT compilation for inner loops
- Euler integration (optional): subvoxel accuracy, slower
- MaskedArray optimization: avoid expensive filled() operations

MIT License:
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from numba import njit
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
import skfmm


# Precompute 26-neighbor offsets (excluding center)
NEIGHBOR_OFFSETS_26 = np.array([
    [0, 0, 1], [1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1],
    [1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1],
    [0, 0, -1], [1, 0, -1], [-1, 0, -1], [0, 1, -1], [0, -1, -1],
    [1, 1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, -1],
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
    [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]
], dtype=np.int32)

# Precompute distances for each neighbor direction
NEIGHBOR_DIST = np.array([
    np.sqrt(ox**2 + oy**2 + oz**2)
    for ox, oy, oz in NEIGHBOR_OFFSETS_26
], dtype=np.float64)


@njit(cache=True)
def _discrete_shortest_path_jit(D, start_i, start_j, start_k, neighbors, max_iter):
    """
    Numba JIT-compiled path tracing kernel.

    Returns preallocated path array and actual path length.
    """
    sz0, sz1, sz2 = D.shape

    # Preallocate output (max possible length)
    path = np.empty((max_iter, 3), dtype=np.int32)

    # Initialize
    path[0, 0] = start_i
    path[0, 1] = start_j
    path[0, 2] = start_k

    ci, cj, ck = start_i, start_j, start_k
    path_len = 1

    for _ in range(max_iter - 1):
        current_val = D[ci, cj, ck]

        # Already at source
        if current_val == 0.0:
            break

        # Find neighbor with minimum D value
        min_val = current_val
        best_i, best_j, best_k = -1, -1, -1

        for n in range(26):
            ni = ci + neighbors[n, 0]
            nj = cj + neighbors[n, 1]
            nk = ck + neighbors[n, 2]

            # Bounds check (explicit for Numba)
            if ni < 0 or ni >= sz0:
                continue
            if nj < 0 or nj >= sz1:
                continue
            if nk < 0 or nk >= sz2:
                continue

            val = D[ni, nj, nk]
            if val < min_val:
                min_val = val
                best_i, best_j, best_k = ni, nj, nk

        # No descent possible
        if best_i < 0:
            break

        # Add to path
        path[path_len, 0] = best_i
        path[path_len, 1] = best_j
        path[path_len, 2] = best_k
        path_len += 1

        ci, cj, ck = best_i, best_j, best_k

    return path[:path_len]


def discrete_shortest_path(D, start_point):
    """
    Trace shortest path by following steepest descent neighbors.

    Much faster than Euler integration:
    - No gradient field computation
    - No trilinear interpolation
    - No floating point arithmetic per step
    - ~10x fewer iterations (1 per voxel vs ~10 per voxel)
    - Numba JIT-compiled inner loop

    Args:
        D: Travel time field (3D array, 0 at sources)
        start_point: Starting voxel coordinates (array-like)

    Returns:
        path: (N, 3) array of voxel coordinates from start to source
    """
    # Ensure D is contiguous float64 for Numba
    D_arr = np.ascontiguousarray(D, dtype=np.float64)

    # Convert start point
    si, sj, sk = int(start_point[0]), int(start_point[1]), int(start_point[2])

    # Safety bound on iterations
    max_iter = (D_arr.shape[0] + D_arr.shape[1] + D_arr.shape[2]) * 2

    return _discrete_shortest_path_jit(D_arr, si, sj, sk, NEIGHBOR_OFFSETS_26, max_iter)


def compute_gradient_field(D):
    """
    Compute gradient field pointing toward minimum distance neighbor.

    Uses in-place updates: iterate through neighbors, track running minimum,
    update gradients where neighbor is smaller.

    Args:
        D: Distance field (3D array)

    Returns:
        Fx, Fy, Fz: Gradient field components (pointing toward min neighbor)
    """
    sz = D.shape
    max_D = np.max(D)

    # Pad D with max value for boundary handling
    D_pad = np.full((sz[0] + 2, sz[1] + 2, sz[2] + 2), max_D, dtype=np.float64)
    D_pad[1:-1, 1:-1, 1:-1] = D

    # Initialize outputs
    Fx = np.zeros(sz, dtype=np.float64)
    Fy = np.zeros(sz, dtype=np.float64)
    Fz = np.zeros(sz, dtype=np.float64)

    # Track running minimum (copy D to avoid modifying input)
    D_min = D.astype(np.float64).copy()

    # Iterate through 26 neighbors with in-place updates
    for n, (ox, oy, oz) in enumerate(NEIGHBOR_OFFSETS_26):
        # Get shifted neighbor values (view, no copy)
        neighbor = D_pad[1+ox:sz[0]+1+ox, 1+oy:sz[1]+1+oy, 1+oz:sz[2]+1+oz]

        # Find where this neighbor is smaller than current minimum
        improved = neighbor < D_min

        # Update minimum and gradient where improved
        D_min[improved] = neighbor[improved]

        inv_d = 1.0 / NEIGHBOR_DIST[n]
        Fx[improved] = ox * inv_d
        Fy[improved] = oy * inv_d
        Fz[improved] = oz * inv_d

    # Negate for descent direction
    return -Fx, -Fy, -Fz


def euler_shortest_path(D, source_points, start_point, step_size, source_tree=None):
    """
    Trace shortest path from start_point to source using Euler integration.

    Provides subvoxel accuracy but is slower than discrete descent.

    Args:
        D: Distance/travel-time field
        source_points: (N, 3) array of source points
        start_point: (1, 3) starting position
        step_size: Integration step size (typically 0.1)
        source_tree: Optional prebuilt KD-tree for source_points

    Returns:
        path: (M, 3) array of path points (floating point)
    """
    # Compute gradient field once
    Fx, Fy, Fz = compute_gradient_field(D)
    sz = np.array(D.shape)

    # Build KD-tree for source distance queries if not provided
    if source_tree is None:
        source_tree = cKDTree(source_points)

    # Path tracing with list accumulation
    path_list = [start_point[0].copy()]
    current = start_point[0].astype(np.float64).copy()

    max_iter = int(np.sum(sz) * 10)  # Reasonable upper bound

    for itr in range(max_iter):
        # Trilinear interpolation using map_coordinates
        coords = current.reshape(3, 1)
        gx = map_coordinates(Fx, coords, order=1, mode='nearest')[0]
        gy = map_coordinates(Fy, coords, order=1, mode='nearest')[0]
        gz = map_coordinates(Fz, coords, order=1, mode='nearest')[0]

        # Normalize gradient
        mag = np.sqrt(gx*gx + gy*gy + gz*gz) + 1e-6
        gx, gy, gz = gx/mag, gy/mag, gz/mag

        # Euler step (follow negative gradient)
        end_point = current - step_size * np.array([gx, gy, gz])

        # Bounds check
        if np.any(end_point < 0) or np.any(end_point >= sz):
            break

        # Check distance to nearest source (KD-tree query is O(log N))
        min_dist, _ = source_tree.query(end_point)

        # Check for stagnation
        if itr >= 10:
            movement = np.linalg.norm(end_point - path_list[itr - 10])
            if movement < step_size:
                break

        path_list.append(end_point.copy())

        # Reached source
        if min_dist < 10 * step_size:
            _, closest_idx = source_tree.query(end_point)
            path_list.append(source_points[closest_idx].copy())
            break

        current = end_point

    return np.array(path_list)


def get_line_length(L):
    """Compute total length of a polyline."""
    if len(L) < 2:
        return 0.0
    diffs = L[1:] - L[:-1]
    return np.sum(np.sqrt(np.sum(diffs**2, axis=1)))


def organize_skeleton(skel_seg, length_th):
    """
    Organize and clean up skeleton segments.

    - Removes segments shorter than length_th
    - Splits segments at junction points
    """
    if not skel_seg:
        return []

    final_skeleton = []
    n = len(skel_seg)

    # Collect all endpoints
    end_points = np.zeros((n * 2, 3))
    for i in range(n):
        ss = skel_seg[i]
        end_points[i * 2] = ss[0]
        end_points[i * 2 + 1] = ss[-1]

    connecting_distance = 2

    for i in range(n):
        ss = np.asarray(skel_seg[i])

        # Compute distances from all endpoints to this segment
        # Vectorized distance computation
        D = np.sum((end_points[:, None, :] - ss[None, :, :])**2, axis=2)

        # Find endpoints close to this segment (excluding own endpoints)
        min_dists = np.amin(D, axis=1)
        check = min_dists < connecting_distance
        check[i * 2] = False
        check[i * 2 + 1] = False

        # Find cut points
        cut_skel = [0, len(ss)]
        if np.any(check):
            for ii in np.where(check)[0]:
                min_ind = np.argmin(D[ii])
                if 2 < min_ind < len(ss) - 2:
                    cut_skel.append(min_ind)

        # Split and filter segments
        cut_skel = sorted(cut_skel)
        for j in range(len(cut_skel) - 1):
            segment = ss[cut_skel[j]:cut_skel[j + 1]]
            if get_line_length(segment) >= length_th:
                final_skeleton.append(segment)

    return final_skeleton


def extract_skeleton(volume, verbose=False, path_method='discrete', step_size=0.1, fmm_order=2):
    """
    Extract skeleton from binary volume using fast marching.

    Args:
        volume: Binary 3D volume (1 = object, 0 = background)
        verbose: Print progress information
        path_method: 'discrete' (fast, voxel-level) or 'euler' (slow, subvoxel)
        step_size: Step size for Euler integration (only used if path_method='euler')
        fmm_order: Finite difference order (1 or 2). order=2 is more accurate.

    Returns:
        List of skeleton segments, each a (N, 3) array of coordinates
    """
    if path_method not in ('discrete', 'euler'):
        raise ValueError(f"path_method must be 'discrete' or 'euler', got '{path_method}'")

    # Compute distance from boundary
    boundary_dist = skfmm.distance(volume.astype(np.float64), order=fmm_order)

    # Find medial axis point (max distance from boundary)
    source_point = np.unravel_index(np.argmax(boundary_dist), boundary_dist.shape)
    max_D = boundary_dist[source_point]

    # Handle degenerate case: very thin axons where max_D ≈ 0
    if max_D < 1e-6:
        raise ValueError(f"Object too thin for skeletonization (max boundary distance = {max_D:.2e})")

    # Speed image based on distance (faster near center)
    speed_im = (boundary_dist / max_D) ** 1.5

    # Initialize arrival time field
    Ax_work = np.ones(volume.shape, dtype=np.float64)
    Ax_work[source_point] = 0

    flag = True
    skeleton_segments = []
    length_threshold = 0
    source_points_list = [np.array(source_point, dtype=np.float64)]

    while True:
        # Compute travel time from current sources
        D = skfmm.travel_time(Ax_work, speed_im, order=fmm_order)

        # Find farthest point - avoid MaskedArray overhead
        # skfmm sets masked values to 0 in D.data, so we can use the data directly
        if isinstance(D, np.ma.MaskedArray):
            # Use underlying data array (masked values are already 0)
            D_data = D.data
            end_point = np.unravel_index(np.argmax(D_data), D_data.shape)
            max_dist = D_data[end_point]
            # Replace masked values with max_dist for path tracing
            D_data[D.mask] = max_dist
            D = D_data
        else:
            end_point = np.unravel_index(np.argmax(D), D.shape)
            max_dist = D[end_point]

        # Trace path back to sources
        if path_method == 'discrete':
            shortest_line = discrete_shortest_path(D, np.array(end_point))
        else:  # euler
            source_points = np.array(source_points_list)
            source_tree = cKDTree(source_points)
            end_point_arr = np.array(end_point, ndmin=2, dtype=np.float64)
            shortest_line = euler_shortest_path(
                D, source_points, end_point_arr,
                step_size=step_size, source_tree=source_tree
            )

        line_length = get_line_length(shortest_line)

        if verbose:
            print(f"  Segment length: {line_length:.1f}")

        # Set length threshold based on first segment
        if flag:
            length_threshold = min(40 * max_D, 0.18 * line_length)
            flag = False

        # Stop if segment too short
        if line_length <= length_threshold:
            break

        skeleton_segments.append(shortest_line)

        # Mark path as source
        if path_method == 'discrete':
            # Integer coordinates - use directly
            valid_mask = np.all(
                (shortest_line >= 0) & (shortest_line < np.array(Ax_work.shape)),
                axis=1
            )
            path_voxels = shortest_line[valid_mask]
            if len(path_voxels) > 0:
                Ax_work[path_voxels[:, 0], path_voxels[:, 1], path_voxels[:, 2]] = 0
        else:
            # Floating point coordinates - floor to integers
            path_voxels = np.floor(shortest_line).astype(int)
            valid_mask = np.all(
                (path_voxels >= 0) & (path_voxels < np.array(Ax_work.shape)),
                axis=1
            )
            path_voxels = path_voxels[valid_mask]
            if len(path_voxels) > 0:
                Ax_work[path_voxels[:, 0], path_voxels[:, 1], path_voxels[:, 2]] = 0

            # Also update source points list for euler method
            for pt in shortest_line:
                source_points_list.append(pt)

    # Clean up skeleton
    if skeleton_segments:
        final_skeleton = organize_skeleton(skeleton_segments, length_threshold)
    else:
        final_skeleton = []

    return final_skeleton


# Backwards compatibility alias
skeleton = extract_skeleton


def warmup():
    """Trigger Numba JIT compilation on a tiny test case."""
    # Small test array to trigger compilation
    D_test = np.ones((5, 5, 5), dtype=np.float64)
    D_test[2, 2, 2] = 0.0  # Source
    D_test[0, 0, 0] = 10.0  # Start point

    # This call triggers JIT compilation (cached afterward)
    _ = discrete_shortest_path(D_test, np.array([0, 0, 0]))
