#!/usr/bin/env python3
"""
Compute fiber morphometry profiles by sampling perpendicular cross-sections along skeletons.

Uses the DeepACSON CSD approach (Abdollahzadeh et al., 2019) for skeleton
extraction and cross-section sampling, reimplemented here without external
package dependency.

The algorithm (per axon):
1. Extract skeleton via FMM + Euler backtracking (step_size=0.1 voxels)
2. Organize skeleton: break at junctions, filter by length threshold
3. For each skeleton segment: sample perpendicular cross-sections using
   fused Numba kernel (rotation + nearest-neighbor interpolation + flood-fill)
4. Measure cross-section area via flood-fill pixel count → equivalent circular radius

Supports both OME-Zarr volumes (canonical format from preparation pipeline)
and legacy .mat files.

Usage:
  # Single Zarr volume (canonical format)
  python compute_3d_axon_profiles.py data/processed/rat/LM/sham_25_ipsi_cc_myelin.zarr \\
      data/processed/rat/LM/sham_25_ipsi_cc_axon_profiles.npz

  # Batch mode with glob pattern
  python compute_3d_axon_profiles.py "data/processed/rat/LM/*_myelin.zarr" \\
      data/processed/rat/LM/ --output-suffix _axon_profiles

  # Legacy .mat file
  python compute_3d_axon_profiles.py data/raw/rat/LM/LM_25_ipsi_myelinated_axons.mat \\
      data/processed/rat/LM/sham_25_ipsi_axon_profiles.npz
"""

import argparse
import glob
import logging
import os
import traceback
from pathlib import Path
import multiprocessing as mp
from typing import Tuple, Union, Optional, List

import numpy as np
import skfmm
from numba import njit
from scipy.ndimage import find_objects, map_coordinates, median_filter
from scipy.spatial import cKDTree
from tqdm import tqdm

# Import from axonometry library (for .mat loading only)
import sys
_root = Path(__file__).resolve().parent
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

from axonometry import (
    load_volume_with_metadata,
    resample_to_isotropic,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global variables for worker processes (inherited via fork)
# ---------------------------------------------------------------------------

_shared_volume = None
_shared_bboxes = None
_base_grid = None      # Precomputed XY sampling grid (N, 2)
_grid_size = None      # Side length of the 2D grid
_g_radius = None        # Grid radius in voxels (computed from max_radius_um / voxel_size)
_g_res = None           # Grid resolution in voxels (default 0.25, matching DeepACSON)
_step_voxels = None     # Step size for skeleton subsampling (in voxels)
_voxel_size_um = None   # Isotropic voxel size in micrometers


# ===========================================================================
# Skeleton extraction (from DeepACSON/CSD/skeleton3D.py)
# ===========================================================================

@njit(cache=True)
def _compute_gradient_field(D):
    """Compute gradient field pointing toward minimum-distance neighbor.

    For each voxel, finds the 26-connected neighbor with the smallest value
    in the travel-time field and records the unit direction toward it.
    Reads from padded snapshot J; does not mutate D.
    """
    sz0, sz1, sz2 = D.shape
    max_D = np.max(D)
    Fx = np.zeros((sz0, sz1, sz2))
    Fy = np.zeros((sz0, sz1, sz2))
    Fz = np.zeros((sz0, sz1, sz2))

    J = np.full((sz0+2, sz1+2, sz2+2), max_D)
    J[1:sz0+1, 1:sz1+1, 1:sz2+1] = D

    nx = np.array([0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 1,-1, 0, 0, 1, 1,-1,-1, 1,-1, 0, 0, 1, 1,-1,-1])
    ny = np.array([0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1])
    nz = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0])

    fx_tab = np.empty(26)
    fy_tab = np.empty(26)
    fz_tab = np.empty(26)
    for i in range(26):
        den = (nx[i]*nx[i] + ny[i]*ny[i] + nz[i]*nz[i])**0.5
        fx_tab[i] = nx[i] / den
        fy_tab[i] = ny[i] / den
        fz_tab[i] = nz[i] / den

    for a in range(sz0):
        for b in range(sz1):
            for c in range(sz2):
                best_val = D[a, b, c]
                best_i = -1
                for i in range(26):
                    val = J[1+a+nx[i], 1+b+ny[i], 1+c+nz[i]]
                    if val < best_val:
                        best_val = val
                        best_i = i
                if best_i >= 0:
                    Fx[a, b, c] = fx_tab[best_i]
                    Fy[a, b, c] = fy_tab[best_i]
                    Fz[a, b, c] = fz_tab[best_i]

    return Fx, Fy, Fz


def _euler_shortest_path(D, source_points, start_point, step_size):
    """Trace shortest path via Euler integration on the gradient field.

    Uses map_coordinates for trilinear interpolation and cKDTree for
    fast source-distance queries. Same algorithm as DeepACSON euler path.
    """
    Fx, Fy, Fz = _compute_gradient_field(D)
    # Negate for descent direction
    Fx = -Fx; Fy = -Fy; Fz = -Fz
    sz = np.array(D.shape)

    source_tree = cKDTree(source_points)

    path_list = [start_point[0].copy()]
    current = start_point[0].astype(np.float64).copy()

    max_iter = int(np.sum(sz) * 10)

    for itr in range(max_iter):
        coords = current.reshape(3, 1)
        gx = map_coordinates(Fx, coords, order=1, mode='nearest')[0]
        gy = map_coordinates(Fy, coords, order=1, mode='nearest')[0]
        gz = map_coordinates(Fz, coords, order=1, mode='nearest')[0]

        mag = np.sqrt(gx*gx + gy*gy + gz*gz) + 1e-6
        gx, gy, gz = gx/mag, gy/mag, gz/mag

        end_point = current - step_size * np.array([gx, gy, gz])

        if np.any(end_point < 0) or np.any(end_point >= sz):
            break

        min_dist, _ = source_tree.query(end_point)

        if itr >= 10:
            movement = np.linalg.norm(end_point - path_list[itr - 10])
            if movement < step_size:
                break

        path_list.append(end_point.copy())

        if min_dist < 10 * step_size:
            _, closest_idx = source_tree.query(end_point)
            path_list.append(source_points[closest_idx].copy())
            break

        current = end_point

    return np.array(path_list)


def _get_line_length(L):
    dist = np.sum(np.sum((L[1:] - L[:-1])**2,axis=1)**0.5)
    return dist


def _organize_skeleton(skel_seg, length_th):
    final_skeleton = []

    n = len(skel_seg)
    end_points = np.zeros((n*2,3))

    l = 0
    for i in range(n):
        ss = skel_seg[i]
        l = max(l,len(ss))
        end_points[i*2] = ss[0]
        end_points[i*2+1] = ss[-1]

    connecting_distance = 2

    for i in range(n):

        ss = np.asarray(skel_seg[i])

        ex = np.reshape(end_points[:,0],(-1,1)); ex = np.repeat(ex,len(ss),axis=1)
        sx = np.reshape(ss[:,0],(1,-1)); sx = np.repeat(sx,len(end_points),axis=0)

        ey = np.reshape(end_points[:,1],(-1,1)); ey = np.repeat(ey,len(ss),axis=1)
        sy = np.reshape(ss[:,1],(1,-1)); sy = np.repeat(sy,len(end_points),axis=0)

        ez = np.reshape(end_points[:,2],(-1,1)); ez = np.repeat(ez,len(ss),axis=1)
        sz = np.reshape(ss[:,2],(1,-1)); sz = np.repeat(sz,len(end_points),axis=0)

        D = (ex-sx)**2 + (ey-sy)**2 + (ez-sz)**2

        check = np.amin(D, axis=1) < connecting_distance
        check[i*2] = False
        check[i*2+1] = False

        cut_skel = [0,len(ss)]
        if(any(check)):
            for ii in range(len(check)):
                if(check[ii]):
                    line = D[ii]
                    min_ind = np.ma.argmin(line)
                    if((min_ind>2) and (min_ind<(len(line)-2))):
                        cut_skel.append(min_ind)

        cut_skel = sorted(cut_skel)
        for j in range(len(cut_skel)-1):
            skel_breaked_seg = ss[cut_skel[j]:cut_skel[j+1]]
            length_skel_seg = _get_line_length(skel_breaked_seg)
            if(length_skel_seg >= length_th):
               final_skeleton.append(skel_breaked_seg)

    return final_skeleton


def _extract_skeleton(Ax):
    """FMM + Euler backtracking skeleton extraction (DeepACSON).

    Returns:
        (skeleton_segments, maxD) where maxD is the maximum inscribed radius
        in voxels (from the boundary distance field).
    """
    boundary_dist = skfmm.distance(Ax)

    source_point = np.unravel_index(np.argmax(boundary_dist), boundary_dist.shape)
    maxD = boundary_dist[source_point]

    if maxD < 1e-6:
        return [], 0.0

    speed_im = (boundary_dist / maxD) ** 1.5

    Ax_work = np.ones(Ax.shape, dtype=np.float64)
    Ax_work[source_point] = 0

    flag = True
    skeleton_segments = []
    source_points_list = [np.array(source_point, dtype=np.float64)]

    while True:
        D = skfmm.travel_time(Ax_work, speed_im)

        # Avoid MaskedArray overhead — work on underlying data directly
        if isinstance(D, np.ma.MaskedArray):
            D_data = D.data
            end_point = np.unravel_index(np.argmax(D_data), D_data.shape)
            max_dist = D_data[end_point]
            D_data[D.mask] = max_dist
            D = D_data
        else:
            end_point = np.unravel_index(np.argmax(D), D.shape)
            max_dist = D[end_point]

        source_points = np.array(source_points_list)
        end_point_arr = np.array(end_point, ndmin=2, dtype=np.float64)
        shortest_line = _euler_shortest_path(D, source_points, end_point_arr, step_size=0.1)

        line_length = _get_line_length(shortest_line)

        if flag:
            length_threshold = min(40 * maxD, 0.18 * line_length)
            flag = False

        if line_length <= length_threshold:
            break

        skeleton_segments.append(shortest_line)

        # Mark path voxels as source
        path_voxels = np.floor(shortest_line).astype(int)
        valid_mask = np.all(
            (path_voxels >= 0) & (path_voxels < np.array(Ax_work.shape)),
            axis=1
        )
        path_voxels = path_voxels[valid_mask]
        if len(path_voxels) > 0:
            Ax_work[path_voxels[:, 0], path_voxels[:, 1], path_voxels[:, 2]] = 0

        for pt in shortest_line:
            source_points_list.append(pt)

    if len(skeleton_segments) != 0:
        final_skeleton = _organize_skeleton(skeleton_segments, length_threshold)
    else:
        final_skeleton = []

    return final_skeleton, maxD


# ===========================================================================
# Tangent vectors (from DeepACSON/CSD/unit_tangent_vector.py)
# ===========================================================================

def _unit_tangent_vector(curve):
    d_curve = np.gradient(curve, axis=0)
    ds = np.expand_dims((np.sum(d_curve**2, axis=1))**0.5, axis=1)
    ds[ds==0] = 1e-5
    u_tang_vec = d_curve/np.repeat(ds, curve.shape[1], axis=1)
    return u_tang_vec



# ===========================================================================
# Cross-section sampling (fused Numba kernel)
# ===========================================================================

def precompute_grid(g_radius, g_res=0.25):
    """Precompute the XY sampling grid (called once).

    Args:
        g_radius: Grid radius in voxels
        g_res: Grid resolution in voxels (default 0.25, matching DeepACSON)

    Returns:
        base_grid: (N, 2) array of grid coordinates in voxel units
        grid_size: Number of points per side
    """
    coords = np.arange(-g_radius, g_radius, g_res)
    x, y = np.meshgrid(coords, coords)
    base_grid = np.ascontiguousarray(
        np.stack([x.ravel(), y.ravel()], axis=1), dtype=np.float64
    )
    grid_size = len(coords)
    return base_grid, grid_size


@njit(cache=True, fastmath=True)
def _sample_and_flood_fill(volume, base_grid, grid_size,
                           point_0, point_1, point_2,
                           tan_0, tan_1, tan_2):
    """
    Fused kernel: rotation + translation + trilinear interpolation + flood-fill.

    Single Numba function with no Python round-trips and no intermediate arrays.
    Uses trilinear interpolation (matching DeepACSON) with threshold >= 0.5.
    Rejects cross-sections spanning fewer than 4 pixels in either axis.
    Returns cross-section area in pixels, or 0 if invalid.
    """
    n_grid = base_grid.shape[0]
    sz0, sz1, sz2 = volume.shape

    # Compute rotation matrix (Euler-Rodrigues) from tangent to z-axis
    dot_zt = tan_2

    if dot_zt > 0.9999:
        r00 = 1.0; r01 = 0.0; r02 = 0.0
        r10 = 0.0; r11 = 1.0; r12 = 0.0
        r20 = 0.0; r21 = 0.0; r22 = 1.0
    elif dot_zt < -0.9999:
        r00 = 1.0; r01 = 0.0; r02 = 0.0
        r10 = 0.0; r11 = 1.0; r12 = 0.0
        r20 = 0.0; r21 = 0.0; r22 = -1.0
    else:
        rn = np.sqrt(tan_0 * tan_0 + tan_1 * tan_1)
        rx = -tan_1 / rn
        ry = tan_0 / rn

        clamped = max(-1.0, min(1.0, dot_zt))
        theta = np.arccos(clamped)

        a = np.cos(theta * 0.5)
        s = np.sin(theta * 0.5)
        b = -rx * s
        c = -ry * s

        aa = a * a; bb = b * b; cc = c * c
        bc = b * c; ac = a * c; ab = a * b

        r00 = aa + bb - cc;  r01 = 2.0 * bc;       r02 = -2.0 * ac
        r10 = 2.0 * bc;      r11 = aa + cc - bb;    r12 = 2.0 * ab
        r20 = 2.0 * ac;      r21 = -2.0 * ab;       r22 = aa - bb - cc

    # Rotate grid, translate to skeleton point, trilinear interpolation
    bw = np.zeros((grid_size, grid_size), dtype=np.uint8)

    for i in range(n_grid):
        gx = base_grid[i, 0]
        gy = base_grid[i, 1]

        sx = r00 * gx + r01 * gy + point_0
        sy = r10 * gx + r11 * gy + point_1
        sz = r20 * gx + r21 * gy + point_2

        # Trilinear interpolation
        ix0 = int(np.floor(sx))
        iy0 = int(np.floor(sy))
        iz0 = int(np.floor(sz))
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        iz1 = iz0 + 1

        if ix0 >= 0 and ix1 < sz0 and iy0 >= 0 and iy1 < sz1 and iz0 >= 0 and iz1 < sz2:
            fx = sx - ix0
            fy = sy - iy0
            fz = sz - iz0
            fx1 = 1.0 - fx
            fy1 = 1.0 - fy
            fz1 = 1.0 - fz

            val = (float(volume[ix0, iy0, iz0]) * fx1 * fy1 * fz1 +
                   float(volume[ix1, iy0, iz0]) * fx  * fy1 * fz1 +
                   float(volume[ix0, iy1, iz0]) * fx1 * fy  * fz1 +
                   float(volume[ix0, iy0, iz1]) * fx1 * fy1 * fz  +
                   float(volume[ix1, iy1, iz0]) * fx  * fy  * fz1 +
                   float(volume[ix1, iy0, iz1]) * fx  * fy1 * fz  +
                   float(volume[ix0, iy1, iz1]) * fx1 * fy  * fz  +
                   float(volume[ix1, iy1, iz1]) * fx  * fy  * fz)

            if val >= 0.5:
                row = i // grid_size
                col = i % grid_size
                bw[row, col] = 1

    # Flood-fill from center (4-connected)
    center = grid_size // 2
    if not bw[center, center]:
        return 0

    queue_r = np.empty(grid_size * grid_size, dtype=np.int32)
    queue_c = np.empty(grid_size * grid_size, dtype=np.int32)
    visited = np.zeros((grid_size, grid_size), dtype=np.uint8)

    queue_r[0] = center
    queue_c[0] = center
    visited[center, center] = 1
    head = 0
    tail = 1
    area = 0

    while head < tail:
        r = queue_r[head]
        c_ = queue_c[head]
        head += 1
        area += 1

        if r > 0 and not visited[r - 1, c_] and bw[r - 1, c_]:
            visited[r - 1, c_] = 1
            queue_r[tail] = r - 1
            queue_c[tail] = c_
            tail += 1
        if r < grid_size - 1 and not visited[r + 1, c_] and bw[r + 1, c_]:
            visited[r + 1, c_] = 1
            queue_r[tail] = r + 1
            queue_c[tail] = c_
            tail += 1
        if c_ > 0 and not visited[r, c_ - 1] and bw[r, c_ - 1]:
            visited[r, c_ - 1] = 1
            queue_r[tail] = r
            queue_c[tail] = c_ - 1
            tail += 1
        if c_ < grid_size - 1 and not visited[r, c_ + 1] and bw[r, c_ + 1]:
            visited[r, c_ + 1] = 1
            queue_r[tail] = r
            queue_c[tail] = c_ + 1
            tail += 1

    # Size filter: reject cross-sections spanning < 4 pixels in either axis
    # (matches DeepACSON's nz_X/nz_Y filter)
    min_r = grid_size
    max_r = 0
    min_c = grid_size
    max_c = 0
    for idx in range(tail):
        rv = queue_r[idx]
        cv = queue_c[idx]
        if rv < min_r:
            min_r = rv
        if rv > max_r:
            max_r = rv
        if cv < min_c:
            min_c = cv
        if cv > max_c:
            max_c = cv

    extent_r = max_r - min_r + 1
    extent_c = max_c - min_c + 1
    if extent_r < 4 or extent_c < 4:
        return 0

    return area


# ===========================================================================
# Bounding boxes and outlier filtering
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


def filter_radius_outliers(radii, window_size=5, threshold=3.0):
    """
    Filter outlier radius measurements using local median comparison.

    Replaces values that are significantly larger than the local median
    with the local median value.
    """
    if len(radii) < window_size:
        return radii

    if window_size % 2 == 0:
        window_size += 1

    local_median = median_filter(radii, size=window_size, mode='reflect')
    outlier_mask = radii > threshold * np.maximum(local_median, 0.01)

    radii_filtered = radii.copy()
    radii_filtered[outlier_mask] = local_median[outlier_mask]

    n_replaced = np.sum(outlier_mask)
    if n_replaced > 0:
        logger.debug(f"  Replaced {n_replaced}/{len(radii)} outlier radius measurements")

    return radii_filtered


# ===========================================================================
# Per-axon processing
# ===========================================================================

def init_worker():
    """Initialize worker process — trigger Numba JIT compilation."""
    D = np.ones((3, 3, 3))
    _compute_gradient_field(D)
    tiny_vol = np.ones((3, 3, 3), dtype=np.uint8)
    tiny_grid = np.array([[0.0, 0.0]], dtype=np.float64)
    _sample_and_flood_fill(tiny_vol, tiny_grid, 1, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0)


def process_single_axon(args):
    """
    Process a single axon using the DeepACSON CSD approach.

    Args:
        args: Tuple of (axon_label,)

    Returns:
        Dict with radius profile and skeleton coords, or None if failed
    """
    (axon_label,) = args

    try:
        bbox = _shared_bboxes.get(axon_label)
        if bbox is None:
            return None

        min_coords, max_coords = bbox
        vol_shape = np.array(_shared_volume.shape)

        # Tight crop for skeleton extraction (only needs axon voxels + small border)
        skel_pad = 5
        min_tight = np.maximum(min_coords - skel_pad, 0)
        max_tight = np.minimum(max_coords + skel_pad, vol_shape)

        subvol_tight = _shared_volume[min_tight[0]:max_tight[0],
                                      min_tight[1]:max_tight[1],
                                      min_tight[2]:max_tight[2]]
        binary_tight = (subvol_tight == axon_label).astype(np.float64)

        n_voxels = np.count_nonzero(binary_tight)
        if n_voxels < 100:
            return None

        # Skeleton extraction (FMM + Euler backtracking) on tight crop
        try:
            skel_segments, maxD = _extract_skeleton(binary_tight)
        except Exception as e:
            logger.debug(f"Axon {axon_label}: skeleton extraction failed - {e}")
            return None

        if len(skel_segments) == 0:
            return None

        # Adaptive cross-section radius from skeleton's max inscribed radius.
        # maxD is the max boundary distance in voxels; use it + margin for
        # the sampling grid instead of the fixed global g_radius.
        axon_g_radius = int(np.ceil(maxD)) + 5
        axon_grid, axon_grid_size = precompute_grid(axon_g_radius, _g_res)

        # Wide crop for cross-section sampling (needs axon_g_radius padding)
        sampling_pad = axon_g_radius + 5
        min_padded = np.maximum(min_coords - sampling_pad, 0)
        max_padded = np.minimum(max_coords + sampling_pad, vol_shape)

        subvol_wide = _shared_volume[min_padded[0]:max_padded[0],
                                     min_padded[1]:max_padded[1],
                                     min_padded[2]:max_padded[2]]
        binary_u8 = (subvol_wide == axon_label).astype(np.uint8)

        # Offset to map skeleton coords from tight crop to wide crop space
        tight_to_wide_offset = min_tight - min_padded

        # Process each skeleton segment — sample cross-sections
        all_radii_um = []
        all_skel_coords_um = []
        main_length_um = 0.0

        for skel_seg in skel_segments:
            if len(skel_seg) < 3:
                continue

            # Subsample skeleton points to match desired step size
            if _step_voxels is not None:
                stride = max(1, round(_step_voxels / 0.1))
                skel_seg = skel_seg[::stride]
                if len(skel_seg) < 3:
                    continue

            # Map skeleton from tight crop to wide crop coordinates
            skel_seg_wide = skel_seg + tight_to_wide_offset

            # Compute tangent vectors
            tangent_vecs = _unit_tangent_vector(skel_seg_wide)

            radii = []
            skel_coords = []

            for point, tangent in zip(skel_seg_wide, tangent_vecs):
                if tangent[0] == 0.0 and tangent[1] == 0.0 and tangent[2] == 0.0:
                    continue

                area = _sample_and_flood_fill(
                    binary_u8, axon_grid, axon_grid_size,
                    point[0], point[1], point[2],
                    tangent[0], tangent[1], tangent[2],
                )

                if area > 0:
                    radius_voxels = np.sqrt(area * (_g_res ** 2) / np.pi)
                    radius_um = radius_voxels * _voxel_size_um
                    radii.append(radius_um)
                    global_coord = (point + min_padded) * _voxel_size_um
                    skel_coords.append(global_coord)

            if len(radii) < 2:
                continue

            radii = np.array(radii)
            radii = filter_radius_outliers(radii, window_size=5, threshold=3.0)

            # Track main trunk length (longest segment)
            seg_length = np.sum(np.sqrt(np.sum(np.diff(skel_seg, axis=0)**2, axis=1))) * _voxel_size_um
            if seg_length > main_length_um:
                main_length_um = seg_length

            all_radii_um.extend(radii)
            all_skel_coords_um.extend(skel_coords)

        if len(all_radii_um) < 2:
            return None

        all_radii_um = np.array(all_radii_um)

        return {
            'label': axon_label,
            'radii_um': all_radii_um,
            'skeleton_um': np.array(all_skel_coords_um),
            'n_points': len(all_radii_um),
            'n_segments': len(skel_segments),
            'mean_radius_um': np.mean(all_radii_um),
            'std_radius_um': np.std(all_radii_um),
            'length_um': main_length_um,
        }

    except Exception as e:
        logger.error(f"Axon {axon_label}: Processing failed - {e}\n{traceback.format_exc()}")
        return None


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
# Main orchestration
# ===========================================================================

def compute_fiber_profiles(input_path: Path,
                           output_file: Path,
                           voxel_size_um: Optional[Union[float, Tuple[float, float, float]]] = None,
                           max_radius_um: float = 5.0,
                           step_size_um: float = 0.05,
                           n_jobs: int = -1,
                           max_axons: int = 0,
                           anisotropy_mode: str = 'simple'):
    """
    Compute morphometry profiles for all fibers in a labeled volume.

    Args:
        input_path: Path to .zarr directory or .mat file with labeled axons
        output_file: Path to save results (.npz)
        voxel_size_um: Voxel size in micrometers (scalar or (vz, vy, vx) tuple).
                       If None, auto-detected from Zarr metadata or companion JSON.
        max_radius_um: Maximum expected axon radius in micrometers (sets grid size, default: 5.0)
        step_size_um: Step size along skeleton in micrometers
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        max_axons: Maximum number of axons to process (0 = all)
        anisotropy_mode: 'simple' (resample to isotropic) or 'none' (use geometric mean).
                         Ignored for Zarr input (already isotropic by convention).
    """
    # Warmup Numba JIT
    logger.info("Warming up Numba JIT compilation...")
    _compute_gradient_field(np.ones((3, 3, 3)))
    tiny_vol = np.ones((3, 3, 3), dtype=np.uint8)
    tiny_grid = np.array([[0.0, 0.0]], dtype=np.float64)
    _sample_and_flood_fill(tiny_vol, tiny_grid, 1, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0)

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
    voxel_size = (vz * vy * vx) ** (1/3)  # geometric mean for isotropic equiv

    # Convert max_radius from μm to voxels for grid sizing
    g_radius = int(np.ceil(max_radius_um / voxel_size))

    logger.info(f"Max axon radius: {max_radius_um:.1f} μm = {g_radius} voxels")

    # Compute bounding boxes
    bboxes = compute_bounding_boxes(volume)
    axon_labels = np.array(sorted(bboxes.keys()))

    logger.info(f"Found {len(axon_labels)} axons")

    if max_axons > 0:
        axon_labels = axon_labels[:max_axons]
        logger.info(f"Processing first {max_axons} axons")

    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    # Precompute sampling grid (g_res=0.25 voxels, matching DeepACSON)
    g_res = 0.25
    base_grid, grid_size = precompute_grid(g_radius, g_res)

    step_voxels = step_size_um / voxel_size if step_size_um is not None else None

    # Set globals for workers
    global _shared_volume, _shared_bboxes
    global _base_grid, _grid_size
    global _g_radius, _g_res, _step_voxels, _voxel_size_um
    _shared_volume = volume
    _shared_bboxes = bboxes
    _base_grid = base_grid
    _grid_size = grid_size
    _g_radius = g_radius
    _g_res = g_res
    _step_voxels = step_voxels
    _voxel_size_um = voxel_size

    args_list = [(lbl,) for lbl in axon_labels]

    logger.info(f"Voxel size: {voxel_size:.4f} μm")
    logger.info(f"Processing {len(args_list)} axons with {n_jobs} workers")
    logger.info(f"Parameters: g_radius={g_radius} voxels, "
                f"step={step_size_um} μm = {step_voxels:.1f} voxels "
                f"(stride ~{max(1, round(step_voxels / 0.1))})")

    results = []
    if n_jobs == 1:
        for args in tqdm(args_list, desc="Processing axons"):
            result = process_single_axon(args)
            if result is not None:
                results.append(result)
    else:
        with mp.Pool(n_jobs, initializer=init_worker) as pool:
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
        method='deepacson_csd',
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
    n_jobs: int,
    max_axons: int,
    anisotropy_mode: str,
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
                n_jobs=n_jobs,
                max_axons=max_axons,
                anisotropy_mode=anisotropy_mode,
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
    parser.add_argument('--voxel-size', type=parse_voxel_size_arg, default=None,
                        help='Voxel size in μm: single value (isotropic) or vz,vy,vx. '
                             'If not specified, loads from companion .json file (default: from JSON or 0.05)')
    parser.add_argument('--max-radius', type=float, default=5.0,
                        help='Maximum expected axon radius in μm (sets cross-section grid size, default: 5.0)')
    parser.add_argument('--step-size', type=float, default=0.05,
                        help='Step size along skeleton in μm (default: 0.05)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 = all CPUs)')
    parser.add_argument('--max-axons', type=int, default=0,
                        help='Maximum axons to process (0 = all)')
    parser.add_argument('--anisotropy-mode', type=str, default='simple',
                        choices=['simple', 'none'],
                        help="'simple' resamples to isotropic, 'none' uses geometric mean")
    parser.add_argument('--output-suffix', type=str, default='_axon_profiles',
                        help='Suffix to append to output filenames in batch mode (default: "_axon_profiles")')

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
            n_jobs=args.n_jobs,
            max_axons=args.max_axons,
            anisotropy_mode=args.anisotropy_mode,
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
            n_jobs=args.n_jobs,
            max_axons=args.max_axons,
            anisotropy_mode=args.anisotropy_mode,
            output_suffix=args.output_suffix,
        )
