"""
DeepACSON skeleton extraction and cross-section sampling — optimized version.

Started as a verbatim copy of deepacson.py. Performance and correctness
improvements will be applied here incrementally, each marked with:
    # --- MODIFIED ---
    # Reason: ...
    # Original: ...
    # ---

Original implementation: https://github.com/aAbdz/DeepACSON
Citation:
    Abdollahzadeh A, Belevich I, Jokitalo E, Tohka J, Sierra A.
    "DeepACSON automated segmentation of white matter in 3D electron microscopy."
    Commun Biol. 2021 Feb 18;4(1):179.

MIT License

Copyright (c) 2020 Ali Abdollahzadeh

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
import skfmm
import sys
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.ndimage import map_coordinates
from numba import njit
from skimage.measure import label, regionprops


# ===================================================================
# plane_rotation.py (verbatim)
# ===================================================================

def rotate_vector(vector, rot_mat):

    "rotating a vector by a rotation matrix"
    rotated_vec = np.dot(vector,rot_mat)
    return rotated_vec


def rotation_matrix_3D(vector, theta):

    """counterclockwise rotation about a unit vector by theta radians using
    Euler-Rodrigues formula: https://en.wikipedia.org/wiki/Euler-Rodrigues_formula"""

    a=np.cos(theta/2.0)
    b,c,d=-vector*np.sin(theta/2.0)
    aa,bb,cc,dd=a**2, b**2, c**2, d**2
    bc,ad,ac,ab,bd,cd=b*c, a*d, a*c, a*b, b*d, c*d

    rot_mat=np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    return rot_mat

def unit_normal_vector(vec1, vec2):

    n = np.cross(vec1, vec2)
    if np.array_equal(n, np.array([0, 0, 0])):
        n = vec1

    s = max(np.sqrt(np.dot(n,n)), 1e-5)
    n = n/s
    return n

def angle(vec1, vec2):

    theta=np.arccos(np.dot(vec1,vec2) / (np.sqrt(np.dot(vec1,vec1)) * np.sqrt(np.dot(vec2, vec2))))
    return theta


# ===================================================================
# unit_tangent_vector.py (verbatim)
# ===================================================================

def unit_tangent_vector(curve):

    d_curve = np.gradient(curve, axis=0)
    ds = np.expand_dims((np.sum(d_curve**2, axis=1))**0.5, axis=1)
    ds[ds==0] = 1e-5
    u_tang_vec = d_curve/np.repeat(ds, curve.shape[1], axis=1)
    return u_tang_vec


# ===================================================================
# skeleton3D.py (verbatim except where marked)
# ===================================================================

def discrete_shortest_path(D,start_point):

    sz = D.shape
    x = [0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 1,-1, 0, 0, 1, 1,-1,-1, 1,-1, 0, 0, 1, 1,-1,-1]
    y = [0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1]
    z = [1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0]

    path = [start_point]

    min_v = np.inf
    while(min_v!=0):

        neighbor_inx = np.array((x,y,z)).T
        ngb = start_point + neighbor_inx

        valid_ngb_inx = np.where(np.all((np.all(ngb>=0,axis=1), np.all(ngb<sz,axis=1)), axis=0))
        ngb = ngb[valid_ngb_inx]

        ngb_value = [D[tuple(i)] for i in ngb]

        min_ind = np.argmin(ngb_value)
        min_v = ngb_value[min_ind]

        start_point = ngb[min_ind]
        path.append(start_point)

    path = np.array(path)
    return path


# --- MODIFIED ---
# Reason: Runtime optimization — numba JIT compilation of pointmin.
#   Neighbor-major loop order (26 passes, one per direction) is ~1.6x
#   faster than pure NumPy (fuses comparison + 4 conditional writes into
#   one pass per direction, no temporary arrays). Voxel-major (single pass,
#   26 neighbors per voxel) was tested but is 2x slower due to scattered
#   J accesses thrashing cache — neighbor-major gives sequential streaming.
# Original: pure-NumPy version in deepacson.py (identical logic, no JIT)
# ---
@njit(cache=True)
def _pointmin_jit(D, J, Fx, Fy, Fz):
    """Numba-compiled 26-neighbor min-propagation (tiled).

    Process the volume in blocks that fit in L3 cache.  For each block,
    run all 26 direction passes before moving to the next block.  The
    innermost c-loop stays full-length for vectorization.
    """
    x = np.array([0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 1,-1, 0, 0, 1, 1,-1,-1, 1,-1, 0, 0, 1, 1,-1,-1])
    y = np.array([0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1])
    z = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0])
    s0, s1, s2 = D.shape
    block_a = 8
    block_b = 16
    for ba in range(0, s0, block_a):
        ba_end = min(ba + block_a, s0)
        for bb in range(0, s1, block_b):
            bb_end = min(bb + block_b, s1)
            for i in range(26):
                In = J[1+x[i]:1+s0+x[i], 1+y[i]:1+s1+y[i], 1+z[i]:1+s2+z[i]]
                den = (x[i]**2 + y[i]**2 + z[i]**2)**0.5
                nx = x[i] / den
                ny = y[i] / den
                nz = z[i] / den
                for a in range(ba, ba_end):
                    for b in range(bb, bb_end):
                        for c in range(s2):
                            if In[a, b, c] < D[a, b, c]:
                                D[a, b, c] = In[a, b, c]
                                Fx[a, b, c] = nx
                                Fy[a, b, c] = ny
                                Fz[a, b, c] = nz
    return Fx, Fy, Fz

def pointmin(D):
    sz = D.shape
    max_D = np.max(D)
    Fx = np.zeros(sz)
    Fy = np.zeros(sz)
    Fz = np.zeros(sz)
    J = max_D * np.ones((sz[0]+2, sz[1]+2, sz[2]+2))
    J[1:-1,1:-1,1:-1] = D
    return _pointmin_jit(D, J, Fx, Fy, Fz)


# --- MODIFIED ---
# Reason: Runtime optimization — fuse Euler_path + euler_shortest_path into
#   a single @njit function. Eliminates per-step Python overhead:
#   - No temporary array allocations (np.array, np.append, list comprehensions)
#   - Pre-allocated path buffer instead of np.append copies
#   - Flat (3,) coordinates instead of (1,3) with squeeze/reshape
#   - Trilinear interpolation inlined with direct indexing
#   The algorithm is identical: gradient descent on the (Fx,Fy,Fz) field with
#   trilinear interpolation, stopping when reaching a source point or stalling.
# Original: Euler_path() + euler_shortest_path() as separate Python functions
#   (see deepacson.py for verbatim original)
# ---
@njit(cache=True)
def _euler_shortest_path_jit(Fx, Fy, Fz, source_point, start_point, step_size,
                              max_iters):
    """JIT-compiled Euler gradient descent path tracer.

    Args:
        Fx, Fy, Fz: negative gradient field arrays (same shape)
        source_point: (N, 3) array of source points to converge toward
        start_point: (3,) starting coordinates
        step_size: Euler integration step size
        max_iters: upper bound on iterations (safety limit)

    Returns:
        path: (M, 3) array of path points
    """
    s0, s1, s2 = Fx.shape

    # Pre-allocate path buffer (will trim at end)
    path = np.empty((max_iters + 2, 3))
    path[0, 0] = start_point[0]
    path[0, 1] = start_point[1]
    path[0, 2] = start_point[2]
    n_path = 1

    cur = np.empty(3)
    cur[0] = start_point[0]
    cur[1] = start_point[1]
    cur[2] = start_point[2]

    n_src = source_point.shape[0]

    for itr in range(max_iters):
        # --- Trilinear interpolation of gradient field (inlined Euler_path) ---
        fi = int(np.floor(cur[0]))
        fj = int(np.floor(cur[1]))
        fk = int(np.floor(cur[2]))

        # Clamp to valid range
        ci = min(max(fi, 0), s0 - 2)
        cj = min(max(fj, 0), s1 - 2)
        ck = min(max(fk, 0), s2 - 2)

        # Fractional distances
        dx = cur[0] - ci
        dy = cur[1] - cj
        dz = cur[2] - ck
        cx = 1.0 - dx
        cy = 1.0 - dy
        cz = 1.0 - dz

        # Trilinear weights (8 corners)
        w0 = cx * cy * cz
        w1 = cx * cy * dz
        w2 = cx * dy * cz
        w3 = cx * dy * dz
        w4 = dx * cy * cz
        w5 = dx * cy * dz
        w6 = dx * dy * cz
        w7 = dx * dy * dz

        gx = (w0 * Fx[ci, cj, ck] + w1 * Fx[ci, cj, ck+1] +
              w2 * Fx[ci, cj+1, ck] + w3 * Fx[ci, cj+1, ck+1] +
              w4 * Fx[ci+1, cj, ck] + w5 * Fx[ci+1, cj, ck+1] +
              w6 * Fx[ci+1, cj+1, ck] + w7 * Fx[ci+1, cj+1, ck+1])
        gy = (w0 * Fy[ci, cj, ck] + w1 * Fy[ci, cj, ck+1] +
              w2 * Fy[ci, cj+1, ck] + w3 * Fy[ci, cj+1, ck+1] +
              w4 * Fy[ci+1, cj, ck] + w5 * Fy[ci+1, cj, ck+1] +
              w6 * Fy[ci+1, cj+1, ck] + w7 * Fy[ci+1, cj+1, ck+1])
        gz = (w0 * Fz[ci, cj, ck] + w1 * Fz[ci, cj, ck+1] +
              w2 * Fz[ci, cj+1, ck] + w3 * Fz[ci, cj+1, ck+1] +
              w4 * Fz[ci+1, cj, ck] + w5 * Fz[ci+1, cj, ck+1] +
              w6 * Fz[ci+1, cj+1, ck] + w7 * Fz[ci+1, cj+1, ck+1])

        # Normalize gradient
        mag = (gx * gx + gy * gy + gz * gz + 1e-12) ** 0.5
        gx /= mag
        gy /= mag
        gz /= mag

        # Step
        nx = cur[0] - step_size * gx
        ny = cur[1] - step_size * gy
        nz = cur[2] - step_size * gz

        # Bounds check — if out of volume, stop (matches original zeros-check)
        if nx < 0.0 or ny < 0.0 or nz < 0.0 or nx > s0 or ny > s1 or nz > s2:
            break

        # Stall check (every 10 steps, compare to path[itr-10])
        if itr >= 10:
            pi = n_path - 10
            ddx = nx - path[pi, 0]
            ddy = ny - path[pi, 1]
            ddz = nz - path[pi, 2]
            movement = (ddx * ddx + ddy * ddy + ddz * ddz) ** 0.5
            if movement < step_size:
                break

        # Append to path
        path[n_path, 0] = nx
        path[n_path, 1] = ny
        path[n_path, 2] = nz
        n_path += 1

        # Distance to nearest source point
        min_dist = 1e30
        min_idx = 0
        for si in range(n_src):
            ddx = source_point[si, 0] - nx
            ddy = source_point[si, 1] - ny
            ddz = source_point[si, 2] - nz
            d = (ddx * ddx + ddy * ddy + ddz * ddz) ** 0.5
            if d < min_dist:
                min_dist = d
                min_idx = si

        if min_dist < 10.0 * step_size:
            # Snap to source point and stop
            path[n_path, 0] = source_point[min_idx, 0]
            path[n_path, 1] = source_point[min_idx, 1]
            path[n_path, 2] = source_point[min_idx, 2]
            n_path += 1
            break

        cur[0] = nx
        cur[1] = ny
        cur[2] = nz

    return path[:n_path].copy()


def euler_shortest_path(D, source_point, start_point, step_size):

    sz = D.shape
    max_D = np.max(D)
    J = max_D * np.ones((sz[0]+2, sz[1]+2, sz[2]+2))
    J[1:-1,1:-1,1:-1] = D

    # Flatten start_point from (1,3) to (3,)
    sp = np.ascontiguousarray(start_point.ravel()[:3])
    src = np.ascontiguousarray(source_point, dtype=np.float64)

    # Conservative upper bound on iterations
    max_iters = max(int(np.sum(np.array(D.shape)) * 10 / step_size), 10000)

    path = _euler_shortest_path_fused_jit(J, src, sp, step_size, max_iters)
    return path


@njit(cache=True)
def _euler_shortest_path_fused_jit(J, source_point, start_point, step_size,
                                    max_iters):
    """Fused pointmin + Euler path: compute gradient direction on-the-fly.

    Instead of precomputing the gradient field for all voxels (pointmin),
    compute it only at the ~2000 voxels the path actually visits.
    For each path point, find the 26-neighbor with minimum value in J
    at each of the 8 trilinear corners, then blend directions with
    trilinear weights.  Identical to pointmin + trilinear Euler integration
    but avoids allocating 3 full-volume gradient arrays.
    """
    # Neighbor offsets (26-connected)
    ox = np.array([0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 1,-1, 0, 0, 1, 1,-1,-1, 1,-1, 0, 0, 1, 1,-1,-1])
    oy = np.array([0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1])
    oz = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0])
    # Precompute unit normals (negated — we want uphill = away from minimum)
    norms = np.empty((26, 3))
    for i in range(26):
        den = (ox[i]**2 + oy[i]**2 + oz[i]**2)**0.5
        norms[i, 0] = -ox[i] / den
        norms[i, 1] = -oy[i] / den
        norms[i, 2] = -oz[i] / den

    # J is padded by 1 on each side, so valid D indices [0..s0) map to J[1..s0+1)
    s0 = J.shape[0] - 2
    s1 = J.shape[1] - 2
    s2 = J.shape[2] - 2

    path = np.empty((max_iters + 2, 3))
    path[0, 0] = start_point[0]
    path[0, 1] = start_point[1]
    path[0, 2] = start_point[2]
    n_path = 1

    cur = np.empty(3)
    cur[0] = start_point[0]
    cur[1] = start_point[1]
    cur[2] = start_point[2]

    n_src = source_point.shape[0]

    for itr in range(max_iters):
        # Trilinear interpolation corners (in D-space coordinates)
        fi = int(np.floor(cur[0]))
        fj = int(np.floor(cur[1]))
        fk = int(np.floor(cur[2]))
        ci = min(max(fi, 0), s0 - 2)
        cj = min(max(fj, 0), s1 - 2)
        ck = min(max(fk, 0), s2 - 2)

        dx = cur[0] - ci
        dy = cur[1] - cj
        dz = cur[2] - ck
        cx = 1.0 - dx
        cy = 1.0 - dy
        cz = 1.0 - dz

        weights = np.empty(8)
        weights[0] = cx * cy * cz
        weights[1] = cx * cy * dz
        weights[2] = cx * dy * cz
        weights[3] = cx * dy * dz
        weights[4] = dx * cy * cz
        weights[5] = dx * cy * dz
        weights[6] = dx * dy * cz
        weights[7] = dx * dy * dz

        # 8 trilinear corners in J-space (offset by 1 for padding)
        corners_i = np.array([ci, ci, ci, ci, ci+1, ci+1, ci+1, ci+1])
        corners_j = np.array([cj, cj, cj+1, cj+1, cj, cj, cj+1, cj+1])
        corners_k = np.array([ck, ck+1, ck, ck+1, ck, ck+1, ck, ck+1])

        # For each corner, find min-neighbor direction, then blend
        gx = 0.0; gy = 0.0; gz = 0.0
        for corn in range(8):
            # J-space index (add 1 for padding)
            ja = corners_i[corn] + 1
            jb = corners_j[corn] + 1
            jc = corners_k[corn] + 1
            best_val = J[ja, jb, jc]
            best_nx = 0.0; best_ny = 0.0; best_nz = 0.0
            for nb in range(26):
                val = J[ja + ox[nb], jb + oy[nb], jc + oz[nb]]
                if val < best_val:
                    best_val = val
                    best_nx = norms[nb, 0]
                    best_ny = norms[nb, 1]
                    best_nz = norms[nb, 2]
            gx += weights[corn] * best_nx
            gy += weights[corn] * best_ny
            gz += weights[corn] * best_nz

        # Normalize
        mag = (gx * gx + gy * gy + gz * gz + 1e-12) ** 0.5
        gx /= mag
        gy /= mag
        gz /= mag

        # Step
        nx = cur[0] - step_size * gx
        ny = cur[1] - step_size * gy
        nz = cur[2] - step_size * gz

        if nx < 0.0 or ny < 0.0 or nz < 0.0 or nx > s0 or ny > s1 or nz > s2:
            break

        # Stall check
        if itr >= 10:
            pi = n_path - 10
            ddx = nx - path[pi, 0]
            ddy = ny - path[pi, 1]
            ddz = nz - path[pi, 2]
            movement = (ddx * ddx + ddy * ddy + ddz * ddz) ** 0.5
            if movement < step_size:
                break

        path[n_path, 0] = nx
        path[n_path, 1] = ny
        path[n_path, 2] = nz
        n_path += 1

        # Distance to nearest source point
        min_dist = 1e30
        min_idx = 0
        for si in range(n_src):
            ddx = source_point[si, 0] - nx
            ddy = source_point[si, 1] - ny
            ddz = source_point[si, 2] - nz
            d = (ddx * ddx + ddy * ddy + ddz * ddz) ** 0.5
            if d < min_dist:
                min_dist = d
                min_idx = si

        if min_dist < 10.0 * step_size:
            path[n_path, 0] = source_point[min_idx, 0]
            path[n_path, 1] = source_point[min_idx, 1]
            path[n_path, 2] = source_point[min_idx, 2]
            n_path += 1
            break

        cur[0] = nx
        cur[1] = ny
        cur[2] = nz

    return path[:n_path].copy()


def get_line_length(L):

    dist = np.sum(np.sum((L[1:] - L[:-1])**2,axis=1)**0.5)
    return dist


def organize_skeleton(skel_seg,length_th):

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
            length_skel_seg = get_line_length(skel_breaked_seg)
            if(length_skel_seg >= length_th):
               final_skeleton.append(skel_breaked_seg)

    return final_skeleton


# --- MODIFIED ---
# Reason: (1) Add verbose parameter to control print output.
#         (2) Return maxD (max inscribed radius) for adaptive grid sizing.
# Original: def skeleton(Ax): ... print(line_length) ... return final_skeleton
# ---
def skeleton(Ax, verbose=True):

    boundary_dist=skfmm.distance(Ax)

    source_point=np.unravel_index(np.argmax(boundary_dist), boundary_dist.shape)
    maxD=boundary_dist[source_point]

    speed_im=(boundary_dist/maxD)**1.5

    Ax=np.ones(Ax.shape)
    Ax[source_point]=0

    flag=True
    skeleton_segments=[]
    source_point = np.array(source_point,ndmin=2)
    while True:

        # --- MODIFIED ---
        # Reason: Runtime optimization — avoid MaskedArray operations.
        #   skfmm.travel_time returns a MaskedArray (masked outside the object).
        #   The original code used np.ma.argmax and MaskedArray.filled, which
        #   have significant Python-level overhead (~14s for 139 calls).
        #   Instead, extract raw data + mask immediately, use np.argmax (masked
        #   voxels have travel_time <= 0 so argmax still finds the farthest
        #   reachable point), then fill masked positions with max_dist.
        # Original:
        #   D=skfmm.travel_time(Ax,speed_im)
        #   end_point=np.unravel_index(np.ma.argmax(D), D.shape)
        #   max_dist=D[end_point]
        #   D=np.ma.MaskedArray.filled(D,max_dist)
        # ---
        D_ma=skfmm.travel_time(Ax,speed_im)
        mask = np.ma.getmask(D_ma)
        D = np.ma.getdata(D_ma).copy()
        end_point=np.unravel_index(np.argmax(D), D.shape)
        max_dist=D[end_point]
        if mask is not np.ma.nomask:
            D[mask] = max_dist

        end_point = np.array(end_point,ndmin=2)
        # --- MODIFIED ---
        # Reason: Runtime optimization — increase Euler step size from 0.1 to 0.5.
        #   The gradient field from pointmin is defined on the voxel grid; trilinear
        #   interpolation provides sub-voxel values but the underlying information
        #   is at 1-voxel resolution. step_size=0.5 traces essentially the same
        #   path with ~5× fewer integration steps. Skeleton points are further
        #   subsampled at step_voxels spacing for cross-section sampling anyway.
        # Original: shortest_line=euler_shortest_path(D,source_point,end_point,step_size=0.1)
        # ---
        shortest_line=euler_shortest_path(D,source_point,end_point,step_size=0.5)
        #shortest_line = discrete_shortest_path(D,end_point)

        line_length=get_line_length(shortest_line)
        if verbose:                                    # MODIFIED: was unconditional print
            print(line_length)

        if flag:
            length_threshold=min(40*maxD, 0.18*line_length)
            flag=False

        if(line_length<=length_threshold):
            break


        source_point=np.append(source_point,shortest_line,axis=0)

        skeleton_segments.append(shortest_line)

        shortest_line=np.floor(shortest_line).astype(int)

        for i in shortest_line:
            Ax[tuple(i)]=0

    if len(skeleton_segments)!=0:
        final_skeleton=organize_skeleton(skeleton_segments,length_threshold)
    else:
        final_skeleton=[]

    return final_skeleton, maxD                        # MODIFIED: was return final_skeleton

if __name__ == "__main__":
    skeleton(sys.argv[1])


# ===================================================================
# Cross-section sampling (from shape_decomposition.py, restructured)
#
# The original tangent_planes_to_zone_of_interest() combines cross-section
# sampling with zone-of-interest tracking (Hausdorff distance, mean curve,
# shift imposition). We extract only the cross-section sampling logic,
# which is the part relevant to radius profiling.
# ===================================================================

def sample_cross_section(binary_vol, point, tangent_vec, g_radius, g_res):
    """
    Sample a perpendicular cross-section and measure its area.

    Uses the DeepACSON approach: rotate a planar grid to be perpendicular to
    the tangent vector, trilinear-interpolate the binary volume, then find the
    connected component at the center.

    Args:
        binary_vol: 3D binary volume (float64, 1=inside axon)
        point: (3,) skeleton point coordinates in volume space
        tangent_vec: (3,) unit tangent vector at this point
        g_radius: grid radius in voxels
        g_res: grid resolution in voxels (e.g. 0.25)

    Returns:
        area_pixels: number of pixels in the central connected component,
                     or 0 if invalid (no center hit, multiple labels, too small)
    """
    sz = binary_vol.shape

    # Build sampling plane grid — verbatim from shape_decomposition.py
    x, y = np.mgrid[-g_radius:g_radius:g_res, -g_radius:g_radius:g_res]
    z = np.zeros_like(x)
    xyz = np.array([np.ravel(x), np.ravel(y), np.ravel(z)]).T

    cent_ball = (x**2+y**2)<g_res*1

    # Rotate plane to be perpendicular to tangent — verbatim
    if np.array_equal(tangent_vec, np.array([0, 0, 0])):
        return 0

    rot_axis = unit_normal_vector(tangent_vec, np.array([0,0,1]))
    theta = angle(tangent_vec, np.array([0,0,1]))
    rot_mat = rotation_matrix_3D(rot_axis, theta)
    rotated_plane = np.squeeze(rotate_vector(xyz, rot_mat))
    cross_section_plane = rotated_plane+point

    # --- MODIFIED ---
    # Reason: Runtime optimization — replace RegularGridInterpolator with
    #   scipy.ndimage.map_coordinates for trilinear interpolation. Same
    #   bilinear (order=1) interpolation, but avoids object construction
    #   and coordinate-lookup overhead of RGI.
    # Original:
    #   interpolating_func = rgi((range(sz[0]),range(sz[1]),range(sz[2])),
    #                               binary_vol,bounds_error=False,fill_value=0)
    #   cross_section = interpolating_func(cross_section_plane)
    # ---
    coords = cross_section_plane.T  # (3, N) for map_coordinates
    cross_section = map_coordinates(binary_vol, coords, order=1,
                                    mode='constant', cval=0.0)
    bw_cross_section = cross_section>=0.5
    bw_cross_section = np.reshape(bw_cross_section, x.shape)

    # Connected component at center — verbatim
    label_cross_section, nn = label(bw_cross_section, return_num=True)
    main_lbl = np.unique(label_cross_section[cent_ball])
    main_lbl = main_lbl[np.nonzero(main_lbl)]

    if len(main_lbl)!=1:
        return 0

    bw_cross_section = label_cross_section==main_lbl[0]

    # Size filter — verbatim
    nz_X = np.count_nonzero(np.sum(bw_cross_section, axis=0))
    nz_Y = np.count_nonzero(np.sum(bw_cross_section, axis=1))
    if (nz_X<4) or (nz_Y<4):
        return 0

    # Area via regionprops — verbatim
    props = regionprops(bw_cross_section.astype(np.int32))
    if len(props) == 0:
        return 0

    return props[0].area


# --- MODIFIED ---
# Reason: Adaptive per-point grid sizing for runtime optimization.
#   Instead of a fixed large g_radius for all cross-sections, start small
#   and grow only when the cross-section touches the grid border.
#   Uses cheap nearest-neighbor sampling of border pixels only.
# Original: not present (DeepACSON uses a fixed g_radius for all points).
# ---
def find_g_radius(binary_vol, point, tangent_vec, g_radius_init, max_g_radius):
    """Find minimum g_radius where cross-section doesn't touch grid border.

    Samples the border of the rotated sampling plane using nearest-neighbor
    interpolation (cheap). If any border voxel is foreground, doubles g_radius
    and retries. Returns the first g_radius with a clear border.

    Args:
        binary_vol: 3D binary volume (float64, 1=inside axon)
        point: (3,) skeleton point coordinates in volume space
        tangent_vec: (3,) unit tangent vector at this point
        g_radius_init: starting grid radius in voxels
        max_g_radius: upper bound on grid radius

    Returns:
        g_radius: adequate grid radius (capped at max_g_radius)
    """
    if np.array_equal(tangent_vec, np.array([0, 0, 0])):
        return g_radius_init

    sz = np.array(binary_vol.shape)

    rot_axis = unit_normal_vector(tangent_vec, np.array([0, 0, 1]))
    theta = angle(tangent_vec, np.array([0, 0, 1]))
    rot_mat = rotation_matrix_3D(rot_axis, theta)

    g_radius = g_radius_init
    while g_radius < max_g_radius:
        # Border points at 1-voxel spacing (cheap)
        edge = np.arange(-g_radius, g_radius + 1, 1.0)
        n = len(edge)
        # Four edges of the square
        top = np.column_stack([edge, np.full(n, -g_radius), np.zeros(n)])
        bot = np.column_stack([edge, np.full(n, g_radius), np.zeros(n)])
        left = np.column_stack([np.full(n - 2, -g_radius), edge[1:-1], np.zeros(n - 2)])
        right = np.column_stack([np.full(n - 2, g_radius), edge[1:-1], np.zeros(n - 2)])
        border_local = np.vstack([top, bot, left, right])

        # Rotate to world space
        border_world = rotate_vector(border_local, rot_mat) + point

        # Nearest-neighbor lookup
        idx = np.floor(border_world).astype(int)
        valid = np.all((idx >= 0) & (idx < sz), axis=1)
        if not valid.any():
            break  # all out of bounds → background

        idx_v = idx[valid]
        values = binary_vol[idx_v[:, 0], idx_v[:, 1], idx_v[:, 2]]
        if not np.any(values >= 0.5):
            break  # border is clear

        g_radius *= 2

    return min(g_radius, max_g_radius)
