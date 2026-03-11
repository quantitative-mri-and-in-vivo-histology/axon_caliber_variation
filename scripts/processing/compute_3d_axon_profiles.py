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
   Euler-Rodrigues rotation + trilinear interpolation + connected-component labeling
4. Measure cross-section area via regionprops → equivalent circular radius

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
from scipy.ndimage import find_objects, median_filter
from scipy.interpolate import RegularGridInterpolator as rgi
from skimage.measure import label, regionprops
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
_grid_xyz = None       # Precomputed sampling plane grid
_grid_cent_ball = None  # Center mask for connected-component check
_grid_shape = None      # Shape of the 2D grid
_g_radius = None        # Grid radius in voxels (computed from max_radius_um / voxel_size)
_step_voxels = None     # Step size for skeleton subsampling (in voxels)
_voxel_size_um = None   # Isotropic voxel size in micrometers


# ===========================================================================
# Skeleton extraction (from DeepACSON/CSD/skeleton3D.py)
# ===========================================================================

@njit(cache=True)
def _pointmin(D):
    sz0, sz1, sz2 = D.shape
    max_D = np.max(D)
    Fx = np.zeros((sz0, sz1, sz2))
    Fy = np.zeros((sz0, sz1, sz2))
    Fz = np.zeros((sz0, sz1, sz2))

    # Padded volume
    J = np.full((sz0+2, sz1+2, sz2+2), max_D)
    J[1:sz0+1, 1:sz1+1, 1:sz2+1] = D

    # 26-connected neighbor offsets
    nx = np.array([0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 1,-1, 0, 0, 1, 1,-1,-1, 1,-1, 0, 0, 1, 1,-1,-1])
    ny = np.array([0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1])
    nz = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0])

    for i in range(26):
        dx, dy, dz = nx[i], ny[i], nz[i]
        den = (dx*dx + dy*dy + dz*dz)**0.5
        fx = dx / den
        fy = dy / den
        fz = dz / den

        for a in range(sz0):
            for b in range(sz1):
                for c in range(sz2):
                    val = J[1+a+dx, 1+b+dy, 1+c+dz]
                    if val < D[a, b, c]:
                        D[a, b, c] = val
                        Fx[a, b, c] = fx
                        Fy[a, b, c] = fy
                        Fz[a, b, c] = fz

    return Fx, Fy, Fz


@njit(cache=True)
def _euler_path(Fx, Fy, Fz, start_point, step_size):
    sz0, sz1, sz2 = Fx.shape

    sp0 = start_point[0, 0]
    sp1 = start_point[0, 1]
    sp2 = start_point[0, 2]
    f0 = int(np.floor(sp0))
    f1 = int(np.floor(sp1))
    f2 = int(np.floor(sp2))

    # Trilinear weights
    d0 = sp0 - f0
    d1 = sp1 - f1
    d2 = sp2 - f2
    c0 = 1.0 - d0
    c1 = 1.0 - d1
    c2 = 1.0 - d2

    perc = np.array([c0*c1*c2, c0*c1*d2, c0*d1*c2, c0*d1*d2,
                     d0*c1*c2, d0*c1*d2, d0*d1*c2, d0*d1*d2])

    # 8 corner offsets
    ox = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    oy = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    oz = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    sum_gx = 0.0
    sum_gy = 0.0
    sum_gz = 0.0
    for i in range(8):
        ix = f0 + ox[i]
        iy = f1 + oy[i]
        iz = f2 + oz[i]
        if ix < 0: ix = 0
        if ix >= sz0: ix = sz0 - 1
        if iy < 0: iy = 0
        if iy >= sz1: iy = sz1 - 1
        if iz < 0: iz = 0
        if iz >= sz2: iz = sz2 - 1

        w = perc[i]
        sum_gx += Fx[ix, iy, iz] * w
        sum_gy += Fy[ix, iy, iz] * w
        sum_gz += Fz[ix, iy, iz] * w

    norm = (sum_gx*sum_gx + sum_gy*sum_gy + sum_gz*sum_gz + 0.000001)**0.5
    gx = sum_gx / norm
    gy = sum_gy / norm
    gz = sum_gz / norm

    ep0 = sp0 - step_size * gx
    ep1 = sp1 - step_size * gy
    ep2 = sp2 - step_size * gz

    end_point = np.zeros((1, 3))
    if ep0 < 0 or ep1 < 0 or ep2 < 0 or ep0 > sz0 or ep1 > sz1 or ep2 > sz2:
        return end_point

    end_point[0, 0] = ep0
    end_point[0, 1] = ep1
    end_point[0, 2] = ep2
    return end_point


def _euler_shortest_path(D, source_point, start_point, step_size):
    Fx, Fy, Fz = _pointmin(D)
    Fx = -Fx
    Fy = -Fy
    Fz = -Fz

    itr = 0
    path = start_point
    while True:

        end_point = _euler_path(Fx,Fy,Fz,start_point,step_size)

        dist_endpoint_to_all = np.sum((source_point-end_point)**2,axis=1)**0.5
        distance_to_endpoint = min(dist_endpoint_to_all)

        if(itr>=10):
            Movement = np.sum((end_point-path[itr-10])**2)**0.5
        else:
            Movement = step_size+1

        if(np.all(end_point==0) or Movement<step_size):
            break

        itr = itr+1

        path = np.append(path,end_point,axis=0)

        if(distance_to_endpoint<10*step_size):
            source_inx = source_point[np.argmin(dist_endpoint_to_all)]
            path = np.append(path,np.array(source_inx,ndmin=2),axis=0)
            break

        start_point = end_point
    return path


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
    """FMM + Euler backtracking skeleton extraction (DeepACSON)."""
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

        D=skfmm.travel_time(Ax,speed_im)
        end_point=np.unravel_index(np.ma.argmax(D), D.shape)
        max_dist=D[end_point]
        D=np.ma.MaskedArray.filled(D,max_dist)

        end_point = np.array(end_point,ndmin=2)
        shortest_line=_euler_shortest_path(D,source_point,end_point,step_size=0.1)

        line_length=_get_line_length(shortest_line)

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
        final_skeleton=_organize_skeleton(skeleton_segments,length_threshold)
    else:
        final_skeleton=[]

    return final_skeleton


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
# Plane rotation (from DeepACSON/CSD/plane_rotation.py)
# ===========================================================================

def _unit_normal_vector(vec1, vec2):
    n = np.cross(vec1, vec2)
    if np.array_equal(n, np.array([0, 0, 0])):
        n = vec1
    s = max(np.sqrt(np.dot(n,n)), 1e-5)
    n = n/s
    return n


def _angle(vec1, vec2):
    theta=np.arccos(np.dot(vec1,vec2) / (np.sqrt(np.dot(vec1,vec1)) * np.sqrt(np.dot(vec2, vec2))))
    return theta


def _rotation_matrix_3D(vector, theta):
    """Euler-Rodrigues rotation matrix."""
    a=np.cos(theta/2.0)
    b,c,d=-vector*np.sin(theta/2.0)
    aa,bb,cc,dd=a**2, b**2, c**2, d**2
    bc,ad,ac,ab,bd,cd=b*c, a*d, a*c, a*b, b*d, c*d

    rot_mat=np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    return rot_mat


def _rotate_vector(vector, rot_mat):
    rotated_vec = np.dot(vector,rot_mat)
    return rotated_vec


# ===========================================================================
# Cross-section sampling (DeepACSON approach)
# ===========================================================================

def precompute_grid(g_radius):
    """Precompute the XY sampling grid and center mask (called once)."""
    coords = np.arange(-g_radius, g_radius + 1)
    x, y = np.meshgrid(coords, coords)
    z = np.zeros_like(x)
    xyz = np.array([np.ravel(x), np.ravel(y), np.ravel(z)]).T
    cent_ball = (x**2 + y**2) < 1
    grid_shape = x.shape
    return xyz, cent_ball, grid_shape


def _sample_cross_section(interpolating_func, point, tangent_vec,
                           xyz, cent_ball, grid_shape):
    """
    Sample a perpendicular cross-section using the DeepACSON approach.

    Uses trilinear interpolation (RegularGridInterpolator), connected-component
    labeling at center, and regionprops for area measurement.

    Returns equivalent radius in voxels, or None if invalid.
    """
    if np.array_equal(tangent_vec, np.array([0, 0, 0])):
        return None

    rot_axis = _unit_normal_vector(tangent_vec, np.array([0, 0, 1]))
    theta = _angle(tangent_vec, np.array([0, 0, 1]))
    rot_mat = _rotation_matrix_3D(rot_axis, theta)
    rotated_plane = np.squeeze(_rotate_vector(xyz, rot_mat))
    cross_section_plane = rotated_plane + point

    # Trilinear interpolation
    cross_section = interpolating_func(cross_section_plane)
    bw_cross_section = cross_section >= 0.5
    bw_cross_section = np.reshape(bw_cross_section, grid_shape)

    # Connected component at center
    label_cross_section, nn = label(bw_cross_section, return_num=True)
    main_lbl = np.unique(label_cross_section[cent_ball])
    main_lbl = main_lbl[np.nonzero(main_lbl)]

    if len(main_lbl) != 1:
        return None

    bw_cross_section = label_cross_section == main_lbl[0]

    # Size check
    nz_X = np.count_nonzero(np.sum(bw_cross_section, axis=0))
    nz_Y = np.count_nonzero(np.sum(bw_cross_section, axis=1))
    if nz_X < 4 or nz_Y < 4:
        return None

    # Measure area via regionprops (1 pixel = 1 voxel² at grid resolution 1.0)
    props = regionprops(bw_cross_section.astype(np.int32))
    if len(props) == 0:
        return None

    area_voxels = props[0].area
    radius_voxels = np.sqrt(area_voxels / np.pi)

    return radius_voxels


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
    # Warmup _pointmin and _euler_path with tiny arrays
    D = np.ones((3, 3, 3))
    _pointmin(D)
    sp = np.array([[1.0, 1.0, 1.0]])
    _euler_path(D, D, D, sp, 0.1)


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

        pad = _g_radius + 5
        min_padded = np.maximum(min_coords - pad, 0)
        max_padded = np.minimum(max_coords + pad, vol_shape)

        subvol = _shared_volume[min_padded[0]:max_padded[0],
                                min_padded[1]:max_padded[1],
                                min_padded[2]:max_padded[2]]
        binary = (subvol == axon_label).astype(np.float64)

        n_voxels = np.count_nonzero(binary)
        if n_voxels < 100:
            return None

        # Skeleton extraction (FMM + Euler backtracking)
        try:
            skel_segments = _extract_skeleton(binary)
        except Exception as e:
            logger.debug(f"Axon {axon_label}: skeleton extraction failed - {e}")
            return None

        if len(skel_segments) == 0:
            return None

        # Create interpolator once per axon
        sz = binary.shape
        interpolating_func = rgi(
            (range(sz[0]), range(sz[1]), range(sz[2])),
            binary,
            bounds_error=False, fill_value=0
        )

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

            # Compute tangent vectors
            tangent_vecs = _unit_tangent_vector(skel_seg)

            radii = []
            skel_coords = []

            for point, tangent in zip(skel_seg, tangent_vecs):
                if np.array_equal(tangent, np.array([0, 0, 0])):
                    continue

                radius_voxels = _sample_cross_section(
                    interpolating_func, point, tangent,
                    _grid_xyz, _grid_cent_ball, _grid_shape
                )

                if radius_voxels is not None and radius_voxels > 0:
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
    D_warmup = np.ones((3, 3, 3))
    _pointmin(D_warmup)
    sp_warmup = np.array([[1.0, 1.0, 1.0]])
    _euler_path(D_warmup, D_warmup, D_warmup, sp_warmup, 0.1)

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

    # Precompute sampling grid (1 pixel = 1 voxel)
    xyz, cent_ball, grid_shape = precompute_grid(g_radius)

    step_voxels = step_size_um / voxel_size if step_size_um is not None else None

    # Set globals for workers
    global _shared_volume, _shared_bboxes
    global _grid_xyz, _grid_cent_ball, _grid_shape
    global _g_radius, _step_voxels, _voxel_size_um
    _shared_volume = volume
    _shared_bboxes = bboxes
    _grid_xyz = xyz
    _grid_cent_ball = cent_ball
    _grid_shape = grid_shape
    _g_radius = g_radius
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
