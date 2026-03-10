#!/usr/bin/env python3
"""
Compute 3D axon profiles using the DeepACSON CSD approach (self-contained).

Reimplements the DeepACSON skeleton extraction and cross-section sampling
without depending on the external DeepACSON package. All functions are copied
verbatim from DeepACSON/CSD/ (Abdollahzadeh et al., 2019).

The algorithm:
1. Skeleton extraction via FMM + Euler backtracking (step_size=0.1)
2. Skeleton organization: break at junction endpoints, filter by length_threshold
3. For each skeleton segment: sample perpendicular cross-sections using
   Euler-Rodrigues rotation + trilinear interpolation + connected-component labeling
4. Measure cross-section area via regionprops

Usage:
    python compute_3d_axon_profiles_ours_v2.py \\
        data/debug/sham_25_ipsi_cg_myelin_small.zarr \\
        data/debug/sham_25_ipsi_cg_ours_v2_profiles.npz
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import skfmm
from numba import njit
from scipy.ndimage import find_objects
from scipy.interpolate import RegularGridInterpolator as rgi
from skimage.measure import label, regionprops
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skeleton extraction (from DeepACSON/CSD/skeleton3D.py)
# ---------------------------------------------------------------------------

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

    # Floor of start point, clamped to valid indices
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
        # Clamp
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


def skeleton(Ax):
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


# ---------------------------------------------------------------------------
# Tangent vectors (from DeepACSON/CSD/unit_tangent_vector.py)
# ---------------------------------------------------------------------------

def unit_tangent_vector(curve):
    d_curve = np.gradient(curve, axis=0)
    ds = np.expand_dims((np.sum(d_curve**2, axis=1))**0.5, axis=1)
    ds[ds==0] = 1e-5
    u_tang_vec = d_curve/np.repeat(ds, curve.shape[1], axis=1)
    return u_tang_vec


# ---------------------------------------------------------------------------
# Plane rotation (from DeepACSON/CSD/plane_rotation.py)
# ---------------------------------------------------------------------------

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


def rotation_matrix_3D(vector, theta):
    """Euler-Rodrigues rotation matrix."""
    a=np.cos(theta/2.0)
    b,c,d=-vector*np.sin(theta/2.0)
    aa,bb,cc,dd=a**2, b**2, c**2, d**2
    bc,ad,ac,ab,bd,cd=b*c, a*d, a*c, a*b, b*d, c*d

    rot_mat=np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    return rot_mat


def rotate_vector(vector, rot_mat):
    rotated_vec = np.dot(vector,rot_mat)
    return rotated_vec


# ---------------------------------------------------------------------------
# Volume loading
# ---------------------------------------------------------------------------

def load_zarr_volume(zarr_path: Path):
    """Load level-0 volume and voxel size from an OME-Zarr store."""
    import zarr
    store = zarr.open_group(str(zarr_path), mode="r")
    volume = np.asarray(store["0"])
    multiscales = store.attrs["multiscales"]
    scale = multiscales[0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    return volume, float(scale[0])


# ---------------------------------------------------------------------------
# Cross-section sampling
# ---------------------------------------------------------------------------

def precompute_grid(g_radius, g_res):
    """Precompute the XY sampling grid and center mask (called once)."""
    x, y = np.mgrid[-g_radius:g_radius:g_res, -g_radius:g_radius:g_res]
    z = np.zeros_like(x)
    xyz = np.array([np.ravel(x), np.ravel(y), np.ravel(z)]).T
    cent_ball = (x**2 + y**2) < g_res * 1
    grid_shape = x.shape
    return xyz, cent_ball, grid_shape


def sample_cross_section(interpolating_func, point, tangent_vec, xyz, cent_ball, grid_shape, g_res):
    """
    Sample a perpendicular cross-section.

    Uses trilinear interpolation (RegularGridInterpolator), connected-component
    labeling at center, and regionprops for area measurement.

    Returns equivalent radius in voxels, or None if invalid.
    """
    # Rotate plane to be perpendicular to tangent
    if np.array_equal(tangent_vec, np.array([0, 0, 0])):
        return None

    rot_axis = unit_normal_vector(tangent_vec, np.array([0, 0, 1]))
    theta = angle(tangent_vec, np.array([0, 0, 1]))
    rot_mat = rotation_matrix_3D(rot_axis, theta)
    rotated_plane = np.squeeze(rotate_vector(xyz, rot_mat))
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

    # Measure area via regionprops
    props = regionprops(bw_cross_section.astype(np.int32))
    if len(props) == 0:
        return None

    area_pixels = props[0].area
    # Convert pixel area to voxel area (g_res is the pixel spacing in voxels)
    area_voxels = area_pixels * (g_res ** 2)
    radius_voxels = np.sqrt(area_voxels / np.pi)

    return radius_voxels


# ---------------------------------------------------------------------------
# Per-axon processing
# ---------------------------------------------------------------------------

def process_single_axon(volume, axon_label, bboxes, voxel_size_um,
                        xyz, cent_ball, grid_shape,
                        g_radius=15, g_res=0.25, step_voxels=None):
    """
    Process one axon:
    1. Crop the axon
    2. Extract skeleton via FMM + Euler
    3. Sample cross-sections along each skeleton segment
    4. Return radius profile
    """
    bbox = bboxes.get(axon_label)
    if bbox is None:
        return None

    min_coords, max_coords = bbox
    pad = g_radius + 5
    vol_shape = np.array(volume.shape)

    min_padded = np.maximum(min_coords - pad, 0)
    max_padded = np.minimum(max_coords + pad, vol_shape)

    subvol = volume[min_padded[0]:max_padded[0],
                    min_padded[1]:max_padded[1],
                    min_padded[2]:max_padded[2]]
    binary = (subvol == axon_label).astype(np.float64)

    n_voxels = np.count_nonzero(binary)
    if n_voxels < 100:
        return None

    # Skeleton extraction (FMM + Euler backtracking)
    try:
        skel_segments = skeleton(binary)
    except Exception as e:
        logger.debug(f"Axon {axon_label}: skeleton extraction failed - {e}")
        return None

    if len(skel_segments) == 0:
        return None

    # Create interpolator once per axon (avoids repeated astype + rgi construction)
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

    for seg_idx, skel_seg in enumerate(skel_segments):
        if len(skel_seg) < 3:
            continue

        # Subsample skeleton points to match desired step size
        if step_voxels is not None:
            # Euler step is ~0.1 voxels; compute stride to approximate step_voxels
            stride = max(1, round(step_voxels / 0.1))
            skel_seg = skel_seg[::stride]
            if len(skel_seg) < 3:
                continue

        # Compute tangent vectors
        tangent_vecs = unit_tangent_vector(skel_seg)

        radii = []
        skel_coords = []

        for point, tangent in zip(skel_seg, tangent_vecs):
            if np.array_equal(tangent, np.array([0, 0, 0])):
                continue

            radius_voxels = sample_cross_section(
                interpolating_func, point, tangent, xyz, cent_ball, grid_shape, g_res
            )

            if radius_voxels is not None and radius_voxels > 0:
                radius_um = radius_voxels * voxel_size_um
                radii.append(radius_um)
                # Convert skeleton coords to physical space
                global_coord = (point + min_padded) * voxel_size_um
                skel_coords.append(global_coord)

        if len(radii) < 2:
            continue

        radii = np.array(radii)

        # Track main trunk length (longest segment)
        seg_length = np.sum(np.sqrt(np.sum(np.diff(skel_seg, axis=0)**2, axis=1))) * voxel_size_um
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compute 3D axon profiles (DeepACSON CSD approach, self-contained)'
    )
    parser.add_argument('input', type=Path,
                        help='Path to .zarr volume with labeled axons')
    parser.add_argument('output', type=Path,
                        help='Output .npz file')
    parser.add_argument('--max-axons', type=int, default=0,
                        help='Maximum axons to process (0 = all)')
    parser.add_argument('--g-radius', type=int, default=15,
                        help='Grid radius for cross-section sampling in voxels (default: 15)')
    parser.add_argument('--g-res', type=float, default=0.25,
                        help='Grid resolution for cross-section sampling in voxels (default: 0.25)')
    parser.add_argument('--step-size', type=float, default=None,
                        help='Step size in μm for subsampling skeleton points (default: use every Euler step ~0.005 μm)')

    args = parser.parse_args()

    # Load volume
    logger.info(f"Loading volume: {args.input}")
    volume, voxel_size_um = load_zarr_volume(args.input)
    logger.info(f"Volume shape: {volume.shape}, voxel size: {voxel_size_um} μm")

    # Precompute bounding boxes via find_objects (once, instead of argwhere per axon)
    logger.info("Computing bounding boxes...")
    slices = find_objects(volume)
    bboxes = {}
    for label_idx, bbox_slices in enumerate(slices):
        if bbox_slices is None:
            continue
        lbl = label_idx + 1
        min_c = np.array([s.start for s in bbox_slices])
        max_c = np.array([s.stop for s in bbox_slices])
        bboxes[lbl] = (min_c, max_c)

    axon_labels = sorted(bboxes.keys())
    logger.info(f"Found {len(axon_labels)} axons")

    if args.max_axons > 0:
        axon_labels = axon_labels[:args.max_axons]
        logger.info(f"Processing first {args.max_axons} axons")

    # Precompute sampling grid (once, reused for all cross-sections)
    xyz, cent_ball, grid_shape = precompute_grid(args.g_radius, args.g_res)

    step_voxels = None
    if args.step_size is not None:
        step_voxels = args.step_size / voxel_size_um
        logger.info(f"Subsampling skeleton at {args.step_size} μm = {step_voxels:.1f} voxels "
                     f"(stride ~{max(1, round(step_voxels / 0.1))})")

    results = []
    for axon_label in tqdm(axon_labels, desc="Processing axons"):
        try:
            result = process_single_axon(
                volume, axon_label, bboxes, voxel_size_um,
                xyz, cent_ball, grid_shape,
                g_radius=args.g_radius, g_res=args.g_res,
                step_voxels=step_voxels,
            )
            if result is not None:
                results.append(result)
        except Exception as e:
            logger.debug(f"Axon {axon_label}: failed - {e}")
            continue

    logger.info(f"Successfully processed {len(results)}/{len(axon_labels)} axons")

    if len(results) == 0:
        logger.error("No axons were successfully processed!")
        return

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)

    labels = np.array([r['label'] for r in results])
    n_points = np.array([r['n_points'] for r in results])
    mean_radii = np.array([r['mean_radius_um'] for r in results])
    std_radii = np.array([r['std_radius_um'] for r in results])
    lengths = np.array([r['length_um'] for r in results])
    radii_profiles = np.array([r['radii_um'] for r in results], dtype=object)
    skeleton_coords = np.array([r['skeleton_um'] for r in results], dtype=object)

    all_radii = np.concatenate([r['radii_um'] for r in results])

    np.savez(
        args.output,
        labels=labels,
        n_points=n_points,
        mean_radii_um=mean_radii,
        std_radii_um=std_radii,
        lengths_um=lengths,
        radii_profiles_um=radii_profiles,
        skeleton_coords_um=skeleton_coords,
        all_radii_um=all_radii,
        voxel_size_um=voxel_size_um,
        source_file=str(args.input),
        method='deepacson_csd',
    )

    logger.info(f"Saved results to {args.output}")

    # Summary
    r_eff = (np.mean(all_radii**6) / np.mean(all_radii**2)) ** 0.25
    logger.info(f"\nSummary:")
    logger.info(f"  Axons processed: {len(results)}")
    logger.info(f"  Total radius samples: {len(all_radii)}")
    logger.info(f"  Mean axon length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f} μm")
    logger.info(f"  r̄: {np.mean(all_radii):.4f} μm")
    logger.info(f"  r_eff: {r_eff:.4f} μm")


if __name__ == '__main__':
    main()
