#!/usr/bin/env python3
"""
Compute 3D axon profiles reproducing DeepACSON cross-section results, fast.

Uses our skeleton extraction (FMM + Euler path tracing) with a Numba-fused
cross-section kernel that matches DeepACSON's approach:
- DeepACSON Euler-Rodrigues rotation (row_vector @ rot_mat)
- Trilinear interpolation (threshold >= 0.5)
- Flood-fill connected component at center
- Sub-voxel grid resolution (g_res=0.25, g_radius=15 voxels)

Usage:
    python compute_3d_axon_profiles_ours.py \
        data/debug/sham_25_ipsi_cg_myelin_small.zarr \
        data/debug/sham_25_ipsi_cg_ours_profiles.npz
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
from numba import njit
from scipy.ndimage import find_objects
from tqdm import tqdm

# axonometry library
_root = Path(__file__).resolve().parent
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

from axonometry import (
    extract_skeleton,
    skeleton_warmup,
    unit_tangent_vector,
    compute_arc_length,
    resample_curve_by_arc_length,
    validate_skeleton_points,
    find_longest_contiguous_segment,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Module-level grid state
_base_grid_xy = None
_grid_size = None


def precompute_base_grid(g_radius, g_res):
    """Precompute the XY plane grid (identical for every cross-section)."""
    global _base_grid_xy, _grid_size
    coords = np.arange(-g_radius, g_radius, g_res)
    x, y = np.meshgrid(coords, coords)
    _base_grid_xy = np.ascontiguousarray(
        np.stack([x.ravel(), y.ravel()], axis=1)
    )
    _grid_size = len(coords)


@njit(cache=True, fastmath=True)
def _sample_trilinear_flood_fill(volume, base_grid, grid_size,
                                  point_0, point_1, point_2,
                                  tan_0, tan_1, tan_2):
    """
    Fused kernel: DeepACSON rotation + trilinear interpolation + flood-fill.

    Matches DeepACSON's cross-section sampling:
    - Euler-Rodrigues rotation with row_vector @ rot_mat convention
    - Trilinear interpolation (threshold >= 0.5) on binary volume
    - Flood-fill from center for connected component
    """
    n_grid = base_grid.shape[0]
    sz0, sz1, sz2 = volume.shape

    # DeepACSON rotation: rot_axis = cross(tangent, z_axis) = (tan_1, -tan_0, 0)
    dot_zt = tan_2

    if dot_zt > 0.9999:
        r00 = 1.0; r01 = 0.0; r02 = 0.0
        r10 = 0.0; r11 = 1.0; r12 = 0.0
    elif dot_zt < -0.9999:
        r00 = 1.0; r01 = 0.0; r02 = 0.0
        r10 = 0.0; r11 = 1.0; r12 = 0.0
    else:
        rn = np.sqrt(tan_0 * tan_0 + tan_1 * tan_1)
        rx = tan_1 / rn
        ry = -tan_0 / rn

        clamped = max(-1.0, min(1.0, dot_zt))
        theta = np.arccos(clamped)

        a = np.cos(theta * 0.5)
        s = np.sin(theta * 0.5)
        b = -rx * s
        c = -ry * s

        aa = a * a; bb = b * b; cc = c * c
        bc = b * c; ac = a * c; ab = a * b

        # DeepACSON rot_mat (d=0):
        r00 = aa + bb - cc;  r01 = 2.0 * bc;       r02 = -2.0 * ac
        r10 = 2.0 * bc;      r11 = aa + cc - bb;    r12 = 2.0 * ab
        r20 = 2.0 * ac;      r21 = -2.0 * ab;       r22 = aa - bb - cc

    bw = np.zeros((grid_size, grid_size), dtype=np.uint8)

    for i in range(n_grid):
        gx = base_grid[i, 0]
        gy = base_grid[i, 1]

        # row_vector @ rot_mat: out_k = gx*R[0,k] + gy*R[1,k]
        sx = gx * r00 + gy * r10 + point_0
        sy = gx * r01 + gy * r11 + point_1
        sz = gx * r02 + gy * r12 + point_2

        # Trilinear interpolation
        ix0 = int(np.floor(sx))
        iy0 = int(np.floor(sy))
        iz0 = int(np.floor(sz))
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        iz1 = iz0 + 1

        # Treat out-of-bounds corners as 0 (matches DeepACSON fill_value=0)
        dx = sx - ix0
        dy = sy - iy0
        dz = sz - iz0

        val = 0.0
        if 0 <= ix0 < sz0 and 0 <= iy0 < sz1 and 0 <= iz0 < sz2:
            val += volume[ix0, iy0, iz0] * (1-dx)*(1-dy)*(1-dz)
        if 0 <= ix0 < sz0 and 0 <= iy0 < sz1 and 0 <= iz1 < sz2:
            val += volume[ix0, iy0, iz1] * (1-dx)*(1-dy)*dz
        if 0 <= ix0 < sz0 and 0 <= iy1 < sz1 and 0 <= iz0 < sz2:
            val += volume[ix0, iy1, iz0] * (1-dx)*dy*(1-dz)
        if 0 <= ix0 < sz0 and 0 <= iy1 < sz1 and 0 <= iz1 < sz2:
            val += volume[ix0, iy1, iz1] * (1-dx)*dy*dz
        if 0 <= ix1 < sz0 and 0 <= iy0 < sz1 and 0 <= iz0 < sz2:
            val += volume[ix1, iy0, iz0] * dx*(1-dy)*(1-dz)
        if 0 <= ix1 < sz0 and 0 <= iy0 < sz1 and 0 <= iz1 < sz2:
            val += volume[ix1, iy0, iz1] * dx*(1-dy)*dz
        if 0 <= ix1 < sz0 and 0 <= iy1 < sz1 and 0 <= iz0 < sz2:
            val += volume[ix1, iy1, iz0] * dx*dy*(1-dz)
        if 0 <= ix1 < sz0 and 0 <= iy1 < sz1 and 0 <= iz1 < sz2:
            val += volume[ix1, iy1, iz1] * dx*dy*dz

        if val >= 0.5:
            row = i // grid_size
            col = i % grid_size
            bw[row, col] = 1

    # Flood-fill from center
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

    return area


def sample_cross_section_fast(volume_bool, point, tangent_vec, g_res):
    """Sample cross-section area using fused Numba kernel."""
    area = _sample_trilinear_flood_fill(
        volume_bool, _base_grid_xy, _grid_size,
        point[0], point[1], point[2],
        tangent_vec[0], tangent_vec[1], tangent_vec[2],
    )
    if area == 0:
        return None
    return area * (g_res ** 2)


def load_zarr_volume(zarr_path: Path):
    """Load level-0 volume and voxel size from an OME-Zarr store."""
    import zarr
    store = zarr.open_group(str(zarr_path), mode="r")
    volume = np.asarray(store["0"])
    multiscales = store.attrs["multiscales"]
    scale = multiscales[0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    return volume, float(scale[0])


def process_single_axon(volume, axon_label, bboxes, voxel_size_um,
                        step_size_um, g_radius, g_res):
    """Process one axon: our skeleton + fast DeepACSON-style cross-sections."""
    bbox = bboxes.get(axon_label)
    if bbox is None:
        return None

    min_coords, max_coords = bbox
    vol_shape = np.array(volume.shape)

    # Tight crop for skeleton
    skel_pad = 5
    min_tight = np.maximum(min_coords - skel_pad, 0)
    max_tight = np.minimum(max_coords + skel_pad, vol_shape)

    subvol_tight = volume[min_tight[0]:max_tight[0],
                          min_tight[1]:max_tight[1],
                          min_tight[2]:max_tight[2]]
    cropped_tight = (subvol_tight == axon_label)

    n_voxels = np.count_nonzero(cropped_tight)
    if n_voxels < 100:
        return None

    # Extract skeleton
    skel_segments = extract_skeleton(cropped_tight, verbose=False, path_method='euler')
    if len(skel_segments) == 0:
        return None

    # Wide crop for cross-section sampling
    sampling_pad = g_radius + 5
    min_padded = np.maximum(min_coords - sampling_pad, 0)
    max_padded = np.minimum(max_coords + sampling_pad, vol_shape)
    tight_to_wide_offset = min_tight - min_padded

    subvol_wide = volume[min_padded[0]:max_padded[0],
                         min_padded[1]:max_padded[1],
                         min_padded[2]:max_padded[2]]
    binary_wide = (subvol_wide == axon_label)

    # Sort segments by length (longest first)
    seg_lengths_px = [len(seg) for seg in skel_segments]
    seg_order = np.argsort(seg_lengths_px)[::-1]
    skel_segments = [skel_segments[i] for i in seg_order]

    voxel_size_tuple = (voxel_size_um, voxel_size_um, voxel_size_um)
    inv_sqrt_pi = 1.0 / np.sqrt(np.pi)
    all_radii_um = []
    all_skel_coords_um = []
    main_length_um = 0.0

    for skel_seg in skel_segments:
        if len(skel_seg) < 3:
            continue

        # Validate skeleton points
        valid_mask = validate_skeleton_points(skel_seg, cropped_tight)
        if not np.any(valid_mask):
            continue
        if not np.all(valid_mask):
            seg_range = find_longest_contiguous_segment(valid_mask)
            if seg_range is None or seg_range[1] - seg_range[0] < 3:
                continue
            skel_seg = skel_seg[seg_range[0]:seg_range[1]]
        if len(skel_seg) < 3:
            continue

        # Convert to wide crop coordinates
        skel_seg = skel_seg.astype(np.float64) + tight_to_wide_offset

        # Arc-length resample
        cumulative_length = compute_arc_length(skel_seg, voxel_size_tuple)
        seg_total_length = cumulative_length[-1]

        if seg_total_length < step_size_um:
            sampled_skel = skel_seg
        else:
            sampled_skel = resample_curve_by_arc_length(skel_seg, step_size_um, voxel_size_tuple)
        if len(sampled_skel) < 2:
            continue

        # Compute tangent vectors
        tangent_vecs = unit_tangent_vector(sampled_skel)

        radii = []
        skel_coords = []

        for point, tangent in zip(sampled_skel, tangent_vecs):
            if tangent[0] == 0 and tangent[1] == 0 and tangent[2] == 0:
                continue

            area = sample_cross_section_fast(binary_wide, point, tangent, g_res)

            if area is not None and area > 0:
                radius_um = np.sqrt(area) * inv_sqrt_pi * voxel_size_um
                radii.append(radius_um)
                global_coord = (point + min_padded) * voxel_size_um
                skel_coords.append(global_coord)

        if len(radii) < 2:
            continue

        radii = np.array(radii)

        seg_length = seg_total_length
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
        'mean_radius_um': np.mean(all_radii_um),
        'std_radius_um': np.std(all_radii_um),
        'length_um': main_length_um,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compute 3D axon profiles: our skeleton + fast DeepACSON cross-sections'
    )
    parser.add_argument('input', type=Path, help='Path to .zarr volume')
    parser.add_argument('output', type=Path, help='Output .npz file')
    parser.add_argument('--max-axons', type=int, default=0,
                        help='Maximum axons to process (0 = all)')
    parser.add_argument('--step-size', type=float, default=0.1,
                        help='Step size along skeleton in μm (default: 0.1)')
    parser.add_argument('--g-radius', type=int, default=15,
                        help='Grid radius for cross-section sampling in voxels (default: 15)')
    parser.add_argument('--g-res', type=float, default=0.25,
                        help='Grid resolution for cross-section sampling in voxels (default: 0.25)')

    args = parser.parse_args()

    # Warmup
    logger.info("Warming up Numba JIT compilation...")
    skeleton_warmup()

    # Warmup fused kernel
    _dummy_vol = np.ones((3, 3, 3), dtype=np.bool_)
    _dummy_grid = np.zeros((1, 2), dtype=np.float64)
    _sample_trilinear_flood_fill(_dummy_vol, _dummy_grid, 1, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0)

    # Load volume
    logger.info(f"Loading Zarr volume: {args.input.name}")
    volume, voxel_size_um = load_zarr_volume(args.input)
    logger.info(f"Volume shape: {volume.shape}, voxel size: {voxel_size_um} μm")

    # Precompute grid
    precompute_base_grid(args.g_radius, args.g_res)
    logger.info(f"Grid: {_grid_size}×{_grid_size} = {_grid_size**2} points")

    # Bounding boxes
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

    logger.info(f"Parameters: step={args.step_size} μm, g_radius={args.g_radius}, g_res={args.g_res}")

    results = []
    for axon_label in tqdm(axon_labels, desc="Processing axons"):
        try:
            result = process_single_axon(
                volume, axon_label, bboxes, voxel_size_um,
                step_size_um=args.step_size,
                g_radius=args.g_radius, g_res=args.g_res,
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

    # Save
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
        method='ours_fast_trilinear',
    )

    logger.info(f"Saved results to {args.output}")

    r_eff = (np.mean(all_radii**6) / np.mean(all_radii**2)) ** 0.25
    logger.info(f"\nSummary:")
    logger.info(f"  Axons processed: {len(results)}")
    logger.info(f"  Total radius samples: {len(all_radii)}")
    logger.info(f"  Mean axon length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f} μm")
    logger.info(f"  r̄: {np.mean(all_radii):.4f} μm")
    logger.info(f"  r_eff: {r_eff:.4f} μm")


if __name__ == '__main__':
    main()
