#!/usr/bin/env python3
"""
Compute 3D axon profiles reproducing the DeepACSON cross-section approach.

Uses our skeleton extraction (FMM + discrete path tracing) but switches to
DeepACSON's cross-section sampling method:
- Trilinear interpolation (RegularGridInterpolator)
- Connected-component labeling at center
- regionprops for area measurement
- Sub-voxel grid resolution (g_res=0.25, g_radius=15 voxels)

This isolates the effect of the cross-section method from the skeleton method.

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
from scipy.ndimage import find_objects
from scipy.interpolate import RegularGridInterpolator as rgi
from skimage.measure import label, regionprops
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


def load_zarr_volume(zarr_path: Path):
    """Load level-0 volume and voxel size from an OME-Zarr store."""
    import zarr
    store = zarr.open_group(str(zarr_path), mode="r")
    volume = np.asarray(store["0"])
    multiscales = store.attrs["multiscales"]
    scale = multiscales[0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    return volume, float(scale[0])


def sample_cross_section(binary_vol, point, tangent_vec, g_radius, g_res):
    """
    Sample a perpendicular cross-section using the DeepACSON approach.

    Trilinear interpolation + connected-component labeling + regionprops area.

    Returns equivalent radius in voxels, or None if invalid.
    """
    sz = binary_vol.shape

    # Build the sampling plane grid
    x, y = np.mgrid[-g_radius:g_radius:g_res, -g_radius:g_radius:g_res]
    z = np.zeros_like(x)
    xyz = np.array([np.ravel(x), np.ravel(y), np.ravel(z)]).T

    c_mesh = int((2 * g_radius) / (2 * g_res))
    cent_ball = (x**2 + y**2) < g_res * 1

    # Rotate plane to be perpendicular to tangent (DeepACSON rotation)
    t = tangent_vec
    z_axis = np.array([0.0, 0.0, 1.0])

    if np.array_equal(t, np.array([0, 0, 0])):
        return None

    # unit_normal_vector (cross product axis)
    rot_axis = np.cross(t, z_axis)
    rot_norm = np.sqrt(np.dot(rot_axis, rot_axis))
    if rot_norm < 1e-5:
        rot_axis = t
    else:
        rot_axis = rot_axis / max(rot_norm, 1e-5)

    # angle between tangent and z-axis
    theta = np.arccos(np.dot(t, z_axis) / (np.sqrt(np.dot(t, t)) * np.sqrt(np.dot(z_axis, z_axis))))

    # Euler-Rodrigues rotation matrix
    a = np.cos(theta / 2.0)
    b, c, d = -rot_axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a**2, b**2, c**2, d**2
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    rot_mat = np.array([
        [aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
        [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
        [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc],
    ])

    # DeepACSON applies as row_vector @ rot_mat
    rotated_plane = np.dot(xyz, rot_mat)
    cross_section_plane = rotated_plane + point

    # Trilinear interpolation
    interpolating_func = rgi(
        (range(sz[0]), range(sz[1]), range(sz[2])),
        binary_vol.astype(np.float64),
        bounds_error=False, fill_value=0
    )
    cross_section = interpolating_func(cross_section_plane)
    bw_cross_section = cross_section >= 0.5
    bw_cross_section = np.reshape(bw_cross_section, x.shape)

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
    area_voxels = area_pixels * (g_res ** 2)
    radius_voxels = np.sqrt(area_voxels / np.pi)

    return radius_voxels


def process_single_axon(volume, axon_label, bboxes, voxel_size_um,
                        step_size_um, g_radius, g_res):
    """
    Process one axon: our skeleton + DeepACSON cross-sections.
    """
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
    binary_wide = (subvol_wide == axon_label).astype(np.float64)

    # Sort segments by length (longest first)
    seg_lengths_px = [len(seg) for seg in skel_segments]
    seg_order = np.argsort(seg_lengths_px)[::-1]
    skel_segments = [skel_segments[i] for i in seg_order]

    voxel_size_tuple = (voxel_size_um, voxel_size_um, voxel_size_um)
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

            radius_voxels = sample_cross_section(
                binary_wide, point, tangent, g_radius, g_res
            )

            if radius_voxels is not None and radius_voxels > 0:
                radius_um = radius_voxels * voxel_size_um
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
        description='Compute 3D axon profiles: our skeleton + DeepACSON cross-sections'
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

    # Load volume
    logger.info(f"Loading Zarr volume: {args.input.name}")
    volume, voxel_size_um = load_zarr_volume(args.input)
    logger.info(f"Volume shape: {volume.shape}, voxel size: {voxel_size_um} μm")

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
        method='ours_trilinear',
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
