#!/usr/bin/env python3
"""
Compute 3D axon profiles using the original DeepACSON CSD approach.

Uses the unmodified DeepACSON skeleton extraction and cross-section sampling
code (from external/DeepACSON/CSD/) to produce axon radius profiles that can
be compared against our compute_3d_axon_profiles.py pipeline.

The DeepACSON approach:
1. Skeleton extraction via FMM + Euler backtracking (step_size=0.1)
2. Skeleton organization: break at junction endpoints, filter by length_threshold
3. For each skeleton segment: sample perpendicular cross-sections using
   trilinear interpolation + connected-component labeling
4. Measure cross-section area via regionprops (equivalent diameter)

Key differences vs our pipeline:
- Skeleton: Euler (subvoxel) vs discrete (voxel-level)
- Cross-section: trilinear interpolation vs nearest-neighbor
- Branch handling: full CSD decomposition vs simple proximity filter
- Filtering: length_threshold on skeleton segments vs n_voxels < 100

Usage:
    python compute_3d_axon_profiles_deepacson.py \\
        data/debug/sham_25_ipsi_cg_myelin_small.zarr \\
        data/debug/sham_25_ipsi_cg_deepacson_profiles.npz
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

# Add DeepACSON CSD to path
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_repo_root / "external" / "DeepACSON" / "CSD"))

from skeleton3D import skeleton as deepacson_skeleton
from unit_tangent_vector import unit_tangent_vector
import plane_rotation as pr

from axonometry.io import load_zarr_volume

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sample_cross_section_deepacson(binary_vol, point, tangent_vec, g_radius, g_res):
    """
    Sample a perpendicular cross-section using the DeepACSON approach.

    Uses trilinear interpolation (RegularGridInterpolator), connected-component
    labeling at center, and regionprops for area measurement.

    Returns equivalent radius in voxels, or None if invalid.
    """
    sz = binary_vol.shape

    # Build the sampling plane grid
    x, y = np.mgrid[-g_radius:g_radius:g_res, -g_radius:g_radius:g_res]
    z = np.zeros_like(x)
    xyz = np.array([np.ravel(x), np.ravel(y), np.ravel(z)]).T

    c_mesh = int((2 * g_radius) / (2 * g_res))
    cent_ball = (x**2 + y**2) < g_res * 1

    # Rotate plane to be perpendicular to tangent
    if np.array_equal(tangent_vec, np.array([0, 0, 0])):
        return None

    rot_axis = pr.unit_normal_vector(tangent_vec, np.array([0, 0, 1]))
    theta = pr.angle(tangent_vec, np.array([0, 0, 1]))
    rot_mat = pr.rotation_matrix_3D(rot_axis, theta)
    rotated_plane = np.squeeze(pr.rotate_vector(xyz, rot_mat))
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

    # Size check (DeepACSON filter)
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


def process_single_axon_deepacson(volume, axon_label, voxel_size_um, g_radius=15, g_res=0.25,
                                  step_voxels=None):
    """
    Process one axon using the DeepACSON approach:
    1. Crop the axon
    2. Extract skeleton via DeepACSON FMM + Euler
    3. Sample cross-sections along each skeleton segment
    4. Return radius profile

    Args:
        volume: Full labeled volume
        axon_label: Integer label for this axon
        voxel_size_um: Isotropic voxel size in micrometers
        g_radius: Grid radius for cross-section sampling (in voxels)
        g_res: Grid resolution for cross-section sampling (in voxels)
        step_voxels: If set, subsample skeleton points at this spacing (in voxels).
            Euler step is ~0.1 voxels, so step_voxels=2 keeps every ~20th point.

    Returns:
        dict with results, or None if failed
    """
    # Find bounding box via argwhere
    coords = np.argwhere(volume == axon_label)
    if len(coords) == 0:
        return None

    pad = g_radius + 5
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1
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

    # DeepACSON skeleton extraction (FMM + Euler backtracking)
    # This returns a list of skeleton segments, each an (N, 3) array
    # Suppress DeepACSON's print statements during skeleton extraction
    import io, contextlib
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            skel_segments = deepacson_skeleton(binary)
    except Exception as e:
        logger.debug(f"Axon {axon_label}: skeleton extraction failed - {e}")
        return None

    if len(skel_segments) == 0:
        return None

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

            radius_voxels = sample_cross_section_deepacson(
                binary, point, tangent, g_radius, g_res
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


def main():
    parser = argparse.ArgumentParser(
        description='Compute 3D axon profiles using original DeepACSON CSD approach'
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

    # Get axon labels (use np.unique since labels may be sparse)
    axon_labels = sorted(np.unique(volume)[1:])  # exclude background 0
    logger.info(f"Found {len(axon_labels)} axons")

    if args.max_axons > 0:
        axon_labels = axon_labels[:args.max_axons]
        logger.info(f"Processing first {args.max_axons} axons")

    step_voxels = None
    if args.step_size is not None:
        step_voxels = args.step_size / voxel_size_um
        logger.info(f"Subsampling skeleton at {args.step_size} μm = {step_voxels:.1f} voxels "
                     f"(stride ~{max(1, round(step_voxels / 0.1))})")

    # Process axons sequentially (DeepACSON skeleton is not easily parallelizable
    # due to print statements and global state in skfmm)
    results = []
    for axon_label in tqdm(axon_labels, desc="Processing axons (DeepACSON)"):
        try:
            result = process_single_axon_deepacson(
                volume, axon_label, voxel_size_um,
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
