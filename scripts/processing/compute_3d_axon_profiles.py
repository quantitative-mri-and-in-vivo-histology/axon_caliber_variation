#!/usr/bin/env python3
"""
Compute fiber morphometry profiles by sampling perpendicular cross-sections along skeletons.

Supports both OME-Zarr volumes (canonical format from preparation pipeline)
and legacy .mat files.

For each axon/fiber:
1. Extract skeleton using fast-marching + path tracing
2. Walk along skeleton points at regular intervals
3. Sample perpendicular cross-section at each point
4. Compute equivalent circular radius from cross-section area

Output: Per-axon arrays of (radius_profile, skeleton_coords, length)

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
from scipy.ndimage import find_objects, median_filter
from tqdm import tqdm

# Import from axonometry library
import sys
# Find repo root (contains pyproject.toml)
_root = Path(__file__).resolve().parent
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

from axonometry import (
    extract_skeleton,
    skeleton_warmup,
    load_volume_with_metadata,
    resample_to_isotropic,
    unit_tangent_vector,
    compute_arc_length,
    resample_curve_by_arc_length,
    validate_skeleton_points,
    find_longest_contiguous_segment,
)
from numba import njit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Global variables for worker processes
_shared_volume = None
_shared_bboxes = None
_base_grid_xyz = None  # Precomputed base grid for cross-section sampling
_grid_size = None


@njit(cache=True, fastmath=True)
def _sample_and_flood_fill(volume, base_grid, grid_size,
                           point_0, point_1, point_2,
                           tan_0, tan_1, tan_2,
                           plane_resolution):
    """
    Fused kernel: rotation + translation + bool interpolation + flood-fill.

    Single Numba function that replaces:
    - create_perpendicular_plane_grid (rotation)
    - nearest_interp_3d_bool (interpolation)
    - flood-fill (connected component at center)

    No Python round-trips, no intermediate arrays except the 2D grid.
    """
    n_grid = base_grid.shape[0]
    sz0, sz1, sz2 = volume.shape

    # Compute rotation matrix from tangent vector
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

    bw = np.zeros((grid_size, grid_size), dtype=np.uint8)

    for i in range(n_grid):
        gx = base_grid[i, 0]
        gy = base_grid[i, 1]

        sx = r00 * gx + r01 * gy + point_0
        sy = r10 * gx + r11 * gy + point_1
        sz = r20 * gx + r21 * gy + point_2

        ix = int(sx + 0.5)
        iy = int(sy + 0.5)
        iz = int(sz + 0.5)

        if ix >= 0 and ix < sz0 and iy >= 0 and iy < sz1 and iz >= 0 and iz < sz2:
            if volume[ix, iy, iz]:
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


def precompute_base_grid(plane_radius, plane_resolution):
    """Precompute the XY plane grid (identical for every cross-section)."""
    global _base_grid_xyz, _grid_size
    coords = np.arange(-plane_radius, plane_radius + plane_resolution, plane_resolution)
    x, y = np.meshgrid(coords, coords)
    z = np.zeros_like(x)
    _base_grid_xyz = np.ascontiguousarray(
        np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    )
    _grid_size = len(coords)


def sample_cross_section_fast(volume_bool, point, tangent_vec, plane_resolution):
    """
    Sample perpendicular cross-section area using fused Numba kernel.

    Single compiled function handles rotation + interpolation + flood-fill
    with no Python round-trips and no intermediate array allocations.
    """
    area = _sample_and_flood_fill(
        volume_bool, _base_grid_xyz, _grid_size,
        point[0], point[1], point[2],
        tangent_vec[0], tangent_vec[1], tangent_vec[2],
        plane_resolution
    )
    if area == 0:
        return None
    return area * (plane_resolution ** 2)


def init_worker():
    """Initialize worker process - globals inherited via fork."""
    skeleton_warmup()


def compute_bounding_boxes(volume: np.ndarray) -> dict:
    """
    Compute bounding boxes for all labels using scipy.ndimage.find_objects.

    This is O(V) for a single pass over the volume, much faster than
    calling np.argwhere for each label separately.
    """
    logger.info("Computing bounding boxes with find_objects (single pass)...")
    slices = find_objects(volume)

    bboxes = {}
    for label_idx, bbox_slices in enumerate(slices):
        if bbox_slices is None:
            continue
        label = label_idx + 1  # find_objects uses 0-indexed, labels are 1-indexed
        min_coords = np.array([s.start for s in bbox_slices])
        max_coords = np.array([s.stop for s in bbox_slices])
        bboxes[label] = (min_coords, max_coords)

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


def process_single_axon(args):
    """
    Process a single axon to extract radius profile along skeleton.

    Args:
        args: Tuple of (axon_label, voxel_size_tuple, plane_radius, plane_resolution,
                       step_size, path_method, skeleton_downsample)

    Returns:
        Dict with radius profile and skeleton coords, or None if failed
    """
    (axon_label, voxel_size_tuple, plane_radius, plane_resolution,
     step_size, path_method, skeleton_downsample) = args

    vz, vy, vx = voxel_size_tuple
    voxel_geom_mean = (vz * vy * vx) ** (1/3)

    try:
        # Get precomputed bounding box
        bbox = _shared_bboxes.get(axon_label)
        if bbox is None:
            return None

        min_coords, max_coords = bbox
        vol_shape = np.array(_shared_volume.shape)

        # Tight crop for skeleton extraction (small padding for boundary distance)
        skel_pad = 5
        min_tight = np.maximum(min_coords - skel_pad, 0)
        max_tight = np.minimum(max_coords + skel_pad, vol_shape)

        subvol_tight = _shared_volume[
            min_tight[0]:max_tight[0],
            min_tight[1]:max_tight[1],
            min_tight[2]:max_tight[2]
        ]
        cropped_tight = (subvol_tight == axon_label)

        # Use count_nonzero instead of argwhere — avoids allocating coordinate array
        n_voxels = np.count_nonzero(cropped_tight)
        if n_voxels < 100:
            return None

        # Extract skeleton on tight crop (FMM runs on much smaller volume)
        if skeleton_downsample > 1:
            cropped_ds = cropped_tight[::skeleton_downsample, ::skeleton_downsample, ::skeleton_downsample]
            skel_segments = extract_skeleton(cropped_ds, verbose=False, path_method=path_method)
            skel_segments = [seg * skeleton_downsample for seg in skel_segments]
        else:
            skel_segments = extract_skeleton(cropped_tight, verbose=False, path_method=path_method)

        # Wide crop for cross-section sampling (needs plane_radius padding)
        sampling_pad = int(plane_radius) + 5
        min_padded = np.maximum(min_coords - sampling_pad, 0)
        max_padded = np.minimum(max_coords + sampling_pad, vol_shape)

        # Offset to convert skeleton coords from tight crop to wide crop space
        tight_to_wide_offset = min_tight - min_padded

        if len(skel_segments) == 0:
            return None

        # Sort segments by length (longest first = main trunk)
        seg_lengths = [len(seg) for seg in skel_segments]
        seg_order = np.argsort(seg_lengths)[::-1]
        skel_segments = [skel_segments[i] for i in seg_order]

        # Extract wide crop for cross-section sampling (shared across all segments)
        subvol_wide = _shared_volume[
            min_padded[0]:max_padded[0],
            min_padded[1]:max_padded[1],
            min_padded[2]:max_padded[2]
        ]
        cropped_wide = (subvol_wide == axon_label)

        # Precompute constants outside the inner loop
        voxel_scale = np.array([vz, vy, vx])
        inv_sqrt_pi = 1.0 / np.sqrt(np.pi)
        voxel_size_tuple_local = (vz, vy, vx)

        # Process all segments (main trunk first, then branches)
        main_result = None
        main_sampled_skel = None  # Store main trunk skeleton for proximity filtering
        branch_radii_list = []
        proximity_threshold_sq = plane_radius ** 2  # squared distance in voxel space

        for skel_seg in skel_segments:
            if len(skel_seg) < 3:
                continue

            # Validate skeleton points are inside the axon (in tight crop space)
            valid_mask = validate_skeleton_points(skel_seg, cropped_tight)
            if not np.any(valid_mask):
                continue

            # Keep only valid skeleton points
            if not np.all(valid_mask):
                segment_range = find_longest_contiguous_segment(valid_mask)
                if segment_range is None or segment_range[1] - segment_range[0] < 3:
                    continue
                skel_seg = skel_seg[segment_range[0]:segment_range[1]]

            if len(skel_seg) < 3:
                continue

            # Convert skeleton from tight crop to wide crop coordinate space
            skel_seg = skel_seg.astype(np.float64) + tight_to_wide_offset

            # Compute arc length and resample
            cumulative_length = compute_arc_length(skel_seg, voxel_size_tuple_local)
            seg_total_length = cumulative_length[-1]

            if seg_total_length < step_size:
                sampled_skel = skel_seg
            else:
                sampled_skel = resample_curve_by_arc_length(skel_seg, step_size, voxel_size_tuple_local)

            if len(sampled_skel) < 2:
                continue

            # For branches: skip points near the main trunk (junction zone)
            if main_sampled_skel is not None:
                diffs = sampled_skel[:, np.newaxis, :] - main_sampled_skel[np.newaxis, :, :]
                min_dist_sq = np.min(np.sum(diffs ** 2, axis=2), axis=1)
                far_mask = min_dist_sq > proximity_threshold_sq
                sampled_skel = sampled_skel[far_mask]
                if len(sampled_skel) < 2:
                    continue

            # Compute tangent vectors
            tangent_vecs = unit_tangent_vector(sampled_skel)

            # Sample cross-sections and compute radii
            radii = []
            valid_skel_points = []

            for point, tangent in zip(sampled_skel, tangent_vecs):
                if tangent[0] == 0 and tangent[1] == 0 and tangent[2] == 0:
                    continue

                area = sample_cross_section_fast(
                    cropped_wide, point, tangent, plane_resolution
                )

                if area is not None and area > 0:
                    radius_um = np.sqrt(area) * inv_sqrt_pi * voxel_geom_mean
                    radii.append(radius_um)
                    valid_skel_points.append((point + min_padded) * voxel_scale)

            if len(radii) < 2:
                continue

            radii = np.array(radii)
            radii = filter_radius_outliers(radii, window_size=5, threshold=3.0)

            if main_result is None:
                # Main trunk (longest segment)
                main_sampled_skel = sampled_skel
                skel_coords = np.array(valid_skel_points)
                main_result = {
                    'label': axon_label,
                    'radii_um': radii,
                    'skeleton_um': skel_coords,
                    'n_points': len(radii),
                    'mean_radius_um': np.mean(radii),
                    'std_radius_um': np.std(radii),
                    'length_um': seg_total_length,
                }
            else:
                # Branch segment — cap at main trunk max to catch residual junction artifacts
                max_main_r = np.max(main_result['radii_um'])
                radii = radii[radii <= max_main_r]
                if len(radii) >= 2:
                    branch_radii_list.append(radii)

        if main_result is None:
            return None

        # Attach branch radii to main result
        main_result['branch_radii_um'] = branch_radii_list

        return main_result

    except Exception as e:
        logger.error(f"Axon {axon_label}: Processing failed - {e}\n{traceback.format_exc()}")
        return None


def load_zarr_volume(zarr_path: Path) -> Tuple[np.ndarray, float]:
    """Load level-0 volume and voxel size from an OME-Zarr store."""
    import zarr

    store = zarr.open_group(str(zarr_path), mode="r")
    volume = np.asarray(store["0"])

    # Read voxel size from OME-NGFF metadata
    multiscales = store.attrs["multiscales"]
    scale = multiscales[0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    voxel_size_z = scale[0]

    if not np.allclose(scale, scale[0]):
        logger.warning(
            f"Zarr voxel size is not isotropic: {scale}. Using Z voxel size ({voxel_size_z})."
        )

    return volume, float(voxel_size_z)


def compute_fiber_profiles(input_path: Path,
                           output_file: Path,
                           voxel_size_um: Optional[Union[float, Tuple[float, float, float]]] = None,
                           max_axon_radius_um: float = 5.0,
                           step_size_um: float = 0.1,
                           n_jobs: int = -1,
                           max_axons: int = 0,
                           anisotropy_mode: str = 'simple',
                           path_method: str = 'discrete',
                           skeleton_downsample: int = 1):
    """
    Compute morphometry profiles for all fibers in a labeled volume.

    Args:
        input_path: Path to .zarr directory or .mat file with labeled axons
        output_file: Path to save results (.npz)
        voxel_size_um: Voxel size in micrometers (scalar or (vz, vy, vx) tuple).
                       If None, auto-detected from Zarr metadata or companion JSON.
        max_axon_radius_um: Maximum expected axon radius in micrometers (sets sampling plane size)
        step_size_um: Step size along skeleton in micrometers
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        max_axons: Maximum number of axons to process (0 = all)
        anisotropy_mode: 'simple' (resample to isotropic) or 'none' (use geometric mean).
                         Ignored for Zarr input (already isotropic by convention).
        path_method: 'discrete' (fast) or 'euler' (subvoxel accuracy)
        skeleton_downsample: Downsample factor for skeleton extraction
    """
    logger.info("Warming up Numba JIT compilation...")
    skeleton_warmup()
    # Warmup fused kernel JIT
    _dummy_vol = np.ones((3, 3, 3), dtype=np.bool_)
    _dummy_grid = np.zeros((1, 3), dtype=np.float64)
    _sample_and_flood_fill(_dummy_vol, _dummy_grid, 1, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0)

    # Load volume
    suffix = input_path.suffix.lower()
    if suffix == ".zarr" or input_path.is_dir():
        logger.info(f"Loading Zarr volume: {input_path.name}")
        volume, iso_voxel_size = load_zarr_volume(input_path)
        voxel_size_tuple = (iso_voxel_size, iso_voxel_size, iso_voxel_size)
    elif suffix == ".mat":
        logger.info(f"Loading .mat volume: {input_path.name}")
        volume, voxel_size_tuple, _ = load_volume_with_metadata(input_path, voxel_size_um)
        # Handle anisotropic voxels
        if anisotropy_mode == 'simple':
            volume, iso_voxel_size = resample_to_isotropic(volume, voxel_size_tuple)
            voxel_size_tuple = (iso_voxel_size, iso_voxel_size, iso_voxel_size)
        elif anisotropy_mode != 'none':
            raise ValueError(f"Unknown anisotropy_mode: {anisotropy_mode}")
    else:
        raise ValueError(f"Unsupported input format: {suffix}. Use .zarr or .mat")

    # Convert max_axon_radius from micrometers to voxels (use geometric mean for anisotropic)
    vz, vy, vx = voxel_size_tuple
    voxel_size_geom_mean = (vz * vy * vx) ** (1/3)
    plane_radius_voxels = int(np.ceil(max_axon_radius_um / voxel_size_geom_mean))
    plane_resolution = 1.0  # Always sample at voxel resolution

    logger.info(f"Max axon radius: {max_axon_radius_um:.2f} μm = {plane_radius_voxels} voxels")

    # Precompute base grid for cross-section sampling (reused for every sample point)
    precompute_base_grid(plane_radius_voxels, plane_resolution)

    # Compute bounding boxes (also gives us all labels — no need for np.unique)
    bboxes = compute_bounding_boxes(volume)

    # Get axon labels from bounding boxes (avoids expensive np.unique on full volume)
    axon_labels = np.array(sorted(bboxes.keys()))

    logger.info(f"Found {len(axon_labels)} axons")

    if max_axons > 0:
        axon_labels = axon_labels[:max_axons]
        logger.info(f"Processing first {max_axons} axons")

    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    args_list = [
        (label, voxel_size_tuple, plane_radius_voxels, plane_resolution,
         step_size_um, path_method, skeleton_downsample)
        for label in axon_labels
    ]

    if vz == vy == vx:
        logger.info(f"Voxel size (isotropic): {vz:.4f} μm")
    else:
        logger.info(f"Voxel size (anisotropic, Z,Y,X): vz={vz:.4f}, vy={vy:.4f}, vx={vx:.4f} μm")

    logger.info(f"Processing {len(args_list)} axons with {n_jobs} workers")
    logger.info(f"Parameters: plane_radius={plane_radius_voxels} voxels ({max_axon_radius_um:.2f} μm), step={step_size_um} μm")

    # Set globals for workers
    global _shared_volume, _shared_bboxes
    _shared_volume = volume
    _shared_bboxes = bboxes

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

    # Pool radii: main trunk only (wo branches) and all segments (w branches)
    all_radii_wo = np.concatenate([r['radii_um'] for r in results])
    all_radii_w_parts = [r['radii_um'] for r in results]
    for r in results:
        all_radii_w_parts.extend(r['branch_radii_um'])
    all_radii_w = np.concatenate(all_radii_w_parts)

    np.savez(
        output_file,
        labels=labels,
        n_points=n_points,
        mean_radii_um=mean_radii,
        std_radii_um=std_radii,
        lengths_um=lengths,
        radii_profiles_um=radii_profiles,
        skeleton_coords_um=skeleton_coords,
        all_radii_wo_branches_um=all_radii_wo,
        all_radii_w_branches_um=all_radii_w,
        voxel_size_um=np.array(voxel_size_tuple),
        max_axon_radius_um=max_axon_radius_um,
        plane_radius_voxels=plane_radius_voxels,
        plane_resolution=plane_resolution,
        step_size_um=step_size_um,
        source_file=str(input_path)
    )

    logger.info(f"Saved results to {output_file}")

    # Summary statistics
    n_branch_radii = len(all_radii_w) - len(all_radii_wo)
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total axons processed: {len(results)}")
    logger.info(f"  Radius samples (main trunk): {len(all_radii_wo)}")
    logger.info(f"  Radius samples (branches): {n_branch_radii}")
    logger.info(f"  Radius samples (total): {len(all_radii_w)}")
    logger.info(f"  Mean axon length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f} μm")
    logger.info(f"  Mean points per axon: {np.mean(n_points):.1f}")

    r_eff_wo = (np.mean(all_radii_wo**6) / np.mean(all_radii_wo**2)) ** 0.25
    r_eff_w = (np.mean(all_radii_w**6) / np.mean(all_radii_w**2)) ** 0.25
    logger.info(f"  r_eff (wo branches): {r_eff_wo:.3f} μm")
    logger.info(f"  r_eff (w branches):  {r_eff_w:.3f} μm")
    logger.info(f"  r̄ (wo branches): {np.mean(all_radii_wo):.3f} μm")
    logger.info(f"  r̄ (w branches):  {np.mean(all_radii_w):.3f} μm")


def batch_compute_fiber_profiles(
    matched_files: List[Path],
    output_root: Path,
    voxel_size_um: Optional[Union[float, Tuple[float, float, float]]],
    max_axon_radius_um: float,
    step_size_um: float,
    n_jobs: int,
    max_axons: int,
    anisotropy_mode: str,
    path_method: str,
    skeleton_downsample: int,
    output_suffix: str = '_axon_profiles',
):
    """
    Batch process multiple .zarr/.mat files matched by glob pattern.

    Args:
        matched_files: List of matched file paths (.zarr directories or .mat files)
        output_root: Root directory for outputs
        voxel_size_um: Voxel size (same as compute_fiber_profiles)
        max_axon_radius_um: Max axon radius in micrometers
        step_size_um: Step size along skeleton in micrometers
        n_jobs: Number of parallel jobs
        max_axons: Max number of axons to process per file
        anisotropy_mode: Anisotropy handling mode
        path_method: Skeleton path extraction method
        skeleton_downsample: Skeleton extraction downsample factor
        output_suffix: Suffix to append to output filenames (default: '_axon_profiles')
    """
    # Find common root directory
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
        # Construct output filename
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
                max_axon_radius_um=max_axon_radius_um,
                step_size_um=step_size_um,
                n_jobs=n_jobs,
                max_axons=max_axons,
                anisotropy_mode=anisotropy_mode,
                path_method=path_method,
                skeleton_downsample=skeleton_downsample
            )
            successful.append(input_file.name)
        except Exception as e:
            error_msg = str(e)
            failed.append((input_file.name, error_msg))
            logger.error(f"Failed to process {input_file.name}: {error_msg}")
            traceback.print_exc()
            continue

    # Print summary
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
        description='Compute fiber morphometry profiles by sampling perpendicular cross-sections'
    )
    parser.add_argument('input', type=str,
                        help="Path to .zarr directory, .mat file, or glob pattern")
    parser.add_argument('output', type=Path,
                        help='Output .npz file (single file) OR output directory (batch mode)')
    parser.add_argument('--voxel-size', type=parse_voxel_size_arg, default=None,
                        help='Voxel size in μm: single value (isotropic) or vz,vy,vx. '
                             'If not specified, loads from companion .json file (default: from JSON or 0.05)')
    parser.add_argument('--max-axon-radius', type=float, default=5.0,
                        help='Maximum expected axon radius in μm (sets sampling plane size, default: 5.0)')
    parser.add_argument('--step-size', type=float, default=0.1,
                        help='Step size along skeleton in μm (default: 0.1)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 = all CPUs)')
    parser.add_argument('--max-axons', type=int, default=0,
                        help='Maximum axons to process (0 = all)')
    parser.add_argument('--anisotropy-mode', type=str, default='simple',
                        choices=['simple', 'none'],
                        help="'simple' resamples to isotropic, 'none' uses geometric mean")
    parser.add_argument('--path-method', type=str, default='discrete',
                        choices=['discrete', 'euler'],
                        help="Skeleton path method: 'discrete' (fast) or 'euler' (subvoxel)")
    parser.add_argument('--skeleton-downsample', type=int, default=1,
                        help="Downsample factor for skeleton extraction (default: 1)")
    parser.add_argument('--output-suffix', type=str, default='_axon_profiles',
                        help='Suffix to append to output filenames in batch mode (default: "_axon_profiles")')

    args = parser.parse_args()

    # Expand glob pattern to find matching files/directories
    input_pattern = args.input
    output_path = args.output

    matched_files = sorted(glob.glob(input_pattern, recursive=True))

    if len(matched_files) == 0:
        parser.error(f"No files matched pattern: {input_pattern}")
    elif len(matched_files) == 1:
        # Single file mode
        input_path = Path(matched_files[0])

        # If output is a directory, construct filename
        if output_path.suffix != ".npz":
            stem = input_path.stem
            if input_path.suffix == ".zarr" or (input_path.is_dir() and ".zarr" in input_path.name):
                stem = input_path.with_suffix("").stem if "." in input_path.stem else input_path.stem
            output_path = output_path / f"{stem}{args.output_suffix}.npz"

        compute_fiber_profiles(
            input_path,
            output_path,
            voxel_size_um=args.voxel_size,
            max_axon_radius_um=args.max_axon_radius,
            step_size_um=args.step_size,
            n_jobs=args.n_jobs,
            max_axons=args.max_axons,
            anisotropy_mode=args.anisotropy_mode,
            path_method=args.path_method,
            skeleton_downsample=args.skeleton_downsample
        )
    else:
        # Batch mode - multiple files matched
        matched_paths = [Path(f) for f in matched_files]
        if output_path.suffix == ".npz":
            parser.error("Output must be a directory in batch mode")
        batch_compute_fiber_profiles(
            matched_paths,
            output_path,
            voxel_size_um=args.voxel_size,
            max_axon_radius_um=args.max_axon_radius,
            step_size_um=args.step_size,
            n_jobs=args.n_jobs,
            max_axons=args.max_axons,
            anisotropy_mode=args.anisotropy_mode,
            path_method=args.path_method,
            skeleton_downsample=args.skeleton_downsample,
            output_suffix=args.output_suffix,
        )
