"""Axon morphometry profiling: skeleton extraction + cross-section sampling.

Two backends for per-axon processing:
- process_single_axon_original(): verbatim DeepACSON (skeleton step=0.1, stride subsampling)
- process_single_axon_fast(): optimized DeepACSON (euler_step = sampling step, two-crop)

Volume-level orchestration:
- compute_axon_radius_profiles(): runs the pipeline on all axons in a labeled 3D volume
"""

import contextlib
import io
import logging
import multiprocessing
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import find_objects
from tqdm import tqdm

from .deepacson.original import (
    skeleton as deepacson_skeleton,
    sample_cross_section as deepacson_sample_cross_section,
    unit_tangent_vector as deepacson_unit_tangent_vector,
    get_line_length as deepacson_get_line_length,
)
from .deepacson.fast import (
    skeleton as fast_skeleton,
    sample_cross_section as fast_sample_cross_section,
    find_g_radius as fast_find_g_radius,
    unit_tangent_vector as fast_unit_tangent_vector,
    get_line_length as fast_get_line_length,
)
from .io import load_zarr_volume

logger = logging.getLogger(__name__)


# ===========================================================================
# Fork-inherited globals for parallel processing
# ===========================================================================

# Module-level references set before Pool fork — workers inherit read-only
# access via copy-on-write. Workers only slice into the volume (no writes),
# so COW pages are not materialized for the volume data.
_shared_volume = None
_shared_bboxes = None


def _worker_process_axon_fast(args):
    """Worker function for fast backend (fork-inherited globals)."""
    axon_label, voxel_size_um, g_radius, g_res, step_voxels = args
    return process_single_axon_fast(
        _shared_volume, axon_label, _shared_bboxes[axon_label],
        voxel_size_um, g_radius, g_res, step_voxels
    )


def _worker_process_axon_original(args):
    """Worker function for original backend (fork-inherited globals)."""
    axon_label, voxel_size_um, g_radius, g_res, step_voxels = args
    return process_single_axon_original(
        _shared_volume, axon_label, _shared_bboxes[axon_label],
        voxel_size_um, g_radius, g_res, step_voxels
    )


# ===========================================================================
# Bounding boxes
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


# ===========================================================================
# Per-axon processing
# ===========================================================================

def process_single_axon_original(volume, axon_label, bbox, voxel_size_um,
                                 g_radius, g_res, step_voxels):
    """Process a single axon using verbatim DeepACSON backend.

    Single crop (bbox + g_radius + 5), skeleton with Euler step=0.1,
    subsampled at step_voxels spacing for cross-section sampling.
    """
    try:
        min_coords, max_coords = bbox
        vol_shape = np.array(volume.shape)

        pad = g_radius + 5
        min_padded = np.maximum(min_coords - pad, 0)
        max_padded = np.minimum(max_coords + pad, vol_shape)

        subvol = volume[min_padded[0]:max_padded[0],
                        min_padded[1]:max_padded[1],
                        min_padded[2]:max_padded[2]]
        binary = (subvol == axon_label).astype(np.float64)

        if np.count_nonzero(binary) == 0:
            return None

        with contextlib.redirect_stdout(io.StringIO()):
            skel_segments = deepacson_skeleton(binary)

        if len(skel_segments) == 0:
            return None

        segments = []

        for skel_seg in skel_segments:
            # np.gradient needs >= 3 points for tangent vector computation
            if len(skel_seg) < 3:
                continue

            if step_voxels is not None:
                stride = max(1, round(step_voxels / 0.1))
                skel_seg = skel_seg[::stride]
                if len(skel_seg) < 3:
                    continue

            tangent_vecs = deepacson_unit_tangent_vector(skel_seg)

            radii = []
            skel_points = []

            for pt, tangent in zip(skel_seg, tangent_vecs):
                area_pixels = deepacson_sample_cross_section(
                    binary, pt, tangent, g_radius, g_res
                )

                if area_pixels > 0:
                    area_voxels = area_pixels * (g_res ** 2)
                    radius_voxels = np.sqrt(area_voxels / np.pi)
                    radii.append(radius_voxels)
                    skel_points.append(pt.copy())

            if len(radii) < 2:
                continue

            seg_length = deepacson_get_line_length(skel_seg)
            radii_um = np.array(radii) * voxel_size_um
            skel_global = (np.array(skel_points) + min_padded) * voxel_size_um
            segments.append({
                'radii_um': radii_um,
                'skeleton_um': skel_global,
                'length_um': seg_length * voxel_size_um,
            })

        if not segments:
            return None

        return {
            'label': axon_label,
            'segments': segments,
            'n_segments': len(segments),
        }

    except Exception as e:
        logger.debug(f"Axon {axon_label}: failed - {e}")
        return None


def process_single_axon_fast(volume, axon_label, bbox, voxel_size_um,
                             g_radius, g_res, step_voxels):
    """Process a single axon using optimized DeepACSON backend.

    Two-crop approach + adaptive per-point grid sizing:
    (1) Tight crop (bbox + 5) for skeleton extraction — FMM Euler step
        equals step_voxels, so every skeleton point is a sampling point.
    (2) Wide crop (bbox + adapted radius) for cross-section sampling.
    (3) Per-point adaptive g_radius via find_g_radius(): starts at
        ceil(maxD) + 5, doubles until the cross-section border is clear.
    """
    try:
        min_coords, max_coords = bbox
        vol_shape = np.array(volume.shape)

        # --- Pass 1: tight crop for skeleton extraction (fast FMM) ---
        tight_pad = 5
        min_tight = np.maximum(min_coords - tight_pad, 0)
        max_tight = np.minimum(max_coords + tight_pad, vol_shape)

        tight_subvol = volume[min_tight[0]:max_tight[0],
                              min_tight[1]:max_tight[1],
                              min_tight[2]:max_tight[2]]
        tight_binary = (tight_subvol == axon_label)

        if np.count_nonzero(tight_binary) == 0:
            return None

        # Euler step 0.1 matches the original DeepACSON integrator resolution.
        # Cost is negligible (~2s extra per 50 axons) vs FMM + cross-sections.
        euler_step = 0.1
        skel_segments, maxD = fast_skeleton(tight_binary, verbose=False,
                                            euler_step=euler_step)
        del tight_binary  # free before allocating wide crop
        if len(skel_segments) == 0:
            return None

        # Initial per-point g_radius from FMM inscribed radius
        g_radius_init = int(np.ceil(maxD)) + 5

        # --- Pass 2: wide crop for cross-section sampling ---
        # Pad by actual inscribed radius (not global max), capped at g_radius.
        wide_pad = min(g_radius_init * 2, g_radius) + 5
        min_wide = np.maximum(min_coords - wide_pad, 0)
        max_wide = np.minimum(max_coords + wide_pad, vol_shape)

        wide_subvol = volume[min_wide[0]:max_wide[0],
                             min_wide[1]:max_wide[1],
                             min_wide[2]:max_wide[2]]
        wide_binary = (wide_subvol == axon_label)

        # Offset skeleton coordinates from tight-crop to wide-crop space
        skel_offset = min_tight - min_wide

        segments = []

        for skel_seg in skel_segments:
            # np.gradient needs >= 3 points for tangent vector computation
            if len(skel_seg) < 3:
                continue

            if step_voxels is not None:
                stride = max(1, round(step_voxels / euler_step))
                skel_seg = skel_seg[::stride]
                if len(skel_seg) < 3:
                    continue

            tangent_vecs = fast_unit_tangent_vector(skel_seg)

            radii = []
            skel_points = []

            for pt, tangent in zip(skel_seg, tangent_vecs):
                pt_wide = pt + skel_offset
                g_r = fast_find_g_radius(
                    wide_binary, pt_wide, tangent, g_radius_init, g_radius
                )
                area_pixels = fast_sample_cross_section(
                    wide_binary, pt_wide, tangent, g_r, g_res
                )

                if area_pixels > 0:
                    area_voxels = area_pixels * (g_res ** 2)
                    radius_voxels = np.sqrt(area_voxels / np.pi)
                    radii.append(radius_voxels)
                    skel_points.append(pt.copy())

            if len(radii) < 2:
                continue

            seg_length = fast_get_line_length(skel_seg)
            radii_um = np.array(radii) * voxel_size_um
            skel_global = (np.array(skel_points) + min_tight) * voxel_size_um
            segments.append({
                'radii_um': radii_um,
                'skeleton_um': skel_global,
                'length_um': seg_length * voxel_size_um,
            })

        if not segments:
            return None

        return {
            'label': axon_label,
            'segments': segments,
            'n_segments': len(segments),
        }

    except Exception as e:
        logger.debug(f"Axon {axon_label}: failed - {e}")
        return None


# ===========================================================================
# Main orchestration
# ===========================================================================

def _compute_axon_radius_profiles_original(volume, axon_labels, bboxes,
                                           voxel_size, g_radius, g_res,
                                           step_voxels, n_jobs):
    """Run original DeepACSON backend on all axons (serial or parallel)."""
    results = []

    if n_jobs != 1:
        global _shared_volume, _shared_bboxes
        workers = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        logger.info(f"Parallel processing with {workers} workers")

        _shared_volume = volume
        _shared_bboxes = bboxes

        work_items = [
            (lbl, voxel_size, g_radius, g_res, step_voxels)
            for lbl in axon_labels
        ]
        try:
            with multiprocessing.Pool(processes=workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(_worker_process_axon_original,
                                        work_items),
                    total=len(work_items),
                    desc=f"Processing axons (original, {workers} cores)",
                ):
                    if result is not None:
                        results.append(result)
        finally:
            _shared_volume = None
            _shared_bboxes = None
    else:
        for axon_label in tqdm(axon_labels, desc="Processing axons (original)"):
            result = process_single_axon_original(
                volume, axon_label, bboxes[axon_label],
                voxel_size, g_radius, g_res, step_voxels
            )
            if result is not None:
                results.append(result)

    return results


def _compute_axon_radius_profiles_fast(volume, axon_labels, bboxes,
                                       voxel_size, g_radius, g_res,
                                       step_voxels, n_jobs):
    """Run optimized DeepACSON backend on all axons (serial or parallel)."""
    results = []

    if n_jobs != 1:
        global _shared_volume, _shared_bboxes
        workers = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        logger.info(f"Parallel processing with {workers} workers")

        # JIT warmup before fork
        logger.info("Warming up Numba JIT (before fork)...")
        _dummy = np.zeros((20, 20, 20), dtype=np.float64)
        _dummy[5:15, 5:15, 5:15] = 1.0
        try:
            fast_skeleton(_dummy, verbose=False)
        except Exception:
            pass
        logger.info("JIT warmup done")

        _shared_volume = volume
        _shared_bboxes = bboxes

        work_items = [
            (lbl, voxel_size, g_radius, g_res, step_voxels)
            for lbl in axon_labels
        ]
        try:
            with multiprocessing.Pool(processes=workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(_worker_process_axon_fast, work_items),
                    total=len(work_items),
                    desc=f"Processing axons (fast, {workers} cores)",
                ):
                    if result is not None:
                        results.append(result)
        finally:
            _shared_volume = None
            _shared_bboxes = None
    else:
        for axon_label in tqdm(axon_labels, desc="Processing axons (fast)"):
            result = process_single_axon_fast(
                volume, axon_label, bboxes[axon_label],
                voxel_size, g_radius, g_res, step_voxels
            )
            if result is not None:
                results.append(result)

    return results


def compute_axon_radius_profiles(input_path: Path,
                                 output_file: Path,
                                 max_radius_um: float = 5.0,
                                 step_size_um: float = 0.05,
                                 max_axons: int = 0,
                                 axon_ids: Optional[list] = None,
                                 backend: str = 'fast',
                                 n_jobs: int = 1):
    """
    Compute radius profiles for all axons in a labeled volume.

    Args:
        input_path: Path to .zarr directory with labeled axons
        output_file: Path to save results (.npz)
        max_radius_um: Maximum expected axon radius in micrometers (sets grid size)
        step_size_um: Step size along skeleton in micrometers
        max_axons: Maximum number of axons to process (0 = all)
        axon_ids: If set, only process these specific axon labels
        backend: 'fast' (optimized) or 'original' (verbatim DeepACSON)
        n_jobs: Number of parallel workers (1 = serial, -1 = all cores)
    """
    # Load volume
    logger.info(f"Loading Zarr volume: {input_path.name}")
    volume, voxel_size = load_zarr_volume(input_path)

    # Convert max_radius from μm to voxels for grid sizing
    g_radius = int(np.ceil(max_radius_um / voxel_size))
    g_res = 0.25

    logger.info(f"Volume shape: {volume.shape}, voxel: {voxel_size:.4f} μm")
    logger.info(f"Backend: {backend}")
    logger.info(f"Max axon radius: {max_radius_um:.1f} μm = {g_radius} voxels")

    # Compute bounding boxes
    bboxes = compute_bounding_boxes(volume)

    if axon_ids is not None:
        axon_labels = sorted([l for l in axon_ids if l in bboxes])
        missing = set(axon_ids) - set(axon_labels)
        if missing:
            logger.warning(f"Axon IDs not found in volume: {missing}")
        logger.info(f"Processing {len(axon_labels)} specified axons")
    else:
        axon_labels = sorted(bboxes.keys())
        logger.info(f"Found {len(axon_labels)} axons")

    if max_axons > 0:
        axon_labels = axon_labels[:max_axons]
        logger.info(f"Processing first {max_axons} axons")

    step_voxels = step_size_um / voxel_size if step_size_um is not None else None

    if backend == 'fast':
        logger.info(f"Parameters: g_radius={g_radius} voxels, g_res={g_res}, "
                    f"step={step_size_um} μm = {step_voxels:.1f} voxels "
                    f"(euler_step = sampling step, no stride)")
    else:
        logger.info(f"Parameters: g_radius={g_radius} voxels, g_res={g_res}, "
                    f"step={step_size_um} μm = {step_voxels:.1f} voxels "
                    f"(stride ~{max(1, round(step_voxels / 0.1))})")

    # Dispatch to backend
    if backend == 'fast':
        results = _compute_axon_radius_profiles_fast(
            volume, axon_labels, bboxes, voxel_size, g_radius, g_res,
            step_voxels, n_jobs
        )
    else:
        results = _compute_axon_radius_profiles_original(
            volume, axon_labels, bboxes, voxel_size, g_radius, g_res,
            step_voxels, n_jobs
        )

    logger.info(f"Successfully processed {len(results)}/{len(axon_labels)} axons")

    if len(results) == 0:
        logger.error("No axons were successfully processed!")
        return

    # Save results — raw per-segment data, no summary stats or filtering.
    # Endpoint trimming, summary stats, and segment selection are deferred
    # to analysis scripts where they can be adjusted without reprocessing.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    labels = np.array([r['label'] for r in results])
    n_segments = np.array([r['n_segments'] for r in results])

    # Per-axon per-segment data (object arrays of object arrays)
    segment_radii = np.array([
        np.array([s['radii_um'] for s in r['segments']], dtype=object)
        for r in results
    ], dtype=object)
    segment_skeletons = np.array([
        np.array([s['skeleton_um'] for s in r['segments']], dtype=object)
        for r in results
    ], dtype=object)
    segment_lengths = np.array([
        np.array([s['length_um'] for s in r['segments']])
        for r in results
    ], dtype=object)

    # Pooled radii from all segments of all axons
    all_radii = np.concatenate([
        np.concatenate([s['radii_um'] for s in r['segments']])
        for r in results
    ])

    np.savez(
        output_file,
        labels=labels,
        n_segments=n_segments,
        segment_radii_um=segment_radii,
        segment_skeletons_um=segment_skeletons,
        segment_lengths_um=segment_lengths,
        all_radii_um=all_radii,
        voxel_size_um=voxel_size,
        max_radius_um=max_radius_um,
        g_radius_voxels=g_radius,
        step_size_um=step_size_um,
        source_file=str(input_path),
        method=f'deepacson_csd_{backend}',
    )

    logger.info(f"Saved results to {output_file}")

    # Summary statistics
    r_eff = (np.mean(all_radii**6) / np.mean(all_radii**2)) ** 0.25
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total axons processed: {len(results)}")
    logger.info(f"  Total radius samples: {len(all_radii)}")
    logger.info(f"  r̄: {np.mean(all_radii):.4f} μm")
    logger.info(f"  r_eff: {r_eff:.4f} μm")
