#!/usr/bin/env python3
"""
Compute fiber morphometry profiles by sampling perpendicular cross-sections along skeletons.

For each axon/fiber:
1. Extract skeleton using fast-marching + path tracing
2. Walk along skeleton points at regular intervals
3. Sample perpendicular cross-section at each point
4. Compute equivalent circular radius from cross-section area

Output: Per-axon arrays of (radius_profile, skeleton_coords, length)
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
from scipy.ndimage import zoom, find_objects, median_filter
from tqdm import tqdm

# Import from axonometry library
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from axonometry import (
    extract_skeleton,
    skeleton_warmup,
    load_volume_with_metadata,
    resample_to_isotropic,
    unit_tangent_vector,
    compute_arc_length,
    resample_curve_by_arc_length,
    sample_perpendicular_cross_section,
    validate_skeleton_points,
    find_longest_contiguous_segment,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Global variables for worker processes
_shared_volume = None
_shared_bboxes = None


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

        # Add padding for cross-section sampling
        padding = int(plane_radius) + 5
        vol_shape = np.array(_shared_volume.shape)
        min_padded = np.maximum(min_coords - padding, 0)
        max_padded = np.minimum(max_coords + padding, vol_shape)

        # Extract subvolume and create binary mask
        subvol = _shared_volume[
            min_padded[0]:max_padded[0],
            min_padded[1]:max_padded[1],
            min_padded[2]:max_padded[2]
        ]

        cropped = (subvol == axon_label).astype(np.uint8)

        local_coords = np.argwhere(cropped)
        if len(local_coords) < 100:
            return None

        # Extract skeleton (optionally downsampled)
        if skeleton_downsample > 1:
            cropped_ds = zoom(cropped, 1.0 / skeleton_downsample, order=0)
            skel_segments = extract_skeleton(cropped_ds, verbose=False, path_method=path_method)
            skel_segments = [seg * skeleton_downsample for seg in skel_segments]
        else:
            skel_segments = extract_skeleton(cropped, verbose=False, path_method=path_method)

        if len(skel_segments) == 0:
            return None

        # Use the longest skeleton segment
        if len(skel_segments) == 1:
            main_skel = skel_segments[0]
        else:
            lengths = [len(seg) for seg in skel_segments]
            main_skel = skel_segments[np.argmax(lengths)]

        if len(main_skel) < 3:
            return None

        # Validate skeleton points are inside the axon
        valid_mask = validate_skeleton_points(main_skel, cropped)
        if not np.any(valid_mask):
            return None

        # Keep only valid skeleton points
        if not np.all(valid_mask):
            segment_range = find_longest_contiguous_segment(valid_mask)
            if segment_range is None or segment_range[1] - segment_range[0] < 3:
                return None
            main_skel = main_skel[segment_range[0]:segment_range[1]]

        if len(main_skel) < 3:
            return None

        # Compute arc length and resample
        voxel_size_tuple = (vz, vy, vx)
        cumulative_length = compute_arc_length(main_skel, voxel_size_tuple)
        total_length = cumulative_length[-1]

        if total_length < step_size:
            sampled_skel = main_skel
        else:
            sampled_skel = resample_curve_by_arc_length(main_skel, step_size, voxel_size_tuple)

        if len(sampled_skel) < 2:
            return None

        # Compute tangent vectors
        tangent_vecs = unit_tangent_vector(sampled_skel)

        # Pre-convert volume to float64 for interpolation
        volume_float = cropped.astype(np.float64)

        # Sample cross-sections and compute radii
        radii = []
        valid_skel_points = []

        for point, tangent in zip(sampled_skel, tangent_vecs):
            if np.allclose(tangent, 0):
                continue

            area = sample_perpendicular_cross_section(
                cropped, point, tangent,
                plane_radius=plane_radius,
                plane_resolution=plane_resolution,
                volume_float=volume_float
            )

            if area is not None and area > 0:
                radius_voxels = np.sqrt(area / np.pi)
                radius_um = radius_voxels * voxel_geom_mean

                radii.append(radius_um)
                original_point = point + min_padded
                valid_skel_points.append(original_point * np.array([vz, vy, vx]))

        if len(radii) < 2:
            return None

        radii = np.array(radii)
        skel_coords = np.array(valid_skel_points)

        # Filter outliers
        radii = filter_radius_outliers(radii, window_size=5, threshold=3.0)

        return {
            'label': axon_label,
            'radii_um': radii,
            'skeleton_um': skel_coords,
            'n_points': len(radii),
            'mean_radius_um': np.mean(radii),
            'std_radius_um': np.std(radii),
            'length_um': total_length
        }

    except Exception as e:
        logger.error(f"Axon {axon_label}: Processing failed - {e}\n{traceback.format_exc()}")
        return None


def compute_fiber_profiles(mat_file: Path,
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
        mat_file: Path to .mat file with labeled axons
        output_file: Path to save results (.npz)
        voxel_size_um: Voxel size in micrometers (scalar or (vz, vy, vx) tuple).
                       If None, loads from companion JSON file.
        max_axon_radius_um: Maximum expected axon radius in micrometers (sets sampling plane size)
        step_size_um: Step size along skeleton in micrometers
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        max_axons: Maximum number of axons to process (0 = all)
        anisotropy_mode: 'simple' (resample to isotropic) or 'none' (use geometric mean)
        path_method: 'discrete' (fast) or 'euler' (subvoxel accuracy)
        skeleton_downsample: Downsample factor for skeleton extraction
    """
    logger.info("Warming up Numba JIT compilation...")
    skeleton_warmup()

    # Load volume and metadata (voxel size from override, JSON, or default)
    logger.info(f"Loading {mat_file}")
    volume, voxel_size_tuple, _ = load_volume_with_metadata(mat_file, voxel_size_um)

    # Handle anisotropic voxels
    if anisotropy_mode == 'simple':
        volume, iso_voxel_size = resample_to_isotropic(volume, voxel_size_tuple)
        voxel_size_tuple = (iso_voxel_size, iso_voxel_size, iso_voxel_size)
    elif anisotropy_mode != 'none':
        raise ValueError(f"Unknown anisotropy_mode: {anisotropy_mode}")

    # Convert max_axon_radius from micrometers to voxels (use geometric mean for anisotropic)
    vz, vy, vx = voxel_size_tuple
    voxel_size_geom_mean = (vz * vy * vx) ** (1/3)
    plane_radius_voxels = int(np.ceil(max_axon_radius_um / voxel_size_geom_mean))
    plane_resolution = 1.0  # Always sample at voxel resolution

    logger.info(f"Max axon radius: {max_axon_radius_um:.2f} μm = {plane_radius_voxels} voxels")

    # Compute bounding boxes
    bboxes = compute_bounding_boxes(volume)

    # Get unique axon labels
    axon_labels = np.unique(volume)
    axon_labels = axon_labels[axon_labels > 0]

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
        voxel_size_um=np.array(voxel_size_tuple),
        max_axon_radius_um=max_axon_radius_um,
        plane_radius_voxels=plane_radius_voxels,
        plane_resolution=plane_resolution,
        step_size_um=step_size_um,
        source_file=str(mat_file)
    )

    logger.info(f"Saved results to {output_file}")

    # Summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total axons processed: {len(results)}")
    logger.info(f"  Total radius samples: {len(all_radii)}")
    logger.info(f"  Mean axon length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f} μm")
    logger.info(f"  Mean points per axon: {np.mean(n_points):.1f}")
    logger.info(f"  Mean radius: {np.mean(all_radii):.3f} ± {np.std(all_radii):.3f} μm")
    logger.info(f"  Median radius: {np.median(all_radii):.3f} μm")

    r_eff = (np.mean(all_radii**6) / np.mean(all_radii**2)) ** 0.25
    logger.info(f"  Effective radius (r_eff): {r_eff:.3f} μm")


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
    output_suffix: str = '_axon_profiles'
):
    """
    Batch process multiple .mat files matched by glob pattern.

    Preserves directory structure relative to common root and derives output filenames.

    Args:
        matched_files: List of matched .mat file paths
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

    for i, mat_file in enumerate(matched_files, 1):
        # Compute relative path and output path
        try:
            relative_path = mat_file.relative_to(common_root)
        except ValueError:
            # If relative_to fails, just use the filename
            relative_path = mat_file.name

        # Construct output path with suffix: stem + suffix + .npz
        output_file = output_root / relative_path.parent / f"{relative_path.stem}{output_suffix}.npz"

        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {i}/{len(matched_files)}: {mat_file.name}")
        logger.info(f"Output: {output_file.relative_to(output_root) if output_file.is_relative_to(output_root) else output_file}")
        logger.info(f"{'='*80}")

        try:
            compute_fiber_profiles(
                mat_file,
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
            successful.append(mat_file.name)
        except Exception as e:
            error_msg = str(e)
            failed.append((mat_file.name, error_msg))
            logger.error(f"Failed to process {mat_file.name}: {error_msg}")
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
    parser.add_argument('mat_file', type=str,
                        help="Path to .mat file or glob pattern (e.g., 'data/raw/**/*.mat')")
    parser.add_argument('output_file', type=Path,
                        help='Output .npz file (single file) OR output root directory (batch mode)')
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

    # Expand glob pattern to find matching files
    input_pattern = args.mat_file
    output_path = Path(args.output_file)

    matched_files = sorted(glob.glob(input_pattern, recursive=True))

    if len(matched_files) == 0:
        parser.error(f"No files matched pattern: {input_pattern}")
    elif len(matched_files) == 1:
        # Single file mode (existing behavior)
        mat_path = Path(matched_files[0])
        compute_fiber_profiles(
            mat_path,
            output_path,  # Treat as output file
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
        batch_compute_fiber_profiles(
            matched_paths,
            output_path,  # Treat as output root directory
            voxel_size_um=args.voxel_size,
            max_axon_radius_um=args.max_axon_radius,
            step_size_um=args.step_size,
            n_jobs=args.n_jobs,
            max_axons=args.max_axons,
            anisotropy_mode=args.anisotropy_mode,
            path_method=args.path_method,
            skeleton_downsample=args.skeleton_downsample,
            output_suffix=args.output_suffix
        )
