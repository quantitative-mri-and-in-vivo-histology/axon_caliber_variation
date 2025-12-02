#!/usr/bin/env python3
"""
Compute per-population slice profiles for LM data with CC/CG identification.

This specialized script for LM (low magnification) data:
1. Identifies CC and CG populations using dominant-axis classification
2. For each population, extracts slices perpendicular to that population's dominant axis
3. Outputs separate NPZ files per population: *_cc_slice_profiles.npz, *_cg_slice_profiles.npz

This integrates the workflows from:
- identify_cc_cg_populations.py (population identification)
- compute_slice_profiles_2d.py (slice extraction and histogram computation)

The combined workflow ensures each population is sliced along its correct dominant axis
for accurate 2D vs 3D metric comparisons.
"""

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

# Import from axonometry library
sys.path.insert(0, str(Path(__file__).parent.parent))
from axonometry.io import load_volume_with_metadata, resample_to_isotropic, construct_output_path
from axonometry.populations import (
    load_volume_downsampled,
    precompute_axon_voxels,
    compute_all_orientations,
    classify_by_dominant_axis,
    filter_sparse_axons,
    create_populations
)

# We'll implement slice processing functions with population filtering inline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Global worker state for population-filtered processing
_worker_volume = None
_worker_slice_axis = None
_worker_label_filter = None


def _init_worker_with_filter(volume, slice_axis, label_filter):
    """Initialize worker process with volume and label filter."""
    global _worker_volume, _worker_slice_axis, _worker_label_filter
    _worker_volume = volume
    _worker_slice_axis = slice_axis
    _worker_label_filter = label_filter


def _process_slice_batch_filtered(args):
    """Process batch of slices with population label filtering."""
    from skimage.measure import regionprops

    slice_indices, bin_edges, voxel_size, max_ellipse_ratio, max_eccentricity = args

    results = []
    for z in slice_indices:
        if _worker_slice_axis == 0:
            slice_2d = _worker_volume[z, :, :]
        elif _worker_slice_axis == 1:
            slice_2d = _worker_volume[:, z, :]
        else:  # axis 2
            slice_2d = _worker_volume[:, :, z]

        # Filter to population labels only
        if _worker_label_filter is not None:
            mask = np.isin(slice_2d, list(_worker_label_filter))
            slice_2d = np.where(mask, slice_2d, 0)

        # Skip empty slices
        if slice_2d.max() == 0:
            hist_circ = np.zeros(len(bin_edges) - 1, dtype=np.int32)
            hist_minor = np.zeros(len(bin_edges) - 1, dtype=np.int32)
            results.append((z, hist_circ, hist_minor, 0))
            continue

        # Extract regions
        props = regionprops(slice_2d)

        radii_circular = []
        radii_minor = []

        for prop in props:
            # Filter by ellipse area ratio
            if prop.area == 0:
                continue

            ellipse_area = np.pi * prop.axis_major_length * prop.axis_minor_length / 4
            area_ratio = ellipse_area / prop.area if prop.area > 0 else np.inf

            if area_ratio > max_ellipse_ratio:
                continue

            # Filter by eccentricity
            if prop.eccentricity > max_eccentricity:
                continue

            # Circular-equivalent radius
            r_circ = np.sqrt(prop.area / np.pi) * voxel_size
            radii_circular.append(r_circ)

            # Minor-axis radius
            r_minor = prop.axis_minor_length / 2 * voxel_size
            radii_minor.append(r_minor)

        # Compute histograms
        hist_circ, _ = np.histogram(radii_circular, bins=bin_edges)
        hist_minor, _ = np.histogram(radii_minor, bins=bin_edges)

        results.append((z, hist_circ.astype(np.int32), hist_minor.astype(np.int32), len(radii_circular)))

    return results


def apply_symmetric_slice_filter(n_axons_per_slice: Dict[int, int],
                                  min_axon_fraction: float) -> List[int]:
    """
    Filter slices to keep a symmetric extent around the peak axon count.

    Args:
        n_axons_per_slice: Dictionary mapping slice index to axon count
        min_axon_fraction: Minimum fraction of max axon count to retain

    Returns:
        List of slice indices that pass the filter
    """
    if not n_axons_per_slice:
        return []

    slice_indices_all = sorted(n_axons_per_slice.keys())
    axon_counts = np.array([n_axons_per_slice[z] for z in slice_indices_all])

    if min_axon_fraction <= 0:
        return slice_indices_all

    max_count = axon_counts.max()
    threshold = min_axon_fraction * max_count

    # Find peak location
    peak_idx = np.argmax(axon_counts)

    # Expand symmetrically from peak
    left = peak_idx
    right = peak_idx

    while left > 0 or right < len(axon_counts) - 1:
        if left > 0 and axon_counts[left - 1] >= threshold:
            left -= 1
        if right < len(axon_counts) - 1 and axon_counts[right + 1] >= threshold:
            right += 1

        # Stop if both sides can't expand
        if (left == 0 or axon_counts[left - 1] < threshold) and \
           (right == len(axon_counts) - 1 or axon_counts[right + 1] < threshold):
            break

    return [slice_indices_all[i] for i in range(left, right + 1)]


def compute_effective_radius_from_histogram(counts: np.ndarray, bin_centers: np.ndarray) -> float:
    """Compute effective radius from histogram."""
    total_counts = counts.sum()
    if total_counts == 0:
        return np.nan

    r2_mean = np.sum(bin_centers**2 * counts) / total_counts
    r6_mean = np.sum(bin_centers**6 * counts) / total_counts

    if r2_mean <= 0 or r6_mean <= 0:
        return np.nan

    return (r6_mean / r2_mean) ** 0.25


def compute_population_slice_profiles(
    mat_file: Path,
    output_root: Path,
    voxel_size_um: Optional[Union[float, Tuple[float, float, float]]] = None,
    downsample: int = 4,
    max_angle_deg: float = 30.0,
    min_length_um: float = 50.0,
    k_neighbors: int = 10,
    max_neighbor_distance_um: float = 30.0,
    n_jobs: int = -1,
    min_axon_fraction: float = 0.0,
    max_ellipse_ratio: float = 2.0,
    max_eccentricity: float = 0.9,
    output_suffix: str = "_slice_profiles",
    organize_by_microscopy: bool = False
) -> List[Path]:
    """
    Identify CC/CG populations and compute slice profiles for each.

    Args:
        mat_file: Path to .mat file with labeled axons
        output_root: Root directory for outputs
        voxel_size_um: Voxel size (if None, loads from JSON)
        downsample: Downsampling factor for population identification
        max_angle_deg: Maximum angle from dominant axis for classification
        min_length_um: Minimum axon length to include
        k_neighbors: KNN neighbors for sparse filtering
        max_neighbor_distance_um: Max distance for KNN filtering
        n_jobs: Number of parallel jobs for slice processing
        min_axon_fraction: Minimum fraction of max axon count per slice
        max_ellipse_ratio: Max ellipse area ratio filter
        max_eccentricity: Max eccentricity filter
        output_suffix: Suffix for output files
        organize_by_microscopy: Organize by HM/LM subdirectories

    Returns:
        List of output file paths (one per population)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing LM data with population identification: {mat_file.name}")
    logger.info(f"{'='*80}\n")

    # Load voxel size
    if voxel_size_um is None:
        _, voxel_size_tuple, _ = load_volume_with_metadata(mat_file, voxel_size_override=None)
        voxel_size_um = (voxel_size_tuple[0] * voxel_size_tuple[1] * voxel_size_tuple[2]) ** (1/3)
        logger.info(f"Loaded voxel size: {voxel_size_tuple} μm (geometric mean: {voxel_size_um:.4f} μm)")

    # Step 1: Identify populations
    logger.info("\nStep 1: Identifying CC/CG populations...")
    volume_ds, _ = load_volume_downsampled(mat_file, downsample)
    voxel_size_ds = voxel_size_um * downsample

    min_voxels = 3
    axon_voxels_ds = precompute_axon_voxels(volume_ds)
    axon_data = compute_all_orientations(axon_voxels_ds, voxel_size_ds, min_voxels, min_length_um)

    if not axon_data:
        raise ValueError("No valid axons found after filtering")

    label_to_bundle, axis_to_bundle = classify_by_dominant_axis(axon_data, max_angle_deg)

    if not label_to_bundle:
        raise ValueError("No axons remain after classification")

    populations = create_populations(
        axon_data, label_to_bundle, axis_to_bundle, axon_voxels_ds, voxel_size_ds,
        k_neighbors, max_neighbor_distance_um
    )

    if len(populations) < 2:
        raise ValueError(f"Only {len(populations)} population(s) found, expected 2 (CC and CG)")

    logger.info(f"\nIdentified {len(populations)} populations:")
    for pop in populations:
        logger.info(f"  {pop['name'].upper()}: {pop['n_axons']} axons, "
                   f"dominant_axis={pop['dominant_axis_name']} ({pop['dominant_axis']})")

    # Load full-resolution volume once for all populations
    logger.info(f"\nLoading full-resolution volume...")
    volume_full, voxel_size_tuple, _ = load_volume_with_metadata(mat_file, voxel_size_override=None)

    # Resample to isotropic if needed
    if len(set(voxel_size_tuple)) > 1:
        logger.info(f"Resampling to isotropic resolution...")
        volume_iso, iso_voxel_size = resample_to_isotropic(volume_full, voxel_size_tuple)
        del volume_full  # Free memory
    else:
        volume_iso = volume_full
        iso_voxel_size = voxel_size_tuple[0]

    # Step 2: Compute slice profiles for each population
    output_files = []

    for pop in populations:
        pop_name = pop['name']
        pop_labels = set(pop['axon_labels'])
        dominant_axis = pop['dominant_axis']

        logger.info(f"\nStep 2.{pop_name.upper()}: Computing slice profiles for {pop_name.upper()} population...")
        logger.info(f"  Dominant axis: {pop['dominant_axis_name']} ({dominant_axis})")
        logger.info(f"  Axon count: {pop['n_axons']}")

        # Construct output path with population suffix
        base_output_path = construct_output_path(
            mat_file,
            output_root,
            output_suffix=f"_{pop_name}{output_suffix}",
            organize_by_microscopy=organize_by_microscopy
        )

        # Process slices with population label filtering (no pre-filtering of volume)
        logger.info(f"  Extracting slices along axis {dominant_axis} ({pop['dominant_axis_name']})...")
        logger.info(f"  Filtering to {len(pop_labels)} axons during slice processing...")

        from multiprocessing import Pool, cpu_count

        # Determine number of jobs
        if n_jobs == -1:
            n_jobs_actual = cpu_count()
        elif n_jobs <= 0:
            n_jobs_actual = max(1, cpu_count() + n_jobs + 1)
        else:
            n_jobs_actual = n_jobs

        n_slices = volume_iso.shape[dominant_axis]
        slice_indices = list(range(n_slices))

        # Histogram parameters
        bin_edges = np.arange(0, 20.02, 0.02)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Initialize histograms
        histogram_circular_per_slice = {}
        histogram_minor_per_slice = {}
        n_axons_per_slice = {}

        # Parallel processing with progress bar using filtered worker
        with Pool(n_jobs_actual, initializer=_init_worker_with_filter,
                  initargs=(volume_iso, dominant_axis, pop_labels)) as pool:
            batch_size = max(1, n_slices // (n_jobs_actual * 4))
            batches = [slice_indices[i:i+batch_size] for i in range(0, len(slice_indices), batch_size)]

            args_list = [(batch, bin_edges, iso_voxel_size, max_ellipse_ratio, max_eccentricity)
                        for batch in batches]

            results = list(tqdm(
                pool.imap(_process_slice_batch_filtered, args_list),
                total=len(batches),
                desc=f"  Processing {pop_name.upper()} slices"
            ))

        # Aggregate results
        for batch_results in results:
            for z, hist_circ, hist_minor, n_axons in batch_results:
                histogram_circular_per_slice[z] = hist_circ
                histogram_minor_per_slice[z] = hist_minor
                n_axons_per_slice[z] = n_axons

        # Apply symmetric slice filtering
        if min_axon_fraction > 0:
            slice_indices_filtered = apply_symmetric_slice_filter(
                n_axons_per_slice, min_axon_fraction
            )
            logger.info(f"  Symmetric filtering: {len(slice_indices_filtered)}/{n_slices} slices retained")
        else:
            slice_indices_filtered = slice_indices

        # Compute aggregated histograms
        total_histogram_circular = np.sum([histogram_circular_per_slice[z] for z in slice_indices_filtered], axis=0).astype(np.int32)
        total_histogram_minor = np.sum([histogram_minor_per_slice[z] for z in slice_indices_filtered], axis=0).astype(np.int32)

        # Compute effective radii
        def compute_r_eff(hist, bin_centers):
            counts = hist.sum()
            if counts == 0:
                return np.nan
            radii = np.repeat(bin_centers, hist.astype(int))
            r2_mean = np.mean(radii ** 2)
            r6_mean = np.mean(radii ** 6)
            if r2_mean == 0:
                return np.nan
            return (r6_mean / r2_mean) ** 0.25

        r_eff_circular_per_slice = np.array([
            compute_r_eff(histogram_circular_per_slice[z], bin_centers)
            for z in slice_indices_filtered
        ])
        r_eff_minor_per_slice = np.array([
            compute_r_eff(histogram_minor_per_slice[z], bin_centers)
            for z in slice_indices_filtered
        ])

        r_eff_circular_global = compute_r_eff(total_histogram_circular, bin_centers)
        r_eff_minor_global = compute_r_eff(total_histogram_minor, bin_centers)

        # Compute mean radii
        total_counts_circ = total_histogram_circular.sum()
        mean_radius_circular = np.sum(bin_centers * total_histogram_circular) / total_counts_circ if total_counts_circ > 0 else np.nan

        total_counts_minor = total_histogram_minor.sum()
        mean_radius_minor = np.sum(bin_centers * total_histogram_minor) / total_counts_minor if total_counts_minor > 0 else np.nan

        # Prepare arrays
        histograms_circular_array = np.array([histogram_circular_per_slice[z] for z in slice_indices_filtered], dtype=np.int32)
        histograms_minor_array = np.array([histogram_minor_per_slice[z] for z in slice_indices_filtered], dtype=np.int32)
        n_axons_array_filtered = np.array([n_axons_per_slice[z] for z in slice_indices_filtered], dtype=np.int32)

        # Save NPZ
        base_output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            base_output_path,
            # Per-slice data
            slice_indices=np.array(slice_indices_filtered, dtype=np.int32),
            histograms=histograms_circular_array,  # Backward compatibility
            histograms_circular=histograms_circular_array,
            histograms_minor=histograms_minor_array,
            n_axons_per_slice=n_axons_array_filtered,
            r_eff_per_slice=r_eff_circular_per_slice,  # Backward compatibility
            r_eff_circular_per_slice=r_eff_circular_per_slice,
            r_eff_minor_per_slice=r_eff_minor_per_slice,
            # Global pooled data
            total_histogram=total_histogram_circular,  # Backward compatibility
            total_histogram_circular=total_histogram_circular,
            total_histogram_minor=total_histogram_minor,
            r_eff_global=np.float64(r_eff_circular_global),  # Backward compatibility
            r_eff_circular_global=np.float64(r_eff_circular_global),
            r_eff_minor_global=np.float64(r_eff_minor_global),
            mean_radius=np.float64(mean_radius_circular),  # Backward compatibility
            mean_radius_circular=np.float64(mean_radius_circular),
            mean_radius_minor=np.float64(mean_radius_minor),
            # Metadata
            bin_edges=bin_edges,
            bin_centers=bin_centers,
            voxel_size_original=np.array(voxel_size_tuple, dtype=np.float64),
            voxel_size_isotropic=np.float64(iso_voxel_size),
            slice_axis=np.int32(dominant_axis),
            slice_axis_name=pop['dominant_axis_name'],
            max_ellipse_ratio=np.float64(max_ellipse_ratio),
            max_eccentricity=np.float64(max_eccentricity),
            source_file=str(mat_file),
            # Population metadata
            population_name=pop_name,
            population_n_axons=np.int32(pop['n_axons']),
            population_dominant_axis=np.int32(dominant_axis),
            population_dominant_axis_name=pop['dominant_axis_name']
        )

        logger.info(f"  Saved: {base_output_path}")
        logger.info(f"  Summary:")
        logger.info(f"    Circular r_eff: {r_eff_circular_global:.3f} μm")
        logger.info(f"    Minor r_eff: {r_eff_minor_global:.3f} μm")
        logger.info(f"    Mean radius (circular): {mean_radius_circular:.3f} μm")

        output_files.append(base_output_path)

    return output_files


def main():
    parser = argparse.ArgumentParser(
        description='Compute per-population slice profiles for LM data with CC/CG identification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python compute_slice_profiles_lm_2d.py \\
      data/raw/LM/LM_25_ipsi_myelinated_axons.mat \\
      data/processed

  # Batch mode
  python compute_slice_profiles_lm_2d.py \\
      "data/raw/**/LM*_myelinated_axons.mat" \\
      data/processed \\
      --organize-by-microscopy
        """
    )
    parser.add_argument('mat_file', type=str,
                        help="Path to .mat file or glob pattern")
    parser.add_argument('output_root', type=Path,
                        help='Output root directory')
    parser.add_argument('--downsample', type=int, default=4,
                        help='Downsampling for population identification (default: 4)')
    parser.add_argument('--max-angle', type=float, default=30.0,
                        help='Max angle from dominant axis (default: 30.0°)')
    parser.add_argument('--min-length', type=float, default=50.0,
                        help='Minimum axon length in μm (default: 50.0)')
    parser.add_argument('--k-neighbors', type=int, default=10,
                        help='KNN neighbors for sparse filtering (default: 10)')
    parser.add_argument('--max-neighbor-distance', type=float, default=30.0,
                        help='Max KNN distance in μm (default: 30.0)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (default: -1 = all CPUs)')
    parser.add_argument('--min-axon-fraction', type=float, default=0.0,
                        help='Minimum fraction of max axon count per slice (default: 0.0)')
    parser.add_argument('--max-ellipse-ratio', type=float, default=2.0,
                        help='Max ellipse area ratio (default: 2.0)')
    parser.add_argument('--max-eccentricity', type=float, default=0.9,
                        help='Max eccentricity (default: 0.9)')
    parser.add_argument('--output-suffix', type=str, default='_slice_profiles',
                        help='Suffix for output files (default: "_slice_profiles")')
    parser.add_argument('--organize-by-microscopy', action='store_true',
                        help='Organize by HM/LM subdirectories')

    args = parser.parse_args()

    # Expand glob pattern
    matched_files = sorted(glob.glob(args.mat_file, recursive=True))

    if len(matched_files) == 0:
        parser.error(f"No files matched pattern: {args.mat_file}")

    logger.info(f"Found {len(matched_files)} file(s) to process\n")

    for i, mat_path in enumerate(matched_files, 1):
        logger.info(f"\n{'#'*80}")
        logger.info(f"Processing file {i}/{len(matched_files)}: {Path(mat_path).name}")
        logger.info(f"{'#'*80}")

        try:
            output_files = compute_population_slice_profiles(
                Path(mat_path),
                args.output_root,
                downsample=args.downsample,
                max_angle_deg=args.max_angle,
                min_length_um=args.min_length,
                k_neighbors=args.k_neighbors,
                max_neighbor_distance_um=args.max_neighbor_distance,
                n_jobs=args.n_jobs,
                min_axon_fraction=args.min_axon_fraction,
                max_ellipse_ratio=args.max_ellipse_ratio,
                max_eccentricity=args.max_eccentricity,
                output_suffix=args.output_suffix,
                organize_by_microscopy=args.organize_by_microscopy
            )

            logger.info(f"\nSuccess! Generated {len(output_files)} output files:")
            for out_file in output_files:
                logger.info(f"  - {out_file}")

        except Exception as e:
            logger.error(f"Failed to process {Path(mat_path).name}: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"\n{'#'*80}")
    logger.info(f"Batch processing complete: {len(matched_files)} files processed")
    logger.info(f"{'#'*80}\n")


if __name__ == '__main__':
    main()
