#!/usr/bin/env python3
"""
Compute per-slice radius histograms from 3D labeled axon volumes.

This script:
1. Loads a labeled volume (.mat file) with optional anisotropic voxel handling
2. Resamples to isotropic resolution if needed (downsampling to coarsest dimension)
3. Slices along a specified axis (Z, Y, or X)
4. For each slice: extracts regions via regionprops → computes circular-equivalent radii → builds histogram
5. Applies symmetric slice filtering based on axon count (min_axon_fraction)
6. Outputs per-slice histograms as NPZ file

This is a refactored version of src/analyze_effective_radius_isotropic.py,
cleaned up for the scripts/ directory with:
- Reusable axonometry library functions
- Batch processing support (glob patterns)
- Cleaner architecture for future extensibility
"""

import argparse
import glob
import logging
import os
import sys
import traceback
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm

# Import from axonometry library
# Find repo root (contains pyproject.toml)
_root = Path(__file__).resolve().parent
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from axonometry.io import load_volume_with_metadata, resample_to_isotropic, construct_output_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for worker processes (shared memory via fork)
_volume_data = None
_slice_axis = None


def _init_worker(volume_data: np.ndarray, slice_axis: int):
    """Initialize worker process with shared volume data and slice axis."""
    global _volume_data, _slice_axis
    _volume_data = volume_data
    _slice_axis = slice_axis


def _process_single_slice(args: Tuple[int, float, np.ndarray, float, float]) -> Tuple[int, np.ndarray, np.ndarray, int]:
    """
    Process a single slice to extract radius histograms (circular-equivalent and minor axis).

    Args:
        args: Tuple of (slice_index, voxel_size_um, bin_edges, max_ellipse_ratio, max_eccentricity)

    Returns:
        Tuple of (slice_index, histogram_counts_circular, histogram_counts_minor, n_axons)
    """
    z, voxel_size_um, bin_edges, max_ellipse_ratio, max_eccentricity = args
    pixel_area_um2 = voxel_size_um * voxel_size_um

    # Extract slice from pre-loaded array
    if _slice_axis == 0:
        slice_2d = _volume_data[z, :, :]
    elif _slice_axis == 1:
        slice_2d = _volume_data[:, z, :]
    else:  # axis == 2
        slice_2d = _volume_data[:, :, z]

    # Use regionprops to efficiently extract areas of all regions
    regions = regionprops(slice_2d.astype(np.int32))

    if len(regions) == 0:
        empty_hist = np.zeros(len(bin_edges) - 1, dtype=np.int32)
        return (z, empty_hist, empty_hist, 0)

    # Compute both circular-equivalent and minor axis radii for each region
    radii_circular = []
    radii_minor = []

    for region in regions:
        area_voxels = region.area

        # Filter by eccentricity if specified (0 = circle, 1 = line)
        if max_eccentricity < 1.0 and region.eccentricity > max_eccentricity:
            continue

        # Filter by ellipse area ratio if specified
        if max_ellipse_ratio > 0:
            # Compute ellipse area from fitted axes (in pixels)
            ellipse_area = np.pi * (region.axis_major_length / 2) * (region.axis_minor_length / 2)
            if ellipse_area > max_ellipse_ratio * area_voxels:
                continue  # Skip sparse/fragmented regions

        # Circular-equivalent radius from area
        area_um2 = area_voxels * pixel_area_um2
        radius_circular_um = np.sqrt(area_um2 / np.pi)
        radii_circular.append(radius_circular_um)

        # Minor axis radius (diameter/2) in micrometers
        radius_minor_um = (region.axis_minor_length / 2) * voxel_size_um
        radii_minor.append(radius_minor_um)

    # Compute histograms
    counts_circular, _ = np.histogram(np.array(radii_circular), bins=bin_edges)
    counts_minor, _ = np.histogram(np.array(radii_minor), bins=bin_edges)

    return (z, counts_circular.astype(np.int32), counts_minor.astype(np.int32), len(radii_circular))


def compute_effective_radius_from_histogram(counts: np.ndarray, bin_centers: np.ndarray) -> float:
    """
    Compute effective MRI-visible radius from histogram.

    Formula: r_eff = (⟨r⁶⟩/⟨r²⟩)^(1/4)

    Args:
        counts: Histogram bin counts
        bin_centers: Histogram bin centers (radius values in μm)

    Returns:
        Effective radius in micrometers
    """
    if counts.sum() == 0:
        return 0.0

    total_count = counts.sum()
    r2_mean = np.sum(counts * bin_centers ** 2) / total_count
    r6_mean = np.sum(counts * bin_centers ** 6) / total_count

    if r2_mean == 0:
        return 0.0

    return (r6_mean / r2_mean) ** 0.25


def compute_slice_profiles(
    mat_file: Path,
    output_file: Path,
    voxel_size_um: Optional[Union[float, Tuple[float, float, float]]] = None,
    slice_axis: int = 0,
    n_jobs: int = -1,
    min_axon_fraction: float = 0.0,
    max_ellipse_ratio: float = 2.0,
    max_eccentricity: float = 1.0
) -> Dict[str, Any]:
    """
    Compute per-slice radius histograms from a labeled volume.

    Computes TWO separate histograms:
    1. Circular-equivalent radius (from area)
    2. Minor axis radius (from fitted ellipse)

    Args:
        mat_file: Path to .mat file with labeled axons
        output_file: Path for output .npz file
        voxel_size_um: Voxel size in micrometers (vz, vy, vx) or scalar.
                       If None, loads from companion JSON metadata file.
        slice_axis: Axis to slice along (0=Z, 1=Y, 2=X).
                    Automatically overridden by 'dominant_axis' from metadata if present.
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        min_axon_fraction: Minimum fraction of max axon count to include slice (0-1)
        max_ellipse_ratio: Max ratio of ellipse area to voxel area (0 = no filter, 2.0 recommended)
        max_eccentricity: Max eccentricity to include (0-1, where 0=circle, 1=line, 1.0=no filter)

    Returns:
        Dictionary with analysis results
    """
    # Load volume and metadata
    logger.info(f"Loading volume from {mat_file.name}")
    volume, voxel_size_tuple, metadata = load_volume_with_metadata(mat_file, voxel_size_um)

    # Check for dominant_axis in metadata and use it if available
    if metadata is not None and 'dominant_axis' in metadata:
        detected_axis = metadata['dominant_axis']
        logger.info(f"Found dominant_axis in metadata: {detected_axis}")
        if slice_axis != detected_axis:
            logger.info(f"Overriding slice_axis {slice_axis} with dominant_axis {detected_axis} from metadata")
            slice_axis = detected_axis
    else:
        logger.info(f"No dominant_axis in metadata, using slice_axis: {slice_axis}")

    # Resample to isotropic if needed
    volume, iso_voxel_size = resample_to_isotropic(volume, voxel_size_tuple)

    logger.info(f"Working with isotropic voxel size: {iso_voxel_size:.4f} μm")
    logger.info(f"Volume shape: {volume.shape}")

    # Get number of slices
    n_slices = volume.shape[slice_axis]
    axis_names = {0: 'Z', 1: 'Y', 2: 'X'}
    logger.info(f"Slicing along axis {slice_axis} ({axis_names[slice_axis]}), {n_slices} slices")

    # Histogram bin edges (0-20 μm with 0.02 μm step)
    bin_edges = np.arange(0, 20.02, 0.02)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Determine number of workers
    if n_jobs == -1:
        n_workers = cpu_count()
    elif n_jobs == 1:
        n_workers = 1
    else:
        n_workers = min(n_jobs, cpu_count())

    logger.info(f"Using {n_workers} parallel workers")
    logger.info(f"Filtering options:")
    logger.info(f"  max_ellipse_ratio: {max_ellipse_ratio:.2f} {'(disabled)' if max_ellipse_ratio == 0 else ''}")
    logger.info(f"  max_eccentricity: {max_eccentricity:.2f} {'(disabled)' if max_eccentricity >= 1.0 else ''}")

    # Prepare arguments for parallel processing
    args_list = [(z, iso_voxel_size, bin_edges, max_ellipse_ratio, max_eccentricity) for z in range(n_slices)]

    # Set globals for workers (works for both parallel and serial)
    global _volume_data, _slice_axis
    _volume_data = volume
    _slice_axis = slice_axis

    # Process slices in parallel
    if n_workers > 1:
        with Pool(processes=n_workers,
                  initializer=_init_worker,
                  initargs=(volume, slice_axis)) as pool:
            results = list(tqdm(
                pool.imap(_process_single_slice, args_list),
                total=n_slices,
                desc="Processing slices"
            ))
    else:
        # Serial fallback for debugging - globals already set above
        results = [_process_single_slice(args) for args in tqdm(args_list, desc="Processing slices")]

    # Convert to dictionaries (separate for circular and minor axis)
    histogram_circular_per_slice = {z: counts_circ for z, counts_circ, _, _ in results}
    histogram_minor_per_slice = {z: counts_minor for z, _, counts_minor, _ in results}
    n_axons_per_slice = {z: n_axons for z, _, _, n_axons in results}

    # Filter slices by axon count (symmetric extent around peak)
    slice_indices = sorted(histogram_circular_per_slice.keys())
    n_axons_array = np.array([n_axons_per_slice[z] for z in slice_indices])

    if min_axon_fraction > 0.0 and n_axons_array.max() > 0:
        max_idx = np.argmax(n_axons_array)
        max_count = n_axons_array[max_idx]
        threshold_count = min_axon_fraction * max_count

        # Find symmetric extent (left and right from peak)
        left_extent = 0
        for i in range(max_idx, -1, -1):
            if n_axons_array[i] >= threshold_count:
                left_extent = max_idx - i
            else:
                break

        right_extent = 0
        for i in range(max_idx, len(n_axons_array)):
            if n_axons_array[i] >= threshold_count:
                right_extent = i - max_idx
            else:
                break

        extent = min(left_extent, right_extent)
        start_idx = max_idx - extent
        end_idx = max_idx + extent + 1

        logger.info(f"Axon count filtering (threshold={min_axon_fraction:.2f}):")
        logger.info(f"  Max count: {max_count} at slice {slice_indices[max_idx]}")
        logger.info(f"  Using slices {start_idx}-{end_idx-1} ({end_idx - start_idx} of {len(slice_indices)})")

        slice_indices = slice_indices[start_idx:end_idx]
        n_axons_array = n_axons_array[start_idx:end_idx]

    # Compute effective radius per slice (for both circular and minor)
    r_eff_circular_per_slice = []
    r_eff_minor_per_slice = []
    for z in slice_indices:
        # Circular-equivalent
        counts_circ = histogram_circular_per_slice[z]
        r_eff_circ = compute_effective_radius_from_histogram(counts_circ, bin_centers)
        r_eff_circular_per_slice.append(r_eff_circ)

        # Minor axis
        counts_minor = histogram_minor_per_slice[z]
        r_eff_minor = compute_effective_radius_from_histogram(counts_minor, bin_centers)
        r_eff_minor_per_slice.append(r_eff_minor)

    r_eff_circular_per_slice = np.array(r_eff_circular_per_slice)
    r_eff_minor_per_slice = np.array(r_eff_minor_per_slice)

    # Compute global effective radius (pooled across all filtered slices)
    total_histogram_circular = np.sum([histogram_circular_per_slice[z] for z in slice_indices], axis=0)
    total_histogram_minor = np.sum([histogram_minor_per_slice[z] for z in slice_indices], axis=0)

    r_eff_circular_global = compute_effective_radius_from_histogram(total_histogram_circular, bin_centers)
    r_eff_minor_global = compute_effective_radius_from_histogram(total_histogram_minor, bin_centers)

    # Statistics
    total_measurements = int(total_histogram_circular.sum())
    mean_radius_circular = np.sum(bin_centers * total_histogram_circular) / total_measurements if total_measurements > 0 else 0.0
    mean_radius_minor = np.sum(bin_centers * total_histogram_minor) / total_measurements if total_measurements > 0 else 0.0

    logger.info(f"\nResults (Circular-Equivalent):")
    logger.info(f"  Global effective radius: {r_eff_circular_global:.3f} μm")
    logger.info(f"  Mean radius: {mean_radius_circular:.3f} μm")
    logger.info(f"\nResults (Minor Axis):")
    logger.info(f"  Global effective radius: {r_eff_minor_global:.3f} μm")
    logger.info(f"  Mean radius: {mean_radius_minor:.3f} μm")
    logger.info(f"\nGeneral:")
    logger.info(f"  Total measurements: {total_measurements}")
    logger.info(f"  Slices used: {len(slice_indices)}/{n_slices}")

    # Save results to NPZ
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for saving
    histograms_circular_array = np.array([histogram_circular_per_slice[z] for z in slice_indices], dtype=np.int32)
    histograms_minor_array = np.array([histogram_minor_per_slice[z] for z in slice_indices], dtype=np.int32)
    n_axons_array_filtered = np.array([n_axons_per_slice[z] for z in slice_indices], dtype=np.int32)

    np.savez(
        output_file,
        # Per-slice data (circular-equivalent)
        slice_indices=np.array(slice_indices, dtype=np.int32),
        histograms=histograms_circular_array,  # Shape: (n_slices, n_bins) - circular-equivalent (for backward compatibility)
        histograms_circular=histograms_circular_array,  # Explicit circular-equivalent
        histograms_minor=histograms_minor_array,  # Minor axis
        n_axons_per_slice=n_axons_array_filtered,
        r_eff_per_slice=r_eff_circular_per_slice,  # Circular (backward compatibility)
        r_eff_circular_per_slice=r_eff_circular_per_slice,
        r_eff_minor_per_slice=r_eff_minor_per_slice,
        # Global pooled data (circular-equivalent)
        total_histogram=total_histogram_circular.astype(np.int32),  # Backward compatibility
        total_histogram_circular=total_histogram_circular.astype(np.int32),
        total_histogram_minor=total_histogram_minor.astype(np.int32),
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
        slice_axis=np.int32(slice_axis),
        slice_axis_name=axis_names[slice_axis],
        max_ellipse_ratio=np.float64(max_ellipse_ratio),
        max_eccentricity=np.float64(max_eccentricity),
        source_file=str(mat_file)
    )

    logger.info(f"Saved results to {output_file}")

    # Return results summary
    return {
        'sample_name': mat_file.stem.replace('_myelinated_axons', ''),
        # Circular-equivalent (for backward compatibility)
        'r_eff_global': float(r_eff_circular_global),
        'r_eff_std': float(np.std(r_eff_circular_per_slice[r_eff_circular_per_slice > 0])) if np.any(r_eff_circular_per_slice > 0) else 0.0,
        'mean_radius': float(mean_radius_circular),
        # Explicit circular
        'r_eff_circular_global': float(r_eff_circular_global),
        'r_eff_circular_std': float(np.std(r_eff_circular_per_slice[r_eff_circular_per_slice > 0])) if np.any(r_eff_circular_per_slice > 0) else 0.0,
        'mean_radius_circular': float(mean_radius_circular),
        # Minor axis
        'r_eff_minor_global': float(r_eff_minor_global),
        'r_eff_minor_std': float(np.std(r_eff_minor_per_slice[r_eff_minor_per_slice > 0])) if np.any(r_eff_minor_per_slice > 0) else 0.0,
        'mean_radius_minor': float(mean_radius_minor),
        # General
        'total_measurements': total_measurements,
        'n_slices_used': len(slice_indices),
        'n_slices_total': n_slices,
        'voxel_size_original': list(voxel_size_tuple),
        'voxel_size_isotropic': float(iso_voxel_size),
        'slice_axis': slice_axis
    }


def batch_compute_slice_profiles(
    matched_files: List[Path],
    output_root: Path,
    voxel_size_um: Optional[Union[float, Tuple[float, float, float]]],
    slice_axis: int,
    n_jobs: int,
    min_axon_fraction: float,
    max_ellipse_ratio: float,
    max_eccentricity: float,
    output_suffix: str = '_slice_profiles',
    organize_by_microscopy: bool = False
):
    """
    Batch process multiple .mat files matched by glob pattern.

    Args:
        matched_files: List of matched .mat file paths
        output_root: Root directory for outputs
        voxel_size_um: Voxel size (optional, loads from JSON if None)
        slice_axis: Axis to slice along (0=Z, 1=Y, 2=X)
        n_jobs: Number of parallel jobs
        min_axon_fraction: Minimum fraction of max axon count
        max_ellipse_ratio: Max ratio of ellipse area to voxel area
        max_eccentricity: Max eccentricity to include (0-1)
        output_suffix: Suffix to append to output filenames (default: '_slice_profiles')
        organize_by_microscopy: If True, organize outputs by microscopy type (HM/LM)
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
        # Construct output path using helper function
        output_file = construct_output_path(
            input_file=mat_file,
            output_root=output_root,
            output_suffix=output_suffix,
            organize_by_microscopy=organize_by_microscopy
        )

        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {i}/{len(matched_files)}: {mat_file.name}")
        logger.info(f"Output: {output_file.relative_to(output_root) if output_file.is_relative_to(output_root) else output_file}")
        logger.info(f"{'='*80}")

        try:
            compute_slice_profiles(
                mat_file,
                output_file,
                voxel_size_um=voxel_size_um,
                slice_axis=slice_axis,
                n_jobs=n_jobs,
                min_axon_fraction=min_axon_fraction,
                max_ellipse_ratio=max_ellipse_ratio,
                max_eccentricity=max_eccentricity
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
    """Parse voxel size CLI argument: single float or comma-separated triple."""
    if ',' in value:
        parts = value.split(',')
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(f"Expected 1 or 3 values, got {len(parts)}")
        return tuple(float(p.strip()) for p in parts)
    return float(value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute per-slice radius histograms from labeled axon volumes'
    )
    parser.add_argument('mat_file', type=str,
                        help="Path to .mat file or glob pattern (e.g., 'data/raw/**/*.mat')")
    parser.add_argument('output_file', type=Path,
                        help='Output .npz file (single file) OR output root directory (batch mode)')
    parser.add_argument('--voxel-size', type=parse_voxel_size_arg, default=None,
                        help='Voxel size in μm: single value (isotropic) or vz,vy,vx. '
                             'If not specified, loads from companion .json file')
    parser.add_argument('--slice-axis', type=int, default=0, choices=[0, 1, 2],
                        help='Axis to slice along: 0=Z, 1=Y, 2=X (default: 0)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (default: -1 = all CPUs)')
    parser.add_argument('--min-axon-fraction', type=float, default=0.0,
                        help='Minimum fraction of max axon count to include slice (default: 0.0)')
    parser.add_argument('--output-suffix', type=str, default='_slice_profiles',
                        help='Suffix to append to output filenames in batch mode (default: "_slice_profiles")')
    parser.add_argument('--organize-by-microscopy', action='store_true',
                        help='Organize outputs by microscopy type (HM/LM folders). '
                             'Files starting with HM_/LM_ will be placed in data/processed/HM/ or data/processed/LM/ '
                             'with the microscopy prefix removed from the filename.')
    parser.add_argument('--max-ellipse-ratio', type=float, default=2.0,
                        help='Max ratio of ellipse area to voxel area (0 = no filter, default: 2.0)')
    parser.add_argument('--max-eccentricity', type=float, default=1.0,
                        help='Max eccentricity to include (0-1, where 0=circle, 1=line, default: 1.0 = no filter)')

    args = parser.parse_args()

    # Expand glob pattern to find matching files
    input_pattern = args.mat_file
    output_path = Path(args.output_file)

    matched_files = sorted(glob.glob(input_pattern, recursive=True))

    if len(matched_files) == 0:
        parser.error(f"No files matched pattern: {input_pattern}")
    elif len(matched_files) == 1:
        # Single file mode
        mat_path = Path(matched_files[0])
        compute_slice_profiles(
            mat_path,
            output_path,
            voxel_size_um=args.voxel_size,
            slice_axis=args.slice_axis,
            n_jobs=args.n_jobs,
            min_axon_fraction=args.min_axon_fraction,
            max_ellipse_ratio=args.max_ellipse_ratio,
            max_eccentricity=args.max_eccentricity
        )
    else:
        # Batch mode
        matched_paths = [Path(f) for f in matched_files]
        batch_compute_slice_profiles(
            matched_paths,
            output_path,
            voxel_size_um=args.voxel_size,
            slice_axis=args.slice_axis,
            n_jobs=args.n_jobs,
            min_axon_fraction=args.min_axon_fraction,
            max_ellipse_ratio=args.max_ellipse_ratio,
            max_eccentricity=args.max_eccentricity,
            output_suffix=args.output_suffix,
            organize_by_microscopy=args.organize_by_microscopy
        )
