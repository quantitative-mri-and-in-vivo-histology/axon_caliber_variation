#!/usr/bin/env python3
"""
Analyze effective MRI-visible radius from labeled axon volumes (.mat files).

Handles anisotropic voxels by downsampling to isotropic resolution first
(using the coarsest dimension as reference).

Computes the effective radius per slice using the formula:
    r_eff = (<r^6>/<r^2>)^(1/4)

This metric is relevant for diffusion MRI sensitivity to axon caliber.

Only supports .mat files (MATLAB v5.0 and v7.3/HDF5 formats).
"""

import argparse
import json
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom
from skimage.measure import regionprops
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for worker processes
_volume_data = None
_slice_axis = None


def parse_voxel_size(voxel_size_um: Union[float, Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """
    Parse voxel size to a (vz, vy, vx) tuple matching array axes (Z, Y, X).

    Args:
        voxel_size_um: Scalar (isotropic) or (vz, vy, vx) tuple (anisotropic)

    Returns:
        Tuple of (vz, vy, vx) in micrometers, matching array axes (Z, Y, X)
    """
    if isinstance(voxel_size_um, (tuple, list)):
        if len(voxel_size_um) == 3:
            return tuple(float(v) for v in voxel_size_um)
        elif len(voxel_size_um) == 1:
            v = float(voxel_size_um[0])
            return (v, v, v)
        else:
            raise ValueError(f"Expected 1 or 3 voxel size values, got {len(voxel_size_um)}")
    v = float(voxel_size_um)
    return (v, v, v)


def parse_voxel_size_arg(value: str) -> Union[float, Tuple[float, float, float]]:
    """Parse voxel size CLI argument: single float or comma-separated triple."""
    if ',' in value:
        parts = value.split(',')
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"Expected 1 or 3 values, got {len(parts)}: '{value}'"
            )
        return tuple(float(p.strip()) for p in parts)
    return float(value)


def resample_to_isotropic(volume: np.ndarray,
                          voxel_size: Tuple[float, float, float]
                          ) -> Tuple[np.ndarray, float]:
    """
    Resample anisotropic volume to isotropic voxels.

    Downsamples to the coarsest voxel dimension to avoid upsampling artifacts.
    Uses nearest-neighbor interpolation (order=0) to preserve label integrity.

    Args:
        volume: 3D labeled volume with shape (Z, Y, X)
        voxel_size: Tuple of (vz, vy, vx) voxel sizes in micrometers, matching array axes

    Returns:
        Tuple of (resampled_volume, isotropic_voxel_size)
    """
    vz, vy, vx = voxel_size

    # Check if already isotropic
    if np.allclose([vz, vy, vx], vz):
        logger.info("Volume is already isotropic, no resampling needed")
        return volume, vz

    # Target voxel size is the coarsest dimension
    target_size = max(vz, vy, vx)

    # Compute zoom factors (< 1 means downsampling)
    zoom_factors = (vz / target_size, vy / target_size, vx / target_size)

    logger.info(f"Resampling anisotropic volume to isotropic:")
    logger.info(f"  Original voxel size (Z, Y, X): vz={vz:.4f}, vy={vy:.4f}, vx={vx:.4f} μm")
    logger.info(f"  Target voxel size: {target_size:.4f} μm (isotropic)")
    logger.info(f"  Zoom factors (Z, Y, X): {zoom_factors}")
    logger.info(f"  Original shape: {volume.shape}")

    # Use order=0 (nearest-neighbor) to preserve label integrity
    resampled = zoom(volume, zoom_factors, order=0, mode='nearest')

    logger.info(f"  Resampled shape: {resampled.shape}")

    return resampled, target_size


def load_metadata(mat_file: Path) -> dict:
    """
    Load metadata JSON file associated with a .mat file.

    Looks for a file with the same name but .json extension.
    For example: HM_25_ipsi_myelinated_axons.mat -> HM_25_ipsi_myelinated_axons.json

    Args:
        mat_file: Path to .mat file

    Returns:
        Dict with metadata (empty dict if not found)
    """
    metadata_file = mat_file.with_suffix('.json')

    if metadata_file.exists():
        logger.info(f"Found metadata file: {metadata_file}")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return metadata
    else:
        logger.info(f"No metadata file found at {metadata_file}")
        return {}


def load_mat_volume(mat_file: Path) -> np.ndarray:
    """
    Load labeled volume from .mat file (v5.0 or v7.3/HDF5).

    Args:
        mat_file: Path to .mat file

    Returns:
        3D labeled volume array
    """
    logger.info(f"Loading {mat_file}")

    # Try HDF5 format first (MATLAB v7.3)
    try:
        with h5py.File(str(mat_file), 'r') as f:
            # Find the labeled volume key
            volume_key = None
            for key in f.keys():
                if not key.startswith('#') and not key.startswith('_'):
                    volume_key = key
                    break

            if volume_key is None:
                raise ValueError(f"No data found in {mat_file}")

            volume = f[volume_key][:]
            logger.info(f"Loaded HDF5 format, volume shape: {volume.shape}, dtype: {volume.dtype}")
            return volume

    except OSError:
        # Try scipy.io for older MATLAB formats (v5/v6/v7)
        logger.info("HDF5 failed, trying scipy.io for older MATLAB format...")
        mat_data = sio.loadmat(str(mat_file))

        # Find the labeled volume key (skip MATLAB metadata keys)
        volume_key = None
        priority_keys = ['myelinated_axons', 'volume', 'labels', 'final_lbl']
        for pkey in priority_keys:
            if pkey in mat_data:
                volume_key = pkey
                break

        if volume_key is None:
            for key in mat_data.keys():
                if not key.startswith('__'):
                    volume_key = key
                    break

        if volume_key is None:
            raise ValueError(f"No data found in {mat_file}")

        volume = mat_data[volume_key]
        logger.info(f"Loaded scipy.io format (key: {volume_key}), volume shape: {volume.shape}, dtype: {volume.dtype}")
        return volume


def _init_worker(volume_data: np.ndarray, slice_axis: int):
    """Initialize worker process with volume data and slice axis."""
    global _volume_data, _slice_axis
    _volume_data = volume_data
    _slice_axis = slice_axis


def _process_single_slice(args: Tuple[int, float, np.ndarray, bool, float]) -> Tuple[int, np.ndarray, int]:
    """
    Process a single slice to extract radii histogram.

    Args:
        args: Tuple of (slice_index, voxel_size_um, bin_edges, use_minor_axis, max_ellipse_ratio)

    Returns:
        Tuple of (slice_index, histogram_counts, n_axons)
    """
    z, voxel_size_um, bin_edges, use_minor_axis, max_ellipse_ratio = args
    pixel_area_um2 = voxel_size_um * voxel_size_um

    # Slice from pre-loaded array
    if _slice_axis == 0:
        slice_2d = _volume_data[z, :, :]
    elif _slice_axis == 1:
        slice_2d = _volume_data[:, z, :]
    else:  # axis == 2
        slice_2d = _volume_data[:, :, z]

    # Use regionprops to efficiently extract areas of all regions
    regions = regionprops(slice_2d.astype(np.int32))

    if len(regions) == 0:
        return (z, np.zeros(len(bin_edges) - 1, dtype=np.int32), 0)

    # Compute radius for each region
    radii = []
    for region in regions:
        area_voxels = region.area

        # Filter by ellipse area ratio if specified
        if max_ellipse_ratio > 0:
            ellipse_area = np.pi * (region.axis_major_length / 2) * (region.axis_minor_length / 2)
            if ellipse_area > max_ellipse_ratio * area_voxels:
                continue

        if use_minor_axis:
            radius_um = (region.axis_minor_length / 2) * voxel_size_um
        else:
            area_um2 = area_voxels * pixel_area_um2
            radius_um = np.sqrt(area_um2 / np.pi)
        radii.append(radius_um)

    # Compute histogram
    radii_array = np.array(radii)
    counts, _ = np.histogram(radii_array, bins=bin_edges)

    return (z, counts.astype(np.int32), len(radii))


def compute_effective_radius_from_histogram(counts: np.ndarray, bin_centers: np.ndarray) -> float:
    """
    Compute effective MRI-visible radius from histogram using formula: (<r^6>/<r^2>)^(1/4)
    """
    if counts.sum() == 0:
        return 0.0

    total_count = counts.sum()
    r2_mean = np.sum(counts * bin_centers ** 2) / total_count
    r6_mean = np.sum(counts * bin_centers ** 6) / total_count

    if r2_mean == 0:
        return 0.0

    return (r6_mean / r2_mean) ** 0.25


def analyze_mat_file(mat_file: Path,
                     output_dir: Path,
                     voxel_size_um: Union[float, Tuple[float, float, float], None] = None,
                     slice_axis: int = None,
                     n_jobs: int = -1,
                     min_axon_fraction: float = 0.0,
                     use_minor_axis: bool = False,
                     max_ellipse_ratio: float = 0.0) -> Dict[str, float]:
    """
    Analyze effective radius from a .mat file with anisotropy handling.

    Args:
        mat_file: Path to .mat file with labeled axons
        output_dir: Directory for output plots
        voxel_size_um: Voxel size in micrometers (vz, vy, vx) or scalar.
                       If None, tries to load from metadata JSON file.
        slice_axis: Axis to slice along (0=Z, 1=Y, 2=X).
                    If None, tries to load dominant_axis from metadata JSON file.
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        min_axon_fraction: Minimum fraction of max axon count to include slice
        use_minor_axis: If True, use ellipse minor axis
        max_ellipse_ratio: Max ratio of ellipse area to voxel area (0 = no filter)

    Returns:
        Dict with analysis results
    """
    # Load metadata if available
    metadata = load_metadata(mat_file)

    # Use metadata values if not provided via arguments
    if slice_axis is None:
        if 'dominant_axis' in metadata:
            slice_axis = metadata['dominant_axis']
            logger.info(f"Using dominant_axis from metadata: {slice_axis} ({metadata.get('dominant_axis_name', '?')})")
        else:
            slice_axis = 0
            logger.warning("No slice_axis specified and no dominant_axis in metadata, defaulting to 0 (Z)")

    if voxel_size_um is None:
        if 'voxel_size_um' in metadata:
            voxel_size_um = tuple(metadata['voxel_size_um'])
            logger.info(f"Using voxel_size from metadata: {voxel_size_um}")
        else:
            voxel_size_um = 0.05
            logger.warning("No voxel_size specified and none in metadata, defaulting to 0.05 (isotropic)")

    # Parse voxel size
    voxel_size_tuple = parse_voxel_size(voxel_size_um)
    vz, vy, vx = voxel_size_tuple

    # Load volume
    volume = load_mat_volume(mat_file)

    # Resample to isotropic if needed
    volume, iso_voxel_size = resample_to_isotropic(volume, voxel_size_tuple)

    logger.info(f"Working with isotropic voxel size: {iso_voxel_size:.4f} μm")
    logger.info(f"Volume shape after resampling: {volume.shape}")

    # Get number of slices
    n_slices = volume.shape[slice_axis]
    axis_names = {0: 'Z', 1: 'Y', 2: 'X'}
    logger.info(f"Slicing along axis {slice_axis} ({axis_names[slice_axis]}), {n_slices} slices")

    # Histogram bin edges
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
    logger.info(f"Radius estimation: {'minor axis' if use_minor_axis else 'circular-equivalent'}")
    if max_ellipse_ratio > 0:
        logger.info(f"Filtering regions with ellipse/voxel area ratio > {max_ellipse_ratio}")

    # Prepare arguments
    args_list = [(z, iso_voxel_size, bin_edges, use_minor_axis, max_ellipse_ratio)
                 for z in range(n_slices)]

    # Process slices
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
        _init_worker(volume, slice_axis)
        results = [_process_single_slice(args) for args in tqdm(args_list, desc="Processing slices")]

    # Convert to dictionaries
    histogram_per_slice = {z: counts for z, counts, _ in results}
    n_axons_per_slice = {z: n_axons for z, _, n_axons in results}

    # Filter slices by axon count
    slice_indices = sorted(histogram_per_slice.keys())
    n_axons_array = np.array([n_axons_per_slice[z] for z in slice_indices])

    if min_axon_fraction > 0.0 and n_axons_array.max() > 0:
        max_idx = np.argmax(n_axons_array)
        max_count = n_axons_array[max_idx]
        threshold_count = min_axon_fraction * max_count

        # Find symmetric extent
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

    # Compute effective radius per slice
    r_eff_per_slice = []
    for z in slice_indices:
        counts = histogram_per_slice[z]
        r_eff = compute_effective_radius_from_histogram(counts, bin_centers)
        r_eff_per_slice.append(r_eff)
    r_eff_per_slice = np.array(r_eff_per_slice)

    # Compute global effective radius
    total_histogram = np.sum([histogram_per_slice[z] for z in slice_indices], axis=0)
    r_eff_global = compute_effective_radius_from_histogram(total_histogram, bin_centers)

    # Statistics
    total_measurements = int(total_histogram.sum())
    mean_radius = np.sum(bin_centers * total_histogram) / total_measurements if total_measurements > 0 else 0.0

    logger.info(f"\nResults:")
    logger.info(f"  Global effective radius: {r_eff_global:.3f} μm")
    logger.info(f"  Mean radius: {mean_radius:.3f} μm")
    logger.info(f"  Total measurements: {total_measurements}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_name = mat_file.stem.replace('_myelinated_axons', '')

    # Plot effective radius profile
    positions_um = np.array(slice_indices) * iso_voxel_size

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(positions_um, r_eff_per_slice, 'b-', linewidth=1.5, label='Per-slice effective radius')
    ax1.axhline(r_eff_global, color='r', linestyle='--', linewidth=2,
                label=f'Global effective radius = {r_eff_global:.3f} μm')
    ax1.set_ylabel('Effective Radius (μm)', fontsize=12)
    ax1.set_title(f'{sample_name}: MRI-Visible Effective Radius Profile\n' +
                  r'$r_{\rm eff} = \langle r^6 \rangle^{1/4} / \langle r^2 \rangle^{1/4}$',
                  fontsize=14, pad=15)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(positions_um, n_axons_array, 'g-', linewidth=1.5)
    ax2.set_xlabel('Position along tract (μm)', fontsize=12)
    ax2.set_ylabel('Number of axons', fontsize=12)
    ax2.set_title('Axon count per slice', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    profile_file = output_dir / f'{sample_name}_effective_radius_profile.png'
    plt.savefig(profile_file, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved profile to {profile_file}")

    # Plot histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    bin_width = bin_centers[1] - bin_centers[0]
    ax.bar(bin_centers, total_histogram, width=bin_width * 0.9, alpha=0.7,
           color='steelblue', edgecolor='black')
    ax.axvline(mean_radius, color='g', linestyle='-', linewidth=2.5,
               label=f'Mean = {mean_radius:.3f} μm')
    ax.axvline(r_eff_global, color='r', linestyle='--', linewidth=2.5,
               label=f'Effective = {r_eff_global:.3f} μm')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlabel('Axon Radius (μm)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{sample_name}: Radii Distribution', fontsize=14)
    ax.set_xlim(0, 3)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    hist_file = output_dir / f'{sample_name}_radii_histogram.png'
    plt.savefig(hist_file, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved histogram to {hist_file}")

    # Return results
    return {
        'sample_name': sample_name,
        'r_eff_global': float(r_eff_global),
        'mean_radius': float(mean_radius),
        'total_measurements': total_measurements,
        'n_slices_used': len(slice_indices),
        'voxel_size_original': list(voxel_size_tuple),
        'voxel_size_isotropic': float(iso_voxel_size),
        'slice_axis': slice_axis
    }


def batch_analyze(input_dir: Path,
                  output_dir: Path,
                  voxel_size_um: Union[float, Tuple[float, float, float], None] = None,
                  slice_axis: int = None,
                  n_jobs: int = -1,
                  min_axon_fraction: float = 0.0,
                  use_minor_axis: bool = False,
                  max_ellipse_ratio: float = 0.0) -> Dict[str, Dict]:
    """
    Batch analyze all .mat files in a directory.

    If voxel_size_um or slice_axis are None, each file will look for its
    own *_metadata.json file to get these values.
    """
    mat_files = sorted(input_dir.glob('*_myelinated_axons.mat'))

    if not mat_files:
        logger.error(f"No *_myelinated_axons.mat files found in {input_dir}")
        return {}

    logger.info(f"Found {len(mat_files)} .mat files to analyze")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for i, mat_file in enumerate(mat_files, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {i}/{len(mat_files)}: {mat_file.name}")
        logger.info(f"{'='*80}")

        try:
            # Pass None for voxel_size and slice_axis to let each file
            # load its own values from its metadata JSON file
            result = analyze_mat_file(
                mat_file,
                output_dir,
                voxel_size_um=voxel_size_um,  # None allows per-file metadata
                slice_axis=slice_axis,  # None allows per-file metadata
                n_jobs=n_jobs,
                min_axon_fraction=min_axon_fraction,
                use_minor_axis=use_minor_axis,
                max_ellipse_ratio=max_ellipse_ratio
            )
            all_results[result['sample_name']] = result

        except Exception as e:
            logger.error(f"Failed to process {mat_file.name}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary
    if all_results:
        summary_file = output_dir / 'effective_radius_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nSaved summary to {summary_file}")

        # Print summary table
        logger.info(f"\n{'='*80}")
        logger.info("SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"{'Sample':<25} {'r_eff (μm)':<12} {'Mean (μm)':<12} {'N':<10}")
        logger.info(f"{'-'*59}")
        for name, result in sorted(all_results.items()):
            logger.info(f"{name:<25} {result['r_eff_global']:<12.3f} {result['mean_radius']:<12.3f} {result['total_measurements']:<10}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze effective MRI-visible radius from .mat files with anisotropy handling'
    )
    parser.add_argument('input', type=Path,
                        help='Path to .mat file or directory containing *_myelinated_axons.mat files')
    parser.add_argument('output_dir', type=Path,
                        help='Output directory for plots and summary')
    parser.add_argument('--voxel-size', type=parse_voxel_size_arg, default=None,
                        help='Voxel size in micrometers: single value for isotropic (e.g., 0.05) '
                             'or vz,vy,vx for anisotropic matching array axes Z,Y,X '
                             '(e.g., 0.015,0.015,0.05). If not specified, reads from *_metadata.json')
    parser.add_argument('--slice-axis', type=int, default=None, choices=[0, 1, 2],
                        help='Axis to slice along: 0=Z, 1=Y, 2=X. '
                             'If not specified, reads dominant_axis from *_metadata.json')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (default: -1 = all CPUs)')
    parser.add_argument('--min-axon-fraction', type=float, default=0.0,
                        help='Minimum fraction of max axon count to include slice (default: 0.0)')
    parser.add_argument('--minor-axis', '--use-minor-axis', dest='use_minor_axis',
                        action='store_true',
                        help='Use ellipse minor axis instead of circular-equivalent radius')
    parser.add_argument('--max-ellipse-ratio', type=float, default=0.0,
                        help='Max ratio of ellipse area to voxel area (0 = no filter)')

    args = parser.parse_args()

    if args.input.is_dir():
        batch_analyze(
            args.input,
            args.output_dir,
            voxel_size_um=args.voxel_size,
            slice_axis=args.slice_axis,
            n_jobs=args.n_jobs,
            min_axon_fraction=args.min_axon_fraction,
            use_minor_axis=args.use_minor_axis,
            max_ellipse_ratio=args.max_ellipse_ratio
        )
    else:
        analyze_mat_file(
            args.input,
            args.output_dir,
            voxel_size_um=args.voxel_size,
            slice_axis=args.slice_axis,
            n_jobs=args.n_jobs,
            min_axon_fraction=args.min_axon_fraction,
            use_minor_axis=args.use_minor_axis,
            max_ellipse_ratio=args.max_ellipse_ratio
        )
