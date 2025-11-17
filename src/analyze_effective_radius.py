#!/usr/bin/env python3
"""
Analyze effective MRI-visible radius along aligned axon volumes.

Computes the effective radius per slice using the formula:
    r_eff = (<r^6>/<r^2>)^(1/4)

This metric is relevant for diffusion MRI sensitivity to axon caliber.
"""

import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _process_single_slice(args: Tuple[int, np.ndarray, float, np.ndarray]) -> Tuple[int, np.ndarray, int]:
    """
    Process a single slice to extract radii histogram. Helper function for parallel processing.

    Args:
        args: Tuple of (slice_index, slice_2d, voxel_size_um, bin_edges)

    Returns:
        Tuple of (slice_index, histogram_counts, n_axons)
    """
    z, slice_2d, voxel_size_um, bin_edges = args

    # Use regionprops to efficiently extract areas of all regions
    regions = regionprops(slice_2d)

    if len(regions) == 0:
        return (z, np.zeros(len(bin_edges) - 1, dtype=np.int32), 0)

    # Compute circular-equivalent radius for each region
    radii = []
    for region in regions:
        area_voxels = region.area
        # Convert to physical area and compute radius
        area_um2 = area_voxels * (voxel_size_um ** 2)
        radius_um = np.sqrt(area_um2 / np.pi)
        radii.append(radius_um)

    # Compute histogram
    radii_array = np.array(radii)
    counts, _ = np.histogram(radii_array, bins=bin_edges)

    return (z, counts.astype(np.int32), len(radii))


def extract_radii_per_slice(volume: np.ndarray,
                            voxel_size_um: float = 0.2,
                            n_jobs: int = -1,
                            bin_edges: np.ndarray = None) -> Tuple[Dict[int, np.ndarray], Dict[int, int], np.ndarray, np.ndarray]:
    """
    Extract circular-equivalent radii histograms for all axons in each z-slice.

    Args:
        volume: 3D labeled volume (z, y, x) where axons are aligned with z
        voxel_size_um: Voxel size in micrometers
        n_jobs: Number of parallel jobs (-1 = use all CPUs, 1 = no parallelization)
        bin_edges: Histogram bin edges in micrometers (default: 0 to 20 with 0.02 step)

    Returns:
        Tuple of (histogram_per_slice, n_axons_per_slice, bin_edges, bin_centers)
    """
    logger.info(f"Extracting radii per slice from volume shape {volume.shape}")

    n_slices = volume.shape[0]

    # Default bin edges: 0 to 20 μm with 0.02 μm step
    if bin_edges is None:
        bin_edges = np.arange(0, 20.02, 0.02)

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    logger.info(f"Using {len(bin_edges)-1} histogram bins from {bin_edges[0]:.2f} to {bin_edges[-1]:.2f} μm")

    # Determine number of workers
    if n_jobs == -1:
        n_workers = cpu_count()
    elif n_jobs == 1:
        n_workers = 1
    else:
        n_workers = min(n_jobs, cpu_count())

    logger.info(f"Using {n_workers} parallel workers")

    # Prepare arguments for parallel processing
    args_list = [(z, volume[z, :, :], voxel_size_um, bin_edges) for z in range(n_slices)]

    # Process in parallel
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_single_slice, args_list),
                total=n_slices,
                desc="Processing slices",
                unit="slice"
            ))
    else:
        # Serial processing (useful for debugging)
        results = [_process_single_slice(args) for args in tqdm(
            args_list,
            desc="Processing slices",
            unit="slice"
        )]

    # Convert list of tuples to dictionaries
    histogram_per_slice = {z: counts for z, counts, _ in results}
    n_axons_per_slice = {z: n_axons for z, _, n_axons in results}

    return histogram_per_slice, n_axons_per_slice, bin_edges, bin_centers


def compute_effective_radius_from_histogram(counts: np.ndarray, bin_centers: np.ndarray) -> float:
    """
    Compute effective MRI-visible radius from histogram using formula: (<r^6>/<r^2>)^(1/4)

    Args:
        counts: Histogram counts
        bin_centers: Bin center values in micrometers

    Returns:
        Effective radius in micrometers
    """
    if counts.sum() == 0:
        return 0.0

    # Weight by counts
    total_count = counts.sum()

    # Compute weighted moments
    r2_mean = np.sum(counts * bin_centers ** 2) / total_count
    r6_mean = np.sum(counts * bin_centers ** 6) / total_count

    if r2_mean == 0:
        return 0.0

    r_eff = (r6_mean / r2_mean) ** 0.25

    return r_eff


def compute_effective_radius(radii: np.ndarray) -> float:
    """
    Compute effective MRI-visible radius using formula: (<r^6>/<r^2>)^(1/4)

    Args:
        radii: Array of radii in micrometers

    Returns:
        Effective radius in micrometers
    """
    if len(radii) == 0:
        return 0.0

    r2_mean = np.mean(radii ** 2)
    r6_mean = np.mean(radii ** 6)

    if r2_mean == 0:
        return 0.0

    r_eff = (r6_mean / r2_mean) ** 0.25

    return r_eff


def plot_effective_radius_profile(histogram_per_slice: Dict[int, np.ndarray],
                                  n_axons_per_slice: Dict[int, int],
                                  bin_centers: np.ndarray,
                                  output_file: Path,
                                  population_name: str = "Axons",
                                  slice_thickness_um: float = 0.2):
    """
    Plot effective radius as a function of position along the tract.

    Args:
        histogram_per_slice: Dict mapping slice_idx -> histogram counts
        n_axons_per_slice: Dict mapping slice_idx -> number of axons
        bin_centers: Bin center values in micrometers
        output_file: Path to save figure
        population_name: Name of the population (CC or CG)
        slice_thickness_um: Thickness of each slice in micrometers
    """
    logger.info(f"Plotting effective radius profile for {population_name}")

    # Compute effective radius per slice
    slice_indices = sorted(histogram_per_slice.keys())
    r_eff_per_slice = []

    for z in slice_indices:
        counts = histogram_per_slice[z]
        r_eff = compute_effective_radius_from_histogram(counts, bin_centers)
        r_eff_per_slice.append(r_eff)

    r_eff_per_slice = np.array(r_eff_per_slice)
    n_axons_array = np.array([n_axons_per_slice[z] for z in slice_indices])

    # Compute global effective radius (pooled over all slices)
    total_histogram = np.sum([histogram_per_slice[z] for z in slice_indices], axis=0)
    r_eff_global = compute_effective_radius_from_histogram(total_histogram, bin_centers)

    total_axons = n_axons_array.sum()
    logger.info(f"Global effective radius: {r_eff_global:.3f} μm")
    logger.info(f"Mean slice effective radius: {np.mean(r_eff_per_slice[r_eff_per_slice > 0]):.3f} μm")
    logger.info(f"Total radii measurements: {total_axons}")

    # Convert slice indices to position in micrometers
    positions_um = np.array(slice_indices) * slice_thickness_um

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Effective radius per slice
    ax1.plot(positions_um, r_eff_per_slice, 'b-', linewidth=1.5, label='Per-slice effective radius')
    ax1.axhline(r_eff_global, color='r', linestyle='--', linewidth=2,
               label=f'Global effective radius = {r_eff_global:.3f} μm')

    ax1.set_ylabel('Effective Radius (μm)', fontsize=12)
    ax1.set_title(f'{population_name}: MRI-Visible Effective Radius Profile\n' +
                 r'$r_{\rm eff} = \langle r^6 \rangle^{1/4} / \langle r^2 \rangle^{1/4}$',
                 fontsize=14, pad=15)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Number of axons per slice
    ax2.plot(positions_um, n_axons_array, 'g-', linewidth=1.5)
    ax2.set_xlabel('Position along tract (μm)', fontsize=12)
    ax2.set_ylabel('Number of axons', fontsize=12)
    ax2.set_title('Axon count per slice', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved effective radius profile to {output_file}")

    return r_eff_global, r_eff_per_slice


def analyze_population(volume_file: Path,
                       output_dir: Path,
                       voxel_size_um: float = 0.2,
                       n_jobs: int = -1) -> Tuple[float, np.ndarray]:
    """
    Analyze effective radius for a population volume.

    Args:
        volume_file: Path to aligned HDF5 volume file
        output_dir: Directory for output plots
        voxel_size_um: Voxel size in micrometers (0.05 * downsample_factor)
        n_jobs: Number of parallel jobs (-1 = use all CPUs)

    Returns:
        Tuple of (global_r_eff, r_eff_per_slice)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing: {volume_file.name}")
    logger.info(f"{'='*80}\n")

    # Load volume
    with h5py.File(volume_file, 'r') as f:
        volume = f['labels'][:]
        population_name = f.attrs.get('population', 'Unknown')

    logger.info(f"Population: {population_name}")
    logger.info(f"Volume shape: {volume.shape}")

    # Extract radii histograms per slice
    histogram_per_slice, n_axons_per_slice, bin_edges, bin_centers = extract_radii_per_slice(
        volume, voxel_size_um, n_jobs=n_jobs
    )

    # Plot effective radius profile
    output_file = output_dir / f'{population_name}_effective_radius_profile.png'
    r_eff_global, r_eff_per_slice = plot_effective_radius_profile(
        histogram_per_slice,
        n_axons_per_slice,
        bin_centers,
        output_file,
        population_name,
        slice_thickness_um=voxel_size_um
    )

    logger.info(f"\n{'='*80}")
    logger.info(f"Completed: {volume_file.name}")
    logger.info(f"{'='*80}\n")

    return r_eff_global, r_eff_per_slice


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze effective MRI-visible radius from aligned axon volumes'
    )
    parser.add_argument('volume_file', type=Path,
                       help='Path to aligned HDF5 volume file (e.g., cc_aligned.h5)')
    parser.add_argument('output_dir', type=Path,
                       help='Output directory for plots')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                       help='Voxel size in micrometers (default: 0.05 for full resolution)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (default: -1 = use all CPUs, 1 = serial)')

    args = parser.parse_args()

    analyze_population(
        args.volume_file,
        args.output_dir,
        voxel_size_um=args.voxel_size,
        n_jobs=args.n_jobs
    )
