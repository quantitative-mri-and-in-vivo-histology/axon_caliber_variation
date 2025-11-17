#!/usr/bin/env python3
"""
Assess axon density along fiber bundle axis to identify valid measurement range.

This script analyzes an aligned HDF5 volume to:
1. Count axons per slice along the fiber axis (z-axis)
2. Identify sparse regions at bundle ends
3. Determine valid measurement range
4. Save metadata for downstream processing
"""

import json
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _process_single_slice_density(args: Tuple) -> Tuple[int, int]:
    """
    Process a single slice to count density. Helper for parallel processing.

    Args:
        args: (z, slice_2d)

    Returns:
        (nonzero_count, unique_axon_count)
    """
    z, slice_2d = args

    # Count axon pixels
    n_nonzero = np.count_nonzero(slice_2d)

    # Count unique axons (excluding background)
    unique_labels = len(np.unique(slice_2d)) - 1  # -1 for background
    n_unique = max(0, unique_labels)

    return (n_nonzero, n_unique)


def assess_density_profile(volume_file: Path,
                           voxel_size_um: float = 0.05,
                           n_jobs: int = -1) -> Dict:
    """
    Analyze axon density along fiber bundle axis.

    Args:
        volume_file: Path to aligned HDF5 volume
        voxel_size_um: Physical voxel size in micrometers
        n_jobs: Number of parallel jobs (-1 = use all CPUs)

    Returns:
        Dictionary with density profile and analysis results
    """
    logger.info(f"Analyzing density profile: {volume_file.name}")

    # Get metadata
    with h5py.File(volume_file, 'r') as f:
        volume_shape = f['labels'].shape
        population = f.attrs.get('population', 'Unknown')
        alignment_axis = f.attrs.get('alignment_axis', 'z')

    n_slices = volume_shape[0]
    logger.info(f"Volume shape: {volume_shape}")

    # Determine number of workers
    if n_jobs == -1:
        n_workers = cpu_count()
    elif n_jobs == 1:
        n_workers = 1
    else:
        n_workers = min(n_jobs, cpu_count())

    logger.info(f"Using {n_workers} parallel workers")

    # Process in batches to avoid loading entire volume
    batch_size = 200  # Process 200 slices at a time
    results = []

    with h5py.File(volume_file, 'r') as f:
        dataset = f['labels']

        for batch_start in tqdm(range(0, n_slices, batch_size), desc="Processing batches", unit="batch"):
            batch_end = min(batch_start + batch_size, n_slices)

            # Load entire batch with single I/O operation (much faster with chunked HDF5)
            batch_3d = dataset[batch_start:batch_end, :, :].copy()

            # Prepare data for parallel processing
            batch_data = [(batch_start + i, batch_3d[i, :, :])
                          for i in range(batch_end - batch_start)]

            # Process batch in parallel
            if n_workers > 1:
                with Pool(processes=n_workers) as pool:
                    batch_results = pool.map(_process_single_slice_density, batch_data)
            else:
                batch_results = [_process_single_slice_density(args) for args in batch_data]

            results.extend(batch_results)

    # Unpack results
    nonzero_counts = np.array([r[0] for r in results])
    unique_axon_counts = np.array([r[1] for r in results])

    # Compute statistics
    max_density = nonzero_counts.max()
    max_axon_count = unique_axon_counts.max()

    logger.info(f"Max density: {max_density} voxels")
    logger.info(f"Max axon count: {max_axon_count} axons")

    # Compute positions in physical units
    positions_um = np.arange(n_slices) * voxel_size_um

    return {
        'volume_file': str(volume_file),
        'population': population,
        'alignment_axis': alignment_axis,
        'volume_shape': volume_shape,
        'voxel_size_um': voxel_size_um,
        'n_slices': n_slices,
        'positions_um': positions_um.tolist(),
        'nonzero_counts': nonzero_counts.tolist(),
        'unique_axon_counts': unique_axon_counts.tolist(),
        'max_density': int(max_density),
        'max_axon_count': int(max_axon_count)
    }


def find_valid_range(density_data: Dict,
                     min_density_fraction: float = 0.1,
                     min_axon_fraction: float = 0.2) -> Tuple[int, int]:
    """
    Identify valid measurement range by excluding sparse ends.

    Args:
        density_data: Density profile dictionary from assess_density_profile
        min_density_fraction: Minimum density as fraction of maximum
        min_axon_fraction: Minimum axon count as fraction of maximum

    Returns:
        (start_slice, end_slice) tuple defining valid range
    """
    nonzero_counts = np.array(density_data['nonzero_counts'])
    unique_axon_counts = np.array(density_data['unique_axon_counts'])

    # Compute thresholds
    density_threshold = density_data['max_density'] * min_density_fraction
    axon_threshold = density_data['max_axon_count'] * min_axon_fraction

    # Find slices exceeding thresholds
    valid_by_density = nonzero_counts >= density_threshold
    valid_by_axons = unique_axon_counts >= axon_threshold

    # Combine criteria (both must be satisfied)
    valid_slices = valid_by_density & valid_by_axons

    if not valid_slices.any():
        logger.warning("No slices meet criteria! Using full range.")
        return 0, len(nonzero_counts) - 1

    # Find continuous valid range
    valid_indices = np.where(valid_slices)[0]
    start_slice = valid_indices[0]
    end_slice = valid_indices[-1]

    logger.info(f"Valid range: slice {start_slice} to {end_slice}")
    logger.info(f"  Position: {density_data['positions_um'][start_slice]:.1f} to "
                f"{density_data['positions_um'][end_slice]:.1f} μm")
    logger.info(f"  Length: {end_slice - start_slice + 1} slices "
                f"({(end_slice - start_slice + 1) * density_data['voxel_size_um']:.1f} μm)")

    return int(start_slice), int(end_slice)


def compute_fiber_direction(volume_file: Path,
                            sample_fraction: float = 0.1,
                            random_seed: int = 42) -> np.ndarray:
    """
    Estimate mean fiber direction from sample of axon voxels.

    Args:
        volume_file: Path to aligned HDF5 volume
        sample_fraction: Fraction of non-zero voxels to sample for PCA
        random_seed: Random seed for reproducible sampling

    Returns:
        Normalized direction vector [vx, vy, vz]
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    logger.info(f"Computing fiber direction from {volume_file.name}")

    with h5py.File(volume_file, 'r') as f:
        dataset = f['labels']
        volume_shape = dataset.shape

        # Collect sample of axon voxel coordinates
        sample_coords = []
        n_samples_target = int(np.prod(volume_shape) * sample_fraction * 0.01)  # Conservative

        logger.info(f"Sampling ~{n_samples_target} voxels for PCA...")

        # Sample slices evenly throughout volume
        sample_slices = np.linspace(0, volume_shape[0]-1, min(50, volume_shape[0]), dtype=int)

        # Read individual slices (they're sparse, so batch reading wouldn't help)
        for z in tqdm(sample_slices, desc="Sampling voxels"):
            slice_2d = dataset[z, :, :]
            nonzero = np.nonzero(slice_2d)

            # Add z coordinate
            z_coords = np.full(len(nonzero[0]), z)
            coords = np.column_stack([z_coords, nonzero[0], nonzero[1]])

            # Subsample if too many
            if len(coords) > n_samples_target // len(sample_slices):
                indices = np.random.choice(len(coords),
                                          size=n_samples_target // len(sample_slices),
                                          replace=False)
                coords = coords[indices]

            sample_coords.append(coords)

        sample_coords = np.vstack(sample_coords)

    logger.info(f"Collected {len(sample_coords)} sample points")

    # Perform PCA
    mean_pos = sample_coords.mean(axis=0)
    centered = sample_coords - mean_pos

    # Compute covariance matrix
    cov_matrix = np.cov(centered.T)

    # Get principal direction (first eigenvector)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    principal_axis = np.real(eigenvectors[:, np.argmax(eigenvalues)])

    # Normalize
    principal_axis = principal_axis / np.linalg.norm(principal_axis)

    logger.info(f"Fiber direction: [{principal_axis[0]:.3f}, {principal_axis[1]:.3f}, {principal_axis[2]:.3f}]")

    # Check if aligned with z-axis (should be for aligned volumes)
    z_alignment = abs(principal_axis[0])
    logger.info(f"Z-axis alignment: {z_alignment:.3f}")

    if z_alignment < 0.9:
        logger.warning(f"Fibers not well-aligned with z-axis! Consider oblique slicing.")

    return principal_axis


def plot_density_profile(density_data: Dict,
                         valid_range: Tuple[int, int],
                         output_file: Path):
    """
    Create visualization of density profile with valid range highlighted.

    Args:
        density_data: Density profile dictionary
        valid_range: (start_slice, end_slice) tuple
        output_file: Path to save figure
    """
    positions = np.array(density_data['positions_um'])
    nonzero_counts = np.array(density_data['nonzero_counts'])
    unique_axon_counts = np.array(density_data['unique_axon_counts'])

    start_slice, end_slice = valid_range
    valid_positions = positions[start_slice:end_slice+1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel 1: Voxel density
    axes[0].plot(positions, nonzero_counts, 'b-', linewidth=1, alpha=0.7)
    axes[0].axvspan(valid_positions[0], valid_positions[-1],
                    color='green', alpha=0.1, label='Valid range')
    axes[0].axhline(density_data['max_density'] * 0.1, color='r',
                   linestyle='--', alpha=0.5, label='10% threshold')
    axes[0].set_ylabel('Non-zero voxels', fontsize=12)
    axes[0].set_title(f"{density_data['population']}: Density Profile Along Fiber Bundle",
                     fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Unique axon count
    axes[1].plot(positions, unique_axon_counts, 'g-', linewidth=1, alpha=0.7)
    axes[1].axvspan(valid_positions[0], valid_positions[-1],
                    color='green', alpha=0.1, label='Valid range')
    axes[1].axhline(density_data['max_axon_count'] * 0.2, color='r',
                   linestyle='--', alpha=0.5, label='20% threshold')
    axes[1].set_xlabel('Position along fiber (μm)', fontsize=12)
    axes[1].set_ylabel('Number of axons', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved density profile plot: {output_file}")


def analyze_volume(volume_file: Path,
                  output_dir: Path,
                  voxel_size_um: float = 0.05,
                  min_density_fraction: float = 0.1,
                  min_axon_fraction: float = 0.2,
                  compute_direction: bool = True,
                  n_jobs: int = -1) -> Dict:
    """
    Complete density analysis workflow for one volume.

    Args:
        volume_file: Path to aligned HDF5 volume
        output_dir: Directory for output files
        voxel_size_um: Physical voxel size
        min_density_fraction: Threshold for valid range
        min_axon_fraction: Threshold for valid range
        compute_direction: Whether to compute fiber direction via PCA
        n_jobs: Number of parallel jobs (-1 = use all CPUs)

    Returns:
        Complete metadata dictionary
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing: {volume_file.name}")
    logger.info(f"{'='*80}\n")

    # Step 1: Assess density profile
    density_data = assess_density_profile(volume_file, voxel_size_um, n_jobs=n_jobs)

    # Step 2: Find valid range
    valid_range = find_valid_range(density_data, min_density_fraction, min_axon_fraction)
    density_data['valid_range'] = {
        'start_slice': valid_range[0],
        'end_slice': valid_range[1],
        'start_position_um': density_data['positions_um'][valid_range[0]],
        'end_position_um': density_data['positions_um'][valid_range[1]],
        'length_slices': valid_range[1] - valid_range[0] + 1,
        'length_um': (valid_range[1] - valid_range[0] + 1) * voxel_size_um
    }

    # Step 3: Compute fiber direction
    if compute_direction:
        fiber_direction = compute_fiber_direction(volume_file)
        density_data['fiber_direction'] = fiber_direction.tolist()

    # Step 4: Save metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    json_file = output_dir / f"{volume_file.stem}_density.json"

    with open(json_file, 'w') as f:
        json.dump(density_data, f, indent=2)

    logger.info(f"Saved metadata: {json_file}")

    # Step 5: Create visualization
    plot_file = output_dir / f"{volume_file.stem}_density_profile.png"
    plot_density_profile(density_data, valid_range, plot_file)

    logger.info(f"\n{'='*80}")
    logger.info(f"Completed: {volume_file.name}")
    logger.info(f"{'='*80}\n")

    return density_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Assess axon density profile and identify valid measurement range'
    )
    parser.add_argument('volume_file', type=Path,
                       help='Path to aligned HDF5 volume (e.g., cc_aligned.h5)')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory (default: same as volume file)')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                       help='Voxel size in micrometers (default: 0.05)')
    parser.add_argument('--min-density-fraction', type=float, default=0.1,
                       help='Minimum density as fraction of max (default: 0.1)')
    parser.add_argument('--min-axon-fraction', type=float, default=0.2,
                       help='Minimum axon count as fraction of max (default: 0.2)')
    parser.add_argument('--no-direction', action='store_true',
                       help='Skip fiber direction computation (faster)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (default: -1 = use all CPUs, 1 = serial)')

    args = parser.parse_args()

    # Default output directory
    if args.output_dir is None:
        args.output_dir = args.volume_file.parent

    # Run analysis
    metadata = analyze_volume(
        args.volume_file,
        args.output_dir,
        voxel_size_um=args.voxel_size,
        min_density_fraction=args.min_density_fraction,
        min_axon_fraction=args.min_axon_fraction,
        compute_direction=not args.no_direction,
        n_jobs=args.n_jobs
    )
