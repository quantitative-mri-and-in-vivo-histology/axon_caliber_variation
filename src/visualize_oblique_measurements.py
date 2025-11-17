#!/usr/bin/env python3
"""
Visualization and quality control for oblique radius measurements.

Creates comprehensive plots to verify:
1. Density profile with valid range
2. Radius distribution evolution (heatmap)
3. Effective radius profile
4. Sample extracted slices
"""

import logging
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_measurement_results(results_file: Path) -> dict:
    """Load oblique measurement results from HDF5 file."""
    logger.info(f"Loading results: {results_file}")

    with h5py.File(results_file, 'r') as f:
        # Get population name (first group)
        population = list(f.keys())[0]
        grp = f[population]

        data = {
            'population': population,
            'positions_um': grp['positions_um'][:],
            'n_axons': grp['n_axons'][:],
            'histograms': grp['histograms'][:],
            'bin_edges': grp['bin_edges'][:],
            'bin_centers': grp['bin_centers'][:],
            'r_eff_per_slice': grp['r_eff_per_slice'][:],
            'r_eff_global': grp.attrs['r_eff_global'],
            'voxel_size_um': grp.attrs['voxel_size_um'],
            'interval_um': grp.attrs['interval_um'],
            'fiber_direction': grp.attrs['fiber_direction']
        }

        # Load extracted slices if available
        if 'extracted_slices' in grp:
            data['extracted_slices'] = grp['extracted_slices'][:]

    logger.info(f"Loaded {len(data['positions_um'])} measurement intervals")
    logger.info(f"Global r_eff: {data['r_eff_global']:.3f} μm")

    return data


def plot_comprehensive_summary(data: dict, output_file: Path):
    """
    Create comprehensive 4-panel summary figure.

    Panels:
    1. Axon count vs position
    2. Radius distribution heatmap
    3. Effective radius profile
    4. Summary statistics
    """
    logger.info("Creating comprehensive summary plot...")

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    population = data['population']
    positions = data['positions_um']
    n_axons = data['n_axons']
    histograms = data['histograms']
    bin_edges = data['bin_edges']
    bin_centers = data['bin_centers']
    r_eff_per_slice = data['r_eff_per_slice']
    r_eff_global = data['r_eff_global']

    # Panel 1: Axon count
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(positions, n_axons, 'g-', linewidth=2)
    ax1.fill_between(positions, n_axons, alpha=0.3, color='green')
    ax1.set_ylabel('Number of axons', fontsize=12, fontweight='bold')
    ax1.set_title(f'{population}: Axon Count Along Fiber Bundle', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(positions[0], positions[-1])

    # Panel 2: Radius distribution heatmap
    ax2 = fig.add_subplot(gs[0, 1])

    # Normalize histograms by axon count for better visualization
    histograms_normalized = histograms.copy().astype(float)
    for i in range(len(histograms_normalized)):
        if n_axons[i] > 0:
            histograms_normalized[i] /= n_axons[i]

    im = ax2.imshow(
        histograms_normalized.T,
        aspect='auto',
        origin='lower',
        extent=[positions[0], positions[-1], bin_edges[0], bin_edges[-1]],
        cmap='viridis',
        interpolation='nearest'
    )
    ax2.set_ylabel('Radius (μm)', fontsize=12, fontweight='bold')
    ax2.set_title('Radius Distribution Evolution (Normalized)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax2, label='Fraction of axons')
    cbar.ax.tick_params(labelsize=10)

    # Panel 3: Effective radius profile
    ax3 = fig.add_subplot(gs[1, :])

    # Filter out NaN values for plotting
    valid_mask = ~np.isnan(r_eff_per_slice)
    valid_positions = positions[valid_mask]
    valid_r_eff = r_eff_per_slice[valid_mask]

    if len(valid_positions) > 0:
        ax3.plot(valid_positions, valid_r_eff, 'b-', linewidth=2, label='Per-slice r_eff')
        ax3.scatter(valid_positions, valid_r_eff, c='blue', s=30, alpha=0.6, zorder=3)

    if not np.isnan(r_eff_global):
        ax3.axhline(r_eff_global, color='r', linestyle='--', linewidth=2.5,
                   label=f'Global r_eff = {r_eff_global:.3f} μm')
    else:
        ax3.text(0.5, 0.5, 'No valid data', transform=ax3.transAxes,
                ha='center', va='center', fontsize=14, color='red')

    ax3.set_xlabel('Position along fiber (μm)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Effective radius (μm)', fontsize=12, fontweight='bold')
    ax3.set_title(r'MRI-Visible Effective Radius: $r_{eff} = (\langle r^6 \rangle / \langle r^2 \rangle)^{1/4}$',
                 fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(positions[0], positions[-1])

    # Panel 4: Summary statistics table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    # Compute summary statistics
    total_measurements = histograms.sum()
    mean_radius = np.sum(histograms * bin_centers[None, :]) / total_measurements if total_measurements > 0 else 0

    # Median from cumulative histogram
    cumsum = np.cumsum(histograms.sum(axis=0))
    median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
    median_radius = bin_centers[median_idx] if median_idx < len(bin_centers) else 0

    # Range
    nonzero_bins = np.where(histograms.sum(axis=0) > 0)[0]
    if len(nonzero_bins) > 0:
        r_min = bin_centers[nonzero_bins[0]]
        r_max = bin_centers[nonzero_bins[-1]]
    else:
        r_min = r_max = 0

    # Mean effective radius across slices (filter out NaN)
    valid_r_eff_for_stats = r_eff_per_slice[~np.isnan(r_eff_per_slice)]
    mean_r_eff_slices = valid_r_eff_for_stats.mean() if len(valid_r_eff_for_stats) > 0 else np.nan
    std_r_eff_slices = valid_r_eff_for_stats.std() if len(valid_r_eff_for_stats) > 0 else np.nan

    stats_text = f"""
    SUMMARY STATISTICS

    Population: {population}
    Fiber Direction: [{data['fiber_direction'][0]:.3f}, {data['fiber_direction'][1]:.3f}, {data['fiber_direction'][2]:.3f}]

    Measurement Range:
        Start: {positions[0]:.1f} μm
        End: {positions[-1]:.1f} μm
        Length: {positions[-1] - positions[0]:.1f} μm
        Intervals: {len(positions)} (spacing: {data['interval_um']:.1f} μm)

    Axon Count:
        Total measurements: {int(total_measurements)}
        Mean per slice: {n_axons.mean():.1f} ± {n_axons.std():.1f}
        Range: {n_axons.min()} - {n_axons.max()}

    Radius Distribution:
        Mean radius: {mean_radius:.3f} μm
        Median radius: {median_radius:.3f} μm
        Range: {r_min:.3f} - {r_max:.3f} μm

    Effective Radius (MRI-visible):
        Global r_eff: {r_eff_global:.3f if not np.isnan(r_eff_global) else 'N/A'} μm
        Mean slice r_eff: {mean_r_eff_slices:.3f if not np.isnan(mean_r_eff_slices) else 'N/A'} ± {std_r_eff_slices:.3f if not np.isnan(std_r_eff_slices) else 'N/A'} μm
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved summary plot: {output_file}")


def plot_extracted_slices(data: dict, output_file: Path, n_samples: int = 9):
    """
    Visualize sample extracted oblique slices.

    Shows a grid of extracted 2D slices to verify oblique slicing quality.
    """
    if 'extracted_slices' not in data:
        logger.warning("No extracted slices available. Run measurement with --save-slices flag.")
        return

    logger.info("Creating extracted slices visualization...")

    slices = data['extracted_slices']
    positions = data['positions_um']
    n_slices = len(slices)

    # Select evenly spaced slices
    if n_slices > n_samples:
        indices = np.linspace(0, n_slices - 1, n_samples, dtype=int)
    else:
        indices = np.arange(n_slices)
        n_samples = n_slices

    # Determine grid layout
    n_cols = min(3, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]
        slice_2d = slices[idx]

        # Display slice
        im = ax.imshow(slice_2d > 0, cmap='gray', interpolation='nearest')
        ax.set_title(f'Position: {positions[idx]:.1f} μm\n{data["n_axons"][idx]} axons',
                    fontsize=11)
        ax.axis('off')

    # Hide unused subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'{data["population"]}: Sample Extracted Oblique Slices',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved slices visualization: {output_file}")


def create_all_visualizations(results_file: Path, output_dir: Path):
    """
    Generate all QC visualizations for oblique measurement results.

    Args:
        results_file: HDF5 file with measurement results
        output_dir: Directory for output plots
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Generating visualizations for: {results_file.name}")
    logger.info(f"{'='*80}\n")

    # Load data
    data = load_measurement_results(results_file)

    # Create comprehensive summary
    summary_file = output_dir / f"{results_file.stem}_summary.png"
    plot_comprehensive_summary(data, summary_file)

    # Create extracted slices visualization if available
    if 'extracted_slices' in data:
        slices_file = output_dir / f"{results_file.stem}_slices.png"
        plot_extracted_slices(data, slices_file)

    logger.info(f"\n{'='*80}")
    logger.info(f"Completed visualizations for: {results_file.name}")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Create QC visualizations for oblique radius measurements'
    )
    parser.add_argument('results_file', type=Path,
                       help='Path to measurement results HDF5 file')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory for plots (default: same as results file)')
    parser.add_argument('--n-slice-samples', type=int, default=9,
                       help='Number of sample slices to display (default: 9)')

    args = parser.parse_args()

    # Default output directory
    if args.output_dir is None:
        args.output_dir = args.results_file.parent

    # Generate visualizations
    create_all_visualizations(args.results_file, args.output_dir)
