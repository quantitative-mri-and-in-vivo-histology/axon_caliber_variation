#!/usr/bin/env python3
"""
Compare 2D slice-based metrics vs 3D axon-based metrics.

This script creates a two-subplot comparison plot showing:
- Subplot 1: Arithmetic mean radius (2D vs 3D)
- Subplot 2: Effective radius (2D vs 3D)

For 2D values: mean ± std across slices (with error bars)
For 3D values: single pooled value (reference)

X-axis: 3D reference value
Y-axis: 2D value with error bars
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_r_eff(radii: np.ndarray) -> float:
    """
    Compute effective MRI-visible radius: r_eff = (⟨r⁶⟩/⟨r²⟩)^(1/4)

    Args:
        radii: Array of radius values in micrometers

    Returns:
        Effective radius in micrometers
    """
    if len(radii) == 0:
        return np.nan

    r2 = radii ** 2
    r6 = radii ** 6

    r2_mean = np.mean(r2)
    r6_mean = np.mean(r6)

    if r2_mean == 0:
        return np.nan

    return (r6_mean / r2_mean) ** 0.25


def load_2d_metrics(npz_file: Path) -> Dict[str, float]:
    """
    Load 2D slice-based metrics from NPZ file.

    Computes:
    - Arithmetic mean radius (from total histogram)
    - Arithmetic mean radius std (from per-slice means)
    - Effective radius (mean across slices)
    - Effective radius std (across slices)

    Args:
        npz_file: Path to 2D slice profiles NPZ file

    Returns:
        Dictionary with keys: mean_radius, mean_radius_std, r_eff, r_eff_std
    """
    data = np.load(npz_file)

    # Extract data
    bin_centers = data['bin_centers']
    histograms = data['histograms']  # Shape: (n_slices, n_bins)
    total_histogram = data['total_histogram']
    r_eff_per_slice = data['r_eff_per_slice']

    # Compute arithmetic mean radius from total histogram
    total_counts = total_histogram.sum()
    if total_counts > 0:
        mean_radius_global = np.sum(bin_centers * total_histogram) / total_counts
    else:
        mean_radius_global = np.nan

    # Compute per-slice arithmetic mean for std calculation
    mean_radius_per_slice = []
    for hist_slice in histograms:
        counts = hist_slice.sum()
        if counts > 0:
            mean_r = np.sum(bin_centers * hist_slice) / counts
            mean_radius_per_slice.append(mean_r)

    if len(mean_radius_per_slice) > 0:
        mean_radius_std = np.std(mean_radius_per_slice)
    else:
        mean_radius_std = np.nan

    # Use global pooled effective radius (computed from total histogram)
    r_eff_global = float(data['r_eff_global'])

    # Compute std of per-slice effective radii for error bars
    valid_r_eff = r_eff_per_slice[r_eff_per_slice > 0]
    if len(valid_r_eff) > 0:
        r_eff_std = np.std(valid_r_eff)
    else:
        r_eff_std = np.nan

    logger.info(f"2D metrics from {npz_file.name}:")
    logger.info(f"  Mean radius: {mean_radius_global:.3f} ± {mean_radius_std:.3f} μm")
    logger.info(f"  Effective radius: {r_eff_global:.3f} ± {r_eff_std:.3f} μm")

    return {
        'mean_radius': mean_radius_global,
        'mean_radius_std': mean_radius_std,
        'r_eff': r_eff_global,
        'r_eff_std': r_eff_std
    }


def load_3d_metrics(npz_file: Path) -> Dict[str, float]:
    """
    Load 3D axon-based metrics from NPZ file.

    Computes:
    - Arithmetic mean radius (from all_radii_um)
    - Effective radius (from all_radii_um)

    Args:
        npz_file: Path to 3D axon profiles NPZ file

    Returns:
        Dictionary with keys: mean_radius, r_eff
    """
    data = np.load(npz_file, allow_pickle=True)

    # Extract all radius measurements
    all_radii = data['all_radii_um']

    # Compute arithmetic mean
    mean_radius = np.mean(all_radii)

    # Compute effective radius
    r_eff = compute_r_eff(all_radii)

    logger.info(f"3D metrics from {npz_file.name}:")
    logger.info(f"  Mean radius: {mean_radius:.3f} μm")
    logger.info(f"  Effective radius: {r_eff:.3f} μm")
    logger.info(f"  Total samples: {len(all_radii)}")

    return {
        'mean_radius': mean_radius,
        'r_eff': r_eff
    }


def plot_2d_vs_3d_comparison(
    metrics_2d: Dict[str, float],
    metrics_3d: Dict[str, float],
    output_file: Path,
    sample_name: str = ""
) -> None:
    """
    Create two-subplot comparison plot.

    Subplot 1: Arithmetic mean radius
    Subplot 2: Effective radius

    Both show:
    - X-axis: 3D reference value
    - Y-axis: 2D value with error bars
    - Identity line for reference

    Args:
        metrics_2d: 2D metrics dictionary
        metrics_3d: 3D metrics dictionary
        output_file: Output PNG file path
        sample_name: Sample name for title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # === Subplot 1: Arithmetic Mean Radius ===
    x_mean = metrics_3d['mean_radius']
    y_mean = metrics_2d['mean_radius']
    yerr_mean = metrics_2d['mean_radius_std']

    ax1.errorbar(
        x_mean, y_mean, yerr=yerr_mean,
        fmt='o', color='#d62728', markersize=10,
        capsize=5, capthick=1.5, elinewidth=1.5,
        markeredgecolor='black', markeredgewidth=0.5,
        alpha=0.8, label='2D vs 3D'
    )

    # Identity line
    all_vals_mean = [x_mean, y_mean]
    min_val_mean = min(all_vals_mean) * 0.92
    max_val_mean = max(all_vals_mean) * 1.08

    ax1.plot([min_val_mean, max_val_mean], [min_val_mean, max_val_mean],
             'k--', alpha=0.5, linewidth=1, label='Identity')

    ax1.set_xlim(min_val_mean, max_val_mean)
    ax1.set_ylim(min_val_mean, max_val_mean)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('3D Mean Radius (μm)', fontsize=11)
    ax1.set_ylabel('2D Mean Radius ± std (μm)', fontsize=11)
    ax1.set_title('Arithmetic Mean Radius', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)

    # === Subplot 2: Effective Radius ===
    x_eff = metrics_3d['r_eff']
    y_eff = metrics_2d['r_eff']
    yerr_eff = metrics_2d['r_eff_std']

    ax2.errorbar(
        x_eff, y_eff, yerr=yerr_eff,
        fmt='o', color='#2ca02c', markersize=10,
        capsize=5, capthick=1.5, elinewidth=1.5,
        markeredgecolor='black', markeredgewidth=0.5,
        alpha=0.8, label='2D vs 3D'
    )

    # Identity line
    all_vals_eff = [x_eff, y_eff]
    min_val_eff = min(all_vals_eff) * 0.92
    max_val_eff = max(all_vals_eff) * 1.08

    ax2.plot([min_val_eff, max_val_eff], [min_val_eff, max_val_eff],
             'k--', alpha=0.5, linewidth=1, label='Identity')

    ax2.set_xlim(min_val_eff, max_val_eff)
    ax2.set_ylim(min_val_eff, max_val_eff)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('3D Effective Radius (μm)', fontsize=11)
    ax2.set_ylabel('2D Effective Radius ± std (μm)', fontsize=11)
    ax2.set_title('Effective Radius (r_eff)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)

    # Overall title
    if sample_name:
        fig.suptitle(f'2D vs 3D Metrics Comparison: {sample_name}',
                     fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plot to {output_file}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Compare 2D slice-based metrics vs 3D axon-based metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_2d_versus_3d_metrics.py \\
      data/processed/HM_25_ipsi_myelinated_axons_slice_profiles.npz \\
      data/processed/HM_25_ipsi_myelinated_axons_axon_profiles.npz \\
      fig/HM_25_ipsi_comparison.png
        """
    )

    parser.add_argument('npz_2d', type=Path,
                        help='2D slice profiles NPZ file (*_slice_profiles.npz)')
    parser.add_argument('npz_3d', type=Path,
                        help='3D axon profiles NPZ file (*_axon_profiles.npz)')
    parser.add_argument('output', type=Path,
                        help='Output plot file path (PNG)')
    parser.add_argument('--sample-name', type=str, default="",
                        help='Sample name for plot title (default: auto-detect from filename)')

    args = parser.parse_args()

    # Validate input files exist
    if not args.npz_2d.exists():
        parser.error(f"2D NPZ file not found: {args.npz_2d}")
    if not args.npz_3d.exists():
        parser.error(f"3D NPZ file not found: {args.npz_3d}")

    # Auto-detect sample name if not provided
    sample_name = args.sample_name
    if not sample_name:
        # Extract from filename (remove suffix)
        sample_name = args.npz_2d.stem.replace('_slice_profiles', '').replace('_myelinated_axons', '')

    logger.info(f"Processing sample: {sample_name}")
    logger.info(f"2D data: {args.npz_2d}")
    logger.info(f"3D data: {args.npz_3d}")

    # Load metrics
    metrics_2d = load_2d_metrics(args.npz_2d)
    metrics_3d = load_3d_metrics(args.npz_3d)

    # Create comparison plot
    plot_2d_vs_3d_comparison(metrics_2d, metrics_3d, args.output, sample_name)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Comparison Summary")
    logger.info("="*60)
    logger.info(f"{'Metric':<25} {'2D Value':<15} {'3D Value':<15} {'Diff %':<10}")
    logger.info("-"*60)

    # Mean radius comparison
    if not np.isnan(metrics_3d['mean_radius']) and metrics_3d['mean_radius'] > 0:
        diff_mean_pct = (metrics_2d['mean_radius'] - metrics_3d['mean_radius']) / metrics_3d['mean_radius'] * 100
    else:
        diff_mean_pct = np.nan

    logger.info(f"{'Arithmetic Mean Radius':<25} "
               f"{metrics_2d['mean_radius']:<15.4f} "
               f"{metrics_3d['mean_radius']:<15.4f} "
               f"{diff_mean_pct:>+9.1f}%")

    # Effective radius comparison
    if not np.isnan(metrics_3d['r_eff']) and metrics_3d['r_eff'] > 0:
        diff_eff_pct = (metrics_2d['r_eff'] - metrics_3d['r_eff']) / metrics_3d['r_eff'] * 100
    else:
        diff_eff_pct = np.nan

    logger.info(f"{'Effective Radius':<25} "
               f"{metrics_2d['r_eff']:<15.4f} "
               f"{metrics_3d['r_eff']:<15.4f} "
               f"{diff_eff_pct:>+9.1f}%")

    logger.info("="*60 + "\n")


if __name__ == '__main__':
    main()
