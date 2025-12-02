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


def load_2d_metrics(npz_file: Path, radius_type: str = 'minor') -> Dict[str, float]:
    """
    Load 2D slice-based metrics from NPZ file.

    Computes:
    - Arithmetic mean radius (from total histogram)
    - Arithmetic mean radius std (from per-slice means)
    - Effective radius (mean across slices)
    - Effective radius std (across slices)

    Args:
        npz_file: Path to 2D slice profiles NPZ file
        radius_type: 'circular' (from area) or 'minor' (from ellipse), default: 'minor'

    Returns:
        Dictionary with keys: mean_radius, mean_radius_std, r_eff, r_eff_std
    """
    data = np.load(npz_file)

    # Extract data based on radius_type
    bin_centers = data['bin_centers']

    if radius_type == 'minor':
        # Load minor axis histograms
        histograms = data['histograms_minor']  # Shape: (n_slices, n_bins)
        total_histogram = data['total_histogram_minor']
        r_eff_per_slice = data['r_eff_minor_per_slice']
        r_eff_global = float(data['r_eff_minor_global'])
    else:  # circular
        # Load circular-equivalent histograms
        histograms = data['histograms_circular']  # Shape: (n_slices, n_bins)
        total_histogram = data['total_histogram_circular']
        r_eff_per_slice = data['r_eff_circular_per_slice']
        r_eff_global = float(data['r_eff_circular_global'])

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

    # Global pooled effective radius was already loaded above based on radius_type

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


def plot_combined_2d_vs_3d_comparison(
    all_metrics: list[tuple[Dict[str, float], Dict[str, float], str]],
    output_file: Path,
    title: str = "2D vs 3D Metrics Comparison"
) -> None:
    """
    Create two-subplot comparison plot with ALL samples combined.

    Subplot 1: Arithmetic mean radius (all samples)
    Subplot 2: Effective radius (all samples)

    Each point represents one sample with error bars.

    Args:
        all_metrics: List of (metrics_2d, metrics_3d, sample_name) tuples
        output_file: Output PNG file path
        title: Overall plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Collect all data points
    x_mean_all = []
    y_mean_all = []
    yerr_mean_all = []

    x_eff_all = []
    y_eff_all = []
    yerr_eff_all = []

    for metrics_2d, metrics_3d, sample_name in all_metrics:
        # Mean radius
        x_mean_all.append(metrics_3d['mean_radius'])
        y_mean_all.append(metrics_2d['mean_radius'])
        yerr_mean_all.append(metrics_2d['mean_radius_std'])

        # Effective radius
        x_eff_all.append(metrics_3d['r_eff'])
        y_eff_all.append(metrics_2d['r_eff'])
        yerr_eff_all.append(metrics_2d['r_eff_std'])

    # === Subplot 1: Arithmetic Mean Radius ===
    ax1.errorbar(
        x_mean_all, y_mean_all, yerr=yerr_mean_all,
        fmt='o', color='#d62728', markersize=8,
        capsize=4, capthick=1.2, elinewidth=1.2,
        markeredgecolor='black', markeredgewidth=0.5,
        alpha=0.7, label=f'2D vs 3D (n={len(all_metrics)})'
    )

    # Identity line
    all_vals_mean = x_mean_all + y_mean_all
    min_val_mean = min(all_vals_mean) * 0.95
    max_val_mean = max(all_vals_mean) * 1.05

    ax1.plot([min_val_mean, max_val_mean], [min_val_mean, max_val_mean],
             'k--', alpha=0.5, linewidth=1.5, label='Identity')

    ax1.set_xlim(min_val_mean, max_val_mean)
    ax1.set_ylim(min_val_mean, max_val_mean)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('3D Mean Radius (μm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('2D Mean Radius ± std (μm)', fontsize=12, fontweight='bold')
    ax1.set_title('Arithmetic Mean Radius', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)

    # === Subplot 2: Effective Radius ===
    ax2.errorbar(
        x_eff_all, y_eff_all, yerr=yerr_eff_all,
        fmt='o', color='#2ca02c', markersize=8,
        capsize=4, capthick=1.2, elinewidth=1.2,
        markeredgecolor='black', markeredgewidth=0.5,
        alpha=0.7, label=f'2D vs 3D (n={len(all_metrics)})'
    )

    # Identity line
    all_vals_eff = x_eff_all + y_eff_all
    min_val_eff = min(all_vals_eff) * 0.95
    max_val_eff = max(all_vals_eff) * 1.05

    ax2.plot([min_val_eff, max_val_eff], [min_val_eff, max_val_eff],
             'k--', alpha=0.5, linewidth=1.5, label='Identity')

    ax2.set_xlim(min_val_eff, max_val_eff)
    ax2.set_ylim(min_val_eff, max_val_eff)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('3D Effective Radius (μm)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('2D Effective Radius ± std (μm)', fontsize=12, fontweight='bold')
    ax2.set_title('Effective Radius (r_eff)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)

    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)

    plt.tight_layout()

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved combined plot to {output_file}")


def find_matching_pairs(pattern_2d: str, pattern_3d: str) -> tuple[list[tuple[Path, Path, str]], list[Path], list[Path]]:
    """
    Find matching pairs of 2D and 3D profile files using glob patterns.

    Args:
        pattern_2d: Glob pattern for 2D slice profile files (e.g., "data/processed/HM/*_slice_profiles.npz")
        pattern_3d: Glob pattern for 3D axon profile files (e.g., "data/processed/HM/*_axon_profiles.npz")

    Returns:
        Tuple of (matched_pairs, unmatched_2d, unmatched_3d) where:
        - matched_pairs: List of (2d_file, 3d_file, sample_name) tuples
        - unmatched_2d: List of 2D files without matching 3D file
        - unmatched_3d: List of 3D files without matching 2D file
    """
    import glob as glob_module

    slice_files = sorted([Path(f) for f in glob_module.glob(pattern_2d, recursive=True)])
    axon_files = sorted([Path(f) for f in glob_module.glob(pattern_3d, recursive=True)])

    # Build mapping from base name to file paths
    slice_map = {f.stem.replace('_slice_profiles', ''): f for f in slice_files}
    axon_map = {f.stem.replace('_axon_profiles', ''): f for f in axon_files}

    # Find matching pairs
    matched_names = sorted(set(slice_map.keys()) & set(axon_map.keys()))
    pairs = [(slice_map[name], axon_map[name], name) for name in matched_names]

    # Find unmatched files
    unmatched_2d = [slice_map[name] for name in sorted(set(slice_map.keys()) - set(axon_map.keys()))]
    unmatched_3d = [axon_map[name] for name in sorted(set(axon_map.keys()) - set(slice_map.keys()))]

    return pairs, unmatched_2d, unmatched_3d


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Compare 2D slice-based metrics vs 3D axon-based metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file mode
  python plot_2d_versus_3d_metrics.py \\
      data/processed/HM/sham_25_ipsi_slice_profiles.npz \\
      data/processed/HM/sham_25_ipsi_axon_profiles.npz \\
      fig/sham_25_ipsi_comparison.png

  # Batch mode with glob patterns (creates single combined plot)
  python plot_2d_versus_3d_metrics.py \\
      "data/processed/HM/*_slice_profiles.npz" \\
      "data/processed/HM/*_axon_profiles.npz" \\
      fig/combined_2d_vs_3d_comparison.png

  # Batch mode with directory output (default filename: combined_2d_vs_3d_comparison.png)
  python plot_2d_versus_3d_metrics.py \\
      "data/processed/HM/*_slice_profiles.npz" \\
      "data/processed/HM/*_axon_profiles.npz" \\
      fig/comparisons
        """
    )

    parser.add_argument('input', type=str,
                        help='Input: 2D NPZ file OR glob pattern for 2D slice profiles')
    parser.add_argument('output', type=str,
                        help='Output: 3D NPZ file (single mode) OR glob pattern for 3D axon profiles (batch mode)')
    parser.add_argument('output_dir', type=Path, nargs='?', default=None,
                        help='Output PNG file (single mode or batch with specific filename) OR output directory (batch mode with default filename)')
    parser.add_argument('--sample-name', type=str, default="",
                        help='Sample name for plot title (default: auto-detect from filename)')
    parser.add_argument('--radius-type', type=str, default='minor', choices=['circular', 'minor'],
                        help='Radius metric to plot: circular (from area) or minor (from ellipse), default: minor')

    args = parser.parse_args()

    # Determine mode based on presence of wildcards or number of matches
    import glob as glob_module

    pattern_2d = args.input
    pattern_3d = args.output

    # Check if patterns match multiple files (batch mode)
    matches_2d = glob_module.glob(pattern_2d, recursive=True)
    matches_3d = glob_module.glob(pattern_3d, recursive=True)

    is_batch_mode = (len(matches_2d) > 1 or len(matches_3d) > 1 or
                     '*' in pattern_2d or '*' in pattern_3d or
                     '?' in pattern_2d or '?' in pattern_3d)

    if is_batch_mode:
        # Batch mode: process all matching pairs
        if args.output_dir is None:
            parser.error("Batch mode requires output file/directory as third argument")

        output_path = args.output_dir

        # Determine if output is a file or directory
        if str(output_path).endswith('.png'):
            # Output is a specific file
            output_file = output_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Output is a directory - create default filename
            output_dir = output_path
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "combined_2d_vs_3d_comparison.png"

        pairs, unmatched_2d, unmatched_3d = find_matching_pairs(pattern_2d, pattern_3d)

        logger.info(f"\n{'='*80}")
        logger.info(f"Batch Mode")
        logger.info(f"{'='*80}")
        logger.info(f"2D pattern: {pattern_2d}")
        logger.info(f"3D pattern: {pattern_3d}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Radius type: {args.radius_type}")
        logger.info(f"Found {len(matches_2d)} 2D files, {len(matches_3d)} 3D files")
        logger.info(f"Matched pairs: {len(pairs)}")

        if unmatched_2d:
            logger.info(f"\nUnmatched 2D files (no corresponding 3D file):")
            for f in unmatched_2d:
                logger.info(f"  - {f.name}")

        if unmatched_3d:
            logger.info(f"\nUnmatched 3D files (no corresponding 2D file):")
            for f in unmatched_3d:
                logger.info(f"  - {f.name}")

        if len(pairs) == 0:
            logger.error(f"\nNo matching profile pairs found!")
            return

        logger.info(f"{'='*80}\n")

        # Collect all metrics from all samples
        all_metrics = []
        logger.info(f"Loading metrics from {len(pairs)} samples...\n")

        for i, (npz_2d, npz_3d, sample_name) in enumerate(pairs, 1):
            logger.info(f"[{i}/{len(pairs)}] Loading {sample_name}...")
            logger.info(f"  2D: {npz_2d.name}")
            logger.info(f"  3D: {npz_3d.name}")

            # Load metrics
            metrics_2d = load_2d_metrics(npz_2d, radius_type=args.radius_type)
            metrics_3d = load_3d_metrics(npz_3d)

            # Add to collection
            all_metrics.append((metrics_2d, metrics_3d, sample_name))

            # Print individual summary
            logger.info(f"  Arithmetic mean radius: 2D={metrics_2d['mean_radius']:.4f}, 3D={metrics_3d['mean_radius']:.4f}")
            logger.info(f"  Effective radius:       2D={metrics_2d['r_eff']:.4f}, 3D={metrics_3d['r_eff']:.4f}\n")

        # Create single combined plot with all samples
        logger.info(f"\n{'='*80}")
        logger.info(f"Creating combined plot with all {len(all_metrics)} samples...")
        logger.info(f"{'='*80}\n")

        plot_combined_2d_vs_3d_comparison(all_metrics, output_file, "2D vs 3D Metrics Comparison")

        # Print overall summary statistics
        logger.info(f"\n{'='*80}")
        logger.info(f"SUMMARY STATISTICS (n={len(all_metrics)} samples)")
        logger.info(f"{'='*80}")

        # Compute mean differences across all samples
        mean_diffs = []
        eff_diffs = []

        for metrics_2d, metrics_3d, _ in all_metrics:
            if not np.isnan(metrics_3d['mean_radius']) and metrics_3d['mean_radius'] > 0:
                diff_pct = (metrics_2d['mean_radius'] - metrics_3d['mean_radius']) / metrics_3d['mean_radius'] * 100
                mean_diffs.append(diff_pct)

            if not np.isnan(metrics_3d['r_eff']) and metrics_3d['r_eff'] > 0:
                diff_pct = (metrics_2d['r_eff'] - metrics_3d['r_eff']) / metrics_3d['r_eff'] * 100
                eff_diffs.append(diff_pct)

        if mean_diffs:
            logger.info(f"\nArithmetic Mean Radius:")
            logger.info(f"  Average difference: {np.mean(mean_diffs):+.2f}% ± {np.std(mean_diffs):.2f}%")
            logger.info(f"  Range: {np.min(mean_diffs):+.2f}% to {np.max(mean_diffs):+.2f}%")

        if eff_diffs:
            logger.info(f"\nEffective Radius:")
            logger.info(f"  Average difference: {np.mean(eff_diffs):+.2f}% ± {np.std(eff_diffs):.2f}%")
            logger.info(f"  Range: {np.min(eff_diffs):+.2f}% to {np.max(eff_diffs):+.2f}%")

        logger.info(f"\n{'='*80}")
        logger.info(f"Combined plot saved to: {output_file}")
        logger.info(f"{'='*80}\n")

    else:
        # Single file mode
        npz_2d = Path(pattern_2d)
        npz_3d = Path(pattern_3d)
        output_file = args.output_dir

        if output_file is None:
            parser.error("Single file mode requires output PNG file as third argument")

        # Validate input files exist
        if not npz_2d.exists():
            parser.error(f"2D NPZ file not found: {npz_2d}")
        if not npz_3d.exists():
            parser.error(f"3D NPZ file not found: {npz_3d}")

        # Auto-detect sample name if not provided
        sample_name = args.sample_name
        if not sample_name:
            # Extract from filename (remove suffix)
            sample_name = npz_2d.stem.replace('_slice_profiles', '')

        logger.info(f"Single File Mode")
        logger.info(f"Processing sample: {sample_name}")
        logger.info(f"2D data: {npz_2d}")
        logger.info(f"3D data: {npz_3d}")
        logger.info(f"Radius type: {args.radius_type}")

        # Load metrics
        metrics_2d = load_2d_metrics(npz_2d, radius_type=args.radius_type)
        metrics_3d = load_3d_metrics(npz_3d)

        # Create comparison plot
        plot_2d_vs_3d_comparison(metrics_2d, metrics_3d, output_file, sample_name)

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
