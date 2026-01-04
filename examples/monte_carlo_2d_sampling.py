#!/usr/bin/env python3
"""
Monte Carlo simulation to assess correlation when sampling random 2D slices.

This script simulates what happens when taking a single 2D histological sample
per ROI (mimicking traditional 2D sampling), rather than using full 3D data.

For each Monte Carlo iteration:
1. Sample one random slice per ROI
2. Compute r_arith and r_eff from that single slice
3. Collect all (2D, 3D) pairs across ROIs
4. Compute Pearson correlation between 2D and 3D values

Output:
- Distribution of R values across iterations
- Fraction of iterations where p < 0.05
- Figure showing R and p-value distributions
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from axonometry import get_plot_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load plot settings
settings = get_plot_settings()


def compute_r_eff_from_histogram(hist: np.ndarray, bin_centers: np.ndarray) -> float:
    """
    Compute effective MRI-visible radius from histogram.

    r_eff = (⟨r⁶⟩/⟨r²⟩)^(1/4)

    Args:
        hist: Histogram counts
        bin_centers: Bin center values (radii in μm)

    Returns:
        Effective radius in μm
    """
    total_count = hist.sum()
    if total_count == 0:
        return np.nan

    # Weighted moments
    r2_mean = np.sum(bin_centers**2 * hist) / total_count
    r6_mean = np.sum(bin_centers**6 * hist) / total_count

    if r2_mean == 0:
        return np.nan

    return (r6_mean / r2_mean) ** 0.25


def compute_r_arith_from_histogram(hist: np.ndarray, bin_centers: np.ndarray) -> float:
    """
    Compute arithmetic mean radius from histogram.

    Args:
        hist: Histogram counts
        bin_centers: Bin center values (radii in μm)

    Returns:
        Arithmetic mean radius in μm
    """
    total_count = hist.sum()
    if total_count == 0:
        return np.nan

    return np.sum(bin_centers * hist) / total_count


def load_roi_data(npz_file: Path, radius_type: str = 'minor') -> Dict:
    """
    Load ROI data from NPZ file.

    Args:
        npz_file: Path to slice profiles NPZ file
        radius_type: 'minor' or 'circular'

    Returns:
        Dictionary with:
        - histograms: Per-slice histograms (n_slices, n_bins)
        - bin_centers: Bin center values
        - r_eff_per_slice: Pre-computed r_eff per slice (from raw radii)
        - r_eff_3d: 3D reference effective radius
        - r_arith_3d: 3D reference arithmetic mean radius
        - n_slices: Number of valid slices
        - name: ROI name from filename
    """
    data = np.load(npz_file)

    # Select histogram type based on radius_type
    if radius_type == 'minor':
        histograms = data['histograms_minor']
        r_eff_per_slice = data['r_eff_minor_per_slice']
        r_eff_3d = float(data['r_eff_minor_global'])
        r_arith_3d = float(data['mean_radius_minor'])
    else:
        histograms = data['histograms_circular']
        r_eff_per_slice = data['r_eff_circular_per_slice']
        r_eff_3d = float(data['r_eff_circular_global'])
        r_arith_3d = float(data['mean_radius_circular'])

    bin_centers = data['bin_centers']

    # Filter to valid slices (non-empty histograms AND valid r_eff)
    slice_counts = histograms.sum(axis=1)
    valid_mask = (slice_counts > 0) & (r_eff_per_slice > 0)
    histograms = histograms[valid_mask]
    r_eff_per_slice = r_eff_per_slice[valid_mask]

    # Extract ROI name from filename
    name = npz_file.stem.replace('_slice_profiles', '')

    return {
        'histograms': histograms,
        'bin_centers': bin_centers,
        'r_eff_per_slice': r_eff_per_slice,  # Pre-computed from raw radii
        'r_eff_3d': r_eff_3d,
        'r_arith_3d': r_arith_3d,
        'n_slices': len(histograms),
        'name': name
    }


def permutation_test_correlation(x: np.ndarray, y: np.ndarray, n_permutations: int,
                                   rng: np.random.Generator) -> float:
    """
    Compute p-value using permutation test.

    Shuffles y values to break the pairing, computes R for each shuffle,
    and returns fraction of shuffled R values >= observed R.

    Args:
        x: First variable (e.g., 2D values)
        y: Second variable (e.g., 3D values)
        n_permutations: Number of permutations
        rng: Random number generator

    Returns:
        p-value (fraction of permuted R >= observed R)
    """
    # Observed correlation
    observed_r = np.corrcoef(x, y)[0, 1]

    # Count how many permuted correlations are >= observed
    count_ge = 0
    for _ in range(n_permutations):
        y_shuffled = rng.permutation(y)
        perm_r = np.corrcoef(x, y_shuffled)[0, 1]
        if perm_r >= observed_r:
            count_ge += 1

    return count_ge / n_permutations


def run_monte_carlo(
    roi_data_list: List[Dict],
    n_iterations: int = 1000,
    n_permutations: int = 1000,
    seed: int = 42
) -> Dict:
    """
    Run Monte Carlo simulation.

    For each iteration:
    1. Sample one random slice from each ROI
    2. Compute 2D metrics from that slice
    3. Compute correlation between 2D and 3D values across all ROIs
    4. Compute permutation-based p-value by shuffling 3D values

    Args:
        roi_data_list: List of ROI data dictionaries
        n_iterations: Number of Monte Carlo iterations
        n_permutations: Number of permutations for p-value estimation
        seed: Random seed for reproducibility

    Returns:
        Dictionary with:
        - r_arith_correlations: Array of Pearson R values for r_arith
        - r_arith_pvalues: Array of permutation p-values for r_arith
        - r_eff_correlations: Array of Pearson R values for r_eff
        - r_eff_pvalues: Array of permutation p-values for r_eff
    """
    rng = np.random.default_rng(seed)

    n_rois = len(roi_data_list)

    # Storage for results
    r_arith_correlations = np.zeros(n_iterations)
    r_arith_pvalues = np.zeros(n_iterations)
    r_eff_correlations = np.zeros(n_iterations)
    r_eff_pvalues = np.zeros(n_iterations)

    # 3D reference values (constant across iterations)
    r_arith_3d = np.array([roi['r_arith_3d'] for roi in roi_data_list])
    r_eff_3d = np.array([roi['r_eff_3d'] for roi in roi_data_list])

    logger.info(f"Running {n_iterations} Monte Carlo iterations with {n_rois} ROIs...")
    logger.info(f"Using {n_permutations} permutations per iteration for p-value estimation")

    for i in range(n_iterations):
        # Sample one slice per ROI
        r_arith_2d = np.zeros(n_rois)
        r_eff_2d = np.zeros(n_rois)

        for j, roi in enumerate(roi_data_list):
            # Random slice index
            slice_idx = rng.integers(0, roi['n_slices'])
            hist = roi['histograms'][slice_idx]
            bin_centers = roi['bin_centers']

            # Compute 2D metrics from single slice
            # r_arith: computed from histogram (sufficient accuracy for mean)
            # r_eff: use pre-computed value from raw radii (critical for r^6 moment)
            r_arith_2d[j] = compute_r_arith_from_histogram(hist, bin_centers)
            r_eff_2d[j] = roi['r_eff_per_slice'][slice_idx]

        # Compute correlations and permutation p-values
        # Filter out any NaN values
        valid_arith = ~np.isnan(r_arith_2d) & ~np.isnan(r_arith_3d)
        valid_eff = ~np.isnan(r_eff_2d) & ~np.isnan(r_eff_3d)

        if valid_arith.sum() >= 3:
            x_arith = r_arith_2d[valid_arith]
            y_arith = r_arith_3d[valid_arith]
            r_arith_correlations[i] = np.corrcoef(x_arith, y_arith)[0, 1]
            r_arith_pvalues[i] = permutation_test_correlation(x_arith, y_arith, n_permutations, rng)
        else:
            r_arith_correlations[i] = np.nan
            r_arith_pvalues[i] = np.nan

        if valid_eff.sum() >= 3:
            x_eff = r_eff_2d[valid_eff]
            y_eff = r_eff_3d[valid_eff]
            r_eff_correlations[i] = np.corrcoef(x_eff, y_eff)[0, 1]
            r_eff_pvalues[i] = permutation_test_correlation(x_eff, y_eff, n_permutations, rng)
        else:
            r_eff_correlations[i] = np.nan
            r_eff_pvalues[i] = np.nan

        if (i + 1) % 100 == 0:
            logger.info(f"  Completed {i + 1}/{n_iterations} iterations")

    return {
        'r_arith_correlations': r_arith_correlations,
        'r_arith_pvalues': r_arith_pvalues,
        'r_eff_correlations': r_eff_correlations,
        'r_eff_pvalues': r_eff_pvalues,
        'n_iterations': n_iterations,
        'n_rois': n_rois,
        'n_permutations': n_permutations
    }


def plot_monte_carlo_results(
    results: Dict,
    output_file: Path,
    title: str = "Monte Carlo Simulation: 2D Slice Sampling"
) -> None:
    """
    Plot Monte Carlo simulation results.

    Creates a 2x2 figure:
    - Row 1: r_arith (R distribution, p-value distribution)
    - Row 2: r_eff (R distribution, p-value distribution)

    Args:
        results: Dictionary from run_monte_carlo
        output_file: Output PNG file path
        title: Figure title
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    font_settings = settings.fonts
    hist_settings = settings.histogram

    # Colors
    color_r = '#2ecc71'  # Green for R distributions
    color_p = '#e74c3c'  # Red for p-value distributions

    metrics = [
        ('r_arith', 'Arithmetic Mean Radius', results['r_arith_correlations'], results['r_arith_pvalues']),
        ('r_eff', 'Effective Radius', results['r_eff_correlations'], results['r_eff_pvalues'])
    ]

    for row, (metric_key, metric_name, r_vals, p_vals) in enumerate(metrics):
        # Filter valid values
        r_valid = r_vals[~np.isnan(r_vals)]
        p_valid = p_vals[~np.isnan(p_vals)]

        # === R distribution (left column) ===
        ax_r = axes[row, 0]

        ax_r.hist(r_valid, bins=50, alpha=0.7, color=color_r, edgecolor='black', linewidth=0.5)

        # Statistics
        r_mean = np.mean(r_valid)
        r_median = np.median(r_valid)
        r_std = np.std(r_valid)
        r_5, r_95 = np.percentile(r_valid, [5, 95])

        # Vertical line for mean
        ax_r.axvline(r_mean, color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {r_mean:.3f}')
        ax_r.axvline(r_median, color='green', linestyle=':', linewidth=2, label=f'Median: {r_median:.3f}')

        ax_r.set_xlabel('Pearson R', fontsize=font_settings['label_size'], fontweight=font_settings['weight'])
        ax_r.set_ylabel('Count', fontsize=font_settings['label_size'], fontweight=font_settings['weight'])
        ax_r.set_title(f'{metric_name}\nR Distribution',
                       fontsize=font_settings['label_size'], fontweight=font_settings['weight'])

        # Text box with stats
        stats_text = f'Mean: {r_mean:.3f}\nStd: {r_std:.3f}\n5-95%: [{r_5:.3f}, {r_95:.3f}]'
        ax_r.text(0.02, 0.98, stats_text, transform=ax_r.transAxes, fontsize=10,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax_r.legend(loc='upper right', fontsize=font_settings['legend_size'])
        ax_r.grid(True, alpha=0.3)

        # === p-value distribution (right column) ===
        ax_p = axes[row, 1]

        ax_p.hist(p_valid, bins=50, alpha=0.7, color=color_p, edgecolor='black', linewidth=0.5)

        # Vertical line for p=0.05
        ax_p.axvline(0.05, color='darkred', linestyle='--', linewidth=2, label='p = 0.05')

        # Fraction significant
        frac_sig = np.mean(p_valid < 0.05)
        frac_sig_01 = np.mean(p_valid < 0.01)

        ax_p.set_xlabel('p-value', fontsize=font_settings['label_size'], fontweight=font_settings['weight'])
        ax_p.set_ylabel('Count', fontsize=font_settings['label_size'], fontweight=font_settings['weight'])
        ax_p.set_title(f'{metric_name}\np-value Distribution',
                       fontsize=font_settings['label_size'], fontweight=font_settings['weight'])

        # Text box with stats
        p_median = np.median(p_valid)
        stats_text = f'Median p: {p_median:.3e}\np < 0.05: {frac_sig*100:.1f}%\np < 0.01: {frac_sig_01*100:.1f}%'
        ax_p.text(0.98, 0.98, stats_text, transform=ax_p.transAxes, fontsize=10,
                  verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax_p.legend(loc='upper center', fontsize=font_settings['legend_size'])
        ax_p.grid(True, alpha=0.3)

    # Overall title
    n_perm = results.get('n_permutations', 'N/A')
    fig.suptitle(f'{title}\n(n = {results["n_iterations"]} iterations, {results["n_rois"]} ROIs, {n_perm} permutations)',
                 fontsize=font_settings['title_size'], fontweight=font_settings['weight'], y=0.98)

    plt.tight_layout()

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"Saved figure to {output_file}")


def print_summary(results: Dict) -> None:
    """Print summary statistics to console."""
    logger.info("\n" + "=" * 70)
    logger.info("MONTE CARLO SIMULATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Iterations: {results['n_iterations']}")
    logger.info(f"ROIs: {results['n_rois']}")
    logger.info(f"Permutations per iteration: {results.get('n_permutations', 'N/A')}")
    logger.info("=" * 70)

    for metric_name, r_key, p_key in [
        ('Arithmetic Mean Radius', 'r_arith_correlations', 'r_arith_pvalues'),
        ('Effective Radius', 'r_eff_correlations', 'r_eff_pvalues')
    ]:
        r_vals = results[r_key]
        p_vals = results[p_key]

        r_valid = r_vals[~np.isnan(r_vals)]
        p_valid = p_vals[~np.isnan(p_vals)]

        logger.info(f"\n{metric_name}:")
        logger.info(f"  R: mean={np.mean(r_valid):.3f}, std={np.std(r_valid):.3f}")
        logger.info(f"  R: median={np.median(r_valid):.3f}, [5%, 95%]=[{np.percentile(r_valid, 5):.3f}, {np.percentile(r_valid, 95):.3f}]")
        logger.info(f"  p < 0.05: {np.mean(p_valid < 0.05)*100:.1f}%")
        logger.info(f"  p < 0.01: {np.mean(p_valid < 0.01)*100:.1f}%")
        logger.info(f"  Median p-value: {np.median(p_valid):.3e}")

    logger.info("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Monte Carlo simulation for 2D slice sampling correlation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (1000 iterations)
  python monte_carlo_2d_sampling.py \\
      "data/processed/LM/*_slice_profiles.npz" \\
      fig/monte_carlo_2d_sampling.png

  # Run with more iterations
  python monte_carlo_2d_sampling.py \\
      "data/processed/LM/*_slice_profiles.npz" \\
      fig/monte_carlo_2d_sampling.png \\
      --n-iterations 5000

  # Use circular radius instead of minor axis
  python monte_carlo_2d_sampling.py \\
      "data/processed/LM/*_slice_profiles.npz" \\
      fig/monte_carlo_2d_sampling.png \\
      --radius-type circular
        """
    )

    parser.add_argument('input_pattern', type=str,
                        help='Glob pattern for slice profile NPZ files')
    parser.add_argument('output', type=Path,
                        help='Output PNG file path')
    parser.add_argument('--n-iterations', type=int, default=1000,
                        help='Number of Monte Carlo iterations (default: 1000)')
    parser.add_argument('--n-permutations', type=int, default=1000,
                        help='Number of permutations per iteration for p-value estimation (default: 1000)')
    parser.add_argument('--radius-type', type=str, default='minor', choices=['minor', 'circular'],
                        help='Radius type: minor (ellipse) or circular (area-based), default: minor')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Find all matching files
    import glob as glob_module
    npz_files = sorted([Path(f) for f in glob_module.glob(args.input_pattern, recursive=True)])

    if not npz_files:
        logger.error(f"No files found matching pattern: {args.input_pattern}")
        return

    logger.info(f"Found {len(npz_files)} ROI files")

    # Load all ROI data
    roi_data_list = []
    for npz_file in npz_files:
        logger.info(f"Loading {npz_file.name}...")
        roi_data = load_roi_data(npz_file, radius_type=args.radius_type)
        roi_data_list.append(roi_data)
        logger.info(f"  {roi_data['name']}: {roi_data['n_slices']} slices, "
                    f"r_arith_3d={roi_data['r_arith_3d']:.3f}, r_eff_3d={roi_data['r_eff_3d']:.3f}")

    # Run Monte Carlo simulation
    results = run_monte_carlo(
        roi_data_list,
        n_iterations=args.n_iterations,
        n_permutations=args.n_permutations,
        seed=args.seed
    )

    # Print summary
    print_summary(results)

    # Plot results
    plot_monte_carlo_results(results, args.output)

    logger.info(f"\nDone! Figure saved to {args.output}")


if __name__ == '__main__':
    main()
