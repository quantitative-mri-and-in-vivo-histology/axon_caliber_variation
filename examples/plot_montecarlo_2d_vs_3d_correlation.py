#!/usr/bin/env python3
"""
Monte Carlo comparison: 2D slice sampling vs i.i.d. sampling.

Compares two approaches for estimating 3D population statistics from 2D samples:
1. 2D slice sampling: Randomly pick one physical 2D slice per ROI
2. i.i.d. sampling: Sample mean_axons_per_slice radii from the pooled 2D distribution

Both methods use the same underlying 2D radii (with minor axis bias from
ellipse fitting), ensuring a fair comparison of sampling strategies.

Creates a 1x2 figure showing 2D vs 3D scatter plots:
(a) Mean radius: 3D vs 2D (slice sampling vs i.i.d. sampling)
(b) Effective radius: 3D vs 2D (slice sampling vs i.i.d. sampling)

Median and IQR are computed over Monte Carlo iterations.

Usage:
    python examples/plot_montecarlo_2d_vs_3d_correlation.py \
        --data-dir data/processed/LM \
        --output fig/montecarlo_2d_vs_3d_correlation.png \
        --n-iterations 100
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy import stats

from axonometry import get_plot_settings, style_axis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load plot settings
settings = get_plot_settings()


def compute_r_eff(radii: np.ndarray) -> float:
    """Compute effective MRI-visible radius: r_eff = (<r^6>/<r^2>)^(1/4)"""
    if len(radii) == 0:
        return np.nan
    r2 = radii ** 2
    r6 = radii ** 6
    r2_mean = np.mean(r2)
    r6_mean = np.mean(r6)
    if r2_mean == 0:
        return np.nan
    return (r6_mean / r2_mean) ** 0.25


def compute_wasserstein(radii_2d: np.ndarray, cdf_3d: np.ndarray,
                        bin_centers: np.ndarray, bin_width: float) -> float:
    """
    Compute Wasserstein distance between 2D sample and 3D ground truth CDF.

    Wasserstein distance = integral of |CDF_2D - CDF_3D|
    For discrete CDFs: sum of |CDF differences| * bin_width

    Args:
        radii_2d: Array of 2D radius samples
        cdf_3d: Pre-computed 3D CDF at bin_centers
        bin_centers: Histogram bin centers
        bin_width: Width of histogram bins

    Returns:
        Wasserstein distance in μm
    """
    # Compute 2D histogram and CDF
    hist_2d, _ = np.histogram(radii_2d, bins=np.append(bin_centers - bin_width/2,
                                                        bin_centers[-1] + bin_width/2))
    if hist_2d.sum() == 0:
        return np.nan
    cdf_2d = np.cumsum(hist_2d) / hist_2d.sum()

    # Wasserstein = integral of |CDF_2D - CDF_3D|
    return np.sum(np.abs(cdf_2d - cdf_3d)) * bin_width


def load_3d_profiles(npz_file: Path, population_labels: Optional[Set[int]] = None
                     ) -> List[np.ndarray]:
    """
    Load 3D axon radius profiles filtered by population.

    Returns:
        List of radius profiles (one array per axon)
    """
    data = np.load(npz_file, allow_pickle=True)
    axon_labels = data['labels']
    radii_profiles = data['radii_profiles_um']

    filtered_profiles = []
    for i, label in enumerate(axon_labels):
        if population_labels is None or int(label) in population_labels:
            profile = radii_profiles[i]
            if len(profile) > 0:
                filtered_profiles.append(profile)

    return filtered_profiles


def compute_3d_metrics(profiles: List[np.ndarray]) -> Dict[str, float]:
    """Compute 3D metrics from all radii in profiles."""
    all_radii = np.concatenate(profiles)
    return {
        'mean_radius': np.mean(all_radii),
        'r_eff': compute_r_eff(all_radii)
    }


def generate_iid_sample(all_radii_3d: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Generate i.i.d. sample from pooled 3D distribution.

    Args:
        all_radii_3d: All radii from 3D profiles (pooled)
        n_samples: Number of samples to draw (mean axons per slice)

    Returns:
        Array of sampled radii
    """
    return np.random.choice(all_radii_3d, size=n_samples, replace=True)


def find_matching_pairs(data_dir: Path) -> List[Tuple[Path, Path, str, Set[int]]]:
    """
    Find all matching 2D/3D file pairs with population labels.

    Returns:
        List of (slice_file, axon_file, sample_name, population_labels) tuples
    """
    pairs = []

    # Find all population-specific slice files
    for pop in ['cc', 'cg']:
        slice_files = sorted(data_dir.glob(f"*_{pop}_slice_profiles.npz"))

        for slice_file in slice_files:
            # Extract base name
            base_name = slice_file.stem.replace(f'_{pop}_slice_profiles', '')
            axon_file = data_dir / f"{base_name}_axon_profiles.npz"
            pop_json = data_dir / f"{base_name}_populations.json"

            if axon_file.exists() and pop_json.exists():
                # Load population labels
                with open(pop_json, 'r') as f:
                    pop_data = json.load(f)

                for p in pop_data['populations']:
                    if p['name'].lower() == pop:
                        labels = set(p['axon_labels'])
                        sample_name = f"{base_name}_{pop.upper()}"
                        pairs.append((slice_file, axon_file, sample_name, labels))
                        break

    return pairs


def load_actual_2d_data(slice_file: Path, radius_type: str = 'circular',
                        min_mean_axons_per_slice: int = 1000) -> Dict[str, np.ndarray]:
    """
    Load actual 2D slice data for Monte Carlo sampling.

    Returns per-slice metrics arrays that can be randomly sampled,
    plus pooled 2D radii for i.i.d. sampling.

    Filters out ROIs with mean axons per slice < min_mean_axons_per_slice.
    """
    data = np.load(slice_file)
    bin_centers = data['bin_centers']
    bin_width = bin_centers[1] - bin_centers[0]
    histograms = data[f'histograms_{radius_type}']
    r_eff_per_slice = data[f'r_eff_{radius_type}_per_slice']
    n_axons_per_slice = data['n_axons_per_slice']

    # First pass: compute mean axon count
    all_counts = [h.sum() for h in histograms if h.sum() > 0]
    if not all_counts:
        return {'n_slices': 0, 'mean_axons_per_slice': 0}

    mean_axon_count = np.mean(all_counts)

    # Skip ROIs with insufficient mean axons per slice
    if mean_axon_count < min_mean_axons_per_slice:
        return {'n_slices': 0, 'mean_axons_per_slice': mean_axon_count}

    min_axon_count = mean_axon_count / 2  # Filter slices below half the mean

    # Second pass: filter slices and compute metrics
    mean_radius_per_slice = []
    valid_slice_indices = []
    axon_counts = []
    pooled_2d_radii = []  # Reconstructed radii from histograms
    valid_histograms = []  # Store per-slice histograms for Wasserstein

    for i, hist_slice in enumerate(histograms):
        counts = hist_slice.sum()
        if counts >= min_axon_count and r_eff_per_slice[i] > 0:
            mean_r = np.sum(bin_centers * hist_slice) / counts
            mean_radius_per_slice.append(mean_r)
            valid_slice_indices.append(i)
            axon_counts.append(int(n_axons_per_slice[i]))
            valid_histograms.append(hist_slice)

            # Reconstruct radii from histogram (sample bin centers by counts)
            for bin_idx, bin_count in enumerate(hist_slice):
                if bin_count > 0:
                    pooled_2d_radii.extend([bin_centers[bin_idx]] * int(bin_count))

    mean_radius_per_slice = np.array(mean_radius_per_slice)
    valid_r_eff = r_eff_per_slice[valid_slice_indices]
    mean_axons_per_slice_filtered = int(np.mean(axon_counts)) if axon_counts else 0
    pooled_2d_radii = np.array(pooled_2d_radii)
    valid_histograms = np.array(valid_histograms)

    return {
        'mean_radius_per_slice': mean_radius_per_slice,
        'r_eff_per_slice': valid_r_eff,
        'n_slices': len(mean_radius_per_slice),
        'mean_axons_per_slice': mean_axons_per_slice_filtered,
        'pooled_2d_radii': pooled_2d_radii,
        'histograms': valid_histograms,
        'bin_centers': bin_centers,
        'bin_width': bin_width,
        'mean_axon_count_raw': mean_axon_count
    }


def run_monte_carlo(roi_slice_data: List[Dict], roi_3d_cdfs: List[np.ndarray],
                    n_iterations: int, seed: int = 42) -> Tuple[Dict, Dict]:
    """
    Run Monte Carlo simulation for both 2D slice sampling and i.i.d. sampling.

    2D slice sampling: Randomly pick one physical 2D slice per ROI
    i.i.d. sampling: Sample mean_axons_per_slice radii from pooled 2D distribution

    Both methods use the same underlying 2D radii (with minor axis bias),
    ensuring a fair comparison of sampling strategies.

    Uses separate RNG streams to ensure slice sampling results are independent
    of changes to the i.i.d. sampling implementation.

    Returns:
        Tuple of (slice_results, iid_results) where each is a dict with:
        - mean_radius: array of shape (n_rois, n_iterations)
        - r_eff: array of shape (n_rois, n_iterations)
        - wasserstein: array of shape (n_rois, n_iterations)
    """
    n_rois = len(roi_slice_data)

    slice_mean = np.zeros((n_rois, n_iterations))
    slice_reff = np.zeros((n_rois, n_iterations))
    slice_wass = np.zeros((n_rois, n_iterations))
    iid_mean = np.zeros((n_rois, n_iterations))
    iid_reff = np.zeros((n_rois, n_iterations))
    iid_wass = np.zeros((n_rois, n_iterations))

    # Separate RNG streams for each method to ensure independence
    rng_slice = np.random.default_rng(seed)
    rng_iid = np.random.default_rng(seed + 1)

    # Run Monte Carlo
    for it in range(n_iterations):
        for j in range(n_rois):
            roi = roi_slice_data[j]
            cdf_3d = roi_3d_cdfs[j]
            bin_centers = roi['bin_centers']
            bin_width = roi['bin_width']

            # 2D slice sampling: sample one random slice
            slice_idx = rng_slice.integers(0, roi['n_slices'])
            slice_mean[j, it] = roi['mean_radius_per_slice'][slice_idx]
            slice_reff[j, it] = roi['r_eff_per_slice'][slice_idx]

            # Wasserstein for slice sampling (from histogram)
            hist_slice = roi['histograms'][slice_idx]
            cdf_2d_slice = np.cumsum(hist_slice) / hist_slice.sum()
            slice_wass[j, it] = np.sum(np.abs(cdf_2d_slice - cdf_3d)) * bin_width

            # i.i.d. sampling: sample mean_axons_per_slice radii from pooled 2D
            n_samples = roi['mean_axons_per_slice']
            iid_radii = rng_iid.choice(roi['pooled_2d_radii'], size=n_samples, replace=False)
            iid_mean[j, it] = np.mean(iid_radii)
            iid_reff[j, it] = compute_r_eff(iid_radii)

            # Wasserstein for i.i.d. sampling
            iid_wass[j, it] = compute_wasserstein(iid_radii, cdf_3d, bin_centers, bin_width)

    return (
        {'mean_radius': slice_mean, 'r_eff': slice_reff, 'wasserstein': slice_wass},
        {'mean_radius': iid_mean, 'r_eff': iid_reff, 'wasserstein': iid_wass}
    )


def plot_scatter_comparison(ax, x_3d: np.ndarray, actual_results: np.ndarray,
                             fake_results: np.ndarray, metric: str,
                             shared_lim: Optional[Tuple[float, float]] = None,
                             p_actual: Optional[float] = None,
                             p_fake: Optional[float] = None) -> Tuple[float, float, float, float]:
    """
    Plot 2D vs 3D scatter comparing slice sampling and i.i.d. sampling.

    Args:
        ax: Matplotlib axes
        x_3d: 3D metric values for each ROI (n_rois,)
        actual_results: Actual 2D results (n_rois, n_iterations)
        fake_results: Fake 2D results (n_rois, n_iterations)
        metric: 'mean_radius' or 'r_eff'
        shared_lim: Optional tuple (min_val, max_val) for shared axis limits
        p_actual: Permutation p-value for slice sampling
        p_fake: Permutation p-value for i.i.d. sampling

    Returns:
        Tuple of (r_actual, r_fake, r_actual_std, r_fake_std)
    """
    font_settings = settings.fonts
    err_settings = settings.error_bars

    # Colors for the two approaches (colored binary comparison)
    color_actual = settings.colors['binary_a']  # Sand/tan for 2D slice sampling
    color_fake = settings.colors['binary_b']    # Dusty teal for i.i.d. sampling

    n_iterations = actual_results.shape[1]

    # Compute R for each MC iteration, then average
    r_actual_per_iter = []
    r_fake_per_iter = []
    for it in range(n_iterations):
        r_act, _ = stats.pearsonr(x_3d, actual_results[:, it])
        r_fake, _ = stats.pearsonr(x_3d, fake_results[:, it])
        r_actual_per_iter.append(r_act)
        r_fake_per_iter.append(r_fake)

    r_actual_per_iter = np.array(r_actual_per_iter)
    r_fake_per_iter = np.array(r_fake_per_iter)

    r_actual_mean = np.mean(r_actual_per_iter)
    r_actual_std = np.std(r_actual_per_iter)
    r_fake_mean = np.mean(r_fake_per_iter)
    r_fake_std = np.std(r_fake_per_iter)

    # For plotting: compute mean and std of 2D values over iterations for each ROI
    actual_mean = np.mean(actual_results, axis=1)
    actual_std = np.std(actual_results, axis=1)
    fake_mean = np.mean(fake_results, axis=1)
    fake_std = np.std(fake_results, axis=1)

    # Plot actual (2D slice sampling)
    ax.errorbar(x_3d, actual_mean, yerr=actual_std, fmt='o', color=color_actual,
                markersize=6, capsize=err_settings['capsize'],
                capthick=err_settings['capthick'], elinewidth=err_settings['linewidth'],
                markeredgecolor='black', markeredgewidth=0.5,
                alpha=0.8, label='2D slice sampling', zorder=10)

    # Plot fake (i.i.d. sampling)
    ax.errorbar(x_3d, fake_mean, yerr=fake_std, fmt='s', color=color_fake,
                markersize=6, capsize=err_settings['capsize'],
                capthick=err_settings['capthick'], elinewidth=err_settings['linewidth'],
                markeredgecolor='black', markeredgewidth=0.5,
                alpha=0.8, label='i.i.d. sampling', zorder=10)

    # Identity line and axis limits
    if shared_lim is not None:
        min_val, max_val = shared_lim
    else:
        all_vals = np.concatenate([x_3d, actual_mean, fake_mean])
        min_val = np.nanmin(all_vals) * 0.95
        max_val = np.nanmax(all_vals) * 1.05

    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5,
            linewidth=1.5, zorder=0)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Format ticks to 2 decimal places
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Labels with math symbols
    if metric == 'mean_radius':
        xlabel = r'$\bar{r}$ (3D) [μm]'
        ylabel = r'$\bar{r}$ (2D) [μm]'
    else:
        xlabel = r'$r_{\mathrm{MRI}}$ (3D) [μm]'
        ylabel = r'$r_{\mathrm{MRI}}$ (2D) [μm]'

    style_axis(ax, xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc='upper left', fontsize=font_settings['legend_size'])

    # Correlation annotation (mean ± std across MC iterations, with p-value)
    if p_actual is not None:
        p_str_actual = 'p < 0.001' if p_actual < 0.001 else f'p = {p_actual:.3f}'
        ax.text(0.95, 0.15, f'R = {r_actual_mean:.2f}±{r_actual_std:.2f}, {p_str_actual}',
                transform=ax.transAxes, fontsize=font_settings['legend_size'],
                ha='right', va='bottom', color=color_actual)
    else:
        ax.text(0.95, 0.15, f'R = {r_actual_mean:.2f}±{r_actual_std:.2f}',
                transform=ax.transAxes, fontsize=font_settings['legend_size'],
                ha='right', va='bottom', color=color_actual)

    if p_fake is not None:
        p_str_fake = 'p < 0.001' if p_fake < 0.001 else f'p = {p_fake:.3f}'
        ax.text(0.95, 0.05, f'R = {r_fake_mean:.2f}±{r_fake_std:.2f}, {p_str_fake}',
                transform=ax.transAxes, fontsize=font_settings['legend_size'],
                ha='right', va='bottom', color=color_fake)
    else:
        ax.text(0.95, 0.05, f'R = {r_fake_mean:.2f}±{r_fake_std:.2f}',
                transform=ax.transAxes, fontsize=font_settings['legend_size'],
                ha='right', va='bottom', color=color_fake)

    ax.set_box_aspect(1)

    return r_actual_mean, r_fake_mean, r_actual_std, r_fake_std


def plot_wasserstein_violin(ax, slice_wass: np.ndarray, iid_wass: np.ndarray) -> None:
    """
    Plot violin comparison of Wasserstein distances (pooled over all MC iterations).

    Args:
        ax: Matplotlib axes
        slice_wass: Wasserstein distances for slice sampling (n_rois, n_iterations)
        iid_wass: Wasserstein distances for i.i.d. sampling (n_rois, n_iterations)
    """
    font_settings = settings.fonts

    # Pool all values across ROIs and iterations
    slice_pooled = slice_wass.flatten()
    iid_pooled = iid_wass.flatten()

    # Colors (colored binary comparison)
    color_slice = settings.colors['binary_a']  # Sand/tan for slice sampling
    color_iid = settings.colors['binary_b']    # Dusty teal for i.i.d. sampling

    # Create violin plots
    parts = ax.violinplot([slice_pooled, iid_pooled], positions=[1, 2],
                           showmeans=False, showmedians=True, widths=0.7)

    # Style violins
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor([color_slice, color_iid][i])
        body.set_edgecolor('black')
        body.set_alpha(0.7)

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(1.5)

    # Add scatter for individual points (subsample for clarity)
    rng = np.random.default_rng(42)
    n_show = min(500, len(slice_pooled))
    idx_slice = rng.choice(len(slice_pooled), n_show, replace=False)
    idx_iid = rng.choice(len(iid_pooled), n_show, replace=False)

    jitter_slice = rng.uniform(-0.15, 0.15, n_show)
    jitter_iid = rng.uniform(-0.15, 0.15, n_show)

    ax.scatter(1 + jitter_slice, slice_pooled[idx_slice], alpha=0.3, s=3,
               color=color_slice, zorder=0)
    ax.scatter(2 + jitter_iid, iid_pooled[idx_iid], alpha=0.3, s=3,
               color=color_iid, zorder=0)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['2D slice\nsampling', 'i.i.d.\nsampling'],
                        fontsize=font_settings['tick_size'])
    style_axis(ax, ylabel='Wasserstein distance [μm]')
    ax.set_ylim(bottom=0)

    ax.set_box_aspect(1)


def compute_3d_cdf(profiles: List[np.ndarray], bin_centers: np.ndarray) -> np.ndarray:
    """Compute 3D CDF at given bin centers."""
    all_radii = np.concatenate(profiles)
    sorted_radii = np.sort(all_radii)
    cdf_y = np.arange(1, len(sorted_radii) + 1) / len(sorted_radii)
    return np.interp(bin_centers, sorted_radii, cdf_y, left=0, right=1)


def compute_permutation_pvalue(x_3d: np.ndarray, roi_slice_data: List[Dict],
                                metric: str, n_perm: int = 10000,
                                seed: int = 42) -> float:
    """
    Compute permutation p-value for correlation.

    Matches main figure approach: shuffle 3D labels AND pick new random slices
    for each permutation.

    Args:
        x_3d: 3D metric values (n_rois,)
        roi_slice_data: List of dicts with per-slice metrics
        metric: 'mean_radius' or 'r_eff' - which metric to use
        n_perm: Number of permutations
        seed: Random seed

    Returns:
        p-value (fraction of permuted R >= observed R)
    """
    n_rois = len(x_3d)
    rng = np.random.default_rng(seed)

    # Key for per-slice data
    slice_key = 'mean_radius_per_slice' if metric == 'mean_radius' else 'r_eff_per_slice'

    # Compute observed R (mean across many random slice picks)
    n_obs_iter = 1000
    r_obs_per_iter = []
    for _ in range(n_obs_iter):
        y_2d = np.zeros(n_rois)
        for j, roi in enumerate(roi_slice_data):
            slice_idx = rng.integers(0, roi['n_slices'])
            y_2d[j] = roi[slice_key][slice_idx]
        r, _ = stats.pearsonr(x_3d, y_2d)
        r_obs_per_iter.append(r)
    r_observed = np.mean(r_obs_per_iter)

    # Permutation test: shuffle 3D AND pick new random slices
    r_perm = np.zeros(n_perm)
    for i in range(n_perm):
        perm_idx = rng.permutation(n_rois)
        x_3d_perm = x_3d[perm_idx]

        y_2d = np.zeros(n_rois)
        for j, roi in enumerate(roi_slice_data):
            slice_idx = rng.integers(0, roi['n_slices'])
            y_2d[j] = roi[slice_key][slice_idx]

        r_perm[i], _ = stats.pearsonr(x_3d_perm, y_2d)

    p_value = np.mean(r_perm >= r_observed)

    return p_value


def main():
    parser = argparse.ArgumentParser(
        description='Monte Carlo comparison: 2D slice sampling vs i.i.d. sampling',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data-dir', type=Path, default=Path('data/processed/LM'),
                        help='Directory containing LM slice and axon profiles')
    parser.add_argument('--output', type=Path,
                        default=Path('fig/montecarlo_2d_vs_3d_correlation.png'),
                        help='Output figure path')
    parser.add_argument('--n-iterations', type=int, default=100,
                        help='Number of Monte Carlo iterations (default: 100)')
    parser.add_argument('--radius-type', type=str, default='circular',
                        choices=['circular', 'minor'],
                        help='Radius type to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    np.random.seed(args.seed)

    logger.info("=" * 70)
    logger.info("Monte Carlo: 2D Slice Sampling vs i.i.d. Sampling")
    logger.info("=" * 70)

    # Find all ROI pairs
    all_pairs = find_matching_pairs(args.data_dir)
    logger.info(f"Found {len(all_pairs)} ROI pairs")

    # Collect data for Monte Carlo
    x_3d_mean, x_3d_reff = [], []
    roi_slice_data = []  # For both slice and i.i.d. sampling (both use 2D data)
    roi_3d_cdfs = []     # 3D CDFs for Wasserstein computation
    roi_profiles = []    # Store profiles for CDF computation

    for slice_file, axon_file, sample_name, pop_labels in all_pairs:
        logger.info(f"Processing {sample_name}...")

        # Load 3D profiles for ground truth
        profiles = load_3d_profiles(axon_file, pop_labels)
        if len(profiles) < 5:
            logger.warning(f"  Skipping - only {len(profiles)} axons")
            continue

        # Load 2D slice data (for both slice sampling and i.i.d. sampling)
        slice_data = load_actual_2d_data(slice_file, args.radius_type)
        if slice_data['n_slices'] < 1:
            logger.warning(f"  Skipping - no valid slices")
            continue

        # 3D metrics (ground truth)
        m3d = compute_3d_metrics(profiles)
        x_3d_mean.append(m3d['mean_radius'])
        x_3d_reff.append(m3d['r_eff'])

        roi_slice_data.append(slice_data)
        roi_profiles.append(profiles)

        logger.info(f"  {len(profiles)} axons, {slice_data['n_slices']} slices, "
                    f"{len(slice_data['pooled_2d_radii'])} pooled 2D radii")

    x_3d_mean = np.array(x_3d_mean)
    x_3d_reff = np.array(x_3d_reff)
    n_rois = len(x_3d_mean)

    # Compute 3D CDFs for each ROI (using bin_centers from first ROI)
    logger.info("\nComputing 3D CDFs for Wasserstein distance...")
    bin_centers = roi_slice_data[0]['bin_centers']
    for profiles in roi_profiles:
        cdf_3d = compute_3d_cdf(profiles, bin_centers)
        roi_3d_cdfs.append(cdf_3d)

    logger.info(f"\nRunning Monte Carlo with {n_rois} ROIs, {args.n_iterations} iterations...")

    # Run Monte Carlo
    actual_results, fake_results = run_monte_carlo(roi_slice_data, roi_3d_cdfs,
                                                    args.n_iterations, seed=args.seed)

    # Compute permutation p-values (slice sampling only - matches main figure approach)
    # For i.i.d. sampling, use simpler approach since there's no slice structure
    logger.info("\nComputing permutation p-values (10k permutations)...")
    p_mean_slice = compute_permutation_pvalue(x_3d_mean, roi_slice_data,
                                               metric='mean_radius',
                                               n_perm=10000, seed=args.seed + 100)
    p_reff_slice = compute_permutation_pvalue(x_3d_reff, roi_slice_data,
                                               metric='r_eff',
                                               n_perm=10000, seed=args.seed + 102)

    # For i.i.d. sampling: use mean 2D values (no slice structure to resample)
    y_iid_mean = np.mean(fake_results['mean_radius'], axis=1)
    y_iid_reff = np.mean(fake_results['r_eff'], axis=1)
    rng_perm = np.random.default_rng(args.seed + 200)

    r_iid_mean_obs, _ = stats.pearsonr(x_3d_mean, y_iid_mean)
    r_iid_reff_obs, _ = stats.pearsonr(x_3d_reff, y_iid_reff)

    r_iid_mean_perm = np.zeros(10000)
    r_iid_reff_perm = np.zeros(10000)
    for i in range(10000):
        perm_idx = rng_perm.permutation(n_rois)
        r_iid_mean_perm[i], _ = stats.pearsonr(x_3d_mean[perm_idx], y_iid_mean)
        r_iid_reff_perm[i], _ = stats.pearsonr(x_3d_reff[perm_idx], y_iid_reff)

    p_mean_iid = np.mean(r_iid_mean_perm >= r_iid_mean_obs)
    p_reff_iid = np.mean(r_iid_reff_perm >= r_iid_reff_obs)

    logger.info(f"  Mean radius p-values: slice={p_mean_slice:.4f}, i.i.d.={p_mean_iid:.4f}")
    logger.info(f"  Effective radius p-values: slice={p_reff_slice:.4f}, i.i.d.={p_reff_iid:.4f}")

    # Create 1x3 figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel (a): Wasserstein distance violin
    logger.info("\nPlotting panel (a): Wasserstein distance comparison...")
    plot_wasserstein_violin(axes[0], actual_results['wasserstein'],
                            fake_results['wasserstein'])

    # Panel (b): Mean radius (auto axis limits)
    logger.info("Plotting panel (b): Mean radius comparison...")
    r_mean_slice, r_mean_iid, r_mean_slice_std, r_mean_iid_std = plot_scatter_comparison(
        axes[1], x_3d_mean, actual_results['mean_radius'],
        fake_results['mean_radius'], 'mean_radius',
        p_actual=p_mean_slice, p_fake=p_mean_iid)

    # Panel (c): Effective radius (auto axis limits)
    logger.info("Plotting panel (c): Effective radius comparison...")
    r_reff_slice, r_reff_iid, r_reff_slice_std, r_reff_iid_std = plot_scatter_comparison(
        axes[2], x_3d_reff, actual_results['r_eff'],
        fake_results['r_eff'], 'r_eff',
        p_actual=p_reff_slice, p_fake=p_reff_iid)

    plt.tight_layout()

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=settings.figure['dpi'], bbox_inches='tight')
    svg_output = args.output.with_suffix('.svg')
    plt.savefig(svg_output, bbox_inches='tight')
    plt.close()
    logger.info(f"\nSaved figure to {args.output} and {svg_output}")

    # Wasserstein stats
    wass_slice = actual_results['wasserstein'].flatten()
    wass_iid = fake_results['wasserstein'].flatten()
    wass_slice_median = np.median(wass_slice)
    wass_iid_median = np.median(wass_iid)
    wass_slice_iqr = np.percentile(wass_slice, 75) - np.percentile(wass_slice, 25)
    wass_iid_iqr = np.percentile(wass_iid, 75) - np.percentile(wass_iid, 25)

    # Format p-values for logging
    def fmt_p(p):
        return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Wasserstein distance (median, IQR):")
    logger.info(f"  2D slice sampling: {wass_slice_median:.4f} ({wass_slice_iqr:.4f})")
    logger.info(f"  i.i.d. sampling:   {wass_iid_median:.4f} ({wass_iid_iqr:.4f})")
    logger.info(f"Correlation with 3D (mean ± std across MC iterations):")
    logger.info(f"  Mean radius:")
    logger.info(f"    2D slice sampling: R = {r_mean_slice:.2f} ± {r_mean_slice_std:.2f}, {fmt_p(p_mean_slice)}")
    logger.info(f"    i.i.d. sampling:   R = {r_mean_iid:.2f} ± {r_mean_iid_std:.2f}, {fmt_p(p_mean_iid)}")
    logger.info(f"  Effective radius:")
    logger.info(f"    2D slice sampling: R = {r_reff_slice:.2f} ± {r_reff_slice_std:.2f}, {fmt_p(p_reff_slice)}")
    logger.info(f"    i.i.d. sampling:   R = {r_reff_iid:.2f} ± {r_reff_iid_std:.2f}, {fmt_p(p_reff_iid)}")

    # Save metadata
    metadata = {
        'n_iterations': args.n_iterations,
        'n_rois': n_rois,
        'radius_type': args.radius_type,
        'seed': args.seed,
        'min_mean_axons_per_slice': 1000,
        'wasserstein': {
            'slice_sampling_median': float(wass_slice_median),
            'slice_sampling_iqr': float(wass_slice_iqr),
            'iid_sampling_median': float(wass_iid_median),
            'iid_sampling_iqr': float(wass_iid_iqr),
        },
        'mean_radius': {
            'r_slice_sampling': float(r_mean_slice),
            'r_slice_sampling_std': float(r_mean_slice_std),
            'p_slice_sampling': float(p_mean_slice),
            'r_iid_sampling': float(r_mean_iid),
            'r_iid_sampling_std': float(r_mean_iid_std),
            'p_iid_sampling': float(p_mean_iid),
        },
        'r_eff': {
            'r_slice_sampling': float(r_reff_slice),
            'r_slice_sampling_std': float(r_reff_slice_std),
            'p_slice_sampling': float(p_reff_slice),
            'r_iid_sampling': float(r_reff_iid),
            'r_iid_sampling_std': float(r_reff_iid_std),
            'p_iid_sampling': float(p_reff_iid),
        }
    }

    json_output = args.output.with_suffix('.json')
    with open(json_output, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {json_output}")


if __name__ == '__main__':
    main()
