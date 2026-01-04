#!/usr/bin/env python3
"""
Monte Carlo null test: compare actual 2D slices vs "fake" 2D slices.

This tests whether the 2D-3D correlation is a statistical property of sampling
along-axon variation, or if physical slices have additional structure.

Two approaches compared:
1. ACTUAL 2D: Randomly pick one physical 2D slice per ROI
2. FAKE 2D: For each axon, randomly sample one radius from its 3D profile,
   combine all samples into a "fake" slice

If fake 2D correlates with 3D as well as actual 2D, then the correlation
is just a statistical property of the data.

Creates a 1×2 figure:
(a) Mean radius: 3D vs 2D (actual=filled, fake=open markers)
(b) Effective radius: 3D vs 2D (actual=filled, fake=open markers)

Usage:
    python examples/plot_montecarlo_2d_vs_3d_correlation.py \
        --data-dir data/processed/LM \
        --output fig/montecarlo_2d_vs_3d_correlation.png \
        --n-iterations 1000
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from axonometry import get_plot_settings, add_panel_labels

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


def generate_fake_2d_slice(profiles: List[np.ndarray]) -> np.ndarray:
    """
    Generate a "fake" 2D slice by randomly sampling one radius from each axon.

    For each axon's 3D profile, pick one random radius measurement.
    Combine all samples into a fake slice.
    """
    return np.array([np.random.choice(profile) for profile in profiles])


def compute_fake_2d_metrics(profiles: List[np.ndarray], n_iterations: int
                            ) -> Dict[str, np.ndarray]:
    """
    Generate many fake 2D slices and compute metrics for each.

    Returns:
        Dict with arrays of mean_radius and r_eff values (n_iterations each)
    """
    mean_radii = []
    r_effs = []

    for _ in range(n_iterations):
        fake_radii = generate_fake_2d_slice(profiles)
        mean_radii.append(np.mean(fake_radii))
        r_effs.append(compute_r_eff(fake_radii))

    return {
        'mean_radius': np.array(mean_radii),
        'r_eff': np.array(r_effs)
    }


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


def extract_group_info(sample_name: str) -> Tuple[str, str]:
    """Extract group (TBI/Sham) and population (CC/CG) from sample name."""
    name_lower = sample_name.lower()

    if 'tbi' in name_lower:
        group = "TBI"
    elif 'sham' in name_lower:
        group = "Sham"
    else:
        group = "Unknown"

    if sample_name.endswith('_CC'):
        population = "CC"
    elif sample_name.endswith('_CG'):
        population = "CG"
    else:
        population = None

    return group, population


def load_actual_2d_data(slice_file: Path, radius_type: str = 'circular'
                        ) -> Dict[str, np.ndarray]:
    """
    Load actual 2D slice data for Monte Carlo sampling.

    Returns per-slice metrics arrays that can be randomly sampled.
    """
    data = np.load(slice_file)
    bin_centers = data['bin_centers']
    histograms = data[f'histograms_{radius_type}']
    r_eff_per_slice = data[f'r_eff_{radius_type}_per_slice']

    # Compute per-slice arithmetic mean
    mean_radius_per_slice = []
    valid_slice_indices = []
    for i, hist_slice in enumerate(histograms):
        counts = hist_slice.sum()
        if counts > 0 and r_eff_per_slice[i] > 0:
            mean_r = np.sum(bin_centers * hist_slice) / counts
            mean_radius_per_slice.append(mean_r)
            valid_slice_indices.append(i)

    mean_radius_per_slice = np.array(mean_radius_per_slice)
    valid_r_eff = r_eff_per_slice[valid_slice_indices]

    return {
        'mean_radius_per_slice': mean_radius_per_slice,
        'r_eff_per_slice': valid_r_eff,
        'n_slices': len(mean_radius_per_slice)
    }


def plot_single_scatter(ax, x_3d: np.ndarray, y_2d: np.ndarray,
                        y_ci: Optional[np.ndarray],
                        groups: List[str], populations: List[str],
                        metric: str, title: str, axis_limits: Tuple[float, float]) -> float:
    """
    Plot scatter for a single comparison (actual or fake 2D vs 3D).

    Returns:
        Pearson R value
    """
    font_settings = settings.fonts

    # Plot each ROI
    for i in range(len(x_3d)):
        color = settings.get_group_color(groups[i])
        marker = settings.get_marker(populations[i])

        if y_ci is not None:
            # Fake 2D with error bars
            yerr = [[y_2d[i] - y_ci[0, i]], [y_ci[1, i] - y_2d[i]]]
            ax.errorbar(x_3d[i], y_2d[i], yerr=yerr, fmt='none',
                        ecolor=color, alpha=0.6, capsize=2, capthick=1, elinewidth=1, zorder=5)

        ax.scatter(x_3d[i], y_2d[i], c=color, marker=marker, s=60,
                   edgecolors='black', linewidths=0.5, zorder=10)

    # Compute correlation
    mask = ~(np.isnan(x_3d) | np.isnan(y_2d))
    r, p = stats.pearsonr(x_3d[mask], y_2d[mask])

    # Identity line
    min_val, max_val = axis_limits
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5,
            linewidth=1.5, zorder=0)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_box_aspect(1)

    # Labels
    if metric == 'mean_radius':
        xlabel = r'3D $\bar{r}$ [μm]'
        ylabel = r'2D $\bar{r}$ [μm]'
    else:
        xlabel = r'3D $r_{\mathrm{MRI}}$ [μm]'
        ylabel = r'2D $r_{\mathrm{MRI}}$ [μm]'

    ax.set_xlabel(xlabel, fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.set_ylabel(ylabel, fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])

    # Title and correlation annotation
    ax.text(0.05, 0.95, title,
            transform=ax.transAxes, fontsize=font_settings['legend_size'] + 1,
            ha='left', va='top', fontweight='bold')
    ax.text(0.05, 0.85, f'R = {r:.3f}',
            transform=ax.transAxes, fontsize=font_settings['legend_size'],
            ha='left', va='top')

    return r


def run_monte_carlo_actual_2d(roi_data: List[Dict], x_3d_mean: np.ndarray,
                               x_3d_reff: np.ndarray, n_iterations: int) -> Dict:
    """
    Run Monte Carlo simulation for actual 2D approach.

    For each iteration:
    - Sample one random slice per ROI
    - Compute R between sampled 2D values and 3D values across ROIs
    """
    r_mean_values = []
    r_reff_values = []

    for _ in range(n_iterations):
        y_2d_mean = np.zeros(len(roi_data))
        y_2d_reff = np.zeros(len(roi_data))

        for j, roi in enumerate(roi_data):
            # Sample one random slice
            slice_idx = np.random.randint(0, roi['n_slices'])
            y_2d_mean[j] = roi['mean_radius_per_slice'][slice_idx]
            y_2d_reff[j] = roi['r_eff_per_slice'][slice_idx]

        # Compute correlations
        r_mean, _ = stats.pearsonr(x_3d_mean, y_2d_mean)
        r_reff, _ = stats.pearsonr(x_3d_reff, y_2d_reff)

        r_mean_values.append(r_mean)
        r_reff_values.append(r_reff)

    return {
        'r_mean': np.array(r_mean_values),
        'r_reff': np.array(r_reff_values)
    }


def run_monte_carlo_fake_2d(roi_profiles: List[List[np.ndarray]], x_3d_mean: np.ndarray,
                             x_3d_reff: np.ndarray, n_iterations: int) -> Dict:
    """
    Run Monte Carlo simulation for fake 2D approach.

    For each iteration:
    - For each ROI, sample one random radius per axon to create "fake" slice
    - Compute R between fake 2D values and 3D values across ROIs
    """
    r_mean_values = []
    r_reff_values = []

    for _ in range(n_iterations):
        y_2d_mean = np.zeros(len(roi_profiles))
        y_2d_reff = np.zeros(len(roi_profiles))

        for j, profiles in enumerate(roi_profiles):
            # Generate fake slice: one random radius per axon
            fake_radii = generate_fake_2d_slice(profiles)
            y_2d_mean[j] = np.mean(fake_radii)
            y_2d_reff[j] = compute_r_eff(fake_radii)

        # Compute correlations
        r_mean, _ = stats.pearsonr(x_3d_mean, y_2d_mean)
        r_reff, _ = stats.pearsonr(x_3d_reff, y_2d_reff)

        r_mean_values.append(r_mean)
        r_reff_values.append(r_reff)

    return {
        'r_mean': np.array(r_mean_values),
        'r_reff': np.array(r_reff_values)
    }


def plot_r_distribution(ax, r_values: np.ndarray, color: str, metric_label: str):
    """Plot histogram of R values with statistics."""
    font_settings = settings.fonts

    # Fixed bins from 0 to 1 for all panels
    bins = np.linspace(0, 1, 31)
    ax.hist(r_values, bins=bins, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)

    # Statistics
    r_median = np.median(r_values)
    r_5, r_95 = np.percentile(r_values, [2.5, 97.5])

    # Vertical line for median
    ax.axvline(r_median, color='black', linestyle='--', linewidth=1.5)

    # Shade 95% CI region
    ax.axvspan(r_5, r_95, alpha=0.2, color='gray')

    ax.set_xlabel(f'R ({metric_label})', fontsize=font_settings['label_size'])
    ax.set_ylabel('Count', fontsize=font_settings['label_size'])
    ax.tick_params(labelsize=font_settings['tick_size'])
    ax.set_xlim(0, 1)

    # Stats text
    stats_text = f'Median = {r_median:.3f}\n95% CI [{r_5:.3f}-{r_95:.3f}]'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=font_settings['legend_size'], va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add legend for CI
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.5),
        Patch(facecolor='gray', alpha=0.2)
    ]
    ax.legend(handles, ['Median', '95% CI'], loc='upper right',
              fontsize=font_settings['legend_size'] - 1)

    ax.set_box_aspect(1)

    return r_median, r_5, r_95


def main():
    parser = argparse.ArgumentParser(
        description='Monte Carlo null test: actual vs fake 2D slices',
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
    logger.info("Monte Carlo: R Distribution for 2D-3D Correlation")
    logger.info("=" * 70)

    # Find all ROI pairs
    all_pairs = find_matching_pairs(args.data_dir)
    logger.info(f"Found {len(all_pairs)} ROI pairs")

    # Collect data for Monte Carlo
    x_3d_mean, x_3d_reff = [], []
    roi_slice_data = []  # For actual 2D sampling
    roi_profiles = []     # For fake 2D sampling

    for slice_file, axon_file, sample_name, pop_labels in all_pairs:
        logger.info(f"Processing {sample_name}...")

        # Load 3D profiles
        profiles = load_3d_profiles(axon_file, pop_labels)
        if len(profiles) < 5:
            logger.warning(f"  Skipping - only {len(profiles)} axons")
            continue

        # Load actual 2D slice data
        slice_data = load_actual_2d_data(slice_file, args.radius_type)
        if slice_data['n_slices'] < 1:
            logger.warning(f"  Skipping - no valid slices")
            continue

        # 3D metrics
        m3d = compute_3d_metrics(profiles)
        x_3d_mean.append(m3d['mean_radius'])
        x_3d_reff.append(m3d['r_eff'])

        roi_slice_data.append(slice_data)
        roi_profiles.append(profiles)

        logger.info(f"  {len(profiles)} axons, {slice_data['n_slices']} slices")

    x_3d_mean = np.array(x_3d_mean)
    x_3d_reff = np.array(x_3d_reff)
    n_rois = len(x_3d_mean)

    logger.info(f"\nRunning Monte Carlo with {n_rois} ROIs, {args.n_iterations} iterations...")

    # Run Monte Carlo for actual 2D
    logger.info("  Running actual 2D Monte Carlo...")
    actual_results = run_monte_carlo_actual_2d(roi_slice_data, x_3d_mean, x_3d_reff,
                                                args.n_iterations)

    # Run Monte Carlo for fake 2D
    logger.info("  Running fake 2D Monte Carlo...")
    fake_results = run_monte_carlo_fake_2d(roi_profiles, x_3d_mean, x_3d_reff,
                                            args.n_iterations)

    # Create 2x2 figure: rows = metric, cols = approach
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))

    color_actual = '#3498db'  # Blue
    color_fake = '#e74c3c'    # Red

    # Row 1: Mean radius
    plot_r_distribution(axes[0, 0], actual_results['r_mean'], color_actual, r'$\bar{r}$')
    axes[0, 0].set_title('Actual 2D', fontsize=settings.fonts['label_size'],
                         fontweight=settings.fonts['weight'])

    plot_r_distribution(axes[0, 1], fake_results['r_mean'], color_fake, r'$\bar{r}$')
    axes[0, 1].set_title('Fake 2D', fontsize=settings.fonts['label_size'],
                         fontweight=settings.fonts['weight'])

    # Row 2: Effective radius
    plot_r_distribution(axes[1, 0], actual_results['r_reff'], color_actual, r'$r_{\mathrm{MRI}}$')
    plot_r_distribution(axes[1, 1], fake_results['r_reff'], color_fake, r'$r_{\mathrm{MRI}}$')

    plt.tight_layout()
    add_panel_labels(axes)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=settings.figure['dpi'], bbox_inches='tight')
    svg_output = args.output.with_suffix('.svg')
    plt.savefig(svg_output, bbox_inches='tight')
    plt.close()
    logger.info(f"\nSaved figure to {args.output}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Mean radius (r̄):")
    logger.info(f"  Actual 2D: R = {np.median(actual_results['r_mean']):.3f} "
                f"[{np.percentile(actual_results['r_mean'], 2.5):.3f}, "
                f"{np.percentile(actual_results['r_mean'], 97.5):.3f}]")
    logger.info(f"  Fake 2D:   R = {np.median(fake_results['r_mean']):.3f} "
                f"[{np.percentile(fake_results['r_mean'], 2.5):.3f}, "
                f"{np.percentile(fake_results['r_mean'], 97.5):.3f}]")
    logger.info(f"Effective radius (r_MRI):")
    logger.info(f"  Actual 2D: R = {np.median(actual_results['r_reff']):.3f} "
                f"[{np.percentile(actual_results['r_reff'], 2.5):.3f}, "
                f"{np.percentile(actual_results['r_reff'], 97.5):.3f}]")
    logger.info(f"  Fake 2D:   R = {np.median(fake_results['r_reff']):.3f} "
                f"[{np.percentile(fake_results['r_reff'], 2.5):.3f}, "
                f"{np.percentile(fake_results['r_reff'], 97.5):.3f}]")

    # Save metadata
    metadata = {
        'n_iterations': args.n_iterations,
        'n_rois': n_rois,
        'radius_type': args.radius_type,
        'seed': args.seed,
        'mean_radius': {
            'actual_median': float(np.median(actual_results['r_mean'])),
            'actual_ci': [float(np.percentile(actual_results['r_mean'], 2.5)),
                          float(np.percentile(actual_results['r_mean'], 97.5))],
            'fake_median': float(np.median(fake_results['r_mean'])),
            'fake_ci': [float(np.percentile(fake_results['r_mean'], 2.5)),
                        float(np.percentile(fake_results['r_mean'], 97.5))],
        },
        'r_eff': {
            'actual_median': float(np.median(actual_results['r_reff'])),
            'actual_ci': [float(np.percentile(actual_results['r_reff'], 2.5)),
                          float(np.percentile(actual_results['r_reff'], 97.5))],
            'fake_median': float(np.median(fake_results['r_reff'])),
            'fake_ci': [float(np.percentile(fake_results['r_reff'], 2.5)),
                        float(np.percentile(fake_results['r_reff'], 97.5))],
        }
    }

    json_output = args.output.with_suffix('.json')
    with open(json_output, 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    main()
