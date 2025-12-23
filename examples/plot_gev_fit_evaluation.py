#!/usr/bin/env python3
"""
Evaluate GEV distribution fit quality against empirical data.

Creates a 2×3 figure:
Row 1 (Human CC):
  (a) PDF comparison: empirical histogram + fitted GEV PDF (largest ROI)
  (b) QQ-plot: fitted vs empirical quantiles with median ± IQR across ROIs
  (c) GEV parameter ranges: mean ± std across ROIs

Row 2 (Rat WM):
  (d-f) Same panels for rat data

Usage:
    python examples/plot_gev_fit_evaluation.py \
        --human-data data/raw_LM \
        --rat-data data/processed/LM \
        --output fig/gev_fit_evaluation.png
"""

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import OptimizeWarning

sys.path.insert(0, str(Path(__file__).parent.parent))
from axonometry import get_plot_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings = get_plot_settings()

# Constants
DEFAULT_BIN_WIDTH = 0.05  # μm
PLOT_XLIM_MAX = 2.5
N_QUANTILES = 100
MIN_BIN_PROB = 1e-300


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class GEVFitResult:
    """Container for GEV fit results."""
    xi: float  # shape parameter
    mu: float  # location parameter
    sigma: float  # scale parameter
    pdf_values: np.ndarray
    quantiles_fitted: np.ndarray  # fitted quantiles at standard quantile points
    quantiles_empirical: np.ndarray  # empirical quantiles at same points
    aic: float


@dataclass
class DatasetResults:
    """Container for all ROI results in a dataset."""
    name: str
    n_rois: int
    bin_centers: np.ndarray
    bin_edges: np.ndarray
    # Per-ROI fit results
    fit_results: List[GEVFitResult]
    # Largest ROI data for PDF panel
    largest_roi_histogram: np.ndarray
    largest_roi_fit: GEVFitResult
    # Aggregated parameters (across ROIs)
    xi_values: np.ndarray
    mu_values: np.ndarray
    sigma_values: np.ndarray


# =============================================================================
# Data Loading (adapted from fit_combined_distributions.py)
# =============================================================================

def rediscretize_histogram(
    bin_edges: np.ndarray,
    counts: np.ndarray,
    new_bin_width: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rediscretize histogram to coarser bins."""
    min_edge = bin_edges[0]
    max_edge = bin_edges[-1]
    new_bin_edges = np.arange(min_edge, max_edge + new_bin_width, new_bin_width)
    old_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    new_counts = np.zeros(len(new_bin_edges) - 1, dtype=counts.dtype)
    for i, center in enumerate(old_bin_centers):
        new_bin_idx = int((center - min_edge) / new_bin_width)
        if 0 <= new_bin_idx < len(new_counts):
            new_counts[new_bin_idx] += counts[i]

    new_bin_centers = (new_bin_edges[:-1] + new_bin_edges[1:]) / 2
    return new_bin_edges, new_bin_centers, new_counts


def load_human_cc_histograms(
    data_dir: Path,
    bin_width: float = DEFAULT_BIN_WIDTH
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load human corpus callosum histogram data.

    Returns:
        Tuple of (bin_edges, bin_centers, counts_matrix, sample_names)
        counts_matrix shape: (n_rois, n_bins)
    """
    bin_edges_file = data_dir / 'desc-binEdges_radii.tsv'
    counts_file = data_dir / 'desc-countsCircularEq_radii.tsv'

    bin_edges_orig = np.loadtxt(bin_edges_file, delimiter='\t', skiprows=1)
    counts_matrix_orig = np.loadtxt(counts_file, delimiter='\t', skiprows=1, dtype=float)
    n_rois = counts_matrix_orig.shape[0]

    # Rediscretize
    first_edges, first_centers, _ = rediscretize_histogram(
        bin_edges_orig, counts_matrix_orig[0], bin_width
    )
    n_bins = len(first_centers)
    counts_matrix = np.zeros((n_rois, n_bins), dtype=float)

    for i in range(n_rois):
        _, _, counts_matrix[i] = rediscretize_histogram(
            bin_edges_orig, counts_matrix_orig[i], bin_width
        )

    sample_names = [f"ROI_{i+1}" for i in range(n_rois)]
    logger.info(f"Human CC: {n_rois} ROIs, {int(counts_matrix.sum()):,} total axons")

    return first_edges, first_centers, counts_matrix, sample_names


def load_rat_histograms(
    data_dir: Path,
    bin_width: float = DEFAULT_BIN_WIDTH,
    r_max: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load rat LM data from NPZ files, split by population.

    Returns:
        Tuple of (bin_edges, bin_centers, counts_matrix, sample_names)
        counts_matrix shape: (n_rois, n_bins)
    """
    npz_files = sorted(data_dir.glob('*_axon_profiles.npz'))
    if not npz_files:
        raise ValueError(f"No *_axon_profiles.npz files found in {data_dir}")

    bin_edges = np.arange(0, r_max + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    all_counts = []
    all_names = []

    for npz_file in npz_files:
        volume_name = npz_file.stem.replace('_axon_profiles', '')
        pop_file = npz_file.parent / f"{volume_name}_populations.json"

        data = np.load(npz_file, allow_pickle=True)
        labels = data['labels']
        radii_profiles = data['radii_profiles_um']

        if pop_file.exists():
            with open(pop_file) as f:
                pop_data = json.load(f)

            for pop in pop_data['populations']:
                pop_name = pop['name'].upper()
                pop_labels = set(pop['axon_labels'])

                mask = np.isin(labels, list(pop_labels))
                if mask.sum() > 0:
                    pop_radii = np.concatenate([radii_profiles[i] for i in np.where(mask)[0]])
                    counts, _ = np.histogram(pop_radii, bins=bin_edges)
                    all_counts.append(counts)
                    all_names.append(f"{volume_name}_{pop_name}")
        else:
            radii = data['all_radii_um']
            counts, _ = np.histogram(radii, bins=bin_edges)
            all_counts.append(counts)
            all_names.append(volume_name)

    counts_matrix = np.array(all_counts, dtype=float)
    logger.info(f"Rat WM: {len(all_names)} ROIs, {int(counts_matrix.sum()):,} total radii")

    return bin_edges, bin_centers, counts_matrix, all_names


# =============================================================================
# GEV Fitting
# =============================================================================

def fit_gev_to_histogram(
    bin_centers: np.ndarray,
    bin_edges: np.ndarray,
    counts: np.ndarray,
    quantile_points: np.ndarray
) -> Optional[GEVFitResult]:
    """
    Fit GEV distribution to histogram data.

    Args:
        bin_centers: Histogram bin centers
        bin_edges: Histogram bin edges
        counts: Histogram counts
        quantile_points: Quantile points for QQ-plot (e.g., [0.01, 0.02, ..., 0.99])

    Returns:
        GEVFitResult or None if fitting fails
    """
    total = counts.sum()
    if total < 100:
        return None

    # Generate samples from histogram for fitting
    n_samples = min(10000, int(total))
    probs = counts / total
    probs = probs / probs.sum()

    bin_width = np.diff(bin_edges).mean()

    try:
        sampled_bins = np.random.choice(len(bin_centers), size=n_samples, p=probs)
        jitter = np.random.uniform(-bin_width/2, bin_width/2, n_samples)
        samples = bin_centers[sampled_bins] + jitter
        samples = samples[samples > 0]

        if len(samples) < 100:
            return None

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', category=OptimizeWarning)
            params = stats.genextreme.fit(samples)
            # If loc is negative, refit with floc=0
            if params[1] < 0:
                params = stats.genextreme.fit(samples, floc=0)

        xi, mu, sigma = params[0], params[1], params[2]

        # Compute PDF values
        pdf_values = stats.genextreme.pdf(bin_centers, xi, loc=mu, scale=sigma)

        # Compute AIC
        edge_cdfs = stats.genextreme.cdf(bin_edges, xi, loc=mu, scale=sigma)
        bin_probs = edge_cdfs[1:] - edge_cdfs[:-1]
        bin_probs = np.maximum(bin_probs, MIN_BIN_PROB)
        nll = -np.dot(counts, np.log(bin_probs))
        aic = 2 * 3 + 2 * nll  # 3 parameters

        # Compute quantiles
        # Empirical quantiles from histogram
        cdf_empirical = np.cumsum(counts) / total
        quantiles_empirical = np.interp(quantile_points, cdf_empirical, bin_centers)

        # Fitted quantiles from GEV
        quantiles_fitted = stats.genextreme.ppf(quantile_points, xi, loc=mu, scale=sigma)

        return GEVFitResult(
            xi=xi,
            mu=mu,
            sigma=sigma,
            pdf_values=pdf_values,
            quantiles_fitted=quantiles_fitted,
            quantiles_empirical=quantiles_empirical,
            aic=aic
        )

    except (ValueError, RuntimeError) as e:
        logger.debug(f"GEV fit failed: {e}")
        return None


def fit_all_rois(
    bin_edges: np.ndarray,
    bin_centers: np.ndarray,
    counts_matrix: np.ndarray,
    sample_names: List[str],
    dataset_name: str
) -> DatasetResults:
    """
    Fit GEV to all ROIs in a dataset.

    Returns:
        DatasetResults with all fit results and aggregated parameters
    """
    n_rois = counts_matrix.shape[0]
    quantile_points = np.linspace(0.01, 0.99, N_QUANTILES)

    # Set seed for reproducibility
    np.random.seed(42)

    fit_results = []
    xi_values = []
    mu_values = []
    sigma_values = []

    logger.info(f"Fitting GEV to {n_rois} ROIs...")

    for i in range(n_rois):
        result = fit_gev_to_histogram(
            bin_centers, bin_edges, counts_matrix[i], quantile_points
        )
        if result is not None:
            fit_results.append(result)
            xi_values.append(result.xi)
            mu_values.append(result.mu)
            sigma_values.append(result.sigma)

    logger.info(f"  Successfully fit {len(fit_results)}/{n_rois} ROIs")

    # Find largest ROI
    sample_counts = counts_matrix.sum(axis=1)
    largest_idx = np.argmax(sample_counts)
    largest_histogram = counts_matrix[largest_idx]

    # Fit largest ROI (should already be in fit_results, but refit for clarity)
    largest_fit = fit_gev_to_histogram(
        bin_centers, bin_edges, largest_histogram, quantile_points
    )

    return DatasetResults(
        name=dataset_name,
        n_rois=n_rois,
        bin_centers=bin_centers,
        bin_edges=bin_edges,
        fit_results=fit_results,
        largest_roi_histogram=largest_histogram,
        largest_roi_fit=largest_fit,
        xi_values=np.array(xi_values),
        mu_values=np.array(mu_values),
        sigma_values=np.array(sigma_values)
    )


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_pdf_comparison(
    ax: plt.Axes,
    results: DatasetResults,
    panel_label: str,
    x_max: float = PLOT_XLIM_MAX
) -> None:
    """
    Plot empirical histogram vs fitted GEV PDF for the largest ROI.
    """
    font_settings = settings.fonts
    bin_width = np.diff(results.bin_edges).mean()

    # Empirical histogram as density
    total = results.largest_roi_histogram.sum()
    density = results.largest_roi_histogram / (total * bin_width)

    # Crop to x_max
    mask = results.bin_centers <= x_max
    x = results.bin_centers[mask]
    y_hist = density[mask]
    y_fit = results.largest_roi_fit.pdf_values[mask]

    # Plot histogram
    ax.bar(x, y_hist, width=bin_width * 0.9, alpha=0.6, color='steelblue',
           edgecolor='white', linewidth=0.5, label='Empirical')

    # Plot fitted GEV
    ax.plot(x, y_fit, color='#e41a1c', linewidth=2.5, label='GEV fit')

    ax.set_xlabel('Radius (μm)', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.set_ylabel('Probability Density', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.set_title(f'{panel_label} {results.name}: PDF Comparison',
                 fontsize=font_settings['label_size'], fontweight=font_settings['weight'])
    ax.legend(loc='upper right', fontsize=font_settings['legend_size'])
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Add fit info
    fit = results.largest_roi_fit
    info_text = f"ξ={fit.xi:.3f}, μ={fit.mu:.3f}, σ={fit.sigma:.3f}"
    ax.text(0.98, 0.75, info_text, transform=ax.transAxes, fontsize=9,
            ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def plot_qq_with_variability(
    ax: plt.Axes,
    results: DatasetResults,
    panel_label: str
) -> None:
    """
    Plot QQ-plot comparing fitted vs empirical quantiles with median ± IQR across ROIs.
    """
    font_settings = settings.fonts
    line_settings = settings.line

    n_fits = len(results.fit_results)
    if n_fits == 0:
        ax.text(0.5, 0.5, 'No valid fits', ha='center', va='center', transform=ax.transAxes)
        return

    # Collect quantiles from all ROIs
    q_empirical_all = np.array([r.quantiles_empirical for r in results.fit_results])
    q_fitted_all = np.array([r.quantiles_fitted for r in results.fit_results])

    # Compute median and IQR across ROIs
    q_emp_median = np.median(q_empirical_all, axis=0)
    q_fit_median = np.median(q_fitted_all, axis=0)
    q_fit_lo = np.percentile(q_fitted_all, 25, axis=0)
    q_fit_hi = np.percentile(q_fitted_all, 75, axis=0)

    # Filter out invalid values
    valid = np.isfinite(q_emp_median) & np.isfinite(q_fit_median)
    q_emp = q_emp_median[valid]
    q_fit = q_fit_median[valid]
    q_lo = q_fit_lo[valid]
    q_hi = q_fit_hi[valid]

    # Identity line
    all_vals = np.concatenate([q_emp, q_fit, q_lo, q_hi])
    min_val = np.nanmin(all_vals) * 0.95
    max_val = np.nanmax(all_vals) * 1.05
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5,
            linewidth=1.5, label='Identity', zorder=0)

    # QQ envelope and median
    ax.fill_between(q_emp, q_lo, q_hi, alpha=0.3, color='#e41a1c', label='IQR')
    ax.plot(q_emp, q_fit, color='#e41a1c', linewidth=line_settings['linewidth'],
            label='Median')

    ax.set_xlabel('Empirical Quantiles (μm)', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.set_ylabel('Fitted Quantiles (μm)', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.set_title(f'{panel_label} {results.name}: QQ-Plot',
                 fontsize=font_settings['label_size'], fontweight=font_settings['weight'])
    ax.legend(loc='upper left', fontsize=font_settings['legend_size'])
    ax.set_aspect('equal')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, alpha=0.3)

    # Add n_rois info
    ax.text(0.98, 0.02, f'n={n_fits} ROIs', transform=ax.transAxes, fontsize=9,
            ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def plot_parameter_ranges(
    ax: plt.Axes,
    results: DatasetResults,
    panel_label: str,
    xlim: Optional[Tuple[float, float]] = None
) -> None:
    """
    Plot GEV parameter ranges as horizontal bars with mean ± std.

    Args:
        ax: Matplotlib axes
        results: Dataset results containing parameter values
        panel_label: Panel label (e.g., '(c)')
        xlim: Optional (x_min, x_max) tuple for synchronized axes
    """
    font_settings = settings.fonts
    err_settings = settings.error_bars

    param_names = ['ξ (shape)', 'μ (location)', 'σ (scale)']
    param_values = [results.xi_values, results.mu_values, results.sigma_values]
    colors = ['#e41a1c', '#377eb8', '#4daf4a']  # Red, Blue, Green

    y_pos = np.arange(len(param_names))

    means = []
    stds = []
    for vals in param_values:
        valid = vals[np.isfinite(vals)]
        means.append(np.mean(valid) if len(valid) > 0 else 0)
        stds.append(np.std(valid) if len(valid) > 0 else 0)

    # Plot horizontal bars
    ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.8,
            edgecolor='black', linewidth=0.5,
            capsize=err_settings['capsize'],
            error_kw={'elinewidth': err_settings['linewidth'],
                      'capthick': err_settings['capthick']})

    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(mean + std + 0.02, i, f'{mean:.3f}±{std:.3f}',
                va='center', ha='left', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names, fontsize=font_settings['label_size'])
    ax.set_xlabel('Parameter Value', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.set_title(f'{panel_label} {results.name}: GEV Parameters',
                 fontsize=font_settings['label_size'], fontweight=font_settings['weight'])
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Set x-axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        x_max = max(m + s for m, s in zip(means, stds)) * 1.5
        ax.set_xlim(left=min(0, min(m - s for m, s in zip(means, stds)) * 1.2), right=x_max)


def create_figure(
    human_results: DatasetResults,
    rat_results: DatasetResults,
    output_file: Path
) -> None:
    """Create the 2×3 figure."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Compute shared x-limits for parameter panels
    all_params = []
    for results in [human_results, rat_results]:
        for vals in [results.xi_values, results.mu_values, results.sigma_values]:
            valid = vals[np.isfinite(vals)]
            if len(valid) > 0:
                all_params.append((np.mean(valid), np.std(valid)))

    x_min = min(m - s for m, s in all_params) * 1.2
    x_max = max(m + s for m, s in all_params) * 1.5
    param_xlim = (min(x_min, -0.1), x_max)  # Ensure negative values visible

    # Row 1: Human CC
    plot_pdf_comparison(axes[0, 0], human_results, '(a)')
    plot_qq_with_variability(axes[0, 1], human_results, '(b)')
    plot_parameter_ranges(axes[0, 2], human_results, '(c)', xlim=param_xlim)

    # Row 2: Rat WM
    plot_pdf_comparison(axes[1, 0], rat_results, '(d)')
    plot_qq_with_variability(axes[1, 1], rat_results, '(e)')
    plot_parameter_ranges(axes[1, 2], rat_results, '(f)', xlim=param_xlim)

    plt.tight_layout()
    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()
    logger.info(f"Saved figure to {output_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate GEV fit quality against empirical data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--human-data', type=Path, required=True,
                        help='Directory containing human CC TSV files')
    parser.add_argument('--rat-data', type=Path, required=True,
                        help='Directory containing rat NPZ files')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output PNG file path')
    parser.add_argument('--bin-width', type=float, default=DEFAULT_BIN_WIDTH,
                        help=f'Bin width in μm (default: {DEFAULT_BIN_WIDTH})')

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load and fit Human CC data
    logger.info("=" * 60)
    logger.info("Processing Human CC data...")
    human_edges, human_centers, human_counts, human_names = load_human_cc_histograms(
        args.human_data, args.bin_width
    )
    human_results = fit_all_rois(
        human_edges, human_centers, human_counts, human_names, "Human CC"
    )

    # Load and fit Rat data
    logger.info("=" * 60)
    logger.info("Processing Rat WM data...")
    rat_edges, rat_centers, rat_counts, rat_names = load_rat_histograms(
        args.rat_data, args.bin_width
    )
    rat_results = fit_all_rois(
        rat_edges, rat_centers, rat_counts, rat_names, "Rat WM"
    )

    # Report parameter statistics
    logger.info("=" * 60)
    logger.info("Human CC GEV parameters:")
    logger.info(f"  ξ: {np.mean(human_results.xi_values):.3f} ± {np.std(human_results.xi_values):.3f}")
    logger.info(f"  μ: {np.mean(human_results.mu_values):.3f} ± {np.std(human_results.mu_values):.3f}")
    logger.info(f"  σ: {np.mean(human_results.sigma_values):.3f} ± {np.std(human_results.sigma_values):.3f}")

    logger.info("Rat WM GEV parameters:")
    logger.info(f"  ξ: {np.mean(rat_results.xi_values):.3f} ± {np.std(rat_results.xi_values):.3f}")
    logger.info(f"  μ: {np.mean(rat_results.mu_values):.3f} ± {np.std(rat_results.mu_values):.3f}")
    logger.info(f"  σ: {np.mean(rat_results.sigma_values):.3f} ± {np.std(rat_results.sigma_values):.3f}")

    # Create figure
    logger.info("=" * 60)
    logger.info("Creating figure...")
    create_figure(human_results, rat_results, args.output)

    # Save JSON metadata
    json_file = args.output.with_suffix('.json')
    output_data = {
        'human_cc': {
            'n_rois': human_results.n_rois,
            'n_successful_fits': len(human_results.fit_results),
            'parameters': {
                'xi': {'mean': float(np.mean(human_results.xi_values)),
                       'std': float(np.std(human_results.xi_values))},
                'mu': {'mean': float(np.mean(human_results.mu_values)),
                       'std': float(np.std(human_results.mu_values))},
                'sigma': {'mean': float(np.mean(human_results.sigma_values)),
                          'std': float(np.std(human_results.sigma_values))}
            }
        },
        'rat_wm': {
            'n_rois': rat_results.n_rois,
            'n_successful_fits': len(rat_results.fit_results),
            'parameters': {
                'xi': {'mean': float(np.mean(rat_results.xi_values)),
                       'std': float(np.std(rat_results.xi_values))},
                'mu': {'mean': float(np.mean(rat_results.mu_values)),
                       'std': float(np.std(rat_results.mu_values))},
                'sigma': {'mean': float(np.mean(rat_results.sigma_values)),
                          'std': float(np.std(rat_results.sigma_values))}
            }
        }
    }
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Saved metadata to {json_file}")


if __name__ == '__main__':
    main()
