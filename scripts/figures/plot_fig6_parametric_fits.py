"""
Combined distribution fitting for Human CC and Rat white matter data.

Creates a 6-panel figure with 2 rows:
- Row 1: Pooled histograms with fitted PDFs (Human CC, Rat)
- Row 2: AIC comparison, Wasserstein distance, radius bias (r_arith, r_eff)

Usage:
    python scripts/figures/plot_fig6_parametric_fits.py \\
        --human-data data/raw/human/lm \\
        --rat-data data/processed/rat/lm \\
        --output fig/main/fig_6.svg
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy import stats

# Find repo root (contains pyproject.toml)
_root = Path(__file__).resolve().parent
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from axonometry import (compute_r_arith, compute_r_eff, get_plot_settings,
                        rediscretize)
from axonometry.distribution_fitting import (FitResult,
                                             compute_distribution_radii,
                                             fit_distribution_mle)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings = get_plot_settings()

# Constants
Y_SPACING = 1.3  # Vertical spacing between distributions in horizontal boxplots
PLOT_XLIM_MAX = 3.0
DEFAULT_BIN_WIDTH = 0.05  # um


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class HistogramData:
    """Container for histogram data."""
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    counts: np.ndarray
    n_samples: int  # number of ROIs
    total_count: int
    name: str = ""


@dataclass
class PerSampleHistogramData:
    """Container for per-ROI histogram data."""
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    counts_matrix: np.ndarray  # (n_samples, n_bins)
    n_samples: int
    sample_counts: np.ndarray  # total count per sample
    sample_names: List[str]


@dataclass
class AggregatedMetrics:
    """Aggregated metrics from per-sample fitting."""
    distribution_names: List[str]
    summed_aic: np.ndarray
    pooled_results: List[FitResult]
    # Per-sample arrays: shape (n_distributions, n_samples)
    all_r_arith: np.ndarray = field(default_factory=lambda: np.array([]))
    all_r_eff: np.ndarray = field(default_factory=lambda: np.array([]))
    all_wasserstein: np.ndarray = field(default_factory=lambda: np.array([]))
    # Per-sample empirical values: shape (n_samples,)
    empirical_r_arith_per_sample: np.ndarray = field(default_factory=lambda: np.array([]))
    empirical_r_eff_per_sample: np.ndarray = field(default_factory=lambda: np.array([]))
    # Distribution name to index mapping
    dist_name_to_idx: Dict[str, int] = field(default_factory=dict)
    # Win rate per distribution (fraction of samples where it has lowest AIC)
    win_rate: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Data Loading - Human CC (TSV histograms)
# =============================================================================


def load_human_cc_data(
    data_dir: Path,
    bin_width: float = DEFAULT_BIN_WIDTH,
) -> Tuple[HistogramData, PerSampleHistogramData]:
    """
    Load human corpus callosum histogram data (MinorAxis).

    Args:
        data_dir: Directory containing human LM histogram files
        bin_width: Target bin width in μm for rediscretization

    Returns:
        Tuple of (pooled HistogramData, per-ROI PerSampleHistogramData)
    """
    bin_edges_file = data_dir / 'desc-binEdges_radii.tsv'
    counts_file = data_dir / 'desc-countsMinorAxis_radii.tsv'

    # Load bin edges
    bin_edges_orig = np.loadtxt(bin_edges_file, delimiter='\t', skiprows=1)
    logger.info(f"Human CC: loaded {len(bin_edges_orig)} bin edges")

    # Load counts matrix (rows=ROIs, columns=bins)
    counts_matrix_orig = np.loadtxt(counts_file, delimiter='\t', skiprows=1, dtype=float)
    n_rois = counts_matrix_orig.shape[0]
    logger.info(f"Human CC: {n_rois} ROIs")

    # Rediscretize each ROI
    first_edges, first_centers, _ = rediscretize(
        bin_edges_orig, counts_matrix_orig[0], bin_width
    )
    n_bins = len(first_centers)
    counts_matrix = np.zeros((n_rois, n_bins), dtype=float)

    for i in range(n_rois):
        _, _, counts_matrix[i] = rediscretize(
            bin_edges_orig, counts_matrix_orig[i], bin_width
        )

    sample_counts = counts_matrix.sum(axis=1)
    pooled_counts = counts_matrix.sum(axis=0)
    total_count = int(pooled_counts.sum())

    logger.info(f"Human CC: {total_count:,} total axons, {n_bins} bins")

    pooled = HistogramData(
        bin_edges=first_edges,
        bin_centers=first_centers,
        counts=pooled_counts,
        n_samples=n_rois,
        total_count=total_count,
        name="Human CC"
    )

    per_sample = PerSampleHistogramData(
        bin_edges=first_edges,
        bin_centers=first_centers,
        counts_matrix=counts_matrix,
        n_samples=n_rois,
        sample_counts=sample_counts,
        sample_names=[f"ROI_{i+1}" for i in range(n_rois)]
    )

    return pooled, per_sample


# =============================================================================
# Data Loading - Rat (NPZ files)
# =============================================================================

def load_rat_data(
    data_dir: Path,
    bin_width: float = DEFAULT_BIN_WIDTH,
    r_max: float = 3.0,
    min_axons: int = 1000
) -> Tuple[HistogramData, PerSampleHistogramData]:
    """
    Load rat LM data from all NPZ files.

    Each ROI contributes one sample. Only includes ROIs with at least
    min_axons axons.

    Returns:
        Tuple of (pooled HistogramData, per-ROI PerSampleHistogramData)
    """
    npz_files = sorted(data_dir.glob('*_axon_profiles.npz'))
    if not npz_files:
        raise ValueError(f"No *_axon_profiles.npz files found in {data_dir}")

    n_rois = len(npz_files)
    logger.info(f"Rat: found {n_rois} NPZ files")

    # Determine bin structure
    bin_edges = np.arange(0, r_max + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_centers)

    all_counts = []
    all_names = []

    from axonometry.io import load_3d_profiles

    for npz_file in npz_files:
        roi_name = npz_file.stem.replace('_axon_profiles', '')
        data = load_3d_profiles(npz_file)
        radii = data['all_radii_um']
        n_axons = len(data['labels'])
        if n_axons < min_axons:
            logger.debug(f"Skipping {roi_name}: only {n_axons} axons (min: {min_axons})")
            continue
        counts, _ = np.histogram(radii, bins=bin_edges)
        all_counts.append(counts)
        all_names.append(roi_name)

    n_samples = len(all_counts)
    counts_matrix = np.array(all_counts, dtype=float)
    sample_counts = counts_matrix.sum(axis=1)
    pooled_counts = counts_matrix.sum(axis=0)
    total_count = int(pooled_counts.sum())

    logger.info(f"Rat: {n_samples} ROIs with ≥{min_axons} axons, {total_count:,} total radii, {n_bins} bins")

    pooled = HistogramData(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        counts=pooled_counts,
        n_samples=n_samples,
        total_count=total_count,
        name="Rat WM"
    )

    per_sample = PerSampleHistogramData(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        counts_matrix=counts_matrix,
        n_samples=n_samples,
        sample_counts=sample_counts,
        sample_names=all_names
    )

    return pooled, per_sample


# =============================================================================
# Distribution Catalog
# =============================================================================

# Distributions from Sepehrband et al. (2016) - top candidates
CANDIDATE_DISTRIBUTIONS = [
    ('genextreme', stats.genextreme),    # Generalized extreme value (GEV)
    ('invgauss', stats.invgauss),        # Inverse gaussian
    ('fatiguelife', stats.fatiguelife),  # Birnbaum-Saunders
    ('lognorm', stats.lognorm),          # Log normal
    ('fisk', stats.fisk),                # Log-logistic
    ('gamma', stats.gamma),              # Gamma
]

# Display names matching Sepehrband et al. (2016) - single line for legends
DIST_DISPLAY_NAMES = {
    'genextreme': 'Gen. Ext. Value',
    'lognorm': 'Log Normal',
    'invgauss': 'Inverse Gaussian',
    'fatiguelife': 'Birnbaum-Saunders',
    'fisk': 'Log-Logistic',
    'gamma': 'Gamma',
}

# Multi-line display names for y-axis labels in bottom row plots
DIST_DISPLAY_NAMES_MULTILINE = {
    'genextreme': 'Gen. Ext.\nValue',
    'lognorm': 'Log\nNormal',
    'invgauss': 'Inverse\nGaussian',
    'fatiguelife': 'Birnbaum-\nSaunders',
    'fisk': 'Log-\nLogistic',
    'gamma': 'Gamma',
}


def get_display_name(scipy_name: str, multiline: bool = False) -> str:
    """Get display name for a distribution."""
    if multiline:
        return DIST_DISPLAY_NAMES_MULTILINE.get(scipy_name, scipy_name)
    return DIST_DISPLAY_NAMES.get(scipy_name, scipy_name)


def compute_empirical_radii(
    bin_centers: np.ndarray, counts: np.ndarray
) -> Tuple[float, float]:
    """Compute empirical r_arith and r_eff from histogram."""
    return (
        compute_r_arith(counts=counts, bin_centers=bin_centers),
        compute_r_eff(counts=counts, bin_centers=bin_centers),
    )


def fit_all_distributions(hist_data: HistogramData) -> List[FitResult]:
    """Fit all candidate scipy distributions."""
    results = []

    for dist_name, dist in CANDIDATE_DISTRIBUTIONS:
        result = fit_distribution_mle(
            dist_name, dist,
            hist_data.bin_centers, hist_data.bin_edges, hist_data.counts
        )
        if result is not None:
            results.append(result)
            logger.debug(f"  {dist_name}: AIC={result.aic:.0f}")
        else:
            logger.debug(f"  {dist_name}: FAILED")

    results.sort(key=lambda x: x.aic)
    return results


# =============================================================================
# Per-Sample Fitting and Aggregation
# =============================================================================

MOMENT_EVAL_PERCENTILE = 0.99999  # p99.999 for truncated moment evaluation


def fit_all_samples(
    per_sample_data: PerSampleHistogramData,
    pooled_data: HistogramData
) -> AggregatedMetrics:
    """Fit all distributions to each sample and aggregate AIC, r_arith, r_eff."""
    n_samples = per_sample_data.n_samples

    all_method_names = [name for name, _ in CANDIDATE_DISTRIBUTIONS]
    n_methods = len(all_method_names)

    all_aics = np.full((n_methods, n_samples), np.nan)
    all_wasserstein = np.full((n_methods, n_samples), np.nan)
    all_r_arith = np.full((n_methods, n_samples), np.nan)
    all_r_eff = np.full((n_methods, n_samples), np.nan)

    # Empirical values per sample
    empirical_r_arith = np.full(n_samples, np.nan)
    empirical_r_eff = np.full(n_samples, np.nan)

    # Data-backed upper bound for moment evaluation (p99.999 of pooled data)
    cdf = np.cumsum(pooled_data.counts) / pooled_data.counts.sum()
    idx = min(np.searchsorted(cdf, MOMENT_EVAL_PERCENTILE), len(pooled_data.bin_centers) - 1)
    r_max_eval = pooled_data.bin_centers[idx]
    logger.info(f"Moment evaluation r_max = {r_max_eval:.2f} μm "
                f"(p{MOMENT_EVAL_PERCENTILE * 100:.3f} of pooled data)")

    logger.info(f"Fitting {n_methods} distributions to {n_samples} samples...")

    for sample_idx in range(n_samples):
        counts = per_sample_data.counts_matrix[sample_idx]
        total = counts.sum()
        if total < 100:
            continue

        # Compute empirical radii for this sample
        emp_r_arith, emp_r_eff = compute_empirical_radii(
            per_sample_data.bin_centers, counts
        )
        empirical_r_arith[sample_idx] = emp_r_arith
        empirical_r_eff[sample_idx] = emp_r_eff

        # Fit all scipy distributions
        for dist_idx, (dist_name, dist) in enumerate(CANDIDATE_DISTRIBUTIONS):
            result = fit_distribution_mle(
                dist_name, dist,
                per_sample_data.bin_centers, per_sample_data.bin_edges, counts
            )
            if result is not None:
                all_aics[dist_idx, sample_idx] = result.aic
                all_wasserstein[dist_idx, sample_idx] = result.wasserstein
                # Compute r_arith and r_eff from fitted distribution
                r_arith, r_eff = compute_distribution_radii(
                    dist, result.params, r_max=r_max_eval
                )
                all_r_arith[dist_idx, sample_idx] = r_arith
                all_r_eff[dist_idx, sample_idx] = r_eff

    # Aggregate AIC
    summed_aic = np.nansum(all_aics, axis=1)

    # Compute win rate: fraction of samples where each distribution has lowest AIC
    win_counts = np.zeros(n_methods)
    for sample_idx in range(n_samples):
        sample_aics = all_aics[:, sample_idx]
        if np.all(np.isnan(sample_aics)):
            continue
        winner_idx = np.nanargmin(sample_aics)
        win_counts[winner_idx] += 1
    n_valid_samples = np.sum(~np.all(np.isnan(all_aics), axis=0))
    win_rate = win_counts / n_valid_samples if n_valid_samples > 0 else win_counts

    # Sort by summed AIC
    sort_idx = np.argsort(summed_aic)
    sorted_names = [all_method_names[i] for i in sort_idx]

    # Fit pooled data for PDF plotting
    pooled_results = fit_all_distributions(pooled_data)

    # Create dist_name_to_idx mapping (original order, not sorted)
    dist_name_to_idx = {name: idx for idx, (name, _) in enumerate(CANDIDATE_DISTRIBUTIONS)}

    # Create win_rate dict
    win_rate_dict = {all_method_names[i]: win_rate[i] for i in range(n_methods)}

    return AggregatedMetrics(
        distribution_names=sorted_names,
        summed_aic=summed_aic[sort_idx],
        pooled_results=pooled_results,
        all_r_arith=all_r_arith,
        all_r_eff=all_r_eff,
        all_wasserstein=all_wasserstein,
        empirical_r_arith_per_sample=empirical_r_arith,
        empirical_r_eff_per_sample=empirical_r_eff,
        dist_name_to_idx=dist_name_to_idx,
        win_rate=win_rate_dict
    )


# =============================================================================
# Visualization
# =============================================================================

# Fixed colors per distribution (consistent across all panels)
# Distribution colors - avoid red/blue which are reserved for species
DIST_COLORS = {
    'genextreme': '#2ca02c',      # Green
    'lognorm': '#ff7f0e',         # Orange
    'invgauss': '#9467bd',        # Purple
    'fatiguelife': '#8c564b',     # Brown
    'fisk': '#e377c2',            # Pink
    'gamma': '#008080',           # Teal
}


def get_dist_color(dist_name: str) -> str:
    """Get fixed color for a distribution."""
    return DIST_COLORS.get(dist_name, '#333333')


def compute_inter_roi_wasserstein(per_sample: PerSampleHistogramData) -> float:
    """Compute median pairwise Wasserstein distance between ROIs.

    This gives a reference for biological variability between ROIs.
    """
    bin_centers = per_sample.bin_centers
    bin_width = np.diff(per_sample.bin_edges).mean()
    n_samples = per_sample.n_samples

    # Compute CDF for each ROI
    cdfs = []
    for i in range(n_samples):
        counts = per_sample.counts_matrix[i]
        total = counts.sum()
        if total > 100:
            cdf = np.cumsum(counts) / total
            cdfs.append(cdf)

    # Compute pairwise Wasserstein distances
    n_rois = len(cdfs)
    pairwise_distances = []
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            w_dist = np.sum(np.abs(cdfs[i] - cdfs[j])) * bin_width
            pairwise_distances.append(w_dist)

    return np.median(pairwise_distances) if pairwise_distances else 0.0


INSET_PERCENTILE_LO = 0.90   # inset shows from p90 to p99.9
INSET_PERCENTILE_HI = 0.999


def _plot_pooled_pdf_with_fits(
    ax: plt.Axes,
    hist_data: HistogramData,
    fit_results: List[FitResult],
    species_color: str,
    distribution_order: List[str] = None
) -> None:
    """Plot pooled histogram with all fitted PDFs in distribution colors."""
    bin_width = np.diff(hist_data.bin_edges).mean()
    density = hist_data.counts / (hist_data.total_count * bin_width)

    # Plot histogram in species color (no label - will use custom legend entry)
    ax.bar(hist_data.bin_centers, density, width=bin_width * 0.9,
           alpha=0.4, color=species_color, edgecolor='white', linewidth=0.5,
           zorder=1)

    # Sort fit_results according to distribution_order if provided
    if distribution_order is not None:
        # Create a mapping from name to result
        result_map = {r.distribution_name: r for r in fit_results}
        # Reorder according to distribution_order
        ordered_results = [result_map[name] for name in distribution_order if name in result_map]
    else:
        ordered_results = fit_results

    # Plot all fitted distributions in their colors (in specified order)
    for result in ordered_results:
        color = get_dist_color(result.distribution_name)
        display_name = get_display_name(result.distribution_name)
        ax.plot(result.pdf_x_fine, result.pdf_values_fine, '-',
                color=color, linewidth=3, label=display_name, zorder=2)

    ax.set_xlim(0, PLOT_XLIM_MAX)
    ax.set_ylim(0, None)
    ax.set_xlabel('Axon radius [μm]', fontsize=settings.fonts['label_size'])
    ax.set_ylabel(r'Probability density [μm$^{-1}$]', fontsize=settings.fonts['label_size'])
    # Title will be added externally
    ax.tick_params(labelsize=settings.fonts['tick_size'])

    # Add tail inset — percentile-based range (consistent across species)
    cdf = np.cumsum(hist_data.counts) / hist_data.total_count
    lo_idx = min(np.searchsorted(cdf, INSET_PERCENTILE_LO), len(hist_data.bin_centers) - 1)
    hi_idx = min(np.searchsorted(cdf, INSET_PERCENTILE_HI), len(hist_data.bin_centers) - 1)
    inset_lo = hist_data.bin_centers[lo_idx] - bin_width / 2
    inset_hi = hist_data.bin_centers[hi_idx]

    tail_mask = (hist_data.bin_centers >= inset_lo) & (hist_data.bin_centers <= inset_hi)
    tail_density = density[tail_mask]
    tail_y_max = tail_density.max() * 1.2 if len(tail_density) > 0 and tail_density.max() > 0 else 0.1

    ax_inset = ax.inset_axes([0.32, 0.32, 0.66, 0.66])
    ax_inset.bar(hist_data.bin_centers, density, width=bin_width * 0.9,
                 alpha=0.4, color=species_color, edgecolor='white', linewidth=0.3)
    for result in ordered_results:
        color = get_dist_color(result.distribution_name)
        ax_inset.plot(result.pdf_x_fine, result.pdf_values_fine, '-',
                      color=color, linewidth=3)
    ax_inset.set_xlim(inset_lo, inset_hi)
    ax_inset.set_ylim(0, tail_y_max)
    ax_inset.tick_params(labelsize=settings.fonts['tick_size'] - 2)
    ax_inset.set_xlabel('')
    ax_inset.set_ylabel('')

    ax.indicate_inset_zoom(ax_inset, edgecolor='gray', linewidth=1.5,
                           linestyle='--', alpha=0.8)


def create_combined_figure(
    human_pooled: HistogramData,
    human_metrics: AggregatedMetrics,
    rat_pooled: HistogramData,
    rat_metrics: AggregatedMetrics,
    human_per_sample: PerSampleHistogramData,
    rat_per_sample: PerSampleHistogramData,
    output_file: Path,
) -> None:
    """Create 6-panel figure with 2 rows.

    Row 1 (top): Pooled PDFs with all fitted distributions
        (a) Human pooled PDF with fits
        (b) Rat pooled PDF with fits

    Row 2 (bottom): Model comparison metrics
        (c) Win rate
        (d) Wasserstein distance (with inter-ROI reference)
        (e) r_arith error
        (f) r_eff error
    """
    from matplotlib.gridspec import GridSpec

    # Species colors from settings
    HUMAN_COLOR = settings.colors['human']
    RAT_COLOR = settings.colors['rat']

    # Create figure with GridSpec: 2 rows, top row has 2 panels, bottom has 4
    fig = plt.figure(figsize=(17, 9.5))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1], hspace=0.35, wspace=0.35)

    # Top row: 2 panels spanning 2 columns each
    ax_a = fig.add_subplot(gs[0, 0:2])  # Rat PDF
    ax_b = fig.add_subplot(gs[0, 2:4])  # Human PDF

    # Bottom row: 4 panels
    ax_c = fig.add_subplot(gs[1, 0])  # Win rate
    ax_d = fig.add_subplot(gs[1, 1])  # Wasserstein
    ax_e = fig.add_subplot(gs[1, 2])  # r_arith error
    ax_f = fig.add_subplot(gs[1, 3])  # r_eff error

    # Set aspect ratio for bottom row panels
    for ax in [ax_c, ax_d, ax_e, ax_f]:
        ax.set_box_aspect(1 / 0.75)

    # Compute inter-ROI Wasserstein for reference lines
    human_inter_roi_w = compute_inter_roi_wasserstein(human_per_sample)
    rat_inter_roi_w = compute_inter_roi_wasserstein(rat_per_sample)

    # Use human distribution order (by AIC) for consistency with bottom row
    dist_order = human_metrics.distribution_names

    # (a) Rat pooled PDF with all fits
    _plot_pooled_pdf_with_fits(
        ax_a, rat_pooled, rat_metrics.pooled_results,
        species_color=RAT_COLOR,
        distribution_order=dist_order
    )

    # (b) Human pooled PDF with all fits
    _plot_pooled_pdf_with_fits(
        ax_b, human_pooled, human_metrics.pooled_results,
        species_color=HUMAN_COLOR,
        distribution_order=dist_order
    )

    # Create shared legend above panels a-b
    # Get handles and labels from ax_a (distribution fits only, no data)
    handles, labels = ax_a.get_legend_handles_labels()

    # Create custom half-blue/half-red patch for empirical data
    class SplitColorHandler(HandlerBase):
        """Custom handler for split-color rectangle."""
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            # Create two rectangles side by side (Rat first, then Human)
            half_width = width / 2
            left_rect = Rectangle(
                (xdescent, ydescent), half_width, height,
                facecolor=RAT_COLOR, edgecolor='white', linewidth=0.5,
                alpha=0.6, transform=trans
            )
            right_rect = Rectangle(
                (xdescent + half_width, ydescent), half_width, height,
                facecolor=HUMAN_COLOR, edgecolor='white', linewidth=0.5,
                alpha=0.6, transform=trans
            )
            return [left_rect, right_rect]

    # Create dummy handle for empirical data
    empirical_patch = Patch(facecolor='gray')  # placeholder, handler will override

    # Prepend empirical data to handles/labels
    all_handles = [empirical_patch] + handles
    all_labels = ['Empirical data'] + labels

    # Create legend with frame
    fig.legend(
        all_handles, all_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(all_handles),
        fontsize=settings.fonts['legend_size'],
        frameon=True,
        edgecolor='gray',
        fancybox=True,
        columnspacing=1.5,
        handler_map={empirical_patch: SplitColorHandler()}
    )

    # (c) Win rate
    _plot_win_rate(
        ax_c, human_metrics, rat_metrics,
        human_color=HUMAN_COLOR, rat_color=RAT_COLOR
    )

    # (d) Wasserstein distance with inter-ROI reference
    _plot_wasserstein_both_species(
        ax_d, human_metrics, rat_metrics,
        human_color=HUMAN_COLOR, rat_color=RAT_COLOR,
        human_inter_roi=human_inter_roi_w, rat_inter_roi=rat_inter_roi_w
    )

    # (e) r_arith error - both species
    _plot_radius_bias_both_species(
        ax_e, human_metrics, rat_metrics,
        radius_type='r_arith',
        human_color=HUMAN_COLOR, rat_color=RAT_COLOR
    )

    # (f) r_eff error - both species
    _plot_radius_bias_both_species(
        ax_f, human_metrics, rat_metrics,
        radius_type='r_eff',
        human_color=HUMAN_COLOR, rat_color=RAT_COLOR
    )

    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()
    logger.info(f"Saved figure to {output_file}")


def _plot_win_rate(
    ax: plt.Axes,
    human_metrics: AggregatedMetrics,
    rat_metrics: AggregatedMetrics,
    human_color: str,
    rat_color: str
) -> None:
    """Plot win rate dot plot with both species on same axes."""
    # Get distribution names (use human order)
    names = human_metrics.distribution_names
    display_names = [get_display_name(n, multiline=True) for n in names]

    # Get win rates for each species (as percentages)
    human_win = np.array([human_metrics.win_rate.get(n, 0) for n in names]) * 100
    rat_win = np.array([rat_metrics.win_rate.get(n, 0) for n in names]) * 100

    y_spacing = Y_SPACING
    y_pos = np.arange(len(names)) * y_spacing

    # Plot dots (rat first so human is on top)
    ax.scatter(rat_win, y_pos, color=rat_color, s=120, marker='s',
               label='Rat', zorder=3, edgecolor='white', linewidth=0.5)
    ax.scatter(human_win, y_pos, color=human_color, s=120, marker='d',
               label='Human', zorder=4, edgecolor='white', linewidth=0.5)

    # Connect dots with lines
    for i in range(len(names)):
        ax.plot([human_win[i], rat_win[i]], [y_pos[i], y_pos[i]],
                color='gray', linewidth=1, alpha=0.5, zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=settings.fonts['tick_size'])
    ax.set_xlabel('Win rate [%]', fontsize=settings.fonts['label_size'])
    ax.set_xlim(-5, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(y_pos[-1] + 0.5, -1.5)  # Inverted, with extra padding at top
    ax.legend(loc='upper right', fontsize=settings.fonts['legend_size'], ncol=2,
              frameon=True, edgecolor='gray', facecolor='white', framealpha=1.0,
              handletextpad=0.3, borderpad=0.3,
              columnspacing=0.8, handlelength=1.2)
    ax.tick_params(labelsize=settings.fonts['tick_size'])


def _plot_wasserstein_both_species(
    ax: plt.Axes,
    human_metrics: AggregatedMetrics,
    rat_metrics: AggregatedMetrics,
    human_color: str,
    rat_color: str,
    human_inter_roi: float = None,
    rat_inter_roi: float = None
) -> None:
    """Plot Wasserstein distance boxplots for both species on same axes.

    Optionally shows inter-ROI Wasserstein as vertical dashed reference lines.
    """
    # Use human distribution order (sorted by AIC)
    names = human_metrics.distribution_names
    display_names = [get_display_name(n, multiline=True) for n in names]

    y_spacing = Y_SPACING
    y_pos = np.arange(len(names)) * y_spacing
    box_width = 0.4

    # Collect Wasserstein data for each distribution
    box_data_human = []
    box_data_rat = []

    for dist_name in names:
        dist_idx = human_metrics.dist_name_to_idx[dist_name]

        # Human Wasserstein values
        human_w = human_metrics.all_wasserstein[dist_idx]
        human_w_valid = human_w[~np.isnan(human_w)]
        box_data_human.append(human_w_valid)

        # Rat Wasserstein values
        rat_dist_idx = rat_metrics.dist_name_to_idx.get(dist_name, -1)
        if rat_dist_idx >= 0:
            rat_w = rat_metrics.all_wasserstein[rat_dist_idx]
            rat_w_valid = rat_w[~np.isnan(rat_w)]
            box_data_rat.append(rat_w_valid)
        else:
            box_data_rat.append(np.array([]))

    # Plot boxplots for human (offset down)
    human_positions = [y_pos[i] - box_width/2 for i in range(len(box_data_human))]
    bp_human = ax.boxplot([d if len(d) > 0 else [np.nan] for d in box_data_human],
                          positions=human_positions, vert=False, widths=box_width * 0.8,
                          patch_artist=True, showfliers=False,
                          medianprops=dict(color=human_color, linewidth=2),
                          whiskerprops=dict(color=human_color, linewidth=1.5),
                          capprops=dict(color=human_color, linewidth=1.5))
    for patch in bp_human['boxes']:
        patch.set_facecolor(human_color)
        patch.set_alpha(0.5)
        patch.set_edgecolor(human_color)

    # Plot boxplots for rat (offset up)
    rat_positions = [y_pos[i] + box_width/2 for i in range(len(box_data_rat))]
    bp_rat = ax.boxplot([d if len(d) > 0 else [np.nan] for d in box_data_rat],
                        positions=rat_positions, vert=False, widths=box_width * 0.8,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color=rat_color, linewidth=2),
                        whiskerprops=dict(color=rat_color, linewidth=1.5),
                        capprops=dict(color=rat_color, linewidth=1.5))
    for patch in bp_rat['boxes']:
        patch.set_facecolor(rat_color)
        patch.set_alpha(0.5)
        patch.set_edgecolor(rat_color)

    # Add inter-ROI reference lines (vertical dashed)
    if human_inter_roi is not None:
        ax.axvline(human_inter_roi, color=human_color, linestyle='--', linewidth=2,
                   alpha=0.8, zorder=1)
    if rat_inter_roi is not None:
        ax.axvline(rat_inter_roi, color=rat_color, linestyle='--', linewidth=2,
                   alpha=0.8, zorder=1)

    # Add legend manually (Rat, Human on first row; Anat. Var. on second row)
    # Custom handler for solid colored rectangle
    class SolidPatchHandler(HandlerBase):
        def __init__(self, color):
            self.color = color
            super().__init__()

        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            rect = Rectangle((xdescent, ydescent), width, height,
                            facecolor=self.color, edgecolor=self.color,
                            linewidth=1, alpha=1.0, transform=trans)
            return [rect]

    # Custom handler for dual colored dashed lines (stacked vertically)
    class DualDashedLineHandler(HandlerBase):
        def __init__(self, color1, color2):
            self.color1 = color1
            self.color2 = color2
            super().__init__()

        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            # Draw two dashed lines stacked on top of each other
            y_offset = height * 0.2
            y_top = ydescent + height / 2 + y_offset
            y_bottom = ydescent + height / 2 - y_offset
            # First line (rat color) on top
            line1 = Line2D([xdescent, xdescent + width],
                          [y_top, y_top],
                          color=self.color1, linestyle='--', linewidth=1.5,
                          transform=trans)
            # Second line (human color) on bottom
            line2 = Line2D([xdescent, xdescent + width],
                          [y_bottom, y_bottom],
                          color=self.color2, linestyle='--', linewidth=1.5,
                          transform=trans)
            return [line1, line2]

    # With ncol=2, matplotlib fills column-wise: (0,0), (1,0), (0,1), (1,1)...
    # To get Row1: [Rat, Human], Row2: [Anat. Var.], order must be: [Rat, Anat. Var., Human]
    rat_handle = Patch(facecolor=rat_color, label='Rat')
    human_handle = Patch(facecolor=human_color, label='Human')

    if human_inter_roi is not None or rat_inter_roi is not None:
        # Placeholder handle for Anat. Var. (handler will draw the actual symbol)
        anat_var_handle = Line2D([0], [0], color='none')
        # Single row: Rat, Human, Anat. Var.
        legend_elements = [rat_handle, human_handle, anat_var_handle]
        labels = ['Rat', 'Human', 'Anat. Var.']
        handler_map = {
            rat_handle: SolidPatchHandler(rat_color),
            human_handle: SolidPatchHandler(human_color),
            anat_var_handle: DualDashedLineHandler(rat_color, human_color)
        }
        ncol = 3
    else:
        legend_elements = [rat_handle, human_handle]
        labels = ['Rat', 'Human']
        handler_map = {
            rat_handle: SolidPatchHandler(rat_color),
            human_handle: SolidPatchHandler(human_color)
        }
        ncol = 2

    ax.legend(handles=legend_elements, labels=labels,
              loc='upper right', fontsize=settings.fonts['legend_size'], ncol=ncol,
              frameon=True, edgecolor='gray', facecolor='white', framealpha=1.0,
              handletextpad=0.2, borderpad=0.2,
              columnspacing=0.5, handlelength=1.0,
              bbox_to_anchor=(1.02, 1.0),
              handler_map=handler_map)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=settings.fonts['tick_size'])
    ax.set_xlabel('Wasserstein distance [μm]', fontsize=settings.fonts['label_size'])
    ax.set_xlim(0, 0.08)
    ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08])
    ax.set_ylim(y_pos[-1] + 0.5, -1.5)  # Inverted, with extra padding at top
    ax.tick_params(labelsize=settings.fonts['tick_size'])


def _plot_radius_bias_both_species(
    ax: plt.Axes,
    human_metrics: AggregatedMetrics,
    rat_metrics: AggregatedMetrics,
    radius_type: str,
    human_color: str,
    rat_color: str
) -> None:
    """Plot radius bias for both species as violins showing spread.

    Args:
        ax: Matplotlib axes
        human_metrics: Human CC aggregated metrics
        rat_metrics: Rat WM aggregated metrics
        radius_type: 'r_arith' or 'r_eff'
        human_color: Color for human data points
        rat_color: Color for rat data points
    """
    # Use human distribution order (sorted by AIC)
    names = human_metrics.distribution_names
    display_names = [get_display_name(n, multiline=True) for n in names]

    y_spacing = Y_SPACING
    y_pos = np.arange(len(names)) * y_spacing

    # Get per-sample values for each species
    if radius_type == 'r_arith':
        human_all = human_metrics.all_r_arith  # (n_dist, n_samples)
        rat_all = rat_metrics.all_r_arith
        human_emp_per_sample = human_metrics.empirical_r_arith_per_sample
        rat_emp_per_sample = rat_metrics.empirical_r_arith_per_sample
        xlabel = r'$\bar{r}$ error [%]'
        x_lim = 5  # ±5%
    else:  # r_eff
        human_all = human_metrics.all_r_eff
        rat_all = rat_metrics.all_r_eff
        human_emp_per_sample = human_metrics.empirical_r_eff_per_sample
        rat_emp_per_sample = rat_metrics.empirical_r_eff_per_sample
        xlabel = r'$r_{\mathrm{MRI}}$ error [%]'
        x_lim = 100  # ±100%

    # Compute per-sample bias for each distribution
    box_width = 0.4
    box_data_human = []
    box_data_rat = []

    for dist_name in names:
        dist_idx = human_metrics.dist_name_to_idx[dist_name]

        # Human: bias = (fitted - empirical) / empirical * 100 for each sample
        human_fitted = human_all[dist_idx]
        human_bias = (human_fitted - human_emp_per_sample) / human_emp_per_sample * 100
        human_bias_valid = human_bias[~np.isnan(human_bias)]
        box_data_human.append(human_bias_valid)

        # Rat
        rat_dist_idx = rat_metrics.dist_name_to_idx.get(dist_name, -1)
        if rat_dist_idx >= 0:
            rat_fitted = rat_all[rat_dist_idx]
            rat_bias = (rat_fitted - rat_emp_per_sample) / rat_emp_per_sample * 100
            rat_bias_valid = rat_bias[~np.isnan(rat_bias)]
            box_data_rat.append(rat_bias_valid)
        else:
            box_data_rat.append(np.array([]))

    # Plot boxplots for human (offset down)
    human_positions = [y_pos[i] - box_width/2 for i in range(len(box_data_human))]
    bp_human = ax.boxplot([d if len(d) > 0 else [np.nan] for d in box_data_human],
                          positions=human_positions, vert=False, widths=box_width * 0.8,
                          patch_artist=True, showfliers=False,
                          medianprops=dict(color=human_color, linewidth=2),
                          whiskerprops=dict(color=human_color, linewidth=1.5),
                          capprops=dict(color=human_color, linewidth=1.5))
    for patch in bp_human['boxes']:
        patch.set_facecolor(human_color)
        patch.set_alpha(0.5)
        patch.set_edgecolor(human_color)

    # Plot boxplots for rat (offset up)
    rat_positions = [y_pos[i] + box_width/2 for i in range(len(box_data_rat))]
    bp_rat = ax.boxplot([d if len(d) > 0 else [np.nan] for d in box_data_rat],
                        positions=rat_positions, vert=False, widths=box_width * 0.8,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color=rat_color, linewidth=2),
                        whiskerprops=dict(color=rat_color, linewidth=1.5),
                        capprops=dict(color=rat_color, linewidth=1.5))
    for patch in bp_rat['boxes']:
        patch.set_facecolor(rat_color)
        patch.set_alpha(0.5)
        patch.set_edgecolor(rat_color)

    # Add zero reference line
    ax.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.7)

    # Annotate out-of-bounds medians with arrows and values
    annot_fontsize = settings.fonts['tick_size']  # Larger, more visible
    for i, (h_data, r_data) in enumerate(zip(box_data_human, box_data_rat)):
        # Human
        if len(h_data) > 0:
            h_median = np.median(h_data)
            if h_median > x_lim:
                ax.annotate(f'{h_median:.0f}%', xy=(x_lim, y_pos[i] - box_width/2),
                            xytext=(x_lim - 8, y_pos[i] - box_width/2),
                            fontsize=annot_fontsize, fontweight='bold',
                            color=human_color, ha='right', va='center',
                            arrowprops=dict(arrowstyle='->', color=human_color, lw=1.5))
            elif h_median < -x_lim:
                ax.annotate(f'{h_median:.0f}%', xy=(-x_lim, y_pos[i] - box_width/2),
                            xytext=(-x_lim + 8, y_pos[i] - box_width/2),
                            fontsize=annot_fontsize, fontweight='bold',
                            color=human_color, ha='left', va='center',
                            arrowprops=dict(arrowstyle='->', color=human_color, lw=1.5))
        # Rat
        if len(r_data) > 0:
            r_median = np.median(r_data)
            if r_median > x_lim:
                ax.annotate(f'{r_median:.0f}%', xy=(x_lim, y_pos[i] + box_width/2),
                            xytext=(x_lim - 8, y_pos[i] + box_width/2),
                            fontsize=annot_fontsize, fontweight='bold',
                            color=rat_color, ha='right', va='center',
                            arrowprops=dict(arrowstyle='->', color=rat_color, lw=1.5))
            elif r_median < -x_lim:
                ax.annotate(f'{r_median:.0f}%', xy=(-x_lim, y_pos[i] + box_width/2),
                            xytext=(-x_lim + 8, y_pos[i] + box_width/2),
                            fontsize=annot_fontsize, fontweight='bold',
                            color=rat_color, ha='left', va='center',
                            arrowprops=dict(arrowstyle='->', color=rat_color, lw=1.5))

    # Add legend with boxplot-style handles (Rat first)
    class BoxplotHandler(HandlerBase):
        """Custom handler that draws a mini boxplot with whisker caps."""
        def __init__(self, facecolor, edgecolor):
            self.facecolor = facecolor
            self.edgecolor = edgecolor
            super().__init__()

        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            # Box (IQR) - centered, takes middle 50% of width
            box_left = xdescent + width * 0.25
            box_width = width * 0.5
            box = Rectangle((box_left, ydescent + height * 0.15),
                           box_width, height * 0.7,
                           facecolor=self.facecolor, edgecolor=self.edgecolor,
                           linewidth=1, alpha=1.0, transform=trans)
            # Median line (inside box)
            median = Line2D([box_left, box_left + box_width],
                          [ydescent + height * 0.5, ydescent + height * 0.5],
                          color='white', linewidth=1.5, transform=trans)
            # Left whisker (horizontal line)
            left_whisker = Line2D([xdescent, box_left],
                                 [ydescent + height * 0.5, ydescent + height * 0.5],
                                 color=self.edgecolor, linewidth=1, transform=trans)
            # Right whisker (horizontal line)
            right_whisker = Line2D([box_left + box_width, xdescent + width],
                                  [ydescent + height * 0.5, ydescent + height * 0.5],
                                  color=self.edgecolor, linewidth=1, transform=trans)
            # Left whisker cap (vertical line)
            left_cap = Line2D([xdescent, xdescent],
                             [ydescent + height * 0.25, ydescent + height * 0.75],
                             color=self.edgecolor, linewidth=1, transform=trans)
            # Right whisker cap (vertical line)
            right_cap = Line2D([xdescent + width, xdescent + width],
                              [ydescent + height * 0.25, ydescent + height * 0.75],
                              color=self.edgecolor, linewidth=1, transform=trans)
            return [box, median, left_whisker, right_whisker, left_cap, right_cap]

    # Create dummy patches for legend (alpha=1.0 for solid)
    rat_patch = Patch(facecolor=rat_color, label='Rat', alpha=1.0)
    human_patch = Patch(facecolor=human_color, label='Human', alpha=1.0)

    ax.legend(handles=[rat_patch, human_patch], loc='upper right',
              fontsize=settings.fonts['legend_size'], ncol=2,
              frameon=True, edgecolor='gray', facecolor='white', framealpha=1.0,
              handletextpad=0.3, borderpad=0.3,
              columnspacing=0.8, handlelength=1.8,
              handler_map={rat_patch: BoxplotHandler(rat_color, rat_color),
                          human_patch: BoxplotHandler(human_color, human_color)})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=settings.fonts['tick_size'])
    ax.set_xlabel(xlabel, fontsize=settings.fonts['label_size'])
    # Symmetric range around 0
    ax.set_xlim(-x_lim, x_lim)
    # Add intermediate ticks
    if radius_type == 'r_arith':
        ax.set_xticks([-4, -2, 0, 2, 4])
    else:  # r_eff
        ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_ylim(y_pos[-1] + 0.5, -1.5)  # Inverted, with extra padding at top
    ax.tick_params(labelsize=settings.fonts['tick_size'])


def main():
    parser = argparse.ArgumentParser(
        description='Combined distribution fitting for Human CC and Rat data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--human-data', type=Path, default=Path('data/raw/human/lm'),
                        help='Directory containing human CC TSV files (default: data/raw/human/lm)')
    parser.add_argument('--rat-data', type=Path, default=Path('data/processed/rat/lm'),
                        help='Directory containing rat NPZ files (default: data/processed/rat/lm)')
    parser.add_argument('--output', type=Path, default=Path('fig/main/fig_6.svg'),
                        help='Output file path (default: fig/main/fig_6.svg)')
    parser.add_argument('--bin-width', type=float, default=DEFAULT_BIN_WIDTH,
                        help=f'Bin width in um (default: {DEFAULT_BIN_WIDTH})')
    parser.add_argument('--r-max', type=float, default=3.0,
                        help='Maximum radius in um (default: 3.0)')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load Human CC data
    logger.info("=" * 60)
    logger.info("Loading Human CC data...")
    human_pooled, human_per_sample = load_human_cc_data(
        args.human_data,
        args.bin_width,
    )

    # Load Rat data
    logger.info("=" * 60)
    logger.info("Loading Rat data...")
    rat_pooled, rat_per_sample = load_rat_data(args.rat_data, args.bin_width, args.r_max)

    # Fit per-sample and aggregate AIC
    logger.info("=" * 60)
    logger.info("Fitting Human CC distributions...")
    human_metrics = fit_all_samples(human_per_sample, human_pooled)

    logger.info("=" * 60)
    logger.info("Fitting Rat distributions...")
    rat_metrics = fit_all_samples(rat_per_sample, rat_pooled)

    # Report results
    logger.info("=" * 60)
    logger.info("Human CC - Top 5 by summed AIC:")
    for i, name in enumerate(human_metrics.distribution_names[:5], 1):
        delta = human_metrics.summed_aic[i-1] - human_metrics.summed_aic[0]
        logger.info(f"  {i}. {name}: Delta AIC = {delta:.0f}")

    logger.info("Rat WM - Top 5 by summed AIC:")
    for i, name in enumerate(rat_metrics.distribution_names[:5], 1):
        delta = rat_metrics.summed_aic[i-1] - rat_metrics.summed_aic[0]
        logger.info(f"  {i}. {name}: Delta AIC = {delta:.0f}")

    # Create main figure (4-panel)
    logger.info("=" * 60)
    logger.info("Creating summary figure...")
    # Slightly smaller fonts for the main figure only; restored afterwards so
    # the per-distribution supplement keeps the global sizes.
    _orig_fonts = dict(settings.fonts)
    settings._settings['fonts'] = {
        **_orig_fonts,
        'label_size': _orig_fonts['label_size'] - 2,
        'tick_size': _orig_fonts['tick_size'] - 2,
        'legend_size': _orig_fonts['legend_size'] - 2,
    }
    try:
        create_combined_figure(
            human_pooled, human_metrics,
            rat_pooled, rat_metrics,
            human_per_sample, rat_per_sample,
            args.output
        )
    finally:
        settings._settings['fonts'] = _orig_fonts

    # Save JSON with full metrics
    json_file = args.output.with_suffix('.json')

    def _metrics_to_dict(metrics, pooled_data):
        """Convert AggregatedMetrics to a JSON-serializable dict."""
        dists = []
        for i, name in enumerate(metrics.distribution_names):
            idx = metrics.dist_name_to_idx[name]
            r_arith_bias = metrics.all_r_arith[idx] - metrics.empirical_r_arith_per_sample
            r_eff_bias = metrics.all_r_eff[idx] - metrics.empirical_r_eff_per_sample
            # Relative bias (%)
            with np.errstate(divide='ignore', invalid='ignore'):
                r_arith_rel = r_arith_bias / metrics.empirical_r_arith_per_sample * 100
                r_eff_rel = r_eff_bias / metrics.empirical_r_eff_per_sample * 100
            r_arith_valid = r_arith_rel[np.isfinite(r_arith_rel)]
            r_eff_valid = r_eff_rel[np.isfinite(r_eff_rel)]
            # Pooled fit params
            pooled_result = next(
                (r for r in metrics.pooled_results if r.distribution_name == name), None
            )
            dists.append({
                'name': name,
                'summed_aic': float(metrics.summed_aic[i]),
                'delta_aic': float(metrics.summed_aic[i] - metrics.summed_aic[0]),
                'win_rate': float(metrics.win_rate.get(name, 0)),
                'pooled_params': [float(p) for p in pooled_result.params] if pooled_result else None,
                'pooled_nll': float(pooled_result.nll) if pooled_result else None,
                'r_arith_bias_pct': {
                    'mean': float(np.mean(r_arith_valid)) if len(r_arith_valid) else None,
                    'median': float(np.median(r_arith_valid)) if len(r_arith_valid) else None,
                    'std': float(np.std(r_arith_valid)) if len(r_arith_valid) else None,
                },
                'r_eff_bias_pct': {
                    'mean': float(np.mean(r_eff_valid)) if len(r_eff_valid) else None,
                    'median': float(np.median(r_eff_valid)) if len(r_eff_valid) else None,
                    'std': float(np.std(r_eff_valid)) if len(r_eff_valid) else None,
                },
                'wasserstein': {
                    'mean': float(np.nanmean(metrics.all_wasserstein[idx])),
                    'median': float(np.nanmedian(metrics.all_wasserstein[idx])),
                },
            })
        return {
            'n_rois': pooled_data.n_samples,
            'total_count': pooled_data.total_count,
            'distributions': dists,
        }

    output_data = {
        'human_cc': _metrics_to_dict(human_metrics, human_pooled),
        'rat': _metrics_to_dict(rat_metrics, rat_pooled),
    }
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Saved results to {json_file}")


if __name__ == '__main__':
    main()
