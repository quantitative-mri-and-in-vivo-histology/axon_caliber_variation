#!/usr/bin/env python3
"""
Study the effect of sample size on radius estimation accuracy.

Creates a 2×2 figure:
  (a) PDFs for Human CC and Rat WM pooled distributions
  (b) CDFs for Human CC and Rat WM pooled distributions
  (c) Arithmetic mean radius error vs sample size (both datasets)
  (d) Effective MRI radius error vs sample size (both datasets)

Sample sizes: 10², 10³, 10⁴, 10⁵, 10⁶
Subsamples: N=50 (configurable)

Usage:
    python scripts/exploratory/distribution_fitting/plot_sample_size_effect.py \
        --human-data data/raw_LM \
        --rat-data data/processed/LM \
        --output fig/sample_size_effect.png
"""

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import OptimizeWarning

# Find repo root (contains pyproject.toml)
_root = Path(__file__).resolve().parent
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from axonometry import get_plot_settings, add_panel_labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings = get_plot_settings()

# Constants
DEFAULT_BIN_WIDTH = 0.05  # μm
# Sample sizes for bias plots (end at 10^5)
SAMPLE_SIZES = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
N_SUBSAMPLES = 50
N_QUANTILES = 50
MIN_BIN_PROB = 1e-300

# Colors for different sample sizes (YlOrBr colormap for QQ plots)
from matplotlib.cm import YlOrBr
_ylorbr_colors = [YlOrBr(x) for x in [0.3, 0.65, 1.0]]
SAMPLE_SIZE_COLORS = {
    100: _ylorbr_colors[0],        # Light yellow
    1_000: '#a6cee3',              # Light blue (not used in QQ)
    10_000: _ylorbr_colors[1],     # Orange
    100_000: '#b2df8a',            # Light green (not used in QQ)
    1_000_000: _ylorbr_colors[2],  # Dark brown
}

SAMPLE_SIZE_LABELS = {
    100: r'$10^2$',
    1_000: r'$10^3$',
    10_000: r'$10^4$',
    100_000: r'$10^5$',
    1_000_000: r'$10^6$',
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SubsampleResults:
    """Results from subsampling analysis for one sample size."""
    sample_size: int
    n_valid_rois: int
    # Quantiles: (n_rois, n_subsamples, n_quantiles)
    raw_quantiles: np.ndarray
    gev_quantiles: np.ndarray
    # Reference quantiles per ROI: (n_rois, n_quantiles)
    reference_quantiles: np.ndarray
    # Radius estimates: (n_rois, n_subsamples)
    r_arith: np.ndarray
    r_eff: np.ndarray
    # Reference values per ROI: (n_rois,)
    reference_r_arith: np.ndarray
    reference_r_eff: np.ndarray


@dataclass
class DatasetSubsampleResults:
    """All subsampling results for a dataset."""
    name: str
    results_by_size: Dict[int, SubsampleResults]
    quantile_points: np.ndarray


# =============================================================================
# Data Loading (reused from previous scripts)
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
    """Load human corpus callosum histogram data (MinorAxis radii)."""
    bin_edges_file = data_dir / 'desc-binEdges_radii.tsv'
    counts_file = data_dir / 'desc-countsMinorAxis_radii.tsv'

    bin_edges_orig = np.loadtxt(bin_edges_file, delimiter='\t', skiprows=1)
    counts_matrix_orig = np.loadtxt(counts_file, delimiter='\t', skiprows=1, dtype=float)
    n_rois = counts_matrix_orig.shape[0]

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
    r_max: float = 3.0,
    min_axons: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load rat LM data from NPZ files, split by population.

    Only includes ROIs with at least min_axons axons.
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
                n_axons = pop.get('n_axons', len(pop['axon_labels']))

                # Skip ROIs with too few axons
                if n_axons < min_axons:
                    logger.debug(f"Skipping {volume_name}_{pop_name}: only {n_axons} axons (min: {min_axons})")
                    continue

                pop_labels = set(pop['axon_labels'])

                mask = np.isin(labels, list(pop_labels))
                if mask.sum() > 0:
                    pop_radii = np.concatenate([radii_profiles[i] for i in np.where(mask)[0]])
                    counts, _ = np.histogram(pop_radii, bins=bin_edges)
                    all_counts.append(counts)
                    all_names.append(f"{volume_name}_{pop_name}")
        else:
            radii = data['all_radii_w_branches_um']
            n_axons = len(data['labels'])
            if n_axons < min_axons:
                logger.debug(f"Skipping {volume_name}: only {n_axons} axons (min: {min_axons})")
                continue
            counts, _ = np.histogram(radii, bins=bin_edges)
            all_counts.append(counts)
            all_names.append(volume_name)

    counts_matrix = np.array(all_counts, dtype=float)
    logger.info(f"Rat WM: {len(all_names)} ROIs with ≥{min_axons} axons, {int(counts_matrix.sum()):,} total radii")

    return bin_edges, bin_centers, counts_matrix, all_names


# =============================================================================
# Subsampling and Analysis
# =============================================================================

def histogram_to_quantiles(counts: np.ndarray, bin_centers: np.ndarray,
                           quantile_points: np.ndarray) -> np.ndarray:
    """Compute quantiles from histogram."""
    total = counts.sum()
    if total == 0:
        return np.full(len(quantile_points), np.nan)
    cdf = np.cumsum(counts) / total
    return np.interp(quantile_points, cdf, bin_centers)


def compute_r_arith_r_eff(counts: np.ndarray, bin_centers: np.ndarray) -> Tuple[float, float]:
    """Compute r_arith and r_eff from histogram."""
    total = counts.sum()
    if total == 0:
        return np.nan, np.nan

    probs = counts / total
    r_arith = np.sum(bin_centers * probs)
    r2 = np.sum(bin_centers**2 * probs)
    r6 = np.sum(bin_centers**6 * probs)
    r_eff = (r6 / r2) ** 0.25 if r2 > 0 else np.nan

    return r_arith, r_eff


def fit_gev_and_get_quantiles(
    counts: np.ndarray,
    bin_centers: np.ndarray,
    bin_edges: np.ndarray,
    quantile_points: np.ndarray
) -> np.ndarray:
    """Fit GEV to histogram and return quantiles."""
    total = counts.sum()
    if total < 100:
        return np.full(len(quantile_points), np.nan)

    n_samples = min(5000, int(total))
    probs = counts / total
    probs = probs / probs.sum()
    bin_width = np.diff(bin_edges).mean()

    try:
        sampled_bins = np.random.choice(len(bin_centers), size=n_samples, p=probs)
        jitter = np.random.uniform(-bin_width/2, bin_width/2, n_samples)
        samples = bin_centers[sampled_bins] + jitter
        samples = samples[samples > 0]

        if len(samples) < 100:
            return np.full(len(quantile_points), np.nan)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', category=OptimizeWarning)
            params = stats.genextreme.fit(samples)
            if params[1] < 0:
                params = stats.genextreme.fit(samples, floc=0)

        xi, mu, sigma = params[0], params[1], params[2]
        return stats.genextreme.ppf(quantile_points, xi, loc=mu, scale=sigma)

    except (ValueError, RuntimeError):
        return np.full(len(quantile_points), np.nan)


def subsample_histogram(
    counts: np.ndarray,
    bin_centers: np.ndarray,
    bin_edges: np.ndarray,
    sample_size: int
) -> np.ndarray:
    """Generate a subsample histogram from a full histogram."""
    total = counts.sum()
    if total < sample_size:
        return None

    probs = counts / total
    bin_width = np.diff(bin_edges).mean()

    # Sample bin indices
    sampled_bins = np.random.choice(len(bin_centers), size=sample_size, p=probs)

    # Create histogram from samples
    sub_counts = np.bincount(sampled_bins, minlength=len(bin_centers))
    return sub_counts.astype(float)


def run_subsampling_analysis(
    bin_edges: np.ndarray,
    bin_centers: np.ndarray,
    counts_matrix: np.ndarray,
    sample_sizes: List[int],
    n_subsamples: int,
    quantile_points: np.ndarray,
    dataset_name: str
) -> DatasetSubsampleResults:
    """Run subsampling analysis for all sample sizes."""
    n_rois = counts_matrix.shape[0]
    n_quantiles = len(quantile_points)

    results_by_size = {}

    for sample_size in sample_sizes:
        logger.info(f"  Processing sample size {sample_size:,}...")

        # Find ROIs with enough axons
        roi_counts = counts_matrix.sum(axis=1)
        valid_roi_mask = roi_counts >= sample_size
        valid_roi_indices = np.where(valid_roi_mask)[0]
        n_valid = len(valid_roi_indices)

        if n_valid == 0:
            logger.warning(f"    No ROIs have >= {sample_size:,} axons, skipping")
            continue

        logger.info(f"    {n_valid}/{n_rois} ROIs have >= {sample_size:,} axons")

        # Initialize arrays
        raw_quantiles = np.full((n_valid, n_subsamples, n_quantiles), np.nan)
        gev_quantiles = np.full((n_valid, n_subsamples, n_quantiles), np.nan)
        r_arith = np.full((n_valid, n_subsamples), np.nan)
        r_eff = np.full((n_valid, n_subsamples), np.nan)
        reference_quantiles = np.full((n_valid, n_quantiles), np.nan)
        reference_r_arith = np.full(n_valid, np.nan)
        reference_r_eff = np.full(n_valid, np.nan)

        for i, roi_idx in enumerate(valid_roi_indices):
            counts = counts_matrix[roi_idx]

            # Reference values from whole ROI
            reference_quantiles[i] = histogram_to_quantiles(counts, bin_centers, quantile_points)
            reference_r_arith[i], reference_r_eff[i] = compute_r_arith_r_eff(counts, bin_centers)

            # Subsampling
            for j in range(n_subsamples):
                sub_counts = subsample_histogram(counts, bin_centers, bin_edges, sample_size)
                if sub_counts is None:
                    continue

                # Raw quantiles
                raw_quantiles[i, j] = histogram_to_quantiles(sub_counts, bin_centers, quantile_points)

                # GEV-fitted quantiles
                gev_quantiles[i, j] = fit_gev_and_get_quantiles(
                    sub_counts, bin_centers, bin_edges, quantile_points
                )

                # Radius estimates
                r_arith[i, j], r_eff[i, j] = compute_r_arith_r_eff(sub_counts, bin_centers)

        results_by_size[sample_size] = SubsampleResults(
            sample_size=sample_size,
            n_valid_rois=n_valid,
            raw_quantiles=raw_quantiles,
            gev_quantiles=gev_quantiles,
            reference_quantiles=reference_quantiles,
            r_arith=r_arith,
            r_eff=r_eff,
            reference_r_arith=reference_r_arith,
            reference_r_eff=reference_r_eff
        )

    return DatasetSubsampleResults(
        name=dataset_name,
        results_by_size=results_by_size,
        quantile_points=quantile_points
    )


# =============================================================================
# Plotting Functions
# =============================================================================

# Font size reduction for this figure (smaller than default)
FONT_REDUCTION = 2


def plot_pdf_combined(
    ax: plt.Axes,
    human_bin_centers: np.ndarray,
    human_pooled_counts: np.ndarray,
    rat_bin_centers: np.ndarray,
    rat_pooled_counts: np.ndarray,
    sample_sizes: List[int],
    n_subsamples: int,
    x_max: float = 2.0
) -> None:
    """Plot PDFs for both datasets with subsampling variability."""
    font_settings = settings.fonts
    label_size = font_settings['label_size'] - FONT_REDUCTION
    tick_size = font_settings['tick_size'] - FONT_REDUCTION
    legend_size = font_settings['legend_size'] - FONT_REDUCTION

    # Species colors
    human_color = settings.colors['human']
    rat_color = settings.colors['rat']

    # Common x-axis for PDF evaluation
    x_eval = np.linspace(0.02, x_max, 200)

    datasets = [
        (rat_bin_centers, rat_pooled_counts, rat_color, 'Rat'),
        (human_bin_centers, human_pooled_counts, human_color, 'Human'),
    ]

    # Line styles for different sample sizes: dashed for subsamples
    line_styles = ['--']
    # Alphas for shaded areas
    fill_alphas = [0.25]

    n_repeats = 25000  # Repeat 25k times for good 95% CI

    # Store legend handles
    legend_handles = []

    for bin_centers, pooled_counts, color, species_label in datasets:
        total_count = int(pooled_counts.sum())
        bin_width = bin_centers[1] - bin_centers[0]
        probs = pooled_counts / total_count

        # Reference PDF (full distribution)
        pdf_ref = pooled_counts / (total_count * bin_width)
        pdf_ref_interp = np.interp(x_eval, bin_centers, pdf_ref, left=0, right=0)

        # Plot subsamples first (so full sample is on top)
        # Sort sample sizes so smaller ones are plotted first (behind)
        sorted_sizes = sorted([s for s in sample_sizes if s <= total_count])

        for idx, sample_size in enumerate(sorted_sizes):
            pdf_subsamples = []
            for _ in range(n_repeats):
                sampled_bins = np.random.choice(len(bin_centers), size=sample_size, p=probs)
                sub_counts = np.bincount(sampled_bins, minlength=len(bin_centers)).astype(float)
                pdf_sub = sub_counts / (sub_counts.sum() * bin_width)
                pdf_sub_interp = np.interp(x_eval, bin_centers, pdf_sub, left=0, right=0)
                pdf_subsamples.append(pdf_sub_interp)

            pdf_subsamples = np.array(pdf_subsamples)
            pdf_lo = np.percentile(pdf_subsamples, 2.5, axis=0)
            pdf_hi = np.percentile(pdf_subsamples, 97.5, axis=0)

            # Smooth only the CI bands
            sigma = max(1, 3 - idx)
            pdf_lo = gaussian_filter1d(pdf_lo, sigma=sigma)
            pdf_hi = gaussian_filter1d(pdf_hi, sigma=sigma)

            # Shaded area
            ax.fill_between(x_eval, pdf_lo, pdf_hi, alpha=fill_alphas[idx], color=color,
                           zorder=1 + idx)

            # Boundary lines with different styles
            linestyle = line_styles[idx] if idx < len(line_styles) else '-'
            ax.plot(x_eval, pdf_lo, color=color, linestyle=linestyle, linewidth=0.8,
                   alpha=0.7, zorder=2 + idx)
            ax.plot(x_eval, pdf_hi, color=color, linestyle=linestyle, linewidth=0.8,
                   alpha=0.7, zorder=2 + idx)

        # Plot full distribution on top (solid line)
        line, = ax.plot(x_eval, pdf_ref_interp, color=color, linewidth=1.5,
                        linestyle='-', zorder=10)

    # Build legend manually
    from matplotlib.lines import Line2D

    def format_sample_size(n: int) -> str:
        """Format sample size as 10^x notation."""
        import math
        exp = math.log10(n)
        if exp == int(exp):
            return rf'$10^{int(exp)}$'
        else:
            # For non-powers of 10, find closest representation
            exp_floor = int(math.floor(exp))
            mantissa = n / (10 ** exp_floor)
            if abs(mantissa - round(mantissa)) < 0.01:
                mantissa = int(round(mantissa))
            return rf'$\sim 10^{exp_floor + 1}$' if mantissa >= 5 else rf'$\sim 10^{exp_floor}$'

    import math
    from matplotlib.patches import Patch, FancyBboxPatch
    import matplotlib.patches as mpatches

    for bin_centers, pooled_counts, color, species_label in datasets:
        total_count = int(pooled_counts.sum())
        # Full sample - format as mantissa × 10^exp (e.g., 5×10^7)
        exp = int(math.floor(math.log10(total_count)))
        mantissa = total_count / (10 ** exp)
        if mantissa >= 9.5:
            exp += 1
            mantissa = 1
        mantissa_rounded = int(round(mantissa))
        if mantissa_rounded == 1:
            full_label = rf'{species_label} ($n \approx 10^{exp}$)'
        else:
            full_label = rf'{species_label} ($n \approx {mantissa_rounded}\times 10^{exp}$)'
        legend_handles.append(Line2D([0], [0], color=color, linewidth=1.5, linestyle='-',
                                      label=full_label))
        # Subsamples - show as shaded patch with dashed edge
        sorted_sizes = sorted([s for s in sample_sizes if s <= total_count])
        for idx, sample_size in enumerate(sorted_sizes):
            linestyle = line_styles[idx] if idx < len(line_styles) else '-'
            alpha = fill_alphas[idx] if idx < len(fill_alphas) else 0.25
            exp_sub = int(math.log10(sample_size))
            # Create a patch with dashed edge to represent shaded area
            # Convert color to RGBA with alpha for facecolor, keep edge fully opaque
            import matplotlib.colors as mcolors
            face_rgba = list(mcolors.to_rgba(color))
            face_rgba[3] = alpha
            legend_handles.append(mpatches.Patch(
                facecolor=face_rgba, edgecolor=color,
                linestyle=linestyle, linewidth=1.5,
                label=rf'{species_label} ($n = 10^{exp_sub}$)'))

    ax.set_xlabel('Axon radius [μm]', fontsize=label_size)
    ax.set_ylabel('Probability density [μm⁻¹]', fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    ax.legend(handles=legend_handles, loc='upper right', fontsize=legend_size)
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_box_aspect(1)


def compute_wasserstein_from_cdfs(cdf1: np.ndarray, cdf2: np.ndarray,
                                   bin_width: float) -> float:
    """Compute Wasserstein distance between two CDFs."""
    return np.sum(np.abs(cdf1 - cdf2)) * bin_width


def plot_within_vs_between_wasserstein(
    ax: plt.Axes,
    human_bin_centers: np.ndarray,
    human_counts_matrix: np.ndarray,
    rat_bin_centers: np.ndarray,
    rat_counts_matrix: np.ndarray,
    sample_sizes: List[int] = [100, 1000, 10000],
    n_subsamples: int = 500
) -> None:
    """
    Plot within-sample Wasserstein distances with inter-ROI medians as reference.

    X-axis: sample sizes
    Grouped violins: Human (blue) and Rat (red) for each sample size
    Reference: Median inter-ROI distances shown as horizontal dashed lines
    """
    font_settings = settings.fonts
    label_size = font_settings['label_size'] - FONT_REDUCTION
    tick_size = font_settings['tick_size'] - FONT_REDUCTION
    legend_size = font_settings['legend_size'] - FONT_REDUCTION

    # Species colors
    human_color = settings.colors['human']
    rat_color = settings.colors['rat']

    datasets = [
        (rat_bin_centers, rat_counts_matrix, 'Rat', rat_color),
        (human_bin_centers, human_counts_matrix, 'Human', human_color),
    ]

    # Compute between-ROI distances per species
    between_median_per_species = {}
    for bin_centers, counts_matrix, name, color in datasets:
        bin_width = bin_centers[1] - bin_centers[0]
        n_rois = counts_matrix.shape[0]

        # Compute CDF for each ROI
        roi_cdfs = []
        for i in range(n_rois):
            roi_total = counts_matrix[i].sum()
            if roi_total > 0:
                roi_cdf = np.cumsum(counts_matrix[i]) / roi_total
                roi_cdfs.append(roi_cdf)

        # Pairwise Wasserstein distances within this species
        species_between = []
        for i in range(len(roi_cdfs)):
            for j in range(i + 1, len(roi_cdfs)):
                w_dist = compute_wasserstein_from_cdfs(roi_cdfs[i], roi_cdfs[j], bin_width)
                species_between.append(w_dist)
        between_median_per_species[name] = np.median(species_between)

    # Compute within-sample distances for each sample size and dataset
    np.random.seed(42)
    width = 0.3
    n_sample_sizes = len(sample_sizes)
    x_positions = np.arange(n_sample_sizes)

    # Collect all data for violin plots
    all_violin_data = []
    all_positions = []
    all_colors = []

    for dataset_idx, (bin_centers, counts_matrix, name, color) in enumerate(datasets):
        bin_width = bin_centers[1] - bin_centers[0]

        # Compute pooled distribution
        pooled_counts = counts_matrix.sum(axis=0)
        pooled_total = pooled_counts.sum()
        pooled_cdf = np.cumsum(pooled_counts) / pooled_total
        pooled_probs = pooled_counts / pooled_total

        for size_idx, sample_size in enumerate(sample_sizes):
            within_distances = []
            for _ in range(n_subsamples):
                sampled_bins = np.random.choice(len(bin_centers), size=sample_size, p=pooled_probs)
                sub_counts = np.bincount(sampled_bins, minlength=len(bin_centers)).astype(float)
                sub_cdf = np.cumsum(sub_counts) / sub_counts.sum()
                w_dist = compute_wasserstein_from_cdfs(sub_cdf, pooled_cdf, bin_width)
                within_distances.append(w_dist)

            all_violin_data.append(within_distances)
            # Offset for grouped violins
            offset = -width/2 if dataset_idx == 0 else width/2
            all_positions.append(size_idx + offset)
            all_colors.append(color)

    # Set x-axis limits
    ax.set_xlim(-0.5, n_sample_sizes - 0.5)

    # Plot thin violins
    parts = ax.violinplot(all_violin_data, positions=all_positions,
                           showmeans=False, showmedians=True, widths=width * 0.8)

    # Color each violin
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(all_colors[i])
        pc.set_alpha(0.7)
        pc.set_zorder(2)

    # Style median lines
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)
    parts['cbars'].set_color('gray')
    parts['cbars'].set_linewidth(0.5)
    parts['cmins'].set_color('gray')
    parts['cmaxes'].set_color('gray')

    # Draw horizontal dashed lines for inter-ROI medians
    ax.axhline(between_median_per_species['Human'], color=human_color,
               linestyle='--', linewidth=2, label='Human inter-ROI', zorder=1)
    ax.axhline(between_median_per_species['Rat'], color=rat_color,
               linestyle='--', linewidth=2, label='Rat inter-ROI', zorder=1)

    # Add legend manually
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor=rat_color, alpha=0.7, label='Rat (sampling)'),
        Patch(facecolor=human_color, alpha=0.7, label='Human (sampling)'),
        Line2D([0], [0], color=rat_color, linestyle='--', linewidth=2, label='Rat (anat. var.)'),
        Line2D([0], [0], color=human_color, linestyle='--', linewidth=2, label='Human (anat. var.)'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=legend_size)

    # X-axis: sample sizes only
    ax.set_xticks(x_positions)
    ax.set_xticklabels([SAMPLE_SIZE_LABELS[s] for s in sample_sizes], fontsize=tick_size)
    ax.set_xlabel('Sample size', fontsize=label_size)
    ax.set_ylabel('Wasserstein distance [μm]', fontsize=label_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    ax.set_ylim(bottom=0)
    # Set y-ticks in 0.05 steps
    y_max = ax.get_ylim()[1]
    ax.set_yticks(np.arange(0, y_max + 0.025, 0.05))

    ax.set_box_aspect(1)


def plot_qq_raw(ax: plt.Axes, dataset_results: DatasetSubsampleResults,
                sample_sizes: List[int] = None) -> None:
    """Plot QQ-plot comparing raw subsample quantiles to whole section."""
    font_settings = settings.fonts

    if sample_sizes is None:
        sample_sizes = SAMPLE_SIZES

    for sample_size in sample_sizes:
        if sample_size not in dataset_results.results_by_size:
            continue

        results = dataset_results.results_by_size[sample_size]
        color = SAMPLE_SIZE_COLORS[sample_size]
        label = SAMPLE_SIZE_LABELS[sample_size]

        # Flatten across ROIs and subsamples for each quantile point
        # raw_quantiles: (n_rois, n_subsamples, n_quantiles)
        # reference_quantiles: (n_rois, n_quantiles)

        n_rois, n_subs, n_q = results.raw_quantiles.shape

        # For each quantile, compute deviation from reference across all ROI×subsample
        # Then compute median and IQR
        q_ref_median = np.median(results.reference_quantiles, axis=0)  # (n_quantiles,)

        # Expand reference to match subsamples: (n_rois, 1, n_quantiles) -> broadcast
        ref_expanded = results.reference_quantiles[:, np.newaxis, :]  # (n_rois, 1, n_quantiles)

        # Compute subsample quantiles relative to their own ROI reference
        # Reshape to (n_rois * n_subs, n_quantiles)
        raw_flat = results.raw_quantiles.reshape(-1, n_q)
        ref_flat = np.tile(results.reference_quantiles, (n_subs, 1))  # repeat for each subsample

        # Compute median and IQR across all ROI×subsample combinations
        sub_median = np.nanmedian(raw_flat, axis=0)
        sub_lo = np.nanpercentile(raw_flat, 25, axis=0)
        sub_hi = np.nanpercentile(raw_flat, 75, axis=0)

        # Plot against reference median
        ax.fill_between(q_ref_median, sub_lo, sub_hi, alpha=0.2, color=color)
        ax.plot(q_ref_median, sub_median, color=color, linewidth=1.5, label=label)

    # Identity line
    all_q = []
    for results in dataset_results.results_by_size.values():
        all_q.extend(results.reference_quantiles.flatten())
    if all_q:
        q_min, q_max = np.nanmin(all_q) * 0.95, np.nanmax(all_q) * 1.05
        ax.plot([q_min, q_max], [q_min, q_max], 'k--', alpha=0.5, linewidth=1, zorder=0)
        ax.set_xlim(q_min, q_max)
        ax.set_ylim(q_min, q_max)

    ax.set_xlabel('Reference quantiles [μm]', fontsize=font_settings['label_size'])
    ax.set_ylabel('Subsample quantiles [μm]', fontsize=font_settings['label_size'])
    ax.tick_params(labelsize=font_settings['tick_size'])
    ax.legend(loc='upper left', fontsize=font_settings['legend_size'])
    ax.set_aspect('equal')


def plot_qq_gev(ax: plt.Axes, dataset_results: DatasetSubsampleResults) -> None:
    """Plot QQ-plot comparing GEV-fitted subsample quantiles to whole section."""
    font_settings = settings.fonts

    for sample_size in SAMPLE_SIZES:
        if sample_size not in dataset_results.results_by_size:
            continue

        results = dataset_results.results_by_size[sample_size]
        color = SAMPLE_SIZE_COLORS[sample_size]
        label = SAMPLE_SIZE_LABELS[sample_size]

        n_rois, n_subs, n_q = results.gev_quantiles.shape
        q_ref_median = np.median(results.reference_quantiles, axis=0)

        gev_flat = results.gev_quantiles.reshape(-1, n_q)

        sub_median = np.nanmedian(gev_flat, axis=0)
        sub_lo = np.nanpercentile(gev_flat, 25, axis=0)
        sub_hi = np.nanpercentile(gev_flat, 75, axis=0)

        ax.fill_between(q_ref_median, sub_lo, sub_hi, alpha=0.2, color=color)
        ax.plot(q_ref_median, sub_median, color=color, linewidth=1.5, label=label)

    # Identity line
    all_q = []
    for results in dataset_results.results_by_size.values():
        all_q.extend(results.reference_quantiles.flatten())
    if all_q:
        q_min, q_max = np.nanmin(all_q) * 0.95, np.nanmax(all_q) * 1.05
        ax.plot([q_min, q_max], [q_min, q_max], 'k--', alpha=0.5, linewidth=1, zorder=0)
        ax.set_xlim(q_min, q_max)
        ax.set_ylim(q_min, q_max)

    ax.set_xlabel('Reference quantiles [μm]', fontsize=font_settings['label_size'])
    ax.set_ylabel('GEV-fitted quantiles [μm]', fontsize=font_settings['label_size'])
    ax.tick_params(labelsize=font_settings['tick_size'])
    ax.legend(loc='upper left', fontsize=font_settings['legend_size'])
    ax.set_aspect('equal')


def plot_bias(ax: plt.Axes, dataset_results: DatasetSubsampleResults, panel_label: str) -> None:
    """Plot relative bias in r_arith and r_eff vs sample size."""
    font_settings = settings.fonts
    err_settings = settings.error_bars

    sample_sizes_present = [s for s in SAMPLE_SIZES if s in dataset_results.results_by_size]
    x_positions = np.arange(len(sample_sizes_present))
    width = 0.35

    r_arith_bias_median = []
    r_arith_bias_iqr = []
    r_eff_bias_median = []
    r_eff_bias_iqr = []

    for sample_size in sample_sizes_present:
        results = dataset_results.results_by_size[sample_size]

        # Compute relative bias per ROI (median across subsamples), then aggregate across ROIs
        # r_arith: (n_rois, n_subsamples), reference_r_arith: (n_rois,)

        # Per-ROI bias: for each subsample, compute (sub - ref) / ref
        ref_arith = results.reference_r_arith[:, np.newaxis]  # (n_rois, 1)
        ref_eff = results.reference_r_eff[:, np.newaxis]

        bias_arith = (results.r_arith - ref_arith) / ref_arith * 100  # (n_rois, n_subs)
        bias_eff = (results.r_eff - ref_eff) / ref_eff * 100

        # Median bias per ROI across subsamples
        roi_bias_arith = np.nanmedian(bias_arith, axis=1)  # (n_rois,)
        roi_bias_eff = np.nanmedian(bias_eff, axis=1)

        # Aggregate across ROIs
        r_arith_bias_median.append(np.nanmedian(roi_bias_arith))
        r_arith_bias_iqr.append([
            np.nanmedian(roi_bias_arith) - np.nanpercentile(roi_bias_arith, 25),
            np.nanpercentile(roi_bias_arith, 75) - np.nanmedian(roi_bias_arith)
        ])

        r_eff_bias_median.append(np.nanmedian(roi_bias_eff))
        r_eff_bias_iqr.append([
            np.nanmedian(roi_bias_eff) - np.nanpercentile(roi_bias_eff, 25),
            np.nanpercentile(roi_bias_eff, 75) - np.nanmedian(roi_bias_eff)
        ])

    r_arith_bias_iqr = np.array(r_arith_bias_iqr).T
    r_eff_bias_iqr = np.array(r_eff_bias_iqr).T

    # Plot bars (binary comparison: teal vs coral)
    bars1 = ax.bar(x_positions - width/2, r_arith_bias_median, width,
                   yerr=r_arith_bias_iqr, label=r'$r_{arith}$',
                   color=settings.colors['category_a'], alpha=0.8,
                   capsize=err_settings['capsize'],
                   error_kw={'elinewidth': err_settings['linewidth']})

    bars2 = ax.bar(x_positions + width/2, r_eff_bias_median, width,
                   yerr=r_eff_bias_iqr, label=r'$r_{eff}$',
                   color=settings.colors['category_b'], alpha=0.8,
                   capsize=err_settings['capsize'],
                   error_kw={'elinewidth': err_settings['linewidth']})

    ax.axhline(0, color='black', linestyle='-', linewidth=1)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([SAMPLE_SIZE_LABELS[s] for s in sample_sizes_present])
    ax.set_xlabel('Sample Size', fontsize=font_settings['label_size'])
    ax.set_ylabel('Relative Bias (%)', fontsize=font_settings['label_size'])
    ax.set_title(f'{panel_label} {dataset_results.name}: Bias',
                 fontsize=font_settings['label_size'], fontweight='bold')
    ax.legend(loc='upper right', fontsize=font_settings['legend_size'])
    ax.grid(True, alpha=0.3, axis='y')


def plot_cov(ax: plt.Axes, dataset_results: DatasetSubsampleResults, panel_label: str) -> None:
    """Plot coefficient of variation in r_arith and r_eff vs sample size."""
    font_settings = settings.fonts
    err_settings = settings.error_bars

    sample_sizes_present = [s for s in SAMPLE_SIZES if s in dataset_results.results_by_size]
    x_positions = np.arange(len(sample_sizes_present))
    width = 0.35

    r_arith_cov_median = []
    r_arith_cov_iqr = []
    r_eff_cov_median = []
    r_eff_cov_iqr = []

    for sample_size in sample_sizes_present:
        results = dataset_results.results_by_size[sample_size]

        # CoV per ROI: std / mean across subsamples
        # r_arith: (n_rois, n_subsamples)

        cov_arith_per_roi = np.nanstd(results.r_arith, axis=1) / np.nanmean(results.r_arith, axis=1) * 100
        cov_eff_per_roi = np.nanstd(results.r_eff, axis=1) / np.nanmean(results.r_eff, axis=1) * 100

        # Aggregate across ROIs
        r_arith_cov_median.append(np.nanmedian(cov_arith_per_roi))
        r_arith_cov_iqr.append([
            np.nanmedian(cov_arith_per_roi) - np.nanpercentile(cov_arith_per_roi, 25),
            np.nanpercentile(cov_arith_per_roi, 75) - np.nanmedian(cov_arith_per_roi)
        ])

        r_eff_cov_median.append(np.nanmedian(cov_eff_per_roi))
        r_eff_cov_iqr.append([
            np.nanmedian(cov_eff_per_roi) - np.nanpercentile(cov_eff_per_roi, 25),
            np.nanpercentile(cov_eff_per_roi, 75) - np.nanmedian(cov_eff_per_roi)
        ])

    r_arith_cov_iqr = np.array(r_arith_cov_iqr).T
    r_eff_cov_iqr = np.array(r_eff_cov_iqr).T

    # Plot bars (binary comparison: teal vs coral)
    bars1 = ax.bar(x_positions - width/2, r_arith_cov_median, width,
                   yerr=r_arith_cov_iqr, label=r'$r_{arith}$',
                   color=settings.colors['category_a'], alpha=0.8,
                   capsize=err_settings['capsize'],
                   error_kw={'elinewidth': err_settings['linewidth']})

    bars2 = ax.bar(x_positions + width/2, r_eff_cov_median, width,
                   yerr=r_eff_cov_iqr, label=r'$r_{eff}$',
                   color=settings.colors['category_b'], alpha=0.8,
                   capsize=err_settings['capsize'],
                   error_kw={'elinewidth': err_settings['linewidth']})

    ax.set_xticks(x_positions)
    ax.set_xticklabels([SAMPLE_SIZE_LABELS[s] for s in sample_sizes_present])
    ax.set_xlabel('Sample Size', fontsize=font_settings['label_size'])
    ax.set_ylabel('CoV (%)', fontsize=font_settings['label_size'])
    ax.set_title(f'{panel_label} {dataset_results.name}: CoV',
                 fontsize=font_settings['label_size'], fontweight='bold')
    ax.legend(loc='upper right', fontsize=font_settings['legend_size'])
    ax.grid(True, alpha=0.3, axis='y')


def plot_combined_bias(
    ax: plt.Axes,
    human_results: DatasetSubsampleResults,
    rat_results: DatasetSubsampleResults,
    metric: str,  # 'r_arith' or 'r_eff'
    ylabel: str
) -> None:
    """Plot percentage error vs sample size for both datasets."""
    font_settings = settings.fonts
    label_size = font_settings['label_size'] - FONT_REDUCTION
    tick_size = font_settings['tick_size'] - FONT_REDUCTION
    legend_size = font_settings['legend_size'] - FONT_REDUCTION

    # Species colors
    human_color = settings.colors['human']
    rat_color = settings.colors['rat']

    datasets = [
        (rat_results, 'Rat', rat_color, 's'),      # Squares on bottom
        (human_results, 'Human', human_color, 'o'),  # Circles on top
    ]

    for dataset_results, label, color, marker in datasets:
        # Use all sample sizes
        sample_sizes_present = [s for s in SAMPLE_SIZES if s in dataset_results.results_by_size]
        if not sample_sizes_present:
            continue

        bias_median = []
        bias_lo = []
        bias_hi = []

        for sample_size in sample_sizes_present:
            results = dataset_results.results_by_size[sample_size]

            if metric == 'r_arith':
                ref = results.reference_r_arith[:, np.newaxis]
                values = results.r_arith
            else:  # r_eff
                ref = results.reference_r_eff[:, np.newaxis]
                values = results.r_eff

            # Compute relative bias: (sub - ref) / ref * 100
            bias = (values - ref) / ref * 100  # (n_rois, n_subs)

            # Median bias per ROI across subsamples, then aggregate
            roi_bias = np.nanmedian(bias, axis=1)  # (n_rois,)

            bias_median.append(np.nanmedian(roi_bias))
            bias_lo.append(np.nanpercentile(roi_bias, 25))
            bias_hi.append(np.nanpercentile(roi_bias, 75))

        bias_median = np.array(bias_median)
        bias_lo = np.array(bias_lo)
        bias_hi = np.array(bias_hi)

        # Plot IQR band
        ax.fill_between(sample_sizes_present, bias_lo, bias_hi, alpha=0.2, color=color)

        # Plot line with markers at all sample sizes
        ax.plot(sample_sizes_present, bias_median, color=color, marker=marker,
                markersize=6, linewidth=2, label=label)

    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('Sample size', fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.legend(loc='best', fontsize=legend_size)
    ax.tick_params(labelsize=tick_size)


def create_figure(
    human_results: DatasetSubsampleResults,
    rat_results: DatasetSubsampleResults,
    human_bin_centers: np.ndarray,
    human_counts_matrix: np.ndarray,
    rat_bin_centers: np.ndarray,
    rat_counts_matrix: np.ndarray,
    output_file: Path,
    n_subsamples: int = N_SUBSAMPLES
) -> None:
    """Create the 2×2 figure."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))

    # Sample sizes for subsampling visualization (10^3 only)
    pdf_sample_sizes = [1_000]

    # Pooled counts for PDF plot
    human_pooled_counts = human_counts_matrix.sum(axis=0)
    rat_pooled_counts = rat_counts_matrix.sum(axis=0)

    # (a) PDFs for both datasets with subsampling variability
    plot_pdf_combined(axes[0, 0], human_bin_centers, human_pooled_counts,
                      rat_bin_centers, rat_pooled_counts,
                      pdf_sample_sizes, n_subsamples, x_max=2.0)

    # (b) Within-sample vs Between-ROI Wasserstein distances
    plot_within_vs_between_wasserstein(axes[0, 1], human_bin_centers, human_counts_matrix,
                                        rat_bin_centers, rat_counts_matrix,
                                        sample_sizes=[100, 1000, 10000, 100000], n_subsamples=500)

    # (c) Arithmetic mean radius: percentage error vs sample size (both datasets)
    plot_combined_bias(axes[1, 0], human_results, rat_results,
                       'r_arith', r'$\bar{r}$ error [%]')

    # (d) Effective radius: percentage error vs sample size (both datasets)
    plot_combined_bias(axes[1, 1], human_results, rat_results,
                       'r_eff', r'$r_{\mathrm{MRI}}$ error [%]')

    # Set y-axis scale symmetric around 0 for each panel independently
    for ax in [axes[1, 0], axes[1, 1]]:
        ymin, ymax = ax.get_ylim()
        ylim_max = max(abs(ymin), abs(ymax))
        ax.set_ylim(-ylim_max, ylim_max)

    # Set aspect ratio 1 for bottom panels
    for ax in [axes[1, 0], axes[1, 1]]:
        ax.set_box_aspect(1)

    plt.tight_layout()

    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')

    # Also save SVG version
    svg_file = output_file.with_suffix('.svg')
    plt.savefig(svg_file, bbox_inches='tight')
    logger.info(f"Saved figure to {output_file} and {svg_file}")

    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Study the effect of sample size on radius estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--human-data', type=Path, default=Path('data/raw/human/lm'),
                        help='Directory containing human CC TSV files (default: data/raw/human/lm)')
    parser.add_argument('--rat-data', type=Path, default=Path('data/processed/rat/lm'),
                        help='Directory containing rat NPZ files (default: data/processed/rat/lm)')
    parser.add_argument('--output', type=Path, default=Path('fig/main/sample_size_effect.svg'),
                        help='Output file path (default: fig/main/sample_size_effect.svg)')
    parser.add_argument('--n-subsamples', type=int, default=N_SUBSAMPLES,
                        help=f'Number of subsamples per ROI (default: {N_SUBSAMPLES})')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    quantile_points = np.linspace(0.01, 0.99, N_QUANTILES)

    # Load Human CC data
    logger.info("=" * 60)
    logger.info("Loading Human CC data...")
    human_edges, human_centers, human_counts, human_names = load_human_cc_histograms(args.human_data)

    logger.info("Running subsampling analysis for Human CC...")
    human_results = run_subsampling_analysis(
        human_edges, human_centers, human_counts,
        SAMPLE_SIZES, args.n_subsamples, quantile_points, "Human CC"
    )

    # Load Rat data
    logger.info("=" * 60)
    logger.info("Loading Rat WM data...")
    rat_edges, rat_centers, rat_counts, rat_names = load_rat_histograms(args.rat_data)

    logger.info("Running subsampling analysis for Rat WM...")
    rat_results = run_subsampling_analysis(
        rat_edges, rat_centers, rat_counts,
        SAMPLE_SIZES, args.n_subsamples, quantile_points, "Rat WM"
    )

    # Create figure
    logger.info("=" * 60)
    logger.info("Creating figure...")
    create_figure(
        human_results, rat_results,
        human_centers, human_counts,
        rat_centers, rat_counts,
        args.output,
        n_subsamples=args.n_subsamples
    )

    # Save JSON metadata
    json_file = args.output.with_suffix('.json')
    output_data = {
        'sample_sizes': SAMPLE_SIZES,
        'n_subsamples': args.n_subsamples,
        'human_cc': {
            size: {
                'n_valid_rois': results.n_valid_rois,
            }
            for size, results in human_results.results_by_size.items()
        },
        'rat_wm': {
            size: {
                'n_valid_rois': results.n_valid_rois,
            }
            for size, results in rat_results.results_by_size.items()
        }
    }
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Saved metadata to {json_file}")


if __name__ == '__main__':
    main()
