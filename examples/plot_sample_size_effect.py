#!/usr/bin/env python3
"""
Study the effect of sample size on radius estimation accuracy.

Creates a 2×4 figure:
Row 1 (Human CC):
  (a) QQ-plot: raw subsample quantiles vs whole section (4 sample sizes)
  (b) QQ-plot: GEV-fitted quantiles vs whole section
  (c) Bias: relative bias in r_arith and r_eff vs sample size
  (d) CoV: coefficient of variation in r_arith and r_eff vs sample size

Row 2 (Rat WM):
  (e-h) Same panels for rat data

Sample sizes: 10³, 10⁴, 10⁵, 10⁶
Subsamples per ROI: N=50 (configurable)
Variability shown as median ± IQR across ROIs and subsamples.

Usage:
    python examples/plot_sample_size_effect.py \
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
from scipy.optimize import OptimizeWarning

sys.path.insert(0, str(Path(__file__).parent.parent))
from axonometry import get_plot_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings = get_plot_settings()

# Constants
DEFAULT_BIN_WIDTH = 0.05  # μm
SAMPLE_SIZES = [100, 1_000, 10_000, 100_000, 1_000_000]  # 10², 10³, 10⁴, 10⁵, 10⁶
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
    """Load human corpus callosum histogram data."""
    bin_edges_file = data_dir / 'desc-binEdges_radii.tsv'
    counts_file = data_dir / 'desc-countsCircularEq_radii.tsv'

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
    r_max: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load rat LM data from NPZ files, split by population."""
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

    # Plot bars
    bars1 = ax.bar(x_positions - width/2, r_arith_bias_median, width,
                   yerr=r_arith_bias_iqr, label=r'$r_{arith}$',
                   color='#1f77b4', alpha=0.8,
                   capsize=err_settings['capsize'],
                   error_kw={'elinewidth': err_settings['linewidth']})

    bars2 = ax.bar(x_positions + width/2, r_eff_bias_median, width,
                   yerr=r_eff_bias_iqr, label=r'$r_{eff}$',
                   color='#ff7f0e', alpha=0.8,
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

    # Plot bars
    bars1 = ax.bar(x_positions - width/2, r_arith_cov_median, width,
                   yerr=r_arith_cov_iqr, label=r'$r_{arith}$',
                   color='#1f77b4', alpha=0.8,
                   capsize=err_settings['capsize'],
                   error_kw={'elinewidth': err_settings['linewidth']})

    bars2 = ax.bar(x_positions + width/2, r_eff_cov_median, width,
                   yerr=r_eff_cov_iqr, label=r'$r_{eff}$',
                   color='#ff7f0e', alpha=0.8,
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

    # Colors for datasets
    human_color = '#1f77b4'  # Blue
    rat_color = '#d62728'    # Red

    datasets = [
        (rat_results, 'Rat WM', rat_color, 's'),      # Squares on bottom
        (human_results, 'Human CC', human_color, 'o'),  # Circles on top
    ]

    for dataset_results, label, color, marker in datasets:
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

        # Plot with error bands
        ax.fill_between(sample_sizes_present, bias_lo, bias_hi, alpha=0.2, color=color)
        ax.plot(sample_sizes_present, bias_median, color=color, marker=marker,
                markersize=8, linewidth=2, label=label)

    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('Sample size', fontsize=font_settings['label_size'])
    ax.set_ylabel(ylabel, fontsize=font_settings['label_size'])
    ax.legend(loc='best', fontsize=font_settings['legend_size'])
    ax.tick_params(labelsize=font_settings['tick_size'])


def create_figure(
    human_results: DatasetSubsampleResults,
    rat_results: DatasetSubsampleResults,
    output_file: Path
) -> None:
    """Create the 2×2 figure."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))

    # Sample sizes for QQ plots (subset to avoid clutter)
    qq_sample_sizes = [100, 10_000, 1_000_000]

    # (a) Human CC: QQ (Raw)
    plot_qq_raw(axes[0, 0], human_results, sample_sizes=qq_sample_sizes)

    # (b) Rat WM: QQ (Raw)
    plot_qq_raw(axes[0, 1], rat_results, sample_sizes=qq_sample_sizes)

    # (c) Arithmetic mean radius: percentage error vs sample size (both datasets)
    plot_combined_bias(axes[1, 0], human_results, rat_results,
                       'r_arith', r'$\bar{r}$ error [%]')

    # (d) Effective radius: percentage error vs sample size (both datasets)
    plot_combined_bias(axes[1, 1], human_results, rat_results,
                       'r_eff', r'$r_{\mathrm{MRI}}$ error [%]')

    # Set same y-axis scale for (c) and (d)
    ymin = min(axes[1, 0].get_ylim()[0], axes[1, 1].get_ylim()[0])
    ymax = max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1])
    # Make symmetric around 0
    ylim_max = max(abs(ymin), abs(ymax))
    axes[1, 0].set_ylim(-ylim_max, ylim_max)
    axes[1, 1].set_ylim(-ylim_max, ylim_max)

    # Set aspect ratio 1 for bottom panels
    for ax in [axes[1, 0], axes[1, 1]]:
        ax.set_box_aspect(1)

    plt.tight_layout()

    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()
    logger.info(f"Saved figure to {output_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Study the effect of sample size on radius estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--human-data', type=Path, required=True,
                        help='Directory containing human CC TSV files')
    parser.add_argument('--rat-data', type=Path, required=True,
                        help='Directory containing rat NPZ files')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output PNG file path')
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
    create_figure(human_results, rat_results, args.output)

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
