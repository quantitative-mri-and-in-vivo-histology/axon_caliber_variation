#!/usr/bin/env python3
"""
Distribution fitting comparison: LM vs EM human corpus callosum data.

Compares light microscopy (LM) data from data/raw_LM with electron microscopy (EM)
data from data/raw_philip.

Creates a combined figure with:
- Top row: Pooled PDFs with fitted distributions for LM and EM
- Bottom row: Model comparison metrics (win rate, Wasserstein, r_arith error, r_eff error)
  with per-subject scatter points

Usage:
    python scripts/figures/fit_philip_cc_distributions.py \
        --lm-data data/raw_LM \
        --em-data data/raw_philip/CC_anonymized.csv \
        --output fig/lm_vs_em_distribution_fits.png
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
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerBase
from scipy import stats
from scipy.optimize import OptimizeWarning

# Find repo root (contains pyproject.toml)
_root = Path(__file__).resolve().parent
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from axonometry import get_plot_settings, style_axis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings = get_plot_settings()

# Constants
MAX_SAMPLES_FOR_INIT = 10000
MIN_BIN_PROB = 1e-300
PLOT_XLIM_MAX = 3.0
EPS = 1e-10
DEFAULT_BIN_WIDTH = 0.05  # um

# Modality colors
MODALITY_COLORS = {
    'LM': settings.colors['binary_a'],  # Sand/tan
    'EM': settings.colors['binary_b'],  # Dusty teal
}

MODALITY_MARKERS = {
    'LM': 'o',
    'EM': 's',
}


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
    counts_matrix: np.ndarray  # (n_rois, n_bins)
    n_samples: int
    sample_counts: np.ndarray  # total count per ROI
    sample_names: List[str]
    subject_ids: Optional[List[str]] = None  # subject ID per ROI


@dataclass
class FitResult:
    """Container for distribution fit results."""
    distribution_name: str
    n_params: int
    params: Tuple
    nll: float
    aic: float
    bic: float
    ks_statistic: float
    rmse: float
    wasserstein: float = 0.0
    pdf_values: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class AggregatedMetrics:
    """Aggregated metrics from per-sample fitting."""
    distribution_names: List[str]
    summed_aic: np.ndarray
    n_successful_fits: np.ndarray
    pooled_results: List[FitResult]
    r_arith_mean: Dict[str, float] = field(default_factory=dict)
    r_arith_std: Dict[str, float] = field(default_factory=dict)
    r_eff_mean: Dict[str, float] = field(default_factory=dict)
    r_eff_std: Dict[str, float] = field(default_factory=dict)
    empirical_r_arith_mean: float = 0.0
    empirical_r_arith_std: float = 0.0
    empirical_r_eff_mean: float = 0.0
    empirical_r_eff_std: float = 0.0
    all_r_arith: np.ndarray = field(default_factory=lambda: np.array([]))
    all_r_eff: np.ndarray = field(default_factory=lambda: np.array([]))
    empirical_r_arith_per_sample: np.ndarray = field(default_factory=lambda: np.array([]))
    empirical_r_eff_per_sample: np.ndarray = field(default_factory=lambda: np.array([]))
    dist_name_to_idx: Dict[str, int] = field(default_factory=dict)
    all_aic: np.ndarray = field(default_factory=lambda: np.array([]))
    all_wasserstein: np.ndarray = field(default_factory=lambda: np.array([]))
    win_rate: Dict[str, float] = field(default_factory=dict)
    # Per-subject aggregates
    subject_ids: List[str] = field(default_factory=list)
    per_subject_r_arith: Dict[str, Dict[str, float]] = field(default_factory=dict)  # dist -> subject -> value
    per_subject_r_eff: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_subject_empirical_r_arith: Dict[str, float] = field(default_factory=dict)
    per_subject_empirical_r_eff: Dict[str, float] = field(default_factory=dict)
    per_subject_wasserstein: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_subject_win_rate: Dict[str, Dict[str, float]] = field(default_factory=dict)


# =============================================================================
# Distribution Catalog
# =============================================================================

CANDIDATE_DISTRIBUTIONS = [
    ('genextreme', stats.genextreme),
    ('invgauss', stats.invgauss),
    ('fatiguelife', stats.fatiguelife),
    ('lognorm', stats.lognorm),
    ('gamma', stats.gamma),
]

DIST_DISPLAY_NAMES = {
    'genextreme': 'Gen. Ext. Value',
    'lognorm': 'Log Normal',
    'invgauss': 'Inverse Gaussian',
    'fatiguelife': 'Birnbaum-Saunders',
    'gamma': 'Gamma',
}

DIST_DISPLAY_NAMES_MULTILINE = {
    'genextreme': 'Gen. Ext.\nValue',
    'lognorm': 'Log\nNormal',
    'invgauss': 'Inverse\nGaussian',
    'fatiguelife': 'Birnbaum-\nSaunders',
    'gamma': 'Gamma',
}

DIST_COLORS = {
    'genextreme': '#2ca02c',
    'lognorm': '#ff7f0e',
    'invgauss': '#9467bd',
    'fatiguelife': '#8c564b',
    'gamma': '#008080',
}


def get_display_name(scipy_name: str, multiline: bool = False) -> str:
    """Get display name for a distribution."""
    if multiline:
        return DIST_DISPLAY_NAMES_MULTILINE.get(scipy_name, scipy_name)
    return DIST_DISPLAY_NAMES.get(scipy_name, scipy_name)


def get_dist_color(dist_name: str) -> str:
    """Get fixed color for a distribution."""
    return DIST_COLORS.get(dist_name, '#333333')


# =============================================================================
# Data Loading
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


def load_lm_data(
    data_dir: Path,
    bin_width: float = DEFAULT_BIN_WIDTH,
    r_max: float = 3.0
) -> Tuple[HistogramData, PerSampleHistogramData]:
    """Load LM human corpus callosum histogram data (MinorAxis)."""
    bin_edges_file = data_dir / 'desc-binEdges_radii.tsv'
    counts_file = data_dir / 'desc-countsMinorAxis_radii.tsv'
    roiinfo_file = data_dir / 'roiinfo.tsv'

    # Load bin edges
    bin_edges_orig = np.loadtxt(bin_edges_file, delimiter='\t', skiprows=1)
    logger.info(f"LM: loaded {len(bin_edges_orig)} bin edges")

    # Load counts matrix (rows=ROIs, columns=bins)
    counts_matrix_orig = np.loadtxt(counts_file, delimiter='\t', skiprows=1, dtype=float)
    n_rois = counts_matrix_orig.shape[0]
    logger.info(f"LM: {n_rois} ROIs")

    # Load ROI info for subject IDs
    roiinfo = pd.read_csv(roiinfo_file, sep='\t')
    subject_ids = roiinfo['subject_id'].tolist()
    roi_names = roiinfo['roi_id'].tolist()

    # Rediscretize each ROI
    first_edges, first_centers, _ = rediscretize_histogram(
        bin_edges_orig, counts_matrix_orig[0], bin_width
    )

    # Clip to r_max
    max_idx = np.searchsorted(first_edges, r_max)
    first_edges = first_edges[:max_idx + 1]
    first_centers = first_centers[:max_idx]
    n_bins = len(first_centers)

    counts_matrix = np.zeros((n_rois, n_bins), dtype=float)

    for i in range(n_rois):
        _, _, rediscretized = rediscretize_histogram(
            bin_edges_orig, counts_matrix_orig[i], bin_width
        )
        counts_matrix[i] = rediscretized[:n_bins]

    sample_counts = counts_matrix.sum(axis=1)
    pooled_counts = counts_matrix.sum(axis=0)
    total_count = int(pooled_counts.sum())

    logger.info(f"LM: {total_count:,} total axons, {n_bins} bins")
    logger.info(f"LM subjects: {sorted(set(subject_ids))}")

    pooled = HistogramData(
        bin_edges=first_edges,
        bin_centers=first_centers,
        counts=pooled_counts,
        n_samples=n_rois,
        total_count=total_count,
        name="LM"
    )

    per_sample = PerSampleHistogramData(
        bin_edges=first_edges,
        bin_centers=first_centers,
        counts_matrix=counts_matrix,
        n_samples=n_rois,
        sample_counts=sample_counts,
        sample_names=roi_names,
        subject_ids=subject_ids
    )

    return pooled, per_sample


def load_em_data(
    csv_path: Path,
    bin_width: float = DEFAULT_BIN_WIDTH,
    r_max: float = 3.0
) -> Tuple[HistogramData, PerSampleHistogramData]:
    """Load EM (Philip's) human CC data."""
    df = pd.read_csv(csv_path)
    df['radius'] = df['inner_axis_minor_length'] / 2

    # Extract sample name as subject ID
    def extract_sample(filename):
        parts = filename.split('_')
        for part in parts:
            if part.startswith('Sample'):
                return part
        return 'Unknown'

    df['subject_id'] = df['file'].apply(extract_sample)

    roi_names = sorted(df['file'].unique())
    n_rois = len(roi_names)
    subject_ids = [extract_sample(r) for r in roi_names]

    logger.info(f"EM: {len(df):,} axons across {n_rois} ROIs")
    logger.info(f"EM subjects: {sorted(set(subject_ids))}")

    bin_edges = np.arange(0, r_max + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_centers)

    counts_matrix = np.zeros((n_rois, n_bins), dtype=float)

    for i, roi_name in enumerate(roi_names):
        roi_radii = df[df['file'] == roi_name]['radius'].values
        counts, _ = np.histogram(roi_radii, bins=bin_edges)
        counts_matrix[i] = counts

    sample_counts = counts_matrix.sum(axis=1)
    pooled_counts = counts_matrix.sum(axis=0)
    total_count = int(pooled_counts.sum())

    pooled = HistogramData(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        counts=pooled_counts,
        n_samples=n_rois,
        total_count=total_count,
        name="EM"
    )

    per_sample = PerSampleHistogramData(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        counts_matrix=counts_matrix,
        n_samples=n_rois,
        sample_counts=sample_counts,
        sample_names=roi_names,
        subject_ids=subject_ids
    )

    return pooled, per_sample


# =============================================================================
# Radius Calculations
# =============================================================================

def compute_empirical_radii(bin_centers: np.ndarray, counts: np.ndarray) -> Tuple[float, float]:
    """Compute empirical r_arith and r_eff from histogram."""
    total = counts.sum()
    if total == 0:
        return np.nan, np.nan

    probs = counts / total
    r = bin_centers

    r_arith = np.sum(r * probs)

    r2 = np.sum(r**2 * probs)
    r6 = np.sum(r**6 * probs)
    r_eff = (r6 / r2) ** 0.25 if r2 > 0 else np.nan

    return r_arith, r_eff


def compute_distribution_radii(
    dist: stats.rv_continuous,
    params: Tuple,
    r_max: float = 10.0,
    n_points: int = 1000
) -> Tuple[float, float]:
    """Compute r_arith and r_eff from fitted distribution."""
    shape_params, loc, scale = _unpack_params(params)

    r = np.linspace(0.001, r_max, n_points)
    dr = r[1] - r[0]

    try:
        pdf = dist.pdf(r, *shape_params, loc=loc, scale=scale)
        pdf = np.maximum(pdf, 0)

        norm = np.sum(pdf) * dr
        if norm < 0.01:
            return np.nan, np.nan
        pdf = pdf / norm

        r_arith = np.sum(r * pdf) * dr

        r2 = np.sum(r**2 * pdf) * dr
        r6 = np.sum(r**6 * pdf) * dr
        r_eff = (r6 / r2) ** 0.25 if r2 > 0 else np.nan

        return r_arith, r_eff
    except Exception:
        return np.nan, np.nan


def _unpack_params(params: Tuple) -> Tuple[Tuple, float, float]:
    """Unpack distribution parameters into (shape_params, loc, scale)."""
    if len(params) == 2:
        return (), params[0], params[1]
    return params[:-2], params[-2], params[-1]


# =============================================================================
# Distribution Fitting
# =============================================================================

def fit_distribution_mle(
    dist_name: str,
    dist: stats.rv_continuous,
    bin_centers: np.ndarray,
    bin_edges: np.ndarray,
    counts: np.ndarray
) -> Optional[FitResult]:
    """Fit distribution to histogram data using maximum likelihood."""
    total = counts.sum()
    if total == 0:
        return None

    n_samples = min(MAX_SAMPLES_FOR_INIT, int(total))
    probs = counts / total
    probs = probs / probs.sum()

    DIST_SEEDS = {name: i * 12345 for i, (name, _) in enumerate(CANDIDATE_DISTRIBUTIONS)}
    saved_state = np.random.get_state()

    try:
        np.random.seed(DIST_SEEDS.get(dist_name, 42))
        bin_width = np.diff(bin_edges).mean()
        jitter = bin_width / 2
        sampled_bins = np.random.choice(len(bin_centers), size=n_samples, p=probs)
        samples = bin_centers[sampled_bins] + np.random.uniform(-jitter, jitter, n_samples)
        samples = samples[samples > 0]

        if len(samples) < 10:
            return None

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', category=OptimizeWarning)
            if dist_name == 'genextreme':
                params = dist.fit(samples)
                if params[1] < 0:
                    params = dist.fit(samples, floc=0)
            else:
                params = dist.fit(samples, floc=0)

        shape_params, loc, scale = _unpack_params(params)
        pdf_values = dist.pdf(bin_centers, *shape_params, loc=loc, scale=scale)

        edge_cdfs = dist.cdf(bin_edges, *shape_params, loc=loc, scale=scale)
        bin_probs = edge_cdfs[1:] - edge_cdfs[:-1]
        bin_probs = np.maximum(bin_probs, MIN_BIN_PROB)

        nll = -np.dot(counts, np.log(bin_probs))
        k = len(params)
        n = total
        aic = 2 * k + 2 * nll
        bic = k * np.log(n) + 2 * nll

        empirical_cdf = np.cumsum(counts) / total
        fitted_cdf = dist.cdf(bin_centers, *shape_params, loc=loc, scale=scale)
        ks_stat = np.max(np.abs(empirical_cdf - fitted_cdf))

        wasserstein = np.sum(np.abs(empirical_cdf - fitted_cdf)) * bin_width

        hist_density = counts / (total * bin_width)
        rmse = np.sqrt(np.mean((pdf_values - hist_density) ** 2))

        result = FitResult(
            distribution_name=dist_name,
            n_params=len(params),
            params=params,
            nll=nll,
            aic=aic,
            bic=bic,
            ks_statistic=ks_stat,
            rmse=rmse,
            wasserstein=wasserstein,
            pdf_values=pdf_values
        )
        np.random.set_state(saved_state)
        return result

    except (ValueError, RuntimeError) as e:
        logger.debug(f"Failed to fit {dist_name}: {e}")
        np.random.set_state(saved_state)
        return None


def fit_all_distributions(hist_data: HistogramData) -> List[FitResult]:
    """Fit all candidate distributions."""
    results = []

    for dist_name, dist in CANDIDATE_DISTRIBUTIONS:
        result = fit_distribution_mle(
            dist_name, dist,
            hist_data.bin_centers, hist_data.bin_edges, hist_data.counts
        )
        if result is not None:
            results.append(result)

    results.sort(key=lambda x: x.aic)
    return results


def fit_all_samples(
    per_sample_data: PerSampleHistogramData,
    pooled_data: HistogramData
) -> AggregatedMetrics:
    """Fit all distributions to each ROI and aggregate, including per-subject stats."""
    n_samples = per_sample_data.n_samples
    subject_ids = per_sample_data.subject_ids or ['unknown'] * n_samples
    unique_subjects = sorted(set(subject_ids))

    all_method_names = [name for name, _ in CANDIDATE_DISTRIBUTIONS]
    n_methods = len(all_method_names)

    all_aics = np.full((n_methods, n_samples), np.nan)
    all_wasserstein = np.full((n_methods, n_samples), np.nan)
    all_r_arith = np.full((n_methods, n_samples), np.nan)
    all_r_eff = np.full((n_methods, n_samples), np.nan)

    empirical_r_arith = np.full(n_samples, np.nan)
    empirical_r_eff = np.full(n_samples, np.nan)

    logger.info(f"Fitting {n_methods} distributions to {n_samples} ROIs...")

    for sample_idx in range(n_samples):
        counts = per_sample_data.counts_matrix[sample_idx]
        total = counts.sum()
        if total < 100:
            continue

        emp_r_arith, emp_r_eff = compute_empirical_radii(
            per_sample_data.bin_centers, counts
        )
        empirical_r_arith[sample_idx] = emp_r_arith
        empirical_r_eff[sample_idx] = emp_r_eff

        for dist_idx, (dist_name, dist) in enumerate(CANDIDATE_DISTRIBUTIONS):
            result = fit_distribution_mle(
                dist_name, dist,
                per_sample_data.bin_centers, per_sample_data.bin_edges, counts
            )
            if result is not None:
                all_aics[dist_idx, sample_idx] = result.aic
                all_wasserstein[dist_idx, sample_idx] = result.wasserstein
                r_arith, r_eff = compute_distribution_radii(dist, result.params)
                all_r_arith[dist_idx, sample_idx] = r_arith
                all_r_eff[dist_idx, sample_idx] = r_eff

    summed_aic = np.nansum(all_aics, axis=1)
    n_successful_fits = np.sum(~np.isnan(all_aics), axis=1)

    # Win rate calculation
    win_counts = np.zeros(n_methods)
    for sample_idx in range(n_samples):
        sample_aics = all_aics[:, sample_idx]
        if np.all(np.isnan(sample_aics)):
            continue
        winner_idx = np.nanargmin(sample_aics)
        win_counts[winner_idx] += 1
    n_valid_samples = np.sum(~np.all(np.isnan(all_aics), axis=0))
    win_rate = win_counts / n_valid_samples if n_valid_samples > 0 else win_counts

    sort_idx = np.argsort(summed_aic)
    sorted_names = [all_method_names[i] for i in sort_idx]

    # Per-distribution mean/std
    r_arith_mean = {}
    r_arith_std = {}
    r_eff_mean = {}
    r_eff_std = {}

    for dist_idx, dist_name in enumerate(all_method_names):
        valid_r_arith = all_r_arith[dist_idx, ~np.isnan(all_r_arith[dist_idx])]
        valid_r_eff = all_r_eff[dist_idx, ~np.isnan(all_r_eff[dist_idx])]

        r_arith_mean[dist_name] = np.mean(valid_r_arith) if len(valid_r_arith) > 0 else np.nan
        r_arith_std[dist_name] = np.std(valid_r_arith) if len(valid_r_arith) > 0 else np.nan
        r_eff_mean[dist_name] = np.mean(valid_r_eff) if len(valid_r_eff) > 0 else np.nan
        r_eff_std[dist_name] = np.std(valid_r_eff) if len(valid_r_eff) > 0 else np.nan

    # Per-subject aggregates
    per_subject_r_arith = {dist: {} for dist in all_method_names}
    per_subject_r_eff = {dist: {} for dist in all_method_names}
    per_subject_empirical_r_arith = {}
    per_subject_empirical_r_eff = {}
    per_subject_wasserstein = {dist: {} for dist in all_method_names}
    per_subject_win_rate = {dist: {} for dist in all_method_names}

    for subj in unique_subjects:
        subj_mask = np.array([s == subj for s in subject_ids])
        subj_indices = np.where(subj_mask)[0]

        # Empirical radii for subject
        subj_emp_r_arith = empirical_r_arith[subj_mask]
        subj_emp_r_eff = empirical_r_eff[subj_mask]
        per_subject_empirical_r_arith[subj] = np.nanmean(subj_emp_r_arith)
        per_subject_empirical_r_eff[subj] = np.nanmean(subj_emp_r_eff)

        # Per-distribution metrics for subject
        for dist_idx, dist_name in enumerate(all_method_names):
            subj_r_arith = all_r_arith[dist_idx, subj_mask]
            subj_r_eff = all_r_eff[dist_idx, subj_mask]
            subj_wasserstein = all_wasserstein[dist_idx, subj_mask]

            per_subject_r_arith[dist_name][subj] = np.nanmean(subj_r_arith)
            per_subject_r_eff[dist_name][subj] = np.nanmean(subj_r_eff)
            per_subject_wasserstein[dist_name][subj] = np.nanmean(subj_wasserstein)

        # Win rate per subject
        subj_win_counts = np.zeros(n_methods)
        for idx in subj_indices:
            sample_aics = all_aics[:, idx]
            if np.all(np.isnan(sample_aics)):
                continue
            winner_idx = np.nanargmin(sample_aics)
            subj_win_counts[winner_idx] += 1
        n_valid_subj = np.sum(~np.all(np.isnan(all_aics[:, subj_mask]), axis=0))
        for dist_idx, dist_name in enumerate(all_method_names):
            per_subject_win_rate[dist_name][subj] = (
                subj_win_counts[dist_idx] / n_valid_subj if n_valid_subj > 0 else 0
            )

    pooled_results = fit_all_distributions(pooled_data)

    dist_name_to_idx = {name: idx for idx, (name, _) in enumerate(CANDIDATE_DISTRIBUTIONS)}
    win_rate_dict = {all_method_names[i]: win_rate[i] for i in range(n_methods)}

    return AggregatedMetrics(
        distribution_names=sorted_names,
        summed_aic=summed_aic[sort_idx],
        n_successful_fits=n_successful_fits[sort_idx],
        pooled_results=pooled_results,
        r_arith_mean=r_arith_mean,
        r_arith_std=r_arith_std,
        r_eff_mean=r_eff_mean,
        r_eff_std=r_eff_std,
        empirical_r_arith_mean=np.nanmean(empirical_r_arith),
        empirical_r_arith_std=np.nanstd(empirical_r_arith),
        empirical_r_eff_mean=np.nanmean(empirical_r_eff),
        empirical_r_eff_std=np.nanstd(empirical_r_eff),
        all_r_arith=all_r_arith,
        all_r_eff=all_r_eff,
        empirical_r_arith_per_sample=empirical_r_arith,
        empirical_r_eff_per_sample=empirical_r_eff,
        dist_name_to_idx=dist_name_to_idx,
        all_aic=all_aics,
        all_wasserstein=all_wasserstein,
        win_rate=win_rate_dict,
        subject_ids=unique_subjects,
        per_subject_r_arith=per_subject_r_arith,
        per_subject_r_eff=per_subject_r_eff,
        per_subject_empirical_r_arith=per_subject_empirical_r_arith,
        per_subject_empirical_r_eff=per_subject_empirical_r_eff,
        per_subject_wasserstein=per_subject_wasserstein,
        per_subject_win_rate=per_subject_win_rate
    )


def compute_inter_roi_wasserstein(per_sample: PerSampleHistogramData) -> float:
    """Compute median pairwise Wasserstein distance between ROIs."""
    bin_centers = per_sample.bin_centers
    bin_width = np.diff(per_sample.bin_edges).mean()
    n_samples = per_sample.n_samples

    cdfs = []
    for i in range(n_samples):
        counts = per_sample.counts_matrix[i]
        total = counts.sum()
        if total > 100:
            cdf = np.cumsum(counts) / total
            cdfs.append(cdf)

    n_rois = len(cdfs)
    pairwise_distances = []
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            w_dist = np.sum(np.abs(cdfs[i] - cdfs[j])) * bin_width
            pairwise_distances.append(w_dist)

    return np.median(pairwise_distances) if pairwise_distances else 0.0


# =============================================================================
# Visualization
# =============================================================================

def _plot_pooled_pdf_with_fits(
    ax,
    hist_data: HistogramData,
    fit_results: List[FitResult],
    modality_color: str,
    inset_xlim: Tuple[float, float] = (1.0, 3.0),
    distribution_order: List[str] = None
) -> None:
    """Plot pooled histogram with all fitted PDFs."""
    bin_width = np.diff(hist_data.bin_edges).mean()
    density = hist_data.counts / (hist_data.total_count * bin_width)

    ax.bar(hist_data.bin_centers, density, width=bin_width * 0.9,
           alpha=0.4, color=modality_color, edgecolor='white', linewidth=0.5,
           zorder=1)

    if distribution_order is not None:
        result_map = {r.distribution_name: r for r in fit_results}
        ordered_results = [result_map[name] for name in distribution_order if name in result_map]
    else:
        ordered_results = fit_results

    for result in ordered_results:
        color = get_dist_color(result.distribution_name)
        display_name = get_display_name(result.distribution_name)
        ax.plot(hist_data.bin_centers, result.pdf_values, '-',
                color=color, linewidth=1.5, label=display_name, zorder=2)

    ax.set_xlim(0, PLOT_XLIM_MAX)
    ax.set_ylim(0, None)
    ax.set_xlabel('Axon radius [μm]', fontsize=settings.fonts['label_size'])
    ax.set_ylabel(r'Probability density [μm$^{-1}$]', fontsize=settings.fonts['label_size'])
    ax.tick_params(labelsize=settings.fonts['tick_size'])

    # Add tail inset
    tail_mask = hist_data.bin_centers >= inset_xlim[0]
    tail_density = density[tail_mask]
    tail_y_max = tail_density.max() * 1.2 if len(tail_density) > 0 and tail_density.max() > 0 else 0.1

    ax_inset = ax.inset_axes([0.32, 0.32, 0.66, 0.66])
    ax_inset.bar(hist_data.bin_centers, density, width=bin_width * 0.9,
                 alpha=0.4, color=modality_color, edgecolor='white', linewidth=0.3)
    for result in ordered_results:
        color = get_dist_color(result.distribution_name)
        ax_inset.plot(hist_data.bin_centers, result.pdf_values, '-',
                      color=color, linewidth=1.5)
    ax_inset.set_xlim(*inset_xlim)
    ax_inset.set_ylim(0, tail_y_max)
    ax_inset.tick_params(labelsize=settings.fonts['tick_size'] - 2)
    ax_inset.set_xlabel('')
    ax_inset.set_ylabel('')

    ax.indicate_inset_zoom(ax_inset, edgecolor='gray', linewidth=1.5,
                           linestyle='--', alpha=0.8)
    style_axis(ax)


def _plot_win_rate(
    ax,
    all_metrics: Dict[str, AggregatedMetrics],
    modality_names: List[str]
) -> None:
    """Plot win rate (pooled across all ROIs)."""
    first_metrics = all_metrics[modality_names[0]]
    names = first_metrics.distribution_names
    display_names = [get_display_name(n, multiline=True) for n in names]

    y_spacing = 1.1
    y_pos = np.arange(len(names)) * y_spacing
    n_modalities = len(modality_names)
    offsets = np.linspace(-0.25, 0.25, n_modalities)

    # Plot pooled win rate (across all ROIs)
    for mod_idx, modality in enumerate(modality_names):
        metrics = all_metrics[modality]
        color = MODALITY_COLORS[modality]
        marker = MODALITY_MARKERS[modality]

        for dist_idx, dist_name in enumerate(names):
            win_rate = metrics.win_rate.get(dist_name, 0) * 100
            ax.scatter(win_rate, y_pos[dist_idx] + offsets[mod_idx],
                      color=color, s=100, marker=marker,
                      label=modality if dist_idx == 0 else None,
                      edgecolor='black', linewidth=1, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=9)
    ax.set_xlabel('Win rate [%]', fontsize=10)
    ax.set_xlim(-5, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(y_pos[-1] + 0.4, -1.4)
    ax.legend(loc='upper right', fontsize=8, ncol=2,
              frameon=True, edgecolor='gray', facecolor='white', framealpha=1.0)
    ax.tick_params(labelsize=9)
    style_axis(ax)


def _plot_wasserstein(
    ax,
    all_metrics: Dict[str, AggregatedMetrics],
    all_per_roi: Dict[str, PerSampleHistogramData],
    modality_names: List[str]
) -> None:
    """Plot Wasserstein distance with per-subject points."""
    first_metrics = all_metrics[modality_names[0]]
    names = first_metrics.distribution_names
    display_names = [get_display_name(n, multiline=True) for n in names]

    y_spacing = 1.1
    y_pos = np.arange(len(names)) * y_spacing
    n_modalities = len(modality_names)
    offsets = np.linspace(-0.25, 0.25, n_modalities)

    # First pass: plot pooled (mean) markers
    for mod_idx, modality in enumerate(modality_names):
        metrics = all_metrics[modality]
        color = MODALITY_COLORS[modality]
        marker = MODALITY_MARKERS[modality]

        for dist_idx, dist_name in enumerate(names):
            dist_idx_orig = metrics.dist_name_to_idx[dist_name]
            w_vals = metrics.all_wasserstein[dist_idx_orig]
            mean_w = np.nanmean(w_vals)
            ax.scatter(mean_w, y_pos[dist_idx] + offsets[mod_idx],
                      color=color, s=100, marker=marker,
                      label=f'{modality} (pooled)' if dist_idx == 0 else None,
                      edgecolor='black', linewidth=1, zorder=3)

    # Second pass: plot per-subject points on top
    for mod_idx, modality in enumerate(modality_names):
        metrics = all_metrics[modality]
        color = MODALITY_COLORS[modality]
        marker = MODALITY_MARKERS[modality]

        for subj_idx, subj in enumerate(metrics.subject_ids):
            for dist_idx, dist_name in enumerate(names):
                w_val = metrics.per_subject_wasserstein[dist_name].get(subj, np.nan)
                if not np.isnan(w_val):
                    ax.scatter(w_val, y_pos[dist_idx] + offsets[mod_idx],
                              color=color, s=40, marker=marker, alpha=0.7,
                              edgecolor='white', linewidth=0.3, zorder=4,
                              label=f'{modality} (per-subj)' if dist_idx == 0 and subj_idx == 0 else None)

    # Add inter-ROI reference lines
    for modality in modality_names:
        per_roi = all_per_roi[modality]
        inter_roi_w = compute_inter_roi_wasserstein(per_roi)
        color = MODALITY_COLORS[modality]
        ax.axvline(inter_roi_w, color=color, linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=9)
    ax.set_xlabel('Wasserstein distance [μm]', fontsize=10)
    ax.set_xlim(0, 0.06)
    ax.set_ylim(y_pos[-1] + 0.4, -1.4)
    ax.tick_params(labelsize=9)
    ax.legend(loc='upper right', fontsize=7, ncol=2,
              frameon=True, edgecolor='gray', facecolor='white', framealpha=1.0)
    style_axis(ax)


def _plot_radius_bias(
    ax,
    all_metrics: Dict[str, AggregatedMetrics],
    modality_names: List[str],
    radius_type: str
) -> None:
    """Plot radius bias with per-subject points."""
    first_metrics = all_metrics[modality_names[0]]
    names = first_metrics.distribution_names
    display_names = [get_display_name(n, multiline=True) for n in names]

    y_spacing = 1.1
    y_pos = np.arange(len(names)) * y_spacing
    n_modalities = len(modality_names)
    offsets = np.linspace(-0.25, 0.25, n_modalities)

    if radius_type == 'r_arith':
        xlabel = r'$\bar{r}$ error [%]'
        x_lim = 5
    else:
        xlabel = r'$r_{\mathrm{MRI}}$ error [%]'
        x_lim = 100

    # Track out-of-bounds values for annotation
    out_of_bounds = []  # List of (bias, y_pos, color, modality)

    # First pass: plot pooled (mean) markers
    for mod_idx, modality in enumerate(modality_names):
        metrics = all_metrics[modality]
        color = MODALITY_COLORS[modality]
        marker = MODALITY_MARKERS[modality]

        if radius_type == 'r_arith':
            all_fitted = metrics.all_r_arith
            emp_per_sample = metrics.empirical_r_arith_per_sample
        else:
            all_fitted = metrics.all_r_eff
            emp_per_sample = metrics.empirical_r_eff_per_sample

        for dist_idx, dist_name in enumerate(names):
            dist_idx_orig = metrics.dist_name_to_idx[dist_name]
            fitted = all_fitted[dist_idx_orig]
            bias = (fitted - emp_per_sample) / emp_per_sample * 100
            mean_bias = np.nanmean(bias)

            # Clip for display
            mean_bias_clipped = np.clip(mean_bias, -x_lim, x_lim)
            ax.scatter(mean_bias_clipped, y_pos[dist_idx] + offsets[mod_idx],
                      color=color, s=100, marker=marker,
                      label=f'{modality} (pooled)' if dist_idx == 0 else None,
                      edgecolor='black', linewidth=1, zorder=3)

            # Track out-of-bounds mean values for arrows
            if mean_bias > x_lim or mean_bias < -x_lim:
                out_of_bounds.append((mean_bias, y_pos[dist_idx] + offsets[mod_idx], color, modality))

    # Second pass: plot per-subject points on top
    for mod_idx, modality in enumerate(modality_names):
        metrics = all_metrics[modality]
        color = MODALITY_COLORS[modality]
        marker = MODALITY_MARKERS[modality]

        if radius_type == 'r_arith':
            per_subj_fitted = metrics.per_subject_r_arith
            per_subj_emp = metrics.per_subject_empirical_r_arith
        else:
            per_subj_fitted = metrics.per_subject_r_eff
            per_subj_emp = metrics.per_subject_empirical_r_eff

        for subj_idx, subj in enumerate(metrics.subject_ids):
            emp_val = per_subj_emp.get(subj, np.nan)
            for dist_idx, dist_name in enumerate(names):
                fitted_val = per_subj_fitted[dist_name].get(subj, np.nan)
                if not np.isnan(fitted_val) and not np.isnan(emp_val) and emp_val > 0:
                    bias = (fitted_val - emp_val) / emp_val * 100
                    # Clip for display but track original
                    bias_clipped = np.clip(bias, -x_lim, x_lim)
                    ax.scatter(bias_clipped, y_pos[dist_idx] + offsets[mod_idx],
                              color=color, s=40, marker=marker, alpha=0.7,
                              edgecolor='white', linewidth=0.3, zorder=4,
                              label=f'{modality} (per-subj)' if dist_idx == 0 and subj_idx == 0 else None)

    ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.7)

    # Add arrows and annotations for out-of-bounds values
    for bias, y, color, modality in out_of_bounds:
        if bias > x_lim:
            ax.annotate(f'{bias:.0f}%', xy=(x_lim * 0.95, y), xytext=(x_lim * 0.7, y),
                       fontsize=7, fontweight='bold', color=color, ha='right', va='center',
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        elif bias < -x_lim:
            ax.annotate(f'{bias:.0f}%', xy=(-x_lim * 0.95, y), xytext=(-x_lim * 0.7, y),
                       fontsize=7, fontweight='bold', color=color, ha='left', va='center',
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(y_pos[-1] + 0.4, -1.4)
    ax.tick_params(labelsize=9)
    ax.legend(loc='upper right', fontsize=7, ncol=2,
              frameon=True, edgecolor='gray', facecolor='white', framealpha=1.0)
    style_axis(ax)


def create_combined_figure(
    all_pooled: Dict[str, HistogramData],
    all_metrics: Dict[str, AggregatedMetrics],
    all_per_roi: Dict[str, PerSampleHistogramData],
    modality_names: List[str],
    output_file: Path
) -> None:
    """Create combined figure with histograms (top) and metrics (bottom)."""
    n_modalities = len(modality_names)

    # Figure with nested GridSpecs
    fig = plt.figure(figsize=(18, 9))
    gs_main = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.35)

    # Top row: 2 histogram panels spanning full width
    gs_top = gs_main[0].subgridspec(1, n_modalities, wspace=0.3)

    # Bottom row: 4 metric panels spanning full width
    gs_bottom = gs_main[1].subgridspec(1, 4, wspace=0.25)

    # Top row: Histograms
    dist_order = all_metrics[modality_names[0]].distribution_names
    hist_axes = []

    for i, modality in enumerate(modality_names):
        ax = fig.add_subplot(gs_top[i])
        hist_axes.append(ax)
        _plot_pooled_pdf_with_fits(
            ax, all_pooled[modality], all_metrics[modality].pooled_results,
            modality_color=MODALITY_COLORS[modality],
            inset_xlim=(1.0, 3.0),
            distribution_order=dist_order
        )
        # Add modality name in top-left corner
        ax.text(0.05, 0.95, modality, transform=ax.transAxes,
                fontsize=settings.fonts['label_size'], fontweight='bold',
                va='top', ha='left')

    # Shared legend above histograms
    handles, labels = hist_axes[0].get_legend_handles_labels()

    # Create split-color patch for empirical data
    class SplitColorHandler(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            n = len(modality_names)
            rects = []
            part_width = width / n
            for j, mod in enumerate(modality_names):
                rect = Rectangle(
                    (xdescent + j * part_width, ydescent), part_width, height,
                    facecolor=MODALITY_COLORS[mod], edgecolor='white', linewidth=0.5,
                    alpha=0.6, transform=trans
                )
                rects.append(rect)
            return rects

    empirical_patch = Patch(facecolor='gray')
    all_handles = [empirical_patch] + handles
    all_labels = ['Empirical data'] + labels

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

    # Bottom row: Metrics
    ax_win = fig.add_subplot(gs_bottom[0])
    ax_wass = fig.add_subplot(gs_bottom[1])
    ax_arith = fig.add_subplot(gs_bottom[2])
    ax_eff = fig.add_subplot(gs_bottom[3])

    # Make bottom row plots narrower
    for ax in [ax_win, ax_wass, ax_arith, ax_eff]:
        ax.set_box_aspect(1.2)

    _plot_win_rate(ax_win, all_metrics, modality_names)
    _plot_wasserstein(ax_wass, all_metrics, all_per_roi, modality_names)
    _plot_radius_bias(ax_arith, all_metrics, modality_names, 'r_arith')
    _plot_radius_bias(ax_eff, all_metrics, modality_names, 'r_eff')

    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.svg'), bbox_inches='tight')
    plt.close()

    logger.info(f"Saved figure to {output_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare LM vs EM distribution fits for human CC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--lm-data', type=Path, default=Path('data/raw_LM'),
                        help='Path to LM data directory')
    parser.add_argument('--em-data', type=Path, default=Path('data/raw_philip/CC_anonymized.csv'),
                        help='Path to EM CSV file')
    parser.add_argument('--output', type=Path, default=Path('fig/lm_vs_em_distribution_fits.png'),
                        help='Output figure path')
    parser.add_argument('--bin-width', type=float, default=DEFAULT_BIN_WIDTH,
                        help=f'Bin width in um (default: {DEFAULT_BIN_WIDTH})')
    parser.add_argument('--r-max', type=float, default=3.0,
                        help='Maximum radius in um (default: 3.0)')

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    modality_names = ['LM', 'EM']

    all_pooled = {}
    all_metrics = {}
    all_per_roi = {}
    all_results = {}

    # Load LM data
    logger.info("=" * 60)
    logger.info("Loading LM data...")
    lm_pooled, lm_per_roi = load_lm_data(args.lm_data, args.bin_width, args.r_max)
    logger.info("Fitting distributions to LM data...")
    lm_metrics = fit_all_samples(lm_per_roi, lm_pooled)

    all_pooled['LM'] = lm_pooled
    all_per_roi['LM'] = lm_per_roi
    all_metrics['LM'] = lm_metrics

    logger.info(f"LM - Top 5 by summed AIC:")
    for i, name in enumerate(lm_metrics.distribution_names[:5], 1):
        delta = lm_metrics.summed_aic[i-1] - lm_metrics.summed_aic[0]
        win_pct = lm_metrics.win_rate.get(name, 0) * 100
        logger.info(f"  {i}. {name}: ΔAIC = {delta:.0f}, win rate = {win_pct:.0f}%")

    # Load EM data
    logger.info("=" * 60)
    logger.info("Loading EM data...")
    em_pooled, em_per_roi = load_em_data(args.em_data, args.bin_width, args.r_max)
    logger.info("Fitting distributions to EM data...")
    em_metrics = fit_all_samples(em_per_roi, em_pooled)

    all_pooled['EM'] = em_pooled
    all_per_roi['EM'] = em_per_roi
    all_metrics['EM'] = em_metrics

    logger.info(f"EM - Top 5 by summed AIC:")
    for i, name in enumerate(em_metrics.distribution_names[:5], 1):
        delta = em_metrics.summed_aic[i-1] - em_metrics.summed_aic[0]
        win_pct = em_metrics.win_rate.get(name, 0) * 100
        logger.info(f"  {i}. {name}: ΔAIC = {delta:.0f}, win rate = {win_pct:.0f}%")

    # Create combined figure
    logger.info("=" * 60)
    logger.info("Creating combined figure...")
    create_combined_figure(all_pooled, all_metrics, all_per_roi, modality_names, args.output)

    # Save results JSON
    for modality in modality_names:
        metrics = all_metrics[modality]
        pooled = all_pooled[modality]
        all_results[modality] = {
            'n_rois': pooled.n_samples,
            'total_count': pooled.total_count,
            'subjects': metrics.subject_ids,
            'distributions': [
                {
                    'name': name,
                    'summed_aic': float(metrics.summed_aic[i]),
                    'win_rate': float(metrics.win_rate.get(name, 0))
                }
                for i, name in enumerate(metrics.distribution_names[:5])
            ]
        }

    json_path = args.output.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved results to {json_path}")


if __name__ == '__main__':
    main()
