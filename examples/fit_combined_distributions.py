#!/usr/bin/env python3
"""
Combined distribution fitting for Human CC and Rat white matter data.

Creates a 2x2 figure showing:
- (a) Human CC: histogram with fitted PDFs (largest ROI)
- (b) Rat: histogram with fitted PDFs (largest volume)
- (c) Human CC: AIC comparison (summed across all ROIs)
- (d) Rat: AIC comparison (summed across all volumes)

This script focuses purely on AIC-based model comparison. Separate scripts
handle r_eff and r_arith evaluation.

Usage:
    python fit_combined_distributions.py \\
        --human-data data/raw_LM \\
        --rat-data data/processed/LM \\
        --output fig/combined_distribution_fits.png
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
MAX_SAMPLES_FOR_INIT = 10000
MIN_BIN_PROB = 1e-300
PLOT_XLIM_MAX = 3.0
EPS = 1e-10
DEFAULT_BIN_WIDTH = 0.05  # um


# =============================================================================
# EM Correction Functions
# =============================================================================

def load_em_radii(em_dir: Path, radius_column: str = 'r_circular_equivalent') -> np.ndarray:
    """
    Load EM (electron microscopy) radii from CSV files.

    Args:
        em_dir: Directory containing ROI subdirectories with *_axon_radii_rp.csv files
        radius_column: Column name for radius measure (default: r_circular_equivalent)

    Returns:
        Array of all EM radii
    """
    import pandas as pd

    radii = []
    for roi_dir in em_dir.iterdir():
        if roi_dir.is_dir():
            csv_file = roi_dir / f'{roi_dir.name}_axon_radii_rp.csv'
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                if radius_column in df.columns:
                    radii.extend(df[radius_column].values)
    return np.array(radii)


def apply_em_correction(
    bin_centers: np.ndarray,
    bin_edges: np.ndarray,
    lm_counts: np.ndarray,
    em_radii: np.ndarray,
    threshold: float = 0.4,
    scale_range: Tuple[float, float] = (0.5, 1.0)
) -> np.ndarray:
    """
    Apply EM-based correction to LM counts for small radii.

    For r < threshold: use scaled EM counts
    For r >= threshold: use original LM counts

    The scale factor is computed by matching total counts in the scale_range
    where both LM and EM are reliable.

    Args:
        bin_centers: Histogram bin centers
        bin_edges: Histogram bin edges
        lm_counts: Original LM counts
        em_radii: Array of EM radii (raw measurements)
        threshold: Radius threshold for correction (default: 0.4 μm)
        scale_range: (r_min, r_max) range for computing scale factor (default: 0.5-1.0 μm)

    Returns:
        Corrected counts array
    """
    # Create EM histogram with same bins
    em_counts, _ = np.histogram(em_radii, bins=bin_edges)

    # Compute scale factor in reliable range
    scale_mask = (bin_centers >= scale_range[0]) & (bin_centers <= scale_range[1])
    em_in_range = em_counts[scale_mask].sum()
    lm_in_range = lm_counts[scale_mask].sum()

    if em_in_range < 10:
        logger.warning(f"Too few EM counts in scale range {scale_range}: {em_in_range}")
        return lm_counts

    scale_factor = lm_in_range / em_in_range
    logger.info(f"EM correction: scale factor = {scale_factor:.1f} (matched at {scale_range[0]}-{scale_range[1]} μm)")

    # Apply correction
    em_scaled = em_counts * scale_factor
    corrected = np.where(bin_centers < threshold, em_scaled, lm_counts)

    # Log summary
    n_added = corrected.sum() - lm_counts.sum()
    logger.info(f"EM correction: threshold = {threshold} μm, added {n_added:.0f} counts ({100*n_added/lm_counts.sum():.1f}%)")

    return corrected.astype(lm_counts.dtype)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class HistogramData:
    """Container for histogram data."""
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    counts: np.ndarray
    n_samples: int  # number of ROIs or volumes
    total_count: int
    name: str = ""


@dataclass
class PerSampleHistogramData:
    """Container for per-sample (ROI or volume) histogram data."""
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    counts_matrix: np.ndarray  # (n_samples, n_bins)
    n_samples: int
    sample_counts: np.ndarray  # total count per sample
    sample_names: List[str]


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
    wasserstein: float = 0.0  # Wasserstein distance between empirical and fitted CDF
    pdf_values: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class AggregatedMetrics:
    """Aggregated metrics from per-sample fitting."""
    distribution_names: List[str]
    summed_aic: np.ndarray
    n_successful_fits: np.ndarray
    pooled_results: List[FitResult]
    # Per-distribution radius estimates (mean ± std across ROIs)
    r_arith_mean: Dict[str, float] = field(default_factory=dict)
    r_arith_std: Dict[str, float] = field(default_factory=dict)
    r_eff_mean: Dict[str, float] = field(default_factory=dict)
    r_eff_std: Dict[str, float] = field(default_factory=dict)
    # Empirical values (mean ± std across samples)
    empirical_r_arith_mean: float = 0.0
    empirical_r_arith_std: float = 0.0
    empirical_r_eff_mean: float = 0.0
    empirical_r_eff_std: float = 0.0
    # Per-sample arrays for scatter plots: shape (n_distributions, n_samples)
    all_r_arith: np.ndarray = field(default_factory=lambda: np.array([]))
    all_r_eff: np.ndarray = field(default_factory=lambda: np.array([]))
    # Per-sample empirical values: shape (n_samples,)
    empirical_r_arith_per_sample: np.ndarray = field(default_factory=lambda: np.array([]))
    empirical_r_eff_per_sample: np.ndarray = field(default_factory=lambda: np.array([]))
    # Distribution name to index mapping (for accessing all_r_arith/all_r_eff)
    dist_name_to_idx: Dict[str, int] = field(default_factory=dict)
    # Per-sample AIC values: shape (n_distributions, n_samples)
    all_aic: np.ndarray = field(default_factory=lambda: np.array([]))
    # Per-sample Wasserstein distances: shape (n_distributions, n_samples)
    all_wasserstein: np.ndarray = field(default_factory=lambda: np.array([]))
    # Win rate per distribution (fraction of samples where it has lowest AIC)
    win_rate: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Data Loading - Human CC (TSV histograms)
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


def load_human_cc_data(
    data_dir: Path,
    bin_width: float = DEFAULT_BIN_WIDTH,
    em_correction_dir: Optional[Path] = None,
    em_correction_threshold: float = 0.4,
    em_correction_scale_range: Tuple[float, float] = (0.5, 1.0)
) -> Tuple[HistogramData, PerSampleHistogramData]:
    """
    Load human corpus callosum histogram data (MinorAxis).

    Args:
        data_dir: Directory containing human LM histogram files
        bin_width: Target bin width in μm for rediscretization
        em_correction_dir: If provided, apply EM-based correction using data from this directory
        em_correction_threshold: Radius threshold for EM correction (default: 0.4 μm)
        em_correction_scale_range: Range for computing scale factor (default: 0.5-1.0 μm)

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
    first_edges, first_centers, _ = rediscretize_histogram(
        bin_edges_orig, counts_matrix_orig[0], bin_width
    )
    n_bins = len(first_centers)
    counts_matrix = np.zeros((n_rois, n_bins), dtype=float)

    for i in range(n_rois):
        _, _, counts_matrix[i] = rediscretize_histogram(
            bin_edges_orig, counts_matrix_orig[i], bin_width
        )

    sample_counts = counts_matrix.sum(axis=1)
    pooled_counts = counts_matrix.sum(axis=0)
    total_count = int(pooled_counts.sum())

    logger.info(f"Human CC: {total_count:,} total axons, {n_bins} bins")

    # Apply EM correction if requested (to pooled counts only for display)
    if em_correction_dir is not None:
        logger.info(f"Loading EM data from {em_correction_dir}")
        em_radii = load_em_radii(em_correction_dir)
        logger.info(f"Loaded {len(em_radii)} EM radii")
        pooled_counts = apply_em_correction(
            first_centers, first_edges, pooled_counts, em_radii,
            threshold=em_correction_threshold,
            scale_range=em_correction_scale_range
        )
        total_count = int(pooled_counts.sum())

        # Also apply to per-ROI counts (for r_eff computation)
        for i in range(n_rois):
            counts_matrix[i] = apply_em_correction(
                first_centers, first_edges, counts_matrix[i], em_radii,
                threshold=em_correction_threshold,
                scale_range=em_correction_scale_range
            )
        sample_counts = counts_matrix.sum(axis=1)

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

def load_radii_from_npz(npz_file: Path) -> np.ndarray:
    """Load all_radii_um from NPZ file."""
    data = np.load(npz_file, allow_pickle=True)
    if 'all_radii_um' in data:
        return data['all_radii_um']
    else:
        raise ValueError(f"NPZ file must contain 'all_radii_um'. Found: {list(data.keys())}")


def radii_to_histogram(
    radii: np.ndarray,
    bin_width: float,
    r_max: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert radius array to histogram."""
    bin_edges = np.arange(0, r_max + bin_width, bin_width)
    counts, _ = np.histogram(radii, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_edges, bin_centers, counts


def load_rat_data(
    data_dir: Path,
    bin_width: float = DEFAULT_BIN_WIDTH,
    r_max: float = 3.0,
    min_axons: int = 1000
) -> Tuple[HistogramData, PerSampleHistogramData]:
    """
    Load rat LM data from all NPZ files, split by population (CC and CG).

    Each volume has 2 populations (CC and CG), giving 10 volumes × 2 = 20 ROIs.
    Only includes ROIs with at least min_axons axons.

    Returns:
        Tuple of (pooled HistogramData, per-ROI PerSampleHistogramData)
    """
    import json

    npz_files = sorted(data_dir.glob('*_axon_profiles.npz'))
    if not npz_files:
        raise ValueError(f"No *_axon_profiles.npz files found in {data_dir}")

    n_volumes = len(npz_files)
    logger.info(f"Rat: found {n_volumes} NPZ files")

    # Determine bin structure
    bin_edges = np.arange(0, r_max + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_centers)

    # Load all volumes, split by population (CC and CG)
    all_counts = []
    all_names = []
    all_radii = []

    for npz_file in npz_files:
        volume_name = npz_file.stem.replace('_axon_profiles', '')
        pop_file = npz_file.parent / f"{volume_name}_populations.json"

        # Load axon data
        data = np.load(npz_file, allow_pickle=True)
        labels = data['labels']
        radii_profiles = data['radii_profiles_um']

        if pop_file.exists():
            # Split by population
            with open(pop_file) as f:
                pop_data = json.load(f)

            for pop in pop_data['populations']:
                pop_name = pop['name'].upper()  # CC or CG
                n_axons = pop.get('n_axons', len(pop['axon_labels']))

                # Skip ROIs with too few axons
                if n_axons < min_axons:
                    logger.debug(f"Skipping {volume_name}_{pop_name}: only {n_axons} axons (min: {min_axons})")
                    continue

                pop_labels = set(pop['axon_labels'])

                # Get radii for this population
                mask = np.isin(labels, list(pop_labels))
                if mask.sum() > 0:
                    pop_radii = np.concatenate([radii_profiles[i] for i in np.where(mask)[0]])
                    counts, _ = np.histogram(pop_radii, bins=bin_edges)
                    all_counts.append(counts)
                    all_names.append(f"{volume_name}_{pop_name}")
                    all_radii.append(pop_radii)
        else:
            # No population file - use all radii
            radii = data['all_radii_um']
            counts, _ = np.histogram(radii, bins=bin_edges)
            all_counts.append(counts)
            all_names.append(volume_name)
            all_radii.append(radii)

    n_rois = len(all_counts)
    counts_matrix = np.array(all_counts, dtype=float)
    sample_counts = counts_matrix.sum(axis=1)
    pooled_counts = counts_matrix.sum(axis=0)
    total_count = int(pooled_counts.sum())

    logger.info(f"Rat: {n_rois} ROIs (CC+CG) with ≥{min_axons} axons, {total_count:,} total radii, {n_bins} bins")

    pooled = HistogramData(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        counts=pooled_counts,
        n_samples=n_rois,
        total_count=total_count,
        name="Rat WM"
    )

    per_sample = PerSampleHistogramData(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        counts_matrix=counts_matrix,
        n_samples=n_rois,
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
    ('gamma', stats.gamma),              # Gamma
]

# Display names matching Sepehrband et al. (2016) - single line for legends
DIST_DISPLAY_NAMES = {
    'genextreme': 'Gen. Ext. Value',
    'lognorm': 'Log Normal',
    'invgauss': 'Inverse Gaussian',
    'fisk': 'Log Logistic',
    'fatiguelife': 'Birnbaum-Saunders',
    'gamma': 'Gamma',
    'nakagami': 'Nakagami',
    'weibull_min': 'Weibull',
    'rayleigh': 'Rayleigh',
    'rice': 'Rician',
    'genpareto': 'Gen. Pareto',
    'expon': 'Exponential',
}

# Multi-line display names for y-axis labels in bottom row plots
DIST_DISPLAY_NAMES_MULTILINE = {
    'genextreme': 'Gen. Ext.\nValue',
    'lognorm': 'Log\nNormal',
    'invgauss': 'Inverse\nGaussian',
    'fisk': 'Log\nLogistic',
    'fatiguelife': 'Birnbaum-\nSaunders',
    'gamma': 'Gamma',
    'nakagami': 'Nakagami',
    'weibull_min': 'Weibull',
    'rayleigh': 'Rayleigh',
    'rice': 'Rician',
    'genpareto': 'Gen.\nPareto',
    'expon': 'Exponential',
}


def get_display_name(scipy_name: str, multiline: bool = False) -> str:
    """Get display name for a distribution."""
    if multiline:
        return DIST_DISPLAY_NAMES_MULTILINE.get(scipy_name, scipy_name)
    return DIST_DISPLAY_NAMES.get(scipy_name, scipy_name)


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

    # r_arith = E[r]
    r_arith = np.sum(r * probs)

    # r_eff = (E[r^6] / E[r^2])^(1/4)
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

    # Use numerical integration over fine grid
    r = np.linspace(0.001, r_max, n_points)
    dr = r[1] - r[0]

    try:
        pdf = dist.pdf(r, *shape_params, loc=loc, scale=scale)
        pdf = np.maximum(pdf, 0)  # Ensure non-negative

        # Normalize (in case of truncation effects)
        norm = np.sum(pdf) * dr
        if norm < 0.01:  # Distribution mostly outside range
            return np.nan, np.nan
        pdf = pdf / norm

        # r_arith = E[r]
        r_arith = np.sum(r * pdf) * dr

        # r_eff = (E[r^6] / E[r^2])^(1/4)
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

    # Deterministic seed based on distribution name (hash() varies across runs)
    # Save and restore random state to avoid corrupting caller's random sequence
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

        # Wasserstein distance = integral of |CDF_empirical - CDF_fitted|
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
        np.random.set_state(saved_state)  # Restore random state
        return result

    except (ValueError, RuntimeError) as e:
        logger.debug(f"Failed to fit {dist_name}: {e}")
        np.random.set_state(saved_state)  # Restore random state
        return None


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
                r_arith, r_eff = compute_distribution_radii(dist, result.params)
                all_r_arith[dist_idx, sample_idx] = r_arith
                all_r_eff[dist_idx, sample_idx] = r_eff

    # Aggregate AIC
    summed_aic = np.nansum(all_aics, axis=1)
    n_successful_fits = np.sum(~np.isnan(all_aics), axis=1)

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

    # Aggregate r_arith and r_eff (mean and std across samples)
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

    # Fit pooled data for PDF plotting
    pooled_results = fit_all_distributions(pooled_data)

    # Create dist_name_to_idx mapping (original order, not sorted)
    dist_name_to_idx = {name: idx for idx, (name, _) in enumerate(CANDIDATE_DISTRIBUTIONS)}

    # Create win_rate dict
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
    'gamma': '#008080',           # Teal (distinct from brown, avoids red/blue)
    'fisk': '#bcbd22',            # Olive
    'nakagami': '#e377c2',        # Pink
    'weibull_min': '#7f7f7f',     # Gray
    'rayleigh': '#98df8a',        # Light green
    'rice': '#c49c94',            # Tan
    'genpareto': '#f7b6d2',       # Light pink
    'expon': '#c7c7c7',           # Light gray
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


def _plot_pooled_pdf_with_fits(
    ax: plt.Axes,
    hist_data: HistogramData,
    fit_results: List[FitResult],
    species_color: str,
    inset_xlim: Tuple[float, float] = (1.0, 3.0)
) -> None:
    """Plot pooled histogram with all fitted PDFs in distribution colors."""
    bin_width = np.diff(hist_data.bin_edges).mean()
    density = hist_data.counts / (hist_data.total_count * bin_width)

    # Plot histogram in species color (no label - will use custom legend entry)
    ax.bar(hist_data.bin_centers, density, width=bin_width * 0.9,
           alpha=0.4, color=species_color, edgecolor='white', linewidth=0.5,
           zorder=1)

    # Plot all fitted distributions in their colors
    for result in fit_results:
        color = get_dist_color(result.distribution_name)
        display_name = get_display_name(result.distribution_name)
        ax.plot(hist_data.bin_centers, result.pdf_values, '-',
                color=color, linewidth=1.5, label=display_name, zorder=2)

    ax.set_xlim(0, PLOT_XLIM_MAX)
    ax.set_ylim(0, None)
    ax.set_xlabel('Axon radius [μm]', fontsize=settings.fonts['label_size'])
    ax.set_ylabel(r'Probability density [μm$^{-1}$]', fontsize=settings.fonts['label_size'])
    # Title will be added externally
    ax.tick_params(labelsize=settings.fonts['tick_size'])

    # Add tail inset - larger, using most of plot area with narrow margin
    # Find appropriate y-limit for inset (max density in tail region)
    tail_mask = hist_data.bin_centers >= inset_xlim[0]
    tail_density = density[tail_mask]
    tail_y_max = tail_density.max() * 1.2 if tail_density.max() > 0 else 0.1

    ax_inset = ax.inset_axes([0.32, 0.32, 0.66, 0.66])  # [x, y, width, height]
    ax_inset.bar(hist_data.bin_centers, density, width=bin_width * 0.9,
                 alpha=0.4, color=species_color, edgecolor='white', linewidth=0.3)
    for result in fit_results:
        color = get_dist_color(result.distribution_name)
        ax_inset.plot(hist_data.bin_centers, result.pdf_values, '-',
                      color=color, linewidth=1.5)
    ax_inset.set_xlim(*inset_xlim)
    ax_inset.set_ylim(0, tail_y_max)
    ax_inset.tick_params(labelsize=settings.fonts['tick_size'] - 2)
    ax_inset.set_xlabel('')
    ax_inset.set_ylabel('')
    # Add subtle border
    for spine in ax_inset.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(0.8)


def create_combined_figure(
    human_pooled: HistogramData,
    human_largest: HistogramData,
    human_metrics: AggregatedMetrics,
    rat_pooled: HistogramData,
    rat_largest: HistogramData,
    rat_metrics: AggregatedMetrics,
    human_per_sample: PerSampleHistogramData,
    rat_per_sample: PerSampleHistogramData,
    output_file: Path,
    top_n: int = 5
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

    # (a) Rat pooled PDF with all fits
    _plot_pooled_pdf_with_fits(
        ax_a, rat_pooled, rat_metrics.pooled_results,
        species_color=RAT_COLOR,
        inset_xlim=(0.5, 1.2)  # Rat tail starts earlier
    )

    # (b) Human pooled PDF with all fits
    _plot_pooled_pdf_with_fits(
        ax_b, human_pooled, human_metrics.pooled_results,
        species_color=HUMAN_COLOR,
        inset_xlim=(1.0, 3.0)  # Human tail
    )

    # Create shared legend above panels a-b
    # Get handles and labels from ax_a (distribution fits only, no data)
    handles, labels = ax_a.get_legend_handles_labels()

    # Create custom half-blue/half-red patch for empirical data
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.legend_handler import HandlerBase

    class SplitColorHandler(HandlerBase):
        """Custom handler for split-color rectangle."""
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            # Create two rectangles side by side (Rat first, then Human)
            from matplotlib.patches import Rectangle
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
    from matplotlib.patches import Patch
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
    svg_file = output_file.with_suffix('.svg')
    plt.savefig(svg_file, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved figure to {output_file} and {svg_file}")


def _plot_histogram_illustrative(
    ax: plt.Axes,
    hist_data: HistogramData,
    fit_results: List[FitResult],
    distributions: List[str] = ['genextreme', 'gamma', 'lognorm']
) -> None:
    """Plot histogram with selected fitted PDFs (illustrative panel)."""
    bin_width = np.diff(hist_data.bin_edges).mean()
    density = hist_data.counts / (hist_data.total_count * bin_width)

    ax.bar(hist_data.bin_centers, density, width=bin_width * 0.9,
           alpha=0.6, color='gray',
           edgecolor='white', linewidth=0.5,
           label='Data')

    # Plot only specified distributions
    for result in fit_results:
        if result.distribution_name in distributions:
            color = get_dist_color(result.distribution_name)
            display_name = get_display_name(result.distribution_name)
            ax.plot(hist_data.bin_centers, result.pdf_values, '-',
                    color=color, linewidth=2, label=display_name)

    ax.set_xlim(0, PLOT_XLIM_MAX)
    ax.set_xlabel('Axon radius [μm]', fontsize=settings.fonts['label_size'])
    ax.set_ylabel('Probability density', fontsize=settings.fonts['label_size'])
    ax.legend(fontsize=settings.fonts['legend_size'], loc='upper right')
    ax.tick_params(labelsize=settings.fonts['tick_size'])


def _plot_summed_delta_aic(
    ax: plt.Axes,
    human_metrics: AggregatedMetrics,
    rat_metrics: AggregatedMetrics,
    human_color: str,
    rat_color: str,
    human_total_count: int,
    rat_total_count: int
) -> None:
    """Plot summed delta AIC dot plot with both species on same axes, normalized by observation count."""
    # Get distribution names (use human order - sorted by AIC)
    names = human_metrics.distribution_names
    display_names = [get_display_name(n) for n in names]

    # Get summed AIC for each species and compute delta from minimum
    human_aic = np.array([
        human_metrics.summed_aic[human_metrics.distribution_names.index(n)]
        if n in human_metrics.distribution_names else np.nan
        for n in names
    ])
    rat_aic = np.array([
        rat_metrics.summed_aic[rat_metrics.distribution_names.index(n)]
        if n in rat_metrics.distribution_names else np.nan
        for n in names
    ])

    # Delta AIC from minimum, normalized by total observation count (per million obs)
    human_delta = (human_aic - np.nanmin(human_aic)) / human_total_count * 1e6
    rat_delta = (rat_aic - np.nanmin(rat_aic)) / rat_total_count * 1e6

    y_pos = np.arange(len(names))

    # Plot dots (rat first so human is on top)
    ax.scatter(rat_delta, y_pos, color=rat_color, s=120, marker='s',
               label='Rat', zorder=3, edgecolor='white', linewidth=0.5)
    ax.scatter(human_delta, y_pos, color=human_color, s=120, marker='d',
               label='Human', zorder=4, edgecolor='white', linewidth=0.5)

    # Connect dots with lines
    for i in range(len(names)):
        ax.plot([human_delta[i], rat_delta[i]], [y_pos[i], y_pos[i]],
                color='gray', linewidth=1, alpha=0.5, zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=settings.fonts['tick_size'])
    ax.set_xlabel(r'$\Delta$AIC per $10^6$ obs.', fontsize=settings.fonts['label_size'])
    # Add margin on left for markers at 0
    x_max = max(np.nanmax(human_delta), np.nanmax(rat_delta))
    ax.set_xlim(-x_max * 0.05, x_max * 1.05)
    ax.set_ylim(len(names) - 0.5, -2.0)  # Inverted, with extra padding at top
    ax.legend(loc='upper right', fontsize=settings.fonts['legend_size'], ncol=2,
              frameon=True, edgecolor='gray', facecolor='white', framealpha=1.0)
    ax.tick_params(labelsize=settings.fonts['tick_size'])


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

    # Get win rates for each species
    human_win = np.array([human_metrics.win_rate.get(n, 0) for n in names])
    rat_win = np.array([rat_metrics.win_rate.get(n, 0) for n in names])

    y_spacing = 1.3  # Spacing between distributions
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
    ax.set_xlabel('Win rate', fontsize=settings.fonts['label_size'])
    ax.set_xlim(-0.05, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
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

    y_spacing = 1.3  # Spacing between distributions
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
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerBase

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


def _compute_gev_quantiles(
    per_sample_data: PerSampleHistogramData,
    quantile_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical and GEV-fitted quantiles for each sample.

    Returns:
        empirical_quantiles: (n_samples, n_quantiles)
        gev_quantiles: (n_samples, n_quantiles)
    """
    n_samples = per_sample_data.n_samples
    n_q = len(quantile_points)

    empirical_q = np.full((n_samples, n_q), np.nan)
    gev_q = np.full((n_samples, n_q), np.nan)

    for i in range(n_samples):
        counts = per_sample_data.counts_matrix[i]
        total = counts.sum()
        if total < 100:
            continue

        # Empirical quantiles
        cdf = np.cumsum(counts) / total
        empirical_q[i] = np.interp(quantile_points, cdf, per_sample_data.bin_centers)

        # GEV fit
        result = fit_distribution_mle(
            'genextreme', stats.genextreme,
            per_sample_data.bin_centers, per_sample_data.bin_edges, counts
        )
        if result is not None:
            shape_params, loc, scale = _unpack_params(result.params)
            gev_q[i] = stats.genextreme.ppf(quantile_points, *shape_params, loc=loc, scale=scale)

    return empirical_q, gev_q


def _plot_tail_deviation(
    ax: plt.Axes,
    human_per_sample: PerSampleHistogramData,
    rat_per_sample: PerSampleHistogramData,
    human_color: str,
    rat_color: str
) -> None:
    """Plot tail deviation: (GEV fitted - empirical) quantiles vs empirical quantile."""
    quantile_points = np.linspace(0.01, 0.99, 50)

    # Compute quantiles for both species
    human_emp, human_gev = _compute_gev_quantiles(human_per_sample, quantile_points)
    rat_emp, rat_gev = _compute_gev_quantiles(rat_per_sample, quantile_points)

    # Compute deviation: fitted - empirical
    human_dev = human_gev - human_emp  # (n_samples, n_quantiles)
    rat_dev = rat_gev - rat_emp

    # Median empirical quantile across samples (x-axis)
    human_emp_median = np.nanmedian(human_emp, axis=0)
    rat_emp_median = np.nanmedian(rat_emp, axis=0)

    # Median and IQR of deviation
    human_dev_median = np.nanmedian(human_dev, axis=0)
    human_dev_lo = np.nanpercentile(human_dev, 25, axis=0)
    human_dev_hi = np.nanpercentile(human_dev, 75, axis=0)

    rat_dev_median = np.nanmedian(rat_dev, axis=0)
    rat_dev_lo = np.nanpercentile(rat_dev, 25, axis=0)
    rat_dev_hi = np.nanpercentile(rat_dev, 75, axis=0)

    # Plot
    ax.fill_between(human_emp_median, human_dev_lo, human_dev_hi,
                    alpha=0.2, color=human_color)
    ax.plot(human_emp_median, human_dev_median, color=human_color,
            linewidth=2, label='Human CC')

    ax.fill_between(rat_emp_median, rat_dev_lo, rat_dev_hi,
                    alpha=0.2, color=rat_color)
    ax.plot(rat_emp_median, rat_dev_median, color=rat_color,
            linewidth=2, label='Rat WM')

    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Empirical quantile [μm]', fontsize=settings.fonts['label_size'])
    ax.set_ylabel('GEV − Empirical [μm]', fontsize=settings.fonts['label_size'])
    ax.legend(loc='best', fontsize=settings.fonts['legend_size'])
    ax.tick_params(labelsize=settings.fonts['tick_size'])


def _plot_dumbbell(
    ax: plt.Axes,
    metrics: AggregatedMetrics,
    title: str,
    color: str
) -> None:
    """Plot dumbbell chart showing r_arith and r_eff bias (%) per distribution."""
    names = metrics.distribution_names
    display_names = [get_display_name(n) for n in names]

    y_pos = np.arange(len(names))

    # Get r_arith and r_eff values
    r_arith = np.array([metrics.r_arith_mean.get(n, np.nan) for n in names])
    r_eff = np.array([metrics.r_eff_mean.get(n, np.nan) for n in names])

    # Empirical values
    emp_r_arith = metrics.empirical_r_arith_mean
    emp_r_eff = metrics.empirical_r_eff_mean

    # Convert to percentage bias: (fitted - empirical) / empirical * 100
    r_arith_bias = (r_arith - emp_r_arith) / emp_r_arith * 100
    r_eff_bias = (r_eff - emp_r_eff) / emp_r_eff * 100

    # Clip extreme values for visualization
    x_max = 150  # Cap at 150%
    x_min = -50  # Allow negative bias
    r_eff_bias_clipped = np.clip(r_eff_bias, x_min, x_max)

    # Plot connecting lines (dumbbells)
    for i in range(len(names)):
        if not np.isnan(r_arith_bias[i]) and not np.isnan(r_eff_bias[i]):
            ax.plot([r_arith_bias[i], r_eff_bias_clipped[i]], [y_pos[i], y_pos[i]],
                    color=color, linewidth=2, alpha=0.6, zorder=1)

    # Plot dots
    ax.scatter(r_arith_bias, y_pos, color=color, s=60, marker='o',
               label=r'$\bar{r}$', zorder=3, edgecolor='white', linewidth=0.5)
    ax.scatter(r_eff_bias_clipped, y_pos, color=color, s=60, marker='s',
               label=r'$r_{\mathrm{MRI}}$', zorder=3, edgecolor='white', linewidth=0.5)

    # Add arrows for clipped values
    for i in range(len(names)):
        if r_eff_bias[i] > x_max:
            ax.annotate('', xy=(x_max, y_pos[i]), xytext=(x_max - 10, y_pos[i]),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # Add zero reference line (empirical)
    ax.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=settings.fonts['tick_size'])
    ax.set_xlabel('Bias [%]', fontsize=settings.fonts['label_size'])
    ax.set_xlim(x_min, x_max)
    ax.invert_yaxis()
    ax.legend(loc='lower right', fontsize=settings.fonts['legend_size'] - 1)
    ax.tick_params(labelsize=settings.fonts['tick_size'])
    ax.text(0.02, 0.98, title, transform=ax.transAxes,
            fontsize=settings.fonts['label_size'], fontweight='bold',
            va='top', ha='left')


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

    y_spacing = 1.3  # Spacing between distributions
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
        x_lim = 50  # ±50%

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

    # Use renamed variable for annotation loop compatibility
    violin_width = box_width
    violin_data_human = box_data_human
    violin_data_rat = box_data_rat

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
    from matplotlib.patches import Patch, FancyBboxPatch
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerBase

    class BoxplotHandler(HandlerBase):
        """Custom handler that draws a mini boxplot with whisker caps."""
        def __init__(self, facecolor, edgecolor):
            self.facecolor = facecolor
            self.edgecolor = edgecolor
            super().__init__()

        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            from matplotlib.patches import Rectangle
            from matplotlib.lines import Line2D
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
        ax.set_xticks([-5, -2.5, 0, 2.5, 5])
    else:  # r_eff
        ax.set_xticks([-50, -25, 0, 25, 50])
    ax.set_ylim(y_pos[-1] + 0.5, -1.5)  # Inverted, with extra padding at top
    ax.tick_params(labelsize=settings.fonts['tick_size'])


def _plot_histogram_with_fits(
    ax: plt.Axes,
    hist_data: HistogramData,
    fit_results: List[FitResult],
    title: str,
    subtitle: str,
    top_n: int = 5
) -> None:
    """Plot histogram with fitted PDFs overlaid."""
    bin_width = np.diff(hist_data.bin_edges).mean()
    density = hist_data.counts / (hist_data.total_count * bin_width)

    ax.bar(hist_data.bin_centers, density, width=bin_width * 0.9,
           alpha=settings.histogram['alpha'], color='steelblue',
           edgecolor=settings.histogram['edgecolor'], linewidth=0.5,
           label='Data')

    for result in fit_results[:top_n]:
        color = get_dist_color(result.distribution_name)
        display_name = get_display_name(result.distribution_name)
        ax.plot(hist_data.bin_centers, result.pdf_values, '-',
                color=color, linewidth=2, label=display_name)

    ax.set_xlim(0, PLOT_XLIM_MAX)
    ax.set_xlabel('Radius (um)', fontsize=settings.fonts['label_size'])
    ax.set_ylabel('Probability density', fontsize=settings.fonts['label_size'])
    ax.set_title(title, fontsize=settings.fonts['title_size'], fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add subtitle with sample info
    ax.text(0.98, 0.85, subtitle, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def _plot_aic_comparison(
    ax: plt.Axes,
    metrics: AggregatedMetrics,
    title: str
) -> None:
    """Plot AIC comparison bar chart."""
    names = metrics.distribution_names
    display_names = [get_display_name(n) for n in names]
    aics = metrics.summed_aic
    delta_aics = aics - aics.min()

    colors = [get_dist_color(name) for name in names]
    y_pos = np.arange(len(names))

    ax.barh(y_pos, delta_aics, color=colors, alpha=0.8,
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=10)
    ax.set_xlabel('Delta AIC (relative to best)', fontsize=settings.fonts['label_size'])
    ax.set_title(title, fontsize=settings.fonts['label_size'], fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Add best model annotation
    best_name = display_names[0]
    ax.text(0.98, 0.02, f'Best: {best_name}', transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            fontweight='bold')


def _plot_radius_comparison(
    ax: plt.Axes,
    metrics: AggregatedMetrics,
    radius_type: str,
    title: str,
    xlim: float = 100
) -> None:
    """Plot radius bias bar chart (r_arith or r_eff) as percent error centered at zero."""
    names = metrics.distribution_names
    display_names = [get_display_name(n) for n in names]

    if radius_type == 'r_arith':
        means = np.array([metrics.r_arith_mean.get(n, np.nan) for n in names])
        stds = np.array([metrics.r_arith_std.get(n, np.nan) for n in names])
        empirical_mean = metrics.empirical_r_arith_mean
        empirical_std = metrics.empirical_r_arith_std
    else:  # r_eff
        means = np.array([metrics.r_eff_mean.get(n, np.nan) for n in names])
        stds = np.array([metrics.r_eff_std.get(n, np.nan) for n in names])
        empirical_mean = metrics.empirical_r_eff_mean
        empirical_std = metrics.empirical_r_eff_std

    # Compute percent bias: (fitted - empirical) / empirical * 100
    bias_pct = (means - empirical_mean) / empirical_mean * 100
    std_pct = stds / empirical_mean * 100
    empirical_std_pct = empirical_std / empirical_mean * 100

    colors = [get_dist_color(name) for name in names]
    y_pos = np.arange(len(names))

    # Plot bars with error bars centered at zero
    ax.barh(y_pos, bias_pct, xerr=std_pct, color=colors, alpha=0.8,
            edgecolor='black', linewidth=0.5,
            capsize=3, error_kw={'elinewidth': 1, 'capthick': 1})

    # Add zero reference line
    ax.axvline(0, color='black', linestyle='-', linewidth=1.5)

    # Add empirical std as shaded region around zero
    ax.axvspan(-empirical_std_pct, empirical_std_pct, alpha=0.2, color='gray',
               label=f'Empirical: {empirical_mean:.3f}±{empirical_std_pct:.0f}%')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=10)
    ax.set_xlabel('Bias (%)', fontsize=settings.fonts['label_size'])
    ax.set_title(title, fontsize=settings.fonts['label_size'], fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(fontsize=7, loc='lower right')

    # Fixed symmetric range for comparability
    ax.set_xlim(-xlim, xlim)


def subsample_and_fit(
    per_sample_data: PerSampleHistogramData,
    n_subsamples: int = 50,
    subsample_size: int = 1000,
    seed: int = 42
) -> Dict:
    """
    Subsample from each ROI/volume and compute r_arith/r_eff.

    For each subsample:
    - Raw: compute r_arith/r_eff directly from subsampled radii
    - Fitted: fit each distribution and compute r_arith/r_eff from fit

    Returns dict with:
    - whole_section_r_arith: (n_samples,) whole section values
    - whole_section_r_eff: (n_samples,) whole section values
    - subsampled: dict[method_name] -> dict with r_arith and r_eff arrays (n_samples, n_subsamples)
    """
    np.random.seed(seed)

    n_samples = per_sample_data.n_samples
    bin_centers = per_sample_data.bin_centers
    bin_edges = per_sample_data.bin_edges
    bin_width = np.diff(bin_edges).mean()

    # Method names: Raw + all distributions
    method_names = ['Raw'] + [name for name, _ in CANDIDATE_DISTRIBUTIONS]

    # Initialize storage
    whole_r_arith = np.full(n_samples, np.nan)
    whole_r_eff = np.full(n_samples, np.nan)

    subsampled = {
        name: {
            'r_arith': np.full((n_samples, n_subsamples), np.nan),
            'r_eff': np.full((n_samples, n_subsamples), np.nan)
        }
        for name in method_names
    }

    logger.info(f"Subsampling {n_subsamples}x{subsample_size} from {n_samples} samples...")

    for sample_idx in range(n_samples):
        counts = per_sample_data.counts_matrix[sample_idx]
        total = counts.sum()
        if total < subsample_size:
            continue

        # Whole section r_arith and r_eff
        whole_r_arith[sample_idx], whole_r_eff[sample_idx] = compute_empirical_radii(
            bin_centers, counts
        )

        # Generate pseudo-radii from histogram for subsampling
        probs = counts / total

        for sub_idx in range(n_subsamples):
            # Subsample bin indices and add jitter
            sampled_bins = np.random.choice(len(bin_centers), size=subsample_size, p=probs)
            jitter = np.random.uniform(-bin_width/2, bin_width/2, subsample_size)
            radii = bin_centers[sampled_bins] + jitter
            radii = radii[radii > 0]

            if len(radii) < 100:
                continue

            # Raw: compute directly from subsampled radii
            raw_r_arith = np.mean(radii)
            r2 = np.mean(radii**2)
            r6 = np.mean(radii**6)
            raw_r_eff = (r6 / r2) ** 0.25 if r2 > 0 else np.nan

            subsampled['Raw']['r_arith'][sample_idx, sub_idx] = raw_r_arith
            subsampled['Raw']['r_eff'][sample_idx, sub_idx] = raw_r_eff

            # Fitted distributions
            sub_counts, _ = np.histogram(radii, bins=bin_edges)

            for dist_name, dist in CANDIDATE_DISTRIBUTIONS:
                result = fit_distribution_mle(
                    dist_name, dist, bin_centers, bin_edges, sub_counts
                )
                if result is not None:
                    r_arith, r_eff = compute_distribution_radii(dist, result.params)
                    subsampled[dist_name]['r_arith'][sample_idx, sub_idx] = r_arith
                    subsampled[dist_name]['r_eff'][sample_idx, sub_idx] = r_eff

    return {
        'whole_section_r_arith': whole_r_arith,
        'whole_section_r_eff': whole_r_eff,
        'subsampled': subsampled,
        'method_names': method_names
    }


def create_scatter_figure(
    subsampling_data: Dict,
    output_file: Path,
    dataset_name: str = "Human CC"
) -> None:
    """Create scatter figure: subsampled vs whole section for one dataset.

    Layout: 2 rows (r_arith, r_eff) x 7 columns (Raw + 6 distributions)
    """
    method_names = subsampling_data['method_names']
    n_methods = len(method_names)

    fig, axes = plt.subplots(2, n_methods, figsize=(2.5 * n_methods, 5))

    for col_idx, method_name in enumerate(method_names):
        display_name = get_display_name(method_name) if method_name != 'Raw' else 'Raw'

        # Row 0: r_arith
        _plot_subsample_scatter(
            axes[0, col_idx],
            subsampling_data['whole_section_r_arith'],
            subsampling_data['subsampled'][method_name]['r_arith'],
            title=display_name,
            ylabel=r"Subsample $r_{arith}$ ($\mu$m)" if col_idx == 0 else None
        )

        # Row 1: r_eff
        _plot_subsample_scatter(
            axes[1, col_idx],
            subsampling_data['whole_section_r_eff'],
            subsampling_data['subsampled'][method_name]['r_eff'],
            title="",
            ylabel=r"Subsample $r_{eff}$ ($\mu$m)" if col_idx == 0 else None,
            xlabel=r"Whole section ($\mu$m)"
        )

    # Add dataset name as suptitle
    fig.suptitle(dataset_name, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()
    logger.info(f"Saved scatter figure to {output_file}")


def _plot_subsample_scatter(
    ax: plt.Axes,
    whole_section: np.ndarray,
    subsampled: np.ndarray,
    title: str,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None
) -> None:
    """Plot subsampled values vs whole section values with error bars."""
    # Get valid samples
    valid_mask = ~np.isnan(whole_section)
    if valid_mask.sum() == 0:
        return

    whole_valid = whole_section[valid_mask]
    sub_valid = subsampled[valid_mask, :]  # (n_valid_samples, n_subsamples)

    # Compute mean and std across subsamples for each ROI/volume
    sub_mean = np.nanmean(sub_valid, axis=1)
    sub_std = np.nanstd(sub_valid, axis=1)

    # Remove any remaining NaN
    valid = ~np.isnan(sub_mean)
    x = whole_valid[valid]
    y_mean = sub_mean[valid]
    y_std = sub_std[valid]

    if len(x) == 0:
        return

    # Get range
    all_vals = np.concatenate([x, y_mean + y_std, y_mean - y_std])
    r_min = np.nanmin(all_vals) * 0.9
    r_max = np.nanmax(all_vals) * 1.1

    # Identity line
    ax.plot([r_min, r_max], [r_min, r_max], 'k--', linewidth=1.5, zorder=1)

    # Error bar plot
    ax.errorbar(x, y_mean, yerr=y_std, fmt='o', color='steelblue',
                markersize=5, capsize=3, capthick=1, elinewidth=1,
                alpha=0.7, zorder=2)

    ax.set_xlim(r_min, r_max)
    ax.set_ylim(r_min, r_max)
    ax.set_aspect('equal', adjustable='box')

    if title:
        ax.set_title(get_display_name(title) if title != 'Raw' else title,
                     fontsize=10, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=settings.fonts['label_size'])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=settings.fonts['label_size'])

    ax.grid(True, alpha=0.3)


def _plot_radius_scatter(
    ax: plt.Axes,
    metrics: AggregatedMetrics,
    radius_type: str,
    title: str
) -> None:
    """Plot scatter of fitted vs empirical radius for each ROI/volume."""
    if radius_type == 'r_arith':
        all_fitted = metrics.all_r_arith
        empirical = metrics.empirical_r_arith_per_sample
    else:  # r_eff
        all_fitted = metrics.all_r_eff
        empirical = metrics.empirical_r_eff_per_sample

    # Get range for identity line
    valid_empirical = empirical[~np.isnan(empirical)]
    if len(valid_empirical) == 0:
        return

    r_min = valid_empirical.min() * 0.8
    r_max = valid_empirical.max() * 1.2

    # Plot identity line
    ax.plot([r_min, r_max], [r_min, r_max], 'k--', linewidth=1.5, zorder=1)

    # Plot raw/empirical values on the identity line
    ax.scatter(
        valid_empirical, valid_empirical,
        color='black', s=50, alpha=0.8, label='Raw (empirical)',
        marker='x', linewidth=1.5, zorder=3
    )

    # Plot each distribution
    for dist_name, dist_idx in metrics.dist_name_to_idx.items():
        fitted = all_fitted[dist_idx]
        valid_mask = ~np.isnan(fitted) & ~np.isnan(empirical)

        if valid_mask.sum() == 0:
            continue

        color = get_dist_color(dist_name)
        display_name = get_display_name(dist_name)

        ax.scatter(
            empirical[valid_mask], fitted[valid_mask],
            color=color, s=30, alpha=0.6, label=display_name,
            edgecolor='white', linewidth=0.5, zorder=2
        )

    ax.set_xlim(r_min, r_max)
    ax.set_ylim(r_min, r_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'Empirical radius ($\mu$m)', fontsize=settings.fonts['label_size'])
    ax.set_ylabel(r'Fitted radius ($\mu$m)', fontsize=settings.fonts['label_size'])
    ax.set_title(title, fontsize=settings.fonts['label_size'], fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)


def create_pooled_histogram_figure(
    human_pooled: HistogramData,
    human_metrics: AggregatedMetrics,
    rat_pooled: HistogramData,
    rat_metrics: AggregatedMetrics,
    output_file: Path
) -> None:
    """Create three-panel figure: (a) human tail, (b) rat tail, (c) full distribution."""
    from matplotlib.patches import ConnectionPatch

    HUMAN_COLOR = '#1f77b4'  # Blue
    RAT_COLOR = '#d62728'    # Red

    fig = plt.figure(figsize=(7, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1], hspace=0.5, wspace=0.3)

    ax_full = fig.add_subplot(gs[0, :])        # Top (spans both columns)
    ax_human_tail = fig.add_subplot(gs[1, 0])  # Bottom left
    ax_rat_tail = fig.add_subplot(gs[1, 1])    # Bottom right

    # Set 1:1 aspect ratio for all panels
    ax_human_tail.set_box_aspect(1)
    ax_rat_tail.set_box_aspect(1)
    ax_full.set_box_aspect(1)

    # Compute densities
    human_bin_width = np.diff(human_pooled.bin_edges).mean()
    human_density = human_pooled.counts / (human_pooled.total_count * human_bin_width)
    rat_bin_width = np.diff(rat_pooled.bin_edges).mean()
    rat_density = rat_pooled.counts / (rat_pooled.total_count * rat_bin_width)

    # Get best fits
    best_fit_human = human_metrics.pooled_results[0] if human_metrics.pooled_results else None
    best_fit_rat = rat_metrics.pooled_results[0] if rat_metrics.pooled_results else None

    tail_xmin, tail_xmax = 1.0, 3.0

    # --- Top left (a): Human tail ---
    tail_mask_human = (human_pooled.bin_centers >= tail_xmin) & (human_pooled.bin_centers <= tail_xmax)
    ax_human_tail.bar(human_pooled.bin_centers[tail_mask_human], human_density[tail_mask_human],
                      width=human_bin_width * 0.9, alpha=0.5, color=HUMAN_COLOR,
                      edgecolor='white', linewidth=0.5)
    if best_fit_human:
        ax_human_tail.plot(human_pooled.bin_centers[tail_mask_human],
                           best_fit_human.pdf_values[tail_mask_human], '-',
                           color=HUMAN_COLOR, linewidth=2)
    ax_human_tail.set_xlim(tail_xmin, tail_xmax)
    ax_human_tail.set_ylim(0, None)
    ax_human_tail.set_ylabel('Prob. density', fontsize=settings.fonts['label_size'])
    ax_human_tail.set_xlabel('Radius [μm]', fontsize=settings.fonts['label_size'])
    ax_human_tail.tick_params(labelsize=settings.fonts['tick_size'])
    ax_human_tail.set_title('Human CC', fontsize=settings.fonts['label_size'], color=HUMAN_COLOR)

    # --- Top right (b): Rat tail ---
    tail_mask_rat = (rat_pooled.bin_centers >= tail_xmin) & (rat_pooled.bin_centers <= tail_xmax)
    ax_rat_tail.bar(rat_pooled.bin_centers[tail_mask_rat], rat_density[tail_mask_rat],
                    width=rat_bin_width * 0.9, alpha=0.5, color=RAT_COLOR,
                    edgecolor='white', linewidth=0.5)
    if best_fit_rat:
        ax_rat_tail.plot(rat_pooled.bin_centers[tail_mask_rat],
                         best_fit_rat.pdf_values[tail_mask_rat], '-',
                         color=RAT_COLOR, linewidth=2)
    ax_rat_tail.set_xlim(tail_xmin, tail_xmax)
    ax_rat_tail.set_ylim(0, None)
    ax_rat_tail.set_xlabel('Radius [μm]', fontsize=settings.fonts['label_size'])
    ax_rat_tail.tick_params(labelsize=settings.fonts['tick_size'])
    ax_rat_tail.set_title('Rat WM', fontsize=settings.fonts['label_size'], color=RAT_COLOR)

    # --- Bottom (c): Full distribution ---
    ax_full.bar(human_pooled.bin_centers, human_density, width=human_bin_width * 0.9,
                alpha=0.5, color=HUMAN_COLOR, edgecolor='white', linewidth=0.5,
                label=f'Human CC (n={human_pooled.total_count:,})')
    if best_fit_human:
        ax_full.plot(human_pooled.bin_centers, best_fit_human.pdf_values, '-',
                     color=HUMAN_COLOR, linewidth=2.5)

    ax_full.bar(rat_pooled.bin_centers, rat_density, width=rat_bin_width * 0.9,
                alpha=0.5, color=RAT_COLOR, edgecolor='white', linewidth=0.5,
                label=f'Rat WM (n={rat_pooled.total_count:,})')
    if best_fit_rat:
        ax_full.plot(rat_pooled.bin_centers, best_fit_rat.pdf_values, '-',
                     color=RAT_COLOR, linewidth=2.5)

    ax_full.set_xlim(0, PLOT_XLIM_MAX)
    ax_full.set_xlabel('Axon radius [μm]', fontsize=settings.fonts['label_size'])
    ax_full.set_ylabel('Probability density', fontsize=settings.fonts['label_size'])
    ax_full.legend(loc='upper right', fontsize=settings.fonts['legend_size'])
    ax_full.tick_params(labelsize=settings.fonts['tick_size'])

    # Connection lines from full plot (top) down to tail panels (bottom)
    # Human: from full plot bottom to human tail top
    con_human_left = ConnectionPatch(
        xyA=(tail_xmin, 0), coordsA=ax_full.transData,
        xyB=(tail_xmin, ax_human_tail.get_ylim()[1]), coordsB=ax_human_tail.transData,
        color='0.5', linestyle='--', linewidth=0.8
    )
    fig.add_artist(con_human_left)

    con_human_right = ConnectionPatch(
        xyA=(tail_xmax, 0), coordsA=ax_full.transData,
        xyB=(tail_xmax, ax_human_tail.get_ylim()[1]), coordsB=ax_human_tail.transData,
        color='0.5', linestyle='--', linewidth=0.8
    )
    fig.add_artist(con_human_right)

    # Rat: from full plot bottom to rat tail top
    con_rat_left = ConnectionPatch(
        xyA=(tail_xmin, 0), coordsA=ax_full.transData,
        xyB=(tail_xmin, ax_rat_tail.get_ylim()[1]), coordsB=ax_rat_tail.transData,
        color='0.5', linestyle='--', linewidth=0.8
    )
    fig.add_artist(con_rat_left)

    con_rat_right = ConnectionPatch(
        xyA=(tail_xmax, 0), coordsA=ax_full.transData,
        xyB=(tail_xmax, ax_rat_tail.get_ylim()[1]), coordsB=ax_rat_tail.transData,
        color='0.5', linestyle='--', linewidth=0.8
    )
    fig.add_artist(con_rat_right)

    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()
    logger.info(f"Saved pooled histogram figure to {output_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Combined distribution fitting for Human CC and Rat data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--human-data', type=Path, required=True,
                        help='Directory containing human CC TSV files (data/raw_LM)')
    parser.add_argument('--rat-data', type=Path, required=True,
                        help='Directory containing rat NPZ files (data/processed/LM)')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output PNG file path')
    parser.add_argument('--bin-width', type=float, default=DEFAULT_BIN_WIDTH,
                        help=f'Bin width in um (default: {DEFAULT_BIN_WIDTH})')
    parser.add_argument('--r-max', type=float, default=3.0,
                        help='Maximum radius in um (default: 3.0)')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top distributions to show in histograms (default: 5)')
    parser.add_argument('--em-correction', type=Path, default=None,
                        help='Path to EM data directory for human CC correction (e.g., data/raw_EM). '
                             'Replaces LM counts below threshold with scaled EM counts.')
    parser.add_argument('--em-correction-threshold', type=float, default=0.4,
                        help='Radius threshold for EM correction in um (default: 0.4)')
    parser.add_argument('--em-correction-scale-range', type=float, nargs=2, default=[0.5, 1.0],
                        metavar=('MIN', 'MAX'),
                        help='Radius range for computing EM/LM scale factor (default: 0.5 1.0)')

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load Human CC data
    logger.info("=" * 60)
    logger.info("Loading Human CC data...")
    human_pooled, human_per_sample = load_human_cc_data(
        args.human_data,
        args.bin_width,
        em_correction_dir=args.em_correction,
        em_correction_threshold=args.em_correction_threshold,
        em_correction_scale_range=tuple(args.em_correction_scale_range)
    )

    # Find largest ROI
    largest_roi_idx = np.argmax(human_per_sample.sample_counts)
    largest_roi_counts = human_per_sample.counts_matrix[largest_roi_idx]
    human_largest = HistogramData(
        bin_edges=human_per_sample.bin_edges,
        bin_centers=human_per_sample.bin_centers,
        counts=largest_roi_counts,
        n_samples=1,
        total_count=int(largest_roi_counts.sum()),
        name=f"Human ROI {largest_roi_idx + 1}"
    )
    logger.info(f"Largest ROI: #{largest_roi_idx + 1} with {human_largest.total_count:,} axons")

    # Load Rat data
    logger.info("=" * 60)
    logger.info("Loading Rat data...")
    rat_pooled, rat_per_sample = load_rat_data(args.rat_data, args.bin_width, args.r_max)

    # Find largest volume
    largest_vol_idx = np.argmax(rat_per_sample.sample_counts)
    largest_vol_counts = rat_per_sample.counts_matrix[largest_vol_idx]
    rat_largest = HistogramData(
        bin_edges=rat_per_sample.bin_edges,
        bin_centers=rat_per_sample.bin_centers,
        counts=largest_vol_counts,
        n_samples=1,
        total_count=int(largest_vol_counts.sum()),
        name=rat_per_sample.sample_names[largest_vol_idx]
    )
    logger.info(f"Largest volume: {rat_largest.name} with {rat_largest.total_count:,} radii")

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
    create_combined_figure(
        human_pooled, human_largest, human_metrics,
        rat_pooled, rat_largest, rat_metrics,
        human_per_sample, rat_per_sample,
        args.output, args.top_n
    )

    # Create pooled histogram figure
    pooled_hist_output = args.output.with_stem(args.output.stem + '_histograms')
    logger.info("Creating pooled histogram figure...")
    create_pooled_histogram_figure(
        human_pooled, human_metrics,
        rat_pooled, rat_metrics,
        pooled_hist_output
    )

    # Subsampling analysis
    logger.info("=" * 60)
    logger.info("Running subsampling analysis...")
    human_subsampling = subsample_and_fit(human_per_sample, n_subsamples=50, subsample_size=1000)
    rat_subsampling = subsample_and_fit(rat_per_sample, n_subsamples=50, subsample_size=1000)

    # Create scatter figures (one per dataset)
    human_scatter_output = args.output.with_stem(args.output.stem + '_scatter_human')
    logger.info("Creating Human CC scatter figure...")
    create_scatter_figure(human_subsampling, human_scatter_output, "Human CC")

    rat_scatter_output = args.output.with_stem(args.output.stem + '_scatter_rat')
    logger.info("Creating Rat WM scatter figure...")
    create_scatter_figure(rat_subsampling, rat_scatter_output, "Rat WM")

    # Save JSON
    json_file = args.output.with_suffix('.json')
    output_data = {
        'human_cc': {
            'n_rois': human_pooled.n_samples,
            'total_count': human_pooled.total_count,
            'largest_roi_idx': int(largest_roi_idx),
            'distributions': [
                {'name': name, 'summed_aic': float(human_metrics.summed_aic[i])}
                for i, name in enumerate(human_metrics.distribution_names)
            ]
        },
        'rat': {
            'n_volumes': rat_pooled.n_samples,
            'total_count': rat_pooled.total_count,
            'largest_volume': rat_largest.name,
            'distributions': [
                {'name': name, 'summed_aic': float(rat_metrics.summed_aic[i])}
                for i, name in enumerate(rat_metrics.distribution_names)
            ]
        }
    }
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Saved results to {json_file}")


if __name__ == '__main__':
    main()
