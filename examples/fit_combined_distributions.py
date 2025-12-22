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
    Load human corpus callosum histogram data (CircularEq).

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
    counts_file = data_dir / 'desc-countsCircularEq_radii.tsv'

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
    r_max: float = 3.0
) -> Tuple[HistogramData, PerSampleHistogramData]:
    """
    Load rat LM data from all NPZ files, split by population (CC and CG).

    Each volume has 2 populations (CC and CG), giving 10 volumes × 2 = 20 ROIs.

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

    logger.info(f"Rat: {n_rois} ROIs (CC+CG), {total_count:,} total radii, {n_bins} bins")

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
    ('fisk', stats.fisk),                # Log logistic
    ('gamma', stats.gamma),              # Gamma
]

# Display names matching Sepehrband et al. (2016)
DIST_DISPLAY_NAMES = {
    'genextreme': 'Gen. Extreme Value',
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


def get_display_name(scipy_name: str) -> str:
    """Get display name for a distribution."""
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
                # Compute r_arith and r_eff from fitted distribution
                r_arith, r_eff = compute_distribution_radii(dist, result.params)
                all_r_arith[dist_idx, sample_idx] = r_arith
                all_r_eff[dist_idx, sample_idx] = r_eff

    # Aggregate AIC
    summed_aic = np.nansum(all_aics, axis=1)
    n_successful_fits = np.sum(~np.isnan(all_aics), axis=1)

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
        dist_name_to_idx=dist_name_to_idx
    )


# =============================================================================
# Visualization
# =============================================================================

# Fixed colors per distribution (consistent across all panels)
DIST_COLORS = {
    'genextreme': '#e41a1c',      # Red
    'lognorm': '#377eb8',         # Blue
    'invgauss': '#4daf4a',        # Green
    'fisk': '#984ea3',            # Purple
    'fatiguelife': '#ff7f00',     # Orange
    'gamma': '#a65628',           # Brown
    'nakagami': '#f781bf',        # Pink
    'weibull_min': '#999999',     # Gray
    'rayleigh': '#66c2a5',        # Teal
    'rice': '#1b9e77',            # Dark teal
    'genpareto': '#d95f02',       # Dark orange
    'expon': '#7570b3',           # Lavender
}


def get_dist_color(dist_name: str) -> str:
    """Get fixed color for a distribution."""
    return DIST_COLORS.get(dist_name, '#333333')


def create_combined_figure(
    human_pooled: HistogramData,
    human_largest: HistogramData,
    human_metrics: AggregatedMetrics,
    rat_pooled: HistogramData,
    rat_largest: HistogramData,
    rat_metrics: AggregatedMetrics,
    output_file: Path,
    top_n: int = 5
) -> None:
    """Create 2x4 combined figure."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Fit distributions to largest samples for plotting
    human_largest_fits = fit_all_distributions(human_largest)
    rat_largest_fits = fit_all_distributions(rat_largest)

    # Row 1: Human CC
    _plot_histogram_with_fits(
        axes[0, 0], human_largest, human_largest_fits,
        title="Human Corpus Callosum",
        subtitle=f"n={human_pooled.n_samples} ROIs",
        top_n=top_n
    )

    _plot_aic_comparison(
        axes[0, 1], human_metrics,
        title="Summed AIC"
    )

    _plot_radius_comparison(
        axes[0, 2], human_metrics, 'r_arith',
        title=r"$r_{arith}$ (mean ± std)", xlim=20
    )

    _plot_radius_comparison(
        axes[0, 3], human_metrics, 'r_eff',
        title=r"$r_{eff}$ (mean ± std)", xlim=100
    )

    # Row 2: Rat WM
    _plot_histogram_with_fits(
        axes[1, 0], rat_largest, rat_largest_fits,
        title="Rat White Matter",
        subtitle=f"n={rat_pooled.n_samples} ROIs",
        top_n=top_n
    )

    _plot_aic_comparison(
        axes[1, 1], rat_metrics,
        title="Summed AIC"
    )

    _plot_radius_comparison(
        axes[1, 2], rat_metrics, 'r_arith',
        title=r"$r_{arith}$ (mean ± std)", xlim=20
    )

    _plot_radius_comparison(
        axes[1, 3], rat_metrics, 'r_eff',
        title=r"$r_{eff}$ (mean ± std)", xlim=100
    )

    # Panel labels
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    for ax, label in zip(axes.flat, labels):
        ax.text(-0.08, 1.05, label, transform=ax.transAxes,
                fontsize=settings.fonts['title_size'], fontweight='bold', va='top')

    plt.tight_layout()
    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()
    logger.info(f"Saved figure to {output_file}")


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

    # Create main figure (2x4 summary)
    logger.info("=" * 60)
    logger.info("Creating summary figure...")
    create_combined_figure(
        human_pooled, human_largest, human_metrics,
        rat_pooled, rat_largest, rat_metrics,
        args.output, args.top_n
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
