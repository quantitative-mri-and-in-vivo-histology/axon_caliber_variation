"""
Compare distribution fits: Human CC LM (raw) vs Human CC LM (EM-corrected).

Applies EM-based correction to human LM data for small radii (below threshold),
replacing LM counts with scaled EM counts, then compares distribution fits.

Creates a 2x2 figure showing:
- (a) Human LM (raw): pooled histogram with fitted PDFs
- (b) Human LM (EM-corrected): pooled histogram with fitted PDFs
- (c)-(f) Model comparison metrics

Usage:
    python scripts/exploratory/parametric_distributions/compare_human_em_corrected_fits.py \\
        --human-data data/raw/human/lm \\
        --em-data data/raw/human/em \\
        --output fig/exploratory/compare_human_em_corrected_fits.svg
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
from scipy import stats
from scipy.optimize import OptimizeWarning

# Find repo root (contains pyproject.toml)
_root = Path(__file__).resolve().parent
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from axonometry import get_plot_settings, rediscretize, compute_r_arith, compute_r_eff

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings = get_plot_settings()

# Constants
MIN_BIN_PROB = 1e-300
PLOT_XLIM_MAX = 3.0
DEFAULT_BIN_WIDTH = 0.05  # um


# =============================================================================
# EM Correction Functions
# =============================================================================

def load_em_radii(em_dir: Path, radius_column: str = 'r_circular_equivalent') -> np.ndarray:
    """Load EM radii from CSV files in em_dir."""
    radii = []
    for csv_file in sorted(em_dir.glob('*_axon_radii_rp.csv')):
        df = pd.read_csv(csv_file)
        if radius_column in df.columns:
            radii.extend(df[radius_column].values)
            logger.info(f"  Loaded {len(df)} radii from {csv_file.name}")
    return np.array(radii)


def apply_em_correction(
    bin_centers: np.ndarray,
    bin_edges: np.ndarray,
    lm_counts: np.ndarray,
    em_radii: np.ndarray,
    threshold: float = 0.4,
    scale_range: Tuple[float, float] = (0.5, 1.0)
) -> np.ndarray:
    """Apply EM-based correction to LM counts for small radii.

    For r < threshold: use scaled EM counts
    For r >= threshold: use original LM counts

    Scale factor is computed by matching total counts in scale_range.
    """
    em_counts, _ = np.histogram(em_radii, bins=bin_edges)

    scale_mask = (bin_centers >= scale_range[0]) & (bin_centers <= scale_range[1])
    em_in_range = em_counts[scale_mask].sum()
    lm_in_range = lm_counts[scale_mask].sum()

    if em_in_range < 10:
        logger.warning(f"Too few EM counts in scale range {scale_range}: {em_in_range}")
        return lm_counts

    scale_factor = lm_in_range / em_in_range
    em_scaled = em_counts * scale_factor
    corrected = np.where(bin_centers < threshold, em_scaled, lm_counts)

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
class FitResult:
    """Container for distribution fit results."""
    distribution_name: str
    n_params: int
    params: Tuple
    nll: float
    aic: float
    wasserstein: float = 0.0  # Wasserstein distance between empirical and fitted CDF
    pdf_values: np.ndarray = field(default_factory=lambda: np.array([]))
    pdf_x_fine: np.ndarray = field(default_factory=lambda: np.array([]))
    pdf_values_fine: np.ndarray = field(default_factory=lambda: np.array([]))


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
# Data Loading - EM-corrected Human LM
# =============================================================================

def load_corrected_human_data(
    lm_data_dir: Path,
    em_data_dir: Path,
    bin_width: float = DEFAULT_BIN_WIDTH,
    threshold: float = 0.4,
    scale_range: Tuple[float, float] = (0.5, 1.0),
) -> Tuple[HistogramData, PerSampleHistogramData]:
    """Load human LM data with EM-based correction for small radii."""
    # First load the raw LM data
    raw_pooled, raw_per_sample = load_human_cc_data(lm_data_dir, bin_width)

    # Load EM radii
    logger.info("Loading EM radii for correction...")
    em_radii = load_em_radii(em_data_dir)
    logger.info(f"  Loaded {len(em_radii):,} EM radii")

    # Apply correction to each ROI
    corrected_matrix = np.zeros_like(raw_per_sample.counts_matrix)
    for i in range(raw_per_sample.n_samples):
        corrected_matrix[i] = apply_em_correction(
            raw_per_sample.bin_centers, raw_per_sample.bin_edges,
            raw_per_sample.counts_matrix[i], em_radii,
            threshold=threshold, scale_range=scale_range
        )

    sample_counts = corrected_matrix.sum(axis=1)
    pooled_counts = corrected_matrix.sum(axis=0)
    total_count = int(pooled_counts.sum())

    logger.info(f"EM-corrected: {raw_per_sample.n_samples} ROIs, {total_count:,} total counts")

    pooled = HistogramData(
        bin_edges=raw_per_sample.bin_edges,
        bin_centers=raw_per_sample.bin_centers,
        counts=pooled_counts,
        n_samples=raw_per_sample.n_samples,
        total_count=total_count,
        name="Human LM (EM-corrected)"
    )

    per_sample = PerSampleHistogramData(
        bin_edges=raw_per_sample.bin_edges,
        bin_centers=raw_per_sample.bin_centers,
        counts_matrix=corrected_matrix,
        n_samples=raw_per_sample.n_samples,
        sample_counts=sample_counts,
        sample_names=raw_per_sample.sample_names
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
    'fatiguelife': 'Birnbaum-Saunders',
    'gamma': 'Gamma',
}

# Multi-line display names for y-axis labels in bottom row plots
DIST_DISPLAY_NAMES_MULTILINE = {
    'genextreme': 'Gen. Ext.\nValue',
    'lognorm': 'Log\nNormal',
    'invgauss': 'Inverse\nGaussian',
    'fatiguelife': 'Birnbaum-\nSaunders',
    'gamma': 'Gamma',
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
    return (
        compute_r_arith(counts=counts, bin_centers=bin_centers),
        compute_r_eff(counts=counts, bin_centers=bin_centers),
    )


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


def _binned_nll(theta, dist, bin_edges, counts, fix_loc):
    """Negative log-likelihood for binned data.

    Args:
        theta: Parameter vector (shape params + loc? + scale).
        dist: scipy.stats distribution object.
        bin_edges: Array of bin edges.
        counts: Observed counts per bin.
        fix_loc: If True, loc is fixed at 0 and not in theta.
    """
    # Reconstruct full parameter tuple
    if fix_loc:
        # theta = (*shape, scale)
        shape_params = theta[:-1]
        loc = 0.0
        scale = theta[-1]
    else:
        # theta = (*shape, loc, scale)
        shape_params = theta[:-2]
        loc = theta[-2]
        scale = theta[-1]

    if scale <= 0:
        return 1e20

    try:
        edge_cdfs = dist.cdf(bin_edges, *shape_params, loc=loc, scale=scale)
        if np.any(np.isnan(edge_cdfs)):
            return 1e20
        bin_probs = np.diff(edge_cdfs)
        bin_probs = np.maximum(bin_probs, MIN_BIN_PROB)
        nll = -np.dot(counts, np.log(bin_probs))
        return nll if np.isfinite(nll) else 1e20
    except Exception:
        return 1e20


N_RESTARTS = 20  # Number of random restarts for binned MLE


def _get_initial_params_multi(dist_name, dist, bin_centers, counts):
    """Get multiple initial parameter estimates from different random seeds."""
    total = counts.sum()
    probs = counts / total
    bin_width = bin_centers[1] - bin_centers[0]
    n_init = min(int(total), 10_000)

    all_params = []
    for seed in range(N_RESTARTS):
        rng = np.random.default_rng(seed)
        sampled_bins = rng.choice(len(bin_centers), size=n_init, p=probs)
        samples = bin_centers[sampled_bins] + rng.uniform(-bin_width / 2, bin_width / 2, n_init)
        samples = samples[samples > 0]

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', category=OptimizeWarning)
            try:
                if dist_name == 'genextreme':
                    params = dist.fit(samples)
                else:
                    params = dist.fit(samples, floc=0)
                all_params.append(params)
            except Exception:
                continue

    return all_params


def fit_distribution_mle(
    dist_name: str,
    dist: stats.rv_continuous,
    bin_centers: np.ndarray,
    bin_edges: np.ndarray,
    counts: np.ndarray
) -> Optional[FitResult]:
    """Fit distribution to histogram data using proper binned MLE.

    Maximizes the multinomial log-likelihood over bin probabilities
    derived from the parametric CDF. Uses multiple random restarts
    to avoid local optima.
    """
    from scipy.optimize import minimize

    total = counts.sum()
    if total == 0:
        return None

    try:
        # Get multiple initial parameter estimates
        all_init_params = _get_initial_params_multi(dist_name, dist, bin_centers, counts)
        if not all_init_params:
            return None

        fix_loc = (dist_name != 'genextreme')

        # Bounds (same for all restarts)
        shape_params_0, _, _ = _unpack_params(all_init_params[0])
        n_shape = len(shape_params_0)
        n_theta = n_shape + (1 if fix_loc else 2)
        bounds = [(None, None)] * n_theta
        if fix_loc:
            bounds[-1] = (1e-8, None)  # scale > 0
        else:
            bounds[-2] = (None, None)  # loc
            bounds[-1] = (1e-8, None)  # scale > 0

        nll_args = (dist, bin_edges, counts, fix_loc)
        best_result = None

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            for init_params in all_init_params:
                shape_params_init, loc_init, scale_init = _unpack_params(init_params)
                if fix_loc:
                    theta0 = np.array([*shape_params_init, scale_init])
                else:
                    theta0 = np.array([*shape_params_init, loc_init, scale_init])

                # Try L-BFGS-B first (fast, uses gradient)
                result = minimize(
                    _binned_nll, theta0, args=nll_args,
                    method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12}
                )

                # Fall back to Nelder-Mead if L-BFGS-B fails
                if not np.isfinite(result.fun) or result.fun >= 1e19:
                    result = minimize(
                        _binned_nll, theta0, args=nll_args,
                        method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
                    )

                if np.isfinite(result.fun) and (best_result is None or result.fun < best_result.fun):
                    best_result = result

        if best_result is None or not np.isfinite(best_result.fun) or best_result.fun >= 1e19:
            logger.debug(f"Optimization failed for {dist_name}")
            return None

        result = best_result

        # Reconstruct full params tuple
        if fix_loc:
            shape_params = tuple(result.x[:-1])
            loc = 0.0
            scale = result.x[-1]
        else:
            shape_params = tuple(result.x[:-2])
            loc = result.x[-2]
            scale = result.x[-1]

        params = (*shape_params, loc, scale)
        nll = result.fun
        k = len(result.x)  # only count free parameters (loc excluded when fixed)
        aic = 2 * k + 2 * nll

        bin_width = np.diff(bin_edges).mean()
        pdf_values = dist.pdf(bin_centers, *shape_params, loc=loc, scale=scale)

        empirical_cdf = np.cumsum(counts) / total
        fitted_cdf = dist.cdf(bin_edges[1:], *shape_params, loc=loc, scale=scale)
        wasserstein = np.sum(np.abs(empirical_cdf - fitted_cdf)) * bin_width

        x_fine = np.linspace(bin_centers[0], bin_centers[-1], 500)
        pdf_fine = dist.pdf(x_fine, *shape_params, loc=loc, scale=scale)

        return FitResult(
            distribution_name=dist_name,
            n_params=k,
            params=params,
            nll=nll,
            aic=aic,
            wasserstein=wasserstein,
            pdf_values=pdf_values,
            pdf_x_fine=x_fine,
            pdf_values_fine=pdf_fine
        )

    except (ValueError, RuntimeError) as e:
        logger.debug(f"Failed to fit {dist_name}: {e}")
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


def _plot_pooled_pdf_with_fits(
    ax: plt.Axes,
    hist_data: HistogramData,
    fit_results: List[FitResult],
    species_color: str,
    inset_xlim: Tuple[float, float] = (1.0, 3.0),
    distribution_order: List[str] = None
) -> None:
    """Plot pooled histogram with all fitted PDFs in distribution colors."""
    from matplotlib.patches import Rectangle

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
    for result in ordered_results:
        color = get_dist_color(result.distribution_name)
        ax_inset.plot(result.pdf_x_fine, result.pdf_values_fine, '-',
                      color=color, linewidth=1.5)
    ax_inset.set_xlim(*inset_xlim)
    ax_inset.set_ylim(0, tail_y_max)
    ax_inset.tick_params(labelsize=settings.fonts['tick_size'] - 2)
    ax_inset.set_xlabel('')
    ax_inset.set_ylabel('')

    # Add indicator rectangle with connector lines to inset
    ax.indicate_inset_zoom(ax_inset, edgecolor='gray', linewidth=1.5,
                           linestyle='--', alpha=0.8)


def create_combined_figure(
    human_pooled: HistogramData,
    human_metrics: AggregatedMetrics,
    corr_pooled: HistogramData,
    corr_metrics: AggregatedMetrics,
    human_per_sample: PerSampleHistogramData,
    corr_per_sample: PerSampleHistogramData,
    output_file: Path,
    top_n: int = 5
) -> None:
    """Create 6-panel figure with 2 rows.

    Row 1 (top): Pooled PDFs with all fitted distributions
        (a) Human pooled PDF with fits
        (b) Corrected pooled PDF with fits

    Row 2 (bottom): Model comparison metrics
        (c) Win rate
        (d) Wasserstein distance (with inter-ROI reference)
        (e) r_arith error
        (f) r_eff error
    """
    from matplotlib.gridspec import GridSpec

    # Species colors from settings
    HUMAN_COLOR = settings.colors['binary_a']   # Sand/tan for raw LM
    CORR_COLOR = settings.colors['binary_b']    # Dusty teal for EM-corrected

    # Create figure with GridSpec: 2 rows, top row has 2 panels, bottom has 4
    fig = plt.figure(figsize=(17, 9.5))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1], hspace=0.35, wspace=0.35)

    # Top row: 2 panels spanning 2 columns each
    ax_a = fig.add_subplot(gs[0, 0:2])  # Corrected PDF
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
    corr_inter_roi_w = compute_inter_roi_wasserstein(corr_per_sample)

    # Use human distribution order (by AIC) for consistency with bottom row
    dist_order = human_metrics.distribution_names

    # (a) Corrected pooled PDF with all fits
    _plot_pooled_pdf_with_fits(
        ax_a, corr_pooled, corr_metrics.pooled_results,
        species_color=CORR_COLOR,
        inset_xlim=(1.0, 3.0),  # Corrected tail
        distribution_order=dist_order
    )

    # (b) Human pooled PDF with all fits
    _plot_pooled_pdf_with_fits(
        ax_b, human_pooled, human_metrics.pooled_results,
        species_color=HUMAN_COLOR,
        inset_xlim=(1.0, 3.0),  # Human tail
        distribution_order=dist_order
    )

    # Create shared legend above panels a-b
    # Get handles and labels from ax_a (distribution fits only, no data)
    handles, labels = ax_a.get_legend_handles_labels()

    # Create custom half-blue/half-red patch for empirical data
    from matplotlib.legend_handler import HandlerBase

    class SplitColorHandler(HandlerBase):
        """Custom handler for split-color rectangle."""
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            # Create two rectangles side by side (Corrected first, then Raw)
            from matplotlib.patches import Rectangle
            half_width = width / 2
            left_rect = Rectangle(
                (xdescent, ydescent), half_width, height,
                facecolor=CORR_COLOR, edgecolor='white', linewidth=0.5,
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
        ax_c, human_metrics, corr_metrics,
        human_color=HUMAN_COLOR, corr_color=CORR_COLOR
    )

    # (d) Wasserstein distance with inter-ROI reference
    _plot_wasserstein_both_species(
        ax_d, human_metrics, corr_metrics,
        human_color=HUMAN_COLOR, corr_color=CORR_COLOR,
        human_inter_roi=human_inter_roi_w, corr_inter_roi=corr_inter_roi_w
    )

    # (e) r_arith error - both species
    _plot_radius_bias_both_species(
        ax_e, human_metrics, corr_metrics,
        radius_type='r_arith',
        human_color=HUMAN_COLOR, corr_color=CORR_COLOR
    )

    # (f) r_eff error - both species
    _plot_radius_bias_both_species(
        ax_f, human_metrics, corr_metrics,
        radius_type='r_eff',
        human_color=HUMAN_COLOR, corr_color=CORR_COLOR
    )

    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()
    logger.info(f"Saved figure to {output_file}")


def _plot_win_rate(
    ax: plt.Axes,
    human_metrics: AggregatedMetrics,
    corr_metrics: AggregatedMetrics,
    human_color: str,
    corr_color: str
) -> None:
    """Plot win rate dot plot with both species on same axes."""
    # Get distribution names (use human order)
    names = human_metrics.distribution_names
    display_names = [get_display_name(n, multiline=True) for n in names]

    # Get win rates for each species (as percentages)
    human_win = np.array([human_metrics.win_rate.get(n, 0) for n in names]) * 100
    corr_win = np.array([corr_metrics.win_rate.get(n, 0) for n in names]) * 100

    y_spacing = 1.3  # Spacing between distributions
    y_pos = np.arange(len(names)) * y_spacing

    # Plot dots (corrected first so raw is on top)
    ax.scatter(corr_win, y_pos, color=corr_color, s=120, marker='s',
               label='Corrected', zorder=3, edgecolor='white', linewidth=0.5)
    ax.scatter(human_win, y_pos, color=human_color, s=120, marker='d',
               label='Raw', zorder=4, edgecolor='white', linewidth=0.5)

    # Connect dots with lines
    for i in range(len(names)):
        ax.plot([human_win[i], corr_win[i]], [y_pos[i], y_pos[i]],
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
    corr_metrics: AggregatedMetrics,
    human_color: str,
    corr_color: str,
    human_inter_roi: float = None,
    corr_inter_roi: float = None
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
    box_data_corr = []

    for dist_name in names:
        dist_idx = human_metrics.dist_name_to_idx[dist_name]

        # Human Wasserstein values
        human_w = human_metrics.all_wasserstein[dist_idx]
        human_w_valid = human_w[~np.isnan(human_w)]
        box_data_human.append(human_w_valid)

        # Corrected Wasserstein values
        corr_dist_idx = corr_metrics.dist_name_to_idx.get(dist_name, -1)
        if corr_dist_idx >= 0:
            corr_w = corr_metrics.all_wasserstein[corr_dist_idx]
            corr_w_valid = corr_w[~np.isnan(corr_w)]
            box_data_corr.append(corr_w_valid)
        else:
            box_data_corr.append(np.array([]))

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

    # Plot boxplots for corrected (offset up)
    corr_positions = [y_pos[i] + box_width/2 for i in range(len(box_data_corr))]
    bp_corr = ax.boxplot([d if len(d) > 0 else [np.nan] for d in box_data_corr],
                        positions=corr_positions, vert=False, widths=box_width * 0.8,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color=corr_color, linewidth=2),
                        whiskerprops=dict(color=corr_color, linewidth=1.5),
                        capprops=dict(color=corr_color, linewidth=1.5))
    for patch in bp_corr['boxes']:
        patch.set_facecolor(corr_color)
        patch.set_alpha(0.5)
        patch.set_edgecolor(corr_color)

    # Add inter-ROI reference lines (vertical dashed)
    if human_inter_roi is not None:
        ax.axvline(human_inter_roi, color=human_color, linestyle='--', linewidth=2,
                   alpha=0.8, zorder=1)
    if corr_inter_roi is not None:
        ax.axvline(corr_inter_roi, color=corr_color, linestyle='--', linewidth=2,
                   alpha=0.8, zorder=1)

    # Add legend manually (Corrected, Raw on first row; Anat. Var. on second row)
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
            # First line (corrected color) on top
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
    # To get Row1: [Corrected, Raw], Row2: [Anat. Var.]
    corr_handle = Patch(facecolor=corr_color, label='Corrected')
    human_handle = Patch(facecolor=human_color, label='Raw')

    if human_inter_roi is not None or corr_inter_roi is not None:
        # Placeholder handle for Anat. Var. (handler will draw the actual symbol)
        anat_var_handle = Line2D([0], [0], color='none')
        # Single row: Corrected, Raw, Anat. Var.
        legend_elements = [corr_handle, human_handle, anat_var_handle]
        labels = ['Corrected', 'Raw', 'Anat. Var.']
        handler_map = {
            corr_handle: SolidPatchHandler(corr_color),
            human_handle: SolidPatchHandler(human_color),
            anat_var_handle: DualDashedLineHandler(corr_color, human_color)
        }
        ncol = 3
    else:
        legend_elements = [corr_handle, human_handle]
        labels = ['Corrected', 'Raw']
        handler_map = {
            corr_handle: SolidPatchHandler(corr_color),
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
    corr_metrics: AggregatedMetrics,
    radius_type: str,
    human_color: str,
    corr_color: str
) -> None:
    """Plot radius bias for both species as violins showing spread.

    Args:
        ax: Matplotlib axes
        human_metrics: Human CC aggregated metrics
        corr_metrics: EM-corrected aggregated metrics
        radius_type: 'r_arith' or 'r_eff'
        human_color: Color for human data points
        corr_color: Color for corrected data points
    """
    # Use human distribution order (sorted by AIC)
    names = human_metrics.distribution_names
    display_names = [get_display_name(n, multiline=True) for n in names]

    y_spacing = 1.3  # Spacing between distributions
    y_pos = np.arange(len(names)) * y_spacing

    # Get per-sample values for each species
    if radius_type == 'r_arith':
        human_all = human_metrics.all_r_arith  # (n_dist, n_samples)
        corr_all = corr_metrics.all_r_arith
        human_emp_per_sample = human_metrics.empirical_r_arith_per_sample
        corr_emp_per_sample = corr_metrics.empirical_r_arith_per_sample
        xlabel = r'$\bar{r}$ error [%]'
        x_lim = 5  # ±5%
    else:  # r_eff
        human_all = human_metrics.all_r_eff
        corr_all = corr_metrics.all_r_eff
        human_emp_per_sample = human_metrics.empirical_r_eff_per_sample
        corr_emp_per_sample = corr_metrics.empirical_r_eff_per_sample
        xlabel = r'$r_{\mathrm{MRI}}$ error [%]'
        x_lim = 60  # ±60%

    # Compute per-sample bias for each distribution
    box_width = 0.4
    box_data_human = []
    box_data_corr = []

    for dist_name in names:
        dist_idx = human_metrics.dist_name_to_idx[dist_name]

        # Human: bias = (fitted - empirical) / empirical * 100 for each sample
        human_fitted = human_all[dist_idx]
        human_bias = (human_fitted - human_emp_per_sample) / human_emp_per_sample * 100
        human_bias_valid = human_bias[~np.isnan(human_bias)]
        box_data_human.append(human_bias_valid)

        # Corrected
        corr_dist_idx = corr_metrics.dist_name_to_idx.get(dist_name, -1)
        if corr_dist_idx >= 0:
            corr_fitted = corr_all[corr_dist_idx]
            corr_bias = (corr_fitted - corr_emp_per_sample) / corr_emp_per_sample * 100
            corr_bias_valid = corr_bias[~np.isnan(corr_bias)]
            box_data_corr.append(corr_bias_valid)
        else:
            box_data_corr.append(np.array([]))

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

    # Plot boxplots for corrected (offset up)
    corr_positions = [y_pos[i] + box_width/2 for i in range(len(box_data_corr))]
    bp_corr = ax.boxplot([d if len(d) > 0 else [np.nan] for d in box_data_corr],
                        positions=corr_positions, vert=False, widths=box_width * 0.8,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color=corr_color, linewidth=2),
                        whiskerprops=dict(color=corr_color, linewidth=1.5),
                        capprops=dict(color=corr_color, linewidth=1.5))
    for patch in bp_corr['boxes']:
        patch.set_facecolor(corr_color)
        patch.set_alpha(0.5)
        patch.set_edgecolor(corr_color)

    # Add zero reference line
    ax.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.7)

    # Annotate out-of-bounds medians with arrows and values
    annot_fontsize = settings.fonts['tick_size']  # Larger, more visible
    for i, (h_data, r_data) in enumerate(zip(box_data_human, box_data_corr)):
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
        # Corrected
        if len(r_data) > 0:
            r_median = np.median(r_data)
            if r_median > x_lim:
                ax.annotate(f'{r_median:.0f}%', xy=(x_lim, y_pos[i] + box_width/2),
                            xytext=(x_lim - 8, y_pos[i] + box_width/2),
                            fontsize=annot_fontsize, fontweight='bold',
                            color=corr_color, ha='right', va='center',
                            arrowprops=dict(arrowstyle='->', color=corr_color, lw=1.5))
            elif r_median < -x_lim:
                ax.annotate(f'{r_median:.0f}%', xy=(-x_lim, y_pos[i] + box_width/2),
                            xytext=(-x_lim + 8, y_pos[i] + box_width/2),
                            fontsize=annot_fontsize, fontweight='bold',
                            color=corr_color, ha='left', va='center',
                            arrowprops=dict(arrowstyle='->', color=corr_color, lw=1.5))

    # Add legend with boxplot-style handles (Corrected first)
    from matplotlib.patches import Patch
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
    corr_patch = Patch(facecolor=corr_color, label='Corrected', alpha=1.0)
    human_patch = Patch(facecolor=human_color, label='Raw', alpha=1.0)

    ax.legend(handles=[corr_patch, human_patch], loc='upper right',
              fontsize=settings.fonts['legend_size'], ncol=2,
              frameon=True, edgecolor='gray', facecolor='white', framealpha=1.0,
              handletextpad=0.3, borderpad=0.3,
              columnspacing=0.8, handlelength=1.8,
              handler_map={corr_patch: BoxplotHandler(corr_color, corr_color),
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
        ax.set_xticks([-60, -30, 0, 30, 60])
    ax.set_ylim(y_pos[-1] + 0.5, -1.5)  # Inverted, with extra padding at top
    ax.tick_params(labelsize=settings.fonts['tick_size'])


def main():
    parser = argparse.ArgumentParser(
        description='Compare distribution fits: Human CC raw vs EM-corrected',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--human-data', type=Path, default=Path('data/raw/human/lm'),
                        help='Directory containing human CC LM TSV files')
    parser.add_argument('--em-data', type=Path, default=Path('data/raw/human/em'),
                        help='Directory containing EM CSV files for correction')
    parser.add_argument('--output', type=Path, default=Path('fig/exploratory/compare_human_em_corrected_fits.svg'),
                        help='Output file path')
    parser.add_argument('--bin-width', type=float, default=DEFAULT_BIN_WIDTH,
                        help=f'Bin width in um (default: {DEFAULT_BIN_WIDTH})')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='EM correction threshold in um (default: 0.4)')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top distributions to show (default: 5)')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load raw Human CC data
    logger.info("=" * 60)
    logger.info("Loading Human CC (raw) data...")
    human_pooled, human_per_sample = load_human_cc_data(
        args.human_data,
        args.bin_width,
    )

    # Load EM-corrected Human CC data
    logger.info("=" * 60)
    logger.info("Loading Human CC (EM-corrected) data...")
    corr_pooled, corr_per_sample = load_corrected_human_data(
        args.human_data, args.em_data,
        args.bin_width, threshold=args.threshold
    )

    # Fit per-sample and aggregate AIC
    logger.info("=" * 60)
    logger.info("Fitting raw distributions...")
    human_metrics = fit_all_samples(human_per_sample, human_pooled)

    logger.info("=" * 60)
    logger.info("Fitting EM-corrected distributions...")
    corr_metrics = fit_all_samples(corr_per_sample, corr_pooled)

    # Report results
    logger.info("=" * 60)
    logger.info("Raw - Top 5 by summed AIC:")
    for i, name in enumerate(human_metrics.distribution_names[:5], 1):
        delta = human_metrics.summed_aic[i-1] - human_metrics.summed_aic[0]
        logger.info(f"  {i}. {name}: Delta AIC = {delta:.0f}")

    logger.info("EM-corrected - Top 5 by summed AIC:")
    for i, name in enumerate(corr_metrics.distribution_names[:5], 1):
        delta = corr_metrics.summed_aic[i-1] - corr_metrics.summed_aic[0]
        logger.info(f"  {i}. {name}: Delta AIC = {delta:.0f}")

    # Create figure
    logger.info("=" * 60)
    logger.info("Creating summary figure...")
    create_combined_figure(
        human_pooled, human_metrics,
        corr_pooled, corr_metrics,
        human_per_sample, corr_per_sample,
        args.output, args.top_n
    )

    # Save JSON
    json_file = args.output.with_suffix('.json')
    output_data = {
        'human_cc_raw': {
            'n_rois': human_pooled.n_samples,
            'total_count': human_pooled.total_count,
            'distributions': [
                {'name': name, 'summed_aic': float(human_metrics.summed_aic[i])}
                for i, name in enumerate(human_metrics.distribution_names)
            ]
        },
        'human_cc_em_corrected': {
            'n_rois': corr_pooled.n_samples,
            'total_count': corr_pooled.total_count,
            'distributions': [
                {'name': name, 'summed_aic': float(corr_metrics.summed_aic[i])}
                for i, name in enumerate(corr_metrics.distribution_names)
            ]
        }
    }
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Saved results to {json_file}")


if __name__ == '__main__':
    main()
