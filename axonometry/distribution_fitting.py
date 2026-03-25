"""Parametric distribution fitting for axon radius histograms.

Fits parametric distributions to binned radius data using proper binned MLE
(multinomial likelihood over CDF-derived bin probabilities). Provides
method-of-moments initialization with perturbed restarts.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from .histogram import compute_r_arith, compute_r_eff

__all__ = [
    "FitResult",
    "fit_distribution_mle",
    "compute_distribution_radii",
]

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

MIN_BIN_PROB = 1e-15  # floor for bin probabilities to avoid log(0)
N_RESTARTS = 10  # perturbed restarts around MoM estimate

# Minimum fraction of PDF mass that must fall within [0, r_max] for the
# truncated moment integration to be considered valid.
MIN_PDF_NORM = 0.01


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class FitResult:
    """Container for distribution fit results."""
    distribution_name: str
    n_params: int
    params: tuple[float, ...]  # (shape..., loc, scale) — scipy convention
    nll: float                 # negative log-likelihood
    aic: float
    wasserstein: float = 0.0
    pdf_x_fine: np.ndarray = field(default_factory=lambda: np.array([]))
    pdf_values_fine: np.ndarray = field(default_factory=lambda: np.array([]))


# =============================================================================
# Radius computation
# =============================================================================

def compute_distribution_radii(
    dist: stats.rv_continuous,
    params: tuple[float, ...],
    r_max: float,
    n_points: int = 2000,
) -> tuple[float, float]:
    """Compute r_arith and r_eff by integrating the fitted PDF over [0, r_max].

    The integration domain is bounded to the observed data range so that
    moments are well-defined even for heavy-tailed distributions (e.g. GEV
    with xi > 1/6 where the analytical 6th moment does not exist).

    Returns:
        (r_arith, r_eff) tuple; NaN if the distribution has negligible mass
        in [0, r_max].
    """
    shape_params, loc, scale = _unpack_params(params)
    try:
        dr = r_max / n_points
        r = np.linspace(dr / 2, r_max - dr / 2, n_points)  # midpoint rule; avoids 0*inf at r=0
        pdf = dist.pdf(r, *shape_params, loc=loc, scale=scale)
        pdf = np.maximum(pdf, 0)

        norm = np.sum(pdf) * dr
        if norm < MIN_PDF_NORM:
            return np.nan, np.nan
        pdf = pdf / norm

        r_arith = np.sum(r * pdf) * dr
        r2 = np.sum(r**2 * pdf) * dr
        r6 = np.sum(r**6 * pdf) * dr
        r_eff = (r6 / r2) ** 0.25 if r2 > 0 else np.nan
        return float(r_arith), float(r_eff)
    except (ValueError, FloatingPointError, OverflowError) as e:
        logger.debug(f"compute_distribution_radii failed: {e}")
        return np.nan, np.nan


# =============================================================================
# Internal helpers
# =============================================================================

def _unpack_params(
    params: tuple[float, ...],
) -> tuple[tuple[float, ...], float, float]:
    """Unpack distribution parameters into (shape_params, loc, scale)."""
    if len(params) == 2:
        return (), params[0], params[1]
    return params[:-2], params[-2], params[-1]


def _binned_nll(theta, dist, bin_edges, counts, fix_loc):
    """Negative log-likelihood for binned data."""
    if fix_loc:
        shape_params = theta[:-1]
        loc = 0.0
        scale = theta[-1]
    else:
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
    except (ValueError, FloatingPointError, OverflowError):
        return 1e20


# =============================================================================
# Method-of-moments initialization
# =============================================================================

def _histogram_moments(bin_centers, counts):
    """Compute mean, variance, and skewness from histogram data."""
    total = counts.sum()
    if total == 0:
        return 0.0, 0.0, 0.0
    probs = counts / total
    mu = np.dot(probs, bin_centers)
    var = np.dot(probs, (bin_centers - mu) ** 2)
    std = np.sqrt(var) if var > 0 else 1e-6
    skew = np.dot(probs, ((bin_centers - mu) / std) ** 3) if std > 0 else 0.0
    return mu, var, skew


def _mom_gamma(mu, var):
    """MoM for Gamma(a, loc=0, scale)."""
    a = max(mu ** 2 / var, 0.1) if var > 0 else 1.0
    scale = max(var / mu, 1e-6) if mu > 0 else 1e-6
    return (a, 0.0, scale)


def _mom_lognorm(mu, var):
    """MoM for LogNormal(s, loc=0, scale)."""
    if mu > 0 and var > 0:
        sigma2 = np.log(1 + var / mu ** 2)
        s = np.sqrt(sigma2)
        scale = mu / np.sqrt(1 + var / mu ** 2)
    else:
        s, scale = 0.5, max(mu, 1e-6)
    return (s, 0.0, scale)


def _mom_invgauss(mu, var):
    """MoM for InvGauss(mu_ig, loc=0, scale).

    scipy invgauss(mu_ig, loc=0, scale):
        mean = mu_ig * scale, var = mu_ig^3 * scale^2
    Solving: mu_ig = var / mean^2, scale = mean^3 / var.
    """
    if mu > 0 and var > 0:
        mu_ig = max(var / mu ** 2, 0.1)
        scale = max(mu ** 3 / var, 1e-6)
    else:
        mu_ig, scale = 1.0, max(mu, 1e-6)
    return (mu_ig, 0.0, scale)


def _mom_fatiguelife(mu, var):
    """MoM for Birnbaum-Saunders / fatiguelife(c, loc=0, scale).

    mean = scale * (1 + c^2/2), var = scale^2 * c^2 * (1 + 5c^2/4).
    First estimate c from the coefficient of variation, then correct scale.
    """
    if var > 0 and mu > 0:
        ratio = var / mu ** 2
        c = np.sqrt(max(ratio / (1 + 5 * ratio / 4), 0.01))
        scale = max(mu / (1 + c**2 / 2), 1e-6)
    else:
        c = 0.5
        scale = max(mu, 1e-6)
    return (c, 0.0, scale)


def _mom_fisk(mu, var):
    """MoM for Log-logistic / fisk(c, loc=0, scale).

    scipy fisk(c, loc=0, scale): mean = scale * pi/c / sin(pi/c) for c > 1.
    Start with c=3 (moderate shape) and solve for scale from the mean.
    """
    c = 3.0
    if mu > 0:
        # mean = scale * pi/c / sin(pi/c)
        scale = max(mu * c * np.sin(np.pi / c) / np.pi, 1e-6)
    else:
        scale = 1e-6
    return (c, 0.0, scale)


def _mom_genextreme(mu, var):
    """MoM for GEV(shape, loc, scale) — Gumbel approximation.

    Starts at shape=0 (Gumbel) and lets the optimizer find the shape.
    """
    std = np.sqrt(var) if var > 0 else 1e-6
    scale = max(std * np.sqrt(6) / np.pi, 1e-6)
    loc = mu - 0.5772 * scale
    shape = 0.0
    return (shape, loc, scale)


_MOM_INITIALIZERS = {
    'gamma': _mom_gamma,
    'lognorm': _mom_lognorm,
    'invgauss': _mom_invgauss,
    'fatiguelife': _mom_fatiguelife,
    'fisk': _mom_fisk,
    'genextreme': _mom_genextreme,
}


def _mom_initial_params(dist_name, bin_centers, counts):
    """Compute method-of-moments initial parameter estimates."""
    if dist_name not in _MOM_INITIALIZERS:
        raise ValueError(f"No MoM initializer for distribution: {dist_name}")
    mu, var, _ = _histogram_moments(bin_centers, counts)
    return _MOM_INITIALIZERS[dist_name](mu, var)


def _get_initial_params_multi(dist_name, bin_centers, counts):
    """Get initial parameter estimates: MoM + perturbed variants."""
    mom_params = _mom_initial_params(dist_name, bin_centers, counts)
    all_params = [mom_params]

    for seed in range(1, N_RESTARTS + 1):
        rng = np.random.default_rng(seed)
        perturbed = list(mom_params)
        for i in range(len(perturbed)):
            if i == len(perturbed) - 2 and dist_name != 'genextreme':
                continue  # skip loc=0 for fixed-loc distributions
            if perturbed[i] == 0.0:
                # Additive perturbation for zero-valued params (e.g. GEV shape
                # starts at 0.0 for the Gumbel initialization)
                perturbed[i] = rng.normal(0, 0.2)
            else:
                perturbed[i] *= rng.lognormal(0, 0.3)
        perturbed[-1] = max(perturbed[-1], 1e-8)
        all_params.append(tuple(perturbed))

    return all_params


# =============================================================================
# Core fitting
# =============================================================================

def fit_distribution_mle(
    dist_name: str,
    dist: stats.rv_continuous,
    bin_centers: np.ndarray,
    bin_edges: np.ndarray,
    counts: np.ndarray,
) -> Optional[FitResult]:
    """Fit a distribution to histogram data using binned MLE.

    Maximizes the multinomial log-likelihood over bin probabilities
    derived from the parametric CDF. Uses method-of-moments initialization
    with perturbed restarts to avoid local optima.

    Args:
        dist_name: Scipy distribution name (e.g. 'gamma', 'genextreme').
            Must have a corresponding entry in the MoM initializer registry.
        dist: Scipy distribution object.
        bin_centers: Array of bin center values.
        bin_edges: Array of bin edge values (len = len(bin_centers) + 1).
        counts: Observed counts per bin (non-negative integers).

    Returns:
        FitResult with fitted parameters and diagnostics, or None if
        fitting fails.
    """
    total = counts.sum()
    if total == 0:
        return None

    try:
        all_init_params = _get_initial_params_multi(dist_name, bin_centers, counts)
        if not all_init_params:
            return None

        fix_loc = (dist_name != 'genextreme')

        # Bounds
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
        k = len(result.x)
        aic = 2 * k + 2 * nll

        # Wasserstein distance using per-bin widths (supports non-uniform bins)
        bin_widths = np.diff(bin_edges)
        empirical_cdf = np.cumsum(counts) / total
        fitted_cdf = dist.cdf(bin_edges[1:], *shape_params, loc=loc, scale=scale)
        wasserstein = float(np.sum(np.abs(empirical_cdf - fitted_cdf) * bin_widths))

        x_fine = np.linspace(bin_centers[0], bin_centers[-1], 500)
        pdf_fine = dist.pdf(x_fine, *shape_params, loc=loc, scale=scale)

        return FitResult(
            distribution_name=dist_name,
            n_params=k,
            params=params,
            nll=nll,
            aic=aic,
            wasserstein=wasserstein,
            pdf_x_fine=x_fine,
            pdf_values_fine=pdf_fine,
        )

    except (ValueError, RuntimeError) as e:
        logger.debug(f"Failed to fit {dist_name}: {e}")
        return None
