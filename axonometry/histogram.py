"""
Histogram utilities for axon radius distributions.

Provides functions for rebinning histograms and computing summary radius
statistics (arithmetic mean, effective MRI-visible radius) from either
histogram data or raw radius arrays.
"""

from typing import Tuple

import numpy as np


def rediscretize(
    bin_edges: np.ndarray,
    counts: np.ndarray,
    new_bin_width: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rediscretize a histogram to coarser bins.

    Args:
        bin_edges: Original bin edges (n_bins + 1,).
        counts: Original bin counts (n_bins,).
        new_bin_width: Width of the new bins.

    Returns:
        Tuple of (new_bin_edges, new_bin_centers, new_counts).
    """
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


def compute_r_arith(
    radii: np.ndarray = None,
    *,
    counts: np.ndarray = None,
    bin_centers: np.ndarray = None,
) -> float:
    """Compute arithmetic mean radius.

    Call with either raw radii or histogram data (counts + bin_centers).

    Args:
        radii: Array of radius values.
        counts: Histogram bin counts.
        bin_centers: Histogram bin centers.

    Returns:
        Arithmetic mean radius, or NaN if input is empty.
    """
    if radii is not None:
        return float(np.mean(radii)) if len(radii) > 0 else np.nan

    if counts is None or bin_centers is None:
        raise ValueError("Provide either radii or both counts and bin_centers")
    total = counts.sum()
    if total == 0:
        return np.nan
    return float(np.sum(bin_centers * counts / total))


def compute_r_eff(
    radii: np.ndarray = None,
    *,
    counts: np.ndarray = None,
    bin_centers: np.ndarray = None,
) -> float:
    """Compute effective MRI-visible radius: r_eff = (⟨r⁶⟩/⟨r²⟩)^(1/4).

    Call with either raw radii or histogram data (counts + bin_centers).

    Args:
        radii: Array of radius values.
        counts: Histogram bin counts.
        bin_centers: Histogram bin centers.

    Returns:
        Effective radius, or NaN if input is empty.
    """
    if radii is not None:
        if len(radii) == 0:
            return np.nan
        r2 = np.mean(radii ** 2)
        r6 = np.mean(radii ** 6)
        return float((r6 / r2) ** 0.25) if r2 > 0 else np.nan

    if counts is None or bin_centers is None:
        raise ValueError("Provide either radii or both counts and bin_centers")
    total = counts.sum()
    if total == 0:
        return np.nan
    probs = counts / total
    r2 = np.sum(bin_centers ** 2 * probs)
    r6 = np.sum(bin_centers ** 6 * probs)
    return float((r6 / r2) ** 0.25) if r2 > 0 else np.nan
