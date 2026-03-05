#!/usr/bin/env python3
"""
Analyze spatial correlation of axon radius profiles between neighboring axons.

Tests the hypothesis of local mechanical compensation: when one axon gets thicker
along its trajectory, do nearby axons get thinner?

Unlike analyze_spatial_correlation_ensemble.py which correlates per-slice deviations,
this script correlates entire radius profiles between axon pairs that remain
neighbors over contiguous z-segments.

Key insight: We use RESIDUAL deviations to remove shared environmental effects.
At each z-position, we subtract the local mean deviation across all axons,
so we're testing: "relative to local conditions, when axon A is thicker than
average, is neighbor B thinner than average?"

Algorithm:
1. Build z-bin index: discretize skeleton points into z-bins
2. Compute local mean deviation profile: mean relative deviation at each z-bin
3. Track neighbor segments: find axon pairs that remain neighbors across z-bins
4. Compute profile correlations: correlate RESIDUAL deviations over shared segments
5. Pool results: bin correlations by inter-axon distance across all samples
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob as glob_module

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import pearsonr, ttest_1samp
from tqdm import tqdm

# Import from axonometry library
# Find repo root (contains pyproject.toml)
_root = Path(__file__).resolve().parent
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from axonometry import get_plot_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load plot settings
settings = get_plot_settings()


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AxonData:
    """Per-axon data for correlation analysis."""
    label: int
    skeleton_um: np.ndarray  # (N, 3) array of (z, y, x) coordinates
    radii_um: np.ndarray     # (N,) array of radii
    mean_radius_um: float
    length_um: float


@dataclass
class ZBinEntry:
    """Entry in the z-bin index."""
    axon_idx: int
    xy_centroid: np.ndarray  # (2,) array of (y, x) centroid
    mean_radius: float
    relative_deviation: float  # (r - r_mean) / r_mean for this axon in this bin


@dataclass
class NeighborSegment:
    """A contiguous z-segment where two axons are neighbors."""
    axon_i: int
    axon_j: int
    z_start: float
    z_end: float
    mean_distance: float
    n_bins: int


@dataclass
class SegmentResult:
    """Result of correlating a neighbor segment."""
    correlation: float
    p_value: float
    n_points: int
    segment_length: float
    mean_distance: float
    axon_i_label: int
    axon_j_label: int
    # Additional: absolute deviation correlation (not normalized)
    correlation_abs: float = np.nan
    p_value_abs: float = np.nan


# =============================================================================
# Data Loading
# =============================================================================

def load_axon_profiles(npz_file: Path, min_length_um: float = 10.0) -> List[AxonData]:
    """
    Load axon profiles from NPZ file.

    Args:
        npz_file: Path to *_axon_profiles.npz file
        min_length_um: Minimum axon length to include (default: 10 um)

    Returns:
        List of AxonData objects
    """
    data = np.load(npz_file, allow_pickle=True)

    labels = data['labels']
    skeleton_coords = data['skeleton_coords_um']
    radii_profiles = data['radii_profiles_um']
    mean_radii = data['mean_radii_um']
    lengths = data['lengths_um']

    axons = []
    for i in range(len(labels)):
        if lengths[i] < min_length_um:
            continue
        if len(skeleton_coords[i]) < 5:  # Need minimum points
            continue

        axons.append(AxonData(
            label=int(labels[i]),
            skeleton_um=skeleton_coords[i],
            radii_um=radii_profiles[i],
            mean_radius_um=float(mean_radii[i]),
            length_um=float(lengths[i])
        ))

    return axons


# =============================================================================
# Z-Index Construction
# =============================================================================

def build_z_index(axons: List[AxonData],
                  z_bin_size: float = 0.5) -> Dict[int, List[ZBinEntry]]:
    """
    Build index mapping z-bin to axons present in that bin.

    Args:
        axons: List of AxonData objects
        z_bin_size: Size of z-bins in micrometers

    Returns:
        Dictionary mapping z-bin index to list of ZBinEntry
    """
    z_index = defaultdict(list)

    for axon_idx, axon in enumerate(axons):
        z_coords = axon.skeleton_um[:, 0]
        xy_coords = axon.skeleton_um[:, 1:3]
        radii = axon.radii_um

        # Discretize z-coordinates
        z_bins = (z_coords / z_bin_size).astype(int)

        # Group by z-bin
        for z_bin in np.unique(z_bins):
            mask = z_bins == z_bin
            xy_centroid = xy_coords[mask].mean(axis=0)
            mean_radius = radii[mask].mean()

            # Compute relative deviation from this axon's global mean
            if axon.mean_radius_um > 0:
                relative_dev = (mean_radius - axon.mean_radius_um) / axon.mean_radius_um
            else:
                relative_dev = 0.0

            z_index[z_bin].append(ZBinEntry(
                axon_idx=axon_idx,
                xy_centroid=xy_centroid,
                mean_radius=mean_radius,
                relative_deviation=relative_dev
            ))

    return z_index


def compute_local_mean_deviations(z_index: Dict[int, List[ZBinEntry]],
                                   z_bin_size: float = 0.5) -> Dict[int, float]:
    """
    Compute the mean relative deviation across all axons at each z-bin.

    This captures the "shared environmental effect" that we want to subtract.

    Args:
        z_index: Dictionary mapping z-bin to list of ZBinEntry
        z_bin_size: Size of z-bins in micrometers

    Returns:
        Dictionary mapping z-bin index to mean relative deviation
    """
    local_means = {}

    for z_bin, entries in z_index.items():
        if entries:
            deviations = [e.relative_deviation for e in entries]
            local_means[z_bin] = np.mean(deviations)
        else:
            local_means[z_bin] = 0.0

    return local_means


# =============================================================================
# Neighbor Segment Tracking
# =============================================================================

def track_neighbor_segments(z_index: Dict[int, List[ZBinEntry]],
                            d_max: float = 10.0,
                            min_segment_bins: int = 10,
                            z_bin_size: float = 0.5,
                            gap_tolerance: int = 1) -> List[NeighborSegment]:
    """
    Find axon pairs that are neighbors over contiguous z-segments.

    Args:
        z_index: Dictionary mapping z-bin to list of ZBinEntry
        d_max: Maximum XY distance for neighbor definition (um)
        min_segment_bins: Minimum number of bins for a valid segment
        z_bin_size: Size of z-bins in micrometers
        gap_tolerance: Allow this many missing bins in continuity

    Returns:
        List of NeighborSegment objects
    """
    z_bins_sorted = sorted(z_index.keys())

    # Track active pairs: (i, j) -> {'start': z_bin, 'last': z_bin, 'distances': []}
    active_pairs: Dict[Tuple[int, int], Dict] = {}
    finalized_segments: List[NeighborSegment] = []

    for z_bin in tqdm(z_bins_sorted, desc="  Tracking neighbors", leave=False):
        entries = z_index[z_bin]
        if len(entries) < 2:
            continue

        # Build KDTree for XY positions
        positions = np.array([e.xy_centroid for e in entries])
        tree = KDTree(positions)

        # Find all pairs within d_max
        pairs_in_range = tree.query_pairs(d_max)

        # Map to axon indices
        axon_indices = [e.axon_idx for e in entries]
        seen_pairs = set()

        for local_i, local_j in pairs_in_range:
            i, j = axon_indices[local_i], axon_indices[local_j]
            if i > j:
                i, j = j, i  # Canonical order

            pair_key = (i, j)
            seen_pairs.add(pair_key)
            dist = np.linalg.norm(positions[local_i] - positions[local_j])

            if pair_key not in active_pairs:
                # Start new segment
                active_pairs[pair_key] = {
                    'start': z_bin,
                    'last': z_bin,
                    'distances': [dist]
                }
            else:
                # Check if gap is acceptable
                if z_bin - active_pairs[pair_key]['last'] <= gap_tolerance:
                    # Extend segment
                    active_pairs[pair_key]['last'] = z_bin
                    active_pairs[pair_key]['distances'].append(dist)
                else:
                    # Gap too large, finalize and restart
                    _finalize_segment(
                        active_pairs[pair_key], pair_key,
                        min_segment_bins, finalized_segments, z_bin_size
                    )
                    active_pairs[pair_key] = {
                        'start': z_bin,
                        'last': z_bin,
                        'distances': [dist]
                    }

        # Check for pairs that went inactive
        pairs_to_remove = []
        for pair_key, data in active_pairs.items():
            if pair_key not in seen_pairs:
                if z_bin - data['last'] > gap_tolerance:
                    _finalize_segment(
                        data, pair_key,
                        min_segment_bins, finalized_segments, z_bin_size
                    )
                    pairs_to_remove.append(pair_key)

        for pair_key in pairs_to_remove:
            del active_pairs[pair_key]

    # Finalize remaining active pairs
    for pair_key, data in active_pairs.items():
        _finalize_segment(
            data, pair_key,
            min_segment_bins, finalized_segments, z_bin_size
        )

    return finalized_segments


def _finalize_segment(data: Dict, pair_key: Tuple[int, int],
                      min_bins: int, results: List[NeighborSegment],
                      z_bin_size: float) -> None:
    """Finalize a neighbor segment if it meets length criteria."""
    n_bins = data['last'] - data['start'] + 1
    if n_bins >= min_bins:
        results.append(NeighborSegment(
            axon_i=pair_key[0],
            axon_j=pair_key[1],
            z_start=data['start'] * z_bin_size,
            z_end=(data['last'] + 1) * z_bin_size,
            mean_distance=np.mean(data['distances']),
            n_bins=n_bins
        ))


# =============================================================================
# Profile Correlation
# =============================================================================

def compute_segment_correlation(axons: List[AxonData],
                                 segment: NeighborSegment,
                                 local_mean_deviations: Dict[int, float],
                                 z_bin_size: float = 0.5,
                                 min_points: int = 10) -> Optional[SegmentResult]:
    """
    Correlate RESIDUAL radius deviations over a shared z-segment.

    Residual = relative deviation - local mean deviation at that z-position.
    This removes the shared environmental effect.

    Args:
        axons: List of AxonData objects
        segment: NeighborSegment defining the z-range and pair
        local_mean_deviations: Dict mapping z-bin to mean deviation across all axons
        z_bin_size: Size of z-bins in micrometers
        min_points: Minimum interpolated points for valid correlation

    Returns:
        SegmentResult or None if insufficient data
    """
    axon_i = axons[segment.axon_i]
    axon_j = axons[segment.axon_j]

    # Extract points in z-range
    i_mask = (axon_i.skeleton_um[:, 0] >= segment.z_start) & \
             (axon_i.skeleton_um[:, 0] < segment.z_end)
    j_mask = (axon_j.skeleton_um[:, 0] >= segment.z_start) & \
             (axon_j.skeleton_um[:, 0] < segment.z_end)

    z_i = axon_i.skeleton_um[i_mask, 0]
    r_i = axon_i.radii_um[i_mask]
    z_j = axon_j.skeleton_um[j_mask, 0]
    r_j = axon_j.radii_um[j_mask]

    if len(z_i) < min_points or len(z_j) < min_points:
        return None

    # Sort by z (skeleton may not be ordered)
    sort_i = np.argsort(z_i)
    z_i, r_i = z_i[sort_i], r_i[sort_i]
    sort_j = np.argsort(z_j)
    z_j, r_j = z_j[sort_j], r_j[sort_j]

    # Create common z-grid
    z_min = max(z_i.min(), z_j.min())
    z_max = min(z_i.max(), z_j.max())

    if z_max <= z_min:
        return None

    n_points = min(len(z_i), len(z_j), 100)  # Cap at 100 points
    common_z = np.linspace(z_min, z_max, n_points)

    if len(common_z) < min_points:
        return None

    # Interpolate radii to common grid
    r_i_interp = np.interp(common_z, z_i, r_i)
    r_j_interp = np.interp(common_z, z_j, r_j)

    # Compute relative deviations
    if axon_i.mean_radius_um <= 0 or axon_j.mean_radius_um <= 0:
        return None

    delta_i = (r_i_interp - axon_i.mean_radius_um) / axon_i.mean_radius_um
    delta_j = (r_j_interp - axon_j.mean_radius_um) / axon_j.mean_radius_um

    # Compute ABSOLUTE deviations (in micrometers, not normalized)
    abs_delta_i = r_i_interp - axon_i.mean_radius_um
    abs_delta_j = r_j_interp - axon_j.mean_radius_um

    # Subtract local mean deviation to get RESIDUALS
    # For each z in common_z, find the corresponding z-bin and subtract its local mean
    z_bins = (common_z / z_bin_size).astype(int)
    local_means = np.array([local_mean_deviations.get(zb, 0.0) for zb in z_bins])

    residual_i = delta_i - local_means
    residual_j = delta_j - local_means

    # Also compute local mean in absolute units for absolute residuals
    # We need to compute mean absolute deviation per z-bin (not stored, so approximate)
    # For absolute residuals, we subtract the mean absolute deviation at each z-bin
    # But the current local_mean_deviations is in relative units
    # Simpler approach: just correlate absolute deviations without subtracting local mean
    # This tests raw mechanical compensation without normalization

    # Pearson correlation of residuals (relative, with local mean subtracted)
    try:
        r, p = pearsonr(residual_i, residual_j)
    except Exception:
        return None

    if np.isnan(r):
        return None

    # Pearson correlation of absolute deviations (without local mean subtraction)
    try:
        r_abs, p_abs = pearsonr(abs_delta_i, abs_delta_j)
    except Exception:
        r_abs, p_abs = np.nan, np.nan

    return SegmentResult(
        correlation=r,
        p_value=p,
        n_points=len(common_z),
        segment_length=z_max - z_min,
        mean_distance=segment.mean_distance,
        axon_i_label=axon_i.label,
        axon_j_label=axon_j.label,
        correlation_abs=r_abs,
        p_value_abs=p_abs
    )


# =============================================================================
# Sample Processing
# =============================================================================

def process_single_sample(npz_file: Path,
                          d_max: float = 10.0,
                          min_segment_length: float = 5.0,
                          z_bin_size: float = 0.5,
                          min_points: int = 10,
                          min_axon_length: float = 10.0) -> Tuple[str, List[SegmentResult], int]:
    """
    Process a single sample and return segment correlations.

    Returns:
        Tuple of (sample_name, list of SegmentResult, n_axons)
    """
    sample_name = npz_file.stem.replace('_axon_profiles', '')
    logger.info(f"Processing {sample_name}...")

    # Load axon profiles
    axons = load_axon_profiles(npz_file, min_length_um=min_axon_length)
    n_axons = len(axons)
    logger.info(f"  Loaded {n_axons} axons (min length: {min_axon_length} um)")

    if n_axons < 2:
        logger.warning(f"  Insufficient axons, skipping")
        return sample_name, [], n_axons

    # Build z-index
    z_index = build_z_index(axons, z_bin_size=z_bin_size)
    n_bins = len(z_index)
    logger.info(f"  Built z-index with {n_bins} bins (z_bin_size: {z_bin_size} um)")

    # Compute local mean deviations (to subtract shared environmental effect)
    local_mean_deviations = compute_local_mean_deviations(z_index, z_bin_size=z_bin_size)
    logger.info(f"  Computed local mean deviations for {len(local_mean_deviations)} z-bins")

    # Track neighbor segments
    min_segment_bins = max(2, int(min_segment_length / z_bin_size))
    segments = track_neighbor_segments(
        z_index, d_max=d_max,
        min_segment_bins=min_segment_bins,
        z_bin_size=z_bin_size
    )
    logger.info(f"  Found {len(segments)} neighbor segments (min length: {min_segment_length} um)")

    if not segments:
        return sample_name, [], n_axons

    # Compute correlations of RESIDUAL deviations
    results = []
    for segment in tqdm(segments, desc="  Computing correlations", leave=False):
        result = compute_segment_correlation(
            axons, segment, local_mean_deviations,
            z_bin_size=z_bin_size, min_points=min_points
        )
        if result is not None:
            results.append(result)

    logger.info(f"  Computed {len(results)} valid correlations")

    return sample_name, results, n_axons


# =============================================================================
# Pooling and Statistics
# =============================================================================

def pool_correlations_by_distance(all_results: List[SegmentResult],
                                   distance_bins: np.ndarray,
                                   n_bootstrap: int = 1000) -> Dict:
    """
    Pool segment correlations by mean inter-axon distance.

    Returns dict with bin_centers, mean_correlations, CIs, n_segments, p_values.
    Also includes absolute deviation correlations.
    """
    n_bins = len(distance_bins) - 1
    bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2

    # Bin results by distance (both relative and absolute correlations)
    binned_correlations = [[] for _ in range(n_bins)]
    binned_correlations_abs = [[] for _ in range(n_bins)]

    for result in all_results:
        bin_idx = np.searchsorted(distance_bins, result.mean_distance, side='right') - 1
        if 0 <= bin_idx < n_bins:
            binned_correlations[bin_idx].append(result.correlation)
            if not np.isnan(result.correlation_abs):
                binned_correlations_abs[bin_idx].append(result.correlation_abs)

    # Compute statistics per bin
    mean_correlations = []
    ci_low = []
    ci_high = []
    n_segments = []
    p_values = []
    # Percentiles for showing distribution spread
    pct_25 = []
    pct_75 = []
    pct_10 = []
    pct_90 = []

    # Also for absolute correlations
    mean_correlations_abs = []
    ci_low_abs = []
    ci_high_abs = []
    n_segments_abs = []
    p_values_abs = []
    pct_25_abs = []
    pct_75_abs = []
    pct_10_abs = []
    pct_90_abs = []

    rng = np.random.default_rng(42)
    min_for_stats = 30

    for i, correlations in enumerate(tqdm(binned_correlations, desc="Computing statistics", leave=False)):
        n_segments.append(len(correlations))

        if len(correlations) < min_for_stats:
            mean_correlations.append(np.nan)
            ci_low.append(np.nan)
            ci_high.append(np.nan)
            p_values.append(np.nan)
            pct_25.append(np.nan)
            pct_75.append(np.nan)
            pct_10.append(np.nan)
            pct_90.append(np.nan)
        else:
            correlations = np.array(correlations)
            mean_correlations.append(np.mean(correlations))

            # Bootstrap CI for the mean
            boot_means = []
            for _ in range(n_bootstrap):
                sample = rng.choice(correlations, size=len(correlations), replace=True)
                boot_means.append(np.mean(sample))
            ci_low.append(np.percentile(boot_means, 2.5))
            ci_high.append(np.percentile(boot_means, 97.5))

            # One-sample t-test against 0
            _, p = ttest_1samp(correlations, 0)
            p_values.append(p)

            # Distribution percentiles (of individual correlations)
            pct_25.append(np.percentile(correlations, 25))
            pct_75.append(np.percentile(correlations, 75))
            pct_10.append(np.percentile(correlations, 10))
            pct_90.append(np.percentile(correlations, 90))

        # Absolute correlations
        corrs_abs = binned_correlations_abs[i]
        n_segments_abs.append(len(corrs_abs))

        if len(corrs_abs) < min_for_stats:
            mean_correlations_abs.append(np.nan)
            ci_low_abs.append(np.nan)
            ci_high_abs.append(np.nan)
            p_values_abs.append(np.nan)
            pct_25_abs.append(np.nan)
            pct_75_abs.append(np.nan)
            pct_10_abs.append(np.nan)
            pct_90_abs.append(np.nan)
        else:
            corrs_abs = np.array(corrs_abs)
            mean_correlations_abs.append(np.mean(corrs_abs))

            # Bootstrap CI for the mean
            boot_means = []
            for _ in range(n_bootstrap):
                sample = rng.choice(corrs_abs, size=len(corrs_abs), replace=True)
                boot_means.append(np.mean(sample))
            ci_low_abs.append(np.percentile(boot_means, 2.5))
            ci_high_abs.append(np.percentile(boot_means, 97.5))

            # One-sample t-test against 0
            _, p = ttest_1samp(corrs_abs, 0)
            p_values_abs.append(p)

            # Distribution percentiles (of individual correlations)
            pct_25_abs.append(np.percentile(corrs_abs, 25))
            pct_75_abs.append(np.percentile(corrs_abs, 75))
            pct_10_abs.append(np.percentile(corrs_abs, 10))
            pct_90_abs.append(np.percentile(corrs_abs, 90))

    return {
        'bin_centers': bin_centers.tolist(),
        'bin_edges': distance_bins.tolist(),
        'mean_correlations': mean_correlations,
        'n_segments': n_segments,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_values': p_values,
        'pct_25': pct_25,
        'pct_75': pct_75,
        'pct_10': pct_10,
        'pct_90': pct_90,
        # Absolute deviation correlations
        'mean_correlations_abs': mean_correlations_abs,
        'n_segments_abs': n_segments_abs,
        'ci_low_abs': ci_low_abs,
        'ci_high_abs': ci_high_abs,
        'p_values_abs': p_values_abs,
        'pct_25_abs': pct_25_abs,
        'pct_75_abs': pct_75_abs,
        'pct_10_abs': pct_10_abs,
        'pct_90_abs': pct_90_abs
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_correlation_by_distance(results: Dict, output_file: Path,
                                  n_samples: int, total_axons: int,
                                  total_segments: int,
                                  correlation_key: str = 'mean_correlations',
                                  ci_low_key: str = 'ci_low',
                                  ci_high_key: str = 'ci_high',
                                  p_values_key: str = 'p_values',
                                  n_segments_key: str = 'n_segments',
                                  pct_25_key: str = 'pct_25',
                                  pct_75_key: str = 'pct_75',
                                  pct_10_key: str = 'pct_10',
                                  pct_90_key: str = 'pct_90',
                                  title_suffix: str = '',
                                  ylabel: str = 'Mean Residual Correlation',
                                  subtitle: str = r'$r = \mathrm{corr}(\Delta r_i - \bar{\Delta r}, \Delta r_j - \bar{\Delta r})$ (local mean subtracted)'):
    """Plot pooled correlation vs distance - two-panel figure."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08}
    )

    font_settings = settings.fonts

    bin_centers = np.array(results['bin_centers'])
    bin_edges = np.array(results['bin_edges'])
    correlations = np.array(results[correlation_key])
    ci_low = np.array(results[ci_low_key])
    ci_high = np.array(results[ci_high_key])
    n_segments = np.array(results[n_segments_key])
    p_values = np.array(results[p_values_key])
    bin_width = bin_edges[1] - bin_edges[0]

    # Percentiles for distribution spread
    pct_25 = np.array(results.get(pct_25_key, [np.nan] * len(bin_centers)))
    pct_75 = np.array(results.get(pct_75_key, [np.nan] * len(bin_centers)))
    pct_10 = np.array(results.get(pct_10_key, [np.nan] * len(bin_centers)))
    pct_90 = np.array(results.get(pct_90_key, [np.nan] * len(bin_centers)))

    valid = ~np.isnan(correlations)

    # Colors
    main_color = '#2E86AB'
    sig_color = '#E94F37'
    nonsig_color = '#A1A1A1'
    band_color = '#2E86AB'

    # ===== Top plot: Correlation vs distance =====
    ax1.axhline(0, color='#666666', linestyle='-', linewidth=0.8, alpha=0.5)

    # Plot percentile bands (distribution of individual correlations)
    if np.any(valid) and not np.all(np.isnan(pct_10)):
        # 10th-90th percentile band (outer, lighter)
        ax1.fill_between(bin_centers[valid], pct_10[valid], pct_90[valid],
                        color=band_color, alpha=0.15, label='10th-90th pct')
        # 25th-75th percentile band (inner, darker)
        ax1.fill_between(bin_centers[valid], pct_25[valid], pct_75[valid],
                        color=band_color, alpha=0.25, label='IQR (25th-75th)')

    if np.any(valid):
        sig_mask = valid & (np.array(p_values) < 0.05)
        nonsig_mask = valid & (np.array(p_values) >= 0.05)

        if np.any(nonsig_mask):
            yerr_low_ns = correlations[nonsig_mask] - ci_low[nonsig_mask]
            yerr_high_ns = ci_high[nonsig_mask] - correlations[nonsig_mask]
            ax1.errorbar(bin_centers[nonsig_mask], correlations[nonsig_mask],
                        yerr=[yerr_low_ns, yerr_high_ns],
                        fmt='o', color=nonsig_color, capsize=3, capthick=1,
                        markersize=6, markerfacecolor='white', markeredgewidth=1.5,
                        linewidth=0, elinewidth=1.2, label='p >= 0.05', zorder=5)

        if np.any(sig_mask):
            yerr_low_s = correlations[sig_mask] - ci_low[sig_mask]
            yerr_high_s = ci_high[sig_mask] - correlations[sig_mask]
            ax1.errorbar(bin_centers[sig_mask], correlations[sig_mask],
                        yerr=[yerr_low_s, yerr_high_s],
                        fmt='o', color=sig_color, capsize=3, capthick=1,
                        markersize=6, markerfacecolor=sig_color, markeredgewidth=1.5,
                        linewidth=0, elinewidth=1.2, label='p < 0.05', zorder=5)

        ax1.plot(bin_centers[valid], correlations[valid],
                '-', color=main_color, alpha=0.6, linewidth=1.5, zorder=4)

    ax1.set_ylabel(ylabel,
                   fontsize=font_settings['label_size'], fontweight=font_settings['weight'])
    ax1.set_title(f'Neighbor Profile Correlation vs Distance (Pooled){title_suffix}\n{subtitle}',
                  fontsize=font_settings['title_size'], fontweight=font_settings['weight'])
    ax1.legend(loc='upper right', fontsize=font_settings['legend_size'], framealpha=0.9)

    if np.any(valid):
        # Consider percentile bands for y-limit
        if not np.all(np.isnan(pct_10)):
            max_abs = max(abs(np.nanmin(pct_10)), abs(np.nanmax(pct_90)))
        else:
            max_abs = max(abs(np.nanmin(correlations)), abs(np.nanmax(correlations)))
        y_limit = max(0.15, min(0.8, max_abs * 1.1))
    else:
        y_limit = 0.3
    ax1.set_ylim(-y_limit, y_limit)
    ax1.grid(True, alpha=0.3)

    # Summary stats
    valid_corrs = [c for c in correlations if not np.isnan(c)]
    nn_corr = valid_corrs[0] if valid_corrs else np.nan

    short_range_corrs = [c for c, d in zip(correlations, bin_centers)
                         if d < 5 and not np.isnan(c)]
    sr_mean = np.mean(short_range_corrs) if short_range_corrs else np.nan

    summary_text = f'n = {n_samples} samples\n{total_axons:,} axons\n{total_segments:,} segments'
    if not np.isnan(nn_corr):
        summary_text += f'\nNN corr: {nn_corr:.3f}'
    if not np.isnan(sr_mean):
        summary_text += f'\n<5um mean: {sr_mean:.3f}'

    ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.9))

    # ===== Bottom plot: Number of segments =====
    colors = [sig_color if p < 0.05 and not np.isnan(p) else nonsig_color for p in p_values]
    ax2.bar(bin_centers, n_segments, width=bin_width * 0.85,
            color=colors, edgecolor='white', linewidth=0.5, alpha=0.8)

    ax2.set_xlabel('Mean Inter-Axon Distance (um)',
                   fontsize=font_settings['label_size'], fontweight=font_settings['weight'])
    ax2.set_ylabel('N Segments', fontsize=font_settings['label_size'])
    ax2.set_yscale('log')
    ax2.set_xlim(bin_edges[0], bin_edges[-1])
    ax2.grid(True, alpha=0.3)

    ax2.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"Saved figure to {output_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze spatial correlation of axon radius profiles between neighbors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all LM axon profile files
  python analyze_neighbor_profile_correlation.py \\
      "data/processed/LM/*_axon_profiles.npz" \\
      fig/neighbor_profile_correlation_pooled.png

  # With custom parameters
  python analyze_neighbor_profile_correlation.py \\
      "data/processed/LM/*_axon_profiles.npz" \\
      fig/neighbor_profile_correlation_pooled.png \\
      --d-max 10.0 --min-segment-length 5.0
        """
    )

    parser.add_argument('input_pattern', type=str,
                        help='Glob pattern for *_axon_profiles.npz files')
    parser.add_argument('output', type=Path,
                        help='Output PNG file path')
    parser.add_argument('--d-max', type=float, default=10.0,
                        help='Maximum XY distance for neighbor definition in um (default: 10.0)')
    parser.add_argument('--min-segment-length', type=float, default=5.0,
                        help='Minimum contiguous neighbor segment length in um (default: 5.0)')
    parser.add_argument('--z-bin-size', type=float, default=0.5,
                        help='Z-bin size in um (default: 0.5)')
    parser.add_argument('--min-points', type=int, default=10,
                        help='Minimum interpolated points for correlation (default: 10)')
    parser.add_argument('--min-axon-length', type=float, default=10.0,
                        help='Minimum axon length to include in um (default: 10.0)')
    parser.add_argument('--distance-bin-width', type=float, default=1.0,
                        help='Width of distance bins for pooling in um (default: 1.0)')
    parser.add_argument('--max-distance', type=float, default=10.0,
                        help='Maximum distance for analysis in um (default: 10.0)')

    args = parser.parse_args()

    # Find all matching files
    npz_files = sorted([Path(f) for f in glob_module.glob(args.input_pattern, recursive=True)])

    if not npz_files:
        logger.error(f"No files found matching pattern: {args.input_pattern}")
        return

    logger.info(f"Found {len(npz_files)} NPZ files")

    # Setup distance bins
    distance_bins = np.arange(0, args.max_distance + args.distance_bin_width,
                              args.distance_bin_width)

    # Process each sample
    all_results: List[SegmentResult] = []
    sample_names = []
    total_axons = 0

    for npz_file in npz_files:
        try:
            sample_name, results, n_axons = process_single_sample(
                npz_file,
                d_max=args.d_max,
                min_segment_length=args.min_segment_length,
                z_bin_size=args.z_bin_size,
                min_points=args.min_points,
                min_axon_length=args.min_axon_length
            )
            sample_names.append(sample_name)
            total_axons += n_axons
            all_results.extend(results)
            logger.info(f"  Accumulated {len(results)} correlations")

        except Exception as e:
            logger.error(f"Failed to process {npz_file.name}: {e}")
            import traceback
            traceback.print_exc()

    if not sample_names:
        logger.error("No samples processed successfully")
        return

    total_segments = len(all_results)
    logger.info(f"\nTotal: {total_segments:,} segment correlations from {len(sample_names)} samples")

    # Pool correlations
    logger.info("Pooling correlations by distance...")
    pooled = pool_correlations_by_distance(all_results, distance_bins)

    # Add metadata
    pooled['n_samples'] = len(sample_names)
    pooled['total_axons'] = total_axons
    pooled['total_segments'] = total_segments
    pooled['sample_names'] = sample_names
    pooled['parameters'] = {
        'd_max_um': args.d_max,
        'min_segment_length_um': args.min_segment_length,
        'z_bin_size_um': args.z_bin_size,
        'min_points': args.min_points,
        'min_axon_length_um': args.min_axon_length,
        'distance_bin_width_um': args.distance_bin_width,
        'max_distance_um': args.max_distance
    }

    # Plot 1: Relative residual correlations (original)
    plot_correlation_by_distance(
        pooled, args.output,
        len(sample_names), total_axons, total_segments,
        correlation_key='mean_correlations',
        ci_low_key='ci_low',
        ci_high_key='ci_high',
        p_values_key='p_values',
        n_segments_key='n_segments',
        pct_25_key='pct_25',
        pct_75_key='pct_75',
        pct_10_key='pct_10',
        pct_90_key='pct_90',
        title_suffix=' - Relative Residuals',
        ylabel='Mean Residual Correlation',
        subtitle=r'$r = \mathrm{corr}(\Delta r_i - \bar{\Delta r}, \Delta r_j - \bar{\Delta r})$ (normalized, local mean subtracted)'
    )

    # Plot 2: Absolute deviation correlations (new)
    abs_output = args.output.parent / (args.output.stem + '_absolute' + args.output.suffix)
    plot_correlation_by_distance(
        pooled, abs_output,
        len(sample_names), total_axons, total_segments,
        correlation_key='mean_correlations_abs',
        ci_low_key='ci_low_abs',
        ci_high_key='ci_high_abs',
        p_values_key='p_values_abs',
        n_segments_key='n_segments_abs',
        pct_25_key='pct_25_abs',
        pct_75_key='pct_75_abs',
        pct_10_key='pct_10_abs',
        pct_90_key='pct_90_abs',
        title_suffix=' - Absolute Deviations',
        ylabel='Mean Absolute Deviation Correlation',
        subtitle=r'$r = \mathrm{corr}(r_i - \bar{r}_i, r_j - \bar{r}_j)$ (in $\mu m$, not normalized)'
    )

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Samples processed: {len(sample_names)}")
    logger.info(f"Total axons: {total_axons:,}")
    logger.info(f"Total segments: {total_segments:,}")

    # Summary for relative residual correlations
    valid_corrs = [c for c in pooled['mean_correlations'] if not np.isnan(c)]
    if valid_corrs:
        nn_corr = valid_corrs[0]
        short_range = [c for c, d in zip(pooled['mean_correlations'], pooled['bin_centers'])
                       if d < 5 and not np.isnan(c)]
        sr_mean = np.mean(short_range) if short_range else np.nan

        logger.info(f"[Relative Residuals] Nearest-neighbor mean correlation: {nn_corr:.4f}")
        if not np.isnan(sr_mean):
            logger.info(f"[Relative Residuals] Short-range (<5um) mean: {sr_mean:.4f}")

    # Summary for absolute deviation correlations
    valid_corrs_abs = [c for c in pooled['mean_correlations_abs'] if not np.isnan(c)]
    if valid_corrs_abs:
        nn_corr_abs = valid_corrs_abs[0]
        short_range_abs = [c for c, d in zip(pooled['mean_correlations_abs'], pooled['bin_centers'])
                           if d < 5 and not np.isnan(c)]
        sr_mean_abs = np.mean(short_range_abs) if short_range_abs else np.nan

        logger.info(f"[Absolute Deviations] Nearest-neighbor mean correlation: {nn_corr_abs:.4f}")
        if not np.isnan(sr_mean_abs):
            logger.info(f"[Absolute Deviations] Short-range (<5um) mean: {sr_mean_abs:.4f}")

    # Save JSON results
    json_file = args.output.with_suffix('.json')
    with open(json_file, 'w') as f:
        json.dump(pooled, f, indent=2)
    logger.info(f"Saved results to {json_file}")


if __name__ == '__main__':
    main()
