#!/usr/bin/env python3
"""
Analyze 2D spatial correlation of axon radius variations within cross-sections.

Implements multiple complementary methods to quantify how axon radius variations
are spatially correlated:

1. Semivariogram: γ(h) = 0.5 * E[(r(x) - r(x+h))²] vs lag distance
   - Outputs: range (correlation length), sill (total variance), nugget

2. Per-slice neighbor correlation: Correlation of relative radii between neighbors
   - Uses KDTree for efficient neighbor queries at multiple kernel radii

3. Local Moran's I: Spatial autocorrelation statistic with hotspot detection
   - Tests whether similar values cluster spatially

4. Radial power spectrum: Fourier analysis for characteristic length scales
   - Identifies dominant wavelengths in the spatial pattern

Usage:
    python examples/analyze_spatial_correlation_2d.py \
        data/processed/LM/sham_25_ipsi_axon_profiles.npz \
        fig/spatial_correlation_2d \
        --mat-file data/raw/Sham_25_ipsi/LM_25_ipsi_myelinated_axons.mat \
        --population cc
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from skimage.measure import regionprops
from tqdm import tqdm

# Import from axonometry library
sys.path.insert(0, str(Path(__file__).parent.parent))
from axonometry import get_plot_settings
from axonometry.io import load_volume_with_metadata, resample_to_isotropic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MIN_POINTS_SEMIVARIOGRAM = 100
MIN_POINTS_NEIGHBOR_CORR = 100
MIN_POINTS_MORANS_I = 50
MIN_POINTS_POWER_SPECTRUM = 100
MIN_POINTS_ANALYSIS = 100
MIN_PAIRS_PER_BIN = 10
MIN_POINTS_CORRELATION = 30

# Load plot settings
settings = get_plot_settings()

# Global variables for worker processes
_volume_data = None
_slice_axis = None
_voxel_size_um = None
_label_filter = None
_label_to_mean = None


# =============================================================================
# Data Loading
# =============================================================================

def load_axon_profiles(npz_file: Path) -> Dict:
    """Load axon profile data."""
    data = np.load(npz_file, allow_pickle=True)
    return {
        'labels': data['labels'],
        'mean_radii': data['mean_radii_um'],
        'voxel_size': float(data['voxel_size_um'][0]),
    }


def load_population_data(npz_file: Path, population: str) -> Tuple[Set[int], int]:
    """Load population labels and dominant axis from populations JSON."""
    sample_name = npz_file.stem.replace('_axon_profiles', '')
    pop_file = npz_file.parent / f'{sample_name}_populations.json'

    if not pop_file.exists():
        raise FileNotFoundError(f"Population file not found: {pop_file}")

    with open(pop_file) as f:
        data = json.load(f)

    for pop in data['populations']:
        if pop['name'].lower() == population.lower():
            labels = set(pop['axon_labels'])
            axis = pop['dominant_axis']
            logger.info(f"Loaded {population.upper()} population: {len(labels)} axons, "
                       f"dominant axis = {axis} ({pop['dominant_axis_name']})")
            return labels, axis

    raise ValueError(f"Population '{population}' not found in {pop_file}")


def load_volume(mat_file: Path) -> Tuple[np.ndarray, float]:
    """Load 3D labeled volume from .mat file."""
    logger.info(f"Loading volume from {mat_file}")
    volume, voxel_size_tuple, _ = load_volume_with_metadata(mat_file)
    volume, voxel_size = resample_to_isotropic(volume, voxel_size_tuple)
    logger.info(f"  Volume shape: {volume.shape}, voxel size: {voxel_size:.4f} um")
    return volume, voxel_size


# =============================================================================
# Slice Processing
# =============================================================================

def _init_worker(volume_data: np.ndarray, slice_axis: int, voxel_size_um: float,
                 label_filter: Optional[Set[int]], label_to_mean: Dict[int, float]):
    """Initialize worker process with shared data."""
    global _volume_data, _slice_axis, _voxel_size_um, _label_filter, _label_to_mean
    _volume_data = volume_data
    _slice_axis = slice_axis
    _voxel_size_um = voxel_size_um
    _label_filter = label_filter
    _label_to_mean = label_to_mean


def _process_slice(z: int) -> Dict[int, Tuple[float, float, float, float]]:
    """
    Extract data for each axon in a slice.

    Returns: {label: (centroid_y_um, centroid_x_um, radius_um, relative_radius)}
    """
    pixel_area_um2 = _voxel_size_um * _voxel_size_um

    if _slice_axis == 0:
        slice_2d = _volume_data[z, :, :]
    elif _slice_axis == 1:
        slice_2d = _volume_data[:, z, :]
    else:
        slice_2d = _volume_data[:, :, z]

    # Filter to population if specified
    if _label_filter is not None:
        mask = np.isin(slice_2d, list(_label_filter))
        slice_2d = np.where(mask, slice_2d, 0)

    regions = regionprops(slice_2d.astype(np.int32))

    result = {}
    for region in regions:
        label = region.label
        if label not in _label_to_mean or _label_to_mean[label] <= 0:
            continue

        area_um2 = region.area * pixel_area_um2
        radius_um = np.sqrt(area_um2 / np.pi)
        mean_radius = _label_to_mean[label]
        relative_radius = radius_um / mean_radius

        cy, cx = region.centroid
        cy_um = cy * _voxel_size_um
        cx_um = cx * _voxel_size_um

        result[label] = (cy_um, cx_um, radius_um, relative_radius)

    return result


def extract_slice_data(volume: np.ndarray, slice_axis: int, voxel_size: float,
                       label_filter: Optional[Set[int]], label_to_mean: Dict[int, float],
                       slice_step: int = 10, n_jobs: int = -1) -> List[Dict]:
    """
    Extract per-slice data for all sampled slices.

    Returns list of dicts, one per slice.
    """
    n_slices = volume.shape[slice_axis]
    slice_indices = list(range(0, n_slices, slice_step))

    if n_jobs == -1:
        n_workers = cpu_count()
    else:
        n_workers = min(n_jobs, cpu_count())

    logger.info(f"Processing {len(slice_indices)} slices (step={slice_step})...")

    if n_workers > 1:
        with Pool(processes=n_workers,
                  initializer=_init_worker,
                  initargs=(volume, slice_axis, voxel_size, label_filter, label_to_mean)) as pool:
            slice_data = list(tqdm(
                pool.imap(_process_slice, slice_indices),
                total=len(slice_indices),
                desc="  Extracting slices",
                leave=False
            ))
    else:
        _init_worker(volume, slice_axis, voxel_size, label_filter, label_to_mean)
        slice_data = [_process_slice(z) for z in tqdm(slice_indices, desc="  Extracting slices", leave=False)]

    # Count total data points
    total_points = sum(len(sd) for sd in slice_data)
    logger.info(f"  Extracted {total_points:,} data points from {len(slice_data)} slices")

    return slice_data


# =============================================================================
# Method 1: Semivariogram
# =============================================================================

def compute_semivariogram(slice_data: List[Dict],
                          max_lag: float = 20.0,
                          n_bins: int = 40,
                          max_pairs: int = 500000) -> Dict:
    """
    Compute empirical semivariogram: γ(h) = 0.5 * E[(r(x) - r(x+h))²]

    Args:
        slice_data: List of per-slice data dicts
        max_lag: Maximum lag distance in μm
        n_bins: Number of lag bins
        max_pairs: Maximum pairs to sample (for efficiency)

    Returns:
        Dict with lag_centers, gamma values, n_pairs, fitted parameters
    """
    logger.info("Computing semivariogram...")

    # Collect all points
    all_points = []
    for slice_dict in slice_data:
        for label, (cy, cx, radius, rel_radius) in slice_dict.items():
            all_points.append((cy, cx, rel_radius))

    if len(all_points) < MIN_POINTS_SEMIVARIOGRAM:
        logger.warning("  Insufficient points for semivariogram")
        return {'lag_centers': [], 'gamma': [], 'n_pairs': [], 'range': np.nan, 'sill': np.nan, 'nugget': np.nan}

    points = np.array(all_points)
    positions = points[:, :2]
    values = points[:, 2]
    n_points = len(points)

    # Compute pairwise distances and squared differences
    lag_bins = np.linspace(0, max_lag, n_bins + 1)
    lag_centers = (lag_bins[:-1] + lag_bins[1:]) / 2
    gamma_sums = np.zeros(n_bins)
    gamma_counts = np.zeros(n_bins, dtype=int)

    # Use random sampling if too many pairs
    n_pairs_total = n_points * (n_points - 1) // 2
    if n_pairs_total > max_pairs:
        # Random sampling of pairs
        rng = np.random.default_rng(42)  # Seed for semivariogram
        n_sample = min(max_pairs, n_pairs_total)
        for _ in tqdm(range(n_sample), desc="  Sampling pairs", leave=False):
            i, j = rng.choice(n_points, size=2, replace=False)
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < max_lag:
                bin_idx = int(dist / max_lag * n_bins)
                if bin_idx < n_bins:
                    sq_diff = (values[i] - values[j]) ** 2
                    gamma_sums[bin_idx] += sq_diff
                    gamma_counts[bin_idx] += 1
    else:
        # Full computation for smaller datasets
        for i in tqdm(range(n_points - 1), desc="  Computing pairs", leave=False):
            dists = np.linalg.norm(positions[i+1:] - positions[i], axis=1)
            sq_diffs = (values[i+1:] - values[i]) ** 2

            for dist, sq_diff in zip(dists, sq_diffs):
                if dist < max_lag:
                    bin_idx = int(dist / max_lag * n_bins)
                    if bin_idx < n_bins:
                        gamma_sums[bin_idx] += sq_diff
                        gamma_counts[bin_idx] += 1

    # Compute gamma = 0.5 * mean(squared differences)
    with np.errstate(invalid='ignore'):
        gamma = 0.5 * gamma_sums / gamma_counts
    gamma = np.where(gamma_counts > MIN_PAIRS_PER_BIN, gamma, np.nan)

    # Estimate parameters: range, sill, nugget
    valid = ~np.isnan(gamma)
    if np.sum(valid) >= 3:
        # Nugget: extrapolate to lag=0
        nugget = gamma[valid][0] if valid[0] else np.nanmin(gamma)

        # Sill: asymptotic variance
        sill = np.nanmax(gamma)

        # Range: lag where gamma reaches ~95% of sill
        threshold = nugget + 0.95 * (sill - nugget)
        range_idx = np.where(valid & (gamma >= threshold))[0]
        if len(range_idx) > 0:
            range_um = lag_centers[range_idx[0]]
        else:
            range_um = max_lag
    else:
        nugget, sill, range_um = np.nan, np.nan, np.nan

    logger.info(f"  Range: {range_um:.2f} μm, Sill: {sill:.4f}, Nugget: {nugget:.4f}")

    return {
        'lag_centers': lag_centers.tolist(),
        'gamma': [float(g) if not np.isnan(g) else None for g in gamma],
        'n_pairs': gamma_counts.tolist(),
        'range': float(range_um) if not np.isnan(range_um) else None,
        'sill': float(sill) if not np.isnan(sill) else None,
        'nugget': float(nugget) if not np.isnan(nugget) else None
    }


# =============================================================================
# Method 2: Per-slice Neighbor Correlation
# =============================================================================

def compute_neighbor_correlation(slice_data: List[Dict],
                                  kernel_radii: List[float] = None,
                                  max_points: int = 50000) -> Dict:
    """
    Compute correlation between own relative radius and neighbor mean.

    For each axon, finds neighbors within kernel radius and correlates
    the axon's relative radius with the mean of its neighbors.

    Args:
        slice_data: List of per-slice data dicts
        kernel_radii: List of kernel radii to test in μm
        max_points: Maximum points to sample

    Returns:
        Dict with kernel_radii, correlations, p_values, n_points
    """
    if kernel_radii is None:
        kernel_radii = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]

    logger.info("Computing neighbor correlations...")

    # Collect all points grouped by slice
    all_points = []
    for slice_idx, slice_dict in enumerate(slice_data):
        for label, (cy, cx, radius, rel_radius) in slice_dict.items():
            all_points.append({
                'slice_idx': slice_idx,
                'label': label,
                'centroid': (cy, cx),
                'relative_radius': rel_radius
            })

    n_points = len(all_points)
    if n_points < MIN_POINTS_NEIGHBOR_CORR:
        logger.warning("  Insufficient points for neighbor correlation")
        return {'kernel_radii': kernel_radii, 'correlations': [np.nan] * len(kernel_radii),
                'p_values': [np.nan] * len(kernel_radii), 'n_valid': [0] * len(kernel_radii)}

    # Subsample if needed
    if n_points > max_points:
        rng = np.random.default_rng(43)  # Seed for neighbor correlation
        indices = rng.choice(n_points, size=max_points, replace=False)
        sampled_points = [all_points[i] for i in indices]
        logger.info(f"  Subsampled to {max_points:,} points")
    else:
        sampled_points = all_points

    # Group by slice and build KDTrees
    points_by_slice = defaultdict(list)
    for pt in all_points:
        points_by_slice[pt['slice_idx']].append(pt)

    trees_by_slice = {}
    for slice_idx, pts in points_by_slice.items():
        if len(pts) > 1:
            centroids = np.array([p['centroid'] for p in pts])
            trees_by_slice[slice_idx] = (cKDTree(centroids), pts)

    results = {
        'kernel_radii': kernel_radii,
        'correlations': [],
        'p_values': [],
        'n_valid': [],
        'mean_n_neighbors': []
    }

    for kernel_r in kernel_radii:
        own_values = []
        neighbor_means = []
        n_neighbors_list = []

        for pt in sampled_points:
            slice_idx = pt['slice_idx']
            if slice_idx not in trees_by_slice:
                continue

            tree, pts_in_slice = trees_by_slice[slice_idx]
            centroid = pt['centroid']

            # Find neighbors within radius
            neighbor_indices = tree.query_ball_point(centroid, kernel_r)

            # Exclude self
            neighbor_values = []
            for ni in neighbor_indices:
                if pts_in_slice[ni]['label'] != pt['label']:
                    neighbor_values.append(pts_in_slice[ni]['relative_radius'])

            n_neighbors_list.append(len(neighbor_values))

            if len(neighbor_values) > 0:
                own_values.append(pt['relative_radius'])
                neighbor_means.append(np.mean(neighbor_values))

        n_valid = len(own_values)
        if n_valid > MIN_POINTS_CORRELATION:
            r, p = pearsonr(own_values, neighbor_means)
        else:
            r, p = np.nan, np.nan

        results['correlations'].append(float(r) if not np.isnan(r) else None)
        results['p_values'].append(float(p) if not np.isnan(p) else None)
        results['n_valid'].append(n_valid)
        results['mean_n_neighbors'].append(float(np.mean(n_neighbors_list)) if n_neighbors_list else 0)

        logger.info(f"  r={kernel_r:.1f}μm: corr={r:.4f}, p={p:.2e}, n={n_valid}")

    return results


# =============================================================================
# Method 3: Moran's I
# =============================================================================

def compute_morans_i(slice_data: List[Dict],
                     distance_thresholds: List[float] = None,
                     max_points: int = 2000,
                     n_permutations: int = 199) -> Dict:
    """
    Compute Moran's I spatial autocorrelation statistic.

    Moran's I = (n/S0) * Σᵢ Σⱼ wᵢⱼ(xᵢ - x̄)(xⱼ - x̄) / Σᵢ(xᵢ - x̄)²

    where wᵢⱼ is the spatial weight (1 if within threshold, 0 otherwise).

    Args:
        slice_data: List of per-slice data dicts
        distance_thresholds: List of distance thresholds in μm
        max_points: Maximum points per slice for efficiency
        n_permutations: Number of permutations for significance testing

    Returns:
        Dict with distance thresholds, Moran's I values, expected I, p-values
    """
    if distance_thresholds is None:
        distance_thresholds = [2.0, 5.0, 10.0, 15.0, 20.0]

    logger.info("Computing Moran's I...")

    results = {
        'distance_thresholds': distance_thresholds,
        'morans_i': [],
        'expected_i': [],
        'p_values': [],
        'z_scores': [],
        'n_points': []
    }

    # Collect all points (pool across slices for global analysis)
    all_points = []
    for slice_dict in slice_data:
        for label, (cy, cx, radius, rel_radius) in slice_dict.items():
            all_points.append((cy, cx, rel_radius))

    if len(all_points) < MIN_POINTS_MORANS_I:
        logger.warning("  Insufficient points for Moran's I")
        for _ in distance_thresholds:
            results['morans_i'].append(None)
            results['expected_i'].append(None)
            results['p_values'].append(None)
            results['z_scores'].append(None)
            results['n_points'].append(0)
        return results

    # Subsample if needed
    if len(all_points) > max_points:
        rng = np.random.default_rng(44)  # Seed for Moran's I
        indices = rng.choice(len(all_points), size=max_points, replace=False)
        all_points = [all_points[i] for i in indices]
        logger.info(f"  Subsampled to {max_points:,} points")

    points = np.array(all_points)
    positions = points[:, :2]
    values = points[:, 2]
    n = len(values)

    # Center values
    mean_val = np.mean(values)
    centered = values - mean_val
    sum_sq = np.sum(centered ** 2)

    # Build KDTree once
    tree = cKDTree(positions)

    # Random generator for permutation tests
    rng_perm = np.random.default_rng(44)

    for threshold in distance_thresholds:
        # Build binary spatial weights matrix
        pairs = tree.query_pairs(threshold)

        if len(pairs) == 0:
            results['morans_i'].append(None)
            results['expected_i'].append(None)
            results['p_values'].append(None)
            results['z_scores'].append(None)
            results['n_points'].append(n)
            continue

        # Compute Moran's I
        # S0 = sum of all weights (2 * n_pairs for symmetric matrix)
        S0 = 2 * len(pairs)

        # Numerator: sum of wij * (xi - xbar)(xj - xbar)
        cross_sum = 0.0
        for i, j in pairs:
            cross_sum += centered[i] * centered[j]
        cross_sum *= 2  # Symmetric

        I = (n / S0) * (cross_sum / sum_sq) if sum_sq > 0 else 0

        # Expected I under null hypothesis
        E_I = -1 / (n - 1)

        # Permutation test for significance
        perm_Is = []
        for _ in range(n_permutations):
            perm_vals = rng_perm.permutation(centered)
            perm_cross = 0.0
            for i, j in pairs:
                perm_cross += perm_vals[i] * perm_vals[j]
            perm_cross *= 2
            perm_I = (n / S0) * (perm_cross / sum_sq) if sum_sq > 0 else 0
            perm_Is.append(perm_I)

        perm_Is = np.array(perm_Is)
        # Two-tailed p-value
        p_value = np.mean(np.abs(perm_Is) >= np.abs(I))
        z_score = (I - E_I) / np.std(perm_Is) if np.std(perm_Is) > 0 else 0

        results['morans_i'].append(float(I))
        results['expected_i'].append(float(E_I))
        results['p_values'].append(float(p_value))
        results['z_scores'].append(float(z_score))
        results['n_points'].append(n)

        sig = '*' if p_value < 0.05 else ''
        logger.info(f"  d={threshold:.1f}μm: I={I:.4f}, E[I]={E_I:.4f}, z={z_score:.2f}, p={p_value:.3f}{sig}")

    return results


# =============================================================================
# Method 4: Radial Power Spectrum
# =============================================================================

def compute_power_spectrum(slice_data: List[Dict],
                           grid_size: float = 1.0,
                           max_wavelength: float = 50.0) -> Dict:
    """
    Compute radial power spectrum of relative radius spatial pattern.

    Interpolates data onto a regular grid, computes 2D FFT, and
    averages power in radial bins to get P(k) vs wavenumber.

    Args:
        slice_data: List of per-slice data dicts
        grid_size: Grid cell size in μm
        max_wavelength: Maximum wavelength to report in μm

    Returns:
        Dict with wavelengths, power values, dominant wavelength
    """
    logger.info("Computing power spectrum...")

    # Collect all points
    all_points = []
    for slice_dict in slice_data:
        for label, (cy, cx, radius, rel_radius) in slice_dict.items():
            all_points.append((cy, cx, rel_radius))

    if len(all_points) < MIN_POINTS_POWER_SPECTRUM:
        logger.warning("  Insufficient points for power spectrum")
        return {'wavelengths': [], 'power': [], 'dominant_wavelength': None}

    points = np.array(all_points)
    positions = points[:, :2]
    values = points[:, 2]

    # Determine grid bounds
    y_min, x_min = positions.min(axis=0)
    y_max, x_max = positions.max(axis=0)

    # Create grid
    ny = int((y_max - y_min) / grid_size) + 1
    nx = int((x_max - x_min) / grid_size) + 1

    if ny < 16 or nx < 16:
        logger.warning("  Grid too small for meaningful FFT")
        return {'wavelengths': [], 'power': [], 'dominant_wavelength': None}

    # Interpolate onto grid using nearest neighbor (simple approach)
    grid = np.zeros((ny, nx))
    counts = np.zeros((ny, nx))

    for (y, x), val in zip(positions, values):
        iy = int((y - y_min) / grid_size)
        ix = int((x - x_min) / grid_size)
        iy = min(iy, ny - 1)
        ix = min(ix, nx - 1)
        grid[iy, ix] += val
        counts[iy, ix] += 1

    # Average where multiple points fall in same cell
    with np.errstate(invalid='ignore'):
        grid = np.where(counts > 0, grid / counts, 1.0)  # Fill empty with mean (1.0)

    # Subtract mean and apply window
    grid = grid - np.mean(grid)
    window_y = np.hanning(ny)
    window_x = np.hanning(nx)
    window = np.outer(window_y, window_x)
    grid = grid * window

    # 2D FFT
    fft2 = np.fft.fft2(grid)
    power2d = np.abs(fft2) ** 2

    # Shift to center
    power2d = np.fft.fftshift(power2d)

    # Compute radial average
    cy, cx = ny // 2, nx // 2
    y_idx, x_idx = np.ogrid[:ny, :nx]
    r = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)

    # Convert to wavelength
    # k = r / (grid_size * n), wavelength = 1/k = grid_size * n / r
    max_r = min(cy, cx)
    n_bins = max_r

    r_bins = np.arange(1, n_bins + 1)
    power_radial = []

    for ri in r_bins:
        mask = (r >= ri - 0.5) & (r < ri + 0.5)
        if np.any(mask):
            power_radial.append(np.mean(power2d[mask]))
        else:
            power_radial.append(0)

    power_radial = np.array(power_radial)

    # Convert r to wavelength
    wavelengths = grid_size * max(ny, nx) / r_bins
    valid = wavelengths <= max_wavelength

    wavelengths = wavelengths[valid]
    power_radial = power_radial[valid]

    # Find dominant wavelength (global maximum)
    if len(wavelengths) > 0:
        peak_idx = np.argmax(power_radial)
        dominant = wavelengths[peak_idx]
    else:
        dominant = None

    logger.info(f"  Dominant wavelength: {dominant:.2f} μm" if dominant else "  No dominant wavelength found")

    return {
        'wavelengths': wavelengths.tolist(),
        'power': power_radial.tolist(),
        'dominant_wavelength': float(dominant) if dominant else None
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_summary(semivariogram: Dict, neighbor_corr: Dict, morans_i: Dict,
                 power_spectrum: Dict, output_file: Path,
                 population: str, sample_name: str, n_slices: int, n_points: int):
    """Create 2x2 summary figure with all methods."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    font_settings = settings.fonts

    # Colors
    main_color = '#2E86AB'
    sig_color = '#E94F37'

    # === Panel 1: Semivariogram ===
    ax1 = axes[0, 0]
    if semivariogram['lag_centers']:
        lags = np.array(semivariogram['lag_centers'])
        gamma = np.array([g if g is not None else np.nan for g in semivariogram['gamma']])
        valid = ~np.isnan(gamma)

        ax1.plot(lags[valid], gamma[valid], 'o-', color=main_color, markersize=5, linewidth=1.5)

        # Add fitted parameters
        if semivariogram['range'] is not None:
            ax1.axvline(semivariogram['range'], color=sig_color, linestyle='--',
                       label=f"Range: {semivariogram['range']:.1f} μm")
        if semivariogram['sill'] is not None:
            ax1.axhline(semivariogram['sill'], color='#666666', linestyle=':',
                       label=f"Sill: {semivariogram['sill']:.3f}")

        ax1.legend(loc='lower right', fontsize=font_settings['legend_size'])

    ax1.set_xlabel('Lag Distance (μm)', fontsize=font_settings['label_size'],
                   fontweight=font_settings['weight'])
    ax1.set_ylabel('γ(h)', fontsize=font_settings['label_size'],
                   fontweight=font_settings['weight'])
    ax1.set_title('Semivariogram', fontsize=font_settings['title_size'],
                  fontweight=font_settings['weight'])
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, None)
    ax1.set_ylim(0, None)

    # === Panel 2: Neighbor Correlation ===
    ax2 = axes[0, 1]
    if neighbor_corr['correlations']:
        radii = np.array(neighbor_corr['kernel_radii'])
        corrs = np.array([c if c is not None else np.nan for c in neighbor_corr['correlations']])
        pvals = np.array([p if p is not None else np.nan for p in neighbor_corr['p_values']])
        valid = ~np.isnan(corrs)

        sig_mask = valid & (pvals < 0.05)
        nonsig_mask = valid & (pvals >= 0.05)

        ax2.axhline(0, color='#666666', linestyle='-', linewidth=0.8, alpha=0.5)

        if np.any(nonsig_mask):
            ax2.plot(radii[nonsig_mask], corrs[nonsig_mask], 'o', color='#A1A1A1',
                    markersize=8, markerfacecolor='white', markeredgewidth=2, label='p ≥ 0.05')
        if np.any(sig_mask):
            ax2.plot(radii[sig_mask], corrs[sig_mask], 'o', color=sig_color,
                    markersize=8, markeredgewidth=2, label='p < 0.05')
        if np.any(valid):
            ax2.plot(radii[valid], corrs[valid], '-', color=main_color, alpha=0.6, linewidth=1.5)

        ax2.legend(loc='upper right', fontsize=font_settings['legend_size'])

    ax2.set_xlabel('Kernel Radius (μm)', fontsize=font_settings['label_size'],
                   fontweight=font_settings['weight'])
    ax2.set_ylabel('Correlation', fontsize=font_settings['label_size'],
                   fontweight=font_settings['weight'])
    ax2.set_title('Neighbor Correlation', fontsize=font_settings['title_size'],
                  fontweight=font_settings['weight'])
    ax2.grid(True, alpha=0.3)

    # === Panel 3: Moran's I ===
    ax3 = axes[1, 0]
    if morans_i['morans_i']:
        thresholds = np.array(morans_i['distance_thresholds'])
        I_vals = np.array([i if i is not None else np.nan for i in morans_i['morans_i']])
        E_I = np.array([e if e is not None else np.nan for e in morans_i['expected_i']])
        pvals = np.array([p if p is not None else np.nan for p in morans_i['p_values']])
        valid = ~np.isnan(I_vals)

        if np.any(valid):
            # Expected I line
            ax3.axhline(E_I[valid][0], color='#666666', linestyle='--', label='Expected I (random)')

            sig_mask = valid & (pvals < 0.05)
            nonsig_mask = valid & (pvals >= 0.05)

            if np.any(nonsig_mask):
                ax3.plot(thresholds[nonsig_mask], I_vals[nonsig_mask], 'o', color='#A1A1A1',
                        markersize=8, markerfacecolor='white', markeredgewidth=2, label='p ≥ 0.05')
            if np.any(sig_mask):
                ax3.plot(thresholds[sig_mask], I_vals[sig_mask], 'o', color=sig_color,
                        markersize=8, markeredgewidth=2, label='p < 0.05')

            ax3.plot(thresholds[valid], I_vals[valid], '-', color=main_color, alpha=0.6, linewidth=1.5)
            ax3.legend(loc='upper right', fontsize=font_settings['legend_size'])

    ax3.set_xlabel('Distance Threshold (μm)', fontsize=font_settings['label_size'],
                   fontweight=font_settings['weight'])
    ax3.set_ylabel("Moran's I", fontsize=font_settings['label_size'],
                   fontweight=font_settings['weight'])
    ax3.set_title("Moran's I (Spatial Autocorrelation)", fontsize=font_settings['title_size'],
                  fontweight=font_settings['weight'])
    ax3.grid(True, alpha=0.3)

    # === Panel 4: Power Spectrum ===
    ax4 = axes[1, 1]
    if power_spectrum['wavelengths']:
        wavelengths = np.array(power_spectrum['wavelengths'])
        power = np.array(power_spectrum['power'])

        ax4.loglog(wavelengths, power, '-', color=main_color, linewidth=1.5)

        if power_spectrum['dominant_wavelength'] is not None:
            ax4.axvline(power_spectrum['dominant_wavelength'], color=sig_color, linestyle='--',
                       label=f"Peak: {power_spectrum['dominant_wavelength']:.1f} μm")
            ax4.legend(loc='upper right', fontsize=font_settings['legend_size'])

    ax4.set_xlabel('Wavelength (μm)', fontsize=font_settings['label_size'],
                   fontweight=font_settings['weight'])
    ax4.set_ylabel('Power', fontsize=font_settings['label_size'],
                   fontweight=font_settings['weight'])
    ax4.set_title('Radial Power Spectrum', fontsize=font_settings['title_size'],
                  fontweight=font_settings['weight'])
    ax4.grid(True, alpha=0.3, which='both')

    # Overall title
    fig.suptitle(f'Spatial Correlation Analysis: {sample_name} - {population.upper()}\n'
                 f'({n_slices} slices, {n_points:,} points)',
                 fontsize=font_settings['title_size'] + 2, fontweight=font_settings['weight'], y=1.02)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"Saved figure to {output_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze 2D spatial correlation of axon radius variations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze CC population
    python examples/analyze_spatial_correlation_2d.py \\
        data/processed/LM/sham_25_ipsi_axon_profiles.npz \\
        fig/spatial_correlation_2d \\
        --mat-file data/raw/Sham_25_ipsi/LM_25_ipsi_myelinated_axons.mat \\
        --population cc

    # Analyze both populations
    python examples/analyze_spatial_correlation_2d.py \\
        data/processed/LM/sham_25_ipsi_axon_profiles.npz \\
        fig/spatial_correlation_2d \\
        --mat-file data/raw/Sham_25_ipsi/LM_25_ipsi_myelinated_axons.mat \\
        --population all
"""
    )

    parser.add_argument('input', type=Path,
                        help='Path to axon profiles NPZ file')
    parser.add_argument('output_dir', type=Path,
                        help='Output directory for figures and results')
    parser.add_argument('--mat-file', type=Path, required=True,
                        help='Path to .mat file with labeled volume')
    parser.add_argument('--population', type=str, default='all', choices=['cc', 'cg', 'all'],
                        help='Population to analyze (default: all)')
    parser.add_argument('--slice-step', type=int, default=10,
                        help='Step between slices (default: 10)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 = all cores)')
    parser.add_argument('--max-lag', type=float, default=20.0,
                        help='Maximum lag distance for semivariogram in μm (default: 20)')
    parser.add_argument('--max-wavelength', type=float, default=50.0,
                        help='Maximum wavelength for power spectrum in μm (default: 50)')

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return

    if not args.mat_file.exists():
        logger.error(f"Mat file not found: {args.mat_file}")
        return

    # Load profiles for mean radius lookup
    profiles = load_axon_profiles(args.input)
    label_to_mean_all = {
        int(lab): rad
        for lab, rad in zip(profiles['labels'], profiles['mean_radii'])
        if rad > 0
    }

    # Load volume
    volume, voxel_size = load_volume(args.mat_file)

    # Determine populations to analyze
    sample_name = args.input.stem.replace('_axon_profiles', '')

    if args.population == 'all':
        populations = ['cc', 'cg']
    else:
        populations = [args.population]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for population in populations:
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing {population.upper()} population")
        logger.info(f"{'='*60}")

        try:
            pop_labels, slice_axis = load_population_data(args.input, population)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"  {e}")
            continue

        # Filter label_to_mean to this population
        label_to_mean = {lab: rad for lab, rad in label_to_mean_all.items() if lab in pop_labels}
        logger.info(f"  {len(label_to_mean)} axons with valid mean radii")

        # Extract slice data
        slice_data = extract_slice_data(
            volume, slice_axis, voxel_size,
            label_filter=pop_labels, label_to_mean=label_to_mean,
            slice_step=args.slice_step, n_jobs=args.n_jobs
        )

        n_points = sum(len(sd) for sd in slice_data)
        if n_points < MIN_POINTS_ANALYSIS:
            logger.warning(f"  Insufficient data points ({n_points}), skipping")
            continue

        # Compute all methods
        semivariogram = compute_semivariogram(slice_data, max_lag=args.max_lag)
        neighbor_corr = compute_neighbor_correlation(slice_data)
        morans_i = compute_morans_i(slice_data)
        power_spectrum = compute_power_spectrum(slice_data, max_wavelength=args.max_wavelength)

        # Create summary figure
        output_file = args.output_dir / f'{sample_name}_{population.upper()}_spatial_correlation.png'
        plot_summary(
            semivariogram, neighbor_corr, morans_i, power_spectrum,
            output_file, population, sample_name,
            n_slices=len(slice_data), n_points=n_points
        )

        # Save JSON results
        results = {
            'sample_name': sample_name,
            'population': population.upper(),
            'n_slices': len(slice_data),
            'n_points': n_points,
            'slice_step': args.slice_step,
            'semivariogram': semivariogram,
            'neighbor_correlation': neighbor_corr,
            'morans_i': morans_i,
            'power_spectrum': power_spectrum
        }

        json_file = output_file.with_suffix('.json')
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {json_file}")

        # Print summary
        logger.info(f"\n--- Summary for {population.upper()} ---")
        if semivariogram['range'] is not None:
            logger.info(f"  Semivariogram range: {semivariogram['range']:.2f} μm")
        if neighbor_corr['correlations'] and neighbor_corr['correlations'][0] is not None:
            logger.info(f"  Nearest-neighbor correlation (r=1μm): {neighbor_corr['correlations'][0]:.4f}")
        if morans_i['morans_i'] and morans_i['morans_i'][0] is not None:
            logger.info(f"  Moran's I (d=2μm): {morans_i['morans_i'][0]:.4f}")
        if power_spectrum['dominant_wavelength'] is not None:
            logger.info(f"  Dominant wavelength: {power_spectrum['dominant_wavelength']:.2f} μm")


if __name__ == '__main__':
    main()
