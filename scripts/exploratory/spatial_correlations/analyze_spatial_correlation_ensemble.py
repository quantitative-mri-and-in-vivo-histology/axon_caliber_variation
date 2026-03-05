#!/usr/bin/env python3
"""
Analyze spatial correlation of axon radius deviations across multiple samples.

Investigates whether radius changes are compensated between neighboring axons:
when one axon gets thicker, do nearby axons get thinner?

This script:
1. Processes multiple .mat files matching a glob pattern
2. Uses dominant axis from populations JSON for slicing direction
3. Pools pairwise correlations across all samples
4. Produces a single combined figure

Uses a two-pass approach per sample:
1. Pass 1: Compute per-axon mean radius across all slices
2. Pass 2: Compute normalized deviations and pairwise correlations by distance
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob as glob_module

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from skimage.measure import regionprops
from tqdm import tqdm

# Import from axonometry library
# Find repo root (contains pyproject.toml)
_root = Path(__file__).resolve().parent
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from axonometry import get_plot_settings
from axonometry.io import load_volume_with_metadata, resample_to_isotropic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load plot settings
settings = get_plot_settings()

# Global variables for worker processes
_volume_data = None
_slice_axis = None
_voxel_size_um = None
_axon_mean_radii = None
_axon_std_radii = None
_metric = None

# Metric options
METRICS = ['relative', 'zscore', 'raw', 'log_ratio']


def find_populations_json(mat_file: Path) -> Path:
    """Find the populations JSON file for a given .mat file."""
    # Extract sample name from mat file
    # e.g., LM_25_ipsi_myelinated_axons.mat -> sham_25_ipsi or tbi_25_ipsi
    stem = mat_file.stem.replace('_myelinated_axons', '')

    # Try to find in processed/LM directory
    # The JSON uses lowercase condition prefix
    parts = stem.split('_')
    if len(parts) >= 3:
        # Format: LM_25_ipsi -> need to find sham_25_ipsi or tbi_25_ipsi
        rat_id = parts[1]
        hemisphere = parts[2]

        # Check parent directory for condition
        parent_name = mat_file.parent.name.lower()
        if 'sham' in parent_name:
            condition = 'sham'
        elif 'tbi' in parent_name:
            condition = 'tbi'
        else:
            condition = 'sham'  # Default

        json_name = f"{condition}_{rat_id}_{hemisphere}_populations.json"
        json_path = Path("data/processed/LM") / json_name

        if json_path.exists():
            return json_path

    raise FileNotFoundError(f"Could not find populations JSON for {mat_file}")


def get_dominant_axis_from_json(json_file: Path) -> int:
    """Get the most common dominant axis from populations JSON."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Get dominant axis from populations (use the one with most axons)
    populations = data.get('populations', [])
    if not populations:
        return 0  # Default to Z axis

    # Find population with most axons
    best_pop = max(populations, key=lambda p: p.get('n_axons', 0))
    return best_pop.get('dominant_axis', 0)


def _init_worker_pass1(volume_data: np.ndarray, slice_axis: int, voxel_size_um: float):
    """Initialize worker for Pass 1."""
    global _volume_data, _slice_axis, _voxel_size_um
    _volume_data = volume_data
    _slice_axis = slice_axis
    _voxel_size_um = voxel_size_um


def _process_slice_pass1(z: int) -> Dict[int, float]:
    """Pass 1: Extract radius for each axon in a slice."""
    pixel_area_um2 = _voxel_size_um * _voxel_size_um

    if _slice_axis == 0:
        slice_2d = _volume_data[z, :, :]
    elif _slice_axis == 1:
        slice_2d = _volume_data[:, z, :]
    else:
        slice_2d = _volume_data[:, :, z]

    regions = regionprops(slice_2d.astype(np.int32))

    result = {}
    for region in regions:
        area_um2 = region.area * pixel_area_um2
        radius_um = np.sqrt(area_um2 / np.pi)
        result[region.label] = radius_um

    return result


def compute_per_axon_stats(volume: np.ndarray, slice_axis: int,
                           voxel_size_um: float, n_jobs: int = -1) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Pass 1: Compute mean and std radius for each axon across all slices."""
    n_slices = volume.shape[slice_axis]

    if n_jobs == -1:
        n_workers = cpu_count()
    elif n_jobs == 1:
        n_workers = 1
    else:
        n_workers = min(n_jobs, cpu_count())

    axon_radii = defaultdict(list)

    if n_workers > 1:
        with Pool(processes=n_workers,
                  initializer=_init_worker_pass1,
                  initargs=(volume, slice_axis, voxel_size_um)) as pool:
            results = list(tqdm(
                pool.imap(_process_slice_pass1, range(n_slices)),
                total=n_slices,
                desc="  Pass 1",
                leave=False
            ))
    else:
        _init_worker_pass1(volume, slice_axis, voxel_size_um)
        results = [_process_slice_pass1(z) for z in tqdm(range(n_slices), desc="  Pass 1", leave=False)]

    for slice_result in results:
        for label, radius in slice_result.items():
            axon_radii[label].append(radius)

    mean_radii = {label: np.mean(radii) for label, radii in axon_radii.items()}
    std_radii = {label: np.std(radii) if len(radii) > 1 else 0.0 for label, radii in axon_radii.items()}

    return mean_radii, std_radii


def _init_worker_pass2(volume_data: np.ndarray, slice_axis: int,
                       voxel_size_um: float, axon_mean_radii: Dict[int, float],
                       axon_std_radii: Dict[int, float], metric: str):
    """Initialize worker for Pass 2."""
    global _volume_data, _slice_axis, _voxel_size_um, _axon_mean_radii, _axon_std_radii, _metric
    _volume_data = volume_data
    _slice_axis = slice_axis
    _voxel_size_um = voxel_size_um
    _axon_mean_radii = axon_mean_radii
    _axon_std_radii = axon_std_radii
    _metric = metric


def _process_slice_pass2(z: int) -> List[Tuple[float, float, float, int]]:
    """Pass 2: Extract centroid, radius, and deviation for each axon."""
    pixel_area_um2 = _voxel_size_um * _voxel_size_um

    if _slice_axis == 0:
        slice_2d = _volume_data[z, :, :]
    elif _slice_axis == 1:
        slice_2d = _volume_data[:, z, :]
    else:
        slice_2d = _volume_data[:, :, z]

    regions = regionprops(slice_2d.astype(np.int32))

    result = []
    for region in regions:
        label = region.label
        if label not in _axon_mean_radii:
            continue

        mean_r = _axon_mean_radii[label]
        if mean_r <= 0:
            continue

        area_um2 = region.area * pixel_area_um2
        radius_um = np.sqrt(area_um2 / np.pi)

        if _metric == 'relative':
            delta_r = (radius_um - mean_r) / mean_r
        elif _metric == 'zscore':
            std_r = _axon_std_radii.get(label, 0.0)
            if std_r > 1e-9:
                delta_r = (radius_um - mean_r) / std_r
            else:
                continue
        elif _metric == 'raw':
            delta_r = radius_um - mean_r
        elif _metric == 'log_ratio':
            if radius_um > 0:
                delta_r = np.log(radius_um / mean_r)
            else:
                continue
        else:
            delta_r = (radius_um - mean_r) / mean_r

        cy, cx = region.centroid
        cy_um = cy * _voxel_size_um
        cx_um = cx * _voxel_size_um

        result.append((cy_um, cx_um, delta_r, label))

    return result


def extract_pairwise_data(slice_data_list: List[List[Tuple[float, float, float, int]]],
                          distance_bins: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract pairwise (delta_r_i, delta_r_j) data binned by distance.

    Returns list of (delta_i_array, delta_j_array) tuples, one per distance bin.
    Using numpy arrays instead of lists of tuples for memory efficiency.
    """
    n_bins = len(distance_bins) - 1

    # Pre-allocate arrays to collect all pairs across slices
    # Estimate capacity: average ~200 axons/slice, ~20k pairs/slice
    all_delta_i = []
    all_delta_j = []
    all_bin_indices = []

    for slice_data in slice_data_list:
        n = len(slice_data)
        if n < 2:
            continue

        positions = np.array([(d[0], d[1]) for d in slice_data])
        delta_rs = np.array([d[2] for d in slice_data])

        # Compute pairwise distances (condensed form)
        distances_condensed = pdist(positions)

        # Vectorized bin assignment for all pairs
        bin_indices = np.searchsorted(distance_bins, distances_condensed, side='right') - 1

        # Filter to valid bins
        valid_mask = (bin_indices >= 0) & (bin_indices < n_bins)
        if not np.any(valid_mask):
            continue

        valid_bin_indices = bin_indices[valid_mask]

        # Build upper triangular indices for valid pairs
        # condensed index k corresponds to (i, j) where k = n*i - i*(i+1)/2 + j - i - 1
        n_pairs = len(distances_condensed)
        pair_indices = np.arange(n_pairs)[valid_mask]

        # Convert condensed indices to (i, j) pairs using vectorized formula
        # For condensed index k: i = n - 2 - floor(sqrt(-8*k + 4*n*(n-1) - 7) / 2 - 0.5)
        # j = k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2
        # Simplified: use cumulative sum approach
        row_end = np.cumsum(np.arange(n - 1, 0, -1))  # [n-1, 2n-3, 3n-6, ...]
        i_indices = np.searchsorted(row_end, pair_indices, side='right')
        # j = pair_index - row_start + i + 1, where row_start = row_end[i-1] for i>0, else 0
        row_starts = np.concatenate([[0], row_end[:-1]])
        j_indices = pair_indices - row_starts[i_indices] + i_indices + 1

        # Gather delta_r values
        all_delta_i.append(delta_rs[i_indices])
        all_delta_j.append(delta_rs[j_indices])
        all_bin_indices.append(valid_bin_indices)

    # Concatenate all slices
    if not all_delta_i:
        return [(np.array([]), np.array([])) for _ in range(n_bins)]

    all_delta_i = np.concatenate(all_delta_i)
    all_delta_j = np.concatenate(all_delta_j)
    all_bin_indices = np.concatenate(all_bin_indices)

    # Group by bin using argsort (much faster than per-element append)
    sort_order = np.argsort(all_bin_indices)
    sorted_bins = all_bin_indices[sort_order]
    sorted_delta_i = all_delta_i[sort_order]
    sorted_delta_j = all_delta_j[sort_order]

    # Find bin boundaries
    bin_counts = np.bincount(sorted_bins, minlength=n_bins)
    bin_ends = np.cumsum(bin_counts)
    bin_starts = np.concatenate([[0], bin_ends[:-1]])

    # Build result as list of numpy array pairs (much more memory efficient)
    bin_pairs = []
    for b in range(n_bins):
        start, end = bin_starts[b], bin_ends[b]
        if end > start:
            bin_pairs.append((sorted_delta_i[start:end].copy(), sorted_delta_j[start:end].copy()))
        else:
            bin_pairs.append((np.array([]), np.array([])))

    return bin_pairs


def process_single_sample(mat_file: Path, voxel_size_um: Optional[float],
                          distance_bins: np.ndarray, metric: str,
                          n_jobs: int) -> Tuple[str, List[Tuple[np.ndarray, np.ndarray]], int]:
    """
    Process a single sample and return binned pairwise data.

    Args:
        mat_file: Path to .mat file
        voxel_size_um: Voxel size override (if None, loads from companion JSON metadata)
        distance_bins: Array of distance bin edges
        metric: Deviation metric ('relative', 'zscore', 'raw', 'log_ratio')
        n_jobs: Number of parallel jobs

    Returns:
        Tuple of (sample_name, bin_pairs, n_axons)
    """
    sample_name = mat_file.stem.replace('_myelinated_axons', '')
    logger.info(f"Processing {sample_name}...")

    # Load volume with metadata (voxel size from JSON if not overridden)
    volume, voxel_tuple, metadata = load_volume_with_metadata(mat_file, voxel_size_um)
    logger.info(f"  Volume shape: {volume.shape}")

    # Get dominant axis from populations JSON or metadata
    slice_axis = 0  # Default
    axis_names = {0: 'Z', 1: 'Y', 2: 'X'}

    # Check metadata first
    if metadata is not None and 'dominant_axis' in metadata:
        slice_axis = metadata['dominant_axis']
        logger.info(f"  Using dominant axis from metadata: {slice_axis} ({axis_names[slice_axis]})")
    else:
        # Try populations JSON as fallback
        try:
            json_file = find_populations_json(mat_file)
            slice_axis = get_dominant_axis_from_json(json_file)
            logger.info(f"  Using dominant axis from populations JSON: {slice_axis} ({axis_names[slice_axis]})")
        except FileNotFoundError:
            logger.warning(f"  No dominant axis found, defaulting to axis 0 (Z)")

    # Resample to isotropic (downsamples to coarsest dimension)
    volume, iso_voxel_size = resample_to_isotropic(volume, voxel_tuple)

    n_slices = volume.shape[slice_axis]
    logger.info(f"  {n_slices} slices along axis {slice_axis}")

    # Pass 1: Per-axon statistics
    axon_mean_radii, axon_std_radii = compute_per_axon_stats(volume, slice_axis, iso_voxel_size, n_jobs)
    n_axons = len(axon_mean_radii)
    logger.info(f"  Found {n_axons} axons")

    # Determine workers for Pass 2
    if n_jobs == -1:
        n_workers = cpu_count()
    elif n_jobs == 1:
        n_workers = 1
    else:
        n_workers = min(n_jobs, cpu_count())

    # Pass 2: Extract centroids and deviations
    if n_workers > 1:
        with Pool(processes=n_workers,
                  initializer=_init_worker_pass2,
                  initargs=(volume, slice_axis, iso_voxel_size, axon_mean_radii, axon_std_radii, metric)) as pool:
            slice_data_list = list(tqdm(
                pool.imap(_process_slice_pass2, range(n_slices)),
                total=n_slices,
                desc="  Pass 2",
                leave=False
            ))
    else:
        _init_worker_pass2(volume, slice_axis, iso_voxel_size, axon_mean_radii, axon_std_radii, metric)
        slice_data_list = [_process_slice_pass2(z) for z in tqdm(range(n_slices), desc="  Pass 2", leave=False)]

    # Extract pairwise data
    bin_pairs = extract_pairwise_data(slice_data_list, distance_bins)

    return sample_name, bin_pairs, n_axons


def _bootstrap_batch_worker(args: Tuple[np.ndarray, np.ndarray, int, int]) -> np.ndarray:
    """
    Worker function for parallel bootstrap computation.

    Args:
        args: Tuple of (x, y, n_samples, seed)

    Returns:
        Array of bootstrap correlation values
    """
    x, y, n_samples, seed = args
    n = len(x)
    rng = np.random.default_rng(seed)

    # Generate bootstrap indices
    indices = rng.integers(0, n, size=(n_samples, n))

    # Gather bootstrap samples
    x_boot = x[indices]
    y_boot = y[indices]

    # Compute means
    x_mean = x_boot.mean(axis=1, keepdims=True)
    y_mean = y_boot.mean(axis=1, keepdims=True)

    # Center the data
    x_centered = x_boot - x_mean
    y_centered = y_boot - y_mean

    # Compute correlation components
    numerator = (x_centered * y_centered).sum(axis=1)
    x_std = np.sqrt((x_centered ** 2).sum(axis=1))
    y_std = np.sqrt((y_centered ** 2).sum(axis=1))

    # Avoid division by zero
    denom = x_std * y_std
    valid = denom > 1e-10
    r_batch = np.full(n_samples, np.nan)
    r_batch[valid] = numerator[valid] / denom[valid]

    return r_batch[~np.isnan(r_batch)]


def _parallel_pearson_bootstrap(x: np.ndarray, y: np.ndarray,
                                 n_bootstrap: int, rng: np.random.Generator,
                                 n_workers: int,
                                 max_batch_bytes: int = 500_000_000) -> np.ndarray:
    """
    Parallel bootstrap Pearson correlation with memory-efficient batching.

    Distributes bootstrap samples across workers, each processing batches.

    Args:
        x, y: Arrays of paired values
        n_bootstrap: Number of bootstrap samples
        rng: Random number generator (used for seed generation)
        n_workers: Number of parallel workers
        max_batch_bytes: Maximum memory per batch per worker (~500 MB default)
    """
    n = len(x)

    # Estimate memory per bootstrap sample: 2 arrays of n float64 values
    bytes_per_sample = 2 * n * 8  # 8 bytes per float64

    # Calculate batch size to stay under memory limit per worker
    batch_size = max(1, max_batch_bytes // bytes_per_sample)
    batch_size = min(batch_size, n_bootstrap)

    # Split bootstrap samples into batches
    n_batches = (n_bootstrap + batch_size - 1) // batch_size
    samples_per_batch = [(n_bootstrap // n_batches) for _ in range(n_batches)]
    # Distribute remainder
    for i in range(n_bootstrap % n_batches):
        samples_per_batch[i] += 1

    # Generate seeds for each batch (deterministic from parent rng)
    seeds = rng.integers(0, 2**31, size=n_batches)

    # Prepare worker arguments
    worker_args = [(x, y, samples_per_batch[i], seeds[i]) for i in range(n_batches)]

    # Run in parallel
    if n_workers > 1 and n_batches > 1:
        with Pool(processes=min(n_workers, n_batches)) as pool:
            results = pool.map(_bootstrap_batch_worker, worker_args)
    else:
        results = [_bootstrap_batch_worker(args) for args in worker_args]

    # Concatenate results
    all_rs = [r for r in results if len(r) > 0]
    return np.concatenate(all_rs) if all_rs else np.array([])


def compute_pooled_correlations(all_bin_pairs: List[Tuple[np.ndarray, np.ndarray]],
                                 distance_bins: np.ndarray,
                                 n_jobs: int = -1) -> Dict:
    """
    Compute correlations from pooled pairwise data.

    Args:
        all_bin_pairs: List of (x_array, y_array) tuples, one per bin
        distance_bins: Array of distance bin edges
        n_jobs: Number of parallel workers for bootstrap (-1 = all CPUs)
    """
    n_bins = len(distance_bins) - 1
    bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2

    # Determine number of workers
    if n_jobs == -1:
        n_workers = cpu_count()
    elif n_jobs == 1:
        n_workers = 1
    else:
        n_workers = min(n_jobs, cpu_count())

    correlations = []
    n_pairs_list = []
    p_values = []
    ci_low = []
    ci_high = []

    rng = np.random.default_rng(42)
    n_bootstrap = 1000
    min_pairs_for_stats = 30

    for bin_idx in tqdm(range(n_bins), desc="Computing correlations", leave=False):
        x, y = all_bin_pairs[bin_idx]
        n_pairs = len(x)
        n_pairs_list.append(n_pairs)

        if n_pairs < min_pairs_for_stats:
            correlations.append(np.nan)
            p_values.append(np.nan)
            ci_low.append(np.nan)
            ci_high.append(np.nan)
            continue

        r, p = pearsonr(x, y)
        correlations.append(r)
        p_values.append(p)

        # Parallel bootstrap 95% CI with memory-efficient batching
        bootstrap_rs = _parallel_pearson_bootstrap(x, y, n_bootstrap, rng, n_workers)

        if len(bootstrap_rs) > 0:
            ci_low.append(np.percentile(bootstrap_rs, 2.5))
            ci_high.append(np.percentile(bootstrap_rs, 97.5))
        else:
            ci_low.append(np.nan)
            ci_high.append(np.nan)

    return {
        'bin_centers': bin_centers.tolist(),
        'bin_edges': distance_bins.tolist(),
        'correlations': correlations,
        'n_pairs': n_pairs_list,
        'p_values': p_values,
        'ci_low': ci_low,
        'ci_high': ci_high
    }


def plot_pooled_correlation(results: Dict, output_file: Path,
                            n_samples: int, total_axons: int, metric: str):
    """Plot pooled correlation vs distance - single combined figure."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08}
    )

    font_settings = settings.fonts

    bin_centers = np.array(results['bin_centers'])
    bin_edges = np.array(results['bin_edges'])
    correlations = np.array(results['correlations'])
    ci_low = np.array(results['ci_low'])
    ci_high = np.array(results['ci_high'])
    n_pairs = np.array(results['n_pairs'])
    p_values = np.array(results['p_values'])
    bin_width = bin_edges[1] - bin_edges[0]

    valid = ~np.isnan(correlations)

    # Colors
    main_color = '#2E86AB'
    sig_color = '#E94F37'
    nonsig_color = '#A1A1A1'

    # ===== Top plot: Correlation vs distance =====
    ax1.axhline(0, color='#666666', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.axhspan(-0.05, 0.05, color='#E8E8E8', alpha=0.5, zorder=0)

    if np.any(valid):
        sig_mask = valid & (np.array(p_values) < 0.05)
        nonsig_mask = valid & (np.array(p_values) >= 0.05)

        if np.any(nonsig_mask):
            yerr_low_ns = correlations[nonsig_mask] - ci_low[nonsig_mask]
            yerr_high_ns = ci_high[nonsig_mask] - correlations[nonsig_mask]
            ax1.errorbar(bin_centers[nonsig_mask], correlations[nonsig_mask],
                        yerr=[yerr_low_ns, yerr_high_ns],
                        fmt='o', color=nonsig_color, capsize=3, capthick=1,
                        markersize=8, markerfacecolor='white', markeredgewidth=1.5,
                        linewidth=0, elinewidth=1.2, label='p ≥ 0.05')

        if np.any(sig_mask):
            yerr_low_s = correlations[sig_mask] - ci_low[sig_mask]
            yerr_high_s = ci_high[sig_mask] - correlations[sig_mask]
            ax1.errorbar(bin_centers[sig_mask], correlations[sig_mask],
                        yerr=[yerr_low_s, yerr_high_s],
                        fmt='o', color=sig_color, capsize=3, capthick=1,
                        markersize=8, markerfacecolor=sig_color, markeredgewidth=1.5,
                        linewidth=0, elinewidth=1.2, label='p < 0.05')

        ax1.plot(bin_centers[valid], correlations[valid],
                '-', color=main_color, alpha=0.4, linewidth=1, zorder=1)

    # Metric labels
    metric_labels = {
        'relative': r'$\Delta r = (r - \bar{r})/\bar{r}$',
        'zscore': r'$\Delta r = (r - \bar{r})/\sigma$',
        'raw': r'$\Delta r = r - \bar{r}$ (μm)',
        'log_ratio': r'$\Delta r = \log(r/\bar{r})$'
    }
    metric_label = metric_labels.get(metric, metric_labels['relative'])

    ax1.set_ylabel('Pearson Correlation of $\\Delta r$',
                   fontsize=font_settings['label_size'], fontweight=font_settings['weight'])
    ax1.set_title(f'Spatial Correlation of Radius Deviations (Pooled)\n{metric_label}',
                  fontsize=font_settings['title_size'], fontweight=font_settings['weight'])
    ax1.legend(loc='upper right', fontsize=font_settings['legend_size'], framealpha=0.9)

    if np.any(valid):
        max_abs = max(abs(np.nanmin(correlations)), abs(np.nanmax(correlations)))
        y_limit = max(0.15, min(0.4, max_abs * 1.5))
    else:
        y_limit = 0.3
    ax1.set_ylim(-y_limit, y_limit)
    ax1.grid(True, alpha=0.3)

    # Summary stats
    nn_idx = next((i for i, c in enumerate(correlations) if not np.isnan(c)), None)
    nn_corr = correlations[nn_idx] if nn_idx is not None else np.nan

    short_range_corrs = [c for c, d in zip(correlations, bin_centers) if d < 10 and not np.isnan(c)]
    sr_mean = np.mean(short_range_corrs) if short_range_corrs else np.nan

    summary_text = f'n = {n_samples} samples\n{total_axons:,} axons total'
    if not np.isnan(nn_corr):
        summary_text += f'\nNN corr: {nn_corr:.3f}'
    if not np.isnan(sr_mean):
        summary_text += f'\n<10μm mean: {sr_mean:.3f}'

    ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.9))

    # ===== Bottom plot: Number of pairs =====
    colors = [sig_color if p < 0.05 and not np.isnan(p) else nonsig_color for p in p_values]
    ax2.bar(bin_centers, n_pairs, width=bin_width * 0.85,
            color=colors, edgecolor='white', linewidth=0.5, alpha=0.8)

    ax2.set_xlabel('Centroid Distance (μm)',
                   fontsize=font_settings['label_size'], fontweight=font_settings['weight'])
    ax2.set_ylabel('N Pairs', fontsize=font_settings['label_size'])
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze spatial correlation of axon radius deviations across multiple samples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all LM samples
  python analyze_spatial_correlation_ensemble.py \\
      "data/raw/**/LM*_myelinated_axons.mat" \\
      fig/spatial_correlation_pooled.png

  # With custom parameters
  python analyze_spatial_correlation_ensemble.py \\
      "data/raw/**/LM*_myelinated_axons.mat" \\
      fig/spatial_correlation_pooled.png \\
      --max-distance 40 --bin-width 2 --metric relative
        """
    )

    parser.add_argument('input_pattern', type=str,
                        help='Glob pattern for .mat files')
    parser.add_argument('output', type=Path,
                        help='Output PNG file path')
    parser.add_argument('--voxel-size', type=float, default=None,
                        help='Voxel size override in um. If not specified, loads from '
                             'companion JSON metadata file for each .mat file')
    parser.add_argument('--max-distance', type=float, default=50.0,
                        help='Maximum distance for correlation analysis in um (default: 50)')
    parser.add_argument('--bin-width', type=float, default=2.0,
                        help='Width of distance bins in um (default: 2)')
    parser.add_argument('--metric', type=str, default='relative', choices=METRICS,
                        help='Deviation metric (default: relative)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (default: -1 = all CPUs)')

    args = parser.parse_args()

    # Find all matching files
    mat_files = sorted([Path(f) for f in glob_module.glob(args.input_pattern, recursive=True)])

    if not mat_files:
        logger.error(f"No files found matching pattern: {args.input_pattern}")
        return

    logger.info(f"Found {len(mat_files)} .mat files")

    # Setup distance bins
    distance_bins = np.arange(0, args.max_distance + args.bin_width, args.bin_width)
    n_bins = len(distance_bins) - 1
    logger.info(f"Distance bins: {n_bins} bins from 0 to {args.max_distance} um")

    # Process each sample and accumulate pairwise data
    # Store lists of arrays per bin, concatenate at the end
    bin_x_arrays = [[] for _ in range(n_bins)]
    bin_y_arrays = [[] for _ in range(n_bins)]
    total_axons = 0
    sample_names = []

    for mat_file in mat_files:
        try:
            sample_name, bin_pairs, n_axons = process_single_sample(
                mat_file, args.voxel_size, distance_bins, args.metric, args.n_jobs
            )
            sample_names.append(sample_name)
            total_axons += n_axons

            # Pool the pairwise data (append arrays, don't concatenate yet)
            sample_pairs = 0
            for bin_idx in range(n_bins):
                x_arr, y_arr = bin_pairs[bin_idx]
                if len(x_arr) > 0:
                    bin_x_arrays[bin_idx].append(x_arr)
                    bin_y_arrays[bin_idx].append(y_arr)
                    sample_pairs += len(x_arr)

            logger.info(f"  Accumulated {sample_pairs:,} pairs")

        except Exception as e:
            logger.error(f"Failed to process {mat_file.name}: {e}")
            import traceback
            traceback.print_exc()

    if not sample_names:
        logger.error("No samples processed successfully")
        return

    # Concatenate all arrays per bin (single memory allocation per bin)
    logger.info("\nConcatenating pooled data...")
    all_bin_pairs = []
    total_pairs = 0
    for bin_idx in range(n_bins):
        if bin_x_arrays[bin_idx]:
            x_pooled = np.concatenate(bin_x_arrays[bin_idx])
            y_pooled = np.concatenate(bin_y_arrays[bin_idx])
            total_pairs += len(x_pooled)
        else:
            x_pooled = np.array([])
            y_pooled = np.array([])
        all_bin_pairs.append((x_pooled, y_pooled))

    # Free the intermediate lists
    del bin_x_arrays, bin_y_arrays

    # Compute pooled correlations
    logger.info(f"Computing pooled correlations from {len(sample_names)} samples...")
    logger.info(f"Total pairs: {total_pairs:,}")

    results = compute_pooled_correlations(all_bin_pairs, distance_bins, args.n_jobs)
    results['n_samples'] = len(sample_names)
    results['total_axons'] = total_axons
    results['sample_names'] = sample_names
    results['metric'] = args.metric

    # Plot
    plot_pooled_correlation(results, args.output, len(sample_names), total_axons, args.metric)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Samples processed: {len(sample_names)}")
    logger.info(f"Total axons: {total_axons:,}")
    logger.info(f"Total pairs: {total_pairs:,}")

    valid_corrs = [c for c in results['correlations'] if not np.isnan(c)]
    if valid_corrs:
        nn_corr = valid_corrs[0]
        short_range = [c for c, d in zip(results['correlations'], results['bin_centers'])
                       if d < 10 and not np.isnan(c)]
        sr_mean = np.mean(short_range) if short_range else np.nan

        logger.info(f"Nearest-neighbor correlation: {nn_corr:.4f}")
        if not np.isnan(sr_mean):
            logger.info(f"Short-range (<10um) mean: {sr_mean:.4f}")

    # Save JSON results
    json_file = args.output.with_suffix('.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {json_file}")


if __name__ == '__main__':
    main()
