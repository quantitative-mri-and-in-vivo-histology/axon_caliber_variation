#!/usr/bin/env python3
"""
Analyze spatial correlation of axon radius deviations across a bundle.

Investigates whether radius changes are compensated between neighboring axons:
when one axon gets thicker, do nearby axons get thinner?

Uses a two-pass approach:
1. Pass 1: Compute per-axon mean radius across all slices
2. Pass 2: Compute normalized deviations and pairwise correlations by distance

Handles anisotropic voxels by downsampling to isotropic resolution first.
"""

import argparse
import json
import logging
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from skimage.measure import regionprops
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for worker processes
_volume_data = None
_slice_axis = None
_voxel_size_um = None
_axon_mean_radii = None
_axon_std_radii = None
_metric = None

# Metric options
METRICS = ['relative', 'zscore', 'raw', 'log_ratio']


def parse_voxel_size(voxel_size_um: Union[float, Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """Parse voxel size to a (vz, vy, vx) tuple matching array axes (Z, Y, X)."""
    if isinstance(voxel_size_um, (tuple, list)):
        if len(voxel_size_um) == 3:
            return tuple(float(v) for v in voxel_size_um)
        elif len(voxel_size_um) == 1:
            v = float(voxel_size_um[0])
            return (v, v, v)
        else:
            raise ValueError(f"Expected 1 or 3 voxel size values, got {len(voxel_size_um)}")
    v = float(voxel_size_um)
    return (v, v, v)


def parse_voxel_size_arg(value: str) -> Union[float, Tuple[float, float, float]]:
    """Parse voxel size CLI argument: single float or comma-separated triple."""
    if ',' in value:
        parts = value.split(',')
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(f"Expected 1 or 3 values, got {len(parts)}: '{value}'")
        return tuple(float(p.strip()) for p in parts)
    return float(value)


def resample_to_isotropic(volume: np.ndarray,
                          voxel_size: Tuple[float, float, float]) -> Tuple[np.ndarray, float]:
    """
    Resample anisotropic volume to isotropic voxels.
    Downsamples to the coarsest voxel dimension using nearest-neighbor interpolation.
    """
    vz, vy, vx = voxel_size

    if np.allclose([vz, vy, vx], vz):
        logger.info("Volume is already isotropic, no resampling needed")
        return volume, vz

    target_size = max(vz, vy, vx)
    zoom_factors = (vz / target_size, vy / target_size, vx / target_size)

    logger.info(f"Resampling anisotropic volume to isotropic:")
    logger.info(f"  Original voxel size (Z, Y, X): vz={vz:.4f}, vy={vy:.4f}, vx={vx:.4f} um")
    logger.info(f"  Target voxel size: {target_size:.4f} um (isotropic)")
    logger.info(f"  Original shape: {volume.shape}")

    resampled = zoom(volume, zoom_factors, order=0, mode='nearest')
    logger.info(f"  Resampled shape: {resampled.shape}")

    return resampled, target_size


def load_metadata(mat_file: Path) -> dict:
    """Load metadata JSON file associated with a .mat file."""
    metadata_file = mat_file.with_suffix('.json')
    if metadata_file.exists():
        logger.info(f"Found metadata file: {metadata_file}")
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return {}


def load_mat_volume(mat_file: Path) -> np.ndarray:
    """Load labeled volume from .mat file (v5.0 or v7.3/HDF5)."""
    logger.info(f"Loading {mat_file}")

    try:
        with h5py.File(str(mat_file), 'r') as f:
            volume_key = None
            for key in f.keys():
                if not key.startswith('#') and not key.startswith('_'):
                    volume_key = key
                    break
            if volume_key is None:
                raise ValueError(f"No data found in {mat_file}")
            volume = f[volume_key][:]
            logger.info(f"Loaded HDF5 format, volume shape: {volume.shape}, dtype: {volume.dtype}")
            return volume

    except OSError:
        logger.info("HDF5 failed, trying scipy.io for older MATLAB format...")
        mat_data = sio.loadmat(str(mat_file))
        priority_keys = ['myelinated_axons', 'volume', 'labels', 'final_lbl']
        volume_key = None
        for pkey in priority_keys:
            if pkey in mat_data:
                volume_key = pkey
                break
        if volume_key is None:
            for key in mat_data.keys():
                if not key.startswith('__'):
                    volume_key = key
                    break
        if volume_key is None:
            raise ValueError(f"No data found in {mat_file}")
        volume = mat_data[volume_key]
        logger.info(f"Loaded scipy.io format (key: {volume_key}), shape: {volume.shape}")
        return volume


def _init_worker_pass1(volume_data: np.ndarray, slice_axis: int, voxel_size_um: float):
    """Initialize worker for Pass 1."""
    global _volume_data, _slice_axis, _voxel_size_um
    _volume_data = volume_data
    _slice_axis = slice_axis
    _voxel_size_um = voxel_size_um


def _process_slice_pass1(z: int) -> Dict[int, float]:
    """
    Pass 1: Extract radius for each axon in a slice.
    Returns dict: {label: radius_um}
    """
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
    """
    Pass 1: Compute mean and std radius for each axon across all slices.

    Returns:
        Tuple of (mean_radii, std_radii) dicts mapping axon label to values in micrometers
    """
    n_slices = volume.shape[slice_axis]

    if n_jobs == -1:
        n_workers = cpu_count()
    elif n_jobs == 1:
        n_workers = 1
    else:
        n_workers = min(n_jobs, cpu_count())

    logger.info(f"Pass 1: Computing per-axon statistics ({n_slices} slices, {n_workers} workers)")

    # Accumulate radii per axon
    axon_radii = defaultdict(list)

    if n_workers > 1:
        with Pool(processes=n_workers,
                  initializer=_init_worker_pass1,
                  initargs=(volume, slice_axis, voxel_size_um)) as pool:
            results = list(tqdm(
                pool.imap(_process_slice_pass1, range(n_slices)),
                total=n_slices,
                desc="Pass 1: Statistics"
            ))
    else:
        _init_worker_pass1(volume, slice_axis, voxel_size_um)
        results = [_process_slice_pass1(z) for z in tqdm(range(n_slices), desc="Pass 1: Statistics")]

    # Aggregate
    for slice_result in results:
        for label, radius in slice_result.items():
            axon_radii[label].append(radius)

    # Compute means and stds
    mean_radii = {label: np.mean(radii) for label, radii in axon_radii.items()}
    std_radii = {label: np.std(radii) if len(radii) > 1 else 0.0 for label, radii in axon_radii.items()}

    logger.info(f"  Found {len(mean_radii)} unique axons")
    logger.info(f"  Mean radius across axons: {np.mean(list(mean_radii.values())):.3f} um")
    logger.info(f"  Mean std within axons: {np.mean(list(std_radii.values())):.3f} um")

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
    """
    Pass 2: Extract centroid, radius, and deviation for each axon.
    Returns list of (centroid_y_um, centroid_x_um, delta_r, label) tuples.
    """
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

        # Compute deviation based on selected metric
        if _metric == 'relative':
            # (r - mean) / mean
            delta_r = (radius_um - mean_r) / mean_r
        elif _metric == 'zscore':
            # (r - mean) / std
            std_r = _axon_std_radii.get(label, 0.0)
            if std_r > 1e-9:
                delta_r = (radius_um - mean_r) / std_r
            else:
                continue  # Skip axons with no variability
        elif _metric == 'raw':
            # r - mean (in um)
            delta_r = radius_um - mean_r
        elif _metric == 'log_ratio':
            # log(r / mean)
            if radius_um > 0:
                delta_r = np.log(radius_um / mean_r)
            else:
                continue
        else:
            delta_r = (radius_um - mean_r) / mean_r  # Default to relative

        # Centroid in physical units
        cy, cx = region.centroid
        cy_um = cy * _voxel_size_um
        cx_um = cx * _voxel_size_um

        result.append((cy_um, cx_um, delta_r, label))

    return result


def compute_pairwise_correlations(slice_data_list: List[List[Tuple[float, float, float, int]]],
                                   distance_bins: np.ndarray) -> Dict:
    """
    Compute pairwise correlations of radius deviations binned by distance.

    Args:
        slice_data_list: List of per-slice data, each containing
                        (cy_um, cx_um, delta_r, label) tuples
        distance_bins: Bin edges for distances

    Returns:
        Dict with correlation results per bin
    """
    n_bins = len(distance_bins) - 1

    # Accumulate (delta_r_i, delta_r_j) pairs for each bin
    bin_pairs = [[] for _ in range(n_bins)]

    for slice_data in tqdm(slice_data_list, desc="Computing pairwise correlations"):
        if len(slice_data) < 2:
            continue

        # Extract arrays
        positions = np.array([(d[0], d[1]) for d in slice_data])
        delta_rs = np.array([d[2] for d in slice_data])

        # Compute pairwise distances using condensed form (memory efficient)
        n = len(slice_data)
        if n > 1:
            distances_condensed = pdist(positions)

            # Iterate over condensed form (upper triangle, row-major order)
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    d = distances_condensed[idx]
                    idx += 1

                    # Find bin
                    bin_idx = np.searchsorted(distance_bins, d, side='right') - 1
                    if 0 <= bin_idx < n_bins:
                        bin_pairs[bin_idx].append((delta_rs[i], delta_rs[j]))

    # Compute correlation per bin
    bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
    correlations = []
    n_pairs_list = []
    p_values = []
    ci_low = []
    ci_high = []

    # Initialize RNG once outside the loop for proper bootstrap statistics
    rng = np.random.default_rng(42)
    n_bootstrap = 1000
    min_pairs_for_stats = 30  # Minimum pairs for reliable bootstrap CIs

    for bin_idx in range(n_bins):
        pairs = bin_pairs[bin_idx]
        n_pairs = len(pairs)
        n_pairs_list.append(n_pairs)

        if n_pairs < min_pairs_for_stats:
            correlations.append(np.nan)
            p_values.append(np.nan)
            ci_low.append(np.nan)
            ci_high.append(np.nan)
            continue

        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])

        r, p = pearsonr(x, y)
        correlations.append(r)
        p_values.append(p)

        # Bootstrap 95% CI
        bootstrap_rs = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n_pairs, size=n_pairs, replace=True)
            x_boot = x[idx]
            y_boot = y[idx]
            if np.std(x_boot) > 0 and np.std(y_boot) > 0:
                r_boot, _ = pearsonr(x_boot, y_boot)
                bootstrap_rs.append(r_boot)

        if bootstrap_rs:
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


def plot_correlation_function(results: Dict, output_file: Path, sample_name: str):
    """Plot correlation vs distance with error bars - publication-ready version."""
    # Set up figure with custom styling
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.12}
    )

    bin_centers = np.array(results['bin_centers'])
    bin_edges = np.array(results['bin_edges'])
    correlations = np.array(results['correlations'])
    ci_low = np.array(results['ci_low'])
    ci_high = np.array(results['ci_high'])
    n_pairs = np.array(results['n_pairs'])
    p_values = np.array(results['p_values'])
    bin_width = bin_edges[1] - bin_edges[0]

    # Valid data mask
    valid = ~np.isnan(correlations)

    # Color scheme
    main_color = '#2E86AB'  # Steel blue
    sig_color = '#E94F37'   # Coral red for significant
    nonsig_color = '#A1A1A1'  # Gray for non-significant
    ci_color = '#2E86AB'

    # ===== Top plot: Correlation vs distance =====
    ax1.axhline(0, color='#666666', linestyle='-', linewidth=0.8, alpha=0.5)

    # Add subtle reference band for "near zero"
    ax1.axhspan(-0.05, 0.05, color='#E8E8E8', alpha=0.5, zorder=0)

    if np.any(valid):
        # Separate significant and non-significant points
        sig_mask = valid & (p_values < 0.05)
        nonsig_mask = valid & (p_values >= 0.05)

        # Plot non-significant points (hollow circles, gray)
        if np.any(nonsig_mask):
            yerr_low_ns = correlations[nonsig_mask] - ci_low[nonsig_mask]
            yerr_high_ns = ci_high[nonsig_mask] - correlations[nonsig_mask]
            ax1.errorbar(bin_centers[nonsig_mask], correlations[nonsig_mask],
                        yerr=[yerr_low_ns, yerr_high_ns],
                        fmt='o', color=nonsig_color, capsize=3, capthick=1,
                        markersize=7, markerfacecolor='white', markeredgewidth=1.5,
                        linewidth=0, elinewidth=1.2, label='p ≥ 0.05')

        # Plot significant points (filled circles, colored)
        if np.any(sig_mask):
            yerr_low_s = correlations[sig_mask] - ci_low[sig_mask]
            yerr_high_s = ci_high[sig_mask] - correlations[sig_mask]
            ax1.errorbar(bin_centers[sig_mask], correlations[sig_mask],
                        yerr=[yerr_low_s, yerr_high_s],
                        fmt='o', color=sig_color, capsize=3, capthick=1,
                        markersize=7, markerfacecolor=sig_color, markeredgewidth=1.5,
                        linewidth=0, elinewidth=1.2, label='p < 0.05')

        # Connect points with a subtle line
        ax1.plot(bin_centers[valid], correlations[valid],
                '-', color=main_color, alpha=0.4, linewidth=1, zorder=1)

    # Metric-specific labels
    metric = results.get('metric', 'relative')
    metric_labels = {
        'relative': r'$\Delta r = (r - \bar{r})/\bar{r}$',
        'zscore': r'$\Delta r = (r - \bar{r})/\sigma$',
        'raw': r'$\Delta r = r - \bar{r}$ (μm)',
        'log_ratio': r'$\Delta r = \log(r/\bar{r})$'
    }
    metric_label = metric_labels.get(metric, metric_labels['relative'])

    ax1.set_ylabel('Pearson Correlation of $\\Delta r$', fontsize=11)
    ax1.set_title(f'{sample_name}: Spatial Correlation of Radius Deviations\n'
                  f'{metric_label}',
                  fontsize=11, fontweight='medium')
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=9)

    # Determine y-axis limits based on data
    if np.any(valid):
        max_abs = max(abs(np.nanmin(correlations)), abs(np.nanmax(correlations)))
        y_limit = max(0.15, min(0.4, max_abs * 1.5))
    else:
        y_limit = 0.3
    ax1.set_ylim(-y_limit, y_limit)
    ax1.tick_params(axis='both', labelsize=10)

    # ===== Bottom plot: Number of pairs per bin =====
    colors = [sig_color if p < 0.05 and not np.isnan(p) else nonsig_color
              for p in p_values]
    bars = ax2.bar(bin_centers, n_pairs, width=bin_width * 0.85,
                   color=colors, edgecolor='white', linewidth=0.5, alpha=0.8)

    ax2.set_xlabel('Centroid Distance (μm)', fontsize=11)
    ax2.set_ylabel('N Pairs', fontsize=10)
    ax2.set_yscale('log')
    ax2.tick_params(axis='both', labelsize=9)
    ax2.set_xlim(bin_edges[0], bin_edges[-1])

    # Remove top spine from bottom plot
    ax2.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    # Add summary statistics as text annotation
    nn_corr = results.get('nearest_neighbor_correlation', None)
    sr_corr = results.get('short_range_mean_correlation', None)
    n_axons = results.get('n_axons', 0)

    summary_text = f'n = {n_axons} axons'
    if nn_corr is not None:
        summary_text += f'\nNN corr: {nn_corr:.3f}'
    if sr_corr is not None:
        summary_text += f'\n<10μm mean: {sr_corr:.3f}'

    ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.9))

    fig.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.12)
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    # Reset style
    plt.style.use('default')
    logger.info(f"Saved correlation plot to {output_file}")


def analyze_spatial_correlation(mat_file: Path,
                                 output_dir: Path,
                                 voxel_size_um: Union[float, Tuple[float, float, float], None] = None,
                                 slice_axis: int = None,
                                 n_jobs: int = -1,
                                 max_distance: float = 50.0,
                                 bin_width: float = 2.0,
                                 metric: str = 'relative') -> Dict:
    """
    Main analysis function: compute spatial correlation of radius deviations.

    Args:
        mat_file: Path to .mat file with labeled axons
        output_dir: Directory for output plots and data
        voxel_size_um: Voxel size in micrometers
        slice_axis: Axis to slice along (0=Z, 1=Y, 2=X)
        n_jobs: Number of parallel jobs
        max_distance: Maximum distance for correlation analysis (um)
        bin_width: Width of distance bins (um)
        metric: Deviation metric ('relative', 'zscore', 'raw', 'log_ratio')

    Returns:
        Dict with analysis results
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: {METRICS}")

    logger.info(f"Using metric: {metric}")

    # Load metadata
    metadata = load_metadata(mat_file)

    if slice_axis is None:
        slice_axis = metadata.get('dominant_axis', 0)
        logger.info(f"Using slice_axis from metadata: {slice_axis}")

    if voxel_size_um is None:
        if 'voxel_size_um' in metadata:
            voxel_size_um = tuple(metadata['voxel_size_um'])
        else:
            voxel_size_um = 0.05
            logger.warning("No voxel_size specified, defaulting to 0.05")

    voxel_size_tuple = parse_voxel_size(voxel_size_um)

    # Load and resample volume
    volume = load_mat_volume(mat_file)
    volume, iso_voxel_size = resample_to_isotropic(volume, voxel_size_tuple)

    logger.info(f"Working with isotropic voxel size: {iso_voxel_size:.4f} um")

    n_slices = volume.shape[slice_axis]
    axis_names = {0: 'Z', 1: 'Y', 2: 'X'}
    logger.info(f"Slicing along axis {slice_axis} ({axis_names[slice_axis]}), {n_slices} slices")

    # Pass 1: Compute per-axon statistics (mean and std)
    axon_mean_radii, axon_std_radii = compute_per_axon_stats(volume, slice_axis, iso_voxel_size, n_jobs)

    # Determine workers for Pass 2
    if n_jobs == -1:
        n_workers = cpu_count()
    elif n_jobs == 1:
        n_workers = 1
    else:
        n_workers = min(n_jobs, cpu_count())

    # Pass 2: Extract centroids and deviations
    logger.info(f"Pass 2: Extracting centroids and deviations ({n_workers} workers)")

    if n_workers > 1:
        with Pool(processes=n_workers,
                  initializer=_init_worker_pass2,
                  initargs=(volume, slice_axis, iso_voxel_size, axon_mean_radii, axon_std_radii, metric)) as pool:
            slice_data_list = list(tqdm(
                pool.imap(_process_slice_pass2, range(n_slices)),
                total=n_slices,
                desc="Pass 2: Deviations"
            ))
    else:
        _init_worker_pass2(volume, slice_axis, iso_voxel_size, axon_mean_radii, axon_std_radii, metric)
        slice_data_list = [_process_slice_pass2(z) for z in tqdm(range(n_slices), desc="Pass 2: Deviations")]

    # Setup distance bins
    distance_bins = np.arange(0, max_distance + bin_width, bin_width)
    logger.info(f"Distance bins: {len(distance_bins)-1} bins from 0 to {max_distance} um (width={bin_width} um)")

    # Compute pairwise correlations
    results = compute_pairwise_correlations(slice_data_list, distance_bins)

    # Add metadata
    sample_name = mat_file.stem.replace('_myelinated_axons', '')
    results['sample_name'] = sample_name
    results['metric'] = metric
    results['n_axons'] = len(axon_mean_radii)
    results['n_slices'] = n_slices
    results['voxel_size_isotropic'] = iso_voxel_size
    results['slice_axis'] = slice_axis

    # Summary statistics
    valid_corrs = [c for c in results['correlations'] if not np.isnan(c)]
    if valid_corrs:
        # Find nearest-neighbor correlation (first valid bin)
        nn_corr = next((c for c in results['correlations'] if not np.isnan(c)), np.nan)
        results['nearest_neighbor_correlation'] = nn_corr
        results['mean_correlation'] = float(np.mean(valid_corrs))

        # Check if significant negative correlation at short distances
        short_range_corrs = [c for c, d in zip(results['correlations'], results['bin_centers'])
                           if d < 10 and not np.isnan(c)]
        if short_range_corrs:
            results['short_range_mean_correlation'] = float(np.mean(short_range_corrs))

    logger.info(f"\nResults:")
    logger.info(f"  Total axon pairs analyzed: {sum(results['n_pairs'])}")
    if 'nearest_neighbor_correlation' in results:
        logger.info(f"  Nearest-neighbor correlation: {results['nearest_neighbor_correlation']:.3f}")
    if 'short_range_mean_correlation' in results:
        logger.info(f"  Short-range (<10um) mean correlation: {results['short_range_mean_correlation']:.3f}")

    # Create output directory and save
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot
    plot_file = output_dir / f'{sample_name}_radius_spatial_correlation.png'
    plot_correlation_function(results, plot_file, sample_name)

    # Save JSON
    json_file = output_dir / f'{sample_name}_radius_spatial_correlation.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {json_file}")

    return results


def batch_analyze(input_dir: Path,
                  output_dir: Path,
                  voxel_size_um: Union[float, Tuple[float, float, float], None] = None,
                  slice_axis: int = None,
                  n_jobs: int = -1,
                  max_distance: float = 50.0,
                  bin_width: float = 2.0,
                  metric: str = 'relative') -> Dict[str, Dict]:
    """Batch analyze all .mat files in a directory."""
    mat_files = sorted(input_dir.glob('*_myelinated_axons.mat'))

    if not mat_files:
        logger.error(f"No *_myelinated_axons.mat files found in {input_dir}")
        return {}

    logger.info(f"Found {len(mat_files)} .mat files to analyze")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for i, mat_file in enumerate(mat_files, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {i}/{len(mat_files)}: {mat_file.name}")
        logger.info(f"{'='*80}")

        try:
            result = analyze_spatial_correlation(
                mat_file, output_dir,
                voxel_size_um=voxel_size_um,
                slice_axis=slice_axis,
                n_jobs=n_jobs,
                max_distance=max_distance,
                bin_width=bin_width,
                metric=metric
            )
            all_results[result['sample_name']] = result
        except Exception as e:
            logger.error(f"Failed to process {mat_file.name}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary
    if all_results:
        summary_file = output_dir / 'spatial_correlation_summary.json'
        # Extract key metrics for summary
        summary = {}
        for name, result in all_results.items():
            summary[name] = {
                'n_axons': result['n_axons'],
                'nearest_neighbor_correlation': result.get('nearest_neighbor_correlation', None),
                'short_range_mean_correlation': result.get('short_range_mean_correlation', None),
                'total_pairs': sum(result['n_pairs'])
            }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nSaved summary to {summary_file}")

        # Print summary table
        logger.info(f"\n{'='*80}")
        logger.info("SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"{'Sample':<25} {'N Axons':<10} {'NN Corr':<12} {'<10um Corr':<12}")
        logger.info(f"{'-'*59}")
        for name, s in sorted(summary.items()):
            nn = f"{s['nearest_neighbor_correlation']:.3f}" if s['nearest_neighbor_correlation'] else "N/A"
            sr = f"{s['short_range_mean_correlation']:.3f}" if s['short_range_mean_correlation'] else "N/A"
            logger.info(f"{name:<25} {s['n_axons']:<10} {nn:<12} {sr:<12}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze spatial correlation of axon radius deviations'
    )
    parser.add_argument('input', type=Path,
                        help='Path to .mat file or directory containing *_myelinated_axons.mat files')
    parser.add_argument('output_dir', type=Path,
                        help='Output directory for plots and results')
    parser.add_argument('--voxel-size', type=parse_voxel_size_arg, default=None,
                        help='Voxel size in um: single value or vz,vy,vx')
    parser.add_argument('--slice-axis', type=int, default=None, choices=[0, 1, 2],
                        help='Axis to slice along: 0=Z, 1=Y, 2=X')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (default: -1 = all CPUs)')
    parser.add_argument('--max-distance', type=float, default=50.0,
                        help='Maximum distance for correlation analysis in um (default: 50)')
    parser.add_argument('--bin-width', type=float, default=2.0,
                        help='Width of distance bins in um (default: 2)')
    parser.add_argument('--metric', type=str, default='relative', choices=METRICS,
                        help='Deviation metric: relative=(r-mean)/mean, zscore=(r-mean)/std, '
                             'raw=r-mean, log_ratio=log(r/mean) (default: relative)')

    args = parser.parse_args()

    if args.input.is_dir():
        batch_analyze(
            args.input, args.output_dir,
            voxel_size_um=args.voxel_size,
            slice_axis=args.slice_axis,
            n_jobs=args.n_jobs,
            max_distance=args.max_distance,
            bin_width=args.bin_width,
            metric=args.metric
        )
    else:
        analyze_spatial_correlation(
            args.input, args.output_dir,
            voxel_size_um=args.voxel_size,
            slice_axis=args.slice_axis,
            n_jobs=args.n_jobs,
            max_distance=args.max_distance,
            bin_width=args.bin_width,
            metric=args.metric
        )
