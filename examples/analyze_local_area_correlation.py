#!/usr/bin/env python3
"""
Analyze local area change correlation between axons and their spatial neighbors.

For each slice along the tract, computes:
1. The area change of each axon relative to its mean area
2. The mean area change of neighboring axons within a 2D spatial kernel
3. Correlates these two quantities

This uses the raw labeled volumes and regionprops for accurate area measurements,
rather than skeleton-based circular approximations.

Hypothesis: If there's local mechanical compensation, when one axon bulges
(positive area change), nearby axons should pinch (negative area change),
yielding a negative correlation.

Usage:
    python analyze_local_area_correlation.py \
        "data/raw/HM/Sham_25_ipsi/HM_25_ipsi_myelinated_axons.mat" \
        fig/local_area_correlation_HM_sham25.png
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

# Load plot settings
settings = get_plot_settings()

# Global variables for worker processes
_volume_data = None
_slice_axis = None
_voxel_size_um = None


def find_populations_json(mat_file: Path) -> Path:
    """Find the populations JSON file for a given .mat file."""
    stem = mat_file.stem.replace('_myelinated_axons', '')
    parts = stem.split('_')
    if len(parts) >= 3:
        microscopy = parts[0]  # HM or LM
        rat_id = parts[1]
        hemisphere = parts[2]
        parent_name = mat_file.parent.name.lower()
        if 'sham' in parent_name:
            condition = 'sham'
        elif 'tbi' in parent_name:
            condition = 'tbi'
        else:
            condition = 'sham'
        json_name = f"{condition}_{rat_id}_{hemisphere}_populations.json"
        json_path = Path(f"data/processed/{microscopy}") / json_name
        if json_path.exists():
            return json_path
    raise FileNotFoundError(f"Could not find populations JSON for {mat_file}")


def get_dominant_axis_from_json(json_file: Path) -> int:
    """Get the most common dominant axis from populations JSON."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    populations = data.get('populations', [])
    if not populations:
        return 0
    best_pop = max(populations, key=lambda p: p.get('n_axons', 0))
    return best_pop.get('dominant_axis', 0)


# =============================================================================
# Worker functions for parallel slice processing
# =============================================================================

def _init_worker(volume_data: np.ndarray, slice_axis: int, voxel_size_um: float):
    """Initialize worker process."""
    global _volume_data, _slice_axis, _voxel_size_um
    _volume_data = volume_data
    _slice_axis = slice_axis
    _voxel_size_um = voxel_size_um


def _process_slice(z: int) -> Dict[int, Tuple[float, float, float]]:
    """
    Extract area and centroid for each axon in a slice.

    Returns: {label: (area_um2, centroid_y_um, centroid_x_um)}
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
        cy, cx = region.centroid
        cy_um = cy * _voxel_size_um
        cx_um = cx * _voxel_size_um
        result[region.label] = (area_um2, cy_um, cx_um)

    return result


def build_axon_profiles(volume: np.ndarray, slice_axis: int,
                        voxel_size_um: float, n_jobs: int = -1,
                        min_slices: int = 10) -> Tuple[Dict[int, Dict], List[Dict]]:
    """
    Build per-axon profiles and per-slice data.

    Returns:
        axon_profiles: Dict mapping label to {
            'mean_area': float,
            'slices': list of slice indices where axon appears
        }
        slice_data: List of dicts, one per slice, each containing:
            {label: {'area': float, 'centroid': (cy, cx), 'area_change_rel': float}}
    """
    n_slices = volume.shape[slice_axis]

    if n_jobs == -1:
        n_workers = cpu_count()
    elif n_jobs == 1:
        n_workers = 1
    else:
        n_workers = min(n_jobs, cpu_count())

    # Process all slices
    if n_workers > 1:
        with Pool(processes=n_workers,
                  initializer=_init_worker,
                  initargs=(volume, slice_axis, voxel_size_um)) as pool:
            slice_results = list(tqdm(
                pool.imap(_process_slice, range(n_slices)),
                total=n_slices,
                desc="  Processing slices",
                leave=False
            ))
    else:
        _init_worker(volume, slice_axis, voxel_size_um)
        slice_results = [_process_slice(z) for z in tqdm(range(n_slices),
                                                          desc="  Processing slices",
                                                          leave=False)]

    # First pass: compute per-axon mean area
    axon_areas = defaultdict(list)
    axon_slices = defaultdict(list)

    for slice_idx, slice_result in enumerate(slice_results):
        for label, (area, cy, cx) in slice_result.items():
            axon_areas[label].append(area)
            axon_slices[label].append(slice_idx)

    # Build axon profiles (filter by min_slices)
    axon_profiles = {}
    for label in axon_areas:
        if len(axon_areas[label]) >= min_slices:
            axon_profiles[label] = {
                'mean_area': np.mean(axon_areas[label]),
                'slices': axon_slices[label]
            }

    # Second pass: compute relative area changes for valid axons
    slice_data = []
    for slice_idx, slice_result in enumerate(slice_results):
        slice_dict = {}
        for label, (area, cy, cx) in slice_result.items():
            if label in axon_profiles:
                mean_area = axon_profiles[label]['mean_area']
                area_change_rel = (area - mean_area) / mean_area if mean_area > 0 else 0.0
                slice_dict[label] = {
                    'area': area,
                    'centroid': (cy, cx),
                    'area_change_rel': area_change_rel
                }
        slice_data.append(slice_dict)

    return axon_profiles, slice_data


# =============================================================================
# Correlation computation
# =============================================================================

def bootstrap_correlation_ci(x: np.ndarray, y: np.ndarray,
                              n_bootstrap: int = 1000,
                              ci: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for Pearson correlation."""
    rng = np.random.default_rng(42)
    n = len(x)
    boot_corrs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = pearsonr(x[idx], y[idx])
        if not np.isnan(r):
            boot_corrs.append(r)

    if len(boot_corrs) < 100:
        return np.nan, np.nan

    alpha = (1 - ci) / 2
    ci_low = np.percentile(boot_corrs, alpha * 100)
    ci_high = np.percentile(boot_corrs, (1 - alpha) * 100)

    return ci_low, ci_high


def compute_total_area_conservation(slice_data: List[Dict],
                                     kernel_radii: List[float],
                                     max_points: int = 100000,
                                     n_bootstrap: int = 1000) -> Dict:
    """
    Test if total axon area within a kernel is conserved.

    For each kernel radius, compute:
    1. Observed variance of total area within kernel
    2. Expected variance if areas varied independently (sum of individual variances)
    3. Ratio: observed/expected (< 1 means compensation, > 1 means positive coupling)

    Returns dict with results.
    """
    results = {
        'kernel_radii': kernel_radii,
        'variance_ratio': [],
        'variance_ratio_ci_low': [],
        'variance_ratio_ci_high': [],
        'observed_cv': [],  # coefficient of variation of total area
        'expected_cv': [],
        'mean_total_area': [],
        'n_valid_points': [],
        'mean_n_neighbors': []
    }

    # Collect all points
    all_points = []
    for slice_idx, slice_dict in enumerate(slice_data):
        for label, data in slice_dict.items():
            all_points.append({
                'slice_idx': slice_idx,
                'label': label,
                'area': data['area'],
                'centroid': data['centroid']
            })

    n_points = len(all_points)
    logger.info(f"  Total data points: {n_points:,}")

    if n_points == 0:
        return results

    # Subsample if needed
    if n_points > max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(n_points, size=max_points, replace=False)
        indices = np.sort(indices)
        sampled_points = [all_points[i] for i in indices]
        logger.info(f"  Subsampled to {max_points:,} points")
    else:
        sampled_points = all_points

    # Group points by slice for efficient neighbor lookup
    points_by_slice = defaultdict(list)
    for pt in all_points:
        points_by_slice[pt['slice_idx']].append(pt)

    # Build KDTree per slice
    trees_by_slice = {}
    arrays_by_slice = {}
    for slice_idx, pts in points_by_slice.items():
        if len(pts) > 1:
            centroids = np.array([p['centroid'] for p in pts])
            trees_by_slice[slice_idx] = cKDTree(centroids)
            arrays_by_slice[slice_idx] = pts

    # For each kernel radius
    for kernel_r in kernel_radii:
        logger.info(f"  Processing kernel radius {kernel_r:.1f} um...")

        total_areas = []  # Total area within kernel for each sampled point
        individual_areas_list = []  # List of individual area arrays for bootstrap
        n_neighbors_list = []

        for pt in tqdm(sampled_points, desc=f"    r={kernel_r:.1f}um", leave=False):
            slice_idx = pt['slice_idx']

            if slice_idx not in trees_by_slice:
                continue

            tree = trees_by_slice[slice_idx]
            pts_in_slice = arrays_by_slice[slice_idx]
            centroid = pt['centroid']

            # Find all axons within radius (including self)
            neighbor_indices = tree.query_ball_point(centroid, kernel_r)

            if len(neighbor_indices) < 2:  # Need at least 2 axons
                continue

            # Get areas of all axons in kernel
            areas_in_kernel = np.array([pts_in_slice[ni]['area'] for ni in neighbor_indices])

            total_areas.append(np.sum(areas_in_kernel))
            individual_areas_list.append(areas_in_kernel)
            n_neighbors_list.append(len(neighbor_indices))

        n_valid = len(total_areas)

        if n_valid < 100:
            results['variance_ratio'].append(np.nan)
            results['variance_ratio_ci_low'].append(np.nan)
            results['variance_ratio_ci_high'].append(np.nan)
            results['observed_cv'].append(np.nan)
            results['expected_cv'].append(np.nan)
            results['mean_total_area'].append(np.nan)
            results['n_valid_points'].append(n_valid)
            results['mean_n_neighbors'].append(np.mean(n_neighbors_list) if n_neighbors_list else 0)
            continue

        total_areas = np.array(total_areas)

        # Observed variance of total area
        observed_var = np.var(total_areas)
        mean_total = np.mean(total_areas)
        observed_cv = np.std(total_areas) / mean_total if mean_total > 0 else np.nan

        # Expected variance if independent: Var(sum) = sum(Var) for independent vars
        # We estimate individual variances from the data
        # For each kernel, compute variance of each "slot" and sum them
        # Approximation: use mean variance per axon * mean number of axons
        all_individual_areas = np.concatenate(individual_areas_list)
        mean_individual_var = np.var(all_individual_areas)
        mean_n_axons = np.mean(n_neighbors_list)
        expected_var = mean_individual_var * mean_n_axons
        expected_cv = np.sqrt(expected_var) / mean_total if mean_total > 0 else np.nan

        # Variance ratio
        variance_ratio = observed_var / expected_var if expected_var > 0 else np.nan

        # Bootstrap CI for variance ratio
        rng = np.random.default_rng(42)
        boot_ratios = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n_valid, size=n_valid, replace=True)
            boot_total = total_areas[idx]
            boot_obs_var = np.var(boot_total)

            # Resample individual areas for expected variance
            boot_indiv = [individual_areas_list[i] for i in idx]
            boot_all_indiv = np.concatenate(boot_indiv)
            boot_indiv_var = np.var(boot_all_indiv)
            boot_n_axons = np.mean([len(individual_areas_list[i]) for i in idx])
            boot_exp_var = boot_indiv_var * boot_n_axons

            if boot_exp_var > 0:
                boot_ratios.append(boot_obs_var / boot_exp_var)

        if len(boot_ratios) > 100:
            ci_low = np.percentile(boot_ratios, 2.5)
            ci_high = np.percentile(boot_ratios, 97.5)
        else:
            ci_low, ci_high = np.nan, np.nan

        results['variance_ratio'].append(variance_ratio)
        results['variance_ratio_ci_low'].append(ci_low)
        results['variance_ratio_ci_high'].append(ci_high)
        results['observed_cv'].append(observed_cv)
        results['expected_cv'].append(expected_cv)
        results['mean_total_area'].append(mean_total)
        results['n_valid_points'].append(n_valid)
        results['mean_n_neighbors'].append(np.mean(n_neighbors_list))

        logger.info(f"    var_ratio={variance_ratio:.4f} [{ci_low:.4f}, {ci_high:.4f}], "
                    f"obs_CV={observed_cv:.4f}, exp_CV={expected_cv:.4f}, n={n_valid:,}")

    return results


def compute_local_correlations(slice_data: List[Dict],
                                kernel_radii: List[float],
                                max_points: int = 100000,
                                n_bootstrap: int = 1000) -> Dict:
    """
    Compute correlation between own area change and neighbor mean area change
    for each kernel radius.

    For each axon in each slice:
    - Find neighbors within kernel radius (2D, based on centroids)
    - Compute mean area change of neighbors
    - Correlate own area change vs neighbor mean area change

    Returns dict with results including bootstrap confidence intervals.
    """
    results = {
        'kernel_radii': kernel_radii,
        'correlations': [],
        'p_values': [],
        'ci_low': [],
        'ci_high': [],
        'n_valid_points': [],
        'mean_n_neighbors': []
    }

    # Collect all (area_change, centroid, slice_idx) tuples
    all_points = []
    for slice_idx, slice_dict in enumerate(slice_data):
        for label, data in slice_dict.items():
            all_points.append({
                'slice_idx': slice_idx,
                'label': label,
                'area_change': data['area_change_rel'],
                'centroid': data['centroid']
            })

    n_points = len(all_points)
    logger.info(f"  Total data points: {n_points:,}")

    if n_points == 0:
        return results

    # Subsample if needed
    if n_points > max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(n_points, size=max_points, replace=False)
        indices = np.sort(indices)
        sampled_points = [all_points[i] for i in indices]
        logger.info(f"  Subsampled to {max_points:,} points")
    else:
        sampled_points = all_points
        indices = np.arange(n_points)

    # Group points by slice for efficient neighbor lookup
    points_by_slice = defaultdict(list)
    for pt in all_points:
        points_by_slice[pt['slice_idx']].append(pt)

    # Build KDTree per slice
    trees_by_slice = {}
    arrays_by_slice = {}
    for slice_idx, pts in points_by_slice.items():
        if len(pts) > 1:
            centroids = np.array([p['centroid'] for p in pts])
            trees_by_slice[slice_idx] = cKDTree(centroids)
            arrays_by_slice[slice_idx] = pts

    # For each kernel radius
    for kernel_r in kernel_radii:
        logger.info(f"  Processing kernel radius {kernel_r:.1f} um...")

        own_changes = []
        neighbor_means = []
        n_neighbors_list = []

        for pt in tqdm(sampled_points, desc=f"    r={kernel_r:.1f}um", leave=False):
            slice_idx = pt['slice_idx']

            if slice_idx not in trees_by_slice:
                continue

            tree = trees_by_slice[slice_idx]
            pts_in_slice = arrays_by_slice[slice_idx]
            centroid = pt['centroid']

            # Find neighbors within radius
            neighbor_indices = tree.query_ball_point(centroid, kernel_r)

            # Exclude self (same label)
            neighbor_changes = []
            for ni in neighbor_indices:
                if pts_in_slice[ni]['label'] != pt['label']:
                    neighbor_changes.append(pts_in_slice[ni]['area_change'])

            n_neighbors = len(neighbor_changes)
            n_neighbors_list.append(n_neighbors)

            if n_neighbors > 0:
                own_changes.append(pt['area_change'])
                neighbor_means.append(np.mean(neighbor_changes))

        # Compute correlation and bootstrap CI
        n_valid = len(own_changes)
        if n_valid > 30:
            own_changes = np.array(own_changes)
            neighbor_means = np.array(neighbor_means)
            r, p = pearsonr(own_changes, neighbor_means)
            ci_low, ci_high = bootstrap_correlation_ci(own_changes, neighbor_means, n_bootstrap)
        else:
            r, p = np.nan, np.nan
            ci_low, ci_high = np.nan, np.nan

        results['correlations'].append(r)
        results['p_values'].append(p)
        results['ci_low'].append(ci_low)
        results['ci_high'].append(ci_high)
        results['n_valid_points'].append(n_valid)
        results['mean_n_neighbors'].append(np.mean(n_neighbors_list) if n_neighbors_list else 0)

        logger.info(f"    r={r:.4f} [{ci_low:.4f}, {ci_high:.4f}], p={p:.2e}, n={n_valid:,}, mean_neighbors={results['mean_n_neighbors'][-1]:.1f}")

    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_variance_ratio(results: Dict, output_file: Path,
                         n_samples: int, total_points: int):
    """Plot variance ratio vs kernel radius with error bars."""
    fig, ax = plt.subplots(figsize=(8, 6))

    font_settings = settings.fonts
    kernel_radii = np.array(results['kernel_radii'])
    variance_ratio = np.array(results['variance_ratio'])
    ci_low = np.array(results['variance_ratio_ci_low'])
    ci_high = np.array(results['variance_ratio_ci_high'])

    # Colors
    main_color = '#2E86AB'
    below_one_color = '#28A745'  # Green for compensation
    above_one_color = '#DC3545'  # Red for positive coupling

    # Reference line at 1 (independent variation)
    ax.axhline(1, color='#666666', linestyle='--', linewidth=1.5, alpha=0.7, label='Independent (ratio=1)')
    ax.axhspan(0.95, 1.05, color='#E8E8E8', alpha=0.5, zorder=0)

    valid = ~np.isnan(variance_ratio)

    # Line connecting points
    ax.plot(kernel_radii[valid], variance_ratio[valid], '-', color=main_color,
            linewidth=2, alpha=0.6, zorder=2)

    # Determine colors based on whether CI excludes 1
    for i, (kr, vr, cl, ch) in enumerate(zip(kernel_radii, variance_ratio, ci_low, ci_high)):
        if np.isnan(vr):
            continue

        yerr_low = vr - cl
        yerr_high = ch - vr

        # Color based on whether significantly different from 1
        if ch < 1:  # Significantly below 1 (compensation)
            color = below_one_color
            mfc = below_one_color
        elif cl > 1:  # Significantly above 1 (positive coupling)
            color = above_one_color
            mfc = above_one_color
        else:  # CI includes 1
            color = '#A1A1A1'
            mfc = 'white'

        ax.errorbar(kr, vr, yerr=[[yerr_low], [yerr_high]],
                    fmt='o', color=color, capsize=4, capthick=1.5,
                    markersize=8, markerfacecolor=mfc, markeredgewidth=2,
                    elinewidth=1.5, zorder=3)

    ax.set_xlabel('Kernel Radius (μm)', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.set_ylabel('Variance Ratio (Observed / Expected)', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.set_title('Total Area Conservation Test\n'
                 r'Ratio < 1 $\Rightarrow$ compensation, Ratio > 1 $\Rightarrow$ positive coupling',
                 fontsize=font_settings['title_size'], fontweight=font_settings['weight'])

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], linestyle='--', color='#666666', label='Independent (ratio=1)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=below_one_color,
               markersize=8, markeredgewidth=2, markeredgecolor=below_one_color, label='Compensation (CI < 1)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=above_one_color,
               markersize=8, markeredgewidth=2, markeredgecolor=above_one_color, label='Positive coupling (CI > 1)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markersize=8, markeredgewidth=2, markeredgecolor='#A1A1A1', label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=font_settings['legend_size'])

    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, kernel_radii.max() * 1.05)

    # Set y-axis limits
    all_vals = np.concatenate([ci_low[~np.isnan(ci_low)], ci_high[~np.isnan(ci_high)]])
    if len(all_vals) > 0:
        y_min = min(0.8, all_vals.min() * 0.95)
        y_max = max(1.2, all_vals.max() * 1.05)
    else:
        y_min, y_max = 0.8, 1.2
    ax.set_ylim(y_min, y_max)

    # Summary text
    summary = f'n = {n_samples} samples\n{total_points:,} points'
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4',
            facecolor='white', edgecolor='#CCCCCC', alpha=0.9))

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"Saved figure to {output_file}")


def plot_kernel_correlations(results: Dict, output_file: Path,
                              n_samples: int, total_points: int):
    """Plot correlation vs kernel radius with error bars."""
    fig, ax = plt.subplots(figsize=(8, 6))

    font_settings = settings.fonts
    kernel_radii = np.array(results['kernel_radii'])
    correlations = np.array(results['correlations'])
    p_values = np.array(results['p_values'])
    ci_low = np.array(results.get('ci_low', [np.nan] * len(correlations)))
    ci_high = np.array(results.get('ci_high', [np.nan] * len(correlations)))

    # Colors
    main_color = '#2E86AB'
    sig_color = '#E94F37'
    nonsig_color = '#A1A1A1'

    # Reference lines
    ax.axhline(0, color='#666666', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axhspan(-0.05, 0.05, color='#E8E8E8', alpha=0.5, zorder=0)

    valid = ~np.isnan(correlations)
    sig_mask = valid & (np.array(p_values) < 0.05)

    # Line connecting points
    ax.plot(kernel_radii[valid], correlations[valid], '-', color=main_color,
            linewidth=2, alpha=0.6, zorder=2)

    # Error bars and points for non-significant
    if np.any(~sig_mask & valid):
        ns_mask = ~sig_mask & valid
        yerr_low = correlations[ns_mask] - ci_low[ns_mask]
        yerr_high = ci_high[ns_mask] - correlations[ns_mask]
        ax.errorbar(kernel_radii[ns_mask], correlations[ns_mask],
                    yerr=[yerr_low, yerr_high],
                    fmt='o', color=nonsig_color, capsize=4, capthick=1.5,
                    markersize=8, markerfacecolor='white', markeredgewidth=2,
                    elinewidth=1.5, label='p >= 0.05', zorder=3)

    # Error bars and points for significant
    if np.any(sig_mask):
        yerr_low = correlations[sig_mask] - ci_low[sig_mask]
        yerr_high = ci_high[sig_mask] - correlations[sig_mask]
        ax.errorbar(kernel_radii[sig_mask], correlations[sig_mask],
                    yerr=[yerr_low, yerr_high],
                    fmt='o', color=sig_color, capsize=4, capthick=1.5,
                    markersize=8, markerfacecolor=sig_color, markeredgewidth=2,
                    elinewidth=1.5, label='p < 0.05', zorder=3)

    ax.set_xlabel('Kernel Radius (μm)', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.set_ylabel('Correlation', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.set_title('Local Area Change Correlation\n'
                 r'$\mathrm{corr}(\Delta A_i / \bar{A}_i, \langle \Delta A_j / \bar{A}_j \rangle_{\mathrm{neighbors}})$',
                 fontsize=font_settings['title_size'], fontweight=font_settings['weight'])
    ax.legend(loc='upper right', fontsize=font_settings['legend_size'])
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, kernel_radii.max() * 1.05)

    # Set y-axis limits based on CI bounds
    all_vals = np.concatenate([ci_low[~np.isnan(ci_low)], ci_high[~np.isnan(ci_high)]])
    if len(all_vals) > 0:
        max_abs = max(abs(all_vals.min()), abs(all_vals.max()))
        y_limit = max(0.15, min(0.5, max_abs * 1.3))
    else:
        y_limit = 0.3
    ax.set_ylim(-y_limit, y_limit)

    # Summary text
    summary = f'n = {n_samples} samples\n{total_points:,} points'
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4',
            facecolor='white', edgecolor='#CCCCCC', alpha=0.9))

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"Saved figure to {output_file}")


# =============================================================================
# Sample processing
# =============================================================================

def process_sample(mat_file: Path, voxel_size_um: Optional[float],
                   n_jobs: int, min_slices: int) -> Tuple[str, Dict[int, Dict], List[Dict]]:
    """Process a single .mat file."""
    sample_name = mat_file.stem.replace('_myelinated_axons', '')
    logger.info(f"Processing {sample_name}...")

    # Load volume
    volume, voxel_tuple, metadata = load_volume_with_metadata(mat_file, voxel_size_um)
    logger.info(f"  Volume shape: {volume.shape}")

    # Get slice axis
    slice_axis = 0
    axis_names = {0: 'Z', 1: 'Y', 2: 'X'}

    if metadata is not None and 'dominant_axis' in metadata:
        slice_axis = metadata['dominant_axis']
        logger.info(f"  Using dominant axis from metadata: {slice_axis} ({axis_names[slice_axis]})")
    else:
        try:
            json_file = find_populations_json(mat_file)
            slice_axis = get_dominant_axis_from_json(json_file)
            logger.info(f"  Using dominant axis from populations JSON: {slice_axis} ({axis_names[slice_axis]})")
        except FileNotFoundError:
            logger.warning(f"  No dominant axis found, defaulting to axis 0 (Z)")

    # Resample to isotropic
    volume, iso_voxel_size = resample_to_isotropic(volume, voxel_tuple)
    n_slices = volume.shape[slice_axis]
    logger.info(f"  {n_slices} slices along axis {slice_axis}, voxel: {iso_voxel_size:.3f} um")

    # Build per-axon profiles and slice data
    axon_profiles, slice_data = build_axon_profiles(
        volume, slice_axis, iso_voxel_size, n_jobs, min_slices=min_slices
    )
    n_axons = len(axon_profiles)
    logger.info(f"  Found {n_axons} axons with >= {min_slices} slices")

    return sample_name, axon_profiles, slice_data


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze local area change correlation vs spatial kernel size',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single .mat file
  python analyze_local_area_correlation.py \\
      "data/raw/HM/Sham_25_ipsi/HM_25_ipsi_myelinated_axons.mat" \\
      fig/local_area_correlation_HM_sham25.png

  # Process all HM samples pooled
  python analyze_local_area_correlation.py \\
      "data/raw/HM/*/HM_*_myelinated_axons.mat" \\
      fig/local_area_correlation_HM_pooled.png
        """
    )

    parser.add_argument('input_pattern', type=str,
                        help='Glob pattern for .mat files')
    parser.add_argument('output', type=Path,
                        help='Output PNG file path')
    parser.add_argument('--kernel-radii', type=float, nargs='+',
                        default=[1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0],
                        help='Kernel radii to test in um (default: 1 2 3 5 7 10 15 20)')
    parser.add_argument('--max-points', type=int, default=100000,
                        help='Maximum points to use for correlation (default: 100000)')
    parser.add_argument('--min-slices', type=int, default=10,
                        help='Minimum slices for an axon to be included (default: 10)')
    parser.add_argument('--voxel-size', type=float, default=None,
                        help='Override voxel size in um (default: from metadata)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 = all cores)')

    args = parser.parse_args()

    # Find all matching files
    mat_files = sorted([Path(f) for f in glob_module.glob(args.input_pattern, recursive=True)])

    if not mat_files:
        logger.error(f"No files found matching pattern: {args.input_pattern}")
        return

    logger.info(f"Found {len(mat_files)} .mat files")

    # Process each sample and collect slice data
    all_slice_data = []
    sample_names = []
    total_axons = 0

    for mat_file in mat_files:
        try:
            sample_name, axon_profiles, slice_data = process_sample(
                mat_file,
                voxel_size_um=args.voxel_size,
                n_jobs=args.n_jobs,
                min_slices=args.min_slices
            )
            sample_names.append(sample_name)
            total_axons += len(axon_profiles)
            all_slice_data.extend(slice_data)

        except Exception as e:
            logger.error(f"Failed to process {mat_file.name}: {e}")
            import traceback
            traceback.print_exc()

    if not sample_names:
        logger.error("No samples processed successfully")
        return

    # Count total points
    total_points = sum(len(sd) for sd in all_slice_data)
    logger.info(f"\nTotal: {total_points:,} data points from {len(sample_names)} samples, {total_axons} axons")

    # Compute correlations
    logger.info("\nComputing local correlations for each kernel radius...")
    corr_results = compute_local_correlations(
        all_slice_data,
        kernel_radii=args.kernel_radii,
        max_points=args.max_points
    )

    # Compute total area conservation test
    logger.info("\nComputing total area conservation test...")
    conservation_results = compute_total_area_conservation(
        all_slice_data,
        kernel_radii=args.kernel_radii,
        max_points=args.max_points
    )

    # Combine results
    results = {
        'correlation': corr_results,
        'conservation': conservation_results,
        'n_samples': len(sample_names),
        'total_points': total_points,
        'total_axons': total_axons,
        'sample_names': sample_names,
        'parameters': {
            'kernel_radii': args.kernel_radii,
            'max_points': args.max_points,
            'min_slices': args.min_slices
        }
    }

    # Plot correlation
    plot_kernel_correlations(corr_results, args.output, len(sample_names), total_points)

    # Plot variance ratio (conservation test)
    conservation_output = args.output.with_stem(args.output.stem + '_conservation')
    plot_variance_ratio(conservation_results, conservation_output, len(sample_names), total_points)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY - CORRELATION ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Samples: {len(sample_names)}")
    logger.info(f"Total axons: {total_axons:,}")
    logger.info(f"Total points: {total_points:,}")
    logger.info(f"Kernel radii tested: {args.kernel_radii}")
    logger.info("\nCorrelations (own vs neighbor area change):")
    for r, corr, p in zip(corr_results['kernel_radii'],
                          corr_results['correlations'],
                          corr_results['p_values']):
        sig = '*' if p < 0.05 else ''
        logger.info(f"  r={r:5.1f} um: {corr:+.4f} (p={p:.2e}){sig}")

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY - TOTAL AREA CONSERVATION")
    logger.info("=" * 70)
    logger.info("Variance ratio: observed/expected (< 1 = compensation, > 1 = positive coupling)")
    for r, vr, cl, ch in zip(conservation_results['kernel_radii'],
                              conservation_results['variance_ratio'],
                              conservation_results['variance_ratio_ci_low'],
                              conservation_results['variance_ratio_ci_high']):
        if ch < 1:
            sig = '** (compensation)'
        elif cl > 1:
            sig = '** (positive coupling)'
        else:
            sig = ''
        logger.info(f"  r={r:5.1f} um: {vr:.4f} [{cl:.4f}, {ch:.4f}] {sig}")

    # Save JSON (convert numpy types for serialization)
    json_file = args.output.with_suffix('.json')

    def convert_numpy(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj

    with open(json_file, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    logger.info(f"\nSaved results to {json_file}")


if __name__ == '__main__':
    main()
