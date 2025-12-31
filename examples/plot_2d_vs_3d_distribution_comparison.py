#!/usr/bin/env python3
"""
Compare 2D slice-based vs 3D skeleton-based radius distributions.

Creates a 2×2 panel figure:
(a) PDF stability across slices for multiple samples (small/medium/large mean radius)
(b) QQ-plot comparing 2D vs 3D distributions with slice variability
(c) Arithmetic mean radius scatter plot (2D vs 3D) for all samples
(d) Effective radius scatter plot (2D vs 3D) for all samples

Uses LM data with CC/CG population separation.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from axonometry import get_plot_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load plot settings
settings = get_plot_settings()

# Constants
MIN_AXON_COUNT = 100  # Minimum axons per slice for valid statistics


def compute_r_eff(radii: np.ndarray) -> float:
    """Compute effective MRI-visible radius: r_eff = (⟨r⁶⟩/⟨r²⟩)^(1/4)"""
    if len(radii) == 0:
        return np.nan
    r2 = radii ** 2
    r6 = radii ** 6
    r2_mean = np.mean(r2)
    r6_mean = np.mean(r6)
    if r2_mean == 0:
        return np.nan
    return (r6_mean / r2_mean) ** 0.25


def find_largest_sample(data_dir: Path, population: str = 'cc') -> Tuple[Path, Path, str]:
    """
    Find the sample with the most total axons.

    Returns:
        Tuple of (slice_profile_path, axon_profile_path, sample_name)
    """
    pattern = f"*_{population}_slice_profiles.npz"
    slice_files = sorted(data_dir.glob(pattern))

    if not slice_files:
        raise FileNotFoundError(f"No {pattern} files found in {data_dir}")

    # Find largest by total axon count
    best_file = None
    best_count = 0

    for f in slice_files:
        npz = np.load(f)
        total = npz['n_axons_per_slice'].sum()
        if total > best_count:
            best_count = total
            best_file = f

    # Extract base name and find corresponding 3D file
    # e.g., sham_25_ipsi_cc_slice_profiles.npz -> sham_25_ipsi
    base_name = best_file.stem.replace(f'_{population}_slice_profiles', '')
    sample_name = f"{base_name}_{population.upper()}"
    axon_file = data_dir / f"{base_name}_axon_profiles.npz"

    logger.info(f"Largest sample: {sample_name} with {best_count:,} total axons")

    return best_file, axon_file, sample_name


def find_samples_by_mean_radius(data_dir: Path, min_axons: int = 1_000_000
                                 ) -> List[Tuple[Path, Path, str, float]]:
    """
    Find samples with small, medium, and high mean radius.

    Args:
        data_dir: Directory containing slice profile files
        min_axons: Minimum total axons to consider a sample

    Returns:
        List of 3 tuples: (slice_file, axon_file, sample_name, mean_radius)
        ordered as [small, medium, high]
    """
    results = []

    for pattern in ["*_cc_slice_profiles.npz", "*_cg_slice_profiles.npz"]:
        for f in sorted(data_dir.glob(pattern)):
            npz = np.load(f)
            total_axons = npz['n_axons_per_slice'].sum()

            if total_axons < min_axons:
                continue

            bin_centers = npz['bin_centers']
            total_hist = npz['total_histogram_circular']
            mean_r = np.sum(bin_centers * total_hist) / total_hist.sum()

            pop = "cc" if "_cc_" in f.name else "cg"
            base_name = f.stem.replace(f'_{pop}_slice_profiles', '')
            sample_name = f"{base_name}_{pop.upper()}"
            axon_file = data_dir / f"{base_name}_axon_profiles.npz"

            if axon_file.exists():
                results.append((f, axon_file, sample_name, mean_r))

    # Sort by mean radius
    results.sort(key=lambda x: x[3])

    if len(results) < 3:
        raise ValueError(f"Need at least 3 samples, found {len(results)}")

    # Pick small (near start), medium (middle), high (near end)
    n = len(results)
    small_idx = 1  # avoid very smallest
    med_idx = n // 2
    high_idx = n - 2  # avoid very largest

    selected = [results[small_idx], results[med_idx], results[high_idx]]

    logger.info("Selected samples by mean radius:")
    for sf, af, name, mean_r in selected:
        logger.info(f"  {name}: mean_r = {mean_r:.3f} μm")

    return selected


def load_population_labels(data_dir: Path, base_name: str, population: str) -> Set[int]:
    """Load axon labels for a specific population from JSON file."""
    pop_json = data_dir / f"{base_name}_populations.json"

    with open(pop_json, 'r') as f:
        data = json.load(f)

    for pop in data['populations']:
        if pop['name'].lower() == population.lower():
            return set(pop['axon_labels'])

    raise ValueError(f"Population '{population}' not found in {pop_json}")


def histogram_to_pdf(histogram: np.ndarray, bin_centers: np.ndarray) -> np.ndarray:
    """Convert histogram counts to probability density."""
    total = histogram.sum()
    if total == 0:
        return np.zeros_like(histogram, dtype=float)

    bin_width = bin_centers[1] - bin_centers[0]
    pdf = histogram / (total * bin_width)
    return pdf


def histogram_to_quantiles(histogram: np.ndarray, bin_centers: np.ndarray,
                           quantiles: np.ndarray) -> np.ndarray:
    """
    Compute quantiles from histogram data.

    Args:
        histogram: Counts per bin
        bin_centers: Center values of bins
        quantiles: Quantile values to compute (e.g., [0.1, 0.25, 0.5, 0.75, 0.9])

    Returns:
        Array of radius values at the specified quantiles
    """
    total = histogram.sum()
    if total == 0:
        return np.full(len(quantiles), np.nan)

    # Compute CDF
    cdf = np.cumsum(histogram) / total

    # Interpolate to find quantile values
    result = np.interp(quantiles, cdf, bin_centers)
    return result


def plot_pdf_stability(ax, slice_file: Path, radius_type: str = 'circular',
                       x_max: float = 2.0) -> None:
    """
    Plot PDF stability across slices with median ± IQR envelope for a single sample.

    Args:
        ax: Matplotlib axes
        slice_file: Path to slice profiles NPZ
        radius_type: 'circular' or 'minor'
        x_max: Maximum x-axis value for cropping
    """
    npz = np.load(slice_file)
    bin_centers = npz['bin_centers']
    histograms = npz[f'histograms_{radius_type}']  # (n_slices, n_bins)

    n_slices, n_bins = histograms.shape
    logger.info(f"Computing PDFs for {n_slices} slices...")

    # Convert each slice histogram to PDF
    pdfs = np.zeros((n_slices, n_bins))
    valid_slices = []

    for i in range(n_slices):
        if histograms[i].sum() > MIN_AXON_COUNT:
            pdfs[i] = histogram_to_pdf(histograms[i], bin_centers)
            valid_slices.append(i)

    logger.info(f"Valid slices with >{MIN_AXON_COUNT} axons: {len(valid_slices)}")

    if len(valid_slices) == 0:
        logger.error("No valid slices found - cannot plot PDF stability")
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        return

    pdfs_valid = pdfs[valid_slices]

    # Compute median and IQR across slices
    pdf_median = np.median(pdfs_valid, axis=0)
    pdf_lo = np.percentile(pdfs_valid, 25, axis=0)
    pdf_hi = np.percentile(pdfs_valid, 75, axis=0)

    # Crop to x_max
    mask = bin_centers <= x_max
    x = bin_centers[mask]
    y_med = pdf_median[mask]
    y_lo = pdf_lo[mask]
    y_hi = pdf_hi[mask]

    # Plot
    font_settings = settings.fonts
    line_settings = settings.line

    ax.fill_between(x, y_lo, y_hi, alpha=0.3, color='steelblue', label='IQR')
    ax.plot(x, y_med, color='steelblue', linewidth=line_settings['linewidth'], label='Median')

    ax.set_xlabel('Axon radius [μm]', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.set_ylabel('Probability density', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.legend(loc='upper right', fontsize=font_settings['legend_size'])
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_box_aspect(1)  # Square subplot box


def plot_pdf_stability_multi(ax, samples: List[Tuple[Path, Path, str, float]],
                              data_dir: Path,
                              radius_type: str = 'circular',
                              x_max: float = 2.0) -> None:
    """
    Plot PDF comparison: 3D (solid) vs 2D (dashed + IQR) for multiple samples.

    Args:
        ax: Matplotlib axes
        samples: List of (slice_file, axon_file, sample_name, mean_radius) tuples
        data_dir: Directory containing population JSON files
        radius_type: 'circular' or 'minor'
        x_max: Maximum x-axis value for cropping
    """
    font_settings = settings.fonts
    line_settings = settings.line

    # Colors for small, medium, high
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # green, blue, red
    sample_info = []  # Store (color, short_name, mean_r) for custom legend

    for idx, (slice_file, axon_file, sample_name, mean_r) in enumerate(samples):
        # Load 2D slice data
        npz_2d = np.load(slice_file)
        bin_centers = npz_2d['bin_centers']
        histograms = npz_2d[f'histograms_{radius_type}']

        n_slices, n_bins = histograms.shape

        # Convert each slice histogram to PDF
        pdfs = np.zeros((n_slices, n_bins))
        valid_slices = []

        for i in range(n_slices):
            if histograms[i].sum() > MIN_AXON_COUNT:
                pdfs[i] = histogram_to_pdf(histograms[i], bin_centers)
                valid_slices.append(i)

        if len(valid_slices) == 0:
            logger.warning(f"No valid slices for {sample_name}")
            continue

        pdfs_valid = pdfs[valid_slices]

        # Compute 2D median and IQR
        pdf_2d_median = np.median(pdfs_valid, axis=0)
        pdf_2d_lo = np.percentile(pdfs_valid, 25, axis=0)
        pdf_2d_hi = np.percentile(pdfs_valid, 75, axis=0)

        # Load 3D axon data and filter by population
        population = sample_name.split('_')[-1].lower()  # e.g., 'cc' or 'cg'
        base_name = slice_file.stem.replace(f'_{population}_slice_profiles', '')
        pop_labels = load_population_labels(data_dir, base_name, population)

        npz_3d = np.load(axon_file, allow_pickle=True)
        axon_labels = npz_3d['labels']
        radii_profiles = npz_3d['radii_profiles_um']

        # Collect all 3D radii for this population
        all_radii_3d = []
        for i, label in enumerate(axon_labels):
            if int(label) in pop_labels:
                all_radii_3d.extend(radii_profiles[i])
        all_radii_3d = np.array(all_radii_3d)

        # Compute 3D PDF using same bin centers
        bin_width = bin_centers[1] - bin_centers[0]
        bin_edges = np.concatenate([
            [bin_centers[0] - bin_width / 2],
            bin_centers + bin_width / 2
        ])
        hist_3d, _ = np.histogram(all_radii_3d, bins=bin_edges)
        pdf_3d = histogram_to_pdf(hist_3d, bin_centers)

        # Crop to x_max
        mask = bin_centers <= x_max
        x = bin_centers[mask]
        y_2d_med = pdf_2d_median[mask]
        y_2d_lo = pdf_2d_lo[mask]
        y_2d_hi = pdf_2d_hi[mask]
        y_3d = pdf_3d[mask]

        # Plot
        color = colors[idx]
        # Extract short name like "25 ipsi CC" from "sham_25_ipsi_CC"
        parts = sample_name.split('_')
        short_name = f"{parts[1]} {parts[2]} {parts[3]}"

        # 3D: solid line (no label, we'll build custom legend)
        ax.plot(x, y_3d, color=color, linewidth=line_settings['linewidth'],
                linestyle='-')
        # 2D: IQR shading only (no median line)
        ax.fill_between(x, y_2d_lo, y_2d_hi, alpha=0.3, color=color)

        # Store info for custom legend
        sample_info.append((color, short_name, mean_r))

    # Build custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    handles = []
    labels = []

    # Style indicators (first row)
    handles.append(Line2D([0], [0], color='gray', linewidth=line_settings['linewidth'], linestyle='-'))
    labels.append('3D')
    handles.append(Patch(facecolor='gray', alpha=0.3, edgecolor='none'))
    labels.append('2D IQR')

    # Sample indicators with colors
    for color, short_name, mean_r in sample_info:
        handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=color,
                              markersize=8, markeredgecolor='none'))
        labels.append(rf"{short_name} ($\bar{{r}}$ = {mean_r:.2f})")

    ax.set_xlabel('Axon radius [μm]', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.set_ylabel('Probability density', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.legend(handles, labels, loc='upper right', fontsize=font_settings['legend_size'] - 1,
              framealpha=0.9)
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_box_aspect(1)  # Square subplot box


def plot_qq_with_variability(ax, slice_file: Path, axon_file: Path,
                             population_labels: Set[int],
                             radius_type: str = 'circular',
                             n_quantiles: int = 100) -> None:
    """
    Plot QQ-plot comparing 2D vs 3D distributions with slice variability.

    Args:
        ax: Matplotlib axes
        slice_file: Path to 2D slice profiles NPZ
        axon_file: Path to 3D axon profiles NPZ
        population_labels: Set of axon labels for filtering 3D data
        radius_type: 'circular' or 'minor'
        n_quantiles: Number of quantile points
    """
    # Load 2D data
    npz_2d = np.load(slice_file)
    bin_centers = npz_2d['bin_centers']
    histograms = npz_2d[f'histograms_{radius_type}']

    # Load 3D data (filtered by population)
    npz_3d = np.load(axon_file, allow_pickle=True)
    axon_labels = npz_3d['labels']
    radii_profiles = npz_3d['radii_profiles_um']

    # Filter 3D radii by population
    filtered_radii = []
    for i, label in enumerate(axon_labels):
        if int(label) in population_labels:
            filtered_radii.append(radii_profiles[i])

    if not filtered_radii:
        logger.error("No axons matched population labels - cannot create QQ plot")
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        return

    all_radii_3d = np.concatenate(filtered_radii)
    logger.info(f"3D radii: {len(all_radii_3d):,} measurements from {len(filtered_radii)} axons")

    # Define quantile points
    quantiles = np.linspace(0.01, 0.99, n_quantiles)

    # Compute 3D quantiles (reference)
    q_3d = np.percentile(all_radii_3d, quantiles * 100)

    # Compute 2D quantiles per slice
    n_slices = histograms.shape[0]
    q_2d_all = []

    for i in range(n_slices):
        if histograms[i].sum() > MIN_AXON_COUNT:
            q_2d = histogram_to_quantiles(histograms[i], bin_centers, quantiles)
            q_2d_all.append(q_2d)

    logger.info(f"Computed quantiles for {len(q_2d_all)} valid slices")

    if len(q_2d_all) == 0:
        logger.error("No valid slices found - cannot create QQ plot")
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        return

    q_2d_all = np.array(q_2d_all)  # (n_valid_slices, n_quantiles)

    # Compute median and IQR across slices
    q_2d_median = np.median(q_2d_all, axis=0)
    q_2d_lo = np.percentile(q_2d_all, 25, axis=0)
    q_2d_hi = np.percentile(q_2d_all, 75, axis=0)

    # Plot
    font_settings = settings.fonts
    line_settings = settings.line

    # Identity line
    max_val = max(q_3d.max(), q_2d_hi.max()) * 1.05
    min_val = min(q_3d.min(), q_2d_lo.min()) * 0.95
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5,
            linewidth=1.5, label='Identity', zorder=0)

    # QQ envelope and median
    ax.fill_between(q_3d, q_2d_lo, q_2d_hi, alpha=0.3, color='steelblue', label='IQR')
    ax.plot(q_3d, q_2d_median, color='steelblue', linewidth=line_settings['linewidth'],
            label='Median')

    ax.set_xlabel('3D axon radius quantile [μm]', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.set_ylabel('2D axon radius quantile [μm]', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.legend(loc='upper left', fontsize=font_settings['legend_size'])
    ax.set_box_aspect(1)  # Square subplot box
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)


def extract_group_info(sample_name: str) -> Tuple[str, str]:
    """Extract group (TBI/Sham) and population (CC/CG) from sample name."""
    name_lower = sample_name.lower()

    if 'tbi' in name_lower:
        group = "TBI"
    elif 'sham' in name_lower:
        group = "Sham"
    else:
        group = "Unknown"

    if sample_name.endswith('_CC'):
        population = "CC"
    elif sample_name.endswith('_CG'):
        population = "CG"
    else:
        population = None

    return group, population


def load_2d_metrics(npz_file: Path, radius_type: str = 'circular') -> Dict[str, float]:
    """Load 2D slice-based metrics from NPZ file."""
    data = np.load(npz_file)
    bin_centers = data['bin_centers']

    histograms = data[f'histograms_{radius_type}']
    r_eff_per_slice = data[f'r_eff_{radius_type}_per_slice']

    # Compute per-slice arithmetic mean
    mean_radius_per_slice = []
    for hist_slice in histograms:
        counts = hist_slice.sum()
        if counts > 0:
            mean_r = np.sum(bin_centers * hist_slice) / counts
            mean_radius_per_slice.append(mean_r)

    mean_radius_per_slice = np.array(mean_radius_per_slice)

    # Compute median and IQR
    mean_radius_median = np.median(mean_radius_per_slice)
    mean_radius_lo = np.percentile(mean_radius_per_slice, 25)
    mean_radius_hi = np.percentile(mean_radius_per_slice, 75)

    valid_r_eff = r_eff_per_slice[r_eff_per_slice > 0]
    r_eff_median = np.median(valid_r_eff)
    r_eff_lo = np.percentile(valid_r_eff, 25)
    r_eff_hi = np.percentile(valid_r_eff, 75)

    return {
        'mean_radius': mean_radius_median,
        'mean_radius_lo': mean_radius_lo,
        'mean_radius_hi': mean_radius_hi,
        'r_eff': r_eff_median,
        'r_eff_lo': r_eff_lo,
        'r_eff_hi': r_eff_hi
    }


def load_3d_metrics(npz_file: Path, population_labels: Optional[Set[int]] = None) -> Dict[str, float]:
    """Load 3D axon-based metrics from NPZ file."""
    data = np.load(npz_file, allow_pickle=True)

    if population_labels is not None:
        axon_labels = data['labels']
        radii_profiles = data['radii_profiles_um']

        filtered_radii_list = []
        for i, label in enumerate(axon_labels):
            if int(label) in population_labels:
                filtered_radii_list.append(radii_profiles[i])

        if not filtered_radii_list:
            logger.warning(f"No axons matched population labels in {npz_file.name}")
            return {'mean_radius': np.nan, 'r_eff': np.nan}

        all_radii = np.concatenate(filtered_radii_list)
    else:
        all_radii = data['all_radii_um']

    return {
        'mean_radius': np.mean(all_radii),
        'r_eff': compute_r_eff(all_radii)
    }


def find_matching_pairs(data_dir: Path) -> List[Tuple[Path, Path, str, Set[int]]]:
    """
    Find all matching 2D/3D file pairs with population labels.

    Returns:
        List of (slice_file, axon_file, sample_name, population_labels) tuples
    """
    import re

    pairs = []

    # Find all population-specific slice files
    for pop in ['cc', 'cg']:
        slice_files = sorted(data_dir.glob(f"*_{pop}_slice_profiles.npz"))

        for slice_file in slice_files:
            # Extract base name
            base_name = slice_file.stem.replace(f'_{pop}_slice_profiles', '')
            axon_file = data_dir / f"{base_name}_axon_profiles.npz"
            pop_json = data_dir / f"{base_name}_populations.json"

            if axon_file.exists() and pop_json.exists():
                # Load population labels
                with open(pop_json, 'r') as f:
                    pop_data = json.load(f)

                for p in pop_data['populations']:
                    if p['name'].lower() == pop:
                        labels = set(p['axon_labels'])
                        sample_name = f"{base_name}_{pop.upper()}"
                        pairs.append((slice_file, axon_file, sample_name, labels))
                        break

    return pairs


def plot_ensemble_scatter(ax, all_metrics: List[Tuple[Dict, Dict, str]],
                          metric: str, panel_label: str) -> None:
    """
    Plot 2D vs 3D scatter for ensemble metrics.

    Args:
        ax: Matplotlib axes
        all_metrics: List of (metrics_2d, metrics_3d, sample_name) tuples
        metric: 'mean_radius' or 'r_eff'
        panel_label: '(c)' or '(d)'
    """
    font_settings = settings.fonts
    err_settings = settings.error_bars

    # Organize by group and population
    data_by_category = {}

    for metrics_2d, metrics_3d, sample_name in all_metrics:
        group, population = extract_group_info(sample_name)
        category = (group, population)

        if category not in data_by_category:
            data_by_category[category] = {'x': [], 'y': [], 'yerr_lo': [], 'yerr_hi': []}

        data_by_category[category]['x'].append(metrics_3d[metric])
        data_by_category[category]['y'].append(metrics_2d[metric])
        data_by_category[category]['yerr_lo'].append(
            metrics_2d[metric] - metrics_2d[f'{metric}_lo'])
        data_by_category[category]['yerr_hi'].append(
            metrics_2d[f'{metric}_hi'] - metrics_2d[metric])

    # Collect all values for axis limits
    all_x, all_y = [], []

    # Plot each category
    for (group, population), data in sorted(data_by_category.items()):
        color = settings.get_group_color(group)
        marker = settings.get_marker(population)

        label = f"{group} - {population}" if population else group

        yerr = [data['yerr_lo'], data['yerr_hi']]
        ax.errorbar(
            data['x'], data['y'], yerr=yerr,
            fmt=marker, color=color, markersize=7,
            capsize=err_settings['capsize'], capthick=err_settings['capthick'],
            elinewidth=err_settings['linewidth'],
            markeredgecolor='black', markeredgewidth=0.5,
            alpha=err_settings['alpha'], label=label
        )
        all_x.extend(data['x'])
        all_y.extend(data['y'])

    # Check for sufficient data
    if len(all_x) < 2:
        logger.warning(f"Insufficient data points ({len(all_x)}) for correlation")
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        return

    # Correlation
    r, p = stats.pearsonr(all_x, all_y)

    # Identity line
    all_vals = all_x + all_y
    min_val = min(all_vals) * 0.95
    max_val = max(all_vals) * 1.05
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5,
            linewidth=1.5, zorder=0)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_box_aspect(1)  # Square subplot box

    if metric == 'mean_radius':
        xlabel = r'3D $\bar{r}$ [μm]'
        ylabel = r'2D $\bar{r}$ [μm]'
    else:
        xlabel = r'3D $r_{\mathrm{MRI}}$ [μm]'
        ylabel = r'2D $r_{\mathrm{MRI}}$ [μm]'

    ax.set_xlabel(xlabel, fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.set_ylabel(ylabel, fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.legend(loc='upper left', fontsize=font_settings['legend_size'] - 1, framealpha=0.9)

    # Add correlation as text annotation
    if p < 0.001:
        p_str = f'p < 0.001'
    else:
        p_str = f'p = {p:.3f}'
    ax.text(0.95, 0.05, f'$R$ = {r:.3f}, {p_str}', transform=ax.transAxes,
            fontsize=font_settings['legend_size'], ha='right', va='bottom')


def main():
    parser = argparse.ArgumentParser(
        description='Compare 2D vs 3D radius distributions with PDF and QQ analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data-dir', type=Path, default=Path('data/processed/LM'),
                        help='Directory containing LM slice and axon profiles')
    parser.add_argument('--output', type=Path, default=Path('fig/distribution_2d_vs_3d_comparison.svg'),
                        help='Output figure path')
    parser.add_argument('--radius-type', type=str, default='circular',
                        choices=['circular', 'minor'],
                        help='Radius type to use')
    parser.add_argument('--population', type=str, default='cc',
                        choices=['cc', 'cg'],
                        help='Population for representative sample')
    parser.add_argument('--x-max', type=float, default=1.5,
                        help='Maximum x-axis value for PDF plot')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("2D vs 3D Distribution Comparison")
    logger.info("=" * 80)

    # Find samples with small, medium, high mean radius for panel (a)
    pdf_samples = find_samples_by_mean_radius(args.data_dir)

    # Find largest sample for panel (b) QQ-plot
    slice_file, axon_file, sample_name = find_largest_sample(
        args.data_dir, args.population)

    # Load population labels for the QQ representative sample
    base_name = slice_file.stem.replace(f'_{args.population}_slice_profiles', '')
    pop_labels = load_population_labels(args.data_dir, base_name, args.population)
    logger.info(f"Population '{args.population.upper()}' has {len(pop_labels)} axons")

    # Find all pairs for panels (c) and (d)
    all_pairs = find_matching_pairs(args.data_dir)
    logger.info(f"Found {len(all_pairs)} sample-population pairs for ensemble analysis")

    # Load metrics for all pairs
    all_metrics = []
    for sf, af, sn, labels in all_pairs:
        m2d = load_2d_metrics(sf, args.radius_type)
        m3d = load_3d_metrics(af, labels)
        all_metrics.append((m2d, m3d, sn))

    # Create figure (2x2 layout)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # Panel (a): PDF stability with 3 samples (small, medium, high mean radius)
    logger.info("\nPlotting panel (a): PDF stability (3 samples)...")
    plot_pdf_stability_multi(axes[0, 0], pdf_samples, args.data_dir, args.radius_type, args.x_max)

    # Panel (b): QQ-plot (using largest sample)
    logger.info("\nPlotting panel (b): QQ-plot...")
    plot_qq_with_variability(axes[0, 1], slice_file, axon_file, pop_labels,
                             args.radius_type)

    # Panel (c): Mean radius scatter
    logger.info("\nPlotting panel (c): Mean radius scatter...")
    plot_ensemble_scatter(axes[1, 0], all_metrics, 'mean_radius', '(c)')

    # Panel (d): Effective radius scatter
    logger.info("\nPlotting panel (d): Effective radius scatter...")
    plot_ensemble_scatter(axes[1, 1], all_metrics, 'r_eff', '(d)')

    plt.subplots_adjust(wspace=0.25, hspace=0.25)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"\nSaved figure to {args.output}")

    # Save metadata
    metadata = {
        'representative_sample': sample_name,
        'n_population_axons': len(pop_labels),
        'n_ensemble_samples': len(all_metrics),
        'radius_type': args.radius_type,
        'x_max_pdf': args.x_max
    }
    json_output = args.output.with_suffix('.json')
    with open(json_output, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {json_output}")


if __name__ == '__main__':
    main()
