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

    # Pick small, medium, high - evenly spaced across the range
    n = len(results)
    small_idx = 1  # avoid very smallest
    high_idx = n - 2  # avoid very largest

    # Find medium closest to midpoint of small and high mean radii
    small_mean_r = results[small_idx][3]
    high_mean_r = results[high_idx][3]
    target_mid = (small_mean_r + high_mean_r) / 2

    # Find index with mean_r closest to target
    med_idx = min(range(small_idx + 1, high_idx),
                  key=lambda i: abs(results[i][3] - target_mid))

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

    ax.fill_between(x, y_lo, y_hi, alpha=0.3, color=settings.colors['category_a'], label='IQR')
    ax.plot(x, y_med, color=settings.colors['category_a'], linewidth=line_settings['linewidth'], label='Median')

    ax.set_xlabel('Axon radius [μm]', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.set_ylabel('Probability density [μm⁻¹]', fontsize=font_settings['label_size'],
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
    Plot PDF comparison: 3D (solid) vs 2D (IQR) for sample(s).
    Shows r̄ and r_MRI with median and IQR across cross-sections.

    Args:
        ax: Matplotlib axes
        samples: List of (slice_file, axon_file, sample_name, mean_radius) tuples
        data_dir: Directory containing population JSON files
        radius_type: 'circular' or 'minor'
        x_max: Maximum x-axis value for cropping
    """
    font_settings = settings.fonts
    line_settings = settings.line

    # Colors - gray for PDF, representative colors for metrics
    pdf_color = settings.colors['single_line']    # Gray for PDF
    r_arith_color = settings.colors['example_2']  # Orange for r̄
    r_eff_color = settings.colors['example_3']    # Purple for r_MRI

    for idx, (slice_file, axon_file, sample_name, mean_r) in enumerate(samples):
        # Load 2D slice data
        npz_2d = np.load(slice_file)
        bin_centers = npz_2d['bin_centers']
        histograms = npz_2d[f'histograms_{radius_type}']
        r_eff_per_slice = npz_2d[f'r_eff_{radius_type}_per_slice']

        n_slices, n_bins = histograms.shape

        # Convert each slice histogram to PDF and compute per-slice metrics
        pdfs = np.zeros((n_slices, n_bins))
        valid_slices = []
        r_arith_per_slice = []

        for i in range(n_slices):
            counts = histograms[i].sum()
            if counts > MIN_AXON_COUNT:
                pdfs[i] = histogram_to_pdf(histograms[i], bin_centers)
                valid_slices.append(i)
                # Compute arithmetic mean for this slice
                r_arith = np.sum(bin_centers * histograms[i]) / counts
                r_arith_per_slice.append(r_arith)

        if len(valid_slices) == 0:
            logger.warning(f"No valid slices for {sample_name}")
            continue

        pdfs_valid = pdfs[valid_slices]
        r_arith_per_slice = np.array(r_arith_per_slice)
        r_eff_valid = r_eff_per_slice[valid_slices]
        r_eff_valid = r_eff_valid[r_eff_valid > 0]  # Filter invalid

        # Compute 2D PDF IQR
        pdf_2d_lo = np.percentile(pdfs_valid, 25, axis=0)
        pdf_2d_hi = np.percentile(pdfs_valid, 75, axis=0)

        # Load 3D axon data and filter by population
        population = sample_name.split('_')[-1].lower()
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
        y_2d_lo = pdf_2d_lo[mask]
        y_2d_hi = pdf_2d_hi[mask]
        y_3d = pdf_3d[mask]

        # Plot PDF: 3D solid line + 2D IQR shading
        ax.plot(x, y_3d, color=pdf_color, linewidth=line_settings['linewidth'],
                linestyle='-')
        ax.fill_between(x, y_2d_lo, y_2d_hi, alpha=0.3, color=pdf_color)

        # Compute median and IQR for r̄ and r_MRI across slices
        r_arith_median = np.median(r_arith_per_slice)
        r_arith_lo = np.percentile(r_arith_per_slice, 25)
        r_arith_hi = np.percentile(r_arith_per_slice, 75)

        r_eff_median = np.median(r_eff_valid)
        r_eff_lo = np.percentile(r_eff_valid, 25)
        r_eff_hi = np.percentile(r_eff_valid, 75)

        logger.info(f"  r̄: median={r_arith_median:.3f}, IQR=[{r_arith_lo:.3f}, {r_arith_hi:.3f}] (width={r_arith_hi-r_arith_lo:.3f})")
        logger.info(f"  r_MRI: median={r_eff_median:.3f}, IQR=[{r_eff_lo:.3f}, {r_eff_hi:.3f}] (width={r_eff_hi-r_eff_lo:.3f})")

        # Get y-axis range for vertical spans
        y_max = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else np.max(y_3d) * 1.1

        # Plot r̄: IQR as shaded region + median line (thin line)
        ax.axvspan(r_arith_lo, r_arith_hi, alpha=0.3, color=r_arith_color, zorder=0)
        ax.axvline(r_arith_median, color=r_arith_color, linestyle='-', linewidth=1.2, alpha=0.9)

        # Plot r_MRI: IQR as shaded region + median line (thin line)
        ax.axvspan(r_eff_lo, r_eff_hi, alpha=0.3, color=r_eff_color, zorder=0)
        ax.axvline(r_eff_median, color=r_eff_color, linestyle='-', linewidth=1.2, alpha=0.9)

    # Build custom legend (compact: 3 entries with line + shaded background)
    from matplotlib.patches import Patch, FancyBboxPatch
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerBase

    # Custom handler to draw line over shaded rectangle
    class HandlerLineWithBackground(HandlerBase):
        def __init__(self, line_color, bg_color, bg_alpha=0.3):
            self.line_color = line_color
            self.bg_color = bg_color
            self.bg_alpha = bg_alpha
            super().__init__()

        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                          width, height, fontsize, trans):
            # Background rectangle
            rect = plt.Rectangle([xdescent, ydescent], width, height,
                                  facecolor=self.bg_color, alpha=self.bg_alpha,
                                  edgecolor='none', transform=trans)
            # Line in center
            line = plt.Line2D([xdescent, xdescent + width],
                              [ydescent + height/2, ydescent + height/2],
                              color=self.line_color, linewidth=1.5, transform=trans)
            return [rect, line]

    handles = []
    labels = []
    handler_map = {}

    # PDF: line + shaded patch
    pdf_dummy = Line2D([0], [0])
    handles.append(pdf_dummy)
    labels.append('PDF (3D ref. + 2D IQR)')
    handler_map[pdf_dummy] = HandlerLineWithBackground(pdf_color, pdf_color, 0.3)

    # r̄: line + shaded background
    r_arith_dummy = Line2D([0], [0])
    handles.append(r_arith_dummy)
    labels.append(r'$\bar{r}$ (median + IQR)')
    handler_map[r_arith_dummy] = HandlerLineWithBackground(r_arith_color, r_arith_color, 0.3)

    # r_MRI: line + shaded background
    r_eff_dummy = Line2D([0], [0])
    handles.append(r_eff_dummy)
    labels.append(r'$r_{\mathrm{MRI}}$ (median + IQR)')
    handler_map[r_eff_dummy] = HandlerLineWithBackground(r_eff_color, r_eff_color, 0.3)

    ax.set_xlabel('Axon radius [μm]', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.set_ylabel('Probability density [μm⁻¹]', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.legend(handles, labels, loc='upper right', fontsize=font_settings['legend_size'] - 1,
              framealpha=0.9, handler_map=handler_map)
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_box_aspect(1)  # Square subplot box


def compute_ks_and_wasserstein_per_slice(slice_file: Path, axon_file: Path,
                                          population_labels: Set[int],
                                          radius_type: str = 'circular'
                                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute KS and Wasserstein distances between each 2D slice and 3D ground truth.

    Returns:
        Tuple of (ks_values, wasserstein_values) arrays (one per valid slice)
    """
    # Load 2D slice data
    npz_2d = np.load(slice_file)
    bin_centers = npz_2d['bin_centers']
    histograms = npz_2d[f'histograms_{radius_type}']
    bin_width = bin_centers[1] - bin_centers[0]

    # Load 3D data and filter by population
    npz_3d = np.load(axon_file, allow_pickle=True)
    axon_labels = npz_3d['labels']
    radii_profiles = npz_3d['radii_profiles_um']

    # Collect 3D radii for this population
    all_radii_3d = []
    for i, label in enumerate(axon_labels):
        if int(label) in population_labels:
            all_radii_3d.extend(radii_profiles[i])
    all_radii_3d = np.array(all_radii_3d)

    # Compute 3D CDF at bin centers
    sorted_3d = np.sort(all_radii_3d)
    cdf_3d_y = np.arange(1, len(sorted_3d) + 1) / len(sorted_3d)
    cdf_3d = np.interp(bin_centers, sorted_3d, cdf_3d_y, left=0, right=1)

    # Compute distances for each valid slice
    ks_values = []
    wasserstein_values = []

    for i in range(histograms.shape[0]):
        if histograms[i].sum() > MIN_AXON_COUNT:
            # Compute 2D CDF
            cdf_2d = np.cumsum(histograms[i]) / histograms[i].sum()

            # KS statistic = max |CDF_2D - CDF_3D|
            ks = np.max(np.abs(cdf_2d - cdf_3d))
            ks_values.append(ks)

            # Wasserstein distance = integral of |CDF_2D - CDF_3D|
            # For discrete CDFs, this is sum of |CDF differences| * bin_width
            wasserstein = np.sum(np.abs(cdf_2d - cdf_3d)) * bin_width
            wasserstein_values.append(wasserstein)

    return np.array(ks_values), np.array(wasserstein_values)


def compute_3d_cdf(axon_file: Path, population_labels: Set[int],
                    bin_centers: np.ndarray) -> np.ndarray:
    """Compute 3D CDF at given bin centers for a population."""
    npz_3d = np.load(axon_file, allow_pickle=True)
    axon_labels = npz_3d['labels']
    radii_profiles = npz_3d['radii_profiles_um']

    all_radii = []
    for i, label in enumerate(axon_labels):
        if int(label) in population_labels:
            all_radii.extend(radii_profiles[i])
    all_radii = np.array(all_radii)

    sorted_radii = np.sort(all_radii)
    cdf_y = np.arange(1, len(sorted_radii) + 1) / len(sorted_radii)
    return np.interp(bin_centers, sorted_radii, cdf_y, left=0, right=1)


def plot_within_vs_between_distances(ax, all_pairs: List[Tuple[Path, Path, str, Set[int]]],
                                      radius_type: str = 'circular') -> None:
    """
    Plot within-ROI vs between-ROI Wasserstein distances.

    Left: 2D slice vs 3D ground truth (within-ROI sampling variation)
    Right: ROI vs ROI (between-ROI biological variation)

    Args:
        ax: Matplotlib axes
        all_pairs: List of (slice_file, axon_file, sample_name, population_labels) tuples
        radius_type: 'circular' or 'minor'
    """
    font_settings = settings.fonts

    # Get common bin centers from first file
    first_npz = np.load(all_pairs[0][0])
    bin_centers = first_npz['bin_centers']
    bin_width = bin_centers[1] - bin_centers[0]

    # Collect within-ROI distances (all 2D slices vs their 3D ground truth)
    within_distances = []
    roi_cdfs = []  # Store 3D CDFs for between-ROI comparison

    for slice_file, axon_file, sample_name, pop_labels in all_pairs:
        _, wasserstein_values = compute_ks_and_wasserstein_per_slice(
            slice_file, axon_file, pop_labels, radius_type)
        within_distances.extend(wasserstein_values)

        # Store 3D CDF for this ROI
        cdf_3d = compute_3d_cdf(axon_file, pop_labels, bin_centers)
        roi_cdfs.append(cdf_3d)

    # Compute between-ROI distances (pairwise comparisons of 3D distributions)
    between_distances = []
    n_rois = len(roi_cdfs)
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            # Wasserstein = integral of |CDF_i - CDF_j|
            w_dist = np.sum(np.abs(roi_cdfs[i] - roi_cdfs[j])) * bin_width
            between_distances.append(w_dist)

    within_distances = np.array(within_distances)
    between_distances = np.array(between_distances)

    # Create violin plots
    parts = ax.violinplot([within_distances, between_distances], positions=[1, 2],
                           showmeans=False, showmedians=True, widths=0.7)

    # Color the violins (binary comparison colors)
    colors = [settings.colors['binary_a'], settings.colors['binary_b']]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.8)

    # Style median lines
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)
    parts['cbars'].set_color('gray')
    parts['cmins'].set_color('gray')
    parts['cmaxes'].set_color('gray')

    # Add individual points with jitter for within (subsample if too many)
    np.random.seed(42)
    if len(within_distances) > 500:
        idx = np.random.choice(len(within_distances), 500, replace=False)
        within_sample = within_distances[idx]
    else:
        within_sample = within_distances
    jitter = np.random.uniform(-0.15, 0.15, len(within_sample))
    ax.scatter(1 + jitter, within_sample, alpha=0.4, s=5, color='#808080', zorder=0)

    # Add points for between
    jitter = np.random.uniform(-0.15, 0.15, len(between_distances))
    ax.scatter(2 + jitter, between_distances, alpha=0.5, s=10, color='#404040', zorder=0)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Sampling error\n(intra-ROI 2D ↔ 3D)', 'Anat. variability\n(inter-ROI 3D)'],
                        fontsize=font_settings['tick_size'] - 1)
    ax.set_ylabel('Wasserstein distance [μm]', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.tick_params(axis='y', labelsize=font_settings['tick_size'])
    ax.set_ylim(bottom=0)
    ax.set_box_aspect(1)


def compute_2d_pooled_cdf(slice_file: Path, bin_centers: np.ndarray,
                           radius_type: str = 'circular') -> Optional[np.ndarray]:
    """Compute pooled 2D CDF from all slices. Returns None if no valid slices."""
    npz_2d = np.load(slice_file)
    histograms = npz_2d[f'histograms_{radius_type}']

    # Pool all valid slices
    pooled_hist = np.zeros(len(bin_centers))
    for i in range(histograms.shape[0]):
        if histograms[i].sum() > MIN_AXON_COUNT:
            pooled_hist += histograms[i]

    if pooled_hist.sum() == 0:
        return None  # No valid slices

    return np.cumsum(pooled_hist) / pooled_hist.sum()


def plot_discriminability_scatter(ax, all_pairs: List[Tuple[Path, Path, str, Set[int]]],
                                   radius_type: str = 'circular') -> None:
    """
    Plot W_2D vs W_3D for all ROI pairs (discriminability test).

    Shows whether 2D preserves the relative differences between ROIs.

    Args:
        ax: Matplotlib axes
        all_pairs: List of (slice_file, axon_file, sample_name, population_labels) tuples
        radius_type: 'circular' or 'minor'
    """
    font_settings = settings.fonts

    # Get common bin centers from first file
    first_npz = np.load(all_pairs[0][0])
    bin_centers = first_npz['bin_centers']
    bin_width = bin_centers[1] - bin_centers[0]

    # Compute CDFs for all ROIs (both 2D and 3D)
    roi_cdfs_3d = []
    roi_cdfs_2d = []
    roi_names = []

    for slice_file, axon_file, sample_name, pop_labels in all_pairs:
        # 2D pooled CDF (check first to skip invalid ROIs)
        cdf_2d = compute_2d_pooled_cdf(slice_file, bin_centers, radius_type)
        if cdf_2d is None:
            logger.warning(f"Skipping {sample_name} - no valid 2D slices")
            continue

        # 3D CDF
        cdf_3d = compute_3d_cdf(axon_file, pop_labels, bin_centers)

        roi_cdfs_3d.append(cdf_3d)
        roi_cdfs_2d.append(cdf_2d)
        roi_names.append(sample_name)

    # Compute pairwise Wasserstein distances
    w_3d_pairs = []
    w_2d_pairs = []

    n_rois = len(roi_cdfs_3d)
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            # 3D distance
            w_3d = np.sum(np.abs(roi_cdfs_3d[i] - roi_cdfs_3d[j])) * bin_width
            w_3d_pairs.append(w_3d)

            # 2D distance
            w_2d = np.sum(np.abs(roi_cdfs_2d[i] - roi_cdfs_2d[j])) * bin_width
            w_2d_pairs.append(w_2d)

    w_3d_pairs = np.array(w_3d_pairs)
    w_2d_pairs = np.array(w_2d_pairs)

    # Scatter plot (single category - use single_line color)
    ax.scatter(w_3d_pairs, w_2d_pairs, alpha=0.6, s=20, color=settings.colors['single_line'],
               edgecolor='black', linewidth=0.3)

    # Identity line
    max_val = max(w_3d_pairs.max(), w_2d_pairs.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=1.5, zorder=0)

    # Correlation
    r, p = stats.pearsonr(w_3d_pairs, w_2d_pairs)

    # Linear regression for slope
    slope, intercept = np.polyfit(w_3d_pairs, w_2d_pairs, 1)

    ax.set_xlabel('3D Wasserstein distance [μm]', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.set_ylabel('2D Wasserstein distance [μm]', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.tick_params(labelsize=font_settings['tick_size'])
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_box_aspect(1)

    # Stats annotation
    if p < 0.001:
        p_str = 'p < 0.001'
    else:
        p_str = f'p = {p:.3f}'
    stats_text = f'R = {r:.3f}, {p_str}\nSlope = {slope:.2f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=font_settings['legend_size'], ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


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


def load_2d_slice_data(npz_file: Path, radius_type: str = 'circular',
                       min_mean_axons_per_slice: int = 1000) -> Dict[str, np.ndarray]:
    """
    Load per-slice 2D metrics for Monte Carlo sampling.

    Returns dict with arrays of per-slice values (only valid slices).
    Filters out edge slices with < half the mean axon count.
    Returns empty if mean axons per slice < min_mean_axons_per_slice.
    """
    data = np.load(npz_file)
    bin_centers = data['bin_centers']
    histograms = data[f'histograms_{radius_type}']
    r_eff_per_slice = data[f'r_eff_{radius_type}_per_slice']

    # First pass: compute mean axon count per slice
    all_counts = []
    for hist_slice in histograms:
        counts = hist_slice.sum()
        if counts > 0:
            all_counts.append(counts)

    mean_axon_count = np.mean(all_counts) if all_counts else 0

    # Skip ROIs with insufficient mean axons per slice
    if mean_axon_count < min_mean_axons_per_slice:
        return {
            'mean_radius_per_slice': np.array([]),
            'r_eff_per_slice': np.array([]),
            'n_valid_slices': 0,
            'mean_axons_per_slice': mean_axon_count
        }

    min_axon_count = mean_axon_count / 2  # Filter slices below half the mean

    # Second pass: filter slices and compute metrics
    mean_radius_per_slice = []
    valid_r_eff = []
    valid_indices = []

    for i, hist_slice in enumerate(histograms):
        counts = hist_slice.sum()
        if counts >= min_axon_count and r_eff_per_slice[i] > 0:
            mean_r = np.sum(bin_centers * hist_slice) / counts
            mean_radius_per_slice.append(mean_r)
            valid_r_eff.append(r_eff_per_slice[i])
            valid_indices.append(i)

    return {
        'mean_radius_per_slice': np.array(mean_radius_per_slice),
        'r_eff_per_slice': np.array(valid_r_eff),
        'n_valid_slices': len(valid_indices),
        'mean_axons_per_slice': mean_axon_count
    }


def run_monte_carlo_correlation(
    all_pairs: List[Tuple[Path, Path, str, Set[int]]],
    radius_type: str = 'circular',
    n_iterations: int = 10000,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Run Monte Carlo simulation to compute correlation between 2D and 3D metrics.

    For each iteration:
    - Randomly pick one slice per ROI
    - Compute correlation R between 2D sample and 3D ground truth

    Args:
        all_pairs: List of (slice_file, axon_file, sample_name, population_labels) tuples
        radius_type: 'circular' or 'minor'
        n_iterations: Number of MC iterations
        seed: Random seed for reproducibility

    Returns:
        Dict with mean and std of R for mean_radius and r_eff
    """
    rng = np.random.default_rng(seed)

    # Load all data
    roi_slice_data = []
    x_3d_mean = []
    x_3d_reff = []

    for slice_file, axon_file, sample_name, pop_labels in all_pairs:
        # Load 2D slice data (filters ROIs with < 1000 mean axons per slice)
        slice_data = load_2d_slice_data(slice_file, radius_type, min_mean_axons_per_slice=1000)
        if slice_data['n_valid_slices'] < 1:
            mean_axons = slice_data.get('mean_axons_per_slice', 0)
            logger.warning(f"Skipping {sample_name} - mean axons/slice too low ({mean_axons:.0f})")
            continue

        # Load 3D metrics
        m3d = load_3d_metrics(axon_file, pop_labels)
        if np.isnan(m3d['mean_radius']):
            continue

        roi_slice_data.append(slice_data)
        x_3d_mean.append(m3d['mean_radius'])
        x_3d_reff.append(m3d['r_eff'])

    x_3d_mean = np.array(x_3d_mean)
    x_3d_reff = np.array(x_3d_reff)
    n_rois = len(roi_slice_data)

    logger.info(f"Running Monte Carlo with {n_rois} ROIs, {n_iterations} iterations...")

    # Run MC iterations
    r_mean_per_iter = []
    r_reff_per_iter = []

    for _ in range(n_iterations):
        y_2d_mean = np.zeros(n_rois)
        y_2d_reff = np.zeros(n_rois)

        for j, roi in enumerate(roi_slice_data):
            # Randomly pick one slice
            slice_idx = rng.integers(0, roi['n_valid_slices'])
            y_2d_mean[j] = roi['mean_radius_per_slice'][slice_idx]
            y_2d_reff[j] = roi['r_eff_per_slice'][slice_idx]

        # Compute correlations
        r_mean, _ = stats.pearsonr(x_3d_mean, y_2d_mean)
        r_reff, _ = stats.pearsonr(x_3d_reff, y_2d_reff)

        r_mean_per_iter.append(r_mean)
        r_reff_per_iter.append(r_reff)

    r_mean_per_iter = np.array(r_mean_per_iter)
    r_reff_per_iter = np.array(r_reff_per_iter)

    # Compute observed R (mean across iterations)
    r_mean_observed = np.mean(r_mean_per_iter)
    r_reff_observed = np.mean(r_reff_per_iter)

    # Permutation test for p-value
    # Null hypothesis: no correlation between 2D and 3D
    # Shuffle 3D values and compute R, count fraction >= observed
    n_perm = n_iterations
    rng_perm = np.random.default_rng(seed + 100)

    r_mean_perm = []
    r_reff_perm = []

    for _ in range(n_perm):
        # Shuffle 3D values
        perm_idx = rng_perm.permutation(n_rois)
        x_3d_mean_perm = x_3d_mean[perm_idx]
        x_3d_reff_perm = x_3d_reff[perm_idx]

        # Pick random slices (same as observed)
        y_2d_mean = np.zeros(n_rois)
        y_2d_reff = np.zeros(n_rois)
        for j, roi in enumerate(roi_slice_data):
            slice_idx = rng_perm.integers(0, roi['n_valid_slices'])
            y_2d_mean[j] = roi['mean_radius_per_slice'][slice_idx]
            y_2d_reff[j] = roi['r_eff_per_slice'][slice_idx]

        r_perm, _ = stats.pearsonr(x_3d_mean_perm, y_2d_mean)
        r_mean_perm.append(r_perm)

        r_perm, _ = stats.pearsonr(x_3d_reff_perm, y_2d_reff)
        r_reff_perm.append(r_perm)

    r_mean_perm = np.array(r_mean_perm)
    r_reff_perm = np.array(r_reff_perm)

    # p-value: fraction of permuted R >= observed R
    p_mean = np.mean(r_mean_perm >= r_mean_observed)
    p_reff = np.mean(r_reff_perm >= r_reff_observed)

    logger.info(f"  Permutation p-values: mean_radius p={p_mean:.4f}, r_eff p={p_reff:.4f}")

    return {
        'mean_radius': {
            'r_mean': r_mean_observed,
            'r_std': np.std(r_mean_per_iter),
            'p_value': p_mean
        },
        'r_eff': {
            'r_mean': r_reff_observed,
            'r_std': np.std(r_reff_per_iter),
            'p_value': p_reff
        },
        'n_iterations': n_iterations,
        'n_rois': n_rois,
        'seed': seed
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
                          metric: str, panel_label: str,
                          mc_results: Optional[Dict] = None) -> None:
    """
    Plot 2D vs 3D scatter for ensemble metrics.

    Args:
        ax: Matplotlib axes
        all_metrics: List of (metrics_2d, metrics_3d, sample_name) tuples
        metric: 'mean_radius' or 'r_eff'
        panel_label: '(c)' or '(d)'
        mc_results: Optional Monte Carlo results dict with r_mean and r_std
    """
    font_settings = settings.fonts
    err_settings = settings.error_bars

    # Collect all data (no differentiation by group/population)
    all_x, all_y, all_yerr_lo, all_yerr_hi = [], [], [], []

    for metrics_2d, metrics_3d, sample_name in all_metrics:
        all_x.append(metrics_3d[metric])
        all_y.append(metrics_2d[metric])
        all_yerr_lo.append(metrics_2d[metric] - metrics_2d[f'{metric}_lo'])
        all_yerr_hi.append(metrics_2d[f'{metric}_hi'] - metrics_2d[metric])

    # Use single default color
    color = settings.colors['single_line']

    yerr = [all_yerr_lo, all_yerr_hi]
    ax.errorbar(
        all_x, all_y, yerr=yerr,
        fmt='o', color=color, markersize=7,
        capsize=err_settings['capsize'], capthick=err_settings['capthick'],
        elinewidth=err_settings['linewidth'],
        markeredgecolor='black', markeredgewidth=0.5,
        alpha=err_settings['alpha']
    )

    # Check for sufficient data
    if len(all_x) < 2:
        logger.warning(f"Insufficient data points ({len(all_x)}) for correlation")
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        return

    # Compute axis limits
    all_vals = all_x + all_y
    min_val = min(all_vals) * 0.95
    max_val = max(all_vals) * 1.05

    if metric == 'mean_radius':
        xlabel = r'$\bar{r}$ (3D) [μm]'
        ylabel = r'$\bar{r}$ (2D) [μm]'
        # Set matching ticks with 0.05 step (within data range)
        tick_step = 0.05
        tick_start = np.ceil(min_val / tick_step) * tick_step
        tick_end = np.floor(max_val / tick_step) * tick_step
        ticks = np.arange(tick_start, tick_end + tick_step/2, tick_step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    else:
        xlabel = r'$r_{\mathrm{MRI}}$ (3D) [μm]'
        ylabel = r'$r_{\mathrm{MRI}}$ (2D) [μm]'

    # Identity line (spans full axis range)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5,
            linewidth=1.5, zorder=0)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_box_aspect(1)  # Square subplot box

    ax.set_xlabel(xlabel, fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax.set_ylabel(ylabel, fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])

    # Add Monte Carlo correlation as text annotation
    if mc_results is not None and metric in mc_results:
        r_mean = mc_results[metric]['r_mean']
        r_std = mc_results[metric]['r_std']
        p_val = mc_results[metric].get('p_value', None)
        if p_val is not None:
            if p_val < 0.001:
                p_str = 'p < 0.001'
            elif p_val == 0:
                p_str = f'p < {1/mc_results["n_iterations"]:.0e}'
            else:
                p_str = f'p = {p_val:.3f}'
            ax.text(0.95, 0.05, f'$R$ = {r_mean:.2f} ± {r_std:.2f}\n{p_str}',
                    transform=ax.transAxes, fontsize=font_settings['legend_size'],
                    ha='right', va='bottom')
        else:
            ax.text(0.95, 0.05, f'$R$ = {r_mean:.2f} ± {r_std:.2f}', transform=ax.transAxes,
                    fontsize=font_settings['legend_size'], ha='right', va='bottom')
    else:
        # Fallback to single correlation from median values
        r, p = stats.pearsonr(all_x, all_y)
        if p < 0.001:
            p_str = f'p < 0.001'
        else:
            p_str = f'p = {p:.3f}'
        ax.text(0.95, 0.05, f'$R$ = {r:.2f}, {p_str}', transform=ax.transAxes,
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
    parser.add_argument('--x-max', type=float, default=1.5,
                        help='Maximum x-axis value for PDF plot')
    parser.add_argument('--n-iterations', type=int, default=10000,
                        help='Number of Monte Carlo iterations for R estimation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for Monte Carlo (sync with supplementary)')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("2D vs 3D Distribution Comparison")
    logger.info("=" * 80)

    # Use sham_25_ipsi_CG - has highest IQR for both r̄ and r_MRI
    slice_file = args.data_dir / "sham_25_ipsi_cg_slice_profiles.npz"
    axon_file = args.data_dir / "sham_25_ipsi_axon_profiles.npz"
    sample_name = "sham_25_ipsi_CG"
    # Load mean_r for the tuple
    npz = np.load(slice_file)
    bin_centers = npz['bin_centers']
    total_hist = npz['total_histogram_circular']
    mean_r = np.sum(bin_centers * total_hist) / total_hist.sum()
    representative_sample = (slice_file, axon_file, sample_name, mean_r)
    logger.info(f"Using representative sample: {sample_name} (r̄={mean_r:.3f})")

    # Find all pairs for panels (b), (c) and (d)
    all_pairs = find_matching_pairs(args.data_dir)
    logger.info(f"Found {len(all_pairs)} sample-population pairs for ensemble analysis")

    # Load metrics for all pairs
    all_metrics = []
    for sf, af, sn, labels in all_pairs:
        m2d = load_2d_metrics(sf, args.radius_type)
        m3d = load_3d_metrics(af, labels)
        all_metrics.append((m2d, m3d, sn))

    # Run Monte Carlo to compute R with uncertainty
    logger.info(f"\nRunning Monte Carlo ({args.n_iterations} iterations, seed={args.seed})...")
    mc_results = run_monte_carlo_correlation(
        all_pairs, args.radius_type, args.n_iterations, args.seed)

    logger.info(f"  Mean radius: R = {mc_results['mean_radius']['r_mean']:.3f} ± "
                f"{mc_results['mean_radius']['r_std']:.3f}")
    logger.info(f"  Effective radius: R = {mc_results['r_eff']['r_mean']:.3f} ± "
                f"{mc_results['r_eff']['r_std']:.3f}")

    # Create figure (2x2 layout)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # Panel (a): PDF stability with single representative sample
    logger.info("\nPlotting panel (a): PDF stability (single sample)...")
    plot_pdf_stability_multi(axes[0, 0], [representative_sample], args.data_dir, args.radius_type, args.x_max)

    # Panel (b): Within-ROI vs Between-ROI Wasserstein distances
    logger.info("\nPlotting panel (b): Within vs Between ROI distances...")
    plot_within_vs_between_distances(axes[0, 1], all_pairs, args.radius_type)

    # Panel (c): Mean radius scatter
    logger.info("\nPlotting panel (c): Mean radius scatter...")
    plot_ensemble_scatter(axes[1, 0], all_metrics, 'mean_radius', '(c)', mc_results)

    # Panel (d): Effective radius scatter
    logger.info("\nPlotting panel (d): Effective radius scatter...")
    plot_ensemble_scatter(axes[1, 1], all_metrics, 'r_eff', '(d)', mc_results)

    plt.subplots_adjust(wspace=0.25, hspace=0.25)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=settings.figure['dpi'], bbox_inches='tight')

    # Also save SVG version
    svg_file = args.output.with_suffix('.svg')
    plt.savefig(svg_file, bbox_inches='tight')

    plt.close()

    logger.info(f"\nSaved figure to {args.output} and {svg_file}")

    # Save metadata
    metadata = {
        'n_ensemble_samples': len(all_metrics),
        'n_qq_rois': len(all_pairs),
        'radius_type': args.radius_type,
        'x_max_pdf': args.x_max,
        'monte_carlo': {
            'n_iterations': mc_results['n_iterations'],
            'n_rois': mc_results['n_rois'],
            'seed': mc_results['seed'],
            'mean_radius': mc_results['mean_radius'],
            'r_eff': mc_results['r_eff']
        }
    }
    json_output = args.output.with_suffix('.json')
    with open(json_output, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {json_output}")


if __name__ == '__main__':
    main()
