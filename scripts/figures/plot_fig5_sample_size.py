"""
Study the effect of sample size on radius estimation accuracy.

Creates a 2×2 figure:
  (a) PDFs for Human CC and Rat WM pooled distributions
  (b) Within-sample vs between-ROI Wasserstein distances
  (c) Arithmetic mean radius error vs sample size (both datasets)
  (d) Effective MRI radius error vs sample size (both datasets)

Usage:
    python scripts/figures/plot_fig5_sample_size.py \
        --human-data data/raw/human/lm \
        --rat-data data/processed/rat/lm \
        --output fig/main/sample_size_effect.svg
"""

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Find repo root (contains pyproject.toml)
_root = Path(__file__).resolve().parent
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from axonometry import (compute_r_arith, compute_r_eff, get_plot_settings,
                        rediscretize)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings = get_plot_settings()

# Constants
DEFAULT_BIN_WIDTH = 0.05  # μm
SAMPLE_SIZES = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
N_SUBSAMPLES = 1000

SAMPLE_SIZE_LABELS = {
    100: r'$10^2$',
    1_000: r'$10^3$',
    10_000: r'$10^4$',
    100_000: r'$10^5$',
    1_000_000: r'$10^6$',
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SubsampleResults:
    """Results from subsampling analysis for one sample size."""
    sample_size: int
    n_valid_rois: int
    # Radius estimates: (n_rois, n_subsamples)
    r_arith: np.ndarray
    r_eff: np.ndarray
    # Reference values per ROI: (n_rois,)
    reference_r_arith: np.ndarray
    reference_r_eff: np.ndarray


@dataclass
class DatasetSubsampleResults:
    """All subsampling results for a dataset."""
    name: str
    results_by_size: Dict[int, SubsampleResults]


# =============================================================================
# Data Loading
# =============================================================================

def load_human_cc_histograms(
    data_dir: Path,
    bin_width: float = DEFAULT_BIN_WIDTH
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load human corpus callosum histogram data (MinorAxis radii)."""
    bin_edges_file = data_dir / 'desc-binEdges_radii.tsv'
    counts_file = data_dir / 'desc-countsMinorAxis_radii.tsv'

    bin_edges_orig = np.loadtxt(bin_edges_file, delimiter='\t', skiprows=1)
    counts_matrix_orig = np.loadtxt(counts_file, delimiter='\t', skiprows=1, dtype=float)
    n_rois = counts_matrix_orig.shape[0]

    first_edges, first_centers, _ = rediscretize(
        bin_edges_orig, counts_matrix_orig[0], bin_width
    )
    n_bins = len(first_centers)
    counts_matrix = np.zeros((n_rois, n_bins), dtype=float)

    for i in range(n_rois):
        _, _, counts_matrix[i] = rediscretize(
            bin_edges_orig, counts_matrix_orig[i], bin_width
        )

    sample_names = [f"ROI_{i+1}" for i in range(n_rois)]
    logger.info(f"Human CC: {n_rois} ROIs, {int(counts_matrix.sum()):,} total axons")

    return first_edges, first_centers, counts_matrix, sample_names


def load_rat_histograms(
    data_dir: Path,
    bin_width: float = DEFAULT_BIN_WIDTH,
    r_max: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load rat LM data from NPZ files and histogram the radii."""
    npz_files = sorted(data_dir.glob('*_axon_profiles.npz'))
    if not npz_files:
        raise ValueError(f"No *_axon_profiles.npz files found in {data_dir}")

    bin_edges = np.arange(0, r_max + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    all_counts = []
    all_names = []

    from axonometry.io import load_3d_profiles

    for npz_file in npz_files:
        volume_name = npz_file.stem.replace('_axon_profiles', '')
        data = load_3d_profiles(npz_file)
        radii = data['all_radii_um']
        counts, _ = np.histogram(radii, bins=bin_edges)
        all_counts.append(counts)
        all_names.append(volume_name)

    counts_matrix = np.array(all_counts, dtype=float)
    logger.info(f"Rat WM: {len(all_names)} ROIs, {int(counts_matrix.sum()):,} total radii")

    return bin_edges, bin_centers, counts_matrix, all_names


# =============================================================================
# Subsampling and Analysis
# =============================================================================

def subsample_histogram(
    counts: np.ndarray,
    bin_centers: np.ndarray,
    sample_size: int,
) -> np.ndarray:
    """Generate a subsample histogram from a full histogram."""
    total = counts.sum()
    if total < sample_size:
        return None

    probs = counts / total

    # Sample bin indices
    sampled_bins = np.random.choice(len(bin_centers), size=sample_size, p=probs)

    # Create histogram from samples
    sub_counts = np.bincount(sampled_bins, minlength=len(bin_centers))
    return sub_counts.astype(float)


def run_subsampling_analysis(
    bin_edges: np.ndarray,
    bin_centers: np.ndarray,
    counts_matrix: np.ndarray,
    sample_sizes: List[int],
    n_subsamples: int,
    dataset_name: str
) -> DatasetSubsampleResults:
    """Run subsampling analysis for all sample sizes."""
    n_rois = counts_matrix.shape[0]

    results_by_size = {}

    for sample_size in sample_sizes:
        logger.info(f"  Processing sample size {sample_size:,}...")

        # Find ROIs with enough axons
        roi_counts = counts_matrix.sum(axis=1)
        valid_roi_mask = roi_counts >= sample_size
        valid_roi_indices = np.where(valid_roi_mask)[0]
        n_valid = len(valid_roi_indices)

        if n_valid == 0:
            logger.warning(f"    No ROIs have >= {sample_size:,} axons, skipping")
            continue

        logger.info(f"    {n_valid}/{n_rois} ROIs have >= {sample_size:,} axons")

        # Initialize arrays
        r_arith = np.full((n_valid, n_subsamples), np.nan)
        r_eff = np.full((n_valid, n_subsamples), np.nan)
        reference_r_arith = np.full(n_valid, np.nan)
        reference_r_eff = np.full(n_valid, np.nan)

        for i, roi_idx in enumerate(valid_roi_indices):
            counts = counts_matrix[roi_idx]

            # Reference values from whole ROI
            reference_r_arith[i] = compute_r_arith(counts=counts, bin_centers=bin_centers)
            reference_r_eff[i] = compute_r_eff(counts=counts, bin_centers=bin_centers)

            # Subsampling
            for j in range(n_subsamples):
                sub_counts = subsample_histogram(counts, bin_centers, sample_size)
                if sub_counts is None:
                    continue
                r_arith[i, j] = compute_r_arith(counts=sub_counts, bin_centers=bin_centers)
                r_eff[i, j] = compute_r_eff(counts=sub_counts, bin_centers=bin_centers)

        results_by_size[sample_size] = SubsampleResults(
            sample_size=sample_size,
            n_valid_rois=n_valid,
            r_arith=r_arith,
            r_eff=r_eff,
            reference_r_arith=reference_r_arith,
            reference_r_eff=reference_r_eff
        )

    return DatasetSubsampleResults(
        name=dataset_name,
        results_by_size=results_by_size,
    )


# =============================================================================
# Plotting Functions
# =============================================================================



def plot_pdf_combined(
    ax: plt.Axes,
    human_bin_centers: np.ndarray,
    human_pooled_counts: np.ndarray,
    rat_bin_centers: np.ndarray,
    rat_pooled_counts: np.ndarray,
    sample_sizes: List[int],
    n_subsamples: int,
    x_max: float = 2.0
) -> None:
    """Plot PDFs for both datasets with subsampling variability."""
    font_settings = settings.fonts
    label_size = font_settings['label_size']
    tick_size = font_settings['tick_size']
    legend_size = font_settings['legend_size']

    # Species colors
    human_color = settings.colors['human']
    rat_color = settings.colors['rat']

    # Common x-axis for PDF evaluation
    x_eval = np.linspace(0.0, x_max, 200)

    datasets = [
        (rat_bin_centers, rat_pooled_counts, rat_color, 'Rat'),
        (human_bin_centers, human_pooled_counts, human_color, 'Human'),
    ]

    # Line styles for different sample sizes: dashed for subsamples
    line_styles = ['--']
    # Alphas for shaded areas
    fill_alphas = [0.25]

    # Store legend handles
    legend_handles = []

    for bin_centers, pooled_counts, color, species_label in datasets:
        total_count = int(pooled_counts.sum())
        bin_width = bin_centers[1] - bin_centers[0]
        probs = pooled_counts / total_count

        # Reference PDF (full distribution)
        pdf_ref = pooled_counts / (total_count * bin_width)
        pdf_ref_interp = np.interp(x_eval, bin_centers, pdf_ref, left=0, right=0)

        # Plot subsamples first (so full sample is on top)
        sorted_sizes = sorted([s for s in sample_sizes if s <= total_count])

        for idx, sample_size in enumerate(sorted_sizes):
            # Build subsample PDFs at bin centers, then take central 95% band
            pdf_subsamples = np.empty((n_subsamples, len(bin_centers)))
            for j in range(n_subsamples):
                sampled_bins = np.random.choice(
                    len(bin_centers), size=sample_size, p=probs
                )
                sub_counts = np.bincount(
                    sampled_bins, minlength=len(bin_centers)
                ).astype(float)
                pdf_subsamples[j] = sub_counts / (sub_counts.sum() * bin_width)

            pdf_lo = np.percentile(pdf_subsamples, 2.5, axis=0)
            pdf_hi = np.percentile(pdf_subsamples, 97.5, axis=0)

            # Interpolate from bin centers onto common x-axis
            lo_interp = np.interp(x_eval, bin_centers, pdf_lo, left=0, right=0)
            hi_interp = np.interp(x_eval, bin_centers, pdf_hi, left=0, right=0)

            # Shaded area
            ax.fill_between(x_eval, lo_interp, hi_interp,
                           alpha=fill_alphas[idx], color=color, zorder=1 + idx)

            # Boundary lines with different styles
            linestyle = line_styles[idx] if idx < len(line_styles) else '-'
            ax.plot(x_eval, lo_interp, color=color, linestyle=linestyle,
                   linewidth=0.8, alpha=0.7, zorder=2 + idx)
            ax.plot(x_eval, hi_interp, color=color, linestyle=linestyle,
                   linewidth=0.8, alpha=0.7, zorder=2 + idx)

        # Plot full distribution on top (solid line)
        ax.plot(x_eval, pdf_ref_interp, color=color,
                linewidth=settings.line['linewidth'],
                linestyle='-', zorder=10)

    # Build legend
    for bin_centers, pooled_counts, color, species_label in datasets:
        total_count = int(pooled_counts.sum())

        # Full sample label: e.g. "Human (n ≈ 5×10^7)" or "Rat (n ≈ 10^6)"
        exp = int(math.floor(math.log10(total_count)))
        mantissa = total_count / (10 ** exp)
        if mantissa >= 9.5:
            exp += 1
            mantissa = 1
        mantissa_rounded = int(round(mantissa))
        if mantissa_rounded == 1:
            full_label = rf'{species_label} ($n \approx 10^{exp}$)'
        else:
            full_label = rf'{species_label} ($n \approx {mantissa_rounded}\times 10^{exp}$)'
        legend_handles.append(Line2D([0], [0], color=color,
                                      linewidth=settings.line['linewidth'], label=full_label))

        # Subsample patches
        sorted_sizes = sorted([s for s in sample_sizes if s <= total_count])
        for idx, sample_size in enumerate(sorted_sizes):
            alpha = fill_alphas[idx] if idx < len(fill_alphas) else 0.25
            linestyle = line_styles[idx] if idx < len(line_styles) else '-'
            face_rgba = list(to_rgba(color))
            face_rgba[3] = alpha
            exp_sub = int(math.log10(sample_size))
            legend_handles.append(Patch(
                facecolor=face_rgba, edgecolor=color,
                linestyle=linestyle, linewidth=1.5,
                label=rf'{species_label} ($n = 10^{exp_sub}$)'))

    ax.set_xlabel('Axon radius [μm]', fontsize=label_size)
    ax.set_ylabel('Probability density [μm⁻¹]', fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    ax.legend(handles=legend_handles, loc='upper right', fontsize=legend_size)
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_box_aspect(1)


def compute_wasserstein_from_cdfs(cdf1: np.ndarray, cdf2: np.ndarray,
                                   bin_width: float) -> float:
    """Compute Wasserstein distance between two CDFs."""
    return np.sum(np.abs(cdf1 - cdf2)) * bin_width


def plot_within_vs_between_wasserstein(
    ax: plt.Axes,
    human_bin_centers: np.ndarray,
    human_counts_matrix: np.ndarray,
    rat_bin_centers: np.ndarray,
    rat_counts_matrix: np.ndarray,
    sample_sizes: List[int] = None,
    n_subsamples: int = N_SUBSAMPLES,
) -> None:
    """
    Plot within-sample Wasserstein distances with inter-ROI medians as reference.

    X-axis: sample sizes
    Grouped violins: Human (blue) and Rat (red) for each sample size
    Reference: Median inter-ROI distances shown as horizontal dashed lines
    """
    if sample_sizes is None:
        sample_sizes = [100, 1000, 10000]

    font_settings = settings.fonts
    label_size = font_settings['label_size']
    tick_size = font_settings['tick_size']
    legend_size = font_settings['legend_size']

    # Species colors
    human_color = settings.colors['human']
    rat_color = settings.colors['rat']

    datasets = [
        (rat_bin_centers, rat_counts_matrix, 'Rat', rat_color),
        (human_bin_centers, human_counts_matrix, 'Human', human_color),
    ]

    # Compute between-ROI distances per species
    between_median_per_species = {}
    for bin_centers, counts_matrix, name, color in datasets:
        bin_width = bin_centers[1] - bin_centers[0]
        n_rois = counts_matrix.shape[0]

        # Compute CDF for each ROI
        roi_cdfs = []
        for i in range(n_rois):
            roi_total = counts_matrix[i].sum()
            if roi_total > 0:
                roi_cdf = np.cumsum(counts_matrix[i]) / roi_total
                roi_cdfs.append(roi_cdf)

        # Pairwise Wasserstein distances within this species
        species_between = []
        for i in range(len(roi_cdfs)):
            for j in range(i + 1, len(roi_cdfs)):
                w_dist = compute_wasserstein_from_cdfs(roi_cdfs[i], roi_cdfs[j], bin_width)
                species_between.append(w_dist)
        between_median_per_species[name] = np.median(species_between)

    # Compute within-sample distances for each sample size and dataset
    width = 0.3
    n_sample_sizes = len(sample_sizes)
    x_positions = np.arange(n_sample_sizes)

    # Collect all data for violin plots
    all_violin_data = []
    all_positions = []
    all_colors = []

    for dataset_idx, (bin_centers, counts_matrix, name, color) in enumerate(datasets):
        bin_width = bin_centers[1] - bin_centers[0]

        # Compute pooled distribution
        pooled_counts = counts_matrix.sum(axis=0)
        pooled_total = pooled_counts.sum()
        pooled_cdf = np.cumsum(pooled_counts) / pooled_total
        pooled_probs = pooled_counts / pooled_total

        for size_idx, sample_size in enumerate(sample_sizes):
            within_distances = []
            for _ in range(n_subsamples):
                sampled_bins = np.random.choice(len(bin_centers), size=sample_size, p=pooled_probs)
                sub_counts = np.bincount(sampled_bins, minlength=len(bin_centers)).astype(float)
                sub_cdf = np.cumsum(sub_counts) / sub_counts.sum()
                w_dist = compute_wasserstein_from_cdfs(sub_cdf, pooled_cdf, bin_width)
                within_distances.append(w_dist)

            all_violin_data.append(within_distances)
            # Offset for grouped violins
            offset = -width/2 if dataset_idx == 0 else width/2
            all_positions.append(size_idx + offset)
            all_colors.append(color)

    # Set x-axis limits
    ax.set_xlim(-0.5, n_sample_sizes - 0.5)

    # Plot thin violins
    parts = ax.violinplot(all_violin_data, positions=all_positions,
                           showmeans=False, showmedians=True, widths=width * 0.8)

    # Color each violin
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(all_colors[i])
        pc.set_alpha(0.7)
        pc.set_zorder(2)

    # Style median lines
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)
    parts['cbars'].set_color('gray')
    parts['cbars'].set_linewidth(0.5)
    parts['cmins'].set_color('gray')
    parts['cmaxes'].set_color('gray')

    # Draw horizontal dashed lines for inter-ROI medians
    ax.axhline(between_median_per_species['Human'], color=human_color,
               linestyle='--', linewidth=settings.line['linewidth'], label='Human inter-ROI', zorder=1)
    ax.axhline(between_median_per_species['Rat'], color=rat_color,
               linestyle='--', linewidth=settings.line['linewidth'], label='Rat inter-ROI', zorder=1)

    # Add legend
    handles = [
        Patch(facecolor=rat_color, alpha=0.7, label='Rat (sampling)'),
        Patch(facecolor=human_color, alpha=0.7, label='Human (sampling)'),
        Line2D([0], [0], color=rat_color, linestyle='--', linewidth=settings.line['linewidth'], label='Rat (anat. var.)'),
        Line2D([0], [0], color=human_color, linestyle='--', linewidth=settings.line['linewidth'], label='Human (anat. var.)'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=legend_size)

    # X-axis: sample sizes only
    ax.set_xticks(x_positions)
    ax.set_xticklabels([SAMPLE_SIZE_LABELS[s] for s in sample_sizes], fontsize=tick_size)
    ax.set_xlabel('Sample size', fontsize=label_size)
    ax.set_ylabel('Wasserstein distance [μm]', fontsize=label_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    ax.set_ylim(bottom=0)
    # Set y-ticks in 0.05 steps
    y_max = ax.get_ylim()[1]
    ax.set_yticks(np.arange(0, y_max + 0.025, 0.05))

    ax.set_box_aspect(1)


def plot_combined_rel_error(
    ax: plt.Axes,
    human_results: DatasetSubsampleResults,
    rat_results: DatasetSubsampleResults,
    metric: str,  # 'r_arith' or 'r_eff'
    ylabel: str
) -> None:
    """Plot percentage error vs sample size for both datasets."""
    font_settings = settings.fonts
    label_size = font_settings['label_size']
    tick_size = font_settings['tick_size']
    legend_size = font_settings['legend_size']

    # Species colors
    human_color = settings.colors['human']
    rat_color = settings.colors['rat']

    datasets = [
        (rat_results, 'Rat', rat_color, 's'),      # Squares on bottom
        (human_results, 'Human', human_color, 'o'),  # Circles on top
    ]

    for dataset_results, label, color, marker in datasets:
        # Use all sample sizes
        sample_sizes_present = [s for s in SAMPLE_SIZES if s in dataset_results.results_by_size]
        if not sample_sizes_present:
            continue

        rel_error_median = []
        rel_error_lo = []
        rel_error_hi = []

        for sample_size in sample_sizes_present:
            results = dataset_results.results_by_size[sample_size]

            if metric == 'r_arith':
                ref = results.reference_r_arith[:, np.newaxis]
                values = results.r_arith
            else:  # r_eff
                ref = results.reference_r_eff[:, np.newaxis]
                values = results.r_eff

            # Compute relative error: (sub - ref) / ref * 100
            rel_error = ((values - ref) / ref * 100).ravel()

            rel_error_median.append(np.nanmedian(rel_error))
            rel_error_lo.append(np.nanpercentile(rel_error, 25))
            rel_error_hi.append(np.nanpercentile(rel_error, 75))

        rel_error_median = np.array(rel_error_median)
        rel_error_lo = np.array(rel_error_lo)
        rel_error_hi = np.array(rel_error_hi)

        # Plot IQR band
        ax.fill_between(sample_sizes_present, rel_error_lo, rel_error_hi, alpha=0.2, color=color)

        # Plot line with markers at all sample sizes
        ax.plot(sample_sizes_present, rel_error_median, color=color, marker=marker,
                markersize=6, linewidth=settings.line['linewidth'], label=label)

    ax.axhline(0, color='black', linestyle='--', alpha=0.5,
               linewidth=settings.line['linewidth'], zorder=0)
    ax.set_xscale('log')
    ax.set_xlabel('Sample size', fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.legend(loc='best', fontsize=legend_size)
    ax.tick_params(labelsize=tick_size)


def create_figure(
    human_results: DatasetSubsampleResults,
    rat_results: DatasetSubsampleResults,
    human_bin_centers: np.ndarray,
    human_counts_matrix: np.ndarray,
    rat_bin_centers: np.ndarray,
    rat_counts_matrix: np.ndarray,
    output_file: Path,
    n_subsamples: int = N_SUBSAMPLES
) -> None:
    """Create the 2×2 figure."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # Sample sizes for subsampling visualization (10^3 only)
    pdf_sample_sizes = [1_000]

    # Pooled counts for PDF plot
    human_pooled_counts = human_counts_matrix.sum(axis=0)
    rat_pooled_counts = rat_counts_matrix.sum(axis=0)

    # (a) PDFs for both datasets with subsampling variability
    plot_pdf_combined(axes[0, 0], human_bin_centers, human_pooled_counts,
                      rat_bin_centers, rat_pooled_counts,
                      pdf_sample_sizes, n_subsamples, x_max=2.0)

    # (b) Within-sample vs Between-ROI Wasserstein distances
    plot_within_vs_between_wasserstein(axes[0, 1], human_bin_centers, human_counts_matrix,
                                        rat_bin_centers, rat_counts_matrix,
                                        sample_sizes=[100, 1000, 10000, 100000], n_subsamples=n_subsamples)

    # (c) Arithmetic mean radius: percentage error vs sample size (both datasets)
    plot_combined_rel_error(axes[1, 0], human_results, rat_results,
                       'r_arith', r'$\bar{r}$ error [%]')

    # (d) Effective radius: percentage error vs sample size (both datasets)
    plot_combined_rel_error(axes[1, 1], human_results, rat_results,
                       'r_eff', r'$r_{\mathrm{MRI}}$ error [%]')

    # Set y-axis scale symmetric around 0 for each panel independently
    for ax in [axes[1, 0], axes[1, 1]]:
        ymin, ymax = ax.get_ylim()
        ylim_max = max(abs(ymin), abs(ymax))
        ax.set_ylim(-ylim_max, ylim_max)

    # Set aspect ratio 1 for bottom panels
    for ax in [axes[1, 0], axes[1, 1]]:
        ax.set_box_aspect(1)

    plt.tight_layout()

    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()
    logger.info(f"Saved figure to {output_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Study the effect of sample size on radius estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--human-data', type=Path, default=Path('data/raw/human/lm'),
                        help='Directory containing human CC TSV files (default: data/raw/human/lm)')
    parser.add_argument('--rat-data', type=Path, default=Path('data/processed/rat/lm'),
                        help='Directory containing rat NPZ files (default: data/processed/rat/lm)')
    parser.add_argument('--output', type=Path, default=Path('fig/main/sample_size_effect.svg'),
                        help='Output file path (default: fig/main/sample_size_effect.svg)')
    parser.add_argument('--n-subsamples', type=int, default=N_SUBSAMPLES,
                        help=f'Number of subsamples per ROI (default: {N_SUBSAMPLES})')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    # Load Human CC data
    logger.info("=" * 60)
    logger.info("Loading Human CC data...")
    human_edges, human_centers, human_counts, human_names = load_human_cc_histograms(args.human_data)

    logger.info("Running subsampling analysis for Human CC...")
    human_results = run_subsampling_analysis(
        human_edges, human_centers, human_counts,
        SAMPLE_SIZES, args.n_subsamples, "Human CC"
    )

    # Load Rat data
    logger.info("=" * 60)
    logger.info("Loading Rat WM data...")
    rat_edges, rat_centers, rat_counts, rat_names = load_rat_histograms(args.rat_data)

    logger.info("Running subsampling analysis for Rat WM...")
    rat_results = run_subsampling_analysis(
        rat_edges, rat_centers, rat_counts,
        SAMPLE_SIZES, args.n_subsamples, "Rat WM"
    )

    # Create figure
    logger.info("=" * 60)
    logger.info("Creating figure...")
    create_figure(
        human_results, rat_results,
        human_centers, human_counts,
        rat_centers, rat_counts,
        args.output,
        n_subsamples=args.n_subsamples
    )

    # Save JSON metadata
    json_file = args.output.with_suffix('.json')
    output_data = {
        'sample_sizes': SAMPLE_SIZES,
        'n_subsamples': args.n_subsamples,
        'human_cc': {
            size: {
                'n_valid_rois': results.n_valid_rois,
            }
            for size, results in human_results.results_by_size.items()
        },
        'rat_wm': {
            size: {
                'n_valid_rois': results.n_valid_rois,
            }
            for size, results in rat_results.results_by_size.items()
        }
    }
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Saved metadata to {json_file}")


if __name__ == '__main__':
    main()
