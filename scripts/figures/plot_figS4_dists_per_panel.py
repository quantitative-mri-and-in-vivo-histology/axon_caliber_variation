"""
Supplementary figure S4: per-panel parametric distribution fits.

Decluttered version of Figure 6 panels a/b: two stacked 2x3 blocks (Rat WM on
top, Human CC below, separated by a horizontal divider), one panel per
distribution showing the pooled histogram with only that single fit in black,
plus a tail inset. Loading and fitting are reused from the Figure 6 script.

Usage:
    python scripts/figures/plot_figS4_dists_per_panel.py \\
        --human-data data/raw/human/lm \\
        --rat-data data/processed/rat/lm \\
        --output fig/supplementary/parametric_fits_per_distribution.svg
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from axonometry import get_plot_settings

# Reuse loaders / fitting / constants from the Figure 6 script (sibling import).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_fig6_parametric_fits import (  # noqa: E402
    CANDIDATE_DISTRIBUTIONS, DEFAULT_BIN_WIDTH, INSET_PERCENTILE_HI,
    INSET_PERCENTILE_LO, PLOT_XLIM_MAX, FitResult, HistogramData,
    fit_all_samples, get_display_name, load_human_cc_data, load_rat_data,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings = get_plot_settings()


def create_per_distribution_figure(
    rat_pooled: HistogramData,
    rat_results: List[FitResult],
    human_pooled: HistogramData,
    human_results: List[FitResult],
    output_file: Path,
) -> None:
    """Supplementary per-distribution grid, decluttered version of panels a/b.

    Two stacked blocks (Rat WM on top, Human CC below), each a 2x3 grid of the
    six distributions. Every panel shows the pooled histogram (species color)
    with only that single fitted PDF in black, plus the tail inset.
    """
    dist_order = [name for name, _ in CANDIDATE_DISTRIBUTIONS]
    n_grid_rows, n_grid_cols = 2, 3  # 6 distributions per species block

    # Blocks: (pooled histogram, pooled fits, species color, header)
    blocks = [
        (rat_pooled, rat_results, settings.colors['rat'], 'Rat WM'),
        (human_pooled, human_results, settings.colors['human'], 'Human CC'),
    ]

    fig = plt.figure(figsize=(15.4, 12.07))
    subfigs = fig.subfigures(len(blocks), 1, hspace=0.10)

    for subfig, (hist_data, results, species_color, header) in zip(subfigs, blocks):
        bin_width = np.diff(hist_data.bin_edges).mean()
        density = hist_data.counts / (hist_data.total_count * bin_width)
        cdf = np.cumsum(hist_data.counts) / hist_data.total_count
        lo_idx = min(np.searchsorted(cdf, INSET_PERCENTILE_LO), len(hist_data.bin_centers) - 1)
        hi_idx = min(np.searchsorted(cdf, INSET_PERCENTILE_HI), len(hist_data.bin_centers) - 1)
        inset_lo = hist_data.bin_centers[lo_idx] - bin_width / 2
        inset_hi = hist_data.bin_centers[hi_idx]
        tail_mask = (hist_data.bin_centers >= inset_lo) & (hist_data.bin_centers <= inset_hi)
        tail_density = density[tail_mask]
        tail_y_max = (tail_density.max() * 1.2
                      if len(tail_density) and tail_density.max() > 0 else 0.1)

        subfig.suptitle(header, fontsize=settings.fonts['label_size'] + 1,
                        fontweight='bold')
        axs = subfig.subplots(n_grid_rows, n_grid_cols,
                              gridspec_kw={'hspace': 0.45})

        for k, dist_name in enumerate(dist_order):
            ax = axs[k // n_grid_cols, k % n_grid_cols]
            ax.bar(hist_data.bin_centers, density, width=bin_width * 0.9,
                   alpha=0.6, color=species_color, edgecolor='white',
                   linewidth=0.5, zorder=1)

            result = next((r for r in results if r.distribution_name == dist_name), None)
            if result is not None:
                ax.plot(result.pdf_x_fine, result.pdf_values_fine, '-',
                        color='black', linewidth=2.0, zorder=2)

            ax.set_xlim(0, PLOT_XLIM_MAX)
            ax.set_ylim(0, None)
            ax.tick_params(labelsize=settings.fonts['tick_size'] - 2)
            ax.set_title(get_display_name(dist_name),
                         fontsize=settings.fonts['tick_size'], fontweight='bold')

            # Tail inset (same percentile range as panels a/b), single fit in black
            ax_inset = ax.inset_axes([0.34, 0.34, 0.63, 0.63])
            ax_inset.bar(hist_data.bin_centers, density, width=bin_width * 0.9,
                         alpha=0.6, color=species_color, edgecolor='white', linewidth=0.3)
            if result is not None:
                ax_inset.plot(result.pdf_x_fine, result.pdf_values_fine, '-',
                              color='black', linewidth=1.5)
            ax_inset.set_xlim(inset_lo, inset_hi)
            ax_inset.set_ylim(0, tail_y_max)
            ax_inset.tick_params(labelsize=settings.fonts['tick_size'] - 3)
            ax.indicate_inset_zoom(ax_inset, edgecolor='gray', linewidth=1.0,
                                   linestyle='--', alpha=0.8)

            ax.set_xlabel('Axon radius [μm]',
                          fontsize=settings.fonts['label_size'] - 5)
            if k % n_grid_cols == 0:
                ax.set_ylabel(r'Probability density [μm$^{-1}$]',
                              fontsize=settings.fonts['label_size'] - 5)

    # Horizontal divider bar between the Rat (top) and Human (bottom) blocks,
    # spanning exactly the panel columns (so it doesn't overhang the sides).
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    xs = []
    for sf in subfigs:
        for ax in sf.axes:
            bb = ax.get_window_extent()
            xs.extend([bb.x0, bb.x1])
    left = inv.transform((min(xs), 0))[0]
    right = inv.transform((max(xs), 0))[0]
    fig.add_artist(Line2D([left, right], [0.5, 0.5], color='black',
                          linewidth=2.0, transform=fig.transFigure))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved per-distribution figure to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Supplementary S4: per-panel parametric distribution fits')
    parser.add_argument('--human-data', type=Path, default=Path('data/raw/human/lm'),
                        help='Directory containing human CC TSV files (default: data/raw/human/lm)')
    parser.add_argument('--rat-data', type=Path, default=Path('data/processed/rat/lm'),
                        help='Directory containing rat NPZ files (default: data/processed/rat/lm)')
    parser.add_argument('--output', type=Path,
                        default=Path('fig/supplementary/parametric_fits_per_distribution.svg'),
                        help='Output file path (default: '
                             'fig/supplementary/parametric_fits_per_distribution.svg)')
    parser.add_argument('--bin-width', type=float, default=DEFAULT_BIN_WIDTH,
                        help=f'Bin width in um (default: {DEFAULT_BIN_WIDTH})')
    parser.add_argument('--r-max', type=float, default=3.0,
                        help='Maximum radius in um (default: 3.0)')
    args = parser.parse_args()

    logger.info("Loading Human CC data...")
    human_pooled, human_per_sample = load_human_cc_data(args.human_data, args.bin_width)
    logger.info("Loading Rat data...")
    rat_pooled, rat_per_sample = load_rat_data(args.rat_data, args.bin_width, args.r_max)

    logger.info("Fitting Human CC distributions...")
    human_metrics = fit_all_samples(human_per_sample, human_pooled)
    logger.info("Fitting Rat distributions...")
    rat_metrics = fit_all_samples(rat_per_sample, rat_pooled)

    create_per_distribution_figure(
        rat_pooled, rat_metrics.pooled_results,
        human_pooled, human_metrics.pooled_results,
        args.output,
    )


if __name__ == '__main__':
    main()
