"""
Plot Figure 3 (2D vs 3D radius distributions) from precomputed source data.

Reads the CSVs written by gen_data_fig3_2d_vs_3d.py and renders the 2x2 figure:
  (a) PDF: 3D reference + 2D median with central-95% band, r̄/r_MRI markers
  (b) Wasserstein distances: within-ROI (sampling) vs between-ROI (biological)
  (c) r̄ scatter (2D vs 3D)      (d) r_MRI scatter (2D vs 3D)

Use --prefix fig_3 for the main figure (minor-axis) or fig_s3 for the
circular-radius supplement.

Usage:
    python scripts/figures/plot_fig3_2d_vs_3d.py
    python scripts/figures/plot_fig3_2d_vs_3d.py --prefix fig_s3 --output fig/supplementary/fig_s3.svg
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from axonometry import get_plot_settings, style_axis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings = get_plot_settings()


class HandlerLineInPatch(HandlerBase):
    """Legend handler: median line centered inside a shaded band."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        patch, line = orig_handle
        rect = Rectangle((xdescent, ydescent), width, height,
                         facecolor=patch.get_facecolor(), edgecolor='none', transform=trans)
        ln = Line2D([xdescent, xdescent + width],
                    [ydescent + height / 2.0, ydescent + height / 2.0],
                    color=line.get_color(), linewidth=line.get_linewidth(),
                    linestyle=line.get_linestyle(), transform=trans)
        return [rect, ln]


def plot_pdf_panel(ax, pdf, markers, x_max):
    color_2d = settings.colors['binary_a']
    color_3d = settings.colors['binary_b']
    vline_lw = settings.line['linewidth']
    x = pdf['bin_center'].to_numpy()

    ax.plot(x, pdf['pdf_3d'], color=color_3d, linewidth=vline_lw, linestyle='-')
    ax.fill_between(x, pdf['pdf_2d_lo'], pdf['pdf_2d_hi'], alpha=0.3, color=color_2d)
    ax.plot(x, pdf['pdf_2d_median'], color=color_2d, linewidth=vline_lw, linestyle='-')

    m = markers.set_index('marker')['value']
    ax.axvline(m['r_arith_3d'], color=color_3d, linewidth=vline_lw, linestyle=':', alpha=0.9)
    ax.axvline(m['r_arith_2d_median'], color=color_2d, linewidth=vline_lw, linestyle=':', alpha=0.9)
    ax.axvline(m['r_eff_3d'], color=color_3d, linewidth=vline_lw, linestyle='--', alpha=0.9)
    if not np.isnan(m['r_eff_2d_median']):
        ax.axvline(m['r_eff_2d_median'], color=color_2d, linewidth=vline_lw, linestyle='--', alpha=0.9)

    handles, labels = [], []
    handles.append(Line2D([0], [0], color=color_3d, linewidth=1.5, linestyle='-')); labels.append('3D PDF')
    handles.append(Line2D([0], [0], color=color_3d, linewidth=vline_lw, linestyle=':')); labels.append(r'3D $\bar{r}$')
    handles.append(Line2D([0], [0], color=color_3d, linewidth=vline_lw, linestyle='--')); labels.append(r'3D $r_{\mathrm{MRI}}$')
    handles.append((Patch(facecolor=color_2d, alpha=0.3),
                    Line2D([0], [0], color=color_2d, linewidth=1.5, linestyle='-')))
    labels.append('2D median PDF\n(+ central 95%)')
    handles.append(Line2D([0], [0], color=color_2d, linewidth=vline_lw, linestyle=':')); labels.append(r'2D $\bar{r}$ median')
    handles.append(Line2D([0], [0], color=color_2d, linewidth=vline_lw, linestyle='--')); labels.append(r'2D $r_{\mathrm{MRI}}$ median')

    style_axis(ax, xlabel='Axon radius [μm]', ylabel='Probability density [μm⁻¹]')
    ax.legend(handles, labels, loc='upper right', fontsize=settings.fonts['legend_size'] - 1,
              framealpha=0.9, handler_map={tuple: HandlerLineInPatch()})
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_box_aspect(1)


def plot_wasserstein_panel(ax, within, between):
    parts = ax.violinplot([within, between], positions=[1, 2],
                          showmeans=False, showmedians=True, widths=0.7)
    colors = [settings.colors['category_a_violin'], settings.colors['category_b_violin']]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.8)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)
    for key in ('cbars', 'cmins', 'cmaxes'):
        parts[key].set_color('gray')

    rng = np.random.default_rng(42)
    n_show = min(500, len(within))
    idx = rng.choice(len(within), n_show, replace=False) if len(within) > n_show else np.arange(len(within))
    jitter = rng.uniform(-0.15, 0.15, len(idx))
    ax.scatter(1 + jitter, within[idx], alpha=0.4, s=5, color='#808080', zorder=0)
    jitter = rng.uniform(-0.15, 0.15, len(between))
    ax.scatter(2 + jitter, between, alpha=0.5, s=10, color='#404040', zorder=0)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Sampling error\n(intra-ROI 2D ↔ 3D)', 'Anat. variability\n(inter-ROI 3D)'],
                       fontsize=settings.fonts['tick_size'] - 1)
    style_axis(ax, ylabel='Wasserstein distance [μm]')
    ax.set_ylim(bottom=0)
    ax.set_box_aspect(1)


def plot_scatter_panel(ax, scatter, stats_row, metric):
    x = scatter[f'{metric}_3d'].to_numpy()
    y = scatter[f'{metric}_2d_median'].to_numpy()
    yerr_lo = y - scatter[f'{metric}_2d_q25'].to_numpy()
    yerr_hi = scatter[f'{metric}_2d_q75'].to_numpy() - y

    color = settings.colors['single_line']
    err = settings.error_bars
    ax.errorbar(x, y, yerr=[yerr_lo, yerr_hi], fmt='o', color=color,
                ecolor=to_rgba(color, 0.7), markersize=8,
                capsize=err['capsize'], capthick=err['capthick'], elinewidth=err['linewidth'],
                markerfacecolor=to_rgba(color, 0.3), markeredgecolor=color, markeredgewidth=1.5)

    all_vals = np.concatenate([x, y])
    lo, hi = all_vals.min() * 0.95, all_vals.max() * 1.05
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, linewidth=settings.line['linewidth'], zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_box_aspect(1)

    if metric == 'r_arith':
        xlabel, ylabel = r'$\bar{r}$ (3D) [μm]', r'$\bar{r}$ (2D) [μm]'
        step = 0.05
        ticks = np.arange(np.ceil(lo / step) * step, np.floor(hi / step) * step + step / 2, step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    else:
        xlabel, ylabel = r'$r_{\mathrm{MRI}}$ (3D) [μm]', r'$r_{\mathrm{MRI}}$ (2D) [μm]'
        ax.set_xticks(ax.get_yticks())
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    style_axis(ax, xlabel=xlabel, ylabel=ylabel)

    p = stats_row['p_value']
    p_str = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
    text = (f"Bias = {stats_row['bias_pct']:+.1f}%\n"
            f"$R$ = {stats_row['r_mean']:.2f} ± {stats_row['r_std']:.2f}\n{p_str}")
    ax.text(0.95, 0.05, text, transform=ax.transAxes,
            fontsize=settings.fonts['legend_size'], ha='right', va='bottom')


def main():
    parser = argparse.ArgumentParser(description="Plot Fig 3 / S3 from source data")
    parser.add_argument('--data-dir', type=Path, default=Path('data/figures'))
    parser.add_argument('--prefix', type=str, default='fig_3')
    parser.add_argument('--output', type=Path, default=Path('fig/main/fig_3.svg'))
    parser.add_argument('--x-max', type=float, default=1.0)
    args = parser.parse_args()

    dd, pf = args.data_dir, args.prefix
    pdf = pd.read_csv(dd / f"{pf}_a_pdf.csv")
    markers = pd.read_csv(dd / f"{pf}_a_markers.csv")
    within = pd.read_csv(dd / f"{pf}_b_wasserstein_within.csv")['wasserstein_um'].to_numpy()
    between = pd.read_csv(dd / f"{pf}_b_wasserstein_between.csv")['wasserstein_um'].to_numpy()
    scatter = pd.read_csv(dd / f"{pf}_cd_scatter.csv")
    cd_stats = pd.read_csv(dd / f"{pf}_cd_stats.csv").set_index('metric')

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    plot_pdf_panel(axes[0, 0], pdf, markers, args.x_max)
    plot_wasserstein_panel(axes[0, 1], within, between)
    plot_scatter_panel(axes[1, 0], scatter, cd_stats.loc['r_arith'], 'r_arith')
    plot_scatter_panel(axes[1, 1], scatter, cd_stats.loc['r_eff'], 'r_eff')

    plt.tight_layout(w_pad=2.5, h_pad=2.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()
    logger.info(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
