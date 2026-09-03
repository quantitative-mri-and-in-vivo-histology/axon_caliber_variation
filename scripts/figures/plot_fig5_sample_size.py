"""
Plot Figure 5 (sample-size effect) from precomputed source data.

Reads the CSVs written by gen_data_fig5_sample_size.py and renders the 2x2 figure:
  (a) pooled PDFs with n=10^3 subsample central-95% band (Rat, Human)
  (b) within-sample vs between-ROI Wasserstein distances (grouped violins)
  (c) r̄ relative error vs sample size      (d) r_MRI relative error vs sample size

Usage:
    python scripts/figures/plot_fig5_sample_size.py
"""

import argparse
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from axonometry import get_plot_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings = get_plot_settings()

SAMPLE_SIZE_LABELS = {100: r'$10^2$', 1_000: r'$10^3$', 10_000: r'$10^4$',
                      100_000: r'$10^5$', 1_000_000: r'$10^6$'}
B_SAMPLE_SIZES = [100, 1_000, 10_000, 100_000]
PDF_X_MAX = 2.0


def _n_label(species, total):
    exp = int(math.floor(math.log10(total)))
    mant = total / (10 ** exp)
    if mant >= 9.5:
        exp += 1
        mant = 1
    mr = int(round(mant))
    if mr == 1:
        return rf'{species} ($n \approx 10^{exp}$)'
    return rf'{species} ($n \approx {mr}\times 10^{exp}$)'


def plot_pdf(ax, pdf, totals):
    fonts = settings.fonts
    colors = {'rat': settings.colors['rat'], 'human': settings.colors['human']}
    x = pdf['x_eval'].to_numpy()
    total_by = totals.set_index('species')['total_count']
    handles = []
    for name in ('rat', 'human'):
        color = colors[name]
        ax.fill_between(x, pdf[f'{name}_sub_lo'], pdf[f'{name}_sub_hi'],
                        alpha=0.25, color=color, zorder=1)
        ax.plot(x, pdf[f'{name}_sub_lo'], color=color, linestyle='--', linewidth=0.8, alpha=0.7, zorder=2)
        ax.plot(x, pdf[f'{name}_sub_hi'], color=color, linestyle='--', linewidth=0.8, alpha=0.7, zorder=2)
        ax.plot(x, pdf[f'{name}_pdf_ref'], color=color,
                linewidth=settings.line['linewidth'], linestyle='-', zorder=10)
    for name in ('rat', 'human'):
        color = colors[name]
        species = name.capitalize()
        handles.append(Line2D([0], [0], color=color, linewidth=settings.line['linewidth'],
                              label=_n_label(species, int(total_by[name]))))
        face = list(to_rgba(color)); face[3] = 0.25
        handles.append(Patch(facecolor=face, edgecolor=color, linestyle='--', linewidth=1.5,
                             label=rf'{species} ($n = 10^3$)'))
    ax.set_xlabel('Axon radius [μm]', fontsize=fonts['label_size'])
    ax.set_ylabel('Probability density [μm⁻¹]', fontsize=fonts['label_size'])
    ax.tick_params(labelsize=fonts['tick_size'])
    ax.legend(handles=handles, loc='upper right', fontsize=fonts['legend_size'])
    ax.set_xlim(0, PDF_X_MAX)
    ax.set_ylim(bottom=0)
    ax.set_box_aspect(1)


def plot_wasserstein(ax, viol, stats_df, between):
    fonts = settings.fonts
    colors = {'rat': settings.colors['rat'], 'human': settings.colors['human']}

    vpstats, positions, body_colors = [], [], []
    for _, s in stats_df.iterrows():
        gv = viol[(viol['species'] == s['species']) & (viol['sample_size'] == s['sample_size'])]
        vpstats.append({'coords': gv['coord'].to_numpy(), 'vals': gv['density'].to_numpy(),
                        'mean': s['mean'], 'median': s['median'], 'min': s['vmin'], 'max': s['vmax']})
        positions.append(s['position'])
        body_colors.append(colors[s['species']])

    n_sizes = len(B_SAMPLE_SIZES)
    ax.set_xlim(-0.5, n_sizes - 0.5)
    parts = ax.violin(vpstats, positions=positions, showmeans=False, showmedians=True, widths=0.3 * 0.8)
    for pc, c in zip(parts['bodies'], body_colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)
        pc.set_zorder(2)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)
    parts['cbars'].set_color('gray')
    parts['cbars'].set_linewidth(0.5)
    parts['cmins'].set_color('gray')
    parts['cmaxes'].set_color('gray')

    bt = between.set_index('species')['median']
    lw = settings.line['linewidth']
    ax.axhline(bt['human'], color=colors['human'], linestyle='--', linewidth=lw, zorder=1)
    ax.axhline(bt['rat'], color=colors['rat'], linestyle='--', linewidth=lw, zorder=1)

    handles = [
        Patch(facecolor=colors['rat'], alpha=0.7, label='Rat (sampling)'),
        Patch(facecolor=colors['human'], alpha=0.7, label='Human (sampling)'),
        Line2D([0], [0], color=colors['rat'], linestyle='--', linewidth=lw, label='Rat (anat. var.)'),
        Line2D([0], [0], color=colors['human'], linestyle='--', linewidth=lw, label='Human (anat. var.)'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=fonts['legend_size'])
    ax.set_xticks(np.arange(n_sizes))
    ax.set_xticklabels([SAMPLE_SIZE_LABELS[s] for s in B_SAMPLE_SIZES], fontsize=fonts['tick_size'])
    ax.set_xlabel('Sample size', fontsize=fonts['label_size'])
    ax.set_ylabel('Wasserstein distance [μm]', fontsize=fonts['label_size'])
    ax.tick_params(axis='y', labelsize=fonts['tick_size'])
    ax.set_ylim(bottom=0)
    y_max = ax.get_ylim()[1]
    ax.set_yticks(np.arange(0, y_max + 0.025, 0.05))
    ax.set_box_aspect(1)


def plot_rel_error(ax, err, metric, ylabel):
    fonts = settings.fonts
    colors = {'rat': settings.colors['rat'], 'human': settings.colors['human']}
    markers = {'rat': 's', 'human': 'o'}
    for name in ('rat', 'human'):
        sub = err[(err['metric'] == metric) & (err['species'] == name)].sort_values('sample_size')
        if sub.empty:
            continue
        sizes = sub['sample_size'].to_numpy()
        ax.fill_between(sizes, sub['q25_pct'], sub['q75_pct'], alpha=0.2, color=colors[name])
        ax.plot(sizes, sub['median_pct'], color=colors[name], marker=markers[name],
                markersize=6, linewidth=settings.line['linewidth'], label=name.capitalize())
    ax.axhline(0, color='black', linestyle='--', alpha=0.5,
               linewidth=settings.line['linewidth'], zorder=0)
    ax.set_xscale('log')
    ax.set_xlabel('Sample size', fontsize=fonts['label_size'])
    ax.set_ylabel(ylabel, fontsize=fonts['label_size'])
    ax.legend(loc='best', fontsize=fonts['legend_size'])
    ax.tick_params(labelsize=fonts['tick_size'])


def main():
    parser = argparse.ArgumentParser(description="Plot Fig 5 from source data")
    parser.add_argument('--data-dir', type=Path, default=Path('data/figures'))
    parser.add_argument('--output', type=Path, default=Path('fig/main/fig_5.svg'))
    args = parser.parse_args()

    dd = args.data_dir
    pdf = pd.read_csv(dd / 'fig_5a_pdf.csv')
    totals = pd.read_csv(dd / 'fig_5a_totals.csv')
    viol = pd.read_csv(dd / 'fig_5b_wasserstein_violin.csv')
    viol_stats = pd.read_csv(dd / 'fig_5b_wasserstein_stats.csv')
    between = pd.read_csv(dd / 'fig_5b_between.csv')
    err = pd.read_csv(dd / 'fig_5cd_error.csv')

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    plot_pdf(axes[0, 0], pdf, totals)
    plot_wasserstein(axes[0, 1], viol, viol_stats, between)
    plot_rel_error(axes[1, 0], err, 'r_arith', r'$\bar{r}$ error [%]')
    plot_rel_error(axes[1, 1], err, 'r_eff', r'$r_{\mathrm{MRI}}$ error [%]')

    for ax in (axes[1, 0], axes[1, 1]):
        ymin, ymax = ax.get_ylim()
        m = max(abs(ymin), abs(ymax))
        ax.set_ylim(-m, m)
        ax.set_box_aspect(1)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()
    logger.info(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
