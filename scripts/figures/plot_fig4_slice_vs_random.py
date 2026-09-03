"""
Plot Figure 4 (2D slice vs random sampling) from precomputed source data.

Reads the CSVs written by gen_data_fig4_slice_vs_random.py and renders the 1x3
figure:
  (a) Wasserstein distance violins: slice vs random sampling
  (b) r̄ scatter (2D vs 3D)      (c) r_MRI scatter (2D vs 3D)

Usage:
    python scripts/figures/plot_fig4_slice_vs_random.py
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.ticker import FormatStrFormatter

from axonometry import get_plot_settings, style_axis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

settings = get_plot_settings()


def _fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def plot_wasserstein_violin(ax, viol, viol_stats, points):
    color_slice = settings.colors["binary_a"]
    color_random = settings.colors["binary_b"]

    vpstats = []
    for g in ("slice", "random"):
        gv = viol[viol["group"] == g]
        s = viol_stats[viol_stats["group"] == g].iloc[0]
        vpstats.append({"coords": gv["coord"].to_numpy(), "vals": gv["density"].to_numpy(),
                        "mean": s["mean"], "median": s["median"],
                        "min": s["vmin"], "max": s["vmax"]})

    parts = ax.violin(vpstats, positions=[1, 2], showmeans=False, showmedians=True, widths=0.7)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor([color_slice, color_random][i])
        body.set_edgecolor("black")
        body.set_alpha(0.7)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_edgecolor("black")
            parts[key].set_linewidth(1.5)

    for g, color in (("slice", color_slice), ("random", color_random)):
        gp = points[points["group"] == g]
        ax.scatter(gp["x"], gp["y"], alpha=0.3, s=3, color=color, zorder=0)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["2D slice-wise\nsampling", "2D random\nsampling"],
                       fontsize=settings.fonts["tick_size"] - 1)
    style_axis(ax, ylabel="Wasserstein distance [μm]")
    ax.set_ylim(bottom=0)
    ax.set_box_aspect(1)


def plot_scatter(ax, scatter, stats_row, metric):
    font_s = settings.fonts
    err_s = settings.error_bars
    color_slice = settings.colors["binary_a"]
    color_random = settings.colors["binary_b"]

    x = scatter[f"{metric}_3d"].to_numpy()
    s_med = scatter[f"{metric}_slice_median"].to_numpy()
    s_lo = s_med - scatter[f"{metric}_slice_q25"].to_numpy()
    s_hi = scatter[f"{metric}_slice_q75"].to_numpy() - s_med
    r_med = scatter[f"{metric}_random_median"].to_numpy()
    r_lo = r_med - scatter[f"{metric}_random_q25"].to_numpy()
    r_hi = scatter[f"{metric}_random_q75"].to_numpy() - r_med

    ax.errorbar(x, s_med, yerr=[s_lo, s_hi], fmt="o", color=color_slice,
                ecolor=to_rgba(color_slice, 0.7), markersize=8,
                capsize=err_s["capsize"], capthick=err_s["capthick"], elinewidth=err_s["linewidth"],
                markerfacecolor=to_rgba(color_slice, 0.3), markeredgecolor=color_slice,
                markeredgewidth=1.5, label="2D slice-wise sampling", zorder=10)
    ax.errorbar(x, r_med, yerr=[r_lo, r_hi], fmt="o", color=color_random,
                ecolor=to_rgba(color_random, 0.7), markersize=4,
                capsize=err_s["capsize"], capthick=err_s["capthick"], elinewidth=err_s["linewidth"],
                markerfacecolor=color_random, markeredgecolor=color_random,
                markeredgewidth=1.5, label="2D random sampling", zorder=11)

    all_vals = np.concatenate([x, s_med, r_med])
    lo, hi = np.nanmin(all_vals) * 0.95, np.nanmax(all_vals) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=settings.line["linewidth"], zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ticks = ax.get_yticks()
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    if metric == "mean_radius":
        xlabel, ylabel = r"$\bar{r}$ (3D) [μm]", r"$\bar{r}$ (2D) [μm]"
    else:
        xlabel, ylabel = r"$r_{\mathrm{MRI}}$ (3D) [μm]", r"$r_{\mathrm{MRI}}$ (2D) [μm]"
    style_axis(ax, xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc="upper left", fontsize=font_s["legend_size"])

    text_slice = (f"Bias = {stats_row['bias_slice']:+.1f}%\n"
                  f"$R$ = {stats_row['r_slice_mean']:.2f} ± {stats_row['r_slice_std']:.2f}\n"
                  f"{_fmt_p(stats_row['p_slice'])}")
    ax.text(0.04, 0.74, text_slice, transform=ax.transAxes,
            fontsize=font_s["legend_size"], ha="left", va="top", color=color_slice)
    text_random = (f"Bias = {stats_row['bias_random']:+.1f}%\n"
                   f"$R$ = {stats_row['r_random_mean']:.2f} ± {stats_row['r_random_std']:.2f}\n"
                   f"{_fmt_p(stats_row['p_random'])}")
    ax.text(0.96, 0.04, text_random, transform=ax.transAxes,
            fontsize=font_s["legend_size"], ha="right", va="bottom", color=color_random)
    ax.set_box_aspect(1)


def main():
    parser = argparse.ArgumentParser(description="Plot Fig 4 from source data")
    parser.add_argument("--data-dir", type=Path, default=Path("data/figures"))
    parser.add_argument("--prefix", type=str, default="fig_4")
    parser.add_argument("--output", type=Path, default=Path("fig/main/fig_4.svg"))
    args = parser.parse_args()

    dd, pf = args.data_dir, args.prefix
    viol = pd.read_csv(dd / f"{pf}a_wasserstein_violin.csv")
    viol_stats = pd.read_csv(dd / f"{pf}a_wasserstein_stats.csv")
    points = pd.read_csv(dd / f"{pf}a_wasserstein_points.csv")
    scatter = pd.read_csv(dd / f"{pf}bc_scatter.csv")
    cd_stats = pd.read_csv(dd / f"{pf}bc_stats.csv").set_index("metric")

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))
    plot_wasserstein_violin(axes[0], viol, viol_stats, points)
    plot_scatter(axes[1], scatter, cd_stats.loc["mean_radius"], "mean_radius")
    plot_scatter(axes[2], scatter, cd_stats.loc["r_eff"], "r_eff")

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=settings.figure["dpi"], bbox_inches="tight")
    plt.close()
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
