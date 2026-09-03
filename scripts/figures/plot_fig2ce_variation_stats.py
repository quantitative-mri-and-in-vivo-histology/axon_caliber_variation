"""
Plot Figure 2c-e (radius variability statistics) from precomputed source data.

Reads the CSVs written by gen_data_fig2ce_variation_stats.py and renders the
1x3 panel figure:
  (c) CoV histogram with mean/median lines
  (d) CoV vs along-axon mean radius (binned median + IQR)
  (e) conduction-velocity & diffusion reduction vs radius

Usage:
    python scripts/figures/plot_fig2ce_variation_stats.py
    python scripts/figures/plot_fig2ce_variation_stats.py --data-dir data/figures
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from axonometry import get_plot_settings, style_axis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

settings = get_plot_settings()


def main():
    parser = argparse.ArgumentParser(description="Plot Fig 2c-e from source data")
    parser.add_argument("--data-dir", type=Path, default=Path("data/figures"),
                        help="Directory containing the fig_2* source-data CSVs")
    parser.add_argument("--output", type=Path,
                        default=Path("fig/main/fig_2ce.svg"))
    args = parser.parse_args()

    dd = args.data_dir
    hist = pd.read_csv(dd / "fig_2c_cov_histogram.csv")
    hist_lines = pd.read_csv(dd / "fig_2c_summary.csv").set_index("statistic")["cov"]
    cvr = pd.read_csv(dd / "fig_2d_cov_vs_radius.csv")
    red = pd.read_csv(dd / "fig_2e_reduction_vs_radius.csv")
    red_med = pd.read_csv(dd / "fig_2e_summary.csv").set_index("statistic")["value"]

    hist_s = settings.histogram
    font_s = settings.fonts
    line_s = settings.line

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax_hist, ax_cv, ax_vel = axes

    # --- (c) CoV histogram ---
    centers = hist["cov_bin_center"].to_numpy()
    counts = hist["count"].to_numpy()
    bin_w = centers[1] - centers[0] if len(centers) > 1 else 0.02
    ax_hist.bar(centers, counts, width=bin_w, align="center",
                color=settings.colors["category_a"], edgecolor=hist_s["edgecolor"],
                alpha=hist_s["alpha"])
    mean_cv, median_cv = float(hist_lines["mean"]), float(hist_lines["median"])
    ax_hist.axvline(mean_cv, color=settings.colors["mean_line"], linestyle="-",
                    linewidth=line_s["linewidth"], label=f"Mean = {mean_cv:.3f}")
    ax_hist.axvline(median_cv, color=settings.colors["median_line"], linestyle="-",
                    linewidth=line_s["linewidth"], label=f"Median = {median_cv:.3f}")
    style_axis(ax_hist, xlabel="CoV", ylabel="Count")
    ax_hist.set_xlim(0, 0.8)
    ax_hist.legend(loc="upper right", fontsize=font_s["legend_size"])
    ax_hist.ticklabel_format(axis="y", style="sci", scilimits=(4, 4), useMathText=True)
    ax_hist.yaxis.get_offset_text().set_fontsize(font_s["tick_size"])

    # --- (d) CoV vs radius (binned median + IQR) ---
    single_color = settings.colors["single_line"]
    ax_cv.plot(cvr["mean_radius_um"], cvr["cov_median"], color=single_color, linestyle="-",
               linewidth=line_s["linewidth"], marker="o", markersize=line_s["marker_size"],
               label="Median")
    ax_cv.fill_between(cvr["mean_radius_um"], cvr["cov_q25"], cvr["cov_q75"],
                       color=single_color, alpha=line_s["fill_alpha"], label="IQR (25-75%)")
    style_axis(ax_cv, xlabel=r"Along-axon mean radius [μm]", ylabel="CoV")
    ax_cv.set_xlim(0.1, 0.6)
    ax_cv.set_ylim(0, 0.385)

    # --- (e) Reduction vs radius ---
    cond_color = settings.colors["binary_a"]
    diff_color = settings.colors["binary_b"]
    med_cond = float(red_med["cond_median_pct"])
    med_diff = float(red_med["diff_median_pct"])

    ax_vel.plot(red["mean_radius_um"], red["cond_median_pct"], color=cond_color, linestyle="-",
                linewidth=line_s["linewidth"], marker="o", markersize=line_s["marker_size"],
                label=f"Conduction velocity\n(median: {med_cond:.1f}%)")
    ax_vel.fill_between(red["mean_radius_um"], red["cond_q25_pct"], red["cond_q75_pct"],
                        color=cond_color, alpha=line_s["fill_alpha"])
    ax_vel.plot(red["mean_radius_um"], red["diff_median_pct"], color=diff_color, linestyle="-",
                linewidth=line_s["linewidth"], marker="s", markersize=line_s["marker_size"],
                label=f"Diffusion along-axon\n(median: {med_diff:.1f}%)")
    ax_vel.fill_between(red["mean_radius_um"], red["diff_q25_pct"], red["diff_q75_pct"],
                        color=diff_color, alpha=line_s["fill_alpha"])
    ax_vel.legend(loc="upper right", fontsize=font_s["legend_size"])
    style_axis(ax_vel, xlabel=r"Along-axon mean radius [μm]", ylabel="Reduction [%]")
    ax_vel.set_xlim(0.1, 0.6)
    ax_vel.set_ylim(0, 36)

    for ax in axes:
        ax.set_box_aspect(1)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=settings.figure["dpi"], bbox_inches="tight")
    plt.close()
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
