#!/usr/bin/env python3
"""
Plot radius variability statistics: CV histogram, CV vs radius, slowdown reduction.

Loads pre-computed NPZ from create_representative_axons.py and creates a 1×3
panel figure using pooled statistics across all axons.

Usage:
    python scripts/figures/plot_radius_variability_stats.py
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from axonometry import get_plot_settings, style_axis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

settings = get_plot_settings()


def load_pooled_stats(npz_path: Path) -> dict:
    """Load pooled statistics from representative axons NPZ."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        "all_mean_radii": data["all_mean_radii"],
        "all_cv": data["all_cv"],
        "all_slowdown": data["all_slowdown"],
    }


def main():
    parser = argparse.ArgumentParser(description="Plot radius variability statistics")
    parser.add_argument("--input", type=Path,
                        default=Path("data/processed/rat/lm/representative_axons.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path("fig/main/radius_variability_stats.svg"))
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        return

    d = load_pooled_stats(args.input)
    all_r = d["all_mean_radii"]
    all_cv = d["all_cv"]
    all_slow = d["all_slowdown"]

    hist_s = settings.histogram
    font_s = settings.fonts
    line_s = settings.line

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.05))
    ax_hist, ax_cv, ax_vel = axes

    # --- (a) CV histogram ---
    main_color = settings.colors["category_a"]
    ax_hist.hist(all_cv, bins=hist_s["bins"], color=main_color,
                 edgecolor=hist_s["edgecolor"], alpha=hist_s["alpha"])
    ax_hist.axvline(np.mean(all_cv), color=settings.colors["mean_line"], linestyle="--",
                    linewidth=line_s["linewidth"], label=f"Mean = {np.mean(all_cv):.3f}")
    ax_hist.axvline(np.median(all_cv), color=settings.colors["median_line"], linestyle=":",
                    linewidth=line_s["linewidth"], label=f"Median = {np.median(all_cv):.3f}")
    style_axis(ax_hist, xlabel="CoV", ylabel="Count")
    ax_hist.set_xlim(0, 0.8)
    ax_hist.legend(loc="upper right", fontsize=font_s["legend_size"])
    ax_hist.ticklabel_format(axis="y", style="sci", scilimits=(4, 4), useMathText=True)
    ax_hist.yaxis.get_offset_text().set_fontsize(font_s["tick_size"])

    # --- (b) CV vs radius (binned median + IQR) ---
    x_max = np.percentile(all_r, 99.5)
    n_bins = 30
    bin_edges = np.linspace(0, x_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_med = np.full(n_bins, np.nan)
    bin_q25 = np.full(n_bins, np.nan)
    bin_q75 = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (all_r >= bin_edges[i]) & (all_r < bin_edges[i + 1])
        if np.sum(mask) >= 100:
            bin_med[i] = np.median(all_cv[mask])
            bin_q25[i] = np.percentile(all_cv[mask], 25)
            bin_q75[i] = np.percentile(all_cv[mask], 75)

    valid = ~np.isnan(bin_med)
    single_color = settings.colors["single_line"]
    ax_cv.plot(bin_centers[valid], bin_med[valid], color=single_color, linestyle="-",
               linewidth=line_s["linewidth"], marker="o", markersize=line_s["marker_size"],
               label="Median")
    ax_cv.fill_between(bin_centers[valid], bin_q25[valid], bin_q75[valid],
                       color=single_color, alpha=line_s["fill_alpha"], label="IQR (25-75%)")
    style_axis(ax_cv, xlabel=r"Along-axon mean radius [μm]", ylabel="CoV")
    ax_cv.set_xlim(0.1, 0.6)
    ax_cv.set_ylim(0, 0.385)

    # --- (c) Slowdown reduction vs radius ---
    size_bins = 30
    sb_edges = np.linspace(0, x_max, size_bins + 1)
    sb_centers = (sb_edges[:-1] + sb_edges[1:]) / 2

    slow_med = np.full(size_bins, np.nan)
    slow_q25 = np.full(size_bins, np.nan)
    slow_q75 = np.full(size_bins, np.nan)
    diff_med = np.full(size_bins, np.nan)
    diff_q25 = np.full(size_bins, np.nan)
    diff_q75 = np.full(size_bins, np.nan)

    for i in range(size_bins):
        mask = (all_r >= sb_edges[i]) & (all_r < sb_edges[i + 1])
        if np.sum(mask) >= 100:
            slow_med[i] = np.median(all_slow[mask])
            slow_q25[i] = np.percentile(all_slow[mask], 25)
            slow_q75[i] = np.percentile(all_slow[mask], 75)
            diff_slow = 1.0 / (1.0 + 4.0 * all_cv[mask] ** 2)
            diff_med[i] = np.median(diff_slow)
            diff_q25[i] = np.percentile(diff_slow, 25)
            diff_q75[i] = np.percentile(diff_slow, 75)

    vb = ~np.isnan(slow_med)

    # Convert to reduction %
    cond_med = (1 - slow_med) * 100
    cond_lo = (1 - slow_q75) * 100   # flipped
    cond_hi = (1 - slow_q25) * 100
    diff_med_pct = (1 - diff_med) * 100
    diff_lo = (1 - diff_q75) * 100
    diff_hi = (1 - diff_q25) * 100

    cond_color = settings.colors["binary_a"]
    diff_color = settings.colors["binary_b"]

    ax_vel.plot(sb_centers[vb], cond_med[vb], color=cond_color, linestyle="-",
                linewidth=line_s["linewidth"], marker="o", markersize=line_s["marker_size"],
                label="Conduction velocity")
    ax_vel.fill_between(sb_centers[vb], cond_lo[vb], cond_hi[vb],
                        color=cond_color, alpha=line_s["fill_alpha"])

    ax_vel.plot(sb_centers[vb], diff_med_pct[vb], color=diff_color, linestyle="-",
                linewidth=line_s["linewidth"], marker="s", markersize=line_s["marker_size"],
                label="Diffusion (along-axon)")
    ax_vel.fill_between(sb_centers[vb], diff_lo[vb], diff_hi[vb],
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
    plt.savefig(args.output.with_suffix(".png"), dpi=settings.figure["dpi"], bbox_inches="tight")
    plt.close()

    logger.info(f"Saved to {args.output}")
    logger.info(f"  N={len(all_cv)} axons, mean CV={np.mean(all_cv):.3f}, "
                f"mean reduction={100*(1-np.mean(all_slow)):.1f}%")


if __name__ == "__main__":
    main()
