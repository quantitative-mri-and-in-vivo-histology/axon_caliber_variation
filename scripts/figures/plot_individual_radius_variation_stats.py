"""
Plot radius variability statistics: CV histogram, CV vs radius, slowdown reduction.

Loads axon profile NPZ files from a data directory and creates a 1×3
panel figure using pooled statistics across all axons.

Usage:
    python scripts/figures/plot_radius_variability_stats.py
    python scripts/figures/plot_radius_variability_stats.py --data-dir data/processed/rat/lm
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


def load_axon_stats(data_dir: Path, min_length_um: float = 20.0) -> dict:
    """Load per-axon statistics from all axon profile NPZ files in data_dir.

    Radius samples from all segments of each axon are pooled.
    Only axons with total length >= min_length_um are included.
    """
    from axonometry.io import load_3d_profiles

    all_mean_radii = []
    all_cv = []
    all_slowdown = []
    total_axons = 0

    for npz_path in sorted(data_dir.glob("*_axon_profiles.npz")):
        filtered = load_3d_profiles(npz_path)
        total_axons += len(filtered['labels'])

        for i in range(len(filtered['labels'])):
            seg_radii = filtered['segment_radii_um'][i]
            seg_lengths = filtered['segment_lengths_um'][i]
            if not seg_radii:
                continue

            total_len = sum(float(sl) for sl in seg_lengths)
            if total_len < min_length_um:
                continue

            # Pool all radius samples across segments
            all_r = np.concatenate([np.asarray(r, dtype=np.float64) for r in seg_radii])
            if len(all_r) < 3:
                continue

            mean_r = np.mean(all_r)
            if mean_r <= 0:
                continue

            cv = np.std(all_r) / mean_r

            # Conduction velocity reduction (Eq. 4 with g-ratio model Eq. 5)
            # v(r) ∝ r * sqrt(-ln(g(r))), g(r) = r/(r+dm), dm = r̄*(1-ḡ)/ḡ
            g_bar = 0.6
            dm = mean_r * (1 - g_bar) / g_bar  # = 2*mean_r/3
            g_r = all_r / (all_r + dm)
            v_r = all_r * np.sqrt(-np.log(g_r))
            v_ideal = mean_r * np.sqrt(-np.log(g_bar))
            # v_eff = 1 / <1/v(r)>
            slowness = 1.0 / v_r
            v_eff = 1.0 / np.mean(slowness)
            slowdown = v_eff / v_ideal  # ratio < 1 means reduction

            all_mean_radii.append(mean_r)
            all_cv.append(cv)
            all_slowdown.append(slowdown)

        logger.info(f"  {npz_path.name}: {len(filtered['labels'])} axons")

    logger.info(f"Total axons: {total_axons}, axons used: {len(all_mean_radii)} "
                f"(length >= {min_length_um} μm)")
    return {
        "all_mean_radii": np.array(all_mean_radii),
        "all_cv": np.array(all_cv),
        "all_slowdown": np.array(all_slowdown),
        "total_axons": total_axons,
        "n_axons_used": len(all_mean_radii),
    }


def main():
    parser = argparse.ArgumentParser(description="Plot radius variability statistics")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/processed/rat/lm"))
    parser.add_argument("--output", type=Path,
                        default=Path("fig/main/radius_variability_stats.svg"))
    parser.add_argument("--min-length", type=float, default=20.0,
                        help="Minimum segment length in μm")
    args = parser.parse_args()

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        return

    d = load_axon_stats(args.data_dir, min_length_um=args.min_length)
    all_r = d["all_mean_radii"]
    all_cv = d["all_cv"]
    all_slow = d["all_slowdown"]

    if len(all_r) == 0:
        logger.error("No axons loaded")
        return

    hist_s = settings.histogram
    font_s = settings.fonts
    line_s = settings.line

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
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
        if np.sum(mask) >= 50:
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
        if np.sum(mask) >= 50:
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

    med_cond = (1 - np.median(all_slow)) * 100
    med_diff = np.median(1.0 / (1.0 + 4.0 * all_cv ** 2))
    med_diff_pct = (1 - med_diff) * 100

    ax_vel.plot(sb_centers[vb], cond_med[vb], color=cond_color, linestyle="-",
                linewidth=line_s["linewidth"], marker="o", markersize=line_s["marker_size"],
                label=f"Conduction velocity\n(median: {med_cond:.1f}%)")
    ax_vel.fill_between(sb_centers[vb], cond_lo[vb], cond_hi[vb],
                        color=cond_color, alpha=line_s["fill_alpha"])

    ax_vel.plot(sb_centers[vb], diff_med_pct[vb], color=diff_color, linestyle="-",
                linewidth=line_s["linewidth"], marker="s", markersize=line_s["marker_size"],
                label=f"Diffusion along-axon\n(median: {med_diff_pct:.1f}%)")
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
