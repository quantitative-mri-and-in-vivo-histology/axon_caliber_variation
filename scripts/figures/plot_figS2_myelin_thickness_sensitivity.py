"""
Plot Figure S2 (myelin-thickness model sensitivity of the conduction-velocity
reduction) from precomputed source data.

Reads the CSVs written by gen_data_figS2_myelin_thickness_sensitivity.py and
draws the two model curves (constant myelin thickness vs empirical Lee et al.
myelin) with IQR bands. Saves SVG and EPS.

Usage:
    python scripts/figures/plot_figS2_myelin_thickness_sensitivity.py
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
    parser = argparse.ArgumentParser(description="Plot Fig S2 from source data")
    parser.add_argument("--data-dir", type=Path, default=Path("data/figures"))
    parser.add_argument("--output", type=Path,
                        default=Path("fig/supplementary/fig_s2.svg"))
    args = parser.parse_args()

    red = pd.read_csv(args.data_dir / "fig_s2_reduction.csv")
    med = pd.read_csv(args.data_dir / "fig_s2_summary.csv").set_index("model")["median_pct"]

    line_s = settings.line
    cur_color = settings.colors["binary_a"]   # red — constant myelin thickness
    lee_color = settings.colors["binary_b"]   # blue — empirical (Lee)
    x = red["mean_radius_um"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, red["cond_current_median"], color=cur_color, linestyle="-",
            linewidth=line_s["linewidth"], marker="o", markersize=line_s["marker_size"],
            label=f"Constant myelin thickness\n(median: {med['constant']:.1f}%)")
    ax.fill_between(x, red["cond_current_q25"], red["cond_current_q75"],
                    color=cur_color, alpha=line_s["fill_alpha"])
    ax.plot(x, red["cond_lee_median"], color=lee_color, linestyle="-",
            linewidth=line_s["linewidth"], marker="s", markersize=line_s["marker_size"],
            label=f"Empirical model (Lee et al.)\n(median: {med['lee']:.1f}%)")
    ax.fill_between(x, red["cond_lee_q25"], red["cond_lee_q75"],
                    color=lee_color, alpha=line_s["fill_alpha"])

    style_axis(ax, xlabel=r"Along-axon mean radius [μm]",
               ylabel="Conduction velocity reduction [%]")
    ax.set_xlim(0.1, 0.6)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=settings.fonts["legend_size"])
    ax.set_box_aspect(1)
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=settings.figure["dpi"], bbox_inches="tight")
    plt.savefig(args.output.with_suffix(".eps"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved to {args.output} (+ .eps)")


if __name__ == "__main__":
    main()
