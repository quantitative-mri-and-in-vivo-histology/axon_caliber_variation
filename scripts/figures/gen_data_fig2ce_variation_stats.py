"""
Generate source data for Figure 2c-e (radius variability statistics).

Pools per-axon radius samples from the 3D axon profiles and computes the
values plotted in Fig 2c-e, writing them as CSV source-data files:

  fig_2c_cov_histogram.csv    CoV histogram (bin centers + counts)
  fig_2c_summary.csv          mean / median CoV (the vertical reference lines)
  fig_2d_cov_vs_radius.csv    CoV vs along-axon mean radius (binned median + IQR)
  fig_2e_reduction_vs_radius.csv  conduction-velocity & diffusion reduction vs radius

Plotting is done separately by plot_fig2ce_variation_stats.py, which reads
these CSVs.

Usage:
    python scripts/figures/gen_data_fig2ce_variation_stats.py
    python scripts/figures/gen_data_fig2ce_variation_stats.py --data-dir data/processed/rat/lm
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from axonometry import get_plot_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

settings = get_plot_settings()

MIN_AXONS_PER_BIN = 50
N_RADIUS_BINS = 30
G_BAR = 0.6  # literature mean g-ratio for rat CNS (Chomiak & Hu, PLoS ONE 2009)


def load_axon_stats(data_dir: Path, min_length_um: float = 20.0) -> dict:
    """Per-axon mean radius, CoV, and conduction/diffusion reduction.

    Radius samples from all segments of each axon are pooled; only axons with
    total length >= min_length_um are kept.
    """
    from axonometry.io import load_3d_profiles

    all_mean_radii, all_cv, all_cond_reduction, all_diff_reduction = [], [], [], []
    total_axons = 0

    for npz_path in sorted(data_dir.glob("*_axon_profiles.npz")):
        filtered = load_3d_profiles(npz_path)
        total_axons += len(filtered["labels"])

        for i in range(len(filtered["labels"])):
            seg_radii = filtered["segment_radii_um"][i]
            seg_lengths = filtered["segment_lengths_um"][i]
            if not seg_radii:
                continue

            total_len = sum(float(sl) for sl in seg_lengths)
            if total_len < min_length_um:
                continue

            all_r = np.concatenate([np.asarray(r, dtype=np.float64) for r in seg_radii])
            all_r = all_r[all_r > 0]
            if len(all_r) < 3:
                continue

            mean_r = np.mean(all_r)
            if mean_r <= 0:
                continue

            cv = np.std(all_r) / mean_r

            # Conduction velocity: v(r) ∝ r * sqrt(-ln(g(r)))
            # (Rushton, J. Physiol. 1951, doi:10.1113/jphysiol.1951.sp004655)
            # g-ratio model: g(r) = r/(r+dm), dm = r̄*(1-ḡ)/ḡ, ḡ = 0.6
            # (Chomiak & Hu, PLoS ONE 2009, doi:10.1371/journal.pone.0007754)
            dm = mean_r * (1 - G_BAR) / G_BAR  # = 2*mean_r/3
            g_r = all_r / (all_r + dm)
            v_r = all_r * np.sqrt(-np.log(g_r))
            v_ideal = mean_r * np.sqrt(-np.log(G_BAR))
            v_eff = 1.0 / np.mean(1.0 / v_r)
            cond_reduction_pct = (1 - v_eff / v_ideal) * 100

            # Along-axon diffusion reduction: 4*CV² / (1 + 4*CV²)
            # (Lee et al., Commun. Biol. 2020, doi:10.1038/s42003-020-1050-x)
            diff_reduction_pct = 4.0 * cv**2 / (1.0 + 4.0 * cv**2) * 100

            all_mean_radii.append(mean_r)
            all_cv.append(cv)
            all_cond_reduction.append(cond_reduction_pct)
            all_diff_reduction.append(diff_reduction_pct)

        logger.info(f"  {npz_path.name}: {len(filtered['labels'])} axons")

    logger.info(f"Total axons: {total_axons}, axons used: {len(all_mean_radii)} "
                f"(length >= {min_length_um} μm)")
    return {
        "all_mean_radii": np.array(all_mean_radii),
        "all_cv": np.array(all_cv),
        "all_cond_reduction": np.array(all_cond_reduction),
        "all_diff_reduction": np.array(all_diff_reduction),
    }


def _binned_median_iqr(x, vals, edges, min_count=MIN_AXONS_PER_BIN):
    """Median and 25/75 percentiles of vals within bins of x (NaN if under-filled)."""
    n = len(edges) - 1
    med = np.full(n, np.nan)
    q25 = np.full(n, np.nan)
    q75 = np.full(n, np.nan)
    for i in range(n):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if np.sum(m) >= min_count:
            med[i] = np.median(vals[m])
            q25[i] = np.percentile(vals[m], 25)
            q75[i] = np.percentile(vals[m], 75)
    return med, q25, q75


def main():
    parser = argparse.ArgumentParser(description="Generate Fig 2c-e source data")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/rat/lm"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/figures"))
    parser.add_argument("--min-length", type=float, default=20.0,
                        help="Minimum total axon length in μm")
    args = parser.parse_args()

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        return

    d = load_axon_stats(args.data_dir, min_length_um=args.min_length)
    all_r = d["all_mean_radii"]
    all_cv = d["all_cv"]
    all_cond = d["all_cond_reduction"]
    all_diff = d["all_diff_reduction"]
    if len(all_r) == 0:
        logger.error("No axons loaded")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- (c) CoV histogram + reference lines ---
    counts, edges = np.histogram(all_cv, bins=settings.histogram["bins"])
    centers = (edges[:-1] + edges[1:]) / 2
    pd.DataFrame({"cov_bin_center": centers, "count": counts}).to_csv(
        args.out_dir / "fig_2c_cov_histogram.csv", index=False)
    pd.DataFrame({"statistic": ["mean", "median"],
                  "cov": [float(np.mean(all_cv)), float(np.median(all_cv))]}).to_csv(
        args.out_dir / "fig_2c_summary.csv", index=False)

    # Shared radius bins for (d) and (e)
    x_max = np.percentile(all_r, 99.5)
    edges_r = np.linspace(0, x_max, N_RADIUS_BINS + 1)
    centers_r = (edges_r[:-1] + edges_r[1:]) / 2

    # --- (d) CoV vs radius ---
    cov_med, cov_q25, cov_q75 = _binned_median_iqr(all_r, all_cv, edges_r)
    valid = ~np.isnan(cov_med)
    pd.DataFrame({
        "mean_radius_um": centers_r[valid],
        "cov_median": cov_med[valid],
        "cov_q25": cov_q25[valid],
        "cov_q75": cov_q75[valid],
    }).to_csv(args.out_dir / "fig_2d_cov_vs_radius.csv", index=False)

    # --- (e) Reduction vs radius ---
    cond_med, cond_q25, cond_q75 = _binned_median_iqr(all_r, all_cond, edges_r)
    diff_med, diff_q25, diff_q75 = _binned_median_iqr(all_r, all_diff, edges_r)
    vb = ~np.isnan(cond_med)
    pd.DataFrame({
        "mean_radius_um": centers_r[vb],
        "cond_median_pct": cond_med[vb], "cond_q25_pct": cond_q25[vb], "cond_q75_pct": cond_q75[vb],
        "diff_median_pct": diff_med[vb], "diff_q25_pct": diff_q25[vb], "diff_q75_pct": diff_q75[vb],
    }).to_csv(args.out_dir / "fig_2e_reduction_vs_radius.csv", index=False)
    # Overall (all-axon) medians shown in the panel-e legend
    pd.DataFrame({"statistic": ["cond_median_pct", "diff_median_pct"],
                  "value": [float(np.median(all_cond)), float(np.median(all_diff))]}).to_csv(
        args.out_dir / "fig_2e_summary.csv", index=False)

    logger.info(f"Wrote Fig 2c-e source data to {args.out_dir}/ "
                f"(N={len(all_cv)} axons, mean CoV={np.mean(all_cv):.3f}, "
                f"median cond. reduction={np.median(all_cond):.1f}%)")


if __name__ == "__main__":
    main()
