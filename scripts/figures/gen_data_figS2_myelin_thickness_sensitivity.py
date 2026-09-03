"""
Generate source data for Figure S2 (g-ratio / myelin-thickness model sensitivity
of the conduction-velocity reduction).

Computes the per-axon conduction-velocity reduction under two myelin models and
writes the binned curves + overall medians:

    data/figures/fig_s2_reduction.csv   mean_radius_um + cond_current_/cond_lee_ median & IQR
    data/figures/fig_s2_summary.csv     overall median reduction per model

Models:
  1. Constant myelin thickness: literature mean g-ratio (g_bar = 0.6) sets a
     per-axon dm; g then follows the local radius.
  2. Empirical myelin thickness: dm from Lee et al. 2019 (Brain Struct Funct
     224:1469-1488, Fig 4d), evaluated per cross-section.
The reference cylinder uses the same relation as the profile in each model.

Plotting is done by plot_figS2_myelin_thickness_sensitivity.py.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MIN_AXONS_PER_BIN = 50
N_RADIUS_BINS = 30
G_BAR = 0.6  # literature mean g-ratio for rat CNS (Chomiak & Hu, PLoS ONE 2009)


def dm_lee(r):
    """Radial myelin thickness [um] from inner radius r [um]. Lee et al. 2019, Fig 4d."""
    d = 2.0 * r
    return 0.35 + 0.006 * d + 0.024 * np.log(d)


def cond_reduction_pct(all_r, mean_r, dm_profile, dm_ref):
    """Conduction velocity reduction [%] from along-axon caliber variation.

    Rushton velocity v(r) = r * sqrt(-ln g), g = r/(r+dm); v_eff = 1/<1/v>;
    reference cylinder uses the same myelin relation (dm_ref).
    """
    g_r = all_r / (all_r + dm_profile)
    v_r = all_r * np.sqrt(-np.log(g_r))
    g_ideal = mean_r / (mean_r + dm_ref)
    v_ideal = mean_r * np.sqrt(-np.log(g_ideal))
    v_eff = 1.0 / np.mean(1.0 / v_r)
    return (1.0 - v_eff / v_ideal) * 100.0


def load_axon_stats(data_dir: Path, min_length_um: float = 20.0) -> dict:
    from axonometry.io import load_3d_profiles

    mean_radii, cond_current, cond_lee = [], [], []
    total_axons = 0
    min_sample_r = np.inf
    for npz_path in sorted(data_dir.glob("*_axon_profiles.npz")):
        filtered = load_3d_profiles(npz_path)
        total_axons += len(filtered["labels"])
        for i in range(len(filtered["labels"])):
            seg_radii = filtered["segment_radii_um"][i]
            seg_lengths = filtered["segment_lengths_um"][i]
            if not seg_radii:
                continue
            if sum(float(sl) for sl in seg_lengths) < min_length_um:
                continue
            all_r = np.concatenate([np.asarray(r, dtype=np.float64) for r in seg_radii])
            all_r = all_r[all_r > 0]
            if len(all_r) < 3:
                continue
            mean_r = float(np.mean(all_r))
            if mean_r <= 0:
                continue
            min_sample_r = min(min_sample_r, float(all_r.min()))
            dm_const = mean_r * (1 - G_BAR) / G_BAR
            cond_current.append(cond_reduction_pct(all_r, mean_r, dm_const, dm_const))
            dm_prof = dm_lee(all_r)
            cond_lee.append(cond_reduction_pct(all_r, mean_r, dm_prof, dm_lee(mean_r)))
            mean_radii.append(mean_r)
        logger.info(f"  {npz_path.name}: {len(filtered['labels'])} axons")

    logger.info(f"Total axons: {total_axons}, axons used: {len(mean_radii)} "
                f"(length >= {min_length_um} um); smallest radius {min_sample_r:.3f} um "
                f"-> dm_lee = {dm_lee(min_sample_r):.3f} um")
    return {"mean_radii": np.array(mean_radii),
            "cond_current": np.array(cond_current),
            "cond_lee": np.array(cond_lee)}


def _binned(x, vals, edges):
    n = len(edges) - 1
    med = np.full(n, np.nan); q25 = np.full(n, np.nan); q75 = np.full(n, np.nan)
    for i in range(n):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if np.sum(m) >= MIN_AXONS_PER_BIN:
            med[i] = np.median(vals[m])
            q25[i] = np.percentile(vals[m], 25)
            q75[i] = np.percentile(vals[m], 75)
    return med, q25, q75


def main():
    parser = argparse.ArgumentParser(description="Generate Fig S2 source data")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/rat/lm"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/figures"))
    parser.add_argument("--min-length", type=float, default=20.0)
    args = parser.parse_args()

    d = load_axon_stats(args.data_dir, min_length_um=args.min_length)
    all_r, cur, lee = d["mean_radii"], d["cond_current"], d["cond_lee"]
    if len(all_r) == 0:
        logger.error("No axons loaded")
        return

    x_max = np.percentile(all_r, 99.5)
    edges = np.linspace(0, x_max, N_RADIUS_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    cur_med, cur_q25, cur_q75 = _binned(all_r, cur, edges)
    lee_med, lee_q25, lee_q75 = _binned(all_r, lee, edges)
    vb = ~np.isnan(cur_med)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "mean_radius_um": centers[vb],
        "cond_current_median": cur_med[vb], "cond_current_q25": cur_q25[vb], "cond_current_q75": cur_q75[vb],
        "cond_lee_median": lee_med[vb], "cond_lee_q25": lee_q25[vb], "cond_lee_q75": lee_q75[vb],
    }).to_csv(args.out_dir / "fig_s2_reduction.csv", index=False)
    pd.DataFrame({"model": ["constant", "lee"],
                  "median_pct": [float(np.median(cur)), float(np.median(lee))]}).to_csv(
        args.out_dir / "fig_s2_summary.csv", index=False)

    logger.info(f"Wrote fig_s2_* source data to {args.out_dir}/ "
                f"(constant {np.median(cur):.2f}%, Lee {np.median(lee):.2f}%)")


if __name__ == "__main__":
    main()
