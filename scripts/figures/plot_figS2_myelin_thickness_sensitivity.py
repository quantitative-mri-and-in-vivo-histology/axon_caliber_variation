"""
Supplementary figure: g-ratio model sensitivity of the conduction velocity reduction.

Reproduces the conduction-velocity-reduction analysis of Figure 2e under two
myelin models and shows the curve barely moves — i.e. the result does not depend
on the g-ratio calibration:

  1. Constant g-ratio (current model): a literature mean g-bar = 0.6 calibrates a
     per-axon myelin thickness dm; g then follows the local radius and varies
     along the axon.
  2. Empirical myelin thickness: dm from Lee et al. 2019 (Brain Struct Funct
     224:1469-1488, mouse corpus callosum genu, 3D SEM, Fig 4d), evaluated per
     cross-section.

The reference (straight) cylinder uses the SAME myelin relation as the profile in
each model, so the comparison isolates the effect of radius variation only.

Usage:
    python scripts/figures/plot_figS2_myelin_thickness_sensitivity.py
    python scripts/figures/plot_figS2_myelin_thickness_sensitivity.py --data-dir data/processed/rat/lm
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from axonometry import get_plot_settings, style_axis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

settings = get_plot_settings()

G_BAR = 0.6  # literature mean g-ratio for rat CNS (Chomiak & Hu, PLoS ONE 2009)


def dm_lee(r):
    """Radial myelin thickness [um] from inner radius r [um]. Lee et al. 2019, Fig 4d."""
    d = 2.0 * r                                   # inner diameter
    return 0.35 + 0.006 * d + 0.024 * np.log(d)   # C0*k, C1*k, C2*k


def cond_reduction_pct(all_r, mean_r, dm_profile, dm_ref):
    """Conduction velocity reduction [%] from along-axon caliber variation.

    Rushton velocity v(r) = r * sqrt(-ln g), g = r / (r + dm)
    (Rushton, J. Physiol. 1951, doi:10.1113/jphysiol.1951.sp004655).
    v_eff = 1 / <1/v> over the pooled samples; v_ideal is the straight reference
    cylinder at the along-axon mean radius. Both use the same myelin relation.

    Args:
        all_r: pooled radius samples along the axon [um].
        mean_r: along-axon mean radius [um].
        dm_profile: myelin thickness along the axon (array per sample, or scalar).
        dm_ref: myelin thickness of the reference cylinder (scalar).
    """
    g_r = all_r / (all_r + dm_profile)
    v_r = all_r * np.sqrt(-np.log(g_r))
    g_ideal = mean_r / (mean_r + dm_ref)
    v_ideal = mean_r * np.sqrt(-np.log(g_ideal))
    v_eff = 1.0 / np.mean(1.0 / v_r)
    return (1.0 - v_eff / v_ideal) * 100.0


def load_axon_stats(data_dir: Path, min_length_um: float = 20.0) -> dict:
    """Per-axon mean radius and conduction reduction under both myelin models.

    Mirrors the pooling/filtering of Figure 2e: radii pooled across all segments
    of each axon, axons with total length >= min_length_um kept.
    """
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

            total_len = sum(float(sl) for sl in seg_lengths)
            if total_len < min_length_um:
                continue

            all_r = np.concatenate([np.asarray(r, dtype=np.float64) for r in seg_radii])
            all_r = all_r[all_r > 0]
            if len(all_r) < 3:
                continue

            mean_r = float(np.mean(all_r))
            if mean_r <= 0:
                continue
            min_sample_r = min(min_sample_r, float(all_r.min()))

            # Model 1: constant g-ratio -> constant dm per axon; reference uses same dm.
            dm_const = mean_r * (1 - G_BAR) / G_BAR
            c_cur = cond_reduction_pct(all_r, mean_r, dm_const, dm_const)

            # Model 2: empirical myelin thickness per sample; reference uses same relation.
            dm_prof = dm_lee(all_r)
            c_lee = cond_reduction_pct(all_r, mean_r, dm_prof, dm_lee(mean_r))

            mean_radii.append(mean_r)
            cond_current.append(c_cur)
            cond_lee.append(c_lee)

        logger.info(f"  {npz_path.name}: {len(filtered['labels'])} axons")

    logger.info(f"Total axons: {total_axons}, axons used: {len(mean_radii)} "
                f"(length >= {min_length_um} um)")
    # Sanity check: Lee myelin thickness must stay positive over the data range.
    logger.info(f"Smallest radius sample: {min_sample_r:.3f} um -> "
                f"dm_lee = {dm_lee(min_sample_r):.3f} um (must be > 0)")

    return {
        "mean_radii": np.array(mean_radii),
        "cond_current": np.array(cond_current),
        "cond_lee": np.array(cond_lee),
        "total_axons": total_axons,
        "n_used": len(mean_radii),
    }


def binned_median_iqr(x, vals, edges, min_count=50):
    """Median and 25/75 percentiles of vals within bins of x."""
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


def _clean(arr):
    """List with NaN -> None for JSON."""
    return [None if np.isnan(v) else round(float(v), 5) for v in arr]


def main():
    parser = argparse.ArgumentParser(
        description="g-ratio model sensitivity of the conduction velocity reduction")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/rat/lm"))
    parser.add_argument("--output", type=Path,
                        default=Path("fig/supplementary/gratio_model_sensitivity.svg"))
    parser.add_argument("--min-length", type=float, default=20.0,
                        help="Minimum total axon length in um")
    args = parser.parse_args()

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        return

    d = load_axon_stats(args.data_dir, min_length_um=args.min_length)
    all_r = d["mean_radii"]
    cur = d["cond_current"]
    lee = d["cond_lee"]

    if len(all_r) == 0:
        logger.error("No axons loaded")
        return

    # Same binning as Figure 2e: 30 bins up to p99.5 of mean radius, >= 50 axons/bin.
    x_max = np.percentile(all_r, 99.5)
    n_bins = 30
    edges = np.linspace(0, x_max, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    cur_med, cur_q25, cur_q75 = binned_median_iqr(all_r, cur, edges)
    lee_med, lee_q25, lee_q75 = binned_median_iqr(all_r, lee, edges)

    med_cur_all = float(np.median(cur))
    med_lee_all = float(np.median(lee))

    line_s = settings.line
    cur_color = settings.colors["binary_a"]   # red — current constant-g model
    lee_color = settings.colors["binary_b"]   # blue — empirical myelin (Lee)

    fig, ax = plt.subplots(figsize=(6, 5))

    vb_c = ~np.isnan(cur_med)
    ax.plot(centers[vb_c], cur_med[vb_c], color=cur_color, linestyle="-",
            linewidth=line_s["linewidth"], marker="o", markersize=line_s["marker_size"],
            label=f"Constant myelin thickness\n(median: {med_cur_all:.1f}%)")
    ax.fill_between(centers[vb_c], cur_q25[vb_c], cur_q75[vb_c],
                    color=cur_color, alpha=line_s["fill_alpha"])

    vb_l = ~np.isnan(lee_med)
    ax.plot(centers[vb_l], lee_med[vb_l], color=lee_color, linestyle="-",
            linewidth=line_s["linewidth"], marker="s", markersize=line_s["marker_size"],
            label=f"Empirical model (Lee et al.)\n(median: {med_lee_all:.1f}%)")
    ax.fill_between(centers[vb_l], lee_q25[vb_l], lee_q75[vb_l],
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

    meta = {
        "n_axons_used": d["n_used"],
        "total_axons": d["total_axons"],
        "min_length_um": args.min_length,
        "g_bar": G_BAR,
        "myelin_reference": "Lee et al. 2019, Brain Struct Funct 224:1469-1488, Fig 4d",
        "median_reduction_pct": {
            "constant_g": round(med_cur_all, 4),
            "lee_empirical": round(med_lee_all, 4),
        },
        "bin_centers_um": _clean(centers),
        "constant_g": {"median": _clean(cur_med), "q25": _clean(cur_q25), "q75": _clean(cur_q75)},
        "lee_empirical": {"median": _clean(lee_med), "q25": _clean(lee_q25), "q75": _clean(lee_q75)},
    }
    json_path = args.output.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved to {args.output} (+ .eps, .json)")
    logger.info(f"Median conduction reduction: constant-g = {med_cur_all:.2f}%, "
                f"Lee = {med_lee_all:.2f}% (delta = {med_lee_all - med_cur_all:+.2f} pp)")


if __name__ == "__main__":
    main()
