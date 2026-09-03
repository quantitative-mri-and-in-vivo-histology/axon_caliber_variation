"""
Generate source data for Figure 4 (2D slice vs random sampling, Monte Carlo).

Writes to data/figures/:
    fig_4a_wasserstein_violin.csv   KDE outline of the slice/random Wasserstein
                                    distributions (compact violin geometry)
    fig_4a_wasserstein_stats.csv    per-group mean/median/min/max (violin markers)
    fig_4a_wasserstein_points.csv   the jittered points overlaid on the violins
    fig_4bc_scatter.csv             per-ROI 2D-vs-3D r̄ and r_MRI (slice & random,
                                    median + IQR over MC iterations)
    fig_4bc_stats.csv               per-metric R (mean±std), bias, permutation p
                                    for slice and random sampling

The 190k-value Wasserstein distributions are summarised with matplotlib's own
violin_stats (the same KDE violinplot uses), so the violin reproduces exactly
without storing the raw arrays. Plotting is done by plot_fig4_slice_vs_random.py.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import cbook
from scipy import stats

from axonometry import compute_r_eff
from axonometry.io import load_2d_profiles, load_3d_profiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MIN_AXON_COUNT = 30
BIN_WIDTH_UM = 0.02
MAX_RADIUS_UM = 5.0


def make_bins() -> Tuple[np.ndarray, np.ndarray, float]:
    bin_centers = np.arange(BIN_WIDTH_UM / 2, MAX_RADIUS_UM, BIN_WIDTH_UM)
    bin_width = bin_centers[1] - bin_centers[0]
    bin_edges = np.concatenate([[bin_centers[0] - bin_width / 2], bin_centers + bin_width / 2])
    return bin_centers, bin_edges, bin_width


def compute_per_slice_stats(data: Dict) -> Dict:
    radii, slices, n_slices = data["radii"], data["slice_index"], data["n_slices"]
    r_arith, r_eff, counts, valid_slice_radii = [], [], [], []
    for z in range(n_slices):
        r_z = radii[slices == z]
        if len(r_z) < MIN_AXON_COUNT:
            continue
        counts.append(len(r_z))
        r_arith.append(np.mean(r_z))
        r_eff.append(compute_r_eff(r_z))
        valid_slice_radii.append(r_z)
    return {"r_arith": np.array(r_arith), "r_eff": np.array(r_eff),
            "counts": np.array(counts), "n_valid": len(counts),
            "valid_slice_radii": valid_slice_radii, "pooled_radii": radii}


def load_3d_radii(npz_path: Path) -> np.ndarray:
    return load_3d_profiles(npz_path)["all_radii_um"]


def find_matching_pairs(data_dir: Path) -> List[Tuple[Path, Path, str]]:
    pairs = []
    for sf in sorted(data_dir.glob("*_myelin_slice_profiles.npz")):
        stem = sf.stem.replace("_slice_profiles", "")
        af = data_dir / f"{stem}_axon_profiles.npz"
        parts = stem.replace("_myelin", "").rsplit("_", 1)
        base_name = parts[0] if len(parts) == 2 else stem
        pop = parts[1].upper() if len(parts) == 2 else ""
        sample_name = f"{base_name}_{pop}" if pop else base_name
        if not af.exists():
            logger.warning(f"  No 3D file for {sf.name}, skipping")
            continue
        pairs.append((sf, af, sample_name))
    logger.info(f"Found {len(pairs)} matching 2D/3D pairs")
    return pairs


def compute_wasserstein(radii_sample, cdf_3d, bin_edges, bin_width):
    hist, _ = np.histogram(radii_sample, bins=bin_edges)
    if hist.sum() == 0:
        return np.nan
    cdf_2d = np.cumsum(hist) / hist.sum()
    return np.sum(np.abs(cdf_2d - cdf_3d)) * bin_width


def compute_3d_cdf(radii, bin_edges):
    hist, _ = np.histogram(radii, bins=bin_edges)
    return np.cumsum(hist) / hist.sum()


def run_monte_carlo(roi_stats, roi_3d_cdfs, bin_edges, n_iterations, seed=42):
    n_rois = len(roi_stats)
    bin_width = bin_edges[1] - bin_edges[0]
    slice_mean = np.zeros((n_rois, n_iterations)); slice_reff = np.zeros((n_rois, n_iterations))
    slice_wass = np.zeros((n_rois, n_iterations)); rand_mean = np.zeros((n_rois, n_iterations))
    rand_reff = np.zeros((n_rois, n_iterations)); rand_wass = np.zeros((n_rois, n_iterations))
    rng_slice = np.random.default_rng(seed)
    rng_random = np.random.default_rng(seed + 1)
    for it in range(n_iterations):
        for j in range(n_rois):
            roi = roi_stats[j]
            cdf_3d = roi_3d_cdfs[j]
            slice_idx = rng_slice.integers(0, roi["n_valid"])
            slice_mean[j, it] = roi["r_arith"][slice_idx]
            slice_reff[j, it] = roi["r_eff"][slice_idx]
            r_slice = roi["valid_slice_radii"][slice_idx]
            slice_wass[j, it] = compute_wasserstein(r_slice, cdf_3d, bin_edges, bin_width)
            rand_radii = rng_random.choice(roi["pooled_radii"], size=len(r_slice), replace=True)
            rand_mean[j, it] = np.mean(rand_radii)
            rand_reff[j, it] = compute_r_eff(rand_radii)
            rand_wass[j, it] = compute_wasserstein(rand_radii, cdf_3d, bin_edges, bin_width)
    return ({"mean_radius": slice_mean, "r_eff": slice_reff, "wasserstein": slice_wass},
            {"mean_radius": rand_mean, "r_eff": rand_reff, "wasserstein": rand_wass})


def compute_permutation_pvalue(x_3d, roi_stats, metric, n_perm=10000, seed=42):
    n_rois = len(x_3d)
    rng = np.random.default_rng(seed)
    slice_key = "r_arith" if metric == "mean_radius" else "r_eff"
    r_obs_list = []
    for _ in range(1000):
        y_2d = np.array([roi[slice_key][rng.integers(roi["n_valid"])] for roi in roi_stats])
        r_obs_list.append(stats.pearsonr(x_3d, y_2d)[0])
    r_observed = np.mean(r_obs_list)
    count = 0
    for _ in range(n_perm):
        perm_idx = rng.permutation(n_rois)
        y_2d = np.array([roi[slice_key][rng.integers(roi["n_valid"])] for roi in roi_stats])
        if stats.pearsonr(x_3d[perm_idx], y_2d)[0] >= r_observed:
            count += 1
    return count / n_perm


def main():
    parser = argparse.ArgumentParser(description="Generate Fig 4 source data")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/rat/lm"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/figures"))
    parser.add_argument("--prefix", type=str, default="fig_4")
    parser.add_argument("--radius-type", type=str, default="minor", choices=["minor", "circular"])
    parser.add_argument("--n-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = find_matching_pairs(args.data_dir)
    if not pairs:
        logger.error("No matching 2D/3D pairs found!")
        return

    bin_centers, bin_edges, bin_width = make_bins()
    x_3d = {"mean_radius": [], "r_eff": []}
    roi_stats, roi_3d_cdfs, roi_names = [], [], []
    for sf, af, sn in pairs:
        per_slice = compute_per_slice_stats(load_2d_profiles(sf, args.radius_type))
        if per_slice["n_valid"] < 1:
            continue
        r3d = load_3d_radii(af)
        if len(r3d) == 0:
            continue
        x_3d["mean_radius"].append(float(np.mean(r3d)))
        x_3d["r_eff"].append(float(compute_r_eff(r3d)))
        roi_stats.append(per_slice)
        roi_3d_cdfs.append(compute_3d_cdf(r3d, bin_edges))
        roi_names.append(sn)
    x_3d = {k: np.array(v) for k, v in x_3d.items()}
    n_rois = len(roi_names)
    logger.info(f"{n_rois} ROIs ready for Monte Carlo")

    logger.info(f"Running Monte Carlo ({args.n_iterations} iterations)...")
    slice_res, rand_res = run_monte_carlo(roi_stats, roi_3d_cdfs, bin_edges,
                                          args.n_iterations, seed=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Panel (a): Wasserstein violins (compact KDE outline) + jitter ─────
    slice_pooled = slice_res["wasserstein"].flatten()
    random_pooled = rand_res["wasserstein"].flatten()
    vpstats = cbook.violin_stats([slice_pooled, random_pooled], points=100)
    groups = ["slice", "random"]
    viol_rows, stat_rows = [], []
    for pos, (g, vs) in enumerate(zip(groups, vpstats), start=1):
        for c, v in zip(vs["coords"], vs["vals"]):
            viol_rows.append({"group": g, "coord": c, "density": v})
        stat_rows.append({"group": g, "position": pos, "mean": vs["mean"],
                          "median": vs["median"], "vmin": vs["min"], "vmax": vs["max"]})
    pd.DataFrame(viol_rows).to_csv(args.out_dir / f"{args.prefix}a_wasserstein_violin.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(args.out_dir / f"{args.prefix}a_wasserstein_stats.csv", index=False)

    # Jittered overlay points (replicate the original rng sequence, seed 42)
    rng = np.random.default_rng(42)
    n_show = min(500, len(slice_pooled))
    idx_s = rng.choice(len(slice_pooled), n_show, replace=False)
    idx_r = rng.choice(len(random_pooled), n_show, replace=False)
    x_s = 1 + rng.uniform(-0.15, 0.15, n_show)
    x_r = 2 + rng.uniform(-0.15, 0.15, n_show)
    pts = pd.DataFrame({
        "group": ["slice"] * n_show + ["random"] * n_show,
        "x": np.concatenate([x_s, x_r]),
        "y": np.concatenate([slice_pooled[idx_s], random_pooled[idx_r]]),
    })
    pts.to_csv(args.out_dir / f"{args.prefix}a_wasserstein_points.csv", index=False)

    # ── Panels (b)/(c): scatter + stats ───────────────────────────────────
    scatter = {"roi": roi_names}
    stat_rows = []
    # permutation p-values for the random arm (both metrics), replicating main
    rng_perm = np.random.default_rng(args.seed + 200)
    y_rand = {m: np.mean(rand_res[m], axis=1) for m in ("mean_radius", "r_eff")}
    r_rand_obs = {m: stats.pearsonr(x_3d[m], y_rand[m])[0] for m in ("mean_radius", "r_eff")}
    p_rand_count = {"mean_radius": 0, "r_eff": 0}
    for _ in range(args.n_iterations):
        perm = rng_perm.permutation(n_rois)
        for m in ("mean_radius", "r_eff"):
            if stats.pearsonr(x_3d[m][perm], y_rand[m])[0] >= r_rand_obs[m]:
                p_rand_count[m] += 1
    p_slice_seed = {"mean_radius": args.seed + 100, "r_eff": args.seed + 102}

    for m in ("mean_radius", "r_eff"):
        sr, rr = slice_res[m], rand_res[m]
        scatter[f"{m}_3d"] = x_3d[m]
        scatter[f"{m}_slice_median"] = np.median(sr, axis=1)
        scatter[f"{m}_slice_q25"] = np.percentile(sr, 25, axis=1)
        scatter[f"{m}_slice_q75"] = np.percentile(sr, 75, axis=1)
        scatter[f"{m}_random_median"] = np.median(rr, axis=1)
        scatter[f"{m}_random_q25"] = np.percentile(rr, 25, axis=1)
        scatter[f"{m}_random_q75"] = np.percentile(rr, 75, axis=1)

        r_slice_iter = np.array([stats.pearsonr(x_3d[m], sr[:, it])[0] for it in range(sr.shape[1])])
        r_rand_iter = np.array([stats.pearsonr(x_3d[m], rr[:, it])[0] for it in range(rr.shape[1])])
        bias_slice = float(np.mean((scatter[f"{m}_slice_median"] - x_3d[m]) / x_3d[m]) * 100)
        bias_random = float(np.mean((scatter[f"{m}_random_median"] - x_3d[m]) / x_3d[m]) * 100)
        p_slice = compute_permutation_pvalue(x_3d[m], roi_stats, m,
                                             n_perm=args.n_iterations, seed=p_slice_seed[m])
        stat_rows.append({
            "metric": m,
            "r_slice_mean": float(np.mean(r_slice_iter)), "r_slice_std": float(np.std(r_slice_iter)),
            "r_random_mean": float(np.mean(r_rand_iter)), "r_random_std": float(np.std(r_rand_iter)),
            "bias_slice": bias_slice, "bias_random": bias_random,
            "p_slice": p_slice, "p_random": p_rand_count[m] / args.n_iterations,
        })

    pd.DataFrame(scatter).to_csv(args.out_dir / f"{args.prefix}bc_scatter.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(args.out_dir / f"{args.prefix}bc_stats.csv", index=False)

    logger.info(f"Wrote {args.prefix}* source data to {args.out_dir}/  "
                f"(slice R̄={stat_rows[0]['r_slice_mean']:.2f}, "
                f"r_MRI={stat_rows[1]['r_slice_mean']:.2f})")


if __name__ == "__main__":
    main()
