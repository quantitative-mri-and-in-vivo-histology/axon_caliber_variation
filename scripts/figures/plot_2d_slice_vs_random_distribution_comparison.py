"""
Monte Carlo comparison: 2D slice sampling vs i.i.d. sampling.

Compares two approaches for estimating 3D population statistics from 2D samples:
1. 2D slice sampling: Randomly pick one physical 2D slice per ROI
2. i.i.d. sampling: Sample mean_axons_per_slice radii from the pooled 2D distribution

Both methods use the same underlying 2D radii (with minor axis bias from
ellipse fitting), ensuring a fair comparison of sampling strategies.

Creates a 1x3 figure showing:
(a) Wasserstein distance violin: slice sampling vs i.i.d. sampling
(b) Mean radius scatter: 2D vs 3D
(c) Effective radius scatter: 2D vs 3D

Median and IQR are computed over Monte Carlo iterations.

Usage:
    python scripts/exploratory/2d_vs_3d/plot_montecarlo_2d_vs_3d_correlation.py \
        --data-dir data/processed/rat/lm \
        --output fig/montecarlo_2d_vs_3d_correlation.png \
        --n-iterations 100
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy import stats

from axonometry import get_plot_settings, style_axis, compute_r_eff

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

settings = get_plot_settings()

# ── Constants ──────────────────────────────────────────────────────────────
MIN_AXON_COUNT = 30        # Min axons per slice for valid statistics
BIN_WIDTH_UM = 0.02
MAX_RADIUS_UM = 5.0


# ── Data loading ──────────────────────────────────────────────────────────

def load_and_filter_2d(npz_path: Path, radius_type: str = "circular") -> Dict:
    """Load 2D slice profiles with quality and label exclusion filters."""
    from axonometry.io import load_2d_profiles
    return load_2d_profiles(npz_path, radius_type=radius_type)


def compute_per_slice_stats(data: Dict) -> Dict:
    """Compute per-slice mean radius, r_eff, and counts."""
    radii = data["radii"]
    slices = data["slice_index"]
    n_slices = data["n_slices"]

    r_arith, r_eff, counts = [], [], []
    valid_slice_radii = []  # store radii per valid slice for MC sampling

    for z in range(n_slices):
        r_z = radii[slices == z]
        if len(r_z) < MIN_AXON_COUNT:
            continue
        counts.append(len(r_z))
        r_arith.append(np.mean(r_z))
        r2 = np.mean(r_z ** 2)
        r6 = np.mean(r_z ** 6)
        r_eff.append((r6 / r2) ** 0.25 if r2 > 0 else np.nan)
        valid_slice_radii.append(r_z)

    return {
        "r_arith": np.array(r_arith),
        "r_eff": np.array(r_eff),
        "counts": np.array(counts),
        "n_valid": len(counts),
        "valid_slice_radii": valid_slice_radii,
        "pooled_radii": radii,
        "mean_axons_per_slice": int(np.mean(counts)) if counts else 0,
    }


def load_3d_radii(npz_path: Path) -> np.ndarray:
    """Load pooled 3D radii with label exclusion."""
    from axonometry.io import load_3d_profiles
    d = load_3d_profiles(npz_path)
    return d['all_radii_um']



# ── File matching ─────────────────────────────────────────────────────────

def find_matching_pairs(data_dir: Path) -> List[Tuple[Path, Path, str]]:
    """
    Find matching 2D slice / 3D axon profile pairs.

    The new canonical format has separate files per population (CC/CG),
    so no population JSON is needed.

    Returns list of (slice_file, axon_file, sample_name) tuples.
    """
    pairs = []
    for sf in sorted(data_dir.glob("*_myelin_slice_profiles.npz")):
        stem = sf.stem.replace("_slice_profiles", "")  # e.g. sham_25_ipsi_cc_myelin
        af = data_dir / f"{stem}_axon_profiles.npz"

        parts = stem.replace("_myelin", "").rsplit("_", 1)  # ['sham_25_ipsi', 'cc']
        base_name = parts[0] if len(parts) == 2 else stem
        pop = parts[1].upper() if len(parts) == 2 else ""
        sample_name = f"{base_name}_{pop}" if pop else base_name

        if not af.exists():
            logger.warning(f"  No 3D file for {sf.name}, skipping")
            continue

        pairs.append((sf, af, sample_name))

    logger.info(f"Found {len(pairs)} matching 2D/3D pairs")
    return pairs


# ── Monte Carlo ───────────────────────────────────────────────────────────

def compute_wasserstein(radii_sample: np.ndarray, cdf_3d: np.ndarray,
                        bin_centers: np.ndarray, bin_width: float) -> float:
    """Compute Wasserstein distance between a 2D sample and 3D CDF."""
    bin_edges = np.concatenate(
        [[bin_centers[0] - bin_width / 2], bin_centers + bin_width / 2]
    )
    hist, _ = np.histogram(radii_sample, bins=bin_edges)
    if hist.sum() == 0:
        return np.nan
    cdf_2d = np.cumsum(hist) / hist.sum()
    return np.sum(np.abs(cdf_2d - cdf_3d)) * bin_width


def compute_3d_cdf(radii: np.ndarray, bin_centers: np.ndarray) -> np.ndarray:
    """Compute 3D CDF at given bin centers."""
    sorted_r = np.sort(radii)
    cdf_y = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
    return np.interp(bin_centers, sorted_r, cdf_y, left=0, right=1)


def run_monte_carlo(roi_stats: List[Dict], roi_3d_cdfs: List[np.ndarray],
                    bin_centers: np.ndarray, n_iterations: int,
                    seed: int = 42) -> Tuple[Dict, Dict]:
    """
    Run Monte Carlo simulation for both 2D slice sampling and i.i.d. sampling.

    Returns:
        Tuple of (slice_results, iid_results), each with keys:
        mean_radius, r_eff, wasserstein — arrays of shape (n_rois, n_iterations)
    """
    n_rois = len(roi_stats)
    bin_width = bin_centers[1] - bin_centers[0]

    slice_mean = np.zeros((n_rois, n_iterations))
    slice_reff = np.zeros((n_rois, n_iterations))
    slice_wass = np.zeros((n_rois, n_iterations))
    iid_mean = np.zeros((n_rois, n_iterations))
    iid_reff = np.zeros((n_rois, n_iterations))
    iid_wass = np.zeros((n_rois, n_iterations))

    rng_slice = np.random.default_rng(seed)
    rng_iid = np.random.default_rng(seed + 1)

    for it in range(n_iterations):
        for j in range(n_rois):
            roi = roi_stats[j]
            cdf_3d = roi_3d_cdfs[j]

            # 2D slice sampling: pick one random valid slice
            slice_idx = rng_slice.integers(0, roi["n_valid"])
            slice_mean[j, it] = roi["r_arith"][slice_idx]
            slice_reff[j, it] = roi["r_eff"][slice_idx]

            # Wasserstein for slice
            r_slice = roi["valid_slice_radii"][slice_idx]
            slice_wass[j, it] = compute_wasserstein(r_slice, cdf_3d, bin_centers, bin_width)

            # Random sampling: sample same N radii as the drawn slice
            n_samples = len(r_slice)
            iid_radii = rng_iid.choice(roi["pooled_radii"], size=n_samples, replace=True)
            iid_mean[j, it] = np.mean(iid_radii)
            iid_reff[j, it] = compute_r_eff(iid_radii)
            iid_wass[j, it] = compute_wasserstein(iid_radii, cdf_3d, bin_centers, bin_width)

    return (
        {"mean_radius": slice_mean, "r_eff": slice_reff, "wasserstein": slice_wass},
        {"mean_radius": iid_mean, "r_eff": iid_reff, "wasserstein": iid_wass},
    )


def compute_permutation_pvalue(x_3d: np.ndarray, roi_stats: List[Dict],
                                metric: str, n_perm: int = 10000,
                                seed: int = 42) -> float:
    """
    Compute permutation p-value for correlation.

    Shuffles 3D labels AND picks new random slices for each permutation.
    """
    n_rois = len(x_3d)
    rng = np.random.default_rng(seed)
    slice_key = "r_arith" if metric == "mean_radius" else "r_eff"

    # Observed R (mean across many random slice picks)
    n_obs_iter = 1000
    r_obs_list = []
    for _ in range(n_obs_iter):
        y_2d = np.array([
            roi[slice_key][rng.integers(roi["n_valid"])] for roi in roi_stats
        ])
        r, _ = stats.pearsonr(x_3d, y_2d)
        r_obs_list.append(r)
    r_observed = np.mean(r_obs_list)

    # Permutation test
    count = 0
    for _ in range(n_perm):
        perm_idx = rng.permutation(n_rois)
        y_2d = np.array([
            roi[slice_key][rng.integers(roi["n_valid"])] for roi in roi_stats
        ])
        r_perm, _ = stats.pearsonr(x_3d[perm_idx], y_2d)
        if r_perm >= r_observed:
            count += 1

    return count / n_perm


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_wasserstein_violin(ax, slice_wass: np.ndarray, iid_wass: np.ndarray):
    """Panel (a): Wasserstein distance violin comparison."""
    slice_pooled = slice_wass.flatten()
    iid_pooled = iid_wass.flatten()

    color_slice = settings.colors["binary_a"]
    color_iid = settings.colors["binary_b"]

    parts = ax.violinplot(
        [slice_pooled, iid_pooled], positions=[1, 2],
        showmeans=False, showmedians=True, widths=0.7,
    )

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor([color_slice, color_iid][i])
        body.set_edgecolor("black")
        body.set_alpha(0.7)

    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_edgecolor("black")
            parts[key].set_linewidth(1.5)

    rng = np.random.default_rng(42)
    n_show = min(500, len(slice_pooled))
    idx_s = rng.choice(len(slice_pooled), n_show, replace=False)
    idx_i = rng.choice(len(iid_pooled), n_show, replace=False)

    ax.scatter(1 + rng.uniform(-0.15, 0.15, n_show), slice_pooled[idx_s],
               alpha=0.3, s=3, color=color_slice, zorder=0)
    ax.scatter(2 + rng.uniform(-0.15, 0.15, n_show), iid_pooled[idx_i],
               alpha=0.3, s=3, color=color_iid, zorder=0)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        ["2D slice-wise\nsampling", "2D random\nsampling"],
        fontsize=settings.fonts["tick_size"] - 1,
    )
    style_axis(ax, ylabel="Wasserstein distance [μm]")
    ax.set_ylim(bottom=0)
    ax.set_box_aspect(1)


def plot_scatter_comparison(ax, x_3d: np.ndarray,
                            actual_results: np.ndarray,
                            fake_results: np.ndarray, metric: str,
                            p_actual: Optional[float] = None,
                            p_fake: Optional[float] = None):
    """Panel (b)/(c): 2D vs 3D scatter comparing slice and i.i.d. sampling."""
    font_s = settings.fonts
    err_s = settings.error_bars

    color_actual = settings.colors["binary_a"]
    color_fake = settings.colors["binary_b"]

    n_iterations = actual_results.shape[1]

    # Compute R per MC iteration
    r_actual_iter = np.array([
        stats.pearsonr(x_3d, actual_results[:, it])[0]
        for it in range(n_iterations)
    ])
    r_fake_iter = np.array([
        stats.pearsonr(x_3d, fake_results[:, it])[0]
        for it in range(n_iterations)
    ])

    r_actual_mean = np.mean(r_actual_iter)
    r_actual_std = np.std(r_actual_iter)
    r_fake_mean = np.mean(r_fake_iter)
    r_fake_std = np.std(r_fake_iter)

    # Mean ± std over iterations per ROI
    actual_mean = np.mean(actual_results, axis=1)
    actual_std = np.std(actual_results, axis=1)
    fake_mean = np.mean(fake_results, axis=1)
    fake_std = np.std(fake_results, axis=1)

    ax.errorbar(x_3d, actual_mean, yerr=actual_std, fmt="o", color=color_actual,
                markersize=6, capsize=err_s["capsize"], capthick=err_s["capthick"],
                elinewidth=err_s["linewidth"], markeredgecolor="black",
                markeredgewidth=0.5, alpha=0.8, label="2D slice-wise sampling", zorder=10)

    ax.errorbar(x_3d, fake_mean, yerr=fake_std, fmt="x", color=color_fake,
                markersize=5, capsize=err_s["capsize"], capthick=err_s["capthick"],
                elinewidth=err_s["linewidth"], markeredgecolor=color_fake,
                markeredgewidth=1.5, alpha=0.8, label="2D random sampling", zorder=10)

    all_vals = np.concatenate([x_3d, actual_mean, fake_mean])
    lo = np.nanmin(all_vals) * 0.95
    hi = np.nanmax(all_vals) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1.5, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    if metric == "mean_radius":
        xlabel = r"$\bar{r}$ (3D) [μm]"
        ylabel = r"$\bar{r}$ (2D) [μm]"
    else:
        xlabel = r"$r_{\mathrm{MRI}}$ (3D) [μm]"
        ylabel = r"$r_{\mathrm{MRI}}$ (2D) [μm]"

    style_axis(ax, xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc="upper left", fontsize=font_s["legend_size"])

    def _fmt_p(p):
        return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

    text_actual = f"R = {r_actual_mean:.2f}±{r_actual_std:.2f}"
    if p_actual is not None:
        text_actual += f", {_fmt_p(p_actual)}"
    ax.text(0.04, 0.78, text_actual, transform=ax.transAxes,
            fontsize=font_s["legend_size"], ha="left", va="top", color=color_actual)

    text_fake = f"R = {r_fake_mean:.2f}±{r_fake_std:.2f}"
    if p_fake is not None:
        text_fake += f", {_fmt_p(p_fake)}"
    ax.text(0.04, 0.72, text_fake, transform=ax.transAxes,
            fontsize=font_s["legend_size"], ha="left", va="top", color=color_fake)

    ax.set_box_aspect(1)

    return r_actual_mean, r_fake_mean, r_actual_std, r_fake_std


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo comparison: 2D slice sampling vs i.i.d. sampling"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/rat/lm"))
    parser.add_argument("--output", type=Path,
                        default=Path("fig/supplementary/montecarlo_2d_vs_3d_correlation.png"))
    parser.add_argument("--n-iterations", type=int, default=100)
    parser.add_argument("--radius-type", type=str, default="circular",
                        choices=["circular", "minor"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Monte Carlo: 2D Slice Sampling vs i.i.d. Sampling")
    logger.info("=" * 70)

    # Find matching pairs
    all_pairs = find_matching_pairs(args.data_dir)
    if not all_pairs:
        logger.error("No matching 2D/3D pairs found!")
        return

    # Load and process each ROI
    x_3d_mean, x_3d_reff = [], []
    roi_stats = []
    roi_3d_cdfs = []

    bin_centers = np.arange(BIN_WIDTH_UM / 2, MAX_RADIUS_UM, BIN_WIDTH_UM)

    for sf, af, sn in all_pairs:
        logger.info(f"Processing {sn}...")

        # 2D data
        data_2d = load_and_filter_2d(sf, args.radius_type)
        per_slice = compute_per_slice_stats(data_2d)

        if per_slice["n_valid"] < 1:
            logger.warning(f"  Skipping - no valid slices")
            continue

        # Skip ROIs with too few axons per slice on average
        if per_slice["mean_axons_per_slice"] < 100:
            logger.warning(f"  Skipping - mean {per_slice['mean_axons_per_slice']} axons/slice")
            continue

        # 3D data
        r3d = load_3d_radii(af)
        if len(r3d) < 100:
            logger.warning(f"  Skipping - only {len(r3d)} 3D radii")
            continue

        x_3d_mean.append(np.mean(r3d))
        x_3d_reff.append(compute_r_eff(r3d))
        roi_stats.append(per_slice)
        roi_3d_cdfs.append(compute_3d_cdf(r3d, bin_centers))

        logger.info(f"  {per_slice['n_valid']} valid slices, "
                    f"{len(per_slice['pooled_radii'])} pooled 2D radii, "
                    f"{len(r3d)} 3D radii")

    x_3d_mean = np.array(x_3d_mean)
    x_3d_reff = np.array(x_3d_reff)
    n_rois = len(x_3d_mean)
    logger.info(f"\n{n_rois} ROIs ready for Monte Carlo")

    # Monte Carlo
    logger.info(f"Running Monte Carlo ({args.n_iterations} iterations)...")
    actual_results, fake_results = run_monte_carlo(
        roi_stats, roi_3d_cdfs, bin_centers, args.n_iterations, seed=args.seed
    )

    # Permutation p-values (slice sampling)
    logger.info("Computing permutation p-values (10k permutations)...")
    p_mean_slice = compute_permutation_pvalue(
        x_3d_mean, roi_stats, "mean_radius", n_perm=10000, seed=args.seed + 100
    )
    p_reff_slice = compute_permutation_pvalue(
        x_3d_reff, roi_stats, "r_eff", n_perm=10000, seed=args.seed + 102
    )

    # Permutation p-values (i.i.d. sampling)
    y_iid_mean = np.mean(fake_results["mean_radius"], axis=1)
    y_iid_reff = np.mean(fake_results["r_eff"], axis=1)
    rng_perm = np.random.default_rng(args.seed + 200)

    r_iid_mean_obs, _ = stats.pearsonr(x_3d_mean, y_iid_mean)
    r_iid_reff_obs, _ = stats.pearsonr(x_3d_reff, y_iid_reff)

    n_perm = 10000
    count_mean, count_reff = 0, 0
    for _ in range(n_perm):
        perm = rng_perm.permutation(n_rois)
        r_m, _ = stats.pearsonr(x_3d_mean[perm], y_iid_mean)
        r_e, _ = stats.pearsonr(x_3d_reff[perm], y_iid_reff)
        if r_m >= r_iid_mean_obs:
            count_mean += 1
        if r_e >= r_iid_reff_obs:
            count_reff += 1
    p_mean_iid = count_mean / n_perm
    p_reff_iid = count_reff / n_perm

    logger.info(f"  Mean radius p-values: slice={p_mean_slice:.4f}, i.i.d.={p_mean_iid:.4f}")
    logger.info(f"  Effective radius p-values: slice={p_reff_slice:.4f}, i.i.d.={p_reff_iid:.4f}")

    # ── Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    logger.info("\nPlotting panel (a): Wasserstein distance...")
    plot_wasserstein_violin(axes[0], actual_results["wasserstein"],
                            fake_results["wasserstein"])

    logger.info("Plotting panel (b): Mean radius...")
    r_mean_s, r_mean_i, r_mean_s_std, r_mean_i_std = plot_scatter_comparison(
        axes[1], x_3d_mean, actual_results["mean_radius"],
        fake_results["mean_radius"], "mean_radius",
        p_actual=p_mean_slice, p_fake=p_mean_iid,
    )

    logger.info("Plotting panel (c): Effective radius...")
    r_reff_s, r_reff_i, r_reff_s_std, r_reff_i_std = plot_scatter_comparison(
        axes[2], x_3d_reff, actual_results["r_eff"],
        fake_results["r_eff"], "r_eff",
        p_actual=p_reff_slice, p_fake=p_reff_iid,
    )

    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=settings.figure["dpi"], bbox_inches="tight")
    plt.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    plt.close()
    logger.info(f"\nSaved figure to {args.output}")

    # Summary
    wass_s = actual_results["wasserstein"].flatten()
    wass_i = fake_results["wasserstein"].flatten()

    def fmt_p(p):
        return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Wasserstein distance (median, IQR):")
    logger.info(f"  2D slice sampling: {np.median(wass_s):.4f} "
                f"({np.percentile(wass_s, 75) - np.percentile(wass_s, 25):.4f})")
    logger.info(f"  i.i.d. sampling:   {np.median(wass_i):.4f} "
                f"({np.percentile(wass_i, 75) - np.percentile(wass_i, 25):.4f})")
    logger.info(f"Correlation with 3D (mean ± std across MC iterations):")
    logger.info(f"  Mean radius:")
    logger.info(f"    slice: R = {r_mean_s:.2f} ± {r_mean_s_std:.2f}, {fmt_p(p_mean_slice)}")
    logger.info(f"    i.i.d.: R = {r_mean_i:.2f} ± {r_mean_i_std:.2f}, {fmt_p(p_mean_iid)}")
    logger.info(f"  Effective radius:")
    logger.info(f"    slice: R = {r_reff_s:.2f} ± {r_reff_s_std:.2f}, {fmt_p(p_reff_slice)}")
    logger.info(f"    i.i.d.: R = {r_reff_i:.2f} ± {r_reff_i_std:.2f}, {fmt_p(p_reff_iid)}")

    # Save metadata
    metadata = {
        "n_iterations": args.n_iterations,
        "n_rois": n_rois,
        "radius_type": args.radius_type,
        "seed": args.seed,
        "wasserstein": {
            "slice_median": float(np.median(wass_s)),
            "iid_median": float(np.median(wass_i)),
        },
        "mean_radius": {
            "r_slice": float(r_mean_s), "r_slice_std": float(r_mean_s_std),
            "p_slice": float(p_mean_slice),
            "r_iid": float(r_mean_i), "r_iid_std": float(r_mean_i_std),
            "p_iid": float(p_mean_iid),
        },
        "r_eff": {
            "r_slice": float(r_reff_s), "r_slice_std": float(r_reff_s_std),
            "p_slice": float(p_reff_slice),
            "r_iid": float(r_reff_i), "r_iid_std": float(r_reff_i_std),
            "p_iid": float(p_reff_iid),
        },
    }

    json_path = args.output.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {json_path}")


if __name__ == "__main__":
    main()
