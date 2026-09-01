"""
Monte Carlo comparison: 2D slice sampling vs random sampling.

Compares two approaches for estimating 3D population statistics from 2D samples:
1. 2D slice sampling: Randomly pick one physical 2D slice per ROI
2. random sampling: Sample mean_axons_per_slice radii from the pooled 2D distribution

Both methods use the same underlying 2D radii (with minor axis bias from
ellipse fitting), ensuring a fair comparison of sampling strategies.

Creates a 1x3 figure showing:
(a) Wasserstein distance violin: slice sampling vs random sampling
(b) Mean radius scatter: 2D vs 3D
(c) Effective radius scatter: 2D vs 3D

Median and IQR are computed over Monte Carlo iterations.

Usage:
    python scripts/figures/plot_2d_slice_vs_random_distribution_comparison.py \
        --data-dir data/processed/rat/lm \
        --output fig/main/2d_slice_vs_random_distribution_comparison.svg \
        --n-iterations 100
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.ticker import FormatStrFormatter
from scipy import stats

from axonometry import compute_r_eff, get_plot_settings, style_axis
from axonometry.io import load_2d_profiles, load_3d_profiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

settings = get_plot_settings()

# ── Constants ──────────────────────────────────────────────────────────────
MIN_AXON_COUNT = 30        # Min axons per slice for valid statistics
BIN_WIDTH_UM = 0.02
MAX_RADIUS_UM = 5.0


# ── Data loading ──────────────────────────────────────────────────────────


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
        r_eff.append(compute_r_eff(r_z))
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
    """Load pooled 3D radii with endpoint trimming."""
    return load_3d_profiles(npz_path)['all_radii_um']



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

def make_bins() -> Tuple[np.ndarray, np.ndarray, float]:
    """Standard bin centers, edges, and width for histogram comparison."""
    bin_centers = np.arange(BIN_WIDTH_UM / 2, MAX_RADIUS_UM, BIN_WIDTH_UM)
    bin_width = bin_centers[1] - bin_centers[0]
    bin_edges = np.concatenate([[bin_centers[0] - bin_width / 2],
                                 bin_centers + bin_width / 2])
    return bin_centers, bin_edges, bin_width


def compute_wasserstein(radii_sample: np.ndarray, cdf_3d: np.ndarray,
                        bin_edges: np.ndarray, bin_width: float) -> float:
    """Compute Wasserstein distance between a 2D sample and 3D CDF."""
    hist, _ = np.histogram(radii_sample, bins=bin_edges)
    if hist.sum() == 0:
        return np.nan
    cdf_2d = np.cumsum(hist) / hist.sum()
    return np.sum(np.abs(cdf_2d - cdf_3d)) * bin_width


def compute_3d_cdf(radii: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Compute 3D CDF using histogram on standard bin edges."""
    hist, _ = np.histogram(radii, bins=bin_edges)
    return np.cumsum(hist) / hist.sum()


def run_monte_carlo(roi_stats: List[Dict], roi_3d_cdfs: List[np.ndarray],
                    bin_edges: np.ndarray, n_iterations: int,
                    seed: int = 42) -> Tuple[Dict, Dict]:
    """
    Run Monte Carlo simulation for both 2D slice sampling and random sampling.

    Returns:
        Tuple of (slice_results, random_results), each with keys:
        mean_radius, r_eff, wasserstein — arrays of shape (n_rois, n_iterations)
    """
    n_rois = len(roi_stats)
    bin_width = bin_edges[1] - bin_edges[0]

    slice_mean = np.zeros((n_rois, n_iterations))
    slice_reff = np.zeros((n_rois, n_iterations))
    slice_wass = np.zeros((n_rois, n_iterations))
    rand_mean = np.zeros((n_rois, n_iterations))
    rand_reff = np.zeros((n_rois, n_iterations))
    rand_wass = np.zeros((n_rois, n_iterations))

    rng_slice = np.random.default_rng(seed)
    rng_random = np.random.default_rng(seed + 1)

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
            slice_wass[j, it] = compute_wasserstein(r_slice, cdf_3d, bin_edges, bin_width)

            # Random sampling: sample same N radii as the drawn slice
            n_samples = len(r_slice)
            rand_radii = rng_random.choice(roi["pooled_radii"], size=n_samples, replace=True)
            rand_mean[j, it] = np.mean(rand_radii)
            rand_reff[j, it] = compute_r_eff(rand_radii)
            rand_wass[j, it] = compute_wasserstein(rand_radii, cdf_3d, bin_edges, bin_width)

    return (
        {"mean_radius": slice_mean, "r_eff": slice_reff, "wasserstein": slice_wass},
        {"mean_radius": rand_mean, "r_eff": rand_reff, "wasserstein": rand_wass},
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

def plot_wasserstein_violin(ax, slice_wass: np.ndarray, rand_wass: np.ndarray):
    """Panel (a): Wasserstein distance violin comparison."""
    slice_pooled = slice_wass.flatten()
    random_pooled = rand_wass.flatten()

    color_slice = settings.colors["binary_a"]
    color_random = settings.colors["binary_b"]

    parts = ax.violinplot(
        [slice_pooled, random_pooled], positions=[1, 2],
        showmeans=False, showmedians=True, widths=0.7,
    )

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor([color_slice, color_random][i])
        body.set_edgecolor("black")
        body.set_alpha(0.7)

    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_edgecolor("black")
            parts[key].set_linewidth(1.5)

    rng = np.random.default_rng(42)
    n_show = min(500, len(slice_pooled))
    idx_s = rng.choice(len(slice_pooled), n_show, replace=False)
    idx_r = rng.choice(len(random_pooled), n_show, replace=False)

    ax.scatter(1 + rng.uniform(-0.15, 0.15, n_show), slice_pooled[idx_s],
               alpha=0.3, s=3, color=color_slice, zorder=0)
    ax.scatter(2 + rng.uniform(-0.15, 0.15, n_show), random_pooled[idx_r],
               alpha=0.3, s=3, color=color_random, zorder=0)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        ["2D slice-wise\nsampling", "2D random\nsampling"],
        fontsize=settings.fonts["tick_size"] - 1,
    )
    style_axis(ax, ylabel="Wasserstein distance [μm]")
    ax.set_ylim(bottom=0)
    ax.set_box_aspect(1)


def plot_scatter_comparison(ax, x_3d: np.ndarray,
                            slice_results: np.ndarray,
                            random_results: np.ndarray, metric: str,
                            p_slice: Optional[float] = None,
                            p_random: Optional[float] = None):
    """Panel (b)/(c): 2D vs 3D scatter comparing slice and random sampling."""
    font_s = settings.fonts
    err_s = settings.error_bars

    color_slice = settings.colors["binary_a"]
    color_random = settings.colors["binary_b"]

    n_iterations = slice_results.shape[1]

    # Compute R per MC iteration
    r_slice_iter = np.array([
        stats.pearsonr(x_3d, slice_results[:, it])[0]
        for it in range(n_iterations)
    ])
    r_random_iter = np.array([
        stats.pearsonr(x_3d, random_results[:, it])[0]
        for it in range(n_iterations)
    ])

    r_slice_mean = np.mean(r_slice_iter)
    r_slice_std = np.std(r_slice_iter)
    r_random_mean = np.mean(r_random_iter)
    r_random_std = np.std(r_random_iter)

    # Median + IQR over iterations per ROI
    slice_med = np.median(slice_results, axis=1)
    slice_q25 = np.percentile(slice_results, 25, axis=1)
    slice_q75 = np.percentile(slice_results, 75, axis=1)
    random_med = np.median(random_results, axis=1)
    random_q25 = np.percentile(random_results, 25, axis=1)
    random_q75 = np.percentile(random_results, 75, axis=1)

    ax.errorbar(x_3d, slice_med,
                yerr=[slice_med - slice_q25, slice_q75 - slice_med],
                fmt="o", color=color_slice, ecolor=to_rgba(color_slice, 0.7),
                markersize=8, capsize=err_s["capsize"], capthick=err_s["capthick"],
                elinewidth=err_s["linewidth"],
                markerfacecolor=to_rgba(color_slice, 0.3), markeredgecolor=color_slice,
                markeredgewidth=1.5, label="2D slice-wise sampling", zorder=10)

    ax.errorbar(x_3d, random_med,
                yerr=[random_med - random_q25, random_q75 - random_med],
                fmt="o", color=color_random, ecolor=to_rgba(color_random, 0.7),
                markersize=4, capsize=err_s["capsize"], capthick=err_s["capthick"],
                elinewidth=err_s["linewidth"],
                markerfacecolor=color_random, markeredgecolor=color_random,
                markeredgewidth=1.5, label="2D random sampling", zorder=11)

    all_vals = np.concatenate([x_3d, slice_med, random_med])
    lo = np.nanmin(all_vals) * 0.95
    hi = np.nanmax(all_vals) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1.5, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # Use identical tick locations on x and y (limits are equal)
    ticks = ax.get_yticks()
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

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

    # Normalized mean bias per method: mean((2D - 3D) / 3D) * 100% (as in Fig 3)
    bias_slice = np.mean((slice_med - x_3d) / x_3d) * 100
    bias_random = np.mean((random_med - x_3d) / x_3d) * 100

    # Slice block: 3 lines (bias / R / p), kept roughly at the current upper-left spot
    text_slice = (f"Bias = {bias_slice:+.1f}%\n"
                  f"$R$ = {r_slice_mean:.2f} ± {r_slice_std:.2f}")
    if p_slice is not None:
        text_slice += f"\n{_fmt_p(p_slice)}"
    ax.text(0.04, 0.74, text_slice, transform=ax.transAxes,
            fontsize=font_s["legend_size"], ha="left", va="top", color=color_slice)

    # Random block: 3 lines (bias / R / p), moved to bottom-right (as in Fig 3)
    text_random = (f"Bias = {bias_random:+.1f}%\n"
                   f"$R$ = {r_random_mean:.2f} ± {r_random_std:.2f}")
    if p_random is not None:
        text_random += f"\n{_fmt_p(p_random)}"
    ax.text(0.96, 0.04, text_random, transform=ax.transAxes,
            fontsize=font_s["legend_size"], ha="right", va="bottom", color=color_random)

    ax.set_box_aspect(1)

    return r_slice_mean, r_random_mean, r_slice_std, r_random_std


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo comparison: 2D slice sampling vs random sampling"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/rat/lm"))
    parser.add_argument("--output", type=Path,
                        default=Path("fig/main/2d_slice_vs_random_distribution_comparison.svg"))
    parser.add_argument("--n-iterations", type=int, default=10000)
    parser.add_argument("--radius-type", type=str, default="minor",
                        choices=["minor", "circular"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Monte Carlo: 2D Slice Sampling vs Random Sampling")
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

    bin_centers, bin_edges, bin_width = make_bins()

    for sf, af, sn in all_pairs:
        logger.info(f"Processing {sn}...")

        # 2D data
        data_2d = load_2d_profiles(sf, args.radius_type)
        per_slice = compute_per_slice_stats(data_2d)

        if per_slice["n_valid"] < 1:
            logger.warning(f"  Skipping - no valid slices")
            continue

        # 3D data
        r3d = load_3d_radii(af)
        if len(r3d) == 0:
            logger.warning(f"  Skipping - no 3D radii")
            continue

        x_3d_mean.append(np.mean(r3d))
        x_3d_reff.append(compute_r_eff(r3d))
        roi_stats.append(per_slice)
        roi_3d_cdfs.append(compute_3d_cdf(r3d, bin_edges))

        logger.info(f"  {per_slice['n_valid']} valid slices, "
                    f"{len(per_slice['pooled_radii'])} pooled 2D radii, "
                    f"{len(r3d)} 3D radii")

    x_3d_mean = np.array(x_3d_mean)
    x_3d_reff = np.array(x_3d_reff)
    n_rois = len(x_3d_mean)
    logger.info(f"\n{n_rois} ROIs ready for Monte Carlo")

    # Monte Carlo
    logger.info(f"Running Monte Carlo ({args.n_iterations} iterations)...")
    slice_results, random_results = run_monte_carlo(
        roi_stats, roi_3d_cdfs, bin_edges, args.n_iterations, seed=args.seed
    )

    # Permutation p-values (slice sampling)
    logger.info("Computing permutation p-values (10k permutations)...")
    p_mean_slice = compute_permutation_pvalue(
        x_3d_mean, roi_stats, "mean_radius", n_perm=10000, seed=args.seed + 100
    )
    p_reff_slice = compute_permutation_pvalue(
        x_3d_reff, roi_stats, "r_eff", n_perm=10000, seed=args.seed + 102
    )

    # Permutation p-values (random sampling)
    y_rand_mean = np.mean(random_results["mean_radius"], axis=1)
    y_rand_reff = np.mean(random_results["r_eff"], axis=1)
    rng_perm = np.random.default_rng(args.seed + 200)

    r_rand_mean_obs, _ = stats.pearsonr(x_3d_mean, y_rand_mean)
    r_rand_reff_obs, _ = stats.pearsonr(x_3d_reff, y_rand_reff)

    n_perm = 10000
    count_mean, count_reff = 0, 0
    for _ in range(n_perm):
        perm = rng_perm.permutation(n_rois)
        r_m, _ = stats.pearsonr(x_3d_mean[perm], y_rand_mean)
        r_e, _ = stats.pearsonr(x_3d_reff[perm], y_rand_reff)
        if r_m >= r_rand_mean_obs:
            count_mean += 1
        if r_e >= r_rand_reff_obs:
            count_reff += 1
    p_mean_random = count_mean / n_perm
    p_reff_random = count_reff / n_perm

    logger.info(f"  Mean radius p-values: slice={p_mean_slice:.4f}, random={p_mean_random:.4f}")
    logger.info(f"  Effective radius p-values: slice={p_reff_slice:.4f}, random={p_reff_random:.4f}")

    # ── Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))

    logger.info("\nPlotting panel (a): Wasserstein distance...")
    plot_wasserstein_violin(axes[0], slice_results["wasserstein"],
                            random_results["wasserstein"])

    logger.info("Plotting panel (b): Mean radius...")
    r_mean_s, r_mean_r, r_mean_s_std, r_mean_r_std = plot_scatter_comparison(
        axes[1], x_3d_mean, slice_results["mean_radius"],
        random_results["mean_radius"], "mean_radius",
        p_slice=p_mean_slice, p_random=p_mean_random,
    )

    logger.info("Plotting panel (c): Effective radius...")
    r_reff_s, r_reff_r, r_reff_s_std, r_reff_r_std = plot_scatter_comparison(
        axes[2], x_3d_reff, slice_results["r_eff"],
        random_results["r_eff"], "r_eff",
        p_slice=p_reff_slice, p_random=p_reff_random,
    )

    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=settings.figure["dpi"], bbox_inches="tight")
    plt.close()
    logger.info(f"\nSaved figure to {args.output}")

    # Summary
    wass_s = slice_results["wasserstein"].flatten()
    wass_r = random_results["wasserstein"].flatten()

    def fmt_p(p):
        return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Wasserstein distance (median, IQR):")
    logger.info(f"  2D slice sampling: {np.median(wass_s):.4f} "
                f"({np.percentile(wass_s, 75) - np.percentile(wass_s, 25):.4f})")
    logger.info(f"  random sampling:   {np.median(wass_r):.4f} "
                f"({np.percentile(wass_r, 75) - np.percentile(wass_r, 25):.4f})")
    logger.info(f"Correlation with 3D (mean ± std across MC iterations):")
    logger.info(f"  Mean radius:")
    logger.info(f"    slice: R = {r_mean_s:.2f} ± {r_mean_s_std:.2f}, {fmt_p(p_mean_slice)}")
    logger.info(f"    random: R = {r_mean_r:.2f} ± {r_mean_r_std:.2f}, {fmt_p(p_mean_random)}")
    logger.info(f"  Effective radius:")
    logger.info(f"    slice: R = {r_reff_s:.2f} ± {r_reff_s_std:.2f}, {fmt_p(p_reff_slice)}")
    logger.info(f"    random: R = {r_reff_r:.2f} ± {r_reff_r_std:.2f}, {fmt_p(p_reff_random)}")

    # Save metadata
    metadata = {
        "n_iterations": args.n_iterations,
        "n_rois": n_rois,
        "radius_type": args.radius_type,
        "seed": args.seed,
        "wasserstein": {
            "slice_median": float(np.median(wass_s)),
            "random_median": float(np.median(wass_r)),
        },
        "mean_radius": {
            "r_slice": float(r_mean_s), "r_slice_std": float(r_mean_s_std),
            "p_slice": float(p_mean_slice),
            "r_random": float(r_mean_r), "r_random_std": float(r_mean_r_std),
            "p_random": float(p_mean_random),
        },
        "r_eff": {
            "r_slice": float(r_reff_s), "r_slice_std": float(r_reff_s_std),
            "p_slice": float(p_reff_slice),
            "r_random": float(r_reff_r), "r_random_std": float(r_reff_r_std),
            "p_random": float(p_reff_random),
        },
    }

    json_path = args.output.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {json_path}")


if __name__ == "__main__":
    main()
