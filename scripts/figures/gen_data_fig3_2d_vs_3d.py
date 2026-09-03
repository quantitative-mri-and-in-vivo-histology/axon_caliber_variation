"""
Generate source data for Figure 3 (2D vs 3D radius distributions) and its
circular-radius supplement (Fig S3).

Run once per radius type / prefix:
    python scripts/figures/gen_data_fig3_2d_vs_3d.py --radius-type minor    --prefix fig_3
    python scripts/figures/gen_data_fig3_2d_vs_3d.py --radius-type circular --prefix fig_s3

Writes to data/figures/:
    {prefix}_a_pdf.csv               representative-ROI PDFs (3D + 2D median + central-95% band)
    {prefix}_a_markers.csv           r̄ / r_MRI vertical markers (3D and 2D median)
    {prefix}_b_wasserstein_within.csv    intra-ROI (2D↔3D) Wasserstein distances
    {prefix}_b_wasserstein_between.csv   inter-ROI (3D) pairwise Wasserstein distances
    {prefix}_cd_scatter.csv          per-ROI 2D-vs-3D r̄ and r_MRI (median + IQR)
    {prefix}_cd_stats.csv            Monte-Carlo R (mean±std), permutation p, and bias

Plotting is done by plot_fig3_2d_vs_3d.py.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from axonometry import compute_r_eff
from axonometry.io import load_2d_profiles, load_3d_profiles

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MIN_AXON_COUNT = 30
BIN_WIDTH_UM = 0.02
MAX_RADIUS_UM = 5.0
PDF_LO_PCT, PDF_HI_PCT = 2.5, 97.5   # central 95% band


def make_bins() -> Tuple[np.ndarray, np.ndarray, float]:
    bin_centers = np.arange(BIN_WIDTH_UM / 2, MAX_RADIUS_UM, BIN_WIDTH_UM)
    bin_width = bin_centers[1] - bin_centers[0]
    bin_edges = np.concatenate([[bin_centers[0] - bin_width / 2], bin_centers + bin_width / 2])
    return bin_centers, bin_edges, bin_width


def compute_per_slice_stats(data: Dict) -> Dict:
    radii, slices, n_slices = data['radii'], data['slice_index'], data['n_slices']
    r_arith, r_eff, counts = [], [], []
    for z in range(n_slices):
        r_z = radii[slices == z]
        if len(r_z) < MIN_AXON_COUNT:
            continue
        counts.append(len(r_z))
        r_arith.append(np.mean(r_z))
        r_eff.append(compute_r_eff(r_z))
    return {'r_arith': np.array(r_arith), 'r_eff': np.array(r_eff),
            'counts': np.array(counts), 'n_valid': len(counts)}


def compute_per_slice_pdfs(data: Dict) -> np.ndarray:
    bin_centers, bin_edges, bin_width = make_bins()
    radii, slices, n_slices = data['radii'], data['slice_index'], data['n_slices']
    pdfs = []
    for z in range(n_slices):
        r_z = radii[slices == z]
        if len(r_z) < MIN_AXON_COUNT:
            continue
        hist, _ = np.histogram(r_z, bins=bin_edges)
        total = hist.sum()
        if total > 0:
            pdfs.append(hist / (total * bin_width))
    return np.array(pdfs) if pdfs else np.empty((0, len(bin_centers)))


def load_3d_radii(npz_path: Path) -> np.ndarray:
    return load_3d_profiles(npz_path)['all_radii_um']


def find_matching_pairs(data_dir: Path) -> List[Tuple[Path, Path, str]]:
    pairs = []
    for sf in sorted(data_dir.glob("*_myelin_slice_profiles.npz")):
        stem = sf.stem.replace('_slice_profiles', '')
        af = data_dir / f"{stem}_axon_profiles.npz"
        parts = stem.replace('_myelin', '').rsplit('_', 1)
        base_name = parts[0] if len(parts) == 2 else stem
        pop = parts[1].upper() if len(parts) == 2 else ''
        sample_name = f"{base_name}_{pop}" if pop else base_name
        if not af.exists():
            logger.warning(f"  No 3D file for {sf.name}, skipping")
            continue
        pairs.append((sf, af, sample_name))
    logger.info(f"Found {len(pairs)} matching 2D/3D pairs")
    return pairs


def run_monte_carlo(roi_data, x_3d_arith, x_3d_reff, n_iterations, seed):
    """Monte Carlo correlation (pick one slice per ROI) + permutation p-value."""
    rng = np.random.default_rng(seed)
    n_rois = len(roi_data)
    r_arith_iter = np.empty(n_iterations)
    r_reff_iter = np.empty(n_iterations)
    for it in range(n_iterations):
        y_arith = np.array([roi['r_arith'][rng.integers(roi['n_valid'])] for roi in roi_data])
        y_reff = np.array([roi['r_eff'][rng.integers(roi['n_valid'])] for roi in roi_data])
        r_arith_iter[it], _ = stats.pearsonr(x_3d_arith, y_arith)
        r_reff_iter[it], _ = stats.pearsonr(x_3d_reff, y_reff)

    rng_perm = np.random.default_rng(seed + 100)
    r_arith_obs, r_reff_obs = np.mean(r_arith_iter), np.mean(r_reff_iter)
    n_perm = n_iterations
    p_a, p_e = 0, 0
    for _ in range(n_perm):
        perm = rng_perm.permutation(n_rois)
        y_arith = np.array([roi['r_arith'][rng_perm.integers(roi['n_valid'])] for roi in roi_data])
        y_reff = np.array([roi['r_eff'][rng_perm.integers(roi['n_valid'])] for roi in roi_data])
        r_a, _ = stats.pearsonr(x_3d_arith[perm], y_arith)
        r_e, _ = stats.pearsonr(x_3d_reff[perm], y_reff)
        p_a += (r_a >= r_arith_obs)
        p_e += (r_e >= r_reff_obs)
    return {
        'r_arith': {'r_mean': r_arith_obs, 'r_std': float(np.std(r_arith_iter)), 'p_value': p_a / n_perm},
        'r_eff': {'r_mean': r_reff_obs, 'r_std': float(np.std(r_reff_iter)), 'p_value': p_e / n_perm},
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Fig 3 / S3 source data")
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed/rat/lm'))
    parser.add_argument('--out-dir', type=Path, default=Path('data/figures'))
    parser.add_argument('--prefix', type=str, default='fig_3', help='Output filename prefix (fig_3 or fig_s3)')
    parser.add_argument('--radius-type', type=str, default='minor', choices=['circular', 'minor'])
    parser.add_argument('--x-max', type=float, default=1.0, help='Max radius for the PDF panel')
    parser.add_argument('--n-iterations', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--representative', type=str, default=None)
    args = parser.parse_args()

    pairs = find_matching_pairs(args.data_dir)
    if not pairs:
        logger.error("No matching 2D/3D pairs found!")
        return

    cache_2d = {sf: load_2d_profiles(sf, args.radius_type) for sf, _, _ in pairs}
    cache_3d = {af: load_3d_radii(af) for _, af, _ in pairs}

    # Representative ROI for panel (a): highest mean axon count per slice
    if args.representative:
        rep_sf = args.data_dir / f"{args.representative}_slice_profiles.npz"
        rep_af = args.data_dir / f"{args.representative}_axon_profiles.npz"
        cache_2d.setdefault(rep_sf, load_2d_profiles(rep_sf, args.radius_type))
        cache_3d.setdefault(rep_af, load_3d_radii(rep_af))
    else:
        best = max(pairs, key=lambda p: len(cache_2d[p[0]]['radii']) / max(cache_2d[p[0]]['n_slices'], 1))
        rep_sf, rep_af, best_name = best
        logger.info(f"Auto-selected representative: {best_name}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bin_centers, bin_edges, bin_width = make_bins()

    # ── Panel (a): representative PDFs + markers ──────────────────────────
    pdfs_2d = compute_per_slice_pdfs(cache_2d[rep_sf])
    stats_2d = compute_per_slice_stats(cache_2d[rep_sf])
    radii_3d = cache_3d[rep_af]
    hist_3d, _ = np.histogram(radii_3d, bins=bin_edges)
    pdf_3d = hist_3d / (hist_3d.sum() * bin_width) if hist_3d.sum() > 0 else hist_3d * 0.0
    mask = bin_centers <= args.x_max
    pd.DataFrame({
        'bin_center': bin_centers[mask],
        'pdf_3d': pdf_3d[mask],
        'pdf_2d_median': np.median(pdfs_2d, axis=0)[mask],
        'pdf_2d_lo': np.percentile(pdfs_2d, PDF_LO_PCT, axis=0)[mask],
        'pdf_2d_hi': np.percentile(pdfs_2d, PDF_HI_PCT, axis=0)[mask],
    }).to_csv(args.out_dir / f"{args.prefix}_a_pdf.csv", index=False)

    valid_reff = stats_2d['r_eff'][~np.isnan(stats_2d['r_eff'])]
    pd.DataFrame({
        'marker': ['r_arith_3d', 'r_arith_2d_median', 'r_eff_3d', 'r_eff_2d_median'],
        'value': [float(np.mean(radii_3d)), float(np.median(stats_2d['r_arith'])),
                  float(compute_r_eff(radii_3d)),
                  float(np.median(valid_reff)) if len(valid_reff) else float('nan')],
    }).to_csv(args.out_dir / f"{args.prefix}_a_markers.csv", index=False)

    # ── Panel (b): within-ROI (2D↔3D) and between-ROI (3D) Wasserstein ────
    within, roi_cdfs_3d = [], []
    for sf, af, _ in pairs:
        r3d = cache_3d[af]
        if len(r3d) == 0:
            continue
        h3, _ = np.histogram(r3d, bins=bin_edges)
        cdf_3d = np.cumsum(h3) / h3.sum()
        roi_cdfs_3d.append(cdf_3d)
        d2 = cache_2d[sf]
        for z in range(d2['n_slices']):
            r_z = d2['radii'][d2['slice_index'] == z]
            if len(r_z) < MIN_AXON_COUNT:
                continue
            hz, _ = np.histogram(r_z, bins=bin_edges)
            if hz.sum() == 0:
                continue
            cdf_2d = np.cumsum(hz) / hz.sum()
            within.append(np.sum(np.abs(cdf_2d - cdf_3d)) * bin_width)
    between = [np.sum(np.abs(roi_cdfs_3d[i] - roi_cdfs_3d[j])) * bin_width
               for i in range(len(roi_cdfs_3d)) for j in range(i + 1, len(roi_cdfs_3d))]
    logger.info(f"  Within: {len(within)} slices, Between: {len(between)} ROI pairs")
    pd.DataFrame({'wasserstein_um': within}).to_csv(
        args.out_dir / f"{args.prefix}_b_wasserstein_within.csv", index=False)
    pd.DataFrame({'wasserstein_um': between}).to_csv(
        args.out_dir / f"{args.prefix}_b_wasserstein_between.csv", index=False)

    # ── Panels (c)/(d): per-ROI 2D-vs-3D scatter + MC stats ───────────────
    rows, roi_data, x3a, x3e = [], [], [], []
    for sf, af, sn in pairs:
        per_slice = compute_per_slice_stats(cache_2d[sf])
        r3d = cache_3d[af]
        if per_slice['n_valid'] < 1 or len(r3d) == 0:
            continue
        vre = per_slice['r_eff'][~np.isnan(per_slice['r_eff'])]
        rows.append({
            'roi': sn,
            'r_arith_3d': float(np.mean(r3d)),
            'r_arith_2d_median': float(np.median(per_slice['r_arith'])),
            'r_arith_2d_q25': float(np.percentile(per_slice['r_arith'], 25)),
            'r_arith_2d_q75': float(np.percentile(per_slice['r_arith'], 75)),
            'r_eff_3d': float(compute_r_eff(r3d)),
            'r_eff_2d_median': float(np.median(vre)) if len(vre) else float('nan'),
            'r_eff_2d_q25': float(np.percentile(vre, 25)) if len(vre) else float('nan'),
            'r_eff_2d_q75': float(np.percentile(vre, 75)) if len(vre) else float('nan'),
        })
        roi_data.append(per_slice)
        x3a.append(np.mean(r3d))
        x3e.append(compute_r_eff(r3d))
    scatter = pd.DataFrame(rows)
    scatter.to_csv(args.out_dir / f"{args.prefix}_cd_scatter.csv", index=False)

    logger.info(f"Running Monte Carlo ({args.n_iterations} iterations)...")
    mc = run_monte_carlo(roi_data, np.array(x3a), np.array(x3e), args.n_iterations, args.seed)
    bias_arith = float(np.mean((scatter['r_arith_2d_median'] - scatter['r_arith_3d'])
                               / scatter['r_arith_3d']) * 100)
    bias_eff = float(np.mean((scatter['r_eff_2d_median'] - scatter['r_eff_3d'])
                             / scatter['r_eff_3d']) * 100)
    pd.DataFrame({
        'metric': ['r_arith', 'r_eff'],
        'r_mean': [mc['r_arith']['r_mean'], mc['r_eff']['r_mean']],
        'r_std': [mc['r_arith']['r_std'], mc['r_eff']['r_std']],
        'p_value': [mc['r_arith']['p_value'], mc['r_eff']['p_value']],
        'bias_pct': [bias_arith, bias_eff],
    }).to_csv(args.out_dir / f"{args.prefix}_cd_stats.csv", index=False)

    logger.info(f"Wrote {args.prefix}_* source data to {args.out_dir}/  "
                f"(r̄ R={mc['r_arith']['r_mean']:.3f}, r_MRI R={mc['r_eff']['r_mean']:.3f})")


if __name__ == '__main__':
    main()
