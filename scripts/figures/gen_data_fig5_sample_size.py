"""
Generate source data for Figure 5 (sample-size effect on radius estimation).

Writes to data/figures/:
    fig_5a_pdf.csv          pooled PDFs + n=10^3 subsample central-95% band (rat, human)
    fig_5a_totals.csv       per-species total axon count (for the n≈ legend labels)
    fig_5b_wasserstein_violin.csv   KDE outlines of within-sample Wasserstein per
                                    (species × sample size)
    fig_5b_wasserstein_stats.csv    per-violin mean/median/min/max + plot position
    fig_5b_between.csv       per-species median inter-ROI Wasserstein (reference lines)
    fig_5cd_error.csv        r̄ / r_MRI relative error vs sample size (median + IQR)

The subsampling draws follow the original RNG order (seed 42) so the figure
reproduces exactly. Plotting is done by plot_fig5_sample_size.py.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import cbook

_root = Path(__file__).resolve().parent
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from axonometry import compute_r_arith, compute_r_eff, rediscretize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_BIN_WIDTH = 0.05
SAMPLE_SIZES = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
N_SUBSAMPLES = 1000
PDF_SAMPLE_SIZE = 1_000
B_SAMPLE_SIZES = [100, 1_000, 10_000, 100_000]
PDF_X_MAX = 2.0


@dataclass
class SubsampleResults:
    sample_size: int
    n_valid_rois: int
    r_arith: np.ndarray
    r_eff: np.ndarray
    reference_r_arith: np.ndarray
    reference_r_eff: np.ndarray


def load_human_cc_histograms(data_dir: Path, bin_width=DEFAULT_BIN_WIDTH):
    bin_edges_orig = np.loadtxt(data_dir / 'desc-binEdges_radii.tsv', delimiter='\t', skiprows=1)
    counts_matrix_orig = np.loadtxt(data_dir / 'desc-countsMinorAxis_radii.tsv',
                                    delimiter='\t', skiprows=1, dtype=float)
    n_rois = counts_matrix_orig.shape[0]
    first_edges, first_centers, _ = rediscretize(bin_edges_orig, counts_matrix_orig[0], bin_width)
    counts_matrix = np.zeros((n_rois, len(first_centers)), dtype=float)
    for i in range(n_rois):
        _, _, counts_matrix[i] = rediscretize(bin_edges_orig, counts_matrix_orig[i], bin_width)
    logger.info(f"Human CC: {n_rois} ROIs, {int(counts_matrix.sum()):,} total axons")
    return first_edges, first_centers, counts_matrix


def load_rat_histograms(data_dir: Path, bin_width=DEFAULT_BIN_WIDTH, r_max=10.0):
    from axonometry.io import load_3d_profiles
    npz_files = sorted(data_dir.glob('*_axon_profiles.npz'))
    if not npz_files:
        raise ValueError(f"No *_axon_profiles.npz files found in {data_dir}")
    bin_edges = np.arange(0, r_max + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    all_counts = []
    for npz_file in npz_files:
        radii = load_3d_profiles(npz_file)['all_radii_um']
        counts, _ = np.histogram(radii, bins=bin_edges)
        all_counts.append(counts)
    counts_matrix = np.array(all_counts, dtype=float)
    logger.info(f"Rat WM: {len(all_counts)} ROIs, {int(counts_matrix.sum()):,} total radii")
    return bin_edges, bin_centers, counts_matrix


def subsample_histogram(counts, bin_centers, sample_size):
    total = counts.sum()
    if total < sample_size:
        return None
    probs = counts / total
    sampled_bins = np.random.choice(len(bin_centers), size=sample_size, p=probs)
    return np.bincount(sampled_bins, minlength=len(bin_centers)).astype(float)


def run_subsampling_analysis(bin_centers, counts_matrix, sample_sizes, n_subsamples):
    n_rois = counts_matrix.shape[0]
    out = {}
    for sample_size in sample_sizes:
        roi_counts = counts_matrix.sum(axis=1)
        valid = np.where(roi_counts >= sample_size)[0]
        if len(valid) == 0:
            continue
        r_arith = np.full((len(valid), n_subsamples), np.nan)
        r_eff = np.full((len(valid), n_subsamples), np.nan)
        ref_a = np.full(len(valid), np.nan)
        ref_e = np.full(len(valid), np.nan)
        for i, roi_idx in enumerate(valid):
            counts = counts_matrix[roi_idx]
            ref_a[i] = compute_r_arith(counts=counts, bin_centers=bin_centers)
            ref_e[i] = compute_r_eff(counts=counts, bin_centers=bin_centers)
            for j in range(n_subsamples):
                sub = subsample_histogram(counts, bin_centers, sample_size)
                if sub is None:
                    continue
                r_arith[i, j] = compute_r_arith(counts=sub, bin_centers=bin_centers)
                r_eff[i, j] = compute_r_eff(counts=sub, bin_centers=bin_centers)
        out[sample_size] = SubsampleResults(sample_size, len(valid), r_arith, r_eff, ref_a, ref_e)
    return out


def wass(cdf1, cdf2, bin_width):
    return np.sum(np.abs(cdf1 - cdf2)) * bin_width


def main():
    parser = argparse.ArgumentParser(description="Generate Fig 5 source data")
    parser.add_argument('--human-data', type=Path, default=Path('data/raw/human/lm'))
    parser.add_argument('--rat-data', type=Path, default=Path('data/processed/rat/lm'))
    parser.add_argument('--out-dir', type=Path, default=Path('data/figures'))
    parser.add_argument('--n-subsamples', type=int, default=N_SUBSAMPLES)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    human_edges, human_centers, human_counts = load_human_cc_histograms(args.human_data)
    rat_edges, rat_centers, rat_counts = load_rat_histograms(args.rat_data)

    # (c)/(d) subsampling analysis — human then rat (original order)
    logger.info("Subsampling Human CC...")
    human_res = run_subsampling_analysis(human_centers, human_counts, SAMPLE_SIZES, args.n_subsamples)
    logger.info("Subsampling Rat WM...")
    rat_res = run_subsampling_analysis(rat_centers, rat_counts, SAMPLE_SIZES, args.n_subsamples)

    human_pooled = human_counts.sum(axis=0)
    rat_pooled = rat_counts.sum(axis=0)
    # datasets order (rat, human) — must match the original plotting RNG order
    ds = [('rat', rat_centers, rat_pooled, rat_counts),
          ('human', human_centers, human_pooled, human_counts)]

    # ── (a) PDFs + n=10^3 subsample central-95% band ──────────────────────
    x_eval = np.linspace(0.0, PDF_X_MAX, 200)
    pdf_out = {'x_eval': x_eval}
    totals = []
    for name, centers, pooled, _ in ds:
        total = int(pooled.sum())
        bw = centers[1] - centers[0]
        probs = pooled / total
        pdf_ref = pooled / (total * bw)
        pdf_out[f'{name}_pdf_ref'] = np.interp(x_eval, centers, pdf_ref, left=0, right=0)
        subs = np.empty((args.n_subsamples, len(centers)))
        for j in range(args.n_subsamples):
            sb = np.random.choice(len(centers), size=PDF_SAMPLE_SIZE, p=probs)
            sc = np.bincount(sb, minlength=len(centers)).astype(float)
            subs[j] = sc / (sc.sum() * bw)
        lo = np.percentile(subs, 2.5, axis=0)
        hi = np.percentile(subs, 97.5, axis=0)
        pdf_out[f'{name}_sub_lo'] = np.interp(x_eval, centers, lo, left=0, right=0)
        pdf_out[f'{name}_sub_hi'] = np.interp(x_eval, centers, hi, left=0, right=0)
        totals.append({'species': name, 'total_count': total})
    pd.DataFrame(pdf_out).to_csv(args.out_dir / 'fig_5a_pdf.csv', index=False)
    pd.DataFrame(totals).to_csv(args.out_dir / 'fig_5a_totals.csv', index=False)

    # ── (b) within-sample vs between-ROI Wasserstein ──────────────────────
    between_rows = []
    for name, centers, pooled, counts_matrix in ds:
        bw = centers[1] - centers[0]
        cdfs = [np.cumsum(counts_matrix[i]) / counts_matrix[i].sum()
                for i in range(counts_matrix.shape[0]) if counts_matrix[i].sum() > 0]
        pair = [wass(cdfs[i], cdfs[j], bw) for i in range(len(cdfs)) for j in range(i + 1, len(cdfs))]
        between_rows.append({'species': name, 'median': float(np.median(pair))})
    pd.DataFrame(between_rows).to_csv(args.out_dir / 'fig_5b_between.csv', index=False)

    width = 0.3
    all_data, meta = [], []
    for d_idx, (name, centers, pooled, counts_matrix) in enumerate(ds):
        bw = centers[1] - centers[0]
        pooled_cdf = np.cumsum(pooled) / pooled.sum()
        pooled_probs = pooled / pooled.sum()
        for size_idx, size in enumerate(B_SAMPLE_SIZES):
            within = []
            for _ in range(args.n_subsamples):
                sb = np.random.choice(len(centers), size=size, p=pooled_probs)
                sc = np.bincount(sb, minlength=len(centers)).astype(float)
                within.append(wass(np.cumsum(sc) / sc.sum(), pooled_cdf, bw))
            all_data.append(np.asarray(within))
            offset = -width / 2 if d_idx == 0 else width / 2
            meta.append({'species': name, 'sample_size': size, 'position': size_idx + offset})
    vpstats = cbook.violin_stats(all_data, points=100)
    viol_rows, stat_rows = [], []
    for m, vs in zip(meta, vpstats):
        for c, v in zip(vs['coords'], vs['vals']):
            viol_rows.append({**m, 'coord': c, 'density': v})
        stat_rows.append({**m, 'mean': vs['mean'], 'median': vs['median'],
                          'vmin': vs['min'], 'vmax': vs['max']})
    pd.DataFrame(viol_rows).to_csv(args.out_dir / 'fig_5b_wasserstein_violin.csv', index=False)
    pd.DataFrame(stat_rows).to_csv(args.out_dir / 'fig_5b_wasserstein_stats.csv', index=False)

    # ── (c)/(d) relative error vs sample size ─────────────────────────────
    err_rows = []
    for metric in ('r_arith', 'r_eff'):
        for name, res in (('rat', rat_res), ('human', human_res)):
            for size in SAMPLE_SIZES:
                if size not in res:
                    continue
                r = res[size]
                ref = (r.reference_r_arith if metric == 'r_arith' else r.reference_r_eff)[:, None]
                vals = r.r_arith if metric == 'r_arith' else r.r_eff
                rel = ((vals - ref) / ref * 100).ravel()
                err_rows.append({'metric': metric, 'species': name, 'sample_size': size,
                                 'median_pct': float(np.nanmedian(rel)),
                                 'q25_pct': float(np.nanpercentile(rel, 25)),
                                 'q75_pct': float(np.nanpercentile(rel, 75))})
    pd.DataFrame(err_rows).to_csv(args.out_dir / 'fig_5cd_error.csv', index=False)

    logger.info(f"Wrote fig_5* source data to {args.out_dir}/")


if __name__ == '__main__':
    main()
