"""
Generate source data for Figure 6 (parametric distribution fits, Human CC + Rat WM).

Loads the histograms, fits all candidate distributions per ROI (reusing the
machinery in plot_fig6_parametric_fits), and writes the plotted values to
data/figures/:

    fig_6_pooled_hist.csv    pooled histogram per species (bin_center, count)
    fig_6_fit_curves.csv     pooled fitted PDF curves (species, distribution, x, pdf)
    fig_6_dist_stats.csv     per-distribution summed AIC + win rate (panel c)
    fig_6_per_sample.csv     per-distribution per-ROI r_arith / r_eff / Wasserstein
                             (panels d/e/f boxplots)
    fig_6_empirical.csv      per-ROI empirical r_arith / r_eff
    fig_6_inter_roi.csv      per-species median inter-ROI Wasserstein (reference lines)

Plotting is done by plot_fig6_parametric_fits.py (and the per-distribution
supplement plot_figS4_dists_per_panel.py).
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_fig6_parametric_fits import (  # noqa: E402
    CANDIDATE_DISTRIBUTIONS, DEFAULT_BIN_WIDTH, compute_inter_roi_wasserstein,
    fit_all_samples, load_human_cc_data, load_rat_data,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate Fig 6 source data")
    parser.add_argument('--human-data', type=Path, default=Path('data/raw/human/lm'))
    parser.add_argument('--rat-data', type=Path, default=Path('data/processed/rat/lm'))
    parser.add_argument('--out-dir', type=Path, default=Path('data/figures'))
    parser.add_argument('--bin-width', type=float, default=DEFAULT_BIN_WIDTH)
    parser.add_argument('--r-max', type=float, default=3.0)
    args = parser.parse_args()

    logger.info("Loading + fitting Human CC...")
    human_pooled, human_ps = load_human_cc_data(args.human_data, args.bin_width)
    human_m = fit_all_samples(human_ps, human_pooled)
    human_iroi = compute_inter_roi_wasserstein(human_ps)

    logger.info("Loading + fitting Rat WM...")
    rat_pooled, rat_ps = load_rat_data(args.rat_data, args.bin_width, args.r_max)
    rat_m = fit_all_samples(rat_ps, rat_pooled)
    rat_iroi = compute_inter_roi_wasserstein(rat_ps)

    species = [('human', human_pooled, human_m, human_iroi),
               ('rat', rat_pooled, rat_m, rat_iroi)]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Pooled histograms
    rows = []
    for name, pooled, _, _ in species:
        for c, cnt in zip(pooled.bin_centers, pooled.counts):
            rows.append({'species': name, 'bin_center': float(c), 'count': float(cnt)})
    pd.DataFrame(rows).to_csv(args.out_dir / 'fig_6_pooled_hist.csv', index=False)

    # Pooled fitted PDF curves
    rows = []
    for name, _, m, _ in species:
        for r in m.pooled_results:
            for x, y in zip(r.pdf_x_fine, r.pdf_values_fine):
                rows.append({'species': name, 'distribution': r.distribution_name,
                             'pdf_x': float(x), 'pdf_val': float(y)})
    pd.DataFrame(rows).to_csv(args.out_dir / 'fig_6_fit_curves.csv', index=False)

    # Per-distribution summed AIC + win rate (in AIC-sorted order)
    rows = []
    for name, _, m, _ in species:
        for i, dname in enumerate(m.distribution_names):
            rows.append({'species': name, 'distribution': dname,
                         'summed_aic': float(m.summed_aic[i]),
                         'win_rate': float(m.win_rate.get(dname, 0.0))})
    pd.DataFrame(rows).to_csv(args.out_dir / 'fig_6_dist_stats.csv', index=False)

    # Per-distribution per-ROI fitted radii + Wasserstein (CANDIDATE order)
    rows = []
    for name, _, m, _ in species:
        n_samples = m.all_r_arith.shape[1]
        for dname, didx in m.dist_name_to_idx.items():
            for s in range(n_samples):
                rows.append({'species': name, 'distribution': dname, 'sample_idx': s,
                             'r_arith': float(m.all_r_arith[didx, s]),
                             'r_eff': float(m.all_r_eff[didx, s]),
                             'wasserstein': float(m.all_wasserstein[didx, s])})
    pd.DataFrame(rows).to_csv(args.out_dir / 'fig_6_per_sample.csv', index=False)

    # Per-ROI empirical radii
    rows = []
    for name, _, m, _ in species:
        for s in range(len(m.empirical_r_arith_per_sample)):
            rows.append({'species': name, 'sample_idx': s,
                         'empirical_r_arith': float(m.empirical_r_arith_per_sample[s]),
                         'empirical_r_eff': float(m.empirical_r_eff_per_sample[s])})
    pd.DataFrame(rows).to_csv(args.out_dir / 'fig_6_empirical.csv', index=False)

    # Inter-ROI Wasserstein reference
    pd.DataFrame([{'species': 'human', 'inter_roi_wasserstein': float(human_iroi)},
                  {'species': 'rat', 'inter_roi_wasserstein': float(rat_iroi)}]).to_csv(
        args.out_dir / 'fig_6_inter_roi.csv', index=False)

    logger.info(f"Wrote fig_6_* source data to {args.out_dir}/")


if __name__ == '__main__':
    main()
