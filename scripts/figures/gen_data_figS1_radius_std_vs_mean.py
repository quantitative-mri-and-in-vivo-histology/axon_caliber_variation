"""
Generate source data for Figure S1 (radius standard deviation vs mean radius).

Pools per-axon (mean radius, std radius) across the 3D axon profiles, bins by
mean radius, and writes the plotted median + IQR to:

    data/figures/fig_s1_std_vs_radius.csv   (mean_radius_um, std_median, std_q25, std_q75)

Plotting is done by plot_figS1_radius_std_vs_mean.py.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MIN_AXONS_PER_BIN = 50
N_BINS = 30


def main():
    parser = argparse.ArgumentParser(description="Generate Fig S1 source data")
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed/rat/lm'))
    parser.add_argument('--out-dir', type=Path, default=Path('data/figures'))
    parser.add_argument('--min-length', type=float, default=20.0,
                        help='Minimum total axon length in μm')
    args = parser.parse_args()

    from axonometry.io import load_3d_profiles

    all_mean, all_std, total = [], [], 0
    for npz in sorted(args.data_dir.glob('*_axon_profiles.npz')):
        data = load_3d_profiles(npz)
        seg_r = data['segment_radii_um']
        seg_l = data['segment_lengths_um']
        total += len(seg_r)
        for i in range(len(seg_r)):
            segs_r, segs_l = seg_r[i], seg_l[i]
            if not segs_r:
                continue
            if sum(float(sl) for sl in segs_l) < args.min_length:
                continue
            all_r = np.concatenate([np.atleast_1d(s).astype(float) for s in segs_r])
            all_r = all_r[all_r > 0]
            if len(all_r) < 3:
                continue
            all_mean.append(np.mean(all_r))
            all_std.append(np.std(all_r))
        logger.info(f"  {npz.name}: {len(seg_r)} axons")

    all_x = np.array(all_mean)
    all_s = np.array(all_std)
    valid = (all_x > 0) & np.isfinite(all_s)
    all_x, all_s = all_x[valid], all_s[valid]
    if len(all_x) == 0:
        logger.error("No axons loaded")
        return

    x_max = np.percentile(all_x, 99.5)
    edges = np.linspace(0, x_max, N_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    med = np.full(N_BINS, np.nan)
    q25 = np.full(N_BINS, np.nan)
    q75 = np.full(N_BINS, np.nan)
    for i in range(N_BINS):
        m = (all_x >= edges[i]) & (all_x < edges[i + 1])
        if np.sum(m) >= MIN_AXONS_PER_BIN:
            med[i] = np.median(all_s[m])
            q25[i] = np.percentile(all_s[m], 25)
            q75[i] = np.percentile(all_s[m], 75)
    vb = ~np.isnan(med)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        'mean_radius_um': centers[vb],
        'std_median': med[vb], 'std_q25': q25[vb], 'std_q75': q75[vb],
    }).to_csv(args.out_dir / 'fig_s1_std_vs_radius.csv', index=False)

    logger.info(f"Total axons: {total}, used: {len(all_x)}; "
                f"corr(mean, std) = {np.corrcoef(all_x, all_s)[0, 1]:.3f}")
    logger.info(f"Wrote fig_s1_std_vs_radius.csv to {args.out_dir}/")


if __name__ == '__main__':
    main()
