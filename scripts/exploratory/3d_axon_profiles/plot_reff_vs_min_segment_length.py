#!/usr/bin/env python3
"""Plot r_arith and r_eff as a function of minimum segment length threshold.

One subplot per ROI. Shows how filtering short segments affects radius estimates.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from axonometry import get_plot_settings, style_axis


def compute_metrics_by_min_length(npz_path, thresholds):
    """Compute r_arith and r_eff for each min segment length threshold.

    Uses only the longest segment per axon.
    """
    d = np.load(npz_path, allow_pickle=True)
    seg_radii = d['segment_radii_um']
    seg_lengths = d['segment_lengths_um']

    # Pick only the longest segment per axon
    flat_radii = []
    flat_seg_len = []
    for i in range(len(seg_radii)):
        lengths = [float(sl) for sl in seg_lengths[i]]
        if len(lengths) == 0:
            continue
        best = int(np.argmax(lengths))
        r = np.atleast_1d(seg_radii[i][best]).astype(float)
        flat_radii.append(r)
        flat_seg_len.append(lengths[best])

    flat_seg_len = np.array(flat_seg_len)

    r_ariths = []
    r_effs = []
    counts = []
    for t in thresholds:
        mask = flat_seg_len >= t
        if mask.sum() == 0:
            r_ariths.append(np.nan)
            r_effs.append(np.nan)
            counts.append(0)
            continue
        r = np.concatenate([flat_radii[i] for i in np.where(mask)[0]])
        r_ariths.append(np.mean(r))
        r2 = np.mean(r**2)
        r6 = np.mean(r**6)
        r_effs.append((r6 / r2) ** 0.25 if r2 > 0 else np.nan)
        counts.append(len(r))

    return np.array(r_ariths), np.array(r_effs), np.array(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=Path('data/processed/rat/lm'))
    parser.add_argument('--output', type=Path,
                        default=Path('fig/debug/reff_vs_min_segment_length.png'))
    args = parser.parse_args()

    settings = get_plot_settings()
    thresholds = np.arange(0, 45, 5, dtype=float)

    files = sorted(args.data_dir.glob('*_axon_profiles.npz'))
    n = len(files)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)

    for idx, f in enumerate(files):
        ax = axes[idx // ncols][idx % ncols]
        name = f.stem.replace('_myelin_axon_profiles', '')

        r_ariths, r_effs, counts = compute_metrics_by_min_length(f, thresholds)

        ax.plot(thresholds, r_ariths, 'o-', color=settings.colors['binary_a'],
                markersize=4, linewidth=1.5, label=r'$\bar{r}$')
        ax.plot(thresholds, r_effs, 's-', color=settings.colors['binary_b'],
                markersize=4, linewidth=1.5, label=r'$r_{\mathrm{MRI}}$')

        style_axis(ax, xlabel='Min segment length [μm]', ylabel='Radius [μm]')
        ax.set_xlim(-1, 42)
        ax.text(0.95, 0.95, name, transform=ax.transAxes,
                ha='right', va='top', fontsize=8)
        if idx == 0:
            ax.legend(fontsize=settings.fonts['legend_size'])

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
