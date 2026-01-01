#!/usr/bin/env python3
"""
Plot three axon radius distributions side by side with different sample sizes.

Creates a 1×3 figure showing histograms from 10², 10⁴, and 10⁶ axons
sampled from a single human corpus callosum section.

Usage:
    python examples/plot_sample_size_histograms.py \
        --human-data data/raw_LM \
        --output fig/sample_size_histograms.png
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from axonometry import get_plot_settings, style_axis

settings = get_plot_settings()

# Sample sizes: 10², 10⁴, 10⁶
SAMPLE_SIZES = [100, 10_000, 1_000_000]
SAMPLE_SIZE_LABELS = [r'$n = 10^2$', r'$n = 10^4$', r'$n = 10^6$']


def load_human_cc_histograms(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load human corpus callosum histogram data."""
    bin_edges_file = data_dir / 'desc-binEdges_radii.tsv'
    counts_file = data_dir / 'desc-countsCircularEq_radii.tsv'

    bin_edges = np.loadtxt(bin_edges_file, delimiter='\t', skiprows=1)
    counts_matrix = np.loadtxt(counts_file, delimiter='\t', skiprows=1, dtype=float)

    return bin_edges, counts_matrix


def sample_radii_from_histogram(
    bin_edges: np.ndarray,
    counts: np.ndarray,
    n_samples: int,
    rng: np.random.Generator
) -> np.ndarray:
    """Sample individual radii from histogram data."""
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = np.diff(bin_edges).mean()

    # Normalize to probabilities
    probs = counts / counts.sum()

    # Sample bin indices
    sampled_bins = rng.choice(len(bin_centers), size=n_samples, p=probs)

    # Add uniform jitter within each bin
    jitter = rng.uniform(-bin_width / 2, bin_width / 2, n_samples)
    samples = bin_centers[sampled_bins] + jitter

    return samples


def main():
    parser = argparse.ArgumentParser(
        description='Plot sample size effect on radius distributions'
    )
    parser.add_argument('--human-data', type=Path, required=True,
                        help='Directory containing human CC TSV files')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output PNG/SVG file path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--roi', type=int, default=None,
                        help='ROI index (0-based). If not specified, picks one randomly.')

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Load data
    bin_edges, counts_matrix = load_human_cc_histograms(args.human_data)

    # Find ROIs with >= 10^6 axons
    roi_totals = counts_matrix.sum(axis=1)
    valid_rois = np.where(roi_totals >= max(SAMPLE_SIZES))[0]

    if len(valid_rois) == 0:
        raise ValueError(f"No ROIs have >= {max(SAMPLE_SIZES):,} axons")

    # Select ROI
    if args.roi is not None:
        roi_idx = args.roi
        if roi_idx not in valid_rois:
            raise ValueError(f"ROI {roi_idx} has only {int(roi_totals[roi_idx]):,} axons")
    else:
        roi_idx = rng.choice(valid_rois)

    counts = counts_matrix[roi_idx]
    print(f"Using ROI {roi_idx + 1} with {int(roi_totals[roi_idx]):,} total axons")

    # Sample radii for each sample size
    samples = []
    for n in SAMPLE_SIZES:
        sampled_radii = sample_radii_from_histogram(bin_edges, counts, n, rng)
        samples.append(sampled_radii)
        print(f"  n={n:,}: mean={sampled_radii.mean():.3f} μm, std={sampled_radii.std():.3f} μm")

    # Create figure (width:height ratio per subplot is 1:1.2)
    fig, axes = plt.subplots(1, 3, figsize=(7, 3), sharex=True, sharey=True,
                             gridspec_kw={'wspace': 0})

    # Common histogram settings
    hist_bins = np.arange(0, 5.05, 0.1)  # 0 to 5 μm in 0.1 μm bins
    from matplotlib.cm import YlOrBr
    colors = [YlOrBr(x) for x in [0.3, 0.65, 1.0]]

    for ax, radii, label, color in zip(axes, samples, SAMPLE_SIZE_LABELS, colors):
        ax.hist(radii, bins=hist_bins, color=color, alpha=0.7, edgecolor='white',
                linewidth=0.5, density=True)

        style_axis(ax, xlabel='Axon radius [μm]', ylabel='Probability density')
        ax.set_xlim(0, 3.5)
        ax.set_title(label, fontsize=settings.fonts['label_size'])

        # Add inset for tail region
        inset = ax.inset_axes([0.4, 0.4, 0.55, 0.55])  # [x, y, width, height] in axes coords
        inset.hist(radii, bins=hist_bins, color=color, alpha=0.7, edgecolor='white',
                   linewidth=0.3, density=True)
        inset.set_xlim(1, 3.5)
        inset.set_ylim(0, 0.15)
        inset.set_yticks([0, 0.05, 0.10, 0.15])
        inset.tick_params(labelsize=8)
        inset.set_xlabel('')
        inset.set_ylabel('')

    # Only show y-label on first panel, x-label on middle panel
    axes[1].set_ylabel('')
    axes[2].set_ylabel('')
    axes[0].set_xlabel('')
    axes[2].set_xlabel('')

    fig.subplots_adjust(wspace=0, left=0.08, right=0.98, bottom=0.15, top=0.9)

    plt.savefig(args.output, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {args.output}")


if __name__ == '__main__':
    main()
