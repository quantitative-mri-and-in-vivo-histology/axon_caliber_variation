#!/usr/bin/env python3
"""
Plot axon radius distributions comparing full LM data vs subsample.

Creates a 1×2 figure showing:
- Left: Full histogram from ROI with most axons
- Right: Subsample of 10³ axons from the same ROI

Usage:
    python scripts/exploratory/distribution_fitting/plot_sample_size_histograms.py \
        --human-data data/raw_LM \
        --output fig/sample_size_histograms.png
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

# Find repo root (contains pyproject.toml)
_root = Path(__file__).resolve().parent
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from axonometry import get_plot_settings, style_axis

settings = get_plot_settings()

# Subsample size
SUBSAMPLE_SIZE = 1000


def load_human_cc_histograms(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
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

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Load data
    bin_edges, counts_matrix = load_human_cc_histograms(args.human_data)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Find ROI with most axons
    roi_totals = counts_matrix.sum(axis=1)
    roi_idx = np.argmax(roi_totals)
    counts = counts_matrix[roi_idx]
    total_axons = int(roi_totals[roi_idx])

    print(f"Using ROI {roi_idx + 1} with {total_axons:,} total axons")

    # Sample 10³ radii from histogram
    sampled_radii = sample_radii_from_histogram(bin_edges, counts, SUBSAMPLE_SIZE, rng)
    print(f"  Subsample n={SUBSAMPLE_SIZE:,}: mean={sampled_radii.mean():.3f} μm")

    # Create 1×2 figure
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))

    # Colors: peach for full, rust for subsample
    color_full = '#F8B878'
    color_subsample = '#B84820'

    # Use 0.1 μm binning for both panels
    common_bin_edges = np.arange(0, 5.05, 0.1)
    common_bin_centers = (common_bin_edges[:-1] + common_bin_edges[1:]) / 2
    common_bin_width = 0.1

    # Rebin full data to 0.1 μm bins (aggregate from finer original bins)
    full_counts_rebinned, _ = np.histogram(
        bin_centers, bins=common_bin_edges, weights=counts
    )
    pdf_full = full_counts_rebinned / (full_counts_rebinned.sum() * common_bin_width)

    # Bin subsample using same 0.1 μm bins
    subsample_counts, _ = np.histogram(sampled_radii, bins=common_bin_edges)
    pdf_subsample = subsample_counts / (subsample_counts.sum() * common_bin_width)

    # Inset range for tail
    inset_xmin, inset_xmax = 1.5, 4.0

    # Left panel: Full LM histogram
    ax = axes[0]
    ax.bar(common_bin_centers, pdf_full, width=common_bin_width * 0.9, color=color_full,
           edgecolor='white', linewidth=0.3)
    style_axis(ax, xlabel='Axon radius [μm]', ylabel='Prob. density [μm⁻¹]')
    ax.set_xlim(0, 2.5)

    # Add tail inset for full data
    inset_left = ax.inset_axes([0.42, 0.42, 0.55, 0.55])
    mask = (common_bin_centers >= inset_xmin) & (common_bin_centers <= inset_xmax)
    inset_left.bar(common_bin_centers[mask], pdf_full[mask], width=common_bin_width * 0.9,
                   color=color_full, edgecolor='white', linewidth=0.3)
    inset_left.set_xlim(inset_xmin, inset_xmax)
    inset_left.set_ylim(0, pdf_full[mask].max() * 1.1)
    inset_left.tick_params(labelsize=settings.fonts['tick_size'] - 3)
    inset_left.set_xlabel('')
    inset_left.set_ylabel('')

    # Right panel: Subsample histogram
    ax = axes[1]
    ax.bar(common_bin_centers, pdf_subsample, width=common_bin_width * 0.9, color=color_subsample,
           edgecolor='white', linewidth=0.3)
    style_axis(ax, xlabel='Axon radius [μm]', ylabel='Prob. density [μm⁻¹]')
    ax.set_xlim(0, 2.5)

    # Add tail inset for subsample
    inset_right = ax.inset_axes([0.42, 0.42, 0.55, 0.55])
    inset_right.bar(common_bin_centers[mask], pdf_subsample[mask], width=common_bin_width * 0.9,
                    color=color_subsample, edgecolor='white', linewidth=0.3)
    inset_right.set_xlim(inset_xmin, inset_xmax)
    inset_right.set_ylim(0, pdf_full[mask].max() * 1.1)  # Same y-scale as left inset
    inset_right.tick_params(labelsize=settings.fonts['tick_size'] - 3)
    inset_right.set_xlabel('')
    inset_right.set_ylabel('')

    plt.tight_layout()

    plt.savefig(args.output, dpi=settings.figure['dpi'], bbox_inches='tight')

    # Also save SVG
    svg_output = args.output.with_suffix('.svg')
    plt.savefig(svg_output, bbox_inches='tight')

    plt.close()
    print(f"Saved figure to {args.output} and {svg_output}")


if __name__ == '__main__':
    main()
