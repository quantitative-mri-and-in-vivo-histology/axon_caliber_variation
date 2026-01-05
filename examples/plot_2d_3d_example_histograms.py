#!/usr/bin/env python3
"""
Plot example histograms comparing 3D vs 2D mid-slice distributions.

Creates a 1×2 figure showing:
- Left: 3D distribution (all radii from axon profiles)
- Right: 2D mid-slice distribution (single slice from center)

Usage:
    python examples/plot_2d_3d_example_histograms.py \
        --data-dir data/processed/LM \
        --output fig/2d_3d_example_histograms.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from axonometry import get_plot_settings, style_axis

settings = get_plot_settings()


def find_largest_sample(data_dir: Path, population: str = 'cc') -> Tuple[Path, Path, str]:
    """
    Find the sample with the most total axons.

    Returns:
        Tuple of (slice_profile_path, axon_profile_path, sample_name)
    """
    pattern = f"*_{population}_slice_profiles.npz"
    slice_files = sorted(data_dir.glob(pattern))

    if not slice_files:
        raise FileNotFoundError(f"No {pattern} files found in {data_dir}")

    best_file = None
    best_count = 0

    for f in slice_files:
        npz = np.load(f)
        total = npz['n_axons_per_slice'].sum()
        if total > best_count:
            best_count = total
            best_file = f

    base_name = best_file.stem.replace(f'_{population}_slice_profiles', '')
    sample_name = f"{base_name}_{population.upper()}"
    axon_file = data_dir / f"{base_name}_axon_profiles.npz"

    print(f"Using sample: {sample_name} with {best_count:,} total axons")

    return best_file, axon_file, sample_name, base_name


def load_population_labels(data_dir: Path, base_name: str, population: str) -> Set[int]:
    """Load axon labels for a specific population from JSON file."""
    pop_json = data_dir / f"{base_name}_populations.json"

    with open(pop_json, 'r') as f:
        data = json.load(f)

    for pop in data['populations']:
        if pop['name'].lower() == population.lower():
            return set(pop['axon_labels'])

    raise ValueError(f"Population '{population}' not found in {pop_json}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot 3D vs 2D mid-slice distribution comparison'
    )
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed/LM'),
                        help='Directory containing LM slice and axon profiles')
    parser.add_argument('--output', type=Path, default=Path('fig/2d_3d_example_histograms.png'),
                        help='Output PNG/SVG file path')
    parser.add_argument('--population', type=str, default='cc',
                        choices=['cc', 'cg'],
                        help='Population to use')
    parser.add_argument('--radius-type', type=str, default='circular',
                        choices=['circular', 'minor'],
                        help='Radius type to use')

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Find largest sample
    slice_file, axon_file, sample_name, base_name = find_largest_sample(
        args.data_dir, args.population)

    # Load population labels
    pop_labels = load_population_labels(args.data_dir, base_name, args.population)
    print(f"Population '{args.population.upper()}' has {len(pop_labels)} axons")

    # Load 2D slice data
    npz_2d = np.load(slice_file)
    bin_centers = npz_2d['bin_centers']
    histograms = npz_2d[f'histograms_{args.radius_type}']
    n_axons_per_slice = npz_2d['n_axons_per_slice']
    bin_width = bin_centers[1] - bin_centers[0]

    # Find mid-slice (with most axons near center)
    n_slices = histograms.shape[0]
    center_region = slice(n_slices // 3, 2 * n_slices // 3)
    center_axon_counts = n_axons_per_slice[center_region]
    mid_slice_idx = n_slices // 3 + np.argmax(center_axon_counts)
    mid_slice_hist = histograms[mid_slice_idx]
    mid_slice_axons = int(n_axons_per_slice[mid_slice_idx])

    print(f"Mid-slice index: {mid_slice_idx} with {mid_slice_axons:,} axons")

    # Load 3D axon data and filter by population
    npz_3d = np.load(axon_file, allow_pickle=True)
    axon_labels = npz_3d['labels']
    radii_profiles = npz_3d['radii_profiles_um']

    all_radii_3d = []
    for i, label in enumerate(axon_labels):
        if int(label) in pop_labels:
            all_radii_3d.extend(radii_profiles[i])
    all_radii_3d = np.array(all_radii_3d)

    print(f"3D distribution: {len(all_radii_3d):,} radius measurements")

    # Use 0.05 μm binning for both panels
    common_bin_edges = np.arange(0, 5.025, 0.05)
    common_bin_centers = (common_bin_edges[:-1] + common_bin_edges[1:]) / 2
    common_bin_width = 0.05

    # Rebin 3D data to 0.05 μm bins
    hist_3d, _ = np.histogram(all_radii_3d, bins=common_bin_edges)
    pdf_3d = hist_3d / (hist_3d.sum() * common_bin_width)

    # Rebin 2D mid-slice to 0.05 μm bins (aggregate from finer original bins)
    hist_2d_rebinned, _ = np.histogram(
        bin_centers, bins=common_bin_edges, weights=mid_slice_hist
    )
    pdf_2d = hist_2d_rebinned / (hist_2d_rebinned.sum() * common_bin_width)

    # Create 1×2 figure
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))

    # Colors: peach for 3D (reference/full), rust for 2D (sample)
    color_3d = '#F8B878'
    color_2d = '#B84820'

    # Inset range for tail (adjusted for visible data)
    inset_xmin, inset_xmax = 0.4, 1.2

    # Left panel: 3D distribution
    ax = axes[0]
    ax.bar(common_bin_centers, pdf_3d, width=common_bin_width * 0.9, color=color_3d,
           edgecolor='white', linewidth=0.3)
    style_axis(ax, xlabel='Axon radius [μm]', ylabel='Prob. density [μm⁻¹]')
    ax.set_xlim(0, 2.5)

    # Add tail inset for 3D
    inset_left = ax.inset_axes([0.42, 0.42, 0.55, 0.55])
    mask = (common_bin_centers >= inset_xmin) & (common_bin_centers <= inset_xmax)
    inset_left.bar(common_bin_centers[mask], pdf_3d[mask], width=common_bin_width * 0.9,
                   color=color_3d, edgecolor='white', linewidth=0.3)
    inset_left.set_xlim(inset_xmin, inset_xmax)
    inset_left.set_ylim(0, pdf_3d[mask].max() * 1.1)
    inset_left.tick_params(labelsize=settings.fonts['tick_size'] - 3)
    inset_left.set_xlabel('')
    inset_left.set_ylabel('')

    # Right panel: 2D mid-slice distribution
    ax = axes[1]
    ax.bar(common_bin_centers, pdf_2d, width=common_bin_width * 0.9, color=color_2d,
           edgecolor='white', linewidth=0.3)
    style_axis(ax, xlabel='Axon radius [μm]', ylabel='Prob. density [μm⁻¹]')
    ax.set_xlim(0, 2.5)

    # Add tail inset for 2D
    inset_right = ax.inset_axes([0.42, 0.42, 0.55, 0.55])
    inset_right.bar(common_bin_centers[mask], pdf_2d[mask], width=common_bin_width * 0.9,
                    color=color_2d, edgecolor='white', linewidth=0.3)
    inset_right.set_xlim(inset_xmin, inset_xmax)
    inset_right.set_ylim(0, pdf_3d[mask].max() * 1.1)  # Same y-scale as left inset
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
