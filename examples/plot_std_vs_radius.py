#!/usr/bin/env python3
"""
Plot standard deviation of radius vs mean radius for axons.

This analysis shows how radius variability (std) scales with axon size.
Supplementary figure for the main individual axon statistics analysis.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from axonometry import get_plot_settings, add_panel_labels, style_axis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings = get_plot_settings()


def load_axon_data(npz_file: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load axon data from NPZ file.

    Args:
        npz_file: Path to 3D axon profiles NPZ file

    Returns:
        Tuple of (mean_radii, std_radii, sample_name)
    """
    data = np.load(npz_file, allow_pickle=True)

    mean_radii = data['mean_radii_um']
    std_radii = data['std_radii_um']

    # Extract sample name from filename
    sample_name = npz_file.stem.replace('_axon_profiles', '')

    # Filter valid data
    valid = (mean_radii > 0) & np.isfinite(std_radii)
    mean_radii = mean_radii[valid]
    std_radii = std_radii[valid]

    logger.info(f"Loaded {len(mean_radii)} axons from {npz_file.name}")

    return mean_radii, std_radii, sample_name


def plot_std_vs_radius(
    all_data: List[Tuple[np.ndarray, np.ndarray, str]],
    output_file: Path,
):
    """
    Plot standard deviation of radius vs mean radius.

    Args:
        all_data: List of (mean_radii, std_radii, sample_name) tuples
        output_file: Output PNG file path
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Pool all data
    all_x = np.concatenate([r for r, _, _ in all_data])
    all_std = np.concatenate([s for _, s, _ in all_data])

    # Get settings
    font_settings = settings.fonts
    line_settings = settings.line
    main_color = settings.get_group_color('sham')

    # Compute binned statistics
    x_max = np.percentile(all_x, 99.5)
    n_bins = 30
    bin_edges = np.linspace(0, x_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_medians = []
    bin_q25 = []
    bin_q75 = []

    for i in range(n_bins):
        mask = (all_x >= bin_edges[i]) & (all_x < bin_edges[i + 1])
        count = np.sum(mask)
        if count >= 100:
            bin_medians.append(np.median(all_std[mask]))
            bin_q25.append(np.percentile(all_std[mask], 25))
            bin_q75.append(np.percentile(all_std[mask], 75))
        else:
            bin_medians.append(np.nan)
            bin_q25.append(np.nan)
            bin_q75.append(np.nan)

    bin_medians = np.array(bin_medians)
    bin_q25 = np.array(bin_q25)
    bin_q75 = np.array(bin_q75)
    valid = ~np.isnan(bin_medians)

    # Plot
    ax.plot(bin_centers[valid], bin_medians[valid], color=main_color, linestyle='-',
            linewidth=line_settings['linewidth'], marker='o',
            markersize=line_settings['marker_size'], label='Median')
    ax.fill_between(bin_centers[valid], bin_q25[valid], bin_q75[valid],
                    color=main_color, alpha=line_settings['fill_alpha'], label='IQR (25-75%)')

    style_axis(ax, xlabel='Along-axon mean radius (μm)', ylabel='Std of Radius (μm)')
    ax.legend(loc='upper left', fontsize=font_settings['legend_size'])
    ax.set_xlim(0, x_max)
    ax.set_box_aspect(1)  # Square subplot

    plt.tight_layout()

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plot to {output_file}")

    # Print summary statistics
    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary Statistics")
    logger.info("=" * 60)
    logger.info(f"Total axons: {len(all_x)}")
    logger.info(f"Mean radius: {np.mean(all_x):.3f} +/- {np.std(all_x):.3f} μm")
    logger.info(f"Mean std of radius: {np.mean(all_std):.3f} +/- {np.std(all_std):.3f} μm")
    logger.info(f"Correlation (mean radius vs std): {np.corrcoef(all_x, all_std)[0, 1]:.3f}")
    logger.info("=" * 60)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Plot standard deviation of radius vs mean radius for axons.'
    )
    parser.add_argument(
        'input_files',
        type=str,
        help='Input NPZ file(s) with axon profiles (glob pattern supported)'
    )
    parser.add_argument(
        'output_file',
        type=Path,
        help='Output PNG file path'
    )

    args = parser.parse_args()

    # Expand glob pattern
    input_path = Path(args.input_files)
    if '*' in str(input_path):
        input_files = sorted(input_path.parent.glob(input_path.name))
    else:
        input_files = [input_path]

    if not input_files:
        logger.error(f"No files found matching: {args.input_files}")
        return

    logger.info(f"Found {len(input_files)} axon profile file(s)")

    # Load all data
    all_data = []
    for npz_file in input_files:
        mean_radii, std_radii, sample_name = load_axon_data(npz_file)
        all_data.append((mean_radii, std_radii, sample_name))

    # Create plot
    plot_std_vs_radius(all_data, args.output_file)


if __name__ == '__main__':
    main()
