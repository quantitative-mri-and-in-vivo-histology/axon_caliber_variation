#!/usr/bin/env python3
"""
Plot coefficient of variation (CV) of axon caliber vs mean axon radius.

This script reads 3D skeleton-based axon profiles and creates a scatter plot showing
the relationship between along-axon caliber variability (CV = std/mean) and mean
axon radius.

CV (coefficient of variation) quantifies how much the axon caliber varies along
its length relative to the mean caliber. Higher CV indicates more variable caliber.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_group(sample_name: str) -> str:
    """
    Extract group (TBI/Sham) from sample name.

    Args:
        sample_name: Sample name like "sham_25_ipsi", "tbi_2_contra", etc.

    Returns:
        "TBI" or "Sham"
    """
    name_lower = sample_name.lower()
    if 'tbi' in name_lower:
        return "TBI"
    elif 'sham' in name_lower:
        return "Sham"
    return "Unknown"


def load_axon_cv_data(npz_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Load axon data and compute CV for each axon.

    Args:
        npz_file: Path to 3D axon profiles NPZ file

    Returns:
        Tuple of (mean_radii, cv_values, std_values, sample_name)
    """
    data = np.load(npz_file, allow_pickle=True)

    mean_radii = data['mean_radii_um']
    std_radii = data['std_radii_um']

    # Compute CV = std / mean (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = std_radii / mean_radii
        cv = np.where(np.isfinite(cv) & (mean_radii > 0), cv, np.nan)

    # Extract sample name from filename
    sample_name = npz_file.stem.replace('_axon_profiles', '')

    # Filter out NaN values
    valid = np.isfinite(cv) & np.isfinite(mean_radii) & np.isfinite(std_radii)
    mean_radii = mean_radii[valid]
    cv = cv[valid]
    std_radii = std_radii[valid]

    logger.info(f"Loaded {len(mean_radii)} axons from {npz_file.name}")
    logger.info(f"  Mean radius range: {mean_radii.min():.3f} - {mean_radii.max():.3f} um")
    logger.info(f"  CV range: {cv.min():.3f} - {cv.max():.3f}")

    return mean_radii, cv, std_radii, sample_name


def find_axon_profile_files(pattern: str) -> List[Path]:
    """
    Find all axon profile NPZ files matching the pattern.

    Args:
        pattern: Glob pattern for axon profile files

    Returns:
        List of matching file paths
    """
    import glob as glob_module
    files = sorted([Path(f) for f in glob_module.glob(pattern, recursive=True)])
    return files


def plot_cv_vs_radius(
    all_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    output_file: Path,
    title: str = "Caliber Variation Analysis"
) -> None:
    """
    Create four-panel plot: radius histogram, CV histogram, CV vs radius, std vs radius.

    Args:
        all_data: List of (mean_radii, cv_values, std_values, sample_name) tuples
        output_file: Output PNG file path
        title: Plot title
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    ax0, ax1, ax2, ax3 = axes

    # Pool all data
    all_x = np.concatenate([r for r, _, _, _ in all_data])
    all_y = np.concatenate([cv for _, cv, _, _ in all_data])
    all_std = np.concatenate([std for _, _, std, _ in all_data])

    # === Panel 0: Radius histogram ===
    ax0.hist(all_x, bins=50, color='#1f77b4', edgecolor='black', alpha=0.7)
    ax0.axvline(np.mean(all_x), color='red', linestyle='--', linewidth=2,
                label=f'Mean = {np.mean(all_x):.3f}')
    ax0.axvline(np.median(all_x), color='orange', linestyle=':', linewidth=2,
                label=f'Median = {np.median(all_x):.3f}')

    ax0.set_xlabel('Mean Axon Radius (um)', fontsize=12, fontweight='bold')
    ax0.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax0.set_title(f'Radius Distribution (n = {len(all_x):,})', fontsize=12, fontweight='bold')
    ax0.legend(loc='upper right', fontsize=10)
    ax0.grid(True, alpha=0.3)

    # === Panel 1: CV histogram ===
    ax1.hist(all_y, bins=50, color='#1f77b4', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(all_y), color='red', linestyle='--', linewidth=2,
                label=f'Mean = {np.mean(all_y):.3f}')
    ax1.axvline(np.median(all_y), color='orange', linestyle=':', linewidth=2,
                label=f'Median = {np.median(all_y):.3f}')

    ax1.set_xlabel('Coefficient of Variation (CV)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('CV Distribution', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # === Panel 2: CV vs radius trend ===
    # Compute binned statistics
    x_max = np.max(all_x) * 1.05
    n_bins = 25
    bin_edges = np.linspace(0, x_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means = []
    bin_stds = []
    bin_counts = []

    for i in range(n_bins):
        mask = (all_x >= bin_edges[i]) & (all_x < bin_edges[i + 1])
        count = np.sum(mask)
        if count >= 10:
            bin_means.append(np.mean(all_y[mask]))
            bin_stds.append(np.std(all_y[mask]))
            bin_counts.append(count)
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
            bin_counts.append(0)

    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    valid = ~np.isnan(bin_means)

    ax2.plot(bin_centers[valid], bin_means[valid], 'b-', linewidth=2, marker='o',
             markersize=5, label='Mean CV')
    ax2.fill_between(bin_centers[valid],
                    bin_means[valid] - bin_stds[valid],
                    bin_means[valid] + bin_stds[valid],
                    color='blue', alpha=0.2, label='±1 std')

    # Compute correlation
    r, p = stats.pearsonr(all_x, all_y)

    ax2.set_xlabel('Mean Axon Radius (um)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Coefficient of Variation (CV)', fontsize=12, fontweight='bold')
    ax2.set_title(f'CV vs Radius\nr = {r:.3f}, p = {p:.2e}', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, x_max)

    # === Panel 3: Absolute std vs radius trend ===
    # Compute binned statistics for std
    bin_means_std = []
    bin_stds_std = []

    for i in range(n_bins):
        mask = (all_x >= bin_edges[i]) & (all_x < bin_edges[i + 1])
        count = np.sum(mask)
        if count >= 10:
            bin_means_std.append(np.mean(all_std[mask]))
            bin_stds_std.append(np.std(all_std[mask]))
        else:
            bin_means_std.append(np.nan)
            bin_stds_std.append(np.nan)

    bin_means_std = np.array(bin_means_std)
    bin_stds_std = np.array(bin_stds_std)
    valid_std = ~np.isnan(bin_means_std)

    ax3.plot(bin_centers[valid_std], bin_means_std[valid_std], 'b-', linewidth=2, marker='o',
             markersize=5, label='Mean std')
    ax3.fill_between(bin_centers[valid_std],
                    bin_means_std[valid_std] - bin_stds_std[valid_std],
                    bin_means_std[valid_std] + bin_stds_std[valid_std],
                    color='blue', alpha=0.2, label='±1 std')

    # Compute correlation for std vs radius
    r_std, p_std = stats.pearsonr(all_x, all_std)

    ax3.set_xlabel('Mean Axon Radius (um)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Std of Radius (um)', fontsize=12, fontweight='bold')
    ax3.set_title(f'Std vs Radius\nr = {r_std:.3f}, p = {p_std:.2e}', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, x_max)

    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plot to {output_file}")
    logger.info(f"Correlation: r = {r:.3f}, p = {p:.2e}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Plot coefficient of variation vs mean axon radius from 3D skeleton data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all HM axon profiles
  python plot_cv_vs_radius.py \\
      "data/processed/HM/*_axon_profiles.npz" \\
      fig/cv_vs_radius.png

  # Process specific files
  python plot_cv_vs_radius.py \\
      "data/processed/HM/sham_*_axon_profiles.npz" \\
      fig/sham_cv_vs_radius.png

  # Single file
  python plot_cv_vs_radius.py \\
      data/processed/HM/sham_25_ipsi_axon_profiles.npz \\
      fig/sham_25_ipsi_cv.png
        """
    )

    parser.add_argument('input', type=str,
                        help='Input: single NPZ file or glob pattern for 3D axon profiles')
    parser.add_argument('output', type=Path,
                        help='Output PNG file path')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom plot title (default: auto-generated)')

    args = parser.parse_args()

    # Find matching files
    files = find_axon_profile_files(args.input)

    if not files:
        logger.error(f"No files found matching pattern: {args.input}")
        return

    logger.info(f"Found {len(files)} axon profile file(s)")

    # Load all data
    all_data = []
    for f in files:
        mean_radii, cv, std_radii, sample_name = load_axon_cv_data(f)
        all_data.append((mean_radii, cv, std_radii, sample_name))

    # Generate title if not provided
    if args.title:
        title = args.title
    else:
        title = "Radius variability along individual axons"

    # Create plot
    plot_cv_vs_radius(all_data, args.output, title)

    # Print summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("Summary Statistics")
    logger.info("=" * 60)

    all_cv = np.concatenate([cv for _, cv, _, _ in all_data])
    all_radii = np.concatenate([r for r, _, _, _ in all_data])
    all_std = np.concatenate([s for _, _, s, _ in all_data])

    logger.info(f"Total axons: {len(all_cv)}")
    logger.info(f"Mean radius: {np.mean(all_radii):.3f} +/- {np.std(all_radii):.3f} um")
    logger.info(f"Mean CV: {np.mean(all_cv):.3f} +/- {np.std(all_cv):.3f}")
    logger.info(f"Median CV: {np.median(all_cv):.3f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
