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

from axonometry import get_plot_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load plot settings
settings = get_plot_settings()


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


def load_axon_cv_data(npz_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, dict]:
    """
    Load axon data and compute CV for each axon.

    Args:
        npz_file: Path to 3D axon profiles NPZ file

    Returns:
        Tuple of (mean_radii, cv_values, std_values, sample_name, profiles_dict)
        profiles_dict contains 'radii_profiles', 'lengths', 'cv' for representative axon selection
    """
    data = np.load(npz_file, allow_pickle=True)

    mean_radii = data['mean_radii_um']
    std_radii = data['std_radii_um']
    lengths = data['lengths_um']
    radii_profiles = data['radii_profiles_um']

    # Compute CV = std / mean (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = std_radii / mean_radii
        cv = np.where(np.isfinite(cv) & (mean_radii > 0), cv, np.nan)

    # Extract sample name from filename
    sample_name = npz_file.stem.replace('_axon_profiles', '')

    # Store profiles for representative axon selection (before filtering)
    profiles_dict = {
        'radii_profiles': radii_profiles,
        'lengths': lengths,
        'cv': cv.copy(),
        'mean_radii': mean_radii.copy()
    }

    # Filter out NaN values
    valid = np.isfinite(cv) & np.isfinite(mean_radii) & np.isfinite(std_radii)
    mean_radii = mean_radii[valid]
    cv = cv[valid]
    std_radii = std_radii[valid]

    logger.info(f"Loaded {len(mean_radii)} axons from {npz_file.name}")
    logger.info(f"  Mean radius range: {mean_radii.min():.3f} - {mean_radii.max():.3f} um")
    logger.info(f"  CV range: {cv.min():.3f} - {cv.max():.3f}")

    return mean_radii, cv, std_radii, sample_name, profiles_dict


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


def select_representative_axons(
    all_profiles: List[dict],
    min_length: float = 50.0,
    seed: int = 42
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Select 3 representative axons with low, mid, and high CV.

    Args:
        all_profiles: List of profile dicts from each file
        min_length: Minimum axon length in μm
        seed: Random seed for reproducibility

    Returns:
        List of (arc_lengths, radii, cv) tuples for 3 representative axons
    """
    np.random.seed(seed)

    # Pool all valid axons with length >= min_length
    candidates = []
    for profiles_dict in all_profiles:
        radii_profiles = profiles_dict['radii_profiles']
        lengths = profiles_dict['lengths']
        cv_values = profiles_dict['cv']
        mean_radii = profiles_dict['mean_radii']

        for i in range(len(lengths)):
            if lengths[i] >= min_length and np.isfinite(cv_values[i]) and mean_radii[i] >= 0.1:
                profile = radii_profiles[i]
                if len(profile) > 10:  # Need enough points
                    candidates.append({
                        'profile': profile,
                        'length': lengths[i],
                        'cv': cv_values[i],
                        'mean_radius': mean_radii[i]
                    })

    if len(candidates) < 3:
        return []

    # Sort by CV
    candidates.sort(key=lambda x: x['cv'])

    # Select low (5th percentile), mid (50th), high (95th) CV
    n = len(candidates)
    idx_low = int(n * 0.05)
    idx_mid = int(n * 0.50)
    idx_high = int(n * 0.95)

    selected = [candidates[idx_low], candidates[idx_mid], candidates[idx_high]]

    result = []
    for axon in selected:
        profile = axon['profile']
        length = axon['length']
        n_points = len(profile)
        arc_lengths = np.linspace(0, length, n_points)
        result.append((arc_lengths, profile, axon['cv']))

    return result


def plot_cv_vs_radius(
    all_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str, dict]],
    output_file: Path,
    title: str = "Caliber Variation Analysis"
) -> None:
    """
    Create four-panel plot: representative axons, CV histogram, CV vs radius, std vs radius.

    Args:
        all_data: List of (mean_radii, cv_values, std_values, sample_name, profiles_dict) tuples
        output_file: Output PNG file path
        title: Plot title
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    ax_rep, ax0, ax1, ax2 = axes

    # Pool all data
    all_x = np.concatenate([r for r, _, _, _, _ in all_data])
    all_y = np.concatenate([cv for _, cv, _, _, _ in all_data])
    all_std = np.concatenate([std for _, _, std, _, _ in all_data])

    # Get all profiles for representative selection
    all_profiles = [p for _, _, _, _, p in all_data]

    # Get settings
    hist_settings = settings.histogram
    font_settings = settings.fonts
    grid_settings = settings.grid
    line_settings = settings.line
    main_color = settings.get_group_color('sham')  # Use sham color as default blue

    # === Panel 0: Representative axon profiles ===
    rep_axons = select_representative_axons(all_profiles, min_length=50.0)
    cv_colors = ['#2ca02c', '#1f77b4', '#d62728']  # green (low), blue (mid), red (high)
    cv_labels = ['Low CV', 'Mid CV', 'High CV']

    for i, (arc, radii, cv_val) in enumerate(rep_axons):
        ax_rep.plot(arc, radii, color=cv_colors[i], linewidth=1.5,
                    label=f'{cv_labels[i]} (CV={cv_val:.2f})')

    ax_rep.set_xlabel('Arc length (μm)', fontsize=font_settings['label_size'],
                      fontweight=font_settings['weight'])
    ax_rep.set_ylabel('Radius (μm)', fontsize=font_settings['label_size'],
                      fontweight=font_settings['weight'])
    ax_rep.set_title('Representative Axons', fontsize=font_settings['label_size'],
                     fontweight=font_settings['weight'])
    ax_rep.legend(loc='upper right', fontsize=font_settings['legend_size'])
    ax_rep.grid(True, alpha=0.3)

    # === Panel 1: CV histogram ===
    ax0.hist(all_y, bins=hist_settings['bins'], color=main_color,
             edgecolor=hist_settings['edgecolor'], alpha=hist_settings['alpha'])
    ax0.axvline(np.mean(all_y), color=settings.colors['mean_line'], linestyle='--',
                linewidth=line_settings['linewidth'], label=f'Mean = {np.mean(all_y):.3f}')
    ax0.axvline(np.median(all_y), color=settings.colors['median_line'], linestyle=':',
                linewidth=line_settings['linewidth'], label=f'Median = {np.median(all_y):.3f}')

    ax0.set_xlabel('Coefficient of Variation (CV)', fontsize=font_settings['label_size'],
                   fontweight=font_settings['weight'])
    ax0.set_ylabel('Count', fontsize=font_settings['label_size'],
                   fontweight=font_settings['weight'])
    ax0.set_title(f'CV Distribution (n = {len(all_x):,})', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax0.legend(loc='upper right', fontsize=font_settings['legend_size'])
    ax0.grid(grid_settings['enabled'], alpha=grid_settings['alpha'])

    # === Panel 1: CV vs radius trend ===
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
            bin_medians.append(np.median(all_y[mask]))
            bin_q25.append(np.percentile(all_y[mask], 25))
            bin_q75.append(np.percentile(all_y[mask], 75))
        else:
            bin_medians.append(np.nan)
            bin_q25.append(np.nan)
            bin_q75.append(np.nan)

    bin_medians = np.array(bin_medians)
    bin_q25 = np.array(bin_q25)
    bin_q75 = np.array(bin_q75)
    valid = ~np.isnan(bin_medians)

    ax1.plot(bin_centers[valid], bin_medians[valid], color=main_color, linestyle='-',
             linewidth=line_settings['linewidth'], marker='o',
             markersize=line_settings['marker_size'], label='Median')
    ax1.fill_between(bin_centers[valid], bin_q25[valid], bin_q75[valid],
                     color=main_color, alpha=line_settings['fill_alpha'], label='IQR (25-75%)')

    # Compute correlation
    r, p = stats.pearsonr(all_x, all_y)

    ax1.set_xlabel('Mean Axon Radius (μm)', fontsize=font_settings['label_size'],
                   fontweight=font_settings['weight'])
    ax1.set_ylabel('Coefficient of Variation (CV)', fontsize=font_settings['label_size'],
                   fontweight=font_settings['weight'])
    ax1.set_title('CV vs Radius', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax1.legend(loc='upper right', fontsize=font_settings['legend_size'])
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, x_max)

    # === Panel 2: Std vs radius trend ===
    # Compute binned statistics for std
    bin_medians_std = []
    bin_q25_std = []
    bin_q75_std = []

    for i in range(n_bins):
        mask = (all_x >= bin_edges[i]) & (all_x < bin_edges[i + 1])
        count = np.sum(mask)
        if count >= 100:
            bin_medians_std.append(np.median(all_std[mask]))
            bin_q25_std.append(np.percentile(all_std[mask], 25))
            bin_q75_std.append(np.percentile(all_std[mask], 75))
        else:
            bin_medians_std.append(np.nan)
            bin_q25_std.append(np.nan)
            bin_q75_std.append(np.nan)

    bin_medians_std = np.array(bin_medians_std)
    bin_q25_std = np.array(bin_q25_std)
    bin_q75_std = np.array(bin_q75_std)
    valid_std = ~np.isnan(bin_medians_std)

    ax2.plot(bin_centers[valid_std], bin_medians_std[valid_std], color=main_color, linestyle='-',
             linewidth=line_settings['linewidth'], marker='o',
             markersize=line_settings['marker_size'], label='Median')
    ax2.fill_between(bin_centers[valid_std], bin_q25_std[valid_std], bin_q75_std[valid_std],
                     color=main_color, alpha=line_settings['fill_alpha'], label='IQR (25-75%)')

    # Compute correlation for std vs radius
    r_std, p_std = stats.pearsonr(all_x, all_std)

    ax2.set_xlabel('Mean Axon Radius (μm)', fontsize=font_settings['label_size'],
                   fontweight=font_settings['weight'])
    ax2.set_ylabel('Std of Radius (μm)', fontsize=font_settings['label_size'],
                   fontweight=font_settings['weight'])
    ax2.set_title('Std vs Radius', fontsize=font_settings['label_size'],
                  fontweight=font_settings['weight'])
    ax2.legend(loc='upper left', fontsize=font_settings['legend_size'])
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, x_max)

    # Overall title
    fig.suptitle(title, fontsize=font_settings['title_size'], fontweight=font_settings['weight'], y=1.02)

    plt.tight_layout()

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plot to {output_file}")
    logger.info(f"Correlation (CV vs radius): r = {r:.3f}, p = {p:.2e}")
    logger.info(f"Correlation (Std vs radius): r = {r_std:.3f}, p = {p_std:.2e}")


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
        mean_radii, cv, std_radii, sample_name, profiles_dict = load_axon_cv_data(f)
        all_data.append((mean_radii, cv, std_radii, sample_name, profiles_dict))

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

    all_cv = np.concatenate([cv for _, cv, _, _, _ in all_data])
    all_radii = np.concatenate([r for r, _, _, _, _ in all_data])
    all_std = np.concatenate([s for _, _, s, _, _ in all_data])

    logger.info(f"Total axons: {len(all_cv)}")
    logger.info(f"Mean radius: {np.mean(all_radii):.3f} +/- {np.std(all_radii):.3f} um")
    logger.info(f"Mean CV: {np.mean(all_cv):.3f} +/- {np.std(all_cv):.3f}")
    logger.info(f"Median CV: {np.median(all_cv):.3f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
