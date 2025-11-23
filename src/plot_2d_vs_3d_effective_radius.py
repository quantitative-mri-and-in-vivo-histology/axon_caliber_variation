#!/usr/bin/env python3
"""
Plot skeleton/tubular r_eff vs orthogonal slice r_eff.

X-axis: Orthogonal slice 3D pooled r_eff (r_eff_global from our analysis)
Y-axis: Skeleton-based r_eff (from tubular cc_morph/cg_morph for LM,
        or from radius_profiles for HM)

HM data only has skeleton r_eff, so plotted on identity line.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_tubular_data(cc_file: Path, cg_file: Path) -> dict:
    """
    Load tubular morphometry data and compute r_eff for each ROI.

    Args:
        cc_file: Path to cc_morph.mat
        cg_file: Path to cg_morph.mat

    Returns:
        Dict mapping (sample_name, population) -> r_eff
    """
    results = {}

    # Mapping from tubular names to our LM names
    name_map = {
        'h_25': 'LM_25',
        'h_49': 'LM_49',
        't_2': 'LM_2',
        't_24': 'LM_24',
        't_28': 'LM_28',
    }

    for mat_file, pop_name in [(cc_file, 'CC'), (cg_file, 'CG')]:
        if not mat_file.exists():
            logger.warning(f"File not found: {mat_file}")
            continue

        mat = sio.loadmat(str(mat_file))
        morph = mat['morph'][0, 0]

        for roi_name in morph.dtype.names:
            parts = roi_name.split('_')
            prefix = f"{parts[0]}_{parts[1]}"
            side = parts[2]

            if prefix not in name_map:
                continue

            sample_name = f"{name_map[prefix]}_{side}"

            roi_data = morph[roi_name]
            all_mean_radii = []

            for i in range(roi_data.shape[1]):
                axon = roi_data[0, i]
                areas = axon['area'].flatten()
                areas = areas[areas > 0]

                if len(areas) > 0:
                    radii = np.sqrt(areas / np.pi)
                    mean_r = np.mean(radii)
                    all_mean_radii.append(mean_r)

            if len(all_mean_radii) == 0:
                continue

            all_mean_radii = np.array(all_mean_radii)
            r2_mean = np.mean(all_mean_radii ** 2)
            r6_mean = np.mean(all_mean_radii ** 6)
            r_eff = (r6_mean / r2_mean) ** 0.25

            results[(sample_name, pop_name)] = r_eff

    return results


def plot_2d_vs_3d(summary_file: Path, output_file: Path,
                  cc_tubular_file: Path = None, cg_tubular_file: Path = None,
                  hm_summary_file: Path = None):
    """
    Plot skeleton/tubular r_eff vs orthogonal slice r_eff.

    Args:
        summary_file: Path to effective_radius_summary.json (LM orthogonal slice data)
        output_file: Path to save output plot
        cc_tubular_file: Path to cc_morph.mat (LM tubular data)
        cg_tubular_file: Path to cg_morph.mat (LM tubular data)
        hm_summary_file: Path to HM radius_profiles_summary.json (skeleton data)
    """
    # Load LM orthogonal slice results
    with open(summary_file, 'r') as f:
        summary = json.load(f)

    logger.info(f"Loaded LM orthogonal slice results for {len(summary)} samples")

    # Load LM tubular results
    tubular_data = {}
    if cc_tubular_file and cg_tubular_file:
        tubular_data = load_tubular_data(cc_tubular_file, cg_tubular_file)
        logger.info(f"Loaded LM tubular results for {len(tubular_data)} ROIs")

    # Load HM skeleton results
    hm_summary = {}
    if hm_summary_file and hm_summary_file.exists():
        with open(hm_summary_file, 'r') as f:
            hm_summary = json.load(f)
        logger.info(f"Loaded HM skeleton results for {len(hm_summary)} samples")

    # Collect LM data points (x = tubular, y = 2D and 3D approaches)
    # Each point: (sample, population, r_tubular, r_2d_mean, r_2d_std, r_3d_global)
    data_points = []

    for sample_name, populations in summary.items():
        for pop_name, pop_data in populations.items():
            # Get tubular r_eff for x-axis
            key = (sample_name, pop_name)
            if key not in tubular_data:
                logger.warning(f"No tubular data for {sample_name} {pop_name}, skipping")
                continue

            r_eff_tubular = tubular_data[key]
            r_eff_2d_mean = pop_data['r_eff_mean']
            r_eff_2d_std = pop_data['r_eff_std']
            r_eff_3d = pop_data['r_eff_global']

            data_points.append((sample_name, pop_name, r_eff_tubular, r_eff_2d_mean, r_eff_2d_std, r_eff_3d))

    # Match HM data to LM tubular x-positions
    # HM_XX_side -> use LM_XX_side CC tubular value as x (HM is all CC)
    hm_data_points = []  # List of (sample, r_eff_hm, r_tubular_ref)
    for hm_sample, sample_data in hm_summary.items():
        # Convert HM_XX_side to LM_XX_side
        lm_sample = hm_sample.replace('HM_', 'LM_')

        # Get CC tubular value for this sample (HM data is CC only)
        cc_key = (lm_sample, 'CC')

        if cc_key in tubular_data:
            r_tubular_ref = tubular_data[cc_key]
            r_eff_hm = sample_data['r_eff']
            hm_data_points.append((hm_sample, r_eff_hm, r_tubular_ref))
        else:
            logger.warning(f"No CC tubular reference for {hm_sample}, skipping")

    if len(data_points) == 0 and len(hm_data_points) == 0:
        logger.error("No data points found")
        return

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # TBI vs Sham rats
    tbi_rats = {'LM_25', 'LM_49', 'HM_25', 'HM_49'}

    # Plot LM data points: 2D (red) and 3D (blue) approaches
    for sample, pop, r_tubular, r_2d_mean, r_2d_std, r_3d in data_points:
        rat = '_'.join(sample.split('_')[:2])
        is_tbi = rat in tbi_rats
        is_cc = pop == 'CC'

        # Marker by population
        marker = 'o' if is_cc else 's'

        # Plot 3D pooled (blue)
        ax.plot(r_tubular, r_3d, marker=marker, color='blue', markersize=8, alpha=0.7,
                markeredgecolor='black', markeredgewidth=0.5)

        # Plot 2D per-slice mean with error bar (red)
        ax.errorbar(r_tubular, r_2d_mean, yerr=r_2d_std,
                    fmt=marker, color='red', markersize=8, alpha=0.7,
                    capsize=3, capthick=1, elinewidth=1,
                    markeredgecolor='black', markeredgewidth=0.5)

    # Plot HM data points (green) at corresponding tubular x-position
    for sample, r_eff_hm, r_tubular_ref in hm_data_points:
        rat = '_'.join(sample.split('_')[:2])
        is_tbi = rat in tbi_rats

        # Diamond marker for HM
        marker = 'D'

        # Plot HM skeleton r_eff (green)
        ax.plot(r_tubular_ref, r_eff_hm, marker=marker, color='green', markersize=8, alpha=0.7,
                markeredgecolor='black', markeredgewidth=0.5)

        # Add label
        ax.annotate(sample.replace('HM_', 'H'), (r_tubular_ref, r_eff_hm), fontsize=7, alpha=0.7,
                    xytext=(5, 5), textcoords='offset points')

    # Calculate axis limits
    all_vals = ([r_tub for _, _, r_tub, _, _, _ in data_points] +
                [r_2d for _, _, _, r_2d, _, _ in data_points] +
                [r_3d for _, _, _, _, _, r_3d in data_points] +
                [r_hm for _, r_hm, _ in hm_data_points] +
                [r_tub for _, _, r_tub in hm_data_points])
    min_val = min(all_vals) * 0.9
    max_val = max(all_vals) * 1.1

    # Add identity line
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)

    # Labels
    ax.set_xlabel('Tubular r_eff (μm)', fontsize=12)
    ax.set_ylabel('Evaluated r_eff (μm)', fontsize=12)
    ax.set_title('Effective Radius: Different Approaches vs Tubular Reference\n' +
                 r'$r_{\rm eff} = (\langle r^6 \rangle / \langle r^2 \rangle)^{1/4}$',
                 fontsize=14)

    # Custom legend - by approach and population
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               markersize=10, label='3D pooled - CC', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',
               markersize=10, label='3D pooled - CG', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=10, label='2D per-slice - CC', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
               markersize=10, label='2D per-slice - CG', markeredgecolor='black'),
    ]
    # Add HM legend entry if HM data exists
    if hm_data_points:
        legend_elements.append(
            Line2D([0], [0], marker='D', color='w', markerfacecolor='green',
                   markersize=10, label='HM skeleton', markeredgecolor='black')
        )
    legend_elements.append(
        Line2D([0], [0], linestyle='--', color='black', alpha=0.5, label='Identity')
    )
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plot to {output_file}")

    # Print summary statistics
    logger.info("\nLM Summary:")
    logger.info(f"{'Sample':<20} {'Pop':<5} {'Tubular':<10} {'2D':<10} {'3D':<10} {'3D/Tub %':<10}")
    logger.info(f"{'-'*65}")
    for sample, pop, r_tub, r_2d, r_2d_std, r_3d in sorted(data_points):
        diff_pct = (r_3d - r_tub) / r_tub * 100 if r_tub > 0 else 0
        logger.info(f"{sample:<20} {pop:<5} {r_tub:<10.4f} {r_2d:<10.4f} {r_3d:<10.4f} {diff_pct:>+8.1f}%")

    # Print HM summary
    if hm_data_points:
        logger.info("\nHM Summary:")
        logger.info(f"{'Sample':<20} {'HM r_eff':<10} {'Tub ref':<10} {'HM/Tub %':<10}")
        logger.info(f"{'-'*50}")
        for sample, r_hm, r_tub_ref in sorted(hm_data_points):
            diff_pct = (r_hm - r_tub_ref) / r_tub_ref * 100 if r_tub_ref > 0 else 0
            logger.info(f"{sample:<20} {r_hm:<10.4f} {r_tub_ref:<10.4f} {diff_pct:>+8.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot orthogonal slice r_eff vs skeleton/tubular r_eff'
    )
    parser.add_argument('summary_file', type=Path,
                        help='Path to effective_radius_summary.json (LM orthogonal slice data)')
    parser.add_argument('output_file', type=Path,
                        help='Output plot file path')
    parser.add_argument('--cc-tubular', type=Path, default=None,
                        help='Path to cc_morph.mat (LM tubular data)')
    parser.add_argument('--cg-tubular', type=Path, default=None,
                        help='Path to cg_morph.mat (LM tubular data)')
    parser.add_argument('--hm-summary', type=Path, default=None,
                        help='Path to HM radius_profiles_summary.json')

    args = parser.parse_args()

    plot_2d_vs_3d(args.summary_file, args.output_file,
                  args.cc_tubular, args.cg_tubular, args.hm_summary)
