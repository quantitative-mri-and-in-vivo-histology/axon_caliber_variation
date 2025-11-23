#!/usr/bin/env python3
"""
Compare effective radii from different approaches against 3D morph reference.

Creates a single plot with 3D morph files (tubular processing) as reference,
and other approaches (our 2D per-slice, our 3D global) evaluated against that.

The tubular data is in data/processed_tubular/{cc,cg}_morph.mat with naming:
- h_XX_ipsi/contra = healthy (corresponds to LM_XX which are TBI: 25, 49)
- t_XX_ipsi/contra = treatment/sham (corresponds to LM_XX which are Sham: 2, 24, 28)

Note: The h/t naming appears inverted relative to TBI/Sham designation.
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
    # h = healthy -> LM_25, LM_49 (TBI)
    # t = treatment -> LM_2, LM_24, LM_28 (Sham)
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
            # Parse roi name (e.g., 'h_25_contra')
            parts = roi_name.split('_')
            prefix = f"{parts[0]}_{parts[1]}"  # 'h_25'
            side = parts[2]  # 'contra' or 'ipsi'

            if prefix not in name_map:
                logger.warning(f"Unknown prefix: {prefix}")
                continue

            sample_name = f"{name_map[prefix]}_{side}"

            # Extract areas and compute r_eff
            roi_data = morph[roi_name]

            all_mean_radii = []
            for i in range(roi_data.shape[1]):
                axon = roi_data[0, i]
                areas = axon['area'].flatten()

                # Filter out zero/negative areas
                areas = areas[areas > 0]

                if len(areas) > 0:
                    radii = np.sqrt(areas / np.pi)
                    mean_r = np.mean(radii)
                    all_mean_radii.append(mean_r)

            if len(all_mean_radii) == 0:
                logger.warning(f"No valid axons in {roi_name}")
                continue

            all_mean_radii = np.array(all_mean_radii)

            # Compute r_eff
            r2_mean = np.mean(all_mean_radii ** 2)
            r6_mean = np.mean(all_mean_radii ** 6)
            r_eff = (r6_mean / r2_mean) ** 0.25

            results[(sample_name, pop_name)] = {
                'r_eff': r_eff,
                'n_axons': len(all_mean_radii),
                'mean_radius': np.mean(all_mean_radii)
            }

            logger.info(f"{sample_name} {pop_name}: r_eff={r_eff:.4f} um ({len(all_mean_radii)} axons)")

    return results


def plot_comparison(summary_file: Path,
                    cc_tubular_file: Path,
                    cg_tubular_file: Path,
                    output_file: Path):
    """
    Create single comparison plot with 3D morph as reference.

    Args:
        summary_file: Path to our effective_radius_summary.json
        cc_tubular_file: Path to cc_morph.mat
        cg_tubular_file: Path to cg_morph.mat
        output_file: Path to save output plot
    """
    # Load our results
    with open(summary_file, 'r') as f:
        our_summary = json.load(f)

    logger.info(f"Loaded our results for {len(our_summary)} samples")

    # Load tubular results
    tubular_data = load_tubular_data(cc_tubular_file, cg_tubular_file)
    logger.info(f"Loaded tubular results for {len(tubular_data)} ROIs")

    # Collect matched data points
    data_points = []  # (sample, pop, our_2d_mean, our_2d_std, our_3d, tubular_3d)

    for sample_name, populations in our_summary.items():
        for pop_name, pop_data in populations.items():
            key = (sample_name, pop_name)

            if key not in tubular_data:
                logger.warning(f"No tubular data for {sample_name} {pop_name}")
                continue

            data_points.append((
                sample_name,
                pop_name,
                pop_data['r_eff_mean'],
                pop_data['r_eff_std'],
                pop_data['r_eff_global'],
                tubular_data[key]['r_eff']
            ))

    if len(data_points) == 0:
        logger.error("No matched data points")
        return

    logger.info(f"Matched {len(data_points)} ROIs")

    # Create figure with single plot
    _, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Extract arrays
    our_2d_mean = np.array([d[2] for d in data_points])
    our_2d_std = np.array([d[3] for d in data_points])
    our_3d = np.array([d[4] for d in data_points])
    tubular_3d = np.array([d[5] for d in data_points])

    # Plot each data point
    for i, (sample, pop, _, _, _, _) in enumerate(data_points):
        is_cc = pop == 'CC'

        # Markers: circle for CC, square for CG
        marker = 'o' if is_cc else 's'

        # Plot Our 3D in blue
        ax.scatter(tubular_3d[i], our_3d[i], c='blue', marker=marker,
                   s=60, alpha=0.8, edgecolors='black', linewidth=0.5)

        # Plot Our 2D in red with error bars
        ax.errorbar(tubular_3d[i], our_2d_mean[i], yerr=our_2d_std[i],
                    fmt=marker, color='red', markersize=7, alpha=0.7,
                    capsize=2, capthick=1, elinewidth=1,
                    markeredgecolor='black', markeredgewidth=0.5)

    # Identity line
    all_vals = list(tubular_3d) + list(our_3d) + list(our_2d_mean)
    min_val = min(all_vals) * 0.9
    max_val = max(all_vals) * 1.1
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)

    ax.set_xlabel('3D Morph Reference r_eff (μm)', fontsize=12)
    ax.set_ylabel('Evaluated r_eff (μm)', fontsize=12)
    ax.set_title('Effective Radius Comparison', fontsize=14)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Create legend showing all combinations
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               markersize=8, label='Our 3D - CC', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',
               markersize=8, label='Our 3D - CG', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=8, label='Our 2D - CC', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
               markersize=8, label='Our 2D - CG', markeredgecolor='black'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plot to {output_file}")

    # Print comparison table
    logger.info("\nComparison Table:")
    logger.info(f"{'Sample':<15} {'Pop':<4} {'Our 2D':<8} {'Our 3D':<8} {'3D Morph':<8} {'3D/Ref %':<10}")
    logger.info(f"{'-'*55}")
    for sample, pop, r2_mean, _, r3_our, r3_tub in sorted(data_points):
        ratio = r3_our / r3_tub * 100 if r3_tub > 0 else 0
        logger.info(f"{sample:<15} {pop:<4} {r2_mean:<8.4f} {r3_our:<8.4f} {r3_tub:<8.4f} {ratio:>8.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare effective radii: our analysis vs 3D morph reference'
    )
    parser.add_argument('summary_file', type=Path,
                        help='Path to our effective_radius_summary.json')
    parser.add_argument('cc_tubular', type=Path,
                        help='Path to cc_morph.mat')
    parser.add_argument('cg_tubular', type=Path,
                        help='Path to cg_morph.mat')
    parser.add_argument('output_file', type=Path,
                        help='Output plot file path')

    args = parser.parse_args()

    plot_comparison(
        args.summary_file,
        args.cc_tubular,
        args.cg_tubular,
        args.output_file
    )
