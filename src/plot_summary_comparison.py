#!/usr/bin/env python3
"""
Plot effective radius comparison between two summary files.

Takes two summary JSON files (from batch processing) and plots
r_eff values from one against the other with identity line.

Useful for comparing:
- Different processing methods (e.g., orthogonal slices vs Kimimaro skeleton)
- Different datasets (e.g., LM vs HM)
- Different processing parameters
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_summary_data(summary_file: Path) -> dict:
    """Load summary JSON file and extract r_eff values with errors.

    Handles various summary formats:
    - {sample: {population: {r_eff_global: val, r_eff_std: err}}} - spatial/bundle format
    - {sample: {r_eff: val, std_radius: err}} - simple format (no population)

    Returns:
        Dict mapping sample_name -> {population_name or 'all' -> (r_eff, error)}
    """
    with open(summary_file, 'r') as f:
        summary = json.load(f)

    results = {}
    for sample_name, sample_data in summary.items():
        if not isinstance(sample_data, dict):
            continue

        results[sample_name] = {}

        # Check if this sample has r_eff directly (simple format)
        if 'r_eff' in sample_data and isinstance(sample_data['r_eff'], (int, float)):
            # Simple format: single r_eff value per sample
            r_eff = sample_data['r_eff']
            # Try to get error from std_radius or std_radii
            error = sample_data.get('std_radius', sample_data.get('std_radii', 0))
            results[sample_name]['all'] = (r_eff, error)
        elif 'r_eff_global' in sample_data and isinstance(sample_data['r_eff_global'], (int, float)):
            # Also check for r_eff_global directly
            r_eff = sample_data['r_eff_global']
            error = sample_data.get('r_eff_std', 0)
            results[sample_name]['all'] = (r_eff, error)
        else:
            # Nested format: iterate over populations
            for key, value in sample_data.items():
                if isinstance(value, dict):
                    # Check for r_eff_global or r_eff
                    if 'r_eff_global' in value:
                        r_eff = value['r_eff_global']
                        error = value.get('r_eff_std', 0)
                        results[sample_name][key] = (r_eff, error)
                    elif 'r_eff' in value and isinstance(value['r_eff'], (int, float)):
                        r_eff = value['r_eff']
                        error = value.get('std_radius', value.get('std_radii', 0))
                        results[sample_name][key] = (r_eff, error)

    return results


def plot_comparison(summary_file1: Path, summary_file2: Path, output_file: Path,
                    label1: str = None, label2: str = None):
    """Plot r_eff comparison between two summary files.

    Args:
        summary_file1: Path to first summary JSON (x-axis)
        summary_file2: Path to second summary JSON (y-axis)
        output_file: Path to save output plot
        label1: Label for x-axis (defaults to filename)
        label2: Label for y-axis (defaults to filename)
    """
    # Load data
    data1 = load_summary_data(summary_file1)
    data2 = load_summary_data(summary_file2)

    logger.info(f"Loaded {len(data1)} samples from {summary_file1.name}")
    logger.info(f"Loaded {len(data2)} samples from {summary_file2.name}")

    # Set default labels
    if label1 is None:
        label1 = summary_file1.stem
    if label2 is None:
        label2 = summary_file2.stem

    # Match samples and populations
    data_points = []  # (sample, population, r_eff1, r_eff2)

    for sample_name in sorted(set(data1.keys()) & set(data2.keys())):
        pop_keys1 = set(data1[sample_name].keys())
        pop_keys2 = set(data2[sample_name].keys())

        # If one file uses 'all' (simple format), match with any population in the other
        if 'all' in pop_keys1 and pop_keys2:
            # file1 is simple, file2 has populations
            for pop_name in sorted(pop_keys2):
                r_eff1, err1 = data1[sample_name]['all']
                r_eff2, err2 = data2[sample_name][pop_name]
                data_points.append((sample_name, pop_name, r_eff1, err1, r_eff2, err2))
        elif 'all' in pop_keys2 and pop_keys1:
            # file2 is simple, file1 has populations
            for pop_name in sorted(pop_keys1):
                r_eff1, err1 = data1[sample_name][pop_name]
                r_eff2, err2 = data2[sample_name]['all']
                data_points.append((sample_name, pop_name, r_eff1, err1, r_eff2, err2))
        else:
            # Both have populations, match by name
            matching_pops = pop_keys1 & pop_keys2
            if not matching_pops:
                logger.warning(f"No matching populations for {sample_name}")
                continue

            for pop_name in sorted(matching_pops):
                r_eff1, err1 = data1[sample_name][pop_name]
                r_eff2, err2 = data2[sample_name][pop_name]
                data_points.append((sample_name, pop_name, r_eff1, err1, r_eff2, err2))

    if len(data_points) == 0:
        logger.error("No matching samples/populations found")
        return

    logger.info(f"Matched {len(data_points)} data points")

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Extract values and errors
    r_eff1_vals = [r1 for _, _, r1, _, _, _ in data_points]
    err1_vals = [e1 for _, _, _, e1, _, _ in data_points]
    r_eff2_vals = [r2 for _, _, _, _, r2, _ in data_points]
    err2_vals = [e2 for _, _, _, _, _, e2 in data_points]

    # Calculate statistics
    r1_mean = np.mean(r_eff1_vals)
    r2_mean = np.mean(r_eff2_vals)
    correlation = np.corrcoef(r_eff1_vals, r_eff2_vals)[0, 1]

    # Plot data points with error bars on y-axis (file1 is y-axis)
    ax.errorbar(r_eff2_vals, r_eff1_vals, yerr=err1_vals, fmt='o',
                markersize=8, alpha=0.6, capsize=5, capthick=1, elinewidth=1,
                ecolor='black', markeredgecolor='black', markeredgewidth=1, color='C0')

    # Plot identity line
    min_val = min(min(r_eff1_vals), min(r_eff2_vals)) * 0.9
    max_val = max(max(r_eff1_vals), max(r_eff2_vals)) * 1.1
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='Identity')

    # Labels and title (swap x and y labels)
    ax.set_xlabel(f'{label2} r_eff (μm)', fontsize=12)
    ax.set_ylabel(f'{label1} r_eff (μm)', fontsize=12)
    ax.set_title(f'Effective Radius Comparison\n' +
                 f'Correlation: {correlation:.3f}',
                 fontsize=14)

    # Set equal axis limits
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(loc='upper left', fontsize=11)

    plt.tight_layout()

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plot to {output_file}")

    # Print summary (swapped x and y)
    logger.info("\nComparison Summary:")
    logger.info(f"{'Sample':<20} {'Population':<10} {label2:<16} {label1:<16} {'Diff (%)':>10}")
    logger.info(f"{'-'*72}")

    for sample, pop, r1, e1, r2, e2 in sorted(data_points):
        diff_pct = (r1 - r2) / r2 * 100 if r2 > 0 else 0
        logger.info(f"{sample:<20} {pop:<10} {r2:<8.4f}±{e2:<6.4f} {r1:<8.4f}±{e1:<6.4f} {diff_pct:>+9.1f}%")

    # Statistics
    logger.info(f"\n{'='*64}")
    logger.info(f"Mean {label2}: {r2_mean:.4f} ± {np.std(r_eff2_vals):.4f} μm")
    logger.info(f"Mean {label1}: {r1_mean:.4f} ± {np.std(r_eff1_vals):.4f} μm")
    logger.info(f"Correlation: {correlation:.4f}")
    logger.info(f"N points: {len(data_points)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot effective radius comparison between two summary files'
    )
    parser.add_argument('summary_file1', type=Path,
                        help='Path to first summary JSON file (x-axis)')
    parser.add_argument('summary_file2', type=Path,
                        help='Path to second summary JSON file (y-axis)')
    parser.add_argument('output_file', type=Path,
                        help='Output plot file path')
    parser.add_argument('--label1', type=str, default=None,
                        help='Label for x-axis (defaults to filename)')
    parser.add_argument('--label2', type=str, default=None,
                        help='Label for y-axis (defaults to filename)')

    args = parser.parse_args()

    plot_comparison(args.summary_file1, args.summary_file2, args.output_file,
                    args.label1, args.label2)
