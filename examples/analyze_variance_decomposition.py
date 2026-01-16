#!/usr/bin/env python3
"""
Analyze across-axon vs along-axon radius variance decomposition.

Decomposes total radius variance into:
- Between-axon (across): variance of per-axon mean radii
- Within-axon (along): mean of per-axon variances

This follows the standard variance decomposition:
    Total variance = Between-axon variance + Within-axon variance

Results are computed per-region (CC, CG) and pooled across all data.

Usage:
    python examples/analyze_variance_decomposition.py \
        --data-dir data/processed/LM \
        --output fig/variance_decomposition.svg
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from axonometry import get_plot_settings, style_axis, add_panel_labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings = get_plot_settings()

# Samples to exclude from analysis
EXCLUDED_SAMPLES = {'tbi_2_ipsi'}


def filter_radii_by_roi(radii_profiles: np.ndarray, skeleton_coords: np.ndarray,
                        lengths: np.ndarray, roi_min: np.ndarray, roi_max: np.ndarray,
                        min_length: float = 0.0) -> List[np.ndarray]:
    """
    Filter radius profiles to only include measurements within ROI bounds.

    Args:
        radii_profiles: Array of radius profile arrays per axon
        skeleton_coords: Array of skeleton coordinate arrays per axon (N, 3) in [Z, Y, X] μm
        lengths: Array of axon lengths in μm
        roi_min: Minimum ROI bounds [Z, Y, X] in μm
        roi_max: Maximum ROI bounds [Z, Y, X] in μm
        min_length: Minimum axon length in μm (default: 0.0, no filtering)

    Returns:
        List of filtered radius profiles (one per axon with measurements in ROI)
    """
    filtered_profiles = []

    for i in range(len(radii_profiles)):
        # Skip axons shorter than min_length
        if lengths[i] < min_length:
            continue

        coords = np.array(skeleton_coords[i])  # (N, 3) in [Z, Y, X]
        radii = np.array(radii_profiles[i])

        if len(coords) == 0 or len(radii) == 0:
            continue

        # Check which points are within ROI bounds
        within_roi = (
            (coords[:, 0] >= roi_min[0]) & (coords[:, 0] <= roi_max[0]) &  # Z
            (coords[:, 1] >= roi_min[1]) & (coords[:, 1] <= roi_max[1]) &  # Y
            (coords[:, 2] >= roi_min[2]) & (coords[:, 2] <= roi_max[2])    # X
        )

        radii_in_roi = radii[within_roi]
        if len(radii_in_roi) > 1:  # Need at least 2 points for variance
            filtered_profiles.append(radii_in_roi)

    return filtered_profiles


def compute_variance_decomposition(profiles: List[np.ndarray]) -> Optional[Dict]:
    """
    Compute variance decomposition for a list of radius profiles.

    Uses the law of total variance: Var(X) = E[Var(X|Y)] + Var(E[X|Y])
    where Y is the axon identity. Within-axon variance is weighted by
    sample count for correct decomposition.

    Args:
        profiles: List of 1D arrays, each containing radius measurements for one axon

    Returns:
        Dictionary with decomposition results, or None if insufficient data
    """
    if len(profiles) < 2:
        return None

    # Collect per-axon statistics
    axon_means = []
    axon_vars = []
    axon_counts = []
    all_radii = []

    for profile in profiles:
        if len(profile) > 1:
            axon_means.append(np.mean(profile))
            axon_vars.append(np.var(profile))
            axon_counts.append(len(profile))
            all_radii.extend(profile)

    if len(axon_means) < 2:
        return None

    axon_means = np.array(axon_means)
    axon_vars = np.array(axon_vars)
    axon_counts = np.array(axon_counts)
    all_radii = np.array(all_radii)

    # Total variance (of all measurements pooled)
    total_var = np.var(all_radii)

    # Between-axon variance (variance of per-axon means)
    between_var = np.var(axon_means)

    # Within-axon variance: E[Var(X|axon)] weighted by sample count
    within_var = np.average(axon_vars, weights=axon_counts)

    # Compute percentages (should sum to ~100% by law of total variance)
    decomposed_total = between_var + within_var
    between_pct = 100 * between_var / decomposed_total if decomposed_total > 0 else 0
    within_pct = 100 * within_var / decomposed_total if decomposed_total > 0 else 0

    return {
        'total_var': total_var,
        'between_var': between_var,
        'within_var': within_var,
        'between_pct': between_pct,
        'within_pct': within_pct,
        'n_axons': len(axon_means),
        'n_measurements': len(all_radii),
        'mean_radius': np.mean(all_radii),
        'axon_means': axon_means,
        'axon_vars': axon_vars,
    }


def load_and_analyze_data(data_dir: Path, min_length: float = 50.0) -> Dict[str, Dict]:
    """
    Load all NPZ files and compute variance decomposition per population.

    Args:
        data_dir: Directory containing *_axon_profiles.npz and *_population_rois.json files
        min_length: Minimum axon length in μm (default: 50.0)

    Returns:
        Dictionary with results per population and pooled
    """
    npz_files = sorted(data_dir.glob('*_axon_profiles.npz'))
    if not npz_files:
        raise ValueError(f"No *_axon_profiles.npz files found in {data_dir}")

    logger.info(f"Found {len(npz_files)} NPZ files")
    logger.info(f"Using min_length filter: {min_length} μm")

    # Collect profiles per population
    cc_profiles = []
    cg_profiles = []

    for npz_file in npz_files:
        volume_name = npz_file.stem.replace('_axon_profiles', '')

        # Skip excluded samples
        if volume_name in EXCLUDED_SAMPLES:
            logger.info(f"Skipping excluded sample: {volume_name}")
            continue

        # Look for population ROI file
        roi_file = npz_file.parent / f"{volume_name}_population_rois.json"
        if not roi_file.exists():
            logger.warning(f"No ROI file for {volume_name}, skipping")
            continue

        # Load data
        data = np.load(npz_file, allow_pickle=True)
        radii_profiles = data['radii_profiles_um']
        skeleton_coords = data['skeleton_coords_um']
        lengths = data['lengths_um']

        with open(roi_file) as f:
            roi_data = json.load(f)

        logger.info(f"Processing {volume_name}...")

        # Process each population
        for pop_name in ['cc', 'cg']:
            if pop_name not in roi_data['rois']:
                continue

            roi = roi_data['rois'][pop_name]
            roi_min = np.array(roi['min_um'])
            roi_max = np.array(roi['max_um'])

            # Filter profiles by ROI and min_length
            filtered = filter_radii_by_roi(radii_profiles, skeleton_coords, lengths,
                                           roi_min, roi_max, min_length)

            if pop_name == 'cc':
                cc_profiles.extend(filtered)
            else:
                cg_profiles.extend(filtered)

            logger.info(f"  {pop_name.upper()}: {len(filtered)} axons with ROI samples")

    # Compute decomposition per population
    results = {}

    if cc_profiles:
        results['CC'] = compute_variance_decomposition(cc_profiles)
        if results['CC']:
            logger.info(f"CC: between={results['CC']['between_pct']:.1f}%, "
                       f"within={results['CC']['within_pct']:.1f}%, "
                       f"n_axons={results['CC']['n_axons']}")

    if cg_profiles:
        results['CG'] = compute_variance_decomposition(cg_profiles)
        if results['CG']:
            logger.info(f"CG: between={results['CG']['between_pct']:.1f}%, "
                       f"within={results['CG']['within_pct']:.1f}%, "
                       f"n_axons={results['CG']['n_axons']}")

    # Pooled across all populations
    all_profiles = cc_profiles + cg_profiles
    if all_profiles:
        results['Pooled'] = compute_variance_decomposition(all_profiles)
        if results['Pooled']:
            logger.info(f"Pooled: between={results['Pooled']['between_pct']:.1f}%, "
                       f"within={results['Pooled']['within_pct']:.1f}%, "
                       f"n_axons={results['Pooled']['n_axons']}")

    return results


def create_figure(results: Dict[str, Dict], output_file: Path) -> None:
    """
    Create stacked bar chart showing variance decomposition.

    Args:
        results: Dictionary with decomposition results per population
        output_file: Path for output figure
    """
    # Filter out None results
    valid_results = {k: v for k, v in results.items() if v is not None}

    if not valid_results:
        logger.error("No valid results to plot")
        return

    # Order: CC, CG, Pooled
    order = ['CC', 'CG', 'Pooled']
    categories = [c for c in order if c in valid_results]

    between_pcts = [valid_results[c]['between_pct'] for c in categories]
    within_pcts = [valid_results[c]['within_pct'] for c in categories]
    n_axons = [valid_results[c]['n_axons'] for c in categories]

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Colors from settings
    between_color = settings.colors['binary_a']  # Sand/tan for between-axon
    within_color = settings.colors['binary_b']   # Dusty teal for within-axon

    y_pos = np.arange(len(categories))
    bar_height = 0.6

    # Create horizontal stacked bars
    bars_between = ax.barh(y_pos, between_pcts, bar_height,
                           label='Between-axon (across)', color=between_color)
    bars_within = ax.barh(y_pos, within_pcts, bar_height, left=between_pcts,
                          label='Within-axon (along)', color=within_color)

    # Add percentage labels on bars
    font_size = settings.fonts['tick_size']
    for i, (b_pct, w_pct) in enumerate(zip(between_pcts, within_pcts)):
        # Between label (left side)
        if b_pct > 15:
            ax.text(b_pct / 2, i, f'{b_pct:.1f}%',
                   ha='center', va='center', fontsize=font_size, fontweight='bold', color='white')
        # Within label (right side)
        if w_pct > 15:
            ax.text(b_pct + w_pct / 2, i, f'{w_pct:.1f}%',
                   ha='center', va='center', fontsize=font_size, fontweight='bold', color='white')

    # Add axon count annotations on right
    for i, n in enumerate(n_axons):
        ax.text(102, i, f'n = {n:,}', ha='left', va='center',
               fontsize=settings.fonts['legend_size'], color='gray')

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.set_xlabel('Percentage of total variance [%]', fontsize=settings.fonts['label_size'])
    ax.set_xlim(0, 100)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2,
              fontsize=settings.fonts['legend_size'], frameon=False)

    style_axis(ax)
    ax.set_box_aspect(0.5)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, right=0.82)

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"Saved figure to {output_file}")


def print_summary(results: Dict[str, Dict]) -> None:
    """Print summary statistics table."""
    print("\n" + "=" * 80)
    print("VARIANCE DECOMPOSITION SUMMARY")
    print("=" * 80)
    print(f"{'Category':<10} {'N Axons':>10} {'N Samples':>12} {'Between %':>12} {'Within %':>12} {'Total Var':>12}")
    print("-" * 80)

    for cat in ['CC', 'CG', 'Pooled']:
        if cat in results and results[cat] is not None:
            r = results[cat]
            print(f"{cat:<10} {r['n_axons']:>10,} {r['n_measurements']:>12,} "
                  f"{r['between_pct']:>11.1f}% {r['within_pct']:>11.1f}% "
                  f"{r['total_var']:>12.4f}")

    print("=" * 80)
    print("\nInterpretation:")
    print("  - Between-axon (across): Variation in mean radius across different axons")
    print("  - Within-axon (along): Variation in radius along individual axon length")
    print("  - Total variance = Between + Within (by variance decomposition)")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze across vs along axon radius variance decomposition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/analyze_variance_decomposition.py \\
      --data-dir data/processed/LM \\
      --output fig/variance_decomposition.svg
        """
    )

    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Directory containing *_axon_profiles.npz files')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output figure path (SVG or PNG)')
    parser.add_argument('--min-length', type=float, default=50.0,
                        help='Minimum axon length in μm (default: 50.0)')

    args = parser.parse_args()

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        return 1

    # Load and analyze data
    results = load_and_analyze_data(args.data_dir, min_length=args.min_length)

    if not results:
        logger.error("No results computed")
        return 1

    # Print summary
    print_summary(results)

    # Create figure
    create_figure(results, args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
