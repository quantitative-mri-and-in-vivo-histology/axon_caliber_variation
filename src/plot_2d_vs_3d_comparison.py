#!/usr/bin/env python3
"""
Compare 2D slice-based effective radius against 3D skeleton-based measurements.

2D data source: fig/effective_radius_LM_spatial/effective_radius_summary.json
  - Has per-sample, per-population (CC/CG) r_eff_global and r_eff_std

3D data source: data/processed/LM/axon_radius_profiles/*.npz
  - Per-axon radius profiles from skeleton tracing
  - CC/CG assignment from data/processed/LM/cc_cg_bundles/*.zarr segment_ids
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import zarr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_r_eff(radii: np.ndarray) -> float:
    """Compute effective MRI-visible radius: r_eff = (<r^6>/<r^2>)^(1/4)."""
    if len(radii) == 0:
        return np.nan
    r2 = radii ** 2
    r6 = radii ** 6
    return (np.mean(r6) / np.mean(r2)) ** 0.25


def load_2d_summary(summary_file: Path) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Load 2D effective radius summary.

    Returns:
        Dict mapping sample_name -> {population -> (r_eff, std)}
    """
    with open(summary_file, 'r') as f:
        summary = json.load(f)

    results = {}
    for sample_name, sample_data in summary.items():
        if not isinstance(sample_data, dict):
            continue

        results[sample_name] = {}
        for pop_name, pop_data in sample_data.items():
            if isinstance(pop_data, dict) and 'r_eff_mean' in pop_data:
                r_eff = pop_data['r_eff_mean']
                std = pop_data.get('r_eff_std', 0)
                results[sample_name][pop_name] = (r_eff, std)

    return results


def load_3d_data(npz_dir: Path, zarr_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load 3D radius profiles and compute CC/CG specific r_eff.

    Args:
        npz_dir: Directory containing *_radius_profiles_accel.npz files
        zarr_dir: Directory containing *_bundles.zarr files with CC/CG segment_ids

    Returns:
        Dict mapping sample_name -> {population -> r_eff}
    """
    results = {}

    # Find all NPZ files
    npz_files = sorted(npz_dir.glob('*_radius_profiles_accel.npz'))
    logger.info(f"Found {len(npz_files)} NPZ files in {npz_dir}")

    for npz_file in npz_files:
        # Extract sample name: LM_25_ipsi_radius_profiles_accel.npz -> LM_25_ipsi
        sample_name = npz_file.stem.replace('_radius_profiles_accel', '')

        # Find corresponding zarr file (use *_spatial.zarr to match 2D analysis)
        zarr_file = zarr_dir / f'{sample_name}_spatial.zarr'
        if not zarr_file.exists():
            logger.warning(f"No zarr file found for {sample_name}, skipping")
            continue

        # Load NPZ data
        npz_data = np.load(npz_file)
        labels = npz_data['labels']
        # Use mean_radii_um (per-axon mean radius) for computing r_eff
        radii_per_axon = npz_data['mean_radii_um']

        # Create label -> index mapping
        label_to_idx = {int(lbl): i for i, lbl in enumerate(labels)}

        # Load CC/CG segment IDs from zarr
        results[sample_name] = {}

        for pop_name in ['CC', 'CG']:
            pop_key = pop_name.lower()
            segment_ids_path = zarr_file / pop_key / 'labels' / 'segment_ids'

            if not segment_ids_path.exists():
                logger.debug(f"No {pop_name} segment_ids for {sample_name}")
                continue

            # Load segment IDs
            segment_ids = zarr.open_array(str(segment_ids_path), mode='r')[:]

            # Filter radii for this population
            pop_indices = [label_to_idx[int(sid)] for sid in segment_ids if int(sid) in label_to_idx]

            if len(pop_indices) == 0:
                logger.warning(f"No matching labels for {sample_name}/{pop_name}")
                continue

            pop_radii = radii_per_axon[pop_indices]
            r_eff = compute_r_eff(pop_radii)
            results[sample_name][pop_name] = r_eff

            logger.debug(f"{sample_name}/{pop_name}: {len(pop_indices)} axons, r_eff={r_eff:.4f}")

    return results


def plot_comparison(data_2d: Dict, data_3d: Dict, output_file: Path,
                    label_2d: str = '2D Slice-based',
                    label_3d: str = '3D Skeleton-based'):
    """Plot 2D vs 3D effective radius comparison.

    Args:
        data_2d: Dict mapping sample -> {pop -> (r_eff, std)}
        data_3d: Dict mapping sample -> {pop -> r_eff}
        output_file: Output plot path
        label_2d: Label for 2D data (y-axis)
        label_3d: Label for 3D data (x-axis)
    """
    # Collect matched data points
    data_points = []  # (sample, pop, r_eff_2d, std_2d, r_eff_3d)
    unmatched_2d = []
    unmatched_3d = []

    # Find all samples and populations
    all_samples = set(data_2d.keys()) | set(data_3d.keys())

    for sample in sorted(all_samples):
        pops_2d = set(data_2d.get(sample, {}).keys())
        pops_3d = set(data_3d.get(sample, {}).keys())

        # Track unmatched
        for pop in pops_2d - pops_3d:
            unmatched_2d.append((sample, pop))
        for pop in pops_3d - pops_2d:
            unmatched_3d.append((sample, pop))

        # Matched populations
        for pop in sorted(pops_2d & pops_3d):
            r_eff_2d, std_2d = data_2d[sample][pop]
            r_eff_3d = data_3d[sample][pop]
            data_points.append((sample, pop, r_eff_2d, std_2d, r_eff_3d))

    # Report unmatched
    if unmatched_2d:
        logger.info(f"Unmatched in 2D only ({len(unmatched_2d)}): {unmatched_2d}")
    if unmatched_3d:
        logger.info(f"Unmatched in 3D only ({len(unmatched_3d)}): {unmatched_3d}")

    if len(data_points) == 0:
        logger.error("No matched data points found")
        return

    logger.info(f"Matched {len(data_points)} data points")

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Extract values
    r_eff_2d_vals = np.array([dp[2] for dp in data_points])
    std_2d_vals = np.array([dp[3] for dp in data_points])
    r_eff_3d_vals = np.array([dp[4] for dp in data_points])
    populations = [dp[1] for dp in data_points]

    # Color by population
    colors = {'CC': 'C0', 'CG': 'C1'}
    markers = {'CC': 'o', 'CG': 's'}

    for pop in ['CC', 'CG']:
        mask = np.array([p == pop for p in populations])
        if not np.any(mask):
            continue

        ax.errorbar(
            r_eff_3d_vals[mask], r_eff_2d_vals[mask],
            yerr=std_2d_vals[mask],
            fmt=markers[pop], color=colors[pop],
            markersize=8, alpha=0.7, capsize=4, capthick=1, elinewidth=1,
            markeredgecolor='black', markeredgewidth=0.5,
            label=pop
        )

    # Identity line
    all_vals = np.concatenate([r_eff_2d_vals, r_eff_3d_vals])
    min_val = min(all_vals) * 0.9
    max_val = max(all_vals) * 1.1
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='Identity')

    # Statistics
    correlation = np.corrcoef(r_eff_3d_vals, r_eff_2d_vals)[0, 1]
    mean_diff = np.mean(r_eff_2d_vals - r_eff_3d_vals)
    mean_diff_pct = np.mean((r_eff_2d_vals - r_eff_3d_vals) / r_eff_3d_vals) * 100

    ax.set_xlabel(f'{label_3d} $r_{{eff}}$ (μm)', fontsize=12)
    ax.set_ylabel(f'{label_2d} $r_{{eff}}$ (μm)', fontsize=12)
    ax.set_title(f'Effective Radius: 2D Slice-based vs 3D Skeleton-based\n'
                 f'Correlation: {correlation:.3f}, Mean diff: {mean_diff_pct:+.1f}%',
                 fontsize=12)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plot to {output_file}")

    # Print detailed comparison table
    logger.info(f"\n{'Sample':<20} {'Pop':<5} {label_3d:<12} {label_2d:<18} {'Diff (%)':>10}")
    logger.info(f"{'-'*70}")

    for sample, pop, r_2d, std_2d, r_3d in sorted(data_points):
        diff_pct = (r_2d - r_3d) / r_3d * 100 if r_3d > 0 else 0
        logger.info(f"{sample:<20} {pop:<5} {r_3d:<12.4f} {r_2d:<8.4f}±{std_2d:<7.4f} {diff_pct:>+9.1f}%")

    logger.info(f"\n{'='*70}")
    logger.info(f"Mean {label_3d}: {np.mean(r_eff_3d_vals):.4f} μm")
    logger.info(f"Mean {label_2d}: {np.mean(r_eff_2d_vals):.4f} μm")
    logger.info(f"Correlation: {correlation:.4f}")
    logger.info(f"Mean difference: {mean_diff:.4f} μm ({mean_diff_pct:+.1f}%)")
    logger.info(f"N points: {len(data_points)}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare 2D slice-based vs 3D skeleton-based effective radius'
    )
    parser.add_argument('--summary-2d', type=Path,
                        default=Path('fig/effective_radius_LM_spatial/effective_radius_summary.json'),
                        help='Path to 2D effective radius summary JSON')
    parser.add_argument('--npz-dir', type=Path,
                        default=Path('data/processed/LM/axon_radius_profiles'),
                        help='Directory containing 3D radius profile NPZ files')
    parser.add_argument('--zarr-dir', type=Path,
                        default=Path('data/processed/LM/cc_cg_spatial'),
                        help='Directory containing CC/CG spatial zarr files')
    parser.add_argument('--output', type=Path,
                        default=Path('fig/r_eff_2d_vs_3d_LM.png'),
                        help='Output plot path')

    args = parser.parse_args()

    # Load data
    logger.info("Loading 2D summary data...")
    data_2d = load_2d_summary(args.summary_2d)
    logger.info(f"Loaded 2D data for {len(data_2d)} samples")

    logger.info("Loading 3D profile data...")
    data_3d = load_3d_data(args.npz_dir, args.zarr_dir)
    logger.info(f"Loaded 3D data for {len(data_3d)} samples")

    # Plot comparison
    plot_comparison(data_2d, data_3d, args.output)


if __name__ == '__main__':
    main()
