#!/usr/bin/env python3
"""
Plot histogram of per-axon effective radii from radius profile data.

For each axon, computes effective radius from its radius profile along the skeleton.
Shows histogram of per-axon effective radii and the joint effective radius
computed over all radius samples from all axons pooled together.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_effective_radius(radii: np.ndarray) -> float:
    """
    Compute MRI-visible effective radius: r_eff = (⟨r⁶⟩/⟨r²⟩)^(1/4)

    Args:
        radii: Array of radii values

    Returns:
        Effective radius
    """
    if len(radii) == 0:
        return 0.0

    r2 = np.mean(radii ** 2)
    r6 = np.mean(radii ** 6)

    if r2 == 0:
        return 0.0

    return (r6 / r2) ** 0.25


def main():
    parser = argparse.ArgumentParser(
        description='Plot histogram of per-axon effective radii from radius profiles'
    )
    parser.add_argument('npz_file', type=Path,
                        help='Path to .npz file from compute_axon_radius_profiles.py')
    parser.add_argument('output_file', type=Path,
                        help='Output PNG file for the plot')
    parser.add_argument('--bins', type=int, default=50,
                        help='Number of histogram bins (default: 50)')
    parser.add_argument('--max-radius', type=float, default=3.0,
                        help='Maximum radius for x-axis in μm (default: 3.0)')

    args = parser.parse_args()

    # Load data
    data = np.load(args.npz_file, allow_pickle=True)

    # Get radius profiles (per-axon arrays of radii along skeleton)
    radii_profiles = data['radii_profiles_um']  # Object array, one array per axon
    all_radii = data['all_radii_um']  # Flattened array of all radii
    labels = data['labels']

    # Compute per-axon effective radius from radius profile
    per_axon_r_eff = []

    for i in range(len(radii_profiles)):
        profile = radii_profiles[i]
        if len(profile) == 0:
            continue

        # Compute effective radius for this axon from its profile
        r_eff_axon = compute_effective_radius(profile)
        per_axon_r_eff.append(r_eff_axon)

    per_axon_r_eff = np.array(per_axon_r_eff)

    print(f"Loaded {len(labels)} axons, {len(per_axon_r_eff)} with valid radii")
    print(f"Per-axon r_eff range: {per_axon_r_eff.min():.3f} - {per_axon_r_eff.max():.3f} μm")
    print(f"Mean per-axon r_eff: {np.mean(per_axon_r_eff):.3f} ± {np.std(per_axon_r_eff):.3f} μm")
    print(f"Median per-axon r_eff: {np.median(per_axon_r_eff):.3f} μm")

    # Compute effective radius over joint distribution (all radii pooled)
    r_eff_joint = compute_effective_radius(all_radii)
    print(f"Joint distribution r_eff: {r_eff_joint:.3f} μm")
    print(f"Total radius samples: {len(all_radii)}")

    # Create histogram
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot histogram of per-axon effective radii
    counts, bin_edges, patches = ax.hist(
        per_axon_r_eff,
        bins=args.bins,
        range=(0, args.max_radius),
        edgecolor='black',
        alpha=0.7,
        label=f'Per-axon $r_{{eff}}$ (n={len(per_axon_r_eff)})'
    )

    # Add vertical line for joint distribution effective radius
    ax.axvline(
        r_eff_joint,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Joint $r_{{eff}}$ = {r_eff_joint:.3f} μm'
    )

    # Add vertical line for mean per-axon r_eff
    mean_r_eff = np.mean(per_axon_r_eff)
    ax.axvline(
        mean_r_eff,
        color='blue',
        linestyle=':',
        linewidth=2,
        label=f'Mean = {mean_r_eff:.3f} μm'
    )

    # Labels and formatting
    ax.set_xlabel('Effective radius (μm)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of per-axon effective radii', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, args.max_radius)

    # Add text with statistics
    stats_text = (
        f"n = {len(per_axon_r_eff)}\n"
        f"Mean = {mean_r_eff:.3f} μm\n"
        f"Median = {np.median(per_axon_r_eff):.3f} μm\n"
        f"Joint $r_{{eff}}$ = {r_eff_joint:.3f} μm"
    )
    ax.text(
        0.95, 0.65, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()

    # Save figure
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_file, dpi=200, bbox_inches='tight')
    print(f"Saved plot to {args.output_file}")

    plt.close()


if __name__ == '__main__':
    main()
