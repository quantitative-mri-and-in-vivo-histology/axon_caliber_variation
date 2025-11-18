#!/usr/bin/env python3
"""
Plot radius profiles of the largest axons in a bundle.

Identifies the N largest axons by their maximum radius along their profile,
then plots individual radius profiles for each.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_axon_radii_per_slice(volume: np.ndarray,
                                  voxel_size_um: float = 0.05) -> Dict[int, Dict[int, float]]:
    """
    Extract radius measurements for each axon in each slice.

    Args:
        volume: 3D labeled volume (z, y, x) where axons are aligned with z
        voxel_size_um: Voxel size in micrometers

    Returns:
        Dict mapping slice_idx -> dict mapping label -> radius (in μm)
    """
    logger.info(f"Extracting radii per axon per slice from volume shape {volume.shape}")

    n_slices = volume.shape[0]
    radii_data = {}

    for z in tqdm(range(n_slices), desc="Processing slices", unit="slice"):
        slice_2d = volume[z, :, :]

        # Use regionprops to efficiently extract areas of all regions
        regions = regionprops(slice_2d)

        if len(regions) == 0:
            radii_data[z] = {}
            continue

        # Compute circular-equivalent radius for each region (axon)
        slice_radii = {}
        for region in regions:
            label = region.label
            area_voxels = region.area
            # Convert to physical area and compute radius
            area_um2 = area_voxels * (voxel_size_um ** 2)
            radius_um = np.sqrt(area_um2 / np.pi)
            slice_radii[label] = radius_um

        radii_data[z] = slice_radii

    return radii_data


def compute_max_radius_per_axon(radii_data: Dict[int, Dict[int, float]]) -> Dict[int, float]:
    """
    Compute the maximum radius for each axon across all slices.

    Args:
        radii_data: Dict mapping slice_idx -> dict mapping label -> radius

    Returns:
        Dict mapping label -> max_radius
    """
    logger.info("Computing maximum radius per axon")

    max_radii = {}

    for z, slice_radii in radii_data.items():
        for label, radius in slice_radii.items():
            if label not in max_radii:
                max_radii[label] = radius
            else:
                max_radii[label] = max(max_radii[label], radius)

    logger.info(f"Found {len(max_radii)} unique axons")

    return max_radii


def get_largest_axons(max_radii: Dict[int, float], n_largest: int = 10) -> List[int]:
    """
    Get the labels of the N largest axons.

    Args:
        max_radii: Dict mapping label -> max_radius
        n_largest: Number of largest axons to return

    Returns:
        List of labels sorted by max radius (descending)
    """
    # Sort by max radius in descending order
    sorted_labels = sorted(max_radii.keys(), key=lambda label: max_radii[label], reverse=True)

    # Take the top N
    largest_labels = sorted_labels[:n_largest]

    logger.info(f"Top {n_largest} axons by max radius:")
    for i, label in enumerate(largest_labels, 1):
        logger.info(f"  {i}. Label {label}: max radius = {max_radii[label]:.3f} μm")

    return largest_labels


def extract_radius_profiles(radii_data: Dict[int, Dict[int, float]],
                           largest_labels: List[int]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract radius profiles for specified axons.

    Args:
        radii_data: Dict mapping slice_idx -> dict mapping label -> radius
        largest_labels: List of axon labels to extract

    Returns:
        Dict mapping label -> (positions, radii) where positions are slice indices
    """
    logger.info(f"Extracting radius profiles for {len(largest_labels)} axons")

    profiles = {}

    for label in largest_labels:
        positions = []
        radii = []

        for z in sorted(radii_data.keys()):
            if label in radii_data[z]:
                positions.append(z)
                radii.append(radii_data[z][label])

        profiles[label] = (np.array(positions), np.array(radii))

    return profiles


def plot_largest_axon_profiles(profiles: Dict[int, Tuple[np.ndarray, np.ndarray]],
                               max_radii: Dict[int, float],
                               output_file: Path,
                               voxel_size_um: float = 0.05,
                               bundle_name: str = "Bundle"):
    """
    Plot radius profiles of the largest axons.

    Args:
        profiles: Dict mapping label -> (positions, radii)
        max_radii: Dict mapping label -> max_radius (for sorting in legend)
        output_file: Path to save figure
        voxel_size_um: Voxel size in micrometers
        bundle_name: Name of the bundle for title
    """
    logger.info(f"Plotting radius profiles for {len(profiles)} axons")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Sort labels by max radius for consistent legend ordering
    sorted_labels = sorted(profiles.keys(), key=lambda label: max_radii[label], reverse=True)

    # Use colormap for distinguishable colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(sorted_labels)))

    for i, label in enumerate(sorted_labels):
        positions, radii = profiles[label]
        positions_um = positions * voxel_size_um

        ax.plot(positions_um, radii, linewidth=1.5, alpha=0.8,
               color=colors[i], label=f'Axon {label} (max: {max_radii[label]:.2f} μm)')

    ax.set_xlabel('Position along tract (μm)', fontsize=12)
    ax.set_ylabel('Radius (μm)', fontsize=12)
    ax.set_title(f'{bundle_name}: Radius Profiles of Largest Axons', fontsize=14, pad=15)
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved radius profiles to {output_file}")


def analyze_largest_axons(volume_file: Path,
                          output_dir: Path,
                          n_largest: int = 10,
                          voxel_size_um: float = 0.05):
    """
    Analyze and plot radius profiles of the largest axons.

    Args:
        volume_file: Path to aligned HDF5 volume file
        output_dir: Directory for output plots
        n_largest: Number of largest axons to plot
        voxel_size_um: Voxel size in micrometers
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing largest axons: {volume_file.name}")
    logger.info(f"{'='*80}\n")

    # Load volume
    with h5py.File(volume_file, 'r') as f:
        volume = f['labels'][:]
        bundle_id = f['labels'].attrs.get('bundle_id', 'Unknown')
        n_axons = f['labels'].attrs.get('n_axons', 'Unknown')

    logger.info(f"Bundle ID: {bundle_id}")
    logger.info(f"Total axons: {n_axons}")
    logger.info(f"Volume shape: {volume.shape}")

    # Extract radii per axon per slice
    radii_data = extract_axon_radii_per_slice(volume, voxel_size_um)

    # Compute max radius per axon
    max_radii = compute_max_radius_per_axon(radii_data)

    # Get N largest axons
    largest_labels = get_largest_axons(max_radii, n_largest)

    # Extract radius profiles
    profiles = extract_radius_profiles(radii_data, largest_labels)

    # Plot radius profiles
    bundle_name = f"Bundle {bundle_id}" if bundle_id != 'Unknown' else volume_file.stem
    output_file = output_dir / f'{volume_file.stem}_largest_{n_largest}_axon_profiles.png'
    plot_largest_axon_profiles(profiles, max_radii, output_file, voxel_size_um, bundle_name)

    logger.info(f"\n{'='*80}")
    logger.info(f"Completed: {volume_file.name}")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot radius profiles of the largest axons in a bundle'
    )
    parser.add_argument('volume_file', type=Path,
                       help='Path to aligned HDF5 volume file')
    parser.add_argument('output_dir', type=Path,
                       help='Output directory for plots')
    parser.add_argument('--n-largest', type=int, default=10,
                       help='Number of largest axons to plot (default: 10)')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                       help='Voxel size in micrometers (default: 0.05)')

    args = parser.parse_args()

    analyze_largest_axons(
        args.volume_file,
        args.output_dir,
        n_largest=args.n_largest,
        voxel_size_um=args.voxel_size
    )
