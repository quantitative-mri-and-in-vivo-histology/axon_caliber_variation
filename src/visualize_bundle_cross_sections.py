#!/usr/bin/env python3
"""
Visualize cross-sections of bundle volumes at various positions along z-axis.

Creates a figure showing multiple cross-sections to illustrate the bundle structure.
"""

import argparse
import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def visualize_cross_sections(input_file: Path,
                            output_file: Path,
                            n_slices: int = 5,
                            figsize: tuple = (20, 8),
                            dpi: int = 300):
    """
    Create visualization of cross-sections along the z-axis.

    Args:
        input_file: Input HDF5 file with bundle volume
        output_file: Output PNG file for visualization
        n_slices: Number of cross-sections to show
        figsize: Figure size (width, height) in inches
        dpi: Resolution in dots per inch
    """
    logger.info(f"Loading volume: {input_file}")

    with h5py.File(input_file, 'r') as f:
        volume = f['labels'][:]

        # Get metadata
        bundle_id = f['labels'].attrs.get('bundle_id', 'Unknown')
        n_axons = f['labels'].attrs.get('n_axons', 'Unknown')
        voxel_size = f['labels'].attrs.get('voxel_size_um', 0.05)

    logger.info(f"Volume shape: {volume.shape}")
    logger.info(f"Bundle ID: {bundle_id}, N axons: {n_axons}")

    # Select slice positions evenly distributed along z-axis
    z_max = volume.shape[0]
    z_positions = np.linspace(0.1 * z_max, 0.9 * z_max, n_slices).astype(int)

    # Create figure
    fig, axes = plt.subplots(1, n_slices, figsize=figsize)
    if n_slices == 1:
        axes = [axes]

    for i, z in enumerate(z_positions):
        ax = axes[i]

        # Get cross-section
        cross_section = volume[z, :, :]

        # Create binary mask (any axon present)
        mask = cross_section > 0

        # Display
        ax.imshow(mask, cmap='binary', origin='lower', interpolation='nearest')
        ax.set_title(f'z = {z * voxel_size:.1f} μm\n(slice {z}/{z_max})', fontsize=10)
        ax.set_xlabel('x (voxels)', fontsize=8)
        if i == 0:
            ax.set_ylabel('y (voxels)', fontsize=8)
        ax.tick_params(labelsize=7)

        # Add axon count for this slice
        n_axons_slice = len(np.unique(cross_section[cross_section > 0]))
        ax.text(0.02, 0.98, f'{n_axons_slice} axons',
               transform=ax.transAxes, fontsize=8,
               verticalalignment='top', bbox=dict(boxstyle='round',
               facecolor='white', alpha=0.8))

    plt.suptitle(f'Bundle {bundle_id} - Cross-sections along z-axis\n'
                f'Total: {n_axons} axons, Voxel size: {voxel_size} μm',
                fontsize=12, y=1.02)

    plt.tight_layout()

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved visualization: {output_file}")
    plt.close()


def visualize_with_labels(input_file: Path,
                         output_file: Path,
                         n_slices: int = 5,
                         figsize: tuple = (20, 8),
                         dpi: int = 300):
    """
    Create visualization showing individual axon labels with random colors.

    Args:
        input_file: Input HDF5 file with bundle volume
        output_file: Output PNG file for visualization
        n_slices: Number of cross-sections to show
        figsize: Figure size (width, height) in inches
        dpi: Resolution in dots per inch
    """
    logger.info(f"Loading volume: {input_file}")

    with h5py.File(input_file, 'r') as f:
        volume = f['labels'][:]

        # Get metadata
        bundle_id = f['labels'].attrs.get('bundle_id', 'Unknown')
        n_axons = f['labels'].attrs.get('n_axons', 'Unknown')
        voxel_size = f['labels'].attrs.get('voxel_size_um', 0.05)

    logger.info(f"Volume shape: {volume.shape}")
    logger.info(f"Bundle ID: {bundle_id}, N axons: {n_axons}")

    # Select slice positions
    z_max = volume.shape[0]
    z_positions = np.linspace(0.1 * z_max, 0.9 * z_max, n_slices).astype(int)

    # Create random colormap for axon labels
    np.random.seed(42)
    all_labels = np.unique(volume[volume > 0])
    n_labels = len(all_labels)
    colors = np.random.rand(n_labels + 1, 3)
    colors[0] = [0, 0, 0]  # Background is black

    # Create figure
    fig, axes = plt.subplots(1, n_slices, figsize=figsize)
    if n_slices == 1:
        axes = [axes]

    for i, z in enumerate(z_positions):
        ax = axes[i]

        # Get cross-section
        cross_section = volume[z, :, :]

        # Map labels to color indices
        colored = np.zeros(cross_section.shape + (3,))
        for j, label in enumerate(all_labels):
            mask = cross_section == label
            colored[mask] = colors[j + 1]

        # Display
        ax.imshow(colored, origin='lower', interpolation='nearest')
        ax.set_title(f'z = {z * voxel_size:.1f} μm\n(slice {z}/{z_max})', fontsize=10)
        ax.set_xlabel('x (voxels)', fontsize=8)
        if i == 0:
            ax.set_ylabel('y (voxels)', fontsize=8)
        ax.tick_params(labelsize=7)

        # Add axon count
        n_axons_slice = len(np.unique(cross_section[cross_section > 0]))
        ax.text(0.02, 0.98, f'{n_axons_slice} axons',
               transform=ax.transAxes, fontsize=8,
               verticalalignment='top', bbox=dict(boxstyle='round',
               facecolor='white', alpha=0.8))

    plt.suptitle(f'Bundle {bundle_id} - Labeled cross-sections\n'
                f'Total: {n_axons} axons (random colors), Voxel size: {voxel_size} μm',
                fontsize=12, y=1.02)

    plt.tight_layout()

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved labeled visualization: {output_file}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize cross-sections of bundle volumes'
    )
    parser.add_argument('input_file', type=Path,
                       help='Input HDF5 bundle file')
    parser.add_argument('output_dir', type=Path,
                       help='Output directory for visualizations')
    parser.add_argument('--n-slices', type=int, default=5,
                       help='Number of cross-sections (default: 5)')
    parser.add_argument('--show-labels', action='store_true',
                       help='Create labeled visualization with random colors')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Resolution in DPI (default: 300)')

    args = parser.parse_args()

    # Get bundle name from filename
    bundle_name = args.input_file.stem

    # Create binary mask visualization
    output_file = args.output_dir / f'{bundle_name}_cross_sections.png'
    visualize_cross_sections(args.input_file, output_file,
                            n_slices=args.n_slices, dpi=args.dpi)

    # Optionally create labeled visualization
    if args.show_labels:
        output_file_labeled = args.output_dir / f'{bundle_name}_cross_sections_labeled.png'
        visualize_with_labels(args.input_file, output_file_labeled,
                             n_slices=args.n_slices, dpi=args.dpi)

    logger.info("Done!")
