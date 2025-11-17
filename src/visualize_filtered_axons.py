#!/usr/bin/env python3
"""
Visualize filtered axons from the JSON file on the original volume.

Creates various visualizations:
- 3D plots of filtered axons colored by alignment axis
- Maximum intensity projections with axon overlays
- Statistics plots
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AxonVisualizer:
    """Visualize filtered axons"""

    def __init__(self, labels: np.ndarray, filtering_results: dict, downsample_factor: int = 1):
        self.labels = labels
        self.results = filtering_results
        self.downsample_factor = downsample_factor

        self.passing_ids = set(filtering_results['passing_axon_ids'])
        self.passing_props = filtering_results['passing_properties']

    def create_filtered_volume(self) -> np.ndarray:
        """Create volume with only passing axons"""
        filtered = np.zeros_like(self.labels)

        for axon_id in self.passing_ids:
            mask = (self.labels == axon_id)
            filtered[mask] = axon_id

        return filtered

    def plot_max_projections(self, output_file: Path):
        """Create maximum intensity projections along each axis"""
        logger.info("Creating maximum intensity projections...")

        filtered_vol = self.create_filtered_volume()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original projections
        axes[0, 0].imshow(self.labels.max(axis=0), cmap='tab20', interpolation='nearest')
        axes[0, 0].set_title('Original - XY projection (max Z)')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(self.labels.max(axis=1), cmap='tab20', interpolation='nearest')
        axes[0, 1].set_title('Original - XZ projection (max Y)')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(self.labels.max(axis=2), cmap='tab20', interpolation='nearest')
        axes[0, 2].set_title('Original - YZ projection (max X)')
        axes[0, 2].axis('off')

        # Filtered projections
        axes[1, 0].imshow(filtered_vol.max(axis=0), cmap='tab20', interpolation='nearest')
        axes[1, 0].set_title('Filtered - XY projection (max Z)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(filtered_vol.max(axis=1), cmap='tab20', interpolation='nearest')
        axes[1, 1].set_title('Filtered - XZ projection (max Y)')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(filtered_vol.max(axis=2), cmap='tab20', interpolation='nearest')
        axes[1, 2].set_title('Filtered - YZ projection (max X)')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved projections to {output_file}")

    def plot_3d_skeleton(self, output_file: Path, max_axons: int = 100):
        """Plot 3D representation of axon centerlines colored by alignment axis"""
        logger.info("Creating 3D skeleton plot...")

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Colors for each axis (RGBA)
        color_rgba = {
            'x': np.array([1.0, 0.0, 0.0, 0.6]),
            'y': np.array([0.0, 1.0, 0.0, 0.6]),
            'z': np.array([0.0, 0.0, 1.0, 0.6])
        }

        # Sample axons if too many
        passing_ids_list = list(self.passing_ids)
        if len(passing_ids_list) > max_axons:
            logger.info(f"Sampling {max_axons} out of {len(passing_ids_list)} axons for visualization")
            np.random.seed(42)
            passing_ids_list = np.random.choice(passing_ids_list, max_axons, replace=False)

        # Collect line segments for each alignment
        segments_by_alignment = {'x': [], 'y': [], 'z': []}

        for axon_id in passing_ids_list:
            # Get all voxels for this axon
            coords = np.argwhere(self.labels == axon_id)

            if len(coords) < 2:
                continue

            # Coords are in [z, y, x] order
            props = self.passing_props[str(axon_id)]
            alignment = props['alignment_axis']

            # Sort along principal axis for better line rendering
            if alignment == 'x':
                sort_idx = coords[:, 2].argsort()  # Sort by x
            elif alignment == 'y':
                sort_idx = coords[:, 1].argsort()  # Sort by y
            else:  # z
                sort_idx = coords[:, 0].argsort()  # Sort by z

            coords_sorted = coords[sort_idx]

            # Convert to x, y, z
            points = coords_sorted[:, [2, 1, 0]]  # [x, y, z]

            # Create line segments
            for i in range(len(points) - 1):
                segments_by_alignment[alignment].append([points[i], points[i+1]])

        # Plot as line collections (much faster than individual scatter points)
        for alignment, segments in segments_by_alignment.items():
            if len(segments) > 0:
                lc = Line3DCollection(
                    segments,
                    colors=color_rgba[alignment],
                    linewidths=1.5,
                    alpha=0.6
                )
                ax.add_collection3d(lc)

        # Set axis limits
        ax.set_xlim(0, self.labels.shape[2])
        ax.set_ylim(0, self.labels.shape[1])
        ax.set_zlim(0, self.labels.shape[0])

        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(f'3D View of Filtered Axons (n={len(passing_ids_list)})\nRed=X-aligned, Green=Y-aligned, Blue=Z-aligned',
                    fontsize=14, pad=20)

        # Create legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=3, label='X-aligned'),
            Line2D([0], [0], color='green', linewidth=3, label='Y-aligned'),
            Line2D([0], [0], color='blue', linewidth=3, label='Z-aligned')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

        # Better viewing angle
        ax.view_init(elev=20, azim=45)

        # Set background color
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.3)

        plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Saved 3D plot to {output_file}")

    def plot_3d_multiview(self, output_file: Path, max_axons: int = 100):
        """Create 3D plot with multiple viewing angles"""
        logger.info("Creating multi-view 3D plot...")

        # Colors for each axis (RGBA)
        color_rgba = {
            'x': np.array([1.0, 0.0, 0.0, 0.6]),
            'y': np.array([0.0, 1.0, 0.0, 0.6]),
            'z': np.array([0.0, 0.0, 1.0, 0.6])
        }

        # Sample axons if too many
        passing_ids_list = list(self.passing_ids)
        if len(passing_ids_list) > max_axons:
            logger.info(f"Sampling {max_axons} out of {len(passing_ids_list)} axons for visualization")
            np.random.seed(42)
            passing_ids_list = np.random.choice(passing_ids_list, max_axons, replace=False)

        # Collect line segments for each alignment
        segments_by_alignment = {'x': [], 'y': [], 'z': []}

        for axon_id in passing_ids_list:
            coords = np.argwhere(self.labels == axon_id)
            if len(coords) < 2:
                continue

            props = self.passing_props[str(axon_id)]
            alignment = props['alignment_axis']

            # Sort along principal axis
            if alignment == 'x':
                sort_idx = coords[:, 2].argsort()
            elif alignment == 'y':
                sort_idx = coords[:, 1].argsort()
            else:
                sort_idx = coords[:, 0].argsort()

            coords_sorted = coords[sort_idx]
            points = coords_sorted[:, [2, 1, 0]]  # [x, y, z]

            for i in range(len(points) - 1):
                segments_by_alignment[alignment].append([points[i], points[i+1]])

        # Create figure with 4 subplots
        fig = plt.figure(figsize=(20, 16))

        # Different viewing angles
        views = [
            (20, 45, 'Oblique View'),
            (0, 0, 'Front View (YZ plane)'),
            (0, 90, 'Side View (XZ plane)'),
            (90, 0, 'Top View (XY plane)')
        ]

        for idx, (elev, azim, title) in enumerate(views):
            ax = fig.add_subplot(2, 2, idx+1, projection='3d')

            # Plot line collections
            for alignment, segments in segments_by_alignment.items():
                if len(segments) > 0:
                    lc = Line3DCollection(
                        segments,
                        colors=color_rgba[alignment],
                        linewidths=1.5,
                        alpha=0.6
                    )
                    ax.add_collection3d(lc)

            # Set axis limits
            ax.set_xlim(0, self.labels.shape[2])
            ax.set_ylim(0, self.labels.shape[1])
            ax.set_zlim(0, self.labels.shape[0])

            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Y', fontsize=10)
            ax.set_zlabel('Z', fontsize=10)
            ax.set_title(title, fontsize=12)

            # Set viewing angle
            ax.view_init(elev=elev, azim=azim)

            # Clean background
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(True, alpha=0.3)

        # Add legend to the figure
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=3, label='X-aligned'),
            Line2D([0], [0], color='green', linewidth=3, label='Y-aligned'),
            Line2D([0], [0], color='blue', linewidth=3, label='Z-aligned')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.98))

        plt.suptitle(f'3D Multi-View of Filtered Axons (n={len(passing_ids_list)})',
                    fontsize=16, y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Saved multi-view 3D plot to {output_file}")

    def plot_statistics(self, output_file: Path):
        """Plot statistics about filtered axons"""
        logger.info("Creating statistics plots...")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Extract properties
        all_props = self.results['all_properties']
        passing_props = self.results['passing_properties']

        # 1. Alignment axis distribution
        axis_counts = {'x': 0, 'y': 0, 'z': 0}
        for props in passing_props.values():
            axis_counts[props['alignment_axis']] += 1

        axes[0, 0].bar(axis_counts.keys(), axis_counts.values(), color=['red', 'green', 'blue'])
        axes[0, 0].set_title('Axons by Alignment Axis')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(axis='y', alpha=0.3)

        # 2. Alignment score distribution
        alignment_scores_all = [p['alignment_score'] for p in all_props.values()]
        alignment_scores_pass = [p['alignment_score'] for p in passing_props.values()]

        axes[0, 1].hist(alignment_scores_all, bins=50, alpha=0.5, label='All', color='gray')
        axes[0, 1].hist(alignment_scores_pass, bins=50, alpha=0.7, label='Passing', color='blue')
        axes[0, 1].axvline(self.results['metadata']['filters']['min_alignment_score'],
                          color='red', linestyle='--', label='Threshold')
        axes[0, 1].set_title('Alignment Score Distribution')
        axes[0, 1].set_xlabel('Alignment Score')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # 3. Straightness distribution
        straightness_all = [p['straightness'] for p in all_props.values()]
        straightness_pass = [p['straightness'] for p in passing_props.values()]

        axes[0, 2].hist(straightness_all, bins=50, alpha=0.5, label='All', color='gray')
        axes[0, 2].hist(straightness_pass, bins=50, alpha=0.7, label='Passing', color='blue')
        axes[0, 2].axvline(self.results['metadata']['filters']['min_straightness'],
                          color='red', linestyle='--', label='Threshold')
        axes[0, 2].set_title('Straightness Distribution')
        axes[0, 2].set_xlabel('Straightness')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3)

        # 4. Max extent distribution
        extents_all = [max(p['extent_x'], p['extent_y'], p['extent_z']) for p in all_props.values()]
        extents_pass = [max(p['extent_x'], p['extent_y'], p['extent_z']) for p in passing_props.values()]

        axes[1, 0].hist(extents_all, bins=50, alpha=0.5, label='All', color='gray', range=(0, 200))
        axes[1, 0].hist(extents_pass, bins=50, alpha=0.7, label='Passing', color='blue', range=(0, 200))
        axes[1, 0].axvline(self.results['metadata']['filters']['min_extent_voxels'],
                          color='red', linestyle='--', label='Threshold')
        axes[1, 0].set_title('Max Extent Distribution')
        axes[1, 0].set_xlabel('Max Extent (voxels)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # 5. Number of voxels distribution
        n_voxels_all = [p['n_voxels'] for p in all_props.values()]
        n_voxels_pass = [p['n_voxels'] for p in passing_props.values()]

        axes[1, 1].hist(n_voxels_all, bins=50, alpha=0.5, label='All', color='gray')
        axes[1, 1].hist(n_voxels_pass, bins=50, alpha=0.7, label='Passing', color='blue')
        axes[1, 1].set_title('Axon Size Distribution')
        axes[1, 1].set_xlabel('Number of Voxels')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        axes[1, 1].set_xscale('log')
        axes[1, 1].grid(alpha=0.3)

        # 6. Summary text
        axes[1, 2].axis('off')
        summary_text = f"""
Filtering Summary
{'='*30}

Total axons: {self.results['n_total']}
Passing axons: {self.results['n_passing']}
Pass rate: {100*self.results['n_passing']/self.results['n_total']:.1f}%

By alignment axis:
  X-aligned: {axis_counts['x']}
  Y-aligned: {axis_counts['y']}
  Z-aligned: {axis_counts['z']}

Filter criteria:
  Min alignment: {self.results['metadata']['filters']['min_alignment_score']:.2f}
  Min extent: {self.results['metadata']['filters']['min_extent_voxels']} voxels
  Min straightness: {self.results['metadata']['filters']['min_straightness']:.2f}

Downsampling: {self.results['metadata']['downsample_factor']}x
        """
        axes[1, 2].text(0.1, 0.5, summary_text, fontfamily='monospace', fontsize=10,
                       verticalalignment='center')

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved statistics to {output_file}")

    def plot_slice_view(self, output_file: Path, slice_axis: str = 'z', slice_idx: int = None):
        """Plot a single slice with passing axons highlighted"""
        logger.info(f"Creating slice view along {slice_axis} axis...")

        if slice_idx is None:
            # Use middle slice
            slice_idx = self.labels.shape[['x', 'y', 'z'].index(slice_axis)] // 2

        # Extract slice
        if slice_axis == 'z':
            slice_orig = self.labels[:, :, slice_idx]
        elif slice_axis == 'y':
            slice_orig = self.labels[:, slice_idx, :]
        else:  # x
            slice_orig = self.labels[slice_idx, :, :]

        # Create colored overlay
        overlay = np.zeros((*slice_orig.shape, 3))

        for axon_id in self.passing_ids:
            mask = (slice_orig == axon_id)
            if mask.any():
                props = self.passing_props[str(axon_id)]
                alignment = props['alignment_axis']

                # Color by alignment
                if alignment == 'x':
                    overlay[mask] = [1, 0, 0]  # Red
                elif alignment == 'y':
                    overlay[mask] = [0, 1, 0]  # Green
                else:  # z
                    overlay[mask] = [0, 0, 1]  # Blue

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(slice_orig > 0, cmap='gray')
        axes[0].set_title(f'Original Slice ({slice_axis}={slice_idx})')
        axes[0].axis('off')

        axes[1].imshow(overlay)
        axes[1].set_title('Filtered Axons by Alignment\n(Red=X, Green=Y, Blue=Z)')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved slice view to {output_file}")


def visualize_filtered_axons(labels_file: Path,
                             filtering_json: Path,
                             output_dir: Path,
                             downsample_factor: int = 4):
    """
    Create visualizations of filtered axons.

    Args:
        labels_file: Path to original .mat file
        filtering_json: Path to JSON file with filtering results
        output_dir: Directory for output figures
        downsample_factor: Downsampling factor used in filtering
    """
    logger.info(f"\n{'='*80}")
    logger.info("Creating visualizations")
    logger.info(f"{'='*80}\n")

    # Load data
    logger.info(f"Loading volume from {labels_file}")
    with h5py.File(labels_file, 'r') as f:
        labels = f['final_lbl'][:]

    # Downsample to match filtering
    if downsample_factor > 1:
        labels = labels[::downsample_factor, ::downsample_factor, ::downsample_factor]
        logger.info(f"Downsampled to {labels.shape}")

    logger.info(f"Loading filtering results from {filtering_json}")
    with open(filtering_json, 'r') as f:
        results = json.load(f)

    # Create visualizer
    viz = AxonVisualizer(labels, results, downsample_factor)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    viz.plot_max_projections(output_dir / 'projections.png')
    viz.plot_statistics(output_dir / 'statistics.png')
    viz.plot_3d_skeleton(output_dir / 'skeleton_3d.png', max_axons=100)
    viz.plot_3d_multiview(output_dir / 'skeleton_3d_multiview.png', max_axons=100)
    viz.plot_slice_view(output_dir / 'slice_z.png', slice_axis='z')

    logger.info(f"\n{'='*80}")
    logger.info("Visualization complete")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    labels_file = Path('/workspace/data/raw/LM/LM_25_ipsi_myelinated_axons.mat')
    filtering_json = Path('/workspace/data/processed/LM_25_ipsi/axon_filtering.json')
    output_dir = Path('/workspace/fig/LM_25_ipsi')

    visualize_filtered_axons(
        labels_file,
        filtering_json,
        output_dir,
        downsample_factor=4
    )
