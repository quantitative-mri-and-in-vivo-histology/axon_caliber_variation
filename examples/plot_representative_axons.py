#!/usr/bin/env python3
"""
Plot representative axons with low, medium, and high coefficient of variation (CV).

This script visualizes 3 representative axons from a single volume:
- One with low CV (stable caliber along length)
- One with medium CV (typical variability)
- One with high CV (highly variable caliber)

Two visualizations are created side-by-side:
1. 3D tube rendering showing the actual axon path with radius-dependent coloring
2. Straightened 3D cylinder showing caliber variation along a straight axis
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

from axonometry import get_plot_settings
from axonometry.geometry import unit_tangent_vector, create_perpendicular_plane_grid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load plot settings
settings = get_plot_settings()

# Define base colors for the three CV categories (will be lightened/darkened by radius)
CV_COLORS = {
    'low': np.array([0.2, 0.6, 0.2]),      # Green - stable
    'medium': np.array([0.2, 0.4, 0.8]),   # Blue - typical
    'high': np.array([0.8, 0.3, 0.2]),     # Red - variable
}


def load_axon_data(npz_file: Path) -> dict:
    """
    Load axon profile data from NPZ file.

    Args:
        npz_file: Path to axon profiles NPZ file

    Returns:
        Dictionary with axon data
    """
    data = np.load(npz_file, allow_pickle=True)

    # Compute CV for each axon
    mean_radii = data['mean_radii_um']
    std_radii = data['std_radii_um']
    n_points = data['n_points']

    with np.errstate(divide='ignore', invalid='ignore'):
        cv = std_radii / mean_radii
        cv = np.where(np.isfinite(cv) & (mean_radii > 0), cv, np.nan)

    return {
        'labels': data['labels'],
        'mean_radii_um': mean_radii,
        'std_radii_um': std_radii,
        'cv': cv,
        'n_points': n_points,
        'lengths_um': data['lengths_um'],
        'skeleton_coords_um': data['skeleton_coords_um'],
        'radii_profiles_um': data['radii_profiles_um'],
        'voxel_size_um': data['voxel_size_um'],
    }


def select_representative_axons(data: dict, min_points: int = 10,
                                 min_length_um: float = 0.0) -> Tuple[int, int, int]:
    """
    Select three representative axons with low, medium, and high CV.

    Args:
        data: Dictionary with axon data
        min_points: Minimum number of skeleton points required
        min_length_um: Minimum axon length in micrometers

    Returns:
        Tuple of (low_cv_idx, medium_cv_idx, high_cv_idx)
    """
    cv = data['cv']
    n_points = data['n_points']
    lengths = data['lengths_um']

    # Filter valid axons (finite CV, enough points, and minimum length)
    valid_mask = np.isfinite(cv) & (n_points >= min_points) & (lengths >= min_length_um)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 3:
        raise ValueError(f"Not enough valid axons (need 3, found {len(valid_indices)})")

    valid_cv = cv[valid_indices]
    valid_n_points = n_points[valid_indices]

    # Compute percentiles for low/medium/high CV
    p25 = np.percentile(valid_cv, 25)
    p50 = np.percentile(valid_cv, 50)
    p75 = np.percentile(valid_cv, 75)

    # Find axons closest to these percentiles, preferring longer axons
    def find_closest(target_cv, exclude_indices=None, cv_tolerance=0.02):
        """Find axon closest to target CV, preferring longer ones within tolerance."""
        distances = np.abs(valid_cv - target_cv)

        # Exclude already selected indices
        if exclude_indices:
            for exc_idx in exclude_indices:
                exc_pos = np.where(valid_indices == exc_idx)[0]
                if len(exc_pos) > 0:
                    distances[exc_pos[0]] = np.inf

        # Find candidates within tolerance of minimum distance
        min_dist = distances.min()
        candidates = distances <= min_dist + cv_tolerance
        # Among candidates, pick the one with most points
        candidate_indices = np.where(candidates)[0]
        best_candidate = candidate_indices[np.argmax(valid_n_points[candidate_indices])]
        return valid_indices[best_candidate]

    low_idx = find_closest(p25)
    medium_idx = find_closest(p50, exclude_indices=[low_idx])
    high_idx = find_closest(p75, exclude_indices=[low_idx, medium_idx])

    logger.info(f"Selected axons:")
    logger.info(f"  Low CV:    idx={low_idx}, CV={cv[low_idx]:.3f}, n_points={n_points[low_idx]}")
    logger.info(f"  Medium CV: idx={medium_idx}, CV={cv[medium_idx]:.3f}, n_points={n_points[medium_idx]}")
    logger.info(f"  High CV:   idx={high_idx}, CV={cv[high_idx]:.3f}, n_points={n_points[high_idx]}")

    return low_idx, medium_idx, high_idx


def generate_tube_mesh(skeleton: np.ndarray, radii: np.ndarray,
                       n_circle_points: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a tube mesh around a skeleton with varying radius.

    Args:
        skeleton: (N, 3) array of skeleton points in physical coordinates
        radii: (N,) array of radii at each skeleton point
        n_circle_points: Number of points around each circle cross-section

    Returns:
        Tuple of (X, Y, Z, R) meshgrid arrays for plotting, where R is radius at each point
    """
    n_points = len(skeleton)

    # Compute tangent vectors
    tangents = unit_tangent_vector(skeleton)

    # Generate circle points
    theta = np.linspace(0, 2 * np.pi, n_circle_points, endpoint=False)

    # Initialize mesh arrays
    X = np.zeros((n_points, n_circle_points))
    Y = np.zeros((n_points, n_circle_points))
    Z = np.zeros((n_points, n_circle_points))
    R = np.zeros((n_points, n_circle_points))  # Store radius for coloring

    for i in range(n_points):
        center = skeleton[i]
        tangent = tangents[i]
        radius = radii[i]

        # Find perpendicular vectors using Gram-Schmidt
        # Start with a vector not parallel to tangent
        if abs(tangent[0]) < 0.9:
            ref = np.array([1, 0, 0])
        else:
            ref = np.array([0, 1, 0])

        # First perpendicular vector
        perp1 = ref - np.dot(ref, tangent) * tangent
        perp1 = perp1 / (np.linalg.norm(perp1) + 1e-10)

        # Second perpendicular vector
        perp2 = np.cross(tangent, perp1)
        perp2 = perp2 / (np.linalg.norm(perp2) + 1e-10)

        # Generate circle points
        for j, t in enumerate(theta):
            offset = radius * (np.cos(t) * perp1 + np.sin(t) * perp2)
            point = center + offset
            X[i, j] = point[2]  # X axis (skeleton coord index 2)
            Y[i, j] = point[1]  # Y axis (skeleton coord index 1)
            Z[i, j] = point[0]  # Z axis (skeleton coord index 0)
            R[i, j] = radius

    return X, Y, Z, R


def straighten_axon(skeleton: np.ndarray, radii: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Straighten an axon by computing arc length along the skeleton.

    Args:
        skeleton: (N, 3) array of skeleton points
        radii: (N,) array of radii

    Returns:
        Tuple of (arc_lengths, radii)
    """
    # Compute cumulative arc length
    diffs = np.diff(skeleton, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])

    return arc_lengths, radii


def generate_straight_tube_mesh(arc_lengths: np.ndarray, radii: np.ndarray,
                                 n_circle_points: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a straight tube mesh with varying radius.

    Args:
        arc_lengths: (N,) array of positions along the straightened axis
        radii: (N,) array of radii at each position
        n_circle_points: Number of points around each circle

    Returns:
        Tuple of (X, Y, Z, R) meshgrid arrays
    """
    n_points = len(arc_lengths)
    theta = np.linspace(0, 2 * np.pi, n_circle_points, endpoint=False)

    X = np.zeros((n_points, n_circle_points))
    Y = np.zeros((n_points, n_circle_points))
    Z = np.zeros((n_points, n_circle_points))
    R = np.zeros((n_points, n_circle_points))

    for i in range(n_points):
        for j, t in enumerate(theta):
            X[i, j] = arc_lengths[i]
            Y[i, j] = radii[i] * np.cos(t)
            Z[i, j] = radii[i] * np.sin(t)
            R[i, j] = radii[i]

    return X, Y, Z, R


def radius_to_color(radii: np.ndarray, base_color: np.ndarray,
                    r_min: float, r_max: float) -> np.ndarray:
    """
    Map radii to colors by varying brightness of base color.

    Args:
        radii: Array of radius values
        base_color: (3,) RGB base color
        r_min: Minimum radius for normalization
        r_max: Maximum radius for normalization

    Returns:
        RGBA color array matching radii shape
    """
    # Normalize radii to [0, 1]
    norm = Normalize(vmin=r_min, vmax=r_max)
    normalized = norm(radii)

    # Vary brightness: thin=darker, thick=lighter
    # Scale from 0.5 to 1.5 of base color
    brightness = 0.5 + normalized  # Range [0.5, 1.5]

    # Apply brightness to base color
    colors = np.zeros(radii.shape + (4,))
    for i in range(3):
        colors[..., i] = np.clip(base_color[i] * brightness, 0, 1)
    colors[..., 3] = 0.9  # Alpha

    return colors


def plot_representative_axons(data: dict, indices: Tuple[int, int, int],
                               output_file: Path) -> None:
    """
    Create side-by-side 3D and straightened visualizations.

    Args:
        data: Dictionary with axon data
        indices: Tuple of (low_cv_idx, medium_cv_idx, high_cv_idx)
        output_file: Output PNG file path
    """
    fig = plt.figure(figsize=(14, 6))

    # Create 3D subplots
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_straight = fig.add_subplot(122, projection='3d')

    font_settings = settings.fonts

    # Collect all radii for consistent color scaling
    all_radii = []
    for idx in indices:
        all_radii.extend(data['radii_profiles_um'][idx])
    r_min, r_max = min(all_radii), max(all_radii)

    # Plot each axon
    cv_labels = ['low', 'medium', 'high']
    legend_handles = []

    for i, (idx, cv_label) in enumerate(zip(indices, cv_labels)):
        skeleton = data['skeleton_coords_um'][idx]
        radii = data['radii_profiles_um'][idx]
        cv_value = data['cv'][idx]
        base_color = CV_COLORS[cv_label]

        # Generate tube mesh for 3D view
        X, Y, Z, R = generate_tube_mesh(skeleton, radii)
        colors = radius_to_color(R, base_color, r_min, r_max)

        # Plot 3D tube
        ax_3d.plot_surface(X, Y, Z, facecolors=colors, shade=False,
                           antialiased=True, linewidth=0)

        # Generate straightened tube
        arc_lengths, straight_radii = straighten_axon(skeleton, radii)
        # Offset each axon vertically for visibility
        offset = i * (r_max * 3)
        Xs, Ys, Zs, Rs = generate_straight_tube_mesh(arc_lengths, straight_radii)
        Zs = Zs + offset  # Apply vertical offset
        colors_s = radius_to_color(Rs, base_color, r_min, r_max)

        ax_straight.plot_surface(Xs, Ys, Zs, facecolors=colors_s, shade=False,
                                  antialiased=True, linewidth=0)

        # Create legend entry
        legend_handles.append(plt.Line2D([0], [0], color=base_color, linewidth=4,
                                          label=f'{cv_label.capitalize()} CV ({cv_value:.2f})'))

    # Configure 3D view
    ax_3d.set_xlabel('X (μm)', fontsize=font_settings['label_size'])
    ax_3d.set_ylabel('Y (μm)', fontsize=font_settings['label_size'])
    ax_3d.set_zlabel('Z (μm)', fontsize=font_settings['label_size'])
    ax_3d.set_title('3D Axon Morphology', fontsize=font_settings['title_size'],
                    fontweight=font_settings['weight'])
    ax_3d.legend(handles=legend_handles, loc='upper left',
                 fontsize=font_settings['legend_size'])

    # Configure straightened view
    ax_straight.set_xlabel('Arc Length (μm)', fontsize=font_settings['label_size'])
    ax_straight.set_ylabel('', fontsize=font_settings['label_size'])
    ax_straight.set_zlabel('', fontsize=font_settings['label_size'])
    ax_straight.set_title('Straightened Axons\n(caliber variation along length)',
                          fontsize=font_settings['title_size'], fontweight=font_settings['weight'])

    # Set viewing angle for straightened view (side view - looking along Y axis)
    ax_straight.view_init(elev=15, azim=-85)

    plt.tight_layout()

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"Saved figure to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Plot representative axons with low, medium, and high CV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize representative axons from a single volume
  python plot_representative_axons.py \\
      data/processed/HM/sham_25_ipsi_axon_profiles.npz \\
      fig/representative_axons.png

  # With custom minimum points threshold
  python plot_representative_axons.py \\
      data/processed/LM/sham_25_ipsi_axon_profiles.npz \\
      fig/lm_representative_axons.png \\
      --min-points 15
        """
    )

    parser.add_argument('input', type=Path,
                        help='Input NPZ file with 3D axon profiles')
    parser.add_argument('output', type=Path,
                        help='Output PNG file path')
    parser.add_argument('--min-points', type=int, default=10,
                        help='Minimum skeleton points required (default: 10)')
    parser.add_argument('--min-length', type=float, default=50.0,
                        help='Minimum axon length in micrometers (default: 50.0)')

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return

    # Load data
    logger.info(f"Loading axon data from {args.input}")
    data = load_axon_data(args.input)
    logger.info(f"Loaded {len(data['labels'])} axons")

    # Select representative axons
    try:
        indices = select_representative_axons(data, min_points=args.min_points,
                                               min_length_um=args.min_length)
    except ValueError as e:
        logger.error(str(e))
        return

    # Create visualization
    plot_representative_axons(data, indices, args.output)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("CV Distribution Summary")
    logger.info("=" * 60)
    valid_cv = data['cv'][np.isfinite(data['cv'])]
    logger.info(f"CV range: {valid_cv.min():.3f} - {valid_cv.max():.3f}")
    logger.info(f"CV percentiles: 25th={np.percentile(valid_cv, 25):.3f}, "
                f"50th={np.percentile(valid_cv, 50):.3f}, "
                f"75th={np.percentile(valid_cv, 75):.3f}")


if __name__ == '__main__':
    main()
