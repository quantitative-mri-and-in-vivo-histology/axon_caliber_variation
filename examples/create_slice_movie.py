#!/usr/bin/env python3
"""
Create a movie through axon cross-sections colored by relative radius.

Generates frames showing how axon radii vary along the bundle, with the image
cropped to the ROI containing the population of interest.

Usage:
    python examples/create_slice_movie.py \
        data/processed/LM/sham_25_ipsi_axon_profiles.npz \
        fig/slice_movie \
        --mat-file data/raw/Sham_25_ipsi/LM_25_ipsi_myelinated_axons.mat \
        --population cg \
        --fps 10
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Tuple, Dict, Optional, Set

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops
from tqdm import tqdm

from axonometry import get_plot_settings

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_axon_profiles(npz_file: Path) -> Dict:
    """Load axon profile data."""
    data = np.load(npz_file, allow_pickle=True)
    return {
        'labels': data['labels'],
        'mean_radii': data['mean_radii_um'],
        'voxel_size': float(data['voxel_size_um'][0]),
    }


def load_population_data(npz_file: Path, population: str) -> Tuple[Set[int], int]:
    """Load population labels and dominant axis from populations JSON."""
    sample_name = npz_file.stem.replace('_axon_profiles', '')
    pop_file = npz_file.parent / f'{sample_name}_populations.json'

    if not pop_file.exists():
        raise FileNotFoundError(f"Population file not found: {pop_file}")

    with open(pop_file) as f:
        data = json.load(f)

    for pop in data['populations']:
        if pop['name'].lower() == population.lower():
            labels = set(pop['axon_labels'])
            axis = pop['dominant_axis']
            logger.info(f"Loaded {population.upper()} population: {len(labels)} axons, "
                       f"dominant axis = {axis} ({pop['dominant_axis_name']})")
            return labels, axis

    raise ValueError(f"Population '{population}' not found in {pop_file}")


def load_volume(mat_file: Path) -> Tuple[np.ndarray, float]:
    """Load 3D labeled volume from .mat file."""
    from axonometry.io import load_volume_with_metadata

    logger.info(f"Loading volume from {mat_file}")
    volume, voxel_size_tuple, _ = load_volume_with_metadata(mat_file)
    voxel_size = voxel_size_tuple[0]
    logger.info(f"  Volume shape: {volume.shape}, voxel size: {voxel_size:.4f} um")
    return volume, voxel_size


def find_roi_bounds(
    volume: np.ndarray,
    slice_axis: int,
    label_filter: Set[int],
    padding: int = 50
) -> Tuple[int, int, int, int, int, int]:
    """
    Find the bounding box containing the population across all slices.

    Returns:
        Tuple of (y_min, y_max, x_min, x_max, slice_min, slice_max)
    """
    n_slices = volume.shape[slice_axis]

    y_mins, y_maxs, x_mins, x_maxs = [], [], [], []
    valid_slices = []

    # Sample slices to find bounds (every 10th slice for speed)
    for i in range(0, n_slices, 10):
        if slice_axis == 0:
            slice_2d = volume[i, :, :]
        elif slice_axis == 1:
            slice_2d = volume[:, i, :]
        else:
            slice_2d = volume[:, :, i]

        # Filter to population
        mask = np.isin(slice_2d, list(label_filter))
        if not mask.any():
            continue

        valid_slices.append(i)
        ys, xs = np.where(mask)
        y_mins.append(ys.min())
        y_maxs.append(ys.max())
        x_mins.append(xs.min())
        x_maxs.append(xs.max())

    if not valid_slices:
        raise ValueError("No axons found in any slice")

    # Get overall bounds with padding
    y_min = max(0, min(y_mins) - padding)
    y_max = min(volume.shape[1] if slice_axis != 1 else volume.shape[0], max(y_maxs) + padding)
    x_min = max(0, min(x_mins) - padding)
    x_max = min(volume.shape[2] if slice_axis != 2 else volume.shape[1], max(x_maxs) + padding)

    slice_min = min(valid_slices)
    slice_max = max(valid_slices)

    return y_min, y_max, x_min, x_max, slice_min, slice_max


def create_movie_frames(
    npz_file: Path,
    mat_file: Path,
    output_dir: Path,
    population: str,
    cmap_name: str = 'RdBu_r',
    vmin: float = 0.7,
    vmax: float = 1.3,
    step: int = 1,
    dpi: int = 150,
    smooth_sigma: float = 10.0,
    grid_res_um: float = 2.0
) -> Tuple[Path, int]:
    """
    Create frames for the movie with two panels:
    - Left: Individual axons colored by relative radius
    - Right: Smoothed heatmap of regional variation

    Args:
        smooth_sigma: Gaussian smoothing sigma in μm for heatmap
        grid_res_um: Grid resolution in μm for heatmap

    Returns:
        Tuple of (frames directory, number of frames)
    """
    settings = get_plot_settings()

    # Load data
    profiles = load_axon_profiles(npz_file)
    pop_labels, slice_axis = load_population_data(npz_file, population)
    volume, voxel_size = load_volume(mat_file)

    # Filter to population
    mask = np.isin(profiles['labels'], list(pop_labels))
    label_to_mean = {
        int(lab): rad
        for lab, rad in zip(profiles['labels'][mask], profiles['mean_radii'][mask])
    }

    # Find ROI bounds
    logger.info("Finding ROI bounds...")
    y_min, y_max, x_min, x_max, slice_min, slice_max = find_roi_bounds(
        volume, slice_axis, pop_labels, padding=30
    )
    logger.info(f"  ROI: y=[{y_min}:{y_max}], x=[{x_min}:{x_max}], slices=[{slice_min}:{slice_max}]")

    # Setup colormap
    cmap = plt.colormaps.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Create frames directory
    frames_dir = output_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Generate frames
    slice_indices = range(slice_min, slice_max + 1, step)
    n_frames = len(list(slice_indices))

    logger.info(f"Creating {n_frames} frames...")

    for frame_idx, slice_idx in enumerate(tqdm(slice_indices, desc="Rendering frames")):
        # Extract slice
        if slice_axis == 0:
            slice_2d = volume[slice_idx, y_min:y_max, x_min:x_max]
        elif slice_axis == 1:
            slice_2d = volume[y_min:y_max, slice_idx, x_min:x_max]
        else:
            slice_2d = volume[y_min:y_max, x_min:x_max, slice_idx]

        # Filter to population
        pop_mask = np.isin(slice_2d, list(pop_labels))
        slice_2d = np.where(pop_mask, slice_2d, 0)

        # Create RGBA image for left panel
        rgba = np.zeros((*slice_2d.shape, 4), dtype=np.float32)

        # Get radii for this slice and collect data for heatmap
        props = regionprops(slice_2d)
        n_axons = 0
        rel_values = []
        centroids = []  # (x_um, y_um, rel_radius)

        for prop in props:
            label_id = prop.label
            if label_id in label_to_mean and label_to_mean[label_id] > 0:
                area_um2 = prop.area * voxel_size * voxel_size
                slice_radius = np.sqrt(area_um2 / np.pi)
                mean_radius = label_to_mean[label_id]
                relative = slice_radius / mean_radius
                rel_values.append(relative)

                # Store centroid in μm for heatmap
                cy, cx = prop.centroid
                centroids.append((cx * voxel_size, cy * voxel_size, relative))

                color = cmap(norm(relative))
                mask = slice_2d == label_id
                rgba[mask] = color
                n_axons += 1

        # Create smoothed heatmap
        height_um = slice_2d.shape[0] * voxel_size
        width_um = slice_2d.shape[1] * voxel_size
        nx = int(width_um / grid_res_um) + 1
        ny = int(height_um / grid_res_um) + 1

        grid_sum = np.zeros((ny, nx))
        grid_count = np.zeros((ny, nx))

        for cx_um, cy_um, rel_r in centroids:
            xi = int(cx_um / grid_res_um)
            yi = int(cy_um / grid_res_um)
            xi = min(xi, nx - 1)
            yi = min(yi, ny - 1)
            grid_sum[yi, xi] += rel_r
            grid_count[yi, xi] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            grid_mean = np.where(grid_count > 0, grid_sum / grid_count, np.nan)

        # Smooth the grid (sigma in grid units)
        sigma_grid = smooth_sigma / grid_res_um
        grid_filled = np.nan_to_num(grid_mean, nan=1.0)
        grid_smooth = gaussian_filter(grid_filled, sigma=sigma_grid)

        # Create two-panel figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Left panel: Individual axons
        ax1 = axes[0]
        ax1.imshow(rgba, interpolation='nearest', origin='lower')
        ax1.set_title('Individual Axons', fontsize=12, color='white', fontweight='bold')
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Right panel: Smoothed heatmap
        ax2 = axes[1]
        extent = [0, width_um, 0, height_um]
        im = ax2.imshow(grid_smooth, origin='lower', extent=extent,
                        cmap=cmap_name, vmin=0.9, vmax=1.1, aspect='auto')
        ax2.set_title(f'Smoothed (σ={smooth_sigma:.0f}μm)', fontsize=12, color='white', fontweight='bold')
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Position info
        pos_um = slice_idx * voxel_size
        axis_names = ['Z', 'Y', 'X']

        # Stats
        if rel_values:
            mean_rel = np.mean(rel_values)
            std_rel = np.std(rel_values)
            pct_below = 100 * np.mean(np.array(rel_values) < 1.0)
            stats_text = f'n={n_axons}, mean={mean_rel:.2f}±{std_rel:.2f}, {pct_below:.0f}% < 1'
        else:
            stats_text = 'n=0'

        fig.suptitle(
            f'{population.upper()} - {axis_names[slice_axis]}={pos_um:.1f} μm\n{stats_text}',
            fontsize=14, color='white', fontweight='bold'
        )

        # Add scale bar to left panel (20 um)
        scale_bar_um = 20
        scale_bar_px = scale_bar_um / voxel_size
        y_pos = slice_2d.shape[0] * 0.05
        ax1.plot([10, 10 + scale_bar_px], [y_pos, y_pos], 'w-', linewidth=2)
        ax1.text(10 + scale_bar_px / 2, y_pos + slice_2d.shape[0] * 0.03,
                f'{scale_bar_um} μm', color='white', fontsize=10, ha='center')

        # Add colorbars
        cbar1 = plt.colorbar(sm, ax=ax1, shrink=0.6, pad=0.02)
        cbar1.set_label('r / ⟨r⟩', fontsize=10, color='white')
        cbar1.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='white')

        cbar2 = plt.colorbar(im, ax=ax2, shrink=0.6, pad=0.02)
        cbar2.set_label('r / ⟨r⟩ (smoothed)', fontsize=10, color='white')
        cbar2.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='white')

        plt.tight_layout()

        # Save frame
        frame_file = frames_dir / f'frame_{frame_idx:04d}.png'
        plt.savefig(frame_file, dpi=dpi, bbox_inches='tight',
                    facecolor='black', edgecolor='none')
        plt.close()

    return frames_dir, n_frames


def create_movie(
    frames_dir: Path,
    output_file: Path,
    fps: int = 10
) -> None:
    """Create movie from frames using ffmpeg."""
    logger.info(f"Creating movie at {fps} fps...")

    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', str(frames_dir / 'frame_%04d.png'),
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',  # Ensure even dimensions for H.264
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        str(output_file)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Saved movie to {output_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed: {e.stderr.decode()}")
        raise
    except FileNotFoundError:
        logger.warning("ffmpeg not found. Frames saved but movie not created.")
        logger.info(f"To create movie manually: ffmpeg -framerate {fps} -i {frames_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Create movie through axon cross-sections colored by relative radius',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create movie for CG population
    python examples/create_slice_movie.py \\
        data/processed/LM/sham_25_ipsi_axon_profiles.npz \\
        fig/slice_movie \\
        --mat-file data/raw/Sham_25_ipsi/LM_25_ipsi_myelinated_axons.mat \\
        --population cg \\
        --fps 10
"""
    )

    parser.add_argument('input', type=Path,
                        help='Path to axon profiles NPZ file')
    parser.add_argument('output_dir', type=Path,
                        help='Output directory for frames and movie')
    parser.add_argument('--mat-file', type=Path, required=True,
                        help='Path to .mat file')
    parser.add_argument('--population', type=str, required=True, choices=['cc', 'cg'],
                        help='Population to visualize')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second (default: 10)')
    parser.add_argument('--step', type=int, default=2,
                        help='Step between slices (default: 2)')
    parser.add_argument('--colormap', type=str, default='RdBu_r',
                        help='Colormap name (default: RdBu_r)')
    parser.add_argument('--vmin', type=float, default=0.7,
                        help='Minimum relative radius for colormap (default: 0.7)')
    parser.add_argument('--vmax', type=float, default=1.3,
                        help='Maximum relative radius for colormap (default: 1.3)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for frames (default: 150)')
    parser.add_argument('--smooth-sigma', type=float, default=10.0,
                        help='Smoothing sigma in μm for heatmap (default: 10)')

    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")

    if not args.mat_file.exists():
        parser.error(f"Mat file not found: {args.mat_file}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create frames
    frames_dir, n_frames = create_movie_frames(
        npz_file=args.input,
        mat_file=args.mat_file,
        output_dir=args.output_dir,
        population=args.population,
        cmap_name=args.colormap,
        vmin=args.vmin,
        vmax=args.vmax,
        step=args.step,
        dpi=args.dpi,
        smooth_sigma=args.smooth_sigma
    )

    # Create movie
    sample_name = args.input.stem.replace('_axon_profiles', '')
    movie_file = args.output_dir / f'{sample_name}_{args.population}_movie.mp4'
    create_movie(frames_dir, movie_file, fps=args.fps)

    logger.info(f"Done! {n_frames} frames created.")


if __name__ == '__main__':
    main()
