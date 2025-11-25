#!/usr/bin/env python3
"""
Plot central X, Y, Z slices from multiple .mat files in a grid layout.

Rows: X, Y, Z slices (or Z, Y, X depending on convention)
Columns: Different samples

Supports anisotropic voxels by downsampling to isotropic first.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom
from matplotlib.colors import ListedColormap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_voxel_size(voxel_size_um: Union[float, Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """Parse voxel size to a (vz, vy, vx) tuple matching array axes (Z, Y, X)."""
    if isinstance(voxel_size_um, (tuple, list)):
        if len(voxel_size_um) == 3:
            return tuple(float(v) for v in voxel_size_um)
        elif len(voxel_size_um) == 1:
            v = float(voxel_size_um[0])
            return (v, v, v)
        else:
            raise ValueError(f"Expected 1 or 3 voxel size values, got {len(voxel_size_um)}")
    v = float(voxel_size_um)
    return (v, v, v)


def parse_voxel_size_arg(value: str) -> Union[float, Tuple[float, float, float]]:
    """Parse voxel size CLI argument: single float or comma-separated triple."""
    if ',' in value:
        parts = value.split(',')
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"Expected 1 or 3 values, got {len(parts)}: '{value}'"
            )
        return tuple(float(p.strip()) for p in parts)
    return float(value)


def resample_to_isotropic(volume: np.ndarray,
                          voxel_size: Tuple[float, float, float]
                          ) -> Tuple[np.ndarray, float]:
    """Resample anisotropic volume to isotropic voxels."""
    vz, vy, vx = voxel_size

    if np.allclose([vz, vy, vx], vz):
        return volume, vz

    target_size = max(vz, vy, vx)
    zoom_factors = (vz / target_size, vy / target_size, vx / target_size)

    resampled = zoom(volume, zoom_factors, order=0, mode='nearest')
    logger.info(f"  Resampling: {volume.shape} -> {resampled.shape}")

    return resampled, target_size


def load_mat_volume(mat_file: Path) -> np.ndarray:
    """Load labeled volume from .mat file (v5.0 or v7.3/HDF5)."""
    try:
        with h5py.File(str(mat_file), 'r') as f:
            volume_key = None
            for key in f.keys():
                if not key.startswith('#') and not key.startswith('_'):
                    volume_key = key
                    break
            if volume_key is None:
                raise ValueError(f"No data found in {mat_file}")
            return f[volume_key][:]
    except OSError:
        mat_data = sio.loadmat(str(mat_file))
        priority_keys = ['myelinated_axons', 'volume', 'labels', 'final_lbl']
        volume_key = None
        for pkey in priority_keys:
            if pkey in mat_data:
                volume_key = pkey
                break
        if volume_key is None:
            for key in mat_data.keys():
                if not key.startswith('__'):
                    volume_key = key
                    break
        if volume_key is None:
            raise ValueError(f"No data found in {mat_file}")
        return mat_data[volume_key]


def create_label_colormap(n_labels: int, seed: int = 42) -> ListedColormap:
    """Create a random colormap for labeled data with background as black."""
    np.random.seed(seed)
    colors = np.random.rand(n_labels + 1, 3)
    colors[0] = [0, 0, 0]  # Background is black
    return ListedColormap(colors)


def get_central_slices(volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get central slices along Z, Y, X axes.

    Returns:
        Tuple of (z_slice, y_slice, x_slice)
        - z_slice: volume[z_mid, :, :] - view in XY plane
        - y_slice: volume[:, y_mid, :] - view in XZ plane
        - x_slice: volume[:, :, x_mid] - view in YZ plane
    """
    z_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    x_mid = volume.shape[2] // 2

    z_slice = volume[z_mid, :, :]  # XY plane
    y_slice = volume[:, y_mid, :]  # XZ plane
    x_slice = volume[:, :, x_mid]  # YZ plane

    return z_slice, y_slice, x_slice


def plot_central_slices_grid(mat_files: List[Path],
                              output_file: Path,
                              voxel_size_um: Union[float, Tuple[float, float, float]] = 0.05,
                              figsize_per_cell: Tuple[float, float] = (3, 3),
                              dpi: int = 150):
    """
    Plot central slices from multiple samples in a grid.

    Rows: Z-slice, Y-slice, X-slice
    Columns: Different samples
    """
    n_samples = len(mat_files)
    voxel_size_tuple = parse_voxel_size(voxel_size_um)

    # Load all volumes and get slices
    all_slices = []  # List of (z_slice, y_slice, x_slice) tuples
    sample_names = []

    for mat_file in mat_files:
        sample_name = mat_file.stem.replace('_myelinated_axons', '')
        sample_names.append(sample_name)
        logger.info(f"Loading {sample_name}...")

        volume = load_mat_volume(mat_file)
        volume, _ = resample_to_isotropic(volume, voxel_size_tuple)

        slices = get_central_slices(volume)
        all_slices.append(slices)

        logger.info(f"  Shape: {volume.shape}, slices: Z={slices[0].shape}, Y={slices[1].shape}, X={slices[2].shape}")

    # Create figure
    fig_width = figsize_per_cell[0] * n_samples
    fig_height = figsize_per_cell[1] * 3  # 3 rows for Z, Y, X

    fig, axes = plt.subplots(3, n_samples, figsize=(fig_width, fig_height))

    # Handle single sample case
    if n_samples == 1:
        axes = axes.reshape(3, 1)

    row_labels = ['Z-slice (XY)', 'Y-slice (XZ)', 'X-slice (YZ)']

    # Find global max label for consistent colormap
    max_label = max(max(s.max() for s in slices) for slices in all_slices)
    cmap = create_label_colormap(int(max_label))

    for col, (slices, name) in enumerate(zip(all_slices, sample_names)):
        for row, (slice_2d, row_label) in enumerate(zip(slices, row_labels)):
            ax = axes[row, col]

            # Plot slice
            ax.imshow(slice_2d, cmap=cmap, interpolation='nearest',
                     vmin=0, vmax=max_label)
            ax.set_aspect('equal')
            ax.axis('off')

            # Add row label on first column
            if col == 0:
                ax.set_ylabel(row_label, fontsize=10, rotation=90, va='center')
                ax.yaxis.set_label_position('left')
                ax.axis('on')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            # Add sample name on first row
            if row == 0:
                ax.set_title(name, fontsize=10)

    plt.tight_layout()

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved grid to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot central X, Y, Z slices from multiple .mat files in a grid'
    )
    parser.add_argument('input', type=Path, nargs='+',
                        help='Path to .mat file(s) or directory containing *_myelinated_axons.mat files')
    parser.add_argument('output', type=Path,
                        help='Output image file (e.g., central_slices_grid.png)')
    parser.add_argument('--voxel-size', type=parse_voxel_size_arg, default=0.05,
                        help='Voxel size in micrometers: single value for isotropic (e.g., 0.05) '
                             'or vz,vy,vx for anisotropic (e.g., 0.015,0.015,0.05). Default: 0.05')
    parser.add_argument('--cell-size', type=float, nargs=2, default=[4, 4],
                        help='Figure size per cell in inches (width height). Default: 4 4')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output DPI (default: 150)')

    args = parser.parse_args()

    # Collect .mat files
    mat_files = []
    for path in args.input:
        if path.is_dir():
            mat_files.extend(sorted(path.glob('*_myelinated_axons.mat')))
        elif path.suffix == '.mat':
            mat_files.append(path)
        else:
            logger.warning(f"Skipping non-.mat file: {path}")

    if not mat_files:
        logger.error("No .mat files found")
        return

    logger.info(f"Found {len(mat_files)} .mat files")

    plot_central_slices_grid(
        mat_files,
        args.output,
        voxel_size_um=args.voxel_size,
        figsize_per_cell=tuple(args.cell_size),
        dpi=args.dpi
    )


if __name__ == '__main__':
    main()
