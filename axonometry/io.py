"""
I/O utilities for loading labeled volumes and handling voxel sizes.

Supports:
- MATLAB .mat files (HDF5 v7.3 and older scipy formats)
- Voxel size parsing (isotropic and anisotropic)
- Resampling anisotropic volumes to isotropic
"""

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import h5py
import scipy.io as sio
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)


def parse_voxel_size(voxel_size_um: Union[float, Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """
    Parse voxel size to a (vz, vy, vx) tuple matching array axis order (Z, Y, X).

    Args:
        voxel_size_um: Scalar (isotropic) or (vz, vy, vx) tuple (anisotropic)

    Returns:
        Tuple of (vz, vy, vx) in micrometers, matching array axes (Z, Y, X)
    """
    if isinstance(voxel_size_um, (tuple, list)):
        if len(voxel_size_um) == 3:
            return tuple(float(v) for v in voxel_size_um)
        elif len(voxel_size_um) == 1:
            v = float(voxel_size_um[0])
            return (v, v, v)
        else:
            raise ValueError(f"Expected 1 or 3 voxel size values, got {len(voxel_size_um)}")
    # Scalar - isotropic
    v = float(voxel_size_um)
    return (v, v, v)


def load_mat_volume(mat_file: Union[str, Path]) -> np.ndarray:
    """
    Load a labeled volume from a MATLAB .mat file.

    Supports both HDF5 (MATLAB v7.3) and older scipy formats.

    Args:
        mat_file: Path to .mat file

    Returns:
        3D numpy array of labels
    """
    mat_file = Path(mat_file)

    if not mat_file.exists():
        raise FileNotFoundError(f"Input file not found: {mat_file}")

    # Try HDF5 format first (MATLAB v7.3)
    try:
        with h5py.File(str(mat_file), 'r') as f:
            # Find the labeled volume key
            volume_key = None
            for key in f.keys():
                if not key.startswith('#') and not key.startswith('_'):
                    volume_key = key
                    break

            if volume_key is None:
                raise ValueError(f"No data found in {mat_file}")

            volume = f[volume_key][:]
            logger.info(f"Loaded HDF5 format, volume shape: {volume.shape}, dtype: {volume.dtype}")
            return volume

    except OSError as e:
        # Try scipy.io for older MATLAB formats (v5/v6/v7)
        logger.info(f"HDF5 failed ({e}), trying scipy.io for older MATLAB format...")
        mat_data = sio.loadmat(str(mat_file))

        # Find the labeled volume key (skip MATLAB metadata keys)
        volume_key = None
        for key in mat_data.keys():
            if not key.startswith('__'):
                volume_key = key
                break

        if volume_key is None:
            raise ValueError(f"No data found in {mat_file}")

        volume = mat_data[volume_key]
        logger.info(f"Loaded scipy.io format, volume shape: {volume.shape}, dtype: {volume.dtype}")
        return volume


def resample_to_isotropic(volume: np.ndarray,
                          voxel_size: Tuple[float, float, float]
                          ) -> Tuple[np.ndarray, float]:
    """
    Resample anisotropic volume to isotropic voxels.

    Downsamples to the coarsest voxel dimension to avoid upsampling artifacts.
    Uses nearest-neighbor interpolation (order=0) to preserve label integrity.

    Args:
        volume: 3D labeled volume with shape (Z, Y, X)
        voxel_size: Tuple of (vz, vy, vx) voxel sizes in micrometers, matching array axes

    Returns:
        Tuple of (resampled_volume, isotropic_voxel_size)
    """
    vz, vy, vx = voxel_size

    # Check if already isotropic
    if np.allclose([vz, vy, vx], vz):
        logger.info("Volume is already isotropic, no resampling needed")
        return volume, vz

    # Target voxel size is the coarsest dimension
    target_size = max(vz, vy, vx)

    # Compute zoom factors (< 1 means downsampling)
    # Volume axes are (Z, Y, X), voxel_size is (vz, vy, vx)
    zoom_factors = (vz / target_size, vy / target_size, vx / target_size)

    logger.info(f"Resampling anisotropic volume to isotropic:")
    logger.info(f"  Original voxel size (Z, Y, X): vz={vz:.4f}, vy={vy:.4f}, vx={vx:.4f} μm")
    logger.info(f"  Target voxel size: {target_size:.4f} μm (isotropic)")
    logger.info(f"  Zoom factors (Z, Y, X): {zoom_factors}")
    logger.info(f"  Original shape: {volume.shape}")

    # Use order=0 (nearest-neighbor) to preserve label integrity
    resampled = zoom(volume, zoom_factors, order=0, mode='nearest')

    logger.info(f"  Resampled shape: {resampled.shape}")

    return resampled, target_size
