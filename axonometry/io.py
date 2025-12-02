"""
I/O utilities for loading labeled volumes and handling voxel sizes.

Supports:
- MATLAB .mat files (HDF5 v7.3 and older scipy formats)
- JSON metadata files with voxel size information
- Voxel size parsing (isotropic and anisotropic)
- Resampling anisotropic volumes to isotropic
"""

import json
import logging
from pathlib import Path
from typing import Tuple, Union, Optional, Dict, Any

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


def load_json_metadata(volume_file: Path) -> Optional[Dict[str, Any]]:
    """
    Load companion JSON metadata file for a volume.

    Args:
        volume_file: Path to the volume file (.mat, .h5, etc.)

    Returns:
        Dictionary with metadata, or None if not found
    """
    json_file = volume_file.with_suffix('.json')

    if not json_file.exists():
        return None

    try:
        with open(json_file, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {json_file.name}")
        return metadata
    except Exception as e:
        logger.warning(f"Failed to load metadata from {json_file}: {e}")
        return None


def load_volume_with_metadata(
    volume_file: Union[str, Path],
    voxel_size_override: Optional[Union[float, Tuple[float, float, float]]] = None
) -> Tuple[np.ndarray, Tuple[float, float, float], Optional[Dict[str, Any]]]:
    """
    Load a labeled volume and its metadata (voxel size, etc.) from disk.

    Priority for voxel size:
    1. voxel_size_override (if provided)
    2. Companion JSON metadata file
    3. Default (0.05 μm isotropic)

    Args:
        volume_file: Path to volume file (.mat, .h5, etc.)
        voxel_size_override: Optional voxel size to override metadata/default

    Returns:
        Tuple of (volume, voxel_size_tuple, metadata_dict)
        - volume: 3D numpy array of labels
        - voxel_size_tuple: (vz, vy, vx) in micrometers
        - metadata_dict: Full metadata dictionary, or None if not found
    """
    volume_file = Path(volume_file)

    # Load volume
    logger.info(f"Loading volume from {volume_file.name}")
    volume = load_mat_volume(volume_file)

    # Load metadata
    metadata = load_json_metadata(volume_file)

    # Determine voxel size with priority: override > JSON > default
    voxel_size: Union[float, Tuple[float, float, float]]

    if voxel_size_override is not None:
        voxel_size = voxel_size_override
        logger.info(f"Using voxel size from override: {voxel_size}")
    elif metadata is not None and 'voxel_size' in metadata:
        vs = metadata['voxel_size']
        if isinstance(vs, list) and len(vs) == 3:
            # Check if isotropic
            if vs[0] == vs[1] == vs[2]:
                voxel_size = float(vs[0])
            else:
                voxel_size = tuple(vs)
        else:
            voxel_size = float(vs)
        logger.info(f"Using voxel size from JSON metadata: {voxel_size}")
    else:
        voxel_size = 0.05
        logger.info(f"No voxel size found, using default: {voxel_size} μm")

    # Parse to tuple
    voxel_size_tuple = parse_voxel_size(voxel_size)

    return volume, voxel_size_tuple, metadata


def construct_output_path(
    input_file: Path,
    output_root: Path,
    output_suffix: str = "",
    organize_by_microscopy: bool = False
) -> Path:
    """
    Construct output path for batch processing with proper filename extraction.

    When organize_by_microscopy is True:
    - Extracts condition (Sham/TBI) from parent directory name (converted to lowercase)
    - Extracts microscopy type (HM/LM) from filename prefix
    - Organizes output into microscopy type subdirectories
    - Removes "_myelinated_axons" suffix from filename (redundant)
    - Constructs filename as: {condition}_{ratid}_{hemisphere}{suffix}.npz (all lowercase)

    Example:
        Input: data/raw/Sham_25_ipsi/HM_25_ipsi_myelinated_axons.mat
        Output (organize_by_microscopy=True, suffix='_axon_profiles'):
               data/processed/HM/sham_25_ipsi_axon_profiles.npz
        Output (organize_by_microscopy=False):
               data/processed/Sham_25_ipsi/HM_25_ipsi_myelinated_axons_axon_profiles.npz

    Args:
        input_file: Input .mat file path
        output_root: Output root directory
        output_suffix: Suffix to append before .npz extension (e.g., '_axon_profiles')
        organize_by_microscopy: If True, organize by microscopy type (HM/LM) subdirectories

    Returns:
        Full output path for the processed file
    """
    filename = input_file.stem  # e.g., "HM_25_ipsi_myelinated_axons"
    parent_name = input_file.parent.name  # e.g., "Sham_25_ipsi"

    if not organize_by_microscopy:
        # Standard mode: preserve directory structure
        output_dir = output_root / parent_name
        output_filename = filename + output_suffix + '.npz'
    else:
        # Microscopy organization mode
        # Extract microscopy type from filename prefix
        if filename.startswith('HM_'):
            microscopy_type = 'HM'
            # Remove HM_ prefix: "HM_25_ipsi_myelinated_axons" -> "25_ipsi_myelinated_axons"
            filename_without_microscopy = filename[3:]
        elif filename.startswith('LM_'):
            microscopy_type = 'LM'
            # Remove LM_ prefix: "LM_25_ipsi_myelinated_axons" -> "25_ipsi_myelinated_axons"
            filename_without_microscopy = filename[3:]
        else:
            # No recognized prefix - fallback to standard mode
            logger.warning(f"Filename '{filename}' does not start with HM_ or LM_. "
                         f"Using standard directory structure.")
            output_dir = output_root / parent_name
            output_filename = filename + output_suffix + '.npz'
            return output_dir / output_filename

        # Remove "_myelinated_axons" suffix if present
        # "25_ipsi_myelinated_axons" -> "25_ipsi"
        if filename_without_microscopy.endswith('_myelinated_axons'):
            filename_base = filename_without_microscopy[:-len('_myelinated_axons')]
        else:
            filename_base = filename_without_microscopy

        # Extract condition from parent directory name
        # Parent name format: "{Condition}_{RatID}_{Hemisphere}"
        # e.g., "Sham_25_ipsi" -> condition = "sham"
        # e.g., "TBI_49_contra" -> condition = "tbi"
        parent_parts = parent_name.split('_')
        if len(parent_parts) >= 1:
            condition = parent_parts[0].lower()  # "sham" or "tbi" (lowercase)
        else:
            # Fallback if parent doesn't match expected pattern
            logger.warning(f"Parent directory '{parent_name}' doesn't match expected format. "
                         f"Cannot extract condition.")
            condition = ""

        # Construct output directory: output_root / microscopy_type
        output_dir = output_root / microscopy_type

        # Construct filename: {condition}_{ratid}_{hemisphere}{suffix}.npz
        # Example: "sham_25_ipsi_axon_profiles.npz" or "tbi_24_ipsi_slice_profiles.npz"
        if condition:
            output_filename = f"{condition}_{filename_base}{output_suffix}.npz"
        else:
            output_filename = f"{filename_base}{output_suffix}.npz"

    return output_dir / output_filename


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
