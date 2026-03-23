"""
I/O utilities for loading and writing labeled volumes.

Supports:
- MATLAB .mat files (HDF5 v7.3 and older scipy formats)
- OME-Zarr with multi-resolution pyramids
- JSON metadata files with voxel size information
- Voxel size parsing (isotropic and anisotropic)
- Resampling anisotropic volumes to isotropic
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union, Optional, Any

import numpy as np
import h5py
import scipy.io as sio
import zarr
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


def load_volume_downsampled(mat_file, downsample: int = 4) -> Tuple[np.ndarray, Dict]:
    """
    Load labeled volume with optional downsampling.

    Args:
        mat_file: Path to .mat file
        downsample: Downsampling factor (default: 4)

    Returns:
        Tuple of (volume, metadata)
    """
    logger.info(f"Loading volume: {mat_file.name}")

    volume_full, _, _ = load_volume_with_metadata(mat_file, voxel_size_override=None)
    logger.info(f"Original shape: {volume_full.shape}")

    if downsample > 1:
        volume = volume_full[::downsample, ::downsample, ::downsample].copy()
        logger.info(f"Downsampled to: {volume.shape} (factor {downsample})")
    else:
        volume = volume_full.copy()

    metadata = {
        'original_shape': list(volume_full.shape),
        'downsampled_shape': list(volume.shape),
        'downsample_factor': downsample,
        'source_file': str(mat_file)
    }

    return volume, metadata


def precompute_axon_voxels(volume: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Pre-compute voxel coordinates for all axons.

    Args:
        volume: Labeled volume array

    Returns:
        Dictionary mapping axon label to voxel coordinates
    """
    logger.info("Pre-computing voxel coordinates for all axons...")
    import time
    start = time.time()

    all_coords = np.argwhere(volume > 0)
    axon_voxels = {}
    for coord in all_coords:
        axon_id = volume[tuple(coord)]
        if axon_id not in axon_voxels:
            axon_voxels[axon_id] = []
        axon_voxels[axon_id].append(coord)

    axon_voxels = {aid: np.array(coords) for aid, coords in axon_voxels.items()}
    logger.info(f"Pre-computed coordinates for {len(axon_voxels)} axons in {time.time() - start:.2f}s")

    return axon_voxels


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


# ===========================================================================
# OME-Zarr I/O
# ===========================================================================

def load_zarr_volume(zarr_path: Union[str, Path]) -> Tuple[np.ndarray, float]:
    """Load level-0 volume and isotropic voxel size from an OME-Zarr store.

    Args:
        zarr_path: Path to .zarr directory

    Returns:
        (volume, voxel_size_um) where volume is the full-resolution array
        and voxel_size_um is the Z voxel size in micrometers.
    """
    store = zarr.open_group(str(zarr_path), mode="r")
    volume = np.asarray(store["0"])

    multiscales = store.attrs["multiscales"]
    scale = multiscales[0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    voxel_size_z = scale[0]

    if not np.allclose(scale, scale[0]):
        logger.warning(
            f"Zarr voxel size is not isotropic: {scale}. Using Z voxel size ({voxel_size_z})."
        )

    return volume, float(voxel_size_z)


def downsample_nearest(volume: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Downsample a 3D volume by striding (nearest-neighbor).

    Suitable for label/segmentation data where averaging is not meaningful.

    Args:
        volume: 3D array (Z, Y, X)
        factor: Downsampling factor (applied to all axes)

    Returns:
        Downsampled array
    """
    return volume[::factor, ::factor, ::factor].copy()


def downsample_mean(volume: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Downsample a 3D volume using local mean (box filter).

    Suitable for grayscale/intensity data.

    Args:
        volume: 3D array (Z, Y, X)
        factor: Downsampling factor (applied to all axes)

    Returns:
        Downsampled array (preserves dtype for integer types via rounding)
    """
    from skimage.transform import downscale_local_mean

    result = downscale_local_mean(volume, (factor, factor, factor))
    if np.issubdtype(volume.dtype, np.integer):
        return np.round(result).astype(volume.dtype)
    return result.astype(volume.dtype)


def compute_pyramid_shapes(
    base_shape: Tuple[int, ...],
    num_levels: int,
    factor: int = 2,
) -> List[Tuple[int, ...]]:
    """
    Compute shapes for all pyramid levels.

    Args:
        base_shape: Shape of level 0 (Z, Y, X)
        num_levels: Number of pyramid levels
        factor: Downsampling factor per level

    Returns:
        List of shapes, one per level
    """
    shapes = [base_shape]
    for _ in range(1, num_levels):
        prev = shapes[-1]
        shapes.append(tuple(max(1, s // factor) for s in prev))
    return shapes


def write_ome_zarr_pyramid(
    data: np.ndarray,
    output_path: Union[str, Path],
    voxel_size_um: Union[float, Tuple[float, float, float]] = 0.05,
    num_levels: int = 4,
    downsample_mode: str = 'nearest',
    chunk_size: int = 64,
    compression_level: int = 3,
    metadata: Optional[Dict] = None,
) -> Path:
    """
    Write a 3D volume as OME-Zarr with multi-resolution pyramid.

    Creates an OME-NGFF v0.4 compliant Zarr store with pyramid levels
    generated by successive downsampling.

    Args:
        data: 3D array (Z, Y, X) to write
        output_path: Path for output .zarr directory
        voxel_size_um: Voxel size in micrometers (scalar or (vz, vy, vx))
        num_levels: Number of pyramid levels (1=full only, 4=full+2x+4x+8x)
        downsample_mode: 'nearest' for labels, 'mean' for grayscale
        chunk_size: Target chunk size per dimension
        compression_level: Zstd compression level (0=none)
        metadata: Optional extra metadata to store in root attrs

    Returns:
        Path to created .zarr directory
    """
    import numcodecs

    output_path = Path(output_path)

    if downsample_mode not in ('nearest', 'mean'):
        raise ValueError(f"downsample_mode must be 'nearest' or 'mean', got {downsample_mode}")
    if num_levels < 1:
        raise ValueError(f"num_levels must be >= 1, got {num_levels}")

    # Parse voxel size
    if isinstance(voxel_size_um, (int, float)):
        voxel_size = (float(voxel_size_um),) * 3
    else:
        voxel_size = tuple(float(v) for v in voxel_size_um)

    # Select downsampler
    downsample_fn = downsample_nearest if downsample_mode == 'nearest' else downsample_mean

    # Create zarr v2 group (Neuroglancer compatible)
    root = zarr.open_group(str(output_path), mode='w', zarr_format=2)

    # Set up compression
    if compression_level > 0:
        compressor = numcodecs.Blosc(cname='zstd', clevel=compression_level, shuffle=numcodecs.Blosc.BITSHUFFLE)
    else:
        compressor = None

    # Build pyramid and write each level
    pyramid_shapes = compute_pyramid_shapes(data.shape, num_levels)
    current = data

    for level in range(num_levels):
        if level > 0:
            current = downsample_fn(current)

        level_chunks = tuple(min(chunk_size, s) for s in current.shape)

        arr = root.create_array(
            name=str(level),
            shape=current.shape,
            chunks=level_chunks,
            dtype=current.dtype,
            compressor=compressor,
            overwrite=True,
        )
        arr[:] = current

        logger.info(f"  Level {level}: {current.shape}, "
                     f"{current.nbytes / 1024**2:.1f} MB")

    # Write OME-NGFF v0.4 metadata
    datasets = []
    for level in range(num_levels):
        scale_factor = 2 ** level
        datasets.append({
            "path": str(level),
            "coordinateTransformations": [{
                "type": "scale",
                "scale": [v * scale_factor for v in voxel_size],
            }],
        })

    ome_metadata = {
        "version": "0.4",
        "name": output_path.stem,
        "axes": [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": datasets,
        "type": downsample_mode,
        "metadata": {
            "original_shape": list(data.shape),
            **(metadata or {}),
        },
    }

    root.attrs["multiscales"] = [ome_metadata]

    logger.info(f"  Wrote OME-Zarr ({num_levels} levels) to {output_path}")

    return output_path


# ── Profile loading ───────────────────────────────────────────────────────

MASKS_DIR = Path('data/masks')
ENDPOINT_TRIM_POINTS = 20    # Points to trim from each segment end
MIN_SEGMENT_LENGTH_UM = 0.0  # Minimum segment length in μm


def find_excluded_labels(npz_path: Path) -> Set[int]:
    """Auto-discover excluded labels file for a given NPZ.

    Maps NPZ path to exclusion file by replacing the data/processed prefix
    with data/masks and stripping suffixes like _axon_profiles, _slice_profiles,
    _myelin.

    Example:
        data/processed/rat/lm/sham_25_ipsi_cc_myelin_axon_profiles.npz
        -> data/masks/rat/lm/sham_25_ipsi_cc_excluded_labels.json
    """
    stem = npz_path.stem
    for suffix in ('_axon_profiles', '_slice_profiles', '_myelin'):
        stem = stem.replace(suffix, '')
    try:
        rel = npz_path.parent.relative_to(Path('data/processed'))
    except ValueError:
        return set()
    excl_path = MASKS_DIR / rel / f"{stem}_excluded_labels.json"
    if excl_path.exists():
        with open(excl_path) as f:
            labels = json.load(f)
        logger.info(f"Excluding {len(labels)} labels from {excl_path.name}")
        return set(labels)
    return set()


def load_3d_profiles(npz_path: Union[str, Path],
                     endpoint_trim: int = ENDPOINT_TRIM_POINTS,
                     min_segment_length_um: float = MIN_SEGMENT_LENGTH_UM,
                     exclude_labels: bool = True) -> dict:
    """
    Load 3D axon profile data and apply quality filters.

    Args:
        npz_path: Path to axon_profiles.npz file
        endpoint_trim: Number of points to trim from each segment end
        min_segment_length_um: Minimum segment length in μm to include
        exclude_labels: If True, auto-discover and exclude labels from
                        corresponding file in data/masks/

    Returns:
        dict with:
            labels: (N,) axon label IDs
            segment_radii_um: per-axon list of per-segment radius arrays
            segment_lengths_um: per-axon list of per-segment lengths
            all_radii_um: pooled radii from all filtered segments
            n_segments: (N,) number of segments per axon after filtering
            voxel_size_um: scalar
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    labels = data['labels']
    raw_seg_radii = data['segment_radii_um']
    raw_seg_lengths = data['segment_lengths_um']
    voxel_size = float(data['voxel_size_um'])

    excluded = find_excluded_labels(npz_path) if exclude_labels else set()

    filtered_seg_radii = []
    filtered_seg_lengths = []
    all_radii = []
    kept_labels = []

    for i in range(len(labels)):
        if int(labels[i]) in excluded:
            continue
        segs_r = raw_seg_radii[i]
        segs_l = raw_seg_lengths[i]
        axon_segs_r = []
        axon_segs_l = []

        for j in range(len(segs_r)):
            r = np.asarray(segs_r[j])
            seg_len = float(segs_l[j])

            if seg_len < min_segment_length_um:
                continue

            if endpoint_trim > 0 and len(r) > 2 * endpoint_trim:
                r = r[endpoint_trim:-endpoint_trim]

            if len(r) == 0:
                continue

            axon_segs_r.append(r)
            axon_segs_l.append(seg_len)
            all_radii.append(r)

        filtered_seg_radii.append(axon_segs_r)
        filtered_seg_lengths.append(axon_segs_l)
        kept_labels.append(labels[i])

    all_radii = np.concatenate(all_radii).astype(np.float64) if all_radii else np.array([], dtype=np.float64)

    return {
        'labels': np.array(kept_labels),
        'segment_radii_um': filtered_seg_radii,
        'segment_lengths_um': filtered_seg_lengths,
        'all_radii_um': all_radii,
        'n_segments': np.array([len(s) for s in filtered_seg_radii]),
        'voxel_size_um': voxel_size,
    }


def load_2d_profiles(npz_path: Union[str, Path],
                     radius_type: str = 'circular',
                     min_radius_um: float = 0.0,
                     max_eccentricity: float = 1.0,
                     min_solidity: float = 0.5,
                     exclude_labels: bool = True) -> dict:
    """
    Load 2D slice profiles and apply quality filters.

    Args:
        npz_path: Path to slice_profiles.npz file
        radius_type: Which radius measure ('circular' or 'minor')
        min_radius_um: Minimum radius filter
        max_eccentricity: Maximum eccentricity filter
        min_solidity: Minimum solidity filter
        exclude_labels: If True, auto-discover and exclude labels

    Returns:
        dict with:
            radii: filtered radius array
            slice_index: corresponding slice indices
            n_slices: total number of slices
    """
    npz_path = Path(npz_path)
    d = np.load(npz_path)

    radius_key = f'radius_{radius_type}_um'
    radii = d[radius_key]
    slices = d['slice_index']
    ecc = d['eccentricity']
    sol = d['solidity']
    labels = d['label']

    excluded = find_excluded_labels(npz_path) if exclude_labels else set()

    mask = (
        (radii >= min_radius_um) &
        (ecc <= max_eccentricity) &
        (sol >= min_solidity)
    )
    if excluded:
        label_mask = np.isin(labels, list(excluded), invert=True)
        mask &= label_mask

    radii = radii[mask]
    slices = slices[mask]
    n_slices = int(d['n_slices'])

    logger.info(f"  {npz_path.stem}: {mask.sum():,}/{len(mask):,} instances after filter "
                f"({mask.sum()/len(mask)*100:.1f}%)")

    return {
        'radii': radii,
        'slice_index': slices,
        'n_slices': n_slices,
    }
