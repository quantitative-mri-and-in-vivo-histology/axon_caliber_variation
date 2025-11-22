#!/usr/bin/env python3
"""
Extract orthogonal cross-sections perpendicular to fiber direction.

This script extracts true orthogonal slices from 3D labeled axon volumes,
using the mean fiber orientation from bundle metadata. Unlike preprocess_1b
which only permutes axes, this performs proper oblique sampling to get
circular cross-sections regardless of fiber orientation.

Input: Original .mat file + bundle metadata JSON (from preprocess_1)
Output: OME-Zarr with pyramids, one per bundle

Slices are extracted at 0.05 μm intervals along the fiber direction,
with perpendicular extent auto-fitted to bundle bounding box.
"""

import gc
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple

import h5py
import numpy as np
import zarr
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import numcodecs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Helper functions (adapted from preprocess_1b)
# =============================================================================

def create_label_lut(axon_labels: Set[int], max_label: int) -> np.ndarray:
    """
    Create a lookup table for fast label filtering.

    Args:
        axon_labels: Set of axon labels to keep
        max_label: Maximum label value in volume

    Returns:
        Boolean LUT where lut[label] = True if label is in axon_labels
    """
    lut = np.zeros(max_label + 1, dtype=bool)
    for label in axon_labels:
        if label <= max_label:
            lut[label] = True
    return lut


def downsample_segmentation_mode_fast(data: np.ndarray, factor: int) -> np.ndarray:
    """
    Fast mode-based downsampling for segmentation labels.

    Takes center voxel of each block (fast approximation of mode).

    Args:
        data: Input 3D segmentation volume
        factor: Downsampling factor

    Returns:
        Downsampled volume
    """
    if factor == 1:
        return data.copy()

    offset = factor // 2
    return data[offset::factor, offset::factor, offset::factor].copy()


# =============================================================================
# Core geometry functions
# =============================================================================

def compute_orthonormal_basis(fiber_direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute orthonormal basis perpendicular to fiber direction using Gram-Schmidt.

    Args:
        fiber_direction: [vz, vy, vx] normalized direction vector

    Returns:
        (v1, v2): Two orthonormal vectors perpendicular to fiber direction
    """
    fiber_direction = fiber_direction / np.linalg.norm(fiber_direction)

    # Choose initial vector not parallel to fiber direction
    if abs(fiber_direction[2]) < 0.9:  # Not parallel to x-axis
        u = np.array([0.0, 0.0, 1.0])
    else:
        u = np.array([0.0, 1.0, 0.0])

    # Gram-Schmidt orthogonalization
    v1 = u - np.dot(u, fiber_direction) * fiber_direction
    v1 = v1 / np.linalg.norm(v1)

    # Second perpendicular vector via cross product
    v2 = np.cross(fiber_direction, v1)
    v2 = v2 / np.linalg.norm(v2)

    return v1, v2


def compute_bundle_bounding_box(volume: np.ndarray,
                                 axon_labels: List[int],
                                 fiber_direction: np.ndarray,
                                 v1: np.ndarray,
                                 v2: np.ndarray) -> Dict:
    """
    Compute bounding box of bundle voxels in fiber-local coordinates.

    Uses memory-efficient approach: iterates slice-by-slice instead of
    materializing all voxel coordinates at once.

    Args:
        volume: 3D labeled volume
        axon_labels: List of axon labels in this bundle
        fiber_direction: Along-fiber direction vector
        v1, v2: Perpendicular basis vectors

    Returns:
        Dictionary with bounding box info in fiber-local coordinates
    """
    logger.info("Computing bundle bounding box (memory-efficient)...")

    # Create LUT for fast filtering
    max_label = int(volume.max())
    lut = create_label_lut(set(axon_labels), max_label)

    # Compute volume center
    volume_center = np.array([
        volume.shape[0] / 2,
        volume.shape[1] / 2,
        volume.shape[2] / 2
    ])

    # Initialize bounds
    fiber_min = np.inf
    fiber_max = -np.inf
    u_min = np.inf
    u_max = -np.inf
    v_min = np.inf
    v_max = -np.inf
    total_voxels = 0

    # Process slice by slice to avoid memory explosion
    for z in tqdm(range(volume.shape[0]), desc="Computing bounding box"):
        slice_2d = volume[z, :, :]
        mask = lut[np.minimum(slice_2d, max_label)]

        if not mask.any():
            continue

        # Get coordinates in this slice
        yx_coords = np.argwhere(mask)  # [N, 2] in (y, x)
        n_voxels = len(yx_coords)
        total_voxels += n_voxels

        # Build 3D coordinates for this slice
        z_coords = np.full(n_voxels, z)
        coords_3d = np.column_stack([z_coords, yx_coords])  # [N, 3] in (z, y, x)

        # Transform to fiber-local coordinates
        centered = coords_3d - volume_center

        # Project onto each axis
        fiber_pos = centered @ fiber_direction
        u_pos = centered @ v1
        v_pos = centered @ v2

        # Update bounds
        fiber_min = min(fiber_min, fiber_pos.min())
        fiber_max = max(fiber_max, fiber_pos.max())
        u_min = min(u_min, u_pos.min())
        u_max = max(u_max, u_pos.max())
        v_min = min(v_min, v_pos.min())
        v_max = max(v_max, v_pos.max())

    if total_voxels == 0:
        raise ValueError("No voxels found for bundle")

    logger.info(f"Found {total_voxels:,} voxels in bundle")

    bbox = {
        'fiber_min': float(fiber_min),
        'fiber_max': float(fiber_max),
        'u_min': float(u_min),
        'u_max': float(u_max),
        'v_min': float(v_min),
        'v_max': float(v_max),
        'volume_center': volume_center,
        'n_voxels': total_voxels
    }

    fiber_extent = bbox['fiber_max'] - bbox['fiber_min']
    u_extent = bbox['u_max'] - bbox['u_min']
    v_extent = bbox['v_max'] - bbox['v_min']

    logger.info(f"Bounding box (voxels):")
    logger.info(f"  Along fiber: {fiber_extent:.0f}")
    logger.info(f"  Perpendicular: {u_extent:.0f} x {v_extent:.0f}")

    return bbox


def compute_slice_positions(bbox: Dict, voxel_size_um: float) -> np.ndarray:
    """
    Compute positions along fiber axis where slices will be extracted.

    Args:
        bbox: Bounding box dictionary
        voxel_size_um: Physical voxel size (determines slice spacing)

    Returns:
        Array of positions in voxels from volume center
    """
    # Slice spacing = 1 voxel along fiber axis
    fiber_extent_voxels = bbox['fiber_max'] - bbox['fiber_min']
    n_slices = int(np.ceil(fiber_extent_voxels)) + 1

    positions = np.linspace(bbox['fiber_min'], bbox['fiber_max'], n_slices)

    fiber_extent_um = fiber_extent_voxels * voxel_size_um
    logger.info(f"Will extract {n_slices} slices over {fiber_extent_um:.1f} μm "
                f"(spacing: {voxel_size_um} μm = 1 voxel)")

    return positions


# =============================================================================
# Axon and slice filtering
# =============================================================================

def compute_valid_slice_range(axon_counts: List[int], min_axon_fraction: float) -> Tuple[int, int]:
    """
    Compute valid slice range using symmetric expansion from maximum.

    Finds the slice with max axon count, then expands symmetrically until
    hitting the threshold (fraction of max). Uses the smaller extent.

    Args:
        axon_counts: List of axon counts per slice
        min_axon_fraction: Minimum fraction of max count (e.g., 0.75)

    Returns:
        (start_idx, end_idx) - valid slice range (inclusive start, exclusive end)
    """
    if not axon_counts:
        return 0, 0

    n_slices = len(axon_counts)
    max_count = max(axon_counts)
    if max_count == 0:
        return 0, 0

    threshold = max_count * min_axon_fraction
    max_idx = axon_counts.index(max_count)

    # Expand left
    left_idx = max_idx
    while left_idx > 0 and axon_counts[left_idx - 1] >= threshold:
        left_idx -= 1

    # Expand right
    right_idx = max_idx
    while right_idx < n_slices - 1 and axon_counts[right_idx + 1] >= threshold:
        right_idx += 1

    # Make symmetric (use smaller extent)
    left_extent = max_idx - left_idx
    right_extent = right_idx - max_idx
    min_extent = min(left_extent, right_extent)

    start_idx = max_idx - min_extent
    end_idx = max_idx + min_extent + 1  # exclusive

    return start_idx, end_idx


def filter_axons_by_coverage(axon_slices: Dict[int, Set[int]],
                              valid_slice_range: Tuple[int, int],
                              min_coverage: float) -> Set[int]:
    """
    Filter axons by their coverage across valid slices.

    Args:
        axon_slices: Dict mapping axon_id -> set of slice indices where it appears
        valid_slice_range: (start_idx, end_idx) of valid slices
        min_coverage: Minimum fraction of valid slices axon must appear in (e.g., 0.5)

    Returns:
        Set of valid axon IDs
    """
    start_idx, end_idx = valid_slice_range
    n_valid_slices = end_idx - start_idx

    if n_valid_slices == 0:
        return set()

    valid_axons = set()
    threshold = n_valid_slices * min_coverage

    for axon_id, slice_indices in axon_slices.items():
        # Count how many valid slices this axon appears in
        count = sum(1 for idx in slice_indices if start_idx <= idx < end_idx)
        if count >= threshold:
            valid_axons.add(axon_id)

    return valid_axons


# =============================================================================
# Batched slice extraction (performance-critical)
# =============================================================================

def extract_slices_batched(volume: np.ndarray,
                           slice_positions: np.ndarray,
                           fiber_direction: np.ndarray,
                           v1: np.ndarray,
                           v2: np.ndarray,
                           bbox: Dict,
                           padding_voxels: int = 10,
                           batch_size: int = 100) -> np.ndarray:
    """
    Extract all orthogonal slices using batched map_coordinates.

    This is the performance-critical function. Batching reduces the number
    of map_coordinates calls from N_slices to N_slices/batch_size.

    Args:
        volume: 3D labeled volume
        slice_positions: Positions along fiber axis (voxels from center)
        fiber_direction: Along-fiber direction vector
        v1, v2: Perpendicular basis vectors
        bbox: Bounding box dictionary
        padding_voxels: Extra padding around bounding box
        batch_size: Number of slices per map_coordinates call

    Returns:
        3D array of shape (ny, nx, n_slices) containing all slices
    """
    n_slices = len(slice_positions)
    volume_center = bbox['volume_center']

    # Compute slice dimensions from perpendicular bounding box
    u_min = bbox['u_min'] - padding_voxels
    u_max = bbox['u_max'] + padding_voxels
    v_min = bbox['v_min'] - padding_voxels
    v_max = bbox['v_max'] + padding_voxels

    nx = int(np.ceil(u_max - u_min)) + 1
    ny = int(np.ceil(v_max - v_min)) + 1

    logger.info(f"Slice dimensions: {ny} x {nx} (with {padding_voxels} voxel padding)")
    logger.info(f"Extracting {n_slices} slices in batches of {batch_size}...")

    # Pre-allocate output (y, x, z convention)
    all_slices = np.zeros((ny, nx, n_slices), dtype=volume.dtype)

    # Pre-compute 2D grid offsets (reused for all slices)
    u_coords = np.linspace(u_min, u_max, nx)
    v_coords = np.linspace(v_min, v_max, ny)
    U, V = np.meshgrid(u_coords, v_coords)  # Shape: (ny, nx)

    # Pre-compute displacement from plane center in 3D
    # Shape: (3, ny, nx)
    displacement = (U[np.newaxis, :, :] * v1[:, np.newaxis, np.newaxis] +
                    V[np.newaxis, :, :] * v2[:, np.newaxis, np.newaxis])

    # Process in batches
    for batch_start in tqdm(range(0, n_slices, batch_size), desc="Extracting slices"):
        batch_end = min(batch_start + batch_size, n_slices)
        batch_positions = slice_positions[batch_start:batch_end]
        batch_n = len(batch_positions)

        # Build coordinates for entire batch
        # Shape: (3, batch_n, ny, nx)
        batch_coords = np.zeros((3, batch_n, ny, nx), dtype=np.float32)

        for i, pos in enumerate(batch_positions):
            # Plane origin for this slice
            origin = volume_center + pos * fiber_direction
            batch_coords[:, i, :, :] = origin[:, np.newaxis, np.newaxis] + displacement

        # Flatten to (3, N) for map_coordinates
        coords_flat = batch_coords.reshape(3, -1)

        # Check bounds
        in_bounds = (
            (coords_flat[0] >= 0) & (coords_flat[0] < volume.shape[0]) &
            (coords_flat[1] >= 0) & (coords_flat[1] < volume.shape[1]) &
            (coords_flat[2] >= 0) & (coords_flat[2] < volume.shape[2])
        )

        # Sample with nearest neighbor (preserves integer labels)
        sampled = np.zeros(coords_flat.shape[1], dtype=volume.dtype)
        if in_bounds.any():
            sampled[in_bounds] = map_coordinates(
                volume,
                coords_flat[:, in_bounds],
                order=0,
                mode='constant',
                cval=0
            )

        # Reshape back to (batch_n, ny, nx) then transpose to (ny, nx, batch_n)
        batch_slices = sampled.reshape(batch_n, ny, nx).transpose(1, 2, 0)
        all_slices[:, :, batch_start:batch_end] = batch_slices

    return all_slices


# =============================================================================
# OME-Zarr output
# =============================================================================

def create_ome_ngff_metadata(bundle: Dict,
                              voxel_size_um: float,
                              n_levels: int,
                              n_slices: int,
                              spatial_info: Dict = None) -> Dict:
    """
    Create OME-NGFF compliant metadata for orthogonal slice output.

    Args:
        bundle: Bundle metadata dictionary
        voxel_size_um: Physical voxel size
        n_levels: Number of pyramid levels
        n_slices: Number of slices extracted
        spatial_info: Dictionary with spatial positioning info (origin, bbox, basis vectors)

    Returns:
        Dictionary for .zattrs
    """
    # Compute translation from spatial info
    if spatial_info is not None:
        # Origin of output volume in original [z, y, x] coordinates (voxels)
        origin_voxels = spatial_info['origin_voxels']
        # Convert to physical coordinates [z, y, x] in micrometers
        origin_um_zyx = origin_voxels * voxel_size_um
        # Translation in output axis order [y, x, z]
        translation_um = [float(origin_um_zyx[1]), float(origin_um_zyx[2]), float(origin_um_zyx[0])]
    else:
        translation_um = [0.0, 0.0, 0.0]

    datasets = []
    for level in range(n_levels):
        scale_factor = 2 ** level
        voxel_size = voxel_size_um * scale_factor
        # Translation is in physical units (micrometers), same for all levels
        datasets.append({
            "path": str(level),
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": [voxel_size, voxel_size, voxel_size]
                },
                {
                    "type": "translation",
                    "translation": translation_um
                }
            ]
        })

    metadata = {
        "multiscales": [{
            "version": "0.4",
            "name": f"bundle_{bundle['bundle_id']:02d}_orthogonal",
            "axes": [
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
                {"name": "z", "type": "space", "unit": "micrometer"}
            ],
            "datasets": datasets,
            "type": "nearest"
        }],
        "bundle_id": int(bundle['bundle_id']),
        "n_axons": int(bundle['n_axons']),
        "fiber_direction": [float(x) for x in bundle['mean_orientation']],
        "slice_spacing_um": float(voxel_size_um),
        "n_slices": int(n_slices),
        "sampling_method": "nearest_neighbor",
        "voxel_size_um": float(voxel_size_um),
        "n_levels": n_levels
    }

    # Add spatial reconstruction metadata
    if spatial_info is not None:
        metadata["origin_voxels_zyx"] = [float(x) for x in spatial_info['origin_voxels']]
        metadata["origin_um_zyx"] = [float(x) for x in origin_um_zyx]
        metadata["v1_zyx"] = [float(x) for x in spatial_info['v1']]
        metadata["v2_zyx"] = [float(x) for x in spatial_info['v2']]
        metadata["volume_center_zyx"] = [float(x) for x in spatial_info['volume_center']]
        metadata["bbox"] = {
            "fiber_min": float(spatial_info['bbox']['fiber_min']),
            "fiber_max": float(spatial_info['bbox']['fiber_max']),
            "u_min": float(spatial_info['bbox']['u_min']),
            "u_max": float(spatial_info['bbox']['u_max']),
            "v_min": float(spatial_info['bbox']['v_min']),
            "v_max": float(spatial_info['bbox']['v_max']),
        }

    return metadata


def write_bundle_slices_zarr(output_path: Path,
                              slices: np.ndarray,
                              bundle: Dict,
                              voxel_size_um: float,
                              n_levels: int = 5):
    """
    Write orthogonal slice stack to OME-Zarr with pyramids.

    Args:
        output_path: Path for output Zarr store
        slices: 3D array of shape (ny, nx, n_slices)
        bundle: Bundle metadata dictionary
        voxel_size_um: Physical voxel size
        n_levels: Number of pyramid levels
    """
    output_shape = slices.shape
    n_slices = output_shape[2]

    logger.info(f"Writing Zarr store: {output_path}")
    logger.info(f"Output shape: {output_shape}")

    # Create Zarr group (v2 format for Neuroglancer compatibility)
    root = zarr.open_group(str(output_path), mode='w', zarr_format=2)

    # Compression codec (numcodecs for Zarr v2)
    compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)

    # Level 0: Full resolution with (ny, nx, 1) chunking for slice access
    chunk_shape_0 = (output_shape[0], output_shape[1], 1)
    level_0 = root.create_array(
        '0',
        shape=output_shape,
        chunks=chunk_shape_0,
        dtype=np.uint16,
        compressor=compressor
    )

    # Write level 0 slice by slice
    logger.info(f"Writing level 0 ({n_slices} slices)...")
    segment_id_set = set()

    for z in tqdm(range(n_slices), desc="Writing level 0", unit="slice"):
        slice_2d = slices[:, :, z]
        level_0[:, :, z] = slice_2d.astype(np.uint16)
        segment_id_set.update(np.unique(slice_2d).tolist())

    # Remove background from segment IDs
    segment_id_set.discard(0)
    segment_ids = np.array(sorted(segment_id_set), dtype=np.uint16)
    logger.info(f"Found {len(segment_ids)} unique segments")

    # Generate pyramid levels
    logger.info("Generating pyramid levels...")
    current_data = level_0[:]

    for level in range(1, n_levels):
        logger.info(f"Generating level {level} (2x downsampling)...")

        downsampled = downsample_segmentation_mode_fast(current_data, 2)

        # 3D chunks for visualization
        chunk_size = min(64, min(downsampled.shape))
        chunk_shape = (chunk_size, chunk_size, chunk_size)

        level_ds = root.create_array(
            str(level),
            shape=downsampled.shape,
            chunks=chunk_shape,
            dtype=np.uint16,
            compressor=compressor
        )
        level_ds[:] = downsampled

        logger.info(f"  Level {level} shape: {downsampled.shape}")

        current_data = downsampled

    del current_data

    # Store segment IDs
    labels_group = root.create_group('labels')
    seg_array = labels_group.create_array('segment_ids', shape=segment_ids.shape, dtype=np.uint16)
    seg_array[:] = segment_ids

    # Write metadata
    root.attrs.update(create_ome_ngff_metadata(bundle, voxel_size_um, n_levels, n_slices))

    # Report size
    total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
    logger.info(f"Saved Zarr store: {output_path}")
    logger.info(f"  Total size: {total_size / 1024**2:.1f} MB")


# =============================================================================
# Main pipeline
# =============================================================================

def extract_and_write_orthogonal_slices(volume: np.ndarray,
                                         bundle: Dict,
                                         output_path: Path,
                                         voxel_size_um: float = 0.05,
                                         padding_um: float = 0.5,
                                         n_levels: int = 5,
                                         n_workers: int = 32,
                                         min_axon_fraction: float = 0.75,
                                         min_axon_coverage: float = 0.5):
    """
    Extract orthogonal slices and write directly to Zarr (streaming).

    This version extracts slices in parallel and streams them to Zarr.
    Pyramids are generated separately to allow freeing the original volume first.

    Two-pass extraction with filtering:
    1. First pass: track axon presence per slice
    2. Filter: compute valid slice range and valid axons
    3. Second pass: write filtered data

    Args:
        volume: Full 3D labeled volume
        bundle: Bundle metadata dictionary
        output_path: Output Zarr path
        voxel_size_um: Physical voxel size
        padding_um: Padding around bounding box in micrometers
        n_levels: Number of pyramid levels (stored in metadata, pyramids generated separately)
        n_workers: Number of parallel workers for extraction
        min_axon_fraction: Minimum axon count fraction for valid slices (default 0.75)
        min_axon_coverage: Minimum fraction of valid slices axon must appear in (default 0.5)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing bundle {bundle['bundle_id']}")
    logger.info(f"{'='*60}")
    logger.info(f"Axons: {bundle['n_axons']}")
    logger.info(f"Fiber direction: {bundle['mean_orientation']}")

    # Compute orthonormal basis
    fiber_direction = np.array(bundle['mean_orientation'])
    fiber_direction = fiber_direction / np.linalg.norm(fiber_direction)
    v1, v2 = compute_orthonormal_basis(fiber_direction)

    # Compute bounding box
    bbox = compute_bundle_bounding_box(
        volume,
        bundle['axon_labels'],
        fiber_direction,
        v1, v2
    )

    # Compute slice positions
    slice_positions = compute_slice_positions(bbox, voxel_size_um)
    n_slices = len(slice_positions)

    # Compute slice dimensions
    padding_voxels = int(padding_um / voxel_size_um)
    u_min = bbox['u_min'] - padding_voxels
    u_max = bbox['u_max'] + padding_voxels
    v_min = bbox['v_min'] - padding_voxels
    v_max = bbox['v_max'] + padding_voxels

    nx = int(np.ceil(u_max - u_min)) + 1
    ny = int(np.ceil(v_max - v_min)) + 1

    logger.info(f"Output dimensions: {ny} x {nx} x {n_slices}")
    output_size_gb = ny * nx * n_slices * 2 / (1024**3)  # uint16 = 2 bytes
    logger.info(f"Output size: {output_size_gb:.2f} GB")

    # Create Zarr store and level 0 (v2 format for Neuroglancer compatibility)
    logger.info(f"Creating Zarr store: {output_path}")
    root = zarr.open_group(str(output_path), mode='w', zarr_format=2)
    compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)

    output_shape = (ny, nx, n_slices)
    chunk_shape_0 = (ny, nx, 1)
    level_0 = root.create_array(
        '0',
        shape=output_shape,
        chunks=chunk_shape_0,
        dtype=np.uint16,
        compressor=compressor
    )

    # Pre-compute 2D grid (reused for all slices)
    volume_center = bbox['volume_center']
    u_coords = np.linspace(u_min, u_max, nx)
    v_coords = np.linspace(v_min, v_max, ny)
    U, V = np.meshgrid(u_coords, v_coords)

    # Pre-compute displacement from plane center
    displacement = (U[np.newaxis, :, :] * v1[:, np.newaxis, np.newaxis] +
                    V[np.newaxis, :, :] * v2[:, np.newaxis, np.newaxis])

    # Create LUT to filter only bundle axons (zero out non-bundle labels)
    bundle_labels = set(bundle['axon_labels'])
    max_label = int(volume.max())
    label_lut = np.zeros(max_label + 1, dtype=np.uint16)
    for label in bundle_labels:
        if label <= max_label:
            label_lut[label] = label  # Keep bundle labels, others stay 0

    logger.info(f"Filtering to {len(bundle_labels)} bundle axons")

    # =========================================================================
    # FIRST PASS: Track axon presence per slice (no writing)
    # =========================================================================
    import threading

    # Track which axons appear in which slices
    axon_slices: Dict[int, Set[int]] = {label: set() for label in bundle_labels}
    axon_counts = [0] * n_slices
    axon_lock = threading.Lock()

    def extract_and_track_slice(slice_idx: int) -> None:
        """Extract slice and track axon presence (no writing)."""
        pos = slice_positions[slice_idx]
        origin = volume_center + pos * fiber_direction

        # Build coordinates for this slice
        coords = origin[:, np.newaxis, np.newaxis] + displacement

        # Flatten for map_coordinates
        coords_flat = coords.reshape(3, -1)

        # Check bounds
        in_bounds = (
            (coords_flat[0] >= 0) & (coords_flat[0] < volume.shape[0]) &
            (coords_flat[1] >= 0) & (coords_flat[1] < volume.shape[1]) &
            (coords_flat[2] >= 0) & (coords_flat[2] < volume.shape[2])
        )

        # Sample with nearest neighbor
        sampled = np.zeros(coords_flat.shape[1], dtype=volume.dtype)
        if in_bounds.any():
            sampled[in_bounds] = map_coordinates(
                volume,
                coords_flat[:, in_bounds],
                order=0,
                mode='constant',
                cval=0
            )

        # Reshape to 2D and filter to bundle
        slice_2d = sampled.reshape(ny, nx)
        slice_2d = label_lut[slice_2d]

        # Track unique axons in this slice
        unique_ids = set(np.unique(slice_2d).tolist())
        unique_ids.discard(0)

        with axon_lock:
            axon_counts[slice_idx] = len(unique_ids)
            for axon_id in unique_ids:
                if axon_id in axon_slices:
                    axon_slices[axon_id].add(slice_idx)

    # Run first pass
    actual_workers = min(n_workers, n_slices)
    logger.info(f"Pass 1: Tracking axon presence in {n_slices} slices...")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(extract_and_track_slice, i) for i in range(n_slices)]
        for future in tqdm(as_completed(futures), total=n_slices, desc="Pass 1 (tracking)"):
            future.result()

    # =========================================================================
    # FILTERING: Compute valid slice range and valid axons
    # =========================================================================
    logger.info(f"Computing filtering criteria...")

    # Compute valid slice range
    valid_start, valid_end = compute_valid_slice_range(axon_counts, min_axon_fraction)
    n_valid_slices = valid_end - valid_start

    max_count = max(axon_counts)
    max_idx = axon_counts.index(max_count)
    logger.info(f"  Max axon count: {max_count} at slice {max_idx}")
    logger.info(f"  Valid slice range: {valid_start}-{valid_end-1} ({n_valid_slices} slices)")
    logger.info(f"  Threshold: {min_axon_fraction * 100:.0f}% of max = {int(max_count * min_axon_fraction)} axons")

    # Filter axons by coverage
    valid_axons = filter_axons_by_coverage(axon_slices, (valid_start, valid_end), min_axon_coverage)
    n_rejected_axons = len(bundle_labels) - len(valid_axons)
    logger.info(f"  Valid axons: {len(valid_axons)} (rejected {n_rejected_axons})")
    logger.info(f"  Coverage threshold: {min_axon_coverage * 100:.0f}% of {n_valid_slices} slices = {int(n_valid_slices * min_axon_coverage)} slices")

    # Create filtered LUT
    filtered_lut = np.zeros(max_label + 1, dtype=np.uint16)
    for label in valid_axons:
        if label <= max_label:
            filtered_lut[label] = label

    # Update output shape for valid slices only
    output_shape = (ny, nx, n_valid_slices)

    # Update slice positions for valid range
    valid_slice_positions = slice_positions[valid_start:valid_end]

    # =========================================================================
    # SECOND PASS: Write filtered slices
    # =========================================================================
    logger.info(f"Creating Zarr store: {output_path}")
    logger.info(f"Output shape: {output_shape}")

    root = zarr.open_group(str(output_path), mode='w', zarr_format=2)
    compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)

    chunk_shape_0 = (ny, nx, 1)
    level_0 = root.create_array(
        '0',
        shape=output_shape,
        chunks=chunk_shape_0,
        dtype=np.uint16,
        compressor=compressor
    )

    # Track segment IDs
    segment_id_set = set()
    segment_lock = threading.Lock()

    def extract_and_write_slice(output_idx: int) -> None:
        """Extract and write a single filtered slice."""
        pos = valid_slice_positions[output_idx]
        origin = volume_center + pos * fiber_direction

        # Build coordinates for this slice
        coords = origin[:, np.newaxis, np.newaxis] + displacement

        # Flatten for map_coordinates
        coords_flat = coords.reshape(3, -1)

        # Check bounds
        in_bounds = (
            (coords_flat[0] >= 0) & (coords_flat[0] < volume.shape[0]) &
            (coords_flat[1] >= 0) & (coords_flat[1] < volume.shape[1]) &
            (coords_flat[2] >= 0) & (coords_flat[2] < volume.shape[2])
        )

        # Sample with nearest neighbor
        sampled = np.zeros(coords_flat.shape[1], dtype=volume.dtype)
        if in_bounds.any():
            sampled[in_bounds] = map_coordinates(
                volume,
                coords_flat[:, in_bounds],
                order=0,
                mode='constant',
                cval=0
            )

        # Reshape to 2D and apply filtered LUT
        slice_2d = sampled.reshape(ny, nx)
        slice_2d = filtered_lut[slice_2d]

        # Write to Zarr
        level_0[:, :, output_idx] = slice_2d.astype(np.uint16)

        # Update segment IDs
        unique_ids = set(np.unique(slice_2d).tolist())
        with segment_lock:
            segment_id_set.update(unique_ids)

    # Run second pass
    logger.info(f"Pass 2: Writing {n_valid_slices} filtered slices...")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(extract_and_write_slice, i) for i in range(n_valid_slices)]
        for future in tqdm(as_completed(futures), total=n_valid_slices, desc="Pass 2 (writing)"):
            future.result()

    # Remove background
    segment_id_set.discard(0)
    segment_ids = np.array(sorted(segment_id_set), dtype=np.uint16)
    logger.info(f"Found {len(segment_ids)} unique segments in output")

    # Clear references to volume data to allow garbage collection
    del displacement, U, V, u_coords, v_coords
    gc.collect()

    # Store segment IDs
    labels_group = root.create_group('labels')
    seg_array = labels_group.create_array('segment_ids', shape=segment_ids.shape, dtype=np.uint16)
    seg_array[:] = segment_ids

    # Compute spatial info for metadata
    # Origin of output volume (voxel 0,0,0) in original [z,y,x] coordinates
    # y_out=0 -> v = v_min (padded), x_out=0 -> u = u_min (padded)
    # z_out=0 -> first valid slice position (valid_slice_positions[0])
    volume_center = bbox['volume_center']
    first_slice_pos = valid_slice_positions[0] if len(valid_slice_positions) > 0 else bbox['fiber_min']
    origin_voxels = (volume_center
                     + first_slice_pos * fiber_direction
                     + u_min * v1
                     + v_min * v2)

    spatial_info = {
        'origin_voxels': origin_voxels,
        'v1': v1,
        'v2': v2,
        'volume_center': volume_center,
        'bbox': bbox
    }

    # Write metadata (pyramids will be generated separately)
    root.attrs.update(create_ome_ngff_metadata(bundle, voxel_size_um, n_levels, n_valid_slices, spatial_info))

    # Log spatial positioning info
    origin_um = origin_voxels * voxel_size_um
    logger.info(f"Level 0 written to: {output_path}")
    logger.info(f"  Shape: {output_shape}")
    logger.info(f"  Origin (voxels, zyx): [{origin_voxels[0]:.1f}, {origin_voxels[1]:.1f}, {origin_voxels[2]:.1f}]")
    logger.info(f"  Origin (μm, zyx): [{origin_um[0]:.2f}, {origin_um[1]:.2f}, {origin_um[2]:.2f}]")
    logger.info(f"  Translation (μm, yxz): [{origin_um[1]:.2f}, {origin_um[2]:.2f}, {origin_um[0]:.2f}]")

    gc.collect()


def generate_pyramids_streamed(zarr_path: Path, n_levels: int = 5):
    """
    Generate pyramid levels by streaming from disk.

    Processes slice pairs to minimize memory usage. Each level is generated
    by reading 2 slices at a time from the previous level.

    Args:
        zarr_path: Path to Zarr store with level 0 already written
        n_levels: Number of pyramid levels to generate
    """
    logger.info(f"\nGenerating pyramids (streamed) for: {zarr_path}")

    root = zarr.open_group(str(zarr_path), mode='r+')
    compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)

    prev_level_key = '0'

    for level in range(1, n_levels):
        prev_level = root[prev_level_key]
        prev_shape = prev_level.shape

        # Output shape after 2x downsampling
        new_shape = (
            (prev_shape[0] + 1) // 2,
            (prev_shape[1] + 1) // 2,
            (prev_shape[2] + 1) // 2
        )

        logger.info(f"Generating level {level}: {prev_shape} -> {new_shape}")

        # Create output array with 3D chunks for visualization
        chunk_size = min(64, min(new_shape))
        chunk_shape = (chunk_size, chunk_size, chunk_size)

        level_ds = root.create_array(
            str(level),
            shape=new_shape,
            chunks=chunk_shape,
            dtype=np.uint16,
            compressor=compressor
        )

        # Process slice pairs from previous level
        n_slices_prev = prev_shape[2]
        n_slices_new = new_shape[2]

        for z_out in tqdm(range(n_slices_new), desc=f"Level {level}", unit="slice"):
            # Get the two input slices that contribute to this output slice
            z_in_0 = z_out * 2
            z_in_1 = min(z_in_0 + 1, n_slices_prev - 1)

            # Load the two slices
            slice_0 = prev_level[:, :, z_in_0]
            slice_1 = prev_level[:, :, z_in_1]

            # Downsample in y and x using center voxel method
            # Take every other pixel starting from offset 1 (center of 2x2)
            ny_prev, nx_prev = slice_0.shape

            # For odd dimensions, we need to handle edge cases
            y_indices = np.arange(0, ny_prev, 2)
            x_indices = np.arange(0, nx_prev, 2)

            # Downsample slice_0 (use this as representative - center voxel approximation)
            downsampled_slice = slice_0[np.ix_(y_indices, x_indices)]

            # Write to output
            level_ds[:, :, z_out] = downsampled_slice

        prev_level_key = str(level)

    gc.collect()

    # Report size
    total_size = sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file())
    logger.info(f"Pyramids complete: {zarr_path}")
    logger.info(f"  Total size: {total_size / 1024**2:.1f} MB")


def extract_all_bundles(mat_file: Path,
                        bundle_metadata_file: Path,
                        output_dir: Path,
                        voxel_size_um: float = 0.05,
                        padding_um: float = 0.5,
                        n_levels: int = 5,
                        n_workers: int = 32,
                        min_axon_fraction: float = 0.75,
                        min_axon_coverage: float = 0.5):
    """
    Extract orthogonal slices for all bundles in a volume.

    Args:
        mat_file: Original .mat file with full labeled volume
        bundle_metadata_file: JSON file from preprocess_1
        output_dir: Directory for output Zarr stores
        voxel_size_um: Physical voxel size
        padding_um: Padding around bounding box
        n_levels: Number of pyramid levels
        n_workers: Number of parallel workers
        min_axon_fraction: Minimum axon count fraction for valid slices
        min_axon_coverage: Minimum fraction of valid slices axon must appear in
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Extracting orthogonal slices")
    logger.info(f"Input: {mat_file.name}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'='*80}\n")

    # Load bundle metadata
    logger.info(f"Loading bundle metadata: {bundle_metadata_file}")
    with open(bundle_metadata_file, 'r') as f:
        metadata = json.load(f)

    bundles = metadata['bundles']
    logger.info(f"Found {len(bundles)} bundles to process")

    # Load full volume - keep original dtype to save memory
    logger.info(f"Loading full volume: {mat_file}")
    with h5py.File(mat_file, 'r') as f:
        dset = f['final_lbl']
        logger.info(f"Volume shape: {dset.shape}, dtype: {dset.dtype}")
        volume = dset[:]

    volume_gb = volume.nbytes / (1024**3)
    logger.info(f"Loaded volume: {volume.shape}, dtype: {volume.dtype}, size: {volume_gb:.2f} GB")

    # Force garbage collection after load
    gc.collect()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each bundle
    for i, bundle in enumerate(bundles):
        logger.info(f"\n--- Bundle {i+1}/{len(bundles)} ---")

        output_path = output_dir / f"bundle_{bundle['bundle_id']:02d}_orthogonal.zarr"

        extract_and_write_orthogonal_slices(
            volume,
            bundle,
            output_path,
            voxel_size_um=voxel_size_um,
            padding_um=padding_um,
            n_levels=n_levels,
            n_workers=n_workers,
            min_axon_fraction=min_axon_fraction,
            min_axon_coverage=min_axon_coverage
        )

    # Free original volume before pyramid generation
    logger.info("\nFreeing original volume...")
    del volume
    gc.collect()

    # Generate pyramids (streamed from Zarr, minimal memory)
    for i, bundle in enumerate(bundles):
        output_path = output_dir / f"bundle_{bundle['bundle_id']:02d}_orthogonal.zarr"
        generate_pyramids_streamed(output_path, n_levels=n_levels)

    logger.info(f"\n{'='*80}")
    logger.info(f"Extraction complete!")
    logger.info(f"Created {len(bundles)} orthogonal slice volumes in {output_dir}")
    logger.info(f"{'='*80}\n")


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract orthogonal cross-sections perpendicular to fiber direction'
    )
    parser.add_argument('mat_file', type=Path,
                        help='Original .mat file with full labeled volume')
    parser.add_argument('bundle_metadata', type=Path,
                        help='Bundle metadata JSON from preprocess_1')
    parser.add_argument('output_dir', type=Path,
                        help='Output directory for Zarr stores')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                        help='Voxel size in micrometers (default: 0.05)')
    parser.add_argument('--padding', type=float, default=0.5,
                        help='Padding around bounding box in μm (default: 0.5)')
    parser.add_argument('--n-levels', type=int, default=5,
                        help='Number of pyramid levels (default: 5)')
    parser.add_argument('--n-workers', type=int, default=32,
                        help='Number of parallel workers for extraction (default: 32)')
    parser.add_argument('--min-axon-fraction', type=float, default=0.75,
                        help='Min axon count fraction for valid slices (default: 0.75)')
    parser.add_argument('--min-axon-coverage', type=float, default=0.5,
                        help='Min fraction of valid slices axon must appear in (default: 0.5)')

    args = parser.parse_args()

    extract_all_bundles(
        args.mat_file,
        args.bundle_metadata,
        args.output_dir,
        voxel_size_um=args.voxel_size,
        padding_um=args.padding,
        n_levels=args.n_levels,
        n_workers=args.n_workers,
        min_axon_fraction=args.min_axon_fraction,
        min_axon_coverage=args.min_axon_coverage
    )
