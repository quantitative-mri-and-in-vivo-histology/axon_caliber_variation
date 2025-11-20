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
from zarr.codecs import BloscCodec
from scipy.ndimage import map_coordinates
from tqdm import tqdm

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
                              n_slices: int) -> Dict:
    """
    Create OME-NGFF compliant metadata for orthogonal slice output.

    Args:
        bundle: Bundle metadata dictionary
        voxel_size_um: Physical voxel size
        n_levels: Number of pyramid levels
        n_slices: Number of slices extracted

    Returns:
        Dictionary for .zattrs
    """
    datasets = []
    for level in range(n_levels):
        scale_factor = 2 ** level
        voxel_size = voxel_size_um * scale_factor
        datasets.append({
            "path": str(level),
            "coordinateTransformations": [{
                "type": "scale",
                "scale": [voxel_size, voxel_size, voxel_size]
            }]
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

    # Create Zarr group
    root = zarr.open_group(str(output_path), mode='w')

    # Compression codec
    compressor = BloscCodec(cname='zstd', clevel=3, shuffle='bitshuffle')

    # Level 0: Full resolution with (ny, nx, 1) chunking for slice access
    chunk_shape_0 = (output_shape[0], output_shape[1], 1)
    level_0 = root.create_array(
        '0',
        shape=output_shape,
        chunks=chunk_shape_0,
        dtype=np.uint32,
        compressors=[compressor]
    )

    # Write level 0 slice by slice
    logger.info(f"Writing level 0 ({n_slices} slices)...")
    segment_id_set = set()

    for z in tqdm(range(n_slices), desc="Writing level 0", unit="slice"):
        slice_2d = slices[:, :, z]
        level_0[:, :, z] = slice_2d.astype(np.uint32)
        segment_id_set.update(np.unique(slice_2d).tolist())

    # Remove background from segment IDs
    segment_id_set.discard(0)
    segment_ids = np.array(sorted(segment_id_set), dtype=np.uint32)
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
            dtype=np.uint32,
            compressors=[compressor]
        )
        level_ds[:] = downsampled

        logger.info(f"  Level {level} shape: {downsampled.shape}")

        current_data = downsampled

    del current_data

    # Store segment IDs
    labels_group = root.create_group('labels')
    seg_array = labels_group.create_array('segment_ids', shape=segment_ids.shape, dtype=np.uint32)
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
                                         n_workers: int = 32):
    """
    Extract orthogonal slices and write directly to Zarr (streaming).

    This version extracts slices in parallel and streams them to Zarr.
    Pyramids are generated separately to allow freeing the original volume first.

    Args:
        volume: Full 3D labeled volume
        bundle: Bundle metadata dictionary
        output_path: Output Zarr path
        voxel_size_um: Physical voxel size
        padding_um: Padding around bounding box in micrometers
        n_levels: Number of pyramid levels (stored in metadata, pyramids generated separately)
        n_workers: Number of parallel workers for extraction
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
    output_size_gb = ny * nx * n_slices * 4 / (1024**3)
    logger.info(f"Output size: {output_size_gb:.2f} GB")

    # Create Zarr store and level 0
    logger.info(f"Creating Zarr store: {output_path}")
    root = zarr.open_group(str(output_path), mode='w')
    compressor = BloscCodec(cname='zstd', clevel=3, shuffle='bitshuffle')

    output_shape = (ny, nx, n_slices)
    chunk_shape_0 = (ny, nx, 1)
    level_0 = root.create_array(
        '0',
        shape=output_shape,
        chunks=chunk_shape_0,
        dtype=np.uint32,
        compressors=[compressor]
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
    label_lut = np.zeros(max_label + 1, dtype=np.uint32)
    for label in bundle_labels:
        if label <= max_label:
            label_lut[label] = label  # Keep bundle labels, others stay 0

    logger.info(f"Filtering to {len(bundle_labels)} bundle axons")

    # Track segment IDs (thread-safe with lock)
    import threading
    segment_id_set = set()
    segment_lock = threading.Lock()
    progress_lock = threading.Lock()
    completed_count = [0]  # Use list for mutable in closure

    # Define function to extract and write a single slice
    def extract_and_write_slice(slice_idx: int) -> None:
        """Extract a single orthogonal slice and write immediately."""
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

        # Reshape to 2D
        slice_2d = sampled.reshape(ny, nx)

        # Filter to keep only bundle axons (apply LUT)
        slice_2d = label_lut[slice_2d]

        # Write immediately to Zarr (Zarr handles concurrent writes to different chunks)
        level_0[:, :, slice_idx] = slice_2d.astype(np.uint32)

        # Update segment IDs (thread-safe)
        unique_ids = set(np.unique(slice_2d).tolist())
        with segment_lock:
            segment_id_set.update(unique_ids)

        # Update progress
        with progress_lock:
            completed_count[0] += 1

    # Extract and write slices in parallel
    actual_workers = min(n_workers, n_slices)
    logger.info(f"Extracting {n_slices} slices using {actual_workers} parallel workers...")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks - they write immediately, no return values accumulate
        futures = [executor.submit(extract_and_write_slice, i) for i in range(n_slices)]

        # Wait for completion with progress bar
        for future in tqdm(as_completed(futures), total=n_slices, desc="Extracting slices"):
            future.result()  # Just check for exceptions, no data returned

    # Remove background
    segment_id_set.discard(0)
    segment_ids = np.array(sorted(segment_id_set), dtype=np.uint32)
    logger.info(f"Found {len(segment_ids)} unique segments")

    # Clear references to volume data to allow garbage collection
    del displacement, U, V, u_coords, v_coords
    gc.collect()

    # Store segment IDs
    labels_group = root.create_group('labels')
    seg_array = labels_group.create_array('segment_ids', shape=segment_ids.shape, dtype=np.uint32)
    seg_array[:] = segment_ids

    # Write metadata (pyramids will be generated separately)
    root.attrs.update(create_ome_ngff_metadata(bundle, voxel_size_um, n_levels, n_slices))

    logger.info(f"Level 0 written to: {output_path}")
    logger.info(f"  Shape: {output_shape}")

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
    compressor = BloscCodec(cname='zstd', clevel=3, shuffle='bitshuffle')

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
            dtype=np.uint32,
            compressors=[compressor]
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
                        n_workers: int = 32):
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
            n_workers=n_workers
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

    args = parser.parse_args()

    extract_all_bundles(
        args.mat_file,
        args.bundle_metadata,
        args.output_dir,
        voxel_size_um=args.voxel_size,
        padding_um=args.padding,
        n_levels=args.n_levels,
        n_workers=args.n_workers
    )
