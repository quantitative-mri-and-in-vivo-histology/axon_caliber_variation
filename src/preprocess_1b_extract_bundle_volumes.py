#!/usr/bin/env python3
"""
Extract individual bundle volumes from bundle metadata as OME-Zarr.

Takes bundle metadata JSON (from preprocess_1) and creates axis-aligned
OME-Zarr volumes with multi-resolution pyramids for each bundle,
optimized for Neuroglancer visualization.

Output format: OME-NGFF (Zarr-based) with 5 pyramid levels.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import h5py
import numcodecs
import numpy as np
import zarr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def determine_alignment_axis(orientation: np.ndarray) -> Tuple[int, Tuple[int, int, int]]:
    """
    Determine which axis is most aligned with orientation for axis permutation.

    After permutation, axons will be aligned along the LAST axis (axis 2),
    following the convention (y, x, z) where z is along-axon direction.

    Args:
        orientation: [vz, vy, vx] normalized direction vector (in original coords)

    Returns:
        (axis_index, permutation) where permutation reorders axes so
        primary axis becomes axis 2 (last)
    """
    orientation = np.array(orientation)
    abs_components = np.abs(orientation)
    primary_axis = np.argmax(abs_components)

    # Permutations to move primary axis to position 2 (last axis)
    # Output convention: (y, x, z) where z is along-axon
    if primary_axis == 0:
        permutation = (1, 2, 0)
    elif primary_axis == 1:
        permutation = (0, 2, 1)
    else:
        permutation = (0, 1, 2)

    return primary_axis, permutation


def create_label_lut(axon_labels: Set[int], max_label: int = None) -> np.ndarray:
    """
    Create a lookup table for fast label filtering.

    Args:
        axon_labels: Set of axon labels to keep
        max_label: Maximum label value (auto-detected if None)

    Returns:
        Boolean LUT where lut[label] = True if label is in axon_labels
    """
    if max_label is None:
        max_label = max(axon_labels) if axon_labels else 0

    lut = np.zeros(max_label + 1, dtype=bool)
    for label in axon_labels:
        if label <= max_label:
            lut[label] = True
    return lut


def filter_with_lut(data: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Filter array using lookup table.

    Args:
        data: Input array (2D or 3D)
        lut: Boolean lookup table

    Returns:
        Filtered array with non-matching labels set to 0
    """
    clamped = np.minimum(data, len(lut) - 1)
    mask = lut[clamped]
    result = np.where(mask, data, 0)
    return result.astype(data.dtype)


def get_input_slice(volume: np.ndarray, z_out: int, permutation: Tuple[int, int, int]) -> np.ndarray:
    """
    Extract the input slice corresponding to output z-index.

    Args:
        volume: Input volume (original axes)
        z_out: Output z-index
        permutation: Axis permutation tuple

    Returns:
        2D slice from input volume, already in correct (y, x) orientation
    """
    input_axis = permutation[2]

    if input_axis == 0:
        slice_2d = volume[z_out, :, :]
    elif input_axis == 1:
        slice_2d = volume[:, z_out, :]
    else:
        slice_2d = volume[:, :, z_out]

    remaining = [i for i in range(3) if i != input_axis]
    expected = [permutation[0], permutation[1]]

    if remaining != expected:
        slice_2d = slice_2d.T

    return slice_2d


def downsample_segmentation_mode(data: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample segmentation using mode filtering.

    For each (factor x factor x factor) block, takes the most common non-zero label.
    This preserves label integrity better than nearest-neighbor for segmentation.

    Args:
        data: Input 3D segmentation volume
        factor: Downsampling factor

    Returns:
        Downsampled volume
    """
    if factor == 1:
        return data

    # Calculate output shape
    out_shape = tuple(s // factor for s in data.shape)

    # Trim input to be evenly divisible
    trimmed = data[:out_shape[0] * factor, :out_shape[1] * factor, :out_shape[2] * factor]

    # Reshape into blocks
    reshaped = trimmed.reshape(
        out_shape[0], factor,
        out_shape[1], factor,
        out_shape[2], factor
    )

    # For segmentation, we want the mode (most common label) in each block
    # Flatten blocks and find mode
    blocks = reshaped.transpose(0, 2, 4, 1, 3, 5).reshape(*out_shape, -1)

    # Find mode for each block (most common non-zero, or zero if all zero)
    result = np.zeros(out_shape, dtype=data.dtype)

    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            for k in range(out_shape[2]):
                block = blocks[i, j, k]
                # Get non-zero values
                non_zero = block[block != 0]
                if len(non_zero) > 0:
                    # Find most common
                    values, counts = np.unique(non_zero, return_counts=True)
                    result[i, j, k] = values[np.argmax(counts)]

    return result


def downsample_segmentation_mode_fast(data: np.ndarray, factor: int) -> np.ndarray:
    """
    Fast mode-based downsampling using scipy's maximum_filter approach.

    For segmentation, uses a majority-vote approximation by taking
    the value at the center of each block (faster than true mode).

    Args:
        data: Input 3D segmentation volume
        factor: Downsampling factor

    Returns:
        Downsampled volume
    """
    if factor == 1:
        return data.copy()

    # Simple approach: take center voxel of each block
    # This is fast and works well when labels are spatially coherent
    offset = factor // 2
    return data[offset::factor, offset::factor, offset::factor].copy()


def compute_valid_slice_range(axon_counts: List[int], min_axon_fraction: float) -> Tuple[int, int]:
    """
    Compute valid slice range using symmetric expansion from maximum.

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
    end_idx = max_idx + min_extent + 1

    return start_idx, end_idx


def filter_axons_by_coverage(axon_slices: Dict[int, Set[int]],
                              valid_slice_range: Tuple[int, int],
                              min_coverage: float) -> Set[int]:
    """
    Filter axons by their coverage across valid slices.

    Args:
        axon_slices: Dict mapping axon_id -> set of slice indices where it appears
        valid_slice_range: (start_idx, end_idx) of valid slices
        min_coverage: Minimum fraction of valid slices axon must appear in

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
        count = sum(1 for idx in slice_indices if start_idx <= idx < end_idx)
        if count >= threshold:
            valid_axons.add(axon_id)

    return valid_axons


def create_ome_ngff_metadata(bundle: Dict, voxel_size_um: float, n_levels: int = 5) -> Dict:
    """
    Create OME-NGFF compliant metadata for .zattrs.

    Args:
        bundle: Bundle metadata dictionary
        voxel_size_um: Physical voxel size in micrometers
        n_levels: Number of pyramid levels

    Returns:
        Dictionary for .zattrs
    """
    datasets = []
    for level in range(n_levels):
        scale_factor = 2 ** level
        voxel_size = voxel_size_um * scale_factor
        datasets.append({
            "path": str(level),
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": [voxel_size, voxel_size, voxel_size]
                },
                {
                    "type": "translation",
                    "translation": [0.0, 0.0, 0.0]
                }
            ]
        })

    metadata = {
        "multiscales": [{
            "version": "0.4",
            "name": f"bundle_{bundle['bundle_id']:02d}",
            "axes": [
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
                {"name": "z", "type": "space", "unit": "micrometer"}
            ],
            "datasets": datasets,
            "type": "gaussian"  # Downsampling method hint
        }],
        # Custom metadata
        "bundle_id": int(bundle['bundle_id']),
        "n_axons": int(bundle['n_axons']),
        "mean_orientation": [float(x) for x in bundle['mean_orientation']],
        "mean_length_um": float(bundle['mean_length_um']),
        "voxel_size_um": float(voxel_size_um),
        "alignment_axis": "z",
        "n_levels": n_levels
    }

    return metadata


def write_bundle_level0(volume: np.ndarray,
                        bundle: Dict,
                        output_path: Path,
                        permutation: Tuple[int, int, int],
                        lut: np.ndarray,
                        voxel_size_um: float = 0.05,
                        min_axon_fraction: float = 0.75,
                        min_axon_coverage: float = 0.5) -> np.ndarray:
    """
    Write level 0 of bundle to Zarr with two-pass filtering.

    Args:
        volume: Full input volume (kept in memory)
        bundle: Bundle metadata dictionary
        output_path: Output Zarr store path
        permutation: Axis permutation tuple
        lut: Boolean lookup table for this bundle
        voxel_size_um: Physical voxel size
        min_axon_fraction: Minimum axon count fraction for valid slices
        min_axon_coverage: Minimum fraction of valid slices axon must appear in

    Returns:
        Array of segment IDs found in this bundle
    """
    # Calculate full output shape after permutation
    full_output_shape = tuple(volume.shape[i] for i in permutation)
    n_slices = full_output_shape[2]

    # Get bundle axon labels
    bundle_labels = set(bundle['axon_labels'])

    # =========================================================================
    # FIRST PASS: Track axon presence per slice
    # =========================================================================
    logger.info(f"Pass 1: Tracking axon presence in {n_slices} slices...")

    axon_slices: Dict[int, Set[int]] = {label: set() for label in bundle_labels}
    axon_counts = []

    for z_out in tqdm(range(n_slices), desc="Pass 1 (tracking)", unit="slice"):
        input_slice = get_input_slice(volume, z_out, permutation)
        filtered_slice = filter_with_lut(input_slice, lut)

        unique_ids = set(np.unique(filtered_slice).tolist())
        unique_ids.discard(0)

        axon_counts.append(len(unique_ids))
        for axon_id in unique_ids:
            if axon_id in axon_slices:
                axon_slices[axon_id].add(z_out)

    # =========================================================================
    # FILTERING: Compute valid slice range and valid axons
    # =========================================================================
    logger.info(f"Computing filtering criteria...")

    valid_start, valid_end = compute_valid_slice_range(axon_counts, min_axon_fraction)
    n_valid_slices = valid_end - valid_start

    max_count = max(axon_counts) if axon_counts else 0
    max_idx = axon_counts.index(max_count) if max_count > 0 else 0
    logger.info(f"  Max axon count: {max_count} at slice {max_idx}")
    logger.info(f"  Valid slice range: {valid_start}-{valid_end-1} ({n_valid_slices} slices)")
    logger.info(f"  Threshold: {min_axon_fraction * 100:.0f}% of max = {int(max_count * min_axon_fraction)} axons")

    valid_axons = filter_axons_by_coverage(axon_slices, (valid_start, valid_end), min_axon_coverage)
    n_rejected = len(bundle_labels) - len(valid_axons)
    logger.info(f"  Valid axons: {len(valid_axons)} (rejected {n_rejected})")
    logger.info(f"  Coverage threshold: {min_axon_coverage * 100:.0f}% of {n_valid_slices} slices")

    # Create filtered LUT
    max_label = len(lut) - 1
    filtered_lut = np.zeros(max_label + 1, dtype=bool)
    for label in valid_axons:
        if label <= max_label:
            filtered_lut[label] = True

    # =========================================================================
    # SECOND PASS: Write filtered slices
    # =========================================================================
    output_shape = (full_output_shape[0], full_output_shape[1], n_valid_slices)

    logger.info(f"Creating Zarr store: {output_path}")
    logger.info(f"Output shape: {output_shape}")

    root = zarr.open_group(str(output_path), mode='w', zarr_format=2)
    compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)

    chunk_shape_0 = (output_shape[0], output_shape[1], 1)
    level_0 = root.create_array(
        '0',
        shape=output_shape,
        chunks=chunk_shape_0,
        dtype=np.uint16,
        compressor=compressor
    )

    logger.info(f"Pass 2: Writing {n_valid_slices} filtered slices...")

    segment_id_set = set()
    for out_idx, z_in in enumerate(tqdm(range(valid_start, valid_end), desc="Pass 2 (writing)", unit="slice")):
        input_slice = get_input_slice(volume, z_in, permutation)
        filtered_slice = filter_with_lut(input_slice, filtered_lut)
        level_0[:, :, out_idx] = filtered_slice
        segment_id_set.update(np.unique(filtered_slice).tolist())

    segment_id_set.discard(0)
    segment_ids = np.array(sorted(segment_id_set), dtype=np.uint16)
    logger.info(f"Found {len(segment_ids)} unique segments in output")

    return segment_ids


def generate_bundle_pyramid(output_path: Path,
                            bundle: Dict,
                            segment_ids: np.ndarray,
                            voxel_size_um: float = 0.05,
                            n_levels: int = 5):
    """
    Generate pyramid levels for a bundle (loads level 0 into memory).

    Args:
        output_path: Zarr store path (must have level 0 written)
        bundle: Bundle metadata dictionary
        segment_ids: Array of segment IDs
        voxel_size_um: Physical voxel size
        n_levels: Number of pyramid levels (default 5)
    """
    # Open existing Zarr store
    root = zarr.open_group(str(output_path), mode='r+')

    # Compression codec (numcodecs for Zarr v2)
    compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)

    # Load level 0 into memory for pyramid generation
    logger.info("Loading level 0 for pyramid generation...")
    current_data = root['0'][:]

    for level in range(1, n_levels):
        logger.info(f"Generating level {level} (2x downsampling from level {level-1})...")

        # Downsample from previous level (cascaded for memory efficiency)
        downsampled = downsample_segmentation_mode_fast(current_data, 2)

        # 3D chunks for visualization
        chunk_size = min(64, min(downsampled.shape))
        chunk_shape = (chunk_size, chunk_size, chunk_size)

        # Create array for this level
        level_ds = root.create_array(
            str(level),
            shape=downsampled.shape,
            chunks=chunk_shape,
            dtype=np.uint16,
            compressor=compressor
        )
        level_ds[:] = downsampled

        logger.info(f"  Level {level} shape: {downsampled.shape}")

        # Use downsampled as input for next level
        current_data = downsampled

    # Free pyramid memory
    del current_data

    # Store segment IDs in a labels group
    labels_group = root.create_group('labels')
    labels_group.create_array('segment_ids', data=segment_ids.astype(np.uint16))

    # Write OME-NGFF metadata
    root.attrs.update(create_ome_ngff_metadata(bundle, voxel_size_um, n_levels))

    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
    logger.info(f"Saved bundle Zarr: {output_path}")
    logger.info(f"  Total size: {total_size / 1024**2:.1f} MB")


def extract_bundles_full_volume(mat_file: Path,
                                 bundles: List[Dict],
                                 output_dir: Path,
                                 voxel_size_um: float = 0.05,
                                 n_levels: int = 5,
                                 min_axon_fraction: float = 0.75,
                                 min_axon_coverage: float = 0.5):
    """
    Extract all bundles as OME-Zarr using full-volume-in-memory approach.

    Args:
        mat_file: Original .mat file with full labeled volume
        bundles: List of bundle metadata dictionaries
        output_dir: Directory for output Zarr stores
        voxel_size_um: Physical voxel size at full resolution
        n_levels: Number of pyramid levels
        min_axon_fraction: Minimum axon count fraction for valid slices
        min_axon_coverage: Minimum fraction of valid slices axon must appear in
    """
    logger.info("Using full-volume mode")

    # Load entire volume into memory once
    logger.info("Loading full volume into memory...")
    with h5py.File(mat_file, 'r') as f:
        dataset = f['final_lbl']
        logger.info(f"Volume shape: {dataset.shape}, dtype: {dataset.dtype}")
        volume = dataset[:]

    volume_gb = volume.nbytes / (1024**3)
    logger.info(f"Loaded {volume_gb:.2f} GB into memory")

    # Get max label for LUT sizing
    max_label = int(volume.max())
    logger.info(f"Max label in volume: {max_label}")

    # Phase 1: Write level 0 for all bundles (needs original volume)
    bundle_info = []  # Store (output_path, bundle, segment_ids) for phase 2

    for i, bundle in enumerate(bundles):
        logger.info(f"\n--- Bundle {i+1}/{len(bundles)} (Phase 1: Write level 0) ---")
        logger.info(f"Bundle ID: {bundle['bundle_id']}")
        logger.info(f"Axons: {bundle['n_axons']}")
        logger.info(f"Mean orientation: {bundle['mean_orientation']}")

        # Determine axis permutation
        primary_axis, permutation = determine_alignment_axis(bundle['mean_orientation'])
        logger.info(f"Primary axis: {primary_axis} -> permutation {permutation}")

        # Create LUT for this bundle
        axon_labels = set(bundle['axon_labels'])
        lut = create_label_lut(axon_labels, max_label)

        # Write level 0 to Zarr
        output_path = output_dir / f"bundle_{bundle['bundle_id']:02d}_aligned.zarr"
        segment_ids = write_bundle_level0(
            volume,
            bundle,
            output_path,
            permutation,
            lut,
            voxel_size_um,
            min_axon_fraction,
            min_axon_coverage
        )
        bundle_info.append((output_path, bundle, segment_ids))

    # Free original volume before pyramid generation
    logger.info("\nFreeing original volume from memory...")
    del volume

    # Phase 2: Generate pyramids (loads each level 0 individually)
    for i, (output_path, bundle, segment_ids) in enumerate(bundle_info):
        logger.info(f"\n--- Bundle {i+1}/{len(bundles)} (Phase 2: Generate pyramid) ---")
        logger.info(f"Bundle ID: {bundle['bundle_id']}")

        generate_bundle_pyramid(
            output_path,
            bundle,
            segment_ids,
            voxel_size_um,
            n_levels
        )

    logger.info(f"\nExtraction complete! Created {len(bundles)} bundle Zarr stores.")


def extract_all_bundles(mat_file: Path,
                        bundle_metadata_file: Path,
                        output_dir: Path,
                        voxel_size_um: float = 0.05,
                        n_levels: int = 5,
                        min_axon_fraction: float = 0.75,
                        min_axon_coverage: float = 0.5):
    """
    Extract all bundles from volume and save as OME-Zarr stores.

    Args:
        mat_file: Original .mat file with full labeled volume
        bundle_metadata_file: JSON file from preprocess_1
        output_dir: Directory for output Zarr stores
        voxel_size_um: Physical voxel size at full resolution
        n_levels: Number of pyramid levels
        min_axon_fraction: Minimum axon count fraction for valid slices
        min_axon_coverage: Minimum fraction of valid slices axon must appear in
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Extracting bundle volumes from: {mat_file.name}")
    logger.info(f"Output format: OME-Zarr with {n_levels} pyramid levels")
    logger.info(f"{'='*80}\n")

    # Load bundle metadata
    logger.info(f"Loading bundle metadata: {bundle_metadata_file.name}")
    with open(bundle_metadata_file, 'r') as f:
        metadata = json.load(f)

    bundles = metadata['bundles']
    logger.info(f"Found {len(bundles)} bundles to extract")

    output_dir.mkdir(parents=True, exist_ok=True)

    extract_bundles_full_volume(
        mat_file,
        bundles,
        output_dir,
        voxel_size_um,
        n_levels,
        min_axon_fraction,
        min_axon_coverage
    )

    logger.info(f"\n{'='*80}")
    logger.info(f"Extraction complete!")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract individual bundle volumes as OME-Zarr with pyramids'
    )
    parser.add_argument('mat_file', type=Path,
                        help='Original .mat file with full labeled volume')
    parser.add_argument('bundle_metadata', type=Path,
                        help='Bundle metadata JSON from preprocess_1')
    parser.add_argument('output_dir', type=Path,
                        help='Output directory for bundle Zarr stores')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                        help='Voxel size in micrometers (default: 0.05)')
    parser.add_argument('--n-levels', type=int, default=5,
                        help='Number of pyramid levels (default: 5)')
    parser.add_argument('--min-axon-fraction', type=float, default=0.75,
                        help='Minimum axon count fraction for valid slices (default: 0.75)')
    parser.add_argument('--min-axon-coverage', type=float, default=0.5,
                        help='Minimum fraction of valid slices axon must appear in (default: 0.5)')

    args = parser.parse_args()

    extract_all_bundles(
        args.mat_file,
        args.bundle_metadata,
        args.output_dir,
        voxel_size_um=args.voxel_size,
        n_levels=args.n_levels,
        min_axon_fraction=args.min_axon_fraction,
        min_axon_coverage=args.min_axon_coverage
    )
