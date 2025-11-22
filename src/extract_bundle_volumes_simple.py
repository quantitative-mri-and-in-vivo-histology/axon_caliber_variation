#!/usr/bin/env python3
"""
Extract individual bundle volumes from bundle metadata as OME-Zarr (simple version).

This is a simplified version without two-pass filtering:
- No slice range filtering (min_axon_fraction)
- No axon coverage filtering (min_axon_coverage)
- Single pass: directly writes all slices

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
    """
    orientation = np.array(orientation)
    abs_components = np.abs(orientation)
    primary_axis = np.argmax(abs_components)

    if primary_axis == 0:
        permutation = (1, 2, 0)
    elif primary_axis == 1:
        permutation = (0, 2, 1)
    else:
        permutation = (0, 1, 2)

    return primary_axis, permutation


def create_label_lut(axon_labels: Set[int], max_label: int = None) -> np.ndarray:
    """Create a lookup table for fast label filtering."""
    if max_label is None:
        max_label = max(axon_labels) if axon_labels else 0

    lut = np.zeros(max_label + 1, dtype=bool)
    for label in axon_labels:
        if label <= max_label:
            lut[label] = True
    return lut


def filter_with_lut(data: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Filter array using lookup table."""
    clamped = np.minimum(data, len(lut) - 1)
    mask = lut[clamped]
    result = np.where(mask, data, 0)
    return result.astype(data.dtype)


def get_input_slice(volume: np.ndarray, z_out: int, permutation: Tuple[int, int, int]) -> np.ndarray:
    """Extract the input slice corresponding to output z-index."""
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


def downsample_segmentation_mode_fast(data: np.ndarray, factor: int) -> np.ndarray:
    """Fast mode-based downsampling using center voxel of each block."""
    if factor == 1:
        return data.copy()

    offset = factor // 2
    return data[offset::factor, offset::factor, offset::factor].copy()


def create_ome_ngff_metadata(bundle: Dict, voxel_size_um: float, n_levels: int = 5) -> Dict:
    """Create OME-NGFF compliant metadata for .zattrs."""
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
            "type": "gaussian"
        }],
        "bundle_id": int(bundle['bundle_id']),
        "n_axons": int(bundle['n_axons']),
        "mean_orientation": [float(x) for x in bundle['mean_orientation']],
        "mean_length_um": float(bundle['mean_length_um']),
        "voxel_size_um": float(voxel_size_um),
        "alignment_axis": "z",
        "n_levels": n_levels
    }

    return metadata


def write_bundle_level0_simple(volume: np.ndarray,
                                bundle: Dict,
                                output_path: Path,
                                permutation: Tuple[int, int, int],
                                lut: np.ndarray,
                                voxel_size_um: float = 0.05) -> np.ndarray:
    """
    Write level 0 of bundle to Zarr (single pass, no filtering).

    Args:
        volume: Full input volume
        bundle: Bundle metadata dictionary
        output_path: Output Zarr store path
        permutation: Axis permutation tuple
        lut: Boolean lookup table for this bundle
        voxel_size_um: Physical voxel size

    Returns:
        Array of segment IDs found in this bundle
    """
    # Calculate output shape after permutation
    output_shape = tuple(volume.shape[i] for i in permutation)
    n_slices = output_shape[2]

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

    logger.info(f"Writing {n_slices} slices...")

    segment_id_set = set()
    for z_out in tqdm(range(n_slices), desc="Writing slices", unit="slice"):
        input_slice = get_input_slice(volume, z_out, permutation)
        filtered_slice = filter_with_lut(input_slice, lut)
        level_0[:, :, z_out] = filtered_slice
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
    """Generate pyramid levels for a bundle."""
    root = zarr.open_group(str(output_path), mode='r+')
    compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)

    logger.info("Loading level 0 for pyramid generation...")
    current_data = root['0'][:]

    for level in range(1, n_levels):
        logger.info(f"Generating level {level}...")

        downsampled = downsample_segmentation_mode_fast(current_data, 2)

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

    labels_group = root.create_group('labels')
    labels_group.create_array('segment_ids', data=segment_ids.astype(np.uint16))

    root.attrs.update(create_ome_ngff_metadata(bundle, voxel_size_um, n_levels))

    total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
    logger.info(f"Saved bundle Zarr: {output_path}")
    logger.info(f"  Total size: {total_size / 1024**2:.1f} MB")


def extract_all_bundles(mat_file: Path,
                        bundle_metadata_file: Path,
                        output_dir: Path,
                        voxel_size_um: float = 0.05,
                        n_levels: int = 5):
    """
    Extract all bundles from volume and save as OME-Zarr stores.

    Args:
        mat_file: Original .mat file with full labeled volume
        bundle_metadata_file: JSON file from preprocess_1
        output_dir: Directory for output Zarr stores
        voxel_size_um: Physical voxel size at full resolution
        n_levels: Number of pyramid levels
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

    # Load entire volume into memory once
    logger.info("Loading full volume into memory...")
    with h5py.File(mat_file, 'r') as f:
        dataset = f['final_lbl']
        logger.info(f"Volume shape: {dataset.shape}, dtype: {dataset.dtype}")
        volume = dataset[:]

    volume_gb = volume.nbytes / (1024**3)
    logger.info(f"Loaded {volume_gb:.2f} GB into memory")

    max_label = int(volume.max())
    logger.info(f"Max label in volume: {max_label}")

    # Phase 1: Write level 0 for all bundles
    bundle_info = []

    for i, bundle in enumerate(bundles):
        logger.info(f"\n--- Bundle {i+1}/{len(bundles)} (Write level 0) ---")
        logger.info(f"Bundle ID: {bundle['bundle_id']}")
        logger.info(f"Axons: {bundle['n_axons']}")

        primary_axis, permutation = determine_alignment_axis(bundle['mean_orientation'])
        logger.info(f"Primary axis: {primary_axis} -> permutation {permutation}")

        axon_labels = set(bundle['axon_labels'])
        lut = create_label_lut(axon_labels, max_label)

        output_path = output_dir / f"bundle_{bundle['bundle_id']:02d}_aligned.zarr"
        segment_ids = write_bundle_level0_simple(
            volume, bundle, output_path, permutation, lut, voxel_size_um
        )
        bundle_info.append((output_path, bundle, segment_ids))

    # Free original volume
    logger.info("\nFreeing original volume from memory...")
    del volume

    # Phase 2: Generate pyramids
    for i, (output_path, bundle, segment_ids) in enumerate(bundle_info):
        logger.info(f"\n--- Bundle {i+1}/{len(bundles)} (Generate pyramid) ---")
        logger.info(f"Bundle ID: {bundle['bundle_id']}")

        generate_bundle_pyramid(output_path, bundle, segment_ids, voxel_size_um, n_levels)

    logger.info(f"\n{'='*80}")
    logger.info(f"Extraction complete! Created {len(bundles)} bundle Zarr stores.")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract individual bundle volumes as OME-Zarr (simple, no filtering)'
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

    args = parser.parse_args()

    extract_all_bundles(
        args.mat_file,
        args.bundle_metadata,
        args.output_dir,
        voxel_size_um=args.voxel_size,
        n_levels=args.n_levels
    )
