#!/usr/bin/env python3
"""
Extract individual bundle volumes from bundle metadata.

Takes bundle metadata JSON (from preprocess_1) and creates axis-aligned
HDF5 volumes for each bundle, ready for oblique slicing analysis.
"""

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Dict, List, Set, Tuple

import h5py
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def determine_alignment_axis(orientation: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Determine which axis is most aligned with orientation for axis permutation.

    Args:
        orientation: [vz, vy, vx] normalized direction vector

    Returns:
        (axis_index, permutation) where permutation reorders (z,y,x) so
        primary axis becomes z
    """
    orientation = np.array(orientation)
    abs_components = np.abs(orientation)
    primary_axis = np.argmax(abs_components)

    # Permutations to move primary axis to position 0 (z)
    if primary_axis == 0:
        # Already z-aligned
        permutation = (0, 1, 2)
    elif primary_axis == 1:
        # y is primary -> move to z
        permutation = (1, 0, 2)  # (y, z, x)
    else:
        # x is primary -> move to z
        permutation = (2, 1, 0)  # (x, y, z)

    return primary_axis, permutation


def create_label_lut(axon_labels: Set[int], max_label: int = None) -> np.ndarray:
    """
    Create a lookup table for fast label filtering.

    Much faster than np.isin() for repeated operations.

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


def filter_volume_with_lut(volume: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Filter volume using lookup table (much faster than np.isin).

    Args:
        volume: Input labeled volume
        lut: Boolean lookup table

    Returns:
        Filtered volume with non-matching labels set to 0
    """
    # Properly handle out-of-range labels (set them to 0, not false positives)
    in_range = volume < len(lut)
    mask = np.zeros(volume.shape, dtype=bool)
    mask[in_range] = lut[volume[in_range]]
    return np.where(mask, volume, 0).astype(volume.dtype)


def process_batch(args):
    """
    Process a batch of slices along any axis (generic worker function).

    Args:
        args: Tuple of (mat_file, start_idx, end_idx, axon_labels_set, max_label, axis, transpose_order)
            - axis: 0 for z, 1 for y, 2 for x
            - transpose_order: None for z, (1,0,2) for y, (2,1,0) for x

    Returns:
        (start_idx, end_idx, filtered_data)
    """
    mat_file, start_idx, end_idx, axon_labels_set, max_label, axis, transpose_order = args

    with h5py.File(mat_file, 'r') as f:
        dataset = f['final_lbl']
        # Build slice tuple dynamically based on axis
        slices = [slice(None)] * 3
        slices[axis] = slice(start_idx, end_idx)
        batch = dataset[tuple(slices)]

    lut = create_label_lut(axon_labels_set, max_label)
    filtered = filter_volume_with_lut(batch, lut)

    # Apply transpose if needed
    if transpose_order is not None:
        filtered = filtered.transpose(transpose_order)

    return start_idx, end_idx, filtered


def extract_bundle_volume_streaming(mat_file: Path,
                                   dataset: h5py.Dataset,
                                   axon_labels: Set[int],
                                   permutation: Tuple[int, int, int],
                                   output_file: Path,
                                   bundle: Dict,
                                   voxel_size_um: float = 0.05,
                                   batch_size: int = 50,
                                   n_workers: int = None):
    """
    Extract and save bundle volume with parallel batched processing.

    Args:
        mat_file: Path to source .mat file (needed for parallel workers)
        dataset: HDF5 dataset reference (not loaded into memory)
        axon_labels: Set of axon labels in bundle
        permutation: Axis permutation tuple
        output_file: Output HDF5 file path
        bundle: Bundle metadata dictionary
        voxel_size_um: Physical voxel size
        batch_size: Number of slices to process per batch
        n_workers: Number of parallel workers (None = CPU count)
    """
    import os
    if n_workers is None:
        n_workers = os.cpu_count() or 4

    logger.info(f"Extracting {len(axon_labels)} axons from volume (optimized)...")
    logger.info(f"Using {n_workers} workers, batch size {batch_size}")

    original_shape = dataset.shape
    # Use max of axon_labels to avoid loading entire dataset into memory
    max_label = max(axon_labels) if axon_labels else 0

    # Determine output shape after permutation
    permuted_shape = tuple(original_shape[i] for i in permutation)

    logger.info(f"Original shape: {original_shape}")
    logger.info(f"Permuted shape: {permuted_shape}")

    # Create output file with optimized chunking based on write pattern
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine which original axis becomes the new z-axis
    # permutation[0] tells us which original axis goes to position 0 (new z)
    read_axis = permutation[0]

    # Optimize chunk layout for write pattern
    if read_axis == 0:
        # Z-aligned: standard chunks
        chunk_shape = (min(100, permuted_shape[0]),
                      min(512, permuted_shape[1]),
                      min(512, permuted_shape[2]))
    elif read_axis == 1:
        # Y-aligned: optimize for y-major writes
        chunk_shape = (min(batch_size, permuted_shape[0]),
                      min(512, permuted_shape[1]),
                      min(512, permuted_shape[2]))
    else:
        # X-aligned: optimize for x-major writes
        chunk_shape = (min(batch_size, permuted_shape[0]),
                      min(512, permuted_shape[1]),
                      min(512, permuted_shape[2]))

    # Axis-specific configuration for transpose
    axis_config = {
        0: None,        # Z-aligned: no transpose
        1: (1, 0, 2),   # Y-aligned: (z, y, x) -> (y, z, x)
        2: (2, 1, 0),   # X-aligned: (z, y, x) -> (x, y, z)
    }

    # Prepare batch tasks with generic function
    n_slices = original_shape[read_axis]
    transpose_order = axis_config[read_axis]
    tasks = [(str(mat_file), i, min(i + batch_size, n_slices), axon_labels, max_label,
              read_axis, transpose_order)
             for i in range(0, n_slices, batch_size)]

    logger.info(f"Processing {len(tasks)} batches of ~{batch_size} slices each")

    with h5py.File(output_file, 'w') as f_out:
        dset_out = f_out.create_dataset(
            'labels',
            shape=permuted_shape,
            dtype=np.uint32,
            chunks=chunk_shape,
            compression='gzip',
            compression_opts=1
        )

        # Add metadata
        dset_out.attrs['bundle_id'] = bundle['bundle_id']
        dset_out.attrs['n_axons'] = bundle['n_axons']
        dset_out.attrs['mean_orientation'] = bundle['mean_orientation']
        dset_out.attrs['mean_length_um'] = bundle['mean_length_um']
        dset_out.attrs['voxel_size_um'] = voxel_size_um
        dset_out.attrs['alignment_axis'] = 'z'

        # Process in parallel
        if n_workers > 1 and len(tasks) > 1:
            # Use ProcessPoolExecutor for parallel processing
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(process_batch, task): task for task in tasks}

                with tqdm(total=len(tasks), desc="Processing batches", unit="batch") as pbar:
                    for future in as_completed(futures):
                        start_idx, end_idx, filtered_data = future.result()
                        dset_out[start_idx:end_idx, :, :] = filtered_data
                        pbar.update(1)
        else:
            # Single-threaded fallback
            for task in tqdm(tasks, desc="Processing batches", unit="batch"):
                start_idx, end_idx, filtered_data = process_batch(task)
                dset_out[start_idx:end_idx, :, :] = filtered_data

    logger.info(f"Saved bundle volume: {output_file}")
    logger.info(f"  Size: {output_file.stat().st_size / 1024**2:.1f} MB")


def save_bundle_hdf5(volume: np.ndarray,
                    bundle: Dict,
                    output_file: Path,
                    voxel_size_um: float = 0.05):
    """
    Save bundle volume to HDF5 with metadata.

    Args:
        volume: Labeled 3D volume (aligned)
        bundle: Bundle metadata dictionary
        output_file: Output HDF5 file path
        voxel_size_um: Physical voxel size
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, 'w') as f:
        # Create dataset with chunking and light compression
        chunk_shape = (min(100, volume.shape[0]),
                      min(512, volume.shape[1]),
                      min(512, volume.shape[2]))

        dset = f.create_dataset(
            'labels',
            data=volume,
            dtype=np.uint32,
            chunks=chunk_shape,
            compression='gzip',
            compression_opts=1
        )

        # Add metadata as attributes
        dset.attrs['bundle_id'] = bundle['bundle_id']
        dset.attrs['n_axons'] = bundle['n_axons']
        dset.attrs['mean_orientation'] = bundle['mean_orientation']
        dset.attrs['mean_length_um'] = bundle['mean_length_um']
        dset.attrs['voxel_size_um'] = voxel_size_um
        dset.attrs['alignment_axis'] = 'z'

    logger.info(f"Saved bundle volume: {output_file}")
    logger.info(f"  Shape: {volume.shape}")
    logger.info(f"  Size: {output_file.stat().st_size / 1024**2:.1f} MB")


def extract_all_bundles(mat_file: Path,
                       bundle_metadata_file: Path,
                       output_dir: Path,
                       voxel_size_um: float = 0.05,
                       batch_size: int = 50,
                       n_workers: int = None):
    """
    Extract all bundles from volume and save as separate HDF5 files.

    Args:
        mat_file: Original .mat file with full labeled volume
        bundle_metadata_file: JSON file from preprocess_1
        output_dir: Directory for output HDF5 files
        voxel_size_um: Physical voxel size at full resolution
        batch_size: Number of slices to process per batch
        n_workers: Number of parallel workers (None = CPU count)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Extracting bundle volumes from: {mat_file.name}")
    logger.info(f"{'='*80}\n")

    # Load bundle metadata
    logger.info(f"Loading bundle metadata: {bundle_metadata_file.name}")
    with open(bundle_metadata_file, 'r') as f:
        metadata = json.load(f)

    bundles = metadata['bundles']
    logger.info(f"Found {len(bundles)} bundles to extract")

    # Open HDF5 file (keep as dataset reference - don't load into memory)
    logger.info(f"Opening volume file: {mat_file.name}")
    with h5py.File(mat_file, 'r') as f:
        dataset = f['final_lbl']
        logger.info(f"Volume shape: {dataset.shape}")

        # Extract each bundle
        for i, bundle in enumerate(bundles):
            logger.info(f"\n--- Bundle {i+1}/{len(bundles)} ---")
            logger.info(f"Bundle ID: {bundle['bundle_id']}")
            logger.info(f"Axons: {bundle['n_axons']}")
            logger.info(f"Mean orientation: {bundle['mean_orientation']}")

            # Determine axis permutation
            primary_axis, permutation = determine_alignment_axis(bundle['mean_orientation'])
            logger.info(f"Primary axis: {primary_axis} -> applying permutation {permutation}")

            # Extract and save bundle volume (streaming - low memory)
            axon_labels = set(bundle['axon_labels'])
            output_file = output_dir / f"bundle_{bundle['bundle_id']:02d}_aligned.h5"
            extract_bundle_volume_streaming(
                mat_file,
                dataset,
                axon_labels,
                permutation,
                output_file,
                bundle,
                voxel_size_um,
                batch_size=batch_size,
                n_workers=n_workers
            )

    logger.info(f"\n{'='*80}")
    logger.info(f"Extraction complete! Created {len(bundles)} bundle volumes.")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract individual bundle volumes from bundle metadata'
    )
    parser.add_argument('mat_file', type=Path,
                       help='Original .mat file with full labeled volume')
    parser.add_argument('bundle_metadata', type=Path,
                       help='Bundle metadata JSON from preprocess_1')
    parser.add_argument('output_dir', type=Path,
                       help='Output directory for bundle HDF5 files')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                       help='Voxel size in micrometers (default: 0.05)')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of slices per batch (default: 50)')
    parser.add_argument('--n-workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')

    args = parser.parse_args()

    extract_all_bundles(
        args.mat_file,
        args.bundle_metadata,
        args.output_dir,
        voxel_size_um=args.voxel_size,
        batch_size=args.batch_size,
        n_workers=args.n_workers
    )
