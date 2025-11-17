#!/usr/bin/env python3
"""
Extract individual bundle volumes from bundle metadata.

Takes bundle metadata JSON (from preprocess_1) and creates axis-aligned
HDF5 volumes for each bundle, ready for oblique slicing analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Set, Tuple

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


def extract_bundle_volume_streaming(dataset: h5py.Dataset,
                                   axon_labels: Set[int],
                                   permutation: Tuple[int, int, int],
                                   output_file: Path,
                                   bundle: Dict,
                                   voxel_size_um: float = 0.05):
    """
    Extract and save bundle volume slice-by-slice (memory efficient).

    Args:
        dataset: HDF5 dataset reference (not loaded into memory)
        axon_labels: Set of axon labels in bundle
        permutation: Axis permutation tuple
        output_file: Output HDF5 file path
        bundle: Bundle metadata dictionary
        voxel_size_um: Physical voxel size
    """
    logger.info(f"Extracting {len(axon_labels)} axons from volume (streaming)...")

    original_shape = dataset.shape
    axon_labels_array = np.array(list(axon_labels), dtype=np.uint32)

    # Determine output shape after permutation
    permuted_shape = tuple(original_shape[i] for i in permutation)

    logger.info(f"Original shape: {original_shape}")
    logger.info(f"Permuted shape: {permuted_shape}")

    # Create output file with chunking
    output_file.parent.mkdir(parents=True, exist_ok=True)

    chunk_shape = (min(100, permuted_shape[0]),
                  min(512, permuted_shape[1]),
                  min(512, permuted_shape[2]))

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

        # Process slice-by-slice based on permutation
        # Read along axis that will become z after permutation
        read_axis = permutation.index(0)

        if read_axis == 0:
            # Reading z slices, they stay as z
            for z in tqdm(range(original_shape[0]), desc="Processing slices"):
                slice_2d = dataset[z, :, :]
                # Filter to bundle axons
                mask = np.isin(slice_2d, axon_labels_array)
                slice_2d = slice_2d.copy()
                slice_2d[~mask] = 0
                dset_out[z, :, :] = slice_2d

        elif read_axis == 1:
            # Reading y slices, they become z (permutation likely (1,0,2))
            for y in tqdm(range(original_shape[1]), desc="Processing slices"):
                slice_2d = dataset[:, y, :]
                mask = np.isin(slice_2d, axon_labels_array)
                slice_2d = slice_2d.copy()
                slice_2d[~mask] = 0
                # Write to output: y becomes new z
                dset_out[y, :, :] = slice_2d

        else:  # read_axis == 2
            # Reading x slices, they become z (permutation likely (2,1,0))
            for x in tqdm(range(original_shape[2]), desc="Processing slices"):
                slice_2d = dataset[:, :, x]
                mask = np.isin(slice_2d, axon_labels_array)
                slice_2d = slice_2d.copy()
                slice_2d[~mask] = 0
                # Write to output: x becomes new z
                dset_out[x, :, :] = slice_2d

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
                       voxel_size_um: float = 0.05):
    """
    Extract all bundles from volume and save as separate HDF5 files.

    Args:
        mat_file: Original .mat file with full labeled volume
        bundle_metadata_file: JSON file from preprocess_1
        output_dir: Directory for output HDF5 files
        voxel_size_um: Physical voxel size at full resolution
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
                dataset,
                axon_labels,
                permutation,
                output_file,
                bundle,
                voxel_size_um
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

    args = parser.parse_args()

    extract_all_bundles(
        args.mat_file,
        args.bundle_metadata,
        args.output_dir,
        voxel_size_um=args.voxel_size
    )
