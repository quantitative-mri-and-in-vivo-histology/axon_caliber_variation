#!/usr/bin/env python3
"""
Identify fiber bundles in 3D axon volumes using orientation-based clustering.

This script discovers fiber bundles without assuming specific tract identities (CC/CG).
Instead, it:
1. Computes orientation for each axon via PCA
2. Clusters axons by orientation similarity
3. Filters bundles by size (≥500 axons) and length (≥50 μm)
4. Saves bundle metadata for downstream processing

Output format enables processing each bundle independently with oblique slicing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_volume_downsampled(mat_file: Path, downsample: int = 4) -> Tuple[np.ndarray, Dict]:
    """
    Load labeled volume from .mat file (HDF5 format) with optional downsampling.

    Args:
        mat_file: Path to .mat file with 'labels' field
        downsample: Downsampling factor (default: 4)

    Returns:
        (volume, metadata) tuple
    """
    logger.info(f"Loading volume: {mat_file.name}")

    # MATLAB v7.3 files are HDF5-based
    with h5py.File(mat_file, 'r') as f:
        # Load final_lbl dataset
        volume_full = f['final_lbl'][()].astype(np.uint32)

        logger.info(f"Original shape: {volume_full.shape}")

        if downsample > 1:
            volume = volume_full[::downsample, ::downsample, ::downsample].copy()
            logger.info(f"Downsampled to: {volume.shape} (factor {downsample})")
        else:
            volume = volume_full.copy()

    metadata = {
        'original_shape': volume_full.shape,
        'downsampled_shape': volume.shape,
        'downsample_factor': downsample,
        'source_file': str(mat_file)
    }

    return volume, metadata


def precompute_axon_voxels(volume: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Pre-compute voxel coordinates for all axons in one pass.

    This is MUCH faster than calling np.argwhere for each axon separately,
    especially for sparse volumes.

    Args:
        volume: Labeled 3D volume

    Returns:
        Dict mapping axon_id -> coordinates array [N, 3] in [z, y, x] order
    """
    logger.info("Pre-computing voxel coordinates for all axons...")
    import time
    start = time.time()

    # Find all non-zero voxels in one pass
    all_coords = np.argwhere(volume > 0)

    # Group by axon ID
    axon_voxels = {}
    for coord in all_coords:
        axon_id = volume[tuple(coord)]
        if axon_id not in axon_voxels:
            axon_voxels[axon_id] = []
        axon_voxels[axon_id].append(coord)

    # Convert to numpy arrays
    axon_voxels = {aid: np.array(coords) for aid, coords in axon_voxels.items()}

    elapsed = time.time() - start
    logger.info(f"Pre-computed coordinates for {len(axon_voxels)} axons in {elapsed:.2f}s")

    return axon_voxels


def compute_axon_orientation_from_coords(coords: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """
    Compute principal orientation of a single axon via PCA from pre-computed coordinates.

    Args:
        coords: Voxel coordinates [N, 3] in [z, y, x] order

    Returns:
        (orientation_vector, length_voxels, n_voxels)
        orientation_vector: [vz, vy, vx] normalized direction
        length_voxels: extent along principal axis
        n_voxels: number of voxels in axon
    """
    n_voxels = len(coords)

    if n_voxels < 3:
        return np.array([0, 0, 0]), 0.0, n_voxels

    # Center coordinates
    mean_pos = coords.mean(axis=0)
    centered = coords - mean_pos

    # PCA
    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Principal direction (first eigenvector)
    principal_axis = np.real(eigenvectors[:, np.argmax(eigenvalues)])

    # Normalize
    principal_axis = principal_axis / np.linalg.norm(principal_axis)

    # Compute length (extent along principal axis)
    projections = np.dot(centered, principal_axis)
    length_voxels = projections.max() - projections.min()

    return principal_axis, float(length_voxels), n_voxels


def compute_all_orientations(volume: np.ndarray,
                            voxel_size_um: float = 0.2,
                            min_voxels: int = 10) -> Dict[int, Dict]:
    """
    Compute orientations for all axons in volume.

    Args:
        volume: Labeled 3D volume
        voxel_size_um: Physical voxel size in micrometers
        min_voxels: Minimum voxels per axon to analyze

    Returns:
        Dictionary mapping label -> {orientation, length_um, n_voxels}
    """
    logger.info("Computing orientations for all axons...")

    # Pre-compute all voxel coordinates (much faster!)
    axon_voxels = precompute_axon_voxels(volume)

    logger.info(f"Processing {len(axon_voxels)} axons...")

    axon_data = {}

    for label, coords in tqdm(axon_voxels.items(), desc="Analyzing axons"):
        orientation, length_voxels, n_voxels = compute_axon_orientation_from_coords(coords)

        if n_voxels < min_voxels:
            continue

        length_um = length_voxels * voxel_size_um

        axon_data[int(label)] = {
            'orientation': orientation.tolist(),
            'length_um': float(length_um),
            'n_voxels': int(n_voxels)
        }

    logger.info(f"Analyzed {len(axon_data)} axons with ≥{min_voxels} voxels")

    return axon_data


def cluster_by_orientation(axon_data: Dict[int, Dict],
                          orientation_threshold: float = 0.3) -> Dict[int, int]:
    """
    Cluster axons by orientation similarity using hierarchical clustering.

    Args:
        axon_data: Dictionary mapping label -> {orientation, length_um, n_voxels}
        orientation_threshold: Distance threshold for clustering (0-2 range)
                              Lower = more similar orientations required
                              Default 0.3 corresponds to ~30° max angle difference

    Returns:
        Dictionary mapping axon_label -> bundle_id
    """
    logger.info("Clustering axons by orientation...")
    import time
    start = time.time()

    labels = list(axon_data.keys())
    orientations = np.array([axon_data[label]['orientation'] for label in labels])

    # Normalize orientations
    orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)

    # Compute pairwise distances using condensed format (memory efficient!)
    # Distance metric: distance = sqrt(2 * (1 - |dot(a, b)|))
    # This treats opposite directions as equivalent (axon orientation is bidirectional)
    logger.info(f"Computing pairwise distances for {len(orientations)} axons...")

    # Use pdist to compute only the upper triangle - saves 50% memory!
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, fcluster

    def orientation_distance(u, v):
        """Distance metric for orientation clustering."""
        dot_product = abs(np.dot(u, v))
        dot_product = min(dot_product, 1.0)  # Numerical safety
        return np.sqrt(2 * (1 - dot_product))

    # Compute condensed distance matrix directly (only N*(N-1)/2 elements)
    condensed_distances = pdist(orientations, metric=orientation_distance)

    logger.info(f"Distance computation: {time.time() - start:.2f}s")

    logger.info("Running hierarchical clustering...")
    start_cluster = time.time()

    # Compute linkage using pre-computed distances
    linkage_matrix = linkage(condensed_distances, method='average')

    # Extract flat clusters
    cluster_ids = fcluster(linkage_matrix, t=orientation_threshold, criterion='distance')

    logger.info(f"Clustering: {time.time() - start_cluster:.2f}s")

    # Map labels to bundle IDs
    label_to_bundle = {labels[i]: int(cluster_ids[i]) for i in range(len(labels))}

    n_bundles = len(np.unique(cluster_ids))
    logger.info(f"Identified {n_bundles} fiber bundles")
    logger.info(f"Total clustering time: {time.time() - start:.2f}s")

    return label_to_bundle


def create_bundles(axon_data: Dict[int, Dict],
                  label_to_bundle: Dict[int, int],
                  min_axons: int = 500,
                  min_length_um: float = 50.0) -> List[Dict]:
    """
    Create bundle metadata with filtering.

    Args:
        axon_data: Dictionary mapping label -> {orientation, length_um, n_voxels}
        label_to_bundle: Dictionary mapping axon_label -> bundle_id
        min_axons: Minimum axons per bundle
        min_length_um: Minimum bundle length in micrometers

    Returns:
        List of bundle dictionaries with metadata
    """
    logger.info("Creating bundle metadata...")

    # Group axons by bundle
    bundles_raw = {}
    for label, bundle_id in label_to_bundle.items():
        if bundle_id not in bundles_raw:
            bundles_raw[bundle_id] = []
        bundles_raw[bundle_id].append(label)

    # Process each bundle
    bundles = []

    for bundle_id, axon_labels in bundles_raw.items():
        n_axons = len(axon_labels)

        # Compute mean orientation
        orientations = np.array([axon_data[label]['orientation'] for label in axon_labels])
        mean_orientation = orientations.mean(axis=0)
        mean_orientation = mean_orientation / np.linalg.norm(mean_orientation)

        # Compute mean length
        lengths = [axon_data[label]['length_um'] for label in axon_labels]
        mean_length = np.mean(lengths)

        # Apply filters
        if n_axons < min_axons:
            continue

        if mean_length < min_length_um:
            continue

        # Create bundle metadata
        bundle = {
            'bundle_id': int(bundle_id),
            'n_axons': int(n_axons),
            'mean_orientation': mean_orientation.tolist(),
            'mean_length_um': float(mean_length),
            'axon_labels': [int(label) for label in axon_labels],
            'length_range_um': [float(min(lengths)), float(max(lengths))],
            'orientation_std': float(np.std(orientations, axis=0).mean())
        }

        bundles.append(bundle)

    # Sort by number of axons (descending)
    bundles.sort(key=lambda b: b['n_axons'], reverse=True)

    logger.info(f"Found {len(bundles)} bundles meeting criteria:")
    logger.info(f"  - Minimum {min_axons} axons")
    logger.info(f"  - Minimum {min_length_um} μm length")

    for i, bundle in enumerate(bundles):
        logger.info(f"  Bundle {i+1}: {bundle['n_axons']} axons, "
                   f"mean length {bundle['mean_length_um']:.1f} μm, "
                   f"orientation [{bundle['mean_orientation'][0]:.3f}, "
                   f"{bundle['mean_orientation'][1]:.3f}, "
                   f"{bundle['mean_orientation'][2]:.3f}]")

    return bundles


def save_bundle_metadata(bundles: List[Dict],
                        metadata: Dict,
                        output_file: Path):
    """
    Save bundle metadata to JSON file.

    Args:
        bundles: List of bundle dictionaries
        metadata: Volume metadata
        output_file: Output JSON file path
    """
    output_data = {
        'volume_metadata': metadata,
        'n_bundles': len(bundles),
        'bundles': bundles
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved bundle metadata: {output_file}")


def identify_bundles(mat_file: Path,
                    output_file: Path,
                    downsample: int = 4,
                    voxel_size_um: float = 0.05,
                    min_axons: int = 500,
                    min_length_um: float = 50.0,
                    orientation_threshold: float = 0.3,
                    min_voxels: int = 10) -> List[Dict]:
    """
    Complete pipeline to identify fiber bundles.

    Args:
        mat_file: Input .mat file with labeled volume
        output_file: Output JSON file for bundle metadata
        downsample: Downsampling factor for orientation computation
        voxel_size_um: Physical voxel size at full resolution
        min_axons: Minimum axons per bundle
        min_length_um: Minimum bundle length
        orientation_threshold: Clustering threshold (lower = stricter)
        min_voxels: Minimum voxels to analyze axon

    Returns:
        List of bundle dictionaries
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Identifying fiber bundles: {mat_file.name}")
    logger.info(f"{'='*80}\n")

    # Load volume
    volume, metadata = load_volume_downsampled(mat_file, downsample)

    # Adjust voxel size for downsampling
    voxel_size_downsampled = voxel_size_um * downsample

    # Compute orientations
    axon_data = compute_all_orientations(volume, voxel_size_downsampled, min_voxels)

    # Cluster by orientation
    label_to_bundle = cluster_by_orientation(axon_data, orientation_threshold)

    # Create and filter bundles
    bundles = create_bundles(axon_data, label_to_bundle, min_axons, min_length_um)

    # Save metadata
    save_bundle_metadata(bundles, metadata, output_file)

    logger.info(f"\n{'='*80}")
    logger.info(f"Bundle identification complete!")
    logger.info(f"{'='*80}\n")

    return bundles


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Identify fiber bundles using orientation-based clustering'
    )
    parser.add_argument('mat_file', type=Path,
                       help='Input .mat file with labeled volume')
    parser.add_argument('output_file', type=Path,
                       help='Output JSON file for bundle metadata')
    parser.add_argument('--downsample', type=int, default=4,
                       help='Downsampling factor for analysis (default: 4)')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                       help='Voxel size at full resolution in μm (default: 0.05)')
    parser.add_argument('--min-axons', type=int, default=500,
                       help='Minimum axons per bundle (default: 500)')
    parser.add_argument('--min-length', type=float, default=50.0,
                       help='Minimum bundle length in μm (default: 50.0)')
    parser.add_argument('--orientation-threshold', type=float, default=0.3,
                       help='Orientation clustering threshold (default: 0.3)')

    args = parser.parse_args()

    bundles = identify_bundles(
        args.mat_file,
        args.output_file,
        downsample=args.downsample,
        voxel_size_um=args.voxel_size,
        min_axons=args.min_axons,
        min_length_um=args.min_length,
        orientation_threshold=args.orientation_threshold
    )
