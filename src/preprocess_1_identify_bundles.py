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
from scipy.spatial import KDTree
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
    length_pca = projections.max() - projections.min()

    # For downsampled/sparse data, bounding box extent is more robust
    # Use maximum extent along any axis as a conservative length estimate
    bbox_extents = coords.max(axis=0) - coords.min(axis=0)
    length_bbox = float(bbox_extents.max())

    # Use the larger of the two (PCA can underestimate for sparse data)
    length_voxels = max(length_pca, length_bbox)

    return principal_axis, float(length_voxels), n_voxels


def compute_all_orientations_from_voxels(axon_voxels: Dict[int, np.ndarray],
                                          voxel_size_um: float = 0.2,
                                          min_voxels: int = 10,
                                          min_length_um: float = 0.0) -> Dict[int, Dict]:
    """
    Compute orientations for all axons from pre-computed voxel coordinates.

    Args:
        axon_voxels: Dict mapping axon_id -> coordinates [N, 3]
        voxel_size_um: Physical voxel size in micrometers
        min_voxels: Minimum voxels per axon to analyze
        min_length_um: Minimum axon length in micrometers (filters before clustering)

    Returns:
        Dictionary mapping label -> {orientation, length_um, n_voxels}
    """
    logger.info("Computing orientations for all axons...")
    logger.info(f"Processing {len(axon_voxels)} axons...")

    axon_data = {}
    rejected_too_short = 0

    for label, coords in tqdm(axon_voxels.items(), desc="Analyzing axons"):
        orientation, length_voxels, n_voxels = compute_axon_orientation_from_coords(coords)

        if n_voxels < min_voxels:
            continue

        length_um = length_voxels * voxel_size_um

        # Filter by minimum length BEFORE clustering
        if length_um < min_length_um:
            rejected_too_short += 1
            continue

        axon_data[int(label)] = {
            'orientation': orientation.tolist(),
            'length_um': float(length_um),
            'n_voxels': int(n_voxels)
        }

    logger.info(f"Analyzed {len(axon_data)} axons with ≥{min_voxels} voxels and ≥{min_length_um:.1f} μm length")
    if rejected_too_short > 0:
        logger.info(f"Rejected {rejected_too_short} axons shorter than {min_length_um:.1f} μm")

    return axon_data


def cluster_by_orientation(axon_data: Dict[int, Dict],
                          orientation_threshold: float = 0.7) -> Dict[int, int]:
    """
    Cluster axons by orientation similarity using hierarchical clustering.

    Args:
        axon_data: Dictionary mapping label -> {orientation, length_um, n_voxels}
        orientation_threshold: Distance threshold for clustering (0-1.4 range)
                              Higher = more permissive clustering
                              0.7 (default) = ~45° tolerance, good for 2-3 main bundles
                              0.3 = very strict (~17°), creates many small clusters

    Returns:
        Dictionary mapping axon_label -> bundle_id
    """
    logger.info("Clustering axons by orientation...")
    logger.info(f"Using orientation_threshold = {orientation_threshold}")
    import time
    start = time.time()

    labels = list(axon_data.keys())
    orientations = np.array([axon_data[label]['orientation'] for label in labels])

    # Normalize orientations
    orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)

    # Compute pairwise distances using condensed format (memory efficient!)
    # Distance metric: distance = sqrt(2 * (1 - |dot(a, b)|))
    # This treats opposite directions as equivalent (axon orientation is bidirectional)
    n = len(orientations)
    logger.info(f"Computing pairwise distances for {n} axons (~{n*(n-1)//2:,} pairs)...")

    from scipy.cluster.hierarchy import fcluster, linkage

    # For large N, compute distances in batches using vectorized operations
    if n > 10000:
        logger.info("Using vectorized batch computation for large dataset...")
        # Pre-allocate condensed distance array
        n_pairs = n * (n - 1) // 2
        condensed_distances = np.zeros(n_pairs, dtype=np.float32)

        idx = 0
        for i in tqdm(range(n), desc="Computing distances", disable=n<1000):
            # Compute distances from point i to all points j > i
            if i + 1 < n:
                remaining = orientations[i+1:]
                # Vectorized computation for this row
                dots = np.abs(remaining @ orientations[i])
                dots = np.clip(dots, 0.0, 1.0)
                dists = np.sqrt(2 * (1 - dots)).astype(np.float32)

                n_dists = len(dists)
                condensed_distances[idx:idx+n_dists] = dists
                idx += n_dists
    else:
        # For smaller datasets, use pdist with custom metric
        from scipy.spatial.distance import pdist

        def orientation_distance(u, v):
            """Distance metric for orientation clustering."""
            dot_product = abs(np.dot(u, v))
            dot_product = min(dot_product, 1.0)
            return np.sqrt(2 * (1 - dot_product))

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


def filter_sparse_axons(axon_labels: List[int],
                        axon_voxels: Dict[int, np.ndarray],
                        voxel_size_um: float,
                        k_neighbors: int = 10,
                        max_distance_um: float = 30.0) -> Tuple[List[int], int]:
    """
    Remove spatially isolated axons using KNN distance threshold.

    Axons whose k-th nearest neighbor is farther than max_distance_um are removed.

    Args:
        axon_labels: List of axon labels in this bundle
        axon_voxels: Dict mapping axon_id -> coordinates [N, 3]
        voxel_size_um: Physical voxel size
        k_neighbors: Number of neighbors to check
        max_distance_um: Maximum allowed distance to k-th neighbor

    Returns:
        (filtered_labels, n_removed) tuple
    """
    if len(axon_labels) <= k_neighbors:
        # Not enough axons to apply filter
        return axon_labels, 0

    # Compute centroids for each axon
    centroids = np.array([axon_voxels[label].mean(axis=0) for label in axon_labels])
    centroids_um = centroids * voxel_size_um

    # Build KDTree and query k+1 neighbors (includes self)
    tree = KDTree(centroids_um)
    distances, _ = tree.query(centroids_um, k=k_neighbors + 1)

    # Distance to k-th neighbor (column k, since column 0 is self with distance 0)
    kth_distances = distances[:, k_neighbors]

    # Keep axons where k-th neighbor is within threshold
    mask = kth_distances <= max_distance_um
    filtered_labels = [axon_labels[i] for i in range(len(axon_labels)) if mask[i]]
    n_removed = len(axon_labels) - len(filtered_labels)

    return filtered_labels, n_removed


def create_bundles(axon_data: Dict[int, Dict],
                  label_to_bundle: Dict[int, int],
                  axon_voxels: Dict[int, np.ndarray],
                  voxel_size_um: float,
                  min_axons: int = 500,
                  min_length_um: float = 50.0,
                  k_neighbors: int = 10,
                  max_neighbor_distance_um: float = 30.0) -> List[Dict]:
    """
    Create bundle metadata with filtering.

    Args:
        axon_data: Dictionary mapping label -> {orientation, length_um, n_voxels}
        label_to_bundle: Dictionary mapping axon_label -> bundle_id
        axon_voxels: Dict mapping axon_id -> coordinates [N, 3]
        voxel_size_um: Physical voxel size
        min_axons: Minimum axons per bundle
        min_length_um: Minimum bundle length in micrometers
        k_neighbors: Number of neighbors for sparse filtering
        max_neighbor_distance_um: Max distance to k-th neighbor

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
    rejected_bundles = []
    total_sparse_removed = 0

    for bundle_id, axon_labels in bundles_raw.items():
        n_axons_before = len(axon_labels)

        # Filter sparse axons (per-bundle)
        axon_labels, n_removed = filter_sparse_axons(
            axon_labels,
            axon_voxels,
            voxel_size_um,
            k_neighbors=k_neighbors,
            max_distance_um=max_neighbor_distance_um
        )
        total_sparse_removed += n_removed

        n_axons = len(axon_labels)

        # Skip if all axons were removed by sparse filtering
        if n_axons == 0:
            continue

        # Compute mean orientation
        orientations = np.array([axon_data[label]['orientation'] for label in axon_labels])
        mean_orientation = orientations.mean(axis=0)
        mean_orientation = mean_orientation / np.linalg.norm(mean_orientation)

        # Compute length statistics
        lengths = [axon_data[label]['length_um'] for label in axon_labels]
        mean_length = np.mean(lengths)
        median_length = np.median(lengths)
        p75_length = np.percentile(lengths, 75)  # 75th percentile - more robust to outliers

        # Create bundle metadata (before filtering)
        bundle_info = {
            'bundle_id': int(bundle_id),
            'n_axons': int(n_axons),
            'mean_orientation': mean_orientation.tolist(),
            'mean_length_um': float(mean_length),
            'median_length_um': float(median_length),
            'p75_length_um': float(p75_length),
            'axon_labels': [int(label) for label in axon_labels],
            'length_range_um': [float(min(lengths)), float(max(lengths))],
            'orientation_std': float(np.std(orientations, axis=0).mean())
        }

        # Apply filters
        # NOTE: Length filtering is disabled when using downsampled volumes because
        # downsampling fragments axons. Only axon count is a reliable discriminator.
        rejected_reason = None
        if n_axons < min_axons:
            rejected_reason = f"too few axons ({n_axons} < {min_axons})"
        # Length filter commented out - unreliable on downsampled data
        # elif median_length < min_length_um:
        #     rejected_reason = f"too short (median {median_length:.1f} < {min_length_um} μm)"

        if rejected_reason:
            bundle_info['rejected_reason'] = rejected_reason
            rejected_bundles.append(bundle_info)
        else:
            bundles.append(bundle_info)

    # Sort by number of axons (descending)
    bundles.sort(key=lambda b: b['n_axons'], reverse=True)
    rejected_bundles.sort(key=lambda b: b['n_axons'], reverse=True)

    logger.info(f"Bundle filtering results:")
    logger.info(f"  Total clusters from hierarchical clustering: {len(bundles_raw)}")
    logger.info(f"  Sparse axons removed (k={k_neighbors}, max_dist={max_neighbor_distance_um}μm): {total_sparse_removed}")
    logger.info(f"  Rejected bundles: {len(rejected_bundles)}")
    logger.info(f"  Final bundles meeting criteria: {len(bundles)}")

    if rejected_bundles:
        logger.info(f"\nRejected bundle details:")
        for rb in rejected_bundles:
            logger.info(f"  Bundle {rb['bundle_id']}: {rb['n_axons']} axons, "
                       f"median {rb['median_length_um']:.1f} μm, mean {rb['mean_length_um']:.1f} μm "
                       f"(range: {rb['length_range_um'][0]:.1f}-{rb['length_range_um'][1]:.1f} μm) - "
                       f"{rb['rejected_reason']}")

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
                    min_length_um: float = 10.0,
                    orientation_threshold: float = 0.7,
                    min_voxels: int = 10,
                    k_neighbors: int = 10,
                    max_neighbor_distance_um: float = 30.0) -> List[Dict]:
    """
    Complete pipeline to identify fiber bundles.

    Args:
        mat_file: Input .mat file with labeled volume
        output_file: Output JSON file for bundle metadata
        downsample: Downsampling factor for orientation computation
        voxel_size_um: Physical voxel size at full resolution
        min_axons: Minimum axons per bundle
        min_length_um: Minimum bundle length
        orientation_threshold: Clustering threshold, range 0-1.4 (higher = more permissive)
                              0.3 = very strict (~17° tolerance)
                              0.7 = moderate (~45° tolerance, recommended)
                              1.0 = permissive (~60° tolerance)
        min_voxels: Minimum voxels to analyze axon
        k_neighbors: Number of neighbors for sparse axon filtering
        max_neighbor_distance_um: Max distance to k-th neighbor for sparse filtering

    Returns:
        List of bundle dictionaries
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Identifying fiber bundles: {mat_file.name}")
    logger.info(f"{'='*80}\n")

    # Load volume
    volume, metadata = load_volume_downsampled(mat_file, downsample)

    logger.info(f"Full resolution volume shape: {metadata['original_shape']}")
    logger.info(f"Downsampled volume shape: {metadata['downsampled_shape']}")

    # Adjust voxel size for downsampling
    voxel_size_downsampled = voxel_size_um * downsample

    # Pre-compute voxel coordinates (needed for orientation and sparse filtering)
    axon_voxels = precompute_axon_voxels(volume)

    # Compute orientations and filter short axons BEFORE clustering
    axon_data = compute_all_orientations_from_voxels(
        axon_voxels, voxel_size_downsampled, min_voxels, min_length_um
    )

    # Cluster by orientation (only long axons remain)
    label_to_bundle = cluster_by_orientation(axon_data, orientation_threshold)

    # Create bundles with sparse axon filtering
    bundles = create_bundles(
        axon_data, label_to_bundle, axon_voxels, voxel_size_downsampled,
        min_axons, min_length_um=0.0,
        k_neighbors=k_neighbors, max_neighbor_distance_um=max_neighbor_distance_um
    )

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
    parser.add_argument('--orientation-threshold', type=float, default=0.7,
                       help='Orientation clustering threshold, 0-1.4 range (default: 0.7, ~45° tolerance)')
    parser.add_argument('--k-neighbors', type=int, default=10,
                       help='Number of neighbors for sparse axon filtering (default: 10)')
    parser.add_argument('--max-neighbor-distance', type=float, default=30.0,
                       help='Max distance to k-th neighbor in μm (default: 30.0)')

    args = parser.parse_args()

    bundles = identify_bundles(
        args.mat_file,
        args.output_file,
        downsample=args.downsample,
        voxel_size_um=args.voxel_size,
        min_axons=args.min_axons,
        min_length_um=args.min_length,
        orientation_threshold=args.orientation_threshold,
        k_neighbors=args.k_neighbors,
        max_neighbor_distance_um=args.max_neighbor_distance
    )
