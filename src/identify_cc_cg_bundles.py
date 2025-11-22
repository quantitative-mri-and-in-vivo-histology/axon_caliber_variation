#!/usr/bin/env python3
"""
Identify CC and CG fiber bundles using K-means clustering on orientation.

This script assumes exactly 2 major bundles (corpus callosum and cingulum) and:
1. Clusters axons by orientation using K-means (n_clusters=2)
2. Rejects orientation outliers (> N standard deviations from cluster mean)
3. Removes spatially isolated axons using per-bundle KNN filtering
4. Saves bundle metadata for downstream processing

Output format is identical to preprocess_1_identify_bundles.py for compatibility.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
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

    with h5py.File(mat_file, 'r') as f:
        volume_full = f['final_lbl'][()]
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

    Args:
        volume: Labeled 3D volume

    Returns:
        Dict mapping axon_id -> coordinates array [N, 3] in [z, y, x] order
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
    """
    n_voxels = len(coords)

    if n_voxels < 3:
        return np.array([0, 0, 0]), 0.0, n_voxels

    mean_pos = coords.mean(axis=0)
    centered = coords - mean_pos

    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    principal_axis = np.real(eigenvectors[:, np.argmax(eigenvalues)])
    principal_axis = principal_axis / np.linalg.norm(principal_axis)

    projections = np.dot(centered, principal_axis)
    length_pca = projections.max() - projections.min()

    bbox_extents = coords.max(axis=0) - coords.min(axis=0)
    length_bbox = float(bbox_extents.max())

    length_voxels = max(length_pca, length_bbox)

    return principal_axis, float(length_voxels), n_voxels


def compute_all_orientations(axon_voxels: Dict[int, np.ndarray],
                             voxel_size_um: float = 0.2,
                             min_voxels: int = 10,
                             min_length_um: float = 50.0) -> Dict[int, Dict]:
    """
    Compute orientations for all axons from pre-computed voxel coordinates.

    Args:
        axon_voxels: Dict mapping axon_id -> coordinates [N, 3]
        voxel_size_um: Physical voxel size in micrometers
        min_voxels: Minimum voxels per axon to analyze
        min_length_um: Minimum axon length in micrometers

    Returns:
        Dictionary mapping label -> {orientation, length_um, n_voxels, centroid_um}
    """
    logger.info("Computing orientations for all axons...")
    logger.info(f"Processing {len(axon_voxels)} axons...")

    axon_data = {}
    n_too_short = 0

    for label, coords in tqdm(axon_voxels.items(), desc="Analyzing axons"):
        orientation, length_voxels, n_voxels = compute_axon_orientation_from_coords(coords)

        if n_voxels < min_voxels:
            continue

        length_um = length_voxels * voxel_size_um

        if length_um < min_length_um:
            n_too_short += 1
            continue

        centroid_um = (coords.mean(axis=0) * voxel_size_um).tolist()

        axon_data[int(label)] = {
            'orientation': orientation.tolist(),
            'length_um': float(length_um),
            'n_voxels': int(n_voxels),
            'centroid_um': centroid_um
        }

    logger.info(f"Analyzed {len(axon_data)} axons with ≥{min_voxels} voxels and ≥{min_length_um:.1f} μm")
    if n_too_short > 0:
        logger.info(f"Rejected {n_too_short} axons shorter than {min_length_um:.1f} μm")

    return axon_data


def cluster_kmeans_2(axon_data: Dict[int, Dict], spatial_weight: float = 0.5) -> Dict[int, int]:
    """
    Cluster axons into exactly 2 bundles using K-means on orientation and position.

    Combines orientation vectors with spatial position (centroid) for clustering.
    Handles orientation sign ambiguity by mapping all orientations to the same hemisphere.

    Args:
        axon_data: Dictionary mapping label -> {orientation, length_um, n_voxels, centroid_um}
        spatial_weight: Weight for spatial position (0-1). 0 = orientation only, 1 = position only.

    Returns:
        Dictionary mapping axon_label -> bundle_id (1 or 2)
    """
    logger.info(f"Clustering axons into 2 bundles using K-means (spatial_weight={spatial_weight})...")

    labels = list(axon_data.keys())
    orientations = np.array([axon_data[label]['orientation'] for label in labels])
    centroids = np.array([axon_data[label]['centroid_um'] for label in labels])

    # Normalize orientations (filter out any with zero norm)
    norms = np.linalg.norm(orientations, axis=1, keepdims=True)
    valid_mask = (norms > 1e-10).flatten()
    if not np.all(valid_mask):
        n_invalid = (~valid_mask).sum()
        logger.warning(f"Filtering {n_invalid} axons with invalid orientations")
        labels = [labels[i] for i in range(len(labels)) if valid_mask[i]]
        orientations = orientations[valid_mask]
        centroids = centroids[valid_mask]
        norms = norms[valid_mask]
    orientations = orientations / norms

    # Handle sign ambiguity: map all to same hemisphere
    # Coordinates are in [z, y, x] order, so index 0 is z-component
    sign_flip = orientations[:, 0] < 0
    orientations[sign_flip] = -orientations[sign_flip]

    # Normalize positions to [0, 1] using min-max scaling
    pos_min = centroids.min(axis=0)
    pos_max = centroids.max(axis=0)
    pos_range = pos_max - pos_min
    pos_range[pos_range < 1e-10] = 1.0  # Avoid division by zero
    positions_normalized = (centroids - pos_min) / pos_range

    # Build feature vector with weighting
    # orientation weight = 1 - spatial_weight, position weight = spatial_weight
    orientation_weight = 1.0 - spatial_weight
    features = np.hstack([
        orientations * orientation_weight,
        positions_normalized * spatial_weight
    ])

    logger.info(f"  Feature vector: 3D orientation (weight {orientation_weight:.2f}) + "
               f"3D position (weight {spatial_weight:.2f})")

    # K-means clustering on 6D features
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(features)

    # Convert to 1-based bundle IDs
    label_to_bundle = {labels[i]: int(cluster_ids[i] + 1) for i in range(len(labels))}

    # Log cluster info
    for cluster_id in [0, 1]:
        mask = cluster_ids == cluster_id
        n_axons = mask.sum()
        center = kmeans.cluster_centers_[cluster_id]
        # Show orientation part of center
        orient_center = center[:3] / (orientation_weight + 1e-10)
        logger.info(f"  Cluster {cluster_id + 1}: {n_axons} axons, "
                   f"orientation [{orient_center[0]:.3f}, {orient_center[1]:.3f}, {orient_center[2]:.3f}]")

    return label_to_bundle


def filter_orientation_outliers(axon_data: Dict[int, Dict],
                                 label_to_bundle: Dict[int, int],
                                 std_threshold: float = 2.0) -> Tuple[Dict[int, int], int]:
    """
    Remove axons whose orientation deviates too much from their cluster mean.

    Args:
        axon_data: Dictionary mapping label -> {orientation, ...}
        label_to_bundle: Dictionary mapping axon_label -> bundle_id
        std_threshold: Number of standard deviations for outlier threshold

    Returns:
        (filtered_label_to_bundle, n_removed) tuple
    """
    logger.info(f"Filtering orientation outliers (>{std_threshold} std deviations)...")

    # Group axons by bundle
    bundles = {1: [], 2: []}
    for label, bundle_id in label_to_bundle.items():
        bundles[bundle_id].append(label)

    filtered_label_to_bundle = {}
    total_removed = 0

    for bundle_id, axon_labels in bundles.items():
        if not axon_labels:
            continue

        # Get orientations for this bundle
        orientations = np.array([axon_data[label]['orientation'] for label in axon_labels])
        orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)

        # Handle sign ambiguity
        sign_flip = orientations[:, 0] < 0
        orientations[sign_flip] = -orientations[sign_flip]

        # Compute mean orientation
        mean_orientation = orientations.mean(axis=0)
        norm = np.linalg.norm(mean_orientation)
        if norm < 1e-10:
            logger.warning(f"Bundle {bundle_id}: Mean orientation has near-zero norm, skipping")
            continue
        mean_orientation = mean_orientation / norm

        # Compute angular deviation for each axon
        # cos(angle) = dot(orientation, mean)
        cos_angles = np.abs(orientations @ mean_orientation)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles_rad = np.arccos(cos_angles)
        angles_deg = np.degrees(angles_rad)

        # Compute statistics
        mean_angle = angles_deg.mean()
        std_angle = angles_deg.std()

        # Filter by standard deviation
        threshold_angle = mean_angle + std_threshold * std_angle
        mask = angles_deg <= threshold_angle

        n_kept = mask.sum()
        n_removed = len(axon_labels) - n_kept
        total_removed += n_removed

        logger.info(f"  Bundle {bundle_id}: mean angle {mean_angle:.1f}°, "
                   f"std {std_angle:.1f}°, threshold {threshold_angle:.1f}°, "
                   f"removed {n_removed}/{len(axon_labels)} axons")

        # Keep only non-outliers
        for i, label in enumerate(axon_labels):
            if mask[i]:
                filtered_label_to_bundle[label] = bundle_id

    logger.info(f"Total orientation outliers removed: {total_removed}")

    return filtered_label_to_bundle, total_removed


def filter_sparse_axons(axon_labels: List[int],
                        axon_voxels: Dict[int, np.ndarray],
                        voxel_size_um: float,
                        k_neighbors: int = 10,
                        max_distance_um: float = 30.0) -> Tuple[List[int], int]:
    """
    Remove spatially isolated axons using KNN distance threshold.

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
        return axon_labels, 0

    centroids = np.array([axon_voxels[label].mean(axis=0) for label in axon_labels])
    centroids_um = centroids * voxel_size_um

    tree = KDTree(centroids_um)
    distances, _ = tree.query(centroids_um, k=k_neighbors + 1)

    kth_distances = distances[:, k_neighbors]

    mask = kth_distances <= max_distance_um
    filtered_labels = [axon_labels[i] for i in range(len(axon_labels)) if mask[i]]
    n_removed = len(axon_labels) - len(filtered_labels)

    return filtered_labels, n_removed


def create_bundles_cc_cg(axon_data: Dict[int, Dict],
                          label_to_bundle: Dict[int, int],
                          axon_voxels: Dict[int, np.ndarray],
                          voxel_size_um: float,
                          k_neighbors: int = 10,
                          max_neighbor_distance_um: float = 30.0) -> List[Dict]:
    """
    Create bundle metadata with per-bundle KNN filtering.

    Args:
        axon_data: Dictionary mapping label -> {orientation, length_um, n_voxels}
        label_to_bundle: Dictionary mapping axon_label -> bundle_id
        axon_voxels: Dict mapping axon_id -> coordinates [N, 3]
        voxel_size_um: Physical voxel size
        k_neighbors: Number of neighbors for sparse filtering
        max_neighbor_distance_um: Max distance to k-th neighbor

    Returns:
        List of bundle dictionaries with metadata
    """
    logger.info("Creating bundle metadata with per-bundle KNN filtering...")

    # Group axons by bundle
    bundles_raw = {1: [], 2: []}
    for label, bundle_id in label_to_bundle.items():
        bundles_raw[bundle_id].append(label)

    bundles = []
    total_sparse_removed = 0

    for bundle_id in [1, 2]:
        axon_labels = bundles_raw[bundle_id]

        if not axon_labels:
            logger.warning(f"Bundle {bundle_id} has no axons after orientation filtering")
            continue

        n_before = len(axon_labels)

        # Per-bundle KNN filtering
        axon_labels, n_removed = filter_sparse_axons(
            axon_labels,
            axon_voxels,
            voxel_size_um,
            k_neighbors=k_neighbors,
            max_distance_um=max_neighbor_distance_um
        )
        total_sparse_removed += n_removed

        if not axon_labels:
            logger.warning(f"Bundle {bundle_id} has no axons after KNN filtering")
            continue

        n_axons = len(axon_labels)

        # Compute mean orientation
        orientations = np.array([axon_data[label]['orientation'] for label in axon_labels])
        orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)

        # Handle sign ambiguity for mean computation
        sign_flip = orientations[:, 0] < 0
        orientations[sign_flip] = -orientations[sign_flip]

        mean_orientation = orientations.mean(axis=0)
        norm = np.linalg.norm(mean_orientation)
        if norm < 1e-10:
            logger.warning(f"Bundle {bundle_id}: Mean orientation has near-zero norm")
            mean_orientation = np.array([1.0, 0.0, 0.0])
        else:
            mean_orientation = mean_orientation / norm

        # Compute length statistics
        lengths = [axon_data[label]['length_um'] for label in axon_labels]
        mean_length = np.mean(lengths)
        median_length = np.median(lengths)
        p75_length = np.percentile(lengths, 75)

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

        bundles.append(bundle_info)

        logger.info(f"  Bundle {bundle_id}: {n_axons} axons "
                   f"(removed {n_removed} sparse from {n_before}), "
                   f"mean length {mean_length:.1f} μm")

    # Sort by number of axons (descending)
    bundles.sort(key=lambda b: b['n_axons'], reverse=True)

    logger.info(f"Total sparse axons removed: {total_sparse_removed}")

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


def identify_cc_cg_bundles(mat_file: Path,
                           output_file: Path,
                           downsample: int = 4,
                           voxel_size_um: float = 0.05,
                           orientation_std_threshold: float = 2.0,
                           spatial_weight: float = 0.5,
                           min_length_um: float = 50.0,
                           min_voxels: int = 10,
                           k_neighbors: int = 10,
                           max_neighbor_distance_um: float = 30.0) -> List[Dict]:
    """
    Complete pipeline to identify CC and CG fiber bundles.

    Args:
        mat_file: Input .mat file with labeled volume
        output_file: Output JSON file for bundle metadata
        downsample: Downsampling factor for orientation computation
        voxel_size_um: Physical voxel size at full resolution
        orientation_std_threshold: Std deviations for orientation outlier rejection
        spatial_weight: Weight for spatial position in clustering (0-1, default 0.5)
        min_length_um: Minimum axon length in micrometers (default 50.0)
        min_voxels: Minimum voxels to analyze axon
        k_neighbors: Number of neighbors for sparse axon filtering
        max_neighbor_distance_um: Max distance to k-th neighbor for sparse filtering

    Returns:
        List of bundle dictionaries
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Identifying CC and CG bundles: {mat_file.name}")
    logger.info(f"{'='*80}\n")

    # Load volume
    volume, metadata = load_volume_downsampled(mat_file, downsample)

    # Adjust voxel size for downsampling
    voxel_size_downsampled = voxel_size_um * downsample

    # Pre-compute voxel coordinates
    axon_voxels = precompute_axon_voxels(volume)

    # Compute orientations with length filtering
    axon_data = compute_all_orientations(
        axon_voxels, voxel_size_downsampled, min_voxels, min_length_um
    )

    # K-means clustering for exactly 2 bundles
    label_to_bundle = cluster_kmeans_2(axon_data, spatial_weight)

    # Filter orientation outliers
    label_to_bundle, _ = filter_orientation_outliers(
        axon_data, label_to_bundle, orientation_std_threshold
    )

    # Check that axons remain after filtering
    if not label_to_bundle:
        logger.error("All axons were filtered out. Check orientation_std_threshold parameter.")
        raise ValueError("No axons remain after orientation filtering")

    # Create bundles with per-bundle KNN filtering
    bundles = create_bundles_cc_cg(
        axon_data, label_to_bundle, axon_voxels, voxel_size_downsampled,
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
        description='Identify CC and CG fiber bundles using K-means clustering'
    )
    parser.add_argument('mat_file', type=Path,
                       help='Input .mat file with labeled volume')
    parser.add_argument('output_file', type=Path,
                       help='Output JSON file for bundle metadata')
    parser.add_argument('--downsample', type=int, default=4,
                       help='Downsampling factor for analysis (default: 4)')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                       help='Voxel size at full resolution in μm (default: 0.05)')
    parser.add_argument('--orientation-std-threshold', type=float, default=2.0,
                       help='Std deviations for orientation outlier rejection (default: 2.0)')
    parser.add_argument('--spatial-weight', type=float, default=0.5,
                       help='Weight for spatial position in clustering, 0-1 (default: 0.5)')
    parser.add_argument('--min-length', type=float, default=50.0,
                       help='Minimum axon length in μm (default: 50.0)')
    parser.add_argument('--min-voxels', type=int, default=10,
                       help='Minimum voxels per axon to analyze (default: 10)')
    parser.add_argument('--k-neighbors', type=int, default=10,
                       help='Number of neighbors for sparse axon filtering (default: 10)')
    parser.add_argument('--max-neighbor-distance', type=float, default=30.0,
                       help='Max distance to k-th neighbor in μm (default: 30.0)')

    args = parser.parse_args()

    bundles = identify_cc_cg_bundles(
        args.mat_file,
        args.output_file,
        downsample=args.downsample,
        voxel_size_um=args.voxel_size,
        orientation_std_threshold=args.orientation_std_threshold,
        spatial_weight=args.spatial_weight,
        min_length_um=args.min_length,
        min_voxels=args.min_voxels,
        k_neighbors=args.k_neighbors,
        max_neighbor_distance_um=args.max_neighbor_distance
    )
