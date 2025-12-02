"""
Population identification for fiber bundles using dominant-axis classification.

This module provides functions for identifying and classifying axon populations
(e.g., CC and CG) based on their principal orientations.
"""

import logging
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from .io import load_volume_with_metadata

logger = logging.getLogger(__name__)


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

    # Use axonometry library instead of direct h5py
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


def compute_axon_orientation_from_coords(coords: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """
    Compute principal orientation via PCA.

    Args:
        coords: Voxel coordinates (N, 3)

    Returns:
        Tuple of (orientation, length_voxels, n_voxels)
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
                             voxel_size_um: float,
                             min_voxels: int = 10,
                             min_length_um: float = 50.0) -> Dict[int, Dict]:
    """
    Compute orientations for all axons with length filtering.

    Args:
        axon_voxels: Dictionary mapping axon label to voxel coordinates
        voxel_size_um: Voxel size in micrometers
        min_voxels: Minimum number of voxels (default: 10)
        min_length_um: Minimum axon length in micrometers (default: 50.0)

    Returns:
        Dictionary mapping axon label to orientation data
    """
    logger.info(f"Computing orientations for {len(axon_voxels)} axons...")

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


def classify_by_dominant_axis(axon_data: Dict[int, Dict],
                              max_angle_deg: float = 30.0) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Classify axons by dominant axis alignment.

    Each axon is assigned to the axis (0=Z, 1=Y, 2=X) with which it aligns best.
    CC is assigned to the axis with most axons, CG to the second-most.

    Args:
        axon_data: Dict mapping axon labels to orientation data
        max_angle_deg: Maximum deviation from axis to be included (default: 30.0)

    Returns:
        Tuple of (label_to_bundle, axis_to_bundle) where:
        - label_to_bundle maps axon label -> bundle ID (1=CC, 2=CG)
        - axis_to_bundle maps axis index -> bundle ID
    """
    logger.info(f"Classifying axons by dominant axis (max angle: {max_angle_deg}°)...")

    cos_threshold = np.cos(np.deg2rad(max_angle_deg))

    # Classify each axon by dominant axis
    axis_counts = {0: 0, 1: 0, 2: 0}  # Z, Y, X
    axon_to_axis = {}
    n_rejected = 0

    for label, data in axon_data.items():
        orient = np.array(data['orientation'])
        norm = np.linalg.norm(orient)
        if norm < 1e-10:
            continue
        orient = orient / norm
        abs_orient = np.abs(orient)

        # Find dominant axis
        dominant_axis = int(np.argmax(abs_orient))
        alignment = abs_orient[dominant_axis]

        # Check if alignment exceeds threshold
        if alignment >= cos_threshold:
            axon_to_axis[label] = dominant_axis
            axis_counts[dominant_axis] += 1
        else:
            n_rejected += 1

    # Sort axes by count (descending)
    sorted_axes = sorted(axis_counts.keys(), key=lambda a: axis_counts[a], reverse=True)

    # Assign: CC = most populated axis, CG = second most
    axis_to_bundle = {
        sorted_axes[0]: 1,  # CC
        sorted_axes[1]: 2,  # CG
        sorted_axes[2]: 0   # Unclassified (third axis)
    }

    # Log axis statistics
    axis_names = {0: 'Z', 1: 'Y', 2: 'X'}
    logger.info(f"  Axis counts:")
    for axis in sorted_axes:
        bundle_name = {1: 'CC', 2: 'CG', 0: 'unclassified'}[axis_to_bundle[axis]]
        logger.info(f"    {axis_names[axis]}-axis: {axis_counts[axis]} axons -> {bundle_name}")
    if n_rejected > 0:
        logger.info(f"  Rejected {n_rejected} axons (angle > {max_angle_deg}°)")

    # Build label_to_bundle mapping (only CC and CG, not third axis)
    label_to_bundle = {}
    for label, axis in axon_to_axis.items():
        bundle_id = axis_to_bundle[axis]
        if bundle_id > 0:  # Only include CC (1) and CG (2)
            label_to_bundle[label] = bundle_id

    logger.info(f"  Total classified: {len(label_to_bundle)} axons")

    return label_to_bundle, axis_to_bundle


def filter_sparse_axons(axon_labels: List[int],
                        axon_voxels: Dict[int, np.ndarray],
                        voxel_size_um: float,
                        k_neighbors: int = 10,
                        max_distance_um: float = 30.0) -> Tuple[List[int], int]:
    """
    Remove spatially isolated axons using KNN.

    Args:
        axon_labels: List of axon labels to filter
        axon_voxels: Dictionary mapping axon label to voxel coordinates
        voxel_size_um: Voxel size in micrometers
        k_neighbors: Number of nearest neighbors to check (default: 10)
        max_distance_um: Maximum distance to k-th neighbor (default: 30.0)

    Returns:
        Tuple of (filtered_labels, n_removed)
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
    return filtered_labels, len(axon_labels) - len(filtered_labels)


def create_populations(axon_data: Dict[int, Dict],
                      label_to_bundle: Dict[int, int],
                      axis_to_bundle: Dict[int, int],
                      axon_voxels: Dict[int, np.ndarray],
                      voxel_size_um: float,
                      k_neighbors: int = 10,
                      max_neighbor_distance_um: float = 30.0) -> List[Dict]:
    """
    Create population metadata with per-population KNN filtering.

    Args:
        axon_data: Dictionary mapping axon label to orientation data
        label_to_bundle: Dictionary mapping axon label to bundle ID
        axis_to_bundle: Dictionary mapping axis index to bundle ID
        axon_voxels: Dictionary mapping axon label to voxel coordinates
        voxel_size_um: Voxel size in micrometers
        k_neighbors: Number of nearest neighbors for filtering (default: 10)
        max_neighbor_distance_um: Maximum distance for filtering (default: 30.0)

    Returns:
        List of population dictionaries
    """
    logger.info("Creating population metadata...")

    # Invert axis_to_bundle mapping to get bundle_to_axis
    bundle_to_axis = {bundle_id: axis for axis, bundle_id in axis_to_bundle.items()}
    axis_names = {0: 'Z', 1: 'Y', 2: 'X'}

    populations_raw = {1: [], 2: []}
    for label, bundle_id in label_to_bundle.items():
        populations_raw[bundle_id].append(label)

    populations = []
    for bundle_id in [1, 2]:
        axon_labels = populations_raw[bundle_id]
        if not axon_labels:
            continue

        # Apply KNN filtering
        axon_labels, n_removed = filter_sparse_axons(
            axon_labels, axon_voxels, voxel_size_um,
            k_neighbors=k_neighbors, max_distance_um=max_neighbor_distance_um
        )

        if not axon_labels:
            continue

        # Compute mean orientation
        orientations = np.array([axon_data[label]['orientation'] for label in axon_labels])
        orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)

        # Flip orientations to point in same direction
        sign_flip = orientations[:, 0] < 0
        orientations[sign_flip] = -orientations[sign_flip]

        mean_orientation = orientations.mean(axis=0)
        norm = np.linalg.norm(mean_orientation)
        mean_orientation = mean_orientation / norm if norm > 1e-10 else np.array([1, 0, 0])

        lengths = [axon_data[label]['length_um'] for label in axon_labels]

        population_name = 'cc' if bundle_id == 1 else 'cg'
        dominant_axis = bundle_to_axis[bundle_id]
        population_info = {
            'name': population_name,
            'n_axons': len(axon_labels),
            'dominant_axis': int(dominant_axis),
            'dominant_axis_name': axis_names[dominant_axis],
            'mean_orientation': mean_orientation.tolist(),
            'mean_length_um': float(np.mean(lengths)),
            'median_length_um': float(np.median(lengths)),
            'axon_labels': [int(label) for label in axon_labels],
            'length_range_um': [float(min(lengths)), float(max(lengths))]
        }
        populations.append(population_info)

        logger.info(f"  {population_name.upper()}: {len(axon_labels)} axons (removed {n_removed} sparse), axis={axis_names[dominant_axis]}")

    # Sort by axon count descending (CC should be first)
    populations.sort(key=lambda p: p['n_axons'], reverse=True)
    return populations
