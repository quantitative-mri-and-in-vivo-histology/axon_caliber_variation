#!/usr/bin/env python3
"""
Segment volume into CC and CG spatial subvolumes using a learned boundary.

Uses orientation-classified axons as seeds to learn a curved spatial boundary
(SVM with RBF kernel), then assigns ALL axons based on their spatial location.
This captures the complete tissue including noise and imperfectly aligned axons.

Output structure:
    output.zarr/
    ├── cc/0, cc/1, ...  (pyramid levels for corpus callosum region)
    ├── cg/0, cg/1, ...  (pyramid levels for cingulum region)
    └── .zattrs          (combined metadata with boundary info)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import h5py
import numcodecs
import numpy as np
import zarr
from scipy.spatial import KDTree
from sklearn.svm import SVC
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Volume Loading and Axon Analysis (reused from extract_cc_cg_bundles.py)
# =============================================================================

def load_volume_downsampled(mat_file: Path, downsample: int = 4) -> Tuple[np.ndarray, Dict]:
    """Load labeled volume with optional downsampling."""
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
        'original_shape': list(volume_full.shape),
        'downsampled_shape': list(volume.shape),
        'downsample_factor': downsample,
        'source_file': str(mat_file)
    }

    return volume, metadata


def precompute_axon_voxels(volume: np.ndarray) -> Dict[int, np.ndarray]:
    """Pre-compute voxel coordinates for all axons."""
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
    """Compute principal orientation via PCA."""
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


def compute_all_axon_data(axon_voxels: Dict[int, np.ndarray],
                          voxel_size_um: float,
                          min_voxels: int = 10,
                          min_length_um: float = 10.0) -> Dict[int, Dict]:
    """Compute orientations and centroids for all axons."""
    logger.info(f"Computing axon data for {len(axon_voxels)} axons...")

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


# =============================================================================
# Seed Classification by Dominant Axis
# =============================================================================

def classify_seeds_by_dominant_axis(axon_data: Dict[int, Dict],
                                     max_angle_deg: float = 30.0) -> Tuple[List[int], List[int], Dict[int, int]]:
    """
    Classify well-aligned axons as seeds for boundary learning.

    Returns:
        Tuple of (cc_seeds, cg_seeds, bundle_to_axis) where bundle_to_axis maps
        bundle_id (1=CC, 2=CG) to dominant axis index (0=Z, 1=Y, 2=X)
    """
    logger.info(f"Classifying seed axons by dominant axis (max angle: {max_angle_deg}°)...")

    cos_threshold = np.cos(np.deg2rad(max_angle_deg))

    # Classify each axon by dominant axis
    axis_counts = {0: 0, 1: 0, 2: 0}  # Z, Y, X
    axon_to_axis = {}

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

        # Only include well-aligned axons as seeds
        if alignment >= cos_threshold:
            axon_to_axis[label] = dominant_axis
            axis_counts[dominant_axis] += 1

    # Sort axes by count (descending)
    sorted_axes = sorted(axis_counts.keys(), key=lambda a: axis_counts[a], reverse=True)

    # Assign: CC = most populated axis, CG = second most
    axis_to_bundle = {
        sorted_axes[0]: 1,  # CC
        sorted_axes[1]: 2,  # CG
        sorted_axes[2]: 0   # Third axis (not used as seeds)
    }

    # Inverse mapping: bundle_id -> axis
    bundle_to_axis = {
        1: sorted_axes[0],  # CC
        2: sorted_axes[1]   # CG
    }

    # Collect seeds
    cc_seeds = [label for label, axis in axon_to_axis.items() if axis_to_bundle[axis] == 1]
    cg_seeds = [label for label, axis in axon_to_axis.items() if axis_to_bundle[axis] == 2]

    # Log statistics
    axis_names = {0: 'Z', 1: 'Y', 2: 'X'}
    logger.info(f"  Seed classification:")
    for axis in sorted_axes:
        bundle_name = {1: 'CC', 2: 'CG', 0: 'third'}[axis_to_bundle[axis]]
        logger.info(f"    {axis_names[axis]}-axis: {axis_counts[axis]} seeds -> {bundle_name}")

    logger.info(f"  Total seeds: {len(cc_seeds)} CC, {len(cg_seeds)} CG")

    return cc_seeds, cg_seeds, bundle_to_axis


# =============================================================================
# Spatial Boundary Learning
# =============================================================================

def learn_spatial_boundary(axon_data: Dict[int, Dict],
                            cc_seeds: List[int],
                            cg_seeds: List[int],
                            svm_gamma: float = None) -> SVC:
    """
    Learn a curved spatial boundary using SVM with RBF kernel.

    Args:
        axon_data: Dict with centroid_um for each axon
        cc_seeds: Labels of CC seed axons
        cg_seeds: Labels of CG seed axons
        svm_gamma: RBF kernel width. Lower = stiffer/smoother boundary.
                   None uses 'scale' (1 / (n_features * X.var()))

    Returns:
        Trained SVM classifier
    """
    logger.info("Learning spatial boundary with SVM (RBF kernel)...")

    # Prepare training data
    X_cc = np.array([axon_data[label]['centroid_um'] for label in cc_seeds])
    X_cg = np.array([axon_data[label]['centroid_um'] for label in cg_seeds])

    X = np.vstack([X_cc, X_cg])
    y = np.array([1] * len(cc_seeds) + [2] * len(cg_seeds))

    # Train SVM with RBF kernel
    # C controls regularization, gamma controls RBF kernel width
    # Lower gamma = stiffer/smoother boundary
    gamma_value = svm_gamma if svm_gamma is not None else 'scale'
    svm = SVC(kernel='rbf', C=1.0, gamma=gamma_value, random_state=42)
    svm.fit(X, y)

    logger.info(f"  Gamma: {gamma_value}")

    # Log training accuracy
    train_acc = svm.score(X, y)
    logger.info(f"  Training accuracy: {train_acc:.1%}")
    logger.info(f"  Support vectors: {svm.n_support_[0]} CC, {svm.n_support_[1]} CG")

    return svm


def assign_all_axons_spatially(axon_data: Dict[int, Dict],
                                svm: SVC) -> Dict[int, int]:
    """
    Assign ALL axons to CC or CG based on spatial location.

    Args:
        axon_data: Dict with centroid_um for each axon
        svm: Trained SVM classifier

    Returns:
        Dict mapping axon label -> bundle ID (1=CC, 2=CG)
    """
    logger.info(f"Assigning {len(axon_data)} axons spatially...")

    labels = list(axon_data.keys())
    centroids = np.array([axon_data[label]['centroid_um'] for label in labels])

    # Predict using learned boundary
    predictions = svm.predict(centroids)

    label_to_bundle = {labels[i]: int(predictions[i]) for i in range(len(labels))}

    # Count assignments
    n_cc = sum(1 for v in label_to_bundle.values() if v == 1)
    n_cg = sum(1 for v in label_to_bundle.values() if v == 2)
    logger.info(f"  Assigned: {n_cc} CC, {n_cg} CG")

    return label_to_bundle


# =============================================================================
# Sparse Axon Filtering (optional post-processing)
# =============================================================================

def filter_sparse_axons(axon_labels: List[int],
                        axon_voxels: Dict[int, np.ndarray],
                        voxel_size_um: float,
                        k_neighbors: int = 10,
                        max_distance_um: float = 30.0) -> Tuple[List[int], int]:
    """Remove spatially isolated axons using KNN."""
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


# =============================================================================
# Bundle Metadata Creation
# =============================================================================

def create_bundle_metadata(axon_data: Dict[int, Dict],
                            label_to_bundle: Dict[int, int],
                            axon_voxels: Dict[int, np.ndarray],
                            voxel_size_um: float,
                            bundle_to_axis: Dict[int, int],
                            k_neighbors: int = 10,
                            max_neighbor_distance_um: float = 30.0) -> List[Dict]:
    """Create bundle metadata with optional KNN filtering."""
    logger.info("Creating bundle metadata...")

    # Axis names and canonical directions
    axis_names = {0: 'Z', 1: 'Y', 2: 'X'}
    axis_directions = {
        0: [1.0, 0.0, 0.0],  # Z-axis (first dimension)
        1: [0.0, 1.0, 0.0],  # Y-axis (second dimension)
        2: [0.0, 0.0, 1.0]   # X-axis (third dimension)
    }

    bundles_raw = {1: [], 2: []}
    for label, bundle_id in label_to_bundle.items():
        bundles_raw[bundle_id].append(label)

    bundles = []
    for bundle_id in [1, 2]:
        axon_labels = bundles_raw[bundle_id]
        if not axon_labels:
            continue

        # Optional: filter sparse axons
        axon_labels, n_removed = filter_sparse_axons(
            axon_labels, axon_voxels, voxel_size_um,
            k_neighbors=k_neighbors, max_distance_um=max_neighbor_distance_um
        )

        if not axon_labels:
            continue

        # Compute mean orientation for reference
        orientations = np.array([axon_data[label]['orientation'] for label in axon_labels])
        norms = np.linalg.norm(orientations, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        orientations = orientations / norms

        # Handle sign ambiguity
        sign_flip = orientations[:, 0] < 0
        orientations[sign_flip] = -orientations[sign_flip]

        mean_orientation = orientations.mean(axis=0)
        norm = np.linalg.norm(mean_orientation)
        mean_orientation = mean_orientation / norm if norm > 1e-10 else np.array([1, 0, 0])

        lengths = [axon_data[label]['length_um'] for label in axon_labels]

        # Get dominant axis for this bundle
        dominant_axis = bundle_to_axis.get(bundle_id, 0)

        bundle_info = {
            'bundle_id': int(bundle_id),
            'n_axons': len(axon_labels),
            'dominant_axis': int(dominant_axis),
            'dominant_axis_name': axis_names[dominant_axis],
            'fiber_direction': axis_directions[dominant_axis],
            'mean_orientation': mean_orientation.tolist(),
            'mean_length_um': float(np.mean(lengths)),
            'median_length_um': float(np.median(lengths)),
            'axon_labels': [int(label) for label in axon_labels],
            'length_range_um': [float(min(lengths)), float(max(lengths))]
        }
        bundles.append(bundle_info)

        bundle_name = 'CC' if bundle_id == 1 else 'CG'
        logger.info(f"  {bundle_name}: {len(axon_labels)} axons (removed {n_removed} sparse)")

    # Sort by bundle_id
    bundles.sort(key=lambda b: b['bundle_id'])
    return bundles


# =============================================================================
# Volume Extraction (reused from extract_cc_cg_bundles.py)
# =============================================================================

def create_group_ome_metadata(group_name: str, voxel_size_um: float, n_levels: int) -> Dict:
    """Create OME-NGFF metadata for a single group."""
    datasets = []
    for level in range(n_levels):
        scale = voxel_size_um * (2 ** level)
        datasets.append({
            "path": str(level),
            "coordinateTransformations": [
                {"type": "scale", "scale": [scale, scale, scale]}
            ]
        })

    return {
        "multiscales": [{
            "version": "0.4",
            "name": group_name,
            "axes": [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ],
            "datasets": datasets
        }]
    }


def create_label_lut(axon_labels: Set[int], max_label: int) -> np.ndarray:
    """Create lookup table for fast label filtering."""
    lut = np.zeros(max_label + 1, dtype=bool)
    for label in axon_labels:
        if label <= max_label:
            lut[label] = True
    return lut


def filter_with_lut(data: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Filter array using lookup table."""
    clamped = np.minimum(data, len(lut) - 1)
    mask = lut[clamped]
    return np.where(mask, data, 0).astype(data.dtype)


def downsample_segmentation(data: np.ndarray, factor: int) -> np.ndarray:
    """Fast downsampling using center voxel."""
    if factor == 1:
        return data.copy()
    offset = factor // 2
    return data[offset::factor, offset::factor, offset::factor].copy()


def write_bundle_to_group(volume: np.ndarray,
                          bundle: Dict,
                          group: zarr.Group,
                          lut: np.ndarray,
                          voxel_size_um: float,
                          n_levels: int = 5) -> np.ndarray:
    """Write bundle to Zarr group with pyramid levels."""
    output_shape = volume.shape
    n_slices = output_shape[0]

    compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)

    # Write level 0
    chunk_shape_0 = (1, output_shape[1], output_shape[2])
    level_0 = group.create_array(
        '0', shape=output_shape, chunks=chunk_shape_0,
        dtype=np.uint16, compressor=compressor
    )

    segment_id_set = set()
    for z_out in tqdm(range(n_slices), desc=f"Writing {group.name}", unit="slice", leave=False):
        input_slice = volume[z_out, :, :]
        filtered_slice = filter_with_lut(input_slice, lut)
        level_0[z_out, :, :] = filtered_slice
        segment_id_set.update(np.unique(filtered_slice).tolist())

    segment_id_set.discard(0)
    segment_ids = np.array(sorted(segment_id_set), dtype=np.uint16)

    # Generate pyramid
    current_data = level_0[:]
    for level in range(1, n_levels):
        downsampled = downsample_segmentation(current_data, 2)
        chunk_size = min(64, min(downsampled.shape))
        level_ds = group.create_array(
            str(level), shape=downsampled.shape,
            chunks=(chunk_size, chunk_size, chunk_size),
            dtype=np.uint16, compressor=compressor
        )
        level_ds[:] = downsampled
        current_data = downsampled

    # Store segment IDs
    labels_group = group.create_group('labels')
    labels_group.create_array('segment_ids', data=segment_ids)

    # Write OME-NGFF metadata to group
    group_name = group.name.strip('/')
    group.attrs.update(create_group_ome_metadata(group_name, voxel_size_um, n_levels))

    return segment_ids


def create_ome_metadata(bundles: List[Dict], voxel_size_um: float, n_levels: int) -> Dict:
    """Create OME-NGFF metadata for combined Zarr."""
    def make_datasets(n_levels, voxel_size):
        datasets = []
        for level in range(n_levels):
            scale = voxel_size * (2 ** level)
            datasets.append({
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [scale, scale, scale]}
                ]
            })
        return datasets

    axes = [
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"}
    ]

    cc_bundle = next((b for b in bundles if b['bundle_id'] == 1), None)
    cg_bundle = next((b for b in bundles if b['bundle_id'] == 2), None)

    metadata = {
        "multiscales": [
            {
                "version": "0.4",
                "name": "cc",
                "axes": axes,
                "datasets": make_datasets(n_levels, voxel_size_um)
            },
            {
                "version": "0.4",
                "name": "cg",
                "axes": axes,
                "datasets": make_datasets(n_levels, voxel_size_um)
            }
        ],
        "cc": {
            "n_axons": cc_bundle['n_axons'] if cc_bundle else 0,
            "dominant_axis": cc_bundle['dominant_axis'] if cc_bundle else 0,
            "dominant_axis_name": cc_bundle['dominant_axis_name'] if cc_bundle else 'Z',
            "fiber_direction": cc_bundle['fiber_direction'] if cc_bundle else [1, 0, 0],
            "mean_orientation": cc_bundle['mean_orientation'] if cc_bundle else [0, 0, 0],
            "mean_length_um": cc_bundle['mean_length_um'] if cc_bundle else 0
        },
        "cg": {
            "n_axons": cg_bundle['n_axons'] if cg_bundle else 0,
            "dominant_axis": cg_bundle['dominant_axis'] if cg_bundle else 1,
            "dominant_axis_name": cg_bundle['dominant_axis_name'] if cg_bundle else 'Y',
            "fiber_direction": cg_bundle['fiber_direction'] if cg_bundle else [0, 1, 0],
            "mean_orientation": cg_bundle['mean_orientation'] if cg_bundle else [0, 0, 0],
            "mean_length_um": cg_bundle['mean_length_um'] if cg_bundle else 0
        },
        "voxel_size_um": voxel_size_um,
        "n_levels": n_levels,
        "segmentation_method": "spatial_boundary_svm"
    }

    return metadata


# =============================================================================
# Main Pipeline
# =============================================================================

def segment_spatial_subvolumes(mat_file: Path,
                                output_zarr: Path,
                                metadata_file: Path,
                                downsample: int = 4,
                                voxel_size_um: float = 0.05,
                                max_angle_deg: float = 30.0,
                                min_length_um: float = 10.0,
                                min_voxels: int = 10,
                                k_neighbors: int = 10,
                                max_neighbor_distance_um: float = 30.0,
                                n_levels: int = 5,
                                svm_gamma: float = None):
    """
    Segment volume into CC and CG spatial subvolumes.

    Uses orientation-classified axons as seeds to learn a curved spatial
    boundary, then assigns ALL axons based on their spatial location.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Segmenting spatial subvolumes: {mat_file.name}")
    logger.info(f"{'='*80}\n")

    # Phase 1: Load and analyze axons
    logger.info("Phase 1: Loading and analyzing axons...")
    volume_ds, vol_metadata = load_volume_downsampled(mat_file, downsample)
    voxel_size_ds = voxel_size_um * downsample

    axon_voxels_ds = precompute_axon_voxels(volume_ds)
    axon_data = compute_all_axon_data(axon_voxels_ds, voxel_size_ds, min_voxels, min_length_um)

    if not axon_data:
        raise ValueError("No valid axons found after filtering")

    # Phase 2: Classify seeds and learn boundary
    logger.info("\nPhase 2: Learning spatial boundary...")
    cc_seeds, cg_seeds, bundle_to_axis = classify_seeds_by_dominant_axis(axon_data, max_angle_deg)

    if len(cc_seeds) < 10 or len(cg_seeds) < 10:
        raise ValueError(f"Not enough seeds: {len(cc_seeds)} CC, {len(cg_seeds)} CG (need ≥10 each)")

    svm = learn_spatial_boundary(axon_data, cc_seeds, cg_seeds, svm_gamma)

    # Phase 3: Assign ALL axons spatially
    logger.info("\nPhase 3: Assigning all axons spatially...")
    label_to_bundle = assign_all_axons_spatially(axon_data, svm)

    # Create bundle metadata
    bundles = create_bundle_metadata(
        axon_data, label_to_bundle, axon_voxels_ds, voxel_size_ds,
        bundle_to_axis, k_neighbors, max_neighbor_distance_um
    )

    if len(bundles) < 2:
        logger.warning(f"Only {len(bundles)} bundle(s) found, expected 2")

    # Free downsampled data
    del volume_ds, axon_voxels_ds

    # Phase 4: Extract to Zarr
    logger.info("\nPhase 4: Extracting subvolumes...")
    logger.info("Loading full volume...")
    with h5py.File(mat_file, 'r') as f:
        volume_full = f['final_lbl'][:]
    max_label = int(volume_full.max())
    logger.info(f"Full volume: {volume_full.shape}, max label: {max_label}")

    # Create output Zarr
    root = zarr.open_group(str(output_zarr), mode='w', zarr_format=2)

    # Extract CC
    cc_bundle = next((b for b in bundles if b['bundle_id'] == 1), None)
    if cc_bundle:
        logger.info(f"\nExtracting CC region: {cc_bundle['n_axons']} axons")
        cc_group = root.create_group('cc')
        lut = create_label_lut(set(cc_bundle['axon_labels']), max_label)
        write_bundle_to_group(volume_full, cc_bundle, cc_group, lut, voxel_size_um, n_levels)

    # Extract CG
    cg_bundle = next((b for b in bundles if b['bundle_id'] == 2), None)
    if cg_bundle:
        logger.info(f"\nExtracting CG region: {cg_bundle['n_axons']} axons")
        cg_group = root.create_group('cg')
        lut = create_label_lut(set(cg_bundle['axon_labels']), max_label)
        write_bundle_to_group(volume_full, cg_bundle, cg_group, lut, voxel_size_um, n_levels)

    # Write metadata
    root.attrs.update(create_ome_metadata(bundles, voxel_size_um, n_levels))

    # Save JSON metadata
    output_metadata = {
        'volume_metadata': vol_metadata,
        'segmentation_method': 'spatial_boundary_svm',
        'n_seeds': {'cc': len(cc_seeds), 'cg': len(cg_seeds)},
        'n_bundles': len(bundles),
        'bundles': [
            {**b, 'name': 'cc'} if b['bundle_id'] == 1 else {**b, 'name': 'cg'}
            for b in bundles
        ]
    }
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, 'w') as f:
        json.dump(output_metadata, f, indent=2)
    logger.info(f"\nSaved metadata: {metadata_file}")

    # Summary
    total_size = sum(f.stat().st_size for f in output_zarr.rglob('*') if f.is_file())
    logger.info(f"Saved Zarr: {output_zarr} ({total_size / 1024**2:.1f} MB)")

    logger.info(f"\n{'='*80}")
    logger.info("Spatial segmentation complete!")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Segment volume into CC and CG spatial subvolumes'
    )
    parser.add_argument('mat_file', type=Path,
                        help='Input .mat file with labeled volume')
    parser.add_argument('output_zarr', type=Path,
                        help='Output Zarr store path')
    parser.add_argument('--metadata', type=Path, required=True,
                        help='Output JSON metadata file')
    parser.add_argument('--downsample', type=int, default=4,
                        help='Downsampling factor for analysis (default: 4)')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                        help='Voxel size in μm (default: 0.05)')
    parser.add_argument('--max-angle', type=float, default=30.0,
                        help='Max angle for seed classification in degrees (default: 30.0)')
    parser.add_argument('--min-length', type=float, default=10.0,
                        help='Minimum axon length in μm (default: 10.0)')
    parser.add_argument('--min-voxels', type=int, default=10,
                        help='Minimum voxels per axon (default: 10)')
    parser.add_argument('--k-neighbors', type=int, default=10,
                        help='K for sparse axon filtering (default: 10)')
    parser.add_argument('--max-neighbor-distance', type=float, default=30.0,
                        help='Max distance to k-th neighbor in μm (default: 30.0)')
    parser.add_argument('--n-levels', type=int, default=5,
                        help='Pyramid levels (default: 5)')
    parser.add_argument('--svm-gamma', type=float, default=0.1,
                        help='SVM gamma (RBF width). Lower = stiffer boundary. Default: auto-scale')

    args = parser.parse_args()

    segment_spatial_subvolumes(
        args.mat_file,
        args.output_zarr,
        args.metadata,
        downsample=args.downsample,
        voxel_size_um=args.voxel_size,
        max_angle_deg=args.max_angle,
        min_length_um=args.min_length,
        min_voxels=args.min_voxels,
        k_neighbors=args.k_neighbors,
        max_neighbor_distance_um=args.max_neighbor_distance,
        n_levels=args.n_levels,
        svm_gamma=args.svm_gamma
    )
