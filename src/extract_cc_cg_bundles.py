#!/usr/bin/env python3
"""
Extract CC and CG fiber bundles to a single OME-Zarr file.

Combines bundle identification (K-means clustering on orientation + position)
with volume extraction into one script. Outputs a single Zarr file with
separate groups for CC (largest bundle) and CG (second largest).

Output structure:
    output.zarr/
    ├── cc/0, cc/1, ...  (pyramid levels for corpus callosum)
    ├── cg/0, cg/1, ...  (pyramid levels for cingulum)
    └── .zattrs          (combined metadata)
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
from sklearn.cluster import KMeans
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Bundle Identification Functions (from identify_cc_cg_bundles.py)
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


def compute_all_orientations(axon_voxels: Dict[int, np.ndarray],
                             voxel_size_um: float,
                             min_voxels: int = 10,
                             min_length_um: float = 50.0) -> Dict[int, Dict]:
    """Compute orientations for all axons with length filtering."""
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


def cluster_kmeans_2(axon_data: Dict[int, Dict], spatial_weight: float = 0.5) -> Dict[int, int]:
    """Cluster axons into 2 bundles using K-means on orientation + position."""
    logger.info(f"Clustering axons into 2 bundles (spatial_weight={spatial_weight})...")

    labels = list(axon_data.keys())
    orientations = np.array([axon_data[label]['orientation'] for label in labels])
    centroids = np.array([axon_data[label]['centroid_um'] for label in labels])

    # Normalize orientations
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

    # Handle sign ambiguity
    sign_flip = orientations[:, 0] < 0
    orientations[sign_flip] = -orientations[sign_flip]

    # Normalize positions to [0, 1]
    pos_min = centroids.min(axis=0)
    pos_max = centroids.max(axis=0)
    pos_range = pos_max - pos_min
    pos_range[pos_range < 1e-10] = 1.0
    positions_normalized = (centroids - pos_min) / pos_range

    # Build 6D features
    orientation_weight = 1.0 - spatial_weight
    features = np.hstack([
        orientations * orientation_weight,
        positions_normalized * spatial_weight
    ])

    # K-means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(features)
    label_to_bundle = {labels[i]: int(cluster_ids[i] + 1) for i in range(len(labels))}

    for cluster_id in [0, 1]:
        n_axons = (cluster_ids == cluster_id).sum()
        logger.info(f"  Cluster {cluster_id + 1}: {n_axons} axons")

    return label_to_bundle


def filter_orientation_outliers(axon_data: Dict[int, Dict],
                                 label_to_bundle: Dict[int, int],
                                 std_threshold: float = 2.0) -> Tuple[Dict[int, int], int]:
    """Remove axons whose orientation deviates too much from cluster mean."""
    logger.info(f"Filtering orientation outliers (>{std_threshold} std)...")

    bundles = {1: [], 2: []}
    for label, bundle_id in label_to_bundle.items():
        bundles[bundle_id].append(label)

    filtered_label_to_bundle = {}
    total_removed = 0

    for bundle_id, axon_labels in bundles.items():
        if not axon_labels:
            continue

        orientations = np.array([axon_data[label]['orientation'] for label in axon_labels])
        orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)
        sign_flip = orientations[:, 0] < 0
        orientations[sign_flip] = -orientations[sign_flip]

        mean_orientation = orientations.mean(axis=0)
        norm = np.linalg.norm(mean_orientation)
        if norm < 1e-10:
            continue
        mean_orientation = mean_orientation / norm

        cos_angles = np.abs(orientations @ mean_orientation)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(cos_angles))

        threshold_angle = angles_deg.mean() + std_threshold * angles_deg.std()
        mask = angles_deg <= threshold_angle
        n_removed = len(axon_labels) - mask.sum()
        total_removed += n_removed

        for i, label in enumerate(axon_labels):
            if mask[i]:
                filtered_label_to_bundle[label] = bundle_id

    logger.info(f"Removed {total_removed} orientation outliers")
    return filtered_label_to_bundle, total_removed


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


def create_bundles(axon_data: Dict[int, Dict],
                   label_to_bundle: Dict[int, int],
                   axon_voxels: Dict[int, np.ndarray],
                   voxel_size_um: float,
                   k_neighbors: int = 10,
                   max_neighbor_distance_um: float = 30.0) -> List[Dict]:
    """Create bundle metadata with per-bundle KNN filtering."""
    logger.info("Creating bundle metadata...")

    bundles_raw = {1: [], 2: []}
    for label, bundle_id in label_to_bundle.items():
        bundles_raw[bundle_id].append(label)

    bundles = []
    for bundle_id in [1, 2]:
        axon_labels = bundles_raw[bundle_id]
        if not axon_labels:
            continue

        axon_labels, n_removed = filter_sparse_axons(
            axon_labels, axon_voxels, voxel_size_um,
            k_neighbors=k_neighbors, max_distance_um=max_neighbor_distance_um
        )

        if not axon_labels:
            continue

        orientations = np.array([axon_data[label]['orientation'] for label in axon_labels])
        orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)
        sign_flip = orientations[:, 0] < 0
        orientations[sign_flip] = -orientations[sign_flip]

        mean_orientation = orientations.mean(axis=0)
        norm = np.linalg.norm(mean_orientation)
        mean_orientation = mean_orientation / norm if norm > 1e-10 else np.array([1, 0, 0])

        lengths = [axon_data[label]['length_um'] for label in axon_labels]

        bundle_info = {
            'bundle_id': int(bundle_id),
            'n_axons': len(axon_labels),
            'mean_orientation': mean_orientation.tolist(),
            'mean_length_um': float(np.mean(lengths)),
            'median_length_um': float(np.median(lengths)),
            'axon_labels': [int(label) for label in axon_labels],
            'length_range_um': [float(min(lengths)), float(max(lengths))]
        }
        bundles.append(bundle_info)

        logger.info(f"  Bundle {bundle_id}: {len(axon_labels)} axons (removed {n_removed} sparse)")

    # Sort by axon count descending
    bundles.sort(key=lambda b: b['n_axons'], reverse=True)
    return bundles


# =============================================================================
# Volume Extraction Functions (from extract_bundle_volumes_simple.py)
# =============================================================================

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


def write_bundle_to_group(volume: np.ndarray,
                          bundle: Dict,
                          group: zarr.Group,
                          lut: np.ndarray,
                          voxel_size_um: float,
                          n_levels: int = 5) -> np.ndarray:
    """Write bundle to Zarr group with pyramid levels and OME-NGFF metadata.

    Keeps original volume orientation [Z, Y, X] for spatial alignment of all bundles.
    """
    # Use original volume shape - no permutation for spatial alignment
    output_shape = volume.shape  # [Z, Y, X]
    n_slices = output_shape[0]

    compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)

    # Write level 0 with (1, Y, X) chunking for efficient slice access
    chunk_shape_0 = (1, output_shape[1], output_shape[2])
    level_0 = group.create_array(
        '0', shape=output_shape, chunks=chunk_shape_0,
        dtype=np.uint16, compressor=compressor
    )

    segment_id_set = set()
    for z_out in tqdm(range(n_slices), desc=f"Writing {group.name}", unit="slice", leave=False):
        # Direct slice extraction - no permutation
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

    # CC is largest (index 0), CG is second (index 1)
    cc_bundle = bundles[0] if len(bundles) > 0 else None
    cg_bundle = bundles[1] if len(bundles) > 1 else None

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
            "mean_orientation": cc_bundle['mean_orientation'] if cc_bundle else [0, 0, 0],
            "mean_length_um": cc_bundle['mean_length_um'] if cc_bundle else 0
        },
        "cg": {
            "n_axons": cg_bundle['n_axons'] if cg_bundle else 0,
            "mean_orientation": cg_bundle['mean_orientation'] if cg_bundle else [0, 0, 0],
            "mean_length_um": cg_bundle['mean_length_um'] if cg_bundle else 0
        },
        "voxel_size_um": voxel_size_um,
        "n_levels": n_levels
    }

    return metadata


# =============================================================================
# Main Pipeline
# =============================================================================

def extract_cc_cg_bundles(mat_file: Path,
                          output_zarr: Path,
                          metadata_file: Path,
                          downsample: int = 4,
                          voxel_size_um: float = 0.05,
                          orientation_std_threshold: float = 2.0,
                          spatial_weight: float = 0.5,
                          min_length_um: float = 50.0,
                          min_voxels: int = 10,
                          k_neighbors: int = 10,
                          max_neighbor_distance_um: float = 30.0,
                          n_levels: int = 5):
    """
    Complete pipeline to identify and extract CC/CG bundles to single Zarr.

    Args:
        mat_file: Input .mat file
        output_zarr: Output Zarr store path
        metadata_file: Output JSON metadata path
        downsample: Downsampling factor for identification
        voxel_size_um: Physical voxel size
        orientation_std_threshold: Std deviations for outlier rejection
        spatial_weight: Weight for spatial position in clustering
        min_length_um: Minimum axon length
        min_voxels: Minimum voxels per axon
        k_neighbors: K for KNN filtering
        max_neighbor_distance_um: Max distance for KNN
        n_levels: Pyramid levels
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Extracting CC and CG bundles: {mat_file.name}")
    logger.info(f"{'='*80}\n")

    # Phase 1: Identify bundles (on downsampled volume)
    logger.info("Phase 1: Identifying bundles...")
    volume_ds, vol_metadata = load_volume_downsampled(mat_file, downsample)
    voxel_size_ds = voxel_size_um * downsample

    axon_voxels_ds = precompute_axon_voxels(volume_ds)
    axon_data = compute_all_orientations(axon_voxels_ds, voxel_size_ds, min_voxels, min_length_um)

    if not axon_data:
        raise ValueError("No valid axons found after filtering")

    label_to_bundle = cluster_kmeans_2(axon_data, spatial_weight)
    label_to_bundle, _ = filter_orientation_outliers(axon_data, label_to_bundle, orientation_std_threshold)

    if not label_to_bundle:
        raise ValueError("No axons remain after orientation filtering")

    bundles = create_bundles(
        axon_data, label_to_bundle, axon_voxels_ds, voxel_size_ds,
        k_neighbors, max_neighbor_distance_um
    )

    if len(bundles) < 2:
        logger.warning(f"Only {len(bundles)} bundle(s) found, expected 2")

    # Free downsampled data
    del volume_ds, axon_voxels_ds

    # Phase 2: Load full volume for extraction
    logger.info("\nPhase 2: Extracting bundles...")
    logger.info("Loading full volume...")
    with h5py.File(mat_file, 'r') as f:
        volume_full = f['final_lbl'][:]
    max_label = int(volume_full.max())
    logger.info(f"Full volume: {volume_full.shape}, max label: {max_label}")

    # Create output Zarr
    root = zarr.open_group(str(output_zarr), mode='w', zarr_format=2)

    # Extract CC (largest bundle) - keep original orientation for spatial alignment
    if len(bundles) >= 1:
        cc_bundle = bundles[0]
        logger.info(f"\nExtracting CC (largest): {cc_bundle['n_axons']} axons")
        cc_group = root.create_group('cc')
        lut = create_label_lut(set(cc_bundle['axon_labels']), max_label)
        write_bundle_to_group(volume_full, cc_bundle, cc_group, lut, voxel_size_um, n_levels)

    # Extract CG (second largest) - keep original orientation for spatial alignment
    if len(bundles) >= 2:
        cg_bundle = bundles[1]
        logger.info(f"\nExtracting CG (second): {cg_bundle['n_axons']} axons")
        cg_group = root.create_group('cg')
        lut = create_label_lut(set(cg_bundle['axon_labels']), max_label)
        write_bundle_to_group(volume_full, cg_bundle, cg_group, lut, voxel_size_um, n_levels)

    # Write metadata
    root.attrs.update(create_ome_metadata(bundles, voxel_size_um, n_levels))

    # Save JSON metadata
    output_metadata = {
        'volume_metadata': vol_metadata,
        'n_bundles': len(bundles),
        'bundles': [
            {**bundles[0], 'name': 'cc'} if len(bundles) > 0 else {},
            {**bundles[1], 'name': 'cg'} if len(bundles) > 1 else {}
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
    logger.info("Extraction complete!")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract CC and CG bundles to single OME-Zarr file'
    )
    parser.add_argument('mat_file', type=Path,
                       help='Input .mat file with labeled volume')
    parser.add_argument('output_zarr', type=Path,
                       help='Output Zarr store path')
    parser.add_argument('--metadata', type=Path, required=True,
                       help='Output JSON metadata file')
    parser.add_argument('--downsample', type=int, default=4,
                       help='Downsampling factor for identification (default: 4)')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                       help='Voxel size in μm (default: 0.05)')
    parser.add_argument('--orientation-std-threshold', type=float, default=2.0,
                       help='Std deviations for outlier rejection (default: 2.0)')
    parser.add_argument('--spatial-weight', type=float, default=0.5,
                       help='Weight for spatial position, 0-1 (default: 0.5)')
    parser.add_argument('--min-length', type=float, default=50.0,
                       help='Minimum axon length in μm (default: 50.0)')
    parser.add_argument('--min-voxels', type=int, default=10,
                       help='Minimum voxels per axon (default: 10)')
    parser.add_argument('--k-neighbors', type=int, default=10,
                       help='K for sparse axon filtering (default: 10)')
    parser.add_argument('--max-neighbor-distance', type=float, default=30.0,
                       help='Max distance to k-th neighbor in μm (default: 30.0)')
    parser.add_argument('--n-levels', type=int, default=5,
                       help='Pyramid levels (default: 5)')

    args = parser.parse_args()

    extract_cc_cg_bundles(
        args.mat_file,
        args.output_zarr,
        args.metadata,
        downsample=args.downsample,
        voxel_size_um=args.voxel_size,
        orientation_std_threshold=args.orientation_std_threshold,
        spatial_weight=args.spatial_weight,
        min_length_um=args.min_length,
        min_voxels=args.min_voxels,
        k_neighbors=args.k_neighbors,
        max_neighbor_distance_um=args.max_neighbor_distance,
        n_levels=args.n_levels
    )
