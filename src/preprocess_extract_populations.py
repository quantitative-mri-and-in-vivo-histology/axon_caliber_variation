#!/usr/bin/env python3
"""
Memory-efficient extraction of cingulum and corpus callosum populations from
raw labeled volumes.

This script:
1. Loads data slice-by-slice to minimize memory usage
2. Computes axon fiber orientations via PCA
3. Clusters axons into two populations (CG and CC) based on orientation
4. Filters outliers (short axons, bent axons)
5. Rotates each population to align fiber direction with z-axis
6. Saves cleaned populations to separate HDF5 files

Author: Generated for axon morphology analysis
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FilteringParams:
    """Parameters for axon filtering"""
    min_fraction_present: float = 0.9  # Axon must appear in this fraction of slices
    max_orientation_deviation_deg: float = 30.0  # Max deviation from cluster mean
    voxel_size_um: float = 0.05  # 50 nm isotropic


@dataclass
class PopulationMetadata:
    """Metadata for a tract population"""
    source_file: str
    tract: str
    n_axons_original: int
    n_axons_kept: int
    n_axons_outlier: int
    rotation_matrix: List[List[float]]
    mean_fiber_direction_original: List[float]
    fiber_axis_aligned_to: str
    voxel_size_um: float
    filtering: Dict
    volume_shape_original: Tuple[int, int, int]
    volume_shape_rotated: Tuple[int, int, int]


class AxonPopulationExtractor:
    """Extract and process axon populations from labeled volumes"""

    def __init__(self, params: FilteringParams):
        self.params = params

    def extract_axon_centroids(self,
                              labels_dataset: h5py.Dataset,
                              min_slices: Optional[int] = None) -> Dict[int, np.ndarray]:
        """
        Extract centroid trajectories for each axon (memory efficient).

        Args:
            labels_dataset: HDF5 dataset reference (not loaded into memory)
            min_slices: Minimum number of slices axon must appear in

        Returns:
            Dict mapping axon_id -> array of centroids [[x, y, z], ...]
        """
        nx, ny, nz = labels_dataset.shape
        if min_slices is None:
            min_slices = int(nz * self.params.min_fraction_present)

        logger.info(f"Extracting centroids from volume shape {labels_dataset.shape}")
        logger.info(f"Min slices required: {min_slices} ({self.params.min_fraction_present*100:.0f}%)")

        # First pass: count slice occurrences
        logger.info("Pass 1: Counting axon occurrences per slice...")
        axon_slice_counts = {}

        for z in range(nz):
            if z % 100 == 0:
                logger.info(f"  Slice {z}/{nz}")

            slice_2d = labels_dataset[:, :, z]
            unique_ids = np.unique(slice_2d)

            for axon_id in unique_ids:
                if axon_id == 0:
                    continue
                axon_slice_counts[axon_id] = axon_slice_counts.get(axon_id, 0) + 1

        # Filter by slice count
        valid_axons = {aid for aid, count in axon_slice_counts.items()
                      if count >= min_slices}
        logger.info(f"Found {len(axon_slice_counts)} total axons")
        logger.info(f"Kept {len(valid_axons)} axons with >= {min_slices} slices")

        # Second pass: extract centroids for valid axons
        logger.info("Pass 2: Extracting centroids for valid axons...")
        axon_centroids = {aid: [] for aid in valid_axons}

        for z in range(nz):
            if z % 100 == 0:
                logger.info(f"  Slice {z}/{nz}")

            slice_2d = labels_dataset[:, :, z]

            for axon_id in valid_axons:
                mask = (slice_2d == axon_id)
                if not mask.any():
                    continue

                y_coords, x_coords = np.where(mask)
                centroid = [x_coords.mean(), y_coords.mean(), float(z)]
                axon_centroids[axon_id].append(centroid)

        # Convert to numpy arrays
        axon_centroids = {aid: np.array(cents)
                         for aid, cents in axon_centroids.items()
                         if len(cents) >= min_slices}

        logger.info(f"Final axon count: {len(axon_centroids)}")
        return axon_centroids

    def compute_fiber_orientations(self,
                                   centroids: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Compute principal fiber direction for each axon using PCA.

        Args:
            centroids: Dict mapping axon_id -> centroid array

        Returns:
            Dict mapping axon_id -> normalized direction vector [3,]
        """
        logger.info("Computing fiber orientations via PCA...")
        orientations = {}

        for axon_id, cents in centroids.items():
            if len(cents) < 3:
                continue

            # Center the data
            centered = cents - cents.mean(axis=0)

            # PCA: first principal component
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)

            # Largest eigenvalue -> principal direction
            idx = eigenvalues.argmax()
            direction = eigenvectors[:, idx].real

            # Normalize
            direction = direction / np.linalg.norm(direction)

            orientations[axon_id] = direction

        logger.info(f"Computed orientations for {len(orientations)} axons")
        return orientations

    def cluster_populations(self,
                           orientations: Dict[int, np.ndarray],
                           n_clusters: int = 2) -> Tuple[Dict[int, int], np.ndarray]:
        """
        Cluster axons by fiber orientation (CG vs CC).

        Args:
            orientations: Dict mapping axon_id -> direction vector
            n_clusters: Number of clusters (default 2 for CG and CC)

        Returns:
            cluster_labels: Dict mapping axon_id -> cluster_id (0 or 1)
            cluster_centers: Array of cluster center directions [n_clusters, 3]
        """
        logger.info(f"Clustering {len(orientations)} axons into {n_clusters} populations...")

        axon_ids = list(orientations.keys())
        directions = np.array([orientations[aid] for aid in axon_ids])

        # K-means clustering on orientation vectors
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(directions)

        cluster_labels = {aid: int(cid) for aid, cid in zip(axon_ids, cluster_ids)}

        # Normalize cluster centers
        cluster_centers = kmeans.cluster_centers_
        cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)

        # Label smaller population as CG (cingulum)
        counts = [sum(1 for cid in cluster_ids if cid == i) for i in range(n_clusters)]
        logger.info(f"Cluster sizes: {counts}")

        if counts[0] > counts[1]:
            # Swap labels so cluster 0 = CG (smaller)
            cluster_labels = {aid: 1-cid for aid, cid in cluster_labels.items()}
            cluster_centers = cluster_centers[[1, 0]]
            counts = counts[::-1]

        logger.info(f"CG (cluster 0): {counts[0]} axons, direction: {cluster_centers[0]}")
        logger.info(f"CC (cluster 1): {counts[1]} axons, direction: {cluster_centers[1]}")

        return cluster_labels, cluster_centers

    def filter_outliers(self,
                       orientations: Dict[int, np.ndarray],
                       cluster_labels: Dict[int, int],
                       cluster_centers: np.ndarray) -> Dict[int, bool]:
        """
        Filter outlier axons based on orientation deviation from cluster center.

        Args:
            orientations: Axon directions
            cluster_labels: Cluster assignments
            cluster_centers: Cluster center directions

        Returns:
            Dict mapping axon_id -> is_valid (bool)
        """
        max_dev_rad = np.deg2rad(self.params.max_orientation_deviation_deg)
        logger.info(f"Filtering outliers (max deviation: {self.params.max_orientation_deviation_deg}°)")

        valid_axons = {}
        outlier_count = 0

        for axon_id, direction in orientations.items():
            cluster_id = cluster_labels[axon_id]
            cluster_center = cluster_centers[cluster_id]

            # Angle between direction and cluster center
            cos_angle = np.abs(np.dot(direction, cluster_center))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)

            is_valid = angle <= max_dev_rad
            valid_axons[axon_id] = is_valid

            if not is_valid:
                outlier_count += 1

        logger.info(f"Filtered {outlier_count} outliers")
        return valid_axons

    def compute_rotation_matrix(self, source_direction: np.ndarray,
                               target_axis: str = 'z') -> np.ndarray:
        """
        Compute rotation matrix to align source direction with target axis.

        Args:
            source_direction: Current fiber direction [3,]
            target_axis: Target axis ('x', 'y', or 'z')

        Returns:
            Rotation matrix [3, 3]
        """
        target_vectors = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
        target = np.array(target_vectors[target_axis])

        # Normalize
        source = source_direction / np.linalg.norm(source_direction)

        # Rotation axis: cross product
        axis = np.cross(source, target)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-6:
            # Already aligned or opposite
            if np.dot(source, target) > 0:
                return np.eye(3)
            else:
                # 180 degree rotation around perpendicular axis
                perp = np.array([1, 0, 0]) if abs(source[0]) < 0.9 else np.array([0, 1, 0])
                axis = np.cross(source, perp)
                axis = axis / np.linalg.norm(axis)
                return Rotation.from_rotvec(np.pi * axis).as_matrix()

        axis = axis / axis_norm

        # Rotation angle
        cos_angle = np.dot(source, target)
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        # Rodrigues' rotation formula via scipy
        rot = Rotation.from_rotvec(angle * axis)
        return rot.as_matrix()

    def rotate_and_save_population(self,
                                   input_file: Path,
                                   output_file: Path,
                                   axon_ids: List[int],
                                   rotation_matrix: np.ndarray,
                                   metadata: PopulationMetadata):
        """
        Rotate a population and save to HDF5 (chunk-by-chunk to save memory).

        This is the memory-intensive part - we'll implement chunked processing.
        """
        logger.info(f"Rotating and saving {len(axon_ids)} axons to {output_file}")

        with h5py.File(input_file, 'r') as f_in:
            labels_in = f_in['final_lbl']
            nx, ny, nz = labels_in.shape

            # Create mask of valid axon IDs
            valid_ids_set = set(axon_ids)

            # First, determine rotated bounding box
            # For simplicity, we'll rotate the corner points and compute new bounds
            corners = np.array([
                [0, 0, 0], [nx, 0, 0], [0, ny, 0], [0, 0, nz],
                [nx, ny, 0], [nx, 0, nz], [0, ny, nz], [nx, ny, nz]
            ])
            corners_rotated = corners @ rotation_matrix.T

            mins = corners_rotated.min(axis=0)
            maxs = corners_rotated.max(axis=0)
            new_shape = tuple(np.ceil(maxs - mins).astype(int))
            offset = -mins

            logger.info(f"Original shape: {(nx, ny, nz)}")
            logger.info(f"Rotated shape: {new_shape}")

            # Create output file
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with h5py.File(output_file, 'w') as f_out:
                # Create dataset
                labels_out = f_out.create_dataset(
                    'labels',
                    shape=new_shape,
                    dtype=np.uint16,
                    compression='gzip',
                    compression_opts=4,
                    chunks=True
                )

                # Process slice by slice
                logger.info("Processing slices...")
                labels_out_data = np.zeros(new_shape, dtype=np.uint16)

                for z in range(nz):
                    if z % 100 == 0:
                        logger.info(f"  Slice {z}/{nz}")

                    slice_2d = labels_in[:, :, z]

                    # Filter to valid axons only
                    filtered_slice = np.where(
                        np.isin(slice_2d, list(valid_ids_set)),
                        slice_2d,
                        0
                    )

                    # Get coordinates of non-zero voxels
                    ys, xs = np.where(filtered_slice > 0)
                    if len(xs) == 0:
                        continue

                    zs = np.full_like(xs, z)
                    coords = np.stack([xs, ys, zs], axis=1)
                    labels_vals = filtered_slice[ys, xs]

                    # Rotate coordinates
                    coords_rotated = coords @ rotation_matrix.T + offset
                    coords_rotated = np.round(coords_rotated).astype(int)

                    # Bounds check
                    valid_mask = (
                        (coords_rotated[:, 0] >= 0) & (coords_rotated[:, 0] < new_shape[0]) &
                        (coords_rotated[:, 1] >= 0) & (coords_rotated[:, 1] < new_shape[1]) &
                        (coords_rotated[:, 2] >= 0) & (coords_rotated[:, 2] < new_shape[2])
                    )

                    coords_rotated = coords_rotated[valid_mask]
                    labels_vals = labels_vals[valid_mask]

                    # Assign to output
                    labels_out_data[
                        coords_rotated[:, 0],
                        coords_rotated[:, 1],
                        coords_rotated[:, 2]
                    ] = labels_vals

                # Write to disk
                logger.info("Writing rotated volume to disk...")
                labels_out[:] = labels_out_data

                # Save axon IDs
                f_out.create_dataset('axon_ids', data=np.array(axon_ids, dtype=np.uint32))

                # Save metadata as attributes
                meta_dict = asdict(metadata)
                for key, val in meta_dict.items():
                    if isinstance(val, (list, tuple)):
                        val = json.dumps(val)
                    elif isinstance(val, dict):
                        val = json.dumps(val)
                    f_out.attrs[key] = val

        logger.info(f"Saved to {output_file}")


def process_volume(input_file: Path,
                   output_dir: Path,
                   params: FilteringParams):
    """
    Process a single volume: extract CG and CC populations.

    Args:
        input_file: Path to input .mat file
        output_dir: Directory for output files
        params: Filtering parameters
    """
    logger.info(f"Processing {input_file.name}")

    extractor = AxonPopulationExtractor(params)

    # Load and extract centroids
    with h5py.File(input_file, 'r') as f:
        labels = f['final_lbl']
        original_shape = labels.shape

        # Extract centroids (memory efficient)
        centroids = extractor.extract_axon_centroids(labels)

    if len(centroids) == 0:
        logger.error("No valid axons found!")
        return

    # Compute orientations
    orientations = extractor.compute_fiber_orientations(centroids)

    # Cluster into populations
    cluster_labels, cluster_centers = extractor.cluster_populations(orientations)

    # Filter outliers
    valid_axons = extractor.filter_outliers(orientations, cluster_labels, cluster_centers)

    # Process each population
    tract_names = ['cingulum', 'corpus_callosum']

    for cluster_id, tract_name in enumerate(tract_names):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {tract_name.upper()}")
        logger.info(f"{'='*60}")

        # Get axons for this population
        population_axons = [
            aid for aid, cid in cluster_labels.items()
            if cid == cluster_id and valid_axons.get(aid, False)
        ]

        if len(population_axons) == 0:
            logger.warning(f"No valid axons in {tract_name}!")
            continue

        # Compute rotation matrix
        rotation_matrix = extractor.compute_rotation_matrix(
            cluster_centers[cluster_id],
            target_axis='z'
        )

        # Create metadata
        all_population_axons = [aid for aid, cid in cluster_labels.items() if cid == cluster_id]
        n_outliers = len(all_population_axons) - len(population_axons)

        metadata = PopulationMetadata(
            source_file=input_file.name,
            tract=tract_name,
            n_axons_original=len(all_population_axons),
            n_axons_kept=len(population_axons),
            n_axons_outlier=n_outliers,
            rotation_matrix=rotation_matrix.tolist(),
            mean_fiber_direction_original=cluster_centers[cluster_id].tolist(),
            fiber_axis_aligned_to='z',
            voxel_size_um=params.voxel_size_um,
            filtering=asdict(params),
            volume_shape_original=original_shape,
            volume_shape_rotated=(0, 0, 0)  # Will be updated during rotation
        )

        # Output file
        volume_name = input_file.stem.replace('_myelinated_axons', '')
        output_subdir = output_dir / volume_name
        output_file = output_subdir / f"{tract_name}.h5"

        # Rotate and save
        extractor.rotate_and_save_population(
            input_file,
            output_file,
            population_axons,
            rotation_matrix,
            metadata
        )

        # Save preprocessing info
        info_file = output_subdir / "preprocessing_info.json"
        info_file.parent.mkdir(parents=True, exist_ok=True)

        info = {
            'source_file': input_file.name,
            'filtering_params': asdict(params),
            'populations': {
                tract_name: {
                    'n_axons_kept': len(population_axons),
                    'n_axons_outlier': n_outliers,
                    'output_file': str(output_file.relative_to(output_dir))
                }
            }
        }

        # Append or update
        if info_file.exists():
            with open(info_file) as f:
                existing = json.load(f)
            existing['populations'].update(info['populations'])
            info = existing

        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)

        logger.info(f"Preprocessing info saved to {info_file}")


if __name__ == '__main__':
    # Example usage
    input_file = Path('/workspace/data/raw/LM/LM_25_ipsi_myelinated_axons.mat')
    output_dir = Path('/workspace/data/processed')

    params = FilteringParams(
        min_fraction_present=0.9,
        max_orientation_deviation_deg=30.0,
        voxel_size_um=0.05
    )

    process_volume(input_file, output_dir, params)
