#!/usr/bin/env python3
"""
Simple and efficient axon filtering based on orientation and length.

No skeletonization needed - just compute basic geometric properties
from the labeled voxels directly.

This is much faster than full skeleton extraction.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AxonProperties:
    """Geometric properties of a single axon"""
    axon_id: int
    n_voxels: int
    bounding_box_min: Tuple[int, int, int]  # (x, y, z)
    bounding_box_max: Tuple[int, int, int]
    extent_x: int
    extent_y: int
    extent_z: int
    principal_direction: Tuple[float, float, float]  # normalized
    alignment_axis: str  # 'x', 'y', or 'z'
    alignment_score: float  # |dot product| with axis
    straightness: float  # ratio of extent to actual path length


class SimpleAxonFilter:
    """Filter axons based on orientation and geometric properties"""

    def __init__(self, downsample_factor: int = 4):
        self.downsample_factor = downsample_factor

    def load_and_downsample(self, input_file: Path) -> np.ndarray:
        """Load volume with optional downsampling"""
        logger.info(f"Loading volume from {input_file}")
        start = time.time()

        with h5py.File(input_file, 'r') as f:
            labels = f['final_lbl'][:]

        logger.info(f"Loaded {labels.shape} in {time.time()-start:.2f}s")
        logger.info(f"Memory: ~{labels.nbytes/1e9:.2f} GB")

        if self.downsample_factor > 1:
            labels = labels[::self.downsample_factor,
                          ::self.downsample_factor,
                          ::self.downsample_factor]
            logger.info(f"Downsampled to {labels.shape}")

        return labels

    def compute_axon_properties_from_coords(self, coords: np.ndarray,
                                            axon_id: int) -> AxonProperties:
        """
        Compute geometric properties from pre-computed coordinates.

        Args:
            coords: Voxel coordinates [N, 3] in [z, y, x] order
            axon_id: Axon ID

        Returns:
            AxonProperties object or None if invalid
        """
        if len(coords) == 0:
            return None

        n_voxels = len(coords)

        # Skip axons with too few voxels for PCA
        if n_voxels < 3:
            return None

        # Bounding box
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)

        extent_z = maxs[0] - mins[0] + 1
        extent_y = maxs[1] - mins[1] + 1
        extent_x = maxs[2] - mins[2] + 1

        # PCA for principal direction
        centered = coords - coords.mean(axis=0)

        # Check if all points are identical (zero variance)
        if np.allclose(centered, 0):
            return None

        try:
            cov = np.cov(centered.T)

            # Check for NaN or inf in covariance matrix
            if not np.isfinite(cov).all():
                return None

            eigenvalues, eigenvectors = np.linalg.eig(cov)

            # Check for NaN or inf in eigenvalues
            if not np.isfinite(eigenvalues).all():
                return None

        except np.linalg.LinAlgError:
            return None

        # Largest eigenvalue = principal direction
        idx = eigenvalues.real.argmax()
        direction = eigenvectors[:, idx].real

        # Normalize
        norm = np.linalg.norm(direction)
        if norm == 0:
            return None
        direction = direction / norm

        # Direction is in [z, y, x] order, reorder to [x, y, z]
        direction_xyz = direction[[2, 1, 0]]

        # Which axis is it most aligned with?
        abs_dir = np.abs(direction_xyz)
        axis_idx = abs_dir.argmax()
        axis_names = ['x', 'y', 'z']
        alignment_axis = axis_names[axis_idx]
        alignment_score = abs_dir[axis_idx]

        # Straightness: how much does it fill its bounding box along principal axis?
        max_extent = max(extent_x, extent_y, extent_z)
        eigenvalue_ratio = eigenvalues[idx] / eigenvalues.sum()
        straightness = eigenvalue_ratio

        return AxonProperties(
            axon_id=int(axon_id),
            n_voxels=int(n_voxels),
            bounding_box_min=tuple(mins[[2, 1, 0]].tolist()),
            bounding_box_max=tuple(maxs[[2, 1, 0]].tolist()),
            extent_x=int(extent_x),
            extent_y=int(extent_y),
            extent_z=int(extent_z),
            principal_direction=tuple(direction_xyz.tolist()),
            alignment_axis=alignment_axis,
            alignment_score=float(alignment_score),
            straightness=float(straightness)
        )

    def compute_axon_properties(self, labels: np.ndarray,
                                axon_id: int) -> AxonProperties:
        """
        Compute geometric properties for a single axon.

        This is much faster than skeletonization!
        """
        # Get all voxels for this axon
        coords = np.argwhere(labels == axon_id)  # Returns [z, y, x]

        if len(coords) == 0:
            return None

        n_voxels = len(coords)

        # Skip axons with too few voxels for PCA
        if n_voxels < 3:
            return None

        # Bounding box
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)

        extent_z = maxs[0] - mins[0] + 1
        extent_y = maxs[1] - mins[1] + 1
        extent_x = maxs[2] - mins[2] + 1

        # PCA for principal direction
        centered = coords - coords.mean(axis=0)

        # Check if all points are identical (zero variance)
        if np.allclose(centered, 0):
            return None

        try:
            cov = np.cov(centered.T)

            # Check for NaN or inf in covariance matrix
            if not np.isfinite(cov).all():
                return None

            eigenvalues, eigenvectors = np.linalg.eig(cov)

            # Check for NaN or inf in eigenvalues
            if not np.isfinite(eigenvalues).all():
                return None

        except np.linalg.LinAlgError:
            return None

        # Largest eigenvalue = principal direction
        idx = eigenvalues.real.argmax()
        direction = eigenvectors[:, idx].real

        # Normalize
        norm = np.linalg.norm(direction)
        if norm == 0:
            return None
        direction = direction / norm

        # Direction is in [z, y, x] order, reorder to [x, y, z]
        direction_xyz = direction[[2, 1, 0]]

        # Which axis is it most aligned with?
        abs_dir = np.abs(direction_xyz)
        axis_idx = abs_dir.argmax()
        axis_names = ['x', 'y', 'z']
        alignment_axis = axis_names[axis_idx]
        alignment_score = abs_dir[axis_idx]

        # Straightness: how much does it fill its bounding box along principal axis?
        # A straight cylinder would have length = max extent
        max_extent = max(extent_x, extent_y, extent_z)
        # Estimate path length from eigenvalue spread
        # If perfectly straight, all variance is along one axis
        eigenvalue_ratio = eigenvalues[idx] / eigenvalues.sum()
        straightness = eigenvalue_ratio  # Close to 1.0 = very straight

        return AxonProperties(
            axon_id=int(axon_id),
            n_voxels=int(n_voxels),
            bounding_box_min=tuple(mins[[2, 1, 0]].tolist()),  # Convert to x,y,z
            bounding_box_max=tuple(maxs[[2, 1, 0]].tolist()),
            extent_x=int(extent_x),
            extent_y=int(extent_y),
            extent_z=int(extent_z),
            principal_direction=tuple(direction_xyz.tolist()),
            alignment_axis=alignment_axis,
            alignment_score=float(alignment_score),
            straightness=float(straightness)
        )

    def precompute_axon_voxels(self, labels: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Pre-compute voxel coordinates for all axons in one pass.

        This is MUCH faster than calling np.argwhere for each axon separately,
        especially for sparse volumes.

        Args:
            labels: Labeled volume

        Returns:
            Dict mapping axon_id -> coordinates array [N, 3] in [z, y, x] order
        """
        logger.info("Pre-computing voxel coordinates for all axons...")
        start = time.time()

        # Find all non-zero voxels in one pass
        all_coords = np.argwhere(labels > 0)

        # Group by axon ID
        axon_voxels = {}
        for coord in all_coords:
            axon_id = labels[tuple(coord)]
            if axon_id not in axon_voxels:
                axon_voxels[axon_id] = []
            axon_voxels[axon_id].append(coord)

        # Convert to numpy arrays
        axon_voxels = {aid: np.array(coords) for aid, coords in axon_voxels.items()}

        elapsed = time.time() - start
        logger.info(f"Pre-computed coordinates for {len(axon_voxels)} axons in {elapsed:.2f}s")

        return axon_voxels

    def filter_axons(self, labels: np.ndarray,
                     min_alignment_score: float = 0.7,
                     min_extent_voxels: int = 50,
                     min_straightness: float = 0.8) -> Dict[int, AxonProperties]:
        """
        Filter axons based on criteria.

        Args:
            labels: Labeled volume
            min_alignment_score: Minimum dot product with nearest axis (0-1)
            min_extent_voxels: Minimum extent along principal axis
            min_straightness: Minimum straightness score (0-1)

        Returns:
            Dict of axon_id -> properties for passing axons
        """
        # Pre-compute all voxel coordinates (much faster!)
        axon_voxels = self.precompute_axon_voxels(labels)

        logger.info(f"Processing {len(axon_voxels)} axons...")

        all_properties = {}
        passing_axons = {}

        for axon_id in tqdm(axon_voxels.keys(), desc="Analyzing axons", unit="axon",
                           bar_format='{desc}: {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]'):

            # Get pre-computed coordinates
            coords = axon_voxels[axon_id]
            props = self.compute_axon_properties_from_coords(coords, axon_id)

            if props is None:
                continue

            all_properties[axon_id] = props

            # Apply filters
            max_extent = max(props.extent_x, props.extent_y, props.extent_z)

            passes = (
                props.alignment_score >= min_alignment_score and
                max_extent >= min_extent_voxels and
                props.straightness >= min_straightness
            )

            if passes:
                passing_axons[axon_id] = props

        logger.info(f"Filtered: {len(passing_axons)}/{len(all_properties)} axons pass criteria")

        # Log statistics by alignment axis
        for axis in ['x', 'y', 'z']:
            count = sum(1 for p in passing_axons.values() if p.alignment_axis == axis)
            logger.info(f"  {axis}-aligned: {count} axons")

        return passing_axons, all_properties

    def save_results(self, output_file: Path,
                     passing_axons: Dict[int, AxonProperties],
                     all_properties: Dict[int, AxonProperties],
                     metadata: dict):
        """Save filtering results to JSON"""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'metadata': metadata,
            'n_total': len(all_properties),
            'n_passing': len(passing_axons),
            'passing_axon_ids': [int(aid) for aid in passing_axons.keys()],
            'passing_properties': {
                int(aid): asdict(props) for aid, props in passing_axons.items()
            },
            'all_properties': {
                int(aid): asdict(props) for aid, props in all_properties.items()
            }
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved results to {output_file}")


def process_volume(input_file: Path, output_dir: Path,
                   downsample_factor: int = 4,
                   min_alignment_score: float = 0.7,
                   min_extent_voxels: int = 50,
                   min_straightness: float = 0.8):
    """
    Process a single volume: filter axons by orientation and geometry,
    then create aligned volumes for CC and CG populations.

    Args:
        input_file: Path to .mat file
        output_dir: Output directory
        downsample_factor: Downsampling factor (4 recommended for filtering)
        min_alignment_score: Minimum alignment with principal axis (0.7 = ~45 degrees)
        min_extent_voxels: Minimum length in downsampled voxels
        min_straightness: Minimum straightness (eigenvalue ratio)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {input_file.name}")
    logger.info(f"{'='*80}\n")

    filter_obj = SimpleAxonFilter(downsample_factor=downsample_factor)

    # Load downsampled volume for filtering
    labels_downsampled = filter_obj.load_and_downsample(input_file)

    # Filter axons on downsampled volume
    passing, all_props = filter_obj.filter_axons(
        labels_downsampled,
        min_alignment_score=min_alignment_score,
        min_extent_voxels=min_extent_voxels,
        min_straightness=min_straightness
    )

    # Prepare metadata
    metadata = {
        'source_file': input_file.name,
        'volume_shape_downsampled': list(labels_downsampled.shape),
        'downsample_factor': downsample_factor,
        'filters': {
            'min_alignment_score': min_alignment_score,
            'min_extent_voxels': min_extent_voxels,
            'min_straightness': min_straightness
        }
    }

    # Save filtering results
    volume_name = input_file.stem.replace('_myelinated_axons', '')
    output_subdir = output_dir / volume_name
    output_file = output_subdir / 'axon_filtering.json'

    filter_obj.save_results(output_file, passing, all_props, metadata)

    # Identify populations
    logger.info(f"\n{'='*80}")
    logger.info(f"Identifying populations and creating aligned volumes")
    logger.info(f"{'='*80}\n")

    populations = identify_populations(passing)

    # Get full volume shape for metadata (without loading full volume)
    with h5py.File(input_file, 'r') as f:
        full_volume_shape = f['final_lbl'].shape
    logger.info(f"Full volume shape: {full_volume_shape}")
    metadata['volume_shape_full'] = list(full_volume_shape)

    # Create and save aligned volumes for each population (slice-by-slice)
    for pop_name, pop_data in populations.items():
        pop_axons = pop_data['axons']
        pop_axis = pop_data['axis']

        if len(pop_axons) == 0:
            logger.info(f"Skipping {pop_name.upper()}: no axons found")
            continue

        logger.info(f"\nCreating aligned volume for {pop_name.upper()}:")
        axon_ids = list(pop_axons.keys())

        # Prepare metadata
        pop_metadata = metadata.copy()
        pop_metadata['n_axons_in_population'] = len(axon_ids)

        # Create and save rotated volume slice-by-slice
        output_volume_file = output_subdir / f'{pop_name}_aligned.h5'
        create_rotated_volume_slicewise(
            input_file,
            output_volume_file,
            axon_ids,
            pop_axis,
            pop_name.upper(),
            pop_metadata
        )

    logger.info(f"\n{'='*80}")
    logger.info(f"Completed: {input_file.name}")
    logger.info(f"{'='*80}\n")


def identify_populations(passing_axons: Dict[int, AxonProperties]) -> Dict[str, Dict[int, AxonProperties]]:
    """
    Identify the top two axon populations based on alignment axis.

    Assumes:
    - Largest population by count is Corpus Callosum (CC)
    - Second largest population is Cingulum (CG)

    Args:
        passing_axons: Dict of filtered axons

    Returns:
        Dict with 'cc' and 'cg' keys mapping to their respective axon dicts
    """
    # Group by alignment axis
    axis_groups = {'x': {}, 'y': {}, 'z': {}}
    for axon_id, props in passing_axons.items():
        axis_groups[props.alignment_axis][axon_id] = props

    # Sort by population size
    sorted_axes = sorted(axis_groups.items(), key=lambda x: len(x[1]), reverse=True)

    # Top two populations
    cc_axis, cc_axons = sorted_axes[0] if len(sorted_axes) > 0 else (None, {})
    cg_axis, cg_axons = sorted_axes[1] if len(sorted_axes) > 1 else (None, {})

    logger.info(f"\nPopulation identification:")
    logger.info(f"  CC (Corpus Callosum): {cc_axis}-aligned, {len(cc_axons)} axons")
    logger.info(f"  CG (Cingulum): {cg_axis}-aligned, {len(cg_axons)} axons")

    return {
        'cc': {'axons': cc_axons, 'axis': cc_axis},
        'cg': {'axons': cg_axons, 'axis': cg_axis}
    }


def create_rotated_volume_slicewise(input_file: Path, output_file: Path,
                                   axon_ids: List[int], alignment_axis: str,
                                   population_name: str, metadata: dict,
                                   chunk_size: int = 100):
    """
    Create axis-permuted volume and write slice-by-slice to avoid memory issues.

    This uses axis permutation, not 3D rotation.

    Args:
        input_file: Path to input HDF5 file
        output_file: Path to output HDF5 file
        axon_ids: List of axon IDs to include
        alignment_axis: 'x', 'y', or 'z' - which axis axons are aligned with
        population_name: 'CC' or 'CG'
        metadata: Metadata dict to save
        chunk_size: Number of slices to process at once
    """
    logger.info(f"  Creating {population_name} volume slice-by-slice...")

    # Open input file to get shape
    with h5py.File(input_file, 'r') as f:
        volume_shape = f['final_lbl'].shape

    logger.info(f"  Original volume shape: {volume_shape}")

    # Convert axon_ids to set for faster lookup
    axon_set = set(axon_ids)

    # Determine output shape based on permutation
    if alignment_axis == 'z':
        output_shape = volume_shape
        perm_desc = "no permutation (already z-aligned)"
    elif alignment_axis == 'y':
        # (z,y,x) -> (y,z,x)
        output_shape = (volume_shape[1], volume_shape[0], volume_shape[2])
        perm_desc = "axes permuted: (z, y, x) -> (y, z, x)"
    elif alignment_axis == 'x':
        # (z,y,x) -> (x,y,z)
        output_shape = (volume_shape[2], volume_shape[1], volume_shape[0])
        perm_desc = "axes permuted: (z, y, x) -> (x, y, z)"
    else:
        raise ValueError(f"Unknown alignment axis: {alignment_axis}")

    logger.info(f"  Output volume shape: {output_shape}")
    logger.info(f"  Permutation: {perm_desc}")

    # Create output file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, 'w') as f_out:
        # Create dataset with chunking for better write performance
        # Chunk along the z-axis (first dimension after permutation)
        chunks = (min(chunk_size, output_shape[0]),
                  min(512, output_shape[1]),
                  min(512, output_shape[2]))

        dset = f_out.create_dataset(
            'labels',
            shape=output_shape,
            dtype=np.uint32,
            compression='gzip',
            compression_opts=1,  # Lower compression for faster writes
            chunks=chunks
        )

        # Save metadata
        f_out.attrs['population'] = population_name
        f_out.attrs['original_alignment_axis'] = alignment_axis
        f_out.attrs['permutation'] = perm_desc
        f_out.attrs['n_axons'] = len(axon_ids)
        for key, val in metadata.items():
            if isinstance(val, (list, tuple, dict)):
                val = json.dumps(val)
            f_out.attrs[key] = val

        # Process slice-by-slice
        with h5py.File(input_file, 'r') as f_in:
            labels_dataset = f_in['final_lbl']

            if alignment_axis == 'z':
                # No permutation: just filter and copy
                for z_start in tqdm(range(0, volume_shape[0], chunk_size),
                                   desc=f"  Writing {population_name} slices",
                                   unit="chunk"):
                    z_end = min(z_start + chunk_size, volume_shape[0])
                    chunk = labels_dataset[z_start:z_end, :, :]

                    # Filter to keep only selected axons
                    mask = np.isin(chunk, list(axon_set))
                    chunk = chunk * mask

                    dset[z_start:z_end, :, :] = chunk

            elif alignment_axis == 'y':
                # Swap z and y: process in chunks for better performance
                for y_start in tqdm(range(0, volume_shape[1], chunk_size),
                                   desc=f"  Writing {population_name} slices",
                                   unit="chunk"):
                    y_end = min(y_start + chunk_size, volume_shape[1])

                    # Read chunk of slices
                    slice_chunk = labels_dataset[:, y_start:y_end, :]  # [z, chunk_y, x]

                    # Filter
                    mask = np.isin(slice_chunk, list(axon_set))
                    slice_chunk = slice_chunk * mask

                    # Write to output (swap y and z)
                    # Input is [z, y, x], output needs [y, z, x]
                    dset[y_start:y_end, :, :] = slice_chunk.transpose(1, 0, 2)

            elif alignment_axis == 'x':
                # Swap z and x: process in chunks for better performance
                for x_start in tqdm(range(0, volume_shape[2], chunk_size),
                                   desc=f"  Writing {population_name} slices",
                                   unit="chunk"):
                    x_end = min(x_start + chunk_size, volume_shape[2])

                    # Read chunk of slices
                    slice_chunk = labels_dataset[:, :, x_start:x_end]  # [z, y, chunk_x]

                    # Filter
                    mask = np.isin(slice_chunk, list(axon_set))
                    slice_chunk = slice_chunk * mask

                    # Write to output (swap x and z)
                    # Input is [z, y, x], output needs [x, y, z]
                    dset[x_start:x_end, :, :] = slice_chunk.transpose(2, 1, 0)

    logger.info(f"  Saved {population_name} volume to {output_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Filter axons by orientation and geometry, create population-specific aligned volumes'
    )
    parser.add_argument('input_file', type=Path,
                       help='Path to input .mat file with labeled axons')
    parser.add_argument('output_dir', type=Path,
                       help='Output directory for results')
    parser.add_argument('--downsample', type=int, default=4,
                       help='Downsampling factor (default: 4)')
    parser.add_argument('--min-alignment', type=float, default=0.7,
                       help='Minimum alignment score 0-1 (default: 0.7, ~45° tolerance)')
    parser.add_argument('--min-extent', type=int, default=50,
                       help='Minimum extent in voxels (default: 50)')
    parser.add_argument('--min-straightness', type=float, default=0.8,
                       help='Minimum straightness 0-1 (default: 0.8)')

    args = parser.parse_args()

    process_volume(
        args.input_file,
        args.output_dir,
        downsample_factor=args.downsample,
        min_alignment_score=args.min_alignment,
        min_extent_voxels=args.min_extent,
        min_straightness=args.min_straightness
    )
