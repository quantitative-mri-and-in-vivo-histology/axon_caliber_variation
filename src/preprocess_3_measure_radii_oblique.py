#!/usr/bin/env python3
"""
Measure axon radius distributions using oblique slicing perpendicular to fiber direction.

This script implements true 3D oblique slicing to extract radius distributions
at regular intervals along the fiber bundle axis, regardless of fiber orientation
relative to volume axes.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.measure import regionprops
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_effective_radius(histogram: np.ndarray, bin_centers: np.ndarray) -> float:
    """
    Compute MRI-visible effective radius: r_eff = (⟨r⁶⟩/⟨r²⟩)^(1/4)

    This metric accounts for the 6th moment weighting in diffusion MRI signal
    attenuation, capturing the sensitivity of MRI to larger axon calibers.

    Reference:
        Alexander et al., "Orientationally invariant indices of axon diameter
        and density from diffusion MRI", NeuroImage 52:1374-1389 (2010).
        DOI: 10.1016/j.neuroimage.2010.05.043

    Args:
        histogram: Counts per bin
        bin_centers: Bin center positions in micrometers

    Returns:
        Effective radius in micrometers, or np.nan if invalid/empty
    """
    if histogram.sum() == 0:
        return np.nan

    r2_mean = np.sum(histogram * bin_centers**2) / histogram.sum()
    r6_mean = np.sum(histogram * bin_centers**6) / histogram.sum()

    if r2_mean <= 0 or r6_mean <= 0:
        return np.nan

    return (r6_mean / r2_mean) ** 0.25


def generate_oblique_plane_grid(
    fiber_direction: np.ndarray,
    volume_shape: Tuple[int, int, int],
    plane_position: float,
    voxel_size_um: float = 0.05,
    grid_spacing_um: float = None
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate 3D coordinates for sampling an oblique plane perpendicular to fiber direction.

    COORDINATE CONVENTION:
    - volume_shape: (nz, ny, nx) matching numpy array indexing order
    - fiber_direction: [vz, vy, vx] matching volume_shape order
    - output coords: [N, 3] in (z, y, x) order for scipy.map_coordinates

    The plane is perpendicular to fiber_direction and passes through:
        plane_origin = volume_center + plane_position * fiber_direction

    The plane is parameterized as:
        P(u, v) = plane_origin + u*v1 + v*v2
    where v1, v2 are orthogonal vectors in the plane (computed via Gram-Schmidt).

    Args:
        fiber_direction: [vz, vy, vx] normalized direction vector
        volume_shape: (nz, ny, nx) volume dimensions in voxels
        plane_position: Position along fiber axis in voxels from volume center
        voxel_size_um: Physical voxel size
        grid_spacing_um: Sampling grid spacing (default: same as voxel_size_um)

    Returns:
        coords_3d: [N, 3] array of (z, y, x) coordinates in continuous space
        (U, V): Grid arrays for reconstructing 2D slice
    """
    # Validate inputs
    assert len(volume_shape) == 3, "volume_shape must be 3D (nz, ny, nx)"
    assert len(fiber_direction) == 3, "fiber_direction must be 3D [vz, vy, vx]"
    if grid_spacing_um is None:
        grid_spacing_um = voxel_size_um

    # Normalize fiber direction
    fiber_direction = fiber_direction / np.linalg.norm(fiber_direction)

    # Step 1: Find two orthogonal vectors in the plane
    # Choose arbitrary starting vector not parallel to fiber_direction
    if abs(fiber_direction[2]) < 0.9:  # Not parallel to z
        u = np.array([0, 0, 1])
    else:
        u = np.array([0, 1, 0])

    # Gram-Schmidt orthogonalization
    v1 = u - np.dot(u, fiber_direction) * fiber_direction
    v1 = v1 / np.linalg.norm(v1)

    # Second perpendicular vector
    v2 = np.cross(fiber_direction, v1)
    v2 = v2 / np.linalg.norm(v2)

    # Step 2: Define plane origin (point on fiber axis through volume center)
    volume_center = np.array([volume_shape[0]/2, volume_shape[1]/2, volume_shape[2]/2])
    plane_origin = volume_center + plane_position * fiber_direction

    # Step 3: Determine plane extent to cover volume
    # Conservative: use diagonal length
    max_extent_voxels = np.linalg.norm(volume_shape)
    max_extent_um = max_extent_voxels * voxel_size_um

    # Step 4: Generate grid of points in plane
    grid_spacing_voxels = grid_spacing_um / voxel_size_um
    n_grid = int(max_extent_um / grid_spacing_um) + 1

    # Create 1D coordinate arrays
    coords_1d = np.linspace(-max_extent_um/2, max_extent_um/2, n_grid)

    # Create 2D meshgrid
    U, V = np.meshgrid(coords_1d, coords_1d)

    # Convert to voxel coordinates and add to plane origin
    U_voxels = U / voxel_size_um
    V_voxels = V / voxel_size_um

    # Compute 3D coordinates: origin + U*v1 + V*v2
    coords_3d = (plane_origin[:, None, None] +
                 U_voxels * v1[:, None, None] +
                 V_voxels * v2[:, None, None])

    # Reshape to [N, 3] for map_coordinates
    coords_flat = coords_3d.reshape(3, -1).T  # [N, 3] in (z, y, x) order

    return coords_flat, (U, V)


def sample_volume_oblique(
    volume: np.ndarray,
    coords: np.ndarray,
    grid_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Sample labeled volume at oblique plane coordinates.

    Args:
        volume: 3D labeled volume
        coords: [N, 3] array of (z, y, x) coordinates in continuous space
        grid_shape: (ny, nx) shape for reshaping to 2D slice

    Returns:
        slice_2d: [ny, nx] labeled 2D slice
    """
    # Transpose coords for map_coordinates (expects [3, N])
    coords_t = coords.T

    # Check bounds
    in_bounds = (
        (coords_t[0] >= 0) & (coords_t[0] < volume.shape[0]) &
        (coords_t[1] >= 0) & (coords_t[1] < volume.shape[1]) &
        (coords_t[2] >= 0) & (coords_t[2] < volume.shape[2])
    )

    # Sample with nearest neighbor (preserves integer labels)
    sampled_values = np.zeros(coords.shape[0], dtype=volume.dtype)
    sampled_values[in_bounds] = map_coordinates(
        volume,
        coords_t[:, in_bounds],
        order=0,  # Nearest neighbor
        mode='constant',
        cval=0
    )

    # Reshape to 2D
    slice_2d = sampled_values.reshape(grid_shape)

    return slice_2d


def extract_radii_from_slice(
    slice_2d: np.ndarray,
    voxel_size_um: float,
    bin_edges: np.ndarray
) -> Dict:
    """
    Extract radius distribution from a 2D labeled slice.

    Args:
        slice_2d: 2D labeled slice
        voxel_size_um: Physical voxel size
        bin_edges: Histogram bin edges

    Returns:
        Dictionary with histogram and statistics
    """
    # Use regionprops for efficient measurement
    regions = regionprops(slice_2d.astype(np.uint32))

    if len(regions) == 0:
        return {
            'n_axons': 0,
            'histogram': np.zeros(len(bin_edges) - 1, dtype=np.int32),
            'radii': np.array([])
        }

    # Compute circular-equivalent radii
    radii = []
    for region in regions:
        area_voxels = region.area
        area_um2 = area_voxels * (voxel_size_um ** 2)
        radius_um = np.sqrt(area_um2 / np.pi)
        radii.append(radius_um)

    radii = np.array(radii)

    # Compute histogram
    histogram, _ = np.histogram(radii, bins=bin_edges)

    return {
        'n_axons': len(radii),
        'histogram': histogram.astype(np.int32),
        'radii': radii  # Keep for QC if needed
    }


def measure_radii_oblique_intervals(
    volume_file: Path,
    density_metadata: Dict,
    output_file: Path,
    interval_um: float = 10.0,
    voxel_size_um: float = 0.05,
    bin_edges: np.ndarray = None,
    save_slices: bool = False
) -> Dict:
    """
    Measure radius distributions at regular intervals using oblique slicing.

    Args:
        volume_file: Path to aligned HDF5 volume
        density_metadata: Density analysis results (from preprocess_2)
        output_file: Output HDF5 file for results
        interval_um: Spacing between measurement planes in micrometers
        voxel_size_um: Physical voxel size
        bin_edges: Histogram bin edges (default: 0-20 μm, 0.02 step)
        save_slices: Whether to save extracted slices for visualization

    Returns:
        Results dictionary
    """
    if bin_edges is None:
        bin_edges = np.arange(0, 20.02, 0.02)

    logger.info(f"Measuring radii with oblique slicing: {volume_file.name}")
    logger.info(f"Interval: {interval_um} μm")

    # Extract metadata
    fiber_direction = np.array(density_metadata['fiber_direction'])
    valid_range = density_metadata['valid_range']
    start_slice = valid_range['start_slice']
    end_slice = valid_range['end_slice']

    # Convert slice range to positions along fiber axis
    # Project slice positions onto fiber direction (handles arbitrary orientations)
    volume_shape = tuple(density_metadata['volume_shape'])
    volume_center = np.array([
        density_metadata['n_slices'] / 2,
        volume_shape[1] / 2,
        volume_shape[2] / 2
    ])

    # Slice positions in 3D space (assuming density assessment used z-axis slicing)
    start_pos_3d = np.array([start_slice, volume_center[1], volume_center[2]])
    end_pos_3d = np.array([end_slice, volume_center[1], volume_center[2]])

    # Project onto fiber direction to get position along fiber axis
    fiber_direction_normalized = fiber_direction / np.linalg.norm(fiber_direction)
    start_position_voxels = np.dot(start_pos_3d - volume_center, fiber_direction_normalized)
    end_position_voxels = np.dot(end_pos_3d - volume_center, fiber_direction_normalized)

    # Define measurement positions
    interval_voxels = interval_um / voxel_size_um
    n_intervals = int((end_position_voxels - start_position_voxels) / interval_voxels)
    plane_positions = np.linspace(start_position_voxels, end_position_voxels, n_intervals)

    logger.info(f"Extracting {n_intervals} oblique slices...")
    logger.info(f"Fiber direction: {fiber_direction}")

    # Load volume
    logger.info(f"Loading volume...")
    with h5py.File(volume_file, 'r') as f:
        volume = f['labels'][:]
        volume_shape = volume.shape
        population = f.attrs.get('population', 'Unknown')

    logger.info(f"Volume shape: {volume_shape}")

    # Process each plane
    results = []
    saved_slices = []

    for i, plane_pos in enumerate(tqdm(plane_positions, desc="Extracting oblique slices")):
        # Generate oblique plane coordinates
        coords, (U, V) = generate_oblique_plane_grid(
            fiber_direction,
            volume_shape,
            plane_pos,
            voxel_size_um
        )

        # Sample volume
        slice_2d = sample_volume_oblique(volume, coords, U.shape)

        # Extract radii
        result = extract_radii_from_slice(slice_2d, voxel_size_um, bin_edges)

        # Add position information
        result['plane_index'] = i
        result['plane_position_voxels'] = float(plane_pos)
        result['plane_position_um'] = float(plane_pos * voxel_size_um)

        results.append(result)

        if save_slices:
            saved_slices.append(slice_2d)

    # Compile results
    positions_um = np.array([r['plane_position_um'] for r in results])
    n_axons = np.array([r['n_axons'] for r in results])
    histograms = np.array([r['histogram'] for r in results])

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute effective radius per slice
    r_eff_per_slice = np.array([compute_effective_radius(hist, bin_centers) for hist in histograms])

    # Global effective radius
    total_histogram = histograms.sum(axis=0)
    r_eff_global = compute_effective_radius(total_histogram, bin_centers)

    if np.isnan(r_eff_global):
        logger.warning("Global effective radius: INVALID (no data)")
    else:
        logger.info(f"Global effective radius: {r_eff_global:.3f} μm")

    valid_r_eff = r_eff_per_slice[~np.isnan(r_eff_per_slice)]
    if len(valid_r_eff) > 0:
        logger.info(f"Mean per-slice effective radius: {valid_r_eff.mean():.3f} μm")
    else:
        logger.warning("Mean per-slice effective radius: INVALID (no valid slices)")

    # Save results to HDF5
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, 'w') as f:
        # Create main group
        grp = f.create_group(population)

        # Save arrays
        grp.create_dataset('positions_um', data=positions_um)
        grp.create_dataset('n_axons', data=n_axons)
        grp.create_dataset('histograms', data=histograms)
        grp.create_dataset('bin_edges', data=bin_edges)
        grp.create_dataset('bin_centers', data=bin_centers)
        grp.create_dataset('r_eff_per_slice', data=r_eff_per_slice)

        # Save metadata
        grp.attrs['population'] = population
        grp.attrs['r_eff_global'] = r_eff_global
        grp.attrs['voxel_size_um'] = voxel_size_um
        grp.attrs['interval_um'] = interval_um
        grp.attrs['n_intervals'] = n_intervals
        grp.attrs['fiber_direction'] = fiber_direction
        grp.attrs['volume_file'] = str(volume_file)

        # Optionally save extracted slices
        if save_slices and saved_slices:
            slices_array = np.array(saved_slices)
            grp.create_dataset('extracted_slices', data=slices_array,
                             compression='gzip', compression_opts=1)

    logger.info(f"Saved results: {output_file}")

    return {
        'output_file': str(output_file),
        'population': population,
        'n_intervals': n_intervals,
        'r_eff_global': r_eff_global,
        'positions_um': positions_um.tolist(),
        'n_axons': n_axons.tolist(),
        'r_eff_per_slice': r_eff_per_slice.tolist()
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Measure axon radii using oblique slicing perpendicular to fiber direction'
    )
    parser.add_argument('volume_file', type=Path,
                       help='Path to aligned HDF5 volume')
    parser.add_argument('density_json', type=Path,
                       help='Path to density metadata JSON (from preprocess_2)')
    parser.add_argument('output_file', type=Path,
                       help='Output HDF5 file for radius measurements')
    parser.add_argument('--interval', type=float, default=10.0,
                       help='Interval between measurement planes in μm (default: 10.0)')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                       help='Voxel size in micrometers (default: 0.05)')
    parser.add_argument('--save-slices', action='store_true',
                       help='Save extracted 2D slices for visualization (increases file size)')

    args = parser.parse_args()

    # Load density metadata
    logger.info(f"Loading density metadata: {args.density_json}")
    with open(args.density_json, 'r') as f:
        density_metadata = json.load(f)

    # Run measurement
    results = measure_radii_oblique_intervals(
        args.volume_file,
        density_metadata,
        args.output_file,
        interval_um=args.interval,
        voxel_size_um=args.voxel_size,
        save_slices=args.save_slices
    )

    logger.info("Completed successfully!")
