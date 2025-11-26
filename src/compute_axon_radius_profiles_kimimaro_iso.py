#!/usr/bin/env python3
"""
Compute axon radius profiles using Kimimaro for skeletonization.

This version combines:
- Kimimaro's efficient batch skeletonization
- Manual perpendicular cross-section sampling (from compute_axon_radius_profiles.py)
- Isotropic resampling for anisotropic volumes

For each axon:
1. Extract skeleton from Kimimaro batch result
2. Walk along skeleton points at regular intervals
3. Sample perpendicular cross-section at each point
4. Compute equivalent circular radius from cross-section area
"""

import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, Union

import h5py
import kimimaro
import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.ndimage import zoom
from skimage.measure import label, regionprops
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_voxel_size(voxel_size_um: Union[float, Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """
    Parse voxel size to a (vz, vy, vx) tuple matching array axis order (Z, Y, X).
    """
    if isinstance(voxel_size_um, (tuple, list)):
        if len(voxel_size_um) == 3:
            return tuple(float(v) for v in voxel_size_um)
        elif len(voxel_size_um) == 1:
            v = float(voxel_size_um[0])
            return (v, v, v)
        else:
            raise ValueError(f"Expected 1 or 3 voxel size values, got {len(voxel_size_um)}")
    v = float(voxel_size_um)
    return (v, v, v)


def parse_voxel_size_arg(value: str) -> Union[float, Tuple[float, float, float]]:
    """Parse voxel size CLI argument: single float or comma-separated triple."""
    if ',' in value:
        parts = value.split(',')
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"Expected 1 or 3 values, got {len(parts)}: '{value}'"
            )
        return tuple(float(p.strip()) for p in parts)
    return float(value)


def resample_to_isotropic(volume: np.ndarray,
                          voxel_size: Tuple[float, float, float]
                          ) -> Tuple[np.ndarray, float]:
    """
    Resample anisotropic volume to isotropic voxels.
    Uses nearest-neighbor interpolation (order=0) to preserve label integrity.
    """
    vz, vy, vx = voxel_size

    if np.allclose([vz, vy, vx], vz):
        logger.info("Volume is already isotropic, no resampling needed")
        return volume, vz

    target_size = max(vz, vy, vx)
    zoom_factors = (vz / target_size, vy / target_size, vx / target_size)

    logger.info(f"Resampling anisotropic volume to isotropic:")
    logger.info(f"  Original voxel size (Z, Y, X): vz={vz:.4f}, vy={vy:.4f}, vx={vx:.4f} μm")
    logger.info(f"  Target voxel size: {target_size:.4f} μm (isotropic)")
    logger.info(f"  Zoom factors (Z, Y, X): {zoom_factors}")
    logger.info(f"  Original shape: {volume.shape}")

    resampled = zoom(volume, zoom_factors, order=0, mode='nearest')
    logger.info(f"  Resampled shape: {resampled.shape}")

    return resampled, target_size


def load_metadata(mat_file: Path) -> dict:
    """Load metadata JSON file associated with a .mat file."""
    metadata_file = mat_file.with_suffix('.json')
    if metadata_file.exists():
        logger.info(f"Found metadata file: {metadata_file}")
        with open(metadata_file, 'r') as f:
            return json.load(f)
    else:
        logger.info(f"No metadata file found at {metadata_file}")
        return {}


def load_mat_volume(mat_file: Path) -> np.ndarray:
    """Load labeled volume from .mat file (v5.0 or v7.3/HDF5)."""
    logger.info(f"Loading {mat_file}")

    try:
        with h5py.File(str(mat_file), 'r') as f:
            volume_key = None
            for key in f.keys():
                if not key.startswith('#') and not key.startswith('_'):
                    volume_key = key
                    break
            if volume_key is None:
                raise ValueError(f"No data found in {mat_file}")
            volume = f[volume_key][:]
            logger.info(f"Loaded HDF5 format, volume shape: {volume.shape}, dtype: {volume.dtype}")
            return volume
    except OSError:
        logger.info("HDF5 failed, trying scipy.io for older MATLAB format...")
        mat_data = sio.loadmat(str(mat_file))
        volume_key = None
        priority_keys = ['myelinated_axons', 'volume', 'labels', 'final_lbl']
        for pkey in priority_keys:
            if pkey in mat_data:
                volume_key = pkey
                break
        if volume_key is None:
            for key in mat_data.keys():
                if not key.startswith('__'):
                    volume_key = key
                    break
        if volume_key is None:
            raise ValueError(f"No data found in {mat_file}")
        volume = mat_data[volume_key]
        logger.info(f"Loaded scipy.io format (key: {volume_key}), volume shape: {volume.shape}, dtype: {volume.dtype}")
        return volume


# ============================================================================
# Cross-section sampling functions (from compute_axon_radius_profiles.py)
# ============================================================================

def unit_tangent_vector(curve):
    """Compute unit tangent vectors along a curve."""
    d_curve = np.gradient(curve, axis=0)
    ds = np.sqrt(np.sum(d_curve**2, axis=1, keepdims=True))
    ds[ds == 0] = 1e-5
    return d_curve / ds


def rotation_matrix_3D(vector, theta):
    """Create rotation matrix for counterclockwise rotation about a unit vector."""
    a = np.cos(theta / 2.0)
    b, c, d = -vector * np.sin(theta / 2.0)
    aa, bb, cc, dd = a**2, b**2, c**2, d**2
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([
        [aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
        [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
        [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]
    ])


def unit_normal_vector(vec1, vec2):
    """Compute unit normal vector from cross product."""
    n = np.cross(vec1, vec2)
    if np.allclose(n, 0):
        n = vec1.copy()
    s = max(np.linalg.norm(n), 1e-5)
    return n / s


def angle_between(vec1, vec2):
    """Compute angle between two vectors."""
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle)


def sample_perpendicular_cross_section(binary_volume, point, tangent_vec,
                                       plane_radius=5.0, plane_resolution=0.5,
                                       interpolator=None):
    """
    Sample a perpendicular cross-section at a skeleton point.

    Args:
        binary_volume: 3D binary array of the axon
        point: (3,) skeleton point coordinates
        tangent_vec: (3,) unit tangent vector at the point
        plane_radius: Radius of sampling plane in voxels
        plane_resolution: Resolution of sampling plane
        interpolator: Optional pre-built RegularGridInterpolator

    Returns:
        area: Cross-section area in voxels^2, or None if sampling failed
    """
    # Create sampling grid in XY plane
    coords = np.arange(-plane_radius, plane_radius + plane_resolution, plane_resolution)
    x, y = np.meshgrid(coords, coords)
    z = np.zeros_like(x)

    # Flatten for rotation
    xyz = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    # Compute rotation to align Z-axis with tangent vector
    z_axis = np.array([0, 0, 1])

    if np.allclose(tangent_vec, z_axis) or np.allclose(tangent_vec, -z_axis):
        if np.dot(tangent_vec, z_axis) < 0:
            rotated_plane = xyz * np.array([1, 1, -1])
        else:
            rotated_plane = xyz
    else:
        rot_axis = unit_normal_vector(z_axis, tangent_vec)
        theta = angle_between(z_axis, tangent_vec)
        rot_mat = rotation_matrix_3D(rot_axis, theta)
        rotated_plane = xyz @ rot_mat.T

    # Translate to skeleton point
    sample_coords = rotated_plane + point

    # Set up interpolator if not provided
    if interpolator is None:
        sz = binary_volume.shape
        interpolator = rgi(
            (range(sz[0]), range(sz[1]), range(sz[2])),
            binary_volume.astype(float),
            bounds_error=False,
            fill_value=0
        )

    # Sample cross-section
    cross_section = interpolator(sample_coords)
    bw_cross_section = (cross_section >= 0.5).reshape(x.shape)

    # Label connected components
    labeled, n_components = label(bw_cross_section, connectivity=1, return_num=True)

    if n_components == 0:
        return None

    # Find the component at the center (should be the axon)
    center_idx = bw_cross_section.shape[0] // 2
    center_label = labeled[center_idx, center_idx]

    if center_label == 0:
        # No component at center - find largest
        props = regionprops(labeled)
        if not props:
            return None
        areas = [p.area for p in props]
        center_label = props[np.argmax(areas)].label

    # Get area of the main component
    main_component = (labeled == center_label)
    area = np.sum(main_component)

    # Convert from grid squares to actual area
    area_voxels = area * (plane_resolution ** 2)

    return area_voxels


# ============================================================================
# Global variables for worker processes
# ============================================================================

_shared_volume = None
_shared_skeletons = None
_shared_params = None


def init_worker(volume, skeletons, params):
    """Initialize worker process with shared data."""
    global _shared_volume, _shared_skeletons, _shared_params
    _shared_volume = volume
    _shared_skeletons = skeletons
    _shared_params = params


def process_single_axon(axon_label):
    """
    Process a single axon to extract radius profile along skeleton.

    Uses Kimimaro skeleton and manual cross-section sampling.
    """
    global _shared_volume, _shared_skeletons, _shared_params

    voxel_size_um = _shared_params['voxel_size_um']
    plane_radius = _shared_params['plane_radius']
    plane_resolution = _shared_params['plane_resolution']
    step_size = _shared_params['step_size']

    try:
        # Get skeleton from Kimimaro results
        if axon_label not in _shared_skeletons:
            logger.debug(f"Axon {axon_label}: No skeleton found")
            return None

        skel = _shared_skeletons[axon_label]
        # Kimimaro returns vertices in physical units (nm) in (x, y, z) order
        # We need to:
        # 1. Convert from nm back to voxels by dividing by anisotropy (in nm)
        # 2. Reverse coordinate order from (x, y, z) to (z, y, x) to match volume axes
        anisotropy_nm = _shared_params['anisotropy_nm']  # (x, y, z) in nm
        vertices_nm = skel.vertices  # In physical nm, (x, y, z) order
        # Convert to voxels: divide by anisotropy
        vertices_voxels_xyz = vertices_nm / np.array(anisotropy_nm)
        # Convert from (x, y, z) to (z, y, x) to match volume indexing
        vertices = vertices_voxels_xyz[:, ::-1]

        if len(vertices) < 3:
            logger.debug(f"Axon {axon_label}: Too few vertices ({len(vertices)})")
            return None

        # Extract binary volume for this axon
        axon_binary = (_shared_volume == axon_label).astype(np.uint8)

        # Get bounding box with padding
        coords = np.argwhere(axon_binary)
        if len(coords) < 100:
            logger.debug(f"Axon {axon_label}: Too few voxels ({len(coords)})")
            return None

        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)

        padding = int(plane_radius) + 5
        min_padded = np.maximum(min_coords - padding, 0)
        max_padded = np.minimum(max_coords + padding, np.array(axon_binary.shape))

        cropped = axon_binary[
            min_padded[0]:max_padded[0],
            min_padded[1]:max_padded[1],
            min_padded[2]:max_padded[2]
        ].copy()

        # Adjust skeleton coordinates to cropped volume
        skel_cropped = vertices - min_padded

        # Subsample skeleton to step_size intervals
        # Compute cumulative arc length in physical units
        diffs = np.diff(skel_cropped, axis=0)
        diffs_physical = diffs * voxel_size_um  # Isotropic, so same for all axes
        segment_lengths = np.sqrt(np.sum(diffs_physical**2, axis=1))
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_length[-1]

        if total_length < step_size:
            sampled_skel = skel_cropped
        else:
            sample_positions = np.arange(0, total_length, step_size)
            sampled_skel = []
            for pos in sample_positions:
                idx = np.searchsorted(cumulative_length, pos) - 1
                idx = max(0, min(idx, len(skel_cropped) - 2))
                t = (pos - cumulative_length[idx]) / (segment_lengths[idx] + 1e-10)
                t = np.clip(t, 0, 1)
                point = skel_cropped[idx] + t * (skel_cropped[idx + 1] - skel_cropped[idx])
                sampled_skel.append(point)
            sampled_skel = np.array(sampled_skel)

        if len(sampled_skel) < 2:
            logger.debug(f"Axon {axon_label}: Too few sampled skeleton points ({len(sampled_skel)})")
            return None

        # Compute tangent vectors
        tangent_vecs = unit_tangent_vector(sampled_skel)

        # Create interpolator once for all cross-sections
        sz = cropped.shape
        interpolator = rgi(
            (range(sz[0]), range(sz[1]), range(sz[2])),
            cropped.astype(float),
            bounds_error=False,
            fill_value=0
        )

        # Sample cross-sections and compute radii
        radii = []
        valid_skel_points = []

        for i, (point, tangent) in enumerate(zip(sampled_skel, tangent_vecs)):
            if np.allclose(tangent, 0):
                continue

            area = sample_perpendicular_cross_section(
                cropped, point, tangent,
                plane_radius=plane_radius,
                plane_resolution=plane_resolution,
                interpolator=interpolator
            )

            if area is not None and area > 0:
                radius_voxels = np.sqrt(area / np.pi)
                radius_um = radius_voxels * voxel_size_um

                radii.append(radius_um)
                original_point = point + min_padded
                valid_skel_points.append(original_point * voxel_size_um)

        if len(radii) < 2:
            logger.debug(f"Axon {axon_label}: Too few valid radii ({len(radii)})")
            return None

        radii = np.array(radii)
        skel_coords = np.array(valid_skel_points)

        return {
            'label': axon_label,
            'radii_um': radii,
            'skeleton_um': skel_coords,
            'n_points': len(radii),
            'mean_radius_um': np.mean(radii),
            'std_radius_um': np.std(radii),
            'length_um': total_length
        }

    except Exception as e:
        logger.debug(f"Axon {axon_label}: Processing failed - {e}")
        return None


def compute_radius_profiles(mat_file: Path,
                            output_file: Path,
                            voxel_size_um: Union[float, Tuple[float, float, float], None] = None,
                            plane_radius: float = 10.0,
                            plane_resolution: float = 0.5,
                            step_size: float = 2.0,
                            n_jobs: int = -1,
                            max_axons: int = 0,
                            anisotropy_mode: str = 'simple',
                            kimimaro_parallel: int = 0):
    """
    Compute radius profiles for all axons using Kimimaro skeletonization
    and manual cross-section sampling.

    Args:
        mat_file: Path to .mat file with labeled axons
        output_file: Path to save results (.npz)
        voxel_size_um: Voxel size in micrometers (vz, vy, vx) or scalar or None (load from metadata)
        plane_radius: Radius of sampling plane in voxels
        plane_resolution: Resolution of sampling plane
        step_size: Step size along skeleton in physical units (μm)
        n_jobs: Number of parallel jobs for cross-section sampling (-1 = all CPUs)
        max_axons: Maximum number of axons to process (0 = all)
        anisotropy_mode: 'simple' (resample to isotropic) or 'none'
        kimimaro_parallel: Parallel workers for Kimimaro (0 = all CPUs)
    """
    if not mat_file.exists():
        raise FileNotFoundError(f"Input file not found: {mat_file}")

    # Load metadata if available
    metadata = load_metadata(mat_file)

    if voxel_size_um is None:
        if 'voxel_size_um' in metadata:
            voxel_size_um = tuple(metadata['voxel_size_um'])
            logger.info(f"Using voxel_size from metadata: {voxel_size_um}")
        else:
            voxel_size_um = 0.05
            logger.warning("No voxel_size specified, defaulting to 0.05 (isotropic)")

    voxel_size_tuple = parse_voxel_size(voxel_size_um)
    vz, vy, vx = voxel_size_tuple

    # Load volume
    volume = load_mat_volume(mat_file)

    # Handle anisotropic voxels
    if anisotropy_mode == 'simple':
        volume, iso_voxel_size = resample_to_isotropic(volume, voxel_size_tuple)
        voxel_size_tuple = (iso_voxel_size, iso_voxel_size, iso_voxel_size)
        vz = vy = vx = iso_voxel_size
    elif anisotropy_mode != 'none':
        raise ValueError(f"Unknown anisotropy_mode: {anisotropy_mode}")

    logger.info(f"Working with isotropic voxel size: {vz:.4f} μm")

    # Get unique axon labels
    axon_labels = np.unique(volume)
    axon_labels = axon_labels[axon_labels > 0]
    n_total = len(axon_labels)
    logger.info(f"Found {n_total} axons")

    if max_axons > 0:
        axon_labels = axon_labels[:max_axons]
        logger.info(f"Processing first {max_axons} axons")

    # Run Kimimaro skeletonization
    logger.info(f"Running Kimimaro skeletonization (parallel={kimimaro_parallel})...")
    logger.info(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")

    # Kimimaro expects (x, y, z) order for anisotropy, in nm
    anisotropy_nm = (vx * 1000, vy * 1000, vz * 1000)
    logger.info(f"Anisotropy for Kimimaro (X,Y,Z in nm): {anisotropy_nm}")

    skels = kimimaro.skeletonize(
        volume,
        # teasar_params={
        #     "scale": 1.5,
        #     "const": 50,
        #     "pdrf_scale": 100000,
        #     "pdrf_exponent": 4,
        #     "soma_acceptance_threshold": 3500,
        #     "soma_detection_threshold": 750,
        #     "soma_invalidation_const": 300,
        #     "soma_invalidation_scale": 2,
        #     "max_paths": 300,
        # },
        progress=True,
        parallel=kimimaro_parallel,
        fix_branching=True,
        parallel_chunk_size=50,
        in_place=False,
        anisotropy=anisotropy_nm,
    )

    logger.info(f"Skeletonization complete. Got {len(skels)} skeletons")

    # Process axons with cross-section sampling
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    params = {
        'voxel_size_um': vz,  # Isotropic after resampling
        'plane_radius': plane_radius,
        'plane_resolution': plane_resolution,
        'step_size': step_size,
        'anisotropy_nm': anisotropy_nm,  # For converting Kimimaro vertices back to voxels
    }

    logger.info(f"Processing {len(axon_labels)} axons with {n_jobs} workers")
    logger.info(f"Parameters: plane_radius={plane_radius}, resolution={plane_resolution}, step={step_size} μm")

    results = []
    if n_jobs == 1:
        init_worker(volume, skels, params)
        for axon_label in tqdm(axon_labels, desc="Processing axons"):
            result = process_single_axon(axon_label)
            if result is not None:
                results.append(result)
    else:
        with mp.Pool(n_jobs, initializer=init_worker, initargs=(volume, skels, params)) as pool:
            for result in tqdm(pool.imap_unordered(process_single_axon, axon_labels),
                               total=len(axon_labels), desc="Processing axons"):
                if result is not None:
                    results.append(result)

    logger.info(f"Successfully processed {len(results)}/{len(axon_labels)} axons")

    if len(results) == 0:
        logger.error("No axons were successfully processed!")
        return

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)

    labels = np.array([r['label'] for r in results])
    n_points = np.array([r['n_points'] for r in results])
    mean_radii = np.array([r['mean_radius_um'] for r in results])
    std_radii = np.array([r['std_radius_um'] for r in results])
    lengths = np.array([r['length_um'] for r in results])

    radii_profiles = np.array([r['radii_um'] for r in results], dtype=object)
    skeleton_coords = np.array([r['skeleton_um'] for r in results], dtype=object)

    all_radii = np.concatenate([r['radii_um'] for r in results])

    np.savez(
        output_file,
        labels=labels,
        n_points=n_points,
        mean_radii_um=mean_radii,
        std_radii_um=std_radii,
        lengths_um=lengths,
        radii_profiles_um=radii_profiles,
        skeleton_coords_um=skeleton_coords,
        all_radii_um=all_radii,
        voxel_size_um=np.array(voxel_size_tuple),
        plane_radius=plane_radius,
        plane_resolution=plane_resolution,
        step_size=step_size,
        source_file=str(mat_file)
    )

    logger.info(f"Saved results to {output_file}")

    # Print summary
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total axons processed: {len(results)}")
    logger.info(f"  Total radius samples: {len(all_radii)}")
    logger.info(f"  Mean axon length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f} μm")
    logger.info(f"  Mean points per axon: {np.mean(n_points):.1f}")
    logger.info(f"  Mean radius: {np.mean(all_radii):.3f} ± {np.std(all_radii):.3f} μm")
    logger.info(f"  Median radius: {np.median(all_radii):.3f} μm")

    r_eff = (np.mean(all_radii**6) / np.mean(all_radii**2)) ** 0.25
    logger.info(f"  Effective radius (r_eff): {r_eff:.3f} μm")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute axon radius profiles using Kimimaro skeletonization and manual cross-section sampling'
    )
    parser.add_argument('mat_file', type=Path,
                        help='Path to .mat file with labeled axons')
    parser.add_argument('output_file', type=Path,
                        help='Output .npz file for results')
    parser.add_argument('--voxel-size', type=parse_voxel_size_arg, default=None,
                        help='Voxel size in micrometers: single value for isotropic (e.g., 0.05) '
                             'or vz,vy,vx for anisotropic (e.g., 0.015,0.015,0.05). '
                             'If not specified, reads from *.json metadata file.')
    parser.add_argument('--plane-radius', type=float, default=10.0,
                        help='Radius of sampling plane in voxels (default: 10.0)')
    parser.add_argument('--plane-resolution', type=float, default=0.5,
                        help='Resolution of sampling plane (default: 0.5)')
    parser.add_argument('--step-size', type=float, default=2.0,
                        help='Step size along skeleton in μm (default: 2.0)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Parallel jobs for cross-section sampling (default: -1 = all CPUs)')
    parser.add_argument('--max-axons', type=int, default=0,
                        help='Maximum axons to process (0 = all)')
    parser.add_argument('--anisotropy-mode', type=str, default='simple',
                        choices=['simple', 'none'],
                        help="'simple' resamples to isotropic, 'none' keeps anisotropic. Default: 'simple'")
    parser.add_argument('--kimimaro-parallel', type=int, default=0,
                        help='Parallel workers for Kimimaro skeletonization (0 = all CPUs)')

    args = parser.parse_args()

    compute_radius_profiles(
        args.mat_file,
        args.output_file,
        voxel_size_um=args.voxel_size,
        plane_radius=args.plane_radius,
        plane_resolution=args.plane_resolution,
        step_size=args.step_size,
        n_jobs=args.n_jobs,
        max_axons=args.max_axons,
        anisotropy_mode=args.anisotropy_mode,
        kimimaro_parallel=args.kimimaro_parallel
    )
