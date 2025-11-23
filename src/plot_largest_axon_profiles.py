#!/usr/bin/env python3
"""
Plot radius profiles of the largest axons in a bundle.

Identifies the N largest axons by their maximum radius along their profile,
then plots individual radius profiles for each.

Supports both HDF5 and Zarr formats, including Zarr files with CC/CG subgroups.
"""

import argparse
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for worker processes
_volume_data = None
_slice_axis = None


def _init_worker(volume_data: np.ndarray, slice_axis: int):
    """Initialize worker process with volume data and slice axis."""
    global _volume_data, _slice_axis
    _volume_data = volume_data
    _slice_axis = slice_axis


def _process_single_slice_axons(args: Tuple[int, float, bool, float]) -> Tuple[int, Dict[int, float]]:
    """
    Process a single slice to extract radii per axon. Helper function for parallel processing.

    Args:
        args: Tuple of (slice_index, voxel_size_um, use_minor_axis, max_ellipse_ratio)

    Returns:
        Tuple of (slice_index, dict mapping label -> radius)
    """
    z, voxel_size_um, use_minor_axis, max_ellipse_ratio = args

    # Slice along the appropriate axis
    if _slice_axis == 0:
        slice_2d = _volume_data[z, :, :]
    elif _slice_axis == 1:
        slice_2d = _volume_data[:, z, :]
    else:  # axis == 2
        slice_2d = _volume_data[:, :, z]

    # Use regionprops to efficiently extract areas of all regions
    regions = regionprops(slice_2d)

    if len(regions) == 0:
        return (z, {})

    # Compute radius for each region (axon)
    slice_radii = {}
    for region in regions:
        label = region.label
        area_voxels = region.area

        # Filter by ellipse area ratio if specified
        if max_ellipse_ratio > 0:
            ellipse_area = np.pi * (region.axis_major_length / 2) * (region.axis_minor_length / 2)
            if ellipse_area > max_ellipse_ratio * area_voxels:
                continue  # Skip sparse/fragmented regions

        if use_minor_axis:
            # Use minor axis of fitted ellipse (diameter -> radius)
            radius_um = (region.axis_minor_length / 2) * voxel_size_um
        else:
            # Compute circular-equivalent radius from area
            area_um2 = area_voxels * (voxel_size_um ** 2)
            radius_um = np.sqrt(area_um2 / np.pi)

        slice_radii[label] = radius_um

    return (z, slice_radii)


def extract_axon_radii_per_slice(volume: np.ndarray,
                                  voxel_size_um: float = 0.05,
                                  slice_axis: int = 0,
                                  use_minor_axis: bool = False,
                                  max_ellipse_ratio: float = 0.0,
                                  n_jobs: int = -1) -> Dict[int, Dict[int, float]]:
    """
    Extract radius measurements for each axon in each slice.

    Args:
        volume: 3D labeled volume
        voxel_size_um: Voxel size in micrometers
        slice_axis: Axis to slice along (0=Z, 1=Y, 2=X)
        use_minor_axis: If True, use ellipse minor axis; otherwise use circular-equivalent radius
        max_ellipse_ratio: Max ratio of ellipse area to voxel area (0 = no filter)
        n_jobs: Number of parallel jobs (-1 = use all CPUs)

    Returns:
        Dict mapping slice_idx -> dict mapping label -> radius (in μm)
    """
    logger.info(f"Extracting radii per axon per slice from volume shape {volume.shape}")
    logger.info(f"Slicing along axis {slice_axis}, radius method: {'minor axis' if use_minor_axis else 'circular-equivalent'}")

    n_slices = volume.shape[slice_axis]

    # Determine number of workers
    if n_jobs == -1:
        n_workers = cpu_count()
    elif n_jobs == 1:
        n_workers = 1
    else:
        n_workers = min(n_jobs, cpu_count())

    logger.info(f"Using {n_workers} parallel workers")

    # Prepare arguments for parallel processing
    args_list = [(z, voxel_size_um, use_minor_axis, max_ellipse_ratio) for z in range(n_slices)]

    # Process in parallel
    if n_workers > 1:
        with Pool(processes=n_workers,
                  initializer=_init_worker,
                  initargs=(volume, slice_axis)) as pool:
            results = list(tqdm(
                pool.imap(_process_single_slice_axons, args_list),
                total=n_slices,
                desc="Processing slices",
                unit="slice"
            ))
    else:
        # Serial processing
        global _volume_data, _slice_axis
        _volume_data = volume
        _slice_axis = slice_axis
        results = [_process_single_slice_axons(args) for args in tqdm(
            args_list,
            desc="Processing slices",
            unit="slice"
        )]

    # Convert list of tuples to dictionary
    radii_data = {z: slice_radii for z, slice_radii in results}

    return radii_data


def compute_max_radius_per_axon(radii_data: Dict[int, Dict[int, float]]) -> Dict[int, float]:
    """
    Compute the maximum radius for each axon across all slices.

    Args:
        radii_data: Dict mapping slice_idx -> dict mapping label -> radius

    Returns:
        Dict mapping label -> max_radius
    """
    logger.info("Computing maximum radius per axon")

    max_radii = {}

    for z, slice_radii in radii_data.items():
        for label, radius in slice_radii.items():
            if label not in max_radii:
                max_radii[label] = radius
            else:
                max_radii[label] = max(max_radii[label], radius)

    logger.info(f"Found {len(max_radii)} unique axons")

    return max_radii


def get_largest_axons(max_radii: Dict[int, float], n_largest: int = 10) -> List[int]:
    """
    Get the labels of the N largest axons.

    Args:
        max_radii: Dict mapping label -> max_radius
        n_largest: Number of largest axons to return

    Returns:
        List of labels sorted by max radius (descending)
    """
    # Sort by max radius in descending order
    sorted_labels = sorted(max_radii.keys(), key=lambda label: max_radii[label], reverse=True)

    # Take the top N
    largest_labels = sorted_labels[:n_largest]

    logger.info(f"Top {n_largest} axons by max radius:")
    for i, label in enumerate(largest_labels, 1):
        logger.info(f"  {i}. Label {label}: max radius = {max_radii[label]:.3f} μm")

    return largest_labels


def extract_radius_profiles(radii_data: Dict[int, Dict[int, float]],
                           largest_labels: List[int]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract radius profiles for specified axons.

    Args:
        radii_data: Dict mapping slice_idx -> dict mapping label -> radius
        largest_labels: List of axon labels to extract

    Returns:
        Dict mapping label -> (positions, radii) where positions are slice indices
    """
    logger.info(f"Extracting radius profiles for {len(largest_labels)} axons")

    profiles = {}

    for label in largest_labels:
        positions = []
        radii = []

        for z in sorted(radii_data.keys()):
            if label in radii_data[z]:
                positions.append(z)
                radii.append(radii_data[z][label])

        profiles[label] = (np.array(positions), np.array(radii))

    return profiles


def plot_largest_axon_profiles(profiles: Dict[int, Tuple[np.ndarray, np.ndarray]],
                               max_radii: Dict[int, float],
                               output_file: Path,
                               voxel_size_um: float = 0.05,
                               bundle_name: str = "Bundle"):
    """
    Plot radius profiles of the largest axons.

    Args:
        profiles: Dict mapping label -> (positions, radii)
        max_radii: Dict mapping label -> max_radius (for sorting in legend)
        output_file: Path to save figure
        voxel_size_um: Voxel size in micrometers
        bundle_name: Name of the bundle for title
    """
    logger.info(f"Plotting radius profiles for {len(profiles)} axons")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Sort labels by max radius for consistent legend ordering
    sorted_labels = sorted(profiles.keys(), key=lambda label: max_radii[label], reverse=True)

    # Use colormap for distinguishable colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(sorted_labels)))

    for i, label in enumerate(sorted_labels):
        positions, radii = profiles[label]
        positions_um = positions * voxel_size_um

        ax.plot(positions_um, radii, linewidth=1.5, alpha=0.8,
               color=colors[i], label=f'Axon {label} (max: {max_radii[label]:.2f} μm)')

    ax.set_xlabel('Position along tract (μm)', fontsize=12)
    ax.set_ylabel('Radius (μm)', fontsize=12)
    ax.set_title(f'{bundle_name}: Radius Profiles of Largest Axons', fontsize=14, pad=15)
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved radius profiles to {output_file}")


def analyze_largest_axons(volume_file: Path,
                          output_dir: Path,
                          n_largest: int = 10,
                          voxel_size_um: float = 0.05,
                          use_minor_axis: bool = False,
                          max_ellipse_ratio: float = 0.0,
                          load_in_memory: bool = False,
                          n_jobs: int = -1):
    """
    Analyze and plot radius profiles of the largest axons.

    Supports both HDF5 and Zarr formats, including Zarr files with CC/CG subgroups.

    Args:
        volume_file: Path to aligned volume file (HDF5 or Zarr)
        output_dir: Directory for output plots
        n_largest: Number of largest axons to plot
        voxel_size_um: Voxel size in micrometers
        use_minor_axis: If True, use ellipse minor axis; otherwise use circular-equivalent radius
        max_ellipse_ratio: Max ratio of ellipse area to voxel area (0 = no filter)
        load_in_memory: If True, load entire volume into memory first (accepted for API consistency)
        n_jobs: Number of parallel jobs (-1 = use all CPUs)
    """
    # Note: load_in_memory is accepted for API consistency but this function
    # always loads the full volume into memory for per-axon tracking
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing largest axons: {volume_file.name}")
    logger.info(f"{'='*80}\n")

    # Detect format
    is_zarr = volume_file.suffix == '.zarr' or (volume_file.is_dir() and (volume_file / 'zarr.json').exists())

    if is_zarr:
        import zarr
        store = zarr.storage.LocalStore(str(volume_file))
        root = zarr.open_group(store, mode='r')

        # Check if this is a CC/CG subvolume structure
        has_cc = 'cc' in root
        has_cg = 'cg' in root

        if has_cc or has_cg:
            # Process CC and CG subgroups
            populations = []
            if has_cc:
                populations.append(('cc', 'CC'))
            if has_cg:
                populations.append(('cg', 'CG'))

            for subgroup, pop_name in populations:
                logger.info(f"\n{'-'*40}")
                logger.info(f"Processing {pop_name} population")
                logger.info(f"{'-'*40}")

                # Get metadata for this population
                pop_meta = root.attrs.get(subgroup, {})
                dominant_axis = pop_meta.get('dominant_axis', 0)
                n_axons = pop_meta.get('n_axons', 0)

                # Load volume
                volume = root[subgroup]['0'][:]

                logger.info(f"Population: {pop_name}")
                logger.info(f"Total axons: {n_axons}")
                logger.info(f"Volume shape: {volume.shape}")
                logger.info(f"Dominant axis: {dominant_axis}")

                # Extract radii per axon per slice
                radii_data = extract_axon_radii_per_slice(
                    volume, voxel_size_um,
                    slice_axis=dominant_axis,
                    use_minor_axis=use_minor_axis,
                    max_ellipse_ratio=max_ellipse_ratio,
                    n_jobs=n_jobs
                )

                # Compute max radius per axon
                max_radii = compute_max_radius_per_axon(radii_data)

                if len(max_radii) == 0:
                    logger.warning(f"No axons found in {pop_name}")
                    continue

                # Get N largest axons
                largest_labels = get_largest_axons(max_radii, n_largest)

                # Extract radius profiles
                profiles = extract_radius_profiles(radii_data, largest_labels)

                # Plot radius profiles
                sample_name = volume_file.stem.replace('_spatial', '')
                bundle_name = f"{sample_name} {pop_name}"
                output_file = output_dir / f'{sample_name}_{pop_name}_largest_{n_largest}_axon_profiles.png'
                plot_largest_axon_profiles(profiles, max_radii, output_file, voxel_size_um, bundle_name)
        else:
            # Single population Zarr (legacy format)
            volume = root['0'][:]
            bundle_id = root.attrs.get('bundle_id', None)
            if bundle_id is not None:
                population_name = f"Bundle_{bundle_id:02d}"
            else:
                population_name = volume_file.stem

            logger.info(f"Population: {population_name}")
            logger.info(f"Volume shape: {volume.shape}")

            # Extract radii per axon per slice
            radii_data = extract_axon_radii_per_slice(
                volume, voxel_size_um,
                slice_axis=0,
                use_minor_axis=use_minor_axis,
                max_ellipse_ratio=max_ellipse_ratio,
                n_jobs=n_jobs
            )

            # Compute max radius per axon
            max_radii = compute_max_radius_per_axon(radii_data)

            # Get N largest axons
            largest_labels = get_largest_axons(max_radii, n_largest)

            # Extract radius profiles
            profiles = extract_radius_profiles(radii_data, largest_labels)

            # Plot radius profiles
            bundle_name = population_name
            output_file = output_dir / f'{volume_file.stem}_largest_{n_largest}_axon_profiles.png'
            plot_largest_axon_profiles(profiles, max_radii, output_file, voxel_size_um, bundle_name)
    else:
        # HDF5 format
        with h5py.File(volume_file, 'r') as f:
            volume = f['labels'][:]
            bundle_id = f['labels'].attrs.get('bundle_id', 'Unknown')
            n_axons = f['labels'].attrs.get('n_axons', 'Unknown')

        logger.info(f"Bundle ID: {bundle_id}")
        logger.info(f"Total axons: {n_axons}")
        logger.info(f"Volume shape: {volume.shape}")

        # Extract radii per axon per slice
        radii_data = extract_axon_radii_per_slice(
            volume, voxel_size_um,
            slice_axis=0,
            use_minor_axis=use_minor_axis,
            max_ellipse_ratio=max_ellipse_ratio,
            n_jobs=n_jobs
        )

        # Compute max radius per axon
        max_radii = compute_max_radius_per_axon(radii_data)

        # Get N largest axons
        largest_labels = get_largest_axons(max_radii, n_largest)

        # Extract radius profiles
        profiles = extract_radius_profiles(radii_data, largest_labels)

        # Plot radius profiles
        bundle_name = f"Bundle {bundle_id}" if bundle_id != 'Unknown' else volume_file.stem
        output_file = output_dir / f'{volume_file.stem}_largest_{n_largest}_axon_profiles.png'
        plot_largest_axon_profiles(profiles, max_radii, output_file, voxel_size_um, bundle_name)

    logger.info(f"\n{'='*80}")
    logger.info(f"Completed: {volume_file.name}")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot radius profiles of the largest axons in a bundle'
    )
    parser.add_argument('volume_file', type=Path,
                       help='Path to aligned volume file (HDF5 or Zarr)')
    parser.add_argument('output_dir', type=Path,
                       help='Output directory for plots')
    parser.add_argument('--n-largest', type=int, default=10,
                       help='Number of largest axons to plot (default: 10)')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                       help='Voxel size in micrometers (default: 0.05)')
    parser.add_argument('--use-minor-axis', action='store_true',
                       help='Use ellipse minor axis instead of circular-equivalent radius')
    parser.add_argument('--max-ellipse-ratio', type=float, default=0.0,
                       help='Max ratio of ellipse area to voxel area to filter sparse regions (0 = no filter)')
    parser.add_argument('--load-in-memory', action='store_true',
                       help='Load entire volume into memory first (accepted for API consistency)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (default: -1 = use all CPUs)')

    args = parser.parse_args()

    analyze_largest_axons(
        args.volume_file,
        args.output_dir,
        n_largest=args.n_largest,
        voxel_size_um=args.voxel_size,
        use_minor_axis=args.use_minor_axis,
        max_ellipse_ratio=args.max_ellipse_ratio,
        load_in_memory=args.load_in_memory,
        n_jobs=args.n_jobs
    )
