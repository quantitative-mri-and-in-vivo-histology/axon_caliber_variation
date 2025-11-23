#!/usr/bin/env python3
"""
Analyze effective MRI-visible radius along aligned axon volumes.

Computes the effective radius per slice using the formula:
    r_eff = (<r^6>/<r^2>)^(1/4)

This metric is relevant for diffusion MRI sensitivity to axon caliber.
"""

import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm

# Global variables for worker processes
_volume_path = None
_is_zarr = None
_slice_axis = None
_subgroup = None
_volume_data = None  # For in-memory mode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _init_worker(volume_path: Path, is_zarr: bool, slice_axis: int, subgroup: str = None, volume_data: np.ndarray = None):
    """Initialize worker process with volume path, format, slice axis, and optional subgroup."""
    global _volume_path, _is_zarr, _slice_axis, _subgroup, _volume_data
    _volume_path = volume_path
    _is_zarr = is_zarr
    _slice_axis = slice_axis
    _subgroup = subgroup
    _volume_data = volume_data


def _process_single_slice(args: Tuple[int, float, np.ndarray, bool, float]) -> Tuple[int, np.ndarray, int]:
    """
    Process a single slice to extract radii histogram. Helper function for parallel processing.

    Reads slice directly from file (memory efficient).

    Args:
        args: Tuple of (slice_index, voxel_size_um, bin_edges, use_minor_axis, max_ellipse_ratio)

    Returns:
        Tuple of (slice_index, histogram_counts, n_axons)
    """
    z, voxel_size_um, bin_edges, use_minor_axis, max_ellipse_ratio = args

    # Use in-memory data if available, otherwise read from file
    if _volume_data is not None:
        # In-memory mode - slice from pre-loaded array
        if _slice_axis == 0:
            slice_2d = _volume_data[z, :, :]
        elif _slice_axis == 1:
            slice_2d = _volume_data[:, z, :]
        else:  # axis == 2
            slice_2d = _volume_data[:, :, z]
    elif _is_zarr:
        import zarr
        root = zarr.open(str(_volume_path), mode='r')
        # Navigate to subgroup if specified (e.g., 'cc' or 'cg')
        if _subgroup:
            data = root[_subgroup]['0']
        else:
            data = root['0']
        # Slice along the appropriate axis
        if _slice_axis == 0:
            slice_2d = data[z, :, :]
        elif _slice_axis == 1:
            slice_2d = data[:, z, :]
        else:  # axis == 2
            slice_2d = data[:, :, z]
    else:
        import h5py
        with h5py.File(_volume_path, 'r') as f:
            data = f['labels']
            if _slice_axis == 0:
                slice_2d = data[z, :, :]
            elif _slice_axis == 1:
                slice_2d = data[:, z, :]
            else:  # axis == 2
                slice_2d = data[:, :, z]

    # Use regionprops to efficiently extract areas of all regions
    regions = regionprops(slice_2d)

    if len(regions) == 0:
        return (z, np.zeros(len(bin_edges) - 1, dtype=np.int32), 0)

    # Compute radius for each region
    radii = []
    for region in regions:
        area_voxels = region.area

        # Filter by ellipse area ratio if specified
        if max_ellipse_ratio > 0:
            # Compute ellipse area from fitted axes (in pixels)
            ellipse_area = np.pi * (region.axis_major_length / 2) * (region.axis_minor_length / 2)
            if ellipse_area > max_ellipse_ratio * area_voxels:
                continue  # Skip sparse/fragmented regions

        if use_minor_axis:
            # Use minor axis of fitted ellipse (diameter -> radius)
            # minor_axis_length is in pixels
            radius_um = (region.axis_minor_length / 2) * voxel_size_um
        else:
            # Compute circular-equivalent radius from area
            # Convert to physical area and compute radius
            area_um2 = area_voxels * (voxel_size_um ** 2)
            radius_um = np.sqrt(area_um2 / np.pi)
        radii.append(radius_um)

    # Compute histogram
    radii_array = np.array(radii)
    counts, _ = np.histogram(radii_array, bins=bin_edges)

    return (z, counts.astype(np.int32), len(radii))


def extract_radii_per_slice(volume_file: Path,
                            voxel_size_um: float = 0.2,
                            n_jobs: int = -1,
                            bin_edges: np.ndarray = None,
                            slice_axis: int = 2,
                            subgroup: str = None,
                            load_in_memory: bool = False,
                            use_minor_axis: bool = False,
                            max_ellipse_ratio: float = 0.0) -> Tuple[Dict[int, np.ndarray], Dict[int, int], np.ndarray, np.ndarray]:
    """
    Extract radii histograms for all axons in each slice.

    Supports both Zarr and HDF5 formats.

    Args:
        volume_file: Path to volume file (Zarr or HDF5)
        voxel_size_um: Voxel size in micrometers
        n_jobs: Number of parallel jobs (-1 = use all CPUs, 1 = no parallelization)
        bin_edges: Histogram bin edges in micrometers (default: 0 to 20 with 0.02 step)
        slice_axis: Axis to slice along (0=Z, 1=Y, 2=X), determined by dominant fiber direction
        subgroup: Optional subgroup name (e.g., 'cc' or 'cg')
        load_in_memory: If True, load entire volume into memory first (faster but uses more RAM)
        use_minor_axis: If True, use ellipse minor axis; otherwise use circular-equivalent radius
        max_ellipse_ratio: Max ratio of ellipse area to voxel area (0 = no filter, 2.0 recommended)

    Returns:
        Tuple of (histogram_per_slice, n_axons_per_slice, bin_edges, bin_centers)
    """
    # Detect format
    is_zarr = volume_file.suffix == '.zarr' or (volume_file.is_dir() and (volume_file / 'zarr.json').exists())

    # Get volume shape from file without loading
    if is_zarr:
        import zarr
        root = zarr.open(str(volume_file), mode='r')
        if subgroup:
            volume_shape = root[subgroup]['0'].shape
        else:
            volume_shape = root['0'].shape
    else:
        import h5py
        with h5py.File(volume_file, 'r') as f:
            volume_shape = f['labels'].shape

    logger.info(f"Extracting radii per slice from volume shape {volume_shape}")
    axis_names = {0: 'Z', 1: 'Y', 2: 'X'}
    logger.info(f"Slicing along axis {slice_axis} ({axis_names[slice_axis]})")

    # Number of slices along the specified axis
    n_slices = volume_shape[slice_axis]

    # Default bin edges: 0 to 20 μm with 0.02 μm step
    if bin_edges is None:
        bin_edges = np.arange(0, 20.02, 0.02)

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    logger.info(f"Using {len(bin_edges)-1} histogram bins from {bin_edges[0]:.2f} to {bin_edges[-1]:.2f} μm")
    logger.info(f"Radius estimation: {'minor axis' if use_minor_axis else 'circular-equivalent'}")
    if max_ellipse_ratio > 0:
        logger.info(f"Filtering regions with ellipse/voxel area ratio > {max_ellipse_ratio}")

    # Determine number of workers
    if n_jobs == -1:
        n_workers = cpu_count()
    elif n_jobs == 1:
        n_workers = 1
    else:
        n_workers = min(n_jobs, cpu_count())

    logger.info(f"Using {n_workers} parallel workers")

    # Load volume into memory if requested
    volume_data = None
    if load_in_memory:
        logger.info("Loading volume into memory...")
        if is_zarr:
            import zarr
            root = zarr.open(str(volume_file), mode='r')
            if subgroup:
                volume_data = root[subgroup]['0'][:]
            else:
                volume_data = root['0'][:]
        else:
            import h5py
            with h5py.File(volume_file, 'r') as f:
                volume_data = f['labels'][:]

        mem_gb = volume_data.nbytes / (1024**3)
        logger.info(f"Loaded volume: {volume_data.shape}, {mem_gb:.2f} GB")

    # Prepare arguments for parallel processing (no slice data, just indices)
    args_list = [(z, voxel_size_um, bin_edges, use_minor_axis, max_ellipse_ratio) for z in range(n_slices)]

    # Process in parallel
    if n_workers > 1:
        with Pool(processes=n_workers,
                  initializer=_init_worker,
                  initargs=(volume_file, is_zarr, slice_axis, subgroup, volume_data)) as pool:
            results = list(tqdm(
                pool.imap(_process_single_slice, args_list),
                total=n_slices,
                desc="Processing slices",
                unit="slice"
            ))
    else:
        # Serial processing (useful for debugging)
        # Set globals for serial mode
        global _volume_path, _is_zarr, _slice_axis, _subgroup, _volume_data
        _volume_path = volume_file
        _is_zarr = is_zarr
        _slice_axis = slice_axis
        _subgroup = subgroup
        _volume_data = volume_data
        results = [_process_single_slice(args) for args in tqdm(
            args_list,
            desc="Processing slices",
            unit="slice"
        )]

    # Convert list of tuples to dictionaries
    histogram_per_slice = {z: counts for z, counts, _ in results}
    n_axons_per_slice = {z: n_axons for z, _, n_axons in results}

    return histogram_per_slice, n_axons_per_slice, bin_edges, bin_centers


def compute_effective_radius_from_histogram(counts: np.ndarray, bin_centers: np.ndarray) -> float:
    """
    Compute effective MRI-visible radius from histogram using formula: (<r^6>/<r^2>)^(1/4)

    Args:
        counts: Histogram counts
        bin_centers: Bin center values in micrometers

    Returns:
        Effective radius in micrometers
    """
    if counts.sum() == 0:
        return 0.0

    # Weight by counts
    total_count = counts.sum()

    # Compute weighted moments
    r2_mean = np.sum(counts * bin_centers ** 2) / total_count
    r6_mean = np.sum(counts * bin_centers ** 6) / total_count

    if r2_mean == 0:
        return 0.0

    r_eff = (r6_mean / r2_mean) ** 0.25

    return r_eff


def compute_effective_radius(radii: np.ndarray) -> float:
    """
    Compute effective MRI-visible radius using formula: (<r^6>/<r^2>)^(1/4)

    Args:
        radii: Array of radii in micrometers

    Returns:
        Effective radius in micrometers
    """
    if len(radii) == 0:
        return 0.0

    r2_mean = np.mean(radii ** 2)
    r6_mean = np.mean(radii ** 6)

    if r2_mean == 0:
        return 0.0

    r_eff = (r6_mean / r2_mean) ** 0.25

    return r_eff


def plot_effective_radius_profile(histogram_per_slice: Dict[int, np.ndarray],
                                  n_axons_per_slice: Dict[int, int],
                                  bin_centers: np.ndarray,
                                  output_file: Path,
                                  population_name: str = "Axons",
                                  slice_thickness_um: float = 0.2,
                                  min_axon_fraction: float = 0.0):
    """
    Plot effective radius as a function of position along the tract.

    Args:
        histogram_per_slice: Dict mapping slice_idx -> histogram counts
        n_axons_per_slice: Dict mapping slice_idx -> number of axons
        bin_centers: Bin center values in micrometers
        output_file: Path to save figure
        population_name: Name of the population (CC or CG)
        slice_thickness_um: Thickness of each slice in micrometers
        min_axon_fraction: Minimum fraction of max axon count to include slice (0.0 = use all)
    """
    logger.info(f"Plotting effective radius profile for {population_name}")

    # Compute effective radius per slice
    slice_indices = sorted(histogram_per_slice.keys())
    r_eff_per_slice = []

    for z in slice_indices:
        counts = histogram_per_slice[z]
        r_eff = compute_effective_radius_from_histogram(counts, bin_centers)
        r_eff_per_slice.append(r_eff)

    r_eff_per_slice = np.array(r_eff_per_slice)
    n_axons_array = np.array([n_axons_per_slice[z] for z in slice_indices])

    # Filter slices based on axon count threshold
    if min_axon_fraction > 0.0:
        # Find slice with maximum axon count
        max_idx = np.argmax(n_axons_array)
        max_count = n_axons_array[max_idx]
        threshold_count = min_axon_fraction * max_count

        # Expand symmetrically from max until we hit threshold
        left_idx = max_idx
        right_idx = max_idx

        # Find extent on each side
        left_extent = 0
        for i in range(max_idx, -1, -1):
            if n_axons_array[i] >= threshold_count:
                left_extent = max_idx - i
            else:
                break

        right_extent = 0
        for i in range(max_idx, len(n_axons_array)):
            if n_axons_array[i] >= threshold_count:
                right_extent = i - max_idx
            else:
                break

        # Use symmetric extent (smaller of the two)
        extent = min(left_extent, right_extent)
        start_idx = max_idx - extent
        end_idx = max_idx + extent + 1  # +1 for inclusive

        logger.info(f"Axon count filtering (threshold={min_axon_fraction:.2f}):")
        logger.info(f"  Max axon count: {max_count} at slice {slice_indices[max_idx]}")
        logger.info(f"  Using slices {slice_indices[start_idx]}-{slice_indices[end_idx-1]} "
                   f"({end_idx - start_idx} of {len(slice_indices)} slices)")

        # Filter arrays
        slice_indices = slice_indices[start_idx:end_idx]
        r_eff_per_slice = r_eff_per_slice[start_idx:end_idx]
        n_axons_array = n_axons_array[start_idx:end_idx]

    # Compute global effective radius (pooled over selected slices)
    total_histogram = np.sum([histogram_per_slice[z] for z in slice_indices], axis=0)
    r_eff_global = compute_effective_radius_from_histogram(total_histogram, bin_centers)

    total_axons = n_axons_array.sum()
    logger.info(f"Global effective radius: {r_eff_global:.3f} μm")
    logger.info(f"Mean slice effective radius: {np.mean(r_eff_per_slice[r_eff_per_slice > 0]):.3f} μm")
    logger.info(f"Total radii measurements: {total_axons}")

    # Convert slice indices to position in micrometers
    positions_um = np.array(slice_indices) * slice_thickness_um

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Effective radius per slice
    ax1.plot(positions_um, r_eff_per_slice, 'b-', linewidth=1.5, label='Per-slice effective radius')
    ax1.axhline(r_eff_global, color='r', linestyle='--', linewidth=2,
               label=f'Global effective radius = {r_eff_global:.3f} μm')

    ax1.set_ylabel('Effective Radius (μm)', fontsize=12)
    ax1.set_title(f'{population_name}: MRI-Visible Effective Radius Profile\n' +
                 r'$r_{\rm eff} = \langle r^6 \rangle^{1/4} / \langle r^2 \rangle^{1/4}$',
                 fontsize=14, pad=15)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Number of axons per slice
    ax2.plot(positions_um, n_axons_array, 'g-', linewidth=1.5)
    ax2.set_xlabel('Position along tract (μm)', fontsize=12)
    ax2.set_ylabel('Number of axons', fontsize=12)
    ax2.set_title('Axon count per slice', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved effective radius profile to {output_file}")

    return r_eff_global, r_eff_per_slice


def analyze_population(volume_file: Path,
                       output_dir: Path,
                       voxel_size_um: float = 0.2,
                       n_jobs: int = -1,
                       min_axon_fraction: float = 0.0,
                       load_in_memory: bool = False,
                       use_minor_axis: bool = False,
                       max_ellipse_ratio: float = 0.0) -> Dict[str, Tuple[float, np.ndarray]]:
    """
    Analyze effective radius for a population volume.

    Supports both Zarr and HDF5 formats.

    For Zarr files with cc/cg subgroups (from segment_spatial_subvolumes.py),
    analyzes both populations using their respective dominant axes.

    Args:
        volume_file: Path to aligned volume file (Zarr or HDF5)
        output_dir: Directory for output plots
        voxel_size_um: Voxel size in micrometers (0.05 * downsample_factor)
        n_jobs: Number of parallel jobs (-1 = use all CPUs)
        min_axon_fraction: Minimum fraction of max axon count to include slice (0.0 = use all)
        load_in_memory: If True, load entire volume into memory first (faster but uses more RAM)
        use_minor_axis: If True, use ellipse minor axis; otherwise use circular-equivalent radius
        max_ellipse_ratio: Max ratio of ellipse area to voxel area (0 = no filter, 2.0 recommended)

    Returns:
        Dict mapping population name -> (global_r_eff, r_eff_per_slice)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing: {volume_file.name}")
    logger.info(f"{'='*80}\n")

    # Detect format
    is_zarr = volume_file.suffix == '.zarr' or (volume_file.is_dir() and (volume_file / 'zarr.json').exists())

    results = {}

    if is_zarr:
        import zarr
        root = zarr.open(str(volume_file), mode='r')

        # Check if this is a CC/CG subvolume structure
        has_cc = 'cc' in root
        has_cg = 'cg' in root

        if has_cc or has_cg:
            # Process CC and CG subgroups with their dominant axes
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
                dominant_axis = pop_meta.get('dominant_axis', 2)  # Default to X-axis
                dominant_axis_name = pop_meta.get('dominant_axis_name', 'X')
                n_axons = pop_meta.get('n_axons', 0)

                volume_shape = root[subgroup]['0'].shape
                logger.info(f"Population: {pop_name}")
                logger.info(f"Volume shape: {volume_shape}")
                logger.info(f"Dominant axis: {dominant_axis} ({dominant_axis_name})")
                logger.info(f"Number of axons: {n_axons}")

                # Extract radii histograms per slice using dominant axis
                histogram_per_slice, n_axons_per_slice, bin_edges, bin_centers = extract_radii_per_slice(
                    volume_file, voxel_size_um, n_jobs=n_jobs,
                    slice_axis=dominant_axis, subgroup=subgroup,
                    load_in_memory=load_in_memory,
                    use_minor_axis=use_minor_axis,
                    max_ellipse_ratio=max_ellipse_ratio
                )

                # Plot effective radius profile
                output_file = output_dir / f'{pop_name}_effective_radius_profile.png'
                r_eff_global, r_eff_per_slice = plot_effective_radius_profile(
                    histogram_per_slice,
                    n_axons_per_slice,
                    bin_centers,
                    output_file,
                    pop_name,
                    slice_thickness_um=voxel_size_um,
                    min_axon_fraction=min_axon_fraction
                )

                results[pop_name] = (r_eff_global, r_eff_per_slice)
        else:
            # Single population (legacy format)
            volume_shape = root['0'].shape
            bundle_id = root.attrs.get('bundle_id', None)
            if bundle_id is not None:
                population_name = f"Bundle_{bundle_id:02d}"
            else:
                population_name = volume_file.stem

            # Default to axis 2 for legacy format
            slice_axis = 2

            logger.info(f"Population: {population_name}")
            logger.info(f"Volume shape: {volume_shape}")

            histogram_per_slice, n_axons_per_slice, bin_edges, bin_centers = extract_radii_per_slice(
                volume_file, voxel_size_um, n_jobs=n_jobs, slice_axis=slice_axis,
                load_in_memory=load_in_memory,
                use_minor_axis=use_minor_axis,
                max_ellipse_ratio=max_ellipse_ratio
            )

            output_file = output_dir / f'{population_name}_effective_radius_profile.png'
            r_eff_global, r_eff_per_slice = plot_effective_radius_profile(
                histogram_per_slice,
                n_axons_per_slice,
                bin_centers,
                output_file,
                population_name,
                slice_thickness_um=voxel_size_um,
                min_axon_fraction=min_axon_fraction
            )

            results[population_name] = (r_eff_global, r_eff_per_slice)
    else:
        # HDF5 format
        import h5py
        with h5py.File(volume_file, 'r') as f:
            volume_shape = f['labels'].shape
            dset = f['labels']
            population_name = dset.attrs.get('bundle_id', None)
            if population_name is not None:
                population_name = f"Bundle_{population_name:02d}"
            else:
                population_name = f.attrs.get('population', 'Unknown')

        logger.info(f"Population: {population_name}")
        logger.info(f"Volume shape: {volume_shape}")

        histogram_per_slice, n_axons_per_slice, bin_edges, bin_centers = extract_radii_per_slice(
            volume_file, voxel_size_um, n_jobs=n_jobs, slice_axis=2,
            load_in_memory=load_in_memory,
            use_minor_axis=use_minor_axis,
            max_ellipse_ratio=max_ellipse_ratio
        )

        output_file = output_dir / f'{population_name}_effective_radius_profile.png'
        r_eff_global, r_eff_per_slice = plot_effective_radius_profile(
            histogram_per_slice,
            n_axons_per_slice,
            bin_centers,
            output_file,
            population_name,
            slice_thickness_um=voxel_size_um,
            min_axon_fraction=min_axon_fraction
        )

        results[population_name] = (r_eff_global, r_eff_per_slice)

    logger.info(f"\n{'='*80}")
    logger.info(f"Completed: {volume_file.name}")
    logger.info(f"{'='*80}\n")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze effective MRI-visible radius from aligned axon volumes'
    )
    parser.add_argument('volume_file', type=Path,
                       help='Path to aligned volume file (Zarr or HDF5, e.g., bundle_07_orthogonal.zarr)')
    parser.add_argument('output_dir', type=Path,
                       help='Output directory for plots')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                       help='Voxel size in micrometers (default: 0.05 for full resolution)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (default: -1 = use all CPUs, 1 = serial)')
    parser.add_argument('--min-axon-fraction', type=float, default=0.75,
                       help='Minimum fraction of max axon count to include slice (default: 0.75)')
    parser.add_argument('--load-in-memory', action='store_true',
                       help='Load entire volume into memory first (faster but uses more RAM)')
    parser.add_argument('--use-minor-axis', action='store_true',
                       help='Use ellipse minor axis instead of circular-equivalent radius')
    parser.add_argument('--max-ellipse-ratio', type=float, default=0.0,
                       help='Max ratio of ellipse area to voxel area to filter sparse regions (0 = no filter, 2.0 recommended)')

    args = parser.parse_args()

    analyze_population(
        args.volume_file,
        args.output_dir,
        voxel_size_um=args.voxel_size,
        n_jobs=args.n_jobs,
        min_axon_fraction=args.min_axon_fraction,
        load_in_memory=args.load_in_memory,
        use_minor_axis=args.use_minor_axis,
        max_ellipse_ratio=args.max_ellipse_ratio
    )
