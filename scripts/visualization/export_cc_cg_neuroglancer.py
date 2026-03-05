#!/usr/bin/env python3
"""
Export CC and CG segmentations as Neuroglancer Precomputed format.

Uses dominant-axis classification to identify CC (corpus callosum) and CG (cingulum)
populations, then exports each as a separate Neuroglancer Precomputed volume for
independent visualization.

Output structure:
    output_dir/
    ├── cc/
    │   ├── info              (JSON metadata)
    │   └── <scale_key>/      (e.g., "1_1_1" for full resolution)
    │       └── chunk_files   (e.g., "0-64_0-64_0-64")
    └── cg/
        ├── info
        └── <scale_key>/
            └── chunk_files

Example usage:
    # Single file
    python scripts/visualization/export_cc_cg_neuroglancer.py \\
        data/raw/Sham_25_ipsi/LM_25_ipsi_myelinated_axons.mat \\
        data/processed/neuroglancer

    # Batch processing with glob
    python scripts/visualization/export_cc_cg_neuroglancer.py \\
        "data/raw/**/LM*_myelinated_axons.mat" \\
        data/processed/neuroglancer
"""

import argparse
import json
import logging
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from axonometry.populations import (
    load_volume_downsampled,
    precompute_axon_voxels,
    compute_all_orientations,
    classify_by_dominant_axis,
    create_populations,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_info_json(
    volume_shape: Tuple[int, int, int],
    voxel_size_nm: Tuple[float, float, float],
    chunk_size: Tuple[int, int, int] = (64, 64, 64),
    num_scales: int = 5,
) -> Dict:
    """
    Create Neuroglancer Precomputed info JSON for segmentation data.

    Args:
        volume_shape: Shape of the volume (z, y, x)
        voxel_size_nm: Voxel size in nanometers (x, y, z)
        chunk_size: Chunk size for each dimension (x, y, z)
        num_scales: Number of downsampling levels

    Returns:
        Dictionary containing info JSON structure
    """
    # Convert to Neuroglancer coordinate order (x, y, z)
    size = [int(volume_shape[2]), int(volume_shape[1]), int(volume_shape[0])]
    resolution = [float(voxel_size_nm[0]), float(voxel_size_nm[1]), float(voxel_size_nm[2])]

    scales = []
    current_size = size.copy()
    current_resolution = resolution.copy()

    for level in range(num_scales):
        # Generate scale key from resolution
        scale_key = f"{int(current_resolution[0])}_{int(current_resolution[1])}_{int(current_resolution[2])}"

        # Adjust chunk size if volume is smaller
        adjusted_chunk = [
            min(chunk_size[i], current_size[i]) for i in range(3)
        ]

        scales.append({
            "key": scale_key,
            "size": current_size.copy(),
            "resolution": current_resolution.copy(),
            "chunk_sizes": [adjusted_chunk],
            "encoding": "raw",
        })

        # Downsample for next level
        current_size = [max(1, s // 2) for s in current_size]
        current_resolution = [r * 2 for r in current_resolution]

        # Stop if volume becomes too small
        if all(s <= 1 for s in current_size):
            break

    info = {
        "@type": "neuroglancer_multiscale_volume",
        "type": "segmentation",
        "data_type": "uint32",
        "num_channels": 1,
        "scales": scales,
        "mesh": "mesh",  # Enable mesh loading from mesh/ subdirectory
    }

    return info


def create_label_lut(axon_labels: Set[int], max_label: int) -> np.ndarray:
    """Create lookup table for fast label filtering."""
    lut = np.zeros(max_label + 1, dtype=bool)
    for label in axon_labels:
        if label <= max_label:
            lut[label] = True
    return lut


def filter_volume_with_lut(data: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Filter array using lookup table, keeping only specified labels."""
    clamped = np.minimum(data, len(lut) - 1)
    mask = lut[clamped]
    return np.where(mask, data, 0).astype(np.uint32)


def _write_chunk(args):
    """Write a single chunk to disk (for parallel execution)."""
    chunk_path, chunk_data = args
    with open(chunk_path, 'wb') as f:
        f.write(chunk_data)


def generate_mesh_for_segment(
    volume: np.ndarray,
    segment_id: int,
    voxel_size_nm: Tuple[float, float, float],
    step_size: int = 1,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate mesh for a single segment using marching cubes.

    Args:
        volume: 3D labeled volume (z, y, x order)
        segment_id: Label ID to mesh
        voxel_size_nm: Voxel size in nanometers (x, y, z)
        step_size: Step size for marching cubes (higher = faster but coarser)

    Returns:
        Tuple of (vertices, faces) or None if segment not found
    """
    from skimage import measure

    # Create binary mask for this segment
    mask = (volume == segment_id)
    if not mask.any():
        return None

    try:
        # Marching cubes expects (z, y, x) order, returns vertices in same order
        verts, faces, _, _ = measure.marching_cubes(
            mask.astype(np.float32),
            level=0.5,
            step_size=step_size,
            allow_degenerate=False,
        )

        # Convert vertices from voxel to nanometer coordinates
        # verts are in (z, y, x) order, need to convert to (x, y, z) for Neuroglancer
        verts_nm = np.zeros_like(verts, dtype=np.float32)
        verts_nm[:, 0] = verts[:, 2] * voxel_size_nm[0]  # x
        verts_nm[:, 1] = verts[:, 1] * voxel_size_nm[1]  # y
        verts_nm[:, 2] = verts[:, 0] * voxel_size_nm[2]  # z

        return verts_nm, faces.astype(np.uint32)
    except Exception as e:
        logger.warning(f"Marching cubes failed for segment {segment_id}: {e}")
        return None


def write_legacy_mesh(mesh_dir: Path, segment_id: int, vertices: np.ndarray, faces: np.ndarray):
    """
    Write mesh in Neuroglancer legacy format.

    Format:
    - Metadata file: <segment_id>:0 (JSON with fragment list)
    - Fragment file: <segment_id>:0:0 (binary: num_verts + vertices + faces)
    """
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Write fragment file
    fragment_name = f"{segment_id}:0:0"
    fragment_path = mesh_dir / fragment_name

    num_vertices = np.uint32(len(vertices))
    with open(fragment_path, 'wb') as f:
        f.write(num_vertices.tobytes())
        f.write(vertices.astype('<f4').tobytes())  # little-endian float32
        f.write(faces.astype('<u4').tobytes())     # little-endian uint32

    # Write metadata file
    metadata_name = f"{segment_id}:0"
    metadata_path = mesh_dir / metadata_name
    with open(metadata_path, 'w') as f:
        json.dump({"fragments": [fragment_name]}, f)


def generate_meshes_from_volume(
    volume: np.ndarray,
    segment_ids: List[int],
    mesh_dir: Path,
    voxel_size_nm: Tuple[float, float, float],
    step_size: int = 2,
    n_workers: int = 8,
) -> int:
    """
    Generate meshes for all segments in a volume.

    Args:
        volume: 3D labeled volume
        segment_ids: List of segment IDs to mesh
        mesh_dir: Output directory for mesh files
        voxel_size_nm: Voxel size in nanometers
        step_size: Marching cubes step size
        n_workers: Number of parallel workers

    Returns:
        Number of meshes generated
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Write mesh info file
    mesh_info = {"@type": "neuroglancer_legacy_mesh"}
    with open(mesh_dir / "info", 'w') as f:
        json.dump(mesh_info, f)

    count = 0

    def process_segment(seg_id):
        result = generate_mesh_for_segment(volume, seg_id, voxel_size_nm, step_size)
        if result is not None:
            verts, faces = result
            write_legacy_mesh(mesh_dir, seg_id, verts, faces)
            return True
        return False

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_segment, sid): sid for sid in segment_ids}
        for future in tqdm(as_completed(futures), total=len(segment_ids), desc="Generating meshes", leave=False):
            if future.result():
                count += 1

    return count


def write_precomputed_chunks_streaming(
    mat_file: Path,
    output_dir: Path,
    info: Dict,
    lut: np.ndarray,
    population_name: str,
    n_workers: int = 8,
) -> None:
    """
    Write volume as Neuroglancer Precomputed chunks using optimized slab-based approach.

    Reads entire z-slabs from HDF5 (much faster than individual chunks),
    then processes and writes chunks in parallel.

    Args:
        mat_file: Source HDF5 .mat file
        output_dir: Output directory for this population
        info: Info JSON dictionary
        lut: Label lookup table for filtering
        population_name: Name for progress display
        n_workers: Number of parallel write workers
    """
    from concurrent.futures import ThreadPoolExecutor

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write info file
    info_path = output_dir / "info"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    logger.info(f"Wrote {info_path}")

    # Get full resolution scale info
    scale = info["scales"][0]
    scale_key = scale["key"]
    scale_dir = output_dir / scale_key
    scale_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = scale["chunk_sizes"][0]
    vol_size = scale["size"]  # (x, y, z) in Neuroglancer order

    # Calculate number of chunks in each dimension
    n_chunks = [
        (vol_size[i] + chunk_size[i] - 1) // chunk_size[i]
        for i in range(3)
    ]

    total_chunks = n_chunks[0] * n_chunks[1] * n_chunks[2]
    desc = f"{population_name} scale 0"

    # Read and process by z-slabs (one z-chunk height at a time)
    # This dramatically reduces HDF5 read calls
    with h5py.File(mat_file, 'r') as f:
        dataset = f['final_lbl']

        with tqdm(total=total_chunks, desc=desc, unit="chunk", leave=False) as pbar:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                for cz in range(n_chunks[2]):
                    z_start = cz * chunk_size[2]
                    z_end = min(z_start + chunk_size[2], vol_size[2])

                    # Read entire z-slab at once (all x, all y, one z-chunk)
                    # HDF5 shape is (z, y, x), so we read [z_start:z_end, :, :]
                    z_slab = dataset[z_start:z_end, :, :]

                    # Filter using LUT (vectorized, fast)
                    z_slab = filter_volume_with_lut(z_slab, lut)

                    # Prepare all chunks from this z-slab
                    write_tasks = []
                    for cy in range(n_chunks[1]):
                        y_start = cy * chunk_size[1]
                        y_end = min(y_start + chunk_size[1], vol_size[1])

                        for cx in range(n_chunks[0]):
                            x_start = cx * chunk_size[0]
                            x_end = min(x_start + chunk_size[0], vol_size[0])

                            # Extract chunk from z-slab (already filtered)
                            # z_slab is (z_chunk_size, full_y, full_x)
                            chunk_data = z_slab[:, y_start:y_end, x_start:x_end]

                            # Transpose to (x, y, z) and convert to bytes in Fortran order
                            chunk_data = chunk_data.transpose(2, 1, 0).astype(np.uint32)
                            chunk_bytes = chunk_data.tobytes(order='F')

                            # Queue for writing
                            chunk_filename = f"{x_start}-{x_end}_{y_start}-{y_end}_{z_start}-{z_end}"
                            chunk_path = scale_dir / chunk_filename
                            write_tasks.append((chunk_path, chunk_bytes))

                    # Write all chunks from this z-slab in parallel
                    list(executor.map(_write_chunk, write_tasks))
                    pbar.update(len(write_tasks))

    logger.info(f"Wrote {population_name} level 0 chunks to {output_dir}")

    # Generate pyramid levels from level 0 chunks
    _generate_pyramid_from_chunks(output_dir, info, population_name)


def _process_pyramid_chunk(args):
    """Process a single pyramid chunk (for parallel execution)."""
    (prev_dir, prev_chunk_size, curr_dir,
     x_start, x_end, y_start, y_end, z_start, z_end) = args

    # Source region in previous level (2x larger)
    src_x_start, src_x_end = x_start * 2, x_end * 2
    src_y_start, src_y_end = y_start * 2, y_end * 2
    src_z_start, src_z_end = z_start * 2, z_end * 2

    # Read source chunks from previous level
    src_data = _read_chunk_region(
        prev_dir, prev_chunk_size,
        src_x_start, src_x_end,
        src_y_start, src_y_end,
        src_z_start, src_z_end,
    )

    # Downsample (take every other voxel)
    chunk_data = src_data[::2, ::2, ::2].copy()

    # Ensure correct shape
    expected_shape = (x_end - x_start, y_end - y_start, z_end - z_start)
    chunk_data = chunk_data[:expected_shape[0], :expected_shape[1], :expected_shape[2]]

    # Write chunk in Fortran order
    chunk_filename = f"{x_start}-{x_end}_{y_start}-{y_end}_{z_start}-{z_end}"
    chunk_path = curr_dir / chunk_filename
    with open(chunk_path, 'wb') as f:
        f.write(chunk_data.tobytes(order='F'))


def _generate_pyramid_from_chunks(
    output_dir: Path,
    info: Dict,
    population_name: str,
    n_workers: int = 8,
) -> None:
    """
    Generate pyramid levels by reading and downsampling from previous level chunks.

    Uses parallel processing for faster generation.
    """
    from concurrent.futures import ThreadPoolExecutor

    for scale_idx in range(1, len(info["scales"])):
        prev_scale = info["scales"][scale_idx - 1]
        curr_scale = info["scales"][scale_idx]

        prev_dir = output_dir / prev_scale["key"]
        curr_dir = output_dir / curr_scale["key"]
        curr_dir.mkdir(parents=True, exist_ok=True)

        prev_chunk_size = prev_scale["chunk_sizes"][0]
        curr_chunk_size = curr_scale["chunk_sizes"][0]
        curr_vol_size = curr_scale["size"]  # (x, y, z)

        # Calculate number of chunks at current level
        n_chunks = [
            (curr_vol_size[i] + curr_chunk_size[i] - 1) // curr_chunk_size[i]
            for i in range(3)
        ]

        total_chunks = n_chunks[0] * n_chunks[1] * n_chunks[2]
        desc = f"{population_name} scale {scale_idx}"

        # Build task list
        tasks = []
        for cz in range(n_chunks[2]):
            z_start = cz * curr_chunk_size[2]
            z_end = min(z_start + curr_chunk_size[2], curr_vol_size[2])

            for cy in range(n_chunks[1]):
                y_start = cy * curr_chunk_size[1]
                y_end = min(y_start + curr_chunk_size[1], curr_vol_size[1])

                for cx in range(n_chunks[0]):
                    x_start = cx * curr_chunk_size[0]
                    x_end = min(x_start + curr_chunk_size[0], curr_vol_size[0])

                    tasks.append((
                        prev_dir, prev_chunk_size, curr_dir,
                        x_start, x_end, y_start, y_end, z_start, z_end
                    ))

        # Process in parallel with progress bar
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            list(tqdm(
                executor.map(_process_pyramid_chunk, tasks),
                total=total_chunks,
                desc=desc,
                unit="chunk",
                leave=False
            ))

    logger.info(f"Generated pyramid levels for {population_name}")


def _read_chunk_region(
    chunk_dir: Path,
    chunk_size: List[int],
    x_start: int, x_end: int,
    y_start: int, y_end: int,
    z_start: int, z_end: int,
) -> np.ndarray:
    """
    Read a region from chunked storage, assembling from multiple chunk files.

    Returns data in (x, y, z) order (Neuroglancer format).
    """
    result = np.zeros((x_end - x_start, y_end - y_start, z_end - z_start), dtype=np.uint32)

    # Find which chunks overlap with this region
    cx_start = x_start // chunk_size[0]
    cx_end = (x_end - 1) // chunk_size[0] + 1
    cy_start = y_start // chunk_size[1]
    cy_end = (y_end - 1) // chunk_size[1] + 1
    cz_start = z_start // chunk_size[2]
    cz_end = (z_end - 1) // chunk_size[2] + 1

    for cz in range(cz_start, cz_end):
        for cy in range(cy_start, cy_end):
            for cx in range(cx_start, cx_end):
                # Chunk boundaries
                chunk_x_start = cx * chunk_size[0]
                chunk_y_start = cy * chunk_size[1]
                chunk_z_start = cz * chunk_size[2]

                # Find chunk file (need to search for actual bounds)
                chunk_files = list(chunk_dir.glob(f"{chunk_x_start}-*_{chunk_y_start}-*_{chunk_z_start}-*"))
                if not chunk_files:
                    continue

                chunk_path = chunk_files[0]
                # Parse actual chunk bounds from filename
                parts = chunk_path.stem.split('_')
                cx_bounds = [int(x) for x in parts[0].split('-')]
                cy_bounds = [int(x) for x in parts[1].split('-')]
                cz_bounds = [int(x) for x in parts[2].split('-')]

                chunk_x_end = cx_bounds[1]
                chunk_y_end = cy_bounds[1]
                chunk_z_end = cz_bounds[1]

                # Read chunk data (stored in Fortran order)
                chunk_shape = (chunk_x_end - chunk_x_start,
                              chunk_y_end - chunk_y_start,
                              chunk_z_end - chunk_z_start)
                chunk_data = np.frombuffer(chunk_path.read_bytes(), dtype=np.uint32).reshape(chunk_shape, order='F')

                # Calculate overlap region
                overlap_x_start = max(x_start, chunk_x_start)
                overlap_x_end = min(x_end, chunk_x_end)
                overlap_y_start = max(y_start, chunk_y_start)
                overlap_y_end = min(y_end, chunk_y_end)
                overlap_z_start = max(z_start, chunk_z_start)
                overlap_z_end = min(z_end, chunk_z_end)

                # Copy overlap to result
                result[
                    overlap_x_start - x_start : overlap_x_end - x_start,
                    overlap_y_start - y_start : overlap_y_end - y_start,
                    overlap_z_start - z_start : overlap_z_end - z_start,
                ] = chunk_data[
                    overlap_x_start - chunk_x_start : overlap_x_end - chunk_x_start,
                    overlap_y_start - chunk_y_start : overlap_y_end - chunk_y_start,
                    overlap_z_start - chunk_z_start : overlap_z_end - chunk_z_start,
                ]

    return result


def export_populations_to_neuroglancer(
    mat_file: Path,
    output_dir: Path,
    voxel_size_um: float = 0.05,
    downsample: int = 4,
    max_angle_deg: float = 30.0,
    min_length_um: float = 50.0,
    min_voxels: int = 10,
    k_neighbors: int = 10,
    max_neighbor_distance_um: float = 30.0,
    chunk_size: int = 64,
    num_scales: int = 5,
    generate_meshes: bool = False,
    mesh_downsample: int = 8,
) -> Dict:
    """
    Export CC and CG populations to Neuroglancer Precomputed format.

    Args:
        mat_file: Input .mat file with labeled volume
        output_dir: Output directory for Neuroglancer volumes
        voxel_size_um: Voxel size in micrometers
        downsample: Downsampling factor for population identification
        max_angle_deg: Maximum angle from dominant axis for classification
        min_length_um: Minimum axon length for inclusion
        min_voxels: Minimum voxels per axon
        k_neighbors: K for KNN sparse filtering
        max_neighbor_distance_um: Max distance for KNN
        chunk_size: Chunk size for Neuroglancer
        generate_meshes: Whether to generate 3D meshes for visualization
        mesh_downsample: Downsampling factor for mesh generation (higher = faster)
        num_scales: Number of pyramid levels

    Returns:
        Dictionary with export metadata
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Exporting CC/CG to Neuroglancer: {mat_file.name}")
    logger.info(f"{'='*80}\n")

    # Phase 1: Identify populations (on downsampled volume)
    logger.info("Phase 1: Identifying populations...")
    volume_ds, vol_metadata = load_volume_downsampled(mat_file, downsample)
    voxel_size_ds = voxel_size_um * downsample

    axon_voxels_ds = precompute_axon_voxels(volume_ds)
    axon_data = compute_all_orientations(
        axon_voxels_ds, voxel_size_ds, min_voxels, min_length_um
    )

    if not axon_data:
        raise ValueError("No valid axons found after filtering")

    label_to_bundle, axis_to_bundle = classify_by_dominant_axis(axon_data, max_angle_deg)

    if not label_to_bundle:
        raise ValueError("No axons remain after classification")

    populations = create_populations(
        axon_data, label_to_bundle, axis_to_bundle, axon_voxels_ds, voxel_size_ds,
        k_neighbors, max_neighbor_distance_um
    )

    if len(populations) < 2:
        logger.warning(f"Only {len(populations)} population(s) found, expected 2")

    # Get max label from axon_data keys (no need to load full volume)
    max_label = max(axon_data.keys())

    # Free downsampled data
    del volume_ds, axon_voxels_ds

    # Phase 2: Get volume metadata (without loading full volume)
    logger.info("\nPhase 2: Reading volume metadata...")
    with h5py.File(mat_file, 'r') as f:
        volume_shape = f['final_lbl'].shape
    logger.info(f"Full volume shape: {volume_shape}, max label: {max_label}")

    # Convert voxel size to nanometers for Neuroglancer
    voxel_size_nm = (voxel_size_um * 1000,) * 3  # isotropic

    # Create info JSON template
    info_template = create_info_json(
        volume_shape,
        voxel_size_nm,
        chunk_size=(chunk_size, chunk_size, chunk_size),
        num_scales=num_scales,
    )

    # Extract sample name for output directory naming
    sample_name = mat_file.stem.replace('_myelinated_axons', '')
    sample_output_dir = output_dir / sample_name

    # Phase 3: Export populations (streaming, memory-efficient)
    logger.info("\nPhase 3: Exporting populations (streaming)...")
    export_metadata = {
        'source_file': str(mat_file),
        'volume_shape': list(volume_shape),
        'voxel_size_um': voxel_size_um,
        'populations': []
    }

    population_names = ['cc', 'cg']
    for i, (pop, name) in enumerate(zip(populations[:2], population_names)):
        logger.info(f"\nExporting {name.upper()}: {pop['n_axons']} axons")

        pop_output_dir = sample_output_dir / name
        lut = create_label_lut(set(pop['axon_labels']), max_label)

        write_precomputed_chunks_streaming(
            mat_file,
            pop_output_dir,
            info_template.copy(),
            lut,
            name.upper(),
        )

        # Generate meshes if requested
        n_meshes = 0
        if generate_meshes:
            logger.info(f"  Generating meshes for {name.upper()} ({mesh_downsample}x downsampled)...")
            mesh_dir = pop_output_dir / "mesh"

            # Load downsampled volume for mesh generation
            with h5py.File(mat_file, 'r') as f:
                volume_ds = f['final_lbl'][::mesh_downsample, ::mesh_downsample, ::mesh_downsample]

            # Filter to this population
            volume_filtered = filter_volume_with_lut(volume_ds, lut)
            del volume_ds

            # Get segment IDs in this population
            segment_ids = list(pop['axon_labels'])

            # Mesh voxel size (scaled by downsample factor)
            mesh_voxel_nm = (voxel_size_um * 1000 * mesh_downsample,) * 3

            n_meshes = generate_meshes_from_volume(
                volume_filtered,
                segment_ids,
                mesh_dir,
                mesh_voxel_nm,
                step_size=1,
                n_workers=8,
            )
            del volume_filtered
            logger.info(f"  Generated {n_meshes} meshes")

        export_metadata['populations'].append({
            'name': name,
            'n_axons': pop['n_axons'],
            'mean_orientation': pop['mean_orientation'],
            'mean_length_um': pop['mean_length_um'],
            'output_path': str(pop_output_dir),
            'n_meshes': n_meshes,
        })

    # Save export metadata
    metadata_path = sample_output_dir / 'export_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(export_metadata, f, indent=2)
    logger.info(f"\nSaved metadata: {metadata_path}")

    # Summary
    total_size = sum(
        f.stat().st_size
        for f in sample_output_dir.rglob('*')
        if f.is_file()
    )
    logger.info(f"Total output size: {total_size / 1024**2:.1f} MB")

    logger.info(f"\n{'='*80}")
    logger.info("Export complete!")
    logger.info(f"{'='*80}\n")

    return export_metadata


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Export CC and CG segmentations as Neuroglancer Precomputed format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single file
    python scripts/visualization/export_cc_cg_neuroglancer.py \\
        data/raw/Sham_25_ipsi/LM_25_ipsi_myelinated_axons.mat \\
        data/processed/neuroglancer

    # Batch processing with glob
    python scripts/visualization/export_cc_cg_neuroglancer.py \\
        "data/raw/**/LM*_myelinated_axons.mat" \\
        data/processed/neuroglancer
        """
    )
    parser.add_argument(
        'input_files',
        type=str,
        help='Input .mat file(s) with labeled volumes (glob pattern supported)'
    )
    parser.add_argument(
        'output_dir',
        type=Path,
        help='Output directory for Neuroglancer volumes'
    )
    parser.add_argument(
        '--voxel-size', type=float, default=0.05,
        help='Voxel size in μm (default: 0.05)'
    )
    parser.add_argument(
        '--downsample', type=int, default=4,
        help='Downsampling factor for population identification (default: 4)'
    )
    parser.add_argument(
        '--max-angle', type=float, default=30.0,
        help='Max deviation from dominant axis in degrees (default: 30.0)'
    )
    parser.add_argument(
        '--min-length', type=float, default=50.0,
        help='Minimum axon length in μm (default: 50.0)'
    )
    parser.add_argument(
        '--min-voxels', type=int, default=10,
        help='Minimum voxels per axon for identification (default: 10)'
    )
    parser.add_argument(
        '--k-neighbors', type=int, default=10,
        help='K for sparse axon filtering (default: 10)'
    )
    parser.add_argument(
        '--max-neighbor-distance', type=float, default=30.0,
        help='Max distance to k-th neighbor in μm (default: 30.0)'
    )
    parser.add_argument(
        '--chunk-size', type=int, default=64,
        help='Chunk size for Neuroglancer (default: 64)'
    )
    parser.add_argument(
        '--num-scales', type=int, default=5,
        help='Number of pyramid levels (default: 5)'
    )
    parser.add_argument(
        '--generate-meshes', action='store_true',
        help='Generate 3D meshes for visualization (slower but enables 3D rendering)'
    )
    parser.add_argument(
        '--mesh-downsample', type=int, default=8,
        help='Downsampling factor for mesh generation (default: 8, higher = faster)'
    )

    args = parser.parse_args()

    # Expand glob pattern
    input_path = args.input_files
    if '*' in input_path:
        input_files = sorted(glob(input_path, recursive=True))
    else:
        input_files = [input_path]

    if not input_files:
        logger.error(f"No files found matching: {args.input_files}")
        return

    logger.info(f"Found {len(input_files)} input file(s)")

    # Process each file
    for mat_file in input_files:
        mat_path = Path(mat_file)
        if not mat_path.exists():
            logger.warning(f"File not found: {mat_file}")
            continue

        try:
            export_populations_to_neuroglancer(
                mat_path,
                args.output_dir,
                voxel_size_um=args.voxel_size,
                downsample=args.downsample,
                max_angle_deg=args.max_angle,
                min_length_um=args.min_length,
                min_voxels=args.min_voxels,
                k_neighbors=args.k_neighbors,
                max_neighbor_distance_um=args.max_neighbor_distance,
                chunk_size=args.chunk_size,
                num_scales=args.num_scales,
                generate_meshes=args.generate_meshes,
                mesh_downsample=args.mesh_downsample,
            )
        except Exception as e:
            logger.error(f"Failed to process {mat_file}: {e}")
            raise


if __name__ == '__main__':
    main()
