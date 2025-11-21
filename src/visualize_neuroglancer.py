#!/usr/bin/env python3
"""
Visualize bundle volumes in Neuroglancer.

Supports both HDF5 and OME-Zarr formats with multi-resolution pyramids.
For Zarr, serves via HTTP for proper multi-resolution support.
"""

import argparse
import logging
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import neuroglancer
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CORSRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler with CORS support for Neuroglancer."""

    def __init__(self, *args, directory=None, **kwargs):
        self.directory = directory
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress HTTP access logs
        pass


def start_http_server(directory: Path, port: int = 8000) -> tuple:
    """
    Start an HTTP server to serve Zarr files.

    Args:
        directory: Directory to serve
        port: Port to use (will try successive ports if busy)

    Returns:
        Tuple of (server, port, thread)
    """
    handler = partial(CORSRequestHandler, directory=str(directory))

    # Try to find an available port
    for attempt in range(10):
        try:
            server = HTTPServer(('localhost', port + attempt), handler)
            actual_port = port + attempt
            break
        except OSError:
            continue
    else:
        raise RuntimeError(f"Could not find available port starting from {port}")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    logger.info(f"HTTP server started on port {actual_port}")
    return server, actual_port, thread


def get_zarr_metadata(zarr_path: Path) -> dict:
    """
    Get metadata from Zarr store without loading volume data.

    Args:
        zarr_path: Path to Zarr directory

    Returns:
        Metadata dictionary
    """
    import zarr

    root = zarr.open(str(zarr_path), mode='r')

    attrs = dict(root.attrs)
    metadata = {
        'bundle_id': attrs.get('bundle_id'),
        'n_axons': attrs.get('n_axons'),
        'voxel_size_um': attrs.get('voxel_size_um', 0.05),
        'mean_orientation': attrs.get('mean_orientation'),
        'n_levels': attrs.get('n_levels', 1),
    }

    # Get shape from level 0
    metadata['shape'] = root['0'].shape

    # Get segment IDs if available
    if 'labels' in root and 'segment_ids' in root['labels']:
        segment_ids = root['labels']['segment_ids'][:]
        metadata['segment_ids'] = segment_ids.tolist()
        logger.info(f"Found {len(segment_ids)} segment IDs")
    else:
        metadata['segment_ids'] = None

    return metadata


def load_zarr_volume(zarr_path: Path, level: int = None) -> tuple:
    """
    Load volume from OME-Zarr store (for single-level fallback).

    Args:
        zarr_path: Path to Zarr directory
        level: Pyramid level to load (None = auto-select based on size)

    Returns:
        Tuple of (volume, metadata_dict, voxel_size_um)
    """
    import zarr

    logger.info(f"Loading Zarr: {zarr_path}")

    root = zarr.open(str(zarr_path), mode='r')

    # Get metadata from .zattrs
    attrs = dict(root.attrs)
    metadata = {
        'bundle_id': attrs.get('bundle_id'),
        'n_axons': attrs.get('n_axons'),
        'voxel_size_um': attrs.get('voxel_size_um', 0.05),
        'mean_orientation': attrs.get('mean_orientation'),
        'n_levels': attrs.get('n_levels', 1),
    }

    n_levels = metadata['n_levels']

    # Auto-select level if not specified
    if level is None:
        # Check level 0 size
        level_0_shape = root['0'].shape
        level_0_size_gb = np.prod(level_0_shape) * 4 / 1024**3  # uint32

        if level_0_size_gb > 10:
            # Use lowest resolution for large volumes
            level = n_levels - 1
            logger.info(f"Auto-selecting level {level} (full res is {level_0_size_gb:.1f} GB)")
        else:
            level = 0

    # Load the selected level
    level_key = str(level)
    if level_key not in root:
        logger.warning(f"Level {level} not found, using level 0")
        level = 0
        level_key = '0'

    volume = root[level_key][:]

    # Adjust voxel size for this level
    voxel_size_um = metadata['voxel_size_um'] * (2 ** level)

    logger.info(f"Loaded level {level}: shape {volume.shape}, voxel size {voxel_size_um} um")
    logger.info(f"Memory: {volume.nbytes / 1024**3:.2f} GB")

    # Get segment IDs if available
    if 'labels' in root and 'segment_ids' in root['labels']:
        segment_ids = root['labels']['segment_ids'][:]
        metadata['segment_ids'] = segment_ids.tolist()
        logger.info(f"Found {len(segment_ids)} segment IDs")
    else:
        metadata['segment_ids'] = None

    return volume, metadata, voxel_size_um


def load_hdf5_volume(volume_file: Path, downsample: int = 1) -> tuple:
    """
    Load volume from HDF5 file (legacy format).

    Args:
        volume_file: Path to HDF5 file
        downsample: Downsampling factor

    Returns:
        Tuple of (volume, metadata_dict, voxel_size_um)
    """
    import h5py

    logger.info(f"Loading HDF5: {volume_file}")

    with h5py.File(volume_file, 'r') as f:
        dset = f['labels']
        metadata = {
            'bundle_id': dset.attrs.get('bundle_id', None),
            'n_axons': dset.attrs.get('n_axons', None),
            'voxel_size_um': dset.attrs.get('voxel_size_um', 0.05),
            'mean_orientation': dset.attrs.get('mean_orientation', None),
            'segment_ids': None,
        }

        voxel_size_um = metadata['voxel_size_um']

        if downsample > 1:
            logger.info(f"Downsampling by {downsample}x")
            volume = dset[::downsample, ::downsample, ::downsample]
            voxel_size_um = voxel_size_um * downsample
        else:
            volume = dset[:]

    logger.info(f"Volume shape: {volume.shape}")
    logger.info(f"Memory: {volume.nbytes / 1024**3:.2f} GB")

    return volume, metadata, voxel_size_um


def visualize_volumes(volume_paths: list, bind_address: str = 'localhost', port: int = 9999,
                      level: int = None, downsample: int = 1, multires: bool = True,
                      grayscale_file: Path = None):
    """
    Visualize one or more volumes in Neuroglancer.

    Args:
        volume_paths: List of paths to volume files (Zarr or HDF5)
        bind_address: Address to bind the viewer to
        port: Port number (0 = auto-select)
        level: Pyramid level for Zarr files (None = auto)
        downsample: Downsampling factor for HDF5 files
        multires: Use multi-resolution HTTP serving for Zarr (default True)
        grayscale_file: Optional path to grayscale HDF5 file with 'raw' dataset
    """
    # Set up Neuroglancer
    neuroglancer.set_server_bind_address(bind_address, port)
    viewer = neuroglancer.Viewer()

    http_servers = []  # Track HTTP servers for cleanup

    # Load and add each volume
    for volume_path in volume_paths:
        # Detect format (Zarr v2 uses .zattrs/.zgroup, v3 uses zarr.json)
        is_zarr = (volume_path.suffix == '.zarr' or
                   (volume_path.is_dir() and
                    ((volume_path / '.zattrs').exists() or
                     (volume_path / '.zgroup').exists() or
                     (volume_path / 'zarr.json').exists())))

        if is_zarr and multires:
            # Use HTTP serving for multi-resolution Zarr
            metadata = get_zarr_metadata(volume_path)

            # Determine layer name
            if metadata['bundle_id'] is not None:
                layer_name = f"Bundle_{metadata['bundle_id']:02d}"
            else:
                layer_name = volume_path.stem

            # Start HTTP server for the parent directory
            parent_dir = volume_path.parent
            zarr_name = volume_path.name

            # Find or start HTTP server for this directory
            server, http_port, thread = start_http_server(parent_dir, port=8000)
            http_servers.append(server)

            # Build Zarr URL for Neuroglancer
            zarr_url = f"zarr://http://localhost:{http_port}/{zarr_name}"

            logger.info(f"Serving Zarr at: {zarr_url}")
            logger.info(f"Test HTTP access: curl http://localhost:{http_port}/{zarr_name}/.zattrs")

            # Verify HTTP server is accessible
            import urllib.request
            try:
                test_url = f"http://localhost:{http_port}/{zarr_name}/.zattrs"
                with urllib.request.urlopen(test_url, timeout=2) as response:
                    if response.status == 200:
                        logger.info("HTTP server verified - .zattrs accessible")
            except Exception as e:
                logger.error(f"HTTP server test failed: {e}")

            # Get voxel size - Zarr metadata uses micrometers
            voxel_size_um = metadata['voxel_size_um']

            # Get segment IDs - only pass if we have them, otherwise let Neuroglancer show all
            segment_ids = metadata.get('segment_ids')

            if segment_ids:
                logger.info(f"Using {len(segment_ids)} stored segment IDs")
            else:
                logger.info("No stored segment IDs - Neuroglancer will show all segments")

            with viewer.txn() as s:
                # Add as segmentation layer with Zarr source
                layer_kwargs = {'source': zarr_url}
                if segment_ids:
                    layer_kwargs['segments'] = segment_ids
                s.layers[layer_name] = neuroglancer.SegmentationLayer(**layer_kwargs)

                # Set initial position to center of volume
                # Position order matches OME-NGFF axes order: [y, x, z]
                # Shape is also [y, x, z]
                shape = metadata['shape']
                center = [shape[0] // 2 * voxel_size_um,  # y
                          shape[1] // 2 * voxel_size_um,  # x
                          shape[2] // 2 * voxel_size_um]  # z
                s.position = center

            logger.info(f"Added multi-resolution layer: {layer_name}")
            logger.info(f"  Shape: {metadata['shape']}")
            logger.info(f"  Levels: {metadata['n_levels']}")
            logger.info(f"  Center position (nm): {center}")
            if metadata['n_axons']:
                logger.info(f"  Expected axons: {metadata['n_axons']}")

        elif is_zarr:
            # Fallback: load single level into memory
            volume, metadata, voxel_size_um = load_zarr_volume(volume_path, level)

            # Determine layer name
            if metadata['bundle_id'] is not None:
                layer_name = f"Bundle_{metadata['bundle_id']:02d}"
            else:
                layer_name = volume_path.stem

            voxel_size_nm = voxel_size_um * 1000

            segment_ids = metadata['segment_ids'] if metadata['segment_ids'] else []
            if not segment_ids:
                logger.info("Finding unique segment IDs...")
                unique_ids = np.unique(volume)
                segment_ids = unique_ids[unique_ids != 0].tolist()

            logger.info(f"Found {len(segment_ids)} unique segments")

            with viewer.txn() as s:
                s.layers[layer_name] = neuroglancer.SegmentationLayer(
                    source=neuroglancer.LocalVolume(
                        data=volume,
                        dimensions=neuroglancer.CoordinateSpace(
                            names=['y', 'x', 'z'],
                            units=['nm', 'nm', 'nm'],
                            scales=[voxel_size_nm, voxel_size_nm, voxel_size_nm],
                        ),
                        voxel_offset=[0, 0, 0],
                    ),
                    segments=segment_ids,
                )

                # Set initial position to center of volume
                # Position order matches dimension names: [y, x, z]
                center = [volume.shape[0] // 2 * voxel_size_nm,  # y
                          volume.shape[1] // 2 * voxel_size_nm,  # x
                          volume.shape[2] // 2 * voxel_size_nm]  # z
                s.position = center

            logger.info(f"Added layer: {layer_name}")
            if metadata['n_axons']:
                logger.info(f"  Expected axons: {metadata['n_axons']}")

        else:
            # HDF5 file
            volume, metadata, voxel_size_um = load_hdf5_volume(volume_path, downsample)

            # Determine layer name
            if metadata['bundle_id'] is not None:
                layer_name = f"Bundle_{metadata['bundle_id']:02d}"
            else:
                layer_name = volume_path.stem

            voxel_size_nm = voxel_size_um * 1000

            segment_ids = metadata['segment_ids'] if metadata['segment_ids'] else []
            if not segment_ids:
                logger.info("Finding unique segment IDs...")
                unique_ids = np.unique(volume)
                segment_ids = unique_ids[unique_ids != 0].tolist()

            logger.info(f"Found {len(segment_ids)} unique segments")

            with viewer.txn() as s:
                s.layers[layer_name] = neuroglancer.SegmentationLayer(
                    source=neuroglancer.LocalVolume(
                        data=volume,
                        dimensions=neuroglancer.CoordinateSpace(
                            names=['y', 'x', 'z'],
                            units=['nm', 'nm', 'nm'],
                            scales=[voxel_size_nm, voxel_size_nm, voxel_size_nm],
                        ),
                        voxel_offset=[0, 0, 0],
                    ),
                    segments=segment_ids,
                )

                # Set initial position to center of volume
                # Position order matches dimension names: [y, x, z]
                center = [volume.shape[0] // 2 * voxel_size_nm,  # y
                          volume.shape[1] // 2 * voxel_size_nm,  # x
                          volume.shape[2] // 2 * voxel_size_nm]  # z
                s.position = center

            logger.info(f"Added layer: {layer_name}")
            if metadata['n_axons']:
                logger.info(f"  Expected axons: {metadata['n_axons']}")

    # Add grayscale image layer if provided
    if grayscale_file is not None:
        import h5py

        logger.info(f"Loading grayscale image: {grayscale_file}")

        with h5py.File(grayscale_file, 'r') as f:
            # Try common dataset names
            if 'raw' in f:
                dset_name = 'raw'
            elif 'data' in f:
                dset_name = 'data'
            else:
                dset_name = list(f.keys())[0]

            dset = f[dset_name]
            logger.info(f"Dataset '{dset_name}': shape {dset.shape}, dtype {dset.dtype}")

            # Load with optional downsampling
            if downsample > 1:
                grayscale = dset[::downsample, ::downsample, ::downsample]
                voxel_size_nm = 50 * downsample  # Assuming 0.05 μm base voxel size
            else:
                grayscale = dset[:]
                voxel_size_nm = 50  # 0.05 μm = 50 nm

        logger.info(f"Grayscale volume: {grayscale.shape}, {grayscale.nbytes / 1024**3:.2f} GB")

        with viewer.txn() as s:
            s.layers['grayscale'] = neuroglancer.ImageLayer(
                source=neuroglancer.LocalVolume(
                    data=grayscale,
                    dimensions=neuroglancer.CoordinateSpace(
                        names=['y', 'x', 'z'],
                        units=['nm', 'nm', 'nm'],
                        scales=[voxel_size_nm, voxel_size_nm, voxel_size_nm],
                    ),
                    voxel_offset=[0, 0, 0],
                ),
            )

            # Set initial position to center if not already set
            # Position order matches dimension names: [y, x, z]
            center = [grayscale.shape[0] // 2 * voxel_size_nm,  # y
                      grayscale.shape[1] // 2 * voxel_size_nm,  # x
                      grayscale.shape[2] // 2 * voxel_size_nm]  # z
            s.position = center

        logger.info("Added grayscale image layer")

    # Print viewer URL
    print(f"\nNeuroglancer viewer URL: {viewer.get_viewer_url()}")
    print("\nPress Ctrl+C to exit\n")

    # Keep the server running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        for server in http_servers:
            server.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize bundle volumes in Neuroglancer'
    )
    parser.add_argument('volume_files', type=Path, nargs='+',
                        help='One or more volume files to visualize (Zarr or HDF5)')
    parser.add_argument('--grayscale', type=Path, default=None,
                        help='Optional grayscale image (HDF5 with "raw" dataset)')
    parser.add_argument('--bind', type=str, default='localhost',
                        help='Address to bind to (default: localhost)')
    parser.add_argument('--port', type=int, default=9999,
                        help='Port number (default: 9999)')
    parser.add_argument('--level', type=int, default=None,
                        help='Pyramid level for Zarr (0=full res, higher=lower res, default: auto)')
    parser.add_argument('--downsample', type=int, default=1,
                        help='Downsampling factor for HDF5 (default: 1)')
    parser.add_argument('--no-multires', action='store_true',
                        help='Disable multi-resolution HTTP serving (load single level into memory)')

    args = parser.parse_args()

    # Verify files exist
    for f in args.volume_files:
        if not f.exists():
            logger.error(f"File not found: {f}")
            return 1

    if args.grayscale and not args.grayscale.exists():
        logger.error(f"Grayscale file not found: {args.grayscale}")
        return 1

    visualize_volumes(args.volume_files, args.bind, args.port, args.level, args.downsample,
                      multires=not args.no_multires, grayscale_file=args.grayscale)
    return 0


if __name__ == '__main__':
    exit(main())
