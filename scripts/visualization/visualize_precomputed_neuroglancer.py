#!/usr/bin/env python3
"""
Visualize Neuroglancer Precomputed volumes.

Auto-detects population subdirectories (e.g., cc/, cg/) and serves them
via HTTP for multi-resolution visualization in Neuroglancer.

Example usage:
    # View exported CC/CG populations (4-panel with 3D)
    python scripts/visualization/visualize_precomputed_neuroglancer.py \
        data/processed/neuroglancer/LM_25_ipsi

    # 3D-only view
    python scripts/visualization/visualize_precomputed_neuroglancer.py \
        data/processed/neuroglancer/LM_25_ipsi --layout 3d

    # With grayscale overlay
    python scripts/visualization/visualize_precomputed_neuroglancer.py \
        data/processed/neuroglancer/LM_25_ipsi \
        --grayscale data/raw/Sham_25_ipsi/LM_25_ipsi_grayscale.h5
"""

import argparse
import json
import logging
import threading
import time
import urllib.request
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
        # Log requests for debugging
        logger.debug(f"HTTP: {format % args}")


def start_http_server(directory: Path, port: int = 8000) -> tuple:
    """
    Start an HTTP server to serve Precomputed files.

    Args:
        directory: Directory to serve
        port: Port to use (will try successive ports if busy)

    Returns:
        Tuple of (server, port, thread)
    """
    handler = partial(CORSRequestHandler, directory=str(directory))

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


def detect_precomputed_volumes(directory: Path) -> list:
    """
    Detect Neuroglancer Precomputed volumes in a directory.

    A precomputed volume is identified by having an 'info' file.

    Args:
        directory: Directory to scan

    Returns:
        List of (name, path) tuples for each detected volume
    """
    volumes = []

    # Check if directory itself is a precomputed volume
    if (directory / 'info').exists():
        volumes.append((directory.name, directory))
        return volumes

    # Scan subdirectories for precomputed volumes
    for subdir in sorted(directory.iterdir()):
        if subdir.is_dir() and (subdir / 'info').exists():
            volumes.append((subdir.name, subdir))

    return volumes


def get_precomputed_metadata(volume_path: Path) -> dict:
    """
    Read metadata from a Precomputed volume's info file.

    Args:
        volume_path: Path to precomputed volume directory

    Returns:
        Metadata dictionary
    """
    info_path = volume_path / 'info'
    with open(info_path) as f:
        info = json.load(f)

    # Extract key metadata
    scales = info.get('scales', [])
    if scales:
        full_res = scales[0]
        shape = full_res.get('size', [0, 0, 0])  # x, y, z order
        resolution = full_res.get('resolution', [1, 1, 1])
        scale_key = full_res.get('key', '1_1_1')
    else:
        shape = [0, 0, 0]
        resolution = [1, 1, 1]
        scale_key = '1_1_1'

    return {
        'shape': shape,  # x, y, z
        'resolution_nm': resolution,
        'n_scales': len(scales),
        'data_type': info.get('data_type', 'uint32'),
        'scale_key': scale_key,
    }


def detect_segment_ids(volume_path: Path, metadata: dict) -> list:
    """
    Detect all segment IDs by reading from the coarsest pyramid level.

    Uses the lowest resolution scale for speed - fewer/smaller chunks to read.

    Args:
        volume_path: Path to precomputed volume directory
        metadata: Volume metadata from get_precomputed_metadata

    Returns:
        List of all non-zero segment IDs
    """
    # Read info to get all scale keys
    info_path = volume_path / 'info'
    with open(info_path) as f:
        info = json.load(f)

    scales = info.get('scales', [])
    if not scales:
        return []

    # Use a mid-resolution level for good balance of speed vs completeness
    # Scale index 2 typically captures 99%+ of segments in <1s
    scale_idx = min(2, len(scales) - 1)
    target_scale = scales[scale_idx]
    scale_key = target_scale['key']
    scale_dir = volume_path / scale_key

    chunk_files = list(scale_dir.glob('*'))
    if not chunk_files:
        # Fall back to finest scale if coarsest is empty
        scale_key = scales[0]['key']
        scale_dir = volume_path / scale_key
        chunk_files = list(scale_dir.glob('*'))

    if not chunk_files:
        return []

    dtype = np.dtype(metadata['data_type'])
    all_ids = set()

    for chunk_path in chunk_files:
        try:
            data = np.fromfile(chunk_path, dtype=dtype)
            unique_ids = np.unique(data)
            non_zero_ids = unique_ids[unique_ids > 0]
            all_ids.update(non_zero_ids.tolist())
        except Exception as e:
            logger.warning(f"Could not read segment IDs from {chunk_path}: {e}")
            continue

    return sorted(all_ids)


def visualize_precomputed(
    data_dir: Path,
    bind_address: str = 'localhost',
    port: int = 9999,
    grayscale_file: Path = None,
    downsample: int = 1,
    layout: str = '4panel',
):
    """
    Visualize Precomputed volumes in Neuroglancer.

    Args:
        data_dir: Directory containing precomputed volumes (or a single volume)
        bind_address: Address to bind the viewer to
        port: Port number for Neuroglancer viewer
        grayscale_file: Optional path to grayscale HDF5 file
        downsample: Downsampling factor for grayscale (if loading into memory)
        layout: Viewer layout ('xy', 'xz', 'yz', '3d', '4panel')
    """
    # Detect precomputed volumes
    volumes = detect_precomputed_volumes(data_dir)

    if not volumes:
        logger.error(f"No precomputed volumes found in {data_dir}")
        logger.error("Expected directories containing 'info' files")
        return

    logger.info(f"Found {len(volumes)} precomputed volume(s): {[v[0] for v in volumes]}")

    # Start HTTP server for the data directory
    # Serve from parent if data_dir itself is a volume, otherwise serve data_dir
    # Use absolute path for HTTP server
    data_dir = data_dir.resolve()
    if (data_dir / 'info').exists():
        serve_dir = data_dir.parent
        url_base = data_dir.name
    else:
        serve_dir = data_dir
        url_base = ""

    server, http_port, _ = start_http_server(serve_dir, port=8000)
    logger.info(f"Serving directory: {serve_dir}")

    # Verify HTTP server is accessible
    time.sleep(0.5)  # Give server time to start
    test_volume = volumes[0][0]
    test_url = f"http://localhost:{http_port}/{test_volume}/info"
    try:
        with urllib.request.urlopen(test_url, timeout=2) as response:
            if response.status == 200:
                logger.info(f"HTTP server verified: {test_url}")
    except Exception as e:
        logger.error(f"HTTP server test failed for {test_url}: {e}")

    # Set up Neuroglancer
    neuroglancer.set_server_bind_address(bind_address, port)
    viewer = neuroglancer.Viewer()

    # Add each volume as a layer
    first_metadata = None
    for name, volume_path in volumes:
        metadata = get_precomputed_metadata(volume_path)

        if first_metadata is None:
            first_metadata = metadata

        # Build precomputed URL
        # When data_dir itself is a volume, url_base == name, so don't duplicate
        if url_base and url_base != name:
            precomputed_url = f"precomputed://http://localhost:{http_port}/{url_base}/{name}"
        else:
            precomputed_url = f"precomputed://http://localhost:{http_port}/{name}"

        logger.info(f"Serving {name} at: {precomputed_url}")
        logger.info(f"  Shape (x,y,z): {metadata['shape']}")
        logger.info(f"  Resolution: {metadata['resolution_nm']} nm")
        logger.info(f"  Scales: {metadata['n_scales']}")

        # Detect segment IDs to pre-select for visibility
        segment_ids = detect_segment_ids(volume_path, metadata)
        if segment_ids:
            logger.info(f"  Found {len(segment_ids)} segments, selecting all")

        with viewer.txn() as s:
            layer = neuroglancer.SegmentationLayer(source=precomputed_url)
            if segment_ids:
                layer.segments = set(segment_ids)
            s.layers[name] = layer

    # Set initial position to center of first volume
    if first_metadata:
        shape = first_metadata['shape']  # x, y, z
        resolution = first_metadata['resolution_nm']
        # Position in physical coordinates (nm) - Neuroglancer expects [x, y, z] order
        center = [
            shape[0] // 2 * resolution[0],  # x
            shape[1] // 2 * resolution[1],  # y
            shape[2] // 2 * resolution[2],  # z
        ]
        logger.info(f"Setting camera position to: {center} nm")
        with viewer.txn() as s:
            s.position = center
            s.layout = layout

    # Add grayscale overlay if provided
    if grayscale_file is not None:
        _add_grayscale_layer(viewer, grayscale_file, downsample)

    # Print viewer URL
    print(f"\nNeuroglancer viewer URL: {viewer.get_viewer_url()}")
    print("\nPress Ctrl+C to exit\n")

    # Keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


def _add_grayscale_layer(viewer, grayscale_file: Path, downsample: int = 1):
    """Add a grayscale image layer from HDF5 file."""
    import h5py

    logger.info(f"Loading grayscale image: {grayscale_file}")

    with h5py.File(grayscale_file, 'r') as f:
        # Try common dataset names
        for dset_name in ['raw', 'data', 'image']:
            if dset_name in f:
                break
        else:
            dset_name = list(f.keys())[0]

        dset = f[dset_name]
        logger.info(f"Dataset '{dset_name}': shape {dset.shape}, dtype {dset.dtype}")

        if downsample > 1:
            grayscale = dset[::downsample, ::downsample, ::downsample]
            voxel_size_nm = 50 * downsample  # Assuming 0.05 um base
        else:
            grayscale = dset[:]
            voxel_size_nm = 50

    logger.info(f"Grayscale volume: {grayscale.shape}, {grayscale.nbytes / 1024**3:.2f} GB")

    with viewer.txn() as s:
        s.layers['grayscale'] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                data=grayscale,
                dimensions=neuroglancer.CoordinateSpace(
                    names=['z', 'y', 'x'],
                    units=['nm', 'nm', 'nm'],
                    scales=[voxel_size_nm, voxel_size_nm, voxel_size_nm],
                ),
                voxel_offset=[0, 0, 0],
            ),
        )

    logger.info("Added grayscale image layer")


def visualize_remote_urls(
    urls: list,
    bind_address: str = 'localhost',
    port: int = 9999,
    layout: str = '4panel',
):
    """
    Visualize remote Neuroglancer Precomputed volumes.

    Args:
        urls: List of precomputed:// URLs
        bind_address: Address to bind the viewer to
        port: Port number for Neuroglancer viewer
        layout: Viewer layout ('xy', 'xz', 'yz', '3d', '4panel')
    """
    neuroglancer.set_server_bind_address(bind_address, port)
    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        s.layout = layout

    for i, url in enumerate(urls):
        name = f"layer_{i}" if len(urls) > 1 else "data"
        # Extract name from URL if possible
        url_parts = url.rstrip('/').split('/')
        if url_parts:
            name = url_parts[-1]

        logger.info(f"Adding remote layer: {url}")

        with viewer.txn() as s:
            # Determine layer type from URL structure
            if 'annotation' in url.lower() or 'seg' in url.lower():
                s.layers[name] = neuroglancer.SegmentationLayer(source=url)
            else:
                # Try as image layer first
                s.layers[name] = neuroglancer.ImageLayer(source=url)

    print(f"\nNeuroglancer viewer URL: {viewer.get_viewer_url()}")
    print("\nPress Ctrl+C to exit\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Neuroglancer Precomputed volumes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # View exported CC/CG populations
    python scripts/visualization/visualize_precomputed_neuroglancer.py \\
        data/processed/neuroglancer/LM_25_ipsi

    # With grayscale overlay
    python scripts/visualization/visualize_precomputed_neuroglancer.py \\
        data/processed/neuroglancer/LM_25_ipsi \\
        --grayscale data/raw/Sham_25_ipsi/LM_25_ipsi_grayscale.h5

    # View remote precomputed URLs
    python scripts/visualization/visualize_precomputed_neuroglancer.py \\
        --url precomputed://https://open-neurodata.s3.amazonaws.com/collman/collman15v2/annotation
        """
    )
    parser.add_argument(
        'data_dir',
        type=Path,
        nargs='?',
        default=None,
        help='Directory containing precomputed volumes (auto-detects cc/, cg/ subdirs)'
    )
    parser.add_argument(
        '--url', type=str, action='append', default=[],
        help='Remote precomputed:// URL (can be specified multiple times)'
    )
    parser.add_argument(
        '--grayscale', type=Path, default=None,
        help='Optional grayscale image (HDF5 with "raw" dataset)'
    )
    parser.add_argument(
        '--bind', type=str, default='localhost',
        help='Address to bind to (default: localhost)'
    )
    parser.add_argument(
        '--port', type=int, default=9999,
        help='Port number for Neuroglancer (default: 9999)'
    )
    parser.add_argument(
        '--downsample', type=int, default=1,
        help='Downsampling factor for grayscale (default: 1)'
    )
    parser.add_argument(
        '--layout', type=str, default='4panel',
        choices=['xy', 'xz', 'yz', '3d', '4panel'],
        help='Viewer layout (default: 4panel). Use "3d" for 3D-only view.'
    )

    args = parser.parse_args()

    # Handle remote URLs
    if args.url:
        visualize_remote_urls(
            args.url,
            bind_address=args.bind,
            port=args.port,
            layout=args.layout,
        )
        return 0

    # Handle local directory
    if args.data_dir is None:
        parser.error("Either data_dir or --url must be provided")

    if not args.data_dir.exists():
        logger.error(f"Directory not found: {args.data_dir}")
        return 1

    if args.grayscale and not args.grayscale.exists():
        logger.error(f"Grayscale file not found: {args.grayscale}")
        return 1

    visualize_precomputed(
        args.data_dir,
        bind_address=args.bind,
        port=args.port,
        grayscale_file=args.grayscale,
        downsample=args.downsample,
        layout=args.layout,
    )
    return 0


if __name__ == '__main__':
    exit(main())
