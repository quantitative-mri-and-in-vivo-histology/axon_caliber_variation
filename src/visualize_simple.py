#!/usr/bin/env python3
"""
Simple Neuroglancer visualization for Zarr labels.

Minimal script to verify coordinate system and positioning.
Serves Zarr via HTTP for multi-resolution support.
"""

import argparse
import logging
import socket
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import neuroglancer
import zarr


def check_port_available(port: int) -> bool:
    """Check if a port is available for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False

# Will be set based on --verbose flag
logger = logging.getLogger(__name__)


class CORSRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler with CORS support for Neuroglancer."""

    verbose = False  # Class variable for verbose mode

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

    def do_GET(self):
        if CORSRequestHandler.verbose:
            logger.debug(f"HTTP GET: {self.path}")
        super().do_GET()

    def log_message(self, format, *args):
        if CORSRequestHandler.verbose:
            logger.debug(f"HTTP: {format % args}")
        # Otherwise suppress HTTP access logs


def main():
    parser = argparse.ArgumentParser(description='Simple Neuroglancer visualization for Zarr labels')
    parser.add_argument('zarr_path', type=Path, help='Path to Zarr store')
    parser.add_argument('--port', type=int, default=9999, help='Neuroglancer port (default: 9999)')
    parser.add_argument('--http-port', type=int, default=8000, help='HTTP server port (default: 8000)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose debug logging')
    args = parser.parse_args()

    # Configure logging based on verbose flag
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # Enable verbose HTTP logging
        CORSRequestHandler.verbose = True
        # Enable neuroglancer debug logging
        logging.getLogger('neuroglancer').setLevel(logging.DEBUG)
        logging.getLogger('tornado').setLevel(logging.DEBUG)
        logging.getLogger('sockjs').setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    if not args.zarr_path.exists():
        logger.error(f"Zarr path not found: {args.zarr_path}")
        return 1

    # Check port availability
    if not check_port_available(args.http_port):
        logger.error(f"Port {args.http_port} is already in use!")
        logger.error(f"Try: --http-port {args.http_port + 1}")
        logger.error(f"Or kill existing process: lsof -ti :{args.http_port} | xargs kill -9")
        return 1

    if not check_port_available(args.port):
        logger.error(f"Port {args.port} is already in use!")
        logger.error(f"Try: --port {args.port + 1}")
        logger.error(f"Or kill existing process: lsof -ti :{args.port} | xargs kill -9")
        return 1

    logger.debug(f"Ports {args.http_port} and {args.port} are available")

    # Load metadata from Zarr
    logger.info(f"Loading Zarr: {args.zarr_path}")
    root = zarr.open(str(args.zarr_path), mode='r')

    # Get shape from level 0
    shape = root['0'].shape
    logger.info(f"Shape: {shape}")

    # Get voxel size from metadata
    attrs = dict(root.attrs)
    voxel_size_um = attrs.get('voxel_size_um', 0.05)
    bundle_id = attrs.get('bundle_id', 0)
    n_levels = attrs.get('n_levels', 1)

    # Get segment IDs if available
    segment_ids = None
    if 'labels' in root and 'segment_ids' in root['labels']:
        segment_ids = root['labels']['segment_ids'][:].tolist()
        logger.info(f"Found {len(segment_ids)} segment IDs")

    logger.info(f"Voxel size: {voxel_size_um} um")
    logger.info(f"Bundle ID: {bundle_id}")
    logger.info(f"Pyramid levels: {n_levels}")

    # Start HTTP server
    parent_dir = args.zarr_path.parent
    zarr_name = args.zarr_path.name

    handler = partial(CORSRequestHandler, directory=str(parent_dir))
    server = HTTPServer(('localhost', args.http_port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    logger.info(f"HTTP server started on port {args.http_port}")

    # Build Zarr URL
    zarr_url = f"zarr://http://localhost:{args.http_port}/{zarr_name}"
    logger.info(f"Zarr URL: {zarr_url}")

    # Wait for HTTP server to be ready
    import urllib.request
    import time
    test_url = f"http://localhost:{args.http_port}/{zarr_name}/.zattrs"
    logger.debug(f"Verifying HTTP server at: {test_url}")
    for attempt in range(10):
        try:
            with urllib.request.urlopen(test_url, timeout=1) as response:
                if response.status == 200:
                    logger.info("HTTP server verified - .zattrs accessible")
                    if args.verbose:
                        content = response.read().decode('utf-8')
                        logger.debug(f".zattrs content ({len(content)} bytes): {content[:200]}...")
                    break
        except Exception as e:
            logger.debug(f"Attempt {attempt+1}/10 failed: {e}")
            time.sleep(0.1)
    else:
        logger.warning("Could not verify HTTP server - proceeding anyway")

    # Additional file checks in verbose mode
    if args.verbose:
        check_files = ['.zgroup', '0/.zarray']
        for check_file in check_files:
            check_url = f"http://localhost:{args.http_port}/{zarr_name}/{check_file}"
            try:
                with urllib.request.urlopen(check_url, timeout=1) as response:
                    logger.debug(f"✓ {check_file} accessible (status {response.status})")
            except Exception as e:
                logger.debug(f"✗ {check_file} not accessible: {e}")

    # Set up Neuroglancer
    neuroglancer.set_server_bind_address('localhost', args.port)
    viewer = neuroglancer.Viewer()

    # Add segmentation layer
    layer_name = f"Bundle_{bundle_id:02d}"

    with viewer.txn() as s:
        layer_kwargs = {'source': zarr_url}
        if segment_ids:
            layer_kwargs['segments'] = segment_ids
        s.layers[layer_name] = neuroglancer.SegmentationLayer(**layer_kwargs)

        # Set initial position to center of volume
        # Position order matches OME-NGFF axes order: [y, x, z]
        # Shape is [y, x, z]
        center = [shape[0] // 2 * voxel_size_um,  # y
                  shape[1] // 2 * voxel_size_um,  # x
                  shape[2] // 2 * voxel_size_um]  # z
        s.position = center

        logger.info(f"Center position (um): {center}")

    logger.info(f"Added layer: {layer_name}")

    # Print viewer URL
    print(f"\nNeuroglancer viewer URL: {viewer.get_viewer_url()}")
    print("\nPress Ctrl+C to exit\n")

    # Keep running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()

    return 0


if __name__ == '__main__':
    exit(main())
