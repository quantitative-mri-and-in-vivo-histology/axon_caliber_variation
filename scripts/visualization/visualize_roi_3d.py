#!/usr/bin/env python3
"""
Visualize an extracted ROI segmentation volume with its grayscale image.

Takes a segmentation OME-Zarr volume and an optional grayscale OME-Zarr volume
(both produced by extract_lm_rois.py) and displays them overlaid in
Neuroglancer with multi-resolution pyramid support via HTTP serving.

Example usage:
    # Segmentation + grayscale overlay
    python scripts/visualization/visualize_roi_3d.py \
        data/raw/rat/LM/sham_25_ipsi_cc_myelin.zarr \
        --grayscale data/raw/rat/LM/sham_25_ipsi_cc_grayscale.zarr

    # Segmentation only, 3D view
    python scripts/visualization/visualize_roi_3d.py \
        data/raw/rat/LM/sham_25_ipsi_cg_myelin.zarr --layout 3d
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
import zarr

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
        pass


def start_http_server(directory: Path, port: int = 8000) -> tuple:
    """Start an HTTP server with CORS support."""
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


def visualize_segmentation(
    segmentation_path: Path,
    grayscale_path: Path = None,
    metadata_path: Path = None,
    bind_address: str = 'localhost',
    port: int = 9999,
    layout: str = '4panel',
):
    """
    Visualize a segmentation volume overlaid on its grayscale image.

    Uses HTTP serving for multi-resolution pyramid support.

    Args:
        segmentation_path: Path to segmentation OME-Zarr volume
        grayscale_path: Optional path to grayscale OME-Zarr volume
        metadata_path: Optional path to JSON metadata (for voxel size)
        bind_address: Address to bind the viewer to
        port: Port for Neuroglancer viewer
        layout: Viewer layout ('xy', 'xz', 'yz', '3d', '4panel')
    """
    name = segmentation_path.stem

    # Read voxel size from metadata
    voxel_size_um = 0.05
    if metadata_path and metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        voxel_size_um = meta.get('voxel_size', [0.05])[0]
        logger.info(f"Population: {meta.get('population', '?')}")
        logger.info(f"Voxel size: {voxel_size_um} um")

    # Collect all zarr paths that need serving
    zarr_paths = [segmentation_path.resolve()]
    if grayscale_path and grayscale_path.exists():
        zarr_paths.append(grayscale_path.resolve())

    # Find common parent directory for HTTP serving
    serve_dir = zarr_paths[0].parent
    for p in zarr_paths[1:]:
        # Find common ancestor
        while not str(p).startswith(str(serve_dir)):
            serve_dir = serve_dir.parent

    serve_dir = serve_dir.resolve()
    server, http_port, _ = start_http_server(serve_dir)
    logger.info(f"Serving directory: {serve_dir}")

    # Verify HTTP accessibility for each zarr store
    for zp in zarr_paths:
        rel = zp.relative_to(serve_dir)
        test_url = f"http://localhost:{http_port}/{rel}/.zattrs"
        try:
            with urllib.request.urlopen(test_url, timeout=2) as resp:
                data = json.loads(resp.read())
                if 'multiscales' in data:
                    logger.info(f"HTTP OK: {rel} (multiscales found)")
                else:
                    logger.warning(f"HTTP OK but no multiscales in .zattrs: {rel}")
        except Exception as e:
            logger.error(f"HTTP FAIL: {test_url} - {e}")
            logger.error("Neuroglancer will not be able to load this zarr store.")

    neuroglancer.set_server_bind_address(bind_address, port)
    viewer = neuroglancer.Viewer()

    # Grayscale image layer (add first so it renders underneath)
    if grayscale_path and grayscale_path.exists():
        gray_rel = grayscale_path.resolve().relative_to(serve_dir)
        gray_url = f"zarr://http://localhost:{http_port}/{gray_rel}"
        logger.info(f"Grayscale URL: {gray_url}")

        # Compute intensity window from coarsest pyramid level
        gray_root = zarr.open_group(str(grayscale_path.resolve()), mode='r')
        ms = gray_root.attrs.get('multiscales', [{}])[0]
        datasets = ms.get('datasets', [])
        coarsest = datasets[-1]['path'] if datasets else '0'
        sample = gray_root[coarsest][:]
        data_min = int(np.percentile(sample, 1))
        data_max = int(np.percentile(sample, 99))
        logger.info(f"  Intensity window: [{data_min}, {data_max}]")

        shader = f"""
#uicontrol invlerp normalized(range=[{data_min}, {data_max}])
void main() {{
  emitGrayscale(normalized());
}}
"""

        with viewer.txn() as s:
            s.layers['image'] = neuroglancer.ImageLayer(
                source=gray_url,
                shader=shader,
            )
        logger.info("Added layer: image")

    # Segmentation layer
    seg_rel = segmentation_path.resolve().relative_to(serve_dir)
    seg_url = f"zarr://http://localhost:{http_port}/{seg_rel}"
    logger.info(f"Segmentation URL: {seg_url}")

    with viewer.txn() as s:
        s.layers['segmentation'] = neuroglancer.SegmentationLayer(source=seg_url)
        s.layout = layout

    logger.info("Added layer: segmentation")

    print(f"\nNeuroglancer viewer URL: {viewer.get_viewer_url()}")
    print(f"\nVolume: {name}")
    print(f"  Segmentation: {segmentation_path.name}")
    if grayscale_path and grayscale_path.exists():
        print(f"  Grayscale:    {grayscale_path.name}")
    print("\nPress Ctrl+C to exit\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize an extracted ROI segmentation with its grayscale image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/visualization/visualize_roi_3d.py \\
        data/raw/rat/LM/sham_25_ipsi_cc_myelin.zarr \\
        --grayscale data/raw/rat/LM/sham_25_ipsi_cc_grayscale.zarr

    python scripts/visualization/visualize_roi_3d.py \\
        data/raw/rat/LM/sham_25_ipsi_cg_myelin.zarr \\
        --grayscale data/raw/rat/LM/sham_25_ipsi_cg_grayscale.zarr
        """
    )
    parser.add_argument(
        'segmentation', type=Path,
        help='Path to segmentation OME-Zarr volume'
    )
    parser.add_argument(
        '--grayscale', type=Path, default=None,
        help='Path to grayscale OME-Zarr volume'
    )
    parser.add_argument(
        '--metadata', type=Path, default=None,
        help='Path to JSON metadata file (for voxel size)'
    )
    parser.add_argument(
        '--bind', type=str, default='localhost',
        help='Address to bind to (default: localhost)'
    )
    parser.add_argument(
        '--port', type=int, default=9999,
        help='Port for Neuroglancer (default: 9999)'
    )
    parser.add_argument(
        '--layout', type=str, default='4panel',
        choices=['xy', 'xz', 'yz', '3d', '4panel'],
        help='Viewer layout (default: 4panel)'
    )

    args = parser.parse_args()

    if not args.segmentation.exists():
        logger.error(f"Not found: {args.segmentation}")
        return 1

    if args.grayscale and not args.grayscale.exists():
        logger.error(f"Grayscale not found: {args.grayscale}")
        return 1

    visualize_segmentation(
        args.segmentation,
        grayscale_path=args.grayscale,
        metadata_path=args.metadata,
        bind_address=args.bind,
        port=args.port,
        layout=args.layout,
    )
    return 0


if __name__ == '__main__':
    exit(main())
