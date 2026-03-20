#!/usr/bin/env python3
"""
Unified 3D volume viewer for Neuroglancer.

Supports grayscale images, label segmentations, and bounding box overlays
from multiple formats (.mat, .h5, .zarr). Non-zarr inputs are automatically
converted to temporary multi-resolution OME-Zarr pyramids.

Example usage:
    # Segmentation only
    python scripts/visualization/visualize_3d.py \
        --label data/raw/rat/LM/sham_25_ipsi_cc_myelin.zarr

    # Grayscale + segmentation
    python scripts/visualization/visualize_3d.py \
        --grayscale data/source/rat/Sham_25_ipsi/LM_25_ipsi.h5 \
        --label data/raw/rat/LM/sham_25_ipsi_cc_myelin.zarr \
                data/raw/rat/LM/sham_25_ipsi_cg_myelin.zarr

    # Raw .mat with bounding box overlay
    python scripts/visualization/visualize_3d.py \
        --grayscale data/source/rat/Sham_25_ipsi/LM_25_ipsi.h5 \
        --label data/source/rat/Sham_25_ipsi/LM_25_ipsi_myelinated_axons.mat \
        --bbox data/raw/rat/LM/sham_25_ipsi_population_rois.json

    # Multiple formats together
    python scripts/visualization/visualize_3d.py \
        --grayscale data/source/rat/Sham_25_ipsi/LM_25_ipsi.h5 \
        --label data/raw/rat/LM/sham_25_ipsi_cc_myelin.zarr \
        --bbox data/raw/rat/LM/sham_25_ipsi_population_rois.json
"""

import argparse
import json
import logging
import tempfile
import threading
import time
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import h5py
import neuroglancer
import numpy as np

from axonometry.io import write_ome_zarr_pyramid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP server with CORS
# ---------------------------------------------------------------------------

class CORSRequestHandler(SimpleHTTPRequestHandler):
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
    logger.info(f"HTTP server on port {actual_port}")
    return server, actual_port


# ---------------------------------------------------------------------------
# Volume loading helpers
# ---------------------------------------------------------------------------

def load_mat(path: Path) -> np.ndarray:
    """Load a volume from a MATLAB .mat file (HDF5-based v7.3)."""
    with h5py.File(str(path), 'r') as f:
        for key in f.keys():
            if not key.startswith('#') and not key.startswith('_'):
                return f[key][:]
    raise ValueError(f"No data found in {path}")


def load_h5(path: Path) -> np.ndarray:
    """Load a volume from an HDF5 .h5 file."""
    with h5py.File(str(path), 'r') as f:
        for dset_name in ['raw', 'data', 'image', 'labels', 'final_lbl']:
            if dset_name in f:
                return f[dset_name][:].squeeze()
        # Fallback to first dataset
        first_key = list(f.keys())[0]
        return f[first_key][:].squeeze()


def is_zarr(path: Path) -> bool:
    """Check if a path is an OME-Zarr store."""
    if path.suffix == '.zarr':
        return True
    if path.is_dir():
        return any((path / f).exists() for f in ['.zattrs', '.zgroup', 'zarr.json'])
    return False


def get_zarr_voxel_size(zarr_path: Path) -> float:
    """Read voxel size from OME-Zarr metadata."""
    import zarr
    root = zarr.open_group(str(zarr_path), mode='r')
    multiscales = root.attrs.get('multiscales', [{}])
    if multiscales:
        datasets = multiscales[0].get('datasets', [])
        if datasets:
            transforms = datasets[0].get('coordinateTransformations', [])
            for t in transforms:
                if t.get('type') == 'scale':
                    return t['scale'][0]  # Assume isotropic
    return 0.05  # Default


def convert_to_zarr(
    volume: np.ndarray,
    tmp_dir: Path,
    name: str,
    voxel_size_um: float,
    mode: str = 'nearest',
    num_levels: int = 4,
) -> Path:
    """Convert an in-memory volume to a temporary OME-Zarr pyramid."""
    zarr_path = tmp_dir / f"{name}.zarr"
    write_ome_zarr_pyramid(
        volume, zarr_path,
        voxel_size_um=voxel_size_um,
        num_levels=num_levels,
        downsample_mode=mode,
    )
    return zarr_path


# ---------------------------------------------------------------------------
# Layer creation
# ---------------------------------------------------------------------------

def add_grayscale_layer(viewer, name: str, zarr_path: Path, http_port: int, tmp_dir: Path):
    """Add a grayscale image layer from an OME-Zarr."""
    import zarr
    rel = zarr_path.relative_to(tmp_dir)
    url = f"zarr://http://localhost:{http_port}/{rel}"

    # Compute display range from level 0
    root = zarr.open_group(str(zarr_path), mode='r')
    sample = root['0'][:]
    data_min = int(np.percentile(sample, 1))
    data_max = int(np.percentile(sample, 99))

    shader = f"""
#uicontrol invlerp normalized(range=[{data_min}, {data_max}])
void main() {{
  emitGrayscale(normalized());
}}
"""
    with viewer.txn() as s:
        s.layers[name] = neuroglancer.ImageLayer(source=url, shader=shader)

    logger.info(f"  Added grayscale layer: {name}")


def add_segmentation_layer(viewer, name: str, zarr_path: Path, http_port: int, tmp_dir: Path):
    """Add a segmentation layer from an OME-Zarr."""
    rel = zarr_path.relative_to(tmp_dir)
    url = f"zarr://http://localhost:{http_port}/{rel}"

    with viewer.txn() as s:
        s.layers[name] = neuroglancer.SegmentationLayer(source=url)

    logger.info(f"  Added segmentation layer: {name}")


def add_bbox_layer(viewer, name: str, roi_json: Path, voxel_size_um: float):
    """Add bounding box annotation from a simplified ROI JSON.

    Expects JSON with: fiber_dir_axis, min, max (in voxel coordinates).
    """
    with open(roi_json) as f:
        roi = json.load(f)

    roi_min = roi['min']
    roi_max = roi['max']

    # Convert voxel coordinates to physical (um) — matches OME-Zarr coordinate space
    point_a = [v * voxel_size_um for v in roi_min]
    point_b = [v * voxel_size_um for v in roi_max]

    # Derive a label from the filename (e.g. sham_25_ipsi_cc_roi.json -> CC)
    roi_label = roi_json.stem.replace('_roi', '').split('_')[-1].upper()

    # Per-population colors
    BBOX_COLORS = {
        'CC': '#ff4444',  # red
        'CG': '#44aaff',  # blue
    }
    color = BBOX_COLORS.get(roi_label, '#ffff00')

    annotation = neuroglancer.AxisAlignedBoundingBoxAnnotation(
        point_a=point_a,
        point_b=point_b,
        id=roi_label,
        description=f"{roi_label} ROI",
    )

    # Explicit coordinate space matching the OME-Zarr layers (micrometers)
    coord_space = neuroglancer.CoordinateSpace(
        names=['z', 'y', 'x'],
        scales=[1, 1, 1],
        units=['um', 'um', 'um'],
    )

    with viewer.txn() as s:
        s.layers[name] = neuroglancer.LocalAnnotationLayer(
            dimensions=coord_space,
            annotations=[annotation],
            annotation_color=color,
        )

    logger.info(f"  Added bbox: {roi_label} {roi_min} -> {roi_max} vox ({color})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Unified 3D volume viewer for Neuroglancer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--grayscale', type=Path, nargs='+', default=[],
                        help='Grayscale image(s): .h5, .mat, or .zarr')
    parser.add_argument('--label', type=Path, nargs='+', default=[],
                        help='Label segmentation(s): .mat, .h5, or .zarr')
    parser.add_argument('--bbox', type=Path, nargs='+', default=[],
                        help='Bounding box ROI(s): *_population_rois.json')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                        help='Voxel size in um for non-zarr inputs (default: 0.05)')
    parser.add_argument('--num-levels', type=int, default=4,
                        help='Pyramid levels for temp zarr conversion (default: 4)')
    parser.add_argument('--bind', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument('--layout', type=str, default='4panel',
                        choices=['xy', 'xz', 'yz', '3d', '4panel'])

    args = parser.parse_args()

    if not args.grayscale and not args.label and not args.bbox:
        parser.error("At least one of --grayscale, --label, or --bbox is required")

    with tempfile.TemporaryDirectory(prefix='neuroglancer_') as tmp_str:
        tmp_dir = Path(tmp_str)
        print(f"Temp directory: {tmp_dir}")

        # Collect all zarr paths (real or converted) for serving
        zarr_paths = []  # (zarr_path, layer_type, name)

        # --- Process grayscale inputs ---
        for i, gray_path in enumerate(args.grayscale):
            name = gray_path.stem
            if is_zarr(gray_path):
                link = tmp_dir / f"{name}_gray.zarr"
                link.symlink_to(gray_path.resolve())
                zarr_paths.append((link, 'grayscale', name))
                print(f"[{i+1}/{len(args.grayscale)}] Grayscale (zarr, symlinked): {gray_path}")
            else:
                print(f"[{i+1}/{len(args.grayscale)}] Grayscale: reading {gray_path} ...",
                      flush=True)
                vol = load_h5(gray_path) if gray_path.suffix == '.h5' else load_mat(gray_path)
                print(f"  shape={vol.shape}, dtype={vol.dtype}, "
                      f"{vol.nbytes / 1024**3:.1f} GB")
                print(f"  Converting to OME-Zarr ({args.num_levels} levels, nearest) ...",
                      flush=True)
                zarr_path = convert_to_zarr(
                    vol, tmp_dir, f"{name}_gray",
                    voxel_size_um=args.voxel_size,
                    mode='nearest',
                    num_levels=args.num_levels,
                )
                del vol
                zarr_paths.append((zarr_path, 'grayscale', name))
                print(f"  Done: {zarr_path.name}")

        # --- Process label inputs ---
        for i, label_path in enumerate(args.label):
            name = label_path.stem.replace('_myelinated_axons', '').replace('_myelin', '')
            if is_zarr(label_path):
                link = tmp_dir / f"{name}_seg.zarr"
                link.symlink_to(label_path.resolve())
                zarr_paths.append((link, 'segmentation', name))
                print(f"[{i+1}/{len(args.label)}] Label (zarr, symlinked): {label_path}")
            else:
                print(f"[{i+1}/{len(args.label)}] Label: reading {label_path} ...",
                      flush=True)
                vol = load_h5(label_path) if label_path.suffix == '.h5' else load_mat(label_path)
                print(f"  shape={vol.shape}, dtype={vol.dtype}, "
                      f"{vol.nbytes / 1024**3:.1f} GB")
                print(f"  Converting to OME-Zarr ({args.num_levels} levels, nearest) ...",
                      flush=True)
                zarr_path = convert_to_zarr(
                    vol, tmp_dir, f"{name}_seg",
                    voxel_size_um=args.voxel_size,
                    mode='nearest',
                    num_levels=args.num_levels,
                )
                del vol
                zarr_paths.append((zarr_path, 'segmentation', name))
                print(f"  Done: {zarr_path.name}")

        # --- Start HTTP server ---
        print("Starting HTTP server ...", flush=True)
        server, http_port = start_http_server(tmp_dir)

        # --- Set up Neuroglancer viewer ---
        print("Setting up Neuroglancer viewer ...", flush=True)
        neuroglancer.set_server_bind_address(args.bind, args.port)
        viewer = neuroglancer.Viewer()

        # Determine voxel size for position centering
        voxel_size_um = args.voxel_size
        if zarr_paths:
            first_zarr = zarr_paths[0][0]
            voxel_size_um = get_zarr_voxel_size(first_zarr)

        # Add layers
        print("Adding layers ...", flush=True)
        for zarr_path, layer_type, name in zarr_paths:
            if layer_type == 'grayscale':
                add_grayscale_layer(viewer, f"{name}_image", zarr_path, http_port, tmp_dir)
            else:
                add_segmentation_layer(viewer, f"{name}_seg", zarr_path, http_port, tmp_dir)

        # Add bounding box layers
        for bbox_path in args.bbox:
            bbox_name = bbox_path.stem.replace('_roi', '')
            add_bbox_layer(viewer, f"{bbox_name}_bbox", bbox_path, voxel_size_um)

        # Set layout and center position
        with viewer.txn() as s:
            s.layout = args.layout

            if zarr_paths:
                import zarr
                first_zarr = zarr_paths[0][0]
                root = zarr.open_group(str(first_zarr), mode='r')
                shape = root['0'].shape
                center = [shape[0] // 2 * voxel_size_um,
                          shape[1] // 2 * voxel_size_um,
                          shape[2] // 2 * voxel_size_um]
                s.position = center

        # Print summary
        url = viewer.get_viewer_url()
        print(f"\nNeuroglancer viewer: {url}")
        print(f"\nLayers:")
        for _, layer_type, name in zarr_paths:
            print(f"  {layer_type}: {name}")
        for bbox_path in args.bbox:
            print(f"  bbox: {bbox_path.stem}")
        print(f"\nVoxel size: {voxel_size_um} um")
        print("Press Ctrl+C to exit\n")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.shutdown()


if __name__ == '__main__':
    main()
