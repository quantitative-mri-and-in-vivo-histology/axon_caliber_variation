#!/usr/bin/env python3
"""
Simple Neuroglancer visualization for Zarr labels using LocalVolume.

Loads the entire volume into memory for direct visualization.
Useful for testing coordinate systems without HTTP serving complexity.
"""

import argparse
import logging
from pathlib import Path

import neuroglancer
import numpy as np
import zarr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Simple Neuroglancer visualization (LocalVolume)')
    parser.add_argument('zarr_path', type=Path, help='Path to Zarr store')
    parser.add_argument('--port', type=int, default=9999, help='Neuroglancer port (default: 9999)')
    parser.add_argument('--level', type=int, default=0, help='Pyramid level to load (default: 0 = full res)')
    args = parser.parse_args()

    if not args.zarr_path.exists():
        logger.error(f"Zarr path not found: {args.zarr_path}")
        return 1

    # Load Zarr
    logger.info(f"Loading Zarr: {args.zarr_path}")
    root = zarr.open(str(args.zarr_path), mode='r')

    # Get metadata
    attrs = dict(root.attrs)
    voxel_size_um = attrs.get('voxel_size_um', 0.05)
    bundle_id = attrs.get('bundle_id', 0)

    # Get segment IDs if available
    segment_ids = None
    if 'labels' in root and 'segment_ids' in root['labels']:
        segment_ids = root['labels']['segment_ids'][:].tolist()
        logger.info(f"Found {len(segment_ids)} segment IDs")

    # Load the specified level
    level_key = str(args.level)
    if level_key not in root:
        logger.error(f"Level {args.level} not found in Zarr")
        return 1

    logger.info(f"Loading level {args.level}...")
    volume = root[level_key][:]

    # Adjust voxel size for pyramid level
    voxel_size_um = voxel_size_um * (2 ** args.level)
    voxel_size_nm = voxel_size_um * 1000

    logger.info(f"Shape: {volume.shape}")
    logger.info(f"Dtype: {volume.dtype}")
    logger.info(f"Memory: {volume.nbytes / 1024**3:.2f} GB")
    logger.info(f"Voxel size: {voxel_size_um} um ({voxel_size_nm} nm)")

    # Set up Neuroglancer
    neuroglancer.set_server_bind_address('localhost', args.port)
    viewer = neuroglancer.Viewer()

    # Add segmentation layer
    layer_name = f"Bundle_{bundle_id:02d}"

    with viewer.txn() as s:
        layer_kwargs = {
            'source': neuroglancer.LocalVolume(
                data=volume,
                dimensions=neuroglancer.CoordinateSpace(
                    names=['y', 'x', 'z'],
                    units=['nm', 'nm', 'nm'],
                    scales=[voxel_size_nm, voxel_size_nm, voxel_size_nm],
                ),
                voxel_offset=[0, 0, 0],
            ),
        }
        if segment_ids:
            layer_kwargs['segments'] = segment_ids
        s.layers[layer_name] = neuroglancer.SegmentationLayer(**layer_kwargs)

        # Set initial position to center of volume
        # Position order matches dimension names: [y, x, z]
        center = [volume.shape[0] // 2 * voxel_size_nm,  # y
                  volume.shape[1] // 2 * voxel_size_nm,  # x
                  volume.shape[2] // 2 * voxel_size_nm]  # z
        s.position = center

        logger.info(f"Center position (nm): {center}")

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

    return 0


if __name__ == '__main__':
    exit(main())
