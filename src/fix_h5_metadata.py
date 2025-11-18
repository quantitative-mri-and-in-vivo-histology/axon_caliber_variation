#!/usr/bin/env python3
"""
Add missing metadata attributes to HDF5 files and optionally convert data type.

Usage:
    python fix_h5_metadata.py input.h5 --voxel-size 0.05 --convert-to-float
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def add_metadata_and_convert(input_file: Path,
                             output_file: Path = None,
                             voxel_size_um: float = 0.05,
                             convert_to_float: bool = False):
    """
    Add missing metadata attributes to HDF5 file and optionally convert to float.

    Args:
        input_file: Input HDF5 file
        output_file: Output HDF5 file (if None, modifies in place)
        voxel_size_um: Voxel size in micrometers
        convert_to_float: Convert dataset to float32
    """
    logger.info(f"Processing: {input_file}")

    # If output file specified, copy first
    if output_file is not None:
        logger.info(f"Creating output file: {output_file}")
        import shutil
        output_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_file, output_file)
        work_file = output_file
    else:
        logger.info("Modifying file in place")
        work_file = input_file

    with h5py.File(work_file, 'r+') as f:
        if 'labels' not in f:
            logger.error("Dataset 'labels' not found!")
            return

        dset = f['labels']
        logger.info(f"Dataset shape: {dset.shape}")
        logger.info(f"Dataset dtype: {dset.dtype}")

        # Add element_size_um attribute
        if 'element_size_um' not in dset.attrs:
            logger.info(f"Adding element_size_um = [{voxel_size_um}, {voxel_size_um}, {voxel_size_um}]")
            dset.attrs['element_size_um'] = [voxel_size_um, voxel_size_um, voxel_size_um]
        else:
            logger.info(f"element_size_um already exists: {dset.attrs['element_size_um']}")

        # Add voxel_size_um if missing
        if 'voxel_size_um' not in dset.attrs:
            logger.info(f"Adding voxel_size_um = {voxel_size_um}")
            dset.attrs['voxel_size_um'] = voxel_size_um
        else:
            logger.info(f"voxel_size_um already exists: {dset.attrs['voxel_size_um']}")

        # Convert to float if requested
        if convert_to_float and dset.dtype != np.float32:
            logger.info("Converting dataset to float32...")
            logger.warning("This will create a temporary copy - ensure sufficient disk space!")

            # Create temporary dataset
            temp_name = 'labels_float32_temp'
            if temp_name in f:
                del f[temp_name]

            # Copy with conversion in chunks to save memory
            chunk_shape = dset.chunks if dset.chunks else (min(100, dset.shape[0]),
                                                            min(512, dset.shape[1]),
                                                            min(512, dset.shape[2]))

            dset_temp = f.create_dataset(
                temp_name,
                shape=dset.shape,
                dtype=np.float32,
                chunks=chunk_shape,
                compression='gzip',
                compression_opts=1
            )

            # Copy attributes
            for key, val in dset.attrs.items():
                dset_temp.attrs[key] = val

            # Copy data in batches
            batch_size = 50
            for z_start in tqdm(range(0, dset.shape[0], batch_size),
                               desc="Converting to float32"):
                z_end = min(z_start + batch_size, dset.shape[0])
                data_batch = dset[z_start:z_end, :, :].astype(np.float32)
                dset_temp[z_start:z_end, :, :] = data_batch

            # Delete original and rename
            logger.info("Replacing original dataset...")
            del f['labels']
            f['labels'] = f[temp_name]
            del f[temp_name]

            logger.info(f"Conversion complete! New dtype: {f['labels'].dtype}")

        logger.info("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add missing metadata to HDF5 files and optionally convert to float'
    )
    parser.add_argument('input_file', type=Path,
                       help='Input HDF5 file')
    parser.add_argument('output_file', type=Path, nargs='?',
                       help='Output HDF5 file (optional, default: modify in place)')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                       help='Voxel size in micrometers (default: 0.05)')
    parser.add_argument('--convert-to-float', action='store_true',
                       help='Convert dataset to float32')

    args = parser.parse_args()

    add_metadata_and_convert(
        args.input_file,
        output_file=args.output_file,
        voxel_size_um=args.voxel_size,
        convert_to_float=args.convert_to_float
    )
