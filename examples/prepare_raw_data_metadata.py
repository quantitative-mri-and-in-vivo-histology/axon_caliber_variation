#!/usr/bin/env python3
"""
Create metadata JSON files for raw data volumes.

For each .mat file in data/raw/, creates a companion .json file with:
- voxel_size: [x, y, z] in micrometers
- unit: "micrometer"

Convention:
- HM (high magnification): [0.015, 0.015, 0.05] μm
- LM (low magnification): [0.05, 0.05, 0.05] μm (isotropic)

Usage:
    python examples/prepare_raw_data_metadata.py data/raw
"""

import argparse
import json
from pathlib import Path
from typing import List


def determine_voxel_size(filename: str) -> List[float]:
    """
    Determine voxel size based on filename prefix.

    Args:
        filename: Name of the .mat file (e.g., "HM_25_ipsi_myelinated_axons.mat")

    Returns:
        Voxel size as [x, y, z] in micrometers, or None if unknown prefix
    """
    if filename.startswith("HM_"):
        # High magnification: anisotropic
        return [0.015, 0.015, 0.05]
    elif filename.startswith("LM_"):
        # Low magnification: isotropic
        return [0.05, 0.05, 0.05]
    else:
        # Unknown prefix - return None to skip
        return None


def create_metadata_file(mat_file: Path, overwrite: bool = False) -> bool:
    """
    Create a .json metadata file for a .mat volume.

    Args:
        mat_file: Path to the .mat file
        overwrite: If True, overwrite existing .json file

    Returns:
        True if metadata file was created, False if skipped
    """
    json_file = mat_file.with_suffix(".json")

    # Skip if already exists and overwrite is False
    if json_file.exists() and not overwrite:
        print(f"Skipping {json_file.name} (already exists)")
        return False

    # Determine voxel size from filename
    voxel_size = determine_voxel_size(mat_file.name)

    # Skip if unknown prefix
    if voxel_size is None:
        print(f"Skipping {mat_file.name} (unknown prefix)")
        return False

    # Create metadata dictionary
    metadata = {
        "voxel_size": voxel_size,
        "unit": "micrometer"
    }

    # Write JSON file
    with open(json_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Created {json_file.relative_to(json_file.parents[2])}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create metadata JSON files for raw data volumes (.mat, .h5, .hdf5)"
    )
    parser.add_argument(
        "raw_data_dir",
        type=Path,
        help="Root directory containing raw data (e.g., data/raw)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files"
    )

    args = parser.parse_args()

    # Find all volume files (.mat, .h5, .hdf5)
    extensions = ["*.mat", "*.h5", "*.hdf5"]
    volume_files = []
    for ext in extensions:
        volume_files.extend(args.raw_data_dir.rglob(ext))

    volume_files = sorted(set(volume_files))  # Remove duplicates and sort

    if not volume_files:
        print(f"No volume files found in {args.raw_data_dir}")
        return

    print(f"Found {len(volume_files)} volume files")
    print()

    # Create metadata file for each volume file
    created_count = 0
    skipped_count = 0
    for volume_file in volume_files:
        if create_metadata_file(volume_file, overwrite=args.overwrite):
            created_count += 1
        else:
            skipped_count += 1

    print()
    print(f"Done! Created {created_count} metadata files")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} files")


if __name__ == "__main__":
    main()
