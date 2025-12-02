#!/usr/bin/env python3
"""
Create metadata JSON files for LM (low magnification) raw data volumes.

For each LM_*.mat file in data/raw/, creates a companion .json file with:
- voxel_size: [0.05, 0.05, 0.05] μm (isotropic)
- unit: "micrometer"

Usage:
    # Create/update metadata files
    python examples/prepare_lm_metadata.py data/raw
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Dict


def create_metadata_file(mat_file: Path) -> Dict:
    """
    Create or update a .json metadata file for an LM .mat volume.

    Args:
        mat_file: Path to the .mat file

    Returns:
        Metadata dictionary
    """
    json_file = mat_file.with_suffix(".json")

    # Load existing metadata if present to preserve additional fields
    existing_metadata = {}
    if json_file.exists():
        try:
            with open(json_file, 'r') as f:
                existing_metadata = json.load(f)
        except:
            pass

    # Create base metadata for LM data (isotropic)
    metadata = {
        "voxel_size": [0.05, 0.05, 0.05],  # Isotropic: X, Y, Z in micrometers
        "unit": "micrometer"
    }

    # Preserve any additional fields from existing metadata
    for key, value in existing_metadata.items():
        if key not in metadata:
            metadata[key] = value

    # Write JSON file
    with open(json_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Created/updated {json_file.relative_to(json_file.parents[2])}")
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Create metadata JSON files for LM (low magnification) raw data volumes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create/update metadata files
  python prepare_lm_metadata.py data/raw
        """
    )
    parser.add_argument(
        "raw_data_dir",
        type=Path,
        help="Root directory containing raw data (e.g., data/raw)"
    )

    args = parser.parse_args()

    # Find all LM .mat files
    mat_files = sorted(args.raw_data_dir.rglob("LM_*.mat"))

    if not mat_files:
        print(f"No LM_*.mat files found in {args.raw_data_dir}")
        return

    print(f"Found {len(mat_files)} LM volume files")
    print()

    # Process each file
    for mat_file in mat_files:
        create_metadata_file(mat_file)

    print()
    print(f"Done! Created/updated {len(mat_files)} metadata files")


if __name__ == "__main__":
    main()
