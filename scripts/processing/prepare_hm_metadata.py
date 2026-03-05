#!/usr/bin/env python3
"""
Create metadata JSON files for HM (high magnification) raw data volumes.

For each HM_*.mat file in data/raw/, creates a companion .json file with:
- voxel_size: [0.015, 0.015, 0.05] μm (anisotropic)
- unit: "micrometer"
- dominant_axis: Axis where axons appear most circular (0=Z, 1=Y, 2=X)

Orientation Analysis Workflow:
1. Analyzes only HM_*_myelinated_axons.mat files
2. Determines dominant axis by computing eccentricity along each axis
   - Axis with lowest eccentricity (most circular) is perpendicular to fiber direction
3. Writes dominant_axis to the myelinated file's JSON
4. Propagates dominant_axis to sibling files in the same directory

Visualization (optional):
- Shows only myelinated_axons files
- Displays central slices along all 3 axes for comparison

Usage:
    # Create/update metadata only (no visualization)
    python scripts/processing/prepare_hm_metadata.py data/raw

    # Create/update metadata with orientation visualization
    python scripts/processing/prepare_hm_metadata.py data/raw --visualize fig/hm_orientation.png

    # Skip orientation analysis (only write voxel size)
    python scripts/processing/prepare_hm_metadata.py data/raw --no-orientation
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from skimage.measure import regionprops

# Import from axonometry library
# Find repo root (contains pyproject.toml)
_root = Path(__file__).resolve().parent
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from axonometry.io import load_volume_with_metadata, resample_to_isotropic


def compute_slice_eccentricity(slice_2d: np.ndarray, min_area: int = 10) -> Tuple[float, int]:
    """
    Compute average eccentricity of axons in a 2D slice.

    Lower eccentricity indicates more circular cross-sections,
    suggesting the slice is perpendicular to fiber direction.

    Args:
        slice_2d: 2D labeled slice
        min_area: Minimum area to consider a region as an axon (filter noise)

    Returns:
        Tuple of (mean_eccentricity, n_axons)
        Returns (1.0, 0) if no valid regions found
    """
    regions = regionprops(slice_2d.astype(np.int32))

    if len(regions) == 0:
        return 1.0, 0

    # Filter by minimum area to exclude noise
    eccentricities = [r.eccentricity for r in regions if r.area >= min_area]

    if len(eccentricities) == 0:
        return 1.0, 0

    return np.mean(eccentricities), len(eccentricities)


def analyze_volume_orientation(
    volume: np.ndarray,
    min_area: int = 10
) -> Dict[int, Dict[str, float]]:
    """
    Analyze axon orientation by computing eccentricity along each axis.

    Args:
        volume: 3D labeled volume (Z, Y, X)
        min_area: Minimum area for valid axons

    Returns:
        Dictionary mapping axis -> {'eccentricity': float, 'n_axons': int}
        Axis with LOWEST eccentricity is the dominant axis (most circular)
    """
    results = {}
    axis_names = ['Z', 'Y', 'X']

    for axis in range(3):
        # Extract central slice along this axis
        if axis == 0:
            central_idx = volume.shape[0] // 2
            slice_2d = volume[central_idx, :, :]
        elif axis == 1:
            central_idx = volume.shape[1] // 2
            slice_2d = volume[:, central_idx, :]
        else:  # axis == 2
            central_idx = volume.shape[2] // 2
            slice_2d = volume[:, :, central_idx]

        # Compute eccentricity
        mean_ecc, n_axons = compute_slice_eccentricity(slice_2d, min_area)

        results[axis] = {
            'eccentricity': mean_ecc,
            'n_axons': n_axons,
            'axis_name': axis_names[axis],
            'central_index': central_idx
        }

    return results


def determine_dominant_axis(orientation_results: Dict[int, Dict[str, float]]) -> int:
    """
    Determine dominant axis (where axons are most circular).

    Args:
        orientation_results: Results from analyze_volume_orientation()

    Returns:
        Axis index (0, 1, or 2) with lowest eccentricity
    """
    # Find axis with minimum eccentricity (most circular)
    min_ecc = float('inf')
    dominant_axis = 0

    for axis, results in orientation_results.items():
        if results['eccentricity'] < min_ecc:
            min_ecc = results['eccentricity']
            dominant_axis = axis

    return dominant_axis


def create_metadata_file(
    mat_file: Path,
    analyze_orientation: bool = True,
    min_area: int = 10
) -> Dict:
    """
    Create or update a .json metadata file for an HM .mat volume.

    Args:
        mat_file: Path to the .mat file
        analyze_orientation: If True, analyze axon orientation and add to metadata
        min_area: Minimum axon area for orientation analysis

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

    # Create base metadata for HM data
    metadata = {
        "voxel_size": [0.015, 0.015, 0.05],  # Anisotropic: X, Y, Z in micrometers
        "unit": "micrometer"
    }

    # Preserve any additional fields from existing metadata
    for key, value in existing_metadata.items():
        if key not in metadata:
            metadata[key] = value

    # Add orientation analysis if requested
    if analyze_orientation:
        print(f"Analyzing orientation for {mat_file.name}...")

        try:
            # Load and resample volume
            volume, voxel_size_tuple, _ = load_volume_with_metadata(mat_file)
            volume, iso_voxel_size = resample_to_isotropic(volume, voxel_size_tuple)

            # Analyze orientation
            orientation_results = analyze_volume_orientation(volume, min_area=min_area)

            # Determine dominant axis
            dominant_axis = determine_dominant_axis(orientation_results)

            # Add to metadata (just the dominant axis integer)
            metadata['dominant_axis'] = int(dominant_axis)

            print(f"  Dominant axis: {dominant_axis} ({orientation_results[dominant_axis]['axis_name']}) "
                  f"with eccentricity {orientation_results[dominant_axis]['eccentricity']:.3f}")

        except Exception as e:
            print(f"  Warning: Failed to analyze orientation: {e}")
            print(f"  Continuing with basic metadata only...")

    # Write JSON file
    with open(json_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Created/updated {json_file.relative_to(json_file.parents[2])}")
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Create metadata JSON files for HM (high magnification) raw data volumes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create metadata only (no visualization)
  python prepare_hm_metadata.py data/raw

  # Create metadata with orientation visualization
  python prepare_hm_metadata.py data/raw --visualize fig/hm_orientation.png

  # Skip orientation analysis (only write voxel size)
  python prepare_hm_metadata.py data/raw --no-orientation

  # Analyze orientation with custom minimum area threshold
  python prepare_hm_metadata.py data/raw --min-area 20
        """
    )
    parser.add_argument(
        "raw_data_dir",
        type=Path,
        help="Root directory containing raw data (e.g., data/raw)"
    )
    parser.add_argument(
        "--no-orientation",
        action="store_true",
        help="Skip orientation analysis (only write voxel size)"
    )
    parser.add_argument(
        "--visualize",
        type=Path,
        default=None,
        help="Create orientation visualization and save to this file (e.g., fig/hm_orientation.png)"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=10,
        help="Minimum axon area in pixels for orientation analysis (default: 10)"
    )

    args = parser.parse_args()

    # Find all HM .mat files
    all_mat_files = sorted(args.raw_data_dir.rglob("HM_*.mat"))

    if not all_mat_files:
        print(f"No HM_*.mat files found in {args.raw_data_dir}")
        return

    # Filter to only myelinated_axons files for orientation analysis
    myelinated_files = [f for f in all_mat_files if '_myelinated_axons' in f.stem]

    print(f"Found {len(all_mat_files)} HM volume files")
    print(f"  - {len(myelinated_files)} myelinated_axons files (for orientation analysis)")
    print()

    # Process myelinated files with orientation analysis
    all_metadata = []
    dominant_axis_map = {}  # Map from directory to dominant_axis

    for mat_file in myelinated_files:
        metadata = create_metadata_file(
            mat_file,
            analyze_orientation=not args.no_orientation,
            min_area=args.min_area
        )
        all_metadata.append((mat_file, metadata))

        # Store dominant_axis for this directory
        if 'dominant_axis' in metadata:
            dominant_axis_map[mat_file.parent] = metadata['dominant_axis']

    # Propagate dominant_axis to sibling files in the same directory
    non_myelinated_files = [f for f in all_mat_files if '_myelinated_axons' not in f.stem]

    if non_myelinated_files:
        print(f"\nPropagating dominant_axis to {len(non_myelinated_files)} sibling files...")
        for mat_file in non_myelinated_files:
            if mat_file.parent in dominant_axis_map:
                # Create/update metadata with the dominant_axis from the myelinated file
                json_file = mat_file.with_suffix(".json")

                # Load existing metadata if present
                existing_metadata = {}
                if json_file.exists():
                    try:
                        with open(json_file, 'r') as f:
                            existing_metadata = json.load(f)
                    except:
                        pass

                # Create base metadata
                metadata = {
                    "voxel_size": [0.015, 0.015, 0.05],
                    "unit": "micrometer"
                }

                # Preserve existing fields
                for key, value in existing_metadata.items():
                    if key not in metadata:
                        metadata[key] = value

                # Add dominant_axis from myelinated sibling
                metadata['dominant_axis'] = dominant_axis_map[mat_file.parent]

                # Write JSON file
                with open(json_file, "w") as f:
                    json.dump(metadata, f, indent=2)

                print(f"  {mat_file.name} -> dominant_axis={metadata['dominant_axis']}")

    print()
    print(f"Done! Created/updated {len(all_metadata) + len(non_myelinated_files)} metadata files")

    # Create visualization if requested
    if args.visualize and all_metadata:
        print(f"\nCreating orientation visualization...")
        try:
            from examples.visualize_axon_orientation import create_combined_visualization

            # Load volumes and prepare data (need to reanalyze for visualization)
            all_results = []
            for mat_file, metadata in all_metadata:
                if 'dominant_axis' in metadata:
                    # Load volume
                    volume, voxel_size_tuple, _ = load_volume_with_metadata(mat_file)
                    volume, _ = resample_to_isotropic(volume, voxel_size_tuple)

                    # Reanalyze orientation for visualization
                    from examples.visualize_axon_orientation import analyze_volume_orientation
                    orientation_results = analyze_volume_orientation(volume, min_area=args.min_area)

                    sample_name = mat_file.stem.replace('_myelinated_axons', '')
                    all_results.append((volume, orientation_results, sample_name))

            if all_results:
                # Create output directory
                args.visualize.parent.mkdir(parents=True, exist_ok=True)

                # Create visualization
                create_combined_visualization(all_results, args.visualize, downsample_size=512)
                print(f"Saved orientation visualization to {args.visualize}")
            else:
                print("No orientation data available for visualization")

        except Exception as e:
            print(f"Failed to create visualization: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
