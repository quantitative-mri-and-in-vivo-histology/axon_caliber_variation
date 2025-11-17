#!/usr/bin/env python3
"""
Batch preprocessing script to extract populations from all volumes.
"""

from pathlib import Path
from preprocess_extract_populations import process_volume, FilteringParams
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    input_dir = Path('/workspace/data/raw/LM')
    output_dir = Path('/workspace/data/processed')

    params = FilteringParams(
        min_fraction_present=0.9,
        max_orientation_deviation_deg=30.0,
        voxel_size_um=0.05
    )

    # Find all .mat files
    mat_files = sorted(input_dir.glob('*_myelinated_axons.mat'))

    logger.info(f"Found {len(mat_files)} volumes to process")

    for mat_file in mat_files:
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING: {mat_file.name}")
        logger.info(f"{'='*80}\n")

        try:
            process_volume(mat_file, output_dir, params)
            logger.info(f"✓ Successfully processed {mat_file.name}")
        except Exception as e:
            logger.error(f"✗ Failed to process {mat_file.name}: {e}", exc_info=True)

    logger.info("\n" + "="*80)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
