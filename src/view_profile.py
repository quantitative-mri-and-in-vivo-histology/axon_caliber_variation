#!/usr/bin/env python3
"""
View cProfile results in a readable format.

Usage:
    python view_profile.py profile_results.prof
"""

import pstats
import sys
from pathlib import Path

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python view_profile.py <profile_file.prof>")
        sys.exit(1)

    profile_file = Path(sys.argv[1])

    if not profile_file.exists():
        print(f"Error: {profile_file} not found")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"Profile Results: {profile_file.name}")
    print(f"{'='*80}\n")

    # Load profile
    stats = pstats.Stats(str(profile_file))

    # Sort by cumulative time and print top 30
    print("\n" + "="*80)
    print("Top 30 functions by cumulative time:")
    print("="*80)
    stats.sort_stats('cumulative').print_stats(30)

    # Sort by total time and print top 30
    print("\n" + "="*80)
    print("Top 30 functions by total time:")
    print("="*80)
    stats.sort_stats('tottime').print_stats(30)

    # Print callers for the slowest function
    print("\n" + "="*80)
    print("Callers of slowest functions:")
    print("="*80)
    stats.sort_stats('tottime').print_callers(10)
