"""Tests for axonometry.sampling module."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_allclose

from axonometry.sampling import (
    nearest_interp_3d,
    sample_perpendicular_cross_section,
    area_to_equivalent_radius,
    validate_skeleton_points,
    find_longest_contiguous_segment,
)


class TestNearestInterp3D:
    """Tests for nearest_interp_3d function."""

    def test_integer_coords(self):
        """Integer coordinates should return exact voxel values."""
        volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
        coords = np.array([[1, 1, 1], [0, 0, 0], [2, 2, 2]], dtype=np.float64).T

        result = nearest_interp_3d(volume, coords)

        assert result[0] == volume[1, 1, 1]
        assert result[1] == volume[0, 0, 0]
        assert result[2] == volume[2, 2, 2]

    def test_fractional_coords_round(self):
        """Fractional coordinates should round to nearest."""
        volume = np.zeros((5, 5, 5), dtype=np.float64)
        volume[2, 2, 2] = 1.0

        # Coords that should round to (2, 2, 2)
        coords = np.array([[2.4, 2.3, 1.6]], dtype=np.float64).T
        result = nearest_interp_3d(volume, coords)

        assert result[0] == 1.0

    def test_out_of_bounds_returns_zero(self):
        """Out of bounds coordinates should return 0."""
        volume = np.ones((5, 5, 5), dtype=np.float64)
        # Use clearly out-of-bounds values (accounting for int(x + 0.5) rounding)
        coords = np.array([
            [-2, 0, 0],  # negative: int(-2 + 0.5) = -1 < 0
            [5, 2, 2],   # too large: int(5 + 0.5) = 5 >= 5
            [2, 2, 10],  # too large
        ], dtype=np.float64).T

        result = nearest_interp_3d(volume, coords)

        # All should be 0 (clearly out of bounds)
        assert result[0] == 0
        assert result[1] == 0
        assert result[2] == 0

    def test_boundary_coords(self):
        """Boundary coordinates should work correctly."""
        volume = np.ones((5, 5, 5), dtype=np.float64)
        volume[0, 0, 0] = 2.0
        volume[4, 4, 4] = 3.0

        coords = np.array([[0, 0, 0], [4, 4, 4]], dtype=np.float64).T
        result = nearest_interp_3d(volume, coords)

        assert result[0] == 2.0
        assert result[1] == 3.0

    def test_large_batch(self):
        """Should handle large batches efficiently."""
        volume = np.random.rand(50, 50, 50).astype(np.float64)
        n_points = 10000
        coords = np.random.rand(3, n_points) * 49  # Random coords in volume

        result = nearest_interp_3d(volume, coords)

        assert result.shape == (n_points,)


class TestSamplePerpendicularCrossSection:
    """Tests for sample_perpendicular_cross_section function."""

    def test_sphere_cross_section(self):
        """Cross-section through sphere center should give circular area."""
        # Create sphere
        volume = np.zeros((21, 21, 21), dtype=np.uint8)
        center = np.array([10, 10, 10])
        radius = 5

        for x in range(21):
            for y in range(21):
                for z in range(21):
                    if (x - 10)**2 + (y - 10)**2 + (z - 10)**2 < radius**2:
                        volume[x, y, z] = 1

        # Sample at center with Z-aligned tangent
        tangent = np.array([0, 0, 1])
        area = sample_perpendicular_cross_section(
            volume, center.astype(float), tangent,
            plane_radius=10, plane_resolution=0.25
        )

        # Expected area ≈ π * r² = π * 25 ≈ 78.5
        expected_area = np.pi * radius**2
        assert area is not None
        assert_allclose(area, expected_area, rtol=0.15)  # Within 15%

    def test_cylinder_cross_section(self):
        """Cross-section of cylinder perpendicular to axis."""
        # Create cylinder along Z
        volume = np.zeros((21, 21, 50), dtype=np.uint8)
        radius = 5

        for z in range(50):
            for y in range(21):
                for x in range(21):
                    if (x - 10)**2 + (y - 10)**2 < radius**2:
                        volume[x, y, z] = 1

        # Sample at midpoint with Z-aligned tangent
        point = np.array([10, 10, 25], dtype=float)
        tangent = np.array([0, 0, 1])

        area = sample_perpendicular_cross_section(
            volume, point, tangent,
            plane_radius=10, plane_resolution=0.25
        )

        expected_area = np.pi * radius**2
        assert area is not None
        assert_allclose(area, expected_area, rtol=0.15)

    def test_empty_cross_section_returns_none(self):
        """Cross-section outside object should return None."""
        volume = np.zeros((21, 21, 21), dtype=np.uint8)
        volume[10, 10, 10] = 1  # Single voxel

        # Sample far from the object
        point = np.array([0, 0, 0], dtype=float)
        tangent = np.array([0, 0, 1])

        area = sample_perpendicular_cross_section(
            volume, point, tangent,
            plane_radius=2, plane_resolution=0.5
        )

        assert area is None or area == 0

    def test_precomputed_volume_float(self):
        """Pre-converted float volume should give same result."""
        volume = np.zeros((15, 15, 15), dtype=np.uint8)
        volume[5:10, 5:10, 5:10] = 1

        point = np.array([7, 7, 7], dtype=float)
        tangent = np.array([0, 0, 1])

        # Without precomputed
        area1 = sample_perpendicular_cross_section(
            volume, point, tangent,
            plane_radius=5, plane_resolution=0.5
        )

        # With precomputed
        volume_float = volume.astype(np.float64)
        area2 = sample_perpendicular_cross_section(
            volume, point, tangent,
            plane_radius=5, plane_resolution=0.5,
            volume_float=volume_float
        )

        assert_allclose(area1, area2)


class TestAreaToEquivalentRadius:
    """Tests for area_to_equivalent_radius function."""

    def test_unit_circle(self):
        """Area of π should give radius 1."""
        area = np.pi
        radius = area_to_equivalent_radius(area)
        assert_allclose(radius, 1.0)

    def test_with_voxel_size(self):
        """Radius should scale with voxel size."""
        area = np.pi  # 1 voxel² circle
        voxel_size = 0.05  # 0.05 μm

        radius = area_to_equivalent_radius(area, voxel_size)
        assert_allclose(radius, 0.05)

    def test_none_area_returns_none(self):
        """None area should return None."""
        assert area_to_equivalent_radius(None) is None

    def test_zero_area_returns_none(self):
        """Zero area should return None."""
        assert area_to_equivalent_radius(0) is None

    def test_negative_area_returns_none(self):
        """Negative area should return None."""
        assert area_to_equivalent_radius(-1) is None

    def test_large_area(self):
        """Large area should give correct radius."""
        radius_expected = 10
        area = np.pi * radius_expected**2

        radius = area_to_equivalent_radius(area)
        assert_allclose(radius, radius_expected)


class TestValidateSkeletonPoints:
    """Tests for validate_skeleton_points function."""

    def test_all_inside(self):
        """Points inside object should be valid."""
        volume = np.zeros((10, 10, 10), dtype=np.uint8)
        volume[3:7, 3:7, 3:7] = 1

        points = np.array([
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
        ], dtype=float)

        valid = validate_skeleton_points(points, volume)
        assert np.all(valid)

    def test_all_outside(self):
        """Points outside object should be invalid."""
        volume = np.zeros((10, 10, 10), dtype=np.uint8)
        volume[5, 5, 5] = 1  # Single voxel

        points = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [9, 9, 9],
        ], dtype=float)

        valid = validate_skeleton_points(points, volume)
        assert not np.any(valid)

    def test_mixed_validity(self):
        """Mix of inside and outside points."""
        volume = np.zeros((10, 10, 10), dtype=np.uint8)
        volume[4:6, 4:6, 4:6] = 1

        points = np.array([
            [5, 5, 5],  # inside
            [0, 0, 0],  # outside
            [4, 4, 4],  # inside
            [9, 9, 9],  # outside
        ], dtype=float)

        valid = validate_skeleton_points(points, volume)
        assert_array_almost_equal(valid, [True, False, True, False])

    def test_out_of_bounds(self):
        """Out of bounds points should be invalid."""
        volume = np.ones((10, 10, 10), dtype=np.uint8)

        points = np.array([
            [-1, 5, 5],   # negative
            [5, 10, 5],   # too large
            [5, 5, 100],  # way too large
        ], dtype=float)

        valid = validate_skeleton_points(points, volume)
        assert not np.any(valid)

    def test_rounding_behavior(self):
        """Fractional coordinates should round to nearest voxel."""
        volume = np.zeros((10, 10, 10), dtype=np.uint8)
        volume[5, 5, 5] = 1

        # Points that round to (5, 5, 5)
        points = np.array([
            [4.6, 5.4, 4.8],
            [5.4, 5.1, 5.3],
        ], dtype=float)

        valid = validate_skeleton_points(points, volume)
        assert np.all(valid)


class TestFindLongestContiguousSegment:
    """Tests for find_longest_contiguous_segment function."""

    def test_all_true(self):
        """All True should return full range."""
        mask = np.array([True, True, True, True, True])
        result = find_longest_contiguous_segment(mask)

        assert result == (0, 5)

    def test_all_false(self):
        """All False should return None."""
        mask = np.array([False, False, False])
        result = find_longest_contiguous_segment(mask)

        assert result is None

    def test_single_true(self):
        """Single True should return None (need at least 2)."""
        mask = np.array([False, True, False])
        result = find_longest_contiguous_segment(mask)

        assert result is None

    def test_gap_in_middle(self):
        """Should find longest segment when there's a gap."""
        mask = np.array([True, True, False, True, True, True, True])
        result = find_longest_contiguous_segment(mask)

        # Longest segment is indices 3-6 (4 elements)
        assert result == (3, 7)

    def test_multiple_gaps(self):
        """Should find longest among multiple segments."""
        mask = np.array([True, True, False, True, False, True, True, True])
        result = find_longest_contiguous_segment(mask)

        # Longest segment is indices 5-7 (3 elements)
        assert result == (5, 8)

    def test_segment_at_start(self):
        """Longest segment at start."""
        mask = np.array([True, True, True, True, False, True, True])
        result = find_longest_contiguous_segment(mask)

        assert result == (0, 4)

    def test_segment_at_end(self):
        """Longest segment at end."""
        mask = np.array([True, True, False, True, True, True, True])
        result = find_longest_contiguous_segment(mask)

        assert result == (3, 7)

    def test_equal_length_segments(self):
        """When segments are equal, should return one of them."""
        mask = np.array([True, True, False, True, True])
        result = find_longest_contiguous_segment(mask)

        # Either (0, 2) or (3, 5) is valid
        assert result in [(0, 2), (3, 5)]
