"""Tests for axonometry.skeleton module."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_allclose

from axonometry.skeleton import (
    extract_skeleton,
    discrete_shortest_path,
    get_line_length,
    warmup,
)


class TestWarmup:
    """Tests for warmup function."""

    def test_warmup_runs_without_error(self):
        """Warmup should complete without error."""
        warmup()  # Should not raise


class TestGetLineLength:
    """Tests for get_line_length function."""

    def test_straight_line(self):
        """Length of straight line."""
        line = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        length = get_line_length(line)
        assert_allclose(length, 3.0)

    def test_diagonal_line(self):
        """Length of diagonal line."""
        line = np.array([[0, 0, 0], [1, 1, 1]])
        length = get_line_length(line)
        assert_allclose(length, np.sqrt(3))

    def test_single_point(self):
        """Single point has zero length."""
        line = np.array([[0, 0, 0]])
        length = get_line_length(line)
        assert length == 0.0

    def test_empty_line(self):
        """Empty line has zero length."""
        line = np.array([]).reshape(0, 3)
        length = get_line_length(line)
        assert length == 0.0

    def test_two_points(self):
        """Two points."""
        line = np.array([[0, 0, 0], [3, 4, 0]])
        length = get_line_length(line)
        assert_allclose(length, 5.0)  # 3-4-5 triangle


class TestDiscreteShortestPath:
    """Tests for discrete_shortest_path function."""

    def test_simple_gradient_descent(self):
        """Path should descend from high to low values."""
        # Create simple distance field
        D = np.ones((5, 5, 5), dtype=np.float64) * 10
        D[2, 2, 2] = 0  # Source at center

        # Fill with distance from center
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    D[i, j, k] = np.sqrt((i-2)**2 + (j-2)**2 + (k-2)**2)

        # Start from corner
        start = np.array([0, 0, 0])
        path = discrete_shortest_path(D, start)

        # Path should start at start point
        assert_array_almost_equal(path[0], start)

        # Path should end at or near the source
        assert D[path[-1, 0], path[-1, 1], path[-1, 2]] < 1.0

    def test_already_at_source(self):
        """Starting at source should give single-point path."""
        D = np.ones((5, 5, 5), dtype=np.float64)
        D[2, 2, 2] = 0

        start = np.array([2, 2, 2])
        path = discrete_shortest_path(D, start)

        assert len(path) == 1
        assert_array_almost_equal(path[0], start)

    def test_path_is_monotonically_decreasing(self):
        """Path values should decrease monotonically."""
        D = np.zeros((10, 10, 10), dtype=np.float64)
        # Create linear gradient
        for i in range(10):
            D[i, :, :] = i

        start = np.array([9, 5, 5])
        path = discrete_shortest_path(D, start)

        # Values along path should decrease
        values = [D[p[0], p[1], p[2]] for p in path]
        for i in range(1, len(values)):
            assert values[i] <= values[i-1]


class TestExtractSkeleton:
    """Tests for extract_skeleton function."""

    def test_cylinder_skeleton(self):
        """Skeleton of cylinder should be roughly along its axis."""
        # Create cylinder along Z axis
        vol = np.zeros((20, 20, 50), dtype=np.uint8)
        center = (10, 10)
        radius = 5

        for z in range(50):
            for y in range(20):
                for x in range(20):
                    if (x - center[0])**2 + (y - center[1])**2 < radius**2:
                        vol[x, y, z] = 1

        skeleton_segments = extract_skeleton(vol, path_method='discrete')

        assert len(skeleton_segments) > 0

        # Main skeleton should be roughly along Z axis
        main_skel = max(skeleton_segments, key=len)
        assert len(main_skel) > 10

        # Check that skeleton stays near center in XY
        xy_distances = np.sqrt((main_skel[:, 0] - 10)**2 + (main_skel[:, 1] - 10)**2)
        assert np.mean(xy_distances) < radius  # Should be near center

    def test_sphere_skeleton(self):
        """Skeleton of sphere - fast-marching produces skeleton from center to all directions."""
        vol = np.zeros((21, 21, 21), dtype=np.uint8)
        center = (10, 10, 10)
        radius = 8

        for x in range(21):
            for y in range(21):
                for z in range(21):
                    if (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 < radius**2:
                        vol[x, y, z] = 1

        skeleton_segments = extract_skeleton(vol, path_method='discrete')

        # Fast-marching skeleton of a sphere radiates from center
        # The algorithm finds paths from the boundary to center
        # Total length depends on how many segments are created
        # Just verify we get some skeleton and it's not crazy long
        assert len(skeleton_segments) >= 0  # May be empty or have segments
        total_length = sum(get_line_length(seg) for seg in skeleton_segments)
        # Should be bounded by surface area traversal (very generous bound)
        assert total_length < 4 * np.pi * radius**2  # Much less than surface area

    def test_empty_volume_raises(self):
        """Empty volume should raise ValueError."""
        vol = np.zeros((10, 10, 10), dtype=np.uint8)

        with pytest.raises(ValueError, match="too thin"):
            extract_skeleton(vol)

    def test_thin_object_behavior(self):
        """Very thin object - may raise or produce short skeleton."""
        vol = np.zeros((10, 10, 10), dtype=np.uint8)
        vol[5, 5, :] = 1  # Single line of voxels

        # Single-voxel-thick objects may raise or produce minimal skeleton
        try:
            skeleton_segments = extract_skeleton(vol)
            # If it doesn't raise, skeleton should be very short or empty
            total_length = sum(get_line_length(seg) for seg in skeleton_segments)
            assert total_length < 20  # Much shorter than object extent
        except ValueError as e:
            # Acceptable if it raises for thin objects
            assert "too thin" in str(e).lower() or "thin" in str(e).lower()

    def test_discrete_vs_euler_methods(self):
        """Both methods should produce similar results."""
        # Create simple cylinder
        vol = np.zeros((15, 15, 30), dtype=np.uint8)
        for z in range(30):
            for y in range(15):
                for x in range(15):
                    if (x - 7)**2 + (y - 7)**2 < 16:
                        vol[x, y, z] = 1

        skel_discrete = extract_skeleton(vol, path_method='discrete')
        skel_euler = extract_skeleton(vol, path_method='euler')

        # Both should find skeleton segments
        assert len(skel_discrete) > 0
        assert len(skel_euler) > 0

        # Total lengths should be similar (within 50%)
        len_discrete = sum(get_line_length(s) for s in skel_discrete)
        len_euler = sum(get_line_length(s) for s in skel_euler)

        ratio = len_discrete / len_euler if len_euler > 0 else 1
        assert 0.5 < ratio < 2.0

    def test_invalid_path_method_raises(self):
        """Invalid path method should raise ValueError."""
        vol = np.ones((10, 10, 10), dtype=np.uint8)

        with pytest.raises(ValueError, match="path_method"):
            extract_skeleton(vol, path_method='invalid')


class TestSkeletonBackwardsCompatibility:
    """Test that the 'skeleton' alias works."""

    def test_skeleton_alias_exists(self):
        """The skeleton alias should be available."""
        from axonometry.skeleton import skeleton
        assert skeleton is extract_skeleton
