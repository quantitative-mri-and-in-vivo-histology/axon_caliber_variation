"""Tests for axonometry.geometry module."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_allclose

from axonometry.geometry import (
    unit_tangent_vector,
    rotation_matrix_3d,
    unit_normal_vector,
    angle_between,
    create_perpendicular_plane_grid,
    compute_arc_length,
    resample_curve_by_arc_length,
)


class TestUnitTangentVector:
    """Tests for unit_tangent_vector function."""

    def test_straight_line_z(self):
        """Tangent along straight line in Z direction should be [0, 0, 1]."""
        curve = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]])
        tangents = unit_tangent_vector(curve)

        # All tangents should point in Z direction
        expected = np.array([0, 0, 1])
        for t in tangents:
            assert_array_almost_equal(t, expected)

    def test_straight_line_x(self):
        """Tangent along straight line in X direction should be [1, 0, 0]."""
        curve = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        tangents = unit_tangent_vector(curve)

        expected = np.array([1, 0, 0])
        for t in tangents:
            assert_array_almost_equal(t, expected)

    def test_unit_length(self):
        """All tangent vectors should have unit length."""
        curve = np.array([
            [0, 0, 0],
            [1, 1, 0],
            [2, 1, 1],
            [3, 2, 1],
            [4, 2, 2],
        ])
        tangents = unit_tangent_vector(curve)

        norms = np.linalg.norm(tangents, axis=1)
        assert_allclose(norms, 1.0, rtol=1e-5)

    def test_single_point_raises_or_handles(self):
        """Single point curve behavior (gradient requires at least 2 points)."""
        curve = np.array([[1, 2, 3]])

        # numpy.gradient requires at least 2 points, so this raises
        with pytest.raises(ValueError):
            unit_tangent_vector(curve)

    def test_two_points(self):
        """Two point curve should work."""
        curve = np.array([[0, 0, 0], [1, 0, 0]])
        tangents = unit_tangent_vector(curve)

        expected = np.array([1, 0, 0])
        assert_array_almost_equal(tangents[0], expected)
        assert_array_almost_equal(tangents[1], expected)


class TestRotationMatrix3D:
    """Tests for rotation_matrix_3d function."""

    def test_identity_rotation(self):
        """Zero rotation should give identity matrix."""
        axis = np.array([0, 0, 1])
        R = rotation_matrix_3d(axis, 0)

        assert_array_almost_equal(R, np.eye(3))

    def test_90_degree_z_rotation(self):
        """90 degree rotation about Z axis."""
        axis = np.array([0, 0, 1])
        R = rotation_matrix_3d(axis, np.pi / 2)

        # Should rotate [1,0,0] to [0,1,0]
        v = np.array([1, 0, 0])
        rotated = R @ v
        expected = np.array([0, 1, 0])
        assert_array_almost_equal(rotated, expected, decimal=5)

    def test_180_degree_rotation(self):
        """180 degree rotation about Z axis."""
        axis = np.array([0, 0, 1])
        R = rotation_matrix_3d(axis, np.pi)

        # Should rotate [1,0,0] to [-1,0,0]
        v = np.array([1, 0, 0])
        rotated = R @ v
        expected = np.array([-1, 0, 0])
        assert_array_almost_equal(rotated, expected, decimal=5)

    def test_rotation_preserves_length(self):
        """Rotation should preserve vector length."""
        axis = np.array([1, 1, 1]) / np.sqrt(3)
        theta = 1.23  # arbitrary angle
        R = rotation_matrix_3d(axis, theta)

        v = np.array([3, 4, 5])
        rotated = R @ v

        assert_allclose(np.linalg.norm(rotated), np.linalg.norm(v))

    def test_rotation_matrix_is_orthogonal(self):
        """Rotation matrix should be orthogonal (R @ R.T = I)."""
        axis = np.array([1, 2, 3]) / np.linalg.norm([1, 2, 3])
        theta = 0.7
        R = rotation_matrix_3d(axis, theta)

        assert_array_almost_equal(R @ R.T, np.eye(3))
        assert_allclose(np.linalg.det(R), 1.0)  # det = 1 for proper rotation


class TestUnitNormalVector:
    """Tests for unit_normal_vector function."""

    def test_perpendicular_vectors(self):
        """Cross product of perpendicular vectors."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        n = unit_normal_vector(v1, v2)

        expected = np.array([0, 0, 1])
        assert_array_almost_equal(n, expected)

    def test_unit_length(self):
        """Result should have unit length."""
        v1 = np.array([3, 0, 0])
        v2 = np.array([0, 4, 0])
        n = unit_normal_vector(v1, v2)

        assert_allclose(np.linalg.norm(n), 1.0)

    def test_parallel_vectors_fallback(self):
        """Parallel vectors should return a fallback (not crash)."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([2, 0, 0])  # parallel to v1
        n = unit_normal_vector(v1, v2)

        # Should return something with unit length
        assert_allclose(np.linalg.norm(n), 1.0)


class TestAngleBetween:
    """Tests for angle_between function."""

    def test_same_direction(self):
        """Same direction vectors have zero angle."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([2, 0, 0])

        assert_allclose(angle_between(v1, v2), 0.0, atol=1e-4)

    def test_opposite_direction(self):
        """Opposite direction vectors have pi angle."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([-1, 0, 0])

        assert_allclose(angle_between(v1, v2), np.pi, rtol=1e-4)

    def test_perpendicular(self):
        """Perpendicular vectors have pi/2 angle."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])

        assert_allclose(angle_between(v1, v2), np.pi / 2)

    def test_45_degrees(self):
        """45 degree angle."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([1, 1, 0])

        assert_allclose(angle_between(v1, v2), np.pi / 4, rtol=1e-5)


class TestCreatePerpendicularPlaneGrid:
    """Tests for create_perpendicular_plane_grid function."""

    def test_z_aligned_tangent(self):
        """Grid for Z-aligned tangent should be in XY plane."""
        tangent = np.array([0, 0, 1])
        grid = create_perpendicular_plane_grid(tangent, plane_radius=2.0, plane_resolution=1.0)

        # All Z coordinates should be 0 (plane perpendicular to Z)
        assert_allclose(grid[:, 2], 0, atol=1e-10)

    def test_grid_centered_at_origin(self):
        """Grid should be centered at origin."""
        tangent = np.array([0, 0, 1])
        grid = create_perpendicular_plane_grid(tangent, plane_radius=2.0, plane_resolution=1.0)

        # Mean should be approximately at origin
        assert_allclose(np.mean(grid, axis=0), [0, 0, 0], atol=1e-10)

    def test_grid_size(self):
        """Grid should have expected number of points."""
        tangent = np.array([0, 0, 1])
        radius = 2.0
        resolution = 1.0
        grid = create_perpendicular_plane_grid(tangent, radius, resolution)

        # Grid should span from -radius to +radius
        expected_points_per_dim = int(2 * radius / resolution) + 1  # 5 points: -2,-1,0,1,2
        expected_total = expected_points_per_dim ** 2
        assert grid.shape[0] == expected_total

    def test_arbitrary_tangent(self):
        """Grid for arbitrary tangent should be perpendicular to it."""
        tangent = np.array([1, 1, 1]) / np.sqrt(3)
        grid = create_perpendicular_plane_grid(tangent, plane_radius=2.0, plane_resolution=0.5)

        # All grid points should be perpendicular to tangent (dot product ≈ 0)
        dots = grid @ tangent
        assert_allclose(dots, 0, atol=1e-9)  # Allow small numerical error


class TestComputeArcLength:
    """Tests for compute_arc_length function."""

    def test_straight_line(self):
        """Arc length of straight line should equal Euclidean distance."""
        curve = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        arc = compute_arc_length(curve)

        expected = np.array([0, 1, 2, 3])
        assert_array_almost_equal(arc, expected)

    def test_diagonal_line(self):
        """Arc length of diagonal line."""
        curve = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        arc = compute_arc_length(curve)

        segment_length = np.sqrt(3)
        expected = np.array([0, segment_length, 2 * segment_length])
        assert_array_almost_equal(arc, expected)

    def test_with_voxel_size(self):
        """Arc length should scale with voxel size."""
        curve = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        voxel_size = (0.1, 0.1, 0.1)  # 0.1 μm voxels
        arc = compute_arc_length(curve, voxel_size)

        expected = np.array([0, 0.1, 0.2])
        assert_array_almost_equal(arc, expected)

    def test_anisotropic_voxel_size(self):
        """Arc length with anisotropic voxels."""
        curve = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        voxel_size = (0.05, 0.05, 0.1)  # Z=0.05, Y=0.05, X=0.1
        arc = compute_arc_length(curve, voxel_size)

        # First segment: 1 voxel in Z direction = 0.05 μm
        # Second segment: 1 voxel in Y direction = 0.05 μm
        expected = np.array([0, 0.05, 0.10])
        assert_array_almost_equal(arc, expected)

    def test_single_point(self):
        """Single point should return [0]."""
        curve = np.array([[1, 2, 3]])
        arc = compute_arc_length(curve)

        assert_array_almost_equal(arc, [0])


class TestResampleCurveByArcLength:
    """Tests for resample_curve_by_arc_length function."""

    def test_straight_line_resampling(self):
        """Resampling straight line should give evenly spaced points."""
        curve = np.array([[0, 0, 0], [10, 0, 0]])
        resampled = resample_curve_by_arc_length(curve, step_size=2.0)

        # Should have points at 0, 2, 4, 6, 8
        expected = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0], [8, 0, 0]])
        assert_array_almost_equal(resampled, expected)

    def test_short_curve_returns_original(self):
        """Curve shorter than step_size should return original."""
        curve = np.array([[0, 0, 0], [0.5, 0, 0]])
        resampled = resample_curve_by_arc_length(curve, step_size=2.0)

        assert_array_almost_equal(resampled, curve)

    def test_with_voxel_size(self):
        """Resampling with physical units."""
        # Curve of 10 voxels in X
        curve = np.array([[0, 0, 0], [0, 0, 10]])
        voxel_size = (0.05, 0.05, 0.05)  # 0.05 μm voxels

        # Physical length = 10 * 0.05 = 0.5 μm
        # Step of 0.1 μm should give 5 samples (0, 0.1, 0.2, 0.3, 0.4)
        resampled = resample_curve_by_arc_length(curve, step_size=0.1, voxel_size=voxel_size)

        assert len(resampled) == 5

    def test_preserves_endpoints(self):
        """First point should be preserved."""
        curve = np.array([[1, 2, 3], [11, 2, 3]])
        resampled = resample_curve_by_arc_length(curve, step_size=2.0)

        assert_array_almost_equal(resampled[0], curve[0])
