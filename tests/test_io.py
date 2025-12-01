"""Tests for axonometry.io module."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from numpy.testing import assert_array_almost_equal, assert_allclose
import h5py
import scipy.io as sio

from axonometry.io import (
    parse_voxel_size,
    load_mat_volume,
    resample_to_isotropic,
)


class TestParseVoxelSize:
    """Tests for parse_voxel_size function."""

    def test_scalar_isotropic(self):
        """Scalar value should give isotropic (v, v, v) tuple."""
        result = parse_voxel_size(0.05)
        assert result == (0.05, 0.05, 0.05)

    def test_scalar_int(self):
        """Integer scalar should work."""
        result = parse_voxel_size(1)
        assert result == (1.0, 1.0, 1.0)

    def test_tuple_anisotropic(self):
        """Tuple should be returned as-is."""
        result = parse_voxel_size((0.05, 0.015, 0.015))
        assert result == (0.05, 0.015, 0.015)

    def test_list_anisotropic(self):
        """List should work like tuple."""
        result = parse_voxel_size([0.05, 0.015, 0.015])
        assert result == (0.05, 0.015, 0.015)

    def test_single_element_list(self):
        """Single element list should give isotropic."""
        result = parse_voxel_size([0.05])
        assert result == (0.05, 0.05, 0.05)

    def test_invalid_length_raises(self):
        """Invalid tuple length should raise ValueError."""
        with pytest.raises(ValueError, match="Expected 1 or 3"):
            parse_voxel_size((0.05, 0.05))  # 2 elements

        with pytest.raises(ValueError, match="Expected 1 or 3"):
            parse_voxel_size((0.05, 0.05, 0.05, 0.05))  # 4 elements


class TestLoadMatVolume:
    """Tests for load_mat_volume function."""

    def test_load_hdf5_format(self):
        """Load HDF5 format .mat file (v7.3)."""
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Create HDF5 format .mat file
            test_volume = np.random.randint(0, 100, size=(10, 20, 30), dtype=np.uint16)
            with h5py.File(str(temp_path), 'w') as hf:
                hf.create_dataset('volume', data=test_volume)

            # Load and verify
            loaded = load_mat_volume(temp_path)
            assert_array_almost_equal(loaded, test_volume)
        finally:
            temp_path.unlink()

    def test_load_scipy_format(self):
        """Load scipy format .mat file (v5/v7)."""
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Create scipy format .mat file
            test_volume = np.random.randint(0, 100, size=(10, 20, 30), dtype=np.uint16)
            sio.savemat(str(temp_path), {'volume': test_volume})

            # Load and verify
            loaded = load_mat_volume(temp_path)
            assert_array_almost_equal(loaded, test_volume)
        finally:
            temp_path.unlink()

    def test_file_not_found_raises(self):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_mat_volume(Path('/nonexistent/path/file.mat'))

    def test_empty_file_raises(self):
        """Empty .mat file should raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Create empty HDF5 file
            with h5py.File(str(temp_path), 'w') as hf:
                pass  # Empty file

            with pytest.raises(ValueError, match="No data found"):
                load_mat_volume(temp_path)
        finally:
            temp_path.unlink()

    def test_string_path_works(self):
        """String path should work same as Path object."""
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
            temp_path = f.name  # String, not Path

        try:
            test_volume = np.random.randint(0, 100, size=(5, 5, 5), dtype=np.uint16)
            with h5py.File(temp_path, 'w') as hf:
                hf.create_dataset('data', data=test_volume)

            loaded = load_mat_volume(temp_path)
            assert_array_almost_equal(loaded, test_volume)
        finally:
            Path(temp_path).unlink()


class TestResampleToIsotropic:
    """Tests for resample_to_isotropic function."""

    def test_already_isotropic_no_change(self):
        """Isotropic input should return unchanged."""
        volume = np.random.randint(0, 10, size=(10, 20, 30), dtype=np.uint16)
        voxel_size = (0.05, 0.05, 0.05)

        result, iso_size = resample_to_isotropic(volume, voxel_size)

        assert_array_almost_equal(result, volume)
        assert iso_size == 0.05

    def test_downsample_to_coarsest(self):
        """Should downsample to coarsest dimension."""
        # Volume with fine resolution in Y,X and coarse in Z
        volume = np.random.randint(0, 10, size=(100, 100, 100), dtype=np.uint16)
        voxel_size = (0.05, 0.015, 0.015)  # Z is 3.33x coarser

        result, iso_size = resample_to_isotropic(volume, voxel_size)

        # Should downsample to 0.05 (the coarsest)
        assert iso_size == 0.05

        # New shape should be smaller in Y and X dimensions
        # Original: 100 voxels at 0.015 = 1.5 physical units
        # Resampled: 1.5 / 0.05 = 30 voxels
        expected_shape_yx = int(100 * 0.015 / 0.05)  # 30
        assert result.shape[1] == expected_shape_yx
        assert result.shape[2] == expected_shape_yx
        assert result.shape[0] == 100  # Z unchanged

    def test_preserves_labels(self):
        """Nearest neighbor should preserve label integrity."""
        volume = np.zeros((20, 20, 20), dtype=np.uint16)
        volume[5:15, 5:15, 5:15] = 1
        volume[8:12, 8:12, 8:12] = 2

        voxel_size = (0.05, 0.025, 0.025)

        result, iso_size = resample_to_isotropic(volume, voxel_size)

        # Labels should only be 0, 1, 2 (no interpolation artifacts)
        unique_labels = np.unique(result)
        assert set(unique_labels).issubset({0, 1, 2})

    def test_returns_correct_voxel_size(self):
        """Should return the target isotropic voxel size."""
        volume = np.ones((10, 10, 10), dtype=np.uint16)
        voxel_size = (0.1, 0.05, 0.02)

        result, iso_size = resample_to_isotropic(volume, voxel_size)

        assert iso_size == 0.1  # Maximum of input sizes
