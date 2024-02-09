"""Test sampling tools for arrays and rasters."""

from __future__ import annotations

import numpy as np
import pytest

import geoutils as gu
from geoutils._typing import NDArrayNum


class TestSubsampling:
    """
    Different examples of 1D to 3D arrays with masked values for testing.
    """

    # Case 1 - 1D array, 1 masked value
    array1D = np.ma.masked_array(np.arange(10), mask=np.zeros(10))
    array1D.mask[3] = True
    assert np.ndim(array1D) == 1
    assert np.count_nonzero(array1D.mask) > 0

    # Case 2 - 2D array, 1 masked value
    array2D = np.ma.masked_array(np.arange(9).reshape((3, 3)), mask=np.zeros((3, 3)))
    array2D.mask[0, 1] = True
    assert np.ndim(array2D) == 2
    assert np.count_nonzero(array2D.mask) > 0

    # Case 3 - 3D array, 1 masked value
    array3D = np.ma.masked_array(np.arange(9).reshape((1, 3, 3)), mask=np.zeros((1, 3, 3)))
    array3D = np.ma.vstack((array3D, array3D + 10))
    array3D.mask[0, 0, 1] = True
    assert np.ndim(array3D) == 3
    assert np.count_nonzero(array3D.mask) > 0

    @pytest.mark.parametrize("array", [array1D, array2D, array3D])  # type: ignore
    def test_subsample(self, array: NDArrayNum) -> None:
        """
        Test gu.raster.subsample_array.
        """
        # Test that subsample > 1 works as expected, i.e. output 1D array, with no masked values, or selected size
        for npts in np.arange(2, np.size(array)):
            random_values = gu.raster.subsample_array(array, subsample=npts)
            assert np.ndim(random_values) == 1
            assert np.size(random_values) == npts
            assert np.count_nonzero(random_values.mask) == 0

        # Test if subsample > number of valid values => return all
        random_values = gu.raster.subsample_array(array, subsample=np.size(array) + 3)
        assert np.all(np.sort(random_values) == array[~array.mask])

        # Test if subsample = 1 => return all valid values
        random_values = gu.raster.subsample_array(array, subsample=1)
        assert np.all(np.sort(random_values) == array[~array.mask])

        # Test if subsample < 1
        random_values = gu.raster.subsample_array(array, subsample=0.5)
        assert np.size(random_values) == int(np.count_nonzero(~array.mask) * 0.5)

        # Test with optional argument return_indices
        indices = gu.raster.subsample_array(array, subsample=0.3, return_indices=True)
        assert np.ndim(indices) == 2
        assert len(indices) == np.ndim(array)
        assert np.ndim(array[indices]) == 1
        assert np.size(array[indices]) == int(np.count_nonzero(~array.mask) * 0.3)

        # Check that we can pass an integer to fix the random state
        sub42 = gu.raster.subsample_array(array, subsample=10, random_state=42)
        # Check by passing a generator directly
        random_gen = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(42)))
        sub42_gen = gu.raster.subsample_array(array, subsample=10, random_state=random_gen)
        # Both should be equal
        assert np.array_equal(sub42, sub42_gen)

    def test_subdivide_array(self) -> None:
        test_shape = (6, 4)
        test_count = 4
        subdivision_grid = gu.raster.subdivide_array(test_shape, test_count)

        assert subdivision_grid.shape == test_shape
        assert np.unique(subdivision_grid).size == test_count

        assert np.unique(gu.raster.subdivide_array((3, 3), 3)).size == 3

        with pytest.raises(ValueError, match=r"Expected a 2D shape, got 1D shape.*"):
            gu.raster.subdivide_array((5,), 2)

        with pytest.raises(ValueError, match=r"Shape.*smaller than.*"):
            gu.raster.subdivide_array((5, 2), 15)
