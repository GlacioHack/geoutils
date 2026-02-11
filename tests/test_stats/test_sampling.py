"""Test sampling statistical tools."""

from __future__ import annotations

import warnings
import numpy as np
import pytest

import geoutils as gu
from geoutils._typing import NDArrayNum
from geoutils.stats.sampling import _dask_subsample, _subsample_numpy

class TestSampling:
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

    @pytest.mark.parametrize("array", [array1D, array2D, array3D])
    def test_subsample(self, array: NDArrayNum) -> None:
        """
        Test gu.stats.subsample_array.
        """
        # Test that subsample > 1 works as expected, i.e. output 1D array, with no masked values, or selected size
        for npts in np.arange(2, np.size(array)):
            random_values = gu.stats.subsample_array(array, subsample=npts)
            assert np.ndim(random_values) == 1
            assert np.size(random_values) == npts
            assert np.count_nonzero(random_values.mask) == 0

        # Test if subsample > number of valid values => return all
        random_values = _subsample_numpy(array, subsample=np.size(array) + 3)
        assert np.all(np.sort(random_values) == array[~array.mask])

        # Test if subsample = 1 => return all valid values
        random_values = _subsample_numpy(array, subsample=1)
        assert np.all(np.sort(random_values) == array[~array.mask])

        # Check that order is preserved for subsample = 1 (no random sampling, simply returns valid mask)
        random_values_2 = _subsample_numpy(array, subsample=1)
        assert np.array_equal(random_values, random_values_2)

        # Test if subsample < 1
        random_values = _subsample_numpy(array, subsample=0.5)
        assert np.size(random_values) == int(np.count_nonzero(~array.mask) * 0.5)

        # Test with optional argument return_indices
        indices = _subsample_numpy(array, subsample=0.3, return_indices=True)
        assert np.ndim(indices) == 2
        assert len(indices) == np.ndim(array)
        assert np.ndim(array[indices]) == 1
        assert np.size(array[indices]) == int(np.count_nonzero(~array.mask) * 0.3)

        # Check that we can pass an integer to fix the random state
        sub42 = _subsample_numpy(array, subsample=10, random_state=42)
        # Check by passing a generator directly
        rng = np.random.default_rng(42)
        sub42_gen = _subsample_numpy(array, subsample=10, random_state=rng)
        # Both should be equal
        assert np.array_equal(sub42, sub42_gen)


class TestDask:
    """
    Testing class for delayed functions.

    We test on a first set of rasters big enough to clearly monitor the memory usage, and a second set small enough
    to run fast to check a wide range of input parameters.

    We compare outputs with the in-memory function specifically for input variables that influence the delayed
    algorithm and might lead to new errors (for example: array shape to get subsample/points locations for
    subsample and interp_points, or destination chunksizes to map output of reproject).
    """

    pytest.importorskip("dask")
    import dask.array as da

    # Define random seed for generating test data
    rng = da.random.default_rng(seed=42)

    # Smaller test files for fast checks, with various shapes and with/without nodata
    list_small_shapes = [(51, 47)]
    with_nodata = [False, True]
    list_small_darr = []
    for small_shape in list_small_shapes:
        for w in with_nodata:
            small_darr = rng.normal(size=small_shape[0] * small_shape[1])
            # Add about half nodata values
            if w:
                ind_nodata = rng.choice(small_darr.size, size=int(small_darr.size / 2), replace=False)
                small_darr[list(ind_nodata)] = np.nan
            small_darr = small_darr.reshape(small_shape[0], small_shape[1])
            list_small_darr.append(small_darr)

    # List of in-memory chunksize for small tests
    list_small_chunksizes_in_mem = [(10, 10), (7, 19)]

    # Create a corresponding boolean array for each numerical dask array
    # Every finite numerical value (valid numerical value) corresponds to True (valid boolean value).
    darr_bool = []
    for small_darr in list_small_darr:
        darr_bool.append(da.where(da.isfinite(small_darr), True, False))

    @pytest.mark.parametrize("darr, darr_bool", list(zip(list_small_darr, darr_bool)))
    @pytest.mark.parametrize("chunksizes_in_mem", list_small_chunksizes_in_mem)
    @pytest.mark.parametrize("subsample_size", [2, 100, 100000])
    def test_dask_subsample__output(
        self, darr: da.Array, darr_bool: da.Array, chunksizes_in_mem: tuple[int, int], subsample_size: int
    ) -> None:
        """
        Checks for delayed subsampling function for output accuracy.
        Variables that influence specifically the delayed function and might lead to new errors are:
        - Input chunksizes,
        - Input array shape,
        - Number of subsampled points.
        """

        import dask.array as da

        warnings.filterwarnings("ignore", category=UserWarning, message="Subsample value*")

        # 1/ We run the delayed function after re-chunking
        darr = darr.rechunk(chunksizes_in_mem)
        sub = _dask_subsample(darr, subsample=subsample_size, random_state=42)

        # 2/ Output checks

        # # The subsample should have exactly the prescribed length, with only valid values
        assert issubclass(sub, da.Array)
        sub.compute()
        assert len(sub) == min(subsample_size, np.count_nonzero(np.isfinite(darr)))
        assert all(np.isfinite(sub))

        # To verify the sampling works correctly, we can get its subsample indices with the argument return_indices
        # And compare to the same subsample with vindex (now that we know the coordinates of valid values sampled)
        indices = _dask_subsample(darr, subsample=subsample_size, random_state=42, return_indices=True)
        assert isinstance(indices, tuple)
        assert issubclass(indices[0], da.Array) and issubclass(indices[1], da.Array)
        sub2 = np.array(darr.vindex[indices[0].compute(), indices[1].compute()])
        assert np.array_equal(sub, sub2)

        # Finally, to verify that a boolean array, with valid values at the same locations as the numerical array,
        # leads to the same results, we compare the samples values and the samples indices.
        darr_bool = darr_bool.rechunk(chunksizes_in_mem)
        indices_bool = _dask_subsample(darr_bool, subsample=subsample_size, random_state=42, return_indices=True)
        assert isinstance(indices, tuple)
        assert issubclass(indices[0], da.Array) and issubclass(indices[1], da.Array)
        indices_bool = (indices_bool[0].compute(), indices_bool[1].compute())
        sub_bool = np.array(darr.vindex[indices_bool])
        assert np.array_equal(sub, sub_bool)
        assert np.array_equal(indices, indices_bool)

