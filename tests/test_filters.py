from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import scipy
from scipy.ndimage import (
    gaussian_filter,
    maximum_filter,
    median_filter,
    minimum_filter,
    uniform_filter,
)

import geoutils as gu
from geoutils._typing import NDArrayNum
from geoutils.raster import get_array_and_mask


class TestGaussianFilter:
    """Tests for the Gaussian filter applied to raster data."""

    landsat_data = gu.Raster(gu.examples.get_path("everest_landsat_b4")).astype(np.float32)

    def test_gauss(self) -> None:
        """Test Gaussian filter on 2D and 3D arrays, including handling of NaNs and invalid input."""
        raster_array = get_array_and_mask(self.landsat_data)[0]
        raster_sm = gu.filters.gaussian_filter(raster_array, sigma=5)
        assert np.min(raster_array) <= np.min(raster_sm)
        assert np.max(raster_array) >= np.max(raster_sm)
        assert raster_array.shape == raster_sm.shape

        # Test with NaNs
        nan_count = 1000
        rng = np.random.default_rng(42)
        cols = rng.integers(0, high=self.landsat_data.width - 1, size=nan_count)
        rows = rng.integers(0, high=self.landsat_data.height - 1, size=nan_count)
        raster_with_nans = np.copy(self.landsat_data.data).squeeze()
        raster_with_nans[rows, cols] = np.nan

        raster_sm = gu.filters.gaussian_filter(raster_with_nans, sigma=10)
        assert np.nanmin(raster_with_nans) <= np.nanmin(raster_sm)
        assert np.nanmax(raster_with_nans) >= np.nanmax(raster_sm)

        # 3D arrays
        array_3d = np.vstack((raster_array[np.newaxis, :], raster_array[np.newaxis, :]))
        raster_sm = gu.filters.gaussian_filter(array_3d, sigma=5)
        assert array_3d.shape == raster_sm.shape

        # 1D array should raise
        data = raster_array[:, 0]
        pytest.raises(ValueError, gu.filters.gaussian_filter, data, sigma=5)


class TestStatisticalFilters:
    """Tests for statistical filters: mean, median, min, max."""

    landsat_data = gu.Raster(gu.examples.get_path("everest_landsat_b4")).astype(np.float32)

    @pytest.mark.parametrize(  # type: ignore
        "name, filter_func",
        [
            ("median", lambda arr: gu.filters.median_filter(arr, size=5)),
            ("mean", lambda arr: gu.filters.mean_filter(arr, size=5)),
            ("min", lambda arr: gu.filters.min_filter(arr, size=5)),
            ("max", lambda arr: gu.filters.max_filter(arr, size=5)),
        ],
    )
    def test_filters(self, name: str, filter_func: Callable[[NDArrayNum], NDArrayNum]) -> None:
        """Generic test for statistical filters with and without NaNs, including 3D and invalid 1D input."""
        raster_array = get_array_and_mask(self.landsat_data)[0]
        raster_filtered = filter_func(raster_array)

        rtol = 1e-5
        atol = 1e-5

        min_a, max_a = np.min(raster_array), np.max(raster_array)
        min_f, max_f = np.min(raster_filtered), np.max(raster_filtered)

        if name in ("median", "mean"):
            assert min_a <= min_f or np.isclose(min_a, min_f, rtol=rtol, atol=atol)
            assert max_a >= max_f or np.isclose(max_a, max_f, rtol=rtol, atol=atol)
        elif name == "min":
            assert np.isclose(min_a, min_f, rtol=rtol, atol=atol)
            assert max_a >= max_f or np.isclose(max_a, max_f, rtol=rtol, atol=atol)
        elif name == "max":
            assert min_a <= min_f or np.isclose(min_a, min_f, rtol=rtol, atol=atol)
            assert np.isclose(max_a, max_f, rtol=rtol, atol=atol)

        assert raster_array.shape == raster_filtered.shape

        # Test with NaNs
        nan_count = 1000
        rng = np.random.default_rng(42)
        cols = rng.integers(0, high=self.landsat_data.width - 1, size=nan_count)
        rows = rng.integers(0, high=self.landsat_data.height - 1, size=nan_count)
        raster_with_nans = np.copy(raster_array).squeeze()
        raster_with_nans[rows, cols] = np.nan

        raster_with_nans_filtered = filter_func(raster_with_nans)
        if name in ("median", "mean"):
            assert np.nanmin(raster_with_nans) <= np.nanmin(raster_with_nans_filtered)
            assert np.min(raster_filtered) == np.nanmin(raster_with_nans_filtered)
        elif name == "min":
            assert np.nanmin(raster_with_nans) == np.nanmin(raster_with_nans_filtered)
            assert np.min(raster_filtered) == np.nanmin(raster_with_nans_filtered)
            assert np.nanmax(raster_with_nans) >= np.nanmax(raster_with_nans_filtered)
        elif name == "max":
            assert np.nanmin(raster_with_nans) <= np.nanmin(raster_with_nans_filtered)
            assert np.nanmax(raster_with_nans) == np.nanmax(raster_with_nans_filtered)
            assert np.max(raster_filtered) == np.nanmax(raster_with_nans_filtered)

        if name != "mean":
            array_3d = np.vstack((raster_array[np.newaxis, :], raster_array[np.newaxis, :]))
            raster_filtered = filter_func(array_3d)
            assert array_3d.shape == raster_filtered.shape
            data = raster_array[:, 0]
            pytest.raises(ValueError, filter_func, data)

    def test_median_filter_nan_consistency(self) -> None:
        """Test that different median filter engines return consistent results with NaNs."""
        arr = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]], dtype=np.float32)
        filtered_scipy = gu.filters.median_filter(arr, size=3, engine="scipy")
        filtered_numba = gu.filters.median_filter(arr, size=3, engine="numba")

        assert filtered_scipy.shape == arr.shape
        assert filtered_numba.shape == arr.shape
        assert np.allclose(filtered_scipy, filtered_numba, equal_nan=True)

    def test_median_filter_even_window_size_raises(self) -> None:
        """Ensure median filter raises with even window size."""
        arr = np.random.rand(10, 10).astype(np.float32)
        with pytest.raises(ValueError):
            gu.filters.median_filter(arr, size=4, engine="scipy")

    def test_min_max_filter_all_nans(self) -> None:
        """Test that min/max filters on all-NaN arrays return NaN arrays."""
        arr = np.full((5, 5), np.nan)
        filtered_min = gu.filters.min_filter(arr, size=3)
        filtered_max = gu.filters.max_filter(arr, size=3)
        assert np.all(np.isnan(filtered_min))
        assert np.all(np.isnan(filtered_max))


class TestDistanceFilter:
    """Tests for the distance-based outlier filter."""

    landsat_data = gu.Raster(gu.examples.get_path("everest_landsat_b4")).astype(np.float32)

    def test_dist_filter(self) -> None:
        """Check that distance filter removes outliers and preserves non-outliers."""
        landsat_data = self.landsat_data.copy()

        count = 1000
        rng = np.random.default_rng(42)
        cols = rng.integers(0, high=self.landsat_data.width - 1, size=count)
        rows = rng.integers(0, high=self.landsat_data.height - 1, size=count)
        landsat_data.data[rows, cols] = 5000

        filtered_landsat_data = gu.filters.distance_filter(landsat_data.data, radius=20, outlier_threshold=50)
        assert np.all(np.isnan(filtered_landsat_data[rows, cols]))
        assert landsat_data.data.shape == filtered_landsat_data.shape
        assert np.all(
            landsat_data.data[np.isfinite(filtered_landsat_data)]
            == filtered_landsat_data[np.isfinite(filtered_landsat_data)]
        )

        landsat_data.data[rows[:500], cols[:500]] = np.nan
        filtered_landsat_data = gu.filters.distance_filter(landsat_data.data, radius=20, outlier_threshold=50)
        assert np.all(np.isnan(filtered_landsat_data[rows, cols]))

    def test_distance_filter_all_nans(self) -> None:
        """Distance filter should return NaNs if all input is NaNs."""
        arr = np.full((10, 10), np.nan)
        filtered = gu.filters.distance_filter(arr, radius=2, outlier_threshold=1)
        assert np.all(np.isnan(filtered))

    def test_distance_filter_no_outliers(self) -> None:
        """Ensure no changes occur when no outliers are present."""
        arr = np.ones((10, 10)) * 10
        filtered = gu.filters.distance_filter(arr, radius=2, outlier_threshold=5)
        np.testing.assert_array_equal(arr, filtered)


class TestGenericFilter:
    """Tests for the generic filter API in GeoUtils."""

    landsat_data = gu.Raster(gu.examples.get_path("everest_landsat_b4")).astype(np.float32)

    def test_generic_filter(self) -> None:
        """Apply a scipy filter using the generic filter interface."""
        raster_array = get_array_and_mask(self.landsat_data)[0]
        raster_filtered = gu.filters.generic_filter(raster_array, scipy.ndimage.minimum_filter, size=5)
        scipy_filtered = scipy.ndimage.minimum_filter(raster_array, size=5)
        np.testing.assert_array_equal(raster_filtered, scipy_filtered)

    def test_generic_filter_1d_input_raises(self) -> None:
        """Generic filter should raise on 1D input."""
        arr = np.arange(10)
        with pytest.raises(ValueError):
            gu.filters.generic_filter(arr, scipy.ndimage.gaussian_filter, sigma=1)

    def test_filter_with_custom_callable(self) -> None:
        """Test using a custom function as filter method."""
        arr = np.arange(9).reshape(3, 3).astype(np.float32)

        def double(arr: NDArrayNum) -> NDArrayNum:
            return arr * 2

        filtered = gu.filters._filter(arr, method=double)
        np.testing.assert_array_equal(filtered, double(arr))

    def test_filter_with_invalid_method_type_raises(self) -> None:
        """Passing an invalid method type should raise a TypeError."""
        arr = np.arange(9).reshape(3, 3).astype(np.float32)
        with pytest.raises(ValueError):
            gu.filters._filter(arr, method="1234")


class TestRasterFilters:  # type: ignore
    """Tests for applying filters directly on `Raster` objects."""

    aster_dem_path = gu.examples.get_path("exploradores_aster_dem")

    @pytest.mark.parametrize(
        "method, kwargs, func",
        [
            ("gaussian", {"sigma": 1}, gaussian_filter),
            ("max", {"size": 3}, maximum_filter),
            ("min", {"size": 3}, minimum_filter),
        ],
    )  # type: ignore
    def test_raster_filter_against_reference(self, method: str, kwargs: dict[str, int], func) -> None:
        """Test raster filter results against SciPy reference implementation."""
        raster = gu.Raster(self.aster_dem_path)

        filtered_raster = raster.filter(method, inplace=False, **kwargs)
        expected = np.ma.masked_array(func(np.ma.filled(raster.data, raster.nodata), **kwargs), mask=raster.data.mask)
        # Compare results
        assert isinstance(filtered_raster, gu.Raster)
        assert filtered_raster.shape == raster.shape
        np.testing.assert_allclose(filtered_raster.data, expected, rtol=1e-1, atol=1e-1)

    def test_raster_filter_median_against_reference(self) -> None:
        """Test raster filter results against SciPy reference implementation."""
        raster = gu.Raster(self.aster_dem_path)

        filtered_raster = raster.filter("median", inplace=False, **{"size": 3})
        expected = np.ma.masked_array(
            median_filter(np.ma.filled(raster.data, raster.nodata), **{"size": 3}), mask=raster.data.mask
        )
        # Compare results
        assert isinstance(filtered_raster, gu.Raster)
        assert filtered_raster.shape == raster.shape
        mask = ~np.ma.getmaskarray(filtered_raster.data)
        np.testing.assert_allclose(filtered_raster.data[mask], expected[mask], rtol=1e-2, atol=1e-2)

    def test_raster_filter_mean_against_reference(self) -> None:
        """Test raster filter results against SciPy reference implementation."""
        raster = gu.Raster(self.aster_dem_path)

        filtered_raster = raster.filter("median", inplace=False, **{"size": 3})
        expected = np.ma.masked_array(
            uniform_filter(np.ma.filled(raster.data, raster.nodata), **{"size": 3}), mask=raster.data.mask
        )
        # Compare results
        assert isinstance(filtered_raster, gu.Raster)
        assert filtered_raster.shape == raster.shape
        mask = ~np.ma.getmaskarray(filtered_raster.data)
        np.testing.assert_allclose(filtered_raster.data[mask], expected[mask], rtol=1e-2, atol=1e-2)

    def test_raster_filter_callable(self) -> None:
        """Apply a custom callable as a filter on a Raster object."""

        def double_filter(arr: NDArrayNum) -> NDArrayNum:
            return arr * 2

        raster = gu.Raster(self.aster_dem_path)
        filtered = raster.filter(double_filter, inplace=False)
        expected_raster = raster.copy()
        expected_raster.data *= 2
        np.testing.assert_allclose(
            np.round(filtered.data.data, 2),
            np.round(expected_raster.data.data, 2),
            rtol=1e-1,
            atol=1e-1,
        )

    def test_raster_filter_inplace(self) -> None:
        """Check that in-place filtering modifies the original raster."""
        raster = gu.Raster(self.aster_dem_path)
        filtered_raster = raster.copy()
        filtered_raster.filter("gaussian", sigma=0, inplace=True)
        np.testing.assert_allclose(filtered_raster.data.data, raster.data.data, rtol=1e-1, atol=1e-1)

    def test_raster_filter_invalid(self) -> None:
        """Ensure invalid filter method raises appropriate exceptions."""
        raster = gu.Raster(self.aster_dem_path)
        with pytest.raises(ValueError, match="Unsupported filter method"):
            raster.filter("unknown_filter")
        with pytest.raises(TypeError, match="`method` must be a string or a callable"):
            raster.filter(12345, inplace=False)


class TestNaNConsistency:

    @pytest.mark.parametrize(  # type: ignore
        "method, func, kwargs",
        [
            (
                "median",
                np.median,
                {"size": 3},
            ),
            (
                "mean",
                np.nanmean,
                {"size": 3},
            ),
            (
                "max",
                np.nanmax,
                {"size": 3},
            ),
            (
                "min",
                np.nanmin,
                {"size": 3},
            ),
        ],
    )
    def test_nan_data(self, method, func, kwargs):

        rng = np.random.default_rng(42)
        arr = rng.normal(size=(3, 3))
        filtered_arr = gu.filters._filter(arr, method=method, **kwargs)
        # Central value of the filter should be the same as the filter on the entire array
        assert np.isclose(filtered_arr[1, 1], func(arr), rtol=1e-08, atol=1e-08)
