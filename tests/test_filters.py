from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import scipy

import geoutils as gu
from geoutils._typing import NDArrayNum
from geoutils.raster import get_array_and_mask


class TestGaussianFilter:
    """Tests for the Gaussian filter applied to raster data."""

    landsat_dem = gu.Raster(gu.examples.get_path("everest_landsat_b4")).astype(np.float32)

    def test_gauss(self) -> None:
        """Test Gaussian filter on 2D and 3D arrays, including handling of NaNs and invalid input."""
        raster_array = get_array_and_mask(self.landsat_dem)[0]
        raster_sm = gu.filters.gaussian_filter(raster_array, sigma=5)
        assert np.min(raster_array) <= np.min(raster_sm)
        assert np.max(raster_array) >= np.max(raster_sm)
        assert raster_array.shape == raster_sm.shape

        # Test with NaNs
        nan_count = 1000
        rng = np.random.default_rng(42)
        cols = rng.integers(0, high=self.landsat_dem.width - 1, size=nan_count)
        rows = rng.integers(0, high=self.landsat_dem.height - 1, size=nan_count)
        raster_with_nans = np.copy(self.landsat_dem.data).squeeze()
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

    landsat_dem = gu.Raster(gu.examples.get_path("everest_landsat_b4")).astype(np.float32)

    @pytest.mark.parametrize(  # type: ignore
        "name, filter_func",
        [
            ("median", lambda arr: gu.filters.median_filter(arr, window_size=5)),
            ("mean", lambda arr: gu.filters.mean_filter(arr, kernel_size=5)),
            ("min", lambda arr: gu.filters.min_filter(arr, size=5)),
            ("max", lambda arr: gu.filters.max_filter(arr, size=5)),
        ],
    )
    def test_filters(self, name: str, filter_func: Callable[[NDArrayNum], NDArrayNum]) -> None:
        """Generic test for statistical filters with and without NaNs, including 3D and invalid 1D input."""
        raster_array = get_array_and_mask(self.landsat_dem)[0]
        raster_filtered = filter_func(raster_array)

        if name in ("median", "mean"):
            assert np.min(raster_array) <= np.min(raster_filtered)
            assert np.max(raster_array) >= np.max(raster_filtered)
        elif name == "min":
            assert np.min(raster_array) == np.min(raster_filtered)
            assert np.max(raster_array) >= np.max(raster_filtered)
        elif name == "max":
            assert np.min(raster_array) <= np.min(raster_filtered)
            assert np.max(raster_array) == np.max(raster_filtered)

        assert raster_array.shape == raster_filtered.shape

        # Test with NaNs
        nan_count = 1000
        rng = np.random.default_rng(42)
        cols = rng.integers(0, high=self.landsat_dem.width - 1, size=nan_count)
        rows = rng.integers(0, high=self.landsat_dem.height - 1, size=nan_count)
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
        filtered_scipy = gu.filters.median_filter(arr, window_size=3, engine="scipy")
        filtered_numba = gu.filters.median_filter(arr, window_size=3, engine="numba")

        assert filtered_scipy.shape == arr.shape
        assert filtered_numba.shape == arr.shape
        mask = np.isfinite(filtered_scipy) & np.isfinite(filtered_numba)
        np.testing.assert_allclose(filtered_scipy[mask], filtered_numba[mask], rtol=1e-5)

    def test_median_filter_even_window_size_raises(self) -> None:
        """Ensure median filter raises with even window size."""
        arr = np.random.rand(10, 10).astype(np.float32)
        with pytest.raises(ValueError):
            gu.filters.median_filter(arr, window_size=4, engine="scipy")

    def test_mean_filter_preserves_nans(self) -> None:
        """Test that mean filter maintains NaNs in the output."""
        arr = np.array([[np.nan, 2, 3], [4, 5, np.nan], [7, 8, 9]], dtype=np.float32)
        filtered = gu.filters.mean_filter(arr, kernel_size=3)
        assert np.isnan(filtered[0, 0])
        assert np.isnan(filtered[1, 2])

    def test_min_max_filter_all_nans(self) -> None:
        """Test that min/max filters on all-NaN arrays return NaN arrays."""
        arr = np.full((5, 5), np.nan)
        filtered_min = gu.filters.min_filter(arr, size=3)
        filtered_max = gu.filters.max_filter(arr, size=3)
        assert np.all(np.isnan(filtered_min))
        assert np.all(np.isnan(filtered_max))


class TestDistanceFilter:
    """Tests for the distance-based outlier filter."""

    landsat_dem = gu.Raster(gu.examples.get_path("everest_landsat_b4")).astype(np.float32)

    def test_dist_filter(self) -> None:
        """Check that distance filter removes outliers and preserves non-outliers."""
        ddem = self.landsat_dem.copy()

        count = 1000
        rng = np.random.default_rng(42)
        cols = rng.integers(0, high=self.landsat_dem.width - 1, size=count)
        rows = rng.integers(0, high=self.landsat_dem.height - 1, size=count)
        ddem.data[rows, cols] = 5000

        filtered_ddem = gu.filters.distance_filter(ddem.data, radius=20, outlier_threshold=50)
        assert np.all(np.isnan(filtered_ddem[rows, cols]))
        assert ddem.data.shape == filtered_ddem.shape
        assert np.all(ddem.data[np.isfinite(filtered_ddem)] == filtered_ddem[np.isfinite(filtered_ddem)])

        ddem.data[rows[:500], cols[:500]] = np.nan
        filtered_ddem = gu.filters.distance_filter(ddem.data, radius=20, outlier_threshold=50)
        assert np.all(np.isnan(filtered_ddem[rows, cols]))

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

    landsat_dem = gu.Raster(gu.examples.get_path("everest_landsat_b4")).astype(np.float32)

    def test_generic_filter(self) -> None:
        """Apply a scipy filter using the generic filter interface."""
        raster_array = get_array_and_mask(self.landsat_dem)[0]
        raster_filtered = gu.filters.generic_filter(raster_array, scipy.ndimage.minimum_filter, size=5)
        assert np.nansum(raster_array) != np.nansum(raster_filtered)

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


class TestRasterFilters:
    """Tests for applying filters directly on `Raster` objects."""

    aster_dem_path = gu.examples.get_path("exploradores_aster_dem")

    @pytest.mark.parametrize(  # type: ignore
        "method, kwargs",
        [
            ("gaussian", {"sigma": 1}),
            ("median", {"window_size": 3}),
            ("mean", {"kernel_size": 3}),
            ("max", {"size": 3}),
            ("min", {"size": 3}),
        ],
    )
    def test_raster_filter(self, method: str, kwargs: dict[str, int]) -> None:
        """Test raster filter methods with standard parameters."""
        raster = gu.Raster(self.aster_dem_path)
        filtered_raster = raster.filter(method, inplace=False, **kwargs)
        assert isinstance(filtered_raster, gu.Raster)
        assert filtered_raster.shape == raster.shape

    def test_raster_filter_callable(self) -> None:
        """Apply a custom callable as a filter on a Raster object."""

        def double_filter(arr: NDArrayNum) -> NDArrayNum:
            return arr * 2

        raster = gu.Raster(self.aster_dem_path)
        filtered = raster.filter(double_filter, inplace=False)
        expected_raster = raster.copy()
        expected_raster.data *= 2
        assert filtered.raster_equal(expected_raster)

    def test_raster_filter_inplace(self) -> None:
        """Check that in-place filtering modifies the original raster."""
        raster = gu.Raster(self.aster_dem_path)
        filtered_raster = raster.copy()
        filtered_raster.filter("gaussian", sigma=0, inplace=True)
        assert raster.raster_equal(filtered_raster)

    def test_raster_filter_invalid(self) -> None:
        """Ensure invalid filter method raises appropriate exceptions."""
        raster = gu.Raster(self.aster_dem_path)
        with pytest.raises(ValueError, match="Unsupported filter method"):
            raster.filter("unknown_filter")
        with pytest.raises(TypeError, match="`method` must be a string or a callable"):
            raster.filter(12345, inplace=False)


class TestNaNConsistency:

    @pytest.mark.parametrize(  # type: ignore
        "method, vt, kwargs",
        [
            (
                "gaussian",
                np.array(
                    [
                        [50.44, 44.78840723, 50.00838795, 55.24, 48.55],
                        [45.24, 46.484448, 51.81481846, 53.84, 50.15],
                        [45.64, 46.84580181, 47.86123291, 45.90860692, 42.38],
                        [46.52, 43.42714287, np.nan, 43.89431209, 41.90],
                        [38.77, 39.21, 41.51, 47.77, 45.75],
                    ],
                ),
                {"sigma": 1},
            ),
            (
                "median",
                np.array(
                    [
                        [39, 42, 52.5, 65.5, 79.5],
                        [39, 45, 56, 60, 42.5],
                        [39.5, 50.5, 58, 61.5, 56],
                        [45, 45, 57, 39.5, 39.5],
                        [46, 34, 58, 49, 56],
                    ]
                ),
                {"window_size": 3},
            ),
            (
                "mean",
                np.array(
                    [
                        [50.33, 44.66, 51.11, 55.44, 52.77],
                        [43.11, 48.11, 52.22, 51.44, 43.44],
                        [48.11, np.nan, np.nan, np.nan, 48.88],
                        [44.44, np.nan, np.nan, np.nan, 37.33],
                        [41.33, np.nan, np.nan, np.nan, 50.55],
                    ]
                ),
                {"kernel_size": 3},
            ),
            (
                "max",
                np.array(
                    [
                        [78.0, 78.0, 88.0, 90.0, 90.0],
                        [78.0, 82.0, 88.0, 90.0, 90.0],
                        [91.0, 91.0, 82.0, 90.0, 90.0],
                        [91.0, 91.0, np.nan, 87.0, 87.0],
                        [91.0, 91.0, 87.0, 87.0, 87.0],
                    ]
                ),
                {"size": 3},
            ),
            (
                "min",
                np.array(
                    [
                        [15.0, 15.0, 15.0, 12.0, 12.0],
                        [15.0, 15.0, 14.0, 7.0, 7.0],
                        [25.0, 25.0, 14.0, 7.0, 7.0],
                        [10.0, 10.0, np.nan, 7.0, 7.0],
                        [10.0, 10.0, 19.0, 19.0, 30.0],
                    ]
                ),
                {"size": 3},
            ),
        ],
    )
    def test_nan_data(self, method, vt, kwargs):
        array = np.array(
            [
                [78, 15, 39, 88, 12],
                [33, 45, 60, 71, 90],
                [25, 56, 82, 14, 7],
                [91, 34, np.nan, 63, 49],
                [10, 58, 19, 87, 30],
            ]
        )

        arr_filtered = gu.filters._filter(array, method=method, **kwargs)
        np.testing.assert_allclose(vt, arr_filtered, rtol=1e-2)
