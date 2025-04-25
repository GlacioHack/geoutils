"""Functions to test the filtering tools."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import geoutils as gu
from geoutils._typing import NDArrayNum
from geoutils.raster import get_array_and_mask


class TestFilters:
    """Test cases for the filter functions."""

    # Load example data.
    landsat_dem = gu.Raster(gu.examples.get_path("everest_landsat_b4")).astype(np.float32)
    aster_dem_path = gu.examples.get_path("exploradores_aster_dem")

    def test_gauss(self) -> None:
        """Test applying the various Gaussian filters on rasters with/without NaNs"""

        # Test applying scipy's Gaussian filter
        # smoothing should not yield values below.above original DEM
        raster_array = get_array_and_mask(self.landsat_dem)[0]
        raster_sm = gu.filters.gaussian_filter(raster_array, sigma=5)
        assert np.min(raster_array) <= np.min(raster_sm)
        assert np.max(raster_array) >= np.max(raster_sm)
        assert raster_array.shape == raster_sm.shape

        # Test that it works with NaNs too
        nan_count = 1000
        rng = np.random.default_rng(42)
        cols = rng.integers(0, high=self.landsat_dem.width - 1, size=nan_count, dtype=int)
        rows = rng.integers(0, high=self.landsat_dem.height - 1, size=nan_count, dtype=int)
        raster_with_nans = np.copy(self.landsat_dem.data).squeeze()
        raster_with_nans[rows, cols] = np.nan

        raster_sm = gu.filters.gaussian_filter(raster_with_nans, sigma=10)
        assert np.nanmin(raster_with_nans) <= np.min(raster_sm)
        assert np.nanmax(raster_with_nans) >= np.max(raster_sm)

        # Test that it works with 3D arrays
        array_3d = np.vstack((raster_array[np.newaxis, :], raster_array[np.newaxis, :]))
        raster_sm = gu.filters.gaussian_filter(array_3d, sigma=5)
        assert array_3d.shape == raster_sm.shape

        # Tests that it fails with 1D arrays with appropriate error
        data = raster_array[:, 0]
        pytest.raises(ValueError, gu.filters.gaussian_filter, data, sigma=5)

    def test_dist_filter(self) -> None:
        """Test that distance_filter works"""

        # Calculate dDEM
        ddem = self.landsat_dem.copy()

        # Add random outliers
        count = 1000
        rng = np.random.default_rng(42)
        cols = rng.integers(0, high=self.landsat_dem.width - 1, size=count, dtype=int)
        rows = rng.integers(0, high=self.landsat_dem.height - 1, size=count, dtype=int)
        ddem.data[rows, cols] = 5000

        # Filter gross outliers
        filtered_ddem = gu.filters.distance_filter(ddem.data, radius=20, outlier_threshold=50)

        # Check that all outliers were properly filtered
        assert np.all(np.isnan(filtered_ddem[rows, cols]))

        # Assert that non filtered pixels remain the same
        assert ddem.data.shape == filtered_ddem.shape
        assert np.all(ddem.data[np.isfinite(filtered_ddem)] == filtered_ddem[np.isfinite(filtered_ddem)])

        # Check that it works with NaNs too
        ddem.data[rows[:500], cols[:500]] = np.nan
        filtered_ddem = gu.filters.distance_filter(ddem.data, radius=20, outlier_threshold=50)
        assert np.all(np.isnan(filtered_ddem[rows, cols]))

    @pytest.mark.parametrize(
        "name, filter_func",
        [
            ("median", lambda arr: gu.filters.median_filter(arr, **{"size": 5})),  # type:ignore
            ("mean", lambda arr: gu.filters.mean_filter(arr, kernel_size=5)),  # type:ignore
            ("min", lambda arr: gu.filters.min_filter(arr, **{"size": 5})),  # type:ignore
            ("max", lambda arr: gu.filters.max_filter(arr, **{"size": 5})),  # type:ignore
        ],
    )
    def test_filters(self, name: str, filter_func: Callable[[NDArrayNum], NDArrayNum]) -> None:
        """Test that all the filters applied on rasters with/without NaNs, work"""
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

        # Test that it works with NaNs too
        nan_count = 1000
        rng = np.random.default_rng(42)
        cols = rng.integers(0, high=self.landsat_dem.width - 1, size=nan_count, dtype=int)
        rows = rng.integers(0, high=self.landsat_dem.height - 1, size=nan_count, dtype=int)
        raster_with_nans = np.copy(raster_array).squeeze()
        raster_with_nans[rows, cols] = np.nan

        raster_with_nans_filtered = filter_func(raster_with_nans)
        if name in ("median", "mean"):
            # smoothing should not yield values below.above original DEM
            assert np.nanmin(raster_with_nans) <= np.nanmin(raster_with_nans_filtered)
            # assert np.nanmax(raster_with_nans) >= np.nanmax(raster_with_nans_filtered)
            assert np.min(raster_filtered) == np.nanmin(raster_with_nans_filtered)
            # assert np.max(raster_filtered) == np.nanmax(raster_with_nans_filtered)
        elif name == "min":
            assert np.nanmin(raster_with_nans) == np.nanmin(raster_with_nans_filtered)
            assert np.min(raster_filtered) == np.nanmin(raster_with_nans_filtered)
            assert np.nanmax(raster_with_nans) >= np.nanmax(raster_with_nans_filtered)
        elif name == "max":
            assert np.nanmin(raster_with_nans) <= np.nanmin(raster_with_nans_filtered)
            assert np.nanmax(raster_with_nans) == np.nanmax(raster_with_nans_filtered)
            assert np.max(raster_filtered) == np.nanmax(raster_with_nans_filtered)

        # Test that it works with 3D arrays
        if name != "mean":
            array_3d = np.vstack((raster_array[np.newaxis, :], raster_array[np.newaxis, :]))
            raster_filtered = filter_func(array_3d)
            assert array_3d.shape == raster_filtered.shape

            # Tests that it fails with 1D arrays with appropriate error
            data = raster_array[:, 0]
            pytest.raises(ValueError, filter_func, data)

    def test_generic_filter(self) -> None:
        """Test that the generic filter applied on rasters works"""

        raster_array = get_array_and_mask(self.landsat_dem)[0]
        raster_filtered = gu.filters.generic_filter(raster_array, np.nanmin, **{"size": 5})  # type:ignore

        assert np.nansum(raster_array) != np.nansum(raster_filtered)

    @pytest.mark.parametrize(  # type: ignore
        "method, kwargs",
        [
            ("gaussian", {"sigma": 1}),
            ("median", {"size": 3}),
            ("mean", {"kernel_size": 3}),
            ("max", {"size": 3}),
            ("min", {"size": 3}),
        ],
    )
    def test_raster_filter(self, method: str, kwargs: dict[str, int]) -> None:
        raster = gu.Raster(self.aster_dem_path)
        filtered_raster = raster.filter(method, inplace=False, **kwargs)
        assert isinstance(filtered_raster, gu.Raster)
        assert filtered_raster.shape == raster.shape

    def test_raster_filter_callable(self) -> None:
        def double_filter(arr: NDArrayNum) -> NDArrayNum:
            return arr * 2

        raster = gu.Raster(self.aster_dem_path)
        filtered = raster.filter(double_filter, inplace=False)
        expected_raster = raster.copy()
        expected_raster.data *= 2
        assert filtered.raster_equal(expected_raster)

    def test_raster_filter_inplace(self) -> None:
        raster = gu.Raster(self.aster_dem_path)
        filtered_raster = raster.copy()
        filtered_raster.filter("gaussian", sigma=0, inplace=True)
        assert raster.raster_equal(filtered_raster)

    def test_raster_filter_invalid(self) -> None:
        raster = gu.Raster(self.aster_dem_path)
        with pytest.raises(ValueError, match="Unsupported filter method"):
            raster.filter("unknown_filter")
        with pytest.raises(TypeError, match="`method` must be a string or a callable"):
            raster.filter(12345, inplace=False)
