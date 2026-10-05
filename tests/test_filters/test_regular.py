"""Test filters on raster grids and arrays."""

from __future__ import annotations

from collections.abc import Callable
from importlib.util import find_spec
from typing import Any, Literal

import numpy as np
import pytest
import rasterio as rio
import scipy
import xarray as xr

import geoutils as gu
from geoutils import Raster
from geoutils._misc import import_optional
from geoutils._typing import NDArrayNum
from geoutils.multiproc import MultiprocConfig
from geoutils.operators import GridNeighbours, LinearCoefficients, LocalData, Reducer
from geoutils.operators.reducer import Count, Maximum, Mean, Median, Minimum, Range, RootMeanSquare, Sum
from geoutils.raster import get_array_and_mask
from tests.operator_helpers import SourceIndexSum


class TestRasterFilters:
    """Test module for Raster.filter() with a callable or an invalid method."""

    aster_dem_path = gu.examples.get_path("exploradores_aster_dem")

    def test_raster_filter__callable(self) -> None:
        """Checks that Raster.filter() applies a custom callable to each raster value."""

        def double_filter(arr: NDArrayNum) -> NDArrayNum:
            return arr * 2

        data = np.array([[1, 1, 1, 1, 1], [2, 2, np.nan, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, np.nan, 4], [5, 5, 5, 5, 5]])
        transform = (30.0, 0.0, 478000.0, 0.0, -30.0, 3108140.0)
        raster = Raster.from_array(data, transform, 32645, np.nan)
        filtered = raster.filter(double_filter)
        expected_raster = raster.copy()
        expected_raster.data *= 2
        np.testing.assert_allclose(filtered.data.data, expected_raster.data.data)

    def test_raster_filter__kernel_shape(self) -> None:
        """Checks that a named mean filter accepts circular option."""

        values = np.zeros((5, 5), dtype=float)
        values[1, 1] = values[1, 3] = values[3, 1] = values[3, 3] = 10
        values[1, 2], values[2, 1], values[2, 2], values[2, 3], values[3, 2] = 1, 2, 3, 4, 5
        raster = Raster.from_array(values, rio.transform.from_origin(0, 5, 1, 1), crs=32631)

        # Compare the center with the two NumPy window means
        circular = raster.filter("mean", size=3, kernel_shape="circular")
        square = raster.filter("mean", size=3, kernel_shape="square")
        cross = np.array([values[1, 2], values[2, 1], values[2, 2], values[2, 3], values[3, 2]])
        np.testing.assert_allclose(circular.data[2, 2], np.mean(cross))
        np.testing.assert_allclose(square.data[2, 2], np.mean(values[1:4, 1:4]))

    def test_raster_filter__invalid(self) -> None:
        """Checks that Raster.filter() rejects an unknown name and a value that is not callable."""
        raster = gu.Raster(self.aster_dem_path)
        with pytest.raises(ValueError, match="Unsupported filter method"):
            raster.filter("unknown_filter")
        with pytest.raises(TypeError, match="`method` must be a string or a callable"):
            raster.filter(12345)


class TestSyntheticsNansFilters:  # type: ignore
    """Test module for window filters that calculate around missing cells."""

    array = np.array([[1, 1, 1, 1, 1], [2, 2, np.nan, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, np.nan, 4], [5, 5, 5, 5, 5]])

    def test_min_filter_with_nan(self) -> None:
        """Checks that min_filter() ignores nearby NaNs but leaves missing center cells as NaN."""
        gt = np.array([[1, 1, 1, 1, 1], [1, 1, np.nan, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, np.nan, 3], [4, 4, 4, 4, 4]])
        test = gu.filters.min_filter(self.array, size=3)

        np.testing.assert_array_equal(gt, test)

    def test_max_filter_with_nan(self) -> None:
        """Checks that max_filter() ignores nearby NaNs but leaves missing center cells as NaN."""

        gt = np.array([[2, 2, 2, 2, 2], [3, 3, np.nan, 3, 3], [4, 4, 4, 4, 4], [5, 5, 5, np.nan, 5], [5, 5, 5, 5, 5]])
        test = gu.filters.max_filter(self.array, size=3)

        np.testing.assert_array_equal(gt, test)

    def test_mean_filter_with_nan(self) -> None:
        """Checks that mean_filter() averages finite neighbors and leaves missing center cells as NaN."""

        gt = np.array(
            [
                [1.5, 1.4, 1.4, 1.4, 1.5],
                [2, 2, np.nan, 2, 2],
                [3, 3.125, 3, 3.0, 2.8],
                [4, 4, 4, np.nan, 4],
                [4.5, 4.5, 4.6, 4.6, 4.66666667],
            ]
        )
        test = gu.filters.mean_filter(self.array, size=3)

        np.testing.assert_allclose(gt, test)

    def test_median_filter_with_nan(self) -> None:
        """Checks that median_filter() uses finite neighbors and leaves missing center cells as NaN."""

        gt = np.array(
            [
                [1.5, 1, 1, 1, 1.5],
                [2, 2, np.nan, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, np.nan, 4],
                [4.5, 4.5, 5, 5, 5],
            ]
        )
        test = gu.filters.median_filter(self.array, size=3)

        np.testing.assert_allclose(gt, test)

    def test_gaussian_filter_with_nan(self) -> None:
        """Checks that gaussian_filter() normalizes weights around missing values in two sample windows."""

        array = np.array(
            [
                [0.0, 2.0, 4.0, 6.0, 8.0],
                [10.0, 12.0, np.nan, 16.0, 18.0],
                [20.0, 22.0, 24.0, 26.0, 28.0],
                [30.0, 32.0, 34.0, np.nan, 38.0],
                [40.0, 42.0, 44.0, 46.0, 48.0],
            ]
        )

        test = gu.filters.gaussian_filter(array, sigma=1)

        # Reference at (2, 2): omit two missing neighbors
        window = np.array([[12, np.nan, 16], [22, 24, 26], [32, 34, np.nan]])
        kernel = np.array([[1, np.nan, 1], [2, 4, 2], [1, 2, np.nan]])
        approx_gt = np.nansum(window * kernel) / np.nansum(kernel)
        np.testing.assert_allclose(approx_gt, test[2, 2], rtol=1e-1)

        # Reference at (3, 1): all nine neighbors
        window = np.array(
            [
                [20.0, 22.0, 24],
                [
                    30.0,
                    32.0,
                    34.0,
                ],
                [40.0, 42.0, 44.0],
            ]
        )
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        approx_gt = np.nansum(window * kernel) / np.nansum(kernel)
        np.testing.assert_allclose(approx_gt, test[3, 1], rtol=1e-1)


class TestGaussianFilter:
    """Test module for Gaussian smoothing of 2D, stacked, and partially missing raster data."""

    landsat_data = gu.Raster(gu.examples.get_path("everest_landsat_b4")).astype(np.float32)

    def test_gauss(self) -> None:
        """Checks that Gaussian smoothing limits extremes, handles NaNs and bands, and rejects 1D input."""

        # Check shape and bounded range after smoothing
        raster_array = get_array_and_mask(self.landsat_data)[0]
        raster_sm = gu.filters.gaussian_filter(raster_array, sigma=5)
        assert np.min(raster_array) <= np.min(raster_sm)
        assert np.max(raster_array) >= np.max(raster_sm)
        assert raster_array.shape == raster_sm.shape

        # Check output range with missing pixels
        nan_count = 1000
        rng = np.random.default_rng(42)
        cols = rng.integers(0, high=self.landsat_data.width - 1, size=nan_count)
        rows = rng.integers(0, high=self.landsat_data.height - 1, size=nan_count)
        raster_with_nans = np.copy(self.landsat_data.data).squeeze()
        raster_with_nans[rows, cols] = np.nan

        raster_sm = gu.filters.gaussian_filter(raster_with_nans, sigma=10)
        assert np.nanmin(raster_with_nans) <= np.nanmin(raster_sm)
        assert np.nanmax(raster_with_nans) >= np.nanmax(raster_sm)

        # Compare joint and separate band filtering
        array_3d = np.stack((raster_array, raster_array + 100))
        raster_sm = gu.filters.gaussian_filter(array_3d, sigma=5)
        expected = np.stack([gu.filters.gaussian_filter(band, sigma=5) for band in array_3d])
        np.testing.assert_allclose(raster_sm, expected)

        # Check error for 1D input
        data = raster_array[:, 0]
        pytest.raises(ValueError, gu.filters.gaussian_filter, data, sigma=5)


class TestStatisticalFilters:
    """Test module for statistical filters with finite and missing values.

    Engine comparisons and edge cases are covered in TestRasterReducerEngines and TestRasterFilterEdgeCases.
    """

    landsat_data = gu.Raster(gu.examples.get_path("everest_landsat_b4")).astype(np.float32)

    @pytest.mark.parametrize(
        "name, filter_func",
        [
            ("median", lambda arr: gu.filters.median_filter(arr, size=5)),
            ("mean", lambda arr: gu.filters.mean_filter(arr, size=5)),
            ("min", lambda arr: gu.filters.min_filter(arr, size=5)),
            ("max", lambda arr: gu.filters.max_filter(arr, size=5)),
        ],
    )
    def test_filters(self, name: str, filter_func: Callable[[NDArrayNum], NDArrayNum]) -> None:
        """Checks that each statistic handles finite and missing values, with band checks where supported."""

        # Check shape and expected extrema for each statistic
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

        # Check extrema with missing pixels
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
            # Check independent bands and 1D input error
            array_3d = np.stack((raster_array, raster_array + 100))
            raster_filtered = filter_func(array_3d)
            expected = np.stack([filter_func(band) for band in array_3d])
            np.testing.assert_allclose(raster_filtered, expected)
            data = raster_array[:, 0]
            pytest.raises(ValueError, filter_func, data)


class TestPatchFilters:
    """Test module for convolution and mean filtering with stacks of raster patches."""

    def test_mean_filter__integer_values(self) -> None:
        """Checks that integer inputs are averaged without truncation or overflow, including partial edge windows."""

        # Window sums exceed uint8's range even though each value fits
        values = np.full((7, 7), 250, dtype=np.uint8)
        result = gu.filters.mean_filter(values, size=5)

        # A constant field has the same mean in every nonempty window
        np.testing.assert_allclose(result, values, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("shape", [(3, 3), (4, 4), (3, 4)])
    def test_stacked_convolution_and_kernel_orientation(self, shape: tuple[int, int]) -> None:
        """Checks that stacked convolution agrees with SciPy for odd, even and asymmetric kernels."""

        # Distinct image/kernel values to check orientation
        images = np.arange(2 * 8 * 9, dtype=float).reshape(2, 8, 9)
        kernels = np.arange(2 * np.prod(shape), dtype=float).reshape(2, *shape)

        # SciPy reference for each image/kernel pair
        expected = np.array(
            [
                [scipy.ndimage.convolve(image, kernel, mode="constant", cval=np.nan) for kernel in kernels]
                for image in images
            ]
        )

        # Check stacked convolution and NaN padding at edges
        actual = gu.filters.convolution(images, kernels)
        np.testing.assert_allclose(actual, expected, equal_nan=True)

        # Compare Numba with SciPy reference
        if find_spec("numba") is not None:
            accelerated = gu.filters.convolution(images, kernels, engine="numba")
            np.testing.assert_allclose(accelerated, expected, equal_nan=True)

    @pytest.mark.parametrize("kernel_shape, expected_count", [("square", 25), ("circular", 9)])
    def test_patch_mean_filter_valid_counts(
        self, kernel_shape: Literal["square", "circular"], expected_count: int
    ) -> None:
        """Checks that patch filtering excludes missing values from the mean and reports finite and total counts."""

        # Constant image with missing center to check finite count
        values = np.ones((11, 11), dtype=float)
        values[5, 5] = np.nan

        # Compute patch means and counts, filling missing center
        mean, counts, kernel_count = gu.filters.mean_filter(
            values,
            5,
            kernel_shape=kernel_shape,
            preserve_nodata=False,
            boundless=False,
            return_counts=True,
        )

        # Check kernel count, one fewer finite pixel and constant mean
        assert kernel_count == expected_count
        assert counts[5, 5] == expected_count - 1
        assert mean[5, 5] == 1

        # Check NaN for incomplete edge window
        assert np.isnan(mean[0, 0])

        # Compare Numba means and counts
        if find_spec("numba") is not None:
            accelerated_mean, accelerated_counts, accelerated_kernel_count = gu.filters.mean_filter(
                values,
                5,
                kernel_shape=kernel_shape,
                engine="numba",
                preserve_nodata=False,
                boundless=False,
                return_counts=True,
            )
            np.testing.assert_allclose(accelerated_mean, mean, equal_nan=True)
            np.testing.assert_allclose(accelerated_counts, counts, equal_nan=True)
            assert accelerated_kernel_count == kernel_count


class TestReducerFilters:
    """
    Test module for filtering with custom reducers.

    Reducer accuracy and fractional area calculations are covered in test_operators/test_reducer.py.
    """

    @pytest.mark.parametrize("shape", ["square", "circular"])
    @pytest.mark.parametrize("fractional", [False, True])
    def test_filter__reducer_window(self, shape: Literal["square", "circular"], fractional: bool) -> None:
        """Checks that filtering with a reducer matches reducing the raster at every cell center."""

        # Two bands expose accidental mixing across the band axis
        values = np.random.default_rng(42).normal(size=(2, 11, 13))
        values[:, 4, 5] = np.nan
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 11, 1, 1), crs=32631, nodata=np.nan)
        neighborhood = GridNeighbours(size=5, shape=shape)
        operator = Mean(neighborhood=neighborhood)
        rows, cols = np.indices(values.shape[-2:])
        points = (cols.ravel() + 0.5, 11 - rows.ravel() - 0.5)

        # Filtering can fill missing centers, as point reduction does by default
        result = raster.filter(operator, coverage=("fractional" if fractional else "center"), preserve_nodata=False)
        expected = np.stack(
            [
                raster.resample_at_points(
                    points, operator, band=band, coverage=("fractional" if fractional else "center"), as_array=True
                ).reshape(values.shape[-2:])
                for band in (1, 2)
            ]
        )

        # Coverage intersections and convolution can change rounding at fractional edges
        np.testing.assert_allclose(result.to_nanarray(), expected, rtol=1e-11, atol=1e-11, equal_nan=True)
        assert operator.default_neighborhood is neighborhood

    def test_filter__window_override_and_output_mask(self) -> None:
        """Checks that explicit sizes override reducer defaults and output masks follow the filter options."""

        # A constant raster separates missing centers, valid counts and incomplete border windows
        values = np.ones((9, 9))
        values[4, 4] = np.nan
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 9, 1, 1), crs=32631, nodata=np.nan)
        neighborhood = GridNeighbours(size=5)
        operator = Count(neighborhood=neighborhood)

        # Override the size without changing the reusable reducer
        result = raster.filter(operator, size=3, preserve_nodata=False, boundless=False).to_nanarray()
        masked = raster.filter(operator).to_nanarray()
        assert operator.default_neighborhood is neighborhood

        # Count eight finite neighbors at the missing center; incomplete border windows stay missing
        assert result[4, 4] == 8
        assert result[2, 2] == 9
        assert np.isnan(result[[0, -1], :]).all()
        assert np.isnan(result[:, [0, -1]]).all()
        assert np.isnan(masked[4, 4])

    def test_filter__custom_reducer_and_median_options(self) -> None:
        """Checks that subclass calculations and nondefault median options are applied to every window."""

        class FirstValue(Mean):
            """Return the first finite value selected by the neighborhood."""

            def coefficients(self, data: LocalData) -> LinearCoefficients:
                """Select the first value in offset order."""
                weights = np.zeros(len(data.values))
                weights[0] = 1
                return LinearCoefficients(weights)

        # Two offsets have an even count, so lower and ordinary medians differ
        values = np.arange(144, dtype=float).reshape(12, 12)
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 12, 1, 1), crs=32631)
        neighborhood = GridNeighbours(((0, 1), (0, 0)))
        first = raster.filter(FirstValue(neighborhood=neighborhood)).to_nanarray()
        lower = raster.filter(Median(method="lower", neighborhood=neighborhood)).to_nanarray()

        # Offset order selects the right-hand value; the lower median selects the current value
        np.testing.assert_array_equal(first[:, :-1], values[:, 1:])
        np.testing.assert_array_equal(lower, values)

    def test_filter__circular_offsets_with_area_coverage(self) -> None:
        """Checks that a custom reducer treats circular offsets as a circle when selecting touched cells."""

        class CustomCount(Count):
            """Count cells through the general reducer evaluation path."""

        # Offsets describe a circle but do not record its shape
        raster = gu.Raster.from_array(np.ones((9, 9)), rio.transform.from_origin(0, 9, 1, 1), crs=32631)
        offsets = GridNeighbours(size=7, shape="circular").offsets
        reducer = CustomCount(neighborhood=GridNeighbours(offsets))

        # A seven-cell circle touches 45 cells; its four corner cells lie outside the circle
        result = raster.filter(reducer, coverage="all_touched")
        assert result.data[4, 4] == 45

    def test_filter__distant_offsets(self) -> None:
        """Checks that distant offsets are evaluated without allocating a mostly empty convolution kernel."""

        # The distant cell is outside every window on this raster
        values = np.arange(100, dtype=float).reshape(10, 10)
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 10, 1, 1), crs=32631, nodata=np.nan)
        neighborhood = GridNeighbours(((0, 0), (10000, 10000)))
        result = raster.filter(Mean(neighborhood=neighborhood)).to_nanarray()
        complete = raster.filter(Mean(neighborhood=neighborhood), boundless=False).to_nanarray()

        # Partial windows contain only their center; no complete window fits
        np.testing.assert_array_equal(result, values)
        assert np.isnan(complete).all()


class TestSieveFilter:
    """Test module for removing small connected regions from categorical rasters."""

    def test_sieve(self) -> None:
        """Checks that sieve() removes one isolated cell but leaves a larger region and masked cells unchanged."""

        # Single-cell and two-cell regions
        array = np.ones((5, 5), dtype=np.uint8)
        array[1, 1] = 2
        array[3, 3:5] = 3
        raster = Raster.from_array(array, transform=rio.transform.from_origin(0, 5, 1, 1), crs=4326)

        # Check replacement below size threshold
        result = raster.sieve(size=2)
        expected = array.copy()
        expected[1, 1] = 1
        assert np.array_equal(result.data, expected)
        assert np.array_equal(raster.data, array)

        # Check original values in excluded cells
        excluded = np.ones(array.shape, dtype=bool)
        excluded[1, 1] = False
        masked_result = raster.sieve(size=2, mask=excluded)
        assert np.array_equal(masked_result.data, array)

    def test_sieve_errors(self) -> None:
        """Checks that sieve() rejects floating values, a zero size, and invalid connectivity."""

        raster = Raster.from_array(
            np.ones((3, 3), dtype=np.float32),
            transform=rio.transform.from_origin(0, 3, 1, 1),
            crs=4326,
        )
        with pytest.raises(ValueError, match="integer or Boolean"):
            raster.sieve(size=2)
        with pytest.raises(ValueError, match="strictly positive integer"):
            raster.astype(np.uint8).sieve(size=0)
        with pytest.raises(ValueError, match="connectivity.*4 or 8"):
            raster.astype(np.uint8).sieve(size=2, connectivity=6)  # type: ignore[arg-type]


class TestDistanceFilter:
    """Test module for distance filtering of outliers and fully missing or constant arrays.

    Gaussian kernels are covered in test_operators/test_reducer.py.
    """

    landsat_data = gu.Raster(gu.examples.get_path("everest_landsat_b4")).astype(np.float32)

    def test_distance_filter(self) -> None:
        """Checks that distance_filter() masks inserted outliers and leaves the other finite cells unchanged."""

        # Insert outliers far above local average
        landsat_data = self.landsat_data.copy()

        count = 1000
        rng = np.random.default_rng(42)
        cols = rng.integers(0, high=self.landsat_data.width - 1, size=count)
        rows = rng.integers(0, high=self.landsat_data.height - 1, size=count)
        landsat_data.data[rows, cols] = 5000

        # Check outlier mask and original values elsewhere
        filtered_landsat_data = gu.filters.distance_filter(landsat_data.data, sigma=20, outlier_threshold=50)
        assert np.all(np.isnan(filtered_landsat_data[rows, cols]))
        assert landsat_data.data.shape == filtered_landsat_data.shape
        assert np.all(
            landsat_data.data[np.isfinite(filtered_landsat_data)]
            == filtered_landsat_data[np.isfinite(filtered_landsat_data)]
        )

        # Check outlier masking with some outliers already NaN
        landsat_data.data[rows[:500], cols[:500]] = np.nan
        filtered_landsat_data = gu.filters.distance_filter(landsat_data.data, sigma=20, outlier_threshold=50)
        assert np.all(np.isnan(filtered_landsat_data[rows, cols]))

    def test_distance_filter_all_nans(self) -> None:
        """Checks that distance_filter() returns NaNs when every input cell is missing."""
        arr = np.full((10, 10), np.nan)
        filtered = gu.filters.distance_filter(arr, sigma=2, outlier_threshold=1)
        assert np.all(np.isnan(filtered))

    def test_distance_filter_no_outliers(self) -> None:
        """Checks that distance_filter() leaves a constant image unchanged."""
        arr = np.ones((10, 10)) * 10
        filtered = gu.filters.distance_filter(arr, sigma=2, outlier_threshold=5)
        np.testing.assert_array_equal(arr, filtered)

    @pytest.mark.parametrize(
        "filter_func, kwargs",
        [
            (gu.filters.distance_filter, {"sigma": 1, "outlier_threshold": 2}),
        ],
    )
    def test_filter_engines_consistent(self, filter_func: Callable[..., NDArrayNum], kwargs: dict[str, Any]) -> None:
        """Checks that the SciPy and Numba engines return consistent results with missing values."""

        pytest.importorskip("numba")

        # Filter same array with SciPy and Numba
        arr = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]], dtype=np.float32)
        filtered_scipy = filter_func(arr, engine="scipy", **kwargs)
        filtered_numba = filter_func(arr, engine="numba", **kwargs)

        # Check equal shapes, values and missing pixels
        assert filtered_scipy.shape == arr.shape
        assert filtered_numba.shape == arr.shape
        np.testing.assert_allclose(filtered_scipy, filtered_numba, equal_nan=True, rtol=1e-6, atol=1e-6)

        # Compare backends on separate bands
        if filter_func is not gu.filters.mean_filter:
            array_3d = np.stack((arr, arr + 10))
            filtered_scipy = filter_func(array_3d, engine="scipy", **kwargs)
            filtered_numba = filter_func(array_3d, engine="numba", **kwargs)
            np.testing.assert_allclose(filtered_scipy, filtered_numba, equal_nan=True, rtol=1e-6, atol=1e-6)


class TestGenericFilter:
    """Test module for generic SciPy filters, custom callables, and invalid inputs."""

    landsat_data = gu.Raster(gu.examples.get_path("everest_landsat_b4")).astype(np.float32)

    def test_generic_filter(self) -> None:
        """Checks that generic_filter() matches SciPy's minimum filter on the same array."""
        raster_array = get_array_and_mask(self.landsat_data)[0]
        raster_filtered = gu.filters.generic_filter(raster_array, scipy.ndimage.minimum_filter, size=5)
        scipy_filtered = scipy.ndimage.minimum_filter(raster_array, size=5)
        np.testing.assert_array_equal(raster_filtered, scipy_filtered)

    def test_generic_filter_1d_input_raises(self) -> None:
        """Checks that generic_filter() rejects a one-dimensional array."""
        arr = np.arange(10)
        with pytest.raises(ValueError):
            gu.filters.generic_filter(arr, scipy.ndimage.gaussian_filter, sigma=1)

    def test_filter_with_custom_callable(self) -> None:
        """Checks that a custom callable doubles every value passed through _filter_base()."""
        arr = np.arange(9).reshape(3, 3).astype(np.float32)

        def double(arr: NDArrayNum) -> NDArrayNum:
            return arr * 2

        filtered = gu.filters._filter_base(arr, method=double)
        np.testing.assert_array_equal(filtered, double(arr))

    def test_filter_with_invalid_method_type_raises(self) -> None:
        """Checks that _filter_base() rejects an unknown filter name with ValueError."""
        arr = np.arange(9).reshape(3, 3).astype(np.float32)
        with pytest.raises(ValueError):
            gu.filters._filter_base(arr, method="1234")


@pytest.mark.skipif(find_spec("dask") is None, reason="Only runs if dask is installed.")
class TestRasterReducerEngines:
    """Test module for SciPy/Numba agreement on whole-cell and fractional raster windows."""

    @pytest.mark.skipif(find_spec("numba") is None, reason="Requires numba")
    @pytest.mark.parametrize(
        "filter_func, kwargs",
        [
            (gu.filters.gaussian_filter, {"sigma": 1}),
            (gu.filters.median_filter, {"size": 3}),
            (gu.filters.mean_filter, {"size": 3}),
            (gu.filters.min_filter, {"size": 3}),
            (gu.filters.max_filter, {"size": 3}),
        ],
    )
    def test_filter_engines_consistent(self, filter_func: Callable[..., NDArrayNum], kwargs: dict[str, Any]) -> None:
        """Checks that the SciPy and Numba engines return consistent results with missing values."""

        import_optional("numba")

        # Filter same array with SciPy and Numba
        arr = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]], dtype=np.float32)
        filtered_scipy = filter_func(arr, engine="scipy", **kwargs)
        filtered_numba = filter_func(arr, engine="numba", **kwargs)

        # Check equal shapes, values and missing pixels
        assert filtered_scipy.shape == arr.shape
        assert filtered_numba.shape == arr.shape
        np.testing.assert_allclose(filtered_scipy, filtered_numba, equal_nan=True, rtol=1e-6, atol=1e-6)

        # Compare backends on separate bands
        if filter_func is not gu.filters.mean_filter:
            array_3d = np.stack((arr, arr + 10))
            filtered_scipy = filter_func(array_3d, engine="scipy", **kwargs)
            filtered_numba = filter_func(array_3d, engine="numba", **kwargs)
            np.testing.assert_allclose(filtered_scipy, filtered_numba, equal_nan=True, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("operator_type", [Mean, Sum, Count, RootMeanSquare, Minimum, Maximum, Range, Median])
    @pytest.mark.parametrize("fractional", [False, True])
    def test_filter__reducer_engines(self, operator_type: type[Reducer], fractional: bool) -> None:
        """Checks that Numba and SciPy apply the same circular reducer window and missing-value rule."""

        from geoutils._misc import import_optional

        import_optional("numba")
        values = np.random.default_rng(42).normal(size=(9, 11))
        values[4, 5] = np.nan
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 9, 1, 1), crs=32631, nodata=np.nan)
        operator = operator_type(neighborhood=GridNeighbours(size=5, shape="circular"))

        # Compare Numba with SciPy; weighted sums may differ by roundoff
        expected = raster.filter(
            operator, coverage=("fractional" if fractional else "center"), engine="scipy"
        ).to_nanarray()
        result = raster.filter(
            operator, coverage=("fractional" if fractional else "center"), engine="numba"
        ).to_nanarray()
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14, equal_nan=True)


class TestRasterFilterEdgeCases:
    """Test module for unsupported raster windows, missing Numba and entirely missing arrays."""

    @pytest.mark.skipif(find_spec("numba") is not None, reason="Only runs if numba is missing.")
    def test_filter_numba__missing_dep(self) -> None:
        """Checks that requesting the Numba engine raises ImportError when Numba is unavailable."""

        arr = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]], dtype=np.float32)

        from geoutils.filters import median_filter

        with pytest.raises(ImportError, match="Optional dependency 'numba' required.*"):
            median_filter(arr, size=3, engine="numba")

    def test_median_filter_even_window_size_raises(self) -> None:
        """Checks that median_filter() rejects an even window size."""
        arr = np.random.rand(10, 10).astype(np.float32)
        with pytest.raises(ValueError):
            gu.filters.median_filter(arr, size=4, engine="scipy")

    def test_min_max_filter_all_nans(self) -> None:
        """Checks that min_filter() and max_filter() return NaNs when every input cell is missing."""
        arr = np.full((5, 5), np.nan)
        filtered_min = gu.filters.min_filter(arr, size=3)
        filtered_max = gu.filters.max_filter(arr, size=3)
        assert np.all(np.isnan(filtered_min))
        assert np.all(np.isnan(filtered_max))


@pytest.mark.skipif(find_spec("dask") is None, reason="Requires Dask")
class TestFilterChunked:
    """Test module for filter values and loading behavior across Dask and multiprocessing tiles.

    Kernel accuracy and SciPy/Numba comparisons are covered in test_operators/test_reducer.py.
    """

    @pytest.mark.parametrize("operator", [Mean(), Sum(), SourceIndexSum()])
    @pytest.mark.parametrize("fractional", [False, True])
    def test_filter__reducer_chunks(self, tmp_path: Any, operator: Reducer, fractional: bool) -> None:
        """Checks that reducer filters preserve lazy inputs and agree across irregular Dask and worker tiles."""

        from geoutils._misc import import_optional
        from geoutils.multiproc.cluster import MpCluster

        da = import_optional("dask").array
        if fractional and isinstance(operator, SourceIndexSum):
            pytest.skip("SourceIndexSum does not accept area weights.")

        # Distinct bands and shorter final chunks reveal band mixing and missing edge windows
        values = np.random.default_rng(42).normal(size=(2, 13, 17))
        values[:, 5, 6] = np.nan
        raster = gu.Raster.from_array(values, rio.transform.from_origin(100, 200, 2, 3), crs=32631, nodata=np.nan)
        path = tmp_path / "reducer-filter.tif"
        raster.to_file(path)
        lazy = gu.open_raster(path, chunks={"band": 1, "x": 7, "y": 5})
        source = gu.Raster(path)
        options = {
            "size": 5,
            "coverage": ("fractional" if fractional else "center"),
            "preserve_nodata": False,
            "boundless": False,
        }

        # Run eager and worker calculations, then build the Dask result without loading its source
        expected = raster.filter(operator, **options).to_nanarray()
        with MpCluster({"nb_workers": 2}) as cluster:
            config = MultiprocConfig(cluster=cluster, chunks=6, outfile=str(tmp_path / "filtered.tif"))
            worker = source.filter(operator, mp_config=config, **options)
        delayed = lazy.rst.filter(operator, **options)
        assert isinstance(delayed.data, da.Array)
        assert not lazy.rst.is_loaded and not delayed.rst.is_loaded
        assert not source.is_loaded and not worker.is_loaded

        # Only floating sums may differ across chunks; cell identities must agree exactly
        tolerance = 0 if isinstance(operator, SourceIndexSum) else 1e-12
        np.testing.assert_allclose(delayed.compute().values, expected, rtol=0, atol=tolerance, equal_nan=True)
        np.testing.assert_allclose(worker.to_nanarray(), expected, rtol=0, atol=tolerance, equal_nan=True)
        assert not lazy.rst.is_loaded and not delayed.rst.is_loaded and not source.is_loaded

    @pytest.mark.parametrize("path_index", [0, 2])
    @pytest.mark.parametrize("method", ["gaussian", "median", "mean", "min", "max"])
    @pytest.mark.parametrize("size", [3, 7])
    @pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
    def test_filter_chunked_backends_equal(
        self,
        tmp_path: Any,
        path_index: int,
        method: str,
        size: int,
        lazy_test_files_tiny: list[str],
    ) -> None:
        """Checks that Dask and worker tiles match eager filters while file-backed Raster objects stay unloaded.

        Gaussian sums can change slightly across tile boundaries, so this case allows an absolute difference of 1e-6.
        The other filters must produce the same values exactly.
        """

        import dask.array as da

        # Shared raster for eager, Dask and MP filters
        path_raster = lazy_test_files_tiny[path_index]

        # 1/ Open test files
        # Load Raster and Xarray references
        raster_base = gu.Raster(path_raster)
        raster_base.load()
        xr_base = gu.open_raster(path_raster)
        xr_base.load()

        # Open lazy Dask/MP inputs (10 x 10 chunks)
        raster_mp = gu.Raster(path_raster)
        mp_config = MultiprocConfig(chunks=10)
        ds = gu.open_raster(path_raster, chunks={"x": 10, "y": 10})
        assert not ds._in_memory
        assert isinstance(ds.data, da.Array)
        assert ds.data.chunks is not None

        # 2/ Run eager filters and build chunked outputs
        # Compute eager references
        base_rst = raster_base.filter(method=method, size=size)
        assert isinstance(base_rst, Raster)
        base_xr = xr_base.rst.filter(method=method, size=size)
        assert isinstance(base_xr, xr.DataArray)

        # Check unloaded MP output and lazy Dask output
        mp_rst = raster_mp.filter(method=method, size=size, mp_config=mp_config)
        assert isinstance(mp_rst, Raster)
        assert not mp_rst.is_loaded
        dask_rst = ds.rst.filter(method=method, size=size)
        assert isinstance(dask_rst, xr.DataArray)
        assert isinstance(dask_rst.data, da.Array)

        # Check inputs still unloaded
        assert not raster_mp.is_loaded
        assert not ds._in_memory
        assert isinstance(ds.data, da.Array)

        # 3/ Compare outputs
        # Allow rounding differences from Gaussian sums across tiles
        if method == "gaussian":
            atol = 1e-6
        else:
            atol = 0.0

        # Compute and compare with eager Raster reference
        dask_rst = dask_rst.compute()
        assert base_rst.raster_allclose(dask_rst, warn_failure_reason=True, strict_masked=False, atol=atol)
        assert base_rst.raster_allclose(mp_rst, warn_failure_reason=True, strict_masked=False, atol=atol)
        assert base_rst.raster_allclose(base_xr, warn_failure_reason=True, strict_masked=False, atol=atol)
