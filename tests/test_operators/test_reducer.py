"""Test reducer calculations and their treatment of missing values and weights."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
from affine import Affine

import geoutils as gu
from benchmarks.comparisons.gdal import build_gdal_grid_command
from geoutils._misc import import_optional
from geoutils._typing import NDArrayNum
from geoutils.interface.gridding import GriddingMethod, _grid_pointcloud
from geoutils.operators import GridNeighbours, LinearCoefficients, LocalData, PointNeighbours, Reducer
from geoutils.operators.nodata import NodataChoice
from geoutils.operators.reducer import (
    AverageDistance,
    AveragePairwiseDistance,
    Count,
    Maximum,
    Mean,
    Median,
    Minimum,
    Mode,
    Quantile,
    Range,
    RootMeanSquare,
    StandardDeviation,
    Sum,
)
from tests.operator_helpers import SupportMassReducer


# Custom subclass implementing only reduce() for tests
class SumReducer(Reducer):
    def reduce(self, data: LocalData) -> float:
        return float(np.sum(data.values))


def _local_data(
    values: list[float],
    *,
    valid: list[bool] | None = None,
    sample_weights: list[float] | None = None,
    support_weights: list[float] | None = None,
) -> LocalData:
    """Shorten creation of LocalData examples for tests."""

    count = len(values)
    return LocalData(
        values=np.asarray(values),
        valid=np.ones(count, dtype=bool) if valid is None else np.asarray(valid),
        source_ids=np.arange(count),
        sample_weights=None if sample_weights is None else np.asarray(sample_weights),
        support_weights=None if support_weights is None else np.asarray(support_weights),
    )


class TestReducer:
    """
    Test module for custom reduce() methods, neighborhood configuration and unsupported weights.

    Raster and point method accuracy, numerical engines and reference comparisons are covered below.
    API options and chunked execution are in test_filters/, test_interface/test_resampling.py,
    test_interface/test_gridding.py and test_raster/test_transformations_raster.py.
    """

    def test_reducer__custom(self) -> None:
        """Checks custom reduce() method uses only valid values by default."""

        # Invalid finite value to check nodata filtering
        data = _local_data([2.0, 100.0, 5.0], valid=[True, False, True])
        result = SumReducer().evaluate(data)

        # Check sum of valid observations
        assert result == 7.0

    def test_reduce_batch__coefficients(self) -> None:
        """Checks that reduce_batch() applies coefficients."""

        class DifferenceReducer(Reducer):
            def coefficients(self, data: LocalData) -> LinearCoefficients:
                return LinearCoefficients(weights=np.array([-1.0, 1.0]), offset=2.0)

        # Two groups to check order
        first = _local_data([3.0, 8.0])
        second = _local_data([7.0, 2.0])
        reducer = DifferenceReducer()
        result = reducer.reduce_batch([first, second])

        # Check inherited reduction from custom coefficients
        # -1 * 3 + 1 * 8 + 2 = 7
        # -1 * 7 + 1 * 2 + 2 = -3
        np.testing.assert_array_equal(result, [7.0, -3.0])
        assert reducer.evaluate(first) == 7.0

    def test_reducer__neighborhood(self) -> None:
        """Checks reducers properly define neighborhoods."""

        # Using grid neighborhoods on different reducers (custom above, built-in)
        window = GridNeighbours(size=5)
        reducers = [
            SumReducer(neighborhood=window),
            Mean(neighborhood=window),
            Quantile(0.5, neighborhood=window),
            Median(neighborhood=window),
            Mode(neighborhood=window),
        ]

        # Check those are stored properly
        assert all(reducer.default_neighborhood is window for reducer in reducers)

        # Using point neighborhoods on different reducers as well
        nearby = PointNeighbours(k=2, radius=5)
        point_reducers = [
            SumReducer(neighborhood=nearby),
            Mean(neighborhood=nearby),
            Quantile(0.5, neighborhood=nearby),
            Median(neighborhood=nearby),
            Mode(neighborhood=nearby),
        ]
        assert all(reducer.default_neighborhood is nearby for reducer in point_reducers)
        with pytest.raises(TypeError, match="GridNeighbours or PointNeighbours"):
            Mean(neighborhood="invalid")  # type: ignore[arg-type]

    def test_reducer__error(self) -> None:
        """Checks error raised when weights not supported."""

        data = _local_data([1.0, 2.0], sample_weights=[1.0, 1.0])

        with pytest.raises(ValueError, match="does not accept sample_weights"):
            SumReducer().evaluate(data)


class TestReducerStatistics:
    """Test module on our built-in statistical reducers (independent of geometry)."""

    @pytest.mark.parametrize(
        "operator,expected",
        [
            (Mean(), 3),
            (Sum(), 9),
            (Minimum(), 1),
            (Maximum(), 5),
            (Range(), 4),
            (Count(), 3),
            (Median(), 3),
            (Quantile(0.25), 2),
            (Mode(), 5),  # Mode is the largest value in a tie break (all equally frequent)
            (RootMeanSquare(), np.sqrt((1 + 3**2 + 5**2) / 3)),
            (StandardDeviation(), np.sqrt((2**2 + 2**2) / 3)),
            (AverageDistance(), 2),
            (AveragePairwiseDistance(), (2 + 2 + 4) / 3),
        ],
    )
    def test_reducer__known(self, operator: Reducer, expected: float) -> None:
        """Checks built-in reducers are accurate."""

        # Reference statistics: mean 3, population variance 8/3, pairwise distances (2, 4, 2)
        data = LocalData(
            values=np.array([1.0, 3.0, 5.0]),
            valid=np.ones(3, dtype=bool),
            source_ids=np.arange(3),
            coordinates=np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]),
            distances=np.array([0.0, 2.0, 4.0]),
        )
        assert operator.evaluate(data) == pytest.approx(expected)

    def test_reducer__masked(self) -> None:
        """Checks that a reducer does not use masked values (even if finite below the mask)."""

        # Masked finite value with validity flag still set
        values = np.ma.array([1.0, 100.0], mask=[False, True])
        data = LocalData(values=values, valid=np.ones(2, dtype=bool), source_ids=np.arange(2))

        # Check output is finite when ignoring nodata, NaN when propagating
        reducer = Sum()
        ignored = reducer.evaluate(data, nodata_propagation="ignore")
        propagated = reducer.evaluate(data, nodata_propagation="propagate")
        assert ignored == 1.0
        assert np.isnan(propagated)


class TestReducerWeights:
    """Test reducer behaviour with sample weights and support weights (fractional areas)."""

    def test_reducer__weighted_mean(self) -> None:
        """Checks mean properly accounts for sample/support weights."""

        # Synthetic data weights to check product
        data = _local_data(
            [4.0, 10.0],
            sample_weights=[2.0, 1.0],
            support_weights=[1.0, 3.0],
        )
        reducer = Mean()

        # Combined weights: 2 * 1 = 2 and 1 * 3 = 3, normalizing for mean into 2/5 = 0.4 and 3/5 = 0.6
        # Check weighted mean: 4 * 0.4 + 10 * 0.6 = 7.6
        coefficients = reducer.coefficients(data)
        assert coefficients is not None
        np.testing.assert_array_equal(coefficients.weights, np.array([0.4, 0.6]))
        assert reducer.evaluate(data) == 7.6

    def test_reducer__weighted_sum(self) -> None:
        """Checks sum properly accounts for sample/support weights."""

        # Synthetic data weights to check product
        data = _local_data(
            [4.0, 10.0],
            sample_weights=[2.0, 1.0],
            support_weights=[1.0, 3.0],
        )
        reducer = Sum()

        # Combine weights: 2 * 1 = 2 and 1 * 3 = 3, without normalization
        # Check weighted sum: 4 * 2 + 10 * 3 = 38
        coefficients = reducer.coefficients(data)
        np.testing.assert_array_equal(coefficients.weights, np.array([2.0, 3.0]))
        assert reducer.evaluate(data) == 38.0

    def test_reducer__weighted_stats(self) -> None:
        """Checks that coverage weights define count, quantile, mode, RMS and standard deviation consistently."""

        # Synthetic data with different values/weights
        data = _local_data([3.0, 4.0], support_weights=[1.0, 3.0])

        # Compute expected values with NumPy directly
        coverage_weights = data.support_weights
        assert coverage_weights is not None
        repeated_values = np.repeat(data.values, coverage_weights.astype(int))
        expected_count = np.sum(coverage_weights)
        expected_quantile = np.quantile(repeated_values, 0.5)
        expected_rms = np.sqrt(np.mean(np.square(repeated_values)))
        expected_std = np.std(repeated_values)

        # Check outputs are as expected
        assert Count().evaluate(data) == expected_count
        assert Quantile(0.5, weighted=True).evaluate(data) == expected_quantile
        np.testing.assert_allclose(RootMeanSquare().evaluate(data), expected_rms)
        np.testing.assert_allclose(StandardDeviation().evaluate(data), expected_std)

        # Weighted mode compares total weights: 4 has 0.5, while the two 2s total 0.2
        weighted_mode = _local_data([2.0, 2.0, 4.0], support_weights=[0.1, 0.1, 0.5])
        mode_weights = weighted_mode.support_weights
        assert mode_weights is not None
        mode_values = np.unique(weighted_mode.values)
        mode_totals = np.array([np.sum(mode_weights[weighted_mode.values == value]) for value in mode_values])
        expected_mode = mode_values[np.argmax(mode_totals)]
        assert Mode().evaluate(weighted_mode) == expected_mode
        # We check that ties select the first input value when counts match (GDAL norm)
        assert Mode(weighted=False, tie_break="first").evaluate(_local_data([4.0, 2.0])) == 4


class TestPointReducerAccuracy:
    """
    Test module for point neighborhood statistics, missing observations and median options.

    Point attributes and chunked filtering are covered in test_filters/test_irregular.py;
    gridding options and chunked grids are in test_interface/test_gridding.py.
    """

    @pytest.mark.parametrize("options", [{"weighted": True}, {"method": "lower"}])
    def test_filter__median_options(self, options: dict[str, Any]) -> None:
        """Checks that weighted and lower medians select the lower middle observation for equal point weights."""

        # Two observations distinguish lower-middle selection from averaging the middle values
        points = gu.PointCloud.from_xyz(np.array([0.0, 1.0]), np.zeros(2), np.array([2.0, 8.0]), crs=32631)
        filtered = points.filter(Median(**options), radius=2)
        np.testing.assert_array_equal(filtered.data, [2, 2])

    def test_filter__nodata_and_minimum_points(self) -> None:
        """Checks that filtering omits or propagates missing values and enforces a minimum valid count."""

        # Shared neighborhood with one missing value
        frame = gpd.GeoDataFrame(
            {"height": [0.0, np.nan, 2.0]},
            geometry=gpd.points_from_xy([0.0, 1.0, 2.0], [0.0] * 3),
            crs=32632,
        )
        points = gu.PointCloud(frame, data_name="height")

        # Apply nodata rules and minimum point count
        omitted = points.filter(method="mean", radius=3.0, nodata_propagation="ignore")
        propagated = points.filter(method="mean", radius=3.0, nodata_propagation="propagate")
        insufficient = points.filter(method="mean", radius=3.0, min_points=3)

        # Check finite mean, propagated NaN and insufficient-point NaN
        np.testing.assert_array_equal(omitted.data, [1.0, 1.0, 1.0])
        assert np.isnan(propagated.data).all()
        assert np.isnan(insufficient.data).all()

    @pytest.mark.parametrize("resampling", ["mean"])
    def test_grid_pc__circular_neighborhood(self, resampling: GriddingMethod) -> None:
        """Checks that moving means match the analytical result for equidistant points."""

        # Two constant value columns place an equal pair of neighbors around the central column
        pc = gpd.GeoDataFrame(
            data={"z": [0.0, 10.0, 0.0, 10.0]},
            geometry=gpd.points_from_xy(x=[0.0, 2.0, 0.0, 2.0], y=[0.0, 0.0, 1.0, 1.0]),
        )
        grid_coords = (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))

        # A radius just over one pixel reaches both same-row neighbors at the center
        result, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            data_name="z",
            resampling=resampling,
            dist_nodata_pixel=1.1,
        )
        expected = np.array([[0.0, 5.0, 10.0], [0.0, 5.0, 10.0]])
        assert np.allclose(result, expected)

    @pytest.mark.parametrize(
        ("resampling", "expected_center"),
        [
            ("mean", 5.0),
            ("average", 5.0),
            ("minimum", 2.0),
            ("min", 2.0),
            ("maximum", 8.0),
            ("max", 8.0),
            ("range", 6.0),
            ("count", 2.0),
            ("stdev", 3.0),
            ("average_distance", 1.0),
            ("average_distance_pts", 2.0),
        ],
    )
    def test_grid_pc__circular_statistics(self, resampling: GriddingMethod, expected_center: float) -> None:
        """Checks that circular statistics and their aliases omit invalid values and match known results."""

        # Two finite points surround the central cell while invalid values cannot contribute
        pc = gpd.GeoDataFrame(
            data={"z": [2.0, 8.0, np.nan, 20.0]},
            geometry=gpd.points_from_xy(x=[0.0, 2.0, 1.0, np.nan], y=[0.0, 0.0, 0.0, 0.0]),
        )
        grid_coords = (np.arange(5, dtype=float), np.array([0.0]))

        # Cells with neighbors use only finite values and cells outside support remain NaN
        result, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            grid_res=(1.0, 1.0),
            data_name="z",
            resampling=resampling,
            dist_nodata_pixel=1.1,
        )
        assert result[0, 1] == pytest.approx(expected_center)
        assert np.isnan(result[0, 4])

    @pytest.mark.parametrize(
        ("resampling", "expected"),
        [
            ("range", 0.0),
            ("count", 1.0),
            ("stdev", 0.0),
            ("average_distance", 0.0),
            ("average_distance_pts", np.nan),
        ],
    )
    def test_grid_pc__circular_single_point(self, resampling: GriddingMethod, expected: float) -> None:
        """Checks that circular spread and distance metrics return the expected result for one point."""

        # Match the central neighborhood used as the large data correctness fingerprint
        pc = gpd.GeoDataFrame(data={"z": [1.0]}, geometry=gpd.points_from_xy(x=[1.0], y=[1.0]))
        result, _ = _grid_pointcloud(
            pc,
            grid_coords=(np.arange(3, dtype=float), np.arange(3, dtype=float)),
            grid_res=(1.0, 1.0),
            data_name="z",
            resampling=resampling,
            dist_nodata_pixel=0.1,
        )

        assert np.isclose(result[1, 1], expected, equal_nan=True)

    def test_grid_pc__minimum_points(self) -> None:
        """Checks that circular outputs need the requested number of finite points."""

        # Only the central cell reaches both points inside its circular support
        pc = gpd.GeoDataFrame(
            data={"z": [2.0, 8.0]},
            geometry=gpd.points_from_xy(x=[0.0, 2.0], y=[0.0, 0.0]),
        )
        result, _ = _grid_pointcloud(
            pc,
            grid_coords=(np.array([0.0, 1.0, 2.0]), np.array([0.0])),
            grid_res=(1.0, 1.0),
            data_name="z",
            resampling="mean",
            dist_nodata_pixel=1.1,
            min_points=2,
        )
        assert np.array_equal(result, np.array([[np.nan, 5.0, np.nan]]), equal_nan=True)

    def test_grid_pc__circular_nodata_propagation(self) -> None:
        """Checks that an invalid point propagates through the complete circular support when requested."""

        # The finite endpoints give every central neighborhood a result when invalid values are ignored
        pc = gpd.GeoDataFrame(
            data={"z": [2.0, np.nan, 8.0]},
            geometry=gpd.points_from_xy(x=[0.0, 1.0, 2.0], y=[0.0, 0.0, 0.0]),
        )
        grid_coords = (np.arange(4, dtype=float), np.array([0.0]))
        default, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            grid_res=(1.0, 1.0),
            data_name="z",
            resampling="mean",
            dist_nodata_pixel=1.1,
        )
        propagated, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            grid_res=(1.0, 1.0),
            data_name="z",
            resampling="mean",
            dist_nodata_pixel=1.1,
            nodata_handling="propagate",
        )

        # The last cell lies outside the invalid point support and therefore remains unchanged
        assert np.all(np.isfinite(default[0, :3]))
        assert np.all(np.isnan(propagated[0, :3]))
        assert propagated[0, 3] == default[0, 3]


class TestRasterReducerAccuracy:
    """Test module for known window statistics, offset orientation and dense/sparse agreement.

    Raster filter options and chunks are covered in test_filters/test_regular.py; point sampling options
    and chunks are in test_interface/test_resampling.py.
    """

    @pytest.mark.parametrize(
        "method, np_filter",
        [
            ("mean", np.nanmean),
            ("median", np.nanmedian),
            ("max", np.nanmax),
            ("min", np.nanmin),
        ],
    )
    def test_filter_against_center_value(self, method: str, np_filter: Callable[[NDArrayNum], NDArrayNum]) -> None:
        """Checks that a filter's center value matches NumPy's result for the full 3 x 3 window."""

        # Unequal values distinguish the statistics in a complete 3 x 3 window
        arr = np.random.default_rng(42).normal(size=(3, 3))
        arr_filtered = getattr(gu.filters, f"{method}_filter")(arr, size=3)

        # Check the center against NumPy's full-window statistic
        assert np.isclose(np_filter(arr), arr_filtered[1, 1], atol=1e-8)

    @pytest.mark.parametrize("reducer_type", [Mean, Sum, Count, RootMeanSquare, Minimum, Maximum, Range, Median])
    @pytest.mark.parametrize(
        "shape,fractional", [("square", False), ("circular", False), ("square", True), ("circular", True)]
    )
    @pytest.mark.parametrize("nodata", ["ignore", "propagate"])
    def test_resample_at_points__dense_and_sparse_windows(
        self,
        reducer_type: type[Reducer],
        shape: Literal["square", "circular"],
        fractional: bool,
        nodata: Literal["ignore", "propagate"],
    ) -> None:
        """Checks that dense window reductions match separate neighborhoods, including fractional edges and nodata."""

        # Missing patches include empty windows, while edge targets have only partial neighborhoods
        values = np.random.default_rng(42).normal(size=(13, 15))
        values[3:9, 4:10] = np.nan
        raster = gu.Raster.from_array(values, rio.transform.from_origin(100, 200, 2, 3), crs=32631, nodata=np.nan)
        rows, cols = np.indices(values.shape)
        phase = 0.25 if fractional else 0.5
        points = (100 + 2 * (cols.ravel() + phase), 200 - 3 * (rows.ravel() + phase))
        neighborhood = GridNeighbours(size=5, shape=shape)
        operator = reducer_type(neighborhood=neighborhood)

        # Compare filtering a dense set of targets with evaluating small groups independently
        result = raster.resample_at_points(
            points, operator, as_array=True, coverage=("fractional" if fractional else "center"), nodata_handling=nodata
        )
        expected = np.concatenate(
            [
                raster.resample_at_points(
                    (points[0][start : start + 7], points[1][start : start + 7]),
                    operator,
                    as_array=True,
                    coverage=("fractional" if fractional else "center"),
                    nodata_handling=nodata,
                )
                for start in range(0, len(points[0]), 7)
            ]
        )

        # Convolution changes summation order; polygon intersections can also differ by roundoff
        np.testing.assert_allclose(result, expected, rtol=1e-11, atol=1e-11, equal_nan=True)
        assert operator.default_neighborhood is neighborhood

    def test_resample_at_points__asymmetric_window(self) -> None:
        """Checks that dense reduction selects the supplied offsets without reversing the window."""

        # The right and lower-right cells have distinct values on this grid
        values = np.arange(144, dtype=float).reshape(12, 12)
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 12, 1, 1), crs=32631)
        neighborhood = GridNeighbours(((0, 0), (0, 1), (1, 2)))
        rows, cols = np.indices((10, 10))
        points = (cols.ravel() + 0.5, 12 - rows.ravel() - 0.5)

        # Derive the sum by slicing the three selected offsets
        result = raster.resample_at_points(points, Sum(neighborhood=neighborhood), as_array=True)
        expected = values[:10, :10] + values[:10, 1:11] + values[1:11, 2:12]
        np.testing.assert_array_equal(result, expected.ravel())


class TestFractionalReducerWindows:
    """Test module for covered area and weighted means in square and circular windows."""

    def test_resample_at_points__coverage(self) -> None:
        """Checks area coverage options (center, all touched, and fractional) for selecting window pixels."""

        # We use a 1-pixel window shifted to lower-right of a 2x2 raster
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 2, 1, 1), crs=32631)
        point = (0.75, 1.25)
        neighborhood = GridNeighbours(size=1)
        operator = Sum(neighborhood=neighborhood)

        # Center selects first pixel, touched selects all four, fractional uses area weights 9/16, 3/16, 3/16, 1/16
        center = raster.resample_at_points(point, operator, as_array=True)
        touched = raster.resample_at_points(point, operator, coverage="all_touched", as_array=True)
        fractional = raster.resample_at_points(point, operator, coverage="fractional", as_array=True)
        assert center == values[0, 0]
        assert touched == np.sum(values)
        assert fractional == pytest.approx((9 * values[0, 0] + 3 * values[0, 1] + 3 * values[1, 0] + values[1, 1]) / 16)
        assert operator.default_neighborhood is neighborhood

        # Check when defining the coverage through the operator directly
        configured = Sum(neighborhood=GridNeighbours(size=1, coverage="all_touched"))
        assert raster.resample_at_points(point, configured, as_array=True) == touched
        wider = Sum(neighborhood=GridNeighbours(size=3))
        assert raster.reduce_at_points(point, reducer_function=wider, as_array=True) == np.sum(values)
        assert raster.reduce_at_points(point, reducer_function=wider, window=1, as_array=True) == values[0, 0]

    def test_filter__fractional_circle_area(self) -> None:
        """Checks that a fractional count measures circular area rather than the number of selected cells."""

        # A 5-pixel circle has radius 2.5 pixels
        raster = gu.Raster.from_array(np.ones((9, 9)), rio.transform.from_origin(0, 9, 1, 1), crs=32631)
        neighborhood = GridNeighbours(size=5, shape="circular")
        operator = Count(neighborhood=neighborhood)
        area = raster.filter(operator, coverage="fractional").to_nanarray()[4, 4]
        touched = raster.filter(operator, coverage="all_touched").to_nanarray()[4, 4]
        count = raster.filter(operator).to_nanarray()[4, 4]

        # The circular footprint uses a 512-sided polygon (line densification)
        # We compare its expected area
        expected_area = 512 * 2.5**2 * np.sin(2 * np.pi / 512) / 2
        assert area == pytest.approx(expected_area, rel=0, abs=1e-6)
        assert touched == 25
        assert count == 21

    @pytest.mark.parametrize(("window", "expected_area"), [(1, np.pi / 4), (3, 2.25 * np.pi), (5, 6.25 * np.pi)])
    def test_reduce_points__fractional_circular_window(self, window: int, expected_area: float) -> None:
        """Checks that circular windows use their covered area for different pixel radii."""

        # Every centered circle fits entirely inside the seven by seven raster
        raster = gu.Raster.from_array(np.ones((7, 7)), rio.transform.from_origin(0, 7, 1, 1), crs=32631)
        point = (3.5, 3.5)
        neighbourhood = GridNeighbours(size=window, shape="circular")

        # The custom reducer adds the covered cell fractions, which should equal the circle's area in pixels
        reduced = raster.reduce_at_points(
            point,
            reducer_function=SupportMassReducer(),
            window=window,
            window_shape="circular",
            coverage="fractional",
            as_array=True,
        )
        resampled = raster.resample_at_points(
            point, SupportMassReducer(neighborhood=neighbourhood), coverage="fractional", as_array=True
        )
        assert reduced == pytest.approx(expected_area, rel=1e-4)
        assert resampled == pytest.approx(reduced)

    def test_reduce_points__fractional_circular_mean(self) -> None:
        """Checks that a shifted circular window weights values by their covered areas."""

        # Split a one-pixel circle between columns with values zero and ten
        values = np.array([[0, 10], [0, 10]], dtype=float)
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 2, 1, 1), crs=32631)
        point = (0.75, 1.5)

        # The right column covers a circular segment beyond x=1; a square window covers one quarter instead
        right_area = np.pi / 12 - np.sqrt(3) / 16
        expected_mean = 10 * right_area / (np.pi / 4)
        circular = raster.reduce_at_points(
            point, window=1, window_shape="circular", coverage="fractional", as_array=True
        )
        square = raster.reduce_at_points(point, window=1, coverage="fractional", as_array=True)
        assert circular == pytest.approx(expected_mean, rel=1e-4)
        assert square == 2.5


class TestPointReducerEngines:
    """Test module for SciPy/Numba agreement on point statistics and radius boundaries.

    Interpolation engines are covered in test_interpolator.py; chunked grids are in test_interface/test_gridding.py.
    """

    @pytest.mark.skipif(find_spec("numba") is None, reason="Requires numba")
    @pytest.mark.parametrize(
        "resampling",
        ["mean", "minimum", "maximum", "range", "count", "stdev", "average_distance"],
    )
    def test_grid_pc__engine(self, resampling: GriddingMethod) -> None:
        """Checks that the SciPy and Numba engines give the same gridded values."""

        import_optional("numba")

        # Uneven point values and positions exercise distance choices and every accumulation
        pc = gpd.GeoDataFrame(
            data={"z": [1.0, 4.0, 8.0]},
            geometry=gpd.points_from_xy(x=[0.0, 1.2, 3.0], y=[0.0, 1.0, 0.0]),
        )
        grid_coords = (np.arange(4, dtype=float), np.arange(2, dtype=float))
        scipy_result, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            data_name="z",
            resampling=resampling,
            dist_nodata_pixel=2,
            engine="scipy",
        )

        # The explicit Numba engine follows the same interface as elsewhere in GeoUtils and xDEM
        numba_result, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            data_name="z",
            resampling=resampling,
            dist_nodata_pixel=2,
            engine="numba",
        )
        assert np.allclose(scipy_result, numba_result, equal_nan=True)

    @pytest.mark.skipif(find_spec("numba") is None, reason="Requires numba")
    @pytest.mark.parametrize(
        ("resampling", "expected"),
        [("average_distance", np.array([[0.0, 1.0, np.nan]]))],
    )
    def test_grid_pc__engines_include_support_boundary(self, resampling: GriddingMethod, expected: NDArrayNum) -> None:
        """Checks that both Numba/SciPy engines include cells on the support radius and exclude cells beyond it."""

        import_optional("numba")
        pc = gpd.GeoDataFrame(data={"z": [4.0]}, geometry=gpd.points_from_xy(x=[0.0], y=[0.0]))
        for engine in ("scipy", "numba"):
            result, _ = _grid_pointcloud(
                pc,
                grid_coords=(np.arange(3, dtype=float), np.array([0.0])),
                grid_res=(1.0, 1.0),
                data_name="z",
                resampling=resampling,
                dist_nodata_pixel=1,
                engine=engine,
            )
            assert np.allclose(result, expected, equal_nan=True)

    @pytest.mark.skipif(find_spec("numba") is None, reason="Requires numba")
    @pytest.mark.parametrize(
        "operator_type",
        [Mean, Minimum, Maximum, Range, Count, StandardDeviation, AverageDistance],
    )
    @pytest.mark.parametrize(
        "neighborhood", [PointNeighbours(k=3), PointNeighbours(radius=1.1), PointNeighbours(k=3, radius=1.1)]
    )
    @pytest.mark.parametrize("nodata_handling", ["ignore", "propagate"])
    def test_grid__numba_matches_scipy_for_explicit_neighborhoods(
        self,
        operator_type: type[Reducer],
        neighborhood: PointNeighbours,
        nodata_handling: NodataChoice,
    ) -> None:
        """Checks that every supported Numba method uses the same selected points and nodata rule as SciPy."""

        import_optional("numba")

        # Include coincident observations, a missing value, radius boundaries and an empty output neighborhood
        points = gu.PointCloud.from_xyz([0, 0, 1, 2, 5], [0, 0, 0, 1, 2], [2, 4, np.nan, 8, 20], crs=32631)
        grid_coords = (np.arange(7, dtype=float), np.arange(4, dtype=float))
        operator = operator_type(neighborhood=neighborhood)
        options: dict[str, Any] = {
            "grid_coords": grid_coords,
            "resampling": operator,
            "nodata_handling": nodata_handling,
        }

        # Both engines must select the same observations, including at exact point locations
        expected = points.grid(**options, engine="scipy")
        result = points.grid(**options, engine="numba")
        np.testing.assert_allclose(result.to_nanarray(), expected.to_nanarray(), rtol=1e-14, atol=1e-14)


class TestPointReducerReferences:
    """Test module for point statistics compared with GDAL and leave-one-out medians compared with PDAL."""

    @pytest.mark.skipif(find_spec("pdal") is None, reason="Requires pdal")
    def test_filter__pdal_zsmooth_accuracy(self) -> None:
        """Checks that leave-one-out median filtering matches PDAL filters.zsmooth on the same neighborhoods."""

        # PDAL reference points with at least one neighbor each
        pdal = import_optional("pdal")
        coordinates = np.arange(5, dtype=np.float64)
        heights = np.array([0.0, 100.0, 2.0, 9.0, 4.0])
        pdal_points = np.zeros(5, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8")])
        pdal_points["X"] = coordinates
        pdal_points["Z"] = heights

        # Run PDAL zsmooth and equivalent leave-one-out median
        pipeline_json = json.dumps(
            [{"type": "filters.zsmooth", "radius": 1.1, "medianpercent": 50, "dim": "SmoothedZ"}]
        )
        pipeline = pdal.Pipeline(pipeline_json, arrays=[pdal_points])
        assert pipeline.execute() == len(pdal_points)
        frame = gpd.GeoDataFrame(
            {"height": heights},
            geometry=gpd.points_from_xy(coordinates, np.zeros(5)),
            crs=32632,
        )
        filtered = gu.PointCloud(frame, data_name="height").filter(method="median", radius=1.1, include_self=False)

        # Check exact agreement for same neighborhood medians
        np.testing.assert_array_equal(filtered.data, pipeline.arrays[0]["SmoothedZ"])

    @pytest.mark.parametrize(
        ("geoutils_method", "gdal_algorithm"),
        [
            ("mean", "average"),
            ("minimum", "minimum"),
            ("maximum", "maximum"),
            ("range", "range"),
            ("count", "count"),
            ("average_distance", "average_distance"),
            ("average_distance_pts", "average_distance_pts"),
        ],
    )
    def test_grid_pc__gdal_circular_methods(
        self, geoutils_method: GriddingMethod, gdal_algorithm: str, tmp_path: Path
    ) -> None:
        """Checks that circular methods match GDAL values and nodata cells."""

        # Two unequal values exercise all statistics while an invalid value checks GDAL's omission rule
        points = gpd.GeoDataFrame(
            {"z": [2.0, np.nan, 8.0]},
            geometry=gpd.points_from_xy(x=[0.0, 1.0, 2.0], y=[0.0, 0.0, 0.0]),
            crs=32631,
        )
        point_file = tmp_path / "circular-points.gpkg"
        points.to_file(point_file, layer="source-points", driver="GPKG")
        x_coords = np.arange(5, dtype=float)
        y_coords = np.array([0.0])

        # GeoUtils expresses the support ellipse in output pixels
        # Use finite points only to match GDAL's point gridding (including IDW)
        expected, _ = _grid_pointcloud(
            points,
            grid_coords=(x_coords, y_coords),
            grid_res=(1.0, 1.0),
            data_name="z",
            resampling=geoutils_method,
            dist_nodata_pixel=1.1,
            nodata_handling="ignore",
        )

        # GDAL receives the equivalent support in coordinate units and the same cell centers
        output_file = tmp_path / f"gdal-{gdal_algorithm}.tif"
        command = build_gdal_grid_command(
            str(point_file),
            str(output_file),
            algorithm=gdal_algorithm,  # type: ignore[arg-type]
            bounds=(-0.5, -0.5, 4.5, 0.5),
            shape=(1, 5),
            radius=(1.1, 1.1),
        )
        subprocess.run(command.command, check=True, capture_output=True, text=True)
        with rio.open(output_file) as dataset:
            actual = dataset.read(1, masked=True).filled(np.nan)

        assert np.allclose(actual, expected, equal_nan=True)


_GDAL_REDUCER_CASES = [
    (Mean(), rio.enums.Resampling.average),
    (Sum(), rio.enums.Resampling.sum),
    (Minimum(), rio.enums.Resampling.min),
    (Maximum(), rio.enums.Resampling.max),
    (RootMeanSquare(), rio.enums.Resampling.rms),
    (Mode(weighted=False, tie_break="first_to_mode"), rio.enums.Resampling.mode),
    (Median(method="inverted_cdf"), rio.enums.Resampling.med),
    (Quantile(0.25, method="inverted_cdf"), rio.enums.Resampling.q1),
    (Quantile(0.75, method="inverted_cdf"), rio.enums.Resampling.q3),
]


_EXACTEXTRACT_REDUCER_CASES = [
    Mean(),
    Sum(),
    Minimum(),
    Maximum(),
    StandardDeviation(),
    Mode(),
    Median(weighted=True),
    Quantile(0.25, weighted=True),
]


class TestRasterReducerCoverage:
    """Test module for source window coverage used by Reducers in reproject()."""

    def test_reproject__source_window_coverage(self) -> None:
        """Checks that the default source window and explicit coverage modes choose the expected cells."""

        # We create a synthetic shifted destination cell that overlaps four source cells with unequal areas
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        source = gu.Raster.from_array(values, rio.transform.from_origin(0, 2, 1, 1), crs=32631)
        reference = gu.Raster.from_array(np.zeros((1, 1)), rio.transform.from_origin(0.25, 1.75, 1, 1), crs=32631)
        operator = Sum()

        # We check the output
        default = source.reproject(reference, resampling=operator)
        center = source.reproject(reference, resampling=operator, coverage="center")
        fractional = source.reproject(reference, resampling=operator, coverage="fractional")
        fixed = source.reproject(reference, resampling=operator, window=1, coverage="all_touched")
        assert default is not None and center is not None and fractional is not None and fixed is not None
        # Default uses fractional areas
        assert default.data[0, 0] == pytest.approx(fractional.data[0, 0])
        # Center selects one
        assert center.data[0, 0] == values[0, 0]
        # Fractional uses areas
        assert fractional.data[0, 0] == pytest.approx(
            (9 * values[0, 0] + 3 * values[0, 1] + 3 * values[1, 0] + values[1, 1]) / 16
        )
        assert fixed.data[0, 0] == np.sum(values)
        assert operator.default_neighborhood is None


class TestRasterReducerReferences:
    """Test module for fractional reducers compared with GDAL and between ExactExtract and Shapely.

    Reprojection grids, source IDs and chunked execution are covered in test_raster/test_transformations_raster.py.
    """

    def test_reproject__mode_tie_matches_gdal(self) -> None:
        """Checks that GDAL mode selects the value that first reaches the tied winning count."""

        # Both two-by-two output cells tie, but the second source value reaches two occurrences first
        values = np.array([[2, 3, 4, 5], [3, 2, 5, 4]], dtype=float)
        source = gu.Raster.from_array(values, rio.transform.from_origin(0, 2, 1, 1), crs=32632)
        reference = gu.Raster.from_array(np.zeros((1, 2)), rio.transform.from_origin(0, 2, 2, 2), crs=32632)

        # Match GDAL's tie rule while preserving the separate first-source rule of Mode(tie_break="first")
        expected = source.reproject(reference, resampling="mode")
        actual = source.reproject(
            reference,
            resampling=Mode(weighted=False, tie_break="first_to_mode"),
            coverage="fractional",
        )
        assert expected is not None and actual is not None
        np.testing.assert_array_equal(actual.to_nanarray(), expected.to_nanarray())
        np.testing.assert_array_equal(actual.to_nanarray(), [[3, 5]])

    def test_reduce_points__fractional_window_matches_gdal(self) -> None:
        """Checks that a shifted three-cell window gives the same area-weighted mean as GDAL."""

        # Use a point between cell centers so the 3 x 3 footprint intersects sixteen source cells
        values = np.arange(1, 26, dtype=float).reshape(5, 5)
        source = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 5, 1, 1), crs=32631)
        point = (2.75, 2.25)
        reference = gu.Raster.from_array(
            np.zeros((1, 1)), transform=rio.transform.from_origin(1.25, 3.75, 3, 3), crs=32631
        )

        # GDAL reduces the same physical square; whole-cell reduction gives a different answer
        expected = source.reproject(reference, resampling=rio.enums.Resampling.average)
        fractional = source.reduce_at_points(
            point, reducer_function=Mean(), window=3, coverage="fractional", as_array=True
        )
        ordinary = source.reduce_at_points(point, reducer_function=Mean(), window=3, as_array=True)
        assert expected is not None
        assert fractional == pytest.approx(float(expected.data[0, 0]))
        assert fractional != ordinary

    @pytest.mark.parametrize(
        "overlap_backend",
        [
            "shapely",
            pytest.param(
                "exactextract",
                marks=pytest.mark.skipif(find_spec("exactextract") is None, reason="Requires exactextract"),
            ),
        ],
    )
    def test_reproject__mean_gdal_nodata(self, overlap_backend: str) -> None:
        """Checks that mean reprojection omits missing cells and returns nodata when no finite cells overlap."""

        if overlap_backend == "exactextract":
            import_optional("exactextract")

        # The upper output cell covers eight finite values and one missing center; the lower cell has no finite data
        values = np.full((6, 3), np.nan)
        values[:3] = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 20]])
        source = gu.Raster.from_array(values, rio.transform.from_origin(0, 6, 1, 1), crs=32632, nodata=-9999)
        reference = gu.Raster.from_array(
            np.zeros((2, 1)), rio.transform.from_origin(0, 6, 3, 3), crs=32632, nodata=-9999
        )

        # Compare the GeoUtils mean to GDAL average; 51 / 8 gives the upper value, and the lower value is nodata
        gdal_average = source.reproject(reference, resampling=rio.enums.Resampling.average)
        geoutils_average = source.reproject(
            reference, resampling=Mean(), nodata_propagation="ignore", overlap_backend=overlap_backend
        )
        assert gdal_average is not None and geoutils_average is not None
        np.testing.assert_allclose(gdal_average.to_nanarray(), [[6.375], [np.nan]], equal_nan=True)
        np.testing.assert_allclose(geoutils_average.to_nanarray(), gdal_average.to_nanarray(), equal_nan=True)

    @pytest.mark.parametrize(
        ("operator", "gdal_resampling"),
        _GDAL_REDUCER_CASES,
    )
    def test_reproject__fractional_reducer_matches_gdal(
        self,
        operator: Reducer,
        gdal_resampling: rio.enums.Resampling,
    ) -> None:
        """Checks that fractional reducer support agrees with GDAL aggregate resampling on a shifted grid."""

        # Shift and resize destination cells so boundary source pixels contribute unequal fractional areas
        values = np.arange(1, 57, dtype=np.float64).reshape(7, 8)
        values[2, 3] = np.nan
        source = gu.Raster.from_array(
            values,
            rio.transform.from_origin(0, 7, 1, 1),
            crs=32632,
            nodata=-9999,
        )
        reference = gu.Raster.from_array(
            np.zeros((3, 3)),
            rio.transform.from_origin(0.35, 6.6, 2.3, 1.7),
            crs=32632,
            nodata=-9999,
        )

        # Compare the weighted reduction against GDAL's established implementation
        expected = source.reproject(reference, resampling=gdal_resampling)
        actual = source.reproject(reference, resampling=operator, coverage="fractional")
        assert expected is not None and actual is not None
        np.testing.assert_allclose(actual.to_nanarray(), expected.to_nanarray(), rtol=1e-12, atol=1e-12)

        # The first weighted mean demonstrates that boundary cells are not treated as equal or center-only inputs
        if isinstance(operator, Mean):
            np.testing.assert_allclose(actual.data[0, 0], 7.647058823529412)

    @pytest.mark.parametrize(("operator", "gdal_resampling"), _GDAL_REDUCER_CASES)
    def test_reproject__fractional_reducer_matches_gdal_for_non_square_multiband(
        self,
        operator: Reducer,
        gdal_resampling: rio.enums.Resampling,
    ) -> None:
        """Checks that every aggregate reducer matches GDAL for multiple bands and non-square footprints."""

        # Give both bands distinct values and use unequal destination X/Y resolution within the source extent
        first = np.arange(1, 26, dtype=np.float64).reshape(5, 5)
        second = first + 50
        values = np.stack((first, second))
        source = gu.Raster.from_array(
            values,
            rio.transform.from_origin(0, 5, 1, 1),
            crs=32632,
            nodata=-9999,
        )
        reference = gu.Raster.from_array(
            np.zeros((3, 3)),
            rio.transform.from_origin(0.35, 4.8, 1.4, 1.3),
            crs=32632,
            nodata=-9999,
        )

        # Calculate the statistic within each rectangular output cell, separately for both bands
        expected = source.reproject(reference, resampling=gdal_resampling)
        actual = source.reproject(reference, resampling=operator, coverage="fractional")
        assert expected is not None and actual is not None
        np.testing.assert_allclose(actual.to_nanarray(), expected.to_nanarray(), rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(
        ("overlap_backend", "rtol", "atol"),
        [
            ("shapely", 1e-12, 1e-12),
            pytest.param(
                "exactextract",
                5e-8,
                2e-5,
                marks=pytest.mark.skipif(find_spec("exactextract") is None, reason="Requires exactextract"),
            ),
        ],
    )
    def test_reproject__rotated_fractional_sum_matches_gdal(
        self,
        overlap_backend: str,
        rtol: float,
        atol: float,
    ) -> None:
        """Checks that each exact-overlap backend matches GDAL sum resampling for rotated footprints."""

        if overlap_backend == "exactextract":
            import_optional("exactextract")

        # Rotate non-square destination cells within a larger source so every footprint requires polygon intersection
        values = np.arange(1, 401, dtype=np.float64).reshape(20, 20)
        values[6:8, 7] = np.nan
        source = gu.Raster.from_array(
            values,
            rio.transform.from_origin(0, 20, 1, 1),
            crs=32632,
            nodata=-9999,
        )
        destination_transform = Affine.translation(3, 17) * Affine.rotation(11) * Affine.scale(2, -2)
        reference = gu.Raster.from_array(
            np.zeros((5, 5)),
            destination_transform,
            crs=32632,
            nodata=-9999,
        )

        # GDAL sum directly reports fractional area contributions and independently validates the overlap fractions
        expected = source.reproject(reference, resampling=rio.enums.Resampling.sum)
        actual = source.reproject(reference, resampling=Sum(), overlap_backend=overlap_backend, coverage="fractional")
        assert expected is not None and actual is not None
        np.testing.assert_allclose(actual.to_nanarray(), expected.to_nanarray(), rtol=rtol, atol=atol)

    @pytest.mark.skipif(find_spec("exactextract") is None, reason="Requires exactextract")
    @pytest.mark.parametrize("operator", _EXACTEXTRACT_REDUCER_CASES)
    def test_reproject__exactextract_reducer_matches_shapely(self, operator: Reducer) -> None:
        """Checks that ExactExtract and Shapely give the same statistics within rotated output cells."""

        import_optional("exactextract")

        # Include repeated and unequal values so weighted mode and quantiles have observable definitions
        values = np.array(
            [
                [1, 1, 4, 4, 9, 9],
                [1, 2, 4, 7, 9, 3],
                [5, 5, 8, 8, 6, 6],
                [5, 2, 8, 7, 6, 3],
                [0, 0, 2, 2, 4, 4],
                [0, 1, 2, 3, 4, 5],
            ],
            dtype=np.float64,
        )
        source = gu.Raster.from_array(
            values,
            rio.transform.from_origin(0, 6, 1, 1),
            crs=32632,
            nodata=-9999,
        )
        destination_transform = Affine.translation(0.4, 5.4) * Affine.rotation(13) * Affine.scale(2.2, -2.2)
        reference = gu.Raster.from_array(np.zeros((2, 2)), destination_transform, crs=32632, nodata=-9999)

        # Compare direct ExactExtract statistics with reductions calculated from Shapely intersection areas
        expected = source.reproject(reference, resampling=operator, overlap_backend="shapely", coverage="fractional")
        actual = source.reproject(reference, resampling=operator, overlap_backend="exactextract", coverage="fractional")
        assert expected is not None and actual is not None
        np.testing.assert_allclose(actual.to_nanarray(), expected.to_nanarray(), rtol=5e-7, atol=5e-7)


class TestReducerUncertainty:
    """Test module for reducer values and uncertainty from observation errors and covered areas."""

    def test_reduce_points__fractional_area_and_uncertainty(self) -> None:
        """Checks that a shifted one-cell window uses covered areas for its mean and uncertainty."""

        # Shift a one-pixel square a quarter cell right and down across four source cells
        values = np.array([[0, 10], [20, 30]], dtype=float)
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 2, 1, 1), crs=32631)
        point = (0.75, 1.25)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # Covered fractions are 9/16, 3/16, 3/16, 1/16 in row order
        ordinary = raster.reduce_at_points(point, reducer_function=Mean(), window=1, as_array=True)
        fractional = raster.reduce_at_points(
            reducer_function=Mean(),
            window=1,
            coverage="fractional",
            as_array=True,
            points=point,
            error_structure=errors,
        )
        summary = gu.uncertainty.propagate(
            raster.reduce_at_points,
            error_structure=errors,
            operation_kwargs={
                "reducer_function": Mean(),
                "window": 1,
                "coverage": "fractional",
                "as_array": True,
                "points": point,
            },
        )
        default_mean = raster.reduce_at_points(point, window=1, coverage="fractional", as_array=True)
        assert ordinary == 0
        assert fractional == pytest.approx(7.5)
        assert default_mean == pytest.approx(fractional)
        assert summary.std == pytest.approx(1.25)

    def test_resample_at_points__reducer_missing_value_and_uncertainty(self) -> None:
        """Checks that a reducer omits missing cells and uses its remaining weights for uncertainty."""

        # Eight of the nine cells around the target have value one; the center cell is missing
        values = np.ones((5, 5), dtype=float)
        values[2, 2] = np.nan
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 5, 1, 1), crs=4326, nodata=-9999)
        x, y = raster.ij2xy(2, 2, force_offset="center")
        source_errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # Compare the default missing-value rule with one that rejects the whole window
        ignored = raster.resample_at_points(as_array=True, points=(x, y), method=Mean(), error_structure=source_errors)
        summary = gu.uncertainty.propagate(
            raster.resample_at_points,
            error_structure=source_errors,
            operation_kwargs={"as_array": True, "points": (x, y), "method": Mean()},
        )
        propagated = raster.resample_at_points((x, y), Mean(), as_array=True, nodata_handling="propagate")

        # Eight independent errors of magnitude two, each weighted 1/8, give a standard deviation of 2 / sqrt(8)
        assert ignored == 1
        assert np.isnan(propagated)
        assert summary.std == pytest.approx(2 / np.sqrt(8))

    @pytest.mark.parametrize(
        "engine",
        ["scipy", pytest.param("numba", marks=pytest.mark.skipif(find_spec("numba") is None, reason="Requires numba"))],
    )
    @pytest.mark.parametrize(
        "neighborhood", [PointNeighbours(k=2), PointNeighbours(radius=1), PointNeighbours(k=2, radius=1)]
    )
    def test_grid__selected_observations_and_uncertainty(
        self, engine: Literal["scipy", "numba"], neighborhood: PointNeighbours
    ) -> None:
        """Checks that point counts and coordinate radii select the same mean and independent-error variance."""

        if engine == "numba":
            import_optional("numba")

        # Two nearby values average to four; the distant value would change the result if selected
        points = gu.PointCloud.from_xyz([0, 1, 4], [0, 0, 0], [2, 6, 100], crs=32631)
        reference = gu.Raster.from_array(np.zeros((1, 1)), rio.transform.from_origin(0, 0, 2, 3), crs=32631)
        operator = Mean(neighborhood=neighborhood)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # The point radius is in map units even though the output pixels have unequal dimensions
        result = points.grid(
            ref=reference,
            resampling=operator,
            engine=engine,
            dist_nodata_pixel=30,
            nodata_handling="ignore",
            error_structure=errors,
        )
        summary = gu.uncertainty.propagate(
            points.grid,
            error_structure=errors,
            operation_kwargs={
                "ref": reference,
                "resampling": operator,
                "engine": engine,
                "dist_nodata_pixel": 30,
                "nodata_handling": "ignore",
            },
        )
        assert result.to_nanarray()[0, 0] == 4
        assert summary.variance.to_nanarray().reshape(-1)[0] == pytest.approx(2)
        assert operator.default_neighborhood is neighborhood

    @pytest.mark.parametrize(
        "operator_type",
        [Mean, Minimum, Maximum, Range, Count, StandardDeviation, AverageDistance],
    )
    @pytest.mark.parametrize(
        "engine",
        ["scipy", pytest.param("numba", marks=pytest.mark.skipif(find_spec("numba") is None, reason="Requires numba"))],
    )
    def test_grid__zero_error_with_explicit_neighborhood(
        self, operator_type: type[Reducer], engine: Literal["scipy", "numba"]
    ) -> None:
        """Checks that zero error reproduces the gridded values for every engine and supported neighborhood method."""

        if engine == "numba":
            import_optional("numba")

        # All targets have three finite neighbors with distinct distances
        points = gu.PointCloud.from_xyz([0, 0.8, 2.1, 3.3], [0, 0.1, 2.2, 0.3], [2, 4, 8, 16], crs=32631)
        operator = operator_type(neighborhood=PointNeighbours(k=3))
        grid_coords = (np.array([0.2, 1.2]), np.array([0.3, 1.3]))
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0)])

        # Small rounding differences can arise from the compiled weighted sums
        result = points.grid(
            grid_coords=grid_coords,
            resampling=operator,
            engine=engine,
            nodata_handling="ignore",
            error_structure=errors,
        )
        summary = gu.uncertainty.propagate(
            points.grid,
            error_structure=errors,
            operation_kwargs={
                "grid_coords": grid_coords,
                "resampling": operator,
                "engine": engine,
                "nodata_handling": "ignore",
            },
            method="numerical",
            n_samples=2,
            random_state=4,
        )
        np.testing.assert_allclose(
            summary.mean.to_nanarray().reshape(-1), result.to_nanarray().reshape(-1), rtol=1e-14, atol=1e-14
        )
        np.testing.assert_array_equal(summary.variance.to_nanarray().reshape(-1), np.zeros(4))

    def test_reproject__fractional_mean_uncertainty(self) -> None:
        """Checks that fractional mean coefficients also define exact analytical reprojection uncertainty."""

        # Center one two-by-two destination cell over nine source cells with quarter, half and complete coverage
        values = np.arange(9, dtype=np.float64).reshape(3, 3)
        source = gu.Raster.from_array(
            values,
            rio.transform.from_origin(0, 3, 1, 1),
            crs=32632,
            nodata=-9999,
        )
        reference = gu.Raster.from_array(
            np.zeros((1, 1)),
            rio.transform.from_origin(0.5, 2.5, 2, 2),
            crs=32632,
            nodata=-9999,
        )
        source_error = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Check that requesting uncertainty returns the same mean values as the ordinary reprojection
        expected = source.reproject(reference, resampling=rio.enums.Resampling.average)
        nominal = source.reproject(
            resampling=Mean(), ref=reference, error_structure=source_error, coverage="fractional"
        )
        summary = gu.uncertainty.propagate(
            source.reproject,
            error_structure=source_error,
            operation_kwargs={"resampling": Mean(), "ref": reference, "coverage": "fractional"},
        )
        assert expected is not None
        assert nominal.raster_equal(expected, strict_masked=False)

        # Four quarter, four half and one complete area weights give variance 2.25 / 4**2
        np.testing.assert_allclose(summary.variance.to_nanarray().reshape(-1), [0.140625])


class TestFractionalReducerEdges:
    """Test module for fractional area and missing values where square or circular windows cross raster edges."""

    def test_reduce_points__fractional_area_at_edge(self) -> None:
        """Checks that partial windows at the raster edge count covered area and outside points return NaN."""

        # Three quarters of a cell's width and height lie inside this one-pixel window
        values = np.full((2, 2), 8, dtype=float)
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 2, 1, 1), crs=32631)
        points = (np.array([0.25, -0.25]), np.array([1.75, 1.75]))

        # Mean divides by the covered area; Sum and the custom reducer use the 9/16 cell fraction directly
        mean = raster.reduce_at_points(points, reducer_function=Mean(), window=1, coverage="fractional", as_array=True)
        total = raster.reduce_at_points(points, reducer_function=Sum(), window=1, coverage="fractional", as_array=True)
        area = raster.reduce_at_points(
            points, reducer_function=SupportMassReducer(), window=1, coverage="fractional", as_array=True
        )
        assert mean[0] == 8
        assert total[0] == pytest.approx(4.5)
        assert area[0] == pytest.approx(9 / 16)
        assert np.isnan(mean[1])
        assert np.isnan(total[1])
        assert np.isnan(area[1])

    def test_reduce_points__fractional_area_with_missing_cell(self) -> None:
        """Checks that the fractional mean uses the remaining covered area when a source cell is missing."""

        # The missing cell would cover 3/16 of a shifted one-pixel window
        values = np.array([[0, np.nan], [20, 30]], dtype=float)
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 2, 1, 1), crs=32631, nodata=-9999)
        point = (0.75, 1.25)

        # The other cells contribute 9/16 + 3/16 + 1/16 of the window
        omitted = raster.reduce_at_points(
            point, reducer_function=Mean(), window=1, coverage="fractional", as_array=True
        )
        propagated = raster.resample_at_points(
            point,
            Mean(neighborhood=GridNeighbours(size=1)),
            coverage="fractional",
            nodata_handling="propagate",
            as_array=True,
        )
        assert omitted == pytest.approx(90 / 13)
        assert np.isnan(propagated)

    def test_reduce_points__fractional_circular_window_at_edge(self) -> None:
        """Checks that a circle centered on the raster edge contributes half its area."""

        # Put the circle's center on the left boundary, away from the other three boundaries
        raster = gu.Raster.from_array(np.full((5, 5), 8.0), rio.transform.from_origin(0, 5, 1, 1), crs=32631)
        point = (0.0, 2.5)

        # A three-cell circle has radius 1.5 pixels, so half its area covers raster cells
        area = raster.reduce_at_points(
            point,
            reducer_function=SupportMassReducer(),
            window=3,
            window_shape="circular",
            coverage="fractional",
            as_array=True,
        )
        total = raster.reduce_at_points(
            point, reducer_function=Sum(), window=3, window_shape="circular", coverage="fractional", as_array=True
        )
        mean = raster.reduce_at_points(point, window=3, window_shape="circular", coverage="fractional", as_array=True)
        assert area == pytest.approx(2.25 * np.pi / 2, rel=1e-4)
        assert total == pytest.approx(8 * 2.25 * np.pi / 2, rel=1e-4)
        assert mean == pytest.approx(8)
