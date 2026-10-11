"""Test operator dispatch and reuse across raster and point inputs."""

from importlib.util import find_spec
from typing import Literal

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio

from geoutils._misc import import_optional
from geoutils.operators import GridNeighbours, Interpolator, PointNeighbours, Reducer
from geoutils.operators.execution import _grid_from_points, _resample_at_points
from geoutils.operators.interpolator import InverseDistance, Nearest
from geoutils.operators.reducer import AverageDistance, Count, Maximum, Mean, Minimum, Range, StandardDeviation
from tests.operator_helpers import LocalMeanInterpolator


class TestOperatorDispatch:
    """
    Test module for reusing operators across raster and point inputs.
    """

    def test_interpolator__selects_neighbours_from_source(self) -> None:
        """
        Checks that a regular/irregular interpolator without defined neighbourhood can be reused for point
        and raster.
        """

        # Raster and point inputs, with an interpolator without neighborhood configuration
        values = np.ones((3, 3))
        transform = rio.transform.from_origin(0, 3, 1, 1)
        points = gpd.GeoDataFrame({"value": [2, 4]}, geometry=gpd.points_from_xy([0, 1], [0, 0]), crs=32631)
        grid_coords = (np.array([0.5]), np.array([0.5]))
        operator = LocalMeanInterpolator()

        # Use with raster
        raster_value = _resample_at_points(
            values,
            transform,
            (np.array([1.5]), np.array([1.5])),
            operator,
            area_or_point=None,
            shift_area_or_point=False,
            nodata_propagation="ignore",
            dist_nodata_spread=None,
        )
        assert raster_value[0] == 1
        assert operator.default_neighborhood is None

        # Reuse with point
        point_grid = _grid_from_points(
            points,
            grid_coords,
            "value",
            operator,
            res_x=1,
            res_y=1,
            radius=1,
            min_points=1,
            nodata_propagation="ignore",
        )
        assert point_grid[0, 0] == 3
        assert operator.default_neighborhood is None


class TestResampleAtPoints:
    """
    Test module for raster resampling.

    This wrapper is necessary because resample_at_points is used both by reproject() and interp/reduce_at_points().
    """

    def test_resample_at_points(self) -> None:
        """Checks that resampling runs, with right output order, including for empty neighborhoods."""

        # Resample 3 points: first cell, last cell, and one point outside
        values = np.arange(9.0).reshape(3, 3)
        transform = rio.transform.from_origin(0, 3, 1, 1)
        points = (np.array([2.5, 0.5, -0.5]), np.array([0.5, 2.5, 2.5]))
        result = _resample_at_points(
            values,
            transform,
            points,
            Mean(),
            area_or_point=None,
            shift_area_or_point=False,
            nodata_propagation="ignore",
            dist_nodata_spread=None,
            neighborhood=GridNeighbours(size=1),
        )

        # Check array output, target order and NaN for empty neighborhood
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [8, 0, np.nan])


class TestGridFromPoints:
    """Test module for point gridding."""

    @pytest.mark.parametrize(
        "engine",
        ["scipy", pytest.param("numba", marks=pytest.mark.skipif(find_spec("numba") is None, reason="Requires Numba"))],
    )
    @pytest.mark.parametrize(
        "operator",
        [
            Nearest(neighborhood=PointNeighbours(radius=1)),
            InverseDistance(neighborhood=PointNeighbours(radius=1)),
            Mean(neighborhood=PointNeighbours(radius=1)),
            Minimum(neighborhood=PointNeighbours(radius=1)),
            Maximum(neighborhood=PointNeighbours(radius=1)),
            Range(neighborhood=PointNeighbours(radius=1)),
            Count(neighborhood=PointNeighbours(radius=1)),
            StandardDeviation(neighborhood=PointNeighbours(radius=1)),
            AverageDistance(neighborhood=PointNeighbours(radius=1)),
        ],
    )
    def test_grid_from_points(self, engine: Literal["scipy", "numba"], operator: Interpolator | Reducer) -> None:
        """Checks that SciPy/Numba agree exactly, and create gaps where relevant."""

        if engine == "numba":
            import_optional("numba")

        # Two point groups near first and last row
        points = gpd.GeoDataFrame(
            {"value": [2.0, 6.0, 10.0, 14.0]},
            geometry=gpd.points_from_xy([0, 0.4, 10, 10.4], [0, 0, 0, 0]),
            crs=32631,
        )
        grid_coords = (np.array([0.0, 5.0, 10.0]), np.array([0.0, 2.0]))
        result = _grid_from_points(
            points,
            grid_coords,
            "value",
            operator,
            res_x=5,
            res_y=2,
            radius=1,
            min_points=1,
            nodata_propagation="ignore",
            engine=engine,
        )

        # Check expected statistics and NaN for empty neighborhoods
        # Only the first and last cells at Y=0 have points within the radius of 1
        assert result.shape == (len(grid_coords[1]), len(grid_coords[0]))
        assert np.isfinite(result[0, [0, -1]]).all()
        assert np.isnan(result[0, 1:-1]).all()
        assert np.isnan(result[1:]).all()

        # Compare Numba with Scipy
        if engine == "numba":
            expected = _grid_from_points(
                points,
                grid_coords,
                "value",
                operator,
                res_x=5,
                res_y=2,
                radius=1,
                min_points=1,
                nodata_propagation="ignore",
                engine="scipy",
            )
            np.testing.assert_array_equal(result, expected)
