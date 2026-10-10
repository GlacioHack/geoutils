"""Test point-cloud gridding values and consistency across calculation backends."""

from collections.abc import Callable, Sequence
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
import xarray as xr
from shapely import geometry

import geoutils as gu
from geoutils import PointCloud, Raster
from geoutils._typing import NDArrayNum
from geoutils.interface.gridding import GriddingMethod, _grid_pointcloud
from geoutils.multiproc import MultiprocConfig
from geoutils.operators import (
    GridNeighbours,
    Interpolator,
    LocalData,
    PointNeighbours,
    Reducer,
)
from geoutils.operators.interpolator import Cubic, InverseDistance, Kriging, Linear, Nearest, ScipyInterpolator
from geoutils.operators.nodata import NodataChoice, NodataHandling, NodataSpread
from geoutils.operators.reducer import (
    Maximum,
    Mean,
    Median,
)
from geoutils.stats.variography import VariogramModel
from tests.operator_helpers import LocalMeanInterpolator, PropagatingLocalMeanInterpolator, PropagatingMeanReducer


class TwoNearestMeanInterpolator(Interpolator):
    """Predict a value from the two nearest point observations."""

    default_neighborhood = PointNeighbours(k=2)

    def __init__(self) -> None:
        """Start with no irregular-source batches evaluated."""

        self.batch_calls = 0

    def predict(self, data: LocalData) -> float:
        """Return the unweighted mean of the two selected source values."""

        return float(np.mean(data.values))

    def predict_batch(
        self,
        data: Sequence[LocalData],
        *,
        nodata_propagation: NodataHandling | None = None,
    ) -> NDArrayNum:
        """Count calls to predict_batch() before predicting each value."""

        self.batch_calls += 1
        return super().predict_batch(data, nodata_propagation=nodata_propagation)


class TestPointCloud:
    """Test module for gridding coordinates, nodata options, validation and uncertainty output.

    Interpolation accuracy and engines are covered in test_operators/test_interpolator.py;
    point statistics and reducer engines are in test_operators/test_reducer.py.
    """

    def test_grid_pc(self) -> None:
        """Test point cloud gridding."""

        # 1/ Check gridding interpolation falls back exactly on original raster

        # Create a point cloud from interpolating a grid, so we can compare back after to check consistency
        rng = np.random.default_rng(42)
        shape = (10, 12)
        rst_arr = np.linspace(0, 10, int(np.prod(shape))).reshape(*shape)
        transform = rio.transform.from_origin(0, shape[0] - 1, 1, 1)
        rst = Raster.from_array(rst_arr, transform=transform, crs=4326, nodata=100)

        # Generate random coordinates to interpolate, to create an irregular point cloud
        points = rng.integers(low=1, high=shape[0] - 1, size=(100, 2)) + rng.normal(0, 0.15, size=(100, 2))
        b1_value = rst.interp_at_points((points[:, 0], points[:, 1]), as_array=True)
        pc = gpd.GeoDataFrame(data={"b1": b1_value}, geometry=gpd.points_from_xy(x=points[:, 0], y=points[:, 1]))
        grid_coords = rst.coords(grid=False)

        # Grid the point cloud
        gridded_pc, output_transform = _grid_pointcloud(pc, grid_coords=grid_coords, data_name="b1")

        # Compare back to raster, all should be very close (but not exact, some info is lost due to interpolations)
        valids = np.isfinite(gridded_pc)
        assert np.allclose(gridded_pc[valids], rst.data.data[valids], rtol=10e-5)
        # And the transform exactly the same
        assert output_transform == transform

        # 2/ Check the propagation of nodata values

        # 2.1/ Grid points outside the convex hull of all points should always be nodata

        # We convert the full raster to a point cloud, keeping all cells even nodata
        rst_pc = rst.to_pointcloud(skip_nodata=False).gdf

        # We define a multi-point geometry from the individual points, and compute its convex hull
        poly = geometry.MultiPoint([[p.x, p.y] for p in pc.geometry])
        chull = poly.convex_hull

        # We compute the index of grid cells intersecting the convex hull
        ind_inters_convhull = rst_pc.intersects(chull)

        # We get corresponding 1D indexes for gridded output
        i, j = rst.xy2ij(x=rst_pc.geometry.x.values, y=rst_pc.geometry.y.values)

        # Check all values outside convex hull are NaNs
        assert all(~np.isfinite(gridded_pc[i[~ind_inters_convhull], j[~ind_inters_convhull]]))

        # 2.2/ For the rest of the points, data should be valid only if a point exists within 1 pixel of their
        # coordinate, that is the closest rounded number
        # TODO: Replace by check with distance, because some pixel not rounded can also be at less than 1 from a point

        # Compute min distance to irregular point cloud for each grid point
        list_min_dist = []
        for p in rst_pc.geometry:
            min_dist = np.min(np.sqrt((p.x - pc.geometry.x.values) ** 2 + (p.y - pc.geometry.y.values) ** 2))
            list_min_dist.append(min_dist)

        ind_close = np.array(list_min_dist) <= 1
        # We get the indexes for these coordinates
        iround, jround = rst.xy2ij(x=rst_pc.geometry.x.values[ind_close], y=rst_pc.geometry.y.values[ind_close])

        # Keep only indexes in the convex hull
        indexes_close = [(iround[k], jround[k]) for k in range(len(iround))]
        indexes_chull = [(i[k], j[k]) for k in range(len(i)) if ind_inters_convhull[k]]
        close_in_chull = [tup for tup in indexes_close if tup in indexes_chull]
        iclosechull, jclosehull = list(zip(*close_in_chull))

        # All values close to pixel in the convex hull should be valid
        assert all(np.isfinite(gridded_pc[iclosechull, jclosehull]))

        # Other values in the convex hull should not be
        far_in_chull = [tup for tup in indexes_chull if tup not in indexes_close]
        ifarchull, jfarchull = list(zip(*far_in_chull))

        assert all(~np.isfinite(gridded_pc[ifarchull, jfarchull]))

        # Check for a different distance value
        gridded_pc, output_transform = _grid_pointcloud(
            pc, grid_coords=grid_coords, dist_nodata_pixel=0.5, data_name="b1"
        )
        ind_close = np.array(list_min_dist) <= 0.5

        # We get the indexes for these coordinates
        iround, jround = rst.xy2ij(x=rst_pc.geometry.x.values[ind_close], y=rst_pc.geometry.y.values[ind_close])

        # Keep only indexes in the convex hull
        indexes_close = [(iround[k], jround[k]) for k in range(len(iround))]
        indexes_chull = [(i[k], j[k]) for k in range(len(i)) if ind_inters_convhull[k]]
        close_in_chull = [tup for tup in indexes_close if tup in indexes_chull]
        iclosechull, jclosehull = list(zip(*close_in_chull))

        # All values close  pixel in the convex hull should be valid
        assert all(np.isfinite(gridded_pc[iclosechull, jclosehull]))

        # Other values in the convex hull should not be
        far_in_chull = [tup for tup in indexes_chull if tup not in indexes_close]
        ifarchull, jfarchull = list(zip(*far_in_chull))

        assert all(~np.isfinite(gridded_pc[ifarchull, jfarchull]))

        # Infinite support skips distance filtering but must match a sufficiently large finite cutoff
        finite_support, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            data_name="b1",
            resampling="nearest",
            dist_nodata_pixel=1e9,
        )
        infinite_support, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            data_name="b1",
            resampling="nearest",
            dist_nodata_pixel=float("inf"),
        )
        assert np.array_equal(finite_support, infinite_support, equal_nan=True)

        # 3/ Errors
        with pytest.raises(TypeError, match="Input grid coordinates must be 1D arrays.*"):
            Raster.from_pointcloud_regular(pc, grid_coords=(1, "lol"))  # type: ignore
        with pytest.raises(ValueError, match="Grid coordinates must be regular*"):
            grid_coords[0][0] += 1
            Raster.from_pointcloud_regular(pc, grid_coords=grid_coords)  # type: ignore

    @pytest.mark.parametrize(
        ("resampling", "spread", "expected_invalid"),
        [
            ("linear", "half_order_down", 1),
            ("linear", "half_order_up", 5),
            ("cubic", "half_order_down", 5),
            ("cubic", "half_order_up", 13),
            ("linear", 2, 13),
        ],
    )
    def test_grid_pc__nodata_spread(
        self,
        resampling: GriddingMethod,
        spread: NodataSpread,
        expected_invalid: int,
    ) -> None:
        """Checks that fixed and half-order distances expand a gridded nodata mask by the requested radius."""

        # Place one invalid observation at the center of an otherwise complete regular point grid
        x, y = np.meshgrid(np.arange(7, dtype=float), np.arange(7, dtype=float))
        values = np.arange(49, dtype=float)
        values[24] = np.nan
        point_cloud = gpd.GeoDataFrame(
            data={"z": values},
            geometry=gpd.points_from_xy(x=x.ravel(), y=y.ravel()),
        )
        grid_coords = (np.arange(7, dtype=float), np.arange(7, dtype=float))

        # Resolve the method-dependent distance and count cells inside its output-pixel radius
        result, _ = _grid_pointcloud(
            point_cloud,
            grid_coords=grid_coords,
            data_name="z",
            resampling=resampling,
            dist_nodata_pixel=np.inf,
            nodata_handling=spread,
        )
        assert np.count_nonzero(np.isnan(result)) == expected_invalid

    def test_grid_pc__zero_distance_replaces_nearest_rule(self) -> None:
        """Checks that zero-distance masking leaves a nearby cell valid even when GDAL would mask it."""

        # The missing point is nearest to both output cells, while the finite point can supply their values
        point_cloud = gpd.GeoDataFrame(
            data={"z": [np.nan, 5.0]},
            geometry=gpd.points_from_xy(x=[0.0, 2.0], y=[0.0, 0.0]),
        )
        grid_coords = (np.array([0.0, 0.4]), np.array([0.0]))

        # Nearest masks both cells, but a zero-pixel distance masks only the cell at the missing point
        nearest_result, _ = _grid_pointcloud(
            point_cloud,
            grid_coords=grid_coords,
            grid_res=(0.4, 1.0),
            data_name="z",
            resampling="nearest",
            dist_nodata_pixel=np.inf,
            nodata_handling="nearest",
        )
        zero_distance_result, _ = _grid_pointcloud(
            point_cloud,
            grid_coords=grid_coords,
            grid_res=(0.4, 1.0),
            data_name="z",
            resampling="nearest",
            dist_nodata_pixel=np.inf,
            nodata_handling=0,
        )
        assert np.isnan(nearest_result).all()
        assert np.isnan(zero_distance_result[0, 0])
        assert zero_distance_result[0, 1] == 5.0

    @pytest.mark.parametrize("resampling", ["linear", "cubic", "average_distance_pts"])
    def test_grid_pc__numba_unsupported_method(self, resampling: GriddingMethod) -> None:
        """Raise a clear error for gridding methods without a Numba implementation."""

        pc = gpd.GeoDataFrame(data={"z": [1.0]}, geometry=gpd.points_from_xy(x=[0.0], y=[0.0]))
        grid_coords = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))

        with pytest.raises(ValueError, match="Numba gridding engine does not support"):
            _grid_pointcloud(
                pc,
                grid_coords=grid_coords,
                data_name="z",
                resampling=resampling,
                engine="numba",
            )

    @pytest.mark.skipif(find_spec("numba") is not None, reason="Only runs if numba is missing.")
    def test_grid_pc__numba_missing_dependency(self) -> None:
        """Raise the standard optional-dependency error when Numba is unavailable."""

        pc = gpd.GeoDataFrame(data={"z": [1.0]}, geometry=gpd.points_from_xy(x=[0.0], y=[0.0]))
        with pytest.raises(ImportError, match="Optional dependency 'numba' required"):
            _grid_pointcloud(
                pc,
                grid_coords=(np.array([0.0, 1.0]), np.array([0.0, 1.0])),
                data_name="z",
                resampling="nearest",
                engine="numba",
            )

    def test_grid_pc__neighborhood_errors(self) -> None:
        """Check explicit validation for unsupported neighborhood definitions."""

        pc = gpd.GeoDataFrame(data={"z": [1.0]}, geometry=gpd.points_from_xy(x=[0.0], y=[0.0]))
        grid_coords = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))

        # Local aggregation cannot have infinite support without building all point-cell pairs
        with pytest.raises(ValueError, match="require a finite dist_nodata_pixel"):
            _grid_pointcloud(
                pc,
                grid_coords=grid_coords,
                data_name="z",
                resampling="mean",
                dist_nodata_pixel=float("inf"),
            )
        with pytest.raises(ValueError, match="distance_power must be finite and strictly positive"):
            _grid_pointcloud(
                pc,
                grid_coords=grid_coords,
                data_name="z",
                resampling="idw",
                distance_power=0,
            )
        with pytest.raises(ValueError, match="min_points.*non-negative integer"):
            _grid_pointcloud(
                pc,
                grid_coords=grid_coords,
                data_name="z",
                resampling="count",
                min_points=-1,
            )
        with pytest.raises(ValueError, match="nodata_handling must be"):
            _grid_pointcloud(
                pc,
                grid_coords=grid_coords,
                data_name="z",
                nodata_handling="invalid",  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="engine.*either 'scipy' or 'numba'"):
            _grid_pointcloud(
                pc,
                grid_coords=grid_coords,
                data_name="z",
                engine="invalid",  # type: ignore[arg-type]
            )

    def test_grid__propagates_reducer_uncertainty(self) -> None:
        """Checks that grid() leaves values unchanged and calculates each neighborhood's uncertainty."""

        # Arrange six independent observations on the same two-by-three grid requested for the output
        x, y = np.meshgrid(np.arange(3, dtype=float), np.arange(2, dtype=float))
        points = PointCloud(
            gpd.GeoDataFrame(
                {"z": np.arange(6, dtype=float)},
                geometry=gpd.points_from_xy(x=x.ravel(), y=y.ravel()),
                crs=32631,
            ),
            data_name="z",
        )
        grid_coords = (np.arange(3, dtype=float), np.arange(2, dtype=float))
        source_error = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # Request uncertainty with the mean method string and compare its raster with the usual gridding result
        expected = points.grid(grid_coords=grid_coords, resampling="mean", dist_nodata_pixel=1.1)
        nominal = points.grid(
            grid_coords=grid_coords, resampling="mean", dist_nodata_pixel=1.1, error_structure=source_error
        )
        summary = gu.uncertainty.propagate(
            points.grid,
            error_structure=source_error,
            operation_kwargs={"grid_coords": grid_coords, "resampling": "mean", "dist_nodata_pixel": 1.1},
        )
        # The local weighted calculation and vectorized mean can differ by rounding in their summation order
        np.testing.assert_allclose(nominal.to_nanarray(), expected.to_nanarray(), rtol=1e-14)
        np.testing.assert_array_equal(summary.estimate.to_nanarray(), nominal.to_nanarray())

        # The top-left mean uses three points and the top-middle mean uses four; two points contribute to both
        np.testing.assert_allclose(summary.variance.to_nanarray().reshape(-1)[:2], [4 / 3, 1])


class TestGridOperatorExecution:
    """Test module for built-in and custom operators used by point-cloud grid()."""

    @pytest.mark.parametrize(
        ("method", "operator_factory"),
        [
            ("nearest", Nearest),
            ("linear", Linear),
            ("cubic", Cubic),
            ("idw", InverseDistance),
            ("mean", Mean),
            ("maximum", Maximum),
        ],
    )
    def test_grid__built_in_operator_matches_string(
        self,
        method: str,
        operator_factory: Callable[[], Interpolator | Reducer],
    ) -> None:
        """Checks that built-in operator objects and their method strings produce exactly the same grid."""

        # Use an irregular finite point set spanning every target cell and supporting cubic triangulation
        rng = np.random.default_rng(42)
        coordinates = rng.uniform(0, 4, size=(40, 2))
        values = coordinates[:, 0] + 2 * coordinates[:, 1]
        points = gpd.GeoDataFrame(
            {"value": values},
            geometry=gpd.points_from_xy(coordinates[:, 0], coordinates[:, 1]),
            crs=4326,
        )
        grid_coords = (np.arange(0.5, 4, 1.0), np.arange(0.5, 4, 1.0))

        # Both public forms must use the same SciPy calculation without altering any output value
        expected, _ = _grid_pointcloud(
            points,
            grid_coords=grid_coords,
            data_name="value",
            resampling=method,  # type: ignore[arg-type]
            dist_nodata_pixel=2,
        )
        result, _ = _grid_pointcloud(
            points,
            grid_coords=grid_coords,
            data_name="value",
            resampling=operator_factory(),
            dist_nodata_pixel=2,
        )
        np.testing.assert_array_equal(result, expected)

    def test_grid__custom_reducer(self) -> None:
        """Checks that grid() applies a custom reducer to points within the requested radius."""

        # Place three values around one target, including a large value outside the one-pixel radius
        points = gpd.GeoDataFrame(
            {"value": [1.0, 5.0, 100.0]},
            geometry=gpd.points_from_xy([0.0, 0.5, 3.0], [0.0, 0.0, 0.0]),
            crs=4326,
        )
        grid_coords = (np.array([0.0]), np.array([0.0]))

        # grid() calls Median.reduce() for each neighborhood because there is no specialized gridding function for it
        result, _ = _grid_pointcloud(
            points,
            grid_coords=grid_coords,
            grid_res=(1.0, 1.0),
            data_name="value",
            resampling=Median(),
            dist_nodata_pixel=1,
        )
        assert result[0, 0] == 3.0

    def test_grid__custom_interpolator(self) -> None:
        """Checks that grid() selects the requested nearest neighbors before calling predict()."""

        # Put values 2 and 6 closest to the target and a third unrelated value farther away
        points = gpd.GeoDataFrame(
            {"value": [2.0, 6.0, 100.0]},
            geometry=gpd.points_from_xy([0.0, 0.5, 4.0], [0.0, 0.0, 0.0]),
            crs=4326,
        )
        grid_coords = (np.array([0.0]), np.array([0.0]))

        # grid() selects the two nearest points and the custom method returns their mean
        operator = TwoNearestMeanInterpolator()
        result, _ = _grid_pointcloud(
            points,
            grid_coords=grid_coords,
            grid_res=(1.0, 1.0),
            data_name="value",
            resampling=operator,
        )
        assert result[0, 0] == 4.0
        assert operator.batch_calls == 1

    def test_grid__custom_interpolator_gdal_nodata(self) -> None:
        """Checks that irregular interpolation applies GDAL's nearest-source nodata rule around a custom method."""

        # Make the nearest observation invalid while leaving the second neighbor available to the custom method
        points = gpd.GeoDataFrame(
            {"value": [np.nan, 6.0, 100.0]},
            geometry=gpd.points_from_xy([0.0, 0.5, 4.0], [0.0, 0.0, 0.0]),
            crs=4326,
        )
        grid_coords = (np.array([0.0]), np.array([0.0]))

        # The default masks the result because the nearest source is invalid; ignore uses the valid neighbor's value
        default, _ = _grid_pointcloud(
            points,
            grid_coords=grid_coords,
            grid_res=(1.0, 1.0),
            data_name="value",
            resampling=TwoNearestMeanInterpolator(),
        )
        ignored, _ = _grid_pointcloud(
            points,
            grid_coords=grid_coords,
            grid_res=(1.0, 1.0),
            data_name="value",
            resampling=TwoNearestMeanInterpolator(),
            nodata_handling="ignore",
        )
        assert np.isnan(default[0, 0])
        assert ignored[0, 0] == 6.0

    @pytest.mark.parametrize(
        ("method", "operator_factory"),
        [("nearest", Nearest), ("idw", InverseDistance), ("mean", Mean)],
    )
    def test_grid__numba_operator_matches_string(
        self,
        method: str,
        operator_factory: Callable[[], Interpolator | Reducer],
    ) -> None:
        """Checks that operator objects and their method strings use the same compiled Numba kernels."""

        pytest.importorskip("numba")
        coordinates = np.array([[0.0, 0.0], [0.4, 0.2], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        points = gpd.GeoDataFrame(
            {"value": np.array([1.0, 3.0, 5.0, 7.0, 9.0])},
            geometry=gpd.points_from_xy(coordinates[:, 0], coordinates[:, 1]),
            crs=4326,
        )
        grid_coords = (np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]))

        # Compare the object and string forms through the same compiled batch calculation
        expected, _ = _grid_pointcloud(
            points,
            grid_coords=grid_coords,
            data_name="value",
            resampling=method,  # type: ignore[arg-type]
            dist_nodata_pixel=1,
            engine="numba",
        )
        result, _ = _grid_pointcloud(
            points,
            grid_coords=grid_coords,
            data_name="value",
            resampling=operator_factory(),
            dist_nodata_pixel=1,
            engine="numba",
        )
        np.testing.assert_array_equal(result, expected)


class TestGriddingOperators:
    """Test module for point neighborhoods, custom gridding operators, missing values and uncertainty."""

    @pytest.mark.parametrize("operator", [PropagatingLocalMeanInterpolator(), PropagatingMeanReducer()])
    def test_grid__default_overrides_operator_rule(self, operator: Interpolator | Reducer) -> None:
        """Checks that point gridding calculates from finite values before applying its nodata rule."""

        # The output center coincides with a valid point and has one nearby point with nodata
        points = gu.PointCloud.from_xyz([0.5, 1.5], [0.5, 0.5], [2.0, np.nan], crs=32631)
        reference = gu.Raster.from_array(np.zeros((1, 1)), rio.transform.from_origin(0, 1, 1, 1), crs=32631)

        # The closest point is valid, so the nodata point does not mask this output cell
        result = points.grid(ref=reference, resampling=operator)
        assert result.to_nanarray()[0, 0] == 2.0

    @pytest.mark.parametrize(
        ("operator", "interpolate"),
        [(PropagatingLocalMeanInterpolator(), True), (PropagatingMeanReducer(), False)],
    )
    def test_grid__default_follows_custom_operator_class(
        self, operator: Interpolator | Reducer, interpolate: bool
    ) -> None:
        """Checks that a custom point interpolator masks a missing nearest point while a reducer omits it."""

        # The missing observation lies at the output center, with a finite point one unit away
        points = gu.PointCloud.from_xyz([0.5, 1.5], [0.5, 0.5], [np.nan, 2.0], crs=32631)
        reference = gu.Raster.from_array(np.zeros((1, 1)), rio.transform.from_origin(0, 1, 1, 1), crs=32631)

        # A two-pixel radius includes the finite point for the reducer; only interpolation masks the output
        result = points.grid(ref=reference, resampling=operator, dist_nodata_pixel=2.0)
        value = result.to_nanarray()[0, 0]
        if interpolate:
            assert np.isnan(value)
        else:
            assert value == 2.0

    def test_grid__automatic_and_custom_point_neighbours(self) -> None:
        """Checks that grid() uses nearby points by default or a requested count or circular distance."""

        # Place two points near the target and two farther away so their means are different
        points = gu.PointCloud.from_xyz([0, 1, 10, 20], [0, 0, 0, 0], [1, 3, 7, 11], crs=32631)
        reference = gu.Raster.from_array(
            np.zeros((1, 1)),
            transform=rio.transform.from_origin(0, 1, 1, 1),
            crs=32631,
        )
        operator = LocalMeanInterpolator()

        # The usual eight-nearest search reaches all four points; both custom searches select the two nearby points
        usual = points.grid(ref=reference, resampling=operator, nodata_handling="ignore")
        nearest_two = points.grid(
            ref=reference,
            resampling=LocalMeanInterpolator(neighborhood=PointNeighbours(k=2)),
            nodata_handling="ignore",
        )
        within_five = points.grid(
            ref=reference,
            resampling=LocalMeanInterpolator(neighborhood=PointNeighbours(radius=5)),
            nodata_handling="ignore",
        )
        assert usual.to_nanarray()[0, 0] == 5.5
        assert nearest_two.to_nanarray()[0, 0] == 2
        assert within_five.to_nanarray()[0, 0] == 2
        assert operator.default_neighborhood is None

    @pytest.mark.parametrize("neighborhood", [PointNeighbours(k=2), PointNeighbours(radius=2)])
    def test_grid__reducer_point_neighbours(self, neighborhood: PointNeighbours) -> None:
        """Checks that a reducer's point count or radius selects values and uncertainty inputs for grid()."""

        # Put two small values near the target and two larger values farther away
        points = gu.PointCloud.from_xyz([0, 1, 10, 20], [0, 0, 0, 0], [1, 3, 7, 11], crs=32631)
        reference = gu.Raster.from_array(np.zeros((1, 1)), transform=rio.transform.from_origin(0, 1, 1, 1), crs=32631)
        reducer = Mean(neighborhood=neighborhood)
        source_error = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # The configured point limits select only the first two observations, despite the larger pixel radius
        usual = points.grid(ref=reference, resampling=Mean(), dist_nodata_pixel=30, nodata_handling="ignore")
        reduced = points.grid(
            ref=reference,
            resampling=reducer,
            dist_nodata_pixel=30,
            nodata_handling="ignore",
            error_structure=source_error,
        )
        summary = gu.uncertainty.propagate(
            points.grid,
            error_structure=source_error,
            operation_kwargs={
                "ref": reference,
                "resampling": reducer,
                "dist_nodata_pixel": 30,
                "nodata_handling": "ignore",
            },
        )

        # Averaging 1 and 3 gives 2; two independent errors of magnitude two give variance 2
        assert usual.to_nanarray()[0, 0] == 5.5
        assert reduced.to_nanarray()[0, 0] == 2
        assert summary.variance.to_nanarray().reshape(-1)[0] == pytest.approx(2)
        assert reducer.default_neighborhood is neighborhood

    def test_grid__error_grid_neighborhood_for_reducer(self) -> None:
        """Checks that point cloud gridding rejects a neighborhood intended for raster cells."""

        # A small point cloud and matching raster are enough to reach neighborhood selection
        points = gu.PointCloud.from_xyz([0, 1], [0, 0], [1, 3], crs=32631)
        reference = gu.Raster.from_array(np.zeros((1, 1)), rio.transform.from_origin(0, 1, 1, 1), crs=32631)
        reducer = Mean(neighborhood=GridNeighbours(size=3))

        # GridNeighbours contains row and column offsets rather than nearby point locations
        with pytest.raises(ValueError, match="GridNeighbours applies to raster cells"):
            points.grid(ref=reference, resampling=reducer)

    def test_grid__zero_radius_includes_coincident_sources(self) -> None:
        """Checks that a zero-radius point search includes coincident sources in values and uncertainty."""

        # Place two independent observations exactly at the requested grid coordinate
        points = gu.PointCloud.from_xyz([0, 0], [0, 0], [2, 4], crs=32631)
        reference = gu.Raster.from_array(
            np.zeros((1, 1)),
            transform=rio.transform.from_origin(0, 0, 1, 1),
            crs=32631,
        )
        operator = InverseDistance(neighborhood=PointNeighbours(k=2, radius=0))
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Both zero-distance values receive half the weight, including in analytical variance
        nominal = points.grid(ref=reference, resampling=operator, nodata_handling="ignore", error_structure=errors)
        summary = gu.uncertainty.propagate(
            points.grid,
            error_structure=errors,
            operation_kwargs={"ref": reference, "resampling": operator, "nodata_handling": "ignore"},
        )
        np.testing.assert_allclose(nominal.to_nanarray(), [[3]])
        np.testing.assert_allclose(summary.variance.to_nanarray().reshape(-1), [0.5])

    def test_grid__builtin_interpolator_uses_custom_point_count(self) -> None:
        """Checks that a custom point count changes inverse-distance gridding instead of using the fast radius rule."""

        # The two nearest points have values one and three, while distant points can change a full-radius average
        points = gu.PointCloud.from_xyz([0, 1, 10, 20], [0, 0, 0, 0], [1, 3, 7, 11], crs=32631)
        reference = gu.Raster.from_array(
            np.zeros((1, 1)),
            transform=rio.transform.from_origin(0, 1, 1, 1),
            crs=32631,
        )

        # A thirty-pixel radius includes all points; k=2 uses only the two nearest ones
        full_radius = points.grid(
            ref=reference,
            resampling=InverseDistance(),
            dist_nodata_pixel=30,
            nodata_handling="ignore",
        )
        nearest_two = points.grid(
            ref=reference,
            resampling=InverseDistance(neighborhood=PointNeighbours(k=2)),
            dist_nodata_pixel=30,
            nodata_handling="ignore",
        )
        assert nearest_two.to_nanarray()[0, 0] == pytest.approx(5 / 3)
        assert full_radius.to_nanarray()[0, 0] > nearest_two.to_nanarray()[0, 0]


class TestGridOperatorNodata:
    """Test module for spatial nodata masks and contributions selected by operator neighborhoods."""

    @pytest.mark.parametrize("method", ["nearest", "linear", "cubic", "idw", "mean"])
    def test_grid__default_follows_operator_class(self, method: str) -> None:
        """Checks that point-grid interpolators mask a missing nearest point and a reducer uses finite points."""

        # Four finite corners can fill the center by every method, while the center observation is missing
        points = gu.PointCloud.from_xyz(
            [-1.0, 1.0, -1.0, 1.0, 0.0],
            [-1.0, -1.0, 1.0, 1.0, 0.0],
            [5.0, 5.0, 5.0, 5.0, np.nan],
            crs=32631,
        )
        grid_coords = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))

        # All calculations find five from finite points; only the interpolators then mask the missing center
        options: dict[str, Any] = {"grid_coords": grid_coords, "resampling": method, "dist_nodata_pixel": 2.0}
        ignored = points.grid(**options, nodata_handling="ignore").to_nanarray()
        default = points.grid(**options).to_nanarray()
        assert ignored[1, 0] == pytest.approx(5.0)
        if method == "mean":
            assert default[1, 0] == pytest.approx(5.0)
        else:
            assert np.isnan(default[1, 0])

    def test_grid__error_nearest_with_reducer(self) -> None:
        """Checks an error is raised for nearest-point masking with a reducer."""

        # A reducer calculates from a group of source points
        points = gu.PointCloud.from_xyz([0.0, 1.0], [0.0, 0.0], [1.0, 2.0], crs=32631)
        grid_coords = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))

        # The nearest rule applies to interpolators
        with pytest.raises(ValueError, match="requires an Interpolator"):
            points.grid(grid_coords=grid_coords, resampling=Mean(), nodata_handling="nearest")


@pytest.mark.skipif(find_spec("gstools") is None, reason="Requires GSTools")
class TestKrigingPoints:
    """Test module for kriging neighborhoods, missing values and propagated uncertainty."""

    def test_grid__kriging_operator_matches_point_krige(self) -> None:
        """Checks that grid() accepts a Kriging object and returns the same value as PointCloud.krige()."""

        # Place four points around one output cell and use a fitted model for both public calls
        points = gu.PointCloud.from_xyz([0, 2, 0, 2], [0, 0, 2, 2], [1, 3, 2, 5], crs=32631)
        reference = gu.Raster.from_array(
            np.zeros((1, 1)),
            transform=rio.transform.from_origin(0.5, 1.5, 1, 1),
            crs=32631,
        )
        variogram = gu.Variogram.from_model("exponential", effective_range=4, partial_sill=2)

        # The configured operator should use its point neighbors on the grid
        actual = points.grid(ref=reference, resampling=Kriging(variogram), nodata_handling="ignore")
        expected = points.krige(variogram, ref=reference)
        np.testing.assert_allclose(actual.to_nanarray(), expected.to_nanarray(), rtol=0, atol=1e-12)

    def test_krige__error_partial_spatial_dimensions(self) -> None:
        """Checks that geospatial kriging rejects a model whose radius omits one spatial coordinate."""

        # A one-dimensional covariance would require an unbounded point search along the omitted Y coordinate
        points = gu.PointCloud.from_xyz([0, 1], [0, 1], [2, 3], crs=32631)
        model = VariogramModel("gaussian", effective_range=2, partial_sill=1, active_dims=(0,))

        # Reject the incomplete neighborhood before gridding rather than silently dropping distant Y observations
        with pytest.raises(NotImplementedError, match="every spatial coordinate"):
            points.krige(model, res=1)
        with pytest.raises(NotImplementedError, match="every spatial coordinate"):
            points.grid(res=1, resampling=Kriging(model))

    def test_krige__point_method_propagates_exact_coefficients(self) -> None:
        """Checks that PointCloud.krige() uses the same weights for predicted values and their uncertainty."""

        # Place four observations around one off-center raster target so every ordinary-kriging weight matters
        coordinates = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        values = np.array([1.0, 3.0, 2.0, 5.0])
        target = np.array([0.8, 1.1])
        points = gu.PointCloud.from_xyz(coordinates[:, 0], coordinates[:, 1], values, crs=32631)
        reference = gu.Raster.from_array(
            np.zeros((1, 1)),
            transform=rio.transform.from_origin(0.8, 1.1, 1, 1),
            crs=32631,
        )
        variogram = gu.Variogram.from_model("exponential", effective_range=4, partial_sill=2)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Observation noise adds an identity matrix to the covariance used for the kriging fit
        local = LocalData(
            values=values,
            valid=np.ones(4, dtype=bool),
            source_ids=np.arange(4),
            coordinates=coordinates,
            target=target,
            distances=np.linalg.norm(coordinates - target, axis=1),
            error_covariance=np.eye(4),
        )
        coefficients = Kriging(variogram).coefficients(local)
        assert coefficients is not None
        expected_value = float(coefficients.weights @ values)
        expected_std = float(np.sqrt(coefficients.weights @ coefficients.weights))

        nominal = points.krige(ref=reference, variogram=variogram, error_structure=errors)
        summary = gu.uncertainty.propagate(
            points.krige,
            error_structure=errors,
            operation_kwargs={"ref": reference, "variogram": variogram},
        )
        assert nominal.to_nanarray()[0, 0] == pytest.approx(expected_value, rel=0, abs=1e-12)
        assert np.asarray(summary.std.to_nanarray().reshape(-1))[0] == pytest.approx(expected_std, rel=0, abs=1e-12)


@pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Only runs if dask-geopandas is installed.")
class TestGridChunked:
    """
    Test module for gridding outputs and loading across eager, Dask and Multiprocessing backends.

    Method accuracy and numerical engines are covered in test_operators/test_interpolator.py
    and test_operators/test_reducer.py.
    """

    # Use a regular point grid so every interpolation method has enough local support
    x, y = np.meshgrid(np.arange(3, dtype=float), np.arange(3, dtype=float))
    points = gpd.GeoDataFrame(
        data={"z": np.arange(9, dtype=float)},
        geometry=gpd.points_from_xy(x=x.ravel(), y=y.ravel()),
        crs=32610,
    )
    grid_coords = (np.arange(3, dtype=float), np.arange(3, dtype=float))

    @pytest.mark.parametrize(
        "resampling",
        [
            "nearest",
            "linear",
            "cubic",
            "idw",
            "mean",
            "minimum",
            "maximum",
            "range",
            "count",
            "stdev",
            "average_distance",
            "average_distance_pts",
            Mean(neighborhood=PointNeighbours(k=2)),
        ],
    )
    @pytest.mark.parametrize("as_type", ["dataarray", "geodataframe"])
    def test_grid__chunked_backends_equal(
        self, as_type: Literal["dataarray", "geodataframe"], resampling: GriddingMethod, tmp_path: Path
    ) -> None:
        """
        Test that grid returns exactly the same output for:
         - PointCloud and the Pandas accessor in memory,
         - Dask through the Pandas accessor with lazy input and output,
         - Multiprocessing through PointCloud with lazy input and output.

        The Dask and Multiprocessing inputs must remain unloaded after their results are read.
        """

        import dask.array as da

        # Store one point source for both backends that read partitions from disk
        point_file = tmp_path / "points.gpkg"
        self.points.to_file(point_file)

        # 1/ Prepare the same point cloud through every public interface
        pointcloud = PointCloud(self.points, data_name="z")
        point_accessor = self.points.copy()
        point_accessor.pc.set_data_name("z")
        dask_points = gu.open_pointcloud(str(point_file), data_name="z", chunks=3, as_type=as_type)
        multiproc_points = PointCloud(point_file, data_name="z")

        assert pointcloud.is_loaded
        assert point_accessor.pc.is_loaded
        assert not dask_points.pc.is_loaded
        assert not multiproc_points.is_loaded

        # 2/ Grid the complete source eagerly and split the other outputs into rectangular chunks
        kwargs = {
            "grid_coords": self.grid_coords,
            "resampling": resampling,
            "dist_nodata_pixel": 2,
        }
        expected = pointcloud.grid(**kwargs)
        accessor_output = point_accessor.pc.grid(**kwargs)
        dask_output = dask_points.pc.grid(**kwargs, chunksizes=(2, 1))
        multiproc_output = multiproc_points.grid(
            **kwargs,
            mp_config=MultiprocConfig(chunks=(2, 1), outfile=str(tmp_path / "grid-multiproc.tif")),
        )

        # 3/ Check output types and loading before evaluating the chunked results
        assert isinstance(expected, Raster)
        assert expected.is_loaded
        assert isinstance(accessor_output, xr.DataArray)
        assert accessor_output._in_memory
        assert isinstance(dask_output, xr.DataArray)
        assert isinstance(dask_output.data, da.Array)
        assert not dask_output._in_memory
        assert dask_output.data.chunks == ((2, 1), (1, 1, 1))
        assert isinstance(multiproc_output, Raster)
        assert not multiproc_output.is_loaded

        # 4/ Read the results and require exact values, masks and georeferencing
        computed_dask = dask_output.compute()
        multiproc_output.load()
        assert expected.raster_equal(accessor_output, warn_failure_reason=True, strict_masked=False)
        assert expected.raster_equal(computed_dask, warn_failure_reason=True, strict_masked=False)
        assert expected.raster_equal(multiproc_output, warn_failure_reason=True, strict_masked=False)

        # 5/ Computing an output must not replace either lazy source with in-memory data
        assert not dask_points.pc.is_loaded
        assert not multiproc_points.is_loaded
        assert not dask_output._in_memory
        assert multiproc_output.is_loaded

    @pytest.mark.parametrize("point_dask", [False, True], ids=["point-eager", "point-dask"])
    @pytest.mark.parametrize("raster_dask", [False, True], ids=["raster-eager", "raster-dask"])
    @pytest.mark.parametrize("as_type", ["dataarray", "geodataframe"])
    def test_grid__point_raster_input_combinations(
        self,
        as_type: Literal["dataarray", "geodataframe"],
        point_dask: bool,
        raster_dask: bool,
        tmp_path: Path,
    ) -> None:
        """
        Checks that grid() returns a lazy output when the point source or raster reference uses Dask.
        """

        import dask.array as da

        # Write both inputs so their Dask variants use the same values and georeferencing
        point_file = tmp_path / "points.gpkg"
        self.points.to_file(point_file)
        # Match the requested grid coordinates to the source points so nearest neighbors have no ties
        reference = Raster.from_array(
            np.zeros((3, 3), dtype=np.uint8),
            transform=rio.transform.from_origin(0, 2, 1, 1),
            crs=self.points.crs,
        )
        raster_file = tmp_path / "reference.tif"
        reference.to_file(raster_file)

        # Select each input independently to cover all four eager and Dask combinations
        points = (
            gu.open_pointcloud(str(point_file), data_name="z", chunks=3, as_type=as_type)
            if point_dask
            else PointCloud(self.points, data_name="z")
        )
        raster = gu.open_raster(str(raster_file), chunks={"x": 2, "y": 2}) if raster_dask else reference
        expected = PointCloud(self.points, data_name="z").grid(
            ref=reference,
            resampling="nearest",
            dist_nodata_pixel=2,
        )

        # Either lazy input selects Dask output, while two eager inputs return an eager raster
        output = (
            points.pc.grid(ref=raster, resampling="nearest", dist_nodata_pixel=2)
            if point_dask
            else points.grid(
                ref=raster,
                resampling="nearest",
                dist_nodata_pixel=2,
            )
        )
        if point_dask or raster_dask:
            assert isinstance(output, xr.DataArray)
            assert isinstance(output.data, da.Array)
            assert not output._in_memory
            computed_output = output.compute()
            if point_dask:
                assert not points.pc.is_loaded
        else:
            assert isinstance(output, Raster)
            assert output.is_loaded
            computed_output = output

        # A Dask reference supplies its spatial chunks without reading any of its raster values
        if raster_dask:
            assert isinstance(raster.data, da.Array)
            assert not raster._in_memory
            assert output.data.chunks == raster.data.chunks

        # Grid coordinates coincide with source points; raster rows reverse the points' increasing Y order
        expected_values = self.points["z"].to_numpy().reshape(3, 3)[::-1]
        np.testing.assert_array_equal(computed_output.data, expected_values)
        assert expected.raster_equal(computed_output, warn_failure_reason=True, strict_masked=False)

    @pytest.mark.parametrize("resampling", ["nearest", "idw", "mean"])
    @pytest.mark.parametrize("as_type", ["dataarray", "geodataframe"])
    def test_grid__numba_chunked_backends(
        self, as_type: Literal["dataarray", "geodataframe"], resampling: GriddingMethod, tmp_path: Path
    ) -> None:
        """Ensure the Numba calculation engine is identical across eager, Dask and Multiprocessing backends."""

        pytest.importorskip("numba")

        # Store one point source so both chunked backends can select local points
        point_file = tmp_path / "points.gpkg"
        self.points.to_file(point_file)
        kwargs = {
            "grid_coords": self.grid_coords,
            "resampling": resampling,
            "dist_nodata_pixel": 2,
            "engine": "numba",
        }

        # Compare both chunked outputs with one complete eager calculation
        expected = PointCloud(self.points, data_name="z").grid(**kwargs)
        dask_points = gu.open_pointcloud(str(point_file), data_name="z", chunks=3, as_type=as_type)
        dask_output = dask_points.pc.grid(**kwargs, chunksizes=(2, 1))
        multiproc_points = PointCloud(point_file, data_name="z")
        multiproc_output = multiproc_points.grid(
            **kwargs,
            mp_config=MultiprocConfig(chunks=(2, 1), outfile=str(tmp_path / "grid-numba.tif")),
        )
        assert expected.raster_equal(dask_output.compute(), warn_failure_reason=True, strict_masked=False)
        assert expected.raster_equal(multiproc_output, warn_failure_reason=True, strict_masked=False)
        assert not dask_points.pc.is_loaded
        assert not multiproc_points.is_loaded

    @pytest.mark.parametrize(
        ("nodata_handling", "resampling"),
        [("propagate", "mean"), (2, "mean"), ("nearest", "idw"), ("ignore", "mean")],
    )
    @pytest.mark.parametrize("as_type", ["dataarray", "geodataframe"])
    def test_grid__nodata_propagation_chunked_backends(
        self,
        as_type: Literal["dataarray", "geodataframe"],
        tmp_path: Path,
        nodata_handling: NodataChoice,
        resampling: GriddingMethod,
    ) -> None:
        """Checks that each nodata choice gives identical results across gridding backends."""

        # Add one invalid center to exercise support across output chunk boundaries
        points = self.points.copy()
        points.loc[4, "z"] = np.nan
        point_file = tmp_path / "points-nodata.gpkg"
        points.to_file(point_file)
        kwargs = {
            "grid_coords": self.grid_coords,
            "resampling": resampling,
            "dist_nodata_pixel": 1.1,
            "nodata_handling": nodata_handling,
        }

        # Compare the same nodata rule before and after splitting either input or output
        expected = PointCloud(points, data_name="z").grid(**kwargs)
        dask_points = gu.open_pointcloud(str(point_file), data_name="z", chunks=3, as_type=as_type)
        dask_output = dask_points.pc.grid(**kwargs, chunksizes=(2, 2))
        multiproc_points = PointCloud(point_file, data_name="z")
        multiproc_output = multiproc_points.grid(
            **kwargs,
            mp_config=MultiprocConfig(chunks=(2, 2), outfile=str(tmp_path / "grid-nodata.tif")),
        )
        assert expected.raster_equal(dask_output.compute(), warn_failure_reason=True, strict_masked=False)
        assert expected.raster_equal(multiproc_output, warn_failure_reason=True, strict_masked=False)
        assert not dask_points.pc.is_loaded
        assert not multiproc_points.is_loaded

    @pytest.mark.parametrize("as_type", ["dataarray", "geodataframe"])
    def test_grid__empty_dask_partitions(self, as_type: Literal["dataarray", "geodataframe"], tmp_path: Path) -> None:
        """Checks that chunked gridding returns empty Dask partitions when points are unreachable."""

        # We create 5 points into partitions of 2/2/1
        points = gpd.GeoDataFrame(
            {"z": np.arange(5, dtype=float)},
            geometry=gpd.points_from_xy(np.arange(5, dtype=float), np.zeros(5)),
            crs=32631,
        )
        filename = tmp_path / "distant-points.gpkg"
        points.to_file(filename, index=False)
        source = gu.open_pointcloud(str(filename), data_name="z", chunks=2, as_type=as_type)

        # We define a grid destination that is beyond the 1-pixel radius of every source point
        reference = Raster.from_array(np.zeros((3, 4)), rio.transform.from_origin(100, 3, 1, 1), crs=32631)
        options = {"ref": reference, "resampling": "nearest", "dist_nodata_pixel": 1}
        expected = PointCloud(points, data_name="z").grid(**options)
        result = source.pc.grid(**options, chunksizes=(2, 3))

        # Check lazy input/output, then that all are NaNs
        assert not source.pc.is_loaded
        assert hasattr(result.data, "compute")
        computed = result.compute()
        assert np.isnan(expected.to_nanarray()).all()
        np.testing.assert_array_equal(np.asarray(computed), expected.to_nanarray())
        assert not source.pc.is_loaded

    @pytest.mark.parametrize("as_type", ["dataarray", "geodataframe"])
    def test_grid__dask_multiprocessing_error(
        self, as_type: Literal["dataarray", "geodataframe"], tmp_path: Path
    ) -> None:
        """Reject two schedulers for one gridding operation before evaluating point partitions."""

        # A Dask point source already owns task scheduling and cannot use Multiprocessing
        point_file = tmp_path / "points.gpkg"
        self.points.to_file(point_file)
        points = gu.open_pointcloud(str(point_file), data_name="z", chunks=3, as_type=as_type)
        with pytest.raises(ValueError, match="Cannot use Multiprocessing and Dask simultaneously"):
            points.pc.grid(
                grid_coords=self.grid_coords,
                mp_config=MultiprocConfig(chunks=(2, 2), outfile=str(tmp_path / "grid-error.tif")),
            )
        assert not points.pc.is_loaded

    def test_grid__error_dask_reference_with_multiprocessing(self, tmp_path: Path) -> None:
        """
        Checks that an eager point source with a Dask raster reference rejects multiprocessing before computing.
        """

        # Place source points at the requested grid coordinates and open the reference with spatial chunks
        points = PointCloud(self.points, data_name="z")
        reference_file = tmp_path / "reference.tif"
        reference = Raster.from_array(
            np.zeros((3, 3), dtype=np.uint8), rio.transform.from_origin(0, 2, 1, 1), self.points.crs
        )
        reference.to_file(reference_file)
        lazy_reference = gu.open_raster(str(reference_file), chunks={"x": 2, "y": 2})

        # Reject competing schedulers without loading the reference or creating a multiprocessing output
        output_file = tmp_path / "grid-error.tif"
        config = MultiprocConfig(chunks=(2, 2), outfile=str(output_file))
        with pytest.raises(ValueError, match="Cannot use Multiprocessing and Dask simultaneously"):
            points.grid(ref=lazy_reference, mp_config=config)
        assert not lazy_reference._in_memory
        assert not output_file.exists()


@pytest.mark.skipif(find_spec("dask") is None, reason="Requires Dask")
class TestGriddingOperatorsChunked:
    """Test module for eager and lazy agreement when gridding operator neighborhoods."""

    def test_grid__automatic_point_neighbours_chunk_invariance(self) -> None:
        """Checks that automatic point searches give the same eager and lazy grid values."""

        # Four points feed a four by four output grid split into 2 x 3 cell chunks
        points = gu.PointCloud.from_xyz([0, 1, 3, 6], [0, 0, 0, 0], [1, 3, 7, 11], crs=32631)
        reference = gu.Raster.from_array(
            np.zeros((4, 4)),
            transform=rio.transform.from_origin(0, 4, 1, 1),
            crs=32631,
        )
        operator = LocalMeanInterpolator()

        # The lazy reference chooses Dask, while the point source remains available for each output chunk
        expected = points.grid(ref=reference, resampling=operator, nodata_handling="ignore")
        lazy_reference = reference.to_xarray().chunk({"y": 2, "x": 3})
        lazy_result = points.grid(ref=lazy_reference, resampling=operator, nodata_handling="ignore")
        assert hasattr(lazy_reference.data, "compute")
        assert hasattr(lazy_result.data, "compute")

        # Computing the grid must agree with the complete eager point search
        np.testing.assert_array_equal(np.asarray(lazy_result.compute()), expected.to_nanarray())

    @pytest.mark.parametrize("as_type", ["dataarray", "geodataframe"])
    def test_grid__custom_reducer_chunk_invariance(
        self, as_type: Literal["dataarray", "geodataframe"], tmp_path: Path
    ) -> None:
        """Checks that a custom reducer receives the same neighbors with eager, Dask and multiprocessing grids."""

        pytest.importorskip("dask_geopandas")

        # Store an irregular source whose circular neighborhoods cross every rectangular output chunk boundary
        coordinates = np.array([[0.0, 0.0], [0.4, 0.2], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        points = gpd.GeoDataFrame(
            {"value": np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0])},
            geometry=gpd.points_from_xy(coordinates[:, 0], coordinates[:, 1]),
            crs=32632,
        )
        point_file = tmp_path / "custom-reducer-points.gpkg"
        points.to_file(point_file)
        grid_coords = (np.arange(3, dtype=float), np.arange(2, dtype=float))
        options = {"grid_coords": grid_coords, "resampling": Median(), "dist_nodata_pixel": 1.1}

        # Evaluate one complete grid and two grids whose output cells are split independently
        expected = gu.PointCloud(points, data_name="value").grid(**options)
        dask_points = gu.open_pointcloud(str(point_file), data_name="value", chunks=3, as_type=as_type)
        dask_result = dask_points.pc.grid(**options, chunksizes=(1, 2))
        assert hasattr(dask_points, "compute")
        assert hasattr(dask_result.data, "compute")
        dask_result = dask_result.compute()
        multiproc_points = gu.PointCloud(point_file, data_name="value")
        multiproc_result = multiproc_points.grid(
            **options,
            mp_config=MultiprocConfig(chunks=(2, 1), outfile=str(tmp_path / "custom-reducer-grid.tif")),
        )

        # Compare values exactly to check that chunk boundaries do not change which neighbors are selected
        assert not multiproc_points.is_loaded
        assert not multiproc_result.is_loaded
        assert expected.raster_equal(dask_result, warn_failure_reason=True, strict_masked=False)
        assert expected.raster_equal(multiproc_result, warn_failure_reason=True, strict_masked=False)


@pytest.mark.skipif(find_spec("dask") is None, reason="Requires Dask")
class TestPointNeighbourMethodsChunked:
    """Test module for lazy and multiprocessing grids with explicit neighborhoods and either numerical engine."""

    @pytest.mark.parametrize("engine", ["scipy", "numba"])
    @pytest.mark.parametrize("operator_type", [Mean, InverseDistance])
    def test_grid__explicit_neighborhood_chunk_invariance(
        self, engine: Literal["scipy", "numba"], operator_type: type[Reducer] | type[Interpolator], tmp_path: Path
    ) -> None:
        """Checks that count-limited neighborhoods cross output chunk boundaries without changing the grid."""

        if engine == "numba":
            pytest.importorskip("numba")

        # An uneven final chunk checks that the point search uses the complete source at each boundary
        points = gu.PointCloud.from_xyz([0, 1, 3, 6], [0, 0, 1, 2], [1, 3, 7, 11], crs=32631)
        reference = gu.Raster.from_array(np.zeros((4, 5)), rio.transform.from_origin(0, 4, 1, 1), crs=32631)
        operator = operator_type(neighborhood=PointNeighbours(k=2, radius=4))
        options: dict[str, Any] = {"resampling": operator, "engine": engine, "nodata_handling": "ignore"}
        expected = points.grid(ref=reference, **options)

        # Dask returns a lazy array; multiprocessing writes its output without loading the returned raster
        lazy_reference = reference.to_xarray().chunk({"y": 3, "x": 2})
        lazy_result = points.grid(ref=lazy_reference, **options)
        assert hasattr(lazy_reference.data, "compute")
        assert hasattr(lazy_result.data, "compute")
        result = points.grid(
            ref=reference,
            **options,
            mp_config=MultiprocConfig(chunks=(2, 3), outfile=str(tmp_path / "point-neighborhood.tif")),
        )
        assert points.is_loaded
        assert not result.is_loaded
        np.testing.assert_array_equal(np.asarray(lazy_result.compute()), expected.to_nanarray())
        np.testing.assert_array_equal(result.to_nanarray(), expected.to_nanarray())


@pytest.mark.skipif(find_spec("dask") is None, reason="Requires Dask")
@pytest.mark.skipif(find_spec("gstools") is None, reason="Requires GSTools")
class TestKrigingPointsChunked:
    """Test module for lazy and multiprocessing kriging across chunk boundaries."""

    def test_krige__point_chunk_invariance(self) -> None:
        """Checks that eager and Dask kriging use the same neighbors within the requested radius."""

        # Place a regular set of known points around a four by four destination grid
        x, y = np.meshgrid(np.arange(5, dtype=float), np.arange(5, dtype=float))
        values = 2 * x.reshape(-1) - y.reshape(-1)
        points = gu.PointCloud.from_xyz(x.reshape(-1), y.reshape(-1), values, crs=32631)
        reference = gu.Raster.from_array(
            np.zeros((4, 4)),
            transform=rio.transform.from_origin(0.5, 4.5, 1, 1),
            crs=32631,
        )
        variogram = gu.Variogram.from_model("exponential", effective_range=2.1, partial_sill=1)

        # Split the output into chunks of 2 x 3 cells, so neighbors cross chunk boundaries and the last chunk is shorter
        expected = points.krige(variogram, ref=reference, max_overlap=2.1)
        lazy_reference = reference.to_xarray().chunk({"y": 2, "x": 3})
        lazy = points.krige(variogram, ref=lazy_reference, max_overlap=2.1)
        assert hasattr(lazy.data, "compute")
        np.testing.assert_allclose(np.asarray(lazy.compute()), expected.to_nanarray(), rtol=0, atol=1e-12)


class TestPointNeighbourMethods:
    """Test module for rejecting raster interpolation methods on point sources.

    Selected-point accuracy and SciPy/Numba comparisons are covered in test_operators/test_interpolator.py
    and test_operators/test_reducer.py.
    """

    @pytest.mark.parametrize("method", ["slinear", "pchip", "quintic", "splinef2d"])
    def test_grid__error_regular_only_method(self, method: str) -> None:
        """Checks that point gridding rejects SciPy methods that require a regular input grid."""

        # Reject the method before trying to prepare a triangulation or a point neighborhood
        points = gu.PointCloud.from_xyz([0, 1, 0], [0, 0, 1], [1, 2, 3], crs=32631)
        with pytest.raises(ValueError, match="regular-grid SciPy method"):
            points.grid(grid_coords=(np.arange(2.0), np.arange(2.0)), resampling=ScipyInterpolator(method))  # type: ignore[arg-type]
