"""Test point cloud filters on irregular neighbors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal

import geoutils as gu
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.cluster import MpCluster
from geoutils.operators import GridNeighbours, PointNeighbours
from geoutils.operators.reducer import Mean, Median, Quantile


class TestPointCloudFilter:
    """
    Test module for filtering point clouds.

    Reducer statistics are covered in test_operators/test_reducer.py.
    """

    def test_filter__radius_reducers(self) -> None:
        """Checks filters apply the radius properly."""

        # We create a synthetic point cloud with 3 nearby points and 1 far point
        x_coordinates = np.array([0.0, 1.0, 2.0, 10.0])
        heights = np.array([0.0, 100.0, 2.0, 10.0])
        frame = gpd.GeoDataFrame(
            {
                "height": heights,
                "label": list("abcd"),
            },
            geometry=gpd.points_from_xy(x_coordinates, np.zeros(4)),
            crs=32632,
        )
        points = gu.PointCloud(frame, data_column="height")

        # We applied varied methods: named median, custom reducers, one with self exclusion
        filtered = points.filter(method="median", radius=1.1)
        leave_one_out = points.filter(method=Median(), radius=1.1, include_self=False)
        lower_quartile = points.filter(method=Quantile(0.25), radius=2.1)

        # To check the statistics manually ourselves, we need to get the neighbours
        distances = np.abs(x_coordinates[:, None] - x_coordinates[None, :])
        median_neighbors = distances <= 1.1
        other_neighbors = median_neighbors & ~np.eye(len(heights), dtype=bool)
        quartile_neighbors = distances <= 2.1

        # Then the expected reduced stats on these neighbours
        expected_median = np.array([np.median(heights[nearby]) for nearby in median_neighbors])
        expected_leave_one_out = np.array(
            [np.median(heights[nearby]) if np.any(nearby) else np.nan for nearby in other_neighbors]
        )
        expected_quartile = np.array([np.quantile(heights[nearby], 0.25) for nearby in quartile_neighbors])

        # Finally, we check filtered values, coordinates and attributes
        np.testing.assert_allclose(filtered.data, expected_median)
        np.testing.assert_allclose(leave_one_out.data, expected_leave_one_out, equal_nan=True)
        np.testing.assert_allclose(lower_quartile.data, expected_quartile)
        expected_attributes = frame.drop(columns="height")
        for result in (filtered, leave_one_out, lower_quartile):
            assert_geodataframe_equal(result.ds.drop(columns="height"), expected_attributes)

    def test_filter__nearest_limit(self) -> None:
        """Checks that k selects the requested number of nearest observations when no radius is given."""

        # Synthetic points with unequal spacing to avoid nearest ties
        frame = gpd.GeoDataFrame(
            {"height": [0.0, 10.0, 40.0]},
            geometry=gpd.points_from_xy([0.0, 1.0, 4.0], [0.0] * 3),
            crs=32632,
        )
        points = gu.PointCloud(frame, data_column="height")

        # Select only 2 nearest
        filtered = points.filter(method="mean", radius=None, k=2)

        # Check first/middle and middle/last pair means
        np.testing.assert_array_equal(filtered.data, [5.0, 5.0, 25.0])

    def test_filter__missing_neigh_min_points(self) -> None:
        """Checks that handling of missing neighbours, and respect of minimum valid points for reducers."""

        # Synthetic point cloud, where the middle point with a radius of 1.1 will see 2 values, the others one
        points = gu.PointCloud.from_xyz([0.0, 1.0, 2.0], [0.0] * 3, [1.0, np.nan, 3.0], crs=32632)

        # We run the filters with different nodata propagation and min_points
        ignored = points.filter(method="mean", radius=1.1, nodata_propagation="ignore")
        propagated = points.filter(method="mean", radius=1.1, nodata_propagation="propagate")
        required = points.filter(method=Quantile(0.5), radius=1.1, min_points=2)

        # We check result is as expected
        expected_ignored = np.array([1.0, np.mean(np.array([1.0, 3.0])), 3.0])
        expected_required = np.array([np.nan, np.median(np.array([1.0, 3.0])), np.nan])
        np.testing.assert_allclose(ignored.data, expected_ignored)
        assert np.isnan(propagated.data).all()
        np.testing.assert_allclose(required.data, expected_required, equal_nan=True)

    def test_filter__reducer_neighb(self) -> None:
        """Checks that point filters use custom neighborhoods in reducers properly."""

        # Synthetic point cloud and neighborhood
        points = gu.PointCloud.from_xyz(np.array([0.0, 1.0, 4.0]), np.zeros(3), np.array([0.0, 10.0, 40.0]), crs=32631)
        neighborhood = PointNeighbours(k=2, radius=None)
        operator = Mean(neighborhood=neighborhood)

        # Different inputs to verify we can override the neighborhood with radius/k
        configured = points.filter(operator)
        limited = points.filter(operator, radius=0.5)
        unlimited = points.filter(operator, k=None, radius=5)

        # Check output
        np.testing.assert_array_equal(configured.data, [5, 5, 25])
        np.testing.assert_array_equal(limited.data, points.data)
        np.testing.assert_allclose(unlimited.data, np.mean(points.data))
        assert operator.default_neighborhood is neighborhood

    def test_filter__zero_radius(self) -> None:
        """Checks edge case of a zero radius."""

        # We create a point cloud with 2 coincident observations (within a zero radisu)
        frame = gpd.GeoDataFrame(
            {"height": [2.0, 4.0]},
            geometry=gpd.points_from_xy([0.0, 0.0], [0.0, 0.0]),
            crs=32632,
        )
        points = gu.PointCloud(frame, data_column="height")

        # Check mean is both is indeed used
        filtered = points.filter(method="mean", radius=0, k=2)
        np.testing.assert_array_equal(filtered.data, [3.0, 3.0])

    def test_filter__z_geometry(self) -> None:
        """Checks that filtering Z geometry works properly."""

        # We store elevations in geometry Z as points
        frame = gpd.GeoDataFrame(
            {"classification": np.array([1, 2, 1], dtype=np.uint8)},
            geometry=gpd.points_from_xy([0.0, 1.0, 2.0], [5.0] * 3, z=[0.0, 100.0, 2.0]),
            crs=32632,
        )
        points = gu.PointCloud(frame, data_column=None)

        # We apply neighborhood median to elevations
        filtered = points.filter(method="median", radius=1.1)

        # Check filtered Z, original X/Y and independent attribute
        np.testing.assert_array_equal(filtered.geometry.x, frame.geometry.x)
        np.testing.assert_array_equal(filtered.geometry.y, frame.geometry.y)
        np.testing.assert_array_equal(filtered.geometry.z, [50.0, 2.0, 51.0])
        np.testing.assert_array_equal(filtered["classification"], frame["classification"])

    def test_filter__temporary_id_collision(self) -> None:
        """Checks edge case that the internal ID used for filtering self doesn't collide with a user ID."""

        # We name a user attribute _geoutils_filter_id, like the filter default temporary row ID column
        heights = np.array([1.0, 3.0])
        frame = gpd.GeoDataFrame(
            {"height": heights, "_geoutils_filter_id": [10, 11]},
            geometry=gpd.points_from_xy([0.0, 1.0], [0.0, 0.0]),
            crs=32632,
        )
        points = gu.PointCloud(frame, data_column="height")

        # We exclude self during filter
        filtered = points.filter(method="mean", radius=1.1, include_self=False)

        # We check it still worked
        np.testing.assert_array_equal(filtered.data, heights[::-1])
        assert_geodataframe_equal(filtered.ds.drop(columns="height"), frame.drop(columns="height"))


class TestPointCloudFilterChunked:
    """Test module for point filter with Dask/MP."""

    points = gpd.GeoDataFrame(
        {
            "height": np.array([0.0, 100.0, 2.0, 9.0, 4.0, 8.0]),
            "row_id": np.arange(6, dtype=np.int32),
            "_geoutils_filter_id": np.arange(20, 26, dtype=np.int32),
        },
        geometry=gpd.points_from_xy(np.arange(6, dtype=float), np.zeros(6)),
        crs=32632,
    )

    def test_filter__chunked_backends_equal(self, tmp_path: Path) -> None:
        """
        Checks that filtering with Dask/MP stays lazy and matches eager results.
        """

        dgpd = pytest.importorskip("dask_geopandas")
        from dask.callbacks import Callback  # To check graph stays lazy during function call

        # Write to file
        source_filename = tmp_path / "filter_source.gpkg"
        output_filename = tmp_path / "filter_output.gpkg"
        self.points.to_file(source_filename, index=False)
        expected = gu.PointCloud(self.points, data_column="height").filter(
            method="median", radius=1.1, include_self=False
        )

        # Read in partitions
        lazy = gu.open_pointcloud(str(source_filename), data_column="height", chunks=2)
        source = gu.PointCloud(source_filename, data_column="height")

        # Dask: build filter graph and record executed tasks
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            dask_filtered = lazy.pc.filter(method="median", radius=1.1, include_self=False, batch_size=2)

        # MP: filter with two workers, then join rows in input order
        with MpCluster({"nb_workers": 2}) as cluster:
            configuration = MultiprocConfig(
                chunks=2,
                outfile=str(output_filename),
                cluster=cluster,
            )
            mp_filtered = source.filter(method="median", radius=1.1, include_self=False, mp_config=configuration)

        # We check zero executed tasks, lazy inputs/outputs and point metadata
        assert tasks == []
        assert isinstance(dask_filtered, dgpd.GeoDataFrame)
        assert not lazy.pc.is_loaded and not dask_filtered.pc.is_loaded
        assert dask_filtered.pc.data_column == "height"
        assert dask_filtered.pc.point_count == len(self.points)

        # Check unloaded inputs/outputs and metadata, then compare rows with eager filtering
        assert not source.is_loaded and not mp_filtered.is_loaded
        assert mp_filtered.point_count == len(self.points)
        assert mp_filtered.data_column == "height"

        # Finally, compare exactly equal outputs
        expected_rows = expected.ds.reset_index(drop=True)
        dask_rows = dask_filtered.compute().reset_index(drop=True)
        mp_rows = mp_filtered.ds.reset_index(drop=True)
        assert_geodataframe_equal(dask_rows, expected_rows, check_dtype=False)
        assert_geodataframe_equal(mp_rows, expected_rows, check_dtype=False)
        assert not lazy.pc.is_loaded and not dask_filtered.pc.is_loaded
        assert not source.is_loaded

    def test_filter__dask_z_geometry(self, tmp_path: Path) -> None:
        """Checks that Dask filtering updates geometry Z across partitions and matches eager values."""

        dgpd = pytest.importorskip("dask_geopandas")

        # We create 3 points a later open with a chunksize of 2
        x_coordinates = np.array([0.0, 1.0, 2.0])
        heights = np.array([2.0, 4.0, 8.0])
        frame = gpd.GeoDataFrame(
            {"label": list("abc")},
            geometry=gpd.points_from_xy(x_coordinates, np.zeros(3), z=heights),
            crs=32632,
        )
        source_filename = tmp_path / "filter_source_3d.gpkg"
        frame.to_file(source_filename, index=False)
        lazy = gu.open_pointcloud(str(source_filename), data_column=None, chunks=2)

        # We filter the same for eager/lazy
        expected = gu.PointCloud(frame, data_column=None).filter(method="mean", radius=1.1)
        filtered = lazy.pc.filter(method="mean", radius=1.1)

        # We check the output stays lazy with Z as data
        assert isinstance(filtered, dgpd.GeoDataFrame)
        assert not lazy.pc.is_loaded and not filtered.pc.is_loaded
        assert filtered.pc.data_column is None
        assert filtered.pc.point_count == len(frame)

        # And check each Z mean and original point attributes after computing
        expected_heights = np.array([np.mean(heights[np.abs(x_coordinates - x) <= 1.1]) for x in x_coordinates])
        computed = filtered.compute().reset_index(drop=True)
        np.testing.assert_allclose(computed.geometry.z, expected_heights)
        assert_geodataframe_equal(computed, expected.ds.reset_index(drop=True), check_dtype=False)
        assert not lazy.pc.is_loaded and not filtered.pc.is_loaded


class TestPointCloudFilterErrors:
    """Test module for errors on point filters."""

    @pytest.mark.parametrize(
        "options, error_type, message",
        [
            ({"method": "unknown"}, ValueError, "Unknown point filter method"),
            ({"method": Mean(neighborhood=GridNeighbours(size=3))}, TypeError, "requires PointNeighbours"),
            ({"nodata_propagation": "invalid"}, ValueError, "nodata_propagation"),
            ({"include_self": 1}, TypeError, "include_self"),
            ({"min_points": -1}, ValueError, "min_points"),
            ({"n_threads": -1}, ValueError, "n_threads"),
            ({"batch_size": 0}, ValueError, "batch_size"),
        ],
    )
    def test_filter__error_invalid_options(
        self, options: dict[str, Any], error_type: type[Exception], message: str
    ) -> None:
        """Checks an error is raised for an invalid point filter method or execution option."""

        points = gu.PointCloud.from_xyz([0.0, 1.0], [0.0, 0.0], [2.0, 4.0], crs=32632)
        with pytest.raises(error_type, match=message):
            points.filter(**options)
