"""Test point cloud filters on irregular neighbors and across data partitions."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal

import geoutils as gu
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.cluster import MpCluster
from geoutils.operators import PointNeighbours
from geoutils.operators.reducer import Mean, Median, Quantile


class TestPointCloudFilter:
    """
    Test module for point neighborhood options, attributes and source row selection.

    Reducer statistics, median options and the PDAL reference are covered in test_operators/test_reducer.py.
    """

    def test_filter__radius_reducers_and_attributes(self) -> None:
        """Checks that radius filters apply named and custom reducers while preserving point rows and attributes."""

        # User attributes, including name matching temporary filter ID
        frame = gpd.GeoDataFrame(
            {
                "height": [0.0, 100.0, 2.0, 10.0],
                "label": list("abcd"),
                "_geoutils_filter_id": [10, 11, 12, 13],
            },
            geometry=gpd.points_from_xy([0.0, 1.0, 2.0, 10.0], [0.0] * 4),
            crs=32632,
        )
        points = gu.PointCloud(frame, data_column="height")

        # Filter with named method and custom reducers, including leave-one-out median
        filtered = points.filter(method="median", radius=1.1)
        leave_one_out = points.filter(method=Median(), radius=1.1, include_self=False)
        lower_quartile = points.filter(method=Quantile(0.25), radius=2.1)

        # Check neighborhood statistics, original coordinates and attributes
        np.testing.assert_allclose(filtered.data, [50.0, 2.0, 51.0, 10.0])
        np.testing.assert_allclose(leave_one_out.data, [100.0, 1.0, 100.0, np.nan], equal_nan=True)
        np.testing.assert_allclose(lower_quartile.data, [1.0, 1.0, 1.0, 10.0])
        assert_geodataframe_equal(filtered.ds.drop(columns="height"), frame.drop(columns="height"))
        assert filtered.ds["label"].tolist() == list("abcd")

    def test_filter__nearest_neighbour_limit(self) -> None:
        """Checks that k selects the requested number of nearest observations when no radius is given."""

        # Unequal spacing to avoid nearest-neighbor ties
        frame = gpd.GeoDataFrame(
            {"height": [0.0, 10.0, 40.0]},
            geometry=gpd.points_from_xy([0.0, 1.0, 4.0], [0.0] * 3),
            crs=32632,
        )
        points = gu.PointCloud(frame, data_column="height")

        # Average each point with nearest neighbor
        filtered = points.filter(method="mean", radius=None, k=2)

        # Check first/middle and middle/last pair means
        np.testing.assert_array_equal(filtered.data, [5.0, 5.0, 25.0])

    def test_filter__reducer_neighborhood_and_overrides(self) -> None:
        """Checks that point filters use the reducer's limits and honor explicit overrides without changing it."""

        # Unequal spacing gives each point a unique nearest neighbor
        points = gu.PointCloud.from_xyz(np.array([0.0, 1.0, 4.0]), np.zeros(3), np.array([0.0, 10.0, 40.0]), crs=32631)
        neighborhood = PointNeighbours(k=2, radius=None)
        operator = Mean(neighborhood=neighborhood)

        # Omitted limits use the reducer; radius and k can each be overridden
        configured = points.filter(operator)
        limited = points.filter(operator, radius=0.5)
        unlimited = points.filter(operator, k=None, radius=5)
        np.testing.assert_array_equal(configured.data, [5, 5, 25])
        np.testing.assert_array_equal(limited.data, points.data)
        np.testing.assert_allclose(unlimited.data, np.mean(points.data))
        assert operator.default_neighborhood is neighborhood

    def test_filter__zero_radius_with_neighbour_limit(self) -> None:
        """Checks that a zero radius and a count limit both include points at the target coordinate."""

        # Coincident observations for zero-radius search
        frame = gpd.GeoDataFrame(
            {"height": [2.0, 4.0]},
            geometry=gpd.points_from_xy([0.0, 0.0], [0.0, 0.0]),
            crs=32632,
        )
        points = gu.PointCloud(frame, data_column="height")

        # Check mean of both coincident observations
        filtered = points.filter(method="mean", radius=0, k=2)
        np.testing.assert_array_equal(filtered.data, [3.0, 3.0])

    def test_filter__geometry_elevation(self) -> None:
        """Checks that filtering native geometry elevation preserves attributes and replaces only Z coordinates."""

        # Store elevations in geometry Z
        frame = gpd.GeoDataFrame(
            {"classification": np.array([1, 2, 1], dtype=np.uint8)},
            geometry=gpd.points_from_xy([0.0, 1.0, 2.0], [5.0] * 3, z=[0.0, 100.0, 2.0]),
            crs=32632,
        )
        points = gu.PointCloud(frame, data_column=None)

        # Apply neighborhood median to elevations
        filtered = points.filter(method="median", radius=1.1)

        # Check filtered Z, original X/Y and classification
        np.testing.assert_array_equal(filtered.geometry.x, frame.geometry.x)
        np.testing.assert_array_equal(filtered.geometry.y, frame.geometry.y)
        np.testing.assert_array_equal(filtered.geometry.z, [50.0, 2.0, 51.0])
        np.testing.assert_array_equal(filtered["classification"], frame["classification"])

    def test_filter__duplicate_points_leave_out_only_the_same_row(self) -> None:
        """Checks that excluding a point from its neighborhood still includes other points at the same coordinates."""

        # Coincident observations with duplicate index to check selection by row
        frame = gpd.GeoDataFrame(
            {"height": [1.0, 3.0]},
            geometry=gpd.points_from_xy([5.0, 5.0], [7.0, 7.0]),
            crs=32632,
            index=[4, 4],
        )
        points = gu.PointCloud(frame, data_column="height")

        # Exclude target row from zero-radius neighborhood
        filtered = points.filter(method="mean", radius=0.0, include_self=False)

        # Check other coincident row's value and original index
        np.testing.assert_array_equal(filtered.data, [3.0, 1.0])
        np.testing.assert_array_equal(filtered.ds.index, [4, 4])


class TestPointCloudFilterChunked:
    """Test module for Dask and multiprocessing filters that read points across chunk boundaries.

    Point statistic accuracy is covered in test_operators/test_reducer.py.
    """

    points = gpd.GeoDataFrame(
        {
            "height": np.array([0.0, 100.0, 2.0, 9.0, 4.0, 8.0]),
            "row_id": np.arange(6, dtype=np.int32),
            "_geoutils_filter_id": np.arange(20, 26, dtype=np.int32),
        },
        geometry=gpd.points_from_xy(np.arange(6, dtype=float), np.zeros(6)),
        crs=32632,
    )

    def test_filter__dask_loading_laziness(self, tmp_path: Path) -> None:
        """Checks that Dask filtering stays lazy and matches eager results across three two-row partitions."""

        dgpd = pytest.importorskip("dask_geopandas")
        from dask.callbacks import Callback

        # Open six ordered points lazily in chunks of two
        source_filename = tmp_path / "filter_source.gpkg"
        self.points.to_file(source_filename, index=False)
        lazy = gu.open_pointcloud(str(source_filename), data_column="height", chunks=2)
        expected = gu.PointCloud(self.points, data_column="height").filter(
            method="median", radius=1.1, include_self=False
        )

        # Build filter graph and record executed tasks
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            filtered = lazy.pc.filter(method="median", radius=1.1, include_self=False, batch_size=2)

        # Check zero executed tasks, lazy inputs/outputs and point metadata
        assert tasks == []
        assert isinstance(filtered, dgpd.GeoDataFrame)
        assert not lazy.pc.is_loaded and not filtered.pc.is_loaded
        assert filtered.pc.data_column == "height"
        assert filtered.pc.point_count == len(self.points)

        # Compare computed rows with eager filtering; check original objects still lazy
        computed = filtered.compute().reset_index(drop=True)
        assert_geodataframe_equal(computed, expected.ds.reset_index(drop=True), check_dtype=False)
        assert not lazy.pc.is_loaded and not filtered.pc.is_loaded

    def test_filter__error_dask_geometry_elevation(self, tmp_path: Path) -> None:
        """Checks that Dask filtering rejects geometry Z values without a named data column."""

        pytest.importorskip("dask_geopandas")

        # Geometry Z without data column: unsupported for lazy point clouds
        source_filename = tmp_path / "filter_source_3d.gpkg"
        frame = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([0.0, 1.0], [0.0, 0.0], z=[2.0, 4.0]),
            crs=32632,
        )
        frame.to_file(source_filename, index=False)
        lazy = gu.open_pointcloud(str(source_filename), data_column=None, chunks=1)

        # Check missing-column error before loading source
        with pytest.raises(ValueError, match="explicit data column"):
            lazy.pc.filter(method="mean", radius=1.1)
        assert not lazy.pc.is_loaded

    def test_filter__multiprocessing_file_backed(self, tmp_path: Path) -> None:
        """
        Checks that multiprocessing reads nearby points across file partitions and returns the eager values without
        loading the output file.
        """

        # Three partitions of two rows (neighbors across boundaries)
        source_filename = tmp_path / "filter_source.gpkg"
        output_filename = tmp_path / "filter_output.gpkg"
        self.points.to_file(source_filename, index=False)
        source = gu.PointCloud(source_filename, data_column="height")
        expected = gu.PointCloud(self.points, data_column="height").filter(
            method="median", radius=1.1, include_self=False
        )

        # MP: filter with two workers, then join rows in input order
        with MpCluster({"nb_workers": 2}) as cluster:
            configuration = MultiprocConfig(
                chunks=2,
                outfile=str(output_filename),
                cluster=cluster,
            )
            filtered = source.filter(method="median", radius=1.1, include_self=False, mp_config=configuration)

        # Check unloaded inputs/outputs and metadata, then compare rows with eager filtering
        assert not source.is_loaded and not filtered.is_loaded
        assert filtered.point_count == len(self.points)
        assert filtered.data_column == "height"
        assert_geodataframe_equal(filtered.ds, expected.ds, check_dtype=False)
        assert not source.is_loaded
