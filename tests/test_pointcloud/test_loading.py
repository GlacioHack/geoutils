"""Test point cloud loading by bounds and row ranges."""

from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from geopandas.testing import assert_geodataframe_equal
from rasterio.coords import BoundingBox

import geoutils as gu
from geoutils.pointcloud.loading import (
    _concat_point_parts,
    _filter_points_by_bounds,
    _load_pointcloud_bounds,
    _load_pointcloud_rows,
)


class TestPointCloudOpening:
    """
    Test module for the ``open_pointcloud()`` function, to open point cloud files as a Pandas geodataframe,
    Xarray dataarray or Xarray dataset.
    """

    @pytest.mark.parametrize("suffix", ["gpkg", "las", "parquet"])
    @pytest.mark.parametrize("as_type", [None, "dataarray", "dataset", "geodataframe"])
    def test_open_pointcloud__return_type(
        self, tmp_path: Path, suffix: str, as_type: Literal["dataarray", "dataset", "geodataframe"] | None
    ) -> None:
        """Checks that opening defaults to a DataArray and every data type/point format loads correct values and CRS."""
        if suffix == "las":
            pytest.importorskip("laspy")
        elif suffix == "parquet":
            pytest.importorskip("pyarrow")

        # Create point cloud and write to file
        # LAS elevations are named Z, other formats store the defined name "height"
        data_name = "Z" if suffix == "las" else "height"
        cloud = gu.PointCloud.from_xyz([0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0], 32633, data_name)
        cloud["quality"] = np.array([4, 5, 6], dtype=np.uint16)
        filename = tmp_path / f"points.{suffix}"
        if suffix == "las":
            cloud.to_las(str(filename))
        else:
            cloud.to_file(str(filename))

        # Check proper data type is opened
        if as_type is None:
            points = gu.open_pointcloud(filename, data_name=data_name, columns=["quality"])
        else:
            points = gu.open_pointcloud(filename, data_name=data_name, columns=["quality"], as_type=as_type)
        if as_type == "dataset":
            assert isinstance(points, xr.Dataset)
            assert points.pc.variables == [data_name, "quality"]
            assert points[data_name].dims == points.quality.dims == ("point",)
            assert "x_point" in points.coords and "y_point" in points.coords
            assert points.quality.pc.crs == cloud.crs
            values = points[data_name]
        else:
            assert isinstance(points, gpd.GeoDataFrame if as_type == "geodataframe" else xr.DataArray)
            values = points

        # Check exact equality with initial point cloud
        np.testing.assert_array_equal(values.pc.to_array(), cloud.to_array())
        quality = points["quality"] if as_type in ("dataset", "geodataframe") else points.coords["quality"]
        np.testing.assert_array_equal(quality, cloud["quality"])
        assert values.pc.crs == cloud.crs
        assert values.pc.point_count == 3
        assert values.pc.is_loaded

    @pytest.mark.parametrize("suffix", ["gpkg", "las", "parquet"])
    @pytest.mark.parametrize("as_type", ["dataarray", "dataset", "geodataframe"])
    def test_open_pointcloud__downsample(
        self, tmp_path: Path, suffix: str, as_type: Literal["dataarray", "dataset", "geodataframe"]
    ) -> None:
        """Checks that every opening type/format uses the same deterministic downsampling."""
        if suffix == "las":
            pytest.importorskip("laspy")
        elif suffix == "parquet":
            pytest.importorskip("pyarrow")

        # Create/write point cloud
        data_name = "Z" if suffix == "las" else "height"
        cloud = gu.PointCloud.from_xyz(np.arange(11.0), np.zeros(11), np.arange(11.0), 32633, data_name)
        filename = tmp_path / f"points.{suffix}"
        if suffix == "las":
            cloud.to_las(str(filename))
        else:
            cloud.to_file(str(filename))
        expected = cloud.subsample(3, random_state=0)

        # Open with same downsampling, and check equality
        points = gu.open_pointcloud(filename, data_name=data_name, downsample=4, as_type=as_type)
        values = points[data_name] if as_type == "dataset" else points
        assert values.pc.point_count == 3
        np.testing.assert_array_equal(values.pc.to_array(), expected.to_array())


class TestPointCloudOpeningChunked:
    """Test module ``open_pointcloud()`` with chunked backends (Dask only)."""

    @pytest.mark.parametrize("suffix", ["gpkg", "las", "parquet"])
    @pytest.mark.parametrize("as_type", ["dataarray", "dataset", "geodataframe"])
    def test_open_pointcloud__loading_laziness(
        self, tmp_path: Path, suffix: str, as_type: Literal["dataarray", "dataset", "geodataframe"]
    ) -> None:
        """Checks that chunked opening is lazy on all point formats/data structure types."""

        pytest.importorskip("dask")
        if as_type == "geodataframe" or suffix == "gpkg":
            pytest.importorskip("dask_geopandas")
        if suffix == "las":
            pytest.importorskip("laspy")
        elif suffix == "parquet":
            pytest.importorskip("pyarrow")

        # Create and write chunked point cloud
        data_name = "Z" if suffix == "las" else "height"
        cloud = gu.PointCloud.from_xyz(np.arange(7.0), np.zeros(7), np.arange(7.0), 32633, data_name)
        filename = tmp_path / f"points.{suffix}"
        if suffix == "parquet":
            cloud.to_parquet(str(filename), chunks=3)
        elif suffix == "las":
            cloud.to_las(str(filename))
        else:
            cloud.to_file(str(filename))

        # Open eagerly/ lazily with an explicit type
        eager = gu.open_pointcloud(filename, data_name=data_name, as_type=as_type)
        lazy = gu.open_pointcloud(filename, data_name=data_name, chunks=3, as_type=as_type)
        eager_values = eager[data_name] if as_type == "dataset" else eager
        lazy_values = lazy[data_name] if as_type == "dataset" else lazy
        assert eager_values.pc.is_loaded
        assert not lazy_values.pc.is_loaded
        assert lazy_values.pc.point_count == 7

        # Check exact equality and laziness
        actual = lazy.compute()
        if as_type in ("dataarray", "dataset"):
            xr.testing.assert_identical(actual, eager)
        else:
            assert_geodataframe_equal(actual, eager)
        actual_values = actual[data_name] if as_type == "dataset" else actual
        assert actual_values.pc.is_loaded
        assert not lazy_values.pc.is_loaded


class TestPointCloudOpeningErrors:
    """Test module for errors in ``open_pointcloud()``."""

    def test_open_pointcloud__error_invalid_type(self, tmp_path: Path) -> None:
        """Checks an error is raised for a return type other than DataArray or GeoDataFrame."""
        filename = tmp_path / "absent.las"
        with pytest.raises(ValueError, match="as_type"):
            gu.open_pointcloud(filename, as_type="scene")  # type: ignore[arg-type]

    @pytest.mark.parametrize("chunks", [0, -1, False, 1.5])
    def test_open_pointcloud__error_invalid_chunks(self, tmp_path: Path, chunks: object) -> None:
        """Checks an error is raised before reading a file when chunks is not a positive integer."""
        filename = tmp_path / "absent.las"
        with pytest.raises(ValueError, match="strictly positive integer"):
            gu.open_pointcloud(filename, chunks=chunks)  # type: ignore[arg-type]

    @pytest.mark.parametrize("as_type", ["dataarray", "dataset", "geodataframe"])
    @pytest.mark.parametrize("downsample", [True, np.bool_(False), 0, -1, np.nan, np.inf, "wrong"])
    def test_open_pointcloud__error_invalid_downsample(
        self, tmp_path: Path, as_type: Literal["dataarray", "dataset", "geodataframe"], downsample: object
    ) -> None:
        """Checks an error is raised for invalid downsampling factors."""
        filename = tmp_path / "absent.las"
        with pytest.raises((TypeError, ValueError), match="downsample must be"):
            gu.open_pointcloud(filename, as_type=as_type, downsample=downsample)  # type: ignore[arg-type]


class TestPointCloudLoading:
    """Test module for selecting point rows, joining partitions, and reading file slices."""

    def test_filter_points_by_bounds__inclusive_and_independent(self) -> None:
        """Checks that bounds include points on their edges and return values independent of the source."""

        # Three points with two on the requested X boundaries
        frame = gpd.GeoDataFrame(
            {"height": [1.0, 2.0, 3.0]},
            geometry=gpd.points_from_xy([0.0, 1.0, 2.0], [0.0, 0.0, 0.0]),
            crs=32631,
            index=[10, 11, 12],
        )

        # Select both boundary points, then modify the original dataframe
        selected = _filter_points_by_bounds(frame, BoundingBox(1.0, 0.0, 2.0, 0.0))
        frame.loc[11, "height"] = 20.0

        # Check boundary inclusion, row indexes, and independent values
        assert selected.index.tolist() == [11, 12]
        np.testing.assert_array_equal(selected.geometry.x, [1.0, 2.0])
        np.testing.assert_array_equal(selected["height"], [2.0, 3.0])

    def test_concat_point_parts__values_and_indexes(self) -> None:
        """Checks that joined partitions preserve point order, duplicate indexes, and independent values."""

        # Two partitions with a repeated source index and an empty partition between them
        first = gpd.GeoDataFrame(
            {"height": [1.0, 2.0]},
            geometry=gpd.points_from_xy([0.0, 1.0], [0.0, 0.0]),
            crs=32631,
            index=[5, 5],
        )
        second = gpd.GeoDataFrame({"height": [3.0]}, geometry=gpd.points_from_xy([2.0], [0.0]), crs=32631, index=[8])

        # Join the partitions, then change the first source partition
        joined = _concat_point_parts([first, first.iloc[:0], second], crs=first.crs)
        first["height"] = [10.0, 20.0]

        # Check the original row order, duplicate indexes, coordinates, and values
        assert joined.index.tolist() == [5, 5, 8]
        assert joined.crs == first.crs
        np.testing.assert_array_equal(joined.geometry.x, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(joined["height"], [1.0, 2.0, 3.0])

    def test_concat_point_parts__empty_partitions(self) -> None:
        """Checks that no partitions or only empty partitions produce an empty point table with a CRS."""

        # Empty input with a data column for the all-empty case
        empty = gpd.GeoDataFrame({"height": np.array([], dtype=float)}, geometry=[], crs=32631)

        # Combine no partitions and one empty partition
        no_parts = _concat_point_parts([], crs=empty.crs)
        empty_parts = _concat_point_parts([empty], crs=empty.crs)

        # Check empty geometry, coordinate system, and the existing data column
        assert len(no_parts) == len(empty_parts) == 0
        assert no_parts.crs == empty_parts.crs == empty.crs
        assert "height" in empty_parts.columns

    def test_concat_point_parts__read_only_geometry(self) -> None:
        """Checks that read-only geometries can be concatenated."""

        # We simulate a Dask partition with geometry array that cannot be modified
        part = gpd.GeoDataFrame(
            {"height": [1.0, 2.0]},
            geometry=gpd.points_from_xy([0.0, 1.0], [0.0, 0.0]),
            crs=32631,
        ).rename_geometry("location")
        part.geometry.array._data.flags.writeable = False

        # We concetenate two partitions
        joined = _concat_point_parts([part, part], crs=part.crs)

        # We check the geometry column, repeated values, and read-only source
        assert joined.geometry.name == "location"
        assert joined.crs == part.crs
        assert not part.geometry.array._data.flags.writeable
        np.testing.assert_array_equal(joined.geometry.x, [0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(joined["height"], [1.0, 2.0, 1.0, 2.0])

    def test_load_pointcloud_bounds__file_ranges(self, tmp_path: Path) -> None:
        """Checks that a lazy point file reads inclusive bounds or all rows for unbounded support."""

        # Write three points so the file reader can select by its spatial index
        frame = gpd.GeoDataFrame(
            {"height": [1.0, 2.0, 3.0]},
            geometry=gpd.points_from_xy([0.0, 1.0, 2.0], [0.0, 0.0, 0.0]),
            crs=32631,
        )
        filename = tmp_path / "points.gpkg"
        frame.to_file(filename, index=False)
        source = gu.PointCloud(filename, data_name="height")

        # Read both boundary points and then the complete file through infinite support
        bounded = _load_pointcloud_bounds(source, BoundingBox(1.0, 0.0, 2.0, 0.0), "height")
        unbounded = _load_pointcloud_bounds(source, BoundingBox(-np.inf, -np.inf, np.inf, np.inf), "height")

        # Check selected values and that neither read loaded the PointCloud
        np.testing.assert_array_equal(bounded["height"], [2.0, 3.0])
        np.testing.assert_array_equal(unbounded["height"], frame["height"])
        assert not source.is_loaded

    @pytest.mark.parametrize("loaded", [True, False], ids=["memory", "file"])
    def test_load_pointcloud_rows__slice_and_empty(self, loaded: bool, tmp_path: Path) -> None:
        """Checks that loaded and file-backed point clouds read the same row slice and zero rows."""

        # Four ordered point rows, available in memory or through a file
        frame = gpd.GeoDataFrame(
            {"height": [1.0, 2.0, 3.0, 4.0]},
            geometry=gpd.points_from_xy([0.0, 1.0, 2.0, 3.0], [0.0] * 4),
            crs=32631,
        )
        filename = tmp_path / "rows.gpkg"
        frame.to_file(filename, index=False)
        source = gu.PointCloud(frame if loaded else filename, data_name="height")

        # Read two rows after the first and an empty slice after the second
        selected = _load_pointcloud_rows(source, start=1, count=2)
        empty = _load_pointcloud_rows(source, start=2, count=0)

        # Check row positions, values, and the file-backed source's lazy state
        np.testing.assert_array_equal(selected.geometry.x, [1.0, 2.0])
        np.testing.assert_array_equal(selected["height"], [2.0, 3.0])
        assert len(empty) == 0
        assert source.is_loaded == loaded
