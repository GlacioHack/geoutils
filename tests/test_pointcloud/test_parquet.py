"""Tests for GeoParquet point arrays and geometry interfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from geopandas.testing import assert_geodataframe_equal
from rasterio.transform import from_origin
from shapely.geometry import box

import geoutils as gu
from geoutils.pointcloud import parquet


class TestPointParquet:
    """Test module for Parquet I/O in-memory, see TestPointParquetChunked for tests out-of-memory."""

    @pytest.mark.parametrize("partitioned", [False, True])
    def test_to_parquet__coords_attrs(self, tmp_path: Path, partitioned: bool) -> None:
        """Checks that GeoParquet stores coordinates/attribute dtypes exactly."""
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        # Create point cloud with mixed dtypes and large IDs (to reveal loss from conversion)
        points = gu.DataArrayPointCloudAccessor.from_xyz(
            np.arange(11.0),
            np.arange(11.0) + 1,
            np.arange(11, dtype=np.float32),
            32633,
            auxiliary={
                "id": np.arange(11, dtype=np.uint64) + 2**63,
                "class": np.arange(11, dtype=np.uint8),
                "label": np.array(["tree"] * 11),
                "time": np.arange(11).astype("datetime64[ns]"),
            },
        )
        filename = tmp_path / "points.parquet"

        # We write it in chunks, and check metadata
        points.pc.to_parquet(str(filename), chunks=4, partitioned=partitioned)
        files = sorted(filename.glob("*.parquet")) if partitioned else [filename]
        file = pq.ParquetFile(files[0])
        geo = json.loads(file.schema_arrow.metadata[b"geo"])
        assert geo["version"] == "1.1.0"
        assert geo["columns"]["geometry"]["encoding"] == "point"
        assert file.schema_arrow.field("geometry").type.num_fields == 2
        assert len(files) == (3 if partitioned else 1)
        assert file.metadata.num_row_groups == (1 if partitioned else 3)

        # Reopen to verify exact equality (through GeoPandas reader)
        reopened = gu.open_pointcloud(str(filename), as_object_type="dataarray", columns="all")
        expected = points.assign_attrs(row_group_sizes=[4, 4, 3])
        xr.testing.assert_identical(reopened, expected)
        frame = gpd.read_parquet(filename)
        np.testing.assert_array_equal(frame.geometry.x, points.coords["x"])
        np.testing.assert_array_equal(frame["id"], points.coords["id"])
        assert frame.crs == points.pc.crs
        assert gu.PointCloud(str(filename)).bounds == (0.0, 1.0, 10.0, 11.0)

    @pytest.mark.parametrize("use_z", [False, True])
    @pytest.mark.parametrize("interface", ["pointcloud", "geopandas"])
    @pytest.mark.parametrize("method", ["to_parquet", "to_file"])
    def test_to_parquet__roundtrip(self, tmp_path: Path, use_z: bool, interface: str, method: str) -> None:
        """Checks that GeoParquet reading/writing keeps consistent geometries in a round trip."""
        pytest.importorskip("pyarrow")

        # We create synthetic geometry elevations as well as a named value column
        cloud = gu.PointCloud.from_xyz([0.0, 1.0], [2.0, 3.0], [4.0, 5.0], 32633, use_z=use_z)
        filename = tmp_path / "points.parquet"

        # We write it through the shared geometry base, then open an unloaded dedicated object
        writer = cloud if interface == "pointcloud" else cloud.gdf.pc
        getattr(writer, method)(str(filename), chunks=1)
        reopened = gu.PointCloud(filename)
        assert not reopened.is_loaded
        assert reopened.point_count == 2
        assert reopened.crs == cloud.crs

        # We check loading reconstructs exactly the same object
        reopened.load()
        np.testing.assert_array_equal(reopened.to_array(), cloud.to_array())
        assert reopened.data_name == cloud.data_name
        assert reopened.geometry.has_z.all() == use_z

    def test_open_pointcloud__wkb(self, tmp_path: Path) -> None:
        """Checks that existing WKB GeoParquet points can be read with the array accessor."""
        pytest.importorskip("pyarrow")

        # GeoPandas defaults to WKB, which needs geometry decoding at the input boundary
        frame = gu.PointCloud.from_xyz([0.0, 1.0], [2.0, 3.0], [4.0, 5.0], 32633).gdf
        filename = tmp_path / "wkb.parquet"
        frame.to_parquet(filename)

        # Compare extracted arrays and CRS with the original geometry points
        points = gu.open_pointcloud(str(filename), data_name="z", as_object_type="dataarray", columns="all")
        np.testing.assert_array_equal(points.pc.to_array(), frame.pc.to_array())
        assert points.pc.crs == frame.crs

    def test_open_pointcloud__deferred_fields(self, tmp_path: Path) -> None:
        """Checks that deferred reads load only requested fields."""

        # Write points with another attribute
        points = gu.pointcloud.from_xyz(
            np.arange(6.0),
            np.zeros(6),
            np.arange(6.0),
            32633,
            auxiliary={"quality": np.arange(6, dtype=np.uint16)},
        )
        filename = tmp_path / "fields.parquet"
        points.pc.to_parquet(str(filename), chunks=3)

        # Open one point cloud eagerly
        expected = gu.open_pointcloud(filename, columns="all", load=True).isel(point=slice(1, 4))

        # Open again with deferred I/O (and record reads)
        with patch.object(parquet, "_read_point_parquet_group", wraps=parquet._read_point_parquet_group) as reads:
            result = gu.open_pointcloud(filename, columns="all").isel(point=slice(1, 4))

            # Ensure nothing was read
            assert reads.call_count == 0

            # Read bounds and ensure point values are not loaded
            assert result.pc.bounds == expected.pc.bounds
            assert all(call.args[2:] == ("geometry", []) for call in reads.call_args_list)
            assert not result.pc.is_loaded
            reads.reset_mock()

            # Load and check equivalence
            xr.testing.assert_identical(result.load(), expected)

            # Check reads: 2 attributes across 2 row groups = 4
            assert reads.call_count == 4
            assert all(call.args[2] is None and len(call.args[3]) == 1 for call in reads.call_args_list)
            assert {call.args[3][0] for call in reads.call_args_list} == {"z", "quality"}

    @pytest.mark.parametrize("write_statistics", [True, False])
    def test_open_pointcloud__nullable_attributes(self, tmp_path: Path, write_statistics: bool) -> None:
        """Checks that deferred reads preserve nullable dtypes."""

        # Write point cloud with nullable attributes
        frame = gu.PointCloud.from_xyz(np.arange(3.0), np.zeros(3), np.arange(3.0), 32633, "height").gdf
        frame["quality"] = pd.array([1, None, 3], dtype="Int16")
        frame["valid"] = pd.array([True, None, False], dtype="boolean")

        # Add large IDs to check integer precision
        frame["identifier"] = np.arange(3, dtype=np.uint64) + 2**63
        filename = tmp_path / "nullable.parquet"
        frame.to_parquet(filename, row_group_size=1, write_statistics=write_statistics)

        # Open eagerly and with deferred I/O
        expected = gu.open_pointcloud(filename, data_name="height", columns="all", load=True)
        result = gu.open_pointcloud(filename, data_name="height", columns="all")

        # Ensure values are not loaded and check dtypes
        assert not result.pc.is_loaded
        assert result.quality.dtype == (np.dtype("float64") if write_statistics else np.dtype("O"))
        assert result.valid.dtype == np.dtype("O")

        # Load selected rows and check equivalence
        selected = result.isel(point=slice(0, 1)).load()
        xr.testing.assert_identical(selected, expected.isel(point=slice(0, 1)))
        xr.testing.assert_identical(result.load(), expected)
        np.testing.assert_array_equal(result.identifier, frame["identifier"])

    @pytest.mark.parametrize("write_statistics", [True, False])
    @pytest.mark.parametrize("row_group_size", [1, 3])
    @pytest.mark.parametrize("load", [False, True])
    def test_open_pointcloud__nullable_large_integers(
        self, tmp_path: Path, write_statistics: bool, row_group_size: int, load: bool
    ) -> None:
        """Checks that nullable integer IDs preserve every bit across row groups and deferred reads."""

        # Write IDs above float64's exact range with a missing row in the middle
        frame = gu.PointCloud.from_xyz(np.arange(3.0), np.zeros(3), np.arange(3.0), 32633, "height").gdf
        expected = [2**63 + 1, None, 2**63 + 3]
        frame["identifier"] = pd.array(expected, dtype="UInt64")
        filename = tmp_path / "nullable_ids.parquet"
        frame.to_parquet(filename, row_group_size=row_group_size, write_statistics=write_statistics)

        # Read all attributes without converting nullable IDs through floating point
        result = gu.open_pointcloud(filename, data_name="height", columns="all", load=load)
        assert result.pc.is_loaded is load
        actual = result.identifier.values

        # Check exact values independently of a second read through the same implementation
        assert actual[0] == expected[0] and actual[2] == expected[2]
        assert pd.isna(actual[1])
        assert result.identifier.dtype == object

    @pytest.mark.parametrize("nullable", [False, True])
    @pytest.mark.parametrize("dtype", ["Int16", "Int64"])
    def test_open_pointcloud__integer_measurements_without_statistics(
        self, tmp_path: Path, nullable: bool, dtype: str
    ) -> None:
        """Checks that integer measurements without Parquet statistics support analysis and sampling."""

        # Write integer measurements, optionally with one missing value and no null statistics
        frame = gu.PointCloud.from_xyz(np.arange(3.0), np.zeros(3), np.arange(3.0), 32633, "height").gdf
        frame["height"] = pd.array([1, None if nullable else 2, 3], dtype=dtype)
        filename = tmp_path / "integer.parquet"
        frame.to_parquet(filename, write_statistics=False)
        source = gu.open_pointcloud(filename, data_name="height")

        # Calculate a mean, sample finite rows and grid at the original point locations
        assert source.pc.stats("mean") == 2
        sampled = source.pc.subsample(2, random_state=0)
        assert sampled.pc.point_count == 2
        assert not pd.isna(sampled.values).any()
        reference = gu.raster.from_array(np.ones((1, 3)), from_origin(0, 0, 1, 1), 32633)
        gridded = source.pc.grid(ref=reference, resampling="nearest", radius=0.25)

        # Check exact samples and missing point positions rather than comparing two reads
        expected = np.array([1.0, np.nan if nullable else 2.0, 3.0])
        np.testing.assert_array_equal(gridded.values[-1], expected)

    def test_open_pointcloud__empty(self, tmp_path: Path) -> None:
        """Checks the edge case of an empty GeoParquet point output ."""
        pytest.importorskip("pyarrow")

        # Create empty result, which still needs readable coordinate/attributes
        values = np.empty(0, dtype=np.float32)
        points = gu.DataArrayPointCloudAccessor.from_xyz(
            np.empty(0),
            np.empty(0),
            values,
            32633,
            auxiliary={"id": np.empty(0, dtype=np.uint64)},
        )
        filename = tmp_path / "empty.parquet"

        # Write and reopen every attribute, check for equality
        points.pc.to_parquet(str(filename))
        reopened = gu.open_pointcloud(str(filename), as_object_type="dataarray", columns="all")
        expected = points.assign_attrs(row_group_sizes=[0])
        xr.testing.assert_identical(reopened, expected)


class TestPointParquetChunked:
    """Test for GeoParquet I/O in chunks (Dask and multiprocessing)."""

    @pytest.mark.parametrize("write_statistics", [True, False])
    def test_open_pointcloud__nullable_large_integers(self, tmp_path: Path, write_statistics: bool) -> None:
        """Checks that chunked nullable IDs stay lazy and preserve exact integers as in an eager read."""
        pytest.importorskip("dask")

        # Put large IDs and a missing value in one row group, then split it into chunks of 2/1 points
        frame = gu.PointCloud.from_xyz(np.arange(3.0), np.zeros(3), np.arange(3.0), 32633, "height").gdf
        expected_ids = [2**63 + 1, None, 2**63 + 3]
        frame["identifier"] = pd.array(expected_ids, dtype="UInt64")
        filename = tmp_path / "nullable_ids.parquet"
        frame.to_parquet(filename, row_group_size=3, write_statistics=write_statistics)
        eager = gu.open_pointcloud(filename, data_name="height", columns="all", load=True)

        # Open lazily and calculate a separate result
        source = gu.open_pointcloud(filename, data_name="height", columns="all", chunks=2)
        assert source.chunks and not source.pc.is_loaded
        result = source.compute()

        # Check exact integers independently, then compare every eager value and metadata field
        assert result.identifier.values.tolist() == expected_ids
        xr.testing.assert_identical(result, eager)
        assert source.chunks and not source.pc.is_loaded

    @pytest.mark.parametrize("dtype", ["Int16", "Int64"])
    def test_open_pointcloud__integer_measurements_without_statistics(self, tmp_path: Path, dtype: str) -> None:
        """Checks that lazy nullable measurements support statistics, gridding and exact eager sampling."""
        pytest.importorskip("dask")
        from dask.callbacks import Callback

        # Split finite measurements around a missing value into chunks of 2/1 points
        frame = gu.PointCloud.from_xyz(np.arange(3.0), np.zeros(3), np.arange(3.0), 32633, "height").gdf
        frame["height"] = pd.array([1, None, 3], dtype=dtype)
        filename = tmp_path / "integer.parquet"
        frame.to_parquet(filename, write_statistics=False)
        eager = gu.open_pointcloud(filename, data_name="height", load=True)
        source = gu.open_pointcloud(filename, data_name="height", chunks=2)
        reference = gu.raster.from_array(np.ones((1, 3)), from_origin(0, 0, 1, 1), 32633)

        # Build gridding lazily; statistics and sampling may read finite counts without loading the source
        with Callback(pretask=lambda *args: pytest.fail("Gridding must build a lazy result")):
            gridded = source.pc.grid(ref=reference, resampling="nearest", radius=0.25)
        assert gridded.chunks and not gridded.rst.is_loaded
        assert source.pc.stats("mean") == eager.pc.stats("mean") == 2
        sampled = source.pc.subsample(2, random_state=0)
        assert sampled.chunks and not sampled.pc.is_loaded

        # Compare every computed result exactly with the same eager calls and check original laziness
        xr.testing.assert_identical(sampled.compute(), eager.pc.subsample(2, random_state=0))
        expected = eager.pc.grid(ref=reference, resampling="nearest", radius=0.25)
        assert gridded.rst.crs == expected.rst.crs
        assert gridded.rst.transform == expected.rst.transform
        # Raster conversion can serialize an equivalent CRS with different WKT text
        xr.testing.assert_identical(gridded.compute().drop_vars("spatial_ref"), expected.drop_vars("spatial_ref"))
        assert source.chunks and not source.pc.is_loaded
        assert not sampled.pc.is_loaded and not gridded.rst.is_loaded

    @pytest.mark.parametrize("write_statistics", [True, False])
    def test_open_pointcloud__nullable_attributes(self, tmp_path: Path, write_statistics: bool) -> None:
        """Checks that chunked reads preserve nullable dtypes."""
        pytest.importorskip("dask")

        # Write nullable attributes in single-row groups
        frame = gu.PointCloud.from_xyz(np.arange(3.0), np.zeros(3), np.arange(3.0), 32633, "height").gdf
        frame["quality"] = pd.array([1, None, 3], dtype="Int16")
        frame["valid"] = pd.array([True, None, False], dtype="boolean")
        filename = tmp_path / "nullable.parquet"
        frame.to_parquet(filename, row_group_size=1, write_statistics=write_statistics)

        # Open eagerly and in chunks
        expected = gu.open_pointcloud(filename, data_name="height", columns="all", load=True)
        result = gu.open_pointcloud(filename, data_name="height", columns="all", chunks=1)
        assert not result.pc.is_loaded
        assert result.quality.dtype == expected.quality.dtype
        assert result.valid.dtype == expected.valid.dtype

        # Load one chunk and full result, then check equivalence
        xr.testing.assert_identical(result.isel(point=slice(0, 1)).compute(), expected.isel(point=slice(0, 1)))
        xr.testing.assert_identical(result.compute(), expected)
        assert not result.pc.is_loaded

    @pytest.mark.parametrize("as_object_type", ["dataarray", "geodataframe"])
    def test_open_pointcloud__select_auxiliary(
        self, tmp_path: Path, as_object_type: Literal["dataarray", "geodataframe"]
    ) -> None:
        """Checks that selecting an auxiliary attribute leaves the original geometry elevations unchanged."""
        pytest.importorskip("pyarrow")
        pytest.importorskip("dask")
        if as_object_type == "geodataframe":
            pytest.importorskip("dask_geopandas")

        # We create + write points with attribute IDs different from default Z elevation
        points = gu.DataArrayPointCloudAccessor.from_xyz(
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            32633,
            data_name="height",
            use_z=True,
            auxiliary={"id": np.array([10, 20, 30], dtype=np.uint16)},
        )
        filename = tmp_path / "points.parquet"
        points.pc.to_parquet(str(filename), chunks=2)

        # Open eager and lazy
        eager = gu.open_pointcloud(filename, data_name="id", as_object_type=as_object_type, load=True)
        reopened = gu.open_pointcloud(filename, data_name="id", chunks=2, as_object_type=as_object_type)

        # Check laziness and exactly equality
        assert eager.pc.is_loaded
        assert not reopened.pc.is_loaded
        actual = reopened.compute()
        if as_object_type == "dataarray":
            xr.testing.assert_identical(actual, eager)
        else:
            assert_geodataframe_equal(actual, eager)
        np.testing.assert_array_equal(actual.pc.data, [10, 20, 30])
        assert actual.pc.data.dtype == np.uint16
        geometry = actual.pc.to_geoutils().geometry
        np.testing.assert_array_equal(geometry.z, [6, 7, 8])
        assert not reopened.pc.is_loaded

    @pytest.mark.parametrize("writer", ["geopandas", "dataarray", "dataset"])
    @pytest.mark.parametrize("index_name", [None, "point_id", "z"])
    @pytest.mark.parametrize("data_name", ["z", "intensity"])
    @pytest.mark.parametrize("as_object_type", ["dataarray", "geodataframe"])
    def test_open_pointcloud__row_labels(
        self,
        tmp_path: Path,
        as_object_type: Literal["dataarray", "geodataframe"],
        writer: str,
        index_name: str | None,
        data_name: str,
    ) -> None:
        """Checks that duplicate row labels stay lazy and are preserved through either point interface."""

        pytest.importorskip("pyarrow")
        pytest.importorskip("dask")
        if as_object_type == "geodataframe":
            pytest.importorskip("dask_geopandas")
        import dask

        # We create + write point with labels distinguishing row positions from a Pandas index
        frame = gu.PointCloud.from_xyz(np.arange(5), np.zeros(5), np.arange(5), 32633).gdf
        frame["intensity"] = np.arange(5, dtype=np.uint16)
        frame.index = pd.Index(["a", "b", "a", "c", "b"], name=index_name)
        filename = tmp_path / "labels.parquet"
        if writer == "geopandas":
            frame.to_parquet(filename, index=True, geometry_encoding="geoarrow", row_group_size=2)
        else:
            points = frame.pc.to_xarray()
            if writer == "dataset":
                # Dataset-level index metadata must also survive when it is absent from the active values
                points = points.rename({"x": "x_point", "y": "y_point"}).to_dataset()
                points.attrs["dataframe_index_name"] = points.z.attrs.pop("dataframe_index_name")
            points.pc.to_parquet(filename, chunks=2)
        eager = gu.open_pointcloud(filename, data_name=data_name, as_object_type=as_object_type, load=True)

        # We open and check laziness/exact equality
        with dask.callbacks.Callback(pretask=lambda *args: pytest.fail("Opening must read only metadata")):
            lazy = gu.open_pointcloud(filename, data_name=data_name, chunks=2, as_object_type=as_object_type)
            assert not lazy.pc.is_loaded
            assert lazy.pc.point_count == len(frame)
        computed = lazy.compute()
        assert computed.pc.pointcloud_equal(eager)
        np.testing.assert_array_equal(computed.pc.data, frame[data_name].to_numpy())
        assert computed.pc.to_geoutils().index.equals(frame.index)
        assert computed.pc.to_geoutils().index.name == index_name
        assert not lazy.pc.is_loaded

    @pytest.mark.parametrize("partitioned", [False, True])
    @pytest.mark.parametrize("as_object_type", ["dataarray", "geodataframe"])
    @pytest.mark.parametrize("use_z", [False, True])
    @pytest.mark.parametrize("columns", ["main", "all"])
    def test_open_pointcloud__loading_laziness(
        self,
        tmp_path: Path,
        partitioned: bool,
        as_object_type: Literal["dataarray", "geodataframe"],
        use_z: bool,
        columns: Literal["main", "all"],
    ) -> None:
        """Checks that GeoParquet reads are lazy and match eager exactly."""
        pytest.importorskip("pyarrow")
        pytest.importorskip("dask")
        if as_object_type == "geodataframe":
            pytest.importorskip("dask_geopandas")
        import dask

        # 1/ Create + write point cloud in chunks
        points = gu.DataArrayPointCloudAccessor.from_xyz(
            np.arange(13.0),
            np.zeros(13),
            np.arange(13, dtype=np.float32),
            32633,
            use_z=use_z,
            auxiliary={"id": np.arange(13, dtype=np.uint64) + 2**63},
        )
        source = tmp_path / "source.parquet"
        points.pc.to_parquet(source, chunks=4)
        lazy = gu.open_pointcloud(source, columns="all", chunks=4, as_object_type=as_object_type)
        filename = tmp_path / "points.parquet"
        lazy.pc.to_parquet(str(filename), chunks=5, partitioned=partitioned)

        # 2/ Reopen
        assert not lazy.pc.is_loaded
        with dask.callbacks.Callback(pretask=lambda *args: pytest.fail("Opening Parquet must read only metadata")):
            reopened = gu.open_pointcloud(str(filename), columns=columns, as_object_type=as_object_type, chunks=3)
            assert reopened.pc.point_count == 13
            assert not reopened.pc.is_loaded
            assert (
                bool(reopened.attrs.get("geometry_z")) if as_object_type == "dataarray" else reopened.pc._has_z
            ) == use_z
            assert ("id" in reopened.pc.columns) == (columns == "all")
        eager = gu.open_pointcloud(filename, columns=columns, as_object_type=as_object_type, load=True)

        # 3/ Compare exact equality and laziness
        computed = reopened.compute()
        if as_object_type == "dataarray":
            xr.testing.assert_identical(computed, eager)
        else:
            assert_geodataframe_equal(computed, eager)
        assert computed.pc.to_geoutils().geometry.has_z.all() == use_z
        if columns == "all":
            np.testing.assert_array_equal(
                computed.coords["id"] if as_object_type == "dataarray" else computed["id"], points.coords["id"]
            )
        assert not reopened.pc.is_loaded
        assert not lazy.pc.is_loaded

    @pytest.mark.parametrize("as_object_type", ["dataarray", "geodataframe"])
    def test_write_parquet_row_groups__formats(
        self, tmp_path: Path, as_object_type: Literal["dataarray", "geodataframe"]
    ) -> None:
        """Checks that both DataArrays and GeoDataFrame partitions write ordered GeoParquet row groups properly."""
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        from geoutils.pointcloud.parquet import _write_parquet_row_groups

        # Create + write groups of points (with a shorter final chunk)
        cloud = gu.PointCloud.from_xyz(
            np.arange(11.0), np.arange(11.0) + 1, np.arange(11, dtype=np.float32), 32633, data_name="height"
        )
        cloud.gdf["id"] = np.arange(11, dtype=np.uint64) + 2**63
        source = cloud.to_xarray() if as_object_type == "dataarray" else cloud.gdf
        filename = tmp_path / "row-groups.parquet"
        partitions = (source[start : start + 4] for start in range(0, 11, 4))
        _write_parquet_row_groups(partitions, filename, data_name=cloud.data_name)
        assert source.pc.is_loaded

        # Load metadata and check
        file = pq.ParquetFile(filename)
        geo = json.loads(file.schema_arrow.metadata[b"geo"])
        assert [file.metadata.row_group(group).num_rows for group in range(file.metadata.num_row_groups)] == [4, 4, 3]
        assert "bbox" not in geo["columns"]["geometry"]
        result = gu.PointCloud(filename)
        assert not result.is_loaded
        assert result.point_count == cloud.point_count
        assert result.bounds == cloud.bounds

        # Explicit loading should give full equality
        result.load(columns="all")
        assert result.pointcloud_equal(cloud)


class TestPointParquetErrors:
    """Test module for errors/warnings of GeoParquet I/P."""

    def test_open_pointcloud__error_invalid_geometry(self, tmp_path: Path) -> None:
        """Checks an error is raised for a GeoParquet file with invalid geometry (e.g. polygons) instead of points."""
        pytest.importorskip("pyarrow")

        filename = tmp_path / "polygon.parquet"
        gpd.GeoDataFrame({"value": [1]}, geometry=[box(0, 0, 1, 1)], crs=32633).to_parquet(filename)
        with pytest.raises(ValueError, match="point geometries"):
            gu.open_pointcloud(str(filename), as_object_type="dataarray")

    @pytest.mark.parametrize("difference", ["crs", "dtype", "geometry_z"])
    def test_open_pointcloud__error_mismatched_partitions(self, tmp_path: Path, difference: str) -> None:
        """Checks an error is raised before combining partitions with different CRS or schemas."""

        pytest.importorskip("pyarrow")

        # Create two point clouds with incompatible CRS/dtype/Z geometry
        first = gu.DataArrayPointCloudAccessor.from_xyz([0.0], [1.0], np.array([2.0], dtype=np.float32), 32633)
        second = gu.DataArrayPointCloudAccessor.from_xyz(
            [3.0],
            [4.0],
            np.array([5.0], dtype=np.float64 if difference == "dtype" else np.float32),
            4326 if difference == "crs" else 32633,
            use_z=difference == "geometry_z",
        )
        first.pc.to_parquet(str(tmp_path / "part-0.parquet"))
        second.pc.to_parquet(str(tmp_path / "part-1.parquet"))

        # Check error is raised
        with pytest.raises(ValueError, match="matching coordinate systems, schemas and attributes"):
            gu.open_pointcloud(str(tmp_path), as_object_type="dataarray", chunks=1)

    def test_to_parquet__error_existing_partition(self, tmp_path: Path) -> None:
        """Checks an error is raised before overwriting an existing partitioned dataset."""
        pytest.importorskip("pyarrow")

        # We write an original file to check survival later
        points = gu.DataArrayPointCloudAccessor.from_xyz([0], [1], [2], 32633)
        destination = tmp_path / "existing"
        destination.mkdir()
        marker = destination / "marker.txt"
        marker.write_text("original")

        # Check error
        with pytest.raises(FileExistsError, match="already exist"):
            points.pc.to_parquet(str(destination), partitioned=True)

        # Existing contents must survive a failed attempt to write a partitioned point cloud
        assert marker.read_text() == "original"

    def test_to_parquet__error_preserves_existing(self, tmp_path: Path) -> None:
        """Checks an error while writing a later row group leaves an existing destination unchanged."""
        pytest.importorskip("pyarrow")

        # We create an erroneous input with last attribute changing Arrow type,
        # causing a schema error after the first row group is written
        points = gu.DataArrayPointCloudAccessor.from_xyz(
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            32633,
            auxiliary={"label": np.array([1, 2, "tree"], dtype=object)},
        )
        filename = tmp_path / "existing.parquet"
        filename.write_bytes(b"original destination")

        # Check error + survival
        with pytest.raises(ValueError, match="schema"):
            points.pc.to_parquet(str(filename), chunks=2)
        assert filename.read_bytes() == b"original destination"
