"""Test point cloud loading by bounds and row ranges."""

from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from geopandas.testing import assert_geodataframe_equal
from rasterio.coords import BoundingBox

import geoutils as gu
from geoutils._misc import import_optional
from geoutils.pointcloud.dataframe import _get_dataframe_attrs
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
    @pytest.mark.parametrize("as_object_type", [None, "dataarray", "dataset", "geodataframe"])
    def test_open_pointcloud__return_type(
        self, tmp_path: Path, suffix: str, as_object_type: Literal["dataarray", "dataset", "geodataframe"] | None
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
        if as_object_type is None:
            points = gu.open_pointcloud(filename, data_name=data_name, columns=["quality"])
        else:
            points = gu.open_pointcloud(
                filename, data_name=data_name, columns=["quality"], as_object_type=as_object_type
            )
        if as_object_type == "dataset":
            assert isinstance(points, xr.Dataset)
            assert list(points.pc.compatible_variables.keys()) == [data_name, "quality"]
            assert points[data_name].dims == points.quality.dims == ("point",)
            assert "x_point" in points.coords and "y_point" in points.coords
            assert points.quality.pc.crs == cloud.crs
            values = points[data_name]
        else:
            assert isinstance(points, gpd.GeoDataFrame if as_object_type == "geodataframe" else xr.DataArray)
            values = points

        # Check exact equality with initial point cloud
        np.testing.assert_array_equal(values.pc.to_array(), cloud.to_array())
        quality = points["quality"] if as_object_type in ("dataset", "geodataframe") else points.coords["quality"]
        np.testing.assert_array_equal(quality, cloud["quality"])
        assert values.pc.crs == cloud.crs
        assert values.pc.point_count == 3
        assert values.pc.is_loaded

    @pytest.mark.parametrize("suffix", ["las", "laz", "parquet", "parquet-partitioned"])
    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset", "geodataframe"])
    def test_open_pointcloud__disk_metadata(
        self, tmp_path: Path, suffix: str, as_object_type: Literal["dataarray", "dataset", "geodataframe"]
    ) -> None:
        """Checks that opening exposes source record sizes, LAZ block limits and Parquet row group sizes."""

        # Write seven points so groups of three have a shorter final group
        cloud = gu.PointCloud.from_xyz(np.arange(7.0), np.zeros(7), np.arange(7.0), 32633, "Z")
        filename = tmp_path / ("parts" if suffix == "parquet-partitioned" else f"points.{suffix}")
        if suffix.startswith("parquet"):
            cloud.to_parquet(filename, chunks=3, partitioned=suffix == "parquet-partitioned")
        else:
            cloud.to_las(filename)

        # Open each representation from the same file metadata
        points = gu.open_pointcloud(filename, as_object_type=as_object_type)
        attrs = points.attrs
        values = points["Z"] if as_object_type == "dataset" else points

        # Compare disk metadata with the independent reader and the chosen output row groups
        if suffix.startswith("parquet"):
            assert attrs["row_group_sizes"] == [3, 3, 1]
        else:
            laspy = import_optional("laspy")
            with laspy.open(filename) as source:
                assert attrs["point_record_size"] == source.header.point_format.size
                if suffix == "laz":
                    lazrs = import_optional("lazrs")
                    compression = lazrs.LazVlr(source.header.vlrs.get("LasZipVlr")[0].record_data)
                    assert attrs["laz_chunk_size"] == compression.chunk_size()
                else:
                    assert attrs["laz_chunk_size"] is None
        if as_object_type == "dataset":
            assert values.attrs == attrs
        assert values.pc.is_loaded
        np.testing.assert_array_equal(values.pc.to_array(), cloud.to_array())

    @pytest.mark.parametrize("suffix", ["gpkg", "las", "parquet"])
    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset", "geodataframe"])
    def test_open_pointcloud__downsample(
        self, tmp_path: Path, suffix: str, as_object_type: Literal["dataarray", "dataset", "geodataframe"]
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
        points = gu.open_pointcloud(filename, data_name=data_name, downsample=4, as_object_type=as_object_type)
        values = points[data_name] if as_object_type == "dataset" else points
        assert values.pc.point_count == 3
        np.testing.assert_array_equal(values.pc.to_array(), expected.to_array())

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset", "geodataframe"])
    @pytest.mark.parametrize("level", [0, 1, range(2), range(0, 3, 2), 99])
    @pytest.mark.parametrize("data_name", ["Z", "intensity"])
    def test_open_pointcloud__copc_levels(
        self,
        copc_file: Path,
        as_object_type: Literal["dataarray", "dataset", "geodataframe"],
        level: int | range,
        data_name: str,
    ) -> None:
        """Checks that COPC levels of detail selected with open_pointcloud() match LasPy."""

        # We query the same levels with LasPy as a reference
        laspy = pytest.importorskip("laspy")
        with laspy.CopcReader.open(copc_file) as reader:
            expected = reader.query(level=level)
            crs = reader.header.parse_crs()
        expected_values = expected.z if data_name == "Z" else expected.intensity
        expected_array = np.stack([expected.x, expected.y, expected_values])

        # Open with levels
        points = gu.open_pointcloud(
            copc_file, level=level, columns="all", data_name=data_name, as_object_type=as_object_type
        )

        # We select only active values and "intensity" for comparison
        values = points[data_name] if as_object_type == "dataset" else points
        if as_object_type in ("dataset", "geodataframe"):
            intensity = points["intensity"]
        else:
            intensity = points if data_name == "intensity" else points.coords["intensity"]

        # Check exact values (and native integer dtype of "intensity")
        np.testing.assert_array_equal(values.pc.to_array(), expected_array)
        np.testing.assert_array_equal(intensity, expected.intensity)
        assert intensity.dtype == np.uint16
        assert values.pc.point_count == len(expected)
        assert values.pc.crs == crs
        assert values.pc.is_loaded

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset", "geodataframe"])
    @pytest.mark.parametrize("downsample", [1, 2, 4.5, 1_000_000])
    def test_open_pointcloud__copc_downsample(
        self, copc_file: Path, as_object_type: Literal["dataarray", "dataset", "geodataframe"], downsample: float
    ) -> None:
        """Checks that COPC downsampling selects levels of detail automatically."""

        # Build a reference with LasPy level read directly + subsampling
        laspy = pytest.importorskip("laspy")
        with laspy.CopcReader.open(copc_file) as reader:
            target_count = int(np.ceil(reader.header.point_count / downsample))
            selected = reader.query(level=0)
            assert 0 < len(selected) < reader.header.point_count
            if len(selected) < target_count:
                selected = reader.query()
        reference = gu.PointCloud.from_xyz(selected.x, selected.y, selected.z, 32633, "Z")
        reference["intensity"] = np.asarray(selected.intensity)
        expected = reference
        if target_count < len(selected):
            request = target_count if target_count > 1 else 1 / len(selected)
            expected = reference.subsample(request, random_state=0)

        # Open with downsampling, and extract Z + intensity
        points = gu.open_pointcloud(
            copc_file, downsample=downsample, columns=["intensity"], as_object_type=as_object_type
        )
        values = points["Z"] if as_object_type == "dataset" else points
        intensity = points["intensity"] if as_object_type in ("dataset", "geodataframe") else points.coords["intensity"]

        # Check we get expected arrays
        assert values.pc.point_count == target_count
        np.testing.assert_array_equal(values.pc.to_array(), expected.to_array())
        np.testing.assert_array_equal(intensity, expected["intensity"])

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset", "geodataframe"])
    @pytest.mark.parametrize(
        "options, selected_levels",
        [
            ({}, [0, 1]),
            ({"level": 0}, [0]),
            ({"level": range(2)}, [0, 1]),
            ({"level": 99}, []),
            ({"downsample": 1_000_000}, [0]),
        ],
    )
    def test_open_pointcloud__copc_metadata(
        self,
        copc_file: Path,
        as_object_type: Literal["dataarray", "dataset", "geodataframe"],
        options: dict[str, Any],
        selected_levels: list[int],
    ) -> None:
        """Checks that COPC attributes distinguish all stored levels from those selected for reading."""

        # Inspect every block in the small fixture's first hierarchy page without reading its points
        laspy = import_optional("laspy")
        with laspy.CopcReader.open(copc_file) as source:
            blocks = sorted(source.root_page.entries.values(), key=lambda entry: entry.offset)
            expected_sizes = [entry.point_count for entry in blocks if entry.point_count > 0]

        # Open full, explicitly selected and automatically sampled levels from the two-level fixture
        points = gu.open_pointcloud(copc_file, as_object_type=as_object_type, **options)

        # Check metadata remains available even for an empty selection or a one-point sample
        assert points.attrs["copc_levels"] == [0, 1]
        assert points.attrs["copc_selected_levels"] == selected_levels
        assert points.attrs["copc_chunk_sizes"] == expected_sizes
        values = points["Z"] if as_object_type == "dataset" else points
        assert values.pc.is_loaded
        if as_object_type == "dataset":
            assert values.attrs["copc_levels"] == [0, 1]
            assert values.attrs["copc_selected_levels"] == selected_levels
            assert values.attrs["copc_chunk_sizes"] == expected_sizes


class TestPointCloudOpeningChunked:
    """Test module ``open_pointcloud()`` with chunked backends (Dask only)."""

    @pytest.mark.parametrize("suffix", ["las", "laz", "parquet", "parquet-partitioned"])
    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset", "geodataframe"])
    def test_open_pointcloud__disk_metadata(
        self, tmp_path: Path, suffix: str, as_object_type: Literal["dataarray", "dataset", "geodataframe"]
    ) -> None:
        """Checks that disk metadata stays available without computing point values and matches eager opening."""
        import_optional("dask")
        from dask.callbacks import Callback

        # Write groups of three points; computation chunks of two split these disk groups differently
        cloud = gu.PointCloud.from_xyz(np.arange(7.0), np.zeros(7), np.arange(7.0), 32633, "Z")
        filename = tmp_path / ("parts" if suffix == "parquet-partitioned" else f"points.{suffix}")
        if suffix.startswith("parquet"):
            cloud.to_parquet(filename, chunks=3, partitioned=suffix == "parquet-partitioned")
        else:
            cloud.to_las(filename)
        eager = gu.open_pointcloud(filename, as_object_type=as_object_type)

        # Inspect metadata while preventing Dask from reading any point values
        with Callback(pretask=lambda *args: pytest.fail("Disk metadata must not compute point values")):
            lazy = gu.open_pointcloud(filename, chunks=2, as_object_type=as_object_type)
            attrs = _get_dataframe_attrs(lazy) if as_object_type == "geodataframe" else lazy.attrs
            keys = ["row_group_sizes"] if suffix.startswith("parquet") else ["point_record_size", "laz_chunk_size"]
            for key in keys:
                assert attrs[key] == eager.attrs[key]

        # Check lazy loading and exact equality with the eager values and metadata
        eager_values = eager["Z"] if as_object_type == "dataset" else eager
        lazy_values = lazy["Z"] if as_object_type == "dataset" else lazy
        assert eager_values.pc.is_loaded and not lazy_values.pc.is_loaded
        computed = lazy.compute(scheduler="synchronous")
        if as_object_type == "geodataframe":
            assert_geodataframe_equal(computed, eager)
        else:
            xr.testing.assert_identical(computed, eager)
        assert not lazy_values.pc.is_loaded

    @pytest.mark.parametrize("suffix", ["gpkg", "las", "parquet"])
    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset", "geodataframe"])
    def test_open_pointcloud__loading_laziness(
        self, tmp_path: Path, suffix: str, as_object_type: Literal["dataarray", "dataset", "geodataframe"]
    ) -> None:
        """Checks that chunked opening is lazy on all point formats/data structure types."""

        pytest.importorskip("dask")
        if as_object_type == "geodataframe" or suffix == "gpkg":
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
        eager = gu.open_pointcloud(filename, data_name=data_name, as_object_type=as_object_type)
        lazy = gu.open_pointcloud(filename, data_name=data_name, chunks=3, as_object_type=as_object_type)
        eager_values = eager[data_name] if as_object_type == "dataset" else eager
        lazy_values = lazy[data_name] if as_object_type == "dataset" else lazy
        assert eager_values.pc.is_loaded
        assert not lazy_values.pc.is_loaded
        assert lazy_values.pc.point_count == 7

        # Check exact equality and laziness
        actual = lazy.compute()
        if as_object_type in ("dataarray", "dataset"):
            xr.testing.assert_identical(actual, eager)
        else:
            assert_geodataframe_equal(actual, eager)
        actual_values = actual[data_name] if as_object_type == "dataset" else actual
        assert actual_values.pc.is_loaded
        assert not lazy_values.pc.is_loaded

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset", "geodataframe"])
    @pytest.mark.parametrize(
        "options",
        [
            {},
            {"level": 0},
            {"level": range(2)},
            {"level": 99},
            {"downsample": 2},
            {"downsample": 4.5},
            {"downsample": 1_000_000},
        ],
    )
    def test_open_pointcloud__copc_loading_laziness(
        self,
        copc_file: Path,
        as_object_type: Literal["dataarray", "dataset", "geodataframe"],
        options: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Checks that COPC reads stay lazy with level selection or downsampling."""

        dask = pytest.importorskip("dask")
        if as_object_type == "geodataframe":
            pytest.importorskip("dask_geopandas")
        laspy = pytest.importorskip("laspy")

        # Open an eager reference with same options
        eager = gu.open_pointcloud(copc_file, columns="all", as_object_type=as_object_type, **options)
        fetch_nodes = laspy.CopcReader._fetch_and_decompress_points_of_nodes

        # Below we'll record node offsets to check which levels are read
        read_offsets = []

        def record_reads(reader: object, nodes: list[Any]) -> object:
            read_offsets.extend(node.offset for node in nodes)
            return fetch_nodes(reader, nodes)

        monkeypatch.setattr(laspy.CopcReader, "_fetch_and_decompress_points_of_nodes", record_reads)

        # Open with chunks (shorter final chunk to reveal potential issues)
        lazy = gu.open_pointcloud(copc_file, columns="all", chunks=7001, as_object_type=as_object_type, **options)
        eager_values = eager["Z"] if as_object_type == "dataset" else eager
        lazy_values = lazy["Z"] if as_object_type == "dataset" else lazy

        # Check loading and metadata
        assert eager_values.pc.is_loaded
        assert not lazy_values.pc.is_loaded
        assert lazy_values.pc.point_count == eager_values.pc.point_count
        assert not read_offsets

        # Check the cached level lists without decompressing any point groups
        eager_attrs = eager.attrs if as_object_type != "geodataframe" else _get_dataframe_attrs(eager)
        lazy_attrs = lazy.attrs if as_object_type != "geodataframe" else _get_dataframe_attrs(lazy)
        assert lazy_attrs["copc_levels"] == eager_attrs["copc_levels"] == [0, 1]
        assert lazy_attrs["copc_selected_levels"] == eager_attrs["copc_selected_levels"]
        assert lazy_attrs["copc_chunk_sizes"] == eager_attrs["copc_chunk_sizes"]
        assert sum(lazy_attrs["copc_chunk_sizes"]) == 100_003
        assert not read_offsets

        # Compute
        with dask.config.set(scheduler="synchronous"):
            actual = lazy.compute()

        # Check every field matches the eager call
        if as_object_type in ("dataarray", "dataset"):
            xr.testing.assert_identical(actual, eager)
        else:
            assert_geodataframe_equal(actual, eager)

        # Check computed result is loaded and the original output stays lazy
        actual_values = actual["Z"] if as_object_type == "dataset" else actual
        assert actual_values.pc.is_loaded
        assert not lazy_values.pc.is_loaded
        if as_object_type != "geodataframe":
            assert max(lazy_values.chunks[0], default=0) <= 7001

        # Check reductions read only the root node
        if options.get("downsample", 1) in (4.5, 1_000_000):
            with laspy.CopcReader.open(copc_file) as reader:
                root_offsets = {
                    entry.offset
                    for entry in reader.root_page.entries.values()
                    if entry.key.level == 0 and entry.point_count > 0
                }
            assert set(read_offsets) == root_offsets


class TestPointCloudOpeningErrors:
    """Test module for errors in ``open_pointcloud()``."""

    def test_open_pointcloud__error_invalid_type(self, tmp_path: Path) -> None:
        """Checks an error is raised for a return type other than DataArray or GeoDataFrame."""
        filename = tmp_path / "absent.las"
        with pytest.raises(ValueError, match="as_object_type"):
            gu.open_pointcloud(filename, as_object_type="scene")  # type: ignore[arg-type]

    @pytest.mark.parametrize("chunks", [0, -1, False, 1.5])
    def test_open_pointcloud__error_invalid_chunks(self, tmp_path: Path, chunks: object) -> None:
        """Checks an error is raised before reading a file when chunks is not a positive integer."""
        filename = tmp_path / "absent.las"
        with pytest.raises(ValueError, match="strictly positive integer"):
            gu.open_pointcloud(filename, chunks=chunks)  # type: ignore[arg-type]

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset", "geodataframe"])
    @pytest.mark.parametrize("downsample", [True, np.bool_(False), 0, -1, np.nan, np.inf, "wrong"])
    def test_open_pointcloud__error_invalid_downsample(
        self, tmp_path: Path, as_object_type: Literal["dataarray", "dataset", "geodataframe"], downsample: object
    ) -> None:
        """Checks an error is raised for invalid downsampling factors."""
        filename = tmp_path / "absent.las"
        with pytest.raises((TypeError, ValueError), match="downsample must be"):
            gu.open_pointcloud(filename, as_object_type=as_object_type, downsample=downsample)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "level", [True, np.bool_(False), -1, 1.5, "wrong", range(0), range(-1, 2), range(2, 0, -1)]
    )
    def test_open_pointcloud__error_invalid_level(self, tmp_path: Path, level: object) -> None:
        """Checks an error is raised for invalid COPC levels before opening a file."""

        # Create an invalid filename (should not matter, arguments are checked before opening)
        filename = tmp_path / "absent.copc.laz"
        # Check invalid levels fail before file access
        with pytest.raises(ValueError, match="Argument 'level'"):
            gu.open_pointcloud(filename, level=level)  # type: ignore[arg-type]

    def test_open_pointcloud__error_level_and_downsample(self, tmp_path: Path) -> None:
        """Checks an error is raised with COPC levels + downsampling at once."""

        # Create an invalid filename (should not matter, arguments are checked before opening)
        filename = tmp_path / "absent.copc.laz"
        # Check downsampling + levels are not accepted together
        with pytest.raises(ValueError, match="downsample and level cannot be used together"):
            gu.open_pointcloud(filename, level=0, downsample=2)

    @pytest.mark.parametrize("suffix", ["las", "parquet", "gpkg"])
    def test_open_pointcloud__error_level_without_copc(self, tmp_path: Path, suffix: str) -> None:
        """Checks an error is raised when COPC levels are requested for other point formats."""

        filename = tmp_path / f"points.{suffix}"
        if suffix == "las":
            laspy = pytest.importorskip("laspy")
            laspy.create().write(filename)
        # Check passing "level" requires a COPC header
        with pytest.raises(ValueError, match="only supported for COPC"):
            gu.open_pointcloud(filename, level=0)


class TestPointCloudLoading:
    """Test module for loading helpers on point clouds."""

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
