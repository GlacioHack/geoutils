"""Test vector opening, loading and writing."""

from __future__ import annotations

import os.path
import pathlib
import tempfile
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from geopandas.testing import assert_geodataframe_equal
from pyproj import CRS
from shapely.geometry import box

import geoutils as gu
import geoutils.vector.io as vector_io
from tests.accessor_helpers import assert_output_equal
from tests.remote_helpers import RemoteFiles


class TestVectorOpening:
    """Test module for opening vector files, deferred loading and layer selection."""

    aster_outlines_path = gu.examples.get_path_test("exploradores_rgi_outlines")

    def test_open_vector(self) -> None:
        """Open a vector file as the same in-memory GeoDataFrame used by Vector."""

        ds = gu.open_vector(self.aster_outlines_path, load=True)

        assert isinstance(ds, gpd.GeoDataFrame)
        assert_geodataframe_equal(ds, gu.Vector(self.aster_outlines_path).gdf)

    def test_lazy_load_from_file(self) -> None:
        """Check that file-backed vectors only load their GeoDataFrame when data are requested."""

        # Read an in-memory reference separately from the lazy Vector constructor
        vector = gu.Vector(self.aster_outlines_path)
        eager = gpd.read_file(self.aster_outlines_path)

        # Public metadata should be available without retaining all features
        assert not vector.is_loaded
        assert vector.crs == eager.crs
        assert np.array_equal(vector.total_bounds, eager.total_bounds)
        assert list(vector.columns) == list(eager.columns)
        assert not vector.is_loaded

        # Accessing the full dataframe triggers loading and matches GeoPandas
        assert_geodataframe_equal(vector.gdf, eager)
        assert vector.is_loaded

    def test_init__select_file_layer(self, tmp_path: pathlib.Path) -> None:
        """Checks that Vector class can select a specific layer, both for lazy metadata and data loading."""

        # Write two layers with distinct values and extents to one GeoPackage
        filename = tmp_path / "multiple-layers.gpkg"
        first = gpd.GeoDataFrame({"value": [1]}, geometry=gpd.points_from_xy([0], [0]), crs=4326)
        second = gpd.GeoDataFrame({"value": [2, 3]}, geometry=gpd.points_from_xy([10, 20], [10, 20]), crs=4326)
        first.to_file(filename, layer="first")
        second.to_file(filename, layer="second")

        # Select the second layer while keeping its features unloaded
        vector = gu.Vector(filename, layer="second")
        assert not vector.is_loaded
        np.testing.assert_array_equal(vector.total_bounds, second.total_bounds)

        # Load the selected layer and compare it with the original frame
        assert_geodataframe_equal(vector.gdf, second)

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="Fiona can return empty data when reading Shapefiles through a filesystem.",
    )
    def test_read_file__shapefile_filesystem(self, tmp_path: pathlib.Path) -> None:
        """Checks that a filesystem read returns all Shapefile features and their attributes."""

        # Filesystem read require Fiona for vectors (for now)
        pytest.importorskip("fiona")
        filesystem = pytest.importorskip("fsspec").filesystem("file")

        # Create and write vector
        frame = gpd.GeoDataFrame({"value": [1, 2]}, geometry=gpd.points_from_xy([0, 1], [2, 3]), crs=4326)
        path = tmp_path / "points.shp"
        frame.to_file(path)

        # Read through Fiona
        result = gpd.read_file(path, engine="fiona", opener=filesystem)

        # Check equality with input
        assert len(result) == len(frame)
        assert_geodataframe_equal(result, frame, check_dtype=False)

    @pytest.mark.parametrize("reader", [gu.Vector, gu.open_vector])
    @pytest.mark.parametrize("partitioned", [False, True])
    def test_open_or_init__parquet_filesystem(
        self, reader: Callable[..., Any], partitioned: bool, tmp_path: pathlib.Path
    ) -> None:
        """Checks that vector Parquet files reopen through their destination filesystem."""
        pytest.importorskip("pyarrow")
        filesystem = pytest.importorskip("fsspec").filesystem("memory")
        frame = gpd.GeoDataFrame(
            {"value": [1, 2], "label": pd.Categorical(["ground", "tree"])},
            geometry=gpd.points_from_xy([0, 1], [0, 1]),
            crs=4326,
        )
        frame.index = pd.Index([4, 8], name="feature")

        # Use forward slashes for memory paths on every platform
        path = f"/{tmp_path.name}/points.parquet"
        if partitioned:
            filesystem.makedirs(path)
            for index in range(len(frame)):
                gu.Vector(frame.iloc[index : index + 1]).to_file(f"{path}/part-{index}.parquet", filesystem=filesystem)
        else:
            gu.Vector(frame).to_file(path, filesystem=filesystem)

        result = reader(path, filesystem=filesystem, **({"load": True} if reader is gu.open_vector else {}))
        if isinstance(result, gu.Vector):
            assert not result.is_loaded
            assert list(result.columns) == list(frame.columns)
            assert result.bounds == tuple(frame.total_bounds)
            assert not result.is_loaded
            result = result.gdf

        assert_geodataframe_equal(result, frame)


class TestVectorOpeningChunked:
    """Test module for vector opening, respecting Dask lazyiness."""

    @pytest.mark.parametrize("filesystem_access", [False, True])
    @pytest.mark.parametrize("chunks", [None, 3])
    @pytest.mark.parametrize(
        "options",
        [
            {},
            {"bbox": (1, 1, 4, 4)},
            {"where": "value >= 2"},
            {"where": "value >= 2", "fid_as_index": True},
            {"fid_as_index": True},
            {"rows": 4},
            {"rows": 0},
            {"rows": slice(1, 6)},
            {"rows": slice(1, 6, 2), "engine": "fiona"},
            {"mask": box(1, 1, 4, 4)},
            {"where": "value >= 2", "rows": slice(1, 4)},
            {"bbox": (20, 20, 21, 21)},
            {"rows": slice(20, 25)},
        ],
    )
    def test_open_vector__options_lazy(
        self, filesystem_access: bool, chunks: int | None, options: dict[str, Any], tmp_path: pathlib.Path
    ) -> None:
        """Checks that vector opening options match between lazy/eager exactly."""
        pytest.importorskip("dask_geopandas")
        fsspec = pytest.importorskip("fsspec")
        if filesystem_access or options.get("engine") == "fiona":
            pytest.importorskip("fiona")

        # Create and write geodataframe
        frame = gpd.GeoDataFrame(
            {"value": np.arange(7)}, geometry=gpd.points_from_xy(np.arange(7), np.arange(7)), crs=4326
        )
        path = tmp_path / "points.gpkg"
        frame.to_file(path, index=False)
        filesystem = fsspec.filesystem("dir", path=str(tmp_path), fs=fsspec.filesystem("file"))
        source = path.name if filesystem_access else str(path)
        access = {"filesystem": filesystem} if filesystem_access else {}

        # Open in-memory and lazily with chunks
        expected = gu.open_vector(source, load=True, **access, **options)
        result = gu.open_vector(source, chunks=chunks, **access, **options)
        assert expected.vct.is_loaded and not result.vct.is_loaded
        computed = result.compute()

        # Check equality and loading
        assert computed.vct.is_loaded and not result.vct.is_loaded
        assert_geodataframe_equal(computed, expected)

    @pytest.mark.parametrize("filesystem_access", [False, True])
    def test_open_vector__timezone(self, filesystem_access: bool, tmp_path: pathlib.Path) -> None:
        """Checks that lazy vector dates preserve their timezone across empty partitions."""
        pytest.importorskip("dask_geopandas")
        if filesystem_access:
            pytest.importorskip("fiona")

        frame = gpd.GeoDataFrame(
            {"date": pd.date_range("2020-01-01", periods=7, tz="UTC"), "value": np.arange(7)},
            geometry=gpd.points_from_xy(np.arange(7), np.arange(7)),
            crs=4326,
        )
        path = tmp_path / "dates.gpkg"
        frame.to_file(path, index=False)
        access = {"filesystem": pytest.importorskip("fsspec").filesystem("file")} if filesystem_access else {}
        eager = gu.open_vector(path, load=True, where="value >= 5", **access)
        # Filtering leaves empty final partitions
        lazy = gu.open_vector(path, chunks=3, where="value >= 5", **access)
        assert eager.vct.is_loaded and not lazy.vct.is_loaded

        computed = lazy.compute()

        assert computed.vct.is_loaded and not lazy.vct.is_loaded
        assert_geodataframe_equal(computed, eager)

    @pytest.mark.parametrize("chunks", [None, 3])
    def test_open_vector__independent_partitions(
        self, chunks: int | None, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks that metadata is read once and selecting a partition reads only its rows."""
        pytest.importorskip("dask_geopandas")
        pytest.importorskip("fiona")
        filesystem = pytest.importorskip("fsspec").filesystem("file")
        frame = gpd.GeoDataFrame(
            {"value": np.arange(7)}, geometry=gpd.points_from_xy(np.arange(7), np.arange(7)), crs=4326
        )
        path = tmp_path / "points.gpkg"
        frame.to_file(path, index=False)
        metadata_reads, row_reads = [], []
        read_info, read_file = vector_io._read_vector_info, vector_io._read_vector_file

        def record_info(*args: Any, **kwargs: Any) -> Any:
            """Count metadata reads."""
            metadata_reads.append(args[0])
            return read_info(*args, **kwargs)

        def record_rows(*args: Any, **kwargs: Any) -> Any:
            """Record the requested feature range."""
            row_reads.append(kwargs.get("rows"))
            return read_file(*args, **kwargs)

        monkeypatch.setattr(vector_io, "_read_vector_info", record_info)
        monkeypatch.setattr(vector_io, "_read_vector_file", record_rows)
        lazy = gu.open_vector(path, filesystem=filesystem, chunks=chunks)
        assert len(metadata_reads) == 1 and not lazy.vct.is_loaded
        row_reads.clear()

        computed = lazy.get_partition(lazy.npartitions - 1).compute(scheduler="synchronous")

        start = 0 if chunks is None else 6
        assert row_reads == [slice(start, 7)]
        assert len(metadata_reads) == 1
        assert computed.vct.is_loaded and not lazy.vct.is_loaded
        assert_geodataframe_equal(computed, frame.iloc[start:])

    @pytest.mark.parametrize("chunks", [None, 3])
    def test_open_vector__parquet_filesystem(self, chunks: int | None, tmp_path: pathlib.Path) -> None:
        """Checks that Parquet filesystem reads stay lazy and match eager vector values."""
        pytest.importorskip("dask_geopandas")
        pytest.importorskip("pyarrow")
        filesystem = pytest.importorskip("fsspec").filesystem("memory")
        frame = gpd.GeoDataFrame(
            {"value": np.arange(7)}, geometry=gpd.points_from_xy(np.arange(7), np.arange(7)), crs=4326
        )
        path = str(tmp_path / "points.parquet")
        gu.Vector(frame).to_file(path, filesystem=filesystem)
        eager = gu.open_vector(path, filesystem=filesystem, load=True)

        lazy = gu.open_vector(path, filesystem=filesystem, chunks=chunks)
        assert eager.vct.is_loaded and not lazy.vct.is_loaded
        computed = lazy.compute()

        assert computed.vct.is_loaded and not lazy.vct.is_loaded
        assert_geodataframe_equal(computed, eager)

    @pytest.mark.parametrize("remote_url", ["http-query"], indirect=True)
    def test_open_vector__url_token(self, remote_files: RemoteFiles, remote_url: Callable[[str], str]) -> None:
        """Checks that a URL access token is preserved for metadata and worker reads."""
        pytest.importorskip("dask_geopandas")
        pytest.importorskip("fiona")
        fsspec = pytest.importorskip("fsspec")

        # Creatr and write vector
        frame = gpd.GeoDataFrame(
            {"value": np.arange(7)}, geometry=gpd.points_from_xy(np.arange(7), np.arange(7)), crs=4326
        )
        path = remote_files.directory / "points.gpkg"
        frame.to_file(path, index=False)

        # Add URL access token
        source = remote_url(path.name)
        filesystem = fsspec.filesystem("http", skip_instance_cache=True)

        # Open in-memory and lazy
        vector = gu.Vector(source, filesystem=filesystem)
        result = gu.open_vector(source, filesystem=filesystem, chunks=3)
        assert not vector.is_loaded and not result.vct.is_loaded
        assert_geodataframe_equal(vector.gdf, frame)

        # Read and check equal output
        # (Compute succeeds here only if token was preserved correctly)
        computed = result.compute(scheduler="processes", num_workers=1)
        assert not result.vct.is_loaded and computed.vct.is_loaded
        assert_geodataframe_equal(computed, frame)

    @pytest.mark.parametrize("file_access", ["http-filesystem"], indirect=True)
    @pytest.mark.parametrize(
        "reader,options,filename",
        [
            (gu.open_vector, {}, "regions.gpkg"),
        ],
    )
    def test_open__private_chunks(
        self,
        reader: Callable[..., Any],
        options: dict[str, Any],
        file_access: tuple[Path, str, dict[str, Any]],
    ) -> None:
        """Checks that chunked readers preserve access options and match eager results."""
        pytest.importorskip("dask_geopandas")
        path, source, access_options = file_access
        eager_options = options if reader is gu.open_raster else {**options, "load": True}
        eager = reader(str(path), **eager_options)
        if isinstance(eager, (xr.DataArray, xr.Dataset)):
            eager.load()

        # Check shorter final chunks
        remote = reader(source, chunks=3, **options, **access_options)
        variable = remote[next(iter(remote.data_vars))] if isinstance(remote, xr.Dataset) else remote
        accessor = "rst" if reader is gu.open_raster else "vct" if reader is gu.open_vector else "pc"
        interface = getattr(variable, accessor)
        eager_variable = eager[next(iter(eager.data_vars))] if isinstance(eager, xr.Dataset) else eager
        assert getattr(eager_variable, accessor).is_loaded
        assert not interface.is_loaded

        # Run process workers
        computed = remote.compute(scheduler="processes", num_workers=1)
        assert not interface.is_loaded
        computed_variable = computed[next(iter(computed.data_vars))] if isinstance(computed, xr.Dataset) else computed
        assert getattr(computed_variable, accessor).is_loaded
        assert_output_equal(computed, eager)
        if isinstance(remote, (xr.DataArray, xr.Dataset)):
            remote.close()
            eager.close()


class TestVectorWriting:
    """Test module for saving and reopening Vector objects."""

    aster_outlines_path = gu.examples.get_path_test("exploradores_rgi_outlines")

    @pytest.mark.parametrize("representation", ["class", "geodataframe"])
    def test_to_file(self, representation: str, tmp_path: Path) -> None:
        """Checks that to_file() preserves vector features through the class and accessor."""

        vector = gu.Vector(self.aster_outlines_path)
        writer = vector if representation == "class" else vector.gdf.vct

        # Create a temporary file in a temporary directory
        temp_file = str(tmp_path / "test.gpkg")

        # Save and check the file exists
        writer.to_file(temp_file)
        assert os.path.exists(temp_file)

        # Open and check the object is the same
        vector_save = gu.Vector(temp_file)
        assert vector_save.vector_equal(vector)

    def test_to_file__shapefile_filesystem(self, tmp_path: pathlib.Path) -> None:
        """
        Checks that writing to a Shapefile using a filesystem properly includes additional "companion files" such as
        .shx/.prj, in addition to the central .shp.
        """

        pytest.importorskip("fiona")
        filesystem = pytest.importorskip("fsspec").filesystem("dir", path=str(tmp_path))
        frame = gpd.GeoDataFrame({"value": [1, 2]}, geometry=gpd.points_from_xy([0, 1], [2, 3]), crs=4326)

        gu.Vector(frame).to_file("points.shp", filesystem=filesystem)

        assert all(filesystem.exists("points" + suffix) for suffix in (".shp", ".shx", ".dbf", ".prj"))
        assert_geodataframe_equal(gpd.read_file(tmp_path / "points.shp"), frame, check_dtype=False)

    @pytest.mark.parametrize("representation", ["class", "geodataframe"])
    @pytest.mark.parametrize("access", ["filesystem", "storage_options"])
    def test_to_file__filesystem(self, representation: str, access: str, tmp_path: pathlib.Path) -> None:
        """Checks that to_file() uses access options and preserves values and CRS."""
        pytest.importorskip("fiona")
        filesystem = pytest.importorskip("fsspec").filesystem("memory")
        path = tmp_path / "output.geojson"
        filename = str(path) if access == "filesystem" else f"memory://{path}"
        options = {"filesystem": filesystem} if access == "filesystem" else {"storage_options": {}}

        # Create test data
        points = gu.PointCloud.from_xyz(np.arange(7.0), np.arange(7.0), np.arange(7.0) + 10, 4326, "height")
        points.gdf["quality"] = np.arange(7, dtype=np.uint16)
        writer = gu.Vector(points.gdf) if representation == "class" else points.gdf.vct

        writer.to_file(filename, **options)
        assert filesystem.exists(str(path)) and not path.exists()

        # Read the filesystem output
        result = gu.open_vector(str(path), filesystem=filesystem, load=True)
        assert_geodataframe_equal(result, points.gdf, check_dtype=False, check_like=True)


class TestVectorWritingChunked:
    """Test module for lazy vector writing to GeoPackage and GeoParquet."""

    aster_outlines_path = gu.examples.get_path_test("exploradores_rgi_outlines")

    @pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Only runs if dask-geopandas is installed.")
    def test_to_file__dask_geopandas_gpkg(self) -> None:
        """Compute lazy vector partitions while writing a single GeoPackage."""

        # Include a lazy reprojection so writing evaluates the complete task graph
        ds = gu.open_vector(self.aster_outlines_path, chunks=1).vct.reproject(crs=CRS.from_epsg(3857))
        assert not ds.vct.is_loaded

        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        ds.vct.to_file(temp_file)

        # Reopen the file eagerly and compare it with the projected source
        assert os.path.exists(temp_file)
        assert gu.Vector(temp_file).vector_equal(
            gu.Vector(gu.open_vector(self.aster_outlines_path, load=True).to_crs(3857))
        )
        assert not ds.vct.is_loaded

    @pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Only runs if dask-geopandas is installed.")
    def test_to_file__dask_geopandas_parquet(self) -> None:
        """Write lazy vector partitions to GeoParquet without changing features."""

        pytest.importorskip("pyarrow")

        # GeoParquet supports a partitioned Dask write directly
        ds = gu.open_vector(self.aster_outlines_path, chunks=1)
        assert not ds.vct.is_loaded

        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.parquet")
        ds.vct.to_file(temp_file)

        # Read through GeoPandas to validate the complete written dataset
        assert os.path.exists(temp_file)
        assert_geodataframe_equal(gpd.read_parquet(temp_file), gu.open_vector(self.aster_outlines_path, load=True))
        assert not ds.vct.is_loaded

    @pytest.mark.parametrize("fragment", ["", "#version"])
    def test_to_file__parquet_filesystem_fragment(self, fragment: str, tmp_path: pathlib.Path) -> None:
        """Checks that a URL fragment uses the Parquet writer without loading the input."""
        pytest.importorskip("dask_geopandas")
        pytest.importorskip("pyarrow")
        filesystem = pytest.importorskip("fsspec").filesystem("memory")
        eager = gu.open_vector(self.aster_outlines_path, load=True)
        lazy = gu.open_vector(self.aster_outlines_path, chunks=1)
        destination = f"memory://{tmp_path}/output.parquet{fragment}"
        assert eager.vct.is_loaded and not lazy.vct.is_loaded

        lazy.vct.to_file(destination, filesystem=filesystem)
        computed = gpd.read_parquet(filesystem._strip_protocol(destination), filesystem=filesystem)

        assert computed.vct.is_loaded and not lazy.vct.is_loaded
        assert_geodataframe_equal(computed, eager)


class TestVectorIOErrors:
    """Test module for missing optional dependencies when opening chunked vector files."""

    aster_outlines_path = gu.examples.get_path_test("exploradores_rgi_outlines")

    @pytest.mark.parametrize(
        "rows", [-1, slice(-1, None), slice(None, -1), slice(None, None, -1), slice(None, None, 0)]
    )
    def test_open_vector__error_lazy_rows(self, rows: int | slice, tmp_path: pathlib.Path) -> None:
        """Checks an error is raised for negative row bounds or non-positive steps in lazy reads."""
        pytest.importorskip("dask_geopandas")
        frame = gpd.GeoDataFrame({"value": [1, 2]}, geometry=gpd.points_from_xy([0, 1], [0, 1]), crs=4326)
        path = tmp_path / "rows.gpkg"
        frame.to_file(path)

        with pytest.raises(ValueError, match="Lazy vector reads require non-negative row bounds"):
            gu.open_vector(path, chunks=1, rows=rows)

    @pytest.mark.parametrize("reader", [gu.Vector, gu.open_vector])
    def test_open_or_init__error_fiona_shapefile(self, reader: Callable[..., object], tmp_path: pathlib.Path) -> None:
        """Checks an error is raised before Fiona filesystem reads can return empty Shapefile data."""
        pytest.importorskip("fiona")
        filesystem = pytest.importorskip("fsspec").filesystem("file")
        frame = gpd.GeoDataFrame({"value": [1, 2]}, geometry=gpd.points_from_xy([0, 1], [2, 3]), crs=4326)
        path = tmp_path / "points.shp"
        frame.to_file(path)

        with pytest.raises(NotImplementedError, match="Fiona cannot reliably read Shapefiles"):
            reader(path, filesystem=filesystem)

    def test_open_vector__dask_geopandas_missing_dep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explain the missing optional dependency when chunked vector opening requests Dask."""

        import_optional = vector_io.import_optional

        def _missing_dask_geopandas(
            import_name: str, package_name: str | None = None, extra_name: str = "opt"
        ) -> object:
            """Fail only the Dask-GeoPandas import while forwarding other imports."""

            if import_name == "dask_geopandas":
                raise ImportError("Optional dependency 'dask-geopandas' required.")
            return import_optional(import_name, package_name=package_name, extra_name=extra_name)

        # Replace the module-level optional importer used by ``open_vector``
        monkeypatch.setattr(vector_io, "import_optional", _missing_dask_geopandas)
        with pytest.raises(ImportError, match="Optional dependency 'dask-geopandas' required.*"):
            gu.open_vector(self.aster_outlines_path, chunks=1)
