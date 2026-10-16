"""Tests for LasPy-backed point-cloud IO helpers."""

from __future__ import annotations

import io
import os
import struct
import tempfile
from contextlib import nullcontext
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

import geoutils as gu
import geoutils.pointcloud.las as las_module
from geoutils._misc import import_optional
from geoutils.multiproc import MultiprocConfig
from geoutils.pointcloud.las import (
    _iter_laspy_spatial_chunks,
    _load_laspy_data_bounds,
    _load_laspy_data_slice,
    _load_laspy_metadata,
    _select_copc_nodes,
    _spatial_bounds_grid,
    _write_laspy_spatial_chunks,
)

laspy = pytest.importorskip("laspy")


class TestLasPyIO:
    """Test LasPy-specific loading, spatial chunking and writing."""

    # Arrange a small regular point cloud with one native LAS attribute
    x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    z = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    intensity = np.array([100, 110, 120, 130, 140, 150], dtype="uint16")
    gdf = gpd.GeoDataFrame(
        data={"z": z, "intensity": intensity},
        geometry=gpd.points_from_xy(x=x, y=y),
        crs=4326,
    )

    @staticmethod
    def _write_source(pc: gu.PointCloud, directory: str, filename: str = "source.las") -> str:
        """Write a precisely scaled LAS source shared by reading and writing tests."""

        # Fixed scales and offsets avoid rounding differences between test runs
        path = os.path.join(directory, filename)
        pc.to_las(
            path,
            point_format=3,
            scales=(0.001, 0.001, 0.001),
            offsets=(0.0, 0.0, 0.0),
        )
        return path

    @staticmethod
    def _assert_roundtrip(pc: gu.PointCloud, filename: str) -> None:
        """Check that a written LAS file retains coordinates, values, attributes and CRS."""

        # Load only the dimensions used in the expected point cloud
        saved = gu.PointCloud(filename)
        saved.load(columns=["Z", "intensity"])

        # Coordinates use approximate equality because LAS applies integer scaling
        assert np.allclose(saved.geometry.x.values, pc.geometry.x.values)
        assert np.allclose(saved.geometry.y.values, pc.geometry.y.values)
        assert np.allclose(saved.data, pc.data)
        # Native attributes and CRS should round-trip exactly
        assert np.array_equal(saved["intensity"].values, pc["intensity"].values)
        assert saved.crs == pc.crs

    def test_load_laspy_metadata_slice_and_bounds(self) -> None:
        """Load metadata, point-index slices and coordinate-filtered chunks."""

        # Write one source used by each increasingly selective reader
        pc = gu.PointCloud(self.gdf, data_name="z")
        with tempfile.TemporaryDirectory() as temp_dir:
            source = self._write_source(pc, temp_dir)

            # Metadata reads the header without loading point records
            metadata = _load_laspy_metadata(source)
            assert metadata.point_count == len(self.gdf)
            assert "Z" in metadata.columns
            assert "intensity" in metadata.columns
            assert not metadata.is_copc

            # Index slicing should return one contiguous range of records
            sliced = _load_laspy_data_slice(source, columns=["Z", "intensity"], start=1, count=3)
            assert len(sliced) == 3
            assert np.allclose(sliced["Z"].values, self.z[1:4])
            assert np.array_equal(sliced["intensity"].values, self.intensity[1:4])

            # Coordinate filtering streams small chunks for a regular non-COPC file
            bounded = _load_laspy_data_bounds(
                source,
                columns=["Z", "intensity"],
                bounds=(0.0, 0.0, 1.0, 1.0),
                chunk_size=2,
                prefer_copc=False,
            )
            # Inclusive outer bounds select the four points in the left square
            assert len(bounded) == 4
            assert set(np.round(bounded.geometry.x.values, 6)) == {0.0, 1.0}
            assert set(np.round(bounded.geometry.y.values, 6)) == {0.0, 1.0}

    def test_crop__deferred_las(self) -> None:
        """Checks that crop() does not read LAS points until they are needed."""

        # Write six points on a regular grid with known coordinates and values
        pointcloud = gu.PointCloud(self.gdf, data_name="z")
        with tempfile.TemporaryDirectory() as temp_dir:
            source = self._write_source(pointcloud, temp_dir)
            unloaded = gu.PointCloud(source)

            # Select four points without reading the source or result
            cropped = unloaded.crop((-0.1, -0.1, 1.1, 1.1))
            assert not unloaded.is_loaded
            assert not cropped.is_loaded

            # Read the result and compare it with the four expected points
            assert np.allclose(cropped.geometry.x.values, self.x[[0, 1, 3, 4]])
            assert np.allclose(cropped.geometry.y.values, self.y[[0, 1, 3, 4]])
            assert np.allclose(cropped.data, self.z[[0, 1, 3, 4]])
            assert not unloaded.is_loaded

    def test_select_copc_nodes__hierarchy_pages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that COPC selection follows child hierarchy pages and orders selected nodes by file position."""

        from laspy.copc import HierarchyPage

        # Create a small COPC index, which locates groups of points (nodes) without reading them
        # We pack entries as (level, voxel X/Y/Z, byte offset, byte size, point count), as in a real file
        # Store one level 1 node in a separate page, linked from the root page by a point count of -1
        child_bytes = struct.pack("<iiiiQii", 1, 0, 0, 0, 10, 8, 3)
        root_bytes = b"".join(
            [
                struct.pack("<iiiiQii", 1, 0, 0, 0, 0, len(child_bytes), -1),
                struct.pack("<iiiiQii", 0, 0, 0, 0, 20, 8, 2),
                struct.pack("<iiiiQii", 2, 0, 0, 0, 30, 8, 7),
                struct.pack("<iiiiQii", 1, 1, 0, 0, 40, 8, 5),
            ]
        )

        # Replace the file reader with our index bytes so node selection needs no actual point data
        # The header counts all 3 + 2 + 7 + 5 points
        reader = SimpleNamespace(
            root_page=HierarchyPage.from_bytes(root_bytes),
            source=io.BytesIO(child_bytes),
            header=SimpleNamespace(point_count=17),
        )
        monkeypatch.setattr(las_module, "_open_las_reader", lambda filename, **kwargs: nullcontext(reader))

        # Run both selection modes: request one level or gather enough points for downsampling
        # Automatic selection adds whole levels from the root until it reaches the target
        explicit = _select_copc_nodes("hierarchy.copc.laz", level=1, target_count=None)
        automatic = _select_copc_nodes("hierarchy.copc.laz", level=None, target_count=6)

        # Check nodes are sorted by file offset, including the node reached through the separate page
        # Level 1 has 8 points: adding the root gives 10, so level 2 is unnecessary for a target of 6
        assert [(node.offset, node.point_count) for node in explicit] == [(10, 3), (40, 5)]
        assert [(node.offset, node.point_count) for node in automatic] == [(10, 3), (20, 2), (40, 5)]

    def test_load_laspy_bounds__copc_errors_are_not_hidden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Propagate indexed-reader failures when the LAS header identifies a COPC file."""

        pc = gu.PointCloud(self.gdf, data_name="z")
        with tempfile.TemporaryDirectory() as temp_dir:
            source = self._write_source(pc, temp_dir)
            metadata = replace(_load_laspy_metadata(source), is_copc=True)
            monkeypatch.setattr(las_module, "_load_laspy_metadata", lambda filename: metadata)

            def _raise_copc_error(**kwargs: object) -> gpd.GeoDataFrame:
                raise RuntimeError("invalid COPC index")

            monkeypatch.setattr(las_module, "_load_laspy_data_bounds_copc", _raise_copc_error)
            with pytest.raises(RuntimeError, match="invalid COPC index"):
                _load_laspy_data_bounds(source, columns=["Z"], bounds=(0, 0, 1, 1))

    def test_spatial_chunks_select_points_by_xy_blocks(self) -> None:
        """Split LAS points into X/Y blocks without edge duplicates."""

        # Divide the source extent into neighboring horizontal blocks
        pc = gu.PointCloud(self.gdf, data_name="z")
        with tempfile.TemporaryDirectory() as temp_dir:
            source = self._write_source(pc, temp_dir)

            blocks = _spatial_bounds_grid(bounds=(0.0, 0.0, 2.0, 1.0), block_size=(1.0, 1.0))
            # Route one streamed source pass into each spatial selection
            chunks = list(
                _iter_laspy_spatial_chunks(
                    source,
                    block_bounds=blocks,
                    columns=["Z", "intensity"],
                    chunk_size=2,
                )
            )

        # Boundary rules must assign every point to exactly one block
        assert len(chunks) == 2
        assert [len(chunk) for _, _, chunk in chunks] == [2, 4]
        chunk_z = np.concatenate([chunk["Z"].values for _, _, chunk in chunks])
        assert np.array_equal(np.sort(chunk_z), np.sort(self.z))

    def test_write_laspy_spatial_chunks(self) -> None:
        """Write LAS points into X/Y block files."""

        # Reuse the two neighboring blocks from the iterator test
        pc = gu.PointCloud(self.gdf, data_name="z")
        with tempfile.TemporaryDirectory() as temp_dir:
            source = self._write_source(pc, temp_dir)
            blocks = _spatial_bounds_grid(bounds=(0.0, 0.0, 2.0, 1.0), block_size=(1.0, 1.0))
            output_dir = os.path.join(temp_dir, "blocks")

            # Write both outputs during one sequential read of the source LAS file
            output_files = _write_laspy_spatial_chunks(
                source,
                output_dir=output_dir,
                block_bounds=blocks,
                chunk_size=2,
            )

            # Reopen each block independently to validate its point count
            chunk_counts = []
            for output_file in output_files:
                chunk = gu.PointCloud(output_file)
                chunk.load(columns=["Z", "intensity"])
                chunk_counts.append(chunk.point_count)

        assert chunk_counts == [2, 4]

    @pytest.mark.parametrize("as_object_type", ["pointcloud", "dataarray", "dataset", "geodataframe"])
    @pytest.mark.parametrize("point_format", [3, 6])
    @pytest.mark.parametrize("laz_chunk_size", [1, 3, 20])
    def test_to_las__disk_chunks(
        self, tmp_path: Path, as_object_type: str, point_format: int, laz_chunk_size: int
    ) -> None:
        """Checks that to_las() writes requested LAZ disk chunks for all representations."""

        lazrs = import_optional("lazrs")

        # Create points with uneven chunks and dtypes
        reference = gu.PointCloud.from_xyz(np.arange(11.0), np.arange(11.0) + 1, np.arange(11.0), 32633, "Z")
        reference["classification"] = np.arange(11, dtype=np.uint8)
        reference["identifier"] = np.arange(11, dtype=np.uint64) + 2**63
        array = reference.to_xarray()
        representations: dict[str, Any] = {
            "pointcloud": reference,
            "geodataframe": reference.gdf,
            "dataarray": array,
            "dataset": array.to_dataset().reset_coords(["classification", "identifier"]),
        }

        # Write with different chunk sizes and formats
        points = representations[as_object_type]
        writer = points if as_object_type == "pointcloud" else points.pc
        filename = tmp_path / "points.laz"
        writer.to_las(filename, point_format=point_format, chunks=4, laz_chunk_size=laz_chunk_size)

        # Read chunk sizes independently to check
        with laspy.open(filename) as reader:
            compression = lazrs.LazVlr(reader.header.vlrs.get("LasZipVlr")[0].record_data)
            assert compression.chunk_size() == laz_chunk_size
            with filename.open("rb") as stream:
                stream.seek(reader.header.offset_to_point_data)
                disk_chunks = lazrs.read_chunk_table(stream, compression)
            assert len(disk_chunks) == (11 + laz_chunk_size - 1) // laz_chunk_size

            # Check equality across compressed chunks
            for start in (0, 2, 3, 4, 9):
                reader.seek(start)
                selected = reader.read_points(2)
                np.testing.assert_array_equal(selected.z, np.arange(11.0)[start : start + 2])
                np.testing.assert_array_equal(selected.identifier, reference["identifier"].values[start : start + 2])

        # Reopen with GeoUtils and check exact equality
        saved = gu.open_pointcloud(filename, columns=["classification", "identifier"])
        assert saved.attrs["laz_chunk_size"] == laz_chunk_size
        assert saved.pc.is_loaded
        if as_object_type == "dataset":
            assert all(variable.pc.is_loaded for variable in points.data_vars.values())
        else:
            assert writer.is_loaded
        assert saved.pc.crs == reference.crs
        np.testing.assert_array_equal(saved.pc.to_array(), reference.to_array())
        np.testing.assert_array_equal(saved.classification, reference["classification"])
        np.testing.assert_array_equal(saved.identifier, reference["identifier"])

    def test_to_las_chunked_pandas_and_multiproc(self) -> None:
        """Write in-memory point clouds by chunks."""

        # Compare both chunk schedulers against the same in-memory source
        pc = gu.PointCloud(self.gdf, data_name="z")
        with tempfile.TemporaryDirectory() as temp_dir:
            # The Pandas path writes row chunks sequentially
            chunked = os.path.join(temp_dir, "chunked.las")
            pc.to_las(
                chunked,
                point_format=3,
                chunks=2,
                scales=(0.001, 0.001, 0.001),
                offsets=(0.0, 0.0, 0.0),
            )
            self._assert_roundtrip(pc, chunked)

            # The multiprocessing path creates chunks in workers before stitching them
            multiproc = os.path.join(temp_dir, "multiproc.las")
            pc.to_las(
                multiproc,
                point_format=3,
                mp_config=MultiprocConfig(chunks=2),
                scales=(0.001, 0.001, 0.001),
                offsets=(0.0, 0.0, 0.0),
            )
            self._assert_roundtrip(pc, multiproc)

    @pytest.mark.parametrize("as_object_type", ["dataarray", "geodataframe"])
    def test_to_las_chunked_dask(self, as_object_type: Literal["dataarray", "geodataframe"]) -> None:
        """Write a Dask-backed LAS point cloud partition by partition."""

        # Skip cleanly when lazy GeoDataFrames are unavailable
        pytest.importorskip("dask")
        if as_object_type == "geodataframe":
            dgpd = pytest.importorskip("dask_geopandas")

        # Write and reopen a source as several lazy LAS partitions
        pc = gu.PointCloud(self.gdf, data_name="z")
        with tempfile.TemporaryDirectory() as temp_dir:
            source = self._write_source(pc, temp_dir, filename="dask-source.las")

            ds = gu.open_pointcloud(source, columns="all", chunks=2, as_object_type=as_object_type)
            assert isinstance(ds, xr.DataArray if as_object_type == "dataarray" else dgpd.GeoDataFrame)
            assert not ds.pc.is_loaded

            # The writer computes one partition at a time into the final LAS stream
            output = os.path.join(temp_dir, "dask-output.las")
            ds.pc.to_las(
                output,
                point_format=3,
                scales=(0.001, 0.001, 0.001),
                offsets=(0.0, 0.0, 0.0),
            )
            # Validate the resulting file through an independent eager reader
            self._assert_roundtrip(pc, output)
            assert not ds.pc.is_loaded

    @pytest.mark.parametrize("suffix", ["las", "laz"])
    def test_to_las__roundtrip(self, tmp_path: Path, suffix: str) -> None:
        """Checks direct array LAS/LAZ opening/writing from the `pc` DataArray accessor in a round trip."""

        if suffix == "laz":
            pytest.importorskip("lazrs")

        # Create a point cloud DataArray with "classification" and "id" variables with different types to check they
        # remain separate from floating values/coords
        points = gu.DataArrayPointCloudAccessor.from_xyz(
            np.arange(11.0),
            np.arange(11.0) + 1,
            np.arange(11.0),
            32633,
            data_name="Z",
            auxiliary={"classification": np.arange(11, dtype=np.uint8), "id": np.arange(11, dtype=np.uint64) + 2**63},
        )
        filename = tmp_path / f"points.{suffix}"

        # We write in chunks, then check we get exactly the same values when reading again
        points.pc.to_las(str(filename), chunks=4, scales=(0.001, 0.001, 0.001))
        result = gu.open_pointcloud(str(filename), columns=["classification", "id"])
        np.testing.assert_array_equal(result.pc.to_array(), points.pc.to_array())
        np.testing.assert_array_equal(result.coords["classification"], points.coords["classification"])
        np.testing.assert_array_equal(result.coords["id"], points.coords["id"])
        assert result.pc.crs == points.pc.crs
        assert result.coords["id"].dtype == np.uint64

    def test_to_las__active_values(self, tmp_path: Path) -> None:
        """Checks that selecting intensity as active values preserves the native LAS elevation dimension."""

        # We create a DataArray where elevation differs from selected intensity values
        points = gu.DataArrayPointCloudAccessor.from_xyz(
            [0.0, 1.0],
            [2.0, 3.0],
            np.array([5, 6], dtype=np.uint16),
            32633,
            data_name="intensity",
            auxiliary={"Z": [10.0, 20.0]},
        )
        filename = tmp_path / "intensity.las"

        # Check that the LAS writer stores each native dimension according to its meaning
        points.pc.to_las(str(filename))
        result = laspy.read(filename)
        np.testing.assert_array_equal(result.z, [10, 20])
        np.testing.assert_array_equal(result.intensity, [5, 6])

    @pytest.mark.parametrize(
        "as_object_type, backend",
        [
            ("dataarray", "dask"),
            ("dataset", "dask"),
            ("geodataframe", "dask"),
            ("pointcloud", "multiproc"),
            ("geodataframe", "multiproc"),
        ],
    )
    def test_to_las__disk_chunks_chunked(
        self,
        tmp_path: Path,
        as_object_type: Literal["pointcloud", "dataarray", "dataset", "geodataframe"],
        backend: Literal["dask", "multiproc"],
    ) -> None:
        """Checks that Dask and multiprocessing writes use specified LAZ disk chunks and match eager writing."""

        import_optional("lazrs")

        # We write uneven chunked point cloud with multiple dtypes
        reference = gu.PointCloud.from_xyz(np.arange(11.0), np.arange(11.0) + 1, np.arange(11.0), 32633, "Z")
        reference["classification"] = np.arange(11, dtype=np.uint8)
        reference["identifier"] = np.arange(11, dtype=np.uint64) + 2**63
        reference["intensity"] = np.arange(11, dtype=np.uint16)
        columns = ["classification", "identifier", "intensity"]

        # Open eager and chunked
        if as_object_type == "pointcloud":
            eager_writer = chunked_writer = reference
            eager_accessors = chunked_accessors = [reference]
        else:
            filename = tmp_path / "source.las"
            reference.to_las(filename)
            eager = gu.open_pointcloud(filename, columns=columns, as_object_type=as_object_type)
            chunked = gu.open_pointcloud(
                filename, columns=columns, as_object_type=as_object_type, chunks=4 if backend == "dask" else None
            )
            eager_writer, chunked_writer = eager.pc, chunked.pc
            eager_variables = list(eager.data_vars.values()) if as_object_type == "dataset" else [eager]
            chunked_variables = list(chunked.data_vars.values()) if as_object_type == "dataset" else [chunked]
            eager_accessors = [variable.pc for variable in eager_variables]
            chunked_accessors = [variable.pc for variable in chunked_variables]
        expected_loaded = backend == "multiproc"
        assert all(accessor.is_loaded for accessor in eager_accessors)
        assert all(accessor.is_loaded == expected_loaded for accessor in chunked_accessors)

        # Write eagerly and with Dask/MP
        eager_filename = tmp_path / "eager.laz"
        chunked_filename = tmp_path / "chunked.laz"
        eager_writer.to_las(eager_filename, chunks=4, laz_chunk_size=3)
        if backend == "dask":
            chunked_writer.to_las(chunked_filename, chunks=4, laz_chunk_size=3)
        else:
            chunked_writer.to_las(chunked_filename, mp_config=MultiprocConfig(chunks=4), laz_chunk_size=3)
        assert all(accessor.is_loaded == expected_loaded for accessor in chunked_accessors)

        # Check exact equality and chunk size metadata
        saved = gu.open_pointcloud(chunked_filename, columns=columns, chunks=4)
        expected = gu.open_pointcloud(eager_filename, columns=columns)
        assert not saved.pc.is_loaded and expected.pc.is_loaded
        assert saved.attrs["laz_chunk_size"] == expected.attrs["laz_chunk_size"] == 3
        xr.testing.assert_identical(saved.compute(), expected)
        np.testing.assert_array_equal(expected.pc.to_array(), reference.to_array())
        assert all(accessor.is_loaded == expected_loaded for accessor in chunked_accessors)
        assert not saved.pc.is_loaded


class TestLasErrors:
    """Test module for errors/warnings for LAS/LAZ/COPC functions."""

    @pytest.mark.parametrize("as_object_type", ["pointcloud", "dataarray"])
    @pytest.mark.parametrize("laz_chunk_size", [0, -1, True, 1.5, "3", 0xFFFFFFFF, 2**32])
    def test_to_las__error_invalid_disk_chunks(self, tmp_path: Path, as_object_type: str, laz_chunk_size: Any) -> None:
        """Checks an error is raised for invalid LAZ disk chunk sizes."""

        cloud = gu.PointCloud.from_xyz([0.0], [1.0], [2.0], 32633, "Z")
        writer = cloud if as_object_type == "pointcloud" else cloud.to_xarray().pc
        filename = tmp_path / "points.laz"
        filename.write_bytes(b"Existing output")

        with pytest.raises(ValueError, match="laz_chunk_size.*integer between"):
            writer.to_las(filename, laz_chunk_size=laz_chunk_size)
        assert filename.read_bytes() == b"Existing output"

    @pytest.mark.parametrize("as_object_type", ["pointcloud", "dataarray"])
    @pytest.mark.parametrize("suffix", ["las", "copc.laz"])
    def test_to_las__error_disk_chunks_other_formats(self, tmp_path: Path, as_object_type: str, suffix: str) -> None:
        """Checks an error is raised for LAZ disk chunk options on LAS or COPC."""

        cloud = gu.PointCloud.from_xyz([0.0], [1.0], [2.0], 32633, "Z")
        writer = cloud if as_object_type == "pointcloud" else cloud.to_xarray().pc
        filename = tmp_path / f"points.{suffix}"

        with pytest.raises(ValueError, match="laz_chunk_size.*only supported for LAZ"):
            writer.to_las(filename, laz_chunk_size=3)
        assert not filename.exists()

    @pytest.mark.skipif(find_spec("lazrs") is not None, reason="Only runs when lazrs is installed")
    def test_to_las__error_disk_chunks_missing_lazrs(self, tmp_path: Path) -> None:
        """Checks an error is raised for a missing Lazrs dependency before output is overwritten."""

        points = gu.DataArrayPointCloudAccessor.from_xyz([0.0], [1.0], [2.0], 32633)
        filename = tmp_path / "points.laz"
        filename.write_bytes(b"Existing output")

        # Check we fail before LasPy can open or truncate the destination
        with pytest.raises(ImportError, match="lazrs"):
            points.pc.to_las(filename, laz_chunk_size=3)
        assert filename.read_bytes() == b"Existing output"
