"""Tests for LasPy-backed point-cloud IO helpers."""

from __future__ import annotations

import os
import tempfile
import warnings
from importlib.util import find_spec

import geopandas as gpd
import numpy as np
import pytest

import geoutils as gu
from geoutils.multiproc import MultiprocConfig
from geoutils.pointcloud.las import (
    iter_laspy_spatial_chunks,
    load_laspy_data_bounds,
    load_laspy_data_slice,
    load_laspy_metadata,
    spatial_bounds_grid,
    write_laspy_spatial_chunks,
)


pytestmark = pytest.mark.skipif(find_spec("laspy") is None, reason="Only runs if laspy is installed.")


class TestLasPyIO:
    """Test LasPy-specific loading, spatial chunking and writing."""

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
        saved = gu.PointCloud(filename)
        saved.load(columns=["Z", "intensity"])

        assert np.allclose(saved.geometry.x.values, pc.geometry.x.values)
        assert np.allclose(saved.geometry.y.values, pc.geometry.y.values)
        assert np.allclose(saved.data, pc.data)
        assert np.array_equal(saved["intensity"].values, pc["intensity"].values)
        assert saved.crs == pc.crs

    def test_load_laspy_metadata_slice_and_bounds(self) -> None:
        """Load metadata, point-index slices and coordinate-filtered chunks."""

        pc = gu.PointCloud(self.gdf, data_column="z")
        with tempfile.TemporaryDirectory() as temp_dir:
            source = self._write_source(pc, temp_dir)

            metadata = load_laspy_metadata(source)
            assert metadata.point_count == len(self.gdf)
            assert "Z" in metadata.columns
            assert "intensity" in metadata.columns

            sliced = load_laspy_data_slice(source, columns=["Z", "intensity"], start=1, count=3)
            assert len(sliced) == 3
            assert np.allclose(sliced["Z"].values, self.z[1:4])
            assert np.array_equal(sliced["intensity"].values, self.intensity[1:4])

            bounded = load_laspy_data_bounds(
                source,
                columns=["Z", "intensity"],
                bounds=(0.0, 0.0, 1.0, 1.0),
                chunk_size=2,
                prefer_copc=False,
            )
            assert len(bounded) == 4
            assert set(np.round(bounded.geometry.x.values, 6)) == {0.0, 1.0}
            assert set(np.round(bounded.geometry.y.values, 6)) == {0.0, 1.0}

    def test_spatial_chunks_select_points_by_xy_blocks(self) -> None:
        """Split LAS points into X/Y blocks without edge duplicates."""

        pc = gu.PointCloud(self.gdf, data_column="z")
        with tempfile.TemporaryDirectory() as temp_dir:
            source = self._write_source(pc, temp_dir)

            blocks = spatial_bounds_grid(bounds=(0.0, 0.0, 2.0, 1.0), block_size=(1.0, 1.0))
            chunks = list(
                iter_laspy_spatial_chunks(
                    source,
                    block_bounds=blocks,
                    columns=["Z", "intensity"],
                    chunk_size=2,
                )
            )

        assert len(chunks) == 2
        assert [len(chunk) for _, _, chunk in chunks] == [2, 4]
        chunk_z = np.concatenate([chunk["Z"].values for _, _, chunk in chunks])
        assert np.array_equal(np.sort(chunk_z), np.sort(self.z))

    def test_write_laspy_spatial_chunks(self) -> None:
        """Write LAS points into X/Y block files."""

        pc = gu.PointCloud(self.gdf, data_column="z")
        with tempfile.TemporaryDirectory() as temp_dir:
            source = self._write_source(pc, temp_dir)
            blocks = spatial_bounds_grid(bounds=(0.0, 0.0, 2.0, 1.0), block_size=(1.0, 1.0))
            output_dir = os.path.join(temp_dir, "blocks")

            output_files = write_laspy_spatial_chunks(
                source,
                output_dir=output_dir,
                block_bounds=blocks,
                columns=["Z", "intensity"],
                chunk_size=2,
            )

            chunk_counts = []
            for output_file in output_files:
                chunk = gu.PointCloud(output_file)
                chunk.load(columns=["Z", "intensity"])
                chunk_counts.append(chunk.point_count)

        assert chunk_counts == [2, 4]

    def test_to_las_chunked_pandas_and_multiproc(self) -> None:
        """Write in-memory point clouds by chunks."""

        pc = gu.PointCloud(self.gdf, data_column="z")
        with tempfile.TemporaryDirectory() as temp_dir:
            chunked = os.path.join(temp_dir, "chunked.las")
            pc.to_las(
                chunked,
                point_format=3,
                chunks=2,
                scales=(0.001, 0.001, 0.001),
                offsets=(0.0, 0.0, 0.0),
            )
            self._assert_roundtrip(pc, chunked)

            multiproc = os.path.join(temp_dir, "multiproc.las")
            pc.to_las(
                multiproc,
                point_format=3,
                mp_config=MultiprocConfig(chunks=2),
                scales=(0.001, 0.001, 0.001),
                offsets=(0.0, 0.0, 0.0),
            )
            self._assert_roundtrip(pc, multiproc)

    def test_to_las_chunked_dask(self) -> None:
        """Write a Dask-backed LAS point cloud partition by partition."""

        pytest.importorskip("dask")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
            import dask.dataframe as dd

        pc = gu.PointCloud(self.gdf, data_column="z")
        with tempfile.TemporaryDirectory() as temp_dir:
            source = self._write_source(pc, temp_dir, filename="dask-source.las")

            ds = gu.open_pointcloud(source, columns="all", chunks=2)
            assert isinstance(ds, dd.DataFrame)
            assert not ds.pc.is_loaded

            output = os.path.join(temp_dir, "dask-output.las")
            ds.pc.to_las(
                output,
                point_format=3,
                scales=(0.001, 0.001, 0.001),
                offsets=(0.0, 0.0, 0.0),
            )
            self._assert_roundtrip(pc, output)
