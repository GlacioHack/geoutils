"""Tests reusable raster and point cloud readers used by the Multiproc backend."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
from numpy.typing import NDArray
from pyproj import CRS
from rasterio.transform import from_origin

import geoutils as gu
from geoutils.multiproc import ClusterGenerator, MultiprocConfig
from geoutils.multiproc.readers import (
    _read_values,
    _reader_from_source,
    _ValueReader,
)


class TestValueReaderChunked:
    """
    Test module for reading values from raster and point cloud files without loading the full objects.

    In the module below, we test:
    - For rasters, the reading of bands, of downsampled rasters, and of nodata and masks.
    - For point clouds, the reading of vector file columns, of geometry heights (Z axis) and of LAS attributes.
    """

    @pytest.mark.parametrize(
        "source_bands,band,downsample,workers",
        [(None, 2, 1, False), ([3, 1], 1, 1, True), ([3, 1], 2, 2, False)],
    )
    def test_value_reader__raster_windows(
        self, source_bands: list[int] | None, band: int, downsample: int, workers: bool, tmp_path: Path
    ) -> None:
        """Checks that Multiproc readers load only the requested raster band and window, and compares with eager."""

        # Write a raster file to disk with three distinct bands, one nodata pixel, and values preserved by downsampling
        base = np.arange(80, dtype=np.int16).reshape(8, 10)
        values = np.ma.array(np.stack((base, base + 100, base + 200)), mask=False)
        values.mask[:, 2, 3] = True
        raster = gu.Raster.from_array(values, from_origin(0, 8, 1, 1), 32633, nodata=-9999)
        filename = tmp_path / "values.tif"
        raster.to_file(filename)
        source = gu.Raster(filename, bands=source_bands, downsample=downsample, load_data=False)
        reference = gu.Raster(filename, bands=source_bands, downsample=downsample, load_data=True)

        # Read one small window either directly or in a worker, without loading the source Raster
        reader = _reader_from_source(source, band, source, MultiprocConfig(chunks=2))
        assert reader is not None and not source.is_loaded
        tile = (slice(1, 3), slice(2, 5))
        block = reader.block(tile)
        assert reader.shape == source.shape
        assert reader.size == np.prod(source.shape)
        assert reader.dtype == values.dtype
        assert not source.is_loaded
        with ClusterGenerator("multi" if workers else "basic", nb_workers=2) as cluster:
            result = cluster.compute(cluster.submit(_read_values, block))

        # Compare the selected values and nodata mask with an independent small eager read
        expected = reference.data[band - 1][tile]
        assert np.array_equal(np.ma.getdata(result), np.ma.getdata(expected))
        assert np.array_equal(np.ma.getmaskarray(result), np.ma.getmaskarray(expected))
        assert result.shape == (2, 3)
        assert not source.is_loaded

    @pytest.mark.parametrize("file_mask", [False, True])
    def test_value_reader__raster_nodata_and_mask(self, file_mask: bool, tmp_path: Path) -> None:
        """Checks that a raster nodata pixel and an array or raster mask are applied to one column."""

        # Write an integer raster file to disk with one nodata pixel
        values = np.ma.array(np.arange(24, dtype=np.int16).reshape(4, 6), mask=False)
        values.mask[1, 2] = True
        transform = from_origin(0, 4, 1, 1)
        filename = tmp_path / "values.tif"
        gu.Raster.from_array(values, transform, 32633, nodata=-9999).to_file(filename)
        source = gu.Raster(filename)
        keep = np.ones(values.shape, dtype=bool)
        keep[2, 2] = False
        mask: NDArray[Any] | _ValueReader = keep
        # Write a boolean raster mask to disk with a different excluded pixel
        if file_mask:
            mask_filename = tmp_path / "mask.tif"
            gu.Raster.from_array(keep, transform, 32633).to_file(mask_filename)
            mask_source = gu.Raster(mask_filename, is_mask=True)
            mask = _ValueReader(mask_source)

        # Apply the array or raster mask while reading one column from three rows
        reader = replace(_ValueReader(source), mask=mask)
        tile = (slice(1, 4), slice(2, 3))
        result = _read_values(reader.block(tile))

        # Nodata and the user mask exclude different pixels, so the remaining integer stays unchanged
        assert result.shape == (3, 1) and result.dtype == values.dtype
        assert np.array_equal(np.ma.getmaskarray(result), [[True], [True], [False]])
        assert result[2, 0] == values[3, 2]
        assert not source.is_loaded
        if file_mask:
            assert not mask_source.is_loaded

    @pytest.mark.parametrize("column", ["height", "weight", None])
    def test_value_reader__geopackage_rows(self, column: str | None, tmp_path: Path) -> None:
        """Checks that row slices select point columns or geometry Z without loading the GeoPackage source."""

        # Write a GeoPackage file to disk with geometry elevations differing from its height/weight columns
        positions = np.arange(6)
        dataframe = gpd.GeoDataFrame(
            {"height": positions.astype(float) + 10, "weight": positions.astype(np.int32) + 100},
            geometry=gpd.points_from_xy(500000 + positions, 5100000 + positions, positions + 1000),
            crs=32633,
        )
        filename = tmp_path / "points.gpkg"
        dataframe.to_file(filename, index=False)
        source = gu.PointCloud(filename)

        # Read the second through fourth rows from height, weight, or geometry Z, using a worker process for weight
        reader = _reader_from_source(source, column, source, MultiprocConfig(chunks=2))
        assert reader is not None and not source.is_loaded
        with ClusterGenerator("multi" if column == "weight" else "basic", nb_workers=2) as cluster:
            result = cluster.compute(cluster.submit(_read_values, reader.block(slice(1, 4))))

        # Read the same rows as point geometries
        point_rows = reader.read_points(slice(1, 4))

        # Check file order, selected values, and that an empty row range returns no data
        expected = dataframe.geometry.z if column is None else dataframe[column]
        assert np.array_equal(result, expected.iloc[1:4])
        assert np.array_equal(point_rows.geometry.x, dataframe.geometry.x.iloc[1:4])
        assert reader.shape == (len(dataframe),)
        assert reader.read(slice(0, 0)).size == 0
        assert reader.read_points(slice(0, 0)).empty
        assert not source.is_loaded

    @pytest.mark.parametrize("column", ["Z", "intensity"])
    def test_value_reader__las_rows(self, column: str, tmp_path: Path) -> None:
        """Checks that LAS row slices return exact elevations and integer attributes while the source is not loaded."""

        laspy = pytest.importorskip("laspy")

        # Write a LAS file to disk with scaled Z elevations and separate integer intensity values
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.scales = np.array([0.01, 0.01, 0.01])
        header.add_crs(CRS.from_epsg(32633))
        records = laspy.LasData(header)
        positions = np.arange(6)
        records.x, records.y, records.z = 500000 + positions, 5100000 + positions, positions / 4 + 10
        records.intensity = positions + 100
        filename = tmp_path / "points.las"
        records.write(filename)
        source = gu.PointCloud(filename)

        # Read the third through fifth rows from either the Z elevations or the intensity attribute
        reader = _ValueReader(source, column)
        result = _read_values(reader.block(slice(2, 5)))
        expected = np.asarray(records.z if column == "Z" else records.intensity)[2:5]
        assert np.array_equal(result, expected)
        assert reader.dtype == result.dtype
        assert reader.read_points(slice(2, 5)).shape[0] == 3
        assert not source.is_loaded

    def test_value_reader__empty_raster_window(self, tmp_path: Path) -> None:
        """Checks that an empty raster window returns the right shape and type without loading its source."""

        # We write a small raster, open the raster object unloaded
        values = np.arange(12, dtype=np.int16).reshape(3, 4)
        filename = tmp_path / "values.tif"
        gu.Raster.from_array(values, from_origin(0, 3, 1, 1), 32633).to_file(filename)
        source = gu.Raster(filename)

        # We read an empty row range
        result = _ValueReader(source).read((slice(1, 1), slice(1, 3)))

        # We check the reader returns an empty array without loading the raster into memory
        assert result.shape == (0, 2)
        assert result.dtype == values.dtype
        assert not source.is_loaded


class TestValueReaderErrors:
    """Test module for errors in _ValueReader."""

    @pytest.mark.parametrize("band", [0, 3, "first"])
    def test_value_reader__error_invalid_raster_band(self, band: int | str, tmp_path: Path) -> None:
        """Checks an error is raised when a raster reader selects a band outside the file."""

        # We define a two band raster, that will accept 1, 2 as band inputs
        values = np.stack([np.ones((2, 3), dtype=np.int16), np.full((2, 3), 2, dtype=np.int16)])
        filename = tmp_path / "bands.tif"
        gu.Raster.from_array(values, from_origin(0, 2, 1, 1), 32633).to_file(filename)
        source = gu.Raster(filename)

        # We should raise an error when using wrong bands integer (0, 3) or a string
        with pytest.raises(ValueError, match="Raster bands must be integers"):
            _ValueReader(source, selector=band)
        assert not source.is_loaded

    def test_value_reader__error_loaded_source(self, tmp_path: Path) -> None:
        """Checks an error is raised because a worker reader needs an unloaded source."""

        # We load a raster
        filename = tmp_path / "values.tif"
        gu.Raster.from_array(np.ones((2, 3)), from_origin(0, 2, 1, 1), 32633).to_file(filename)
        source = gu.Raster(filename, load_data=True)

        # And check we raise an error if not unloaded
        with pytest.raises(ValueError, match="require an unloaded raster"):
            _ValueReader(source)

    @pytest.mark.parametrize(
        "slices, error_type, message",
        [
            (slice(0, 1), TypeError, "one slice per input dimension"),
            ((slice(0, 2, 2), slice(None)), ValueError, "contiguous increasing slices"),
        ],
    )
    def test_value_reader__error_invalid_raster_slices(
        self, slices: slice | tuple[slice, ...], error_type: type[Exception], message: str, tmp_path: Path
    ) -> None:
        """Checks an error is raised for a raster block with missing dimensions or skipped rows."""

        # We create a single band raster (2D, not fit for point reading)
        filename = tmp_path / "values.tif"
        gu.Raster.from_array(np.ones((3, 4)), from_origin(0, 3, 1, 1), 32633).to_file(filename)
        reader = _ValueReader(gu.Raster(filename))

        # Fail on discontinuous slices, or trying to read points
        with pytest.raises(error_type, match=message):
            reader.block(slices)
        with pytest.raises(TypeError, match="Point rows require a point cloud reader"):
            reader.read_points(slice(0, 1))
