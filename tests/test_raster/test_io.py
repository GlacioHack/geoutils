"""Test raster opening, loading, storage encoding and writing."""

from __future__ import annotations

import os
import pathlib
import tempfile
from collections.abc import Callable
from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryFile
from typing import Any, Literal

import numpy as np
import pytest
import rasterio as rio
import xarray as xr
from affine import Affine
from PIL import Image
from rasterio.transform import from_origin

import geoutils as gu
from geoutils import examples, open_raster
from tests.accessor_helpers import assert_output_equal
from tests.remote_helpers import RemoteFiles


class TestRasterOpening:
    """Test module for raster opening, downsampling, overviews and source metadata."""

    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")
    landsat_rgb_path = examples.get_path_test("everest_landsat_rgb")

    @pytest.mark.parametrize("as_object_type", [None, "dataarray", "dataset"])
    @pytest.mark.parametrize("data_name", [None, "elevation"])
    @pytest.mark.parametrize("bands", [1, 2])
    @pytest.mark.parametrize("is_mask", [False, True])
    @pytest.mark.parametrize("squeeze", [False, True])
    def test_open_raster__return_type(
        self,
        tmp_path: Path,
        as_object_type: Literal["dataarray", "dataset"] | None,
        data_name: str | None,
        bands: int,
        is_mask: bool,
        squeeze: bool,
    ) -> None:
        """Checks that opening returns the requested type and name with exact values, bands and georeferencing."""

        # We write 1/2 bands with a NaN to check masking/nodata behaviour
        values = np.arange(bands * 35, dtype=np.float32).reshape(bands, 5, 7)
        values[0, 2, 3] = np.nan
        transform = from_origin(500000, 8600000, 20, 20)
        reference = gu.Raster.from_array(values, transform, 32633, nodata=-9999)
        path = tmp_path / "opening.tif"
        reference.to_file(path)

        # Check we open with the right type
        if as_object_type is None:
            result = open_raster(str(path), data_name=data_name, is_mask=is_mask, squeeze=squeeze)
        else:
            result = open_raster(
                str(path), as_object_type=as_object_type, data_name=data_name, is_mask=is_mask, squeeze=squeeze
            )
        if as_object_type == "dataset":
            assert isinstance(result, xr.Dataset)
            name = data_name or "raster"
            assert list(result.rst.compatible_variables.keys()) == [name]
            variable = result[name]
        else:
            assert isinstance(result, xr.DataArray)
            variable = result
            if data_name is not None:
                assert variable.name == data_name
        assert variable._in_memory is is_mask

        # Then we check proper metadata attribute, and equal data
        expected = values[0] if bands == 1 and squeeze else values
        if is_mask:
            expected = expected.astype(bool)
        else:
            assert variable.rio.encoded_nodata == reference.nodata
        assert variable.dims == (("y", "x") if bands == 1 and squeeze else ("band", "y", "x"))
        assert variable.rio.crs == reference.crs
        assert variable.rio.transform() == transform
        np.testing.assert_array_equal(variable.data, expected)

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset"])
    @pytest.mark.parametrize("downsample", [1, 2])
    @pytest.mark.parametrize("mask_and_scale", [False, True])
    def test_open_raster__unmasked(
        self, tmp_path: Path, as_object_type: Literal["dataarray", "dataset"], downsample: int, mask_and_scale: bool
    ) -> None:
        """Checks unmasked opening works, but raises a warning."""

        # Create integer values to open
        values = np.arange(48, dtype=np.int16).reshape(6, 8)
        values[1, 1] = -9999
        filename = tmp_path / "stored_values.tif"
        array = np.ma.masked_equal(values, -9999)
        gu.Raster.from_array(array, from_origin(0, 6, 1, 1), 32631, nodata=-9999).to_file(filename)
        with rio.open(filename, "r+") as source:
            source.scales, source.offsets = (2,), (10,)

        # Check warning is raised opening with masked=Fasle
        with pytest.warns(UserWarning, match="masked=False"):
            result = open_raster(
                str(filename),
                masked=False,
                downsample=downsample,
                mask_and_scale=mask_and_scale,
                as_object_type=as_object_type,
                squeeze=False,
            )

        # Raster should be integer, with nodata value not replaced by NaNs
        variable = result["raster"] if as_object_type == "dataset" else result
        assert not variable.rst.is_loaded
        selected = values if downsample == 1 else values[1::2, 1::2]
        expected = selected.astype(float) * 2 + 10 if mask_and_scale else selected
        if mask_and_scale:
            expected[selected == -9999] = np.nan
        assert variable.dtype == (np.float32 if mask_and_scale else np.int16)
        np.testing.assert_array_equal(variable.values[0], expected)
        assert variable.rst.nodata == -9999

        # Writing back to file should respect the nodata
        output = tmp_path / "copied.tif"
        result.rst.to_file(str(output), tiled=True, blockxsize=16, blockysize=16)
        with rio.open(output) as saved:
            assert saved.dtypes == ("int16",)
            assert saved.nodata == -9999
            np.testing.assert_array_equal(saved.read(1), selected)
        result.close()

    @pytest.mark.parametrize("shape", [(1, 3), (3, 1), (1, 1)])
    @pytest.mark.parametrize("bands", [1, 2])
    def test_open_raster__single_row_or_column(self, tmp_path: Path, shape: tuple[int, int], bands: int) -> None:
        """Checks that opening a raster returns spatial dimensions of length one and every requested band."""

        # A single row or column is still a two-dimensional grid, including for several bands
        values = np.arange(bands * np.prod(shape), dtype=np.float32).reshape((bands, *shape))
        path = tmp_path / "narrow.tif"
        transform = from_origin(0, 3, 1, 1)
        gu.Raster.from_array(values, transform, 32631).to_file(path)

        # Opening removes only a single band dimension, while retaining the grid coordinates and values
        result = open_raster(str(path))
        expected = values[0] if bands == 1 else values
        assert result.dims == (("y", "x") if bands == 1 else ("band", "y", "x"))
        assert result.rst.shape == shape
        assert result.rst.transform == transform
        np.testing.assert_array_equal(result.data, expected)

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset"])
    def test_open_raster__sensor_metadata(self, as_object_type: Literal["dataarray", "dataset"]) -> None:
        """Checks that opening parses the same sensor metadata as Raster without loading pixels."""

        # Open the Landsat file through both interfaces with sensor parsing enabled
        expected = gu.Raster(self.landsat_b4_path, parse_sensor_metadata=True, silent=True)
        result = open_raster(self.landsat_b4_path, parse_sensor_metadata=True, as_object_type=as_object_type)
        variable = result["raster"] if as_object_type == "dataset" else result

        # Compare parsed fields and check that reading metadata left the raster unloaded
        assert "sensor" in expected.tags
        for key in ("sensor", "platform", "datetime", "product", "tile_name"):
            assert variable.attrs[key] == expected.tags[key]
            assert result.attrs[key] == expected.tags[key]
        assert not variable.rst.is_loaded and not expected.is_loaded

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset"])
    @pytest.mark.parametrize("downsample", [1, 2])
    def test_open_raster__disk_chunk_metadata(
        self, tmp_path: Path, as_object_type: Literal["dataarray", "dataset"], downsample: int
    ) -> None:
        """Checks that cached disk block shapes describe the source file and refresh after writing new tiles."""

        # Write rectangular tiles so swapped row/column dimensions produce a different result
        values = np.arange(2 * 64 * 64, dtype=np.float32).reshape(2, 64, 64)
        filename = tmp_path / "source_tiles.tif"
        gu.Raster.from_array(values, from_origin(0, 64, 1, 1), 32631).to_file(
            filename, co_opts={"TILED": "YES", "BLOCKYSIZE": "16", "BLOCKXSIZE": "32"}
        )

        # Open metadata from the original file, including when returning a reduced grid
        result = open_raster(str(filename), as_object_type=as_object_type, downsample=downsample)
        variable = result["raster"] if as_object_type == "dataset" else result
        raster = gu.Raster(filename, downsample=downsample)
        for attrs in (result.attrs, variable.attrs, variable.rst.tags, raster.tags):
            assert attrs["block_shape"] == (16, 32)
        assert not variable.rst.is_loaded and not raster.is_loaded

        # Convert through a memory file without replacing the cached source tile shapes
        converted = raster.to_xarray()
        assert converted.attrs["block_shape"] == (16, 32)
        converted.close()

        # Write different tiles and check reopening replaces the source's cached shapes
        output = tmp_path / "new_tiles.tif"
        variable.rst.to_file(str(output), tiled=True, blockysize=32, blockxsize=16)
        reopened = open_raster(str(output))
        assert reopened.attrs["block_shape"] == (32, 16)
        assert gu.Raster(output).tags["block_shape"] == (32, 16)
        reopened.close()
        result.close()

    @pytest.mark.parametrize("storage", ["netcdf", "zarr"])
    def test_tags__disk_chunks(self, tmp_path: Path, storage: str) -> None:
        """Checks that array storage chunks are cached in tags and refreshed after writing a new layout."""

        # Create a georeferenced variable with metadata from an older disk layout
        values = np.arange(32 * 64, dtype=np.float32).reshape(32, 64)
        array = gu.Raster.from_array(values, from_origin(0, 32, 1, 1), 32631).to_xarray()
        array = array.squeeze("band", drop=True).rename("height")
        array.attrs["disk_chunks"] = (8, 8)
        dataset = array.to_dataset()
        encoding_key = "chunks" if storage == "zarr" else "chunksizes"
        encoding = {"height": {encoding_key: (16, 32), "grid_mapping": array.rio.grid_mapping}}
        filename = tmp_path / storage

        # Write known disk chunks and reopen without Dask or pixel loading
        if storage == "zarr":
            pytest.importorskip("zarr")
            dataset.to_zarr(filename, zarr_format=2, encoding=encoding)
            opened = xr.open_zarr(filename, chunks=None, decode_coords="all")
        else:
            pytest.importorskip("netCDF4")
            dataset.to_netcdf(filename, engine="netcdf4", encoding=encoding)
            opened = xr.open_dataset(filename, engine="netcdf4", decode_coords="all")
        variable = opened["height"]
        assert variable.rst.tags["disk_chunks"] == (16, 32)
        assert not variable.rst.is_loaded

        # Rewrite with rectangular chunks in the opposite orientation, including the old cached tags
        output = tmp_path / f"new_{storage}"
        encoding["height"][encoding_key] = (32, 16)
        if storage == "zarr":
            opened.to_zarr(output, zarr_format=2, encoding=encoding)
            rewritten = xr.open_zarr(output, chunks=None, decode_coords="all")
        else:
            opened.to_netcdf(output, engine="netcdf4", encoding=encoding)
            rewritten = xr.open_dataset(output, engine="netcdf4", decode_coords="all")

        # Check actual saved chunks replace the old tags and all pixel values remain unchanged
        assert rewritten["height"].rst.tags["disk_chunks"] == (32, 16)
        np.testing.assert_array_equal(rewritten["height"].values, values)
        rewritten.close()
        opened.close()
        array.close()

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset"])
    @pytest.mark.parametrize("factors", [[], [2, 4]])
    @pytest.mark.parametrize("resampling", [None, "average"])
    def test_open_raster__overview_metadata(
        self,
        tmp_path: Path,
        as_object_type: Literal["dataarray", "dataset"],
        factors: list[int],
        resampling: str | None,
    ) -> None:
        """Checks that file opening caches first-band overview factors and known resampling without loading pixels."""

        # Write two bands so cached overview factors also describe a multi-band raster
        values = np.arange(512, dtype=np.float32).reshape(2, 16, 16)
        path = tmp_path / "overview_metadata.tif"
        gu.Raster.from_array(values, from_origin(0, 16, 1, 1), 32631).to_file(path)
        with rio.open(path, "r+") as source:
            if factors:
                source.build_overviews(factors, rio.enums.Resampling.average)
            if resampling is not None:
                source.update_tags(ns="rio_overview", resampling=resampling)

        # Open both GeoUtils representations from the same source metadata
        result = open_raster(str(path), as_object_type=as_object_type)
        variable = result["raster"] if as_object_type == "dataset" else result
        raster = gu.Raster(path)

        # Check empty overview lists and unknown methods without inventing a resampling method
        for attrs in (result.attrs, variable.attrs, variable.rst.tags, raster.tags):
            assert attrs["overview_factors"] == factors
            assert attrs["overview_selected_level"] == -1
            assert attrs["overview_selected_factor"] == 1
            if resampling is None:
                assert "overview_resampling" not in attrs
            else:
                assert attrs["overview_resampling"] == resampling
        assert not variable.rst.is_loaded and not raster.is_loaded

        # Convert the Raster through its in-memory file without turning cached factors into strings
        converted = raster.to_xarray()
        assert converted.attrs["overview_factors"] == factors
        if resampling is not None:
            assert converted.attrs["overview_resampling"] == resampling
        converted.close()

        # Write tiles without overviews, then check reopening refreshes the cached source metadata
        output_path = tmp_path / "exported.tif"
        variable.rst.to_file(str(output_path), tiled=True, blockxsize=16, blockysize=16)
        reopened = open_raster(str(output_path))
        for attrs in (reopened.attrs, gu.Raster(output_path).tags):
            assert attrs["overview_factors"] == []
            assert attrs["overview_selected_level"] == -1
            assert attrs["overview_selected_factor"] == 1
            assert "overview_resampling" not in attrs
        reopened.close()
        result.close()

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset"])
    @pytest.mark.parametrize(
        "options, selected_level, selected_factor",
        [
            ({}, -1, 1),
            ({"overview_level": -1}, -1, 1),
            ({"overview_level": 0}, 0, 2),
            ({"overview_level": 1}, 1, 4),
            ({"downsample": 1.3}, -1, 1),
            ({"downsample": 3}, 0, 2),
            ({"downsample": 6}, 1, 4),
        ],
    )
    def test_open_raster__overview_selection_metadata(
        self,
        tmp_path: Path,
        as_object_type: Literal["dataarray", "dataset"],
        options: dict[str, Any],
        selected_level: int,
        selected_factor: int,
    ) -> None:
        """Checks that cached overview selection identifies the source grid used before any output resampling."""

        # Write factor-2/4 overviews so automatic downsampling can select either level or the native grid
        path = tmp_path / "selected_overview.tif"
        values = np.arange(256, dtype=np.float32).reshape(16, 16)
        gu.Raster.from_array(values, from_origin(0, 16, 1, 1), 32631).to_file(path)
        with rio.open(path, "r+") as source:
            source.build_overviews([2, 4], rio.enums.Resampling.average)

        # Open the requested grid without loading its pixels
        result = open_raster(str(path), as_object_type=as_object_type, **options)
        variable = result["raster"] if as_object_type == "dataset" else result
        assert not variable.rst.is_loaded

        # Check source selection independently of the requested output spacing
        for attrs in (result.attrs, variable.attrs):
            assert attrs["overview_selected_level"] == selected_level
            assert attrs["overview_selected_factor"] == selected_factor
        raster = gu.Raster(path, **options)
        assert not raster.is_loaded
        assert raster.tags["overview_selected_level"] == selected_level
        assert raster.tags["overview_selected_factor"] == selected_factor
        assert raster.tags["block_shape"] == variable.attrs["block_shape"]
        converted = raster.to_xarray()
        assert converted.attrs["overview_selected_level"] == selected_level
        assert converted.attrs["overview_selected_factor"] == selected_factor
        converted.close()
        result.close()

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset"])
    def test_open_raster__downsample(self, tmp_path: Path, as_object_type: Literal["dataarray", "dataset"]) -> None:
        """Checks that downsampling uses a suitable overview while keeping the requested output grid."""

        # Write input for full array, then open with downsampling (without overview)
        values = np.arange(5_000, dtype=np.int32).reshape(50, 100)
        path = tmp_path / "gradient.tif"
        gu.Raster.from_array(values, from_origin(0, 50, 1, 1), 4326).to_file(path)
        without_overview = open_raster(str(path), downsample=6, as_object_type=as_object_type)
        without_overview = without_overview["raster"] if as_object_type == "dataset" else without_overview
        without_overview_values = without_overview.values.copy()

        # Add overview for factors of 2 and 4, factor 4 should be the closest suitable overview
        with rio.open(path, "r+") as dataset:
            dataset.build_overviews([2, 4], rio.enums.Resampling.nearest)

        # Check that downsampling the new file uses the overview (slightly changes the sampled values)
        # without changing the size/transform of the requested grid
        result = open_raster(str(path), downsample=6, as_object_type=as_object_type)
        result = result["raster"] if as_object_type == "dataset" else result
        assert not result._in_memory
        assert result.attrs["overview_factors"] == [2, 4]
        output = result.values
        expected = gu.Raster(path, downsample=6).data.data

        assert result.rst.shape == (8, 16)
        assert result.rst.transform == from_origin(0, 50, 6, 6)
        np.testing.assert_array_equal(output, expected)
        assert not np.array_equal(output, without_overview_values)

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset"])
    @pytest.mark.parametrize("overviews", [False, True])
    @pytest.mark.parametrize(
        "downsample,shape,factors",
        [
            (1, (50, 100), (1, 1)),
            (6, (8, 16), (6, 6)),
            ({"factor": 6}, (8, 16), (6, 6)),
            ({"res": (12, 18)}, (8, 16), (6, 6)),
            ({"res": 12}, (12, 16), (6, 4)),
            ({"shape": (8, 16)}, (8, 16), (100 / 16, 50 / 8)),
            ({"shape": (12, 16)}, (12, 16), (100 / 16, 50 / 12)),
            ({"shape": (7, 13)}, (7, 13), (100 / 13, 50 / 7)),
            (6.5, (7, 15), (6.5, 6.5)),
        ],
    )
    def test_open_raster__downsample_grid(
        self,
        tmp_path: Path,
        as_object_type: Literal["dataarray", "dataset"],
        overviews: bool,
        downsample: Any,
        shape: tuple[int, int],
        factors: tuple[float, float],
    ) -> None:
        """Checks that factors, resolutions and shapes use nearest reads on the requested grid."""

        values = np.arange(10_000, dtype=np.int16).reshape(2, 50, 100)
        filename = tmp_path / "grid.tif"
        transform = from_origin(500000, 1000000, 2, 3)
        gu.Raster.from_array(values, transform, 32631).to_file(filename)
        with rio.open(filename, "r+") as source:
            if overviews:
                source.build_overviews([2, 4], rio.enums.Resampling.nearest)
            window = rio.windows.Window(0, 0, shape[1] * factors[0], shape[0] * factors[1])
            expected = source.read(out_shape=(2, *shape), window=window)

        result = open_raster(str(filename), downsample=downsample, as_object_type=as_object_type)
        variable = result["raster"] if as_object_type == "dataset" else result
        raster = gu.Raster(filename, downsample=downsample)
        assert not variable.rst.is_loaded and not raster.is_loaded
        assert variable.rst.shape == raster.shape == shape
        assert variable.rst.transform == raster.transform == transform * Affine.scale(*factors)
        np.testing.assert_array_equal(variable.values, expected)
        np.testing.assert_array_equal(raster.data, expected)

        # Deferred crops and mask reads use the same reduced grid
        deferred = gu.Raster(filename, downsample=downsample).icrop((1, 1, 5, 4))
        assert not deferred.is_loaded
        np.testing.assert_array_equal(deferred._load_only_mask(), np.zeros((2, 3, 4), dtype=bool))
        np.testing.assert_array_equal(deferred.data, expected[:, 1:4, 1:5])
        loaded = gu.Raster(filename, downsample=downsample, load_data=True)
        np.testing.assert_array_equal(loaded.data, expected)
        result.close()

    @pytest.mark.parametrize("overviews", [False, True])
    @pytest.mark.parametrize("masked", [False, True])
    def test_open_raster__downsample_source_metadata(self, tmp_path: Path, overviews: bool, masked: bool) -> None:
        """Checks that downsampling preserves source tags, band metadata, masks and storage encoding."""

        filename = tmp_path / "metadata.tif"
        values = np.arange(48, dtype=np.int16).reshape(6, 8)
        with rio.open(
            filename,
            "w",
            driver="GTiff",
            height=6,
            width=8,
            count=1,
            dtype="int16",
            crs=32631,
            transform=from_origin(0, 6, 1, 1),
        ) as source:
            source.write(values, 1)
            source.update_tags(AREA_OR_POINT="Point", OVR_RESAMPLING_ALG="NEAREST", DataType="Elevation")
            source.update_tags(1, survey="valley")
            source.set_band_description(1, "Elevation")
            source.set_band_unit(1, "m")
            source.scales, source.offsets = (0.5,), (100,)
            # A mask without numeric nodata checks explicit GDAL mask bands
            mask = np.full(values.shape, 255, dtype=np.uint8)
            mask[1:3, 1:3] = 0
            source.write_mask(mask)
            if overviews:
                source.build_overviews([2], rio.enums.Resampling.nearest)

        with pytest.warns(UserWarning, match="masked=False") if not masked else nullcontext():
            original = open_raster(str(filename), masked=masked)
            reduced = open_raster(str(filename), masked=masked, downsample=2)
        for key, value in original.attrs.items():
            if key not in ("overview_selected_level", "overview_selected_factor"):
                assert reduced.attrs[key] == value
        assert reduced.encoding["source"] == original.encoding["source"]
        assert reduced.rst.driver == original.rst.driver
        with rio.open(filename) as source:
            expected = source.read(1, out_shape=(3, 4), masked=masked)
        if masked:
            expected = expected.astype(np.float32).filled(np.nan)
        np.testing.assert_array_equal(reduced.values, expected)
        reduced.close()
        original.close()

    @pytest.mark.parametrize("downsample,shape", [(4, (3, 5)), ({"shape": (1, 5)}, (1, 5))])
    @pytest.mark.parametrize("as_reader", [False, True])
    def test_open_raster__downsample_rotated(
        self, tmp_path: Path, downsample: Any, shape: tuple[int, int], as_reader: bool
    ) -> None:
        """Checks that reduced rotated grids preserve their affine transform and spatial dimensions."""

        values = np.arange(240, dtype=np.int16).reshape(12, 20)
        filename = tmp_path / "rotated.tif"
        transform = Affine(2, 0.5, 500000, 0.25, -2, 1000000)
        gu.Raster.from_array(values, transform, 32631).to_file(filename)

        with rio.open(filename) as reader:
            result = open_raster(reader if as_reader else str(filename), downsample=downsample)
            assert not reader.closed and not result.rst.is_loaded
            expected = reader.read(1, out_shape=shape)
        assert result.dims == ("y", "x")
        assert result.rst.shape == shape
        assert result.rst.transform == transform * Affine.scale(20 / shape[1], 12 / shape[0])
        np.testing.assert_array_equal(result.values, expected)
        result.close()

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset"])
    def test_open_raster__overview_level(self, tmp_path: Path, as_object_type: Literal["dataarray", "dataset"]) -> None:
        """Checks that overview selection is distinct from downsampling."""

        # Store two overview levels (the level 0 represents the factor-2 grid)
        values = np.arange(16 * 16, dtype=np.int16).reshape(16, 16)
        path = tmp_path / "overviews.tif"
        gu.Raster.from_array(values, from_origin(0, 16, 1, 1), 4326).to_file(path)
        with rio.open(path, "r+") as dataset:
            dataset.build_overviews([2, 4], rio.enums.Resampling.nearest)

        # Check error is raised when combining exact overview selection with downsampling
        overview = open_raster(str(path), overview_level=0, as_object_type=as_object_type)
        overview = overview["raster"] if as_object_type == "dataset" else overview
        assert overview.attrs["overview_factors"] == [2, 4]
        assert overview.rst.shape == (8, 8)
        assert overview.rst.transform == from_origin(0, 16, 2, 2)
        with pytest.raises(ValueError, match="downsample and overview_level cannot be used together"):
            open_raster(str(path), downsample=3, overview_level=0, as_object_type=as_object_type)

    @pytest.mark.parametrize("source_type", ["reader", "memory_file"])
    def test_open_raster__rasterio_objects(self, raster_gcp_rpc: gu.Raster, source_type: str) -> None:
        """Checks that Rasterio readers and memory files open lazily with their GCP/RPC metadata and values."""

        # Open the same GCP/RPC image from its filename and a Rasterio object
        path = Path(raster_gcp_rpc.name)
        expected = open_raster(str(path))
        source_context = rio.open(path) if source_type == "reader" else rio.MemoryFile(path.read_bytes())
        with source_context as source:
            # Read metadata without loading pixels or closing the supplied Rasterio object
            result = open_raster(source)  # type: ignore[arg-type]
            assert not source.closed
            assert not result.rst.is_loaded
            assert not expected.rst.is_loaded

            # Compare coordinates, GCPs/RPCs and values with the file opened directly
            xr.testing.assert_identical(result, expected)
            assert not source.closed
            result.close()
        expected.close()

    @pytest.mark.parametrize("path_raster", [landsat_b4_path, aster_dem_path])
    def test_open__loaded(self, path_raster: str) -> None:
        """
        Test that a DataArray opened using "open_raster" maintains implicit loading logic.

        Tests checking loading for all attributes and methods are done in TestBase.

        Note: this is different from using lazy Dask arrays: for any array type, Xarray only loads metadata, and
        implicitly loads data in memory when .data or .load() is called.
        """

        # Open raster with/without chunks, should not load in memory yet
        ds = open_raster(path_raster)
        assert not ds._in_memory

        # The array should be NumPy
        assert isinstance(ds.data, np.ndarray)
        ds.load()
        assert ds._in_memory

    @pytest.mark.parametrize("example", [aster_dem_path, landsat_b4_path, landsat_rgb_path])
    def test_downsampling(self, example: str) -> None:
        """
        Check that self metadata are correctly updated when using downsampling
        """
        # Load raster at full resolution
        rst_orig = gu.Raster(example)
        bounds_orig = rst_orig.bounds

        # -- Tries various downsampling factors to ensure rounding is needed in at least one case --
        for down_fact in np.arange(2, 8):
            rst_down = gu.Raster(example, downsample=int(down_fact))

            # - Check that output resolution is as intended -
            assert rst_down.res[0] == rst_orig.res[0] * down_fact
            assert rst_down.res[1] == rst_orig.res[1] * down_fact

            # - Check that downsampled width and height are as intended -
            # Due to rounding, width/height can be up to 1 pixel larger than unrounded value
            assert abs(rst_down.width - rst_orig.width / down_fact) < 1
            assert abs(rst_down.height - rst_orig.height / down_fact) < 1
            assert rst_down.shape == (rst_down.height, rst_down.width)

            # - Check that bounds are updated accordingly -
            # left/top bounds should be the same, right/bottom should be rounded to nearest pixel
            bounds_down = rst_down.bounds
            assert bounds_down.left == bounds_orig.left
            assert bounds_down.top == bounds_orig.top
            assert abs(bounds_down.right - bounds_orig.right) < rst_down.res[0]
            assert abs(bounds_down.bottom - bounds_orig.bottom) < rst_down.res[1]

            # - Check that metadata are consistent, with/out loading data -
            assert not rst_down.is_loaded
            width_unload = rst_down.width
            height_unload = rst_down.height
            bounds_unload = rst_down.bounds
            rst_down.load()
            width_load = rst_down.width
            height_load = rst_down.height
            bounds_load = rst_down.bounds
            assert width_load == width_unload
            assert height_load == height_unload
            assert bounds_load == bounds_unload

            # - Test that xy2ij are consistent with new image -
            # Upper left
            assert rst_down.xy2ij(rst_down.bounds.left, rst_down.bounds.top) == (-0.5, -0.5)
            # Upper right
            assert rst_down.xy2ij(rst_down.bounds.right + rst_down.res[0], rst_down.bounds.top) == (
                -0.5,
                rst_down.width + 0.5,
            )
            # Bottom right
            assert rst_down.xy2ij(rst_down.bounds.right + rst_down.res[0], rst_down.bounds.bottom) == (
                rst_down.height - 0.5,
                rst_down.width + 0.5,
            )
            # One pixel right and down
            assert rst_down.xy2ij(rst_down.bounds.left + rst_down.res[0], rst_down.bounds.top - rst_down.res[1]) == (
                0.5,
                0.5,
            )

        # -- Check that error is raised when downsampling value is not valid --
        with pytest.raises(TypeError, match="downsample must be of type int or float."):
            gu.Raster(example, downsample=[1, 1])  # type: ignore
        with pytest.raises(ValueError, match="downsample must be >=1."):
            gu.Raster(example, downsample=0)  # type: ignore

    @pytest.mark.parametrize("load_data", [False, True])
    def test_init__downsample_regular_interval(self, tmp_path: pathlib.Path, load_data: bool) -> None:
        """Checks that file opening downsampling keeps a constant interval."""

        # Write input on dimensions not divisible by 6
        values = np.arange(5_000, dtype=np.int32).reshape(50, 100)
        source = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 50, 1, 1), crs=4326)
        filename = tmp_path / "gradient.tif"
        source.to_file(filename)

        # Open with downsampling of 6
        downsampled = gu.Raster(filename, downsample=6, load_data=load_data)
        result = downsampled.data.data

        # Check size is as expected, and values match the constant intervals
        assert downsampled.shape == (8, 16)
        assert np.all(np.diff(result, axis=0) == 600)
        assert np.all(np.diff(result, axis=1) == 6)

    @pytest.mark.parametrize("load_data", [False, True])
    def test_init__downsample_uses_overview(self, tmp_path: pathlib.Path, load_data: bool) -> None:
        """Checks that downsampling reads an overview when it can."""

        # Write input for full array, then open with downsampling (without overview)
        values = np.arange(5_000, dtype=np.int32).reshape(50, 100)
        filename = tmp_path / "gradient.tif"
        gu.Raster.from_array(values, rio.transform.from_origin(0, 50, 1, 1), 4326).to_file(filename)
        without_overview = gu.Raster(filename, downsample=6, load_data=load_data).data.data.copy()

        # Add overview for factors of 2 and 4, factor 4 should be the closest suitable overview
        with rio.open(filename, "r+") as dataset:
            dataset.build_overviews([2, 4], rio.enums.Resampling.nearest)

        # Check that downsampling the new file uses the overview (slightly changes the sampled values)
        # without changing the size/transform of the requested grid
        downsampled = gu.Raster(filename, downsample=6, load_data=load_data)
        result = downsampled.data.data
        assert downsampled.shape == (8, 16)
        assert downsampled.transform == rio.transform.from_origin(0, 50, 6, 6)
        assert not np.array_equal(result, without_overview)

    @pytest.mark.parametrize("overview_level", [-1, 0, 1])
    @pytest.mark.parametrize("load_data", [False, True])
    @pytest.mark.parametrize("source_type", ["path", "reader", "memory_file"])
    @pytest.mark.parametrize("bands", [None, 1])
    def test_init__overview_level(
        self,
        tmp_path: Path,
        overview_level: int,
        load_data: bool,
        source_type: str,
        bands: int | None,
    ) -> None:
        """Checks overview reading with Raster opening."""

        # We build some overviews to open later on
        values = np.arange(2 * 18 * 26, dtype=np.int16).reshape(2, 18, 26)
        values[:, :4, :4] = -9999
        array = np.ma.masked_equal(values, -9999)
        path = tmp_path / "overviews.tif"
        gu.Raster.from_array(array, rio.transform.from_origin(0, 18, 1, 1), 32631, nodata=-9999).to_file(path)
        with rio.open(path, "r+") as dataset:
            dataset.build_overviews([2, 4], rio.enums.Resampling.average)
            dataset.update_tags(ns="rio_overview", resampling="average")
        with rio.open(path, overview_level=overview_level) as dataset:
            expected = dataset.read(indexes=bands, masked=True)
            expected_transform = dataset.transform

        # Open with each supported input
        context: Any
        if source_type == "reader":
            context = rio.open(path)
        elif source_type == "memory_file":
            context = rio.MemoryFile(path.read_bytes())
        else:
            context = nullcontext(path)
        with context as source:
            raster = gu.Raster(source, bands=bands, overview_level=overview_level, load_data=load_data)
            assert raster.is_loaded == (load_data or source_type == "memory_file")
            if source_type != "path":
                assert not source.closed
            assert raster.shape == expected.shape[-2:]
            assert raster.transform == expected_transform
            assert raster.crs == rio.crs.CRS.from_epsg(32631)
            assert raster.dtype == values.dtype

            # Check equality and deferred I/O is maintained
            if not raster.is_loaded:
                np.testing.assert_array_equal(raster._load_only_mask(), np.ma.getmaskarray(expected))
                copied = raster.copy()
                cropped = raster.icrop((1, 1, 5, 4))
                assert not raster.is_loaded and not copied.is_loaded and not cropped.is_loaded
                np.testing.assert_array_equal(cropped._load_only_mask(), np.ma.getmaskarray(expected[..., 1:4, 1:5]))
                np.testing.assert_array_equal(copied.data.filled(-9999), expected.filled(-9999))
                np.testing.assert_array_equal(cropped.data.filled(-9999), expected[..., 1:4, 1:5].filled(-9999))

                # Check that crop() uses the overview
                bbox = rio.windows.bounds(rio.windows.Window(1, 1, 4, 3), expected_transform)
                georeferenced_crop = raster.crop(bbox)
                np.testing.assert_array_equal(
                    georeferenced_crop.data.filled(-9999), expected[..., 1:4, 1:5].filled(-9999)
                )
                assert not raster.is_loaded

            # Check values and source overview metadata
            np.testing.assert_array_equal(raster.data.filled(-9999), expected.filled(-9999))
            np.testing.assert_array_equal(np.ma.getmaskarray(raster.data), np.ma.getmaskarray(expected))
            assert raster.tags["overview_factors"] == [2, 4]
            assert raster.tags["overview_selected_level"] == overview_level
            assert raster.tags["overview_selected_factor"] == (1 if overview_level == -1 else 2 ** (overview_level + 1))
            assert raster.tags["overview_resampling"] == "average"

    @pytest.mark.parametrize("filesystem_access", [False, True])
    @pytest.mark.parametrize(
        "options",
        [
            {},
            {"squeeze": False},
            {"is_mask": True},
            {"as_object_type": "dataset"},
            {"data_name": "height"},
            {"downsample": 2},
        ],
    )
    def test_open_raster__close(
        self, filesystem_access: bool, options: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks that closing returned raster arrays and Datasets closes their source files."""
        path = tmp_path / "source.tif"
        gu.Raster.from_array(np.ones((4, 6), dtype=np.float32), from_origin(0, 4, 1, 1), 4326).to_file(path)
        access: dict[str, Any] = {}
        if filesystem_access:
            access["filesystem"] = pytest.importorskip("fsspec").filesystem("dir", path=str(tmp_path))
        datasets = []
        open_dataset = rio.open

        def record_open(*args: Any, **kwargs: Any) -> Any:
            """Record the opened datasets so their lifetime can be checked."""
            dataset = open_dataset(*args, **kwargs)
            datasets.append(dataset)
            return dataset

        monkeypatch.setattr(rio, "open", record_open)
        source = path.name if filesystem_access else str(path)
        with gu.open_raster(source, **access, **options) as result:
            assert any(not dataset.closed for dataset in datasets)
            result.load()

        assert datasets and all(dataset.closed for dataset in datasets)

    @pytest.mark.parametrize("as_dataset", [False, True])
    @pytest.mark.parametrize("is_mask", [False, True])
    def test_open_raster__filesystem_metadata(self, tmp_path: Path, as_dataset: bool, is_mask: bool) -> None:
        """Checks that filesystem source metadata survives mask conversion, copying and Dataset opening."""
        fsspec = pytest.importorskip("fsspec")
        filesystem = fsspec.filesystem("dir", path=str(tmp_path), fs=fsspec.filesystem("file"))
        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        path = tmp_path / "source.tif"
        gu.Raster.from_array(values, from_origin(0, 3, 1, 1), 4326).to_file(path, co_opts={"BIGTIFF": "YES"})

        result = gu.open_raster(
            path.name,
            filesystem=filesystem,
            is_mask=is_mask,
            as_object_type="dataset" if as_dataset else "dataarray",
        )
        variable = result["raster"] if as_dataset else result
        copied = variable.copy()

        assert variable.encoding["source"] == path.name
        assert variable.encoding["_filesystem"] is filesystem
        if as_dataset:
            assert result.encoding["_filesystem"] is filesystem
        assert copied.rst.driver == "GTiff"
        assert copied.rst._is_bigtiff()
        np.testing.assert_array_equal(copied.values, values.astype(bool) if is_mask else values)
        result.close()

    def test_open_raster__filesystem_external_mask(self, tmp_path: Path) -> None:
        """Checks that a filesystem opener reads a TIFF mask stored in a companion file."""
        filesystem = pytest.importorskip("fsspec").filesystem("dir", path=str(tmp_path))
        path = tmp_path / "masked.tif"
        values = np.ones((3, 4), dtype=np.uint8)
        mask = np.full(values.shape, 255, dtype=np.uint8)
        mask[1, 2] = 0
        with rio.Env(GDAL_TIFF_INTERNAL_MASK=False):
            with rio.open(
                path,
                "w",
                driver="GTiff",
                width=4,
                height=3,
                count=1,
                dtype="uint8",
                crs=4326,
                transform=from_origin(0, 3, 1, 1),
            ) as output:
                output.write(values, 1)
                output.write_mask(mask)
        assert path.with_suffix(".tif.msk").exists()

        with gu.open_raster(path.name, filesystem=filesystem) as result:
            expected = values.astype(np.float32)
            expected[1, 2] = np.nan
            np.testing.assert_array_equal(result.values, expected)


class TestRasterLoading:
    """Test module for reading raster pixels, selected bands and missing-value masks."""

    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    landsat_rgb_path = examples.get_path_test("everest_landsat_rgb")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")

    def test_load(self) -> None:
        """
        Test that loading metadata and data works for all possible cases.
        """
        # Test 0 - test that loading explicitly via load() or implicitly via .data is similar
        r_explicit = gu.Raster(self.landsat_b4_path, load_data=True)
        r_implicit = gu.Raster(self.landsat_b4_path)

        assert r_explicit.raster_equal(r_implicit)

        # Same for a multi-band raster
        r_explicit_rgb = gu.Raster(self.landsat_rgb_path, load_data=True)
        r_implicit_rgb = gu.Raster(self.landsat_rgb_path)

        assert r_explicit_rgb.raster_equal(r_implicit_rgb)

        # Test 1 - loading metadata only, single band
        # For the first example with Landsat B4
        r = gu.Raster(self.landsat_b4_path)

        assert not r.is_loaded
        assert r.driver == "GTiff"
        assert r.width == 193
        assert r.height == 167
        assert r.shape == (r.height, r.width)
        assert r.count == 1
        assert r._count_on_disk == 1
        assert r.bands == (1,)
        assert r._bands_on_disk == (1,)
        assert r.dtype == "uint8"
        assert r.transform == rio.transform.Affine(30.0, 0.0, 489340.0, 0.0, -30.0, 3098570.0)
        assert np.array_equal(r.res, [30.0, 30.0])
        assert r.bounds == rio.coords.BoundingBox(left=489340.0, bottom=3093560.0, right=495130.0, top=3098570.0)
        assert r.crs == rio.crs.CRS.from_epsg(32645)
        assert r.footprint.vector_equal(r.get_footprint_projected(r.crs))

        # And the second example with ASTER DEM
        r2 = gu.Raster(self.aster_dem_path)

        assert not r2.is_loaded
        assert r2.driver == "GTiff"
        assert r2.width == 165
        assert r2.height == 124
        assert r2.shape == (r2.height, r2.width)
        assert r2.count == 1
        assert r._count_on_disk == 1
        assert r.bands == (1,)
        assert r._bands_on_disk == (1,)
        assert r2.dtype == "float32"
        assert r2.transform == rio.transform.Affine(30.0, 0.0, 626785.0, 0.0, -30.0, 4837025.0)
        assert np.array_equal(r2.res, [30.0, 30.0])
        assert r2.bounds == rio.coords.BoundingBox(left=626785.0, bottom=4833305.0, right=631735.0, top=4837025.0)
        assert r2.crs == rio.crs.CRS.from_epsg(32718)
        assert r2.footprint.vector_equal(r2.get_footprint_projected(r2.crs))

        # Test 2 - loading the data afterward
        r.load()
        assert r.is_loaded
        assert r.count == 1
        assert r._count_on_disk == 1
        assert r.bands == (1,)
        assert r._bands_on_disk == (1,)
        assert r.data.shape == (r.height, r.width)

        # Test 3 - single band, loading data
        r = gu.Raster(self.landsat_b4_path, load_data=True)
        assert r.is_loaded
        assert r.count == 1
        assert r._count_on_disk == 1
        assert r.bands == (1,)
        assert r._bands_on_disk == (1,)
        assert r.data.shape == (r.height, r.width)

        # Test 4 - multiple bands, load all bands
        r = gu.Raster(self.landsat_rgb_path, load_data=True)
        assert r.count == 3
        assert r._count_on_disk == 3
        assert r.bands == (
            1,
            2,
            3,
        )
        assert r._bands_on_disk == (
            1,
            2,
            3,
        )
        assert r.data.shape == (r.count, r.height, r.width)

        # Test 5 - multiple bands, load one band only
        r = gu.Raster(self.landsat_rgb_path, load_data=True, bands=1)
        assert r.count == 1
        assert r._count_on_disk == 3
        assert r.bands == (1,)
        assert r._bands_on_disk == (1, 2, 3)
        assert r.data.shape == (r.height, r.width)

        # Test 6 - multiple bands, load a list of bands
        r = gu.Raster(self.landsat_rgb_path, load_data=True, bands=[2, 3])
        assert r.count == 2
        assert r._count_on_disk == 3
        assert r.bands == (1, 2)
        assert r._bands_on_disk == (1, 2, 3)
        assert r.data.shape == (r.count, r.height, r.width)

        # Test 7 - load a single band a posteriori calling load()
        r = gu.Raster(self.landsat_rgb_path)
        r.load(bands=1)
        assert r.count == 1
        assert r._count_on_disk == 3
        assert r.bands == (1,)
        assert r._bands_on_disk == (1, 2, 3)
        assert r.data.shape == (r.height, r.width)

        # Test 8 - load a list of band a posteriori calling load()
        r = gu.Raster(self.landsat_rgb_path)
        r.load(bands=[2, 3])
        assert r.count == 2
        assert r._count_on_disk == 3
        assert r.bands == (1, 2)
        assert r._bands_on_disk == (1, 2, 3)
        assert r.data.shape == (r.count, r.height, r.width)

        # Check that errors are raised when appropriate
        with pytest.raises(ValueError, match="Data are already loaded."):
            r.load()
        with pytest.raises(
            AttributeError,
            match="Cannot load as name is not set anymore. Did you manually update the name attribute?",
        ):
            r = gu.Raster(self.landsat_b4_path)
            r._name = None
            r.load()

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path, landsat_rgb_path])
    def test_load_only_mask(self, example: str) -> None:
        """
        Test that loading only mask works properly.
        """

        # Load raster with and without loading
        r_loaded = gu.Raster(example, load_data=True)
        r_notloaded = gu.Raster(example)

        # Get the mask for the two options
        mask_loaded = np.ma.getmaskarray(r_loaded.data)
        mask_notloaded = r_notloaded._load_only_mask()

        # Data should not be loaded and masks should be equal
        assert not r_notloaded.is_loaded
        assert np.array_equal(mask_notloaded, mask_loaded)

    @pytest.mark.parametrize("bands", [2, [2, 3]])
    def test_load_only_mask__selected_bands_downsample(self, bands: int | list[int]) -> None:
        """Checks that selected band masks match loaded data after downsampling without loading the source."""

        # Read a reduced mask from a multiband file with either one or two selected bands
        source = gu.Raster(self.landsat_rgb_path, downsample=2)
        selected_mask = source._load_only_mask(bands=bands)

        # Compare with the mask of the same bands loaded as raster values
        reference = gu.Raster(self.landsat_rgb_path, downsample=2, bands=bands)
        expected_mask = np.ma.getmaskarray(reference.data)
        assert np.array_equal(selected_mask, expected_mask)
        assert not source.is_loaded


class TestRasterWriting:
    """Test module for Raster.to_file() storage options, values and georeferencing."""

    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])
    def test_to_file(self, example: str) -> None:
        """Checks that Raster.to_file() preserves values and applies the requested storage options."""

        # Read single band raster
        img = gu.Raster(example)

        # Temporary folder
        temp_dir = tempfile.TemporaryDirectory()

        # Save file to temporary file, with defaults opts
        temp_file = os.path.join(temp_dir.name, "test.tif")
        img.to_file(temp_file)
        saved = gu.Raster(temp_file)
        assert img.raster_equal(saved)
        assert (Image.open(temp_file)).info["compression"] == "tiff_adobe_deflate"  # test no default compression
        assert not saved._is_bigtiff()  # test default BIGTIFF param (IF_SAFER)

        # Try to save with a pathlib path (create a new temp file for Windows)
        path = pathlib.Path(temp_file)
        img.to_file(path)

        # Test additional option: one co_opts with tiled (with no compression)
        co_opts = {"COMPRESS": "NONE"}
        img.to_file(temp_file, co_opts=co_opts)
        assert len((Image.open(temp_file)).tile) > 1  # test {TILED": "NO"} default value
        saved = gu.Raster(temp_file)
        assert not saved._is_bigtiff()  # test default BIGTIFF param (IF_SAFER)

        # Test additional options: several co_opts with compress and bigtiff + metadata
        co_opts = {"COMPRESS": "LZW", "BIGTIFF": "YES"}
        metadata = {"Type": "test"}
        img.to_file(temp_file, co_opts=co_opts, metadata=metadata)
        saved = gu.Raster(temp_file)
        assert img.raster_equal(saved)
        assert (Image.open(temp_file)).info["compression"] == "tiff_lzw"  # test {"COMPRESS": "LZW"}
        assert saved._is_bigtiff()  # test {"BIGTIFF": "YES"}
        assert saved.tags["Type"] == "test"

        # Test saving file in COG format
        img.to_file(temp_file, driver="COG")
        saved = gu.Raster(temp_file)
        assert img.raster_equal(saved)
        assert saved.tags["LAYOUT"] == "COG"

        # Test that nodata value is enforced when masking - since value 0 is not used, data should be unchanged
        img.to_file(temp_file, nodata=0)
        saved = gu.Raster(temp_file)
        assert np.ma.allequal(img.data, saved.data)
        assert saved.nodata == 0

        # Test that mask is preserved if nodata value is valid
        mask = img.data == np.min(img.data)
        img.set_mask(mask)
        if img.nodata is not None:
            img.to_file(temp_file)
            saved = gu.Raster(temp_file)
            assert np.array_equal(img.data.mask, saved.data.mask)

        # Test that a warning is raised if nodata is not set and a mask exists (defined above)
        if img.nodata is None:
            with pytest.warns(UserWarning):
                img.to_file(TemporaryFile())

        # Test with blank argument
        img.to_file(temp_file, blank_value=0)
        saved = gu.Raster(temp_file)

        assert np.array_equal(saved.data.data, np.zeros(np.shape(saved.data)))

        # Clean up temporary folder - fails on Windows
        try:
            temp_dir.cleanup()
        except (NotADirectoryError, PermissionError):
            pass

    def test_to_file__dtype(self, tmp_path: pathlib.Path) -> None:
        """Checks that to_file() casts stored values and metadata to the requested dtype."""

        # Create a byte raster with values that can be represented exactly as floats
        values = np.arange(4, dtype=np.uint8).reshape(2, 2)
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 2, 1, 1), crs=4326)
        filename = tmp_path / "float-output.tif"

        # Save forcing floating point type
        raster.to_file(filename, dtype=np.float32)
        saved = gu.Raster(filename)

        # Check both file metadata and stored values use the requested type
        assert np.dtype(saved.dtype) == np.dtype("float32")
        assert np.array_equal(saved.data, values.astype(np.float32))

    @pytest.mark.parametrize("explicit", [False, True])
    def test_to_file__georeferencing(self, raster_gcp_rpc: gu.Raster, tmp_path: Path, explicit: bool) -> None:
        """Checks that GCPs/RPCs are written properly (whether stored in the Raster object or passed during writing)."""

        # Create an image with a CRS differing from the GCP CRS, and we add ordinary metadata
        source = raster_gcp_rpc
        image = gu.Raster.from_array(source.data, rio.Affine.identity(), 3857, nodata=source.nodata)
        image.tags["label"] = "source"
        if source.rpcs is not None:
            image.tags["RPC"] = source.rpcs.to_gdal()
        options = {}
        if explicit:
            options = {"gcps": source.gcps[0], "gcps_crs": source.gcps[1], "rpcs": source.rpcs}
        else:
            image.gcps, image.rpcs = source.gcps, source.rpcs
        filename = tmp_path / "gcp_rcp_metadata.tif"

        # We write to file, passing GCPs/RCPs in options
        image.to_file(filename, metadata={"label": "output"}, **options)

        # As TIFF rewrites GCP identifiers, we compare their coordinates and CRS separately
        with rio.open(filename) as dataset:
            points, point_crs = dataset.gcps
            assert [(g.row, g.col, g.x, g.y, g.z) for g in points] == [
                (g.row, g.col, g.x, g.y, g.z) for g in source.gcps[0]
            ]
            assert point_crs == source.gcps[1]
            assert dataset.rpcs == source.rpcs
            assert "RPC" not in dataset.tags()
            assert bool(dataset.tags(ns="RPC")) == (source.rpcs is not None)
            assert dataset.tags()["label"] == "output"
            np.testing.assert_array_equal(dataset.read(), source.data.filled(source.nodata))
        assert image.tags["label"] == "source"
        if source.rpcs is not None:
            assert image.tags["RPC"] == source.rpcs.to_gdal()

    def test_to_file__legacy_gcp_tuples(self, tmp_path: Path) -> None:
        """Checks that passing a GCP when writing overwrites the affine transform of the raster, with a warning."""

        # We create GCPs
        source = gu.Raster.from_array(np.ones((4, 5)), rio.transform.from_origin(0, 4, 1, 1), 4326)
        points = [(0, 0, 10, 50), (0, 5, 10.05, 50), (4, 0, 10, 49.96), (4, 5, 10.05, 49.96)]
        filename = tmp_path / "legacy_gcps.tif"

        # Check warning is raised, and affine transform is overwritten
        with pytest.warns(UserWarning, match="geotransform previously set"):
            source.to_file(filename, gcps=points)
        result = gu.Raster(filename)
        assert result.transform == rio.Affine.identity()
        assert result.crs is None
        assert result.gcps[1] == source.crs
        assert [(g.row, g.col, g.x, g.y) for g in result.gcps[0]] == points

    @pytest.mark.parametrize(
        "raster_gcp_rpc",
        ["gcp_projected", "gcp_nonlinear", "rpc_polynomial", "rpc_rational", "gcp_rpc_polynomial", "gcp_rpc_rational"],
        indirect=True,
    )
    @pytest.mark.parametrize("method", ["to_file", "to_rio_dataset", "to_xarray", "to_xarray_file", "from_xarray_file"])
    def test_metadata__roundtrip(self, raster_gcp_rpc: gu.Raster, tmp_path: Path, method: str) -> None:
        """Checks that GCPs/RPCs and values survive both disk writing/reading, and in-memory conversions."""

        # Export, then read the result back as a Raster
        source = raster_gcp_rpc
        if method == "to_file":
            filename = tmp_path / "roundtrip.tif"
            source.to_file(filename)
            result = gu.Raster(filename)
        elif method == "to_rio_dataset":
            with source.to_rio_dataset() as dataset:
                assert "RPC" not in dataset.tags()
                result = gu.Raster(dataset, load_data=True)
        else:
            ds = source.to_xarray()
            result = ds.rst.to_geoutils()
            assert [g.asdict() for g in ds.rst.gcps[0]] == [g.asdict() for g in source.gcps[0]]
            assert ds.rst.gcps[1] == source.gcps[1]
            assert ds.rst.rpcs == source.rpcs
            if method in ("to_xarray_file", "from_xarray_file"):
                filename = tmp_path / "xarray_roundtrip.tif"
                if method == "to_xarray_file":
                    ds.rst.to_file(filename, tiled=True, blockxsize=256, blockysize=256)
                else:
                    result.to_file(filename)
                result = gu.Raster(filename)
                assert ds.rst.rpcs == source.rpcs
                assert ds.rst.gcps[1] == source.gcps[1]
                assert [g.asdict() for g in ds.rst.gcps[0]] == [g.asdict() for g in source.gcps[0]]

        # We also read file metadata directly -so an RPC dictionary cannot be hidden in a default string tag)
        if method in ("to_file", "to_xarray_file", "from_xarray_file"):
            with rio.open(filename) as dataset:
                assert "RPC" not in dataset.tags()
                assert dataset.rpcs == source.rpcs

        # TIFF rewrites GCP identifiers, so compare pixel and ground coordinates independently
        expected_points = [(g.row, g.col, g.x, g.y, g.z) for g in source.gcps[0]]
        actual_points = [(g.row, g.col, g.x, g.y, g.z) for g in result.gcps[0]]
        assert actual_points == expected_points
        assert result.gcps[1] == source.gcps[1]
        assert result.rpcs == source.rpcs
        np.testing.assert_array_equal(result.data.filled(source.nodata), source.data.filled(source.nodata))

    @pytest.mark.parametrize("representation", ["class", "dataarray", "dataset"])
    @pytest.mark.parametrize("access", ["filesystem", "storage_options"])
    def test_to_file__filesystem(self, representation: str, access: str, tmp_path: Path) -> None:
        """Checks that to_file() uses access filesystem options and preserves values and CRS."""
        filesystem = pytest.importorskip("fsspec").filesystem("memory")
        path = tmp_path / "output.tif"
        filename = str(path) if access == "filesystem" else f"memory://{path}"
        options = {"filesystem": filesystem} if access == "filesystem" else {"storage_options": {}}

        # Create test data
        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        raster = gu.Raster.from_array(values, from_origin(0, 3, 1, 1), 4326)
        array = raster.to_xarray().squeeze("band")
        writer = raster if representation == "class" else array.rst
        if representation == "dataset":
            writer = array.to_dataset(name="elevation").rst

        writer.to_file(filename, **options)
        assert filesystem.exists(str(path)) and not path.exists()

        # Read the filesystem output
        with gu.open_raster(str(path), filesystem=filesystem) as result:
            np.testing.assert_array_equal(result.values, values)
            assert result.rst.crs == raster.crs


class TestRasterArrayWriting:
    """Test module for DataArray raster writing, storage encoding and missing values."""

    @pytest.mark.parametrize("nodata", [np.iinfo(np.int32).min, np.iinfo(np.int32).max])
    def test_from_array__encoding(self, nodata: int, tmp_path: Path) -> None:
        """Checks that from_array() and copy() preserve encoding."""

        # Create values with custom encoding
        # Large integer nodata can lose precision in float32
        values = np.arange(6, dtype=np.float32).reshape(2, 3)
        encoding = {
            "dtype": "int32",
            "scale_factor": 0.5,
            "add_offset": 100,
            "_FillValue": nodata,
            "grid_mapping": "previous_grid",
        }
        original_encoding = encoding.copy()

        # Create raster and copy with new values
        result = gu.DataArrayRasterAccessor.from_array(
            values, from_origin(0, 2, 1, 1), 32631, nodata=nodata, encoding=encoding, name="height"
        )
        copied = result.rst.copy(new_array=values + 1)

        # Check output encoding and unchanged input
        expected_encoding = dict(encoding, grid_mapping="spatial_ref")
        assert result.encoding == expected_encoding
        assert copied.encoding == expected_encoding
        assert result.rst.nodata == nodata and copied.rst.nodata == nodata
        assert "_FillValue" not in result.attrs and "_FillValue" not in copied.attrs
        assert result.name == copied.name == "height"
        np.testing.assert_array_equal(result.values, values)
        np.testing.assert_array_equal(copied.values, values + 1)
        assert encoding == original_encoding

        # Modify copied encoding and check input is unchanged
        copied.encoding["add_offset"] = 200
        assert result.encoding["add_offset"] == encoding["add_offset"] == 100

        # Check an unchanged grid returns the original array, then save a missing pixel
        with pytest.warns(UserWarning, match="Output projection, bounds and grid size are identical"):
            assert result.rst.reproject(res=1) is result
        missing = values.copy()
        missing[0, 0] = np.nan
        filename = tmp_path / "encoded.tif"
        result.rst.copy(new_array=missing).rst.to_file(filename)
        with rio.open(filename) as saved:
            assert saved.nodata == nodata
            assert saved.dtypes == ("int32",)
            decoded = saved.read(1, masked=True).astype(float) * saved.scales[0] + saved.offsets[0]
            np.testing.assert_array_equal(decoded.filled(np.nan), missing)

    @pytest.mark.parametrize("setter", ["accessor", "property", "method"])
    def test_nodata__packed_roundtrip(self, tmp_path: Path, setter: str) -> None:
        """Checks that nodata access and setters preserve missing pixels."""

        # Write packed raster with a gap
        values = np.array([[100.0, 101.0], [102.0, np.nan]])
        source = gu.DataArrayRasterAccessor.from_array(values, from_origin(0, 2, 1, 1), 32631).rename("height")
        source.rio.write_nodata(-32768, encoded=True, inplace=True)
        source.encoding.update(dtype="int16", scale_factor=0.1, add_offset=100)
        filename = tmp_path / "packed.nc"
        source.to_dataset().to_netcdf(filename, engine="scipy")

        # Read nodata and optionally change its value
        with xr.open_dataset(filename, engine="scipy", decode_coords="all") as opened:
            array = opened.height
            encoding = array.encoding.copy()
            assert array.rst.nodata == -32768
            assert array.encoding == encoding
            if setter == "property":
                array.rst.nodata = -32767
            elif setter == "method":
                array.rst.set_nodata(-32767)
            assert "_FillValue" not in array.attrs
            output_file = tmp_path / "rewritten.nc"
            array.to_dataset().to_netcdf(output_file, engine="scipy")

        # Reopen file and check values and nodata
        with xr.open_dataset(output_file, engine="scipy", decode_coords="all") as saved:
            np.testing.assert_array_equal(saved.height.values, values)
            assert saved.height.rst.nodata == (-32768 if setter == "accessor" else -32767)
        with xr.open_dataset(output_file, engine="scipy", decode_cf=False) as raw:
            assert raw.height.dtype.kind == "i" and raw.height.dtype.itemsize == 2
            assert raw.height.values[1, 1] == raw.height.attrs["_FillValue"]

    @pytest.mark.parametrize("setter", ["property", "method"])
    def test_nodata__remove_encoded_fill_value_roundtrip(self, tmp_path: Path, setter: str) -> None:
        """Checks that clearing nodata preserves missing pixels."""

        # Create raster with integer encoding and a gap
        values = np.array([[100.0, 101.0], [102.0, np.nan]])
        source = gu.DataArrayRasterAccessor.from_array(values, from_origin(0, 2, 1, 1), 32631).rename("height")
        source.rio.write_nodata(-32768, encoded=True, inplace=True)
        source.encoding.update(dtype="int16", scale_factor=0.1, add_offset=100)

        # Clear nodata and check encoding
        if setter == "property":
            source.rst.nodata = None
        else:
            source.rst.set_nodata(None)
        assert source.rst.nodata is None
        assert "_FillValue" not in source.attrs and "_FillValue" not in source.encoding
        assert "dtype" not in source.encoding
        assert source.encoding["scale_factor"] == 0.1 and source.encoding["add_offset"] == 100

        # Write and reopen raster to check missing pixels
        filename = tmp_path / "without_sentinel.nc"
        source.to_dataset().to_netcdf(filename, engine="scipy")
        with xr.open_dataset(filename, engine="scipy", decode_coords="all") as saved:
            np.testing.assert_array_equal(saved.height.values, values)
            assert saved.height.encoding["dtype"].kind == "f"

    @pytest.mark.parametrize("raster_gcp_rpc", ["rpc_polynomial", "rpc_rational"], indirect=True)
    def test_to_file__explicit_rpcs(self, raster_gcp_rpc: gu.Raster, tmp_path: Path) -> None:
        """Checks that to_file() writes explicit RPCs without assigning them to the input DataArray."""

        # Create a DataArray with nodata sentinels and no stored GCPs/RPCs
        source = raster_gcp_rpc
        image = gu.DataArrayRasterAccessor.from_array(
            source.data.filled(source.nodata), rio.Affine.identity(), None, nodata=source.nodata
        )
        filename = tmp_path / "explicit_rpcs.tif"

        # Write the coefficients directly through the accessor
        image.rst.to_file(filename, rpcs=source.rpcs)

        # Rasterio must read the native RPC namespace while the input has no RPCs
        with rio.open(filename) as dataset:
            assert dataset.rpcs == source.rpcs
            assert "RPC" not in dataset.tags()
            np.testing.assert_array_equal(dataset.read(), source.data.filled(source.nodata))
        assert image.rst.rpcs is None


class TestMaskWriting:
    """Test module for saving and reopening boolean rasters."""

    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    everest_outlines_path = examples.get_path_test("everest_rgi_outlines")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")
    mask_landsat_b4 = gu.Raster(landsat_b4_path) > 125
    mask_aster_dem = gu.Raster(aster_dem_path) > 2000
    mask_everest = gu.Vector(everest_outlines_path).create_mask(gu.Raster(landsat_b4_path))

    @pytest.mark.parametrize("mask", [mask_landsat_b4, mask_aster_dem, mask_everest])
    def test_to_file(self, mask: gu.Raster) -> None:
        """Test saving for masks"""

        # Temporary folder
        temp_dir = tempfile.TemporaryDirectory()

        # Save file to temporary file, with defaults opts
        temp_file = os.path.join(temp_dir.name, "test.tif")
        mask.to_file(temp_file)
        saved = gu.Raster(temp_file, is_mask=True)

        # A raster (or mask) in-memory has more information than on disk, we need to update it before checking equality
        # The values in its .data.data that are masked in .data.mask are not necessarily equal to the nodata value
        mask.data.data[mask.data.mask] = True  # The default nodata 255 is converted to boolean True on masked values

        # Check all attributes are equal
        assert mask.raster_equal(saved)

        # Clean up temporary folder - fails on Windows
        try:
            temp_dir.cleanup()
        except (NotADirectoryError, PermissionError):
            pass


class TestRasterOpeningChunked:
    """Test module for inspecting disk chunk metadata without loading Dask raster values."""

    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset"])
    @pytest.mark.parametrize("squeeze", [False, True])
    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("downsample", [6, 6.5, {"res": 6}, {"shape": (7, 13)}])
    def test_open_raster__reduced_reads(
        self,
        tmp_path: Path,
        as_object_type: Literal["dataarray", "dataset"],
        squeeze: bool,
        masked: bool,
        downsample: Any,
    ) -> None:
        """Checks that reduced reads stay lazy across uneven chunks and match eager values and metadata."""
        pytest.importorskip("dask")
        from dask.callbacks import Callback

        values = np.arange(5_000, dtype=np.int16).reshape(50, 100)
        values[::3, ::5] = -9999
        filename = tmp_path / "chunked_grid.tif"
        array = np.ma.masked_equal(values, -9999)
        gu.Raster.from_array(array, from_origin(0, 50, 1, 1), 32631, nodata=-9999).to_file(filename)
        with rio.open(filename, "r+") as source:
            source.build_overviews([2, 4], rio.enums.Resampling.nearest)

        options: dict[str, Any] = {
            "downsample": downsample,
            "squeeze": squeeze,
            "masked": masked,
            "as_object_type": as_object_type,
        }
        with pytest.warns(UserWarning, match="masked=False") if not masked else nullcontext():
            eager = open_raster(str(filename), **options)
            # Short final chunks expose boundaries within the reduced grid
            with Callback(pretask=lambda *args: pytest.fail("Opening must not compute pixels")):
                lazy = open_raster(str(filename), chunks={"y": 3, "x": 5}, **options)
        eager_values = eager["raster"] if as_object_type == "dataset" else eager
        lazy_values = lazy["raster"] if as_object_type == "dataset" else lazy
        assert not eager_values.rst.is_loaded and not lazy_values.rst.is_loaded
        assert lazy_values.chunks
        xr.testing.assert_identical(lazy.compute(), eager)
        assert not lazy_values.rst.is_loaded
        eager.close()
        lazy.close()

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset"])
    def test_open_raster__disk_chunk_metadata(
        self, tmp_path: Path, as_object_type: Literal["dataarray", "dataset"]
    ) -> None:
        """Checks that disk tile metadata remains independent of computation chunks and matches in-memory opening."""
        pytest.importorskip("dask")
        from dask.callbacks import Callback

        # Write 16/32 pixel disk tiles, then use chunks of 7/9 to split those tiles differently
        values = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        filename = tmp_path / "tiles.tif"
        gu.Raster.from_array(values, from_origin(0, 64, 1, 1), 32631).to_file(
            filename, co_opts={"TILED": "YES", "BLOCKYSIZE": "16", "BLOCKXSIZE": "32"}
        )
        eager = open_raster(str(filename), as_object_type=as_object_type)

        # Inspect disk shapes while preventing computation of raster pixels
        with Callback(pretask=lambda *args: pytest.fail("Disk metadata must not compute raster values")):
            lazy = open_raster(str(filename), chunks={"y": 7, "x": 9}, as_object_type=as_object_type)
            variable = lazy["raster"] if as_object_type == "dataset" else lazy
            assert lazy.attrs["block_shape"] == variable.attrs["block_shape"] == (16, 32)
        assert not variable.rst.is_loaded

        # Compare exact computed values and metadata with the same in-memory call
        xr.testing.assert_identical(lazy.compute(scheduler="synchronous"), eager)
        assert not variable.rst.is_loaded
        lazy.close()
        eager.close()

    @pytest.mark.parametrize("storage", ["netcdf", "zarr"])
    def test_tags__disk_chunks(self, tmp_path: Path, storage: str) -> None:
        """Checks that inspecting stored array chunks stays lazy and computed values match in-memory opening."""
        pytest.importorskip("dask")
        from dask.callbacks import Callback

        # Write 16/32 pixel disk chunks, then combine them into 32/64 pixel computation chunks
        values = np.arange(32 * 64, dtype=np.float32).reshape(32, 64)
        array = gu.Raster.from_array(values, from_origin(0, 32, 1, 1), 32631).to_xarray()
        array = array.squeeze("band", drop=True).rename("height")
        encoding_key = "chunks" if storage == "zarr" else "chunksizes"
        encoding = {"height": {encoding_key: (16, 32), "grid_mapping": array.rio.grid_mapping}}
        filename = tmp_path / storage
        if storage == "zarr":
            pytest.importorskip("zarr")
            array.to_dataset().to_zarr(filename, zarr_format=2, encoding=encoding)
            eager = xr.open_zarr(filename, chunks=None, decode_coords="all")
        else:
            pytest.importorskip("netCDF4")
            array.to_dataset().to_netcdf(filename, engine="netcdf4", encoding=encoding)
            eager = xr.open_dataset(filename, engine="netcdf4", decode_coords="all")

        # Inspect physical chunk sizes without letting Dask read any pixels
        with Callback(pretask=lambda *args: pytest.fail("Disk metadata must not compute array values")):
            if storage == "zarr":
                lazy = xr.open_zarr(filename, chunks={"y": 32, "x": 64}, decode_coords="all")
            else:
                lazy = xr.open_dataset(filename, engine="netcdf4", chunks={"y": 32, "x": 64}, decode_coords="all")
            assert lazy["height"].rst.tags["disk_chunks"] == (16, 32)
        assert not lazy["height"].rst.is_loaded

        # Compare the computed result exactly with in-memory values and cached disk metadata
        assert eager["height"].rst.tags["disk_chunks"] == (16, 32)
        xr.testing.assert_identical(lazy.compute(scheduler="synchronous"), eager)
        assert not lazy["height"].rst.is_loaded
        lazy.close()
        eager.close()
        array.close()

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset"])
    @pytest.mark.parametrize("downsample", [1, 6])
    def test_open_raster__downsample_loading_laziness(
        self, tmp_path: Path, as_object_type: Literal["dataarray", "dataset"], downsample: int
    ) -> None:
        """Checks that chunked downsampling stays lazy and exactly matches in-memory opening."""

        pytest.importorskip("dask.array")
        from dask.callbacks import Callback

        # Write input on dimensions not divisible by 6
        values = np.arange(5_000, dtype=np.int32).reshape(50, 100)
        path = tmp_path / "gradient.tif"
        gu.Raster.from_array(values, from_origin(0, 50, 1, 1), 4326).to_file(path)

        # Uneven tiles expose shorter final chunks on both the native and reduced grids
        eager = open_raster(str(path), downsample=downsample, as_object_type=as_object_type)
        with Callback(pretask=lambda *args: pytest.fail("Opening rasters must read only metadata")):
            lazy = open_raster(str(path), downsample=downsample, chunks={"y": 3, "x": 7}, as_object_type=as_object_type)
        eager_values = eager["raster"] if as_object_type == "dataset" else eager
        lazy_values = lazy["raster"] if as_object_type == "dataset" else lazy
        assert not eager_values._in_memory
        assert not lazy_values._in_memory
        expected_chunks = ((3,) * 16 + (2,), (7,) * 14 + (2,)) if downsample == 1 else ((3, 3, 2), (7, 7, 2))
        assert lazy_values.chunks == expected_chunks

        # Check result leaves source lazy and matches exactly with in-memory resampling
        computed = lazy.compute()
        xr.testing.assert_identical(computed, eager)
        computed_values = computed["raster"] if as_object_type == "dataset" else computed
        assert computed_values._in_memory
        assert not lazy_values._in_memory

    @pytest.mark.parametrize("path_raster", [landsat_b4_path, aster_dem_path])
    def test_open__dask(self, path_raster: str) -> None:
        """
        Check that a DataArray opened with chunks using "open_raster" maintains Dask laziness.

        Note: this is different from loading mechanism of Xarray (triggers when calling .data).
        """
        pytest.importorskip("dask")
        import dask.array as da

        # Open raster lazily with chunks
        ds = open_raster(path_raster, chunks={"band": 1, "x": 10, "y": 10})

        # Array should be a Dask array (chunks exist)
        ds_arr = ds.data
        assert not ds._in_memory
        assert isinstance(ds_arr, da.Array)
        assert ds_arr.chunks is not None

        # After compute, it should be a NumPy array
        ds_comp = ds.compute()
        assert isinstance(ds_comp.data, np.ndarray)
        assert ds_comp._in_memory

    @pytest.mark.parametrize("remote_url", ["http-query"], indirect=True)
    @pytest.mark.parametrize("representation", ["class", "dataarray", "dask"])
    @pytest.mark.parametrize("downsample", [1, 2])
    def test_open_or_init__signed_external_mask(
        self, remote_files: RemoteFiles, remote_url: Any, representation: str, downsample: int
    ) -> None:
        """Checks that signed TIFF masks match eager reads through every raster reader."""
        filesystem = pytest.importorskip("fsspec").filesystem("http", skip_instance_cache=True)
        path = remote_files.directory / "masked.tif"
        values = np.ones((6, 8), dtype=np.float32)
        mask = np.full(values.shape, 255, dtype=np.uint8)
        mask[2:4, 2:4] = 0
        with rio.Env(GDAL_TIFF_INTERNAL_MASK=False):
            gu.Raster.from_array(values, from_origin(0, 6, 1, 1), 4326).to_file(path)
            with rio.open(path, "r+") as dataset:
                dataset.write_mask(mask)
        source = remote_url(path.name)

        with rio.Env(GDAL_DISABLE_READDIR_ON_OPEN="FALSE"):
            expected = gu.Raster(path, downsample=downsample, load_data=True)
            if representation == "class":
                remote = gu.Raster(source, filesystem=filesystem, downsample=downsample)
                assert expected.is_loaded and not remote.is_loaded
                result = remote.to_nanarray()
                assert remote.is_loaded
            else:
                options: dict[str, Any] = {"chunks": 2} if representation == "dask" else {}
                with gu.open_raster(source, filesystem=filesystem, downsample=downsample, **options) as remote:
                    assert not remote.rst.is_loaded
                    computed = remote.compute(scheduler="synchronous")
                    assert computed.rst.is_loaded
                    if representation == "dask":
                        assert not remote.rst.is_loaded
                    result = computed.values

        np.testing.assert_array_equal(result, expected.to_nanarray())

    @pytest.mark.parametrize("file_access", ["http-filesystem"], indirect=True)
    @pytest.mark.parametrize(
        "reader,options,filename",
        [
            (gu.open_raster, {"downsample": 2}, "elevation.tif"),
            (gu.open_raster, {"as_object_type": "dataset", "overview_level": 0}, "elevation.tif"),
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


class TestRasterWritingChunked:
    """Test module for chunked raster writing and storage encoding compared with eager reads and writes."""

    aster_dem_path = examples.get_path_test("exploradores_aster_dem")

    def test_open__dask_nodata_can_be_written(self, tmp_path: Path) -> None:
        """Write a lazily opened nodata raster without conflicting xarray metadata."""

        pytest.importorskip("dask")

        # Open a masked file without changing its encoded nodata
        ds = open_raster(self.aster_dem_path, chunks={"band": 1, "x": 100, "y": 100})
        assert ds.rst.nodata is not None
        assert ds.encoding["_FillValue"] == ds.rst.nodata
        assert "_FillValue" not in ds.attrs

        # The final writer must preserve nodata while computing the Dask array
        output_file = tmp_path / "dask-nodata.tif"
        ds.rst.to_file(output_file)
        with rio.open(output_file) as output:
            assert output.nodata == ds.rst.nodata

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("representation", ["dataarray", "dataset"])
    @pytest.mark.parametrize("storage", ["netcdf", "geotiff"])
    @pytest.mark.parametrize(
        "method,options",
        [
            ("copy", {}),
            ("crop", {"bbox": (1, 1, 5, 4)}),
            ("icrop", {"bbox": (1, 1, 5, 4)}),
            ("clip", {"mask": (1, 1, 5, 4)}),
            ("translate", {"xoff": 10000, "yoff": 10000}),
            ("reproject", {"res": 0.5, "resampling": "nearest"}),
            ("reproject", {"res": 0.5, "resampling": "nearest", "backend": "scipy"}),
            ("filter", {"method": "mean", "size": 3}),
            ("filter", {"method": "median", "size": 3}),
            ("fill_nodata", {"interpolation": "nearest"}),
        ],
    )
    def test_methods__encoding_roundtrip(
        self, tmp_path: Path, method: str, options: dict[str, Any], storage: str, representation: str, lazy: bool
    ) -> None:
        """Checks that operations preserve metadata and encoding when writing files."""

        # Write raster with custom metadata and encoding
        values = np.arange(35, dtype=np.float32).reshape(5, 7)
        values[2, 3] = np.nan
        array = gu.DataArrayRasterAccessor.from_array(values, from_origin(0, 5, 1, 1), 32631, nodata=255)
        array = array.rename("height")
        array.attrs.update(title="elevation survey", units="m")
        if storage == "netcdf":
            pytest.importorskip("netCDF4")
            array.rio.write_nodata(0, encoded=True, inplace=True)
            array.encoding.update(
                dtype="int16", scale_factor=0.1, add_offset=100, zlib=True, complevel=1, chunksizes=(3, 4)
            )
            # Store coordinates as integers so large translations must discard their old encoding
            for name in ("x", "y"):
                array.coords[name].encoding.update(dtype="int16", scale_factor=0.1, _FillValue=-32768)
            filename = tmp_path / "input.nc"
            array.to_dataset().to_netcdf(filename, engine="netcdf4")
        else:
            array.rio.write_nodata(255, encoded=True, inplace=True)
            array.encoding["dtype"] = "uint8"
            filename = tmp_path / "input.tif"
            array.rst.to_file(filename)

        # Open raster eagerly or in chunks of 3 rows and 4 columns
        chunks = {"y": 3, "x": 4} if lazy else None
        if lazy:
            pytest.importorskip("dask")
        if storage == "netcdf":
            opened = xr.open_dataset(filename, engine="netcdf4", decode_coords="all", chunks=chunks)
        else:
            opened = open_raster(str(filename), as_object_type="dataset", data_name="height", chunks=chunks)
        source = opened.height if representation == "dataarray" else opened
        if not lazy:
            source.load()
        _ = opened.height.rst
        original = source.copy(deep=True)
        encoding = opened.height.encoding.copy()

        # Run each operation and check metadata is unchanged
        if method == "copy" and representation == "dataset":
            result = source.copy()
        else:
            result = getattr(source.rst, method)(**options)
        variable = result if representation == "dataarray" else result.height
        assert variable.encoding == encoding
        assert opened.height.encoding == encoding
        assert variable.attrs["title"] == "elevation survey"
        assert variable.attrs["units"] == "m"
        assert "_FillValue" not in variable.attrs
        assert variable.rst.nodata == encoding["_FillValue"]
        assert variable.encoding["grid_mapping"] in variable.coords
        if lazy:
            assert opened.height.chunks and not opened.height.rst.is_loaded
            # DataArray filling runs immediately; Dataset filling stays lazy
            if method == "fill_nodata" and representation == "dataarray":
                assert variable.chunks is None and variable.rst.is_loaded
            else:
                assert variable.chunks and not variable.rst.is_loaded
            eager = original.compute()
            if method != "copy" or representation == "dataarray":
                eager = getattr(eager.rst, method)(**options)
            xr.testing.assert_identical(result.compute(), eager)
        xr.testing.assert_identical(source, original)

        # Check valid zeros survive reprojection even when the stored nodata value is zero
        if method == "reproject":
            np.testing.assert_array_equal(variable.values, np.repeat(np.repeat(values, 2, axis=0), 2, axis=1))

        # Write and reopen result to check values and encoding
        dataset = variable.rename("height").to_dataset()
        if storage == "netcdf":
            output_file = tmp_path / "output.nc"
            dataset.to_netcdf(output_file, engine="netcdf4")
            saved = xr.open_dataset(output_file, engine="netcdf4", decode_coords="all")
        else:
            output_file = tmp_path / "output.tif"
            result.rst.to_file(output_file)
            saved = open_raster(str(output_file), as_object_type="dataset", data_name="height")
        scale = encoding.get("scale_factor", 1)
        offset = encoding.get("add_offset", 0)
        expected = np.rint((variable.values - offset) / scale) * scale + offset
        np.testing.assert_allclose(saved.height.values, expected, rtol=0, atol=1e-12, equal_nan=True)
        assert saved.height.encoding["dtype"] == encoding["dtype"]
        assert saved.height.rst.nodata == variable.rst.nodata
        assert saved.height.rst.crs == variable.rst.crs
        assert saved.height.rst.transform == variable.rst.transform
        saved.close()
        opened.close()

    @pytest.mark.skipif(find_spec("dask") is None, reason="Dask is not installed")
    @pytest.mark.parametrize("compute", [True, False])
    def test_to_file__georeferencing(self, raster_gcp_rpc: gu.Raster, tmp_path: Path, compute: bool) -> None:
        """Checks that immediate/deferred chunked writes preserve GCP/RPC metadata and exactly match in-memory files."""

        dask = pytest.importorskip("dask")
        from dask.callbacks import Callback

        # Load an in-memory reference and split the same image into chunks of 7/9, including shorter final chunks
        source = raster_gcp_rpc
        eager = gu.open_raster(source.name).load()
        lazy = gu.open_raster(source.name, chunks={"band": 1, "y": 7, "x": 9})
        lazy.attrs["label"] = "input"
        graph = lazy.data
        assert eager.rst.is_loaded
        assert eager.rst._chunks is None
        assert lazy.rst._chunks is not None
        assert not lazy.rst.is_loaded
        eager_file = tmp_path / "eager.tif"
        lazy_file = tmp_path / "chunked.tif"

        # Write both files while leaving the Dask input lazy and its metadata unchanged
        eager.rst.to_file(eager_file, tags={"label": "output"}, tiled=True, blockxsize=256, blockysize=256)
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(1)):
            pending = lazy.rst.to_file(
                lazy_file, tags={"label": "output"}, tiled=True, blockxsize=256, blockysize=256, compute=compute
            )
            # A deferred write must leave every chunk unread until compute() is called
            if compute:
                assert pending is None
            else:
                assert dask.is_dask_collection(pending)
                assert not tasks
                pending.compute()
            assert tasks
        assert eager.rst.is_loaded
        assert lazy.data is graph
        assert not lazy.rst.is_loaded
        assert lazy.attrs["label"] == "input"
        assert lazy.rst.rpcs == source.rpcs

        # Reopen unloaded results and compare values, missing pixels and GCPs/RPCs with in-memory output
        expected = gu.Raster(eager_file)
        result = gu.Raster(lazy_file)
        assert not expected.is_loaded
        assert not result.is_loaded
        assert result.raster_equal(expected)
        assert not source.is_loaded
        assert not lazy.rst.is_loaded
        with rio.open(lazy_file) as dataset:
            assert "RPC" not in dataset.tags()
            assert bool(dataset.tags(ns="RPC")) == (source.rpcs is not None)
            assert dataset.tags()["label"] == "output"

    @pytest.mark.parametrize("variable_count", [1, 2])
    @pytest.mark.parametrize("packed", [False, True])
    @pytest.mark.parametrize("lazy", [False, True])
    def test_to_file__shared_value_encoding(
        self, tmp_path: Path, variable_count: int, packed: bool, lazy: bool
    ) -> None:
        """Checks that Dataset writing preserves metadata and encoding."""

        # Create raster variables with shared encoding
        values = np.arange(35, dtype=np.float32).reshape(5, 7) + 100
        values[2, 3] = np.nan
        source = gu.DataArrayRasterAccessor.from_array(values, from_origin(0, 5, 1, 1), 32631)
        nodata = -32768 if packed else 255
        source.rio.write_nodata(nodata, encoded=True, inplace=True)
        source.encoding["dtype"] = "int16" if packed else "uint8"
        if packed:
            source.encoding.update(scale_factor=0.5, add_offset=100)
        dataset = xr.Dataset({f"height_{index}": source.copy(data=values + index) for index in range(variable_count)})
        for name in dataset.data_vars:
            dataset[name].attrs["title"] = name
        if lazy:
            pytest.importorskip("dask")
            dataset = dataset.chunk({"x": 4, "y": 3})
        encodings = {name: variable.encoding.copy() for name, variable in dataset.data_vars.items()}

        # Write Dataset and check stored values and metadata
        filename = tmp_path / "encoded_bands.tif"
        dataset.rst.to_file(filename)
        scale = 0.5 if packed else 1
        offset = 100 if packed else 0
        with rio.open(filename) as saved:
            assert saved.count == variable_count
            assert saved.dtypes == (source.encoding["dtype"],) * variable_count
            assert saved.nodata == nodata
            assert saved.scales == (scale,) * variable_count
            assert saved.offsets == (offset,) * variable_count
            for index, name in enumerate(dataset.data_vars):
                encoded = np.rint((values + index - offset) / scale)
                encoded[np.isnan(values)] = nodata
                np.testing.assert_array_equal(saved.read(index + 1), encoded.astype(source.encoding["dtype"]))
                assert saved.tags(index + 1).get("title", saved.tags().get("title")) == name

        # Check input encoding and loading are unchanged
        assert all(dataset[name].encoding == encoding for name, encoding in encodings.items())
        if lazy:
            assert all(variable.chunks and not variable.rst.is_loaded for variable in dataset.data_vars.values())

    @pytest.mark.parametrize("representation", ["dataarray", "dataset"])
    @pytest.mark.parametrize("scheduler", ["synchronous", "threads", "processes"])
    @pytest.mark.parametrize("compute", [True, False])
    def test_to_file__chunked_writing(
        self,
        representation: str,
        scheduler: str,
        compute: bool,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Checks that chunk writing match in-memory files, and stays lazy."""

        dask = pytest.importorskip("dask")
        import dask.array as da
        from dask.callbacks import Callback

        # 1/ Create in-memory and chunked rasters
        # We create 2 bands with a NaN (to test nodata encoding when writing chunks)
        values = np.arange(70, dtype=np.float32).reshape(2, 5, 7) / 4
        values[0, 0, 0] = np.nan
        eager: xr.DataArray | xr.Dataset = gu.DataArrayRasterAccessor.from_array(
            values,
            from_origin(7, 46, 0.25, 0.25),
            4326,
            nodata=np.iinfo(np.int32).max,
            encoding={"dtype": "int32", "_FillValue": np.iinfo(np.int32).max, "scale_factor": 0.25},
        )
        if representation == "dataset":
            eager = xr.Dataset({"first": eager.isel(band=0, drop=True), "second": eager.isel(band=1, drop=True)})

        # Split the grid into chunks of 2/3 pixels (to leave shorter edge chunks, to check for incorrect writes)
        source = eager.chunk({"y": 2, "x": 3})
        if representation == "dataset":
            source["second"] = source.second.chunk({"y": 3, "x": 2})

        # Check every source is lazy
        variables = source.data_vars.values() if isinstance(source, xr.Dataset) else (source,)
        original_chunks = [variable.chunks for variable in variables]
        assert all(not variable.rst.is_loaded for variable in variables)

        # Write in-memory reference file
        expected_file = tmp_path / "eager.tif"
        eager.rst.to_file(expected_file)
        output_file = tmp_path / "chunked.tif"

        # 2/ Write chunked rasters with the selected scheduler
        task_keys = []

        def fail_array_conversion(*args: Any, **kwargs: Any) -> Any:
            """Fail if the writer gathers the Dask raster into a NumPy array."""
            raise AssertionError("to_file() must write chunks without converting the complete Dask array to NumPy")

        # We write with each scheduler
        # (Allows thread/process schedulers to test the shared file lock with two workers)
        with dask.config.set(scheduler=scheduler, num_workers=2), monkeypatch.context() as patch:
            patch.setattr(da.Array, "__array__", fail_array_conversion)
            with Callback(pretask=lambda key, graph, state: task_keys.append(key)):
                pending = source.rst.to_file(output_file, compute=compute)
                if compute:
                    # Check that a direct write executes tasks and returns no pending result
                    assert pending is None
                    assert task_keys
                else:
                    # Check that a deferred write stays lazy until we explicitly compute it
                    assert dask.is_dask_collection(pending)
                    assert not task_keys
                    assert all(not variable.rst.is_loaded for variable in variables)
                    pending.compute()
                    assert task_keys

        # 3/ Compare loading behaviour and written files
        # Check that writing leaves the source arrays lazy with their original chunks
        assert [variable.chunks for variable in variables] == original_chunks
        assert all(not variable.rst.is_loaded for variable in variables)

        # We can then open and check for exact equality (using Rasterio directly to avoid any decoding)
        with rio.open(output_file) as actual, rio.open(expected_file) as expected:
            np.testing.assert_array_equal(actual.read(), expected.read())
            assert actual.transform == expected.transform
            assert actual.crs == expected.crs
            assert actual.nodata == expected.nodata
            assert actual.count == expected.count

    @pytest.mark.parametrize("as_dataset", [False, True])
    @pytest.mark.parametrize("backend", ["memory", "directory"])
    def test_to_file__deferred_filesystem(self, as_dataset: bool, backend: str, tmp_path: Path) -> None:
        """Checks that deferred raster writes preserve values and CRS with threads or process workers."""
        fsspec = pytest.importorskip("fsspec")
        filesystem = (
            fsspec.filesystem("memory")
            if backend == "memory"
            else fsspec.filesystem("dir", path=str(tmp_path), fs=fsspec.filesystem("file"))
        )
        filename = str(tmp_path / "memory.tif") if backend == "memory" else "directory.tif"
        values = np.arange(48, dtype=np.float32).reshape(6, 8)
        eager = gu.DataArrayRasterAccessor.from_array(values, from_origin(0, 6, 1, 1), 4326)
        # Check shorter final chunks
        lazy = eager.chunk({"y": 4, "x": 3})
        source = lazy.to_dataset(name="elevation") if as_dataset else lazy
        assert eager.rst.is_loaded and not lazy.rst.is_loaded

        write = source.rst.to_file(filename, filesystem=filesystem, compute=False)
        assert not lazy.rst.is_loaded
        write.compute(scheduler="threads" if backend == "memory" else "processes", num_workers=2)

        with gu.open_raster(filename, filesystem=filesystem) as result:
            np.testing.assert_array_equal(result.values, eager.values)
            assert result.rst.crs == eager.rst.crs
        assert not lazy.rst.is_loaded


class TestRasterIOErrors:
    """Test module for invalid opening options, missing file sources and incompatible storage encoding."""

    landsat_b4_path = examples.get_path_test("everest_landsat_b4")

    @pytest.mark.parametrize(
        "downsample",
        [
            {},
            {"factor": 2, "shape": (2, 3)},
            {"factor": 2, "unknown": True},
            {"factor": 2, "preserve": "bounds"},
            {"factor": 2, "preserve": "resolution"},
            {"factor": 0},
            {"factor": np.inf},
            {"res": 0},
            {"res": np.nan},
            {"res": 0.5},
            {"res": 2, "preserve": "bounds"},
            {"shape": (2, 3), "preserve": "bounds"},
            {"shape": (2, 3), "preserve": "resolution"},
            {"shape": (0, 3)},
            {"shape": (2.5, 3)},
            {"shape": (True, 3)},
            {"shape": (1000, 1000)},
        ],
    )
    def test_open_raster__error_downsample_options(self, tmp_path: Path, downsample: Any) -> None:
        """Checks an error is raised for invalid or conflicting downsampling options."""

        filename = tmp_path / "options.tif"
        gu.Raster.from_array(np.ones((6, 8)), from_origin(0, 6, 1, 1), 32631).to_file(filename)
        with pytest.raises(ValueError):
            open_raster(str(filename), downsample=downsample)
        with pytest.raises(ValueError):
            gu.Raster(filename, downsample=downsample)

    @pytest.mark.parametrize("as_object_type", ["geodataframe", "scene"])
    def test_open_raster__error_invalid_type(self, tmp_path: Path, as_object_type: str) -> None:
        """Checks an error is raised before reading a file for an unsupported raster return type."""

        path = tmp_path / "absent.tif"
        with pytest.raises(ValueError, match="as_object_type"):
            open_raster(str(path), as_object_type=as_object_type)  # type: ignore[arg-type]

    @pytest.mark.parametrize("setter", ["property", "method"])
    def test_nodata__error_incompatible_storage_dtype(self, setter: str) -> None:
        """Checks an error is raised for nodata outside the storage dtype."""

        # Create raster with byte encoding
        array = gu.DataArrayRasterAccessor.from_array(np.ones((2, 2)), from_origin(0, 2, 1, 1), 32631)
        array.rio.write_nodata(255, encoded=True, inplace=True)
        array.encoding["dtype"] = "uint8"
        original = array.copy(deep=True)

        # Try each setter and check metadata is unchanged
        with pytest.raises(ValueError, match="Nodata value.*incompatible"):
            if setter == "property":
                array.rst.nodata = -1
            else:
                array.rst.set_nodata(-1)
        xr.testing.assert_identical(array, original)
        assert array.encoding == original.encoding

    @pytest.mark.parametrize("downsample", [2, {"factor": 2}, {"res": 2}, {"shape": (4, 4)}])
    @pytest.mark.parametrize("overview_level", [-1, 0])
    def test_init__error_overview_with_downsample(self, downsample: Any, overview_level: int) -> None:
        """Checks an error is raised for using overview_level + downsampling at once."""

        with pytest.raises(ValueError, match="downsample and overview_level cannot be used together"):
            gu.Raster(self.landsat_b4_path, downsample=downsample, overview_level=overview_level)

    def test_init__error_gcp_rcp_downsample(self, raster_gcp_rpc: gu.Raster) -> None:
        """Checks an error is raised for opening downsampling before GCP/RPC orthorectification."""
        with pytest.raises(NotImplementedError, match="before opening with downsample"):
            gu.Raster(raster_gcp_rpc.name, downsample=2)
        with pytest.raises(NotImplementedError, match="before opening with downsample"):
            gu.open_raster(raster_gcp_rpc.name, downsample=2)

    def test_load_only_mask__error_loaded_or_missing_name(self) -> None:
        """Checks an error is raised when mask-only reading has no unloaded file to read."""

        # A loaded raster already has its mask in memory
        loaded = gu.Raster(self.landsat_b4_path, load_data=True)
        with pytest.raises(ValueError, match="Data are already loaded"):
            loaded._load_only_mask()

        # Removing the filename leaves no source for a mask-only read
        unnamed = gu.Raster(self.landsat_b4_path)
        unnamed._name = None
        with pytest.raises(AttributeError, match="Cannot load as name is not set anymore"):
            unnamed._load_only_mask()

    @pytest.mark.parametrize("different", ["dtype", "scale_factor", "add_offset", "_FillValue", "unpacked"])
    def test_to_file__error_different_encoding(self, tmp_path: Path, different: str) -> None:
        """Checks an error is raised before writing variables with incompatible storage settings."""

        # Create variables with different storage settings
        first = gu.DataArrayRasterAccessor.from_array(np.ones((2, 3)), from_origin(0, 2, 1, 1), 32631)
        first.encoding.update(dtype="int16", scale_factor=0.1, add_offset=100, _FillValue=-32768)
        second = first.copy()
        if different == "unpacked":
            second.encoding = {}
        else:
            second.encoding[different] = "int32" if different == "dtype" else 2

        # Attempt the write and check that no output file was created
        filename = tmp_path / "mixed.tif"
        with pytest.raises(ValueError, match="share storage encoding"):
            xr.Dataset({"first": first, "second": second}).rst.to_file(filename)
        assert not filename.exists()
