"""
Tests for multiprocessing functions
"""

import os

import numpy as np
import pytest

from geoutils import Raster, examples
from geoutils.raster import RasterType
from geoutils.raster.distributed_computing import (
    ClusterGenerator,
    load_raster_tile,
    map_block,
    map_multiproc,
    remove_tile_padding,
)


# Define a simple function (copy the raster)
def _copy_func(r: RasterType, cast_nodata: bool) -> RasterType:
    return r.copy(cast_nodata=cast_nodata)


class TestMultiproc:
    aster_dem_path = examples.get_path("exploradores_aster_dem")
    landsat_rgb_path = examples.get_path("everest_landsat_rgb")

    @pytest.mark.parametrize("example", [aster_dem_path, landsat_rgb_path])  # type: ignore
    def test_load_raster_tile(self, example) -> None:
        """
        Test loading a specific tile (spatial subset) from the raster.
        """
        raster = Raster(example)
        # Define a tile (bounding box) to load
        tile = np.array([50, 125, 100, 200])  # [xmin, xmax, ymin, ymax]

        # Load the tile and verify dimensions
        raster_tile = load_raster_tile(raster, tile)
        assert np.array_equal(raster_tile.data, raster.data[..., tile[0] : tile[1], tile[2] : tile[3]])

    @pytest.mark.parametrize("example", [aster_dem_path, landsat_rgb_path])  # type: ignore
    @pytest.mark.parametrize("padding", [0, 1, 10])  # type: ignore
    def test_remove_tile_padding(self, example, padding) -> None:
        """
        Test removing padding from a raster tile after processing.
        """
        raster = Raster(example)
        # Extract a tile with padding
        tile = np.array([0, 100, 50, 150])
        tile_pad = tile + np.array([-1, 1, -1, 1]) * padding

        raster_tile = load_raster_tile(raster, tile)
        raster_tile_with_padding = load_raster_tile(raster, tile_pad)

        # Remove padding and ensure it's back to the original size
        remove_tile_padding(raster, raster_tile_with_padding, tile, padding)
        assert raster_tile_with_padding.raster_equal(raster_tile)

    @pytest.mark.parametrize("example", [aster_dem_path, landsat_rgb_path])  # type: ignore
    def test_map_block(self, example):
        """
        Test applying a function to a raster tile and handling padding removal.
        """
        raster = Raster(example)
        tile = np.array([100, 200, 100, 200])  # [xmin, xmax, ymin, ymax]
        padding = 10

        cast_nodata = True
        # Apply map_block
        result_tile, _ = map_block(_copy_func, raster, tile, padding, (cast_nodata,))

        # Check if the result is the same as the original tile
        original_tile = load_raster_tile(raster, tile)
        assert result_tile.raster_equal(original_tile)

    @pytest.mark.parametrize("example", [aster_dem_path, landsat_rgb_path])  # type: ignore
    @pytest.mark.parametrize("tile_size", [100, 200])  # type: ignore
    @pytest.mark.parametrize("cluster", [None, ClusterGenerator("multi", 4)])
    def test_map_multiproc(self, example, tile_size, cluster):
        """
        Test the multiprocessing map function with a simple copy operation.
        """
        raster = Raster(example)
        output_file = "output.tif"

        cast_nodata = True
        # Apply the multiproc map function
        map_multiproc(_copy_func, raster, tile_size, output_file, (cast_nodata,), depth=10, cluster=cluster)

        # Ensure raster has not been loading during process
        assert not raster.is_loaded

        # Ensure the output file is created and valid
        assert os.path.exists(output_file)

        # Open the output file and compare with the original raster
        output_raster = Raster(output_file)
        assert output_raster.raster_equal(raster)

        # Remove output file
        os.remove(output_file)
