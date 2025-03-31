"""
Tests for multiprocessing functions
"""

import os
from typing import Any

import numpy as np
import pytest
import scipy
from numpy import floating

from geoutils import Raster, examples
from geoutils.raster import RasterType
from geoutils.raster.distributed_computing import (
    ClusterGenerator,
    MultiprocConfig,
    apply_func_block,
    load_raster_tile,
    map_multiproc_collect,
    map_overlap_multiproc_save,
    remove_tile_padding,
)


# Define a simple function where overlap is needed
def _custom_func_overlap(raster: RasterType, size: int) -> RasterType:
    new_data = scipy.ndimage.maximum_filter(raster.data, size)
    if raster.nodata is not None:
        new_data = np.ma.masked_equal(new_data, raster.nodata)
    raster.data = new_data
    return raster


# Define a simple function with some args
def _custom_func(raster: RasterType, addition: float, factor: float) -> RasterType:
    return (raster + addition) * factor


# Define a simple function which do not return a Raster
def _custom_func_stats(raster: RasterType) -> dict[str, floating[Any]]:
    return raster.get_stats(stats_name=["mean", "valid_count"])


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
        tile = np.array([50, 125, 100, 200])  # [rowmin, rowmax, colmin, colmax]

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
        remove_tile_padding((raster.height, raster.width), raster_tile_with_padding, tile, padding)
        assert raster_tile_with_padding.raster_equal(raster_tile)

    @pytest.mark.parametrize("example", [aster_dem_path, landsat_rgb_path])  # type: ignore
    @pytest.mark.parametrize("padding", [0, 1, 10])  # type: ignore
    def test_apply_func_block(self, example, padding):
        """
        Test applying a function to a raster tile and handling padding removal.
        """
        raster = Raster(example)
        tile = np.array([100, 200, 100, 200])  # [rowmin, rowmax, colmin, colmax]
        size = 2

        # Apply map_block
        result_tile, _ = apply_func_block(_custom_func_overlap, raster, tile, padding, size)

        raster = _custom_func_overlap(raster, size)
        # If padding >=1, The result should be the equal to the original tile filtered
        original_tile_filtered = load_raster_tile(raster, tile)
        if padding >= size - 1:
            assert result_tile.raster_equal(original_tile_filtered)
        else:
            assert not result_tile.raster_equal(original_tile_filtered)

    @pytest.mark.parametrize("example", [aster_dem_path, landsat_rgb_path])  # type: ignore
    @pytest.mark.parametrize("tile_size", [100, 200])  # type: ignore
    @pytest.mark.parametrize("cluster", [None, ClusterGenerator("multi", 4)])
    def test_map_overlap_multiproc_save(self, example, tile_size, cluster):
        """
        Test the multiprocessing map function with a simple operation returning a raster.
        """
        raster = Raster(example)
        output_file = "output.tif"
        depth = 10
        config = MultiprocConfig(tile_size, output_file, cluster)

        addition = 5
        factor = 0.5
        # Apply the multiproc map function
        map_overlap_multiproc_save(_custom_func, raster, config, addition, factor, depth=depth)

        # Ensure raster has not been loading during process
        assert not raster.is_loaded

        # Ensure the output file is created and valid
        assert os.path.exists(output_file)

        # Open the output file and compare with the operation on full raster
        output_raster = Raster(output_file)
        new_raster = _custom_func(raster, addition, factor)
        assert output_raster.raster_equal(new_raster)

        # Remove output file
        os.remove(output_file)

    @pytest.mark.parametrize("example", [aster_dem_path, landsat_rgb_path])  # type: ignore
    @pytest.mark.parametrize("tile_size", [100, 200])  # type: ignore
    @pytest.mark.parametrize("cluster", [None, ClusterGenerator("multi", 4)])
    @pytest.mark.parametrize("return_tile", [False, True])
    def test_map_multiproc_collect(self, example, tile_size, cluster, return_tile):
        """
        Test the multiprocessing map function with a simple operation returning not a raster.
        """
        raster = Raster(example)
        config = MultiprocConfig(tile_size, cluster=cluster)

        # Apply the multiproc map function
        results = map_multiproc_collect(_custom_func_stats, raster, config, return_tile=return_tile)
        if return_tile:
            list_stats = [result[0] for result in results]
        else:
            list_stats = results

        # Ensure raster has not been loading during process
        assert not raster.is_loaded

        # Compare tiled_stats with the stats on full raster
        total_stats = _custom_func_stats(raster)

        tiled_count = sum([stats["valid_count"] for stats in list_stats])
        tiled_mean = sum([stats["mean"] * stats["valid_count"] for stats in list_stats]) / tiled_count
        assert abs(total_stats["mean"] - tiled_mean) < tiled_mean * 1e-5
        assert total_stats["valid_count"] == tiled_count
