"""
Tests for multiprocessing functions
"""

import os
from multiprocessing import cpu_count
from typing import Any

import numpy as np
import pytest
import rasterio as rio
import scipy
from numpy import floating

import geoutils as gu
from geoutils import Raster, examples
from geoutils.raster import RasterType
from geoutils.raster.distributed_computing import (
    ClusterGenerator,
    MultiprocConfig,
    map_multiproc_collect,
    map_overlap_multiproc_save,
)
from geoutils.raster.distributed_computing.multiproc import (
    _apply_func_block,
    _load_raster_tile,
    _remove_tile_padding,
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


# Define a simple function which return a Mask
def _custom_func_mask(raster: RasterType) -> gu.Raster:
    mask_array = raster.get_mask()
    return gu.Raster.from_array(mask_array, raster.transform, raster.crs)


class TestMultiproc:
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")
    landsat_rgb_path = examples.get_path_test("everest_landsat_rgb")

    num_workers = min(2, cpu_count())  # Safer limit for CI
    cluster = ClusterGenerator("test", nb_workers=num_workers)

    @pytest.mark.parametrize("example", [aster_dem_path, landsat_rgb_path])  # type: ignore
    def test_load_raster_tile(self, example) -> None:
        """
        Test loading a specific tile (spatial subset) from the raster.
        """
        raster = Raster(example)
        # Define a tile (bounding box) to load
        tile = np.array([50, 125, 100, 200])  # [rowmin, rowmax, colmin, colmax]

        # Load the tile and verify dimensions
        raster_tile = _load_raster_tile(raster, tile)
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

        raster_tile = _load_raster_tile(raster, tile)
        raster_tile_with_padding = _load_raster_tile(raster, tile_pad)

        # Remove padding and ensure it's back to the original size
        _remove_tile_padding((raster.height, raster.width), raster_tile_with_padding, tile, padding)
        assert raster_tile_with_padding.raster_equal(raster_tile)

    @pytest.mark.parametrize("example", [aster_dem_path, landsat_rgb_path])  # type: ignore
    @pytest.mark.parametrize("padding", [0, 1, 3])  # type: ignore
    def test_apply_func_block(self, example, padding):
        """
        Test applying a function to a raster tile and handling padding removal.
        """
        raster = Raster(example)
        tile = np.array([10, 20, 10, 20])  # [rowmin, rowmax, colmin, colmax]
        size = 2

        # Apply map_block
        result_tile, _ = _apply_func_block(_custom_func_overlap, raster, tile, padding, size)

        raster = _custom_func_overlap(raster, size)
        # If padding >=1, The result should be the equal to the original tile filtered
        original_tile_filtered = _load_raster_tile(raster, tile)
        if padding >= size - 1:
            assert result_tile.raster_equal(original_tile_filtered)
        else:
            assert not result_tile.raster_equal(original_tile_filtered)

    @pytest.mark.parametrize("example", [aster_dem_path, landsat_rgb_path])  # type: ignore
    @pytest.mark.parametrize("tile_size", [100, 200])  # type: ignore
    @pytest.mark.parametrize("cluster", [None, cluster])
    def test_map_overlap_multiproc_save(self, example, tile_size, cluster):
        """
        Test the multiprocessing map function with a simple operation returning a raster.
        """
        raster = Raster(example)
        output_file = "output.tif"
        depth = 10
        config = MultiprocConfig(tile_size, output_file, cluster=cluster)

        addition = 5
        factor = 0.5
        # Apply the multiproc map function
        output_raster = map_overlap_multiproc_save(_custom_func, raster, config, addition, factor, depth=depth)

        # Ensure raster has not been loading during process
        assert not raster.is_loaded

        # Ensure the output file is created and valid
        assert os.path.exists(output_file)

        # Compare with the operation on full raster
        new_raster = _custom_func(raster, addition, factor)
        assert output_raster.raster_equal(new_raster)

        # Assert raster has been properly saved in output_file
        output_raster_saved = Raster(output_file)
        assert output_raster_saved.raster_equal(output_raster)

        # Remove output file
        os.remove(output_file)

        # With a tempfile :
        config = MultiprocConfig(tile_size)
        output_raster = map_overlap_multiproc_save(_custom_func, raster, config, addition, factor, depth=depth)
        output_raster_saved = Raster(config.outfile)
        assert output_raster_saved.raster_equal(output_raster)

        if raster.count == 1:
            # With a wrapper returning a Mask
            output_mask = map_overlap_multiproc_save(_custom_func_mask, raster, config, depth=depth)
            assert np.array_equal(raster.get_mask(), output_mask.data)

    @pytest.mark.parametrize("example", [aster_dem_path, landsat_rgb_path])  # type: ignore
    @pytest.mark.parametrize("tile_size", [10, 20])  # type: ignore
    @pytest.mark.parametrize("cluster", [None, cluster])
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
            list_tiles = [result[1] for result in results]
            assert np.array_equal(list_tiles[0], np.array([0, tile_size, 0, tile_size]))
        else:
            list_stats = results

        # Ensure raster has not been loading during process
        assert not raster.is_loaded

        # Compare tiled_stats with the stats on full raster
        total_stats = _custom_func_stats(raster)

        tiled_count = np.nansum([stats["valid_count"] for stats in list_stats])
        tiled_mean = np.nansum([stats["mean"] * stats["valid_count"] for stats in list_stats]) / tiled_count
        assert abs(total_stats["mean"] - tiled_mean) < tiled_mean * 1e-5
        assert total_stats["valid_count"] == tiled_count

    @pytest.mark.skip()
    @pytest.mark.parametrize("example", [aster_dem_path])  # type: ignore
    @pytest.mark.parametrize("tile_size", [10, 20])  # type: ignore
    @pytest.mark.parametrize("cluster", [None, cluster])  # type: ignore
    def test_multiproc_reproject(self, example, tile_size, cluster):
        """Test for multiproc_reproject"""

        r = gu.Raster(example)
        config = MultiprocConfig(tile_size, cluster=cluster)

        # specific for the landsat test case, default nodata 255 cannot be used (see above), so use 0
        if r.nodata is None:
            r.set_nodata(0)

        # - Test reprojection with bounds and resolution -
        dst_bounds = rio.coords.BoundingBox(
            left=r.bounds.left, bottom=r.bounds.bottom + r.res[0], right=r.bounds.right - 2 * r.res[1], top=r.bounds.top
        )
        res_tuple = (r.res[0] * 0.5, r.res[1] * 3)

        # Multiprocessing reprojection
        r_multi = r.reproject(bounds=dst_bounds, res=res_tuple, multiproc_config=config)

        # Assert that the raster has not been loaded during reprojection
        assert not r.is_loaded

        # Single-process reprojection
        r_single = r.reproject(bounds=dst_bounds, res=res_tuple)

        # Assert the results are the same
        assert r_single.raster_equal(r_multi)

        # - Test reprojection with CRS change -
        for out_crs in [rio.crs.CRS.from_epsg(4326)]:

            # Single-process reprojection
            r_single = r.reproject(crs=out_crs)

            # Multiprocessing reprojection
            r_multi = r.reproject(crs=out_crs, multiproc_config=config)

            # Assert the results are the same
            assert r_single.raster_equal(r_multi)

        # Check that reprojection works for several bands in multiproc as well
        for n in [2, 3, 4]:
            img1 = Raster.from_array(
                np.ones((n, 500, 500), dtype="uint8"),
                transform=rio.transform.from_origin(0, 500, 1, 1),
                crs=4326,
                nodata=0,
            )
            img2 = Raster.from_array(
                np.ones((n, 500, 500), dtype="uint8"),
                transform=rio.transform.from_origin(50, 500, 1, 1),
                crs=4326,
                nodata=0,
            )

            out_img_single = img2.reproject(img1)
            out_img_multi = img2.reproject(ref=img1, multiproc_config=config)

            assert out_img_multi.count == n
            assert out_img_multi.shape == (500, 500)
            assert out_img_single.raster_equal(out_img_multi)
