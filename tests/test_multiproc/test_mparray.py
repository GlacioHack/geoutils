"""
Tests for multiprocessing functions
"""

from multiprocessing import cpu_count

import numpy as np
import pytest
<<<<<<<< HEAD:tests/test_multiproc/test_mparray.py
import scipy
from numpy import floating

import geoutils as gu
from geoutils import Raster, examples
from geoutils.multiproc.cluster import (
    AbstractCluster,
    ClusterGenerator,
)
from geoutils.multiproc.mparray import (
    MultiprocConfig,
    _apply_func_block,
    _generate_tiling_grid,
    _load_raster_tile,
    _remove_tile_padding,
    map_multiproc_collect,
    map_overlap_multiproc_save,
)
from geoutils.raster import RasterType
========
import rasterio as rio

import geoutils as gu
from geoutils import Raster, examples
from geoutils.multiproc import AbstractCluster, ClusterGenerator, MultiprocConfig
>>>>>>>> 0027c89 (Improve typing, input checks and import management for package scalability (#855)):tests/test_raster/test_distributing_computing/test_multiproc.py


class TestMultiProc:

<<<<<<<< HEAD:tests/test_multiproc/test_mparray.py

# Define a simple function with some args
def _custom_func(raster: Raster, addition: float, factor: float) -> Raster:
    return (raster + addition) * factor


# Define a simple function which do not return a Raster
def _custom_func_stats(raster: RasterType) -> dict[str, floating[Any]]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Empty raster")
        return raster.get_stats(band=1, stats_name=["mean", "valid_count"])


# Define a simple function which return a Mask
def _custom_func_mask(raster: RasterType) -> gu.Raster:
    mask_array = raster.get_mask()
    return gu.Raster.from_array(mask_array, raster.transform, raster.crs)


class TestTiling:

    landsat_b4_path = examples.get_path_test("everest_landsat_b4")

    @pytest.mark.parametrize("overlap", [0, 5])
    def test_tiling(self, overlap: int) -> None:

        # Test with mock data
        tiling_grid_mock = _generate_tiling_grid(0, 0, 100, 100, 50, 50, overlap)
        if overlap == 0:
            expected_tiling = np.array([[[0, 50, 0, 50], [0, 50, 50, 100]], [[50, 100, 0, 50], [50, 100, 50, 100]]])
            assert np.array_equal(tiling_grid_mock, expected_tiling)
        elif overlap == 5:
            expected_tiling = np.array(
                [
                    [[0, 55, 0, 55], [0, 55, 45, 100]],
                    [[45, 100, 0, 55], [45, 100, 45, 100]],
                ]
            )
            assert np.array_equal(tiling_grid_mock, expected_tiling)

        tiling_grid_mock = _generate_tiling_grid(0, 0, 55, 55, 50, 50, overlap)
        if overlap == 0:
            expected_tiling = np.array([[[0, 50, 0, 50], [0, 50, 50, 55]], [[50, 55, 0, 50], [50, 55, 50, 55]]])
            assert np.array_equal(tiling_grid_mock, expected_tiling)
        elif overlap == 5:
            expected_tiling = np.array([[[0, 55, 0, 55]]])
            assert np.array_equal(tiling_grid_mock, expected_tiling)

        # Test with real data
        img = gu.Raster(self.landsat_b4_path)

        # Define tiling parameters
        row_split, col_split = 100, 100
        row_max, col_max = img.shape

        # Generate the tiling grid
        tiling_grid = _generate_tiling_grid(0, 0, row_max, col_max, row_split, col_split, overlap)

        # Calculate expected number of tiles
        nb_row_tiles = np.ceil(row_max / row_split).astype(int)
        nb_col_tiles = np.ceil(col_max / col_split).astype(int)

        if 0 < col_max % col_split <= overlap:
            nb_col_tiles = max(nb_col_tiles - 1, 1)
        if 0 < row_max % row_split <= overlap:
            nb_row_tiles = max(nb_row_tiles - 1, 1)

        # Check that the tiling grid has the expected shape
        assert tiling_grid.shape == (nb_row_tiles, nb_col_tiles, 4)

        # Check the boundaries of the first and last tile
        assert np.array_equal(
            tiling_grid[0, 0],
            np.array(
                [
                    0,
                    min(row_split + overlap, row_max),
                    0,
                    min(col_split + overlap, col_max),
                ]
            ),
        )
        assert np.array_equal(
            tiling_grid[-1, -1],
            np.array(
                [
                    (nb_row_tiles - 1) * row_split - overlap,
                    row_max,
                    (nb_col_tiles - 1) * col_split - overlap,
                    col_max,
                ]
            ),
        )

        # Check if overlap is consistent between tiles
        for row in range(nb_row_tiles - 1):
            assert tiling_grid[row + 1, 0, 0] == tiling_grid[row, 0, 1] - 2 * overlap

        for col in range(nb_col_tiles - 1):
            assert tiling_grid[0, col + 1, 2] == tiling_grid[0, col, 3] - 2 * overlap

    def test_tiling_overlap_errors(self) -> None:
        with pytest.raises(ValueError):
            _generate_tiling_grid(0, 0, 100, 100, 50, 50, -1)
        with pytest.raises(TypeError):
            _generate_tiling_grid(0, 0, 100, 100, 50, 50, 0.5)  # type: ignore


class TestMultiproc:
========
>>>>>>>> 0027c89 (Improve typing, input checks and import management for package scalability (#855)):tests/test_raster/test_distributing_computing/test_multiproc.py
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")
    landsat_rgb_path = examples.get_path_test("everest_landsat_rgb")
    num_workers = min(2, cpu_count())  # Safer limit for CI
    cluster = ClusterGenerator("test", nb_workers=num_workers)

<<<<<<<< HEAD:tests/test_multiproc/test_mparray.py
    @pytest.mark.parametrize("example", [aster_dem_path, landsat_rgb_path])
    def test_load_raster_tile(self, example: str) -> None:
        """
        Test loading a specific tile (spatial subset) from the raster.
        """
        raster = Raster(example)
        # Define a tile (bounding box) to load
        tile = np.array([50, 125, 100, 200])  # [rowmin, rowmax, colmin, colmax]

        # Load the tile and verify dimensions
        raster_tile = _load_raster_tile(raster, tile)
        assert np.array_equal(raster_tile.data, raster.data[..., tile[0] : tile[1], tile[2] : tile[3]])

    @pytest.mark.parametrize("example", [aster_dem_path, landsat_rgb_path])
    @pytest.mark.parametrize("padding", [0, 1, 10])
    def test_remove_tile_padding(self, example: str, padding: int) -> None:
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

    @pytest.mark.parametrize("example", [aster_dem_path, landsat_rgb_path])
    @pytest.mark.parametrize("padding", [0, 1, 3])
    def test_apply_func_block(self, example: str, padding: int) -> None:
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

    @pytest.mark.parametrize("example", [aster_dem_path, landsat_rgb_path])
    @pytest.mark.parametrize("tile_size", [100, 200])
    @pytest.mark.parametrize("cluster", [None, cluster])
    def test_map_overlap_multiproc_save(self, example: str, tile_size: int, cluster: None | AbstractCluster) -> None:
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

    @pytest.mark.parametrize("example", [aster_dem_path, landsat_rgb_path])
    @pytest.mark.parametrize("tile_size", [10, 20])
    @pytest.mark.parametrize("cluster", [None, cluster])
    @pytest.mark.parametrize("return_tile", [False, True])
    def test_map_multiproc_collect(
        self, example: str, tile_size: int, cluster: None | AbstractCluster, return_tile: bool
    ) -> None:
        """
        Test the multiprocessing map function with a simple operation returning not a raster.
        """
        raster = Raster(example)
        config = MultiprocConfig(tile_size, cluster=cluster)

        # Apply the multiproc map function
        results = map_multiproc_collect(_custom_func_stats, raster, config, return_tile=return_tile)  # type: ignore
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
========
    @pytest.mark.skip()
    @pytest.mark.parametrize("example", [aster_dem_path])
    @pytest.mark.parametrize("tile_size", [10, 20])
    @pytest.mark.parametrize("cluster", [None, cluster])
    def test_multiproc_reproject(self, example: str, tile_size: int, cluster: None | AbstractCluster) -> None:
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

    @pytest.mark.skip(reason="Not implemented yet")
    def test_map_overlap_multiproc_save_bigTiff(self) -> None:
        """
        Test the multiprocessing map function with a simple operation returning a raster > 4go (BigTIFF format)
        """
>>>>>>>> 0027c89 (Improve typing, input checks and import management for package scalability (#855)):tests/test_raster/test_distributing_computing/test_multiproc.py
