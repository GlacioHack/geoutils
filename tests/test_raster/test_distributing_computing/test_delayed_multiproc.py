"""Tests for multiprocessing functions."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio as rio

from geoutils import Raster, examples
from geoutils.raster import compute_tiling
from geoutils.raster.distributed_computing import (
    AbstractCluster,
    ClusterGenerator,
    multiproc_interp_points,
    multiproc_reproject,
)


class TestDelayedMultiproc:
    aster_dem_path = examples.get_path("exploradores_aster_dem")

    cluster = ClusterGenerator("multi", nb_workers=4)

    @pytest.mark.parametrize("example", [aster_dem_path])  # type: ignore
    @pytest.mark.parametrize("tile_size", [50, 150])  # type: ignore
    @pytest.mark.parametrize("cluster", [None, cluster])  # type: ignore
    def test_delayed_interp_points__output(self, example: str, tile_size: int, cluster: AbstractCluster) -> None:
        raster = Raster(example)

        tiling_grid = compute_tiling(tile_size, raster.shape, raster.shape)

        # Get raster dimensions and transformation to world coordinates
        rows, cols = raster.shape

        # Generate random points within the raster bounds (in real-world coordinates)
        num_points = 10  # Number of points to generate
        in_raster_i = np.random.default_rng(42).uniform(0, rows - 1, size=num_points)
        in_raster_j = np.random.default_rng(42).uniform(0, cols - 1, size=num_points)

        # Generate random points outside the raster bounds (in real-world coordinates)
        out_raster_i = np.random.default_rng(42).uniform(rows, rows + 100, size=num_points)
        out_raster_j = np.random.default_rng(42).uniform(cols, cols + 100, size=num_points)

        # Combine in-bounds and out-of-bounds points
        i_points = np.concatenate((in_raster_i, out_raster_i))
        j_points = np.concatenate((in_raster_j, out_raster_j))

        # convert points from pixel-base to crs coordinates
        x_points, y_points = raster.ij2xy(i_points, j_points)
        points = [(x, y) for (x, y) in zip(x_points, y_points)]

        tiles_interp_points = multiproc_interp_points(raster, tiling_grid, points, cluster=cluster)
        full_interp_points = raster.interp_points((x_points, y_points))

        assert len(tiles_interp_points) == num_points * 2
        assert np.sum(np.isnan(tiles_interp_points)) == num_points
        assert np.allclose(tiles_interp_points, full_interp_points, equal_nan=True, rtol=1e-3)

    @pytest.mark.parametrize("example", [aster_dem_path])  # type: ignore
    @pytest.mark.parametrize("tile_size", [200, 500])  # type: ignore
    @pytest.mark.parametrize("cluster", [None, cluster])  # type: ignore
    @pytest.mark.parametrize("overlap", [10, 50])  # type: ignore
    def test_delayed_reproject__output(
        self, example: str, tile_size: int, cluster: AbstractCluster, overlap: int
    ) -> None:
        r = Raster(example)
        tiling_grid = compute_tiling(tile_size, r.shape, r.shape, overlap=overlap)

        outfile = "test.tif"

        # - Test reprojection with bounds and resolution -
        dst_bounds = rio.coords.BoundingBox(
            left=r.bounds.left, bottom=r.bounds.bottom + r.res[0], right=r.bounds.right - 2 * r.res[1], top=r.bounds.top
        )
        res_tuple = (r.res[0] * 4, r.res[1] * 4)

        # Single-process reprojection
        r_single = r.reproject(bounds=dst_bounds, res=res_tuple)

        # Multiprocessing reprojection
        multiproc_reproject(r, tiling_grid, outfile, bounds=dst_bounds, res=res_tuple)
        r_multi = Raster(outfile)

        # Assert the results are the same
        assert r_single.raster_equal(r_multi)

        # - Test reprojection with CRS change -
        out_crs = rio.crs.CRS.from_epsg(4326)

        # Single-process reprojection
        r_single = r.reproject(crs=out_crs)

        # Multiprocessing reprojection
        multiproc_reproject(r, tiling_grid, outfile, crs=out_crs)
        r_multi = Raster(outfile)

        plt.figure()
        r_single.plot()
        plt.figure()
        r_multi.plot()
        plt.figure()
        diff = r_single - r_multi
        diff.plot()
        plt.show()

        # Assert the results are the same
        assert r_single.raster_equal(r_multi)

        # -- Additional Tests --

        # Check that multiproc version does not fail with different data types
        r_float = r.astype("float32")  # type: ignore
        r_single_float = r_float.reproject(crs=out_crs)
        multiproc_reproject(r, tiling_grid, outfile, crs=out_crs)
        r_multi_float = Raster(outfile)

        plt.figure()
        r_single_float.plot()
        plt.figure()
        r_multi_float.plot()
        plt.figure()
        diff = r_single_float - r_multi_float
        diff.plot()
        plt.show()

        assert r_single_float.raster_equal(r_multi_float)

        # Check that reprojection works for several bands in multiproc as well
        tiling_grid = compute_tiling(tile_size, (500, 500), (500, 500), overlap=overlap)
        for n in [2, 3, 4]:
            img1 = Raster.from_array(
                np.ones((n, 500, 500), dtype="uint8"), transform=rio.transform.from_origin(0, 500, 1, 1), crs=4326
            )
            img2 = Raster.from_array(
                np.ones((n, 500, 500), dtype="uint8"), transform=rio.transform.from_origin(50, 500, 1, 1), crs=4326
            )

            out_img_single = img2.reproject(img1)
            multiproc_reproject(img2, tiling_grid, outfile, ref=img1)
            out_img_multi = Raster(outfile)

            assert out_img_multi.count == n
            assert out_img_multi.shape == (500, 500)
            assert out_img_single.raster_equal(out_img_multi)

        os.remove(outfile)
