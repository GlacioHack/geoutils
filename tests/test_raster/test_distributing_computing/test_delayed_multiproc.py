"""Tests for multiprocessing functions."""

import os
import re
import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio as rio

import geoutils as gu
from geoutils import Raster, examples
from geoutils.raster.distributed_computing import (
    AbstractCluster,
    ClusterGenerator,
    multiproc_interp_points,
    multiproc_reproject,
)
from geoutils.raster.georeferencing import _default_nodata


class TestDelayedMultiproc:
    landsat_b4_path = examples.get_path("everest_landsat_b4")
    landsat_b4_crop_path = examples.get_path("everest_landsat_b4_cropped")
    landsat_rgb_path = examples.get_path("everest_landsat_rgb")
    everest_outlines_path = examples.get_path("everest_rgi_outlines")
    aster_dem_path = examples.get_path("exploradores_aster_dem")
    aster_outlines_path = examples.get_path("exploradores_rgi_outlines")

    cluster = ClusterGenerator("multi", nb_workers=4)

    @pytest.mark.parametrize("example", [aster_dem_path])  # type: ignore
    @pytest.mark.parametrize("cluster", [None, cluster])  # type: ignore
    @pytest.mark.parametrize(
        "method", ["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"]
    )  # type: ignore
    def test_delayed_interp_points__output(
        self,
        example: str,
        method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"],
        cluster: AbstractCluster,
    ) -> None:
        raster = Raster(example)

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

        tiles_interp_points = multiproc_interp_points(raster, points, method=method, cluster=cluster)
        full_interp_points = raster.interp_points((x_points, y_points), method=method)

        assert len(tiles_interp_points) == num_points * 2
        assert np.sum(np.isnan(tiles_interp_points)) == num_points
        assert np.allclose(np.array(tiles_interp_points), full_interp_points, equal_nan=True, rtol=1e-3)

    @pytest.mark.parametrize("example", [aster_dem_path])  # type: ignore
    @pytest.mark.parametrize("tile_size", [200])  # type: ignore
    @pytest.mark.parametrize("cluster", [None])  # type: ignore
    @pytest.mark.parametrize("overlap", [5])  # type: ignore
    def test_delayed_reproject__output(
        self, example: str, tile_size: int, cluster: AbstractCluster, overlap: int
    ) -> None:
        DO_PLOT = False

        r = Raster(example)
        outfile = "test.tif"

        # # specific for the landsat test case, default nodata 255 cannot be used (see above), so use 0
        # if r.nodata is None:
        #     r.set_nodata(0)
        #
        # # - Test reprojection with bounds and resolution -
        # dst_bounds = rio.coords.BoundingBox(
        #   left=r.bounds.left, bottom=r.bounds.bottom + r.res[0], right=r.bounds.right - 2 * r.res[1], top=r.bounds.top
        # )
        # res_tuple = (r.res[0] * 0.5, r.res[1] * 3)
        #
        # # Single-process reprojection
        # r_single = r.reproject(bounds=dst_bounds, res=res_tuple)
        #
        # # Multiprocessing reprojection
        # multiproc_reproject(r, outfile, tile_size, bounds=dst_bounds, res=res_tuple)
        # r_multi = Raster(outfile)
        #
        # # Assert the results are the same
        # assert r_single.raster_equal(r_multi)
        #
        # # - Test reprojection with CRS change -
        # out_crs = rio.crs.CRS.from_epsg(4326)
        #
        # # Single-process reprojection
        # r_single = r.reproject(crs=out_crs)
        #
        # # Multiprocessing reprojection
        # multiproc_reproject(r, outfile, tile_size, crs=out_crs)
        # r_multi = Raster(outfile)
        #
        # plt.figure()
        # r_multi.plot()
        # plt.figure()
        # diff = r_single - r_multi
        # diff.plot()
        # plt.show()
        #
        # # Assert the results are the same
        # # assert r_single.raster_equal(r_multi, rtol=1e-3)
        #
        # # Check that reprojection works for several bands in multiproc as well
        # for n in [2, 3, 4]:
        #     img1 = Raster.from_array(
        #         np.ones((n, 500, 500), dtype="uint8"), transform=rio.transform.from_origin(0, 500, 1, 1), crs=4326
        #     )
        #     img2 = Raster.from_array(
        #         np.ones((n, 500, 500), dtype="uint8"), transform=rio.transform.from_origin(50, 500, 1, 1), crs=4326
        #     )
        #
        #     out_img_single = img2.reproject(img1)
        #     multiproc_reproject(img2, outfile, tile_size, ref=img1)
        #     out_img_multi = Raster(outfile)
        #
        #     assert out_img_multi.count == n
        #     assert out_img_multi.shape == (500, 500)
        #     assert out_img_single.raster_equal(out_img_multi)

        warnings.simplefilter("error")

        # -- Check proper errors are raised if nodata are not set -- #
        r_nodata = r.copy()
        r_nodata.set_nodata(None)

        # Make sure at least one pixel is masked for test 1
        rand_indices = gu.raster.subsample_array(r_nodata.data, 10, return_indices=True)
        r_nodata.data[rand_indices] = np.ma.masked
        assert np.count_nonzero(r_nodata.data.mask) > 0

        # make sure at least one pixel is set at default nodata for test
        default_nodata = _default_nodata(r_nodata.dtype)
        rand_indices = gu.raster.subsample_array(r_nodata.data, 10, return_indices=True)
        r_nodata.data[rand_indices] = default_nodata
        assert np.count_nonzero(r_nodata.data == default_nodata) > 0

        # 1 - if no force_source_nodata is set and masked values exist, raises an error
        with pytest.raises(
            ValueError,
            match=re.escape(
                "No nodata set, set one for the raster with self.set_nodata() or use a "
                "temporary one with `force_source_nodata`."
            ),
        ):
            _ = r_nodata.reproject(res=r_nodata.res[0] / 2, nodata=0)

        # 2 - if no nodata is set and default value conflicts with existing value, a warning is raised
        with pytest.warns(
            UserWarning,
            match=re.escape(
                f"For reprojection, nodata must be set. Default chosen value "
                f"{_default_nodata(r_nodata.dtype)} exists in self.data. This may have unexpected "
                f"consequences. Consider setting a different nodata with self.set_nodata()."
            ),
        ):
            r_test = r_nodata.reproject(res=r_nodata.res[0] / 2, force_source_nodata=default_nodata)
        assert r_test.nodata == default_nodata

        # 3 - if default nodata does not conflict, should not raise a warning
        r_nodata.data[r_nodata.data == default_nodata] = 3
        r_test = r_nodata.reproject(res=r_nodata.res[0] / 2, force_source_nodata=default_nodata)
        assert r_test.nodata == default_nodata

        # -- Test setting each combination of georeferences bounds, res and size -- #

        # specific for the landsat test case, default nodata 255 cannot be used (see above), so use 0
        if r.nodata is None:
            r.set_nodata(0)

        # - Test size - this should modify the shape, and hence resolution, but not the bounds -
        out_size = (r.shape[1] // 2, r.shape[0] // 2)  # Outsize is (ncol, nrow)
        multiproc_reproject(r, outfile, tile_size, grid_size=out_size)
        r_test = Raster(outfile)
        assert r_test.shape == (out_size[1], out_size[0])
        assert r_test.res != r.res
        assert r_test.bounds == r.bounds

        # - Test bounds -
        # if bounds is a multiple of res, outptut res should be preserved
        bounds = np.copy(r.bounds)
        dst_bounds = rio.coords.BoundingBox(
            left=bounds[0], bottom=bounds[1] + r.res[0], right=bounds[2] - 2 * r.res[1], top=bounds[3]
        )
        multiproc_reproject(r, outfile, tile_size, bounds=dst_bounds)
        r_test = Raster(outfile)
        assert r_test.bounds == dst_bounds
        assert r_test.res == r.res

        # Create bounds with 1/2 and 1/3 pixel extra on the right/bottom.
        bounds = np.copy(r.bounds)
        dst_bounds = rio.coords.BoundingBox(
            left=bounds[0], bottom=bounds[1] - r.res[0] / 3.0, right=bounds[2] + r.res[1] / 2.0, top=bounds[3]
        )

        # If bounds are not a multiple of res, the latter will be updated accordingly
        multiproc_reproject(r, outfile, tile_size, bounds=dst_bounds)
        r_test = Raster(outfile)
        assert r_test.bounds == dst_bounds
        assert r_test.res != r.res

        # - Test size and bounds -
        multiproc_reproject(r, outfile, tile_size, grid_size=out_size, bounds=dst_bounds)
        r_test = Raster(outfile)
        assert r_test.shape == (out_size[1], out_size[0])
        assert r_test.bounds == dst_bounds

        # - Test res -
        # Using a single value, output res will be enforced, resolution will be different
        res_single = r.res[0] * 2
        multiproc_reproject(r, outfile, tile_size, res=res_single)
        r_test = Raster(outfile)
        assert r_test.res == (res_single, res_single)
        assert r_test.shape != r.shape

        # Using a tuple
        res_tuple = (r.res[0] * 0.5, r.res[1] * 4)
        multiproc_reproject(r, outfile, tile_size, res=res_tuple)
        r_test = Raster(outfile)
        assert r_test.res == res_tuple
        assert r_test.shape != r.shape

        # - Test res and bounds -
        # Bounds will be enforced for upper-left pixel, but adjusted by up to one pixel for the lower right bound.
        # for single res value
        multiproc_reproject(r, outfile, tile_size, bounds=dst_bounds, res=res_single)
        r_test = Raster(outfile)
        assert r_test.res == (res_single, res_single)
        assert r_test.bounds.left == dst_bounds.left
        assert r_test.bounds.top == dst_bounds.top
        assert np.abs(r_test.bounds.right - dst_bounds.right) < res_single
        assert np.abs(r_test.bounds.bottom - dst_bounds.bottom) < res_single

        # For tuple
        multiproc_reproject(r, outfile, tile_size, bounds=dst_bounds, res=res_tuple)
        r_test = Raster(outfile)
        assert r_test.res == res_tuple
        assert r_test.bounds.left == dst_bounds.left
        assert r_test.bounds.top == dst_bounds.top
        assert np.abs(r_test.bounds.right - dst_bounds.right) < res_tuple[0]
        assert np.abs(r_test.bounds.bottom - dst_bounds.bottom) < res_tuple[1]

        # - Test crs -
        out_crs = rio.crs.CRS.from_epsg(4326)
        multiproc_reproject(r, outfile, tile_size, crs=out_crs)
        r_test = Raster(outfile)
        assert r_test.crs.to_epsg() == 4326

        # -- Additional tests --
        # First, make sure dst_bounds extend beyond current extent to create nodata
        dst_bounds = rio.coords.BoundingBox(
            left=bounds[0], bottom=bounds[1] - r.res[0], right=bounds[2] + 2 * r.res[1], top=bounds[3]
        )
        multiproc_reproject(r, outfile, tile_size, bounds=dst_bounds)
        r_test = Raster(outfile)
        assert np.count_nonzero(r_test.data.mask) > 0

        # If nodata falls outside the original image range, check range is preserved (with nearest interpolation)
        r_float = r.astype("float32")  # type: ignore
        if r_float.nodata is not None:
            if (r_float.nodata < np.min(r_float.data)) or (r_float.nodata > np.max(r_float.data)):
                multiproc_reproject(r_float, outfile, tile_size, bounds=dst_bounds, resampling="nearest")
                r_test = Raster(outfile)
                assert r_test.nodata == r_float.nodata
                assert np.count_nonzero(r_test.data.data == r_test.nodata) > 0  # Some values should be set to nodata
                assert np.min(r_test.data) == np.min(r_float.data)  # But min and max should not be affected
                assert np.max(r_test.data) == np.max(r_float.data)

        # Check that nodata works as expected
        multiproc_reproject(r_float, outfile, tile_size, bounds=dst_bounds, nodata=9999)
        r_test = Raster(outfile)
        assert r_test.nodata == 9999
        assert np.count_nonzero(r_test.data.data == r_test.nodata) > 0

        # Test that reprojection works for several bands
        for n in [2, 3, 4]:
            img1 = gu.Raster.from_array(
                np.ones((n, 500, 500), dtype="uint8"), transform=rio.transform.from_origin(0, 500, 1, 1), crs=4326
            )

            img2 = gu.Raster.from_array(
                np.ones((n, 500, 500), dtype="uint8"), transform=rio.transform.from_origin(50, 500, 1, 1), crs=4326
            )

            multiproc_reproject(img2, outfile, tile_size, ref=img1)
            out_img = Raster(outfile)
            assert (out_img.count, *out_img.shape) == (n, 500, 500)

        # -- Test match reference functionalities --

        # - Create 2 artificial rasters -
        # for r2b, bounds are cropped to the upper left by an integer number of pixels (i.e. crop)
        # for r2, resolution is also set to 2/3 the input res
        min_size = min(r.shape)
        rng = np.random.default_rng(42)
        rand_int = rng.integers(int(min_size / 10), int(min(r.shape) - min_size / 10))
        new_transform = rio.transform.from_origin(
            r.bounds.left + rand_int * r.res[0], r.bounds.top - rand_int * abs(r.res[1]), r.res[0], r.res[1]
        )

        # data is cropped to the same extent
        new_data = r.data[rand_int::, rand_int::]
        r2b = gu.Raster.from_array(data=new_data, transform=new_transform, crs=r.crs, nodata=r.nodata)

        # Create a raster with different resolution
        dst_res = r.res[0] * 2 / 3
        multiproc_reproject(r2b, outfile, tile_size, res=dst_res)
        r2 = Raster(outfile)
        assert r2.res == (dst_res, dst_res)

        # Assert the initial rasters are different
        assert r.bounds != r2b.bounds
        assert r.shape != r2b.shape
        assert r.bounds != r2.bounds
        assert r.shape != r2.shape
        assert r.res != r2.res

        # Test reprojecting with ref=r2b (i.e. crop) -> output should have same shape, bounds and data, i.e. be the
        # same object
        multiproc_reproject(r, outfile, tile_size, ref=r2b)
        r3 = Raster(outfile)

        assert r3.raster_equal(r2b)

        if DO_PLOT:
            fig1, ax1 = plt.subplots()
            r.plot(ax=ax1, title="Raster 1")

            fig2, ax2 = plt.subplots()
            r2b.plot(ax=ax2, title="Raster 2")

            fig3, ax3 = plt.subplots()
            r3.plot(ax=ax3, title="Raster 1 reprojected to Raster 2")

            plt.show()

        # Test reprojecting with ref=r2 -> output should have same shape, bounds and transform
        # Data should be slightly different due to difference in input resolution
        multiproc_reproject(r, outfile, tile_size, ref=r2)
        r3 = Raster(outfile)

        assert r3.bounds == r2.bounds
        assert r3.shape == r2.shape
        assert r3.bounds == r2.bounds
        assert r3.transform == r2.transform
        assert not np.array_equal(r3.data.data, r2.data.data, equal_nan=True)

        if DO_PLOT:
            fig1, ax1 = plt.subplots()
            r.plot(ax=ax1, title="Raster 1")

            fig2, ax2 = plt.subplots()
            r2.plot(ax=ax2, title="Raster 2")

            fig3, ax3 = plt.subplots()
            r3.plot(ax=ax3, title="Raster 1 reprojected to Raster 2")

            plt.show()

        # -- Check that if mask is modified afterward, it is taken into account during reproject -- #
        # Create a raster with (additional) random gaps
        r_gaps = r.copy()
        nsamples = 200
        rand_indices = gu.raster.subsample_array(r_gaps.data, nsamples, return_indices=True)
        r_gaps.data[rand_indices] = np.ma.masked
        assert np.sum(r_gaps.data.mask) - np.sum(r.data.mask) == nsamples  # sanity check

        # reproject raster, and reproject mask. Check that both have same number of masked pixels
        # TODO: should test other resampling algo
        multiproc_reproject(r, outfile, tile_size, res=dst_res, resampling="nearest")
        r_gaps_reproj = Raster(outfile)
        mask = gu.Raster.from_array(
            r_gaps.data.mask.astype("uint8"), crs=r_gaps.crs, transform=r_gaps.transform, nodata=None
        )
        multiproc_reproject(mask, outfile, tile_size, res=dst_res, nodata=255, resampling="nearest")
        mask_reproj = Raster(outfile)
        # Final masked pixels are those originally masked (=1) and the values masked during reproject, e.g. edges
        tot_masked_true = np.count_nonzero(mask_reproj.data.mask) + np.count_nonzero(mask_reproj.data == 1)
        assert np.count_nonzero(r_gaps_reproj.data.mask) == tot_masked_true

        # If a nodata is set, make sure it is preserved
        r_nodata = r.copy()

        r_nodata.set_nodata(0)

        r3 = r_nodata.reproject(r2)
        assert r_nodata.nodata == r3.nodata

        os.remove(outfile)
