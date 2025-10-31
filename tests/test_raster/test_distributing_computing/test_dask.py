"""Tests for dask-delayed functions."""

from __future__ import annotations

import warnings

import dask.array as da
import numpy as np
import pytest
import rasterio as rio
import xarray as xr
from pyproj import CRS

from geoutils.raster.distributed_computing.dask import (
    delayed_interp_points,
    delayed_reproject,
    delayed_subsample,
)


def _build_dst_transform_shifted_newres(
    src_transform: rio.transform.Affine,
    src_shape: tuple[int, int],
    src_crs: CRS,
    dst_crs: CRS,
    bounds_rel_shift: tuple[float, float],
    res_rel_fac: tuple[float, float],
) -> rio.transform.Affine:
    """
    Build a destination transform intersecting the source transform given source/destination shapes,
    and possibly introducing a relative shift in upper-left bound and multiplicative change in resolution.
    """

    # Get bounding box in source CRS
    bounds = rio.coords.BoundingBox(*rio.transform.array_bounds(src_shape[0], src_shape[1], src_transform))

    # Construct an aligned transform in the destination CRS assuming the same resolution
    tmp_transform = rio.warp.calculate_default_transform(
        src_crs,
        dst_crs,
        src_shape[1],
        src_shape[0],
        left=bounds.left,
        right=bounds.right,
        top=bounds.top,
        bottom=bounds.bottom,
        dst_width=src_shape[1],
        dst_height=src_shape[0],
    )[0]
    # This allows us to get bounds and resolution in the units of the new CRS
    tmp_res = (tmp_transform[0], abs(tmp_transform[4]))
    tmp_bounds = rio.coords.BoundingBox(*rio.transform.array_bounds(src_shape[0], src_shape[1], tmp_transform))
    # Now we can create a shifted/different-res destination grid
    dst_transform = rio.transform.from_origin(
        west=tmp_bounds.left + bounds_rel_shift[0] * tmp_res[0] * src_shape[1],
        north=tmp_bounds.top + 150 * bounds_rel_shift[0] * tmp_res[1] * src_shape[0],
        xsize=tmp_res[0] / res_rel_fac[0],
        ysize=tmp_res[1] / res_rel_fac[1],
    )

    return dst_transform


class TestDelayed:
    """
    Testing class for delayed functions.

    We test on a first set of rasters big enough to clearly monitor the memory usage, and a second set small enough
    to run fast to check a wide range of input parameters.

    We compare outputs with the in-memory function specifically for input variables that influence the delayed
    algorithm and might lead to new errors (for example: array shape to get subsample/points locations for
    subsample and interp_points, or destination chunksizes to map output of reproject).
    """

    # Define random seed for generating test data
    rng = da.random.default_rng(seed=42)

    # Smaller test files for fast checks, with various shapes and with/without nodata
    list_small_shapes = [(51, 47)]
    with_nodata = [False, True]
    list_small_darr = []
    for small_shape in list_small_shapes:
        for w in with_nodata:
            small_darr = rng.normal(size=small_shape[0] * small_shape[1])
            # Add about half nodata values
            if w:
                ind_nodata = rng.choice(small_darr.size, size=int(small_darr.size / 2), replace=False)
                small_darr[list(ind_nodata)] = np.nan
            small_darr = small_darr.reshape(small_shape[0], small_shape[1])
            list_small_darr.append(small_darr)

    # List of in-memory chunksize for small tests
    list_small_chunksizes_in_mem = [(10, 10), (7, 19)]

    # Create a corresponding boolean array for each numerical dask array
    # Every finite numerical value (valid numerical value) corresponds to True (valid boolean value).
    darr_bool = []
    for small_darr in list_small_darr:
        darr_bool.append(da.where(da.isfinite(small_darr), True, False))

    @pytest.mark.parametrize("darr, darr_bool", list(zip(list_small_darr, darr_bool)))  # type: ignore
    @pytest.mark.parametrize("chunksizes_in_mem", list_small_chunksizes_in_mem)  # type: ignore
    @pytest.mark.parametrize("subsample_size", [2, 100, 100000])  # type: ignore
    def test_delayed_subsample__output(
        self, darr: da.Array, darr_bool: da.Array, chunksizes_in_mem: tuple[int, int], subsample_size: int
    ):
        """
        Checks for delayed subsampling function for output accuracy.
        Variables that influence specifically the delayed function and might lead to new errors are:
        - Input chunksizes,
        - Input array shape,
        - Number of subsampled points.
        """
        warnings.filterwarnings("ignore", category=UserWarning, message="Subsample value*")

        # 1/ We run the delayed function after re-chunking
        darr = darr.rechunk(chunksizes_in_mem)
        sub = delayed_subsample(darr, subsample=subsample_size, random_state=42)
        # 2/ Output checks

        # # The subsample should have exactly the prescribed length, with only valid values
        assert len(sub) == min(subsample_size, np.count_nonzero(np.isfinite(darr)))
        assert all(np.isfinite(sub))

        # To verify the sampling works correctly, we can get its subsample indices with the argument return_indices
        # And compare to the same subsample with vindex (now that we know the coordinates of valid values sampled)
        indices = delayed_subsample(darr, subsample=subsample_size, random_state=42, return_indices=True)
        sub2 = np.array(darr.vindex[indices[0], indices[1]])
        assert np.array_equal(sub, sub2)

        # Finally, to verify that a boolean array, with valid values at the same locations as the numerical array,
        # leads to the same results, we compare the samples values and the samples indices.
        darr_bool = darr_bool.rechunk(chunksizes_in_mem)
        indices_bool = delayed_subsample(darr_bool, subsample=subsample_size, random_state=42, return_indices=True)
        sub_bool = np.array(darr.vindex[indices_bool])
        assert np.array_equal(sub, sub_bool)
        assert np.array_equal(indices, indices_bool)

    @pytest.mark.parametrize("darr", list_small_darr)  # type: ignore
    @pytest.mark.parametrize("chunksizes_in_mem", list_small_chunksizes_in_mem)  # type: ignore
    @pytest.mark.parametrize("ninterp", [2, 100])  # type: ignore
    @pytest.mark.parametrize("res", [(0.5, 2), (1, 1)])  # type: ignore
    def test_delayed_interp_points__output(
        self, darr: da.Array, chunksizes_in_mem: tuple[int, int], ninterp: int, res: tuple[float, float]
    ):
        """
        Checks for delayed interpolate points function.
        Variables that influence specifically the delayed function are:
        - Input chunksizes,
        - Input array shape,
        - Number of interpolated points,
        - The resolution of the regular grid.
        """

        # 1/ Define points to interpolate given the size and resolution
        darr = darr.rechunk(chunksizes_in_mem)
        rng = np.random.default_rng(seed=42)
        interp_x = (rng.choice(darr.shape[0], ninterp) + rng.random(ninterp)) * res[0]
        interp_y = (rng.choice(darr.shape[1], ninterp) + rng.random(ninterp)) * res[1]

        interp1 = delayed_interp_points(darr, points=(interp_x, interp_y), resolution=res)  # type: ignore

        # 2/ Output checks

        # Interpolate directly with Xarray (loads a lot in memory) and check results are exactly the same
        xx = xr.DataArray(interp_x, dims="z", name="x")
        yy = xr.DataArray(interp_y, dims="z", name="y")
        ds = xr.DataArray(
            data=darr,
            dims=["x", "y"],
            coords={
                "x": np.arange(0, darr.shape[0] * res[0], res[0]),
                "y": np.arange(0, darr.shape[1] * res[1], res[1]),
            },
        )
        interp2 = ds.interp(x=xx, y=yy)
        interp2.compute()
        interp2 = np.array(interp2.values)

        assert np.array_equal(interp1, interp2, equal_nan=True)

    @pytest.mark.parametrize("darr", list_small_darr)  # type: ignore
    @pytest.mark.parametrize("chunksizes_in_mem", list_small_chunksizes_in_mem)  # type: ignore
    @pytest.mark.parametrize("dst_chunksizes", list_small_chunksizes_in_mem)  # type: ignore
    # Shift upper left corner of output bounds (relative to projected input bounds) by fractions of the raster size
    @pytest.mark.parametrize("dst_bounds_rel_shift", [(0, 0), (-0.2, 0.5)])  # type: ignore
    # Modify output resolution (relative to projected input resolution) by a factor
    @pytest.mark.parametrize("dst_res_rel_fac", [(1, 1), (2.1, 0.54)])  # type: ignore
    # Same for shape
    @pytest.mark.parametrize("dst_shape_diff", [(0, 0), (-28, 117)])  # type: ignore
    def test_delayed_reproject__output(
        self,
        darr: da.Array,
        chunksizes_in_mem: tuple[int, int],
        dst_chunksizes: tuple[int, int],
        dst_bounds_rel_shift: tuple[float, float],
        dst_res_rel_fac: tuple[float, float],
        dst_shape_diff: tuple[int, int],
    ):
        """
        Checks for the delayed reproject function.
        Variables that influence specifically the delayed function are:
        - Input/output chunksizes,
        - Input array shape,
        - Output geotransform relative to projected input geotransform,
        - Output array shape relative to input.
        """

        # Keeping this commented here if we need to redo local tests due to Rasterio errors
        # darr = list_small_darr[0]
        # chunksizes_in_mem = list_small_chunksizes_in_mem[0]
        # dst_chunksizes = list_small_chunksizes_in_mem[0]  # (2000, 2000)
        # dst_bounds_rel_shift = (0, 0)
        # dst_res_rel_fac = (0.45, 0.45)  # (1, 1)
        # dst_shape_diff = (0, 0)
        # cluster = LocalCluster(n_workers=1, threads_per_worker=1, dashboard_address=None)

        warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)

        # 0/ Define input parameters

        # Get input and output shape
        darr = darr.rechunk(chunksizes_in_mem)
        src_shape = darr.shape
        dst_shape = (src_shape[0] + dst_shape_diff[0], src_shape[1] + dst_shape_diff[1])

        # Define arbitrary input transform, as we only care about the relative difference with the output transform
        src_transform = rio.transform.from_bounds(10, 10, 15, 15, src_shape[0], src_shape[1])

        # Other arguments having (normally) no influence
        src_crs = CRS(4326)
        dst_crs = CRS(32630)
        src_nodata = -9999
        dst_nodata = 99999
        resampling = rio.enums.Resampling.bilinear

        # Get shifted dst_transform with new resolution
        dst_transform = _build_dst_transform_shifted_newres(
            src_transform=src_transform,
            src_crs=src_crs,
            dst_crs=dst_crs,
            src_shape=src_shape,
            bounds_rel_shift=dst_bounds_rel_shift,
            res_rel_fac=dst_res_rel_fac,
        )

        # 2/ Run delayed reproject with memory monitoring

        reproj_arr = delayed_reproject(
            darr,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_shape=dst_shape,
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
            resampling=resampling,
            dst_chunksizes=dst_chunksizes,
        )

        # 3/ Outputs check: load in memory and compare with a direct Rasterio reproject
        reproj_arr = np.array(reproj_arr)

        dst_arr = np.zeros(dst_shape)
        _ = rio.warp.reproject(
            np.array(darr),
            dst_arr,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
        )

        # Keeping this to visualize Rasterio resampling issue
        # if PLOT:
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow((reproj_arr - dst_arr), cmap="RdYlBu", vmin=-0.2, vmax=0.2, interpolation="None")
        # plt.colorbar()
        # plt.savefig("/home/atom/ongoing/diff_close_zero.png", dpi=500)
        # plt.figure()
        # plt.imshow(np.abs(reproj_arr - dst_arr), cmap="RdYlBu", vmin=99997, vmax=100001, interpolation="None")
        # plt.colorbar()
        # plt.savefig("/home/atom/ongoing/diff_nodata.png", dpi=500)
        # plt.figure()
        # plt.imshow(dst_arr, cmap="RdYlBu", vmin=-1, vmax=1, interpolation="None")
        # plt.colorbar()
        # plt.savefig("/home/atom/ongoing/dst.png", dpi=500)

        # Due to (what appears to be) Rasterio errors, we have to remain imprecise for the checks here:
        # even though some reprojections are pretty good, some can get a bit nasty

        # Check that little data (less than 10% of pixels) are significantly different
        ind_signif_diff = np.abs(reproj_arr - dst_arr) > 0.5
        assert np.count_nonzero(ind_signif_diff) < 0.1 * reproj_arr.size

        # The median difference should be negligible compared to the amplitude of the signal (+/- 1 std)
        assert np.nanmedian(np.abs(reproj_arr - dst_arr)) < 0.1

        # # Replace with allclose once Rasterio issue fixed? For some cases we get a good match
        # (less than 0.01 for all pixels)
        # assert np.allclose(reproj_arr[~ind_both_nodata], dst_arr[~ind_both_nodata], atol=0.02)
