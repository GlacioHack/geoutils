"""Tests for dask-delayed functions."""
import os
from tempfile import NamedTemporaryFile
from typing import Callable, Any

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import rasterio as rio
import xarray as xr
from pyproj import CRS
from dask.distributed import Client, LocalCluster

from geoutils.examples import _EXAMPLES_DIRECTORY
from geoutils.raster.delayed import (
    delayed_interp_points,
    delayed_reproject,
    delayed_subsample,
)

from pluggy import PluggyTeardownRaisedWarning
from dask_memusage import install

# Ignore teardown warning given by Dask when closing the local cluster (due to dask-memusage plugin)
pytestmark = pytest.mark.filterwarnings("ignore", category=PluggyTeardownRaisedWarning)

# Fixture to use a single cluster for the entire module
@pytest.fixture(scope='module')
def cluster():
    # Need cluster to be single-threaded to use dask-memusage confidently
    dask_cluster = LocalCluster(n_workers=2, threads_per_worker=1, dashboard_address=None)
    yield dask_cluster
    dask_cluster.close()

def _run_dask_measuring_memusage(cluster, dask_func: Callable, *args_dask_func: Any, **kwargs_dask_func: Any) -> tuple[Any, float]:
    """Run a dask function monitoring its memory usage."""

    # Create a name temporary file that won't delete immediatley
    fn_tmp_csv = NamedTemporaryFile(suffix=".csv", delete=False).name

    # Setup cluster and client within context managers for a clean shutdown
    install(cluster.scheduler, fn_tmp_csv)
    with Client(cluster) as client:
        outputs = dask_func(*args_dask_func, **kwargs_dask_func)

    # Read memusage file and cleanup
    df = pd.read_csv(fn_tmp_csv)
    os.remove(fn_tmp_csv)

    # Keep only non-zero memory usage
    ind_nonzero = df.max_memory_mb != 0

    # Compute peak additional memory usage from min baseline
    memusage_mb = np.max(df.max_memory_mb[ind_nonzero]) - np.min(df.max_memory_mb[ind_nonzero])

    return outputs, memusage_mb

def _estimate_subsample_memusage(darr: da.Array, chunksizes_in_mem: tuple[int, int], subsample_size: int) -> float:
    """
    Estimate the theoretical memory usage of the delayed subsampling method.
    (we don't need to be super precise, just within a factor of ~2 to check memory usage performs as expected)
    """

    # TOTAL SIZE = Single chunk operations + Subsample indexes + Metadata passed to dask + Outputs

    # On top of the rest is added the Dask graph, we will multiply by a factor of 2 to get a good safety margin
    fac_dask_margin = 2
    num_chunks = np.prod(darr.numblocks)

    # Single chunk operation = (data type bytes + boolean from np.isfinite) * chunksize **2
    chunk_memusage = (darr.dtype.itemsize + np.dtype("bool").itemsize) * np.prod(chunksizes_in_mem)

    # 1D index subsample size: integer type * subsample_size
    sample_memusage = np.dtype("int32").itemsize * subsample_size

    # Outputs: number of valid pixels + subsample
    valids_memusage = np.dtype("int32").itemsize * num_chunks
    subout_memusage = np.dtype(darr.dtype).itemsize * subsample_size
    out_memusage = valids_memusage + subout_memusage

    # Size of metadata passed to dask: number of blocks times its content
    # Content of a metadata block = list (block size) of list (subsample size) of integer indexes
    size_index_int = 28  # Python size for int
    list_all_blocks = 64 + 8 * num_chunks  # A list is 64 + 8 bits per element, without memory of contained elements
    list_per_block = 64 * num_chunks + 8 * subsample_size + size_index_int * subsample_size  # Rough max estimate
    meta_memusage = list_per_block + list_all_blocks

    # Final estimate of memory usage of operation in MB
    max_op_memusage = fac_dask_margin * (chunk_memusage + sample_memusage + out_memusage + meta_memusage) / (2 ** 20)
    # We add a base memory usage of ~50 MB + 10MB per 1000 chunks (loaded in background by Dask even on tiny data)
    max_op_memusage += 50 + 10 * (num_chunks / 1000)

    return max_op_memusage

def _estimate_interp_points_memusage(darr: da.Array, chunksizes_in_mem: tuple[int, int], ninterp: int) -> float:
    """
    Estimate the theoretical memory usage of the delayed interpolation method.
    (we don't need to be super precise, just within a factor of ~2 to check memory usage performs as expected)
    """

    # TOTAL SIZE = Single chunk operations + Chunk overlap + Metadata passed to dask + Outputs

    # On top of the rest is added the Dask graph, we will multiply by a factor of 2 to get a good safety margin
    fac_dask_margin = 2
    num_chunks = np.prod(darr.numblocks)

    # Single chunk operation = (data type bytes + boolean from np.isfinite) * chunksize **2
    chunk_memusage = (darr.dtype.itemsize + np.dtype("bool").itemsize) * np.prod(chunksizes_in_mem)
    # For interpolation, chunks have to overlap and temporarily load each neighbouring chunk,
    # we add 8 neighbouring chunks
    chunk_memusage += darr.dtype.itemsize * np.prod(chunksizes_in_mem) * 8

    # Outputs: interpolate coordinates
    out_memusage = np.dtype(darr.dtype).itemsize * ninterp * 2

    # Size of metadata passed to dask: number of blocks times its content
    # Content of a metadata block = list (block size) of list (subsample size) of integer
    size_index_int = 28  # Python size for int
    size_index_float = 24  # Python size for float
    list_all_blocks = 64 + 8 * num_chunks  # A list is 64 + 8 bits per element, without memory of contained elements
    list_per_block = 64 * num_chunks + 8 * ninterp + size_index_int * ninterp  # Rough max estimate
    # And a list for each block of dict with 4 floats (xres, yres, xstart, ystart)
    dict_all_blocks = 64 * num_chunks + 4 * size_index_float * 64 * num_chunks
    meta_memusage = list_per_block + list_all_blocks + dict_all_blocks

    # Final estimate of memory usage of operation in MB
    max_op_memusage = fac_dask_margin * (chunk_memusage + out_memusage + meta_memusage) / (2 ** 20)
    # We add a base memory usage of ~50 MB + 10MB per 1000 chunks (loaded in background by Dask even on tiny data)
    max_op_memusage += 50 + 10 * (num_chunks / 1000)

    return max_op_memusage

def _estimate_reproject_memusage(darr: da.Array, chunksizes_in_mem: tuple[int, int], dst_chunksizes: tuple[int, int],
                                 rel_res_fac: tuple[float, float]) -> float:
    """
    Estimate the theoretical memory usage of the delayed reprojection method.
    (we don't need to be super precise, just within a factor of ~2 to check memory usage performs as expected)
    """

    # TOTAL SIZE = Combined source chunk operations + Building geopandas mapping + Metadata passed to dask + Outputs

    # On top of the rest is added the Dask graph, we will multiply by a factor of 2 to get a good safety margin
    fac_dask_margin = 2
    num_chunks = np.prod(darr.numblocks)

    # THE BIG QUESTION: how many maximum source chunks might be loaded for a single destination chunk?
    # It depends on the relative ratio of input chunksizes to destination chunksizes, accounting for resolution change
    x_rel_source_chunks = dst_chunksizes[0] / chunksizes_in_mem[0] * rel_res_fac[0]
    y_rel_source_chunks = dst_chunksizes[1] / chunksizes_in_mem[1] * rel_res_fac[1]
    # There is also some overlap needed for resampling and due to warping in different CRS, so let's multiply this by 8
    # (all neighbouring tiles)
    nb_source_chunks_per_dest = 8 * x_rel_source_chunks * y_rel_source_chunks

    # Combined memory usage of one chunk operation = squared array made from combined chunksize + original chunks
    total_nb = np.ceil(np.sqrt(nb_source_chunks_per_dest))**2 + nb_source_chunks_per_dest
    # We multiply the memory usage of a single chunk to the number of loaded/combined chunks
    chunk_memusage = darr.dtype.itemsize * np.prod(chunksizes_in_mem) * total_nb

    # Outputs: reprojected raster
    out_memusage = np.dtype(darr.dtype).itemsize * np.prod(dst_chunksizes)

    # Size of metadata passed to dask: number of blocks times its content
    # For each block, we pass a dict with 4 integers per source chunk (rxs, rxe, rys, rye)
    size_index_float = 24  # Python size for float
    size_index_int = 28  # Python size for float
    dict_all_blocks = (64 + 4 * size_index_int * nb_source_chunks_per_dest) * num_chunks
    # Passing the 2 CRS, 2 transforms, resampling methods and 2 nodatas
    combined_meta = (112 + 112 + 56 + 56 + 44 + 28 + 28) * size_index_float * num_chunks
    meta_memusage = combined_meta + dict_all_blocks

    # Final estimate of memory usage of operation in MB
    max_op_memusage = fac_dask_margin * (chunk_memusage + out_memusage + meta_memusage) / (2 ** 20)
    # We add a base memory usage of ~50 MB + 10MB per 1000 chunks (loaded in background by Dask even on tiny data)
    max_op_memusage += 50 + 10 * (num_chunks / 1000)

    return max_op_memusage


def _build_dst_transform_shifted_newres(src_transform: rio.transform.Affine,
                                        src_shape: tuple[int, int],
                                        src_crs: CRS,
                                        dst_crs: CRS,
                                        bounds_rel_shift: tuple[float, float],
                                        res_rel_fac: tuple[float, float]) -> rio.transform.Affine:
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
        ysize=tmp_res[1] / res_rel_fac[1])

    return dst_transform

class TestDelayed:
    """
    Testing delayed functions is pretty straightforward.

    We test on rasters big enough to clearly monitor the memory usage, but small enough to fit in-memory to check
    the function outputs against the ones from in-memory methods.

    In details:
    1. We compare with the in-memory function output only for the set of input variables that influence the delayed
        algorithm and might lead to new errors (for example: array shape to get subsample/points locations for
        subsample and interp_points, or destination chunksizes to map output of reproject).
    2. During execution, we capture memory usage and check that only the expected amount of memory
        (one or several chunk combinations + metadata) is indeed used during the compute() call.
     """

    # Write big test files on disk out-of-memory, with different input shapes not necessarily aligned between themselves
    # or with chunks
    fn_nc_shape = {"test_square.nc": (10000, 10000),
                   "test_complex.nc": (5511, 6768)}
    # We can use a constant value for storage chunks, it doesn't have any influence on the accuracy of delayed methods
    # (can change slightly RAM usage, but pretty stable as long as chunksizes in memory are larger and
    # significantly bigger)
    chunksizes_on_disk = (200, 200)
    list_fn = []
    for fn_basename in fn_nc_shape.keys():
        fn = os.path.join(_EXAMPLES_DIRECTORY, fn_basename)
        list_fn.append(fn)
        if not os.path.exists(fn):
            # Create random array in the right shape
            test_shape = fn_nc_shape[fn_basename]
            rng = da.random.default_rng(seed=42)
            data = rng.normal(size=test_shape[0] * test_shape[1]).reshape(test_shape[0], test_shape[1])
            data_arr = xr.DataArray(data=data, dims=["x", "y"])
            ds = xr.Dataset(data_vars={"test": data_arr})
            encoding_kwargs = {"test": {"chunksizes": chunksizes_on_disk}}
            writer = ds.to_netcdf(fn, encoding=encoding_kwargs, compute=False)
            writer.compute()

    @pytest.mark.parametrize("fn", list_fn)
    @pytest.mark.parametrize("chunksizes_in_mem", [(2000, 2000), (1241, 3221)])
    @pytest.mark.parametrize("subsample_size", [100, 100000])
    def test_delayed_subsample(self, fn: str, chunksizes_in_mem: tuple[int, int], subsample_size: int, cluster: Any):
        """
        Checks for delayed subsampling function, both for output and memory usage.
        Variables that influence specifically the delayed function are:
        - Input chunksizes,
        - Input array shape,
        - Number of subsampled points.
        """

        # 0/ Open dataset with chunks
        ds = xr.open_dataset(fn, chunks={"x": chunksizes_in_mem[0], "y": chunksizes_in_mem[1]})
        darr = ds["test"].data

        # 1/ Estimation of theoretical memory usage of the subsampling script

        max_op_memusage = _estimate_subsample_memusage(darr=darr, chunksizes_in_mem=chunksizes_in_mem,
                                                       subsample_size=subsample_size)

        # 2/ Run delayed subsample with dask memory usage monitoring

        # Derive subsample from delayed function
        # (passed to wrapper function to measure memory usage during execution)
        sub, measured_op_memusage = _run_dask_measuring_memusage(cluster, delayed_subsample, darr,
                                                                 subsample=subsample_size, random_state=42)

        # Check the measured memory usage is smaller than the maximum estimated one
        assert measured_op_memusage < max_op_memusage

        # 3/ Output checks

        # # The subsample should have exactly the prescribed length, with only valid values
        assert len(sub) == subsample_size
        assert all(np.isfinite(sub))

        # To verify the sampling works correctly, we can get its subsample indices with the argument return_indices
        # And compare to the same subsample with vindex (now that we know the coordinates of valid values sampled)
        indices = delayed_subsample(darr, subsample=subsample_size, random_state=42, return_indices=True)
        sub2 = np.array(darr.vindex[indices[0], indices[1]])
        assert np.array_equal(sub, sub2)

    @pytest.mark.parametrize("fn", list_fn)
    @pytest.mark.parametrize("chunksizes_in_mem", [(2000, 2000), (1241, 3221)])
    @pytest.mark.parametrize("ninterp", [100, 100000])
    def test_delayed_interp_points(self, fn: str, chunksizes_in_mem: tuple[int, int], ninterp: int, cluster: Any):
        """
        Checks for delayed interpolate points function.
        Variables that influence specifically the delayed function are:
        - Input chunksizes,
        - Input array shape,
        - Number of interpolated points.
        """

        # 0/ Open dataset with chunks and create random point locations to interpolate
        ds = xr.open_dataset(fn, chunks={"x": chunksizes_in_mem[0], "y": chunksizes_in_mem[1]})
        darr = ds["test"].data

        rng = np.random.default_rng(seed=42)
        interp_x = rng.choice(ds.x.size, ninterp) + rng.random(ninterp)
        interp_y = rng.choice(ds.y.size, ninterp) + rng.random(ninterp)

        # 1/ Estimation of theoretical memory usage of the subsampling script
        max_op_memusage = _estimate_interp_points_memusage(darr=darr, chunksizes_in_mem=chunksizes_in_mem,
                                                           ninterp=ninterp)


        # 2/ Run interpolation of random point coordinates with memory monitoring

        interp1, measured_op_memusage = _run_dask_measuring_memusage(cluster, delayed_interp_points, darr,
                                                                     points=(interp_x, interp_y), resolution=(1, 1))
        # Check the measured memory usage is smaller than the maximum estimated one
        assert measured_op_memusage < max_op_memusage

        # 3/ Output checks

        # Interpolate directly with Xarray (loads a lot in memory) and check results are exactly the same
        xx = xr.DataArray(interp_x, dims="z", name="x")
        yy = xr.DataArray(interp_y, dims="z", name="y")
        interp2 = ds.test.interp(x=xx, y=yy)
        interp2.compute()
        interp2 = np.array(interp2.values)

        assert np.array_equal(interp1, interp2, equal_nan=True)

    @pytest.mark.parametrize("fn", list_fn)
    @pytest.mark.parametrize("chunksizes_in_mem", [(2000, 2000), (1241, 3221)])
    @pytest.mark.parametrize("dst_chunksizes", [(2000, 2000), (1398, 2983)])
    # Shift upper left corner of output bounds (relative to projected input bounds) by fractions of the raster size
    @pytest.mark.parametrize("dst_bounds_rel_shift", [(0, 0), (-0.2, 0.5)])
    # Modify output resolution (relative to projected input resolution) by a factor
    @pytest.mark.parametrize("dst_res_rel_fac", [(1, 1), (2.1, 0.54)])
    # Same for shape
    @pytest.mark.parametrize("dst_shape_diff", [(0, 0), (-28, 117)])
    def test_delayed_reproject(self, fn: str, chunksizes_in_mem: tuple[int, int],
                               dst_chunksizes: tuple[int, int], dst_bounds_rel_shift: tuple[float, float],
                               dst_res_rel_fac: tuple[float, float], dst_shape_diff: tuple[int, int],
                               cluster: Any):
        """
        Checks for the delayed reproject function.
        Variables that influence specifically the delayed function are:
        - Input/output chunksizes,
        - Input array shape,
        - Output geotransform relative to projected input geotransform,
        - Output array shape relative to input.
        """

        fn = list_fn[0]
        chunksizes_in_mem = (2000, 2000)
        dst_chunksizes = (1398, 2983)  # (2000, 2000)
        dst_bounds_rel_shift = (0, 0)
        dst_res_rel_fac = (0.45, 0.45)  # (1, 1)
        dst_shape_diff = (0, 0)
        cluster = LocalCluster(n_workers=1, threads_per_worker=1, dashboard_address=None)

        # 0/ Open dataset with chunks and define variables
        ds = xr.open_dataset(fn, chunks={"x": chunksizes_in_mem[0], "y": chunksizes_in_mem[1]})
        darr = ds["test"].data

        # Get input and output shape
        src_shape = darr.shape
        dst_shape = (src_shape[0] + dst_shape_diff[0], src_shape[1] + dst_shape_diff[1])

        # Define arbitrary input/output CRS, they don't have a direct influence on the delayed method
        # (as long as the input/output transforms intersect if projected in the same CRS)
        src_crs = CRS(4326)
        dst_crs = CRS(32630)

        # Define arbitrary input transform, as we only care about the relative difference of the output transform
        src_transform = rio.transform.from_bounds(10, 10, 15, 15, src_shape[0], src_shape[1])

        # Other arguments having no influence
        src_nodata = -9999
        dst_nodata = 99999
        resampling = rio.enums.Resampling.cubic

        # Get shifted dst_transform with new resolution
        dst_transform = _build_dst_transform_shifted_newres(src_transform=src_transform, src_crs=src_crs, dst_crs=dst_crs,
                                                            src_shape=src_shape, bounds_rel_shift=dst_bounds_rel_shift,
                                                            res_rel_fac=dst_res_rel_fac)

        # 1/ Estimation of theoretical memory usage of the subsampling script

        max_op_memusage = _estimate_reproject_memusage(darr, chunksizes_in_mem=chunksizes_in_mem, dst_chunksizes=dst_chunksizes,
                                                       rel_res_fac=dst_res_rel_fac)

        # 2/ Run delayed reproject with memory monitoring

        # We define a function where computes happens during writing to be able to measure memory usage
        # (delayed_reproject returns a delayed array that might not fit in memory, unlike subsampling/interpolation)
        fn_tmp_out = os.path.join(_EXAMPLES_DIRECTORY, os.path.splitext(os.path.basename(fn))[0] + "_reproj.nc")

        def reproject_and_write(*args, **kwargs):

            # Run delayed reprojection
            reproj_arr_tmp = delayed_reproject(*args, **kwargs)

            # Save file out-of-memory and compute
            data_arr = xr.DataArray(data=reproj_arr_tmp, dims=["x", "y"])
            ds_out = xr.Dataset(data_vars={"test_reproj": data_arr})
            write_delayed = ds_out.to_netcdf(fn_tmp_out, compute=False)
            write_delayed.compute()

        # And call this function with memory usage monitoring
        _, measured_op_memusage = _run_dask_measuring_memusage(
            cluster,
            reproject_and_write,
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

        # Check the measured memory usage is smaller than the maximum estimated one
        assert measured_op_memusage < max_op_memusage

        # 3/ Outputs check: load in memory and compare with a direct Rasterio reproject
        reproj_arr = xr.open_dataset(fn_tmp_out)["test_reproj"].values

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

        # Keeping this to debug in case this is not only a Rasterio issue
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

        # Check that very little data (less than 0.01% of pixels) are significantly different
        # (it seems to be mostly some pixels that are nodata in one and not the other)
        ind_signif_diff = np.abs(reproj_arr - dst_arr) > 0.5
        assert np.count_nonzero(ind_signif_diff) < 0.01 / 100 * reproj_arr.size

        # The median difference is negligible compared to the amplitude of the signal (+/- 1 std)
        assert np.nanmedian(np.abs(reproj_arr - dst_arr)) < 0.02

        # # Replace with allclose once Rasterio issue fixed?
        # assert np.allclose(reproj_arr[~ind_both_nodata], dst_arr[~ind_both_nodata], atol=0.02)
