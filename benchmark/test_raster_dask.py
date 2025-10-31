"""
CURRENTLY UNUSED: Temporarily taken out of tests and saved for future performance benchmarking

Test RAM usage of Dask functions
."""

from __future__ import annotations

import os
import sys
from tempfile import NamedTemporaryFile
from typing import Any, Callable

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import rasterio as rio
import xarray as xr
from dask.distributed import Client, LocalCluster
from dask_memusage import install
from pluggy import PluggyTeardownRaisedWarning
from pyproj import CRS

from geoutils.examples import _EXAMPLES_DIRECTORY
from geoutils.raster.distributed_computing.dask import (
    delayed_interp_points,
    delayed_reproject,
    delayed_subsample,
)

# Ignore teardown warning given by Dask when closing the local cluster (due to dask-memusage plugin)
pytestmark = pytest.mark.filterwarnings("ignore", category=PluggyTeardownRaisedWarning)


@pytest.fixture(scope="module")  # type: ignore
def cluster():
    """Fixture to use a single cluster for the entire module (otherwise raise runtime errors)."""
    # Need cluster to be single-threaded to use dask-memusage confidently
    dask_cluster = LocalCluster(n_workers=1, threads_per_worker=1, dashboard_address=None)
    yield dask_cluster
    dask_cluster.close()


def _run_dask_measuring_memusage(
    cluster: Any, dask_func: Callable[..., Any], *args_dask_func: Any, **kwargs_dask_func: Any
) -> tuple[Any, float]:
    """Run a dask function monitoring its memory usage."""

    # Create a name temporary file that won't delete immediately
    fn_tmp_csv = NamedTemporaryFile(suffix=".csv", delete=False).name

    # Setup cluster and client within context managers for a clean shutdown
    install(cluster.scheduler, fn_tmp_csv)
    with Client(cluster) as _:
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

    # On top of the rest is added the Dask graph, we will multiply by a factor of 2.5 to get a good safety margin
    fac_dask_margin = 2.5
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
    max_op_memusage = fac_dask_margin * (chunk_memusage + sample_memusage + out_memusage + meta_memusage) / (2**20)
    # We add a base memory usage of ~130 MB + 10MB per 1000 chunks (loaded in background by Dask even on tiny data)
    max_op_memusage += 150 + 10 * (num_chunks / 1000)

    return max_op_memusage


def _estimate_interp_points_memusage(darr: da.Array, chunksizes_in_mem: tuple[int, int], ninterp: int) -> float:
    """
    Estimate the theoretical memory usage of the delayed interpolation method.
    (we don't need to be super precise, just within a factor of ~2 to check memory usage performs as expected)
    """

    # TOTAL SIZE = Single chunk operations + Chunk overlap + Metadata passed to dask + Outputs

    # On top of the rest is added the Dask graph, we will multiply by a factor of 2.5 to get a good safety margin
    fac_dask_margin = 2.5
    num_chunks = np.prod(darr.numblocks)

    # Single chunk operation = (data type bytes + boolean from np.isfinite + its subset) * overlapping chunksize **2
    chunk_memusage = (darr.dtype.itemsize + 2 * np.dtype("bool").itemsize) * np.prod(chunksizes_in_mem)
    # For interpolation, chunks have to overlap and temporarily load each neighbouring chunk,
    # we add 8 neighbouring chunks, and double the size due to the memory used during interpolation
    chunk_memusage *= 9

    # Outputs: pair of interpolated coordinates
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
    max_op_memusage = fac_dask_margin * (chunk_memusage + out_memusage + meta_memusage) / (2**20)
    # We add a base memory usage of ~80 MB + 10MB per 1000 chunks (loaded in background by Dask even on tiny data)
    max_op_memusage += 80 + 10 * (num_chunks / 1000)

    return max_op_memusage


def _estimate_reproject_memusage(
    darr: da.Array,
    chunksizes_in_mem: tuple[int, int],
    dst_chunksizes: tuple[int, int],
    rel_res_fac: tuple[float, float],
) -> float:
    """
    Estimate the theoretical memory usage of the delayed reprojection method.
    (we don't need to be super precise, just within a factor of ~2 to check memory usage performs as expected)
    """

    # TOTAL SIZE = Combined source chunk operations + Building geopandas mapping + Metadata passed to dask + Outputs

    # On top of the rest is added the Dask graph, we will multiply by a factor of 2.5 to get a good safety margin
    fac_dask_margin = 2.5
    num_chunks = np.prod(darr.numblocks)

    # THE BIG QUESTION: how many maximum source chunks might be loaded for a single destination chunk?
    # It depends on the relative ratio of input chunksizes to destination chunksizes, accounting for resolution change
    x_rel_source_chunks = dst_chunksizes[0] / chunksizes_in_mem[0] * rel_res_fac[0]
    y_rel_source_chunks = dst_chunksizes[1] / chunksizes_in_mem[1] * rel_res_fac[1]
    # There is also some overlap needed for resampling and due to warping in different CRS, so let's multiply this by 8
    # (all neighbouring tiles)
    nb_source_chunks_per_dest = 8 * x_rel_source_chunks * y_rel_source_chunks

    # Combined memory usage of one chunk operation = squared array made from combined chunksize + original chunks
    total_nb = np.ceil(np.sqrt(nb_source_chunks_per_dest)) ** 2 + nb_source_chunks_per_dest
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
    max_op_memusage = fac_dask_margin * (chunk_memusage + out_memusage + meta_memusage) / (2**20)
    # We add a base memory usage of ~80 MB + 10MB per 1000 chunks (loaded in background by Dask even on tiny data)
    max_op_memusage += 80 + 10 * (num_chunks / 1000)

    return max_op_memusage


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

    In details:
    Set 1. We capture memory usage during the .compute() calls and check that only the expected amount of memory that
        we estimate independently (bytes used by one or several chunk combinations + metadata) is indeed used.
    Set 2. We compare outputs with the in-memory function specifically for input variables that influence the delayed
        algorithm and might lead to new errors (for example: array shape to get subsample/points locations for
        subsample and interp_points, or destination chunksizes to map output of reproject).

    We start with set 2: output checks which run faster when ordered before
    (maybe due to the cluster memory monitoring after).
    """

    # Define random seed for generating test data
    rng = da.random.default_rng(seed=42)

    # 1/ Set 1: Memory usage checks

    # Big test files written on disk in an out-of-memory fashion,
    # with different input shapes not necessarily aligned between themselves
    large_shape = (10000, 10000)
    # We can use a constant value for storage chunks, as it doesn't have any influence on the accuracy of delayed
    # methods (can change slightly RAM usage, but pretty stable as long as chunksizes in memory are larger and
    # significantly bigger)
    chunksizes_on_disk = (500, 500)
    fn_large = os.path.join(_EXAMPLES_DIRECTORY, "test_large.nc")
    if not os.path.exists(fn_large):
        # Create random array in the right shape
        data = rng.normal(size=large_shape[0] * large_shape[1]).reshape(large_shape[0], large_shape[1])
        data_arr = xr.DataArray(data=data, dims=["x", "y"])
        ds = xr.Dataset(data_vars={"test": data_arr})
        encoding_kwargs = {"test": {"chunksizes": chunksizes_on_disk}}
        # Write to disk out-of-memory
        writer = ds.to_netcdf(fn_large, encoding=encoding_kwargs, compute=False)
        writer.compute()

    @pytest.mark.parametrize("fn", [fn_large])  # type: ignore
    @pytest.mark.parametrize("chunksizes_in_mem", [(1000, 1000), (2500, 2500)])  # type: ignore
    @pytest.mark.parametrize("subsample_size", [100, 100000])  # type: ignore
    def test_delayed_subsample__memusage(
        self, fn: str, chunksizes_in_mem: tuple[int, int], subsample_size: int, cluster: Any
    ):
        """
        Checks for delayed subsampling function for memory usage on big file.
        (and also runs output checks as not long or too memory intensive in this case)
        Variables that influence memory usage are:
        - Subsample sizes,
        - Chunksizes in memory.
        """

        # Only check on linux
        if sys.platform == "linux":

            # 0/ Open dataset with chunks
            ds = xr.open_dataset(fn, chunks={"x": chunksizes_in_mem[0], "y": chunksizes_in_mem[1]})
            darr = ds["test"].data

            # 1/ Estimation of theoretical memory usage of the subsampling script

            max_op_memusage = _estimate_subsample_memusage(
                darr=darr, chunksizes_in_mem=chunksizes_in_mem, subsample_size=subsample_size
            )

            # 2/ Run delayed subsample with dask memory usage monitoring

            # Derive subsample from delayed function
            # (passed to wrapper function to measure memory usage during execution)
            sub, measured_op_memusage = _run_dask_measuring_memusage(
                cluster, delayed_subsample, darr, subsample=subsample_size, random_state=42
            )

            # Check the measured memory usage is smaller than the maximum estimated one
            assert measured_op_memusage < max_op_memusage

            # 3/ Output checks
            # The subsample should have exactly the prescribed length, with only valid values
            assert len(sub) == subsample_size
            assert all(np.isfinite(sub))

            # To verify the sampling works correctly, we can get its subsample indices with the argument return_indices
            # And compare to the same subsample with vindex (now that we know the coordinates of valid values sampled)
            indices = delayed_subsample(darr, subsample=subsample_size, random_state=42, return_indices=True)
            sub2 = np.array(darr.vindex[indices[0], indices[1]])
            assert np.array_equal(sub, sub2)

    @pytest.mark.parametrize("fn", [fn_large])  # type: ignore
    @pytest.mark.parametrize("chunksizes_in_mem", [(2000, 2000)])  # type: ignore
    @pytest.mark.parametrize("ninterp", [100, 100000])  # type: ignore
    def test_delayed_interp_points__memusage(
        self, fn: str, chunksizes_in_mem: tuple[int, int], ninterp: int, cluster: Any
    ):
        """
        Checks for delayed interpolate points function for memory usage on a big file.
        Variables that influence memory usage are:
        - Number of interpolated points,
        - Chunksizes in memory.
        """

        # Only check on linux
        if sys.platform == "linux":

            # 0/ Open dataset with chunks and create random point locations to interpolate
            ds = xr.open_dataset(fn, chunks={"x": chunksizes_in_mem[0], "y": chunksizes_in_mem[1]})
            darr = ds["test"].data

            rng = np.random.default_rng(seed=42)
            interp_x = rng.choice(ds.x.size, ninterp) + rng.random(ninterp)
            interp_y = rng.choice(ds.y.size, ninterp) + rng.random(ninterp)

            # 1/ Estimation of theoretical memory usage of the subsampling script
            max_op_memusage = _estimate_interp_points_memusage(
                darr=darr, chunksizes_in_mem=chunksizes_in_mem, ninterp=ninterp
            )

            # 2/ Run interpolation of random point coordinates with memory monitoring
            interp1, measured_op_memusage = _run_dask_measuring_memusage(
                cluster, delayed_interp_points, darr, points=(interp_x, interp_y), resolution=(1, 1)
            )
            # Check the measured memory usage is smaller than the maximum estimated one
            assert measured_op_memusage < max_op_memusage

    @pytest.mark.parametrize("fn", [fn_large])  # type: ignore
    @pytest.mark.parametrize("chunksizes_in_mem", [(1000, 1000), (2500, 2500)])  # type: ignore
    @pytest.mark.parametrize("dst_chunksizes", [(1000, 1000), (2500, 2500)])  # type: ignore
    @pytest.mark.parametrize("dst_bounds_rel_shift", [(1, 1), (2, 2)])  # type: ignore
    def test_delayed_reproject__memusage(
        self,
        fn: str,
        chunksizes_in_mem: tuple[int, int],
        dst_chunksizes: tuple[int, int],
        dst_bounds_rel_shift: tuple[float, float],
        cluster: Any,
    ):
        """
        Checks for the delayed reproject function for memory usage on a big file.
        Variables that influence memory usage are:
        - Source chunksizes in memory,
        - Destination chunksizes in memory,
        - Relative difference in resolution (potentially more/less source chunks to load for a destination chunk).
        """

        # Only check on linux
        if sys.platform == "linux":

            # We fix arbitrary changes to the destination shape/resolution/bounds
            # (already checked in details in the output tests)
            dst_shape_diff = (25, -25)
            dst_res_rel_fac = (1.5, 0.5)

            # 0/ Open dataset with chunks and define variables
            ds = xr.open_dataset(fn, chunks={"x": chunksizes_in_mem[0], "y": chunksizes_in_mem[1]})
            darr = ds["test"].data

            # Get input and output shape
            src_shape = darr.shape
            dst_shape = (src_shape[0], src_shape[1] + dst_shape_diff[1])

            # Define arbitrary input/output CRS, they don't have a direct influence on the delayed method
            # (as long as the input/output transforms intersect if projected in the same CRS)
            src_crs = CRS(4326)
            dst_crs = CRS(32630)

            # Define arbitrary input transform, as we only care about the relative difference with the output transform
            src_transform = rio.transform.from_bounds(10, 10, 15, 15, src_shape[0], src_shape[1])

            # Other arguments having no influence
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

            # 1/ Estimation of theoretical memory usage of the subsampling script

            max_op_memusage = _estimate_reproject_memusage(
                darr, chunksizes_in_mem=chunksizes_in_mem, dst_chunksizes=dst_chunksizes, rel_res_fac=dst_res_rel_fac
            )

            # 2/ Run delayed reproject with memory monitoring

            # We define a function where computes happens during writing to be able to measure memory usage
            # (delayed_reproject returns a delayed array that might not fit in memory, unlike subsampling/interpolation)
            fn_tmp_out = os.path.join(_EXAMPLES_DIRECTORY, os.path.splitext(os.path.basename(fn))[0] + "_reproj.nc")

            def reproject_and_write(*args: Any, **kwargs: Any) -> None:
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
