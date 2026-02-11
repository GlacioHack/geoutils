# Copyright (c) 2025 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functionalities for interpolating a regular grid at points (raster to point cloud)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Literal, overload
import warnings

import numpy as np
import rasterio as rio
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.ndimage import binary_dilation, distance_transform_edt, map_coordinates

from geoutils import profiler
from geoutils._dispatch import _check_match_points
from geoutils._typing import NDArrayNum, Number
from geoutils._misc import import_optional
from geoutils.projtools import reproject_from_latlon
from geoutils.raster.referencing import _coords, _outside_bounds, _res, _xy2ij, _bounds
from geoutils.multiproc import MultiprocConfig, compute_tiling

method_to_order = {"nearest": 0, "linear": 1, "cubic": 3, "quintic": 5, "slinear": 1, "pchip": 3, "splinef2d": 3}

if TYPE_CHECKING:
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterBase
    from geoutils.raster.raster import Raster

# Dask as optional dependency
try:
    import dask
    import dask.array as da
    from dask import delayed
    from dask.utils import cached_cumsum
except ImportError:

    def delayed(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Fake delayed decorator if dask is not installed
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

####################################################
# 1/ REGULAR GRID INTERPOLATION AT POINT COORDINATES
####################################################

def _get_dist_nodata_spread(order: int, dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int) -> int:
    """
    Derive distance of nodata spreading based on interpolation order.

    :param order: Interpolation order.
    :param dist_nodata_spread: Spreading distance of nodata, either half-order rounded up (default), rounded down, or
        fixed integer.
    """

    if dist_nodata_spread == "half_order_up":
        dist_nodata_spread = int(np.ceil(order / 2))
    elif dist_nodata_spread == "half_order_down":
        dist_nodata_spread = int(np.floor(order / 2))

    return dist_nodata_spread


def _interpn_interpolator(
    points: tuple[NDArrayNum, NDArrayNum],
    values: NDArrayNum,
    fill_value: Number = np.nan,
    bounds_error: bool = False,
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int = "half_order_up",
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
) -> Callable[[tuple[NDArrayNum, NDArrayNum]], NDArrayNum]:
    """
    Create SciPy interpolator with nodata spreading. Default is spreading at distance of half the method order
    rounded up (i.e., linear spreads 1 nodata in each direction, cubic spreads 2, quintic 3).

    Gives the exact same result as scipy.interpolate.interpn, and allows interpolator to be re-used if required (
    for speed).
    In practice, returns either a NaN-modified RegularGridInterpolator or a NaN-modified RectBivariateSpline object,
    both expecting a tuple of X/Y coordinates to be evaluated.

    For input arguments, see scipy.interpolate.RegularGridInterpolator.
    For additional argument "dist_nodata_spread", see description of Raster.interp_points.

    Adapted from:
    https://github.com/scipy/scipy/blob/44e4ebaac992fde33f04638b99629d23973cb9b2/scipy/interpolate/_rgi.py#L743.
    """

    # Derive distance to spread nodata to depending on method order
    order = method_to_order[method]
    d = _get_dist_nodata_spread(order=order, dist_nodata_spread=dist_nodata_spread)

    # We compute the nodata mask and dilate it to the distance to spread nodatas
    mask_nan = ~np.isfinite(values)
    if d != 0:
        new_mask = binary_dilation(mask_nan, iterations=d).astype("uint8")
    # Zero iterations has a different behaviour in binary_dilation than doing nothing, we want the original array
    else:
        new_mask = mask_nan.astype("uint8")

    # We create an interpolator for the nodata mask using nearest
    interp_mask = RegularGridInterpolator(points, new_mask, method="nearest", bounds_error=bounds_error, fill_value=1)

    # Most methods (cubic, quintic, etc) do not support NaNs and require an array full of valid values
    # We replace thus replace all NaN values by nearest neighbours to give surrounding values of the same order of
    # magnitude and minimize interpolation errors near NaNs (errors of 10e-2/e-5 relative to the values)
    # Elegant solution from: https://stackoverflow.com/questions/5551286/filling-gaps-in-a-numpy-array for a fast
    # nearest neighbour fill
    indices = distance_transform_edt(mask_nan, return_distances=False, return_indices=True)
    values = values[tuple(indices)]

    # For the RegularGridInterpolator
    if method in RegularGridInterpolator._ALL_METHODS:

        # We create the classic interpolator
        interp = RegularGridInterpolator(
            points, values, method=method, bounds_error=bounds_error, fill_value=fill_value
        )

        # We create a new interpolator callable that propagates nodata as defined above
        def regulargrid_interpolator_with_nan(xi: tuple[NDArrayNum, NDArrayNum]) -> NDArrayNum:

            # Get results
            results = interp(xi)
            # Get invalids
            invalids = interp_mask(xi)
            results[invalids.astype(bool)] = np.nan

            return results

        return regulargrid_interpolator_with_nan

    # For the RectBivariateSpline
    else:

        # The coordinates must be in ascending order, which requires flipping the array too (more costly)
        interp = RectBivariateSpline(np.flip(points[0]), points[1], np.flip(values[:], axis=0))

        # We create a new interpolator callable that propagates nodata as defined above, and supports fill_value
        def rectbivariate_interpolator_with_fillvalue(xi: tuple[NDArrayNum, NDArrayNum]) -> NDArrayNum:

            # Get invalids
            invalids = interp_mask(xi)

            # RectBivariateSpline doesn't support fill_value, so we need to wrap here to add them
            xi_arr = np.array(xi).T
            xi_shape = xi_arr.shape
            xi_arr = xi_arr.reshape(-1, xi_arr.shape[-1])
            idx_valid = np.all(
                (
                    points[0][-1] <= xi_arr[:, 0],
                    xi_arr[:, 0] <= points[0][0],
                    points[1][0] <= xi_arr[:, 1],
                    xi_arr[:, 1] <= points[1][-1],
                ),
                axis=0,
            )
            # Make a copy of values for RectBivariateSpline
            result = np.empty_like(xi_arr[:, 0])
            result[idx_valid] = interp.ev(xi_arr[idx_valid, 0], xi_arr[idx_valid, 1])
            result[np.logical_not(idx_valid)] = fill_value

            # Add back NaNs from dilated mask
            results = np.atleast_1d(result.reshape(xi_shape[:-1]))
            results[invalids.astype(bool)] = np.nan

            return results

        return rectbivariate_interpolator_with_fillvalue


def _map_coordinates_nodata_propag(
    values: NDArrayNum,
    indices: tuple[NDArrayNum, NDArrayNum],
    order: int,
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int = "half_order_up",
    **kwargs: Any,
) -> NDArrayNum:
    """
    Perform map_coordinates with nodata spreading. Default is spreading at distance of half the method order rounded
    up (i.e., linear spreads 1 nodata in each direction, cubic spreads 2, quintic 3).

    For map_coordinates, only nearest and linear are used.

    For input arguments, see scipy.ndimage.map_coordinates.
    For additional argument "dist_nodata_spread", see description of Raster.interp_points.
    """

    # Derive distance of nodata spreading
    d = _get_dist_nodata_spread(order=order, dist_nodata_spread=dist_nodata_spread)

    # We compute the mask and dilate it to the distance to spread nodatas
    mask_nan = ~np.isfinite(values)
    if d != 0:
        new_mask = binary_dilation(mask_nan, iterations=d).astype("uint8")
    # Zero iterations has a different behaviour in binary_dilation than doing nothing, here we want the original array
    else:
        new_mask = mask_nan.astype("uint8")

    # We replace all NaN values by nearest neighbours to minimize interpolation errors near NaNs
    # Elegant solution from: https://stackoverflow.com/questions/5551286/filling-gaps-in-a-numpy-array
    ind = distance_transform_edt(mask_nan, return_distances=False, return_indices=True)
    values = values[tuple(ind)]

    # We interpolate the dilated array at the coordinates with nearest, and transform it back to a boolean to mask NaNs
    rmask = map_coordinates(new_mask, indices, order=0, cval=1, prefilter=False)

    # Interpolate at indices
    rpoints = map_coordinates(values, indices, order=order, **kwargs)

    # Set to NaNs based on spreading distance
    rpoints[rmask.astype(bool)] = np.nan

    return rpoints


# BASE FUNCTION FOR INTERP POINTS (WHOLE ARRAY IN MEMORY, USED BY CHUNKED FUNCTIONS + MAIN API)

@overload
def _interp_points_base(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum],
    area_or_point: Literal["Area", "Point"] | None = None,
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int = "half_order_up",
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    *,
    return_interpolator: Literal[False] = False,
    **kwargs: Any,
) -> NDArrayNum: ...


@overload
def _interp_points_base(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum],
    area_or_point: Literal["Area", "Point"] | None = None,
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int = "half_order_up",
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    *,
    return_interpolator: Literal[True],
    **kwargs: Any,
) -> Callable[[tuple[NDArrayNum, NDArrayNum]], NDArrayNum]: ...


@overload
def _interp_points_base(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum],
    area_or_point: Literal["Area", "Point"] | None = None,
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int = "half_order_up",
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    *,
    return_interpolator: bool = False,
    **kwargs: Any,
) -> NDArrayNum | Callable[[tuple[NDArrayNum, NDArrayNum]], NDArrayNum]: ...


def _interp_points_base(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum] | None,
    area_or_point: Literal["Area", "Point"] | None = None,
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int = "half_order_up",
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    return_interpolator: bool = False,
    **kwargs: Any,
) -> NDArrayNum | Callable[[tuple[NDArrayNum, NDArrayNum]], NDArrayNum]:

    # If array is not a floating dtype (to support NaNs), convert dtype
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    # If array is masked, fill with NaN without copy
    if np.ma.isMaskedArray(array):
        array = array.filled(np.nan)

    # If the raster is on an equal grid, use scipy.ndimage.map_coordinates
    force_map_coords = force_scipy_function is not None and force_scipy_function == "map_coordinates"
    force_interpn = force_scipy_function is not None and force_scipy_function == "interpn"

    # Map method name to spline order in map_coordinates, and use only is method compatible
    method_to_order_mapcoords = {"nearest": 0, "linear": 1}
    mapcoords_supported = method in method_to_order_mapcoords.keys()

    res = _res(transform)
    use_mapcoords = ((res[0] == res[1] or force_map_coords) and not force_interpn and
                     mapcoords_supported and not return_interpolator)

    if not return_interpolator:
        x, y = points

    if use_mapcoords:
        # Convert method name into order
        order = method_to_order_mapcoords[method]

        # Remove default spline pre-filtering that is activated by default
        if "prefilter" not in kwargs.keys():
            kwargs.update({"prefilter": False})
        # Change default constant value to NaN for interpolation outside the image bounds
        if "cval" not in kwargs.keys():
            kwargs.update({"cval": np.nan})

        # Use map coordinates with nodata propagation
        i, j = _xy2ij(x, y, transform=transform, area_or_point=area_or_point, shift_area_or_point=shift_area_or_point)
        rpoints = _map_coordinates_nodata_propag(
            values=array, indices=(i, j), order=order, dist_nodata_spread=dist_nodata_spread, **kwargs
        )

    # Otherwise, use scipy.interpolate.interpn
    else:
        # Get lower-left corner coordinates
        xycoords = _coords(
            transform=transform,
            shape=array.shape[0:2],
            area_or_point=area_or_point,
            grid=False,
            shift_area_or_point=shift_area_or_point,
        )

        # Let interpolation outside the bounds not raise any error by default
        if "bounds_error" not in kwargs.keys():
            kwargs.update({"bounds_error": False})
        # Return NaN outside image bounds
        if "fill_value" not in kwargs.keys():
            kwargs.update({"fill_value": np.nan})

        # Using direct coordinates, Y is the first axis, and we need to flip it
        interpolator = _interpn_interpolator(
            points=(np.flip(xycoords[1], axis=0), xycoords[0]),
            values=array,
            method=method,
            dist_nodata_spread=dist_nodata_spread,
            bounds_error=kwargs["bounds_error"],
            fill_value=kwargs["fill_value"],
        )
        if return_interpolator:
            return interpolator
        else:
            rpoints = interpolator((y, x))  # type: ignore

    return rpoints

# CHUNKED LOGIC: POINT INTERPOLATION ON REGULAR OR EQUAL GRID
# Notes at the date of April 2024:
# This functionality is not covered efficiently by Dask/Xarray, because they need to support rectilinear grids, which
# is difficult when interpolating in the chunked dimensions, and loads nearly all array memory when using .interp().

# Here we harness the fact that rasters are always on regular (or sometimes equal) grids to efficiently map
# the location of the blocks required for interpolation, which requires little memory usage.

# Code structure inspired by https://blog.dask.org/2021/07/02/ragged-output and the "block_id" in map_blocks

def _get_interp_indices_per_block(
    interp_x: NDArrayNum,
    interp_y: NDArrayNum,
    starts: list[tuple[int, ...]],
    num_chunks: tuple[int, int],
    chunksize: tuple[int, int],
    xres: float,
    yres: float,
    left: float,
    top: float,
) -> list[list[int]]:
    """Map blocks where each pair of interpolation coordinates will have to be computed."""

    # The argument "starts" contains the list of chunk first X/Y index for the full array, plus the last index
    ny, nx = num_chunks
    y_chunksize, x_chunksize = chunksize
    y_starts, x_starts = starts

    # We use one bucket per block, assuming a flattened blocks shape
    ind_per_block = [[] for _ in range(ny * nx)]
    for i, (x, y) in enumerate(zip(interp_x, interp_y)):
        # Because it is a regular grid, we know exactly in which block ID the coordinate will fall
        xb = int(np.floor((x - left) / (xres * x_chunksize)))
        yb = int(np.floor((top - y) / (yres * y_chunksize)))

        if 0 <= xb < nx and 0 <= yb < ny:
            ind_per_block[yb * nx + xb].append(i)

    return ind_per_block

@delayed
def _delayed_interp_points_block(
    arr_chunk: NDArrayNum, block_id: dict[str, Any], interp_coords: NDArrayNum, **kwargs,
) -> NDArrayNum:
    """
    Interpolate block in 2D out-of-memory for a regular or equal grid.
    """

    # Extract information out of block_id dictionary
    xs, ys, xres, yres = (block_id["xstart"], block_id["ystart"], block_id["xres"], block_id["yres"])

    # Reconstruct the transform from xi/yi/xres/yres
    transform = rio.transform.from_origin(xs, ys, xres, yres)

    # Interpolate to points by dispatching to base function
    interp_chunk = _interp_points_base(
        array=arr_chunk,
        transform=transform,
        points=(interp_coords[0, :], interp_coords[1, :]),
        **kwargs,
    )

    # And return the interpolated array
    return interp_chunk

def _dask_interp_points(
    darr: da.Array,
    transform: rio.transform.Affine,
    points: tuple[NDArrayNum, NDArrayNum],
    **kwargs: Any,
) -> NDArrayNum:
    """
    Interpolate raster at point coordinates on out-of-memory chunks.

    This function harnesses the fact that a raster is defined on a regular (or equal) grid, and it is therefore
    faster than Xarray.interpn (especially for small sample sizes) and uses only a fraction of the memory usage.

    :param darr: Input dask array.
    :param transform: Geotransform of array.
    :param points: Point(s) at which to interpolate raster value. If points fall outside of image, value
            returned is nan. Shape should be tuple of arrays.
    :param kwargs: Keyword arguments passed to interp_points_base.

    :return: Array of raster value(s) interpolated at the given points.
    """

    # To raise appropriate error on missing optional dependency
    import_optional("dask")

    # Convert input to 2D array
    points_arr = np.vstack((points[0], points[1]))

    # Map depth of overlap required for each interpolation method
    depth = method_to_order[kwargs["method"]] + 1  # The overlap size is the order + 1
    res = _res(transform)
    bounds = _bounds(transform=transform, shape=darr.shape)
    left, right, top, bottom = bounds.left, bounds.right, bounds.top, bounds.bottom

    # Expand dask array for overlapping computations
    chunksize = darr.chunksize
    # TODO: Figure out how to get NaNs in exact same places
    expanded = da.overlap.overlap(darr, depth=depth, boundary="nearest")

    # Get starting 2D index for each chunk of the full array
    # (mirroring what is done in block_id of dask.array.map_blocks)
    starts = [cached_cumsum(c, initial_zero=True) for c in darr.chunks]
    num_chunks = expanded.numblocks

    # Get samples indices per blocks
    ind_per_block = _get_interp_indices_per_block(
        points_arr[0, :], points_arr[1, :], starts, num_chunks, chunksize, res[0], res[1], left, top,
    )

    # Create a delayed object for each block, and flatten the blocks into a 1d shape
    blocks = expanded.to_delayed().ravel()

    # Build the block IDs by unravelling starting indexes for each block (Y is first axis)
    indexes_yi, indexes_xi = np.unravel_index(np.arange(len(blocks)), shape=(num_chunks[0], num_chunks[1]))
    block_ids = [
        {
            "xstart": left + (starts[1][indexes_xi[i]] - depth) * res[0],
            "ystart": top - (starts[0][indexes_yi[i]] - depth) * res[1],
            "xres": res[0],
            "yres": res[1],
        }
        for i in range(len(blocks))
    ]

    # Compute values delayed
    used = [i for i in range(len(blocks)) if len(ind_per_block[i]) > 0]
    list_interp = [
        _delayed_interp_points_block(blocks[i], block_ids[i], points_arr[:, ind_per_block[i]], **kwargs)
        for i in used
    ]

    # We concatenate and re-order in a delayed manner
    def _concat_reorder(list_vals, list_inds):
        # Flatten outputs to 1D and concatenate
        vals = [np.asarray(v).ravel() for v in list_vals]
        vcat = np.concatenate(vals) if vals else np.array([], dtype=np.float32)

        # Build index array and argsort
        inds = np.concatenate([np.asarray(ii, dtype=np.int64) for ii in list_inds]) if list_inds else np.array([],
                                                                                                     dtype=np.int64)
        order = np.argsort(inds)
        return vcat[order]

    # Get list of indexes only for used blocks
    list_inds_used = [ind_per_block[i] for i in used]
    joined = dask.delayed(_concat_reorder)(list_interp, list_inds_used)

    # Join into full length array of input dtype
    interp_points = da.from_delayed(joined, shape=(len(points[0]),), dtype=darr.dtype)

    return interp_points

# SAME WITH MULTIPROCESSING

def _wrapper_multiproc_interp_per_block(
    rst: Raster,
    block_id: dict[str, Any],
    interp_coords: NDArrayNum,
    **kwargs,
) -> NDArrayNum:
    """Wrapper to use interpolation per block."""

    # Extract information out of block_id dictionary
    tile_idx = block_id["tile_idx"]

    # Crop input raster for the given block
    rst_block = rst.icrop((tile_idx[2], tile_idx[0], tile_idx[3], tile_idx[1]))

    # Interpolate to points by dispatching to base function
    interp_chunk = _interp_points_base(
        array=rst_block.data,
        transform=rst_block.transform,
        points=(interp_coords[0, :], interp_coords[1, :]),
        **kwargs,
    )

    # And return the interpolated array
    return interp_chunk

def _multiproc_interp_points(
    rst: Raster,
    points: tuple[list[float], list[float]],
    config: MultiprocConfig,
    **kwargs: Any,
) -> NDArrayNum:
    """
    Interpolate raster at point coordinates on out-of-memory chunks.
    """

    # Convert input to 2D array
    points_arr = np.vstack((points[0], points[1]))

    # Map depth of overlap required for each interpolation method
    depth = method_to_order[kwargs["method"]] + 1  # The overlap size is the order + 1
    res = _res(rst.transform)
    bounds = _bounds(transform=rst.transform, shape=rst.shape)
    left, right, top, bottom = bounds.left, bounds.right, bounds.top, bounds.bottom

    # Get multiprocessing chunk sizes
    chunksize = (config.chunk_size, config.chunk_size)

    def _chunk_sizes_1d(n: int, chunk: int) -> tuple[int, ...]:
        """Return tuple of chunk lengths that partition length n."""
        if chunk <= 0:
            raise ValueError("chunk must be > 0")
        full = n // chunk
        rem = n % chunk
        sizes = (chunk,) * full + ((rem,) if rem else ())
        return sizes if sizes else (0,)

    def cached_cumsum(chunks: tuple[int, ...], initial_zero: bool = True) -> list[int]:
        """Like dask's cumulative chunk starts. For (3,3,1) -> [0,3,6,7] if initial_zero."""
        out = [0] if initial_zero else []
        s = 0
        for c in chunks:
            s += c
            out.append(s)
        return out

    # Get starting 2D index for each chunk of the full array
    # (mirroring what is done in block_id of dask.array.map_blocks)
    tiling = compute_tiling(tile_size=config.chunk_size, raster_shape=rst.shape, overlap=depth)
    starts = [cached_cumsum(_chunk_sizes_1d(n=rst.shape[0], chunk=config.chunk_size), initial_zero=True),
              cached_cumsum(_chunk_sizes_1d(n=rst.shape[1], chunk=config.chunk_size), initial_zero=True)]
    num_chunks = (tiling.shape[0], tiling.shape[1])
    num_blocks = np.prod(num_chunks)

    # Get samples indices per blocks
    ind_per_block = _get_interp_indices_per_block(
        points_arr[0, :], points_arr[1, :], starts, num_chunks, chunksize, res[0], res[1], left, top,
    )

    # Build the block IDs by unravelling starting indexes for each block
    indexes_xi, indexes_yi = np.unravel_index(np.arange(num_blocks), shape=(num_chunks[0], num_chunks[1]))
    block_ids = [{"tile_idx": tiling[indexes_xi[i], indexes_yi[i], :]} for i in range(num_blocks)]

    # Create tasks for multiprocessing
    tasks = []
    for i in range(len(block_ids)):
        # Launch the task on the cluster to process each tile
        tasks.append(
            config.cluster.launch_task(fun=_wrapper_multiproc_interp_per_block,
                                       args=[rst, block_ids[i], points_arr[:, ind_per_block[i]]],
                                       kwargs=kwargs)
        )

    # Collect results
    try:
        list_interp = []
        # Iterate over the tasks and retrieve the processed results
        for results in tasks:
            interp = config.cluster.get_res(results)
            list_interp.append(interp)
    except Exception as e:
        raise RuntimeError(f"Error retrieving interpolated segments from multiprocessing tasks: {e}")

    # Concatenate outputs
    interp_points = np.concatenate(list_interp, axis=0)

    # Re-order per-block output points to match their original indices
    indices = np.concatenate(ind_per_block).astype(int)
    argsort = np.argsort(indices)
    interp_points = np.array(interp_points)[argsort]
    print(interp_points)

    return interp_points


# MAIN API FUNCTION CHECKING USER INPUTS AND DISPATCHING TO BASE, DASK OR MULTIPROCESSING

def _interp_points(
    source_raster: RasterBase,
    points: tuple[NDArrayNum, NDArrayNum] | tuple[Number, Number] | PointCloudLike,
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
    band: int = 1,
    input_latlon: bool = False,
    as_array: bool = False,
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int = "half_order_up",
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    return_interpolator: bool = False,
    mp_config: MultiprocConfig | None = None,
    **kwargs: Any,
):
    """See description of Raster.interp_points."""

    # 1/ Input checks

    # Check and normalize input points
    pts, input_scalar = _check_match_points(source_raster, points)

    # Extract raster metadata for later checks and conversions
    transform = source_raster.transform
    area_or_point = source_raster.area_or_point
    shape = source_raster.shape

    # Convert from latlon if necessary
    if input_latlon:
        pts = reproject_from_latlon(pts, out_crs=source_raster.crs)

    # If we evaluate points (not returning interpolator), remove those outside of bounds
    # (Out of bounds points are hard to deal with for chunked operations otherwise)
    if not return_interpolator:
        if pts is None:
            raise ValueError("Input 'points' cannot be None if 'return_interpolator' is False.")
        x, y = pts
        i, j = _xy2ij(x, y, transform=transform, area_or_point=area_or_point, shift_area_or_point=shift_area_or_point)

        # Get index of points outside of bounds
        ind_outofbounds = np.vectorize(
            lambda k1, k2: _outside_bounds(
                k1, k2, transform=transform, area_or_point=area_or_point,
                shape=shape, index=True
            )
        )(j, i)

        # If all points fell outside of bounds
        if np.count_nonzero(~ind_outofbounds) == 0:
            warnings.warn("All provided points were outside of raster bounds, returning only NaNs.")
            output = np.full(x.shape[0], np.nan)
            if as_array:
                return output
            else:
                # If point cloud input
                from geoutils.pointcloud import (
                    PointCloud,  # Runtime import to avoid circular issues
                )
                return PointCloud.from_xyz(x=points[0], y=points[1], z=output, crs=source_raster.crs)

        # Only work on points inside bounds
        pts_inbounds = x[~ind_outofbounds], y[~ind_outofbounds]
    else:
        pts_inbounds = None

    # 2/ Dispatch to either base (in-memory) function, Dask function, or Multiprocessing function
    interp_kwargs = {"area_or_point": area_or_point, "method": method, "dist_nodata_spread": dist_nodata_spread,
                     "shift_area_or_point": shift_area_or_point, "force_scipy_function": force_scipy_function,
                     "return_interpolator": return_interpolator}
    interp_kwargs.update(kwargs)

    # Cannot use Multiprocessing backend and Dask backend simultaneously
    mp_backend = mp_config is not None
    # The check below can only run on Xarray
    dask_backend = da is not None and source_raster._chunks is not None

    if mp_backend and dask_backend:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove mp_config parameter "
            "from reproject(). To use Multiprocessing, open the file without chunks."
        )

    if (dask_backend or mp_backend) and return_interpolator:
        raise ValueError("Option 'return_interpolator' of interp_points cannot be used with Dask or Multiprocessing, "
                         "only with in-memory array.")

    # If using Multiprocessing backend, process and return NumPy array (ragged output)
    if mp_backend:
        # Temporary switch bands
        orig_bands = source_raster.bands
        source_raster._bands = (band,)
        z_inbounds = _multiproc_interp_points(rst=source_raster, points=pts_inbounds,
                                              config=mp_config, **interp_kwargs)
        # Rewrite original bands
        source_raster._bands = orig_bands
    # For both Dask and NumPy array:
    else:
        if source_raster.data.ndim != 2:
            arr = source_raster.data[band - 1, :, :]
        else:
            arr = source_raster.data
        # If using Dask backend, process and return NumPy array (ragged output)
        if dask_backend:
            z_inbounds = _dask_interp_points(darr=arr, transform=transform, points=pts_inbounds, **interp_kwargs)
        # If using direct reprojection, process and return NumPy array
        else:
            z_inbounds = _interp_points_base(array=arr, transform=transform, points=pts_inbounds, **interp_kwargs)

    # 3/ Output preparation and return

    # If interpolator, return directly
    if return_interpolator:
        return z_inbounds
    # Otherwise, return array of input length with NaNs for outside-bound points
    else:

        # Get output length and dtype
        n = len(x)
        dtype = source_raster.dtype

        # Rebuild array (delayed if Dask, normal if NumPy)
        def _rebuild_with_nans(z_inbounds: np.ndarray, mask_out: np.ndarray, n: int, dtype):
            out = np.full(n, np.nan, dtype=np.float32 if np.issubdtype(dtype, np.integer) else dtype)
            out[~mask_out] = z_inbounds
            return out

        if dask_backend:
            out_del = dask.delayed(_rebuild_with_nans)(z_inbounds, ind_outofbounds, n, dtype)
            z = da.from_delayed(out_del, shape=(n,), dtype=np.float32 if np.issubdtype(dtype, np.integer) else dtype)
        else:
            z = _rebuild_with_nans(z_inbounds, ind_outofbounds, n, dtype)

        # Return array or pointcloud
        if as_array:
            return z
        else:
            # If point cloud input
            from geoutils.pointcloud import (
                PointCloud,  # Runtime import to avoid circular issues
            )
            return PointCloud.from_xyz(x=points[0], y=points[1], z=z, crs=source_raster.crs)


##############################################################
# 2/ REGULAR GRID REDUCTION IN WINDOW AROUND POINT COORDINATES
##############################################################

def _reduce_points(
    source_raster: RasterBase,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum] | PointCloudLike,
    reducer_function: Callable[[NDArrayNum], float] = np.ma.mean,
    window: int | None = None,
    input_latlon: bool = False,
    band: int | None = None,
    masked: bool = False,
    return_window: bool = False,
    as_array: bool = False,
    boundless: bool = True,
) -> NDArrayNum | tuple[NDArrayNum, NDArrayNum]:

    # Check and normalize input points
    pts, input_scalar = _check_match_points(source_raster, points)

    # Convert from latlon if necessary
    if input_latlon:
        pts = reproject_from_latlon(pts, out_crs=source_raster.crs)

    x, y = pts

    # Check window parameter
    if window is not None:
        if not float(window).is_integer():
            raise ValueError("Window must be a whole number.")
        if window % 2 != 1:
            raise ValueError("Window must be an odd number.")
        window = int(window)

    # Define subfunction for reducing the window array
    def format_value(value: Any) -> Any:
        """Check if valid value has been extracted"""
        if type(value) in [np.ndarray, np.ma.core.MaskedArray]:
            if window is not None:
                value = np.atleast_1d(reducer_function(value.flatten()))
            else:
                value = np.atleast_1d(value[0, 0])
        else:
            value = None
        return value

    # Initiate output lists
    list_values = []
    if return_window:
        list_windows = []

    # Convert coordinates to pixel space
    rows, cols = rio.transform.rowcol(source_raster.transform, x, y, op=math.floor)

    # Loop over all coordinates passed
    for k in range(len(rows)):  # type: ignore
        value: float | dict[int, float] | tuple[float | dict[int, float] | tuple[list[float], NDArrayNum] | Any]

        row = rows[k]  # type: ignore
        col = cols[k]  # type: ignore

        # Decide what pixel coordinates to read:
        if window is not None:
            half_win = (window - 1) / 2
            # Subtract start coordinates back to top left of window
            col = col - half_win
            row = row - half_win
            # Offset to read to == window
            width = window
            height = window
        else:
            # Start reading at col,row and read 1px each way
            width = 1
            height = 1

        # If center is out of image, continue and return only NaNs
        if _outside_bounds(
            row,
            col,
            transform=source_raster.transform,
            shape=source_raster.shape,
            area_or_point=source_raster.area_or_point,
        ):
            list_values.append(np.atleast_1d(np.nan))
            if return_window:
                list_windows.append(np.ones((height, width)) * np.nan)
            continue

        # Make sure coordinates are int
        col = int(col)
        row = int(row)

        if True:
            if source_raster.count == 1:
                data = source_raster.data[row : row + height, col : col + width]
            else:
                data = source_raster.data[
                    slice(None) if band is None else band - 1, row : row + height, col : col + width
                ]
            if np.ma.isMaskedArray(data) and not masked:
                data = data.astype(np.float32).filled(np.nan)
            value = format_value(data)
            win: NDArrayNum | dict[int, NDArrayNum] = data

        else:

            # Create rasterio's window for reading
            rio_window = rio.windows.Window(col, row, width, height)

            with rio.open(source_raster.name) as raster:
                data = raster.read(
                    window=rio_window,
                    fill_value=source_raster.nodata,
                    boundless=boundless,
                    masked=masked,
                    indexes=band,
                )
            value = format_value(data)
            win = data

        list_values.append(value)  # type: ignore
        if return_window:
            list_windows.append(win)  # type: ignore

    # If for a single value, unwrap output list
    if input_scalar:
        output_val = list_values[0][0]
        if return_window:
            output_win = list_windows[0]
    else:
        output_val = np.array(list_values)
        output_val = output_val.squeeze()
        if return_window:
            output_win = list_windows  # type: ignore

    # Return array or pointcloud
    from geoutils.pointcloud import (
        PointCloud,  # Runtime import to avoid circularity issues
    )

    if not as_array:
        output_val = PointCloud.from_xyz(x=points[0], y=points[1], z=output_val, crs=source_raster.crs)

    if return_window:
        return (output_val, output_win)
    else:
        return output_val
