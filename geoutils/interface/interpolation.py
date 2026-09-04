# Copyright (c) 2026 GeoUtils developers
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
import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict, cast, overload

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.ndimage import binary_dilation, distance_transform_edt, map_coordinates

from geoutils._config import config
from geoutils._dispatch import _check_match_points, is_dask_geodataframe
from geoutils._misc import import_optional
from geoutils._typing import DTypeLike, NDArrayBool, NDArrayNum, Number
from geoutils.interface._nodata import NodataPropagation, _validate_nodata_propagation
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.chunked import cached_cumsum, normalize_chunks
from geoutils.multiproc.mparray import block_bounds_from_chunks
from geoutils.projtools import reproject_from_latlon
from geoutils.raster.referencing import _bounds, _coords, _outside_bounds, _res, _xy2ij

method_to_order = {"nearest": 0, "linear": 1, "cubic": 3, "quintic": 5, "slinear": 1, "pchip": 3, "splinef2d": 3}

if TYPE_CHECKING:
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterBase
    from geoutils.raster.raster import Raster


def _interp_output_dtype(dtype: DTypeLike) -> DTypeLike:
    """Return an interpolation dtype that can represent NaNs."""

    return np.float32 if np.issubdtype(dtype, np.integer) else dtype


def _destination_pixel_indices(
    src_transform: rio.transform.Affine,
    dst_transform: rio.transform.Affine,
    dst_shape: tuple[int, int],
) -> tuple[NDArrayNum, NDArrayNum]:
    """
    Return source array indices at the centers of destination pixels.

    :param src_transform: Geotransform of the source array.
    :param dst_transform: Geotransform of the destination array.
    :param dst_shape: Height and width of the destination array.

    :return: Source row and column indices for every destination pixel center.
    """

    # Build destination pixel-center positions without retaining coordinate pairs as Python objects
    dst_cols, dst_rows = np.meshgrid(np.arange(dst_shape[1]) + 0.5, np.arange(dst_shape[0]) + 0.5)
    dst_x = dst_transform.a * dst_cols + dst_transform.b * dst_rows + dst_transform.c
    dst_y = dst_transform.d * dst_cols + dst_transform.e * dst_rows + dst_transform.f

    # Transform coordinates back to source pixels and place array index zero at the first center
    inverse = ~src_transform
    src_cols = inverse.a * dst_x + inverse.b * dst_y + inverse.c - 0.5
    src_rows = inverse.d * dst_x + inverse.e * dst_y + inverse.f - 0.5
    return src_rows, src_cols


def _interpolate_array_band(
    array: NDArrayNum,
    src_rows: NDArrayNum,
    src_cols: NDArrayNum,
    method: Literal["nearest", "linear"],
    nodata_propagation: NodataPropagation,
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int | None = None,
) -> NDArrayNum:
    """
    Interpolate one array band using normalized finite-value weights.

    :param array: Two-dimensional source values.
    :param src_rows: Source row indices of destination pixel centers.
    :param src_cols: Source column indices of destination pixel centers.
    :param method: Nearest-neighbor or linear interpolation.
    :param nodata_propagation: Rule used to handle invalid source values.
    :param dist_nodata_spread: Optional extra distance for spreading invalid cells.

    :return: Interpolated floating-point array.
    """

    # Convert masked and integer inputs to floating values where invalid cells are represented by NaN
    source = np.array(np.ma.getdata(array), dtype=_interp_output_dtype(array.dtype), copy=True)
    if np.ma.isMaskedArray(array):
        source[np.ma.getmaskarray(array)] = np.nan
    valid = np.isfinite(source)

    # Interpolate values and validity separately so invalid neighbors never contribute numerically
    order = 0 if method == "nearest" else 1
    filled = np.where(valid, source, 0)
    numerator = map_coordinates(filled, (src_rows, src_cols), order=order, mode="nearest", prefilter=False)
    weights = map_coordinates(
        valid.astype(np.float32), (src_rows, src_cols), order=order, mode="nearest", prefilter=False
    )

    # Normalize the remaining finite weights as GDAL does for nearest and bilinear resampling
    output = np.full(numerator.shape, np.nan, dtype=_interp_output_dtype(source.dtype))
    np.divide(numerator, weights, out=output, where=weights > 0)

    # Pixel areas extend half a cell beyond their centers but do not extend farther
    inside = (src_rows >= -0.5) & (src_rows < source.shape[0] - 0.5)
    inside &= (src_cols >= -0.5) & (src_cols < source.shape[1] - 0.5)
    output[~inside] = np.nan

    if nodata_propagation == "propagate":
        # A propagated output is invalid when any weighted source value is invalid
        output[weights < 1 - np.finfo(np.float32).eps] = np.nan
    elif nodata_propagation == "gdal":
        # GDAL invalidates an output when its nearest source cell is invalid
        invalid_center = map_coordinates(
            (~valid).astype(np.uint8),
            (src_rows, src_cols),
            order=0,
            mode="nearest",
            prefilter=False,
        )
        output[invalid_center.astype(bool)] = np.nan

    if dist_nodata_spread is not None:
        # An explicit distance adds a predictable mask around invalid cells after interpolation
        distance = _get_dist_nodata_spread(
            order=order,
            dist_nodata_spread=dist_nodata_spread,
        )
        invalid = ~valid
        if distance != 0:
            invalid = binary_dilation(invalid, iterations=distance)
        spread_mask = map_coordinates(
            invalid.astype(np.uint8),
            (src_rows, src_cols),
            order=0,
            mode="nearest",
            prefilter=False,
        )
        output[spread_mask.astype(bool)] = np.nan
    return output


def _interpolate_array(
    array: NDArrayNum,
    src_transform: rio.transform.Affine,
    dst_transform: rio.transform.Affine,
    dst_shape: tuple[int, int] | None = None,
    method: Literal["nearest", "linear", "bilinear"] = "linear",
    nodata_propagation: NodataPropagation = "gdal",
) -> NDArrayNum:
    """
    Interpolate an array onto another grid in the same coordinate reference system.

    This function separates coordinate mapping from value interpolation so it can be reused by same-CRS
    reprojection. The default reproduces GDAL nearest and bilinear nodata behavior, while ``ignore`` always uses
    available finite neighbors and ``propagate`` rejects outputs influenced by an invalid neighbor.

    :param array: Two- or three-dimensional source array, with bands on the first axis.
    :param src_transform: Geotransform of the source array.
    :param dst_transform: Geotransform of the destination array.
    :param dst_shape: Height and width of the destination array. Defaults to the source shape.
    :param method: Nearest-neighbor or linear interpolation. ``bilinear`` is an alias for ``linear``.
    :param nodata_propagation: Rule used to handle invalid source values.

    :return: Interpolated floating-point array.
    """

    # Normalize inputs before building the shared destination-to-source coordinate mapping
    source = np.asanyarray(array)
    if source.ndim not in (2, 3):
        raise ValueError("array must have two or three dimensions.")
    resolved_dst_shape = (source.shape[-2], source.shape[-1]) if dst_shape is None else dst_shape
    normalized_method = "linear" if method == "bilinear" else method
    propagation = _validate_nodata_propagation(nodata_propagation)
    src_rows, src_cols = _destination_pixel_indices(src_transform, dst_transform, resolved_dst_shape)

    # Interpolate each band independently because nodata locations can differ between bands
    if source.ndim == 2:
        return _interpolate_array_band(source, src_rows, src_cols, normalized_method, propagation)
    bands = [
        _interpolate_array_band(source[band], src_rows, src_cols, normalized_method, propagation)
        for band in range(source.shape[0])
    ]
    return np.stack(bands)


# Dask as optional dependency
try:
    import dask
    import dask.array as da
    from dask import delayed
except ImportError:

    da = None

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
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int | None = None,
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = None,
) -> Callable[[tuple[NDArrayNum, NDArrayNum]], NDArrayNum]:
    """
    Create SciPy interpolator with nodata spreading. Default method is linear and default spreading is at distance of
    half the method order rounded up (i.e., linear spreads 1 nodata in each direction, cubic spreads 2, quintic 3).
    They can be configured with the global settings geoutils.config["interpolation_method"] and
    geoutils.config["interpolation_dist_nodata_spread"] respectively.

    Gives the exact same result as scipy.interpolate.interpn, and allows interpolator to be re-used if required (
    for speed).
    In practice, returns either a NaN-modified RegularGridInterpolator or a NaN-modified RectBivariateSpline object,
    both expecting a tuple of X/Y coordinates to be evaluated.

    For input arguments, see scipy.interpolate.RegularGridInterpolator.
    For additional argument "dist_nodata_spread", see description of Raster.interp_points.

    Adapted from:
    https://github.com/scipy/scipy/blob/44e4ebaac992fde33f04638b99629d23973cb9b2/scipy/interpolate/_rgi.py#L743.
    """

    # If interpolation method undefined, default to the global system config
    if method is None:
        method = config["interpolation_method"]

    # If dist_nodata_spread undefined, default to the global system config
    if dist_nodata_spread is None:
        dist_nodata_spread = config["interpolation_dist_nodata_spread"]

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
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int | None = None,
    **kwargs: Any,
) -> NDArrayNum:
    """
    Perform map_coordinates with nodata spreading. Default is spreading at distance of half the method order rounded
    up (i.e., linear spreads 1 nodata in each direction, cubic spreads 2, quintic 3).

    For map_coordinates, only nearest and linear are used.

    For input arguments, see scipy.ndimage.map_coordinates.
    For additional argument "dist_nodata_spread", see description of Raster.interp_points.
    """

    # If dist_nodata_spread undefined, default to the global system config
    if dist_nodata_spread is None:
        dist_nodata_spread = config["interpolation_dist_nodata_spread"]

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
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] | None = None,
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int | None = None,
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    nodata_propagation: NodataPropagation = "gdal",
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
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] | None = None,
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int | None = None,
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    nodata_propagation: NodataPropagation = "gdal",
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
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] | None = None,
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int | None = None,
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    nodata_propagation: NodataPropagation = "gdal",
    *,
    return_interpolator: bool = False,
    **kwargs: Any,
) -> NDArrayNum | Callable[[tuple[NDArrayNum, NDArrayNum]], NDArrayNum]: ...


def _interp_points_base(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum] | None,
    area_or_point: Literal["Area", "Point"] | None = None,
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] | None = None,
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int | None = None,
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    nodata_propagation: NodataPropagation = "gdal",
    return_interpolator: bool = False,
    **kwargs: Any,
) -> NDArrayNum | Callable[[tuple[NDArrayNum, NDArrayNum]], NDArrayNum]:
    # If interpolation method undefined, default to the global system config
    if method is None:
        method = config["interpolation_method"]

    # If array is not a floating dtype (to support NaNs), convert dtype
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    # If array is masked, fill with NaN without copy
    if np.ma.isMaskedArray(array):
        array = array.filled(np.nan)

    # Nearest and linear interpolation share the same finite-weight rules as same-grid interpolation
    propagation = _validate_nodata_propagation(nodata_propagation)
    if method in ("nearest", "linear"):
        normalized_method = cast(Literal["nearest", "linear"], method)

        def interpolate_nearest_or_linear(x: NDArrayNum, y: NDArrayNum) -> NDArrayNum:
            """Interpolate point coordinates with the shared nearest or linear policy."""

            # Convert georeferenced coordinates to array indices before applying the common numeric kernel
            i, j = _xy2ij(
                x,
                y,
                transform=transform,
                area_or_point=area_or_point,
                shift_area_or_point=shift_area_or_point,
            )
            return _interpolate_array_band(
                array=array,
                src_rows=i,
                src_cols=j,
                method=normalized_method,
                nodata_propagation=propagation,
                dist_nodata_spread=dist_nodata_spread,
            )

        if return_interpolator:
            # Interpolators receive coordinates in array-axis order to match SciPy's existing interface
            def point_interpolator(xi: tuple[NDArrayNum, NDArrayNum]) -> NDArrayNum:
                return interpolate_nearest_or_linear(x=np.asarray(xi[1]), y=np.asarray(xi[0]))

            return point_interpolator

        assert points is not None
        return interpolate_nearest_or_linear(x=np.asarray(points[0]), y=np.asarray(points[1]))

    # Higher-order methods retain their configurable mask spread until a GDAL-equivalent kernel is available
    if dist_nodata_spread is None:
        dist_nodata_spread = config["interpolation_dist_nodata_spread"]

    # If the raster is on an equal grid, use scipy.ndimage.map_coordinates
    force_map_coords = force_scipy_function is not None and force_scipy_function == "map_coordinates"
    force_interpn = force_scipy_function is not None and force_scipy_function == "interpn"

    # Map method name to spline order in map_coordinates, and use only is method compatible
    method_to_order_mapcoords = {"nearest": 0, "linear": 1}
    mapcoords_supported = method in method_to_order_mapcoords.keys()

    res = _res(transform)
    use_mapcoords = (
        (res[0] == res[1] or force_map_coords) and not force_interpn and mapcoords_supported and not return_interpolator
    )

    if not return_interpolator:
        assert points is not None
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
            shape=(array.shape[0], array.shape[1]),
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
        scipy_interpolator = _interpn_interpolator(
            points=(np.flip(xycoords[1], axis=0), xycoords[0]),
            values=array,
            method=method,
            dist_nodata_spread=dist_nodata_spread,
            bounds_error=kwargs["bounds_error"],
            fill_value=kwargs["fill_value"],
        )
        if return_interpolator:
            return scipy_interpolator
        else:
            rpoints = scipy_interpolator((y, x))  # type: ignore
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
    xres: float,
    yres: float,
    left: float,
    top: float,
) -> list[list[int]]:
    """Map blocks where each pair of interpolation coordinates will have to be computed."""

    # The argument "starts" contains the list of chunk first X/Y index for the full array, plus the last index
    ny, nx = num_chunks
    y_starts, x_starts = starts

    # We use one bucket per block, assuming a flattened blocks shape
    ind_per_block = [[] for _ in range(ny * nx)]
    for i, (x, y) in enumerate(zip(interp_x, interp_y)):
        # Use actual chunk boundaries because overlap can merge small edge chunks
        xb = int(np.searchsorted(x_starts, (x - left) / xres, side="right") - 1)
        yb = int(np.searchsorted(y_starts, (top - y) / yres, side="right") - 1)

        # Assign outer half pixels to the first block, matching the interpolation kernel's finite support
        if left - xres / 2 <= x < left:
            xb = 0
        if top < y <= top + yres / 2:
            yb = 0

        if 0 <= xb < nx and 0 <= yb < ny:
            ind_per_block[yb * nx + xb].append(i)

    return ind_per_block


@delayed
def _delayed_interp_points_block(
    arr_chunk: NDArrayNum,
    block_id: dict[str, Any],
    interp_coords: NDArrayNum,
    **kwargs: Any,
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
    left, top = bounds.left, bounds.top

    # Expand dask array for overlapping computations
    expanded = da.overlap.overlap(darr, depth=depth, boundary="nearest")

    # Recover core chunk boundaries after any automatic merging required for overlap
    core_chunks = [tuple(size - 2 * depth for size in axis) for axis in expanded.chunks]
    starts = [cached_cumsum(axis, initial_zero=True) for axis in core_chunks]
    num_chunks = expanded.numblocks

    # Get samples indices per blocks
    ind_per_block = _get_interp_indices_per_block(
        points_arr[0, :],
        points_arr[1, :],
        starts,
        num_chunks,
        res[0],
        res[1],
        left,
        top,
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
        _delayed_interp_points_block(blocks[i], block_ids[i], points_arr[:, ind_per_block[i]], **kwargs) for i in used
    ]

    # We concatenate and re-order in a delayed manner
    def _concat_reorder(list_vals, list_inds):  # type: ignore
        # Flatten outputs to 1D and concatenate
        vals = [np.asarray(v).ravel() for v in list_vals]
        vcat = np.concatenate(vals) if vals else np.array([], dtype=np.float32)

        # Build index array and argsort
        inds = (
            np.concatenate([np.asarray(ii, dtype=np.int64) for ii in list_inds])
            if list_inds
            else np.array([], dtype=np.int64)
        )
        order = np.argsort(inds)
        return vcat[order]

    # Get list of indexes only for used blocks
    list_inds_used = [ind_per_block[i] for i in used]
    joined = dask.delayed(_concat_reorder)(list_interp, list_inds_used)

    # Join into one array using a floating type whenever source values cannot represent NaN
    output_dtype = _interp_output_dtype(darr.dtype)
    interp_points = da.from_delayed(joined, shape=(len(points[0]),), dtype=output_dtype)

    # Padded edge chunks repeat their outer cells, so restore the bounds of the complete source raster
    src_rows, src_cols = _xy2ij(
        points[0],
        points[1],
        transform=transform,
        area_or_point=kwargs["area_or_point"],
        shift_area_or_point=kwargs["shift_area_or_point"],
    )
    inside = (src_rows >= -0.5) & (src_rows < darr.shape[0] - 0.5)
    inside &= (src_cols >= -0.5) & (src_cols < darr.shape[1] - 0.5)
    interp_points = da.where(inside, interp_points, np.nan)

    return interp_points


def _empty_pointcloud_meta(data_column: str, crs: Any, dtype: DTypeLike) -> gpd.GeoDataFrame:
    """Build an empty GeoDataFrame for Dask point-cloud outputs."""

    # Dask uses this empty object to infer columns, geometry and data types
    return gpd.GeoDataFrame(
        data={data_column: pd.Series(dtype=dtype)},
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )


def _interp_points_partition(
    part: gpd.GeoDataFrame,
    source_raster: RasterBase,
    interp_options: dict[str, Any],
    extra_kwargs: dict[str, Any],
    data_column: str,
    out_crs: Any,
) -> gpd.GeoDataFrame:
    """Interpolate one point partition and return a point-cloud partition."""

    # Preserve the planned output structure even when Dask sends an empty partition
    out_dtype = _interp_output_dtype(source_raster.dtype)
    if len(part) == 0:
        return _empty_pointcloud_meta(data_column=data_column, crs=out_crs, dtype=out_dtype)

    # Convert partition geometries to the coordinate arrays used by raster interpolation
    x = np.atleast_1d(np.asarray(part.geometry.x.values))
    y = np.atleast_1d(np.asarray(part.geometry.y.values))
    i, j = _xy2ij(
        x,
        y,
        transform=source_raster.transform,
        area_or_point=source_raster.area_or_point,
        shift_area_or_point=interp_options["shift_area_or_point"],
    )
    # Detect partitions with no raster overlap before constructing interpolation work
    # Include outer half pixels accepted by nearest and linear interpolation when selecting partitions
    margin = 0.5 if interp_options["method"] in {"nearest", "linear"} else 0
    ind_outofbounds: NDArrayBool = (i < -margin) | (j < -margin)
    ind_outofbounds |= (i >= source_raster.shape[0] - margin) | (j >= source_raster.shape[1] - margin)

    if np.count_nonzero(~ind_outofbounds) == 0:
        z = np.full(len(part), np.nan, dtype=out_dtype)
    else:
        # Reuse the regular interpolation path within the current point partition
        z = _interp_points(
            source_raster=source_raster,
            points=(x, y),
            as_array=True,
            **interp_options,
            **extra_kwargs,
        )
        # A Dask raster may return a lazy array that must finish inside this task
        if hasattr(z, "compute"):
            z = z.compute()

    # Retain the original geometry and index while adding interpolated values
    return gpd.GeoDataFrame(
        data={data_column: np.asarray(z)},
        geometry=part.geometry,
        crs=out_crs,
        index=part.index,
    )


def _interp_points_dask_pointcloud(
    source_raster: RasterBase,
    points: Any,
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"],
    band: int,
    input_latlon: bool,
    as_array: bool,
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int | None,
    nodata_propagation: NodataPropagation,
    shift_area_or_point: bool | None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None,
    return_interpolator: bool,
    extra_kwargs: dict[str, Any],
) -> Any:
    """Interpolate raster values at a Dask-GeoPandas point cloud."""

    # Reject options whose eager return type cannot be represented by partitions
    if return_interpolator:
        raise ValueError("Option 'return_interpolator' of interp_points cannot be used with Dask point-cloud inputs.")
    if input_latlon:
        raise ValueError("Argument 'input_latlon' is only supported for tuple point inputs.")

    # Import only after identifying a Dask input so the dependency remains optional
    import_optional("dask_geopandas", package_name="dask-geopandas")

    # Reproject lazily so every partition reaches the raster in the same CRS
    out_crs = source_raster.crs
    points_in_crs = points if points.crs == out_crs else points.to_crs(out_crs)
    data_column = "z"
    out_dtype = _interp_output_dtype(source_raster.dtype)
    meta = _empty_pointcloud_meta(data_column=data_column, crs=out_crs, dtype=out_dtype)

    # Package stable interpolation options once for each partition task
    interp_options = {
        "method": method,
        "band": band,
        "input_latlon": False,
        "dist_nodata_spread": dist_nodata_spread,
        "nodata_propagation": nodata_propagation,
        "shift_area_or_point": shift_area_or_point,
        "force_scipy_function": force_scipy_function,
        "return_interpolator": False,
    }
    # Map the eager partition helper while keeping the complete point cloud lazy
    out = points_in_crs.map_partitions(
        _interp_points_partition,
        source_raster,
        interp_options,
        extra_kwargs,
        data_column,
        out_crs,
        meta=meta,
    )
    # Import at runtime to avoid the point-cloud base importing this interpolation module in return
    from geoutils.pointcloud.base import _set_dataframe_attrs

    # Restore the metadata expected by the GeoUtils ``pc`` accessor
    _set_dataframe_attrs(
        out,
        {
            "crs": out_crs,
            "bounds": None,
            "point_count": None,
            "data_column": data_column,
            "geometry_type": "Point",
        },
    )

    if as_array:
        # Expose values as a Dask array without computing point partitions
        return out[data_column].to_dask_array(lengths=True)
    return out


# SAME WITH MULTIPROCESSING


def _wrapper_multiproc_interp_per_block(
    rst: Raster,
    block_id: dict[str, Any],
    interp_coords: NDArrayNum,
    **kwargs: Any,
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
    rst: RasterBase,
    points: tuple[NDArrayNum, NDArrayNum],
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
    left, top = bounds.left, bounds.top

    # Get multiprocessing chunk sizes
    chunks = normalize_chunks(chunks=config.chunks, shape=rst.shape)

    # Get starting 2D index for each chunk of the full array
    # (mirroring what is done in block_id of dask.array.map_blocks)
    tiling = block_bounds_from_chunks(chunks=chunks, shape=rst.shape, overlap=depth)
    starts = [
        cached_cumsum(chunks[0], initial_zero=True),
        cached_cumsum(chunks[1], initial_zero=True),
    ]
    num_chunks = (tiling.shape[0], tiling.shape[1])
    num_blocks = np.prod(num_chunks)

    # Get samples indices per blocks
    ind_per_block = _get_interp_indices_per_block(
        points_arr[0, :],
        points_arr[1, :],
        starts,  # type: ignore
        num_chunks,
        res[0],
        res[1],
        left,
        top,
    )

    # Build the block IDs by unravelling starting indexes for each block
    indexes_xi, indexes_yi = np.unravel_index(np.arange(num_blocks), shape=(num_chunks[0], num_chunks[1]))
    block_ids = [{"tile_idx": tiling[indexes_xi[i], indexes_yi[i], :]} for i in range(num_blocks)]

    # Create tasks for multiprocessing
    tasks = []
    for i in range(len(block_ids)):
        # Launch the task on the cluster to process each tile
        tasks.append(
            config.cluster.submit(
                _wrapper_multiproc_interp_per_block,
                rst,
                block_ids[i],
                points_arr[:, ind_per_block[i]],
                **kwargs,
            )
        )

    # Collect results
    try:
        list_interp = []
        # Iterate over the tasks and retrieve the processed results
        for results in tasks:
            interp = config.cluster.compute(results)
            list_interp.append(interp)
    except Exception as e:
        raise RuntimeError(f"Error retrieving interpolated segments from multiprocessing tasks: {e}")

    # Concatenate outputs
    interp_points = np.concatenate(list_interp, axis=0)

    # Re-order per-block output points to match their original indices
    indices = np.concatenate(ind_per_block).astype(int)
    argsort = np.argsort(indices)
    interp_points = np.array(interp_points)[argsort]

    return interp_points


# MAIN API FUNCTION CHECKING USER INPUTS AND DISPATCHING TO BASE, DASK OR MULTIPROCESSING


def _interp_points(
    source_raster: RasterBase,
    points: tuple[NDArrayNum, NDArrayNum] | tuple[Number, Number] | PointCloudLike,
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = None,
    band: int = 1,
    input_latlon: bool = False,
    as_array: bool = False,
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int | None = None,
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    return_interpolator: bool = False,
    mp_config: MultiprocConfig | None = None,
    nodata_propagation: NodataPropagation = "gdal",
    **kwargs: Any,
) -> Any:
    """See description of Raster.interp_points."""

    # If interpolation method undefined, default to the global system config
    if method is None:
        method = config["interpolation_method"]

    propagation = _validate_nodata_propagation(nodata_propagation)

    # 1/ Input checks
    if is_dask_geodataframe(points):
        if mp_config is not None:
            raise ValueError("Dask point-cloud inputs cannot be combined with Multiprocessing interpolation.")
        return _interp_points_dask_pointcloud(
            source_raster=source_raster,
            points=points,
            method=method,
            band=band,
            input_latlon=input_latlon,
            as_array=as_array,
            dist_nodata_spread=dist_nodata_spread,
            nodata_propagation=propagation,
            shift_area_or_point=shift_area_or_point,
            force_scipy_function=force_scipy_function,
            return_interpolator=return_interpolator,
            extra_kwargs=kwargs,
        )

    # Check and normalize input points
    pts_xy, input_scalar = _check_match_points(source_raster, points)

    # Extract raster metadata for later checks and conversions
    transform = source_raster.transform
    area_or_point = source_raster.area_or_point
    shape = source_raster.shape

    # Convert from latlon if necessary
    pts = pts_xy
    if input_latlon:
        pts = reproject_from_latlon(pts_xy, out_crs=source_raster.crs)

    # If we evaluate points (not returning interpolator), remove those outside of bounds
    # (Out of bounds points are hard to deal with for chunked operations otherwise)
    pts_inbounds: tuple[NDArrayNum, NDArrayNum] | None
    if not return_interpolator:
        if pts is None:
            raise ValueError("Input 'points' cannot be None if 'return_interpolator' is False.")
        x0, y0 = pts
        # Normalize to 1D arrays for typing + uniform downstream logic
        x: NDArrayNum = np.atleast_1d(np.asarray(x0))
        y: NDArrayNum = np.atleast_1d(np.asarray(y0))

        i, j = _xy2ij(x, y, transform=transform, area_or_point=area_or_point, shift_area_or_point=shift_area_or_point)

        # Retain the outer half pixels accepted by nearest and linear array interpolation
        margin = 0.5 if method in {"nearest", "linear"} else 0
        ind_outofbounds: NDArrayBool = (i < -margin) | (j < -margin)
        ind_outofbounds |= (i >= shape[0] - margin) | (j >= shape[1] - margin)

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
    class _InterpKwargs(TypedDict):
        area_or_point: Literal["Area", "Point"] | None
        method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"]
        dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int | None
        nodata_propagation: NodataPropagation
        shift_area_or_point: bool | None
        force_scipy_function: Literal["map_coordinates", "interpn"] | None
        return_interpolator: bool

    interp_kwargs: _InterpKwargs = {
        "area_or_point": area_or_point,
        "method": method,
        "dist_nodata_spread": dist_nodata_spread,
        "nodata_propagation": propagation,
        "shift_area_or_point": shift_area_or_point,
        "force_scipy_function": force_scipy_function,
        "return_interpolator": return_interpolator,
    }

    # Cannot use Multiprocessing backend and Dask backend simultaneously
    mp_backend = mp_config is not None
    # The check below can only run on Xarray
    dask_backend = da is not None and source_raster._chunks is not None

    if mp_backend and dask_backend:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove mp_config parameter "
            "from interp_points(). To use Multiprocessing, open the file without chunks."
        )

    if (dask_backend or mp_backend) and (return_interpolator or pts_inbounds is None):
        raise ValueError(
            "Option 'return_interpolator' of interp_points cannot be used with Dask or Multiprocessing, "
            "only with in-memory array."
        )

    # If using Multiprocessing backend, process and return NumPy array (ragged output)
    if mp_backend:
        assert mp_config is not None
        assert pts_inbounds is not None
        # Temporary switch bands
        orig_bands = source_raster.bands
        source_raster._bands = (band,)
        z_inbounds = _multiproc_interp_points(
            rst=source_raster, points=pts_inbounds, config=mp_config, **interp_kwargs, **kwargs
        )
        # Rewrite original bands
        source_raster._bands = orig_bands
    # For both Dask and NumPy array:
    else:
        if source_raster.data.ndim != 2:
            arr = source_raster.data[band - 1, :, :]
        else:
            arr = source_raster.data
        # If using Dask backend, process and return a lazy array with one value per point
        if dask_backend:
            assert pts_inbounds is not None
            z_inbounds = _dask_interp_points(
                darr=arr, transform=transform, points=pts_inbounds, **interp_kwargs, **kwargs
            )
        # If using direct reprojection, process and return NumPy array
        else:
            z_inbounds = _interp_points_base(
                array=arr, transform=transform, points=pts_inbounds, **interp_kwargs, **kwargs  # type: ignore
            )

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
        def _rebuild_with_nans(z_inbounds: NDArrayNum, mask_out: NDArrayBool, n: int, dtype: DTypeLike) -> NDArrayNum:
            out = np.full(n, np.nan, dtype=_interp_output_dtype(dtype))
            out[~mask_out] = z_inbounds
            return out

        if dask_backend:
            out_del = dask.delayed(_rebuild_with_nans)(z_inbounds, ind_outofbounds, n, dtype)
            z = da.from_delayed(out_del, shape=(n,), dtype=_interp_output_dtype(dtype))
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
