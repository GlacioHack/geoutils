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

from typing import Any, Callable, Literal, overload

import numpy as np
import rasterio as rio
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.ndimage import binary_dilation, distance_transform_edt, map_coordinates

from geoutils import profiler
from geoutils._typing import NDArrayNum, Number
from geoutils.raster.georeferencing import _coords, _outside_image, _res, _xy2ij

method_to_order = {"nearest": 0, "linear": 1, "cubic": 3, "quintic": 5, "slinear": 1, "pchip": 3, "splinef2d": 3}


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


@overload
def _interp_points(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum],
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int = "half_order_up",
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    *,
    return_interpolator: Literal[False] = False,
    **kwargs: Any,
) -> NDArrayNum: ...


@overload
def _interp_points(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum],
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int = "half_order_up",
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    *,
    return_interpolator: Literal[True],
    **kwargs: Any,
) -> Callable[[tuple[NDArrayNum, NDArrayNum]], NDArrayNum]: ...


@overload
def _interp_points(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum],
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int = "half_order_up",
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    *,
    return_interpolator: bool = False,
    **kwargs: Any,
) -> NDArrayNum | Callable[[tuple[NDArrayNum, NDArrayNum]], NDArrayNum]: ...


@profiler.profile("geoutils.interface.interpolate._interp_points", memprof=True)  # type: ignore
def _interp_points(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum] | None,
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int = "half_order_up",
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    return_interpolator: bool = False,
    **kwargs: Any,
) -> NDArrayNum | Callable[[tuple[NDArrayNum, NDArrayNum]], NDArrayNum]:
    """See description of Raster.interp_points."""

    # If array is not a floating dtype (to support NaNs), convert dtype
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    shape: tuple[int, int] = array.shape[0:2]  # type: ignore

    # TODO: Add check about None for "points" depending on "return_interpolator"
    if not return_interpolator:
        if points is None:
            raise ValueError("Input 'points' cannot be None if 'return_interpolator' is False.")
        x, y = points
        i, j = _xy2ij(x, y, transform=transform, area_or_point=area_or_point, shift_area_or_point=shift_area_or_point)

        ind_invalid = np.vectorize(
            lambda k1, k2: _outside_image(
                k1, k2, transform=transform, area_or_point=area_or_point, shape=shape, index=True
            )
        )(j, i)

    # If the raster is on an equal grid, use scipy.ndimage.map_coordinates
    force_map_coords = force_scipy_function is not None and force_scipy_function == "map_coordinates"
    force_interpn = force_scipy_function is not None and force_scipy_function == "interpn"

    # Map method name to spline order in map_coordinates, and use only is method compatible
    method_to_order_mapcoords = {"nearest": 0, "linear": 1}
    mapcoords_supported = method in method_to_order_mapcoords.keys()

    res = _res(transform)
    if (res[0] == res[1] or force_map_coords) and not force_interpn and mapcoords_supported and not return_interpolator:

        # Convert method name into order
        order = method_to_order_mapcoords[method]

        # Remove default spline pre-filtering that is activated by default
        if "prefilter" not in kwargs.keys():
            kwargs.update({"prefilter": False})
        # Change default constant value to NaN for interpolation outside the image bounds
        if "cval" not in kwargs.keys():
            kwargs.update({"cval": np.nan})

        # Use map coordinates with nodata propagation
        rpoints = _map_coordinates_nodata_propag(
            values=array, indices=(i, j), order=order, dist_nodata_spread=dist_nodata_spread, **kwargs
        )

    # Otherwise, use scipy.interpolate.interpn
    else:
        # Get lower-left corner coordinates
        xycoords = _coords(
            transform=transform,
            shape=shape,
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

    rpoints = np.array(np.atleast_1d(rpoints), dtype=np.float32)
    rpoints[np.array(ind_invalid)] = np.nan

    return rpoints
