from __future__ import annotations

from typing import Any, Callable, Literal, overload

import numpy as np
import rasterio as rio
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.ndimage import binary_dilation, map_coordinates

from geoutils._typing import NDArrayNum, Number
from geoutils.raster.georeferencing import _coords, _outside_image, _res, _xy2ij

method_to_order = {"nearest": 0, "linear": 1, "cubic": 3, "quintic": 5, "slinear": 1, "pchip": 3, "splinef2d": 3}


def _interpn_interpolator(
    coords: tuple[NDArrayNum, NDArrayNum],
    values: NDArrayNum,
    fill_value: Number = np.nan,
    bounds_error: bool = False,
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
) -> Callable[[tuple[NDArrayNum, NDArrayNum]], NDArrayNum]:
    """
    Create SciPy interpolator with nodata spreading at distance of half the method order rounded up (i.e., linear
    spreads 1 nodata in each direction, cubic spreads 2, quintic 3).

    Gives the exact same result as scipy.interpolate.interpn, and allows interpolator to be re-used if required (
    for speed).
    In practice, returns either a NaN-modified RegularGridInterpolator or a NaN-modified RectBivariateSpline object,
    both expecting a tuple of X/Y coordinates to be evaluated.

    Adapted from:
    https://github.com/scipy/scipy/blob/44e4ebaac992fde33f04638b99629d23973cb9b2/scipy/interpolate/_rgi.py#L743.
    """

    # Adding masking of NaNs for methods not supporting it
    method_support_nan = method in ["nearest"]
    order = method_to_order[method]
    dist_nodata_spread = int(np.ceil(order / 2))

    # If NaNs are not supported
    if not method_support_nan:
        # We compute the mask and dilate it to the order of interpolation (propagating NaNs)
        mask_nan = ~np.isfinite(values)
        new_mask = binary_dilation(mask_nan, iterations=dist_nodata_spread).astype("uint8")

        # We create an interpolator for the mask too, using nearest
        interp_mask = RegularGridInterpolator(
            coords, new_mask, method="nearest", bounds_error=bounds_error, fill_value=1
        )

        # Replace NaN values by nearest neighbour to avoid biasing interpolation near NaNs with placeholder value
        values[mask_nan] = 0

    # For the RegularGridInterpolator
    if method in RegularGridInterpolator._ALL_METHODS:

        # We create the interpolator
        interp = RegularGridInterpolator(
            coords, values, method=method, bounds_error=bounds_error, fill_value=fill_value
        )

        # We create a new interpolator callable
        def regulargrid_interpolator_with_nan(xi: tuple[NDArrayNum, NDArrayNum]) -> NDArrayNum:

            results = interp(xi)

            if not method_support_nan:
                invalids = interp_mask(xi)
                results[invalids.astype(bool)] = np.nan

            return results

        return regulargrid_interpolator_with_nan

    # For the RectBivariateSpline
    else:

        # The coordinates must be in ascending order, which requires flipping the array too (more costly)
        interp = RectBivariateSpline(np.flip(coords[0]), coords[1], np.flip(values[:], axis=0))

        # We create a new interpolator callable
        def rectbivariate_interpolator_with_fillvalue(xi: tuple[NDArrayNum, NDArrayNum]) -> NDArrayNum:

            # Get invalids
            invalids = interp_mask(xi)

            # RectBivariateSpline doesn't support fill_value, so we need to wrap here to add them
            xi_arr = np.array(xi).T
            xi_shape = xi_arr.shape
            xi_arr = xi_arr.reshape(-1, xi_arr.shape[-1])
            idx_valid = np.all(
                (
                    coords[0][-1] <= xi_arr[:, 0],
                    xi_arr[:, 0] <= coords[0][0],
                    coords[1][0] <= xi_arr[:, 1],
                    xi_arr[:, 1] <= coords[1][-1],
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


@overload
def _interp_points(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum],
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    *,
    return_interpolator: Literal[False] = False,
    **kwargs: Any,
) -> NDArrayNum:
    ...


@overload
def _interp_points(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum],
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    *,
    return_interpolator: Literal[True],
    **kwargs: Any,
) -> Callable[[tuple[NDArrayNum, NDArrayNum]], NDArrayNum]:
    ...


@overload
def _interp_points(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum],
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    *,
    return_interpolator: bool = False,
    **kwargs: Any,
) -> NDArrayNum | Callable[[tuple[NDArrayNum, NDArrayNum]], NDArrayNum]:
    ...


def _interp_points(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum],
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    return_interpolator: bool = False,
    **kwargs: Any,
) -> NDArrayNum | Callable[[tuple[NDArrayNum, NDArrayNum]], NDArrayNum]:
    """See description of Raster.interp_points."""

    # Get coordinates
    x, y = points

    i, j = _xy2ij(x, y, transform=transform, area_or_point=area_or_point, shift_area_or_point=shift_area_or_point)

    shape: tuple[int, int] = array.shape[0:2]  # type: ignore
    ind_invalid = np.vectorize(
        lambda k1, k2: _outside_image(k1, k2, transform=transform, area_or_point=area_or_point, shape=shape, index=True)
    )(j, i)

    # If the raster is on an equal grid, use scipy.ndimage.map_coordinates
    force_map_coords = force_scipy_function is not None and force_scipy_function == "map_coordinates"
    force_interpn = force_scipy_function is not None and force_scipy_function == "interpn"

    # Map method name to spline order in map_coordinates, and use only is method compatible
    method_to_order_mapcoords = {"nearest": 0, "linear": 1}
    mapcoords_supported = method in method_to_order_mapcoords.keys()

    res = _res(transform)
    if (res[0] == res[1] or force_map_coords) and not force_interpn and mapcoords_supported:

        # Convert method name into order
        order = method_to_order_mapcoords[method]

        # Remove default spline pre-filtering that is activated by default
        if "prefilter" not in kwargs.keys():
            kwargs.update({"prefilter": False})
        # Change default constant value to NaN for interpolation outside the image bounds
        if "cval" not in kwargs.keys():
            kwargs.update({"cval": np.nan})

        rpoints = map_coordinates(array, [i, j], order=order, **kwargs)

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
            coords=(np.flip(xycoords[1], axis=0), xycoords[0]),
            values=array,
            method=method,
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
