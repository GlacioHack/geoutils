
from typing import Literal, Any, Callable

import pyproj
import numpy as np
from scipy.interpolate import interpn, RegularGridInterpolator, RectBivariateSpline
from scipy.ndimage import map_coordinates
import rasterio as rio

from geoutils._typing import Number, NDArrayNum
from geoutils.raster.georeferencing import _coords, _xy2ij, _outside_image, _res

def _interpn_interpolator(
    coords: tuple[NDArrayNum, NDArrayNum],
    values: NDArrayNum,
    fill_value: Number = np.nan,
    bounds_error: bool = False,
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear") -> \
        Callable[[NDArrayNum, NDArrayNum], NDArrayNum]:
    """
    Mirroring scipy.interpn function but returning interpolator directly: either a RegularGridInterpolator or
    a RectBivariateSpline object. (required for speed when interpolating multiple times)

    From: https://github.com/scipy/scipy/blob/44e4ebaac992fde33f04638b99629d23973cb9b2/scipy/interpolate/_rgi.py#L743
    """

    # Easy for the RegularGridInterpolator
    if method in RegularGridInterpolator._ALL_METHODS:
        interp = RegularGridInterpolator(coords, values, method=method,
                                         bounds_error=bounds_error,
                                         fill_value=fill_value)
        return interp

    # Otherwise need to wrap the fill value around RectBivariateSpline
    elif method == "splinef2d":

        interp = RectBivariateSpline(np.flip(coords[0]), coords[1], np.flip(values[:], axis=0))

        def rectbivariate_interpolator_with_fillvalue(xi):

            # RectBivariateSpline doesn't support fill_value; we need to wrap here
            xi = np.array(xi)
            xi_shape = xi.shape
            xi = xi.reshape(-1, xi.shape[-1])
            idx_valid = np.all((coords[0][-1] <= xi[:, 0], xi[:, 0] <= coords[0][0],
                                coords[1][0] <= xi[:, 1], xi[:, 1] <= coords[1][-1]),
                               axis=0)
            # make a copy of values for RectBivariateSpline
            result = np.empty_like(xi[:, 0])
            result[idx_valid] = interp.ev(xi[idx_valid, 0], xi[idx_valid, 1])
            result[np.logical_not(idx_valid)] = fill_value

            return result.reshape(xi_shape[:-1])

        return rectbivariate_interpolator_with_fillvalue


def _interp_points(
    array: NDArrayNum,
    crs: pyproj.CRS,
    transform: rio.transform.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum],
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
    input_latlon: bool = False,
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    return_interpolator: bool = False,
    **kwargs: Any,
) -> NDArrayNum | Callable:
    """See description of Raster.interp_points."""

    # Get coordinates
    x, y = points

    # If those are in latlon, convert to Raster CRS
    if input_latlon:
        init_crs = pyproj.CRS(4326)
        dest_crs = pyproj.CRS(crs)
        transformer = pyproj.Transformer.from_crs(init_crs, dest_crs)
        x, y = transformer.transform(x, y)

    i, j = _xy2ij(x, y, transform=transform, area_or_point=area_or_point, shift_area_or_point=shift_area_or_point)

    ind_invalid = np.vectorize(lambda k1, k2: _outside_image(k1, k2, transform=transform, area_or_point=area_or_point,
                                                             shape=array.shape, index=True))(j, i)

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
        xycoords = _coords(transform=transform, shape=array.shape, area_or_point=area_or_point,
                           grid=False, shift_area_or_point=shift_area_or_point)

        # Let interpolation outside the bounds not raise any error by default
        if "bounds_error" not in kwargs.keys():
            kwargs.update({"bounds_error": False})
        # Return NaN outside image bounds
        if "fill_value" not in kwargs.keys():
            kwargs.update({"fill_value": np.nan})

        # Using direct coordinates, Y is the first axis, and we need to flip it
        interpolator = _interpn_interpolator(coords=(np.flip(xycoords[1], axis=0), xycoords[0]), values=array,
                                             method=method, bounds_error=kwargs["bounds_error"],
                                             fill_value=kwargs["fill_value"])
        if return_interpolator:
            return interpolator
        else:
            rpoints = interpolator((y, x))

    rpoints = np.array(np.atleast_1d(rpoints), dtype=np.float32)
    rpoints[np.array(ind_invalid)] = np.nan

    return rpoints


