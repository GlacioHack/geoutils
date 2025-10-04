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

"""
Functions for manipulating georeferencing of the raster objects.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Literal

import numpy as np
import rasterio as rio

from geoutils._config import config
from geoutils._typing import ArrayLike, DTypeLike, NDArrayNum


def _ij2xy(
    i: ArrayLike,
    j: ArrayLike,
    transform: rio.transform.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    shift_area_or_point: bool | None = None,
    force_offset: str | None = None,
) -> tuple[NDArrayNum, NDArrayNum]:
    """See description of Raster.ij2xy."""

    # If undefined, default to the global system config
    if shift_area_or_point is None:
        shift_area_or_point = config["shift_area_or_point"]

    # Shift by half a pixel back for "Point" interpretation
    if shift_area_or_point and force_offset is None:
        if area_or_point is not None and area_or_point == "Point":
            i = np.asarray(i) - 0.5
            j = np.asarray(j) - 0.5

    # Default offset is upper-left for raster coordinates
    if force_offset is None:
        force_offset = "ul"

    x, y = rio.transform.xy(transform, i, j, offset=force_offset)

    return x, y


def _xy2ij(
    x: ArrayLike,
    y: ArrayLike,
    transform: rio.transform.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    op: type = np.float32,
    precision: float | None = None,
    shift_area_or_point: bool | None = None,
) -> tuple[NDArrayNum, NDArrayNum]:
    """See description of Raster.xy2ij."""

    # If undefined, default to the global system config
    if shift_area_or_point is None:
        shift_area_or_point = config["shift_area_or_point"]

    # Input checks
    if op not in [np.float32, np.float64, float]:
        raise UserWarning(
            "Operator is not of type float: rio.Dataset.index might "
            "return unreliable indexes due to rounding issues."
        )

    i, j = rio.transform.rowcol(transform, x, y, op=op, precision=precision)

    # Necessary because rio.Dataset.index does not return abc.Iterable for a single point
    if not isinstance(i, Iterable):
        i, j = (
            np.asarray(
                [
                    i,
                ]
            ),
            np.asarray(
                [
                    j,
                ]
            ),
        )
    else:
        i, j = (np.asarray(i), np.asarray(j))

    # AREA_OR_POINT GDAL attribute, i.e. does the value refer to the upper left corner "Area" or
    # the center of pixel "Point". This normally has no influence on georeferencing, it's only
    # about the interpretation of the raster values, and thus can affect sub-pixel interpolation,
    # for more details see: https://gdal.org/user/raster_data_model.html#metadata

    # If the user wants to shift according to the interpretation
    if shift_area_or_point:

        # Shift by half a pixel if the AREA_OR_POINT attribute is "Point", otherwise leave as is
        if area_or_point is not None and area_or_point == "Point":
            if not isinstance(i.flat[0], (np.floating, float)):
                raise ValueError("Operator must return np.floating values to perform pixel interpretation shifting.")

            i += 0.5
            j += 0.5

    # Convert output indexes to integer if they are all whole numbers
    if np.all(np.mod(i, 1) == 0) and np.all(np.mod(j, 1) == 0):
        i = i.astype(int)
        j = j.astype(int)

    return i, j


def _coords(
    transform: rio.transform.Affine,
    shape: tuple[int, int],
    area_or_point: Literal["Area", "Point"] | None,
    grid: bool = True,
    shift_area_or_point: bool | None = None,
    force_offset: str | None = None,
) -> tuple[NDArrayNum, NDArrayNum]:
    """See description of Raster.coords."""

    # The coordinates are extracted from indexes 0 to shape
    _, yy = _ij2xy(
        i=np.arange(shape[0] - 1, -1, -1),
        j=0,
        transform=transform,
        area_or_point=area_or_point,
        shift_area_or_point=shift_area_or_point,
        force_offset=force_offset,
    )
    xx, _ = _ij2xy(
        i=0,
        j=np.arange(shape[1]),
        transform=transform,
        area_or_point=area_or_point,
        shift_area_or_point=shift_area_or_point,
        force_offset=force_offset,
    )

    # If grid is True, return coordinate grids
    if grid:
        meshgrid = tuple(np.meshgrid(xx, np.flip(yy)))
        return meshgrid  # type: ignore
    else:
        return np.asarray(xx), np.asarray(yy)


def _outside_image(
    xi: ArrayLike,
    yj: ArrayLike,
    transform: rio.transform.Affine,
    shape: tuple[int, int],
    area_or_point: Literal["Area", "Point"] | None,
    index: bool = True,
) -> bool:
    """See description of Raster.outside_image."""

    if not index:
        yj, xi = _xy2ij(xi, yj, transform=transform, area_or_point=area_or_point)

    if np.any(np.array((xi, yj)) < 0):
        return True
    elif np.asanyarray(xi) > shape[1] or np.asanyarray(yj) > shape[0]:
        return True
    else:
        return False


def _res(transform: rio.transform.Affine) -> tuple[float, float]:
    """See description of Raster.res"""

    return transform[0], abs(transform[4])


def _bounds(transform: rio.transform.Affine, shape: tuple[int, int]) -> rio.coords.BoundingBox:
    """See description of Raster.bounds."""

    return rio.coords.BoundingBox(*rio.transform.array_bounds(height=shape[0], width=shape[1], transform=transform))


def _cast_pixel_interpretation(
    area_or_point1: Literal["Area", "Point"] | None, area_or_point2: Literal["Area", "Point"] | None
) -> Literal["Area", "Point"] | None:
    """
    Cast two pixel interpretations and warn if not castable.

    Casts to:
     - "Area" if both are "Area",
     - "Point" if both are "Point",
     - None if any of the interpretation is None, or
     - None if one is "Area" and the other "Point" (and raises a warning).
    """

    # If one is None, cast to None
    if area_or_point1 is None or area_or_point2 is None:
        area_or_point_out = None
    # If both are equal and not None
    elif area_or_point1 == area_or_point2:
        area_or_point_out = area_or_point1
    else:
        area_or_point_out = None
        msg = (
            'One raster has a pixel interpretation "Area" and the other "Point". To silence this warning, '
            "either correct the pixel interpretation of one raster, or deactivate "
            'warnings of pixel interpretation with geoutils.config["warn_area_or_point"]=False.'
        )
        if config["warn_area_or_point"]:
            warnings.warn(message=msg, category=UserWarning)

    return area_or_point_out


# Function to set the default nodata values for any given dtype
# Similar to GDAL for int types, but without absurdly long nodata values for floats.
# For unsigned types, the maximum value is chosen (with a max of 99999).
# For signed types, the minimum value is chosen (with a min of -99999).
def _default_nodata(dtype: DTypeLike) -> int:
    """
    Set the default nodata value for any given dtype, when this is not provided.
    """
    default_nodata_lookup = {
        "bool": 255,
        "uint8": 255,
        "int8": -128,
        "uint16": 65535,
        "int16": -32768,
        "uint32": 99999,
        "int32": -99999,
        "uint64": 99999,
        "int64": -99999,
        "float16": -99999,
        "float32": -99999,
        "float64": -99999,
        "float128": -99999,
        "longdouble": -99999,  # This is float64 on Windows, float128 on other systems, for compatibility
    }
    # Check argument dtype is as expected
    if not isinstance(dtype, (str, np.dtype, type)):
        raise TypeError(f"dtype {dtype} not understood.")

    # Convert numpy types to string
    if isinstance(dtype, type):
        dtype = np.dtype(dtype).name

    # Convert np.dtype to string
    if isinstance(dtype, np.dtype):
        dtype = dtype.name

    if dtype in default_nodata_lookup.keys():
        return default_nodata_lookup[dtype]
    else:
        raise NotImplementedError(f"No default nodata value set for dtype {dtype}.")


def _cast_nodata(out_dtype: DTypeLike, nodata: int | float | None) -> int | float | None:
    """
    Cast nodata value for output data type to default nodata if incompatible.

    :param out_dtype: Dtype of output array.
    :param nodata: Nodata value.

    :return: Cast nodata value.
    """

    if out_dtype == bool:
        nodata = None
    if nodata is not None and not rio.dtypes.can_cast_dtype(nodata, out_dtype):
        nodata = _default_nodata(out_dtype)
    else:
        nodata = nodata

    return nodata
