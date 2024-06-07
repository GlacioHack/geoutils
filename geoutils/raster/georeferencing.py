from typing import Iterable, Literal

import numpy as np
import rasterio as rio

from geoutils._typing import ArrayLike, NDArrayNum
from geoutils._config import config

def _ij2xy(
    i: ArrayLike,
    j: ArrayLike,
    transform: rio.transform.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    shift_area_or_point: bool | None = None,
    force_offset: str | None = None
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
                raise ValueError(
                    "Operator must return np.floating values to perform pixel interpretation shifting."
                )

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
    force_offset: str | None = None
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
        force_offset=force_offset
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
    index: bool = True) -> bool:
    """See description of Raster.outside_image."""

    if not index:
        xi, xj = _xy2ij(xi, yj, transform=transform, area_or_point=area_or_point)

    if np.any(np.array((xi, yj)) < 0):
        return True
    elif np.asanyarray(xi) > shape[1] or np.asanyarray(yj) > shape[0]:
        return True
    else:
        return False

def _res(transform: rio.transform.Affine):
    """See description of Raster.res"""

    return transform[0], abs(transform[4])