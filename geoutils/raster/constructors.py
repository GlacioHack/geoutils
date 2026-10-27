"""Constructor of raster from arrays (for accessors, otherwise Raster.from_array() is used)."""

from __future__ import annotations

from typing import Any, Literal

import xarray as xr
from affine import Affine
from rasterio.crs import CRS

from geoutils._config import config
from geoutils._typing import MArrayNum, NDArrayBool, NDArrayNum


def from_array(
    data: NDArrayNum | MArrayNum | NDArrayBool,
    transform: tuple[float, ...] | Affine,
    crs: CRS | int | str | None,
    nodata: int | float | None = None,
    area_or_point: Literal["Area", "Point"] | None = None,
    tags: dict[str, Any] | None = None,
    cast_nodata: bool = True,
    *,
    data_name: str | None = None,
    as_object_type: Literal["dataarray", "dataset"] | None = None,
) -> xr.DataArray | xr.Dataset:
    """
    Create a georeferenced raster from a single-band or multiband array.

    A single band requires a two-dimensional array.
    Multiband arrays must have shape (bands, rows, columns), a singleton band dimension is removed.
    Masked values become NaNs, and Dask arrays remain lazy.

    :param data: Raster values with shape (rows, columns) or (bands, rows, columns).
    :param transform: Affine transform, or its six coefficients in Affine order.
    :param crs: Coordinate reference system, or None for unspecified coordinates.
    :param nodata: Value representing missing raster data.
    :param area_or_point: Pixel interpretation stored in AREA_OR_POINT metadata.
    :param tags: Additional raster metadata.
    :param cast_nodata: Nodata conversion option passed to the accessor constructor.
    :param data_name: DataArray or Dataset variable name. An unnamed Dataset variable defaults to raster.
    :param as_object_type: Returned representation: dataarray or dataset. None uses config["raster_return_type"],
        which defaults to dataarray.
    :returns: A DataArray or a Dataset containing one raster variable, with values and georeferencing.
    """
    from geoutils.raster.xr_accessor import DataArrayRasterAccessor

    # Check inputs
    if as_object_type is None:
        as_object_type = config["raster_return_type"]
    if as_object_type not in ("dataarray", "dataset"):
        raise ValueError("Argument 'as_object_type' must be 'dataarray' or 'dataset'.")
    if not isinstance(transform, Affine):
        transform = Affine(*transform)
    raster_crs = None if crs is None else CRS.from_user_input(crs)
    metadata = {} if tags is None else tags.copy()

    # We dispatch to accessor construction
    raster = DataArrayRasterAccessor.from_array(
        data, transform, raster_crs, nodata, area_or_point, metadata, cast_nodata
    )
    if data_name is not None:
        raster = raster.rename(data_name)
    if as_object_type == "dataset":
        return raster.to_dataset(name=raster.name or "raster")
    return raster
