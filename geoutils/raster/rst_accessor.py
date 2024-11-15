"""
Module for the Xarray accessor "rst" mirroring the API of the Raster class.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import rasterio as rio
from affine import Affine
from rasterio.crs import CRS
import rioxarray as rioxr
from rioxarray.rioxarray import affine_to_coords
import xarray as xr

import geoutils as gu
from geoutils.raster.base import RasterBase

from geoutils._typing import NDArrayNum, NDArrayBool, MArrayNum

def open_raster(filename: str, **kwargs):

    # Open with Rioxarray
    ds = rioxr.open_rasterio(filename, masked=False, **kwargs)

    # Change all nodata to NaNs
    ds = ds.astype(dtype=np.float32)
    ds.data[ds.data == ds.rio.nodata] = np.nan

    # Remove the band dimension if there is only one
    if ds.rio.count == 1:
        ds = ds.isel(band=0, drop=True)  # Delete band coordinate (only one dimension)

    # Store disk attributes? Didn't follow the logic in Rioxarray on this, need to get more into code details
    ds.rst._count_on_disk = 1

    return ds


@xr.register_dataarray_accessor("rst")
class RasterAccessor(RasterBase):
    def __init__(self, xarray_obj: xr.DataArray):

        super().__init__()

        self._obj = xarray_obj
        self._area_or_point = self._obj.attrs.get("AREA_OR_POINT", None)

    def copy(self, new_array: NDArrayNum | None = None) -> xr.DataArray:

        return self._obj.copy(data=new_array)

    def from_array(
        self,
        data: NDArrayNum | MArrayNum | NDArrayBool,
        transform: tuple[float, ...] | Affine,
        crs: CRS | int | None,
        nodata: int | float | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
        tags: dict[str, Any] = None,
        cast_nodata: bool = True,) -> xr.DataArray:

        # Add area_or_point
        if tags is None:
            tags = {}
        if area_or_point is not None:
            tags.update({"AREA_OR_POINT": area_or_point})

        data = data.squeeze()

        # Get netCDF coordinates from transform and shape
        coords = affine_to_coords(affine=transform, width=data.shape[0], height=data.shape[1])

        # Build a data array
        out_ds = xr.DataArray(
            data=data,
            coords={"x": coords["x"], "y": coords["y"]},  # Somehow need to re-order the coords...
            attrs=tags,
        )

        # Set other attributes
        out_ds.rio.write_transform(transform)
        out_ds.rio.set_crs(crs)
        out_ds.rio.set_nodata(nodata)

        return out_ds



    def to_raster(self) -> RasterBase:
        """
        Convert to Raster object.

        :return:
        """
        return gu.Raster.from_array(data=self._obj.data, crs=self.crs, transform=self.transform, nodata=self.nodata)


@xr.register_dataarray_accessor("sat")
class SatelliteImageAccessor(RasterAccessor):
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj
