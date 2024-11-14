"""
Module for the Xarray accessor "rst" mirroring the API of the Raster class.
"""

import numpy as np
import rasterio as rio
import rioxarray as rioxr
import xarray as xr

import geoutils as gu
from geoutils._typing import NDArrayNum
from geoutils.raster.base import RasterBase, RasterType


def open_raster(filename: str, **kwargs):

    # Open with Rioxarray
    ds = rioxr.open_rasterio(filename, **kwargs)

    # Remove the band dimension if there is only one
    if ds.rio.count == 1:
        del ds.coords["band"]  # Delete band coordinate (only one dimension)

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
