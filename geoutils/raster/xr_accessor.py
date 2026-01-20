"""
Module for the Xarray accessor "rst" mirroring the API of the Raster class.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import rioxarray as rioxr
import xarray as xr
from affine import Affine
from rasterio.crs import CRS
from rioxarray.rioxarray import affine_to_coords

import geoutils as gu
from geoutils._typing import MArrayNum, NDArrayBool, NDArrayNum
from geoutils.raster.base import RasterBase


def open_raster(filename: str, **kwargs: Any) -> xr.DataArray:

    # Open with Rioxarray
    ds = rioxr.open_rasterio(filename, masked=False, **kwargs)

    # Cast array to float32 is its dtype is integer (cannot be filled with NaNs otherwise)
    if "int" in str(ds.data.dtype):
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
    """
    This class defines the Xarray accessor 'rst' for rasters.

    Most attributes and functionalities are inherited from the RasterBase class (also parent of the Raster class).
    Only methods specific to the functioning of the Xarray accessor live in this class: mostly initialization, I/O or
    copying.
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        """
        Instantiate the raster accessor.
        """

        super().__init__()

        self._obj: xr.DataArray = xarray_obj
        self._area_or_point = self._obj.attrs.get("AREA_OR_POINT", None)

    def copy(self, new_array: NDArrayNum | None = None, cast_nodata: bool = True) -> xr.DataArray:
        """
        Copy the raster in-memory.

        :param new_array: New array to use in the copied raster.
        :param cast_nodata: Automatically cast nodata value to the default nodata for the new array type if not
          compatible. If False, will raise an error when incompatible.

        :return: Copy of the raster.
        """

        # Define new array
        if new_array is not None:
            data = new_array
        else:
            data = self.data.copy()

        # Send to from_array
        cp = self.from_array(
            data=data,
            transform=self.transform,
            crs=self.crs,
            nodata=self.nodata,
            area_or_point=self.area_or_point,
            tags=self.tags,
            cast_nodata=cast_nodata,
        )

        return cp

    @classmethod
    def from_array(
        cls,
        data: NDArrayNum | MArrayNum | NDArrayBool,
        transform: tuple[float, ...] | Affine,
        crs: CRS | int | None,
        nodata: int | float | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
        tags: dict[str, Any] = None,
        cast_nodata: bool = True,
    ) -> xr.DataArray:

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
        out_ds.rio.write_transform(transform, inplace=True)
        out_ds.rio.set_crs(crs, inplace=True)
        out_ds.rio.set_nodata(nodata, inplace=True)

        return out_ds

    def to_geoutils(self) -> RasterBase:
        """
        Convert to Raster object from GeoUtils.

        :return:
        """
        return gu.Raster.from_array(
            data=self._obj.data,
            crs=self.crs,
            transform=self.transform,
            nodata=self.nodata,
            tags=self.tags,
            area_or_point=self.area_or_point,
        )
