"""
Module for the Xarray accessor "rst" mirroring the API of the Raster class.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import rasterio
import rioxarray as rioxr
import xarray as xr
from affine import Affine
from rasterio.crs import CRS
from rioxarray.rioxarray import affine_to_coords

import geoutils as gu
from geoutils._typing import DTypeLike, MArrayNum, NDArrayBool, NDArrayNum
from geoutils.raster.base import RasterBase


def open_raster(filename: str, is_mask: bool = False, **kwargs: Any) -> xr.DataArray:
    """
    Open a raster using Rioxarray, always masked and squeezed.

    :param filename: Path to the raster file to open.
    :param kwargs: Keyword to pass to rioxarray.open().
    """

    # TODO: Wrap the chunk argument to accept 2D chunks? Right now need to pass 3D even for single-band raster

    # Open with Rioxarray, cast to float32 if integer type
    ds = rioxr.open_rasterio(filename, masked=True, **kwargs)

    # Remove the band dimension if there is only one
    ds = ds.squeeze()  # Delete band coordinate (only one dimension)

    # If input needs to be interpreted as a boolean mask
    if is_mask:
        ds = ds.astype(bool)

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
        Instantiate the raster accessor. This function is called on the first "ds.rst" call of a given DataArray.
        """

        super().__init__()

        # Base instantiation of Xarray accessor
        self._obj: xr.DataArray = xarray_obj

        # We are never returning a DataArray that is unmasked, so the nodata plays a different role than in Rioxarray
        # It is only used for file writing = always the encoded value if it exists
        if self._obj.rio.encoded_nodata is not None:
            # Write encoded nodata as "_FillValue" attribute
            self._obj.rio.write_nodata(self._obj.rio.encoded_nodata, inplace=True)

    @property
    def data(self) -> xr.DataArray:
        # Overloads abstract method in RasterBase
        return self._obj.data

    @data.setter
    def data(self, new_data: xr.DataArray) -> None:
        self._obj.data = new_data

    @property
    def transform(self) -> Affine:
        # Overloads abstract method in RasterBase
        return self._obj.rio.transform()

    @transform.setter
    def transform(self, new_transform: Affine) -> None:
        self.set_transform(new_transform)

    def _set_transform(self, new_transform: Affine) -> None:

        # Rioxarray prioritizes coordinates over transform to re-define transform,
        # so we need to overwrite coordinates
        # See https://github.com/corteva/rioxarray/issues/698
        from rioxarray.rioxarray import affine_to_coords

        # Derive coordinate from new transform
        coords = affine_to_coords(affine=new_transform, width=self._obj.sizes["x"], height=self._obj.sizes["y"])
        # This is lazy (doesn't load data), while calling the arrays directly is
        self._obj = self._obj.assign_coords({"x": coords["x"], "y": coords["y"]})

        # Need to call write transform now
        self._obj.rio.write_transform(new_transform, inplace=True)

    def _set_crs(self, new_crs: CRS) -> None:
        # Overloads abstract method in RasterBase
        self._obj.rio.set_crs(new_crs)

    @property
    def crs(self) -> CRS:
        return self._obj.rio.crs

    @crs.setter
    def crs(self, new_crs: CRS) -> None:
        self.set_crs(new_crs)

    @property
    def nodata(self) -> int | float | None:
        # Overloads abstract method in RasterBase
        return self._obj.attrs.get("_FillValue", None)

    @nodata.setter
    def nodata(self, new_nodata: int | float | None) -> None:
        # self.set_nodata(new_nodata=new_nodata)
        self._nodata = new_nodata
        # Update the Xarray attributes with a new "_FillValue"
        self._obj.rio.write_nodata(self._nodata, inplace=True)

    @property
    def area_or_point(self) -> Literal["Area", "Point"] | None:
        return self._obj.attrs.get("AREA_OR_POINT", None)

    @area_or_point.setter
    def area_or_point(self, new_area_or_point: Literal["Area", "Point"] | None) -> None:
        self.set_area_or_point(new_area_or_point=new_area_or_point)

    def _set_area_or_point(self, new_area_or_point: Literal["Area", "Point"] | None) -> None:
        self._obj.attrs.update({"AREA_OR_POINT": new_area_or_point})

    @property
    def tags(self) -> dict[str, Any]:
        # Overloads abstract method in RasterBase
        return self._obj.attrs

    @tags.setter
    def tags(self, new_tags: dict[str, Any] | None) -> None:
        if new_tags is None:
            new_tags = {}
        self._obj.attrs = new_tags

    @property
    def shape(self) -> tuple[int, int]:
        return self._obj.rio.shape

    @property
    def width(self) -> int:
        return self._obj.rio.width

    @property
    def height(self) -> int:
        return self._obj.rio.height

    @property
    def count(self) -> int:
        return self._obj.rio.count

    @property
    def _count_on_disk(self) -> None | int:
        return None

    @property
    def bands(self) -> tuple[int, ...]:
        if "band" not in self._obj.dims:
            return (1,)
        return tuple(self._obj["band"])

    @property
    def driver(self) -> str | None:
        # Check if driver exists in encoding (inconsistent in Rioxarray)
        xr_driver = self._obj.encoding.get("driver")
        if xr_driver is not None:
            driver = xr_driver
        # Otherwise, if filename exists, get it from Rasterio directly
        elif self.name is not None:
            with rasterio.open(self.name) as ds:
                driver = ds.driver
            # Add it to encoding
            self._obj.encoding.update({"driver": driver})
        else:
            driver = None

        return driver

    @property
    def name(self) -> str | None:
        return self._obj.encoding.get("source")

    @property
    def dtype(self) -> DTypeLike:
        return self._obj.dtype

    @property
    def is_mask(self) -> bool:
        return np.dtype(self.dtype) == np.bool_

    def load(self) -> None:
        self._obj.load()

    def copy(self, new_array: NDArrayNum | None = None, cast_nodata: bool = True, deep: bool = True) -> xr.DataArray:
        """
        Copy the raster in-memory.

        :param new_array: New array to use in the copied raster.
        :param cast_nodata: Automatically cast nodata value to the default nodata for the new array type if not
          compatible. If False, will raise an error when incompatible.


        :return: Copy of the raster.
        """

        # For a Xarray object, all the metadata should be stored (in .attrs, .encoding, or dimensions/variables),
        # so we simply wrap the copy function
        return self._obj.copy(data=new_array, deep=deep)

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

        # Squeeze data
        data = data.squeeze()

        # For a 2-d array
        if data.ndim == 2:
            # Get netCDF coordinates from transform and shape
            coords = affine_to_coords(affine=transform, width=data.shape[0], height=data.shape[1])
            # Need to order the coords as X then Y in dict, or it fails...
            out_ds = xr.DataArray(
                data=data,
                coords={"x": coords["x"], "y": coords["y"]},
                attrs=tags,
            )
        elif data.ndim == 3:
            # Get netCDF coordinates from transform and shape
            coords = affine_to_coords(affine=transform, width=data.shape[1], height=data.shape[2])
            # Need to order the coords as band, then X, then Y in dict, or it fails...
            out_ds = xr.DataArray(
                data=data,
                coords={"band": np.arange(1, data.shape[0] + 1), "x": coords["x"], "y": coords["y"]},
                attrs=tags,
            )

        # Set other attributes
        out_ds.rio.write_transform(transform, inplace=True)
        out_ds.rio.write_crs(crs, inplace=True)
        out_ds.rio.write_nodata(nodata, inplace=True)

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
