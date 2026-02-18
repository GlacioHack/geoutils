# Copyright (c) 2025 GeoUtils developers
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES)
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
Module for Raster class.
"""

from __future__ import annotations

import copy
import math
import pathlib
import warnings
from collections import abc
from contextlib import ExitStack
from typing import IO, TYPE_CHECKING, Any, Callable, overload

import numpy as np
import rasterio as rio
import rasterio.windows
import rioxarray
import xarray as xr
from affine import Affine
from packaging.version import Version
from rasterio.crs import CRS

import geoutils as gu
from geoutils import profiler
from geoutils._misc import deprecate, import_optional
from geoutils._typing import (
    ArrayLike,
    DTypeLike,
    MArrayNum,
    NDArrayBool,
    NDArrayNum,
    Number,
)
from geoutils.projtools import reproject_from_latlon, reproject_points
from geoutils.raster.base import RasterBase, RasterType
from geoutils.raster.georeferencing import (
    _cast_nodata,
    _cast_pixel_interpretation,
    _default_nodata,
)
from geoutils.raster.satimg import (
    decode_sensor_metadata,
    parse_and_convert_metadata_from_filename,
)

# If python38 or above, Literal is builtin. Otherwise, use typing_extensions
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

if TYPE_CHECKING:
    import matplotlib

# List of NumPy "array" functions that are handled.
# Note: all universal function are supported: https://numpy.org/doc/stable/reference/ufuncs.html
# Array functions include: NaN math and stats, classic math and stats, logical, sorting/counting:
_HANDLED_FUNCTIONS_1NIN = (
    # NaN math: https://numpy.org/doc/stable/reference/routines.math.html
    # and NaN stats: https://numpy.org/doc/stable/reference/routines.statistics.html
    [
        "nansum",
        "nanmax",
        "nanmin",
        "nanargmax",
        "nanargmin",
        "nanmean",
        "nanmedian",
        "nanpercentile",
        "nanvar",
        "nanstd",
        "nanprod",
        "nancumsum",
        "nancumprod",
        "nanquantile",
    ]
    # Classic math and stats (same links as above)
    + [
        "sum",
        "amax",
        "amin",
        "max",
        "min",
        "argmax",
        "argmin",
        "mean",
        "median",
        "percentile",
        "var",
        "std",
        "prod",
        "cumsum",
        "cumprod",
        "quantile",
        "abs",
        "absolute",
        "gradient",
    ]
    # Sorting, searching and counting: https://numpy.org/doc/stable/reference/routines.sort.html
    + ["sort", "count_nonzero", "unique"]
    # Logic functions: https://numpy.org/doc/stable/reference/routines.logic.html
    + ["all", "any", "isfinite", "isinf", "isnan", "logical_not"]
)

_HANDLED_FUNCTIONS_2NIN = [
    "logical_and",
    "logical_or",
    "logical_xor",
    "allclose",
    "isclose",
    "array_equal",
    "array_equiv",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "equal",
    "not_equal",
]
handled_array_funcs = _HANDLED_FUNCTIONS_1NIN + _HANDLED_FUNCTIONS_2NIN

# Set default attributes to be kept from rasterio's DatasetReader
_default_rio_attrs = [
    "bounds",
    "count",
    "crs",
    "driver",
    "height",
    "indexes",
    "name",
    "nodata",
    "res",
    "shape",
    "transform",
    "width",
    "profile",
]


def _load_rio(
    dataset: rio.io.DatasetReader,
    only_mask: bool = False,
    convert_to_mask: bool = False,
    indexes: int | list[int] | None = None,
    masked: bool = False,
    transform: Affine | None = None,
    shape: tuple[int, int] | None = None,
    out_count: int | None = None,
    **kwargs: Any,
) -> MArrayNum:
    r"""
    Load specific bands of the dataset, using :func:`rasterio.read`.

    Ensure that ``self.data.ndim=3`` for ease of use (needed e.g. in show).

    :param dataset: Dataset to read (opened with :func:`rasterio.open`).
    :param only_mask: Read only the dataset mask.
    :param convert_to_mask: Whether to convert to mask after reading.
    :param indexes: Band(s) to load. Note that rasterio begins counting at 1, not 0.
    :param masked: Whether the mask should be read (if any exists) to use the nodata to mask values.
    :param transform: Create a window from the given transform (to read only parts of the raster)
    :param shape: Expected shape of the read ndarray. Must be given together with the `transform` argument.
    :param out_count: Specify the count for a downsampled version (to be used with kwargs out_shape).

    :raises ValueError: If only one of ``transform`` and ``shape`` are given.

    :returns: An unmasked array if ``masked`` is ``False``, or a masked array otherwise.

    \*\*kwargs: any additional arguments to rasterio.io.DatasetReader.read.
    Useful ones are:
    .. hlist::
    * out_shape : to load a downsampled version, always use with out_count
    * window : to load a cropped version
    * resampling : to set the resampling algorithm
    """
    # If out_shape is passed, no need to account for transform and shape
    if kwargs.get("out_shape") is not None:
        window = None
        # If multi-band raster, the out_shape needs to contain the count
        if out_count is not None and out_count > 1:
            kwargs["out_shape"] = (out_count, *kwargs["out_shape"])
    else:
        if transform is not None and shape is not None:
            if transform == dataset.transform:
                row_off, col_off = 0, 0
            else:
                row_off, col_off = (round(val) for val in dataset.index(transform[2], abs(transform[4])))

            window = rio.windows.Window(col_off, row_off, *shape[::-1])
        elif sum(param is None for param in [shape, transform]) == 1:
            raise ValueError("If 'shape' or 'transform' is provided, BOTH must be given.")

    if indexes is None:
        if only_mask:
            data = dataset.read_masks(window=window, **kwargs)
        else:
            data = dataset.read(masked=masked, window=window, **kwargs)
    else:
        if only_mask:
            data = dataset.read_masks(indexes=indexes, window=window, **kwargs)
        else:
            data = dataset.read(indexes=indexes, masked=masked, window=window, **kwargs)

    if convert_to_mask:
        return data.astype(bool)

    return data


def _cast_numeric_array_raster(
    raster: RasterType, other: RasterType | NDArrayNum | Number, operation_name: str
) -> tuple[MArrayNum, MArrayNum | NDArrayNum | Number, float | int | None, Literal["Area", "Point"] | None]:
    """
    Cast a raster and another raster or array or number to arrays with proper metadata, or raise an error message.

    :param raster: Raster.
    :param other: Raster or array or number.
    :param operation_name: Name of operation to raise in the error message.

    :return: Returns array objects, nodata value and pixel interpretation.
    """

    # Check first input is a raster
    if not isinstance(raster, Raster):
        raise ValueError("Developer error: Only a raster should be passed as first argument to this function.")

    # Check that other is of correct type
    # If not, a NotImplementedError should be raised, in case other's class has a method implemented.
    # See https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
    if not isinstance(other, (Raster, np.ndarray, float, int, np.floating, np.integer)):
        raise NotImplementedError(
            f"Operation between an object of type {type(other)} and a Raster impossible. Must be a Raster, "
            f"np.ndarray or single number."
        )

    # If other is a raster
    if isinstance(other, Raster):

        nodata2 = other.nodata
        dtype2 = other.data.dtype
        other_data: NDArrayNum | MArrayNum | Number = other.data

        # Check that both rasters have the same shape and georeferences
        if raster.georeferenced_grid_equal(other):  # type: ignore
            pass
        else:
            raise ValueError(
                "Both rasters must have the same shape, transform and CRS for " + operation_name + ". "
                "For example, use raster1 = raster1.reproject(raster2) to reproject raster1 on the "
                "same grid and CRS than raster2."
            )

    # If other is an array
    elif isinstance(other, np.ndarray):

        # Squeeze first axis of other data if possible
        if other.ndim == 3 and other.shape[0] == 1:
            other_data = other.squeeze(axis=0)
        else:
            other_data = other
        nodata2 = None
        dtype2 = other.dtype

        if raster.shape == other_data.shape:
            pass
        else:
            raise ValueError(
                "The raster and array must have the same shape for " + operation_name + ". "
                "For example, if the array comes from another raster, use raster1 = "
                "raster1.reproject(raster2) beforehand to reproject raster1 on the same grid and CRS "
                "than raster2. Or, if the array does not come from a raster, define one with raster = "
                "Raster.from_array(array, array_transform, array_crs, array_nodata) then reproject."
            )

    # If other is a single number
    else:
        other_data = other
        nodata2 = None
        dtype2 = rio.dtypes.get_minimum_dtype(other_data)

    # Get raster dtype and nodata
    nodata1 = raster.nodata
    dtype1 = raster.data.dtype

    # 1/ Output nodata depending on common data type
    out_dtype = np.promote_types(dtype1, dtype2)

    out_nodata = None
    # Priority to nodata of first raster if both match the output dtype
    if (nodata2 is not None) and (out_dtype == dtype2):
        out_nodata = nodata2
    elif (nodata1 is not None) and (out_dtype == dtype1):
        out_nodata = nodata1
    # In some cases the promoted output type does not match any inputs
    # (e.g. for inputs "uint8" and "int8", output is "int16")
    elif (nodata1 is not None) or (nodata2 is not None):
        out_nodata = nodata1 if nodata1 is not None else nodata2

    # 2/ Output pixel interpretation
    if isinstance(other, Raster):
        area_or_point = _cast_pixel_interpretation(raster.area_or_point, other.area_or_point)
    else:
        area_or_point = raster.area_or_point

    return raster.data, other_data, out_nodata, area_or_point


class Raster(RasterBase):
    """
    The georeferenced raster.

    Main attributes:
        data: :class:`np.ndarray`
            Data array of the raster, with dimensions corresponding to (count, height, width).
        transform: :class:`affine.Affine`
            Geotransform of the raster.
        crs: :class:`pyproj.crs.CRS`
            Coordinate reference system of the raster.
        nodata: :class:`int` or :class:`float`
            Nodata value of the raster.

    All other attributes are derivatives of those attributes, or read from the file on disk.
    See the API for more details.
    """

    @profiler.profile("geoutils.raster.raster.__init__", memprof=True)  # type: ignore
    def __init__(
        self,
        filename_or_dataset: (
            str | pathlib.Path | RasterType | rio.io.DatasetReader | rio.io.MemoryFile | dict[str, Any]
        ),
        bands: int | list[int] | None = None,
        is_mask: bool = False,
        load_data: bool = False,
        parse_sensor_metadata: bool = False,
        silent: bool = True,
        downsample: Number = 1,
        force_nodata: int | float | None = None,
    ) -> None:
        """
        Instantiate a raster from a filename or rasterio dataset.

        :param filename_or_dataset: Path to file or Rasterio dataset.
        :param bands: Band(s) to load into the object. Default loads all bands.
        :param load_data: Whether to load the array during instantiation. Default is False.
        :param parse_sensor_metadata: Whether to parse sensor metadata from filename and similarly-named metadata files.
        :param silent: Whether to parse metadata silently or with console output.
        :param downsample: Downsample the array once loaded by a round factor. Default is no downsampling.
        :param force_nodata: Force nodata value to be used (overwrites the metadata). Default reads from metadata.
        """

        # Sets attributes as None in RasterBase
        super().__init__()

        # Only attributes defined from instantiation
        self._nodata = force_nodata
        self._bands = bands
        self._is_mask = is_mask
        self._masked = True

        # This is for Raster.from_array to work.
        if isinstance(filename_or_dataset, dict):

            self.tags = filename_or_dataset["tags"]
            # To have "area_or_point" user input go through checks of the set() function without shifting the transform
            self.set_area_or_point(filename_or_dataset["area_or_point"], shift_area_or_point=False)

            # Need to set nodata before the data setter, which uses it
            # We trick set_nodata into knowing the data type by setting self._disk_dtype, then unsetting it
            # (as a raster created from an array doesn't have a disk dtype)
            if np.dtype(filename_or_dataset["data"].dtype) != bool:  # Exception for Mask class
                self._disk_dtype = filename_or_dataset["data"].dtype
                self.set_nodata(filename_or_dataset["nodata"], update_array=False, update_mask=False)
                self._disk_dtype = None

            # Then, we can set the data, transform and crs
            self.data = filename_or_dataset["data"]
            self.transform: rio.transform.Affine = filename_or_dataset["transform"]
            self.crs: rio.crs.CRS = filename_or_dataset["crs"]

            for key in filename_or_dataset:
                if key in ["data", "transform", "crs", "nodata", "area_or_point", "tags"]:
                    continue
                setattr(self, key, filename_or_dataset[key])
            return

        # If Raster is passed, simply point back to Raster
        if isinstance(filename_or_dataset, Raster):
            for key in filename_or_dataset.__dict__:
                setattr(self, key, filename_or_dataset.__dict__[key])

        # Image is a file on disk.
        elif isinstance(filename_or_dataset, (str, pathlib.Path, rio.io.DatasetReader, rio.io.MemoryFile)):
            # ExitStack is used instead of "with rio.open(filename_or_dataset) as ds:".
            # This is because we might not actually want to open it like that, so this is equivalent
            # to the pseudocode:
            # "with rio.open(filename_or_dataset) as ds if isinstance(filename_or_dataset, str) else ds:"
            # This is slightly black magic, but it works!
            with ExitStack():
                if isinstance(filename_or_dataset, (str, pathlib.Path)):
                    ds: rio.io.DatasetReader = rio.open(filename_or_dataset)
                elif isinstance(filename_or_dataset, rio.io.DatasetReader):
                    ds = filename_or_dataset
                # This is if it's a MemoryFile
                else:
                    ds = filename_or_dataset.open()
                    # In that case, data has to be loaded
                    load_data = True

                self._transform = ds.transform
                self._crs = ds.crs
                # Allow user to manually override the nodata value which may be specified in the file.
                if force_nodata is not None:
                    self._nodata = force_nodata
                else:
                    self._nodata = ds.nodata
                self._name = ds.name
                self._driver = ds.driver
                self._tags.update(ds.tags())
                self._profile = ds.profile

                # For tags saved from sensor metadata, convert from string to practical type (datetime, etc)
                converted_tags = decode_sensor_metadata(self.tags)
                self._tags.update(converted_tags)
                # Add image structure in tags
                self._tags.update(ds.tags(ns="IMAGE_STRUCTURE"))

                self._area_or_point = self.tags.get("AREA_OR_POINT", None)

                self._disk_shape = (ds.count, ds.height, ds.width)
                self._disk_bands = ds.indexes
                self._disk_dtype = ds.dtypes[0]
                self._disk_transform = ds.transform

            # Check number of bands to be loaded
            if bands is None:
                count = self.count
            elif isinstance(bands, int):
                count = 1
            else:
                count = len(bands)

            # Downsampled image size
            if not isinstance(downsample, (int, float)):
                raise TypeError("downsample must be of type int or float.")
            if downsample < 1:
                raise ValueError("downsample must be >=1.")

            if downsample == 1:
                out_shape = (self.height, self.width)
            else:
                down_width = int(np.ceil(self.width / downsample))
                down_height = int(np.ceil(self.height / downsample))
                out_shape = (down_height, down_width)
                res = tuple(np.asarray(self.res) * downsample)
                self.transform = rio.transform.from_origin(self.bounds.left, self.bounds.top, res[0], res[1])
                self._downsample = downsample

            # This will record the downsampled out_shape is data is only loaded later on by .load()
            self._out_shape = out_shape
            self._out_count = count

            if load_data:
                # Mypy doesn't like the out_shape for some reason. I can't figure out why! (erikmannerfelt, 14/01/2022)
                # Don't need to pass shape and transform, because out_shape overrides it
                self.data = _load_rio(
                    ds,
                    indexes=bands,
                    masked=self._masked,
                    convert_to_mask=is_mask,
                    out_shape=out_shape,
                    out_count=count,
                )  # type: ignore

            # Probably don't want to use set_nodata that can update array, setting self._nodata is sufficient
            # Set nodata only if data is loaded
            # if nodata is not None and self._data is not None:
            #     self.set_nodata(self._nodata)

        # Provide a catch in case trying to load from data array
        elif isinstance(filename_or_dataset, np.ndarray):
            raise TypeError("The filename is an array, did you mean to call Raster.from_array(...) instead?")

        # Don't recognise the input, so stop here.
        else:
            raise TypeError("The filename argument is not recognised, should be a path or a Rasterio dataset.")

        # Parse metadata and add to tags
        if parse_sensor_metadata and self.name is not None:
            sensor_meta = parse_and_convert_metadata_from_filename(self.name, silent=silent)
            self._tags.update(sensor_meta)

    @property
    def data(self) -> MArrayNum:
        # Overloads abstract method in RasterBase
        if not self.is_loaded:
            self.load()
        return self._data  # type: ignore

    @data.setter
    def data(self, new_data: NDArrayNum | MArrayNum) -> None:
        """
        Set the contents of .data and possibly update .nodata.

        The data setter behaviour is the following:

        1. Writes the data in a masked array, whether the input is a classic array or a masked_array,
        2. Reshapes the data to a 2D array if it is single band,
        3. If new dtype is different from Raster and nodata is not compatible, casts the nodata value to new dtype,
        4. Sets a new nodata value to the Raster if none is set and if the provided array contains non-finite values
            that are unmasked (including if there is no mask at all, e.g. NaNs in a classic array),
        5. Masks non-finite values that are unmasked, whether the input is a classic array or a masked_array. Note that
            these values are not overwritten and can still be accessed in .data.data.

        :param new_data: New data to assign to this instance of Raster.

        """
        # Check that new_data is a NumPy array
        if not isinstance(new_data, np.ndarray):
            raise ValueError("New data must be a numpy array.")

        if new_data.ndim not in [2, 3]:
            raise ValueError("Data array must have 2 or 3 dimensions.")

        # Squeeze 3D data if the band axis is of length 1
        if new_data.ndim == 3 and new_data.shape[0] == 1:
            new_data = new_data.squeeze(axis=0)

        # Check that new_data has correct shape

        # If data is loaded
        if self._data is not None:
            dtype = str(self._data.dtype)
            orig_shape = self._data.shape
        # If filename exists
        elif self._disk_dtype is not None:
            dtype = str(self._disk_dtype)
            if self._out_count == 1:
                orig_shape = self._out_shape
            else:
                orig_shape = (self._out_count, *self._out_shape)  # type: ignore
        else:
            dtype = str(new_data.dtype)
            orig_shape = new_data.shape

        if new_data.shape != orig_shape:
            raise ValueError(
                f"New data must be of the same shape as existing data: {orig_shape}. Given: {new_data.shape}."
            )

        # Cast nodata if the new array has incompatible dtype with the old nodata value
        # (we accept setting an array with new dtype to mirror NumPy behaviour)
        self._nodata = _cast_nodata(new_data.dtype, self.nodata)

        # If the new data is not masked and has non-finite values, we define a default nodata value
        if (not np.ma.is_masked(new_data) and self.nodata is None and np.count_nonzero(~np.isfinite(new_data)) > 0) or (
            np.ma.is_masked(new_data)
            and self.nodata is None
            and np.count_nonzero(~np.isfinite(new_data.data[~new_data.mask])) > 0
        ):
            warnings.warn(
                "Setting default nodata {:.0f} to mask non-finite values found in the array, as "
                "no nodata value was defined.".format(_default_nodata(dtype)),
                category=UserWarning,
            )
            self._nodata = _default_nodata(dtype)

        # Now comes the important part, the data setting!
        # Several cases to consider:

        # 1/ If the new data is not a masked array and contains non-finite values such as NaNs, define a mask
        if not np.ma.isMaskedArray(new_data):

            # Have to write it this way, because wrapper np.ma.mask_invalid always creates a boolean array,
            # instead of attributing nomask (mask = False, single boolean) when no invalids exist
            mask = ~np.isfinite(new_data)
            if not mask.any():
                m = np.ma.array(new_data, mask=np.ma.nomask, copy=True, fill_value=self.nodata)
            else:
                m = np.ma.array(new_data, mask=mask, copy=True, fill_value=self.nodata)
            self._data = m

        elif not np.ma.is_masked(new_data) and np.count_nonzero(~np.isfinite(new_data)) > 0:
            self._data = np.ma.masked_array(
                data=np.asarray(new_data), mask=~np.isfinite(new_data.data), fill_value=self.nodata
            )

        # 2/ If the new data is masked but some non-finite values aren't masked, add them to the mask
        elif np.ma.is_masked(new_data) and np.count_nonzero(~np.isfinite(new_data.data[~new_data.mask])) > 0:
            self._data = np.ma.masked_array(
                data=new_data.data,
                mask=np.logical_or(~np.isfinite(new_data.data), new_data.mask),
                fill_value=self.nodata,
            )

        # 3/ If the new data is a Masked Array, we pass data.data and data.mask independently (passing directly the
        # masked array to data= has a strange behaviour that redefines fill_value)
        elif np.ma.isMaskedArray(new_data):
            self._data = np.ma.masked_array(data=new_data.data, mask=new_data.mask, fill_value=self.nodata)

        # 4/ If the new data is classic ndarray
        else:
            self._data = np.ma.masked_array(data=new_data, fill_value=self.nodata)

        # Finally, mask values equal to the nodata value in case they weren't masked, but raise a warning
        if np.count_nonzero(np.logical_and(~self._data.mask, self._data.data == self.nodata)) > 0:
            # This can happen during a numerical operation, especially for integer values that max out with a modulo
            # It can also happen with from_array()
            warnings.warn(
                category=UserWarning,
                message="Unmasked values equal to the nodata value found in data array. They are now masked.\n "
                "If this happened when creating or updating the array, to silence this warning, "
                "convert nodata values in the array to np.nan or mask them with np.ma.masked prior "
                "to creating or updating the raster.\n"
                "If this happened during a numerical operation, use astype() prior to the operation "
                "to convert to a data type that won't derive the nodata values (e.g., a float type).",
            )
            self._data[self._data.data == self.nodata] = np.ma.masked

    def _set_transform(self, new_transform: Affine) -> None:
        # Overloads abstract method in RasterBase
        self._transform = new_transform

    @property
    def transform(self) -> Affine:
        # Overloads abstract method in RasterBase
        return self._transform

    @transform.setter
    def transform(self, new_transform: Affine) -> None:
        self.set_transform(new_transform)

    def _set_crs(self, new_crs: CRS) -> None:
        # Overloads abstract method in RasterBase
        self._crs = new_crs

    @property
    def crs(self) -> CRS:
        # Overloads abstract method in RasterBase
        return self._crs

    @crs.setter
    def crs(self, new_crs: CRS) -> None:
        self.set_crs(new_crs)

    @property
    def nodata(self) -> int | float | None:
        # Overloads abstract method in RasterBase
        return self._nodata

    @nodata.setter
    def nodata(self, new_nodata: int | float | None) -> None:
        self.set_nodata(new_nodata=new_nodata)

    @property
    def tags(self) -> dict[str, Any]:
        return self._tags

    @tags.setter
    def tags(self, new_tags: dict[str, Any] | None) -> None:
        if new_tags is None:
            new_tags = {}
        self._tags = new_tags

    @property
    def area_or_point(self) -> Literal["Area", "Point"] | None:
        return self._area_or_point

    @area_or_point.setter
    def area_or_point(self, new_area_or_point: Literal["Area", "Point"] | None) -> None:
        self.set_area_or_point(new_area_or_point=new_area_or_point)

    def _set_area_or_point(self, new_area_or_point: Literal["Area", "Point"] | None) -> None:
        self._area_or_point = new_area_or_point
        # Update tag only if not None
        if new_area_or_point is not None:
            self.tags.update({"AREA_OR_POINT": new_area_or_point})
        else:
            if "AREA_OR_POINT" in self.tags:
                self.tags.pop("AREA_OR_POINT")

    @property
    def _count_on_disk(self) -> None | int:
        if self._disk_shape is not None:
            return self._disk_shape[0]
        return None

    @property
    def count(self) -> int:
        if self.is_loaded:
            if self.data.ndim == 2:
                return 1
            else:
                return int(self.data.shape[0])
        #  This can only happen if data is not loaded, with a DatasetReader on disk is open, never returns None
        if self._bands is not None:
            if isinstance(self._bands, (int, np.integer)):
                return 1
            else:
                return len(self._bands)
        return self._count_on_disk  # type: ignore

    @property
    def height(self) -> int:
        if not self.is_loaded:
            if self._out_shape is not None:
                return self._out_shape[0]
            else:
                return self._disk_shape[1]  # type: ignore
        else:
            # If the raster is single-band
            if self.data.ndim == 2:
                return int(self.data.shape[0])
            # Or multi-band
            else:
                return int(self.data.shape[1])

    @property
    def width(self) -> int:
        if not self.is_loaded:
            if self._out_shape is not None:
                return self._out_shape[1]
            else:
                return self._disk_shape[2]  # type: ignore
        else:
            # If the raster is single-band
            if self.data.ndim == 2:
                return int(self.data.shape[1])
            # Or multi-band
            else:
                return int(self.data.shape[2])

    @property
    def shape(self) -> tuple[int, int]:
        # If a downsampling argument was defined but data not loaded yet
        if self._out_shape is not None and not self.is_loaded:
            return self._out_shape
        # If data loaded or not, pass the disk/data shape through height and width
        return self.height, self.width

    @property
    def bands(self) -> tuple[int, ...]:
        if self._bands is not None and not self.is_loaded:
            if isinstance(self._bands, (int, np.integer)):
                return (self._bands,)
            return tuple(self._bands)
        # if self._indexes_loaded is not None:
        #     if isinstance(self._indexes_loaded, int):
        #         return (self._indexes_loaded, )
        #     return tuple(self._indexes_loaded)
        if self.is_loaded:
            return tuple(range(1, self.count + 1))
        return self._bands_on_disk  # type: ignore

    @property
    def driver(self) -> str | None:
        return self._driver

    @property
    def profile(self) -> dict[str, Any] | None:
        """Basic metadata and creation options of this dataset.
        May be passed as keyword arguments to rasterio.open()
        to create a clone of this dataset."""
        return self._profile

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def dtype(self) -> DTypeLike:
        if not self.is_loaded and self._disk_dtype is not None:
            return self._disk_dtype
        return self.data.dtype

    @property
    def is_mask(self) -> bool:
        # Follow user input if not loaded
        if not self.is_loaded:
            return self._is_mask
        # Otherwise check data type
        else:
            return np.dtype(self.dtype) == np.bool_

    def _load_only_mask(self, bands: int | list[int] | None = None, **kwargs: Any) -> NDArrayBool:
        """
        Load only the raster mask from disk and return as independent array (not stored in any class attributes).

        :param bands: Band(s) to load. Note that rasterio begins counting at 1, not 0.
        :param kwargs: Optional keyword arguments sent to '_load_rio()'.

        :raises ValueError: If the data are already loaded.
        :raises AttributeError: If no 'filename' attribute exists.
        """
        if self.is_loaded:
            raise ValueError("Data are already loaded.")

        if self.name is None:
            raise AttributeError("Cannot load as name is not set anymore. Did you manually update the name attribute?")

        out_count = self._out_count
        # If no index is passed, use all of them
        if bands is None:
            valid_bands = self.bands
        # If a new index was pass, redefine out_shape
        elif isinstance(bands, (int, list)):
            # Rewrite properly as a tuple
            if isinstance(bands, int):
                valid_bands = (bands,)
            else:
                valid_bands = tuple(bands)
            # Update out_count if out_shape exists (when a downsampling has been passed)
            if self._out_shape is not None:
                out_count = len(valid_bands)

        # If a downsampled out_shape was defined during instantiation
        with rio.open(self.name) as dataset:
            mask = _load_rio(
                dataset,
                only_mask=True,
                indexes=list(valid_bands),
                masked=self._masked,
                transform=self.transform,
                shape=self.shape,
                out_shape=self._out_shape,
                out_count=out_count,
                **kwargs,
            )

        # Rasterio says the mask should be returned in 2D for a single band but it seems not
        mask = mask.squeeze()

        # Valid data is equal to 255, invalid is equal to zero (see Rasterio doc)
        mask_bool = mask == 0

        return mask_bool

    def load(self, bands: int | list[int] | None = None, **kwargs: Any) -> None:
        """
        Load the raster array from disk.

        :param bands: Band(s) to load. Note that rasterio begins counting at 1, not 0.
        :param kwargs: Optional keyword arguments sent to '_load_rio()'.

        :raises ValueError: If the data are already loaded.
        :raises AttributeError: If no 'name' attribute exists.
        """
        if self.is_loaded:
            raise ValueError("Data are already loaded.")

        if self.name is None:
            raise AttributeError("Cannot load as name is not set anymore. Did you manually update the name attribute?")

        # If no index is passed, use all of them
        if bands is None:
            valid_bands = self.bands
        # If a new index was pass, redefine out_shape
        elif isinstance(bands, (int, list)):
            # Rewrite properly as a tuple
            if isinstance(bands, int):
                valid_bands = (bands,)
            else:
                valid_bands = tuple(bands)
            # Update out_count if out_shape exists (when a downsampling has been passed)
            if self._out_shape is not None:
                self._out_count = len(valid_bands)

        # Save which bands are loaded
        self._bands_loaded = valid_bands

        # If a downsampled out_shape was defined during instantiation
        with rio.open(self.name) as dataset:
            self.data = _load_rio(
                dataset,
                indexes=list(valid_bands),
                masked=self._masked,
                convert_to_mask=self._is_mask,
                transform=self.transform,
                shape=self.shape,
                out_shape=self._out_shape,
                out_count=self._out_count,
                **kwargs,
            )

        # Probably don't want to use set_nodata() that updates the array
        # Set nodata value with the loaded array
        # if self._nodata is not None:
        #     self.set_nodata(nodata=self._nodata)

    @classmethod
    def from_array(
        cls: type[RasterType],
        data: NDArrayNum | MArrayNum | NDArrayBool,
        transform: tuple[float, ...] | Affine,
        crs: CRS | int | None,
        nodata: int | float | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
        tags: dict[str, Any] = None,
        cast_nodata: bool = True,
    ) -> RasterType:
        """Create a raster from a numpy array and the georeferencing information.

        Expects a 2D (single band) or 3D (multi-band) array of the raster. The first axis corresponds to bands.

        :param data: Input array, 2D for single band or 3D for multi-band (bands should be first axis).
        :param transform: Affine 2D transform. Either a tuple(x_res, 0.0, top_left_x,
            0.0, y_res, top_left_y) or an affine.Affine object.
        :param crs: Coordinate reference system. Any CRS supported by Pyproj (e.g., CRS object, EPSG integer).
        :param nodata: Nodata value.
        :param area_or_point: Pixel interpretation of the raster, will be stored in AREA_OR_POINT metadata.
        :param tags: Metadata stored in a dictionary.
        :param cast_nodata: Automatically cast nodata value to the default nodata for the new array type if not
            compatible. If False, will raise an error when incompatible.

        :returns: Raster created from the provided array and georeferencing.

        Example:

            You have a data array in EPSG:32645. It has a spatial resolution of
            30 m in x and y, and its top left corner is X=478000, Y=3108140.

            >>> data = np.ones((500, 500), dtype="uint8")
            >>> transform = (30.0, 0.0, 478000.0, 0.0, -30.0, 3108140.0)
            >>> myim = Raster.from_array(data, transform, 32645)
        """
        # Define tags as empty dictionary if not defined
        if tags is None:
            tags = {}

        # Cast nodata if the new array has incompatible type with the old nodata value
        if cast_nodata:
            nodata = _cast_nodata(data.dtype, nodata)

        return cls(
            {
                "data": data,
                "transform": transform,
                "crs": crs,
                "nodata": nodata,
                "area_or_point": area_or_point,
                "tags": tags,
            }
        )

    def to_rio_dataset(self) -> rio.io.DatasetReader:
        """Export to a Rasterio in-memory dataset."""

        # Create handle to new memory file
        mfh = rio.io.MemoryFile()

        # Write info to the memory file
        with rio.open(
            mfh,
            "w",
            height=self.height,
            width=self.width,
            count=self.count,
            dtype=self.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=self.nodata,
            driver="GTiff",
        ) as ds:
            if self.count == 1:
                ds.write(self.data[np.newaxis, :, :])
            else:
                ds.write(self.data)

        # Then open as a DatasetReader
        return mfh.open()

    def __repr__(self) -> str:
        """Convert raster to string representation."""

        # If data not loaded, return and string and avoid calling .data
        if not self.is_loaded:
            str_data = "not_loaded; shape on disk " + str(self._disk_shape)
            if self._out_shape is not None:
                # Shape to load
                if self.count == 1:
                    shape_to_load = self._out_shape
                else:
                    shape_to_load = (self._out_count, *self._out_shape)  # type: ignore
                str_data = str_data + "; will load " + str(shape_to_load)
        else:
            str_data = "\n       ".join(self.data.__str__().split("\n"))

        # Above and below, we align left spaces for multi-line object representation (arrays) after return to lines
        # And call the class name for easier inheritance to subclasses (avoid overloading to just rewrite subclass name)
        s = (
            self.__class__.__name__
            + "(\n"
            + "  data="
            + str_data
            + "\n  transform="
            + "\n            ".join(self.transform.__str__().split("\n"))
            + "\n  crs="
            + self.crs.__str__()
            + "\n  nodata="
            + self.nodata.__str__()
            + ")"
        )

        return str(s)
        # L = [getattr(self, item) for item in self._saved_attrs]
        # s: str = "{}.{}({})".format(type(self).__module__, type(self).__qualname__, ", ".join(map(str, L)))

        # return s

    def _repr_html_(self) -> str:
        """Convert raster to HTML representation for documentation."""

        # If data not loaded, return and string and avoid calling .data
        if not self.is_loaded:
            str_data = "<i>not_loaded; shape on disk " + str(self._disk_shape)
            if self._out_shape is not None:
                # Shape to load
                if self._out_count == 1:
                    shape_to_load = self._out_shape
                else:
                    shape_to_load = (self._out_count, *self._out_shape)  # type: ignore
                str_data = str_data + "; will load " + str(shape_to_load) + "</i>"

        else:
            str_data = "\n       ".join(self.data.__str__().split("\n"))

        # Use <pre> to keep white spaces, <span> to keep line breaks
        s = (
            '<pre><span style="white-space: pre-wrap"><b><em>'
            + self.__class__.__name__
            + "</em></b>(\n"
            + "  <b>data=</b>"
            + str_data
            + "\n  <b>transform=</b>"
            + "\n            ".join(self.transform.__str__().split("\n"))
            + "\n  <b>crs=</b>"
            + self.crs.__str__()
            + "\n  <b>nodata=</b>"
            + self.nodata.__str__()
            + ")</span></pre>"
        )

        return str(s)

    def __str__(self) -> str:
        """Provide simplified raster string representation for print()."""

        if not self.is_loaded:
            s = "not_loaded"
        else:
            s = self.data.__str__()

        return str(s)

    def __getitem__(self, index: Raster | NDArrayBool | Any) -> NDArrayNum | Raster:
        """
        Index the raster.

        In addition to all index types supported by NumPy, also supports a mask of same georeferencing or a
        boolean array of the same shape as the raster.
        """

        if isinstance(index, (Raster, np.ndarray)):
            _cast_numeric_array_raster(self, index, operation_name="an indexing operation")  # type: ignore

        # If input is Mask with the same shape and georeferencing
        if isinstance(index, Raster):
            if not index.is_mask:
                ind = index.astype(bool)
                warnings.warn(message="Input raster was cast to boolean for indexing.", category=UserWarning)
            else:
                ind = index
            if self.count == 1:
                return self.data[ind.data.squeeze()]
            else:
                return self.data[:, ind.data.squeeze()]
        # If input is array with the same shape
        elif isinstance(index, np.ndarray):
            if str(index.dtype) != "bool":
                index = index.astype(bool)
                warnings.warn(message="Input array was cast to boolean for indexing.", category=UserWarning)
            if self.count == 1:
                return self.data[index]
            else:
                return self.data[:, index]

        # Otherwise, use any other possible index and leave it to NumPy
        else:
            return self.data[index]

    def __setitem__(self, index: Raster | NDArrayBool | Any, assign: NDArrayNum | Number) -> None:
        """
        Perform index assignment on the raster.

        In addition to all index types supported by NumPy, also supports a mask of same georeferencing or a
        boolean array of the same shape as the raster.
        """

        # First, check index
        if isinstance(index, (Raster, np.ndarray)):
            _cast_numeric_array_raster(self, index, operation_name="an index assignment operation")  # type: ignore

        # If input is Mask with the same shape and georeferencing
        if isinstance(index, Raster):
            if not index.is_mask:
                index = index.astype(bool)
                warnings.warn(message="Input raster was cast to boolean for indexing.", category=UserWarning)
            ind = index.data.data
            use_all_bands = False
        # If input is array with the same shape
        elif isinstance(index, np.ndarray):
            if str(index.dtype) != "bool":
                ind = index.astype(bool)
                warnings.warn(message="Input array was cast to boolean for indexing.", category=UserWarning)
            else:
                ind = index
            use_all_bands = False
        # Otherwise, use the index, NumPy will raise appropriate errors itself
        else:
            ind = index
            use_all_bands = True

        # Second, assign the data, here also let NumPy do the job

        # We need to explicitly load here, as we cannot call the data getter/setter directly
        if not self.is_loaded:
            self.load()

        # Assign the values to the index (single band raster with mask/array, or other NumPy index)
        if self.count == 1 or use_all_bands:
            self._data[ind] = assign  # type: ignore
        # For multi-band rasters with a mask/array
        else:
            self._data[:, ind] = assign  # type: ignore
        return None

    def __add__(self: RasterType, other: RasterType | NDArrayNum | Number) -> RasterType:
        """
        Sum two rasters, or a raster and a numpy array, or a raster and single number.

        If other is a Raster, it must have the same shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        # Check inputs and return compatible data, output dtype and nodata value
        self_data, other_data, nodata, aop = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"
        )

        # Run calculation
        out_data = self_data + other_data

        # Save to output Raster
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata, area_or_point=aop)

        return out_rst

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __radd__(self: RasterType, other: NDArrayNum | Number) -> RasterType:  # type: ignore
        """
        Sum two rasters, or a raster and a numpy array, or a raster and single number.

        For when other is first item in the operation (e.g. 1 + rst).
        """
        return self.__add__(other)  # type: ignore

    def __neg__(self: RasterType) -> RasterType:
        """
        Take the raster negation.

        Returns a raster with -self.data.
        """
        return self.copy(-self.data)

    def __sub__(self, other: Raster | NDArrayNum | Number) -> Raster:
        """
        Subtract two rasters, or a raster and a numpy array, or a raster and single number.

        If other is a Raster, it must have the same shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, nodata, aop = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"
        )
        out_data = self_data - other_data
        return self.from_array(out_data, self.transform, self.crs, nodata=nodata, area_or_point=aop)

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __rsub__(self: RasterType, other: NDArrayNum | Number) -> RasterType:  # type: ignore
        """
        Subtract two rasters, or a raster and a numpy array, or a raster and single number.

        For when other is first item in the operation (e.g. 1 - rst).
        """
        self_data, other_data, nodata, aop = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"
        )
        out_data = other_data - self_data
        return self.from_array(out_data, self.transform, self.crs, nodata=nodata, area_or_point=aop)

    def __mul__(self: RasterType, other: RasterType | NDArrayNum | Number) -> RasterType:
        """
        Multiply two rasters, or a raster and a numpy array, or a raster and single number.

        If other is a Raster, it must have the same shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, nodata, aop = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"
        )
        out_data = self_data * other_data
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata, area_or_point=aop)
        return out_rst

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __rmul__(self: RasterType, other: NDArrayNum | Number) -> RasterType:  # type: ignore
        """
        Multiply two rasters, or a raster and a numpy array, or a raster and single number.

        For when other is first item in the operation (e.g. 2 * rst).
        """
        return self.__mul__(other)  # type: ignore

    def __truediv__(self: RasterType, other: RasterType | NDArrayNum | Number) -> RasterType:
        """
        True division of two rasters, or a raster and a numpy array, or a raster and single number.

        If other is a Raster, it must have the same shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, nodata, aop = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"
        )
        out_data = self_data / other_data
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata, area_or_point=aop)
        return out_rst

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __rtruediv__(self: RasterType, other: NDArrayNum | Number) -> RasterType:  # type: ignore
        """
        True division of two rasters, or a raster and a numpy array, or a raster and single number.

        For when other is first item in the operation (e.g. 1/rst).
        """
        self_data, other_data, nodata, aop = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"
        )
        out_data = other_data / self_data
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata, area_or_point=aop)
        return out_rst

    def __floordiv__(self: RasterType, other: RasterType | NDArrayNum | Number) -> RasterType:
        """
        Floor division of two rasters, or a raster and a numpy array, or a raster and single number.

        If other is a Raster, it must have the same shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, nodata, aop = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"
        )
        out_data = self_data // other_data
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata, area_or_point=aop)
        return out_rst

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __rfloordiv__(self: RasterType, other: NDArrayNum | Number) -> RasterType:  # type: ignore
        """
        Floor division of two rasters, or a raster and a numpy array, or a raster and single number.

        For when other is first item in the operation (e.g. 1/rst).
        """
        self_data, other_data, nodata, aop = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"
        )
        out_data = other_data // self_data
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata, area_or_point=aop)
        return out_rst

    def __mod__(self: RasterType, other: RasterType | NDArrayNum | Number) -> RasterType:
        """
        Modulo of two rasters, or a raster and a numpy array, or a raster and single number.

        If other is a Raster, it must have the same shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, nodata, aop = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"
        )
        out_data = self_data % other_data  # type: ignore
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata, area_or_point=aop)
        return out_rst

    def __pow__(self: RasterType, power: int | float) -> RasterType:
        """
        Power of a raster to a number.
        """
        # Check that input is a number
        if not isinstance(power, (float, int, np.floating, np.integer)):
            raise ValueError("Power needs to be a number.")

        # Calculate the product of arrays and save to new Raster
        out_data = self.data**power
        nodata = self.nodata
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata, area_or_point=self.area_or_point)
        return out_rst

    def __eq__(self: RasterType, other: RasterType | NDArrayNum | Number) -> RasterType:  # type: ignore
        """
        Element-wise equality of two rasters, or a raster and a numpy array, or a raster and single number.

        This operation casts the result into a Mask.

        If other is a Raster, it must have the same shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, _, aop = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"
        )
        out_data = self_data == other_data
        out_mask = self.from_array(out_data, self.transform, self.crs, nodata=None, area_or_point=aop)
        return out_mask

    def __ne__(self: RasterType, other: RasterType | NDArrayNum | Number) -> RasterType:  # type: ignore
        """
        Element-wise negation of two rasters, or a raster and a numpy array, or a raster and single number.

        This operation casts the result into a Mask.

        If other is a Raster, it must have the same shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, _, aop = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"
        )
        out_data = self_data != other_data
        out_mask = self.from_array(out_data, self.transform, self.crs, nodata=None, area_or_point=aop)
        return out_mask

    def __lt__(self: RasterType, other: RasterType | NDArrayNum | Number) -> RasterType:
        """
        Element-wise lower than comparison of two rasters, or a raster and a numpy array,
        or a raster and single number.

        This operation casts the result into a Mask.

        If other is a Raster, it must have the same shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, _, aop = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"
        )
        out_data = self_data < other_data
        out_mask = self.from_array(out_data, self.transform, self.crs, nodata=None, area_or_point=aop)
        return out_mask

    def __le__(self: RasterType, other: RasterType | NDArrayNum | Number) -> RasterType:
        """
        Element-wise lower or equal comparison of two rasters, or a raster and a numpy array,
        or a raster and single number.

        This operation casts the result into a Mask.

        If other is a Raster, it must have the same shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, _, aop = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"
        )
        out_data = self_data <= other_data
        out_mask = self.from_array(out_data, self.transform, self.crs, nodata=None, area_or_point=aop)
        return out_mask

    def __gt__(self: RasterType, other: RasterType | NDArrayNum | Number) -> RasterType:
        """
        Element-wise greater than comparison of two rasters, or a raster and a numpy array,
        or a raster and single number.

        This operation casts the result into a Mask.

        If other is a Raster, it must have the same shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, _, aop = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"
        )
        out_data = self_data > other_data
        out_mask = self.from_array(out_data, self.transform, self.crs, nodata=None, area_or_point=aop)
        return out_mask

    def __ge__(self: RasterType, other: RasterType | NDArrayNum | Number) -> RasterType:
        """
        Element-wise greater or equal comparison of two rasters, or a raster and a numpy array,
        or a raster and single number.

        This operation casts the result into a Mask.

        If other is a Raster, it must have the same shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, _, aop = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"
        )
        out_data = self_data >= other_data
        out_mask = self.from_array(out_data, self.transform, self.crs, nodata=None, area_or_point=aop)
        return out_mask

    def __and__(self: RasterType, other: RasterType | NDArrayBool) -> RasterType:
        """Bitwise and between masks, or a mask and an array."""
        self_data, other_data = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"  # type: ignore
        )[0:2]

        return self.copy(self_data & other_data)  # type: ignore

    def __rand__(self: RasterType, other: RasterType | NDArrayBool) -> RasterType:
        """Bitwise and between masks, or a mask and an array."""

        return self.__and__(other)  # type: ignore

    def __or__(self: RasterType, other: RasterType | NDArrayBool) -> RasterType:
        """Bitwise or between masks, or a mask and an array."""

        self_data, other_data = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"  # type: ignore
        )[0:2]

        return self.copy(self_data | other_data)  # type: ignore

    def __ror__(self: RasterType, other: RasterType | NDArrayBool) -> RasterType:
        """Bitwise or between masks, or a mask and an array."""

        return self.__or__(other)  # type: ignore

    def __xor__(self: RasterType, other: RasterType | NDArrayBool) -> RasterType:
        """Bitwise xor between masks, or a mask and an array."""

        self_data, other_data = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"  # type: ignore
        )[0:2]

        return self.copy(self_data ^ other_data)  # type: ignore

    def __rxor__(self: RasterType, other: RasterType | NDArrayBool) -> RasterType:
        """Bitwise xor between masks, or a mask and an array."""

        return self.__xor__(other)  # type: ignore

    def __invert__(self: RasterType) -> RasterType:
        """Bitwise inversion of a mask."""

        return self.copy(~self.data)

    @overload
    def astype(
        self: RasterType, dtype: DTypeLike, convert_nodata: bool = True, *, inplace: Literal[False] = False
    ) -> RasterType: ...

    @overload
    def astype(self: RasterType, dtype: DTypeLike, convert_nodata: bool = True, *, inplace: Literal[True]) -> None: ...

    @overload
    def astype(
        self: RasterType, dtype: DTypeLike, convert_nodata: bool = True, *, inplace: bool = False
    ) -> RasterType | None: ...

    def astype(
        self: RasterType, dtype: DTypeLike, convert_nodata: bool = True, inplace: bool = False
    ) -> RasterType | None:
        """
        Convert data type of the raster.

        By default, converts the nodata value to the default of the new data type.

        :param dtype: Any numpy dtype or string accepted by numpy.astype.
        :param convert_nodata: Whether to convert the nodata value to the default of the new dtype.
        :param inplace: Whether to modify the raster in-place.

        :returns: Raster with updated dtype (or None if inplace).
        """

        # Check for all data type except boolean, that we support in addition to other types
        if np.dtype(dtype) != np.bool_:
            # Check that dtype is supported by rasterio
            if not rio.dtypes.check_dtype(dtype):
                raise TypeError(f"{dtype} is not supported by rasterio")

            # Check that data type change will not result in a loss of information
            if not rio.dtypes.can_cast_dtype(self.data, dtype):
                warnings.warn(
                    "dtype conversion will result in a loss of information. "
                    f"{rio.dtypes.get_minimum_dtype(self.data)} is the minimum type to represent the data.",
                    category=UserWarning,
                )

        out_data = self.data.astype(dtype)

        if inplace:
            self._data = out_data  # type: ignore
            if convert_nodata:
                self.set_nodata(new_nodata=_default_nodata(dtype))
            return None
        else:
            if not convert_nodata:
                nodata = self.nodata
            else:
                nodata = _default_nodata(dtype)
            return self.from_array(out_data, self.transform, self.crs, nodata=nodata, area_or_point=self.area_or_point)

    def set_mask(self, mask: NDArrayBool | Raster) -> None:
        """
        Set a mask on the raster array.

        All pixels where `mask` is set to True or > 0 will be masked (in addition to previously masked pixels).

        Masking is performed in place. The mask must have the same shape as loaded data,
        unless the first dimension is 1, then it is ignored.

        :param mask: The raster array mask.
        """
        # Check that mask is a Numpy array
        if not isinstance(mask, (np.ndarray, Raster)):
            raise ValueError("mask must be a numpy array or a raster.")

        # Check that new_data has correct shape
        if self.is_loaded:
            orig_shape = self.data.shape
        else:
            raise AttributeError("self.data must be loaded first, with e.g. self.load()")

        # If the mask is a Mask instance, pass the boolean array
        if isinstance(mask, Raster) and mask.is_mask:
            mask_arr = mask.data.filled(False)
        else:
            mask_arr = mask
        mask_arr = mask_arr.squeeze()

        if mask_arr.shape != orig_shape:
            # In case first dimension is more than one (several bands) and other dimensions match
            if orig_shape[1:] == mask_arr.shape:
                self.data[:, mask_arr > 0] = np.ma.masked
            else:
                raise ValueError(f"mask must be of the same shape as existing data: {orig_shape}.")
        else:
            self.data[mask_arr > 0] = np.ma.masked

    def copy(
        self: RasterType, new_array: NDArrayNum | None = None, cast_nodata: bool = True, deep: bool = True
    ) -> RasterType:

        # Core attributes to copy and set again
        dict_copy = [
            "_crs",
            "_nodata",
            "_tags",
            "_area_or_point",
            "_bands",
            "_transform",
            "_masked",
            "_is_mask",
            "_disk_shape",
            "_disk_bands",
            "_disk_dtype",
            "_disk_transform",
            "_downsample",
            "_name",
            "_driver",
            "_out_shape",
            "_out_count",
            "_obj",
        ]

        # Create empty instance without calling __init__
        cls = self.__class__
        new_obj = cls.__new__(cls)

        # Looping through non-data attributes
        for attr_name in dict_copy:
            # Automatically copy other attributes
            attr_value = getattr(self, attr_name)
            if deep:
                setattr(new_obj, attr_name, copy.deepcopy(attr_value))
            else:
                setattr(new_obj, attr_name, attr_value)

        # Cast nodata if the new array has incompatible type with the old nodata value
        if new_array is not None and cast_nodata:
            nodata = _cast_nodata(new_array.dtype, self.nodata)
            new_obj._nodata = copy.deepcopy(nodata)

        # Then, set the data attribute depending on deep copy, new array or none of those, ensuring laziness
        if new_array is None:
            if self.is_loaded:
                if deep:
                    new_obj._data = copy.deepcopy(self._data)  # Deep copy NumPy masked array (np.ma.copy insufficient)
                else:
                    new_obj._data = self._data  # Otherwise just point towards it
            else:
                # No need to load here: the array loaded implicitly later will already be different for the new object!
                new_obj._data = None  # If unloaded, set to None (as for input data)
        else:
            new_obj._data = None
            new_obj.data = new_array  # Using data.setter to trigger input checks for new array

        return new_obj

    def get_mask(self) -> NDArrayBool:
        """
        Get mask of invalid values from the raster.

        If the raster is not loaded, reads only the mask from disk to optimize memory usage.

        The mask is always returned as a boolean array, even if there is no mask associated to .data (nomask property
        of masked arrays).

        :return: The mask of invalid values in the raster.
        """
        # If it is loaded, use NumPy's getmaskarray function to deal with False values
        if self.is_loaded:
            mask = np.ma.getmaskarray(self.data)
        # Otherwise, load from Rasterio and deal with the possibility of having a single value "False" mask manually
        else:
            mask = self._load_only_mask()
            if isinstance(mask, np.bool_):
                mask = np.zeros(self.shape, dtype=bool)

        return mask

    # This is interfering with __array_ufunc__ and __array_function__, so better to leave out and specify
    # behaviour directly in those.
    # def __array__(self) -> np.ndarray:
    #     """Method to cast np.array() or np.asarray() function directly on Raster classes."""
    #
    #     return self._data

    def __array_ufunc__(
        self,
        ufunc: Callable[[NDArrayNum | tuple[NDArrayNum, NDArrayNum]], NDArrayNum | tuple[NDArrayNum, NDArrayNum]],
        method: str,
        *inputs: Raster | tuple[Raster, Raster] | tuple[NDArrayNum, Raster] | tuple[Raster, NDArrayNum],
        **kwargs: Any,
    ) -> Raster | tuple[Raster, Raster]:
        """
        Method to cast NumPy universal functions directly on Raster classes, by passing to the masked array.
        This function basically applies the ufunc (with its method and kwargs) to .data, and rebuilds the Raster from
        self.__class__. The cases separate the number of input nin and output nout, to properly feed .data and return
        Raster objects.
        See more details in NumPy doc, e.g., https://numpy.org/doc/stable/user/basics.dispatch.html#basics-dispatch.
        """

        # In addition to running ufuncs, this function takes over arithmetic operations (__add__, __multiply__, etc...)
        # when the first input provided is a NumPy array and second input a Raster.

        # The Raster ufuncs behave exactly as arithmetic operations (+, *, .) of NumPy masked array (call np.ma instead
        # of np when available). There is an inconsistency when calling np.ma: operations return a full boolean mask
        # even when there is no invalid value (true_divide and floor_divide).
        # We find one exception, however, for modulo: np.ma.remainder is not called but np.remainder instead one the
        # masked array is the second input (an inconsistency in NumPy!), so we mirror this exception below:
        if "remainder" in ufunc.__name__:
            final_ufunc = getattr(ufunc, method)
        else:
            try:
                final_ufunc = getattr(getattr(np.ma, ufunc.__name__), method)
            except AttributeError:
                final_ufunc = getattr(ufunc, method)

        # If the universal function takes only one input
        if ufunc.nin == 1:
            # If the universal function has only one output
            if ufunc.nout == 1:
                return self.copy(new_array=final_ufunc(inputs[0].data, **kwargs))  # type: ignore

            # If the universal function has two outputs (Note: no ufunc exists that has three outputs or more)
            else:
                output = final_ufunc(inputs[0].data, **kwargs)  # type: ignore
                return self.copy(new_array=output[0]), self.copy(new_array=output[1])

        # If the universal function takes two inputs (Note: no ufunc exists that has three inputs or more)
        else:

            # Check the casting between Raster and array inputs, and return error messages if not consistent
            if isinstance(inputs[0], Raster):
                raster = inputs[0]
                other = inputs[1]
            else:
                raster = inputs[1]  # type: ignore
                other = inputs[0]
            nodata, aop = _cast_numeric_array_raster(raster, other, "an arithmetic operation")[-2:]  # type: ignore

            if ufunc.nout == 1:
                return self.from_array(
                    data=final_ufunc(inputs[0].data, inputs[1].data, **kwargs),  # type: ignore
                    transform=self.transform,
                    crs=self.crs,
                    nodata=self.nodata,
                    area_or_point=aop,
                )

            # If the universal function has two outputs (Note: no ufunc exists that has three outputs or more)
            else:
                output = final_ufunc(inputs[0].data, inputs[1].data, **kwargs)  # type: ignore
                return self.from_array(
                    data=output[0],
                    transform=self.transform,
                    crs=self.crs,
                    nodata=self.nodata,
                    area_or_point=aop,
                ), self.from_array(
                    data=output[1], transform=self.transform, crs=self.crs, nodata=self.nodata, area_or_point=aop
                )

    def __array_function__(
        self, func: Callable[[NDArrayNum, Any], Any], types: tuple[type], args: Any, kwargs: Any
    ) -> Any:
        """
        Method to cast NumPy array function directly on a Raster object by applying it to the masked array.
        A limited number of function is supported, listed in raster.handled_array_funcs.
        """

        # If function is not implemented
        if func.__name__ not in _HANDLED_FUNCTIONS_1NIN + _HANDLED_FUNCTIONS_2NIN:
            return NotImplemented

        # For subclassing
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented

        # We now choose the behaviour of array functions
        # For median, np.median ignores masks of masked array, so we force np.ma.median
        if func.__name__ in ["median", "nanmedian"]:
            func = np.ma.median
            first_arg = args[0].data

        # For percentiles and quantiles, there exist no masked array version, so we compute on the valid data directly
        elif func.__name__ in ["percentile", "nanpercentile"]:
            first_arg = args[0].data.compressed()

        elif func.__name__ in ["quantile", "nanquantile"]:
            first_arg = args[0].data.compressed()

        elif func.__name__ in ["gradient"]:
            if self.count == 1:
                first_arg = args[0].data
            else:
                warnings.warn("Applying np.gradient to first raster band only.", category=UserWarning)
                first_arg = args[0].data[0, :, :]

        # Otherwise, we run the numpy function normally (most take masks into account)
        else:
            first_arg = args[0].data

        # Separate one and two input functions
        cast_required = False
        aop = None  # The None value is never used (aop only used when cast_required = True)
        if func.__name__ in _HANDLED_FUNCTIONS_1NIN:
            outputs = func(first_arg, *args[1:], **kwargs)  # type: ignore
        # Two input functions require casting
        else:
            # Check the casting between Raster and array inputs, and return error messages if not consistent
            if isinstance(args[0], Raster):
                raster = args[0]
                other = args[1]
            else:
                raster = args[1]
                other = args[0]
            nodata, aop = _cast_numeric_array_raster(raster, other, operation_name="an arithmetic operation")[-2:]
            cast_required = True
            second_arg = args[1].data
            outputs = func(first_arg, second_arg, *args[2:], **kwargs)  # type: ignore

        # Below, we recast to Raster if the shape was preserved, otherwise return an array
        # First, if there are several outputs in a tuple which are arrays
        if isinstance(outputs, tuple) and isinstance(outputs[0], np.ndarray):
            if all(output.shape == args[0].data.shape for output in outputs):

                # If casting was not necessary, copy all attributes except array
                # Otherwise update array, nodata and
                if cast_required:
                    return tuple(
                        self.from_array(
                            data=output, transform=self.transform, crs=self.crs, nodata=self.nodata, area_or_point=aop
                        )
                        for output in outputs
                    )
                else:
                    return tuple(self.copy(new_array=output) for output in outputs)
            else:
                return outputs
        # Second, if there is a single output which is an array
        elif isinstance(outputs, np.ndarray):
            if outputs.shape == args[0].data.shape:

                # If casting was not necessary, copy all attributes except array
                if cast_required:
                    return self.from_array(
                        data=outputs, transform=self.transform, crs=self.crs, nodata=self.nodata, area_or_point=aop
                    )
                else:
                    return self.copy(new_array=outputs)
            else:
                return outputs
        # Else, return outputs directly
        else:
            return outputs

    def to_file(
        self,
        filename: str | pathlib.Path | IO[bytes],
        driver: str = "GTiff",
        dtype: DTypeLike | None = None,
        nodata: Number | None = None,
        blank_value: int | float | None = None,
        co_opts: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        gcps: list[tuple[float, ...]] | None = None,
        gcps_crs: CRS | None = None,
    ) -> None:
        """
        Write the raster to file.

        If blank_value is set to an integer or float, then instead of writing
        the contents of self.data to disk, write this provided value to every
        pixel instead.

        Compression default value is set to 'deflate' (equal to GDALs: COMPRESS=DEFLATE in co_opts).
        Tiled default value is set to 'NO' as the GDAL default value.
        Raster is saved as a BigTIFF if the output file might exceed 4GB and as classical TIFF otherwise.

        Example: dem.to_file(to_file, co_opts={'TILED':'YES', 'COMPRESS':'LZW'})

        :param filename: Filename to write the file to.
        :param driver: Driver to write file with.
        :param dtype: Data type to write the image as (defaults to dtype of image data).
        :param nodata: Force a nodata value to be used (default to that of raster).
        :param blank_value: Use to write an image out with every pixel's value.
            corresponding to this value, instead of writing the image data to disk.
        :param co_opts: GDAL creation options provided as a dictionary, e.g. {'TILED':'YES', 'COMPRESS':'LZW'}.
        :param metadata: Pairs of metadata to save to disk, in addition to existing metadata in self.tags.
        :param gcps: List of gcps, each gcp being [row, col, x, y, (z)].
        :param gcps_crs: CRS of the GCPS.

        :returns: None.
        """

        if co_opts is None:
            co_opts = {}

        # Set COMPRESS default value to DEFLATE
        if "COMPRESS" not in map(str.upper, co_opts.keys()):
            co_opts["COMPRESS"] = "DEFLATE"

        # Set BIGTIFF default value to IF_SAFER, to save the output image as a BigTIFF if it might exceed 4GB.
        if "BIGTIFF" not in map(str.upper, co_opts.keys()):
            co_opts["BIGTIFF"] = "IF_SAFER"

        meta = self.tags if self.tags is not None else {}
        if metadata is not None:
            meta.update(metadata)
        if gcps is None:
            gcps = []

        # Use nodata set by user, otherwise default to self's
        nodata = nodata if nodata is not None else self.nodata

        # Declare type of save_data to work in all occurrences
        save_data: NDArrayNum

        # Define save_data depending on blank_value argument
        if (self.data is None) & (blank_value is None):
            raise AttributeError("No data loaded, and alternative blank_value not set.")
        elif blank_value is not None:
            if isinstance(blank_value, int) | isinstance(blank_value, float):
                save_data = np.zeros(self.data.shape)
                save_data[:] = blank_value
            else:
                raise ValueError("blank_values must be one of int, float (or None).")
        else:
            save_data = self.data

            # If the raster is a mask, convert to uint8 before saving and force nodata to 255
            if self.data.dtype == bool:
                save_data = save_data.astype("uint8")
                nodata = 255

            # If masked array, save with masked values replaced by nodata
            if isinstance(save_data, np.ma.masked_array):
                # In this case, nodata=None is not compatible, so revert to default values, only if masked values exist
                if (nodata is None) & (np.count_nonzero(save_data.mask) > 0):
                    nodata = _default_nodata(save_data.dtype)
                    warnings.warn(f"No nodata set, will use default value of {nodata}", category=UserWarning)
                save_data = save_data.filled(nodata)

        # Cast to 3D before saving if single band
        if self.count == 1:
            save_data = save_data[np.newaxis, :, :]

        with rio.open(
            filename,
            "w",
            driver=driver,
            height=self.height,
            width=self.width,
            count=self.count,
            dtype=save_data.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=nodata,
            **co_opts,
        ) as dst:
            dst.write(save_data)

            # Add metadata (tags in rio)
            dst.update_tags(**meta)

            # Save GCPs
            if not isinstance(gcps, list):
                raise ValueError("gcps must be a list")

            if len(gcps) > 0:
                rio_gcps = []
                for gcp in gcps:
                    rio_gcps.append(rio.control.GroundControlPoint(*gcp))

                # Warning: this will overwrite the transform
                if dst.transform != rio.transform.Affine(1, 0, 0, 0, 1, 0):
                    warnings.warn(
                        "A geotransform previously set is going to be cleared due to the setting of GCPs.",
                        category=UserWarning,
                    )

                dst.gcps = (rio_gcps, gcps_crs)

    @deprecate(
        removal_version=Version("0.3.0"),
        details="The function .save() will be soon deprecated, use .to_file() instead.",
    )  # type: ignore
    def save(
        self,
        filename: str | pathlib.Path | IO[bytes],
        driver: str = "GTiff",
        dtype: DTypeLike | None = None,
        nodata: Number | None = None,
        blank_value: int | float | None = None,
        co_opts: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        gcps: list[tuple[float, ...]] | None = None,
        gcps_crs: CRS | None = None,
    ) -> None:
        self.to_file(filename, driver, dtype, nodata, blank_value, co_opts, metadata, gcps, gcps_crs)

    @classmethod
    def from_xarray(cls: type[RasterType], ds: xr.DataArray, dtype: DTypeLike | None = None) -> RasterType:
        """
        Create raster from a xarray.DataArray.

        This conversion loads the xarray.DataArray in memory. Use functions of the Xarray accessor directly
        to avoid this behaviour.

        :param ds: Data array.
        :param dtype: Cast the array to a certain dtype.

        :return: Raster.
        """

        # Define main attributes
        crs = ds.rio.crs
        transform = ds.rio.transform(recalc=True)
        nodata = ds.rio.nodata
        area_or_point = ds.attrs.get("AREA_OR_POINT", None)
        tags = ds.attrs

        raster = cls.from_array(
            data=ds.data, transform=transform, crs=crs, nodata=nodata, area_or_point=area_or_point, tags=tags
        )

        if dtype is not None:
            raster = raster.astype(dtype)

        return raster

    def to_xarray(self, name: str | None = None) -> xr.DataArray:
        """
        Convert a xarray.DataArray raster.

        Integer-type rasters are cast to float32.

        :param name: Name attribute for the data array.

        :returns: Data array.
        """

        # If type was integer, cast to float to be able to save nodata values in the xarray data array
        if np.issubdtype(self.dtype, np.integer):
            # Nodata conversion is not needed in this direction (integer towards float), we can maintain the original
            updated_raster = self.astype(np.float32, convert_nodata=False)
        else:
            updated_raster = self

        ds = rioxarray.open_rasterio(updated_raster.to_rio_dataset(), masked=True)
        # When reading as masked, the nodata is not written to the dataset so we do it manually
        ds.rio.set_nodata(self.nodata)

        if name is not None:
            ds.name = name

        return ds

    @overload
    def plot(
        self,
        bands: int | tuple[int, ...] | None = None,
        cmap: matplotlib.colors.Colormap | str | None = None,
        vmin: float | int | None = None,
        vmax: float | int | None = None,
        alpha: float | int | None = None,
        title: str | None = None,
        cbar_title: str | None = None,
        add_cbar: bool = True,
        ax: matplotlib.axes.Axes | Literal["new"] | None = None,
        *,
        return_axes: Literal[False] = False,
        savefig_fname: str | None = None,
        **kwargs: Any,
    ) -> None: ...

    @overload
    def plot(
        self,
        bands: int | tuple[int, ...] | None = None,
        cmap: matplotlib.colors.Colormap | str | None = None,
        vmin: float | int | None = None,
        vmax: float | int | None = None,
        alpha: float | int | None = None,
        title: str | None = None,
        cbar_title: str | None = None,
        add_cbar: bool = True,
        ax: matplotlib.axes.Axes | Literal["new"] | None = None,
        *,
        return_axes: Literal[True],
        savefig_fname: str | None = None,
        **kwargs: Any,
    ) -> tuple[matplotlib.axes.Axes, matplotlib.colors.Colormap]: ...

    def plot(
        self,
        bands: int | tuple[int, ...] | None = None,
        cmap: matplotlib.colors.Colormap | str | None = None,
        vmin: float | int | None = None,
        vmax: float | int | None = None,
        alpha: float | int | None = None,
        title: str | None = None,
        cbar_title: str | None = None,
        add_cbar: bool = True,
        ax: matplotlib.axes.Axes | Literal["new"] | None = None,
        return_axes: bool = False,
        savefig_fname: str | None = None,
        **kwargs: Any,
    ) -> None | tuple[matplotlib.axes.Axes, matplotlib.colors.Colormap]:
        r"""
        Plot the raster, with axes in projection of image.

        This method is a wrapper to matplotlib.imshow with modifications to work on raster (flip Y-axis, lower origin,
        equal scale). Any \*\*kwargs which you give this method will be passed to matplotlib.imshow.
        If the raster is passed with 3(4) bands, it is plotted as RGB(Alpha).

        :param bands: Bands to plot, counting from 1 to self.count (default is all bands).
        :param cmap: Colormap to use. Default is plt.rcParams['image.cmap'].
        :param vmin: Minimum value for colorbar. Default is data min.
        :param vmax: Maximum value for colorbar. Default is data max.
        :param alpha: Transparency of raster and colorbar. Default is None.
        :param title: Title of the plot. Default is None.
        :param cbar_title: Colorbar label title. Default is None.
        :param add_cbar: Set to True to display a colorbar. Default is True.
        :param ax: A figure ax to be used for plotting. If None, will plot on current axes.
            If "new", will create a new axis.
        :param return_axes: Whether to return axes.
        :param savefig_fname: Path to quick save the output figure (previously created if an ax is give, new if not)
            with a default DPI, no transparency and no metadata. Use `plt.savefig()` to specify other save
            parameters or after other customizations. Warning: `plt.close()` or `plt.show()` still needs to be called
            to close the figure.

        :returns: None, or (ax, caxes) if return_axes is True.
        """

        matplotlib = import_optional("matplotlib")
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # If data is not loaded, need to load it
        if not self.is_loaded:
            self.load()

        # Set matplotlib interpolation to None by default, to avoid spreading gaps in plots
        if "interpolation" not in kwargs.keys():
            kwargs.update({"interpolation": None})

        # Check if specific band selected, or take all
        # if self.count=3 (4) => plotted as RGB(A)
        if bands is None or isinstance(bands, tuple):
            # Use all if None was specified
            if bands is None:
                bands = tuple(range(1, self.count + 1))
            # Check the number of bands is 1, 3 or 4
            if len(bands) not in [1, 3, 4]:
                raise ValueError(
                    f"Only single-band or 3/4-band (RGB-A) plotting is supported. "
                    f"Found {len(bands)} bands. Use the `bands` argument to specify bands."
                )
            if len(bands) == 1:
                bands = bands[0]
        elif isinstance(bands, int):
            if bands > self.count:
                raise ValueError(f"Index must be in range 1-{self.count:d}")
            pass
        else:
            raise ValueError("Index must be int, tuple or None")

        # Get data
        if self.count == 1:
            data = self.data
        else:
            data = self.data[np.array(bands) - 1, :, :]

        # If multiple bands (RGB), cbar does not make sense
        if isinstance(bands, abc.Sequence):
            if len(bands) > 1:
                add_cbar = False
            # Re-order axes for RGB plotting
            data = np.moveaxis(data, 0, -1)  # type: ignore

        # Create colorbar
        # Use rcParam default
        if cmap is None and len(bands) == 1:
            # ONLY set a cmap arg for single band images
            cmap = plt.get_cmap(plt.rcParams["image.cmap"])
        elif cmap is None and len(bands) > 1:
            # Leave cmap as None for multi-band image, because if a cmap
            # is passed then imshow treats this as an instruction to apply scalar
            # mapping, which is not a desirable behaviour (it can result in color-casted
            # RGB images for example).
            pass
        elif isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        elif isinstance(cmap, matplotlib.colors.Colormap):
            pass

        # Set colorbar min/max values (needed for ScalarMappable)
        if vmin is None:
            vmin = float(np.nanmin(data))

        if vmax is None:
            vmax = float(np.nanmax(data))

        # Make sure they are numbers, to avoid mpl error
        try:
            vmin = float(vmin)
            vmax = float(vmax)
        except ValueError:
            raise ValueError("vmin or vmax cannot be converted to float")

        # Create axes
        if ax is None:
            ax0 = plt.gca()
        elif isinstance(ax, str) and ax.lower() == "new":
            _, ax0 = plt.subplots()
        elif isinstance(ax, matplotlib.axes.Axes):
            ax0 = ax
        else:
            raise ValueError("ax must be a matplotlib.axes.Axes instance, 'new' or None.")

        # Use data array directly, as rshow on self.ds will re-load data
        extent = [self.bounds.left, self.bounds.right, self.bounds.bottom, self.bounds.top]
        ax0.imshow(
            np.flip(data, axis=0),
            extent=extent,
            origin="lower",  # So that the array is not upside-down
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            **kwargs,
        )
        if title is not None:
            ax0.set_title(title)

        # Add colorbar
        if add_cbar:
            divider = make_axes_locatable(ax0)
            cax = divider.append_axes("right", size="5%", pad="2%")
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
            cbar.solids.set_alpha(alpha)

            if cbar_title is not None:
                cbar.set_label(cbar_title)
        else:
            cbar = None

        plt.sca(ax0)
        plt.tight_layout()

        # if savefig_fname filled, save the plot
        if savefig_fname:
            plt.savefig(savefig_fname)

        # If returning axes
        if return_axes:
            return ax0, cax
        return None

    def reduce_points(
        self,
        points: tuple[ArrayLike, ArrayLike] | gu.PointCloud,
        reducer_function: Callable[[NDArrayNum], float] = np.ma.mean,
        window: int | None = None,
        input_latlon: bool = False,
        band: int | None = None,
        masked: bool = False,
        return_window: bool = False,
        as_array: bool = False,
        boundless: bool = True,
    ) -> Any:
        """
        Reduce raster values around point coordinates.

        By default, samples pixel value of each band. Can be passed a band index to sample from.

        Uses Rasterio's windowed reading to keep memory usage low (for a raster not loaded).

        :param points: Point(s) at which to interpolate raster value. Can be either a tuple of array-like of X/Y
            coordinates (same CRS as raster or latitude/longitude, see "input_latlon") or a pointcloud in any CRS.
            If points fall outside of image, value returned is nan.
        :param reducer_function: Reducer function to apply to the values in window (defaults to np.mean).
        :param window: Window size to read around coordinates. Must be odd.
        :param input_latlon: (Only for tuple point input) Whether to convert input coordinates from latlon to raster
            CRS.
        :param band: Band number to extract from (from 1 to self.count).
        :param masked: Whether to return a masked array, or classic array.
        :param return_window: Whether to return the windows (in addition to the reduced value).
        :param as_array: Whether to return an array of reduced values (defaults to a point cloud containing input
            coordinates).
        :param boundless: Whether to allow windows that extend beyond the extent.

        :returns: Point cloud of interpolated points, or 1D array of interpolated values.
            In addition, if return_window=True, return tuple of (values, arrays).

        :examples:

            >>> self.value_at_coords(-48.125, 67.8901, window=3)  # doctest: +SKIP
            Returns mean of a 3*3 window:
                v v v \
                v c v  | = float(mean)
                v v v /
            (c = provided coordinate, v= value of surrounding coordinate)

        """

        if isinstance(points, gu.PointCloud):
            points = reproject_points((points.ds.geometry.x.values, points.ds.geometry.y.values), points.crs, self.crs)
            # Otherwise
        else:
            if input_latlon:
                points = reproject_from_latlon((points[1], points[0]), out_crs=self.crs)  # type: ignore

        x, y = points

        # Check for array-like inputs
        if (
            not isinstance(x, (float, np.floating, int, np.integer))
            and isinstance(y, (float, np.floating, int, np.integer))
            or isinstance(x, (float, np.floating, int, np.integer))
            and not isinstance(y, (float, np.floating, int, np.integer))
        ):
            raise TypeError("Coordinates must be both numbers or both array-like.")

        # If for a single value, wrap in a list
        if isinstance(x, (float, np.floating, int, np.integer)):
            x = [x]  # type: ignore
            y = [y]  # type: ignore
            # For the end of the function
            unwrap = True
        else:
            unwrap = False
            # Check that array-like objects are the same length
            if len(x) != len(y):  # type: ignore
                raise ValueError("Coordinates must be of the same length.")

        # Check window parameter
        if window is not None:
            if not float(window).is_integer():
                raise ValueError("Window must be a whole number.")
            if window % 2 != 1:
                raise ValueError("Window must be an odd number.")
            window = int(window)

        # Define subfunction for reducing the window array
        def format_value(value: Any) -> Any:
            """Check if valid value has been extracted"""
            if type(value) in [np.ndarray, np.ma.core.MaskedArray]:
                if window is not None:
                    value = reducer_function(value.flatten())
                else:
                    value = value[0, 0]
            else:
                value = None
            return value

        # Initiate output lists
        list_values = []
        if return_window:
            list_windows = []

        # Convert to latlon if asked
        # if input_latlon:
        #     x, y = reproject_from_latlon((y, x), self.crs)  # type: ignore

        # Convert coordinates to pixel space
        rows, cols = rio.transform.rowcol(self.transform, x, y, op=math.floor)

        # Loop over all coordinates passed
        for k in range(len(rows)):  # type: ignore
            value: float | dict[int, float] | tuple[float | dict[int, float] | tuple[list[float], NDArrayNum] | Any]

            row = rows[k]  # type: ignore
            col = cols[k]  # type: ignore

            # Decide what pixel coordinates to read:
            if window is not None:
                half_win = (window - 1) / 2
                # Subtract start coordinates back to top left of window
                col = col - half_win
                row = row - half_win
                # Offset to read to == window
                width = window
                height = window
            else:
                # Start reading at col,row and read 1px each way
                width = 1
                height = 1

            # Make sure coordinates are int
            col = int(col)
            row = int(row)

            # Create rasterio's window for reading
            rio_window = rio.windows.Window(col, row, width, height)

            if self.is_loaded:
                if self.count == 1:
                    data = self.data[row : row + height, col : col + width]
                else:
                    data = self.data[slice(None) if band is None else band - 1, row : row + height, col : col + width]
                if not masked:
                    data = data.astype(np.float32).filled(np.nan)
                value = format_value(data)
                win: NDArrayNum | dict[int, NDArrayNum] = data

            else:
                # if self.count == 1:
                with rio.open(self.name) as raster:
                    data = raster.read(
                        window=rio_window,
                        fill_value=self.nodata,
                        boundless=boundless,
                        masked=masked,
                        indexes=band,
                    )
                value = format_value(data)
                win = data
                # else:
                #     value = {}
                #     win = {}
                #     with rio.open(self.name) as raster:
                #         for b in self.indexes:
                #             data = raster.read(
                #                 window=rio_window, fill_value=self.nodata, boundless=boundless,
                #                 masked=masked, indexes=b
                #             )
                #             val = format_value(data)
                #             value[b] = val
                #             win[b] = data  # type: ignore

            list_values.append(value)
            if return_window:
                list_windows.append(win)

        # If for a single value, unwrap output list
        if unwrap:
            output_val = list_values[0]
            if return_window:
                output_win = list_windows[0]
        else:
            output_val = np.array(list_values)  # type: ignore
            if return_window:
                output_win = list_windows  # type: ignore

        # Return array or pointcloud
        if not as_array:
            output_val = gu.PointCloud.from_xyz(x=points[0], y=points[1], z=output_val, crs=self.crs)

        if return_window:
            return (output_val, output_win)
        else:
            return output_val

    def split_bands(self: RasterType, bands: list[int] | int | None = None, deep: bool = True) -> list[RasterType]:
        """
        Split the bands into separate rasters.

        :param bands: Optional. A list of band indices to extract (from 1 to self.count). Defaults to all.
        :param deep: Whether to return a deep or shallow copy.

        :returns: A list of Rasters for each band.
        """

        raster_bands: list[RasterType] = []

        if bands is None:
            indices = list(np.arange(1, self.count + 1))
        elif isinstance(bands, int):
            indices = [bands]
        elif isinstance(bands, list):
            indices = bands
        else:
            raise ValueError(f"'subset' got invalid type: {type(bands)}. Expected list[int], int or None")

        raster_bands = []
        # If not loaded, we do a copy that points towards the right bands for future loading
        if not self.is_loaded:
            for band_n in indices:
                rast_band = self.copy(deep=deep, cast_nodata=False)
                rast_band._bands = band_n
                rast_band._out_count = 1
                raster_bands.append(rast_band)
        else:
            for band_n in indices:
                # Little trick again: overwrite the array after a shallow copy
                rast_band = self.copy(deep=False, cast_nodata=False)
                rast_band._bands = band_n
                if deep:
                    band_arr = copy.deepcopy(self.data[band_n - 1, :, :])
                else:
                    band_arr = self.data[band_n - 1, :, :]
                rast_band._data = band_arr
                raster_bands.append(rast_band)

        return raster_bands


class Mask(Raster):
    """
    The georeferenced raster mask.

    DEPRECATED: USE RASTER(IS_MASK=TRUE) INSTEAD.

    A raster mask is a raster with a boolean array (True or False), that can serve to index or assign values to other
    rasters with the same georeferenced grid.

    Note: As boolean arrays cannot be saved to file, Masks are converted to uint8 type with default nodata of 255 when
    saving to file.

    Subclasses :class:`geoutils.Raster`.

     Main attributes:
        data: :class:`np.ndarray`
            Boolean data array of the mask, with dimensions corresponding to (height, width).
        transform: :class:`affine.Affine`
            Geotransform of the raster.
        crs: :class:`pyproj.crs.CRS`
            Coordinate reference system of the raster.

    All other attributes are derivatives of those attributes, or read from the file on disk.
    See the API for more details.
    """

    @deprecate(
        removal_version=Version("0.3.0"), details="The Mask class is deprecated, use Raster(is_mask=True) instead."
    )  # type: ignore
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        super().__init__(*args, **kwargs, is_mask=True)  # type: ignore
