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

import logging
import math
import pathlib
import warnings
from collections import abc
from contextlib import ExitStack
from math import floor
from typing import IO, Any, Callable, TypeVar, overload

import affine
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import rasterio.windows
import rioxarray
import xarray as xr
from affine import Affine
from mpl_toolkits.axes_grid1 import make_axes_locatable
from packaging.version import Version
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.plot import show as rshow

import geoutils as gu
from geoutils import profiler
from geoutils._config import config
from geoutils._typing import (
    ArrayLike,
    DTypeLike,
    MArrayNum,
    NDArrayBool,
    NDArrayNum,
    Number,
)
from geoutils.filters import _filter
from geoutils.interface.distance import _proximity_from_vector_or_raster
from geoutils.interface.interpolate import _interp_points
from geoutils.interface.raster_point import (
    _raster_to_pointcloud,
    _regular_pointcloud_to_raster,
)
from geoutils.interface.raster_vector import _polygonize
from geoutils.misc import deprecate
from geoutils.projtools import (
    _get_bounds_projected,
    _get_footprint_projected,
    _get_utm_ups_crs,
    reproject_from_latlon,
    reproject_points,
)
from geoutils.raster.distributed_computing.multiproc import MultiprocConfig
from geoutils.raster.georeferencing import (
    _bounds,
    _cast_nodata,
    _cast_pixel_interpretation,
    _coords,
    _default_nodata,
    _ij2xy,
    _outside_image,
    _res,
    _xy2ij,
)
from geoutils.raster.geotransformations import _crop, _reproject, _translate
from geoutils.raster.satimg import (
    decode_sensor_metadata,
    parse_and_convert_metadata_from_filename,
)
from geoutils.stats.sampling import subsample_array
from geoutils.stats.stats import _statistics

# If python38 or above, Literal is builtin. Otherwise, use typing_extensions
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

RasterType = TypeVar("RasterType", bound="Raster")

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


class Raster:
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
        nodata: int | float | None = None,
    ) -> None:
        """
        Instantiate a raster from a filename or rasterio dataset.

        :param filename_or_dataset: Path to file or Rasterio dataset.
        :param bands: Band(s) to load into the object. Default loads all bands.
        :param load_data: Whether to load the array during instantiation. Default is False.
        :param parse_sensor_metadata: Whether to parse sensor metadata from filename and similarly-named metadata files.
        :param silent: Whether to parse metadata silently or with console output.
        :param downsample: Downsample the array once loaded by a round factor. Default is no downsampling.
        :param nodata: Nodata value to be used (overwrites the metadata). Default reads from metadata.
        """
        self._driver: str | None = None
        self._name: str | None = None
        self.filename: str | None = None
        self._tags: dict[str, Any] = {}

        self._data: MArrayNum | None = None
        self._transform: affine.Affine | None = None
        self._crs: CRS | None = None
        self._nodata: int | float | None = nodata
        self._bands = bands
        self._bands_loaded: int | tuple[int, ...] | None = None
        self._masked = True
        self._out_count: int | None = None
        self._out_shape: tuple[int, int] | None = None
        self._disk_hash: int | None = None
        self._is_modified = True
        self._disk_shape: tuple[int, int, int] | None = None
        self._disk_bands: tuple[int] | None = None
        self._disk_dtype: str | None = None
        self._disk_transform: affine.Affine | None = None
        self._downsample: int | float = 1
        self._area_or_point: Literal["Area", "Point"] | None = None
        self._profile: dict[str, Any] | None = None
        self._is_mask: bool = is_mask

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
                    self.filename = str(filename_or_dataset)
                elif isinstance(filename_or_dataset, rio.io.DatasetReader):
                    ds = filename_or_dataset
                    self.filename = filename_or_dataset.files[0]
                # This is if it's a MemoryFile
                else:
                    ds = filename_or_dataset.open()
                    # In that case, data has to be loaded
                    load_data = True
                    self.filename = None

                self._transform = ds.transform
                self._crs = ds.crs
                # Allow user to manually override the nodata value which may be specified in the file.
                if nodata is not None:
                    self._nodata = nodata
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

            # If data was loaded explicitly, initiate is_modified and save disk hash
            if load_data and isinstance(filename_or_dataset, str):
                self._is_modified = False
                self._disk_hash = hash((self.data.tobytes(), self.transform, self.crs, self.nodata))

        # Provide a catch in case trying to load from data array
        elif isinstance(filename_or_dataset, np.ndarray):
            raise TypeError("The filename is an array, did you mean to call Raster.from_array(...) instead?")

        # Don't recognise the input, so stop here.
        else:
            raise TypeError("The filename argument is not recognised, should be a path or a Rasterio dataset.")

        # Parse metadata and add to tags
        if parse_sensor_metadata and self.filename is not None:
            sensor_meta = parse_and_convert_metadata_from_filename(self.filename, silent=silent)
            self._tags.update(sensor_meta)

    @property
    def count_on_disk(self) -> None | int:
        """Count of bands on disk if it exists."""
        if self._disk_shape is not None:
            return self._disk_shape[0]
        return None

    @property
    def count(self) -> int:
        """Count of bands loaded in memory if they are, otherwise the one on disk."""
        if self.is_loaded:
            if self.data.ndim == 2:
                return 1
            else:
                return int(self.data.shape[0])
        #  This can only happen if data is not loaded, with a DatasetReader on disk is open, never returns None
        return self.count_on_disk  # type: ignore

    @property
    def height(self) -> int:
        """Height of the raster in pixels."""
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
        """Width of the raster in pixels."""
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
        """Shape (i.e., height, width) of the raster in pixels."""
        # If a downsampling argument was defined but data not loaded yet
        if self._out_shape is not None and not self.is_loaded:
            return self._out_shape
        # If data loaded or not, pass the disk/data shape through height and width
        return self.height, self.width

    @property
    def res(self) -> tuple[float | int, float | int]:
        """Resolution (X, Y) of the raster in georeferenced units."""
        return _res(self.transform)

    @property
    def bounds(self) -> rio.coords.BoundingBox:
        """Bounding coordinates of the raster."""
        return _bounds(transform=self.transform, shape=self.shape)

    @property
    def footprint(self) -> gu.Vector:
        """Footprint of the raster."""
        return self.get_footprint_projected(self.crs)

    @property
    def is_loaded(self) -> bool:
        """Whether the raster array is loaded."""
        return self._data is not None

    @property
    def dtype(self) -> str:
        """Data type of the raster (string representation)."""
        if not self.is_loaded and self._disk_dtype is not None:
            return self._disk_dtype
        return str(self.data.dtype)

    @property
    def bands_on_disk(self) -> None | tuple[int, ...]:
        """Band indexes on disk if a file exists."""
        if self._disk_bands is not None:
            return self._disk_bands
        return None

    @property
    def bands(self) -> tuple[int, ...]:
        """Band indexes loaded in memory if they are, otherwise on disk."""
        if self._bands is not None and not self.is_loaded:
            if isinstance(self._bands, int):
                return (self._bands,)
            return tuple(self._bands)
        # if self._indexes_loaded is not None:
        #     if isinstance(self._indexes_loaded, int):
        #         return (self._indexes_loaded, )
        #     return tuple(self._indexes_loaded)
        if self.is_loaded:
            return tuple(range(1, self.count + 1))
        return self.bands_on_disk  # type: ignore

    @property
    def indexes(self) -> tuple[int, ...]:
        """
        Band indexes (duplicate of .bands attribute, mirroring Rasterio naming "indexes").
        Loaded in memory if they are, otherwise on disk.
        """
        return self.bands

    @property
    def name(self) -> str | None:
        """Name of the raster file on disk, if it exists."""
        return self._name

    @property
    def profile(self) -> dict[str, Any] | None:
        """Basic metadata and creation options of this dataset.
        May be passed as keyword arguments to rasterio.open()
        to create a clone of this dataset."""
        return self._profile

    @property
    def is_mask(self) -> bool:
        """Whether the raster array is a boolean data type (a mask)."""

        # Follow user input if not loaded
        if not self.is_loaded:
            return self._is_mask
        # Otherwise check data type
        else:
            return np.dtype(self.dtype) == np.bool_

    def set_area_or_point(
        self, new_area_or_point: Literal["Area", "Point"] | None, shift_area_or_point: bool | None = None
    ) -> None:
        """
        Set new pixel interpretation of the raster.

        Overwrites the `area_or_point` attribute and updates "AREA_OR_POINT" in raster metadata tags.

        Optionally, shifts the raster to correct value coordinates in relation to interpretation:

        - By half a pixel (right and downwards) if old interpretation was "Area" and new is "Point",
        - By half a pixel (left and upwards) if old interpretration was "Point" and new is "Area",
        - No shift for all other cases.

        :param new_area_or_point: New pixel interpretation "Area", "Point" or None.
        :param shift_area_or_point: Whether to shift with pixel interpretation, which shifts to center of pixel
            indexes if self.area_or_point is "Point" and maintains corner pixel indexes if it is "Area" or None.
            Defaults to True. Can be configured with the global setting geoutils.config["shift_area_or_point"].

        :return: None.
        """

        # If undefined, default to the global system config
        if shift_area_or_point is None:
            shift_area_or_point = config["shift_area_or_point"]

        # Check input
        if new_area_or_point is not None and not (
            isinstance(new_area_or_point, str) and new_area_or_point.lower() in ["area", "point"]
        ):
            raise ValueError("New pixel interpretation must be 'Area', 'Point' or None.")

        # Update string input as exactly "Area" or "Point"
        if new_area_or_point is not None:
            if new_area_or_point.lower() == "area":
                new_area_or_point = "Area"
            else:
                new_area_or_point = "Point"

        # Save old area or point
        old_area_or_point = self.area_or_point

        # Set new interpretation
        self._area_or_point = new_area_or_point
        # Update tag only if not None
        if new_area_or_point is not None:
            self.tags.update({"AREA_OR_POINT": new_area_or_point})
        else:
            if "AREA_OR_POINT" in self.tags:
                self.tags.pop("AREA_OR_POINT")

        # If shift is True, and both interpretation were different strings, a change is needed
        if (
            shift_area_or_point
            and isinstance(old_area_or_point, str)
            and isinstance(new_area_or_point, str)
            and old_area_or_point != new_area_or_point
        ):
            # The shift below represents +0.5/+0.5 or opposite in indexes (as done in xy2ij)

            # If the new one is Point, we shift back by half a pixel
            if new_area_or_point == "Point":
                xoff = 0.5
                yoff = 0.5
            # Otherwise we shift forward half a pixel
            else:
                xoff = -0.5
                yoff = -0.5
            # We perform the shift in place
            self.translate(xoff=xoff, yoff=yoff, distance_unit="pixel", inplace=True)

    @property
    def area_or_point(self) -> Literal["Area", "Point"] | None:
        """
        Pixel interpretation of the raster.

        Based on the "AREA_OR_POINT" raster metadata:

        - If pixel interpretation is "Area", the value of the pixel is associated with its upper left corner.
        - If pixel interpretation is "Point", the value of the pixel is associated with its center.

        When setting with self.area_or_point = new_area_or_point, uses the default arguments of
        self.set_area_or_point().
        """
        return self._area_or_point

    @area_or_point.setter
    def area_or_point(self, new_area_or_point: Literal["Area", "Point"] | None) -> None:
        """
        Setter for pixel interpretation.

        Uses default arguments of self.set_area_or_point(): shifts by half a pixel going from "Area" to "Point",
        or the opposite.

        :param new_area_or_point: New pixel interpretation "Area", "Point" or None.

        :return: None.
        """
        self.set_area_or_point(new_area_or_point=new_area_or_point)

    @property
    def driver(self) -> str | None:
        """Driver used to read a file on disk."""
        return self._driver

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

        if self.filename is None:
            raise AttributeError(
                "Cannot load as filename is not set anymore. Did you manually update the filename attribute?"
            )

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
        with rio.open(self.filename) as dataset:
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
        :raises AttributeError: If no 'filename' attribute exists.
        """
        if self.is_loaded:
            raise ValueError("Data are already loaded.")

        if self.filename is None:
            raise AttributeError(
                "Cannot load as filename is not set anymore. Did you manually update the filename attribute?"
            )

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
        with rio.open(self.filename) as dataset:
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

        # To have is_modified work correctly when data is loaded implicitly (not in init)
        self._is_modified = False
        self._disk_hash = hash((self.data.tobytes(), self.transform, self.crs, self.nodata))

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

    def raster_equal(self, other: RasterType, strict_masked: bool = True, warn_failure_reason: bool = False) -> bool:
        """
        Check if two rasters are equal.

        This means that are equal:
        - The raster's masked array's data (including masked values), mask, fill_value and dtype,
        - The raster's transform, crs and nodata values.

        :param other: Other raster.
        :param strict_masked: Whether to check if masked cells (in .data.mask) have the same value (in .data.data).
        :param warn_failure_reason: Whether to warn for the reason of failure if the check does not pass.
        """

        if not isinstance(other, Raster):
            raise NotImplementedError("Equality with other object than Raster not supported by raster_equal.")

        if strict_masked:
            names = ["data.data", "data.mask", "data.fill_value", "dtype", "transform", "crs", "nodata"]
            equalities = [
                np.array_equal(self.data.data, other.data.data, equal_nan=True),
                # Use getmaskarray to avoid comparing boolean with array when mask=False
                np.array_equal(np.ma.getmaskarray(self.data), np.ma.getmaskarray(other.data)),
                self.data.fill_value == other.data.fill_value,
                self.data.dtype == other.data.dtype,
                self.transform == other.transform,
                self.crs == other.crs,
                self.nodata == other.nodata,
            ]
        else:
            names = ["data", "data.fill_value", "dtype", "transform", "crs", "nodata"]
            equalities = [
                np.ma.allequal(self.data, other.data),
                self.data.fill_value == other.data.fill_value,
                self.data.dtype == other.data.dtype,
                self.transform == other.transform,
                self.crs == other.crs,
                self.nodata == other.nodata,
            ]

        complete_equality = all(equalities)

        if not complete_equality and warn_failure_reason:
            where_fail = np.nonzero(~np.array(equalities))[0]
            warnings.warn(
                category=UserWarning, message=f"Equality failed for: {', '.join([names[w] for w in where_fail])}."
            )

        return complete_equality

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
        return self.__mul__(other)

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

        return self.__and__(other)

    def __or__(self: RasterType, other: RasterType | NDArrayBool) -> RasterType:
        """Bitwise or between masks, or a mask and an array."""

        self_data, other_data = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"  # type: ignore
        )[0:2]

        return self.copy(self_data | other_data)  # type: ignore

    def __ror__(self: RasterType, other: RasterType | NDArrayBool) -> RasterType:
        """Bitwise or between masks, or a mask and an array."""

        return self.__or__(other)

    def __xor__(self: RasterType, other: RasterType | NDArrayBool) -> RasterType:
        """Bitwise xor between masks, or a mask and an array."""

        self_data, other_data = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"  # type: ignore
        )[0:2]

        return self.copy(self_data ^ other_data)  # type: ignore

    def __rxor__(self: RasterType, other: RasterType | NDArrayBool) -> RasterType:
        """Bitwise xor between masks, or a mask and an array."""

        return self.__xor__(other)

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
                    f"{rio.dtypes.get_minimum_dtype(self.data)} is the minimum type to represent the data."
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

    @property
    def is_modified(self) -> bool:
        """Whether the array has been modified since it was loaded from disk.

        :returns: True if raster has been modified.

        """
        if not self.is_loaded:
            return False

        if not self._is_modified:
            new_hash = hash(
                (self._data.tobytes() if self._data is not None else 0, self.transform, self.crs, self.nodata)
            )
            self._is_modified = not (self._disk_hash == new_hash)

        return self._is_modified

    @property
    def nodata(self) -> int | float | None:
        """
        Nodata value of the raster.

        When setting with self.nodata = new_nodata, uses the default arguments of self.set_nodata().

        :returns: Nodata value
        """
        return self._nodata

    @nodata.setter
    def nodata(self, new_nodata: int | float | None) -> None:
        """
        Set .nodata and update .data by calling set_nodata() with default parameters.

        By default, the old nodata values are updated into the new nodata in the data array .data.data, and the
        mask .data.mask is updated to mask all new nodata values (i.e., the mask from old nodata stays and is extended
        to potential new values of new nodata found in the array).

        To set nodata for more complex cases (e.g., redefining a wrong nodata that has a valid value in the array),
        call the function set_nodata() directly to set the arguments update_array and update_mask adequately.

        :param new_nodata: New nodata to assign to this instance of Raster.
        """

        self.set_nodata(new_nodata=new_nodata)

    def set_nodata(
        self,
        new_nodata: int | float | None,
        update_array: bool = True,
        update_mask: bool = True,
    ) -> None:
        """
        Set a new nodata value for all bands. This updates the old nodata into a new nodata value in the metadata,
        replaces the nodata values in the data of the masked array, and updates the mask of the masked array.

        Careful! If the new nodata value already exists in the array, the related grid cells will be masked by default.

        If the nodata value was not defined in the raster, run this function with a new nodata value corresponding to
        the value of nodata that exists in the data array and is not yet accounted for. All those values will be masked.

        If a nodata value was correctly defined in the raster, and you wish to change it to a new value, run
        this function with that new value. All values having either the old or new nodata value will be masked.

        If the nodata value was wrongly defined in the raster, and you wish to change it to a new value without
        affecting data that might have the value of the old nodata, run this function with the update_array
        argument as False. Only the values of the new nodata will be masked.

        If you wish to set nodata value without updating the mask, run this function with the update_mask argument as
        False.

        If None is passed as nodata, only the metadata is updated and the mask of old nodata unset.

        :param new_nodata: New nodata value.
        :param update_array: Update the old nodata values into new nodata values in the data array.
        :param update_mask: Update the old mask by unmasking old nodata and masking new nodata (if array is updated,
            old nodata are changed to new nodata and thus stay masked).
        """
        if new_nodata is not None and not isinstance(new_nodata, (int, float, np.integer, np.floating)):
            raise ValueError("Type of nodata not understood, must be float or int.")

        if new_nodata is not None:
            if not rio.dtypes.can_cast_dtype(new_nodata, self.dtype):
                raise ValueError(f"Nodata value {new_nodata} incompatible with self.dtype {self.dtype}.")

        # If we update mask or array, get the masked array
        if update_array or update_mask:

            # Extract the data variable, so the self.data property doesn't have to be called a bunch of times
            imgdata = self.data

            # Get the index of old nodatas
            index_old_nodatas = imgdata.data == self.nodata

            # Get the index of new nodatas, if it is defined
            index_new_nodatas = imgdata.data == new_nodata

            if np.count_nonzero(index_new_nodatas) > 0:
                if update_array and update_mask:
                    warnings.warn(
                        message="New nodata value cells already exist in the data array. These cells will now be "
                        "masked, and the old nodata value cells will update to the same new value. "
                        "Use set_nodata() with update_array=False or update_mask=False to change "
                        "this behaviour.",
                        category=UserWarning,
                    )
                elif update_array:
                    warnings.warn(
                        "New nodata value cells already exist in the data array. The old nodata cells will update to "
                        "the same new value. Use set_nodata() with update_array=False to change this behaviour.",
                        category=UserWarning,
                    )
                elif update_mask:
                    warnings.warn(
                        "New nodata value cells already exist in the data array. These cells will now be masked. "
                        "Use set_nodata() with update_mask=False to change this behaviour.",
                        category=UserWarning,
                    )

            if update_array:
                # Only update array with new nodata if it is defined
                if new_nodata is not None:
                    # Replace the nodata value in the Raster
                    imgdata.data[index_old_nodatas] = new_nodata

            if update_mask:
                # If a mask already exists, unmask the old nodata values before masking the new ones
                # Can be skipped if array is updated (nodata is transferred from old to new, this part of the mask
                # stays the same)
                if np.ma.is_masked(imgdata) and (not update_array or new_nodata is None):
                    # No way to unmask a value from the masked array, so we modify the mask directly
                    imgdata.mask[index_old_nodatas] = False

                # Masking like this works from the masked array directly, whether a mask exists or not
                imgdata[index_new_nodatas] = np.ma.masked

            # Update the data
            self._data = imgdata

        # Update the nodata value
        self._nodata = new_nodata

        # Update the fill value only if the data is loaded
        if self.is_loaded:
            self.data.fill_value = new_nodata

    @property
    def data(self) -> MArrayNum:
        """
        Array of the raster.

        :returns: Raster array.

        """
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
            dtype = self._disk_dtype
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
                UserWarning,
            )
            self._nodata = _default_nodata(dtype)

        # Now comes the important part, the data setting!
        # Several cases to consider:

        # 1/ If the new data is not masked (either classic array or masked array with no mask, hence the use of
        # as array) and contains non-finite values such as NaNs, define a mask
        if not np.ma.is_masked(new_data) and np.count_nonzero(~np.isfinite(new_data)) > 0:
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

    @property
    def transform(self) -> affine.Affine:
        """
        Geotransform of the raster.

        :returns: Affine matrix geotransform.
        """
        return self._transform

    @transform.setter
    def transform(self, new_transform: tuple[float, ...] | Affine | None) -> None:
        """
        Set the geotransform of the raster.
        """

        if new_transform is None:
            self._transform = None
            return

        if not isinstance(new_transform, Affine):
            if isinstance(new_transform, tuple):
                new_transform = Affine(*new_transform)
            else:
                raise TypeError("The transform argument needs to be Affine or tuple.")

        self._transform = new_transform

    @property
    def crs(self) -> CRS:
        """
        Coordinate reference system of the raster.

        :returns: Pyproj coordinate reference system.
        """
        return self._crs

    @crs.setter
    def crs(self, new_crs: CRS | int | str | None) -> None:
        """
        Set the coordinate reference system of the raster.
        """

        if new_crs is None:
            self._crs = None
            return

        new_crs = CRS.from_user_input(value=new_crs)
        self._crs = new_crs

    @property
    def tags(self) -> dict[str, Any]:
        """
        Metadata tags of the raster.

        :returns: Dictionary of raster metadata, potentially including sensor information.
        """
        return self._tags

    @tags.setter
    def tags(self, new_tags: dict[str, Any] | None) -> None:
        """
        Set the metadata tags of the raster.
        """

        if new_tags is None:
            new_tags = {}
        self._tags = new_tags

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

    @overload
    def get_stats(
        self,
        stats_name: str | Callable[[NDArrayNum], np.floating[Any]],
        inlier_mask: Raster | NDArrayBool | None = None,
        band: int = 1,
        counts: tuple[int, int] | None = None,
    ) -> np.floating[Any]: ...

    @overload
    def get_stats(
        self,
        stats_name: list[str | Callable[[NDArrayNum], np.floating[Any]]] | None = None,
        inlier_mask: Raster | NDArrayBool | None = None,
        band: int = 1,
        counts: tuple[int, int] | None = None,
    ) -> dict[str, np.floating[Any]]: ...

    @profiler.profile("geoutils.raster.raster.get_stats", memprof=True)  # type: ignore
    def get_stats(
        self,
        stats_name: (
            str | Callable[[NDArrayNum], np.floating[Any]] | list[str | Callable[[NDArrayNum], np.floating[Any]]] | None
        ) = None,
        inlier_mask: Raster | NDArrayBool | None = None,
        band: int = 1,
        counts: tuple[int, int] | None = None,
    ) -> np.floating[Any] | dict[str, np.floating[Any]]:
        """
        Retrieve specified statistics or all available statistics for the raster data. Allows passing custom callables
        to calculate custom stats.

        Common statistics are :

        - Mean: arithmetic mean of the data, ignoring masked values.
        - Median: middle value when the valid data points are sorted in increasing order, ignoring masked values.
        - Max: maximum value among the data, ignoring masked values.
        - Min: minimum value among the data, ignoring masked values.
        - Sum: sum of all data, ignoring masked values.
        - Sum of squares: sum of the squares of all data, ignoring masked values.
        - 90th percentile: point below which 90% of the data falls, ignoring masked values.
        - IQR (Interquartile Range): difference between the 75th and 25th percentile of a dataset, \
        ignoring masked values.
        - LE90 (Linear Error with 90% confidence): difference between the 95th and 5th percentiles of a dataset, \
          representing the range within which 90% of the data points lie. Ignore masked values.
        - NMAD (Normalized Median Absolute Deviation): robust measure of variability in the data, \
        less sensitive to outliers compared to standard deviation. Ignore masked values.
        - RMSE (Root Mean Square Error): commonly used to express the magnitude of errors or variability and can give \
          insight into the spread of the data. Only relevant when the raster represents a difference of two objects. \
          Ignore masked values.
        - Std (Standard deviation): measures the spread or dispersion of the data around the mean, \
        ignoring masked values.
        - Valid count: number of finite data points in the array. It counts the non-masked elements.
        - Total count: total size of the raster.
        - Percentage valid points: ratio between Valid count and Total count.

        For all statistics up to and including "Std", NumPy Masked functions are used (directly or in the calculation)
        in case of a masked array, NumPy module otherwise.

        "Valid count" represents all non zero and not masked pixels in the input data (final_count_nonzero),
        calculated before the mask application in case of an inlier_mask. NumPy Masked functions are used is this case
        or if the Raster was already a masked array. "Percentage valid points" is calculated accordingly.

        If an inlier mask is passed:

        - Total inlier count: number of data points in the inlier mask.
        - Valid inlier count: number of unmasked data points in the array after applying the inlier mask.
        - Percentage inlier points: ratio between Valid inlier count and Valid count. Useful for classification \
        statistics.
        - Percentage valid inlier points: ratio between Valid inlier count and Total inlier count.

        They are all computed based on the previously stated final_count_nonzero.

        Callable functions are supported as well.

        :param stats_name: Name or list of names of the statistics to retrieve. If None, all statistics are returned.
            Accepted names include:
            `mean`, `median`, `max`, `min`, `sum`, `sum of squares`, `90th percentile`, `iqr`, `LE90`, `nmad`, `rmse`,
            `std`, `valid count`, `total count`, `percentage valid points` and if an inlier mask is passed :
            `valid inlier count`, `total inlier count`, `percentage inlier point`, `percentage valid inlier points`.
            Custom callables can also be provided.
        :param inlier_mask: Mask or boolean array of areas to include (inliers=True).
        :param band: The index of the band for which to compute statistics. Default is 1.
        :param counts: (number of finite data points in the array, number of valid points (=True, to keep)
            in inlier_mask), initialize in case of a inlier_mask. DO NOT USE.
        :returns: The requested statistic or a dictionary of statistics if multiple or all are requested.
        """
        # Force load if not loaded
        if not self.is_loaded:
            self.load()

        # Get data band
        data = self.data[band - 1, :, :] if self.count > 1 else self.data

        # Derive inlier mask
        if inlier_mask is not None:
            valid_points = np.count_nonzero(np.logical_and(np.isfinite(data), ~data.mask))
            if isinstance(inlier_mask, Raster) and inlier_mask.is_mask:
                inlier_points = np.count_nonzero(inlier_mask.data)
            else:
                inlier_points = np.count_nonzero(inlier_mask)  # type: ignore
            dem_masked = self.copy()

            # Mask pixels from the inlier_mask
            dem_masked.set_mask(~inlier_mask)
            return dem_masked.get_stats(stats_name=stats_name, band=band, counts=(valid_points, inlier_points))

        # Given list or all attributes to compute if None
        if isinstance(stats_name, list) or stats_name is None:
            return _statistics(data, stats_name, counts)  # type: ignore
        else:
            # Single attribute to compute
            if isinstance(stats_name, str):
                return _statistics(data, [stats_name], counts)[stats_name]  # type: ignore
            elif callable(stats_name):
                return stats_name(data)  # type: ignore
            else:
                logging.warning("Statistic name '%s' is a not recognized string", stats_name)

    @overload
    def info(self, stats: bool = False, *, verbose: Literal[True] = ...) -> None: ...

    @overload
    def info(self, stats: bool = False, *, verbose: Literal[False]) -> str: ...

    def info(self, stats: bool = False, verbose: bool = True) -> None | str:
        """
        Print summary information about the raster.

        :param stats: Add statistics for each band of the dataset (max, min, median, mean, std. dev.). Default is to
            not calculate statistics.
        :param verbose: If set to True (default) will directly print to screen and return None

        :returns: Summary string or None.
        """
        as_str = [
            f"Driver:               {self.driver} \n",
            f"Opened from file:     {self.filename} \n",
            f"Filename:             {self.name} \n",
            f"Loaded?               {self.is_loaded} \n",
            f"Modified since load?  {self.is_modified} \n",
            f"Grid size:            {self.width}, {self.height}\n",
            f"Number of bands:      {self.count:d}\n",
            f"Data types:           {self.dtype}\n",
            f"Coordinate system:    {[self.crs.to_string() if self.crs is not None else None]}\n",
            f"Nodata value:         {self.nodata}\n",
            f"Pixel interpretation: {self.area_or_point}\n",
            "Pixel size:           {}, {}\n".format(*self.res),
            f"Upper left corner:    {self.bounds.left}, {self.bounds.top}\n",
            f"Lower right corner:   {self.bounds.right}, {self.bounds.bottom}\n",
        ]

        if stats:
            as_str.append("\nStatistics:\n")
            if not self.is_loaded:
                self.load()

            if self.count == 1:
                statistics = self.get_stats()

                # Determine the maximum length of the stat names for alignment
                max_len = max(len(name) for name in statistics.keys())

                # Format the stats with aligned names
                for name, value in statistics.items():
                    as_str.append(f"{name.ljust(max_len)}: {value:.2f}\n")
            else:
                for b in range(self.count):
                    # try to keep with rasterio convention.
                    as_str.append(f"Band {b + 1}:\n")
                    statistics = self.get_stats(band=b)
                    if isinstance(statistics, dict):
                        max_len = max(len(name) for name in statistics.keys())
                        for name, value in statistics.items():
                            as_str.append(f"{name.ljust(max_len)}: {value:.2f}\n")

        if verbose:
            print("".join(as_str))
            return None
        else:
            return "".join(as_str)

    def copy(self: RasterType, new_array: NDArrayNum | None = None, cast_nodata: bool = True) -> RasterType:
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

    def georeferenced_grid_equal(self: RasterType, raster: RasterType) -> bool:
        """
        Check that raster shape, geotransform and CRS are equal.

        :param raster: Another raster.

        :return: Whether the two objects have the same georeferenced grid.
        """

        return all([self.shape == raster.shape, self.transform == raster.transform, self.crs == raster.crs])

    @overload
    def get_nanarray(
        self, floating_dtype: DTypeLike = "float32", *, return_mask: Literal[False] = False
    ) -> NDArrayNum: ...

    @overload
    def get_nanarray(
        self, floating_dtype: DTypeLike = "float32", *, return_mask: Literal[True]
    ) -> tuple[NDArrayNum, NDArrayBool]: ...

    def get_nanarray(
        self, floating_dtype: DTypeLike = "float32", *, return_mask: bool = False
    ) -> NDArrayNum | tuple[NDArrayNum, NDArrayBool]:
        """
        Get NaN array from the raster.

        Optionally, return the mask from the masked array.

        :param floating_dtype: Floating dtype to convert to, if masked array is not of floating type.
        :param return_mask: Whether to return the mask of valid data.

        :returns Array with masked data as NaNs, (Optional) Mask of invalid data.
        """

        # Cast array to float32 is its dtype is integer (cannot be filled with NaNs otherwise)
        if "int" in str(self.data.dtype):
            # Get the array with masked value fill with NaNs
            nanarray = self.data.astype(floating_dtype).filled(fill_value=np.nan).squeeze()
        else:
            # Same here
            nanarray = self.data.filled(fill_value=np.nan).squeeze()

        # The function np.ma.filled() only returns a copy if the array is masked, copy the array if it's not the case
        if not np.ma.is_masked(self.data):
            nanarray = np.copy(nanarray)

        # Return the NaN array, and possibly the mask as well
        if return_mask:
            return nanarray, np.copy(np.ma.getmaskarray(self.data).squeeze())
        else:
            return nanarray

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
            # TODO: Use nodata value derived here after fixing issue #517
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
                warnings.warn("Applying np.gradient to first raster band only.")
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
            # TODO: Use nodata below, but after fixing issue #517
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

    # Note the star is needed because of the default argument 'mode' preceding non default arg 'inplace'
    # Then the final overload must be duplicated
    # Also note that in the first overload, only "inplace: Literal[False]" does not work. The ellipsis is
    # essential, otherwise MyPy gives incompatible return type Optional[Raster].
    @overload
    def crop(
        self: RasterType,
        bbox: RasterType | gu.Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        *,
        inplace: Literal[False] = False,
    ) -> RasterType: ...

    @overload
    def crop(
        self: RasterType,
        bbox: RasterType | gu.Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        *,
        inplace: Literal[True],
    ) -> None: ...

    @overload
    def crop(
        self: RasterType,
        bbox: RasterType | gu.Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        *,
        inplace: bool = False,
    ) -> RasterType | None: ...

    @profiler.profile("geoutils.raster.raster.crop", memprof=True)  # type: ignore
    def crop(
        self: RasterType,
        bbox: RasterType | gu.Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        *,
        inplace: bool = False,
    ) -> RasterType | None:
        """
        Crop the raster to a given extent.

        **Match-reference:** a reference raster or vector can be passed to match bounds during cropping.

        Reprojection is done on the fly if georeferenced objects have different projections.

        :param bbox: Geometry to crop raster to. Can use either a raster or vector as match-reference, or a list of
            coordinates. If ``bbox`` is a raster or vector, will crop to the bounds. If ``bbox`` is a
            list of coordinates, the order is assumed to be [xmin, ymin, xmax, ymax].
        :param mode: Whether to match within pixels or exact extent. ``'match_pixel'`` will preserve the original pixel
            resolution, cropping to the extent that most closely aligns with the current coordinates. ``'match_extent'``
            will match the extent exactly, adjusting the pixel resolution to fit the extent.
        :param inplace: Whether to update the raster in-place.

        :returns: A new raster (or None if inplace).
        """

        crop_img, tfm = _crop(source_raster=self, bbox=bbox, mode=mode)

        if inplace:
            self._data = crop_img
            self.transform = tfm
            return None
        else:
            newraster = self.from_array(crop_img, tfm, self.crs, self.nodata, self.area_or_point)
            return newraster

    @overload
    def icrop(
        self: RasterType,
        bbox: list[int] | tuple[int, ...],
        *,
        inplace: Literal[True],
    ) -> None: ...

    @overload
    def icrop(
        self: RasterType,
        bbox: list[int] | tuple[int, ...],
        *,
        inplace: Literal[False] = False,
    ) -> RasterType: ...

    @profiler.profile("geoutils.raster.raster.icrop", memprof=True)  # type: ignore
    def icrop(
        self: RasterType,
        bbox: list[int] | tuple[int, ...],
        *,
        inplace: bool = False,
    ) -> RasterType | None:
        """
        Crop raster based on pixel indices (bbox), converting them into georeferenced coordinates.

        :param bbox: Bounding box based on indices of the raster array (colmin, rowmin, colmax, rowmax).
        :param inplace: If True, modify the raster in place. Otherwise, return a new cropped raster.

        :returns: Cropped raster or None (if inplace=True).
        """
        crop_img, tfm = _crop(source_raster=self, bbox=bbox, distance_unit="pixel")

        if inplace:
            self._data = crop_img
            self.transform = tfm
            return None
        else:
            newraster = self.from_array(crop_img, tfm, self.crs, self.nodata, self.area_or_point)
            return newraster

    @overload
    def reproject(
        self: RasterType,
        ref: RasterType | str | None = None,
        crs: CRS | str | int | None = None,
        res: float | abc.Iterable[float] | None = None,
        grid_size: tuple[int, int] | None = None,
        bounds: dict[str, float] | rio.coords.BoundingBox | None = None,
        nodata: int | float | None = None,
        dtype: DTypeLike | None = None,
        resampling: Resampling | str = Resampling.bilinear,
        force_source_nodata: int | float | None = None,
        *,
        inplace: Literal[False] = False,
        silent: bool = False,
        n_threads: int = 0,
        memory_limit: int = 64,
        multiproc_config: MultiprocConfig | None = None,
    ) -> RasterType: ...

    @overload
    def reproject(
        self: RasterType,
        ref: RasterType | str | None = None,
        crs: CRS | str | int | None = None,
        res: float | abc.Iterable[float] | None = None,
        grid_size: tuple[int, int] | None = None,
        bounds: dict[str, float] | rio.coords.BoundingBox | None = None,
        nodata: int | float | None = None,
        dtype: DTypeLike | None = None,
        resampling: Resampling | str = Resampling.bilinear,
        force_source_nodata: int | float | None = None,
        *,
        inplace: Literal[True],
        silent: bool = False,
        n_threads: int = 0,
        memory_limit: int = 64,
        multiproc_config: MultiprocConfig | None = None,
    ) -> None: ...

    @profiler.profile("geoutils.raster.raster.reproject", memprof=True)  # type: ignore
    def reproject(
        self: RasterType,
        ref: RasterType | str | None = None,
        crs: CRS | str | int | None = None,
        res: float | abc.Iterable[float] | None = None,
        grid_size: tuple[int, int] | None = None,
        bounds: dict[str, float] | rio.coords.BoundingBox | None = None,
        nodata: int | float | None = None,
        dtype: DTypeLike | None = None,
        resampling: Resampling | str = Resampling.bilinear,
        force_source_nodata: int | float | None = None,
        inplace: bool = False,
        silent: bool = False,
        n_threads: int = 0,
        memory_limit: int = 64,
        multiproc_config: MultiprocConfig | None = None,
    ) -> RasterType | None:
        """
        Reproject raster to a different geotransform (resolution, bounds) and/or coordinate reference system (CRS).

        **Match-reference**: a reference raster can be passed to match resolution, bounds and CRS during reprojection.

        Alternatively, the destination resolution, bounds and CRS can be passed individually.

        Any resampling algorithm implemented in Rasterio can be passed as a string.

        The reprojection can be computed out-of-memory in multiprocessing by passing a
        :class:`~geoutils.raster.MultiprocConfig` object.
        The reprojected raster is written to disk under the path specified in the configuration

        :param ref: Reference raster to match resolution, bounds and CRS.
        :param crs: Destination coordinate reference system as a string or EPSG. If ``ref`` not set,
            defaults to this raster's CRS.
        :param res: Destination resolution (pixel size) in units of destination CRS. Single value or (xres, yres).
            Do not use with ``grid_size``.
        :param grid_size: Destination grid size as (x, y). Do not use with ``res``.
        :param bounds: Destination bounds as a Rasterio bounding box, or a dictionary containing left, bottom,
            right, top bounds in the destination CRS.
        :param nodata: Destination nodata value. If set to ``None``, will use the same as source. If source does
            not exist, will use GDAL's default.
        :param dtype: Destination data type of array.
        :param resampling: A Rasterio resampling method, can be passed as a string.
            See https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling
            for the full list.
        :param inplace: Whether to update the raster in-place.
        :param force_source_nodata: Force a source nodata value (read from the metadata by default).
        :param silent: Whether to print warning statements.
        :param n_threads: Number of threads. Defaults to (os.cpu_count() - 1).
        :param memory_limit: Memory limit in MB for warp operations. Larger values may perform better.
        :param multiproc_config: Configuration object containing chunk size, output file path, and an optional cluster.

        :returns: Reprojected raster (or None if inplace or computed out-of-memory).

        """
        # Reproject
        return_copy, data, transformed, crs, nodata = _reproject(
            source_raster=self,
            ref=ref,
            crs=crs,
            res=res,
            grid_size=grid_size,
            bounds=bounds,
            nodata=nodata,
            dtype=dtype,
            resampling=resampling,
            force_source_nodata=force_source_nodata,
            silent=silent,
            n_threads=n_threads,
            memory_limit=memory_limit,
            multiproc_config=multiproc_config,
        )

        # If return copy is True (target georeferenced grid was the same as input)
        if return_copy:
            if inplace:
                return None
            else:
                return self

        # If multiprocessing -> results on disk -> load metadata
        if multiproc_config:
            result_raster = Raster(multiproc_config.outfile)
            if inplace:
                crs = result_raster.crs
                nodata = result_raster.nodata
                transformed = result_raster.transform
                data = result_raster.data
            else:
                return result_raster  # type: ignore

        # To make MyPy happy without overload for _reproject (as it might re-structured soon anyway)
        assert data is not None
        assert transformed is not None
        assert crs is not None

        # Write results to a new Raster.
        if inplace:
            # Order is important here, because calling self.data will use nodata to mask the array properly
            self._crs = crs
            self._nodata = nodata
            self._transform = transformed
            # A little trick to force the right shape of data in, then update the mask properly through the data setter
            self._data = data.squeeze()
            self.data = data
            return None
        else:
            return self.from_array(data, transformed, crs, nodata, self.area_or_point)

    @overload
    def translate(
        self: RasterType,
        xoff: float,
        yoff: float,
        distance_unit: Literal["georeferenced"] | Literal["pixel"] = "georeferenced",
        *,
        inplace: Literal[False] = False,
    ) -> RasterType: ...

    @overload
    def translate(
        self: RasterType,
        xoff: float,
        yoff: float,
        distance_unit: Literal["georeferenced"] | Literal["pixel"] = "georeferenced",
        *,
        inplace: Literal[True],
    ) -> None: ...

    @overload
    def translate(
        self: RasterType,
        xoff: float,
        yoff: float,
        distance_unit: Literal["georeferenced"] | Literal["pixel"] = "georeferenced",
        *,
        inplace: bool = False,
    ) -> RasterType | None: ...

    def translate(
        self: RasterType,
        xoff: float,
        yoff: float,
        distance_unit: Literal["georeferenced", "pixel"] = "georeferenced",
        inplace: bool = False,
    ) -> RasterType | None:
        """
        Translate a raster by a (x,y) offset.

        The translation only updates the geotransform (no resampling is performed).

        :param xoff: Translation x offset.
        :param yoff: Translation y offset.
        :param distance_unit: Distance unit, either 'georeferenced' (default) or 'pixel'.
        :param inplace: Whether to modify the raster in-place.

        :returns: Translated raster (or None if inplace).
        """

        translated_transform = _translate(self.transform, xoff=xoff, yoff=yoff, distance_unit=distance_unit)

        if inplace:
            # Overwrite transform by translated transform
            self.transform = translated_transform
            return None
        else:
            raster_copy = self.copy()
            raster_copy.transform = translated_transform
            return raster_copy

    def to_file(
        self,
        filename: str | pathlib.Path | IO[bytes],
        driver: str = "GTiff",
        dtype: DTypeLike | None = None,
        nodata: Number | None = None,
        compress: str = "deflate",
        tiled: bool = False,
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

        :param filename: Filename to write the file to.
        :param driver: Driver to write file with.
        :param dtype: Data type to write the image as (defaults to dtype of image data).
        :param nodata: Force a nodata value to be used (default to that of raster).
        :param compress: Compression type. Defaults to 'deflate' (equal to GDALs: COMPRESS=DEFLATE).
        :param tiled: Whether to write blocks in tiles instead of strips. Improves read performance on large files,
            but increases file size.
        :param blank_value: Use to write an image out with every pixel's value.
            corresponding to this value, instead of writing the image data to disk.
        :param co_opts: GDAL creation options provided as a dictionary,
            e.g. {'TILED':'YES', 'COMPRESS':'LZW'}.
        :param metadata: Pairs of metadata to save to disk, in addition to existing metadata in self.tags.
        :param gcps: List of gcps, each gcp being [row, col, x, y, (z)].
        :param gcps_crs: CRS of the GCPS.

        :returns: None.
        """

        if co_opts is None:
            co_opts = {}
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
                    warnings.warn(f"No nodata set, will use default value of {nodata}")
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
            compress=compress,
            tiled=tiled,
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
                    warnings.warn("A geotransform previously set is going to be cleared due to the setting of GCPs.")

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
        compress: str = "deflate",
        tiled: bool = False,
        blank_value: int | float | None = None,
        co_opts: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        gcps: list[tuple[float, ...]] | None = None,
        gcps_crs: CRS | None = None,
    ) -> None:
        self.to_file(filename, driver, dtype, nodata, compress, tiled, blank_value, co_opts, metadata, gcps, gcps_crs)

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

        # TODO: Add tags and area_or_point with PR #509
        raster = cls.from_array(data=ds.data, transform=transform, crs=crs, nodata=nodata)

        if dtype is not None:
            raster = raster.astype(dtype)

        return raster

    def to_xarray(self, name: str | None = None) -> xr.DataArray:
        """
        Convert raster to a xarray.DataArray.

        This converts integer-type rasters into float32.

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

    def get_bounds_projected(self, out_crs: CRS, densify_points: int = 5000) -> rio.coords.BoundingBox:
        """
        Get raster bounds projected in a specified CRS.

        :param out_crs: Output CRS.
        :param densify_points: Maximum points to be added between image corners to account for non linear edges.
         Reduce if time computation is really critical (ms) or increase if extent is not accurate enough.

        """
        # Max points to be added between image corners to account for non linear edges
        # rasterio's default is a bit low for very large images
        # instead, use image dimensions, with a maximum of 50000
        densify_points = min(max(self.width, self.height), densify_points)

        # Calculate new bounds
        new_bounds = _get_bounds_projected(self.bounds, in_crs=self.crs, out_crs=out_crs, densify_points=densify_points)

        return new_bounds

    def get_footprint_projected(self, out_crs: CRS, densify_points: int = 5000) -> gu.Vector:
        """
        Get raster footprint projected in a specified CRS.

        The polygon points of the vector are densified during reprojection to warp
        the rectangular square footprint of the original projection into the new one.

        :param out_crs: Output CRS.
        :param densify_points: Maximum points to be added between image corners to account for non linear edges.
         Reduce if time computation is really critical (ms) or increase if extent is not accurate enough.
        """

        return gu.Vector(
            _get_footprint_projected(
                bounds=self.bounds, in_crs=self.crs, out_crs=out_crs, densify_points=densify_points
            )
        )

    def get_metric_crs(
        self,
        local_crs_type: Literal["universal"] | Literal["custom"] = "universal",
        method: Literal["centroid"] | Literal["geopandas"] = "centroid",
    ) -> CRS:
        """
        Get local metric coordinate reference system for the raster (UTM, UPS, or custom Mercator or Polar).

        :param local_crs_type: Whether to get a "universal" local CRS (UTM or UPS) or a "custom" local CRS
            (Mercator or Polar centered on centroid).
        :param method: Method to choose the zone of the CRS, either based on the centroid of the footprint
            or the extent as implemented in :func:`geopandas.GeoDataFrame.estimate_utm_crs`.
            Forced to centroid if `local_crs="custom"`.
        """

        # For universal CRS (UTM or UPS)
        if local_crs_type == "universal":
            return _get_utm_ups_crs(self.get_footprint_projected(out_crs=self.crs).ds, method=method)
        # For a custom CRS
        else:
            raise NotImplementedError("This is not implemented yet.")

    def intersection(self, raster: str | Raster, match_ref: bool = True) -> tuple[float, float, float, float]:
        """
        Returns the bounding box of intersection between this image and another.

        If the rasters have different projections, the intersection extent is given in self's projection system.

        :param raster : path to the second image (or another Raster instance)
        :param match_ref: if set to True, returns the smallest intersection that aligns with that of self, i.e. same \
        resolution and offset with self's origin is a multiple of the resolution
        :returns: extent of the intersection between the 2 images \
        (xmin, ymin, xmax, ymax) in self's coordinate system.

        """
        from geoutils import projtools

        # If input raster is string, open as Raster
        if isinstance(raster, str):
            raster = Raster(raster, load_data=False)

        # Reproject the bounds of raster to self's
        raster_bounds_sameproj = raster.get_bounds_projected(self.crs)  # type: ignore

        # Calculate intersection of bounding boxes
        intersection = projtools.merge_bounds([self.bounds, raster_bounds_sameproj], merging_algorithm="intersection")

        # Check that intersection is not void (changed to NaN instead of empty tuple end 2022)
        if intersection == () or all(math.isnan(i) for i in intersection):
            warnings.warn("Intersection is void")
            return (0.0, 0.0, 0.0, 0.0)

        # if required, ensure the intersection is aligned with self's georeferences
        if match_ref:
            intersection = projtools.align_bounds(self.transform, intersection)

        # mypy raises a type issue, not sure how to address the fact that output of merge_bounds can be ()
        return intersection  # type: ignore

    @overload
    def plot(
        self,
        bands: int | tuple[int, ...] | None = None,
        cmap: matplotlib.colors.Colormap | str | None = None,
        vmin: float | int | None = None,
        vmax: float | int | None = None,
        alpha: float | int | None = None,
        cbar_title: str | None = None,
        add_cbar: bool = True,
        ax: matplotlib.axes.Axes | Literal["new"] | None = None,
        *,
        return_axes: Literal[False] = False,
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
        cbar_title: str | None = None,
        add_cbar: bool = True,
        ax: matplotlib.axes.Axes | Literal["new"] | None = None,
        *,
        return_axes: Literal[True],
        **kwargs: Any,
    ) -> tuple[matplotlib.axes.Axes, matplotlib.colors.Colormap]: ...

    def plot(
        self,
        bands: int | tuple[int, ...] | None = None,
        cmap: matplotlib.colors.Colormap | str | None = None,
        vmin: float | int | None = None,
        vmax: float | int | None = None,
        alpha: float | int | None = None,
        cbar_title: str | None = None,
        add_cbar: bool = True,
        ax: matplotlib.axes.Axes | Literal["new"] | None = None,
        return_axes: bool = False,
        **kwargs: Any,
    ) -> None | tuple[matplotlib.axes.Axes, matplotlib.colors.Colormap]:
        r"""
        Plot the raster, with axes in projection of image.

        This method is a wrapper to rasterio.plot.show. Any \*\*kwargs which
        you give this method will be passed to it.

        :param bands: Bands to plot, from 1 to self.count (default is all).
        :param cmap: The figure's colormap. Default is plt.rcParams['image.cmap'].
        :param vmin: Colorbar minimum value. Default is data min.
        :param vmax: Colorbar maximum value. Default is data max.
        :param alpha: Transparency of raster and colorbar.
        :param cbar_title: Colorbar label. Default is None.
        :param add_cbar: Set to True to display a colorbar. Default is True.
        :param ax: A figure ax to be used for plotting. If None, will plot on current axes.
            If "new", will create a new axis.
        :param return_axes: Whether to return axes.

        :returns: None, or (ax, caxes) if return_axes is True.


        You can also pass in \*\*kwargs to be used by the underlying imshow or
        contour methods of matplotlib. The example below shows provision of
        a kwarg for rasterio.plot.show, and a kwarg for matplotlib as well::

            import matplotlib.pyplot as plt
            ax1 = plt.subplot(111)
            mpl_kws = {'cmap':'seismic'}
            myimage.plot(ax=ax1, mpl_kws)
        """
        # If data is not loaded, need to load it
        if not self.is_loaded:
            self.load()

        # Set matplotlib interpolation to None by default, to avoid spreading gaps in plots
        if "interpolation" not in kwargs.keys():
            kwargs.update({"interpolation": None})

        # Check if specific band selected, or take all
        # rshow takes care of image dimensions
        # if self.count=3 (4) => plotted as RGB(A)
        if bands is None:
            bands = tuple(range(1, self.count + 1))
        elif isinstance(bands, int):
            if bands > self.count:
                raise ValueError(f"Index must be in range 1-{self.count:d}")
            pass
        else:
            raise ValueError("Index must be int or None")

        # Get data
        if self.count == 1:
            data = self.data
        else:
            data = self.data[np.array(bands) - 1, :, :]

        # If multiple bands (RGB), cbar does not make sense
        if isinstance(bands, abc.Sequence):
            if len(bands) > 1:
                add_cbar = False

        # Create colorbar
        # Use rcParam default
        if cmap is None:
            cmap = plt.get_cmap(plt.rcParams["image.cmap"])
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
        rshow(
            data,
            transform=self.transform,
            ax=ax0,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            **kwargs,
        )

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
            # TODO: Check conversion is not done for nothing?
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
        rows, cols = rio.transform.rowcol(self.transform, x, y, op=floor)

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
                # TODO: if we want to allow sampling multiple bands, need to do it also when data is loaded
                # if self.count == 1:
                with rio.open(self.filename) as raster:
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
                #     with rio.open(self.filename) as raster:
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

    def xy2ij(
        self,
        x: ArrayLike,
        y: ArrayLike,
        op: type = np.float32,
        precision: float | None = None,
        shift_area_or_point: bool | None = None,
    ) -> tuple[NDArrayNum, NDArrayNum]:
        """
        Get indexes (row,column) of coordinates (x,y).

        By default, the indexes are shifted with the interpretation of pixel coordinates "AREA_OR_POINT" of the raster,
        to ensure that the indexes of points represent the right location. See parameter description of
        shift_area_or_point for more details.

        This function is reversible with ij2xy for any pixel interpretation.

        :param x: X coordinates.
        :param y: Y coordinates.
        :param op: Operator to compute index.
        :param precision: Precision passed to :func:`rasterio.transform.rowcol`.
        :param shift_area_or_point: Whether to shift with pixel interpretation, which shifts to center of pixel
            indexes if self.area_or_point is "Point" and maintains corner pixel indexes if it is "Area" or None.
            Defaults to True. Can be configured with the global setting geoutils.config["shift_area_or_point"].

        :returns i, j: Indices of (x,y) in the image.
        """

        return _xy2ij(
            x=x,
            y=y,
            transform=self.transform,
            area_or_point=self.area_or_point,
            op=op,
            precision=precision,
            shift_area_or_point=shift_area_or_point,
        )

    def ij2xy(
        self, i: ArrayLike, j: ArrayLike, shift_area_or_point: bool | None = None, force_offset: str | None = None
    ) -> tuple[NDArrayNum, NDArrayNum]:
        """
        Get coordinates (x,y) of indexes (row,column).

        By default, the indexes are shifted with the interpretation of pixel coordinates "AREA_OR_POINT" of the
        raster, to ensure that the indexes of points represent the right location. See parameter description of
        shift_area_or_point for more details.

        This function is reversible with xy2ij for any pixel interpretation.

        :param i: Row (i) index of pixel.
        :param j: Column (j) index of pixel.
        :param shift_area_or_point: Whether to shift with pixel interpretation, which shifts to center of pixel
            coordinates if self.area_or_point is "Point" and maintains corner pixel coordinate if it is "Area" or None.
            Defaults to True. Can be configured with the global setting geoutils.config["shift_area_or_point"].
        :param force_offset: Ignore pixel interpretation and force coordinate to a certain offset: "center" of pixel, or
            any corner (upper-left "ul", "ur", "ll", lr"). Default coordinate of a raster is upper-left.

        :returns x, y: x,y coordinates of i,j in reference system.
        """

        return _ij2xy(
            i=i,
            j=j,
            transform=self.transform,
            area_or_point=self.area_or_point,
            shift_area_or_point=shift_area_or_point,
            force_offset=force_offset,
        )

    def coords(
        self, grid: bool = True, shift_area_or_point: bool | None = None, force_offset: str | None = None
    ) -> tuple[NDArrayNum, NDArrayNum]:
        """
        Get coordinates (x,y) of all pixels in the raster.

        :param grid: Whether to return mesh grids of coordinates matrices.
        :param shift_area_or_point: Whether to shift with pixel interpretation, which shifts to center of pixel
            coordinates if self.area_or_point is "Point" and maintains corner pixel coordinate if it is "Area" or None.
            Defaults to True. Can be configured with the global setting geoutils.config["shift_area_or_point"].
        :param force_offset: Ignore pixel interpretation and force coordinate to a certain offset: "center" of pixel, or
            any corner (upper-left "ul", "ur", "ll", lr"). Default coordinate of a raster is upper-left.

        :returns x,y: Arrays of the (x,y) coordinates.
        """

        return _coords(
            transform=self.transform,
            shape=self.shape,
            area_or_point=self.area_or_point,
            grid=grid,
            shift_area_or_point=shift_area_or_point,
            force_offset=force_offset,
        )

    def outside_image(self, xi: ArrayLike, yj: ArrayLike, index: bool = True) -> bool:
        """
        Check whether a given point falls outside the raster.

        :param xi: Indices (or coordinates) of x direction to check.
        :param yj: Indices (or coordinates) of y direction to check.
        :param index: Interpret ij as raster indices (default is ``True``). If False, assumes ij is coordinates.

        :returns is_outside: ``True`` if ij is outside the image.
        """

        return _outside_image(
            xi=xi, yj=yj, transform=self.transform, shape=self.shape, area_or_point=self.area_or_point, index=index
        )

    @overload
    def interp_points(
        self,
        points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum] | gu.PointCloud,
        method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
        dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int = "half_order_up",
        band: int = 1,
        input_latlon: bool = False,
        *,
        as_array: Literal[False] = False,
        shift_area_or_point: bool | None = None,
        force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
        **kwargs: Any,
    ) -> gu.PointCloud: ...

    @overload
    def interp_points(
        self,
        points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum] | gu.PointCloud,
        method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
        dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int = "half_order_up",
        band: int = 1,
        input_latlon: bool = False,
        *,
        as_array: Literal[True],
        shift_area_or_point: bool | None = None,
        force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
        **kwargs: Any,
    ) -> NDArrayNum: ...

    @overload
    def interp_points(
        self,
        points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum] | gu.PointCloud,
        method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
        dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int = "half_order_up",
        band: int = 1,
        input_latlon: bool = False,
        *,
        as_array: bool = False,
        shift_area_or_point: bool | None = None,
        force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
        **kwargs: Any,
    ) -> NDArrayNum | gu.PointCloud: ...

    @profiler.profile("geoutils.raster.raster.interp_points", memprof=True)  # type: ignore
    def interp_points(
        self,
        points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum] | gu.PointCloud,
        method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
        dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int = "half_order_up",
        band: int = 1,
        input_latlon: bool = False,
        as_array: bool = False,
        shift_area_or_point: bool | None = None,
        force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
        **kwargs: Any,
    ) -> NDArrayNum | gu.PointCloud:
        """
         Interpolate raster values at a set of points.

         Returns a point cloud with data column the interpolated values at the point coordinates, or optionally just
         the array of interpolated rvalues.

         Uses scipy.ndimage.map_coordinates if the Raster is on an equal grid using "nearest" or "linear" (for speed),
         otherwise uses scipy.interpn on a regular grid.

         Optionally, user can enforce the interpretation of pixel coordinates in self.tags['AREA_OR_POINT']
         to ensure that the interpolation of points is done at the right location. See parameter description
         of shift_area_or_point for more details.

        :param points: Point(s) at which to interpolate raster value. Can be either a tuple of array-like of X/Y
            coordinates (same CRS as raster or latitude/longitude, see "input_latlon") or a pointcloud in any CRS.
            If points fall outside of image, value returned is nan.
        :param method: Interpolation method, one of 'nearest', 'linear', 'cubic', 'quintic', 'slinear', 'pchip' or
            'splinef2d'. For more information, see scipy.ndimage.map_coordinates and scipy.interpolate.interpn.
            Default is linear.
        :param dist_nodata_spread: Distance of nodata spreading during interpolation, either half-interpolation order
            rounded up (default; equivalent to 0 for nearest, 1 for linear methods, 2 for cubic methods and 3 for
            quintic method), or rounded down, or a fixed integer.
        :param band: Band to use (from 1 to self.count).
        :param input_latlon: (Only for tuple point input) Whether to convert input coordinates from latlon to raster
            CRS.
        :param as_array: Whether to return a point cloud with data column the interpolated values (default) or an
            array of interpolated values.
        :param shift_area_or_point: Whether to shift with pixel interpretation, which shifts to center of pixel
            coordinates if self.area_or_point is "Point" and maintains corner pixel coordinate if it is "Area" or None.
            Defaults to True. Can be configured with the global setting geoutils.config["shift_area_or_point"].
        :param force_scipy_function: Force to use either map_coordinates or interpn. Mainly for testing purposes.

        :returns Point cloud of interpolated points, or 1D array of interpolated values.
        """

        # Extract array supporting NaNs
        array = self.get_nanarray()
        if self.count != 1:
            array = array[band - 1, :, :]

        # If point cloud input
        if isinstance(points, gu.PointCloud):
            # TODO: Check conversion is not done for nothing?
            points = reproject_points((points.ds.geometry.x.values, points.ds.geometry.y.values), points.crs, self.crs)
        # Otherwise
        else:
            if input_latlon:
                points = reproject_from_latlon(points, out_crs=self.crs)  # type: ignore

        z = _interp_points(
            array,
            transform=self.transform,
            area_or_point=self.area_or_point,
            points=points,
            method=method,
            shift_area_or_point=shift_area_or_point,
            dist_nodata_spread=dist_nodata_spread,
            force_scipy_function=force_scipy_function,
            **kwargs,
        )

        # Return array or pointcloud
        if as_array:
            return z
        else:
            return gu.PointCloud.from_xyz(x=points[0], y=points[1], z=z, crs=self.crs)

    @overload
    def filter(
        self: RasterType,
        method: str | Callable[..., NDArrayNum],
        *,
        inplace: Literal[False] = False,
        size: int = 3,
        **kwargs: dict[str, Any],
    ) -> RasterType: ...

    @overload
    def filter(
        self: RasterType,
        method: str | Callable[..., NDArrayNum],
        *,
        inplace: Literal[True],
        size: int = 3,
        **kwargs: dict[str, Any],
    ) -> None: ...

    def filter(
        self: RasterType,
        method: str | Callable[..., NDArrayNum],
        inplace: bool = False,
        size: int = 3,
        **kwargs: dict[str, Any],
    ) -> RasterType | None:
        """
        Apply a filter to the array.

        :param method: The filter to apply. Can be a string ("gaussian", "median", "mean", "max", "min", "distance")
                       for built-in filters, or a custom callable that takes a 2D ndarray and returns one.
        :param inplace: Whether to modify the raster in-place.
        :param size: window size for filter

        :return: A new Raster instance with the filtered data (or None if inplace)

        :raises ValueError: If the filter name is not one of the predefined options.
        :raises TypeError: If `method` is neither a string nor a callable.
        """
        # Convert data to float to avoid integer issues with nodata
        array = self.data.astype(float)

        # Mask nodata values
        masked_array = np.ma.masked_equal(array, self.nodata)

        # Fill masked values with nodata for filtering, to match SciPy behavior
        filled_array = masked_array.filled(self.nodata)

        # Apply filter
        filtered_array = _filter(filled_array, method, size, **kwargs)

        # Mask nodata again after filtering
        final_masked = np.ma.masked_equal(filtered_array, self.nodata)
        final_masked.set_fill_value(self.nodata)
        final_masked = final_masked.astype(float)

        if inplace:
            self._data = final_masked
            return None
        else:
            return self.from_array(final_masked, self.transform, self.crs, self.nodata, self.area_or_point)

    def split_bands(self: RasterType, copy: bool = False, bands: list[int] | int | None = None) -> list[Raster]:
        """
        Split the bands into separate rasters.

        :param copy: Copy the bands or return slices of the original data.
        :param bands: Optional. A list of band indices to extract (from 1 to self.count). Defaults to all.

        :returns: A list of Rasters for each band.
        """
        raster_bands: list[Raster] = []

        if bands is None:
            indices = list(np.arange(1, self.count + 1))
        elif isinstance(bands, int):
            indices = [bands]
        elif isinstance(bands, list):
            indices = bands
        else:
            raise ValueError(f"'subset' got invalid type: {type(bands)}. Expected list[int], int or None")

        if copy:
            for band_n in indices:
                # Generate a new Raster from a copy of the band's data
                raster_bands.append(
                    self.copy(
                        self.data[band_n - 1, :, :].copy(),
                    )
                )
        else:
            for band_n in indices:
                # Set the data to a slice of the original array
                raster_bands.append(
                    self.copy(
                        self.data[band_n - 1, :, :],
                    )
                )

        return raster_bands

    @deprecate(
        Version("0.3.0"),
        "Raster.to_points() is deprecated in favor of Raster.to_pointcloud() and " "will be removed in v0.3.",
    )
    def to_points(self, **kwargs):  # type: ignore

        self.to_pointcloud(**kwargs)  # type: ignore

    @overload
    def to_pointcloud(
        self,
        data_column_name: str = "b1",
        data_band: int = 1,
        auxiliary_data_bands: list[int] | None = None,
        auxiliary_column_names: list[str] | None = None,
        subsample: float | int = 1,
        skip_nodata: bool = True,
        *,
        as_array: Literal[False] = False,
        random_state: int | np.random.Generator | None = None,
        force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"] = "ul",
    ) -> NDArrayNum: ...

    @overload
    def to_pointcloud(
        self,
        data_column_name: str = "b1",
        data_band: int = 1,
        auxiliary_data_bands: list[int] | None = None,
        auxiliary_column_names: list[str] | None = None,
        subsample: float | int = 1,
        skip_nodata: bool = True,
        *,
        as_array: Literal[True],
        random_state: int | np.random.Generator | None = None,
        force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"] = "ul",
    ) -> gu.Vector: ...

    @overload
    def to_pointcloud(
        self,
        data_column_name: str = "b1",
        data_band: int = 1,
        auxiliary_data_bands: list[int] | None = None,
        auxiliary_column_names: list[str] | None = None,
        subsample: float | int = 1,
        skip_nodata: bool = True,
        *,
        as_array: bool = False,
        random_state: int | np.random.Generator | None = None,
        force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"] = "ul",
    ) -> NDArrayNum | gu.Vector: ...

    def to_pointcloud(
        self,
        data_column_name: str = "b1",
        data_band: int = 1,
        auxiliary_data_bands: list[int] | None = None,
        auxiliary_column_names: list[str] | None = None,
        subsample: float | int = 1,
        skip_nodata: bool = True,
        as_array: bool = False,
        random_state: int | np.random.Generator | None = None,
        force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"] = "ul",
    ) -> NDArrayNum | gu.PointCloud:
        """
        Convert raster to point cloud.

        A point cloud is a vector of point geometries associated to a data column, and possibly other auxiliary data
        columns, see geoutils.PointCloud.

        For a single band raster, the main data column name of the point cloud defaults to "b1" and stores values of
        that single band.
        For a multi-band raster, the main data column name of the point cloud defaults to "bX" where X is the data band
        index chosen by the user (defaults to 1, the first band).
        Optionally, all other bands can also be stored in columns "b1", "b2", etc. For more specific band selection,
        use Raster.split_bands previous to converting to point cloud.

        Optionally, randomly subsample valid pixels for the data band (nodata values can be skipped, but only for the
        band that will be used as data column of the point cloud).
        If 'subsample' is either 1, or is equal to the pixel count, all (valid) points are returned.
        If 'subsample' is smaller than 1 (for fractions), or smaller than the pixel count, a random subsample
        of (valid) points is returned.

        If the raster is not loaded, sampling will be done from disk using rasterio.sample after loading only the masks
        of the dataset.

        Formats:
            * `as_array` == False: A vector with dataframe columns ["b1", "b2", ..., "geometry"],
            * `as_array` == True: A numpy ndarray of shape (N, 2 + count) with the columns [x, y, b1, b2..].

        :param data_column_name: Name to use for point cloud data column, defaults to "bX" where X is the data band
            number.
        :param data_band: (Only for multi-band rasters) Band to use for data column, defaults to first. Band counting
            starts at 1.
        :param auxiliary_data_bands: (Only for multi-band rasters) Whether to save other band numbers as auxiliary data
            columns, defaults to none.
        :param auxiliary_column_names: (Only for multi-band rasters) Names to use for auxiliary data bands, only if
            auxiliary data bands is not none, defaults to "b1", "b2", etc.
        :param subsample: Subsample size. If > 1, parsed as a count, otherwise a fraction.
        :param skip_nodata: Whether to skip nodata values.
        :param as_array: Return an array instead of a vector.
        :param random_state: Random state or seed number.
        :param force_pixel_offset: Force offset to derive point coordinate with. Raster coordinates normally only
            associate to upper-left corner "ul" ("Area" definition) or center ("Point" definition).

        :raises ValueError: If the sample count or fraction is poorly formatted.

        :returns: A point cloud, or array of the shape (N, 2 + count) where N is the sample count.
        """

        return _raster_to_pointcloud(
            source_raster=self,
            data_column_name=data_column_name,
            data_band=data_band,
            auxiliary_data_bands=auxiliary_data_bands,
            auxiliary_column_names=auxiliary_column_names,
            subsample=subsample,
            skip_nodata=skip_nodata,
            as_array=as_array,
            random_state=random_state,
            force_pixel_offset=force_pixel_offset,
        )

    @classmethod
    def from_pointcloud_regular(
        cls: type[RasterType],
        pointcloud: gpd.GeoDataFrame | gu.PointCloud,
        grid_coords: tuple[NDArrayNum, NDArrayNum] = None,
        transform: rio.transform.Affine = None,
        shape: tuple[int, int] = None,
        nodata: int | float | None = None,
        data_column_name: str = "b1",
        area_or_point: Literal["Area", "Point"] = "Point",
    ) -> RasterType:
        """
        Create a raster from a point cloud with coordinates on a regular grid.

        To inform on what grid to create the raster, either pass a tuple of X/Y grid coordinates, or the expected
        transform and shape. All point cloud coordinates must fall exactly at one of the coordinates of this grid.

        :param pointcloud: Point cloud.
        :param grid_coords: Regular coordinate vectors for the raster, from which the geotransform and shape are
            deduced.
        :param transform: Geotransform of the raster.
        :param shape: Shape of the raster.
        :param nodata: Nodata value of the raster.
        :param data_column_name: Name to use for point cloud data column, defaults to "bX" where X is the data band
            number.
        :param area_or_point: Whether to set the pixel interpretation of the raster to "Area" or "Point".
        """

        arr, transform, crs, nodata, aop = _regular_pointcloud_to_raster(
            pointcloud=pointcloud,
            grid_coords=grid_coords,
            transform=transform,
            shape=shape,
            nodata=nodata,
            data_column_name=data_column_name,
            area_or_point=area_or_point,
        )

        return cls.from_array(data=arr, transform=transform, crs=crs, nodata=nodata, area_or_point=area_or_point)

    def polygonize(
        self,
        target_values: Number | tuple[Number, Number] | list[Number] | NDArrayNum | Literal["all"] = "all",
        data_column_name: str = "id",
    ) -> gu.Vector:
        """
        Polygonize the raster into a vector.

        :param target_values: Value or range of values of the raster from which to
          create geometries (defaults to "all", for which all unique pixel values of the raster are used).
        :param data_column_name: Data column name to be associated with target values in the output vector
            (defaults to "id").

        :returns: gu.Vector containing the polygonized geometries associated to target values.
        """

        return _polygonize(source_raster=self, target_values=target_values, data_column_name=data_column_name)

    def proximity(
        self,
        vector: gu.Vector | None = None,
        target_values: list[float] | None = None,
        geometry_type: str = "boundary",
        in_or_out: Literal["in"] | Literal["out"] | Literal["both"] = "both",
        distance_unit: Literal["pixel"] | Literal["georeferenced"] = "georeferenced",
    ) -> Raster:
        """
        Compute proximity distances to the raster target pixels, or to a vector geometry on the raster grid.

        **Match-reference**: a raster can be passed to match its resolution, bounds and CRS for computing
        proximity distances.

        When passing a vector, by default, the boundary of the geometry will be used. The full geometry can be used by
        passing "geometry", or any lower dimensional geometry attribute such as "centroid", "envelope" or "convex_hull".
        See all geometry attributes in the Shapely documentation at https://shapely.readthedocs.io/.

        :param vector: gu.Vector for which to compute the proximity to geometry,
            if not provided computed on this raster target pixels.
        :param target_values: (Only with raster) List of target values to use for the proximity,
            defaults to all non-zero values.
        :param geometry_type: (Only with a vector) Type of geometry to use for the proximity, defaults to 'boundary'.
        :param in_or_out: (Only with a vector) Compute proximity only 'in' or 'out'-side the geometry, or 'both'.
        :param distance_unit: Distance unit, either 'georeferenced' or 'pixel'.

        :return: Proximity distances raster.
        """

        proximity = _proximity_from_vector_or_raster(
            raster=self,
            vector=vector,
            target_values=target_values,
            geometry_type=geometry_type,
            in_or_out=in_or_out,
            distance_unit=distance_unit,
        )

        out_nodata = _default_nodata(proximity.dtype)
        return self.from_array(
            data=proximity,
            transform=self.transform,
            crs=self.crs,
            nodata=out_nodata,
            area_or_point=self.area_or_point,
            tags=self.tags,
        )

    @overload
    def subsample(
        self,
        subsample: int | float,
        return_indices: Literal[False] = False,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> NDArrayNum: ...

    @overload
    def subsample(
        self,
        subsample: int | float,
        return_indices: Literal[True],
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> tuple[NDArrayNum, ...]: ...

    @overload
    def subsample(
        self,
        subsample: float | int,
        return_indices: bool = False,
        random_state: int | np.random.Generator | None = None,
    ) -> NDArrayNum | tuple[NDArrayNum, ...]: ...

    @profiler.profile("geoutils.raster.raster.subsample", memprof=True)  # type: ignore
    def subsample(
        self,
        subsample: float | int,
        return_indices: bool = False,
        random_state: int | np.random.Generator | None = None,
    ) -> NDArrayNum | tuple[NDArrayNum, ...]:
        """
        Randomly sample the raster. Only valid values are considered.

        :param subsample: Subsample size. If <= 1, a fraction of the total pixels to extract.
            If > 1, the number of pixels.
        :param return_indices: Whether to return the extracted indices only.
        :param random_state: Random state or seed number.

        :return: Array of sampled valid values, or array of sampled indices.
        """

        return subsample_array(
            array=self.data, subsample=subsample, return_indices=return_indices, random_state=random_state
        )


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
