"""
geoutils.georaster provides a toolset for working with raster data.
"""
from __future__ import annotations

import os
import warnings
from collections import abc
from collections.abc import Iterable
from contextlib import ExitStack
from numbers import Number
from typing import IO, Any, Callable, TypeVar, overload

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio as rio
import rasterio.mask
import rasterio.transform
import rasterio.warp
import rasterio.windows
from affine import Affine
from matplotlib import cm, colors
from rasterio.crs import CRS
from rasterio.features import shapes
from rasterio.plot import show as rshow
from rasterio.warp import Resampling
from scipy.ndimage import map_coordinates

import geoutils.geovector as gv
from geoutils._typing import AnyNumber, ArrayLike, DTypeLike
from geoutils.geovector import Vector

# If python38 or above, Literal is builtin. Otherwise, use typing_extensions
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

try:
    import rioxarray
except ImportError:
    _has_rioxarray = False
else:
    _has_rioxarray = True

RasterType = TypeVar("RasterType", bound="Raster")

# List of numpy functions that are handled: nan statistics function, normal statistics function and sorting/counting
_HANDLED_FUNCTIONS = (
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
    + [
        "sum",
        "amax",
        "amin",
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
    ]
    + ["sort", "count_nonzero", "unique"]
)


# Function to set the default nodata values for any given dtype
# Similar to GDAL for int types, but without absurdly long nodata values for floats.
# For unsigned types, the maximum value is chosen (with a max of 99999).
# For signed types, the minimum value is chosen (with a min of -99999).
def _default_nodata(dtype: str | np.dtype | type) -> int:
    """
    Set the default nodata value for any given dtype, when this is not provided.
    """
    default_nodata_lookup = {
        "uint8": 255,
        "int8": -128,
        "uint16": 65535,
        "int16": -32768,
        "uint32": 99999,
        "int32": -99999,
        "float16": -99999,
        "float32": -99999,
        "float64": -99999,
        "float128": -99999,
        "longdouble": -99999,  # This is float64 on Windows, float128 on other systems, for compatibility
    }
    # Check argument dtype is as expected
    if not isinstance(dtype, (str, np.dtype, type)):
        raise ValueError(f"dtype {dtype} not understood")

    # Convert numpy types to string
    if isinstance(dtype, type):
        dtype = np.dtype(dtype).name

    # Convert np.dtype to string
    if isinstance(dtype, np.dtype):
        dtype = dtype.name

    if dtype in default_nodata_lookup.keys():
        return default_nodata_lookup[dtype]
    else:
        raise NotImplementedError(f"No default nodata value set for dtype {dtype}")


# Set default attributes to be kept from rasterio's DatasetReader
_default_rio_attrs = [
    "bounds",
    "count",
    "crs",
    "driver",
    "dtypes",
    "height",
    "indexes",
    "name",
    "nodata",
    "res",
    "shape",
    "transform",
    "width",
]


def _load_rio(
    dataset: rio.io.DatasetReader,
    bands: int | list[int] | None = None,
    masked: bool = False,
    transform: Affine | None = None,
    shape: tuple[int, int] | None = None,
    **kwargs: Any,
) -> np.ma.masked_array:
    r"""
    Load specific bands of the dataset, using rasterio.read().

    Ensure that self.data.ndim = 3 for ease of use (needed e.g. in show)

    :param dataset: The dataset to read (opened with "rio.open(filename)")
    :param bands: The band(s) to load. Note that rasterio begins counting at 1, not 0.
    :param masked: Should the mask be read (if any exists), and/or should the nodata be used to mask values
    :param transform: Create a window from the given transform (to read only parts of the raster)
    :param shape: The expected shape of the read ndarray. Must be given together with the 'transform' argument.

    :raises ValueError: If only one of 'transform' and 'shape' are given.

    :returns: A numpy array if masked == False or a masked_array

    \*\*kwargs: any additional arguments to rasterio.io.DatasetReader.read.
    Useful ones are:
    .. hlist::
    * out_shape : to load a subsampled version
    * window : to load a cropped version
    * resampling : to set the resampling algorithm
    """
    if transform is not None and shape is not None:
        if transform == dataset.transform:
            row_off, col_off = 0, 0
        else:
            row_off, col_off = (round(val) for val in dataset.index(transform[2], abs(transform[4])))

        window = rio.windows.Window(col_off, row_off, *shape[::-1])
    elif sum(param is None for param in [shape, transform]) == 1:
        raise ValueError("If 'shape' or 'transform' is provided, BOTH must be given.")
    else:
        window = None

    if bands is None:
        data = dataset.read(masked=masked, window=window, **kwargs)
    else:
        data = dataset.read(bands, masked=masked, window=window, **kwargs)
    if len(data.shape) == 2:
        data = data[np.newaxis, :, :]
    return np.ma.masked_array(data)


class Raster:
    """
    Create a Raster object from a rasterio-supported raster dataset.

    If not otherwise specified below, attribute types and contents correspond
    to the attributes defined by rasterio.

    Attributes:
        filename_or_dataset : str
            The path/filename of the loaded, file, only set if a disk-based file is read in.
        data : np.array
            Loaded image. Dimensions correspond to (bands, height, width).
        nbands : int
            Number of bands loaded into .data
        bands : tuple
            The indexes of the opened dataset which correspond to the bands loaded into data.
        is_loaded : bool
            True if the image data have been loaded into this Raster.

        bounds

        count

        crs

        driver

        dtypes

        height

        indexes

        name

        nodata

        res

        shape

        transform

        width
    """

    def __init__(
        self,
        filename_or_dataset: str | RasterType | rio.io.DatasetReader | rio.io.MemoryFile | dict[str, Any],
        bands: None | int | list[int] = None,
        load_data: bool = True,
        downsample: AnyNumber = 1,
        masked: bool = True,
        nodata: int | float | list[int] | list[float] | None = None,
        attrs: list[str] | None = None,
    ) -> None:
        """
        Load a rasterio-supported dataset, given a filename.

        :param filename_or_dataset: The filename of the dataset.

        :param bands: The band(s) to load into the object. Default is to load all bands.

        :param load_data: Load the raster data into the object. Default is True.

        :param downsample: Reduce the size of the image loaded by this factor. Default is 1

        :param masked: the data is loaded as a masked array, with no data values masked. Default is True.

        :param nodata: nodata to be used (overwrites the metadata). Default is None, i.e. reads from metadata.

        :param attrs: Additional attributes from rasterio's DataReader class to add to the Raster object.
            Default list is set by geoutils.georaster.raster._default_rio_attrs, i.e.
            ['bounds', 'count', 'crs', 'driver', 'dtypes', 'height', 'indexes',
            'name', 'nodata', 'res', 'shape', 'transform', 'width'] - if no attrs are specified, these will be added.

        :return: A Raster object
        """
        self.driver: str | None = None
        self.name: str | None = None
        self.filename: str | None = None
        self.tags: dict[str, Any] = {}

        self._data: np.ma.masked_array | None = None
        self._nodata: int | float | list[int] | list[float] | None = nodata
        self._bands = bands
        self._masked = masked
        self._disk_hash: int | None = None
        self._is_modified = True
        self._disk_shape: tuple[int, int, int] | None = None
        self._disk_indexes: tuple[int] | None = None
        self._disk_dtypes: tuple[str] | None = None

        # This is for Raster.from_array to work.
        if isinstance(filename_or_dataset, dict):
            # Important to pass the nodata before the data setter, which uses it in turn
            self._nodata = filename_or_dataset["nodata"]
            self.data = filename_or_dataset["data"]
            self.transform: rio.transform.Affine = filename_or_dataset["transform"]
            self.crs: rio.crs.CRS = filename_or_dataset["crs"]
            for key in filename_or_dataset:
                if key in ["data", "transform", "crs", "nodata"]:
                    continue
                setattr(self, key, filename_or_dataset[key])
            return

        # If Raster is passed, simply point back to Raster
        if isinstance(filename_or_dataset, Raster):
            for key in filename_or_dataset.__dict__:
                setattr(self, key, filename_or_dataset.__dict__[key])
            return
        # Image is a file on disk.
        elif isinstance(filename_or_dataset, (str, rio.io.DatasetReader, rio.io.MemoryFile)):

            # ExitStack is used instead of "with rio.open(filename_or_dataset) as ds:".
            # This is because we might not actually want to open it like that, so this is equivalent
            # to the pseudocode:
            # "with rio.open(filename_or_dataset) as ds if isinstance(filename_or_dataset, str) else ds:"
            # This is slightly black magic, but it works!
            with ExitStack():
                if isinstance(filename_or_dataset, str):
                    ds: rio.io.DatasetReader = rio.open(filename_or_dataset)
                    self.filename = filename_or_dataset
                elif isinstance(filename_or_dataset, rio.io.DatasetReader):
                    ds = filename_or_dataset
                    self.filename = filename_or_dataset.files[0]
                else:  # This is if it's a MemoryFile
                    ds = filename_or_dataset.open()
                    self.filename = None

                self.transform = ds.transform
                self.crs = ds.crs
                self._nodata = ds.nodata
                self.name = ds.name
                self.driver = ds.driver
                self.tags.update(ds.tags())

                self._disk_shape = (ds.count, ds.height, ds.width)
                self._disk_indexes = ds.indexes
                self._disk_dtypes = ds.dtypes

                if attrs is not None:
                    for attr in attrs:
                        self.__setattr__(attr, ds.__getattr__(attr))

            # Check number of bands to be loaded
            if bands is None:
                nbands = self.nbands
            elif isinstance(bands, int):
                nbands = 1
            elif isinstance(bands, abc.Iterable):
                nbands = len(bands)

            # Downsampled image size
            if not isinstance(downsample, (int, float)):
                raise ValueError("downsample must be of type int or float")
            if downsample == 1:
                out_shape = (nbands, self.height, self.width)
            else:
                down_width = int(np.ceil(self.width / downsample))
                down_height = int(np.ceil(self.height / downsample))
                out_shape = (nbands, down_height, down_width)
                res = tuple(np.asarray(self.res) * downsample)
                self.transform = rio.transform.from_origin(self.bounds.left, self.bounds.top, res[0], res[1])

            if load_data:
                # Mypy doesn't like the out_shape for some reason. I can't figure out why! (erikmannerfelt, 14/01/2022)
                self._data = _load_rio(ds, bands=bands, masked=masked, out_shape=out_shape)  # type: ignore
                if isinstance(filename_or_dataset, str):
                    self._is_modified = False
                    self._disk_hash = hash((self.data.tobytes(), self.transform, self.crs, self.nodata))

            # Set nodata
            if nodata is not None:
                self.set_nodata(nodata)

        # Provide a catch in case trying to load from data array
        elif isinstance(filename_or_dataset, np.ndarray):
            raise ValueError("np.array provided as filename. Did you mean to call Raster.from_array(...) instead? ")

        # Don't recognise the input, so stop here.
        else:
            raise ValueError("filename argument not recognised.")

    @property
    def nbands(self) -> int:
        if not self.is_loaded and self._disk_shape is not None:
            return self._disk_shape[0]
        return int(self.data.shape[0])

    @property
    def count(self) -> int:
        if self._disk_shape is not None:
            return self._disk_shape[0]
        return self.nbands

    @property
    def height(self) -> int:
        """Return the height of the Raster in pixels."""
        if not self.is_loaded:
            return self._disk_shape[1]  # type: ignore
        return int(self.data.shape[1])

    @property
    def width(self) -> int:
        """Return the width of the Raster in pixels."""
        if not self.is_loaded:
            return self._disk_shape[2]  # type: ignore
        return int(self.data.shape[2])

    @property
    def shape(self) -> tuple[int, int]:
        """Return a (height, width) tuple of the data shape in pixels."""
        if not self.is_loaded:
            return self._disk_shape[1], self._disk_shape[2]  # type: ignore
        return int(self.data.shape[1]), int(self.data.shape[2])

    @property
    def res(self) -> tuple[float | int, float | int]:
        """Return the X/Y resolution in georeferenced units of the Raster."""
        return self.transform[0], abs(self.transform[4])

    @property
    def bounds(self) -> rio.coords.BoundingBox:
        """Return the bounding coordinates of the Raster."""
        return rio.coords.BoundingBox(*rio.transform.array_bounds(self.height, self.width, self.transform))

    @property
    def is_loaded(self) -> bool:
        """Return False if the data attribute is None, and True if data exists."""
        return self._data is not None

    @property
    def dtypes(self) -> tuple[str, ...]:
        """Return the string representations of the data types for each band."""
        if not self.is_loaded and self._disk_dtypes is not None:
            return self._disk_dtypes
        return (str(self.data.dtype),) * self.nbands

    @property
    def indexes(self) -> tuple[int, ...]:
        if self._disk_indexes is not None:
            return self._disk_indexes
        return tuple(range(1, self.nbands + 1))

    @property
    def bands(self) -> tuple[int, ...]:
        if self._bands is not None:
            if isinstance(self._bands, int):
                return (self._bands,)
            return tuple(self._bands)
        return self.indexes

    def load(self, **kwargs: Any) -> None:
        """
        Load the data from disk.

        :param kwargs: Optional keyword arguments sent to '_load_rio()'

        :raises ValueError: If the data are already loaded.
        :raises AttributeError: If no 'filename' attribute exists.
        """
        if self.is_loaded:
            raise ValueError("Data are already loaded")

        if self.filename is None:
            raise AttributeError("'filename' is not set")

        with rio.open(self.filename) as dataset:
            self.data = _load_rio(
                dataset, bands=self._bands, masked=self._masked, transform=self.transform, shape=self.shape, **kwargs
            )

    @classmethod
    def from_array(
        cls: type[RasterType],
        data: np.ndarray | np.ma.masked_array,
        transform: tuple[float, ...] | Affine,
        crs: CRS | int | None,
        nodata: int | float | list[int] | list[float] | None = None,
    ) -> RasterType:
        """Create a Raster from a numpy array and some geo-referencing information.

        :param data: data array

        :param transform: the 2-D affine transform for the image mapping.
            Either a tuple(x_res, 0.0, top_left_x, 0.0, y_res, top_left_y) or
            an affine.Affine object.

        :param crs: Coordinate Reference System for image. Either a rasterio CRS,
            or the EPSG integer.

        :param nodata: nodata value


        :returns: A Raster object containing the provided data.


        Example:

            You have a data array in EPSG:32645. It has a spatial resolution of
            30 m in x and y, and its top left corner is X=478000, Y=3108140.

            >>> data = np.ones((500, 500), dtype="uint8")
            >>> transform = (30.0, 0.0, 478000.0, 0.0, -30.0, 3108140.0)
            >>> myim = Raster.from_array(data, transform, 32645)
        """
        if not isinstance(transform, Affine):
            if isinstance(transform, tuple):
                transform = Affine(*transform)
            else:
                raise ValueError("transform argument needs to be Affine or tuple.")

        # Enable shortcut to create CRS from an EPSG ID.
        if isinstance(crs, int):
            crs = CRS.from_epsg(crs)

        return cls({"data": data, "transform": transform, "crs": crs, "nodata": nodata})

    def __repr__(self) -> str:
        """Convert object to formal string representation."""
        return self.__str__()
        # L = [getattr(self, item) for item in self._saved_attrs]
        # s: str = "{}.{}({})".format(type(self).__module__, type(self).__qualname__, ", ".join(map(str, L)))

        # return s

    def __str__(self) -> str:
        """Provide string of information about Raster."""
        return self.info()

    def __eq__(self, other: object) -> bool:
        """Check if a Raster masked array's data (including masked values), mask, fill_value and dtype are equal,
        as well as the Raster's nodata, and georeferencing."""

        if not isinstance(other, type(self)):  # TODO: Possibly add equals to SatelliteImage?
            return NotImplemented
        return all(
            [
                np.array_equal(self.data.data, other.data.data, equal_nan=True),
                np.array_equal(self.data.mask, other.data.mask),
                self.data.fill_value == other.data.fill_value,
                self.data.dtype == other.data.dtype,
                self.transform == other.transform,
                self.crs == other.crs,
                self.nodata == other.nodata,
            ]
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def _overloading_check(
        self: RasterType, other: RasterType | np.ndarray | Number
    ) -> tuple[np.ma.masked_array, np.ma.masked_array | Number, float | int | list[int] | list[float] | None]:
        """
        Before any operation overloading, check input data type and return both self and other data as either \
a np.ndarray or number, converted to the minimum compatible dtype between both datasets.
        Also returns the best compatible nodata value.

        The nodata value is set in the following order:
        - to nodata of self, if output dtype is same as self's dtype
        - to nodata of other, if other is of Raster type and output dtype is same as other's dtype
        - otherwise falls to default nodata value for the output dtype (only if masked values -> done externally)

        Inputs:
        :param other: The other data set to be used in the operation.

        :returns: a tuple containing, self.data converted to the compatible dtype, other data converted to \
np.ndarray or number and correct dtype, the compatible nodata value.
        """
        # Check that other is of correct type
        # If not, a NotImplementedError should be raised, in case other's class has a method implemented.
        # See https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
        if not isinstance(other, (Raster, np.ndarray, Number)):
            raise NotImplementedError(
                f"Operation between an object of type {type(other)} and a Raster impossible. Must be a Raster, "
                f"np.ndarray or single number."
            )

        # Get self's dtype and nodata
        nodata1 = self.nodata
        dtype1 = self.data.dtype

        # Case 1 - other is a Raster
        if isinstance(other, Raster):
            # Check that both data are loaded
            if not (self.is_loaded & other.is_loaded):
                raise ValueError("Raster's data must be loaded with self.load().")

            # Check that both rasters have the same shape and georeferences
            if (self.data.shape == other.data.shape) & (self.transform == other.transform) & (self.crs == other.crs):
                pass
            else:
                raise ValueError("Both rasters must have the same shape, transform and CRS.")

            other_data = other.data
            nodata2 = other.nodata
            dtype2 = other_data.dtype

        # Case 2 - other is a numpy array
        elif isinstance(other, np.ndarray):
            # Check that both array have the same shape

            if len(other.shape) == 2:
                other_data = other[np.newaxis, :, :]
            else:
                other_data = other

            if self.data.shape == other_data.shape:
                pass
            else:
                raise ValueError("Both rasters must have the same shape.")

            nodata2 = None
            dtype2 = other_data.dtype

        # Case 3 - other is a single number
        else:
            other_data = other
            nodata2 = None
            dtype2 = rio.dtypes.get_minimum_dtype(other_data)

        # Figure out output dtype
        out_dtype = np.find_common_type([dtype1, dtype2], [])

        # Figure output nodata
        out_nodata = None
        if (nodata2 is not None) and (out_dtype == dtype2):
            out_nodata = nodata2
        if (nodata1 is not None) and (out_dtype == dtype1):
            out_nodata = nodata1

        self_data = self.data

        return self_data, other_data, out_nodata

    def __add__(self: RasterType, other: RasterType | np.ndarray | Number) -> RasterType:
        """
        Sum up the data of two rasters or a raster and a numpy array, or a raster and single number.
        If other is a Raster, it must have the same data.shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        # Check inputs and return compatible data, output dtype and nodata value
        self_data, other_data, nodata = self._overloading_check(other)

        # Run calculation
        out_data = self_data + other_data

        # Save to output Raster
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata)

        return out_rst

    def __radd__(self: RasterType, other: np.ndarray | Number) -> RasterType:
        """
        Addition overloading when other is first item in the operation (e.g. 1 + rst).
        """
        return self.__add__(other)

    def __neg__(self: RasterType) -> RasterType:
        """Return self with self.data set to -self.data"""
        return self.from_array(-self.data, self.transform, self.crs, nodata=self.nodata)

    def __sub__(self, other: Raster | np.ndarray | Number) -> Raster:
        """
        Subtract two rasters. Both rasters must have the same data.shape, transform and crs.
        """
        self_data, other_data, nodata = self._overloading_check(other)
        out_data = self_data - other_data
        return self.from_array(out_data, self.transform, self.crs, nodata=nodata)

    def __rsub__(self: RasterType, other: np.ndarray | Number) -> RasterType:
        """
        Subtraction overloading when other is first item in the operation (e.g. 1 - rst).
        """
        self_data, other_data, nodata = self._overloading_check(other)
        out_data = other_data - self_data
        return self.from_array(out_data, self.transform, self.crs, nodata=nodata)

    def __mul__(self: RasterType, other: RasterType | np.ndarray | Number) -> RasterType:
        """
        Multiply the data of two rasters or a raster and a numpy array, or a raster and single number.
        If other is a Raster, it must have the same data.shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, nodata = self._overloading_check(other)
        out_data = self_data * other_data
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata)
        return out_rst

    def __rmul__(self: RasterType, other: np.ndarray | Number) -> RasterType:
        """
        Multiplication overloading when other is first item in the operation (e.g. 2 * rst).
        """
        return self.__mul__(other)

    def __truediv__(self: RasterType, other: RasterType | np.ndarray | Number) -> RasterType:
        """
        True division of the data of two rasters or a raster and a numpy array, or a raster and single number.
        If other is a Raster, it must have the same data.shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, nodata = self._overloading_check(other)
        out_data = self_data / other_data
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata)
        return out_rst

    def __rtruediv__(self: RasterType, other: np.ndarray | Number) -> RasterType:
        """
        True division overloading when other is first item in the operation (e.g. 1/rst).
        """
        self_data, other_data, nodata = self._overloading_check(other)
        out_data = other_data / self_data
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata)
        return out_rst

    def __floordiv__(self: RasterType, other: RasterType | np.ndarray | Number) -> RasterType:
        """
        Floor division of the data of two rasters or a raster and a numpy array, or a raster and single number.
        If other is a Raster, it must have the same data.shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, nodata = self._overloading_check(other)
        out_data = self_data // other_data
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata)
        return out_rst

    def __rfloordiv__(self: RasterType, other: np.ndarray | Number) -> RasterType:
        """
        Floor division overloading when other is first item in the operation (e.g. 1/rst).
        """
        self_data, other_data, nodata = self._overloading_check(other)
        out_data = other_data // self_data
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata)
        return out_rst

    def __mod__(self: RasterType, other: RasterType | np.ndarray | Number) -> RasterType:
        """
        Modulo of the data of two rasters or a raster and a numpy array, or a raster and single number.
        If other is a Raster, it must have the same data.shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, nodata = self._overloading_check(other)
        out_data = self_data % other_data
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata)
        return out_rst

    def __pow__(self: RasterType, power: int | float) -> RasterType:
        """
        Calculate the power of self.data and returns a Raster.
        """
        # Check that input is a number
        if not isinstance(power, Number):
            raise ValueError("Power needs to be a number.")

        # Calculate the product of arrays and save to new Raster
        out_data = self.data**power
        nodata = self.nodata
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata)
        return out_rst

    @overload
    def astype(self, dtype: DTypeLike, inplace: Literal[False] = False) -> Raster:
        ...

    @overload
    def astype(self, dtype: DTypeLike, inplace: Literal[True]) -> None:
        ...

    def astype(self, dtype: DTypeLike, inplace: bool = False) -> Raster | None:
        """
        Converts the data type of a Raster object.

        :param dtype: Any numpy dtype or string accepted by numpy.astype
        :param inplace: Set to True to modify the raster in place.

        :returns: the output Raster with dtype changed.
        """
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
            self._data = out_data
            return None
        else:
            return self.from_array(out_data, self.transform, self.crs, nodata=self.nodata)

    @property
    def is_modified(self) -> bool:
        """Check whether file has been modified since it was created/opened.

        :returns: True if Raster has been modified.

        """
        if not self._is_modified:
            new_hash = hash(
                (self._data.tobytes() if self._data is not None else 0, self.transform, self.crs, self.nodata)
            )
            self._is_modified = not (self._disk_hash == new_hash)

        return self._is_modified

    @property
    def nodata(self) -> int | float | list[int] | list[float] | None:
        """
        Get nodata value.

        :returns: Nodata value
        """
        return self._nodata

    @nodata.setter
    def nodata(self, new_nodata: int | float | list[int] | list[float] | None) -> None:
        """
        Set .nodata and update .data by calling set_nodata() with default parameters.

        By default, the old nodata values are updated into the new nodata in the data array .data.data, and the
        mask .data.mask is updated to mask all new nodata values (i.e., the mask from old nodata stays and is extended
        to potential new values of new nodata found in the array).

        To set nodata for more complex cases (e.g., redefining a wrong nodata that has a valid value in the array),
        call the function set_nodata() directly to set the arguments update_array and update_mask adequately.

        :param new_nodata: New nodata to assign to this instance of Raster
        """

        self.set_nodata(nodata=new_nodata)

    def set_nodata(
        self, nodata: int | float | list[int] | list[float] | None, update_array: bool = True, update_mask: bool = True
    ) -> None:
        """
        Set a new nodata value for each band. This updates the old nodata into a new nodata value in the metadata,
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

        If None is passed as nodata, only the metadata is updated and the mask of oldnodata unset.

        :param nodata: Nodata values
        :param update_array: Update the old nodata values into new nodata values in the data array
        :param update_mask: Update the old mask by unmasking old nodata and masking new nodata (if array is updated,
            old nodata are changed to new nodata and thus stay masked)
        """
        if nodata is not None and not isinstance(nodata, (list, int, float, np.integer, np.floating)):
            raise ValueError("Type of nodata not understood, must be list or float or int")

        elif (isinstance(nodata, (int, float, np.integer, np.floating))) and self.count > 1:
            nodata = [nodata] * self.count

        elif isinstance(nodata, list) and self.count == 1:
            nodata = list(nodata)[0]

        elif nodata is None:
            nodata = None

        # Check that nodata has same length as number of bands in self
        if isinstance(nodata, list):
            if len(nodata) != self.count:
                raise ValueError(f"Length of nodata ({len(nodata)}) incompatible with number of bands ({self.count})")
            # Check that nodata value is compatible with dtype
            for k in range(len(nodata)):
                if not rio.dtypes.can_cast_dtype(nodata[k], self.dtypes[k]):
                    raise ValueError(f"nodata value {nodata[k]} incompatible with self.dtype {self.dtypes[k]}")
        elif isinstance(nodata, (int, float, np.integer, np.floating)):
            if not rio.dtypes.can_cast_dtype(nodata, self.dtypes[0]):
                raise ValueError(f"nodata value {nodata} incompatible with self.dtype {self.dtypes[0]}")

        # If we update mask or array, get the masked array
        if update_array or update_mask:

            # Extract the data variable, so the self.data property doesn't have to be called a bunch of times
            imgdata = self.data

            # Loop through the bands
            for i, new_nodata in enumerate(nodata if isinstance(nodata, Iterable) else [nodata]):

                # Get the index of old nodatas
                index_old_nodatas = imgdata.data[i, :, :] == self.nodata

                # Get the index of new nodatas, if it is defined
                index_new_nodatas = imgdata.data[i, :, :] == new_nodata

                if np.count_nonzero(index_new_nodatas) > 0:
                    if update_array and update_mask:
                        warnings.warn(
                            message="New nodata value found in the data array. Those will be masked, and the old "
                            "nodata cells will now take the same value. Use set_nodata() with update_array=False "
                            "and/or update_mask=False to change this behaviour.",
                            category=UserWarning,
                        )
                    elif update_array:
                        warnings.warn(
                            "New nodata value found in the data array. The old nodata cells will now take the same "
                            "value. Use set_nodata() with update_array=False to change this behaviour.",
                            category=UserWarning,
                        )
                    elif update_mask:
                        warnings.warn(
                            "New nodata value found in the data array. Those will be masked. Use set_nodata() "
                            "with update_mask=False to change this behaviour.",
                            category=UserWarning,
                        )

                if update_array:
                    # Only update array with new nodata if it is defined
                    if nodata is not None:
                        # Replace the nodata value in the Raster
                        imgdata.data[i, index_old_nodatas] = new_nodata

                if update_mask:
                    # If a mask already exists, unmask the old nodata values before masking the new ones
                    # Can be skipped if array is updated (nodata is transferred from old to new, this part of the mask
                    # stays the same)
                    if np.ma.is_masked(imgdata) and (not update_array or nodata is None):
                        # No way to unmask a value from the masked array, so we modify the mask directly
                        imgdata.mask[i, index_old_nodatas] = False

                    # Masking like this works from the masked array directly, whether a mask exists or not
                    imgdata[i, index_new_nodatas] = np.ma.masked

            # Update the data
            self._data = imgdata

        # Update the nodata value
        self._nodata = nodata

    @property
    def data(self) -> np.ma.masked_array:
        """
        Get data.

        :returns: data array.

        """
        if not self.is_loaded and self._data is None:
            raise ValueError("Data are not loaded")
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray | np.ma.masked_array) -> None:
        """
        Set the contents of .data and possibly update .nodata.

        The data setter behaviour is the following:

        1. Writes the data in a masked array, whether the input is a classic array or a masked_array,
        2. Reshapes the data in a 3D array if it is 2D that can be broadcasted, raises an error otherwise,
        3. Raises an error if the dtype is different from that of the Raster, and points towards .copy() or .astype(),
        4. Sets a new nodata value to the Raster if none is set and if the provided array contains non-finite values
            that are unmasked (including if there is no mask at all, e.g. NaNs in a classic array),
        5. Masks non-finite values that are unmasked, whether the input is a classic array or a masked_array. Note that
            these values are not overwritten and can still be accessed in .data.data.

        :param new_data: New data to assign to this instance of Raster

        """
        # Check that new_data is a Numpy array
        if not isinstance(new_data, np.ndarray):
            raise ValueError("New data must be a numpy array.")

        if len(new_data.shape) == 2:
            new_data = new_data[np.newaxis, :, :]

        # Check that new_data has correct shape
        if self._data is not None:
            dtype = str(self._data.dtype)
            orig_shape = self._data.shape[1:]
        elif self.filename is not None:
            dtype = self.dtypes[0]
            orig_shape = self.shape
        else:
            dtype = str(new_data.dtype)
            orig_shape = new_data.shape[1:]

        # Check that new_data has the right type
        if str(new_data.dtype) != dtype:
            raise ValueError(
                "New data must be of the same type as existing data: {}. Use copy() to set a new array with "
                "different dtype, or astype() to change type.".format(dtype)
            )

        if new_data.shape[1:] != orig_shape:
            raise ValueError(
                f"New data must be of the same shape as existing data: {orig_shape}. Given: {new_data.shape[1:]}."
            )

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

    def set_mask(self, mask: np.ndarray) -> None:
        """
        Mask all pixels of self.data where `mask` is set to True or > 0.

        Masking is performed in place.
        `mask` must have the same shape as loaded data, unless the first dimension is 1, then it is ignored.

        :param mask: The data mask
        """
        # Check that mask is a Numpy array
        if not isinstance(mask, np.ndarray):
            raise ValueError("mask must be a numpy array.")

        # Check that new_data has correct shape
        if self.is_loaded:
            orig_shape = self.data.shape
        else:
            raise AttributeError("self.data must be loaded first, with e.g. self.load()")

        if mask.shape != orig_shape:
            # In case first dimension is empty and other dimensions match -> reshape mask
            if (orig_shape[0] == 1) & (orig_shape[1:] == mask.shape):
                mask = mask.reshape(orig_shape)
            else:
                raise ValueError(f"mask must be of the same shape as existing data: {orig_shape}.")

        self.data[mask > 0] = np.ma.masked

    def info(self, stats: bool = False) -> str:
        """
        Returns string of information about the raster (filename, coordinate system, number of columns/rows, etc.).

        :param stats: Add statistics for each band of the dataset (max, min, median, mean, std. dev.). Default is to
            not calculate statistics.


        :returns: text information about Raster attributes.

        """
        as_str = [
            f"Driver:               {self.driver} \n",
            f"Opened from file:     {self.filename} \n",
            f"Filename:             {self.name} \n",
            f"Raster modified since disk load?  {self._is_modified} \n",
            f"Size:                 {self.width}, {self.height}\n",
            f"Number of bands:      {self.count:d}\n",
            f"Data types:           {self.dtypes}\n",
            f"Coordinate System:    {[self.crs.to_string() if self.crs is not None else None]}\n",
            f"NoData Value:         {self.nodata}\n",
            "Pixel Size:           {}, {}\n".format(*self.res),
            "Upper Left Corner:    {}, {}\n".format(*self.bounds[:2]),
            "Lower Right Corner:   {}, {}\n".format(*self.bounds[2:]),
        ]

        if stats:
            if self.is_loaded:
                if self.nbands == 1:
                    as_str.append(f"[MAXIMUM]:          {np.nanmax(self.data):.2f}\n")
                    as_str.append(f"[MINIMUM]:          {np.nanmin(self.data):.2f}\n")
                    as_str.append(f"[MEDIAN]:           {np.ma.median(self.data):.2f}\n")
                    as_str.append(f"[MEAN]:             {np.nanmean(self.data):.2f}\n")
                    as_str.append(f"[STD DEV]:          {np.nanstd(self.data):.2f}\n")
                else:
                    for b in range(self.nbands):
                        # try to keep with rasterio convention.
                        as_str.append(f"Band {b + 1}:")
                        as_str.append(f"[MAXIMUM]:          {np.nanmax(self.data[b, :, :]):.2f}\n")
                        as_str.append(f"[MINIMUM]:          {np.nanmin(self.data[b, :, :]):.2f}\n")
                        as_str.append(f"[MEDIAN]:           {np.ma.median(self.data[b, :, :]):.2f}\n")
                        as_str.append(f"[MEAN]:             {np.nanmean(self.data[b, :, :]):.2f}\n")
                        as_str.append(f"[STD DEV]:          {np.nanstd(self.data[b, :, :]):.2f}\n")

        return "".join(as_str)

    def copy(self: RasterType, new_array: np.ndarray | None = None) -> RasterType:
        """
        Copy the Raster object in memory

        :param new_array: New array to use for the copied Raster

        :return:
        """
        if new_array is not None:
            data = new_array
        else:
            data = self.data.copy()

        cp = self.from_array(data=data, transform=self.transform, crs=self.crs, nodata=self.nodata)

        return cp

    @overload
    def get_nanarray(self, return_mask: Literal[False] = False) -> np.ndarray:
        ...

    @overload
    def get_nanarray(self, return_mask: Literal[True]) -> tuple[np.ndarray, np.ndarray]:
        ...

    def get_nanarray(self, return_mask: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Method to return the squeezed masked array filled with NaNs and associated squeezed mask, both as a copy.

        :param return_mask: Whether to return the mask of valid data

        :returns Array with masked data as NaNs, (Optional) Mask of valid data
        """

        # Get the array with masked value fill with NaNs
        nanarray = self.data.filled(fill_value=np.nan).squeeze()

        # The function np.ma.filled() only returns a copy if the array is masked, copy the array if it's not the case
        if not np.ma.is_masked(self.data):
            nanarray = np.copy(nanarray)

        # Return the NaN array, and possibly the mask as well
        if return_mask:
            return nanarray, np.copy(np.ma.getmaskarray(self.data).squeeze())
        else:
            return nanarray

    def __array__(self) -> np.ndarray:
        """Method to cast np.array() or np.asarray() function directly on Raster classes."""

        return self._data

    def __array_ufunc__(
        self,
        ufunc: Callable[[np.ndarray | tuple[np.ndarray, np.ndarray]], np.ndarray | tuple[np.ndarray, np.ndarray]],
        method: str,
        *inputs: Raster | tuple[Raster, Raster] | tuple[np.ndarray, Raster] | tuple[Raster, np.ndarray],
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
                return self.from_array(
                    data=final_ufunc(inputs[0].data, **kwargs),  # type: ignore
                    transform=self.transform,
                    crs=self.crs,
                    nodata=self.nodata,
                )  # type: ignore

            # If the universal function has two outputs (Note: no ufunc exists that has three outputs or more)
            else:
                output = final_ufunc(inputs[0].data, **kwargs)  # type: ignore
                return self.from_array(
                    data=output[0], transform=self.transform, crs=self.crs, nodata=self.nodata
                ), self.from_array(data=output[1], transform=self.transform, crs=self.crs, nodata=self.nodata)

        # If the universal function takes two inputs (Note: no ufunc exists that has three inputs or more)
        else:
            if ufunc.nout == 1:
                return self.from_array(
                    data=final_ufunc(inputs[0].data, inputs[1].data, **kwargs),  # type: ignore
                    transform=self.transform,
                    crs=self.crs,
                    nodata=self.nodata,
                )

            # If the universal function has two outputs (Note: no ufunc exists that has three outputs or more)
            else:
                output = final_ufunc(inputs[0].data, inputs[1].data, **kwargs)  # type: ignore
                return self.from_array(
                    data=output[0], transform=self.transform, crs=self.crs, nodata=self.nodata
                ), self.from_array(data=output[1], transform=self.transform, crs=self.crs, nodata=self.nodata)

    def __array_function__(
        self, func: Callable[[np.ndarray, Any], Any], types: tuple[type], args: Any, kwargs: Any
    ) -> Any:
        """
        Method to cast NumPy array function directly on a Raster object by applying it to the masked array.
        A limited number of function is supported, listed in raster._HANDLED_FUNCTIONS.
        """

        # If function is not implemented
        if func.__name__ not in _HANDLED_FUNCTIONS:
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

        # Otherwise, we run the numpy function normally (most take masks into account)
        else:
            first_arg = args[0].data

        return func(first_arg, *args[1:], **kwargs)  # type: ignore

    # Note the star is needed because of the default argument 'mode' preceding non default arg 'inplace'
    # Then the final overload must be duplicated
    @overload
    def crop(
        self: RasterType,
        cropGeom: RasterType | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        *,
        inplace: Literal[True],
    ) -> None:
        ...

    @overload
    def crop(
        self: RasterType,
        cropGeom: RasterType | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        *,
        inplace: Literal[False],
    ) -> RasterType:
        ...

    @overload
    def crop(
        self: RasterType,
        cropGeom: RasterType | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        inplace: bool = True,
    ) -> RasterType | None:
        ...

    def crop(
        self: RasterType,
        cropGeom: RasterType | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        inplace: bool = True,
    ) -> RasterType | None:
        """
        Crop the Raster to a given extent.

        :param cropGeom: Geometry to crop raster to, as either a Raster object, a Vector object, or a list of
            coordinates. If cropGeom is a Raster, crop() will crop to the boundary of the raster as returned by
            Raster.ds.bounds. If cropGeom is a Vector, crop() will crop to the bounding geometry. If cropGeom is a
            list of coordinates, the order is assumed to be [xmin, ymin, xmax, ymax].
        :param mode: one of 'match_pixel' (default) or 'match_extent'. 'match_pixel' will preserve the original pixel
            resolution, cropping to the extent that most closely aligns with the current coordinates. 'match_extent'
            will match the extent exactly, adjusting the pixel resolution to fit the extent.
        :param inplace: Update the raster inplace or return copy.

        :returns: None if inplace=True and a new Raster if inplace=False
        """
        assert mode in [
            "match_extent",
            "match_pixel",
        ], "mode must be one of 'match_pixel', 'match_extent'"
        if isinstance(cropGeom, (Raster, Vector)):
            xmin, ymin, xmax, ymax = cropGeom.bounds
        elif isinstance(cropGeom, (list, tuple)):
            xmin, ymin, xmax, ymax = cropGeom
        else:
            raise ValueError("cropGeom must be a Raster, Vector, or list of coordinates.")

        if mode == "match_pixel":
            ref_win = rio.windows.from_bounds(xmin, ymin, xmax, ymax, transform=self.transform)
            self_win = rio.windows.from_bounds(*self.bounds, transform=self.transform)
            final_window = ref_win.intersection(self_win).round_lengths().round_offsets()
            new_xmin, new_ymin, new_xmax, new_ymax = rio.windows.bounds(final_window, transform=self.transform)
            tfm = rio.transform.from_origin(new_xmin, new_ymax, *self.res)

            if self.is_loaded:
                (rowmin, rowmax), (colmin, colmax) = final_window.toranges()
                crop_img = self.data[:, rowmin:rowmax, colmin:colmax]
            else:
                with rio.open(self.filename) as raster:
                    crop_img = raster.read(
                        self._bands,
                        masked=self._masked,
                        window=final_window,
                    )

        else:
            bbox = rio.coords.BoundingBox(left=xmin, bottom=ymin, right=xmax, top=ymax)
            out_rst = self.reproject(dst_bounds=bbox)  # should we instead raise an issue and point to reproject?
            crop_img = out_rst.data
            tfm = out_rst.transform

        if inplace:
            self._data = crop_img
            self.transform = tfm
            self.tags["AREA_OR_POINT"] = "Area"  # TODO: Explain why this should have an area interpretation now
            return None
        else:
            newraster = self.from_array(crop_img, tfm, self.crs, self.nodata)
            newraster.tags["AREA_OR_POINT"] = "Area"
            return newraster

    def reproject(
        self: RasterType,
        dst_ref: RasterType | rio.io.Dataset | str | None = None,
        dst_crs: CRS | str | None = None,
        dst_size: tuple[int, int] | None = None,
        dst_bounds: dict[str, float] | rio.coords.BoundingBox | None = None,
        dst_res: float | abc.Iterable[float] | None = None,
        dst_nodata: int | float | list[int] | list[float] | None = None,
        src_nodata: int | float | list[int] | list[float] | None = None,
        dtype: np.dtype | None = None,
        resampling: Resampling | str = Resampling.bilinear,
        silent: bool = False,
        n_threads: int = 0,
        memory_limit: int = 64,
    ) -> RasterType:
        """
        Reproject raster to a specified grid.

        The output grid can either be given by a reference Raster (using `dst_ref`),
        or by manually providing the output CRS (`dst_crs`), dimensions (`dst_size`),
        resolution (with `dst_size`) and/or bounds (`dst_bounds`).
        Any resampling algorithm implemented in rasterio can be used.

        To reproject a Raster with different source bounds, first run Raster.crop.

        :param dst_ref: a reference raster. If set will use the attributes of this
            raster for the output grid. Can be provided as Raster/rasterio data set or as path to the file.
        :param crs: Specify the Coordinate Reference System to reproject to. If dst_ref not set, defaults to self.crs.
        :param dst_size: Raster size to write to (x, y). Do not use with dst_res.
        :param dst_bounds: a BoundingBox object or a dictionary containing\
                left, bottom, right, top bounds in the source CRS.
        :param dst_res: Pixel size in units of target CRS. Either 1 value or (xres, yres). Do not use with dst_size.
        :param dst_nodata: nodata value of the destination. If set to None, will use the same as source, \
        and if source is None, will use GDAL's default.
        :param src_nodata: nodata value of the source. If set to None, will read from the metadata.
        :param resampling: A rasterio Resampling method
        :param silent: If True, will not print warning statements
        :param n_threads: The number of worker threads. Defaults to (os.cpu_count() - 1).
        :param memory_limit: The warp operation memory limit in MB. Larger values may perform better.

        :returns: Raster

        """

        # Check that either dst_ref or dst_crs is provided
        if dst_ref is not None:
            if dst_crs is not None:
                raise ValueError("Either of `dst_ref` or `dst_crs` must be set. Not both.")
        else:
            # In case dst_res or dst_size is set, use original CRS
            if dst_crs is None:
                dst_crs = self.crs

        # Case a raster is provided as reference
        if dst_ref is not None:

            # Check that dst_ref type is either str, Raster or rasterio data set
            # Preferably use Raster instance to avoid rasterio data set to remain open. See PR #45
            if isinstance(dst_ref, Raster):
                ds_ref = dst_ref
            elif isinstance(dst_ref, rio.io.MemoryFile) or isinstance(dst_ref, rasterio.io.DatasetReader):
                ds_ref = dst_ref
            elif isinstance(dst_ref, str):
                assert os.path.exists(dst_ref), "Reference raster does not exist"
                ds_ref = Raster(dst_ref, load_data=False)
            else:
                raise ValueError(
                    "Type of dst_ref not understood, must be path to file (str), Raster or rasterio data set"
                )

            # Read reprojecting params from ref raster
            dst_crs = ds_ref.crs
            dst_size = (ds_ref.width, ds_ref.height)
            dst_res = None
            dst_bounds = ds_ref.bounds
        else:
            # Determine target CRS
            dst_crs = CRS.from_user_input(dst_crs)

        # Set output dtype
        if dtype is None:
            # Warning: this will not work for multiple bands with different dtypes
            dtype = self.dtypes[0]

        # Set source nodata if provided
        if src_nodata is None:
            src_nodata = self.nodata

        # Set destination nodata if provided. This is needed in areas not covered by the input data.
        # If None, will use GeoUtils' default, as rasterio's default is unknown, hence cannot be handled properly.
        if dst_nodata is None:
            dst_nodata = self.nodata
            if dst_nodata is None:
                dst_nodata = _default_nodata(dtype)
                # if dst_nodata is already being used, raise a warning.
                # TODO: for uint8, if all values are used, apply rio.warp to mask to identify invalid values
                if not self.is_loaded:
                    warnings.warn(
                        f"For reprojection, dst_nodata must be set. Setting default nodata to {dst_nodata}. You may "
                        f"set a different nodata with `dst_nodata`."
                    )

                elif dst_nodata in self.data:
                    warnings.warn(
                        f"For reprojection, dst_nodata must be set. Default chosen value {dst_nodata} exists in "
                        f"self.data. This may have unexpected consequences. Consider setting a different nodata with "
                        f"self.set_nodata()."
                    )

        from geoutils.misc import resampling_method_from_str

        # Basic reprojection options, needed in all cases.
        reproj_kwargs = {
            "src_transform": self.transform,
            "src_crs": self.crs,
            "dst_crs": dst_crs,
            "resampling": resampling if isinstance(resampling, Resampling) else resampling_method_from_str(resampling),
            "src_nodata": src_nodata,
            "dst_nodata": dst_nodata,
        }

        # If dst_ref is None, check other input arguments
        if dst_size is not None and dst_res is not None:
            raise ValueError("dst_size and dst_res both specified. Specify only one.")

        # Create a BoundingBox if required
        if dst_bounds is not None:
            if not isinstance(dst_bounds, rio.coords.BoundingBox):
                dst_bounds = rio.coords.BoundingBox(
                    dst_bounds["left"],
                    dst_bounds["bottom"],
                    dst_bounds["right"],
                    dst_bounds["top"],
                )

        # Determine target raster size/resolution
        dst_transform = None
        if dst_res is not None:
            if dst_bounds is None:
                # Let rasterio determine the maximum bounds of the new raster.
                reproj_kwargs.update({"dst_resolution": dst_res})
            else:

                # Bounds specified. First check if xres and yres are different.
                if isinstance(dst_res, tuple):
                    xres = dst_res[0]
                    yres = dst_res[1]
                else:
                    xres = dst_res
                    yres = dst_res

                # Calculate new raster size which ensures that pixels have
                # precisely the resolution specified.
                dst_width = np.ceil((dst_bounds.right - dst_bounds.left) / xres)
                dst_height = np.ceil(np.abs(dst_bounds.bottom - dst_bounds.top) / yres)
                dst_size = (int(dst_width), int(dst_height))

                # As a result of precise pixel size, the destination bounds may
                # have to be adjusted.
                x1 = dst_bounds.left + (xres * dst_width)
                y1 = dst_bounds.top - (yres * dst_height)
                dst_bounds = rio.coords.BoundingBox(top=dst_bounds.top, left=dst_bounds.left, bottom=y1, right=x1)

        # Set output shape (Note: dst_size is (ncol, nrow))
        if dst_size is not None:
            dst_shape = (self.count, dst_size[1], dst_size[0])
            dst_data = np.ones(dst_shape, dtype=dtype)
            reproj_kwargs.update({"destination": dst_data})
        else:
            dst_shape = (self.count, self.height, self.width)

        # If dst_bounds is set, will enforce dst_bounds
        if dst_bounds is not None:

            if dst_size is None:
                # Calculate new raster size which ensures that pixels resolution is as close as possible to original
                # Raster size is increased by up to one pixel if needed
                yres, xres = self.res
                dst_width = int(np.ceil((dst_bounds.right - dst_bounds.left) / xres))
                dst_height = int(np.ceil(np.abs(dst_bounds.bottom - dst_bounds.top) / yres))
                dst_size = (dst_width, dst_height)

            # Calculate associated transform
            dst_transform = rio.transform.from_bounds(*dst_bounds, width=dst_size[0], height=dst_size[1])

            # Specify the output bounds and shape, let rasterio handle the rest
            reproj_kwargs.update({"dst_transform": dst_transform})
            dst_data = np.ones((dst_size[1], dst_size[0]), dtype=dtype)
            reproj_kwargs.update({"destination": dst_data})

        # Check that reprojection is actually needed
        # Caution, dst_size is (width, height) while shape is (height, width)
        if all(
            [
                (dst_transform == self.transform) or (dst_transform is None),
                (dst_crs == self.crs) or (dst_crs is None),
                (dst_size == self.shape[::-1]) or (dst_size is None),
                np.all(dst_res == self.res) or (dst_res == self.res[0] == self.res[1]) or (dst_res is None),
            ]
        ):
            if (dst_nodata == self.nodata) or (dst_nodata is None):
                if not silent:
                    warnings.warn("Output projection, bounds and size are identical -> return self (not a copy!)")
                return self

            elif dst_nodata is not None:
                if not silent:
                    warnings.warn(
                        "Only nodata is different, consider using the 'set_nodata()' method instead'\
                    ' -> return self (not a copy!)"
                    )
                return self

        # Set the performance keywords
        if n_threads == 0:
            # Default to cpu count minus one. If the cpu count is undefined, num_threads will be 1
            cpu_count = os.cpu_count() or 2
            num_threads = cpu_count - 1
        else:
            num_threads = n_threads
        reproj_kwargs.update({"num_threads": num_threads, "warp_mem_limit": memory_limit})

        # If data is loaded, reproject the numpy array directly
        if self.is_loaded:

            # All masked values must be set to a nodata value for rasterio's reproject to work properly
            # TODO: another option is to apply rio.warp.reproject to the mask to identify invalid pixels
            if src_nodata is None and np.sum(self.data.mask) > 0:
                raise ValueError("No nodata set, use `src_nodata`.")

            # Mask not taken into account by rasterio, need to fill with src_nodata
            dst_data, dst_transformed = rio.warp.reproject(self.data.filled(src_nodata), **reproj_kwargs)

        # If not, uses the dataset instead
        else:
            dst_data = []
            for k in range(self.count):
                with rio.open(self.filename) as ds:
                    band = rio.band(ds, k + 1)
                    dst_band, dst_transformed = rio.warp.reproject(band, **reproj_kwargs)
                    dst_data.append(dst_band.squeeze())

            dst_data = np.array(dst_data)

        # Enforce output type
        dst_data = np.ma.masked_array(dst_data.astype(dtype), fill_value=dst_nodata)

        if dst_nodata is not None:
            dst_data.mask = dst_data == dst_nodata

        # Check for funny business.
        if dst_transform is not None:
            assert dst_transform == dst_transformed

        # Write results to a new Raster.
        dst_r = self.from_array(dst_data, dst_transformed, dst_crs, dst_nodata)

        return dst_r

    def shift(self, xoff: float, yoff: float) -> None:
        """
        Translate the Raster by a given x,y offset.

        :param xoff: Translation x offset.
        :param yoff: Translation y offset.


        """
        dx, b, xmin, d, dy, ymax = list(self.transform)[:6]

        self.transform = rio.transform.Affine(dx, b, xmin + xoff, d, dy, ymax + yoff)

    def save(
        self,
        filename: str | IO[bytes],
        driver: str = "GTiff",
        dtype: DTypeLike | None = None,
        nodata: AnyNumber | None = None,
        compress: str = "deflate",
        tiled: bool = False,
        blank_value: None | int | float = None,
        co_opts: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        gcps: list[tuple[float, ...]] | None = None,
        gcps_crs: CRS | None = None,
    ) -> None:
        """Write the Raster to a geo-referenced file.

        Given a filename to save the Raster to, create a geo-referenced file
        on disk which contains the contents of self.data.

        If blank_value is set to an integer or float, then instead of writing
        the contents of self.data to disk, write this provided value to every
        pixel instead.

        :param filename: Filename to write the file to.
        :param driver: the 'GDAL' driver to use to write the file as.
        :param dtype: Data Type to write the image as (defaults to dtype of image data)
        :param nodata: nodata value to be used.
        :param compress: Compression type. Defaults to 'deflate' (equal to GDALs: COMPRESS=DEFLATE)
        :param tiled: Whether to write blocks in tiles instead of strips. Improves read performance on large files,
                      but increases file size.
        :param blank_value: Use to write an image out with every pixel's value
            corresponding to this value, instead of writing the image data to disk.
        :param co_opts: GDAL creation options provided as a dictionary,
            e.g. {'TILED':'YES', 'COMPRESS':'LZW'}
        :param metadata: pairs of metadata key, value
        :param gcps: list of gcps, each gcp being [row, col, x, y, (z)]
        :param gcps_crs: the CRS of the GCPS (Default is None)


        :returns: None.
        """
        dtype = self.data.dtype if dtype is None else dtype

        if co_opts is None:
            co_opts = {}
        if metadata is None:
            metadata = {}
        if gcps is None:
            gcps = []

        # Use nodata set by user, otherwise default to self's
        nodata = nodata if nodata is not None else self.nodata

        if (self.data is None) & (blank_value is None):
            raise AttributeError("No data loaded, and alternative blank_value not set.")
        elif blank_value is not None:
            if isinstance(blank_value, int) | isinstance(blank_value, float):
                save_data = np.zeros(self.data.shape)
                save_data[:, :, :] = blank_value
            else:
                raise ValueError("blank_values must be one of int, float (or None).")
        else:
            save_data = self.data

            # if masked array, save with masked values replaced by nodata
            if isinstance(save_data, np.ma.masked_array):

                # In this case, nodata=None is not compatible, so revert to default values, only if masked values exist
                if (nodata is None) & (np.count_nonzero(save_data.mask) > 0):
                    nodata = _default_nodata(save_data.dtype)
                    warnings.warn(f"No nodata set, will use default value of {nodata}")
                save_data = save_data.filled(nodata)

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
            dst.update_tags(**metadata)

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

    def to_xarray(self, name: str | None = None) -> rioxarray.DataArray:
        """Convert this Raster into an xarray DataArray using rioxarray.

        This method uses rioxarray to generate a DataArray with associated
        geo-referencing information.

        See the documentation of rioxarray and xarray for more information on
        the methods and attributes of the resulting DataArray.

        :param name: Set the name of the DataArray.

        :returns: xarray DataArray
        """
        if not _has_rioxarray:
            raise ImportError("rioxarray is required for this functionality.")

        xr = rioxarray.open_rasterio(self.ds)
        if name is not None:
            xr.name = name

        return xr

    def get_bounds_projected(self, out_crs: CRS, densify_pts_max: int = 5000) -> rio.coords.BoundingBox:
        """
        Return self's bounds in the given CRS.

        :param out_crs: Output CRS
        :param densify_pts_max: Maximum points to be added between image corners to account for non linear edges.
                                Reduce if time computation is really critical (ms) or increase if extent is \
                                        not accurate enough.

        """
        # Max points to be added between image corners to account for non linear edges
        # rasterio's default is a bit low for very large images
        # instead, use image dimensions, with a maximum of 50000
        densify_pts = min(max(self.width, self.height), densify_pts_max)

        # Calculate new bounds
        left, bottom, right, top = self.bounds
        new_bounds = rio.warp.transform_bounds(self.crs, out_crs, left, bottom, right, top, densify_pts)
        new_bounds = rio.coords.BoundingBox(*new_bounds)

        return new_bounds

    def intersection(self, rst: str | Raster, match_ref: bool = True) -> tuple[float, float, float, float]:
        """
        Returns the bounding box of intersection between this image and another.

        If the rasters have different projections, the intersection extent is given in self's projection system.

        :param rst : path to the second image (or another Raster instance)
        :param match_ref: if set to True, returns the smallest intersection that aligns with that of self, i.e. same \
        resolution and offset with self's origin is a multiple of the resolution
        :returns: extent of the intersection between the 2 images \
        (xmin, ymin, xmax, ymax) in self's coordinate system.

        """
        from geoutils import projtools

        # If input rst is string, open as Raster
        if isinstance(rst, str):
            rst = Raster(rst, load_data=False)

        # Reproject the bounds of rst to self's
        rst_bounds_sameproj = rst.get_bounds_projected(self.crs)

        # Calculate intersection of bounding boxes
        intersection = projtools.merge_bounds([self.bounds, rst_bounds_sameproj], merging_algorithm="intersection")

        # check that intersection is not void, otherwise return 0 everywhere
        if intersection == ():
            warnings.warn("Intersection is void")
            return (0.0, 0.0, 0.0, 0.0)

        # if required, ensure the intersection is aligned with self's georeferences
        if match_ref:
            intersection = projtools.align_bounds(self.transform, intersection)

        # mypy raises a type issue, not sure how to address the fact that output of merge_bounds can be ()
        return intersection  # type: ignore

    def show(
        self,
        band: int | None = None,
        cmap: matplotlib.colors.Colormap | str | None = None,
        vmin: float | int | None = None,
        vmax: float | int | None = None,
        cb_title: str | None = None,
        add_cb: bool = True,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> None | tuple[matplotlib.axes.Axes, matplotlib.colors.Colormap]:
        r"""Show/display the image, with axes in projection of image.

        This method is a wrapper to rasterio.plot.show. Any \*\*kwargs which
        you give this method will be passed to rasterio.plot.show.

        :param band: which band to plot, from 0 to self.count-1 (default is all)
        :param cmap: The figure's colormap. Default is plt.rcParams['image.cmap']
        :param vmin: Colorbar minimum value. Default is data min.
        :param vmax: Colorbar maximum value. Default is data min.
        :param cb_title: Colorbar label. Default is None.
        :param add_cb: Set to True to display a colorbar. Default is True.
        :param ax: A figure ax to be used for plotting. If None, will create default figure and axes,\
                and plot figure directly.

        :returns: if ax is not None, returns (ax, cbar) where cbar is the colorbar (None if add_cb is False)


        You can also pass in \*\*kwargs to be used by the underlying imshow or
        contour methods of matplotlib. The example below shows provision of
        a kwarg for rasterio.plot.show, and a kwarg for matplotlib as well::

            import matplotlib.pyplot as plt
            ax1 = plt.subplot(111)
            mpl_kws = {'cmap':'seismic'}
            myimage.show(ax=ax1, mpl_kws)
        """
        # If data is not loaded, need to load it
        if not self.is_loaded:
            self.load()

        # Check if specific band selected, or take all
        # rshow takes care of image dimensions
        # if self.count=3 (4) => plotted as RGB(A)
        if band is None:
            band = np.arange(self.count)
        elif isinstance(band, int):
            if band >= self.count:
                raise ValueError(f"band must be in range 0-{self.count - 1:d}")
            pass
        else:
            raise ValueError("band must be int or None")

        # If multiple bands (RGB), cbar does not make sense
        if isinstance(band, abc.Sequence):
            if len(band) > 1:
                add_cb = False

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
            vmin = np.nanmin(self.data[band, :, :])

        if vmax is None:
            vmax = np.nanmax(self.data[band, :, :])

        # Make sure they are numbers, to avoid mpl error
        try:
            vmin = float(vmin)
            vmax = float(vmax)
        except ValueError:
            raise ValueError("vmin or vmax cannot be converted to float")

        # Create axes
        if ax is None:
            fig, ax0 = plt.subplots()
        elif isinstance(ax, matplotlib.axes.Axes):
            ax0 = ax
            fig = ax.figure
        else:
            raise ValueError("ax must be a matplotlib.axes.Axes instance or None")

        # Use data array directly, as rshow on self.ds will re-load data
        rshow(
            self.data[band, :, :],
            transform=self.transform,
            ax=ax0,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

        # Add colorbar
        if add_cb:
            cbar = fig.colorbar(
                cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                ax=ax0,
            )

            if cb_title is not None:
                cbar.set_label(cb_title)
        else:
            cbar = None

        # If ax not set, figure should be plotted directly
        if ax is None:
            plt.show()
            return None

        return ax0, cbar

    def value_at_coords(
        self,
        x: float | ArrayLike,
        y: float | ArrayLike,
        latlon: bool = False,
        band: int | None = None,
        masked: bool = False,
        window: int | None = None,
        return_window: bool = False,
        boundless: bool = True,
        reducer_function: Callable[[np.ndarray], float] = np.ma.mean,
    ) -> Any:
        """ Extract the pixel value(s) at the nearest pixel(s) from the specified coordinates.

        Extract pixel value of each band in dataset at the specified
        coordinates. Alternatively, if band is specified, return only that
        band's pixel value.

        Optionally, return mean of pixels within a square window.

        :param x: x (or longitude) coordinate.
        :param y: y (or latitude) coordinate.
        :param latlon: Set to True if coordinates provided as longitude/latitude.
        :param band: the band number to extract from.
        :param masked: If `masked` is `True` the return value will be a masked
            array. Otherwise (the default) the return value will be a
            regular array.
        :param window: expand area around coordinate to dimensions \
                  window * window. window must be odd.
        :param return_window: If True when window=int, returns (mean,array) \
            where array is the dataset extracted via the specified window size.
        :param boundless: If `True`, windows that extend beyond the dataset's extent
            are permitted and partially or completely filled arrays (with self.nodata) will
            be returned as appropriate.
        :param reducer_function: a function to apply to the values in window.

        :returns: When called on a Raster or with a specific band \
            set, return value of pixel.
        :returns: If multiple band Raster and the band is not specified, a \
            dictionary containing the value of the pixel in each band.
        :returns: In addition, if return_window=True, return tuple of \
            (values, arrays)

        :examples:

            >>> self.value_at_coords(-48.125, 67.8901, window=3)  # doctest: +SKIP
            Returns mean of a 3*3 window:
                v v v \
                v c v  | = float(mean)
                v v v /
            (c = provided coordinate, v= value of surrounding coordinate)

        """
        value: float | dict[int, float] | tuple[float | dict[int, float] | tuple[list[float], np.ndarray] | Any]
        if window is not None:
            if window % 2 != 1:
                raise ValueError("Window must be an odd number.")

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

        # Need to implement latlon option later
        if latlon:
            from geoutils import projtools

            x, y = projtools.reproject_from_latlon((y, x), self.crs)

        # Convert coordinates to pixel space
        row, col = rio.transform.rowcol(self.transform, x, y, op=round)

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
        window = rio.windows.Window(col, row, width, height)

        if self.is_loaded:
            data = self.data[slice(None) if band is None else band + 1, row : row + height, col : col + width]
            value = format_value(data)
            win: np.ndarray | dict[int, np.ndarray] = data

        else:
            if self.nbands == 1:
                with rio.open(self.filename) as raster:
                    data = raster.read(window=window, fill_value=self.nodata, boundless=boundless, masked=masked)
                value = format_value(data)
                win = data
            else:
                value = {}
                win = {}
                with rio.open(self.filename) as raster:
                    for b in self.indexes:
                        data = raster.read(
                            window=window, fill_value=self.nodata, boundless=boundless, masked=masked, indexes=b
                        )
                        val = format_value(data)
                        value[b] = val
                        win[b] = data  # type: ignore

        if return_window:
            return (value, win)

        return value

    def coords(self, offset: str = "corner", grid: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Get x,y coordinates of all pixels in the raster.

        :param offset: coordinate type. If 'corner', returns corner coordinates of pixels.
            If 'center', returns center coordinates. Default is corner.
        :param grid: Return grid

        :returns x,y: numpy arrays corresponding to the x,y coordinates of each pixel.
        """
        assert offset in [
            "corner",
            "center",
        ], f"ctype is not one of 'corner', 'center': {offset}"

        dx = self.res[0]
        dy = self.res[1]

        xx = np.linspace(self.bounds.left, self.bounds.right, self.width + 1)[:: int(np.sign(dx))]
        yy = np.linspace(self.bounds.bottom, self.bounds.top, self.height + 1)[:: int(np.sign(dy))]

        if offset == "center":
            xx += dx / 2  # shift by half a pixel
            yy += dy / 2
        if grid:
            meshgrid: tuple[np.ndarray, np.ndarray] = np.meshgrid(xx[:-1], yy[:-1])  # drop the last element
            return meshgrid
        else:
            return xx[:-1], yy[:-1]

    def xy2ij(
        self,
        x: ArrayLike,
        y: ArrayLike,
        op: type = np.float32,
        area_or_point: str | None = None,
        precision: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return row, column indices for a given x,y coordinate pair.

        :param x: x coordinates
        :param y: y coordinates
        :param op: operator to calculate index
        :param precision: precision for rio.Dataset.index
        :param area_or_point: shift index according to GDAL AREA_OR_POINT attribute (None) or \
                force position ('Point' or 'Area') of the interpretation of where the raster value \
                corresponds to in the pixel ('Area' = lower left or 'Point' = center)

        :returns i, j: indices of x,y in the image.


        """
        if op not in [np.float32, np.float64, float]:
            raise UserWarning(
                "Operator is not of type float: rio.Dataset.index might "
                "return unreliable indexes due to rounding issues."
            )
        if area_or_point not in [None, "Area", "Point"]:
            raise ValueError(
                'Argument "area_or_point" must be either None (falls back to GDAL metadata), "Point" or "Area".'
            )

        i, j = rio.transform.rowcol(self.transform, x, y, op=op, precision=precision)

        # # necessary because rio.Dataset.index does not return abc.Iterable for a single point
        if not isinstance(i, abc.Iterable):
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

        # AREA_OR_POINT GDAL attribute, i.e. does the value refer to the upper left corner (AREA) or
        # the center of pixel (POINT)
        # This has no influence on georeferencing, it's only about the interpretation of the raster values,
        # and thus only affects sub-pixel interpolation

        # if input is None, default to GDAL METADATA
        if area_or_point is None:
            area_or_point = self.tags.get("AREA_OR_POINT", "Point")

        if area_or_point == "Point":
            if not isinstance(i.flat[0], np.floating):
                raise ValueError(
                    "Operator must return np.floating values to perform AREA_OR_POINT subpixel index shifting"
                )

            # if point, shift index by half a pixel
            i += 0.5
            j += 0.5
            # otherwise, leave as is

        return i, j

    def ij2xy(self, i: ArrayLike, j: ArrayLike, offset: str = "center") -> tuple[np.ndarray, np.ndarray]:
        """
        Return x,y coordinates for a given row, column index pair.

        :param i: row (i) index of pixel.
        :param j: column (j) index of pixel.
        :param offset: return coordinates as "corner" or "center" of pixel

        :returns x, y: x,y coordinates of i,j in reference system.
        """

        x, y = rio.transform.xy(self.transform, i, j, offset=offset)

        return x, y

    def outside_image(self, xi: ArrayLike, yj: ArrayLike, index: bool = True) -> bool:
        """
        Check whether a given point falls outside of the raster.

        :param xi: Indices (or coordinates) of x direction to check.
        :param yj: Indices (or coordinates) of y direction to check.
        :param index: Interpret ij as raster indices (default is True). If False, assumes ij is coordinates.

        :returns is_outside: True if ij is outside of the image.
        """
        if not index:
            xi, xj = self.xy2ij(xi, yj)

        if np.any(np.array((xi, yj)) < 0):
            return True
        elif np.asanyarray(xi) > self.width or np.asanyarray(yj) > self.height:
            return True
        else:
            return False

    def interp_points(
        self,
        pts: ArrayLike,
        input_latlon: bool = False,
        mode: str = "linear",
        band: int = 1,
        area_or_point: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:

        """
         Interpolate raster values at a given point, or sets of points.

        :param pts: Point(s) at which to interpolate raster value. If points fall outside of image,
        value returned is nan. Shape should be (N,2)'
        :param input_latlon: Whether the input is in latlon, unregarding of Raster CRS
        :param mode: One of 'linear', 'cubic', or 'quintic'. Determines what type of spline is
             used to interpolate the raster value at each point. For more information, see
             scipy.interpolate.interp2d. Default is linear.
        :param band: Raster band to use
        :param area_or_point: shift index according to GDAL AREA_OR_POINT attribute (None) or force position\
                ('Point' or 'Area') of the interpretation of where the raster value corresponds to in the pixel\
                ('Area' = lower left or 'Point' = center)

        :returns rpts: Array of raster value(s) for the given points.
        """
        assert mode in [
            "mean",
            "linear",
            "cubic",
            "quintic",
            "nearest",
        ], "mode must be mean, linear, cubic, quintic or nearest."

        # get coordinates
        x, y = list(zip(*pts))

        # if those are in latlon, convert to Raster crs
        if input_latlon:
            init_crs = pyproj.CRS(4326)
            dest_crs = pyproj.CRS(self.crs)
            transformer = pyproj.Transformer.from_crs(init_crs, dest_crs)
            x, y = transformer.transform(x, y)

        i, j = self.xy2ij(x, y, op=np.float32, area_or_point=area_or_point)

        ind_invalid = np.vectorize(lambda k1, k2: self.outside_image(k1, k2, index=True))(j, i)

        rpts = map_coordinates(self.data[band - 1, :, :].astype(np.float32), [i, j], **kwargs)
        rpts = np.array(rpts, dtype=np.float32)
        rpts[np.array(ind_invalid)] = np.nan

        return rpts

        # #TODO: right now it's a loop... could add multiprocessing parallel loop outside,
        # # but such a method probably exists already within scipy/other interpolation packages?
        # for pt in pts:
        #     i,j = self.xy2ij(pt[0],pt[1])
        #     if self.outside_image(i,j, index=True):
        #         rpts.append(np.nan)
        #         continue
        #     else:
        #         x = xx[j - nsize:j + nsize + 1]
        #         y = yy[i - nsize:i + nsize + 1]
        #
        #         #TODO: read only that window?
        #         z = self.data[band-1, i - nsize:i + nsize + 1, j - nsize:j + nsize + 1]
        #         if mode in ['linear', 'cubic', 'quintic', 'nearest']:
        #             X, Y = np.meshgrid(x, y)
        #             try:
        #                 zint = griddata((X.flatten(), Y.flatten()), z.flatten(), list(pt), method=mode)[0]
        #             except:
        #                 #TODO: currently fails when dealing with the edges
        #                 print('Interpolation failed for:')
        #                 print(pt)
        #                 print(i,j)
        #                 print(x)
        #                 print(y)
        #                 print(z)
        #                 zint = np.nan
        #         else:
        #             zint = np.nanmean(z.flatten())
        #         rpts.append(zint)
        # rpts = np.array(rpts)

    def split_bands(self: RasterType, copy: bool = False, subset: list[int] | int | None = None) -> list[Raster]:
        """
        Split the bands into separate copied rasters.

        :param copy: Copy the bands or return slices of the original data.
        :param subset: Optional. A subset of band indices to extract. Defaults to all.

        :returns: A list of Rasters for each band.
        """
        bands: list[Raster] = []

        if subset is None:
            indices = list(range(self.nbands))
        elif isinstance(subset, int):
            indices = [subset]
        elif isinstance(subset, list):
            indices = subset
        else:
            raise ValueError(f"'subset' got invalid type: {type(subset)}. Expected list[int], int or None")

        if copy:
            for band_n in indices:
                # Generate a new Raster from a copy of the band's data
                bands.append(
                    self.from_array(
                        self.data[band_n, :, :].copy(),
                        transform=self.transform,
                        crs=self.crs,
                        nodata=self.nodata,
                    )
                )
        else:
            for band_n in indices:
                # Generate a new instance with the same underlying values.
                raster = Raster(self)
                # Set the data to a slice of the original array
                raster._data = self.data[band_n, :, :].reshape((1,) + self.data.shape[1:])
                # Set the nbands
                bands.append(raster)

        return bands

    @overload
    def to_points(
        self, subset: float | int, as_frame: Literal[True], pixel_offset: Literal["center", "corner"]
    ) -> gpd.GeoDataFrame:
        ...

    @overload
    def to_points(
        self, subset: float | int, as_frame: Literal[False], pixel_offset: Literal["center", "corner"]
    ) -> np.ndarray:
        ...

    def to_points(
        self, subset: float | int = 1, as_frame: bool = False, pixel_offset: Literal["center", "corner"] = "center"
    ) -> np.ndarray:
        """
        Subset a point cloud of the raster.

        If 'subset' is either 1 or is equal to the pixel count, all points are returned in order.
        If 'subset' is smaller than 1 (for fractions) or the pixel count, a random sample is returned.

        If the raster is not loaded, sampling will be done from disk without loading the entire Raster.

        Formats:
            * `as_frame` == None | False: A numpy ndarray of shape (N, 2 + nbands) with the columns [x, y, b1, b2..].
            * `as_frame` == True: A GeoPandas GeoDataFrame with the columns ["b1", "b2", ..., "geometry"]

        :param subset: The point count or fraction. If 'subset' > 1, it's parsed as a count.
        :param as_frame: Return a GeoDataFrame with a geometry column and crs instead of an ndarray.
        :param pixel_offset: The point at which to associate the pixel coordinate with ('corner' == upper left).

        :raises ValueError: If the subset count or fraction is poorly formatted.

        :returns: An ndarray/GeoDataFrame of the shape (N, 2 + nbands) where N is the subset count.
        """
        data_size = self.width * self.height

        # Validate the subset argument.
        if subset <= 0.0:
            raise ValueError(f"Subset cannot be zero or negative (given value: {subset})")
        # If the subset is equal to or less than 1, it is assumed to be a fraction.
        if subset <= 1.0:
            subset = int(data_size * subset)
        else:
            subset = int(subset)
        if subset > data_size:
            raise ValueError(f"Subset cannot exceed the size of the dataset ({subset} vs {data_size})")

        # If the subset is smaller than the max size, take a random subset of indices, otherwise take the whole.
        choice = np.random.randint(0, data_size - 1, subset) if subset != data_size else np.arange(data_size)

        cols = choice % self.width
        rows = (choice / self.width).astype(int)

        # Extract the coordinates of the pixels and filter by the chosen pixels.
        x_coords, y_coords = (np.array(a) for a in self.ij2xy(rows, cols, offset=pixel_offset))

        # If the Raster is loaded, pick from the data, otherwise use the disk-sample method from rasterio.
        if self.is_loaded:
            pixel_data = self.data[:, rows, cols]
        else:
            with rio.open(self.filename) as raster:
                pixel_data = np.array(list(raster.sample(zip(x_coords, y_coords)))).T

        if isinstance(pixel_data, np.ma.masked_array):
            pixel_data = np.where(pixel_data.mask, np.nan, pixel_data.data)

        # Merge the coordinates and pixel data into a point cloud.
        points = np.vstack((x_coords.reshape(1, -1), y_coords.reshape(1, -1), pixel_data)).T

        if as_frame:
            points = gpd.GeoDataFrame(
                points[:, 2:],
                columns=[f"b{i}" for i in range(1, pixel_data.shape[0] + 1)],
                geometry=gpd.points_from_xy(points[:, 0], points[:, 1]),
                crs=self.crs,
            )

        return points

    def polygonize(
        self, in_value: Number | tuple[Number, Number] | list[Number] | np.ndarray | Literal["all"] = 1
    ) -> Vector:
        """
        Return a GeoDataFrame polygonized from a raster.

        :param in_value: Value or range of values of the raster from which to
          create geometries (Default is 1). If 'all', all unique pixel values of the raster are used.

        :returns: Vector containing the polygonized geometries.
        """

        # mask a unique value set by a number
        if isinstance(in_value, Number):

            if np.sum(self.data == in_value) == 0:
                raise ValueError(f"no pixel with in_value {in_value}")

            bool_msk = np.array(self.data == in_value).astype(np.uint8)

        # mask values within boundaries set by a tuple
        elif isinstance(in_value, tuple):

            if np.sum((self.data > in_value[0]) & (self.data < in_value[1])) == 0:
                raise ValueError(f"no pixel with in_value between {in_value[0]} and {in_value[1]}")

            bool_msk = ((self.data > in_value[0]) & (self.data < in_value[1])).astype(np.uint8)

        # mask specific values set by a sequence
        elif isinstance(in_value, list) or isinstance(in_value, np.ndarray):

            if np.sum(np.isin(self.data, in_value)) == 0:
                raise ValueError("no pixel with in_value " + ", ".join(map("{}".format, in_value)))

            bool_msk = np.isin(self.data, in_value).astype("uint8")

        # mask all valid values
        elif in_value == "all":

            vals_for_msk = list(set(self.data.flatten()))
            bool_msk = np.isin(self.data, vals_for_msk).astype("uint8")

        else:

            raise ValueError("in_value must be a number, a tuple or a sequence")

        results = (
            {"properties": {"raster_value": v}, "geometry": s}
            for i, (s, v) in enumerate(shapes(self.data, mask=bool_msk, transform=self.transform))
        )

        gdf = gpd.GeoDataFrame.from_features(list(results))
        gdf.insert(0, "New_ID", range(0, 0 + len(gdf)))
        gdf.set_geometry(col="geometry", inplace=True)
        gdf.set_crs(self.crs, inplace=True)

        return gv.Vector(gdf)
