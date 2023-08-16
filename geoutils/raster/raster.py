"""
geoutils.raster provides a toolset for working with raster data.
"""
from __future__ import annotations

import math
import os
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
import pyproj
import rasterio as rio
import rasterio.warp
import rasterio.windows
from affine import Affine
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.plot import show as rshow
from scipy.ndimage import distance_transform_edt, map_coordinates

import geoutils.vector as gv
from geoutils._typing import (
    ArrayLike,
    DTypeLike,
    MArrayBool,
    MArrayNum,
    NDArrayBool,
    NDArrayNum,
    Number,
)
from geoutils.projtools import (
    _get_bounds_projected,
    _get_footprint_projected,
    _get_utm_ups_crs,
)
from geoutils.raster.sampling import subsample_array
from geoutils.vector import Vector

# If python38 or above, Literal is builtin. Otherwise, use typing_extensions
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

try:
    import rioxarray
    import xarray as xr

    _has_rioxarray = True
except ImportError:
    rioxarray = None
    _has_rioxarray = False

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


# Function to set the default nodata values for any given dtype
# Similar to GDAL for int types, but without absurdly long nodata values for floats.
# For unsigned types, the maximum value is chosen (with a max of 99999).
# For signed types, the minimum value is chosen (with a min of -99999).
def _default_nodata(dtype: DTypeLike) -> int:
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
    :param indexes: Band(s) to load. Note that rasterio begins counting at 1, not 0.
    :param masked: Whether the mask should be read (if any exists) to use the nodata to mask values.
    :param transform: Create a window from the given transform (to read only parts of the raster)
    :param shape: Expected shape of the read ndarray. Must be given together with the `transform` argument.
    :param out_count: Specify the count for a subsampled version (to be used with kwargs out_shape).

    :raises ValueError: If only one of ``transform`` and ``shape`` are given.

    :returns: An unmasked array if ``masked`` is ``False``, or a masked array otherwise.

    \*\*kwargs: any additional arguments to rasterio.io.DatasetReader.read.
    Useful ones are:
    .. hlist::
    * out_shape : to load a subsampled version, always use with out_count
    * window : to load a cropped version
    * resampling : to set the resampling algorithm
    """
    # If out_shape is passed, no need to account for transform and shape
    if kwargs["out_shape"] is not None:
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
        else:
            window = None

    if indexes is None:
        data = dataset.read(masked=masked, window=window, **kwargs)
    else:
        data = dataset.read(indexes=indexes, masked=masked, window=window, **kwargs)
    return data


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

    def __init__(
        self,
        filename_or_dataset: str
        | pathlib.Path
        | RasterType
        | rio.io.DatasetReader
        | rio.io.MemoryFile
        | dict[str, Any],
        indexes: int | list[int] | None = None,
        load_data: bool = False,
        downsample: Number = 1,
        masked: bool = True,
        nodata: int | float | None = None,
    ) -> None:
        """
        Instantiate a raster from a filename or rasterio dataset.

        :param filename_or_dataset: Path to file or Rasterio dataset.

        :param indexes: Band(s) to load into the object. Default loads all bands.

        :param load_data: Whether to load the array during instantiation. Default is False.

        :param downsample: Downsample the array once loaded by a round factor. Default is no downsampling.

        :param masked: Whether to load the array as a NumPy masked-array, with nodata values masked. Default is True.

        :param nodata: Nodata value to be used (overwrites the metadata). Default reads from metadata.
        """
        self._driver: str | None = None
        self._name: str | None = None
        self.filename: str | None = None
        self.tags: dict[str, Any] = {}

        self._data: MArrayNum | None = None
        self._transform: affine.Affine | None = None
        self._crs: CRS | None = None
        self._nodata: int | float | None = nodata
        self._indexes = indexes
        self._indexes_loaded: int | tuple[int, ...] | None = None
        self._masked = masked
        self._out_count: int | None = None
        self._out_shape: tuple[int, int] | None = None
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
                self._nodata = ds.nodata
                self._name = ds.name
                self._driver = ds.driver
                self.tags.update(ds.tags())

                self._disk_shape = (ds.count, ds.height, ds.width)
                self._disk_indexes = ds.indexes
                self._disk_dtypes = ds.dtypes

            # Check number of bands to be loaded
            if indexes is None:
                count = self.count
            elif isinstance(indexes, int):
                count = 1
            else:
                count = len(indexes)

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

            # This will record the downsampled out_shape is data is only loaded later on by .load()
            self._out_shape = out_shape
            self._out_count = count

            if load_data:
                # Mypy doesn't like the out_shape for some reason. I can't figure out why! (erikmannerfelt, 14/01/2022)
                # Don't need to pass shape and transform, because out_shape overrides it
                self.data = _load_rio(
                    ds,
                    indexes=indexes,
                    masked=masked,
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
            if len(self.data.shape) == 2:
                return 1
            else:
                return int(self.data.shape[0])
        #  This can only happen if data is not loaded, with a DatasetReader on disk is open, never returns None
        return self.count_on_disk  # type: ignore

    @property
    def height(self) -> int:
        """Height of the raster in pixels."""
        if not self.is_loaded:
            return self._disk_shape[1]  # type: ignore
        else:
            # If the raster is single-band
            if len(self.data.shape) == 2:
                return int(self.data.shape[0])
            # Or multi-band
            else:
                return int(self.data.shape[1])

    @property
    def width(self) -> int:
        """Width of the raster in pixels."""
        if not self.is_loaded:
            return self._disk_shape[2]  # type: ignore
        else:
            # If the raster is single-band
            if len(self.data.shape) == 2:
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
        return self.transform[0], abs(self.transform[4])

    @property
    def bounds(self) -> rio.coords.BoundingBox:
        """Bounding coordinates of the raster."""
        return rio.coords.BoundingBox(*rio.transform.array_bounds(self.height, self.width, self.transform))

    @property
    def is_loaded(self) -> bool:
        """Whether the raster array is loaded."""
        return self._data is not None

    @property
    def dtypes(self) -> tuple[str, ...]:
        """Data type for each raster band (string representation)."""
        if not self.is_loaded and self._disk_dtypes is not None:
            return self._disk_dtypes
        return (str(self.data.dtype),) * self.count

    @property
    def indexes_on_disk(self) -> None | tuple[int, ...]:
        """Indexes of bands on disk if it exists."""
        if self._disk_indexes is not None:
            return self._disk_indexes
        return None

    @property
    def indexes(self) -> tuple[int, ...]:
        """Indexes of bands loaded in memory if they are, otherwise on disk."""
        if self._indexes is not None and not self.is_loaded:
            if isinstance(self._indexes, int):
                return (self._indexes,)
            return tuple(self._indexes)
        # if self._indexes_loaded is not None:
        #     if isinstance(self._indexes_loaded, int):
        #         return (self._indexes_loaded, )
        #     return tuple(self._indexes_loaded)
        if self.is_loaded:
            return tuple(range(1, self.count + 1))
        return self.indexes_on_disk  # type: ignore

    @property
    def name(self) -> str | None:
        """Name of the file on disk, if it exists."""
        return self._name

    @property
    def driver(self) -> str | None:
        """Driver used to read a file on disk."""
        return self._driver

    def load(self, indexes: int | list[int] | None = None, **kwargs: Any) -> None:
        """
        Load the raster array from disk.

        :param kwargs: Optional keyword arguments sent to '_load_rio()'.
        :param indexes: Band(s) to load. Note that rasterio begins counting at 1, not 0.

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
        if indexes is None:
            valid_indexes = self.indexes
        # If a new index was pass, redefine out_shape
        elif isinstance(indexes, (int, list)):
            # Rewrite properly as a tuple
            if isinstance(indexes, int):
                valid_indexes = (indexes,)
            else:
                valid_indexes = tuple(indexes)
            # Update out_count if out_shape exists (when a downsampling has been passed)
            if self._out_shape is not None:
                self._out_count = len(valid_indexes)

        # Save which indexes are loaded
        self._indexes_loaded = valid_indexes

        # If a downsampled out_shape was defined during instantiation
        with rio.open(self.filename) as dataset:
            self.data = _load_rio(
                dataset,
                indexes=list(valid_indexes),
                masked=self._masked,
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
        nodata: int | float | tuple[int, ...] | tuple[float, ...] | None = None,
    ) -> RasterType:
        """Create a raster from a numpy array and the georeferencing information.

        :param data: Input array.
        :param transform: Affine 2D transform. Either a tuple(x_res, 0.0, top_left_x,
            0.0, y_res, top_left_y) or an affine.Affine object.
        :param crs: Coordinate reference system. Either a rasterio CRS,
            or an EPSG integer.
        :param nodata: Nodata value.

        :returns: Raster created from the provided array and georeferencing.

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
                raise TypeError("The transform argument needs to be Affine or tuple.")

        # Enable shortcut to create CRS from an EPSG ID.
        if isinstance(crs, int):
            crs = CRS.from_epsg(crs)

        # If the data was transformed into boolean, re-initialize as a Mask subclass
        # Typing: we can specify this behaviour in @overload once we add the NumPy plugin of MyPy
        if data.dtype == bool:
            return Mask({"data": data, "transform": transform, "crs": crs, "nodata": nodata})  # type: ignore
        # Otherwise, keep as a given RasterType subclass
        else:
            return cls({"data": data, "transform": transform, "crs": crs, "nodata": nodata})

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
            dtype=self.dtypes[0],
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

        # Over-ride Raster's method to remove nodata value (always None)
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

    def __getitem__(self, index: Raster | Vector | NDArrayNum | list[float] | tuple[float, ...]) -> NDArrayNum | Raster:
        """
        Index or subset the raster.

        Two cases:
        - If a mask of same georeferencing or array of same shape is passed, return the indexed raster array.
        - If a raster, vector, list or tuple of bounds is passed, return the cropped raster matching those objects.
        """

        # If input is Mask with the same shape and georeferencing
        if isinstance(index, Mask):
            if not self.georeferenced_grid_equal(index):
                raise ValueError("Indexing a raster with a mask requires the two being on the same georeferenced grid.")
            if self.count == 1:
                return self.data[index.data.squeeze()]
            else:
                return self.data[:, index.data.squeeze()]
        # If input is array with the same shape
        elif isinstance(index, np.ndarray):
            if np.shape(index) != self.shape:
                raise ValueError("Indexing a raster with an array requires the two having the same shape.")
            if str(index.dtype) != "bool":
                index = index.astype(bool)
                warnings.warn(message="Input array was cast to boolean for indexing.", category=UserWarning)
            if self.count == 1:
                return self.data[index]
            else:
                return self.data[:, index]

        # Otherwise, subset with crop
        else:
            return self.crop(crop_geom=index, inplace=False)

    def __setitem__(self, index: Mask | NDArrayBool, assign: NDArrayNum | Number) -> None:
        """
        Perform index assignment on the raster.

        If a mask of same georeferencing or array of same shape is passed,
        it is used as index to assign values to the raster array.
        """

        # First, check index

        # If input is Mask with the same shape and georeferencing
        if isinstance(index, Mask):
            if not self.georeferenced_grid_equal(index):
                raise ValueError("Indexing a raster with a mask requires the two being on the same georeferenced grid.")

            ind = index.data.data
        # If input is array with the same shape
        elif isinstance(index, np.ndarray):
            if np.shape(index) != self.shape:
                raise ValueError("Indexing a raster with an array requires the two having the same shape.")
            if str(index.dtype) != "bool":
                ind = index.astype(bool)
                warnings.warn(message="Input array was cast to boolean for indexing.", category=UserWarning)
            else:
                ind = index
        # Otherwise, raise an error
        else:
            raise ValueError(
                "Indexing a raster requires a mask of same georeferenced grid, or a boolean array of same shape."
            )

        # Second, assign, NumPy will raise appropriate errors itself

        # We need to explicitly load here, as we cannot call the data getter/setter directly
        if not self.is_loaded:
            self.load()
        # Assign the values to the index
        if self.count == 1:
            self._data[ind] = assign  # type: ignore
        else:
            self._data[:, ind] = assign  # type: ignore
        return None

    def raster_equal(self, other: object) -> bool:
        """
        Check if two rasters are equal.

        This means that are equal:
        - The raster's masked array's data (including masked values), mask, fill_value and dtype,
        - The raster's transform, crs and nodata values.
        """

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

    def _overloading_check(
        self: RasterType, other: RasterType | NDArrayNum | Number
    ) -> tuple[MArrayNum, MArrayNum | NDArrayNum | Number, float | int | tuple[int, ...] | tuple[float, ...] | None]:
        """
        Before any operation overloading, check input data type and return both self and other data as either
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
        if not isinstance(other, (Raster, np.ndarray, float, int, np.floating, np.integer)):
            raise NotImplementedError(
                f"Operation between an object of type {type(other)} and a Raster impossible. Must be a Raster, "
                f"np.ndarray or single number."
            )

        # Get self's dtype and nodata
        nodata1 = self.nodata
        dtype1 = self.data.dtype

        # Case 1 - other is a Raster
        if isinstance(other, Raster):
            # Not necessary anymore with implicit loading
            # # Check that both data are loaded
            # if not (self.is_loaded & other.is_loaded):
            #     raise ValueError("Raster's data must be loaded with self.load().")

            # Check that both rasters have the same shape and georeferences
            if (self.data.shape == other.data.shape) & (self.transform == other.transform) & (self.crs == other.crs):
                pass
            else:
                raise ValueError("Both rasters must have the same shape, transform and CRS.")

            nodata2 = other.nodata
            dtype2 = other.data.dtype
            other_data: NDArrayNum | MArrayNum | Number = other.data

        # Case 2 - other is a numpy array
        elif isinstance(other, np.ndarray):
            # Check that both array have the same shape

            # Squeeze first axis of other data if possible
            if len(other.shape) == 3 and other.shape[0] == 1:
                other_data = other.squeeze(axis=0)
            else:
                other_data = other

            if self.data.shape == other_data.shape:
                pass
            else:
                raise ValueError("The raster and array must have the same shape.")

            nodata2 = None
            dtype2 = other.dtype

        # Case 3 - other is a single number
        else:
            other_data = other
            nodata2 = None
            dtype2 = rio.dtypes.get_minimum_dtype(other_data)

        # Figure out output dtype
        out_dtype = np.promote_types(dtype1, dtype2)

        # Figure output nodata
        out_nodata = None
        if (nodata2 is not None) and (out_dtype == dtype2):
            out_nodata = nodata2
        if (nodata1 is not None) and (out_dtype == dtype1):
            out_nodata = nodata1

        self_data = self.data

        return self_data, other_data, out_nodata

    def __add__(self: RasterType, other: RasterType | NDArrayNum | Number) -> RasterType:
        """
        Sum two rasters, or a raster and a numpy array, or a raster and single number.

        If other is a Raster, it must have the same shape, transform and crs as self.
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
        return self.from_array(-self.data, self.transform, self.crs, nodata=self.nodata)

    def __sub__(self, other: Raster | NDArrayNum | Number) -> Raster:
        """
        Subtract two rasters, or a raster and a numpy array, or a raster and single number.

        If other is a Raster, it must have the same shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, nodata = self._overloading_check(other)
        out_data = self_data - other_data
        return self.from_array(out_data, self.transform, self.crs, nodata=nodata)

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __rsub__(self: RasterType, other: NDArrayNum | Number) -> RasterType:  # type: ignore
        """
        Subtract two rasters, or a raster and a numpy array, or a raster and single number.

        For when other is first item in the operation (e.g. 1 - rst).
        """
        self_data, other_data, nodata = self._overloading_check(other)
        out_data = other_data - self_data
        return self.from_array(out_data, self.transform, self.crs, nodata=nodata)

    def __mul__(self: RasterType, other: RasterType | NDArrayNum | Number) -> RasterType:
        """
        Multiply two rasters, or a raster and a numpy array, or a raster and single number.

        If other is a Raster, it must have the same shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, nodata = self._overloading_check(other)
        out_data = self_data * other_data
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata)
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
        self_data, other_data, nodata = self._overloading_check(other)
        out_data = self_data / other_data
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata)
        return out_rst

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __rtruediv__(self: RasterType, other: NDArrayNum | Number) -> RasterType:  # type: ignore
        """
        True division of two rasters, or a raster and a numpy array, or a raster and single number.

        For when other is first item in the operation (e.g. 1/rst).
        """
        self_data, other_data, nodata = self._overloading_check(other)
        out_data = other_data / self_data
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata)
        return out_rst

    def __floordiv__(self: RasterType, other: RasterType | NDArrayNum | Number) -> RasterType:
        """
        Floor division of two rasters, or a raster and a numpy array, or a raster and single number.

        If other is a Raster, it must have the same shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, nodata = self._overloading_check(other)
        out_data = self_data // other_data
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata)
        return out_rst

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __rfloordiv__(self: RasterType, other: NDArrayNum | Number) -> RasterType:  # type: ignore
        """
        Floor division of two rasters, or a raster and a numpy array, or a raster and single number.

        For when other is first item in the operation (e.g. 1/rst).
        """
        self_data, other_data, nodata = self._overloading_check(other)
        out_data = other_data // self_data
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata)
        return out_rst

    def __mod__(self: RasterType, other: RasterType | NDArrayNum | Number) -> RasterType:
        """
        Modulo of two rasters, or a raster and a numpy array, or a raster and single number.

        If other is a Raster, it must have the same shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, nodata = self._overloading_check(other)
        out_data = self_data % other_data
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata)
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
        out_rst = self.from_array(out_data, self.transform, self.crs, nodata=nodata)
        return out_rst

    def __eq__(self: RasterType, other: RasterType | NDArrayNum | Number) -> RasterType:  # type: ignore
        """
        Element-wise equality of two rasters, or a raster and a numpy array, or a raster and single number.

        This operation casts the result into a Mask.

        If other is a Raster, it must have the same shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, _ = self._overloading_check(other)
        out_data = self_data == other_data
        out_mask = self.from_array(out_data, self.transform, self.crs, nodata=None)
        return out_mask

    def __ne__(self: RasterType, other: RasterType | NDArrayNum | Number) -> RasterType:  # type: ignore
        """
        Element-wise negation of two rasters, or a raster and a numpy array, or a raster and single number.

        This operation casts the result into a Mask.

        If other is a Raster, it must have the same shape, transform and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        self_data, other_data, _ = self._overloading_check(other)
        out_data = self_data != other_data
        out_mask = self.from_array(out_data, self.transform, self.crs, nodata=None)
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
        self_data, other_data, _ = self._overloading_check(other)
        out_data = self_data < other_data
        out_mask = self.from_array(out_data, self.transform, self.crs, nodata=None)
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
        self_data, other_data, _ = self._overloading_check(other)
        out_data = self_data <= other_data
        out_mask = self.from_array(out_data, self.transform, self.crs, nodata=None)
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
        self_data, other_data, _ = self._overloading_check(other)
        out_data = self_data > other_data
        out_mask = self.from_array(out_data, self.transform, self.crs, nodata=None)
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
        self_data, other_data, _ = self._overloading_check(other)
        out_data = self_data >= other_data
        out_mask = self.from_array(out_data, self.transform, self.crs, nodata=None)
        return out_mask

    @overload
    def astype(self, dtype: DTypeLike, inplace: Literal[False] = False) -> Raster:
        ...

    @overload
    def astype(self, dtype: DTypeLike, inplace: Literal[True]) -> None:
        ...

    def astype(self, dtype: DTypeLike, inplace: bool = False) -> Raster | None:
        """
        Convert data type of the raster.

        :param dtype: Any numpy dtype or string accepted by numpy.astype.
        :param inplace: Whether to modify the raster in-place.

        :returns: Raster with updated dtype.
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
            self._data = out_data  # type: ignore
            return None
        else:
            return self.from_array(out_data, self.transform, self.crs, nodata=self.nodata)

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
            if not rio.dtypes.can_cast_dtype(new_nodata, self.dtypes[0]):
                raise ValueError(f"nodata value {new_nodata} incompatible with self.dtype {self.dtypes[0]}")

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
        3. Raises an error if the dtype is different from that of the Raster, and points towards .copy() or .astype(),
        4. Sets a new nodata value to the Raster if none is set and if the provided array contains non-finite values
            that are unmasked (including if there is no mask at all, e.g. NaNs in a classic array),
        5. Masks non-finite values that are unmasked, whether the input is a classic array or a masked_array. Note that
            these values are not overwritten and can still be accessed in .data.data.

        :param new_data: New data to assign to this instance of Raster.

        """
        # Check that new_data is a NumPy array
        if not isinstance(new_data, np.ndarray):
            raise ValueError("New data must be a numpy array.")

        if len(new_data.shape) not in [2, 3]:
            raise ValueError("Data array must have 2 or 3 dimensions.")

        # Squeeze 3D data if the band axis is of length 1
        if len(new_data.shape) == 3 and new_data.shape[0] == 1:
            new_data = new_data.squeeze(axis=0)

        # Check that new_data has correct shape

        # If data is loaded
        if self._data is not None:
            dtype = str(self._data.dtype)
            orig_shape = self._data.shape
        # If filename exists
        elif self._disk_dtypes is not None:
            dtype = self._disk_dtypes[0]
            if self._out_count == 1:
                orig_shape = self._out_shape
            else:
                orig_shape = (self._out_count, *self._out_shape)  # type: ignore
        else:
            dtype = str(new_data.dtype)
            orig_shape = new_data.shape

        # Check that new_data has the right type
        if str(new_data.dtype) != dtype:
            raise ValueError(
                "New data must be of the same type as existing data: {}. Use copy() to set a new array with "
                "different dtype, or astype() to change type.".format(dtype)
            )

        if new_data.shape != orig_shape:
            raise ValueError(
                f"New data must be of the same shape as existing data: {orig_shape}. Given: {new_data.shape}."
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

    def set_mask(self, mask: NDArrayBool | Mask) -> None:
        """
        Set a mask on the raster array.

        All pixels where `mask` is set to True or > 0 will be masked (in addition to previously masked pixels).

        Masking is performed in place. The mask must have the same shape as loaded data,
        unless the first dimension is 1, then it is ignored.

        :param mask: The raster array mask.
        """
        # Check that mask is a Numpy array
        if not isinstance(mask, (np.ndarray, Mask)):
            raise ValueError("mask must be a numpy array or a Mask.")

        # Check that new_data has correct shape
        if self.is_loaded:
            orig_shape = self.data.shape
        else:
            raise AttributeError("self.data must be loaded first, with e.g. self.load()")

        # If the mask is a Mask instance, pass the boolean array
        if isinstance(mask, Mask):
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

    def info(self, stats: bool = False) -> str:
        """
        Summarize information about the raster.

        :param stats: Add statistics for each band of the dataset (max, min, median, mean, std. dev.). Default is to
            not calculate statistics.


        :returns: text information about Raster attributes.

        """
        as_str = [
            f"Driver:               {self.driver} \n",
            f"Opened from file:     {self.filename} \n",
            f"Filename:             {self.name} \n",
            f"Loaded?               {self.is_loaded} \n",
            f"Modified since load?  {self.is_modified} \n",
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
            if not self.is_loaded:
                self.load()

            if self.count == 1:
                as_str.append(f"[MAXIMUM]:          {np.nanmax(self.data):.2f}\n")
                as_str.append(f"[MINIMUM]:          {np.nanmin(self.data):.2f}\n")
                as_str.append(f"[MEDIAN]:           {np.ma.median(self.data):.2f}\n")
                as_str.append(f"[MEAN]:             {np.nanmean(self.data):.2f}\n")
                as_str.append(f"[STD DEV]:          {np.nanstd(self.data):.2f}\n")
            else:
                for b in range(self.count):
                    # try to keep with rasterio convention.
                    as_str.append(f"Band {b + 1}:\n")
                    as_str.append(f"[MAXIMUM]:          {np.nanmax(self.data[b, :, :]):.2f}\n")
                    as_str.append(f"[MINIMUM]:          {np.nanmin(self.data[b, :, :]):.2f}\n")
                    as_str.append(f"[MEDIAN]:           {np.ma.median(self.data[b, :, :]):.2f}\n")
                    as_str.append(f"[MEAN]:             {np.nanmean(self.data[b, :, :]):.2f}\n")
                    as_str.append(f"[STD DEV]:          {np.nanstd(self.data[b, :, :]):.2f}\n")

        return "".join(as_str)

    def copy(self: RasterType, new_array: NDArrayNum | None = None) -> RasterType:
        """
        Copy the raster in-memory.

        :param new_array: New array to use in the copied raster.

        :return: Copy of the raster.
        """
        if new_array is not None:
            data = new_array
        else:
            data = self.data.copy()

        cp = self.from_array(data=data, transform=self.transform, crs=self.crs, nodata=self.nodata)

        return cp

    def georeferenced_grid_equal(self: RasterType, raster: RasterType) -> bool:
        """
        Check that raster shape, geotransform and CRS are equal.

        :param raster: Another raster.

        :return: Whether the two objects have the same georeferenced grid.
        """

        return all([self.shape == raster.shape, self.transform == raster.transform, self.crs == raster.crs])

    @overload
    def get_nanarray(self, return_mask: Literal[False] = False) -> NDArrayNum:
        ...

    @overload
    def get_nanarray(self, return_mask: Literal[True]) -> tuple[NDArrayNum, NDArrayBool]:
        ...

    def get_nanarray(self, return_mask: bool = False) -> NDArrayNum | tuple[NDArrayNum, NDArrayBool]:
        """
        Get NaN array from the raster.

        Optionally, return the mask from the masked array.

        :param return_mask: Whether to return the mask of valid data.

        :returns Array with masked data as NaNs, (Optional) Mask of valid data.
        """

        # Cast array to float32 is its dtype is integer (cannot be filled with NaNs otherwise)
        if "int" in str(self.data.dtype):
            # Get the array with masked value fill with NaNs
            nanarray = self.data.astype("float32").filled(fill_value=np.nan).squeeze()
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
        if func.__name__ in _HANDLED_FUNCTIONS_1NIN:
            outputs = func(first_arg, *args[1:], **kwargs)  # type: ignore
        else:
            second_arg = args[1].data
            outputs = func(first_arg, second_arg, *args[2:], **kwargs)  # type: ignore

        # Below, we recast to Raster if the shape was preserved, otherwise return an array
        # First, if there are several outputs in a tuple which are arrays
        if isinstance(outputs, tuple) and isinstance(outputs[0], np.ndarray):
            if all(output.shape == args[0].data.shape for output in outputs):
                return (
                    self.from_array(data=output, transform=self.transform, crs=self.crs, nodata=self.nodata)
                    for output in outputs
                )
            else:
                return outputs
        # Second, if there is a single output which is an array
        elif isinstance(outputs, np.ndarray):
            if outputs.shape == args[0].data.shape:
                return self.from_array(data=outputs, transform=self.transform, crs=self.crs, nodata=self.nodata)
            else:
                return outputs
        # Else, return outputs directly
        else:
            return outputs

    # Note the star is needed because of the default argument 'mode' preceding non default arg 'inplace'
    # Then the final overload must be duplicated
    @overload
    def crop(
        self: RasterType,
        crop_geom: RasterType | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        *,
        inplace: Literal[True],
    ) -> None:
        ...

    @overload
    def crop(
        self: RasterType,
        crop_geom: RasterType | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        *,
        inplace: Literal[False],
    ) -> RasterType:
        ...

    @overload
    def crop(
        self: RasterType,
        crop_geom: RasterType | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        inplace: bool = True,
    ) -> RasterType | None:
        ...

    def crop(
        self: RasterType,
        crop_geom: RasterType | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        inplace: bool = True,
    ) -> RasterType | None:
        """
        Crop the raster to a given extent.

        **Match-reference:** a reference raster or vector can be passed to match bounds during cropping.

        Reprojection is done on the fly if georeferenced objects have different projections.

        :param crop_geom: Geometry to crop raster to. Can use either a raster or vector as match-reference, or a list of
            coordinates. If ``crop_geom`` is a raster or vector, will crop to the bounds. If ``crop_geom`` is a
            list of coordinates, the order is assumed to be [xmin, ymin, xmax, ymax].
        :param mode: Whether to match within pixels or exact extent. ``'match_pixel'`` will preserve the original pixel
            resolution, cropping to the extent that most closely aligns with the current coordinates. ``'match_extent'``
            will match the extent exactly, adjusting the pixel resolution to fit the extent.
        :param inplace: Whether to update the raster in-place.

        :returns: None for in-place cropping (defaults) or a new raster otherwise.
        """
        assert mode in [
            "match_extent",
            "match_pixel",
        ], "mode must be one of 'match_pixel', 'match_extent'"

        if isinstance(crop_geom, (Raster, Vector)):
            # For another Vector or Raster, we reproject the bounding box in the same CRS as self
            xmin, ymin, xmax, ymax = crop_geom.get_bounds_projected(out_crs=self.crs)
        elif isinstance(crop_geom, (list, tuple)):
            xmin, ymin, xmax, ymax = crop_geom
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
                if self.count == 1:
                    crop_img = self.data[rowmin:rowmax, colmin:colmax]
                else:
                    crop_img = self.data[:, rowmin:rowmax, colmin:colmax]
            else:
                with rio.open(self.filename) as raster:
                    crop_img = raster.read(
                        indexes=self._indexes,
                        masked=self._masked,
                        window=final_window,
                    )
                # Squeeze first axis for single-band
                if len(crop_img.shape) == 3 and crop_img.shape[0] == 1:
                    crop_img = crop_img.squeeze(axis=0)

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
        dst_ref: RasterType | str | None = None,
        dst_crs: CRS | str | int | None = None,
        dst_size: tuple[int, int] | None = None,
        dst_bounds: dict[str, float] | rio.coords.BoundingBox | None = None,
        dst_res: float | abc.Iterable[float] | None = None,
        dst_nodata: int | float | tuple[int, ...] | tuple[float, ...] | None = None,
        src_nodata: int | float | tuple[int, ...] | tuple[float, ...] | None = None,
        dst_dtype: DTypeLike | None = None,
        resampling: Resampling | str = Resampling.bilinear,
        silent: bool = False,
        n_threads: int = 0,
        memory_limit: int = 64,
    ) -> RasterType:
        """
        Reproject raster to a different geotransform (resolution, bounds) and/or coordinate reference system (CRS).

        **Match-reference**: a reference raster can be passed to match resolution, bounds and CRS during reprojection.

        Alternatively, the destination resolution, bounds and CRS can be passed individually.

        Any resampling algorithm implemented in Rasterio can be passed as a string.


        :param dst_ref: Reference raster to match resolution, bounds and CRS.
        :param dst_crs: Destination coordinate reference system as a string or EPSG. If ``dst_ref`` not set,
            defaults to this raster's CRS.
        :param dst_size: Destination size as (x, y). Do not use with ``dst_res``.
        :param dst_bounds: Destination bounds as a Rasterio bounding box, or a dictionary containing left, bottom,
            right, top bounds in the destination CRS.
        :param dst_res: Destination resolution (pixel size) in units of destination CRS. Single value or (xres, yres).
            Do not use with ``dst_size``.
        :param dst_nodata: Destination nodata value. If set to ``None``, will use the same as source. If source does
            not exist, will use GDAL's default.
        :param dst_dtype: Destination data type of array.
        :param src_nodata: Force a source nodata value (read from the metadata by default).
        :param resampling: A Rasterio resampling method, can be passed as a string.
            See https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling
            for the full list.
        :param silent: Whether to print warning statements.
        :param n_threads: Number of threads. Defaults to (os.cpu_count() - 1).
        :param memory_limit: Memory limit in MB for warp operations. Larger values may perform better.

        :returns: Reprojected raster.

        """

        # Check that either dst_ref or dst_crs is provided
        if dst_ref is not None and dst_crs is not None:
            raise ValueError("Either of `dst_ref` or `dst_crs` must be set. Not both.")
        # If none are provided, simply preserve the CRS
        elif dst_ref is None and dst_crs is None:
            dst_crs = self.crs

        # Set output dtype
        if dst_dtype is None:
            # Warning: this will not work for multiple bands with different dtypes
            dst_dtype = self.dtypes[0]

        # Set source nodata if provided
        if src_nodata is None:
            src_nodata = self.nodata

        # Set destination nodata if provided. This is needed in areas not covered by the input data.
        # If None, will use GeoUtils' default, as rasterio's default is unknown, hence cannot be handled properly.
        if dst_nodata is None:
            dst_nodata = self.nodata
            if dst_nodata is None:
                dst_nodata = _default_nodata(dst_dtype)
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

        dst_transform = None
        # Case a raster is provided as reference
        if dst_ref is not None:
            # Check that dst_ref type is either str, Raster or rasterio data set
            # Preferably use Raster instance to avoid rasterio data set to remain open. See PR #45
            if isinstance(dst_ref, Raster):
                ds_ref = dst_ref
            elif isinstance(dst_ref, str):
                if not os.path.exists(dst_ref):
                    raise ValueError("Reference raster does not exist.")
                ds_ref = Raster(dst_ref, load_data=False)
            else:
                raise TypeError("Type of dst_ref not understood, must be path to file (str), Raster.")

            # Read reprojecting params from ref raster
            dst_crs = ds_ref.crs
            dst_transform = ds_ref.transform
            reproj_kwargs.update({"dst_transform": dst_transform})
            dst_data = np.ones((self.count, ds_ref.shape[0], ds_ref.shape[1]), dtype=dst_dtype)
            reproj_kwargs.update({"destination": dst_data})
            reproj_kwargs.update({"dst_crs": ds_ref.crs})
        else:
            # Determine target CRS
            dst_crs = CRS.from_user_input(dst_crs)

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
            dst_data = np.ones(dst_shape, dtype=dst_dtype)
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
            dst_data = np.ones((self.count, dst_size[1], dst_size[0]), dtype=dst_dtype)
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
            dst_data = []  # type: ignore
            for k in range(self.count):
                with rio.open(self.filename) as ds:
                    band = rio.band(ds, k + 1)
                    dst_band, dst_transformed = rio.warp.reproject(band, **reproj_kwargs)
                    dst_data.append(dst_band.squeeze())

            dst_data = np.array(dst_data)

        # Enforce output type
        dst_data = np.ma.masked_array(dst_data.astype(dst_dtype), fill_value=dst_nodata)

        if dst_nodata is not None:
            dst_data.mask = dst_data == dst_nodata

        # Check for funny business.
        if dst_transform is not None:
            assert dst_transform == dst_transformed

        # Write results to a new Raster.
        dst_r = self.from_array(dst_data, dst_transformed, dst_crs, dst_nodata)

        return dst_r

    def shift(
        self, xoff: float, yoff: float, distance_unit: Literal["georeferenced"] | Literal["pixel"] = "georeferenced"
    ) -> None:
        """
        Shift the raster by a (x,y) offset.

        The shifting only updates the geotransform (no resampling is performed).

        :param xoff: Translation x offset.
        :param yoff: Translation y offset.
        :param distance_unit: Distance unit, either 'georeferenced' (default) or 'pixel'.
        """
        if distance_unit not in ["georeferenced", "pixel"]:
            raise ValueError("Argument 'distance_unit' should be either 'pixel' or 'georeferenced'.")

        # Get transform
        dx, b, xmin, d, dy, ymax = list(self.transform)[:6]

        # Convert pixel offsets to georeferenced units
        if distance_unit == "pixel":
            xoff *= self.res[0]
            yoff *= self.res[1]

        # Overwrite transform by shifted transform
        self.transform = rio.transform.Affine(dx, b, xmin + xoff, d, dy, ymax + yoff)

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
        :param metadata: Pairs of metadata key, value.
        :param gcps: List of gcps, each gcp being [row, col, x, y, (z)].
        :param gcps_crs: CRS of the GCPS.

        :returns: None.
        """

        if co_opts is None:
            co_opts = {}
        if metadata is None:
            metadata = {}
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
            if save_data.dtype == bool:
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

    def to_xarray(self, name: str | None = None) -> xr.DataArray:
        """
        Convert raster to a xarray.DataArray.

        This method uses rioxarray to generate a DataArray with associated
        geo-referencing information.

        See the documentation of rioxarray and xarray for more information on
        the methods and attributes of the resulting DataArray.

        :param name: Name attribute for the DataArray.

        :returns: xarray DataArray
        """
        if not _has_rioxarray:
            raise ImportError("rioxarray is required for this functionality.")

        ds = rioxarray.open_rasterio(self.to_rio_dataset())
        if name is not None:
            ds.name = name

        return ds

    def get_bounds_projected(self, out_crs: CRS, densify_pts: int = 5000) -> rio.coords.BoundingBox:
        """
        Get raster bounds projected in a specified CRS.

        :param out_crs: Output CRS.
        :param densify_pts: Maximum points to be added between image corners to account for non linear edges.
         Reduce if time computation is really critical (ms) or increase if extent is not accurate enough.

        """
        # Max points to be added between image corners to account for non linear edges
        # rasterio's default is a bit low for very large images
        # instead, use image dimensions, with a maximum of 50000
        densify_pts = min(max(self.width, self.height), densify_pts)

        # Calculate new bounds
        new_bounds = _get_bounds_projected(self.bounds, in_crs=self.crs, out_crs=out_crs, densify_pts=densify_pts)

        return new_bounds

    def get_footprint_projected(self, out_crs: CRS, densify_pts: int = 5000) -> Vector:
        """
        Get raster footprint projected in a specified CRS.

        The polygon points of the vector are densified during reprojection to warp
        the rectangular square footprint of the original projection into the new one.

        :param out_crs: Output CRS.
        :param densify_pts: Maximum points to be added between image corners to account for non linear edges.
         Reduce if time computation is really critical (ms) or increase if extent is not accurate enough.
        """

        return Vector(
            _get_footprint_projected(bounds=self.bounds, in_crs=self.crs, out_crs=out_crs, densify_pts=densify_pts)
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

        # Check that intersection is not void (changed to NaN instead of empty tuple end 2022)
        if intersection == () or all(math.isnan(i) for i in intersection):
            warnings.warn("Intersection is void")
            return (0.0, 0.0, 0.0, 0.0)

        # if required, ensure the intersection is aligned with self's georeferences
        if match_ref:
            intersection = projtools.align_bounds(self.transform, intersection)

        # mypy raises a type issue, not sure how to address the fact that output of merge_bounds can be ()
        return intersection  # type: ignore

    def show(
        self,
        index: int | tuple[int, ...] | None = None,
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

        :param index: Band to plot, from 1 to self.count (default is all).
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
            myimage.show(ax=ax1, mpl_kws)
        """
        # If data is not loaded, need to load it
        if not self.is_loaded:
            self.load()

        # Check if specific band selected, or take all
        # rshow takes care of image dimensions
        # if self.count=3 (4) => plotted as RGB(A)
        if index is None:
            index = tuple(range(1, self.count + 1))
        elif isinstance(index, int):
            if index > self.count:
                raise ValueError(f"Index must be in range 1-{self.count:d}")
            pass
        else:
            raise ValueError("Index must be int or None")

        # Get data
        if self.count == 1:
            data = self.data
        else:
            data = self.data[np.array(index) - 1, :, :]

        # If multiple bands (RGB), cbar does not make sense
        if isinstance(index, abc.Sequence):
            if len(index) > 1:
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
            vmin = np.nanmin(data)

        if vmax is None:
            vmax = np.nanmax(data)

        # Make sure they are numbers, to avoid mpl error
        try:
            vmin = float(vmin)
            vmax = float(vmax)
        except ValueError:
            raise ValueError("vmin or vmax cannot be converted to float")

        # Create axes
        if ax is None:
            # If no figure exists, get a new axis
            if len(plt.get_fignums()) == 0:
                ax0 = plt.gca()
            # Otherwise, get first axis
            else:
                ax0 = plt.gcf().axes[0]
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
            cax = divider.append_axes("right", size="5%", pad=0.05)
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
            cbar.solids.set_alpha(alpha)

            if cbar_title is not None:
                cbar.set_label(cbar_title)
        else:
            cbar = None

        # If returning axes
        if return_axes:
            return ax, cax
        else:
            return None

    def value_at_coords(
        self,
        x: Number | ArrayLike,
        y: Number | ArrayLike,
        latlon: bool = False,
        index: int | None = None,
        masked: bool = False,
        window: int | None = None,
        reducer_function: Callable[[NDArrayNum], float] = np.ma.mean,
        return_window: bool = False,
        boundless: bool = True,
    ) -> Any:
        """
        Extract raster values at the nearest pixels from the specified coordinates,
        or reduced (e.g., mean of pixels) from a window around the specified coordinates.

        By default, samples pixel value of each band. Can be passed a band index to sample from.

        :param x: X (or longitude) coordinate(s).
        :param y: Y (or latitude) coordinate(s).
        :param latlon: Whether coordinates are provided as longitude-latitude.
        :param index: Band number to extract from (from 1 to self.count).
        :param masked: Whether to return a masked array, or classic array.
        :param window: Window size to read around coordinates. Must be odd.
        :param reducer_function: Reducer function to apply to the values in window (defaults to np.mean).
        :param return_window: Whether to return the windows (in addition to the reduced value).
        :param boundless: Whether to allow windows that extend beyond the extent.

        :returns: When called on a raster or with a specific band set, return value of pixel.
        :returns: If multiple band raster and the band is not specified, a
            dictionary containing the value of the pixel in each band.
        :returns: In addition, if return_window=True, return tuple of (values, arrays)

        :examples:

            >>> self.value_at_coords(-48.125, 67.8901, window=3)  # doctest: +SKIP
            Returns mean of a 3*3 window:
                v v v \
                v c v  | = float(mean)
                v v v /
            (c = provided coordinate, v= value of surrounding coordinate)

        """
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
        if latlon:
            from geoutils import projtools

            x, y = projtools.reproject_from_latlon((y, x), self.crs)  # type: ignore

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
                    data = self.data[slice(None) if index is None else index - 1, row : row + height, col : col + width]
                if not masked:
                    data = data.filled()
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
                        indexes=index,
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

        if return_window:
            return (output_val, output_win)
        else:
            return output_val

    def coords(self, offset: str = "corner", grid: bool = True) -> tuple[NDArrayNum, ...]:
        """
        Get coordinates (x,y) of all pixels in the raster.

        :param offset: coordinate type. If 'corner', returns corner coordinates of pixels.
            If 'center', returns center coordinates. Default is corner.
        :param grid: Return grid

        :returns x,y: Arrays of the (x,y) coordinates.
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
            # Drop the last element
            meshgrid = tuple(np.meshgrid(xx[:-1], np.flip(yy[:-1])))
            return meshgrid
        else:
            return xx[:-1], yy[:-1]

    def xy2ij(
        self,
        x: ArrayLike,
        y: ArrayLike,
        op: type = np.float32,
        precision: float | None = None,
        shift_area_or_point: bool = False,
    ) -> tuple[NDArrayNum, NDArrayNum]:
        """
        Get indexes (row,column) of coordinates (x,y).

        Optionally, user can enforce the interpretation of pixel coordinates in self.tags['AREA_OR_POINT']
        to ensure that the indexes of points represent the right location. See parameter description of
        shift_area_or_point for more details.

        :param x: X coordinates.
        :param y: Y coordinates.
        :param op: Operator to compute index.
        :param precision: Precision passed to :func:`rasterio.transform.rowcol`.
        :param shift_area_or_point: Shifts index to center pixel coordinates if GDAL's AREA_OR_POINT
            attribute (in self.tags) is "Point", keeps the corner pixel coordinate for "Area".

        :returns i, j: Indices of (x,y) in the image.
        """
        # Input checks
        if op not in [np.float32, np.float64, float]:
            raise UserWarning(
                "Operator is not of type float: rio.Dataset.index might "
                "return unreliable indexes due to rounding issues."
            )

        i, j = rio.transform.rowcol(self.transform, x, y, op=op, precision=precision)

        # Necessary because rio.Dataset.index does not return abc.Iterable for a single point
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

        # AREA_OR_POINT GDAL attribute, i.e. does the value refer to the upper left corner "Area" or
        # the center of pixel "Point". This normally has no influence on georeferencing, it's only
        # about the interpretation of the raster values, and thus can affect sub-pixel interpolation,
        # for more details see: https://gdal.org/user/raster_data_model.html#metadata

        # If the user wants to shift according to the interpretation
        if shift_area_or_point:
            # If AREA_OR_POINT attribute does not exist, use the most typical "Area"
            if self.tags.get("AREA_OR_POINT") is not None:
                area_or_point = self.tags.get("AREA_OR_POINT")
                if not isinstance(area_or_point, str):
                    raise TypeError('Attribute self.tags["AREA_OR_POINT"] must be a string.')
                if area_or_point.lower() not in ["area", "point"]:
                    raise ValueError('Attribute self.tags["AREA_OR_POINT"] must be one of "Area" or "Point".')
            else:
                area_or_point = "Area"
                warnings.warn(
                    category=UserWarning,
                    message='Attribute AREA_OR_POINT undefined in self.tags, using "Area" as default (no shift).',
                )

            # Shift by half a pixel if the AREA_OR_POINT attribute is "Point", otherwise leave as is
            if area_or_point.lower() == "point":
                if not isinstance(i.flat[0], (np.floating, float)):
                    raise ValueError(
                        "Operator must return np.floating values to perform area_or_point subpixel index shifting."
                    )

                i += 0.5
                j += 0.5

        # Convert output indexes to integer if they are all whole numbers
        if np.all(np.mod(i, 1) == 0) and np.all(np.mod(j, 1) == 0):
            i = i.astype(int)
            j = j.astype(int)

        return i, j

    def ij2xy(self, i: ArrayLike, j: ArrayLike, offset: str = "ul") -> tuple[NDArrayNum, NDArrayNum]:
        """
        Get coordinates (x,y) of indexes (row,column).

        Defaults to upper-left, for which this function is fully reversible with xy2ij.

        :param i: Row (i) index of pixel.
        :param j: Column (j) index of pixel.
        :param offset: Return coordinates as "center" of pixel, or any corner (upper-left "ul", "ur", "ll", lr").

        :returns x, y: x,y coordinates of i,j in reference system.
        """

        x, y = rio.transform.xy(self.transform, i, j, offset=offset)

        return x, y

    def outside_image(self, xi: ArrayLike, yj: ArrayLike, index: bool = True) -> bool:
        """
        Check whether a given point falls outside the raster.

        :param xi: Indices (or coordinates) of x direction to check.
        :param yj: Indices (or coordinates) of y direction to check.
        :param index: Interpret ij as raster indices (default is ``True``). If False, assumes ij is coordinates.

        :returns is_outside: ``True`` if ij is outside the image.
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
        pts: tuple[list[float], list[float]],
        input_latlon: bool = False,
        mode: str = "linear",
        index: int = 1,
        shift_area_or_point: bool = False,
        **kwargs: Any,
    ) -> NDArrayNum:
        """
         Interpolate raster values at a set of points.

         Optionally, user can enforce the interpretation of pixel coordinates in self.tags['AREA_OR_POINT']
         to ensure that the interpolation of points is done at the right location. See parameter description
         of shift_area_or_point for more details.

        :param pts: Point(s) at which to interpolate raster value. If points fall outside of image, value
            returned is nan. Shape should be (N,2).
        :param input_latlon: Whether the input is in latlon, unregarding of Raster CRS
        :param mode: One of 'linear', 'cubic', or 'quintic'. Determines what type of spline is used to
            interpolate the raster value at each point. For more information, see scipy.interpolate.interp2d.
            Default is linear.
        :param index: The band to use (from 1 to self.count).
        :param shift_area_or_point: Shifts index to center pixel coordinates if GDAL's AREA_OR_POINT
            attribute (in self.tags) is "Point", keeps the corner pixel coordinate for "Area".

        :returns rpts: Array of raster value(s) for the given points.
        """
        assert mode in [
            "mean",
            "linear",
            "cubic",
            "quintic",
            "nearest",
        ], "mode must be mean, linear, cubic, quintic or nearest."

        # Get coordinates
        x, y = list(zip(*pts))

        # If those are in latlon, convert to Raster crs
        if input_latlon:
            init_crs = pyproj.CRS(4326)
            dest_crs = pyproj.CRS(self.crs)
            transformer = pyproj.Transformer.from_crs(init_crs, dest_crs)
            x, y = transformer.transform(x, y)

        i, j = self.xy2ij(x, y, op=np.float32, shift_area_or_point=shift_area_or_point)

        ind_invalid = np.vectorize(lambda k1, k2: self.outside_image(k1, k2, index=True))(j, i)

        if self.count == 1:
            rpts = map_coordinates(self.data.astype(np.float32), [i, j], **kwargs)
        else:
            rpts = map_coordinates(self.data[index - 1, :, :].astype(np.float32), [i, j], **kwargs)

        rpts = np.array(rpts, dtype=np.float32)
        rpts[np.array(ind_invalid)] = np.nan

        return rpts

    def split_bands(self: RasterType, copy: bool = False, subset: list[int] | int | None = None) -> list[Raster]:
        """
        Split the bands into separate rasters.

        :param copy: Copy the bands or return slices of the original data.
        :param subset: Optional. A subset of band indices to extract (from 1 to self.count). Defaults to all.

        :returns: A list of Rasters for each band.
        """
        bands: list[Raster] = []

        if subset is None:
            indices = list(np.arange(1, self.count + 1))
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
                        self.data[band_n - 1, :, :].copy(),
                        transform=self.transform,
                        crs=self.crs,
                        nodata=self.nodata,
                    )
                )
        else:
            for band_n in indices:
                # Set the data to a slice of the original array
                bands.append(
                    self.from_array(
                        self.data[band_n - 1, :, :],
                        transform=self.transform,
                        crs=self.crs,
                        nodata=self.nodata,
                    )
                )

        return bands

    @overload
    def to_points(
        self,
        subset: float | int,
        as_array: Literal[False] = False,
        pixel_offset: Literal["center", "corner"] = "center",
    ) -> NDArrayNum:
        ...

    @overload
    def to_points(
        self,
        subset: float | int,
        as_array: Literal[True],
        pixel_offset: Literal["center", "corner"] = "center",
    ) -> Vector:
        ...

    def to_points(
        self, subset: float | int = 1, as_array: bool = False, pixel_offset: Literal["center", "corner"] = "center"
    ) -> NDArrayNum | Vector:
        """
        Convert raster to points.

        Optionally, randomly subset the raster.

        If 'subset' is either 1 or is equal to the pixel count, all points are returned in order.
        If 'subset' is smaller than 1 (for fractions) or the pixel count, a random sample is returned.

        If the raster is not loaded, sampling will be done from disk without loading the entire Raster.

        Formats:
            * `as_array` == False: A vector with dataframe columns ["b1", "b2", ..., "geometry"],
            * `as_array` == True: A numpy ndarray of shape (N, 2 + count) with the columns [x, y, b1, b2..].

        :param subset: The point count or fraction. If 'subset' > 1, it's parsed as a count.
        :param as_array: Return an array instead of a vector.
        :param pixel_offset: The point at which to associate the pixel coordinate with ('corner' == upper left).

        :raises ValueError: If the subset count or fraction is poorly formatted.

        :returns: A ndarray/GeoDataFrame of the shape (N, 2 + count) where N is the subset count.
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
            if self.count == 1:
                pixel_data = self.data[rows, cols]
            else:
                pixel_data = self.data[:, rows, cols]
        else:
            with rio.open(self.filename) as raster:
                pixel_data = np.array(list(raster.sample(zip(x_coords, y_coords)))).T

        if isinstance(pixel_data, np.ma.masked_array):
            pixel_data = np.where(pixel_data.mask, np.nan, pixel_data.data)

        # Merge the coordinates and pixel data into a point cloud.
        points_arr = np.vstack((x_coords.reshape(1, -1), y_coords.reshape(1, -1), pixel_data)).T

        if not as_array:
            points = Vector(
                gpd.GeoDataFrame(
                    points_arr[:, 2:],
                    columns=[f"b{i}" for i in range(1, self.count + 1)],
                    geometry=gpd.points_from_xy(points_arr[:, 0], points_arr[:, 1]),
                    crs=self.crs,
                )
            )
            return points
        else:
            return points_arr

    def polygonize(
        self, target_values: Number | tuple[Number, Number] | list[Number] | NDArrayNum | Literal["all"] = "all"
    ) -> Vector:
        """
        Polygonize the raster into a vector.

        :param target_values: Value or range of values of the raster from which to
          create geometries (defaults to 'all', for which all unique pixel values of the raster are used).

        :returns: Vector containing the polygonized geometries.
        """

        # Mask a unique value set by a number
        if isinstance(target_values, (int, float, np.integer, np.floating)):
            if np.sum(self.data == target_values) == 0:
                raise ValueError(f"no pixel with in_value {target_values}")

            bool_msk = np.array(self.data == target_values).astype(np.uint8)

        # Mask values within boundaries set by a tuple
        elif isinstance(target_values, tuple):
            if np.sum((self.data > target_values[0]) & (self.data < target_values[1])) == 0:
                raise ValueError(f"no pixel with in_value between {target_values[0]} and {target_values[1]}")

            bool_msk = ((self.data > target_values[0]) & (self.data < target_values[1])).astype(np.uint8)

        # Mask specific values set by a sequence
        elif isinstance(target_values, list) or isinstance(target_values, np.ndarray):
            if np.sum(np.isin(self.data, np.array(target_values))) == 0:
                raise ValueError("no pixel with in_value " + ", ".join(map("{}".format, target_values)))

            bool_msk = np.isin(self.data, np.array(target_values)).astype("uint8")

        # Mask all valid values
        elif target_values == "all":
            bool_msk = (~self.data.mask).astype("uint8")

        else:
            raise ValueError("in_value must be a number, a tuple or a sequence")

        # GeoPandas.from_features() only supports certain dtypes, we find the best common dtype to optimize memory usage
        # TODO: this should be a function independent of polygonize, reused in several places
        gpd_dtypes = ["uint8", "uint16", "int16", "int32", "float32"]
        list_common_dtype_index = []
        for gpd_type in gpd_dtypes:
            polygonize_dtype = np.promote_types(gpd_type, self.dtypes[0])
            if str(polygonize_dtype) in gpd_dtypes:
                list_common_dtype_index.append(gpd_dtypes.index(gpd_type))
        if len(list_common_dtype_index) == 0:
            final_dtype = "float32"
        else:
            final_dtype_index = min(list_common_dtype_index)
            final_dtype = gpd_dtypes[final_dtype_index]

        results = (
            {"properties": {"raster_value": v}, "geometry": s}
            for i, (s, v) in enumerate(shapes(self.data.astype(final_dtype), mask=bool_msk, transform=self.transform))
        )

        gdf = gpd.GeoDataFrame.from_features(list(results))
        gdf.insert(0, "New_ID", range(0, 0 + len(gdf)))
        gdf.set_geometry(col="geometry", inplace=True)
        gdf.set_crs(self.crs, inplace=True)

        return gv.Vector(gdf)

    def proximity(
        self,
        vector: Vector | None = None,
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

        :param vector: Vector for which to compute the proximity to geometry,
            if not provided computed on this raster target pixels.
        :param target_values: (Only with raster) List of target values to use for the proximity,
            defaults to all non-zero values.
        :param geometry_type: (Only with a vector) Type of geometry to use for the proximity, defaults to 'boundary'.
        :param in_or_out: (Only with a vector) Compute proximity only 'in' or 'out'-side the geometry, or 'both'.
        :param distance_unit: Distance unit, either 'georeferenced' or 'pixel'.

        :return: Proximity distances raster.
        """

        proximity = proximity_from_vector_or_raster(
            raster=self,
            vector=vector,
            target_values=target_values,
            geometry_type=geometry_type,
            in_or_out=in_or_out,
            distance_unit=distance_unit,
        )

        return self.copy(new_array=proximity)

    @overload
    def subsample(
        self,
        subsample: int | float,
        return_indices: Literal[False] = False,
        *,
        random_state: np.random.RandomState | int | None = None,
    ) -> NDArrayNum:
        ...

    @overload
    def subsample(
        self,
        subsample: int | float,
        return_indices: Literal[True],
        *,
        random_state: np.random.RandomState | int | None = None,
    ) -> tuple[NDArrayNum, ...]:
        ...

    @overload
    def subsample(
        self,
        subsample: float | int,
        return_indices: bool = False,
        random_state: np.random.RandomState | int | None = None,
    ) -> NDArrayNum | tuple[NDArrayNum, ...]:
        ...

    def subsample(
        self,
        subsample: float | int,
        return_indices: bool = False,
        random_state: np.random.RandomState | int | None = None,
    ) -> NDArrayNum | tuple[NDArrayNum, ...]:
        """
        Randomly subsample the raster. Only valid values are considered.

        :param subsample: If <= 1, a fraction of the total pixels to extract. If > 1, the number of pixels.
        :param return_indices: Whether to return the extracted indices only.
        :param random_state: Random state or seed number.

        :return: Array of subsampled valid values, or array of subsampled indices.
        """

        return subsample_array(
            array=self.data, subsample=subsample, return_indices=return_indices, random_state=random_state
        )


class Mask(Raster):
    """
    The georeferenced mask.

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

    def __init__(
        self,
        filename_or_dataset: str | RasterType | rio.io.DatasetReader | rio.io.MemoryFile | dict[str, Any],
        **kwargs: Any,
    ) -> None:

        self._data: MArrayNum | MArrayBool | None = None  # type: ignore

        # If a Mask is passed, simply point back to Mask
        if isinstance(filename_or_dataset, Mask):
            for key in filename_or_dataset.__dict__:
                setattr(self, key, filename_or_dataset.__dict__[key])
            return
        # Else rely on parent Raster class options (including raised errors)
        else:
            super().__init__(filename_or_dataset, **kwargs)

            # If nbands larger than one, use only first band and raise a warning
            if self.count > 1:
                warnings.warn(
                    category=UserWarning,
                    message="Multi-band raster provided to create a Mask, only the first band will be used.",
                )
                self._data = self.data[0, :, :]

            # Convert masked array to boolean
            self._data = self.data.astype(bool)  # type: ignore

            # Fix nodata to None
            self._nodata = None

            # Define in dtypes
            self._dtypes = (bool,)

    def __repr__(self) -> str:
        """Convert mask to string representation."""

        # If data not loaded, return and string and avoid calling .data
        if not self.is_loaded:
            str_data = "not_loaded"
        else:
            str_data = "\n       ".join(self.data.__str__().split("\n"))

        # Over-ride Raster's method to remove nodata value (always None)
        s = (
            "Mask(\n"
            + "  data="
            + str_data
            + "\n  transform="
            + "\n            ".join(self.transform.__str__().split("\n"))
            + "\n  crs="
            + self.crs.__str__()
            + ")"
        )

        return str(s)

    def _repr_html_(self) -> str:
        """Convert mask to HTML representation for notebooks and doc"""

        # If data not loaded, return and string and avoid calling .data
        if not self.is_loaded:
            str_data = "<i>not_loaded</i>"
        else:
            str_data = "\n       ".join(self.data.__str__().split("\n"))

        # Over-ride Raster's method to remove nodata value (always None)
        # Use <pre> to keep white spaces, <span> to keep line breaks
        s = (
            '<pre><span style="white-space: pre-wrap"><b><em>Mask</em></b>(\n'
            + "  <b>data=</b>"
            + str_data
            + "\n  <b>transform=</b>"
            + "\n            ".join(self.transform.__str__().split("\n"))
            + "\n  <b>crs=</b>"
            + self.crs.__str__()
            + ")</span></pre>"
        )

        return str(s)

    def reproject(
        self: Mask,
        dst_ref: RasterType | str | None = None,
        dst_crs: CRS | str | int | None = None,
        dst_size: tuple[int, int] | None = None,
        dst_bounds: dict[str, float] | rio.coords.BoundingBox | None = None,
        dst_res: float | abc.Iterable[float] | None = None,
        dst_nodata: int | float | tuple[int, ...] | tuple[float, ...] | None = None,
        src_nodata: int | float | tuple[int, ...] | tuple[float, ...] | None = None,
        dst_dtype: DTypeLike | None = None,
        resampling: Resampling | str = Resampling.nearest,
        silent: bool = False,
        n_threads: int = 0,
        memory_limit: int = 64,
    ) -> Mask:
        # Depending on resampling, adjust to rasterio supported types
        if resampling in [Resampling.nearest, "nearest"]:
            self._data = self.data.astype("uint8")  # type: ignore
        else:
            warnings.warn(
                "Reprojecting a mask with a resampling method other than 'nearest', "
                "the boolean array will be converted to float during interpolation."
            )
            self._data = self.data.astype("float32")  # type: ignore

        # Call Raster.reproject()
        output = super().reproject(
            dst_ref=dst_ref,  # type: ignore
            dst_crs=dst_crs,
            dst_size=dst_size,
            dst_bounds=dst_bounds,
            dst_res=dst_res,
            dst_nodata=dst_nodata,
            src_nodata=src_nodata,
            dst_dtype=dst_dtype,
            resampling=resampling,
            silent=silent,
            n_threads=n_threads,
            memory_limit=memory_limit,
        )

        # Transform back to a boolean array
        output._data = output.data.astype(bool)  # type: ignore

        return output

    # Note the star is needed because of the default argument 'mode' preceding non default arg 'inplace'
    # Then the final overload must be duplicated
    @overload
    def crop(
        self: Mask,
        crop_geom: Mask | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        *,
        inplace: Literal[True],
    ) -> None:
        ...

    @overload
    def crop(
        self: Mask,
        crop_geom: Mask | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        *,
        inplace: Literal[False],
    ) -> Mask:
        ...

    @overload
    def crop(
        self: Mask,
        crop_geom: Mask | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        inplace: bool = True,
    ) -> Mask | None:
        ...

    def crop(
        self: Mask,
        crop_geom: Mask | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        inplace: bool = True,
    ) -> Mask | None:
        # If there is resampling involved during cropping, encapsulate type as in reproject()
        if mode == "match_extent":
            raise ValueError(NotImplementedError)
            # self._data = self.data.astype("float32")
            # if inplace:
            #     super().crop(crop_geom=crop_geom, mode=mode, inplace=inplace)
            #     self._data = self.data.astype(bool)
            #     return None
            # else:
            #     output = super().crop(crop_geom=crop_geom, mode=mode, inplace=inplace)
            #     output._data = output.data.astype(bool)
            #     return output
        # Otherwise, run a classic crop
        else:
            if not inplace:
                return super().crop(crop_geom=crop_geom, mode=mode, inplace=inplace)
            else:
                super().crop(crop_geom=crop_geom, mode=mode, inplace=inplace)
                return None

    def polygonize(
        self, target_values: Number | tuple[Number, Number] | list[Number] | NDArrayNum | Literal["all"] = 1
    ) -> Vector:
        # If target values is passed but does not correspond to 0 or 1, raise a warning
        if not isinstance(target_values, (int, np.integer, float, np.floating)) or target_values not in [0, 1]:
            warnings.warn("In-value converted to 1 for polygonizing boolean mask.")
            target_values = 1

        # Convert to unsigned integer and pass to parent method
        self._data = self.data.astype("uint8")  # type: ignore
        return super().polygonize(target_values=target_values)

    def proximity(
        self,
        vector: Vector | None = None,
        target_values: list[float] | None = None,
        geometry_type: str = "boundary",
        in_or_out: Literal["in"] | Literal["out"] | Literal["both"] = "both",
        distance_unit: Literal["pixel"] | Literal["georeferenced"] = "georeferenced",
    ) -> Raster:
        # By default, target True values of the mask
        if vector is None and target_values is None:
            target_values = [1]

        # TODO: Adapt target_value into int | list in Raster.proximity
        # If target values is passed but does not correspond to 0 or 1, raise a warning
        # if target_values is not None and not isinstance(target_values, (int, np.integer,
        # float, np.floating)) or target_values not in [0, 1]:
        #     warnings.warn("In-value converted to 1 for polygonizing boolean mask.")
        #     target_values = [1]

        # Convert to unsigned integer and pass to parent method
        self._data = self.data.astype("uint8")  # type: ignore

        # Need to cast output to Raster before computing proximity, as output will not be boolean
        # (super() would instantiate Mask() again)
        raster = Raster({"data": self.data, "transform": self.transform, "crs": self.crs, "nodata": self.nodata})
        return raster.proximity(
            vector=vector,
            target_values=target_values,
            geometry_type=geometry_type,
            in_or_out=in_or_out,
            distance_unit=distance_unit,
        )

    def __and__(self: Mask, other: Mask | NDArrayBool) -> Mask:
        """Bitwise and between masks, or a mask and an array."""
        self_data, other_data = self._overloading_check(other)[0:2]  # type: ignore

        return self.from_array(
            data=(self_data & other_data), transform=self.transform, crs=self.crs, nodata=self.nodata  # type: ignore
        )

    def __rand__(self: Mask, other: Mask | NDArrayBool) -> Mask:
        """Bitwise and between masks, or a mask and an array."""

        return self.__and__(other)

    def __or__(self: Mask, other: Mask | NDArrayBool) -> Mask:
        """Bitwise or between masks, or a mask and an array."""

        self_data, other_data = self._overloading_check(other)[0:2]  # type: ignore

        return self.from_array(
            data=(self_data | other_data), transform=self.transform, crs=self.crs, nodata=self.nodata  # type: ignore
        )

    def __ror__(self: Mask, other: Mask | NDArrayBool) -> Mask:
        """Bitwise or between masks, or a mask and an array."""

        return self.__or__(other)

    def __xor__(self: Mask, other: Mask | NDArrayBool) -> Mask:
        """Bitwise xor between masks, or a mask and an array."""

        self_data, other_data = self._overloading_check(other)[0:2]  # type: ignore

        return self.from_array(
            data=(self_data ^ other_data), transform=self.transform, crs=self.crs, nodata=self.nodata  # type: ignore
        )

    def __rxor__(self: Mask, other: Mask | NDArrayBool) -> Mask:
        """Bitwise xor between masks, or a mask and an array."""

        return self.__xor__(other)

    def __invert__(self: Mask) -> Mask:
        """Bitwise inversion of a mask."""

        return self.from_array(data=~self.data, transform=self.transform, crs=self.crs, nodata=self.nodata)


# -----------------------------------------
# Additional stand-alone utility functions
# -----------------------------------------


def proximity_from_vector_or_raster(
    raster: Raster,
    vector: Vector | None = None,
    target_values: list[float] | None = None,
    geometry_type: str = "boundary",
    in_or_out: Literal["in"] | Literal["out"] | Literal["both"] = "both",
    distance_unit: Literal["pixel"] | Literal["georeferenced"] = "georeferenced",
) -> NDArrayNum:
    """
    (This function is defined here as mostly raster-based, but used in a class method for both Raster and Vector)
    Proximity to a Raster's target values if no Vector is provided, otherwise to a Vector's geometry type
    rasterized on the Raster.

    :param raster: Raster to burn the proximity grid on.
    :param vector: Vector for which to compute the proximity to geometry,
        if not provided computed on the Raster target pixels.
    :param target_values: (Only with a Raster) List of target values to use for the proximity,
        defaults to all non-zero values.
    :param geometry_type: (Only with a Vector) Type of geometry to use for the proximity, defaults to 'boundary'.
    :param in_or_out: (Only with a Vector) Compute proximity only 'in' or 'out'-side the geometry, or 'both'.
    :param distance_unit: Distance unit, either 'georeferenced' or 'pixel'.
    """

    # 1/ First, if there is a vector input, we rasterize the geometry type
    # (works with .boundary that is a LineString (.exterior exists, but is a LinearRing)
    if vector is not None:
        # We create a geodataframe with the geometry type
        boundary_shp = gpd.GeoDataFrame(geometry=vector.ds.__getattr__(geometry_type), crs=vector.crs)
        # We mask the pixels that make up the geometry type
        mask_boundary = Vector(boundary_shp).create_mask(raster, as_array=True)

    else:
        # We mask target pixels
        if target_values is not None:
            mask_boundary = np.logical_or.reduce([raster.get_nanarray() == target_val for target_val in target_values])
        # Otherwise, all non-zero values are considered targets
        else:
            mask_boundary = raster.get_nanarray().astype(bool)

    # 2/ Now, we compute the distance matrix relative to the masked geometry type
    if distance_unit.lower() == "georeferenced":
        sampling: int | tuple[float | int, float | int] = raster.res
    elif distance_unit.lower() == "pixel":
        sampling = 1
    else:
        raise ValueError('Distance unit must be either "georeferenced" or "pixel".')

    # If not all pixels are targets, then we compute the distance
    non_targets = np.count_nonzero(mask_boundary)
    if non_targets > 0:
        proximity = distance_transform_edt(~mask_boundary, sampling=sampling)
    # Otherwise, pass an array full of nodata
    else:
        proximity = np.ones(np.shape(mask_boundary)) * np.nan

    # 3/ If there was a vector input, apply the in_and_out argument to optionally mask inside/outside
    if vector is not None:
        if in_or_out == "both":
            pass
        elif in_or_out in ["in", "out"]:
            mask_polygon = Vector(vector.ds).create_mask(raster, as_array=True)
            if in_or_out == "in":
                proximity[~mask_polygon] = 0
            else:
                proximity[mask_polygon] = 0
        else:
            raise ValueError('The type of proximity must be one of "in", "out" or "both".')

    return proximity
