"""
Module for Raster class.
"""

from __future__ import annotations

import pathlib
import warnings
from collections import abc
from contextlib import ExitStack
from typing import IO, Any, Callable, overload

import affine
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import rasterio.windows
import rioxarray
import xarray as xr
from affine import Affine
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.plot import show as rshow

from geoutils._typing import (
    DTypeLike,
    MArrayBool,
    MArrayNum,
    NDArrayBool,
    NDArrayNum,
    Number,
)
from geoutils.raster.base import RasterBase, RasterType
from geoutils.raster.georeferencing import (
    _cast_nodata,
    _cast_pixel_interpretation,
    _default_nodata,
)
from geoutils.vector.vector import Vector

# If python38 or above, Literal is builtin. Otherwise, use typing_extensions
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

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
]


def _load_rio(
    dataset: rio.io.DatasetReader,
    only_mask: bool = False,
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
        if only_mask:
            data = dataset.read_masks(window=window, **kwargs)
        else:
            data = dataset.read(masked=masked, window=window, **kwargs)
    else:
        if only_mask:
            data = dataset.read_masks(indexes=indexes, window=window, **kwargs)
        else:
            data = dataset.read(indexes=indexes, masked=masked, window=window, **kwargs)
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
        out_nodata = nodata1 if not None else nodata2

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

    def __init__(
        self,
        filename_or_dataset: (
            str | pathlib.Path | RasterType | rio.io.DatasetReader | rio.io.MemoryFile | dict[str, Any]
        ),
        bands: int | list[int] | None = None,
        load_data: bool = False,
        downsample: Number = 1,
        force_nodata: int | float | None = None,
    ) -> None:
        """
        Instantiate a raster from a filename or rasterio dataset.

        :param filename_or_dataset: Path to file or Rasterio dataset.

        :param bands: Band(s) to load into the object. Default loads all bands.

        :param load_data: Whether to load the array during instantiation. Default is False.

        :param downsample: Downsample the array once loaded by a round factor. Default is no downsampling.

        :param force_nodata: Force nodata value to be used (overwrites the metadata). Default reads from metadata.
        """

        super().__init__()

        self._data: MArrayNum | None = None
        self._nodata = force_nodata
        self._bands = bands
        self._masked = True

        # This is for Raster.from_array to work.
        if isinstance(filename_or_dataset, dict):

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
                if key in ["data", "transform", "crs", "nodata", "area_or_point"]:
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

        # If the data was transformed into boolean, re-initialize as a Mask subclass
        # Typing: we can specify this behaviour in @overload once we add the NumPy plugin of MyPy
        if data.dtype == bool:
            return Mask(
                {
                    "data": data,
                    "transform": transform,
                    "crs": crs,
                    "nodata": nodata,
                    "area_or_point": area_or_point,
                    "tags": tags,
                }
            )  # type: ignore
        # Otherwise, keep as a given RasterType subclass
        else:
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

    def __getitem__(self, index: Mask | NDArrayBool | Any) -> NDArrayBool | Raster:
        """
        Index the raster.

        In addition to all index types supported by NumPy, also supports a mask of same georeferencing or a
        boolean array of the same shape as the raster.
        """

        if isinstance(index, (Mask, np.ndarray)):
            _cast_numeric_array_raster(self, index, operation_name="an indexing operation")  # type: ignore

        # If input is Mask with the same shape and georeferencing
        if isinstance(index, Mask):
            if self.count == 1:
                return self.data[index.data.squeeze()]
            else:
                return self.data[:, index.data.squeeze()]
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

    def __setitem__(self, index: Mask | NDArrayBool | Any, assign: NDArrayNum | Number) -> None:
        """
        Perform index assignment on the raster.

        In addition to all index types supported by NumPy, also supports a mask of same georeferencing or a
        boolean array of the same shape as the raster.
        """

        # First, check index
        if isinstance(index, (Mask, np.ndarray)):
            _cast_numeric_array_raster(self, index, operation_name="an index assignment operation")  # type: ignore

        # If input is Mask with the same shape and georeferencing
        if isinstance(index, Mask):
            ind = index.data.data
            use_all_bands = False
        # If input is array with the same shape
        elif isinstance(index, np.ndarray):
            if str(index.dtype) != "bool":
                ind = index.astype(bool)
                warnings.warn(message="Input array was cast to boolean for indexing.", category=UserWarning)
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
        out_data = self_data % other_data
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

    @overload
    def get_nanarray(self, return_mask: Literal[False] = False) -> NDArrayNum: ...

    @overload
    def get_nanarray(self, return_mask: Literal[True]) -> tuple[NDArrayNum, NDArrayBool]: ...

    def get_nanarray(self, return_mask: bool = False) -> NDArrayNum | tuple[NDArrayNum, NDArrayBool]:
        """
        Get NaN array from the raster.

        Optionally, return the mask from the masked array.

        :param return_mask: Whether to return the mask of valid data.

        :returns Array with masked data as NaNs, (Optional) Mask of invalid data.
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
                    return (
                        self.from_array(
                            data=output, transform=self.transform, crs=self.crs, nodata=self.nodata, area_or_point=aop
                        )
                        for output in outputs
                    )
                else:
                    return (self.copy(new_array=output) for output in outputs)
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
            kwargs.update({"interpolation": "None"})

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
        else:
            return None

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

            # Force dtypes
            self._dtypes = (bool,)

            # Fix nodata to None
            self._nodata = None

            # Convert masked array to boolean
            self._data = self.data.astype(bool)  # type: ignore

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

    @overload
    def reproject(
        self: Mask,
        ref: RasterType | str | None = None,
        crs: CRS | str | int | None = None,
        res: float | abc.Iterable[float] | None = None,
        grid_size: tuple[int, int] | None = None,
        bounds: dict[str, float] | rio.coords.BoundingBox | None = None,
        nodata: int | float | None = None,
        dtype: DTypeLike | None = None,
        resampling: Resampling | str = Resampling.nearest,
        force_source_nodata: int | float | None = None,
        *,
        inplace: Literal[False] = False,
        silent: bool = False,
        n_threads: int = 0,
        memory_limit: int = 64,
    ) -> Mask: ...

    @overload
    def reproject(
        self: Mask,
        ref: RasterType | str | None = None,
        crs: CRS | str | int | None = None,
        res: float | abc.Iterable[float] | None = None,
        grid_size: tuple[int, int] | None = None,
        bounds: dict[str, float] | rio.coords.BoundingBox | None = None,
        nodata: int | float | None = None,
        dtype: DTypeLike | None = None,
        resampling: Resampling | str = Resampling.nearest,
        force_source_nodata: int | float | None = None,
        *,
        inplace: Literal[True],
        silent: bool = False,
        n_threads: int = 0,
        memory_limit: int = 64,
    ) -> None: ...

    @overload
    def reproject(
        self: Mask,
        ref: RasterType | str | None = None,
        crs: CRS | str | int | None = None,
        res: float | abc.Iterable[float] | None = None,
        grid_size: tuple[int, int] | None = None,
        bounds: dict[str, float] | rio.coords.BoundingBox | None = None,
        nodata: int | float | None = None,
        dtype: DTypeLike | None = None,
        resampling: Resampling | str = Resampling.nearest,
        force_source_nodata: int | float | None = None,
        *,
        inplace: bool = False,
        silent: bool = False,
        n_threads: int = 0,
        memory_limit: int = 64,
    ) -> Mask | None: ...

    def reproject(
        self: Mask,
        ref: RasterType | str | None = None,
        crs: CRS | str | int | None = None,
        res: float | abc.Iterable[float] | None = None,
        grid_size: tuple[int, int] | None = None,
        bounds: dict[str, float] | rio.coords.BoundingBox | None = None,
        nodata: int | float | None = None,
        dtype: DTypeLike | None = None,
        resampling: Resampling | str = Resampling.nearest,
        force_source_nodata: int | float | None = None,
        inplace: bool = False,
        silent: bool = False,
        n_threads: int = 0,
        memory_limit: int = 64,
    ) -> Mask | None:
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
            ref=ref,  # type: ignore
            crs=crs,
            res=res,
            grid_size=grid_size,
            bounds=bounds,
            nodata=nodata,
            dtype=dtype,
            resampling=resampling,
            inplace=False,
            force_source_nodata=force_source_nodata,
            silent=silent,
            n_threads=n_threads,
            memory_limit=memory_limit,
        )

        # Transform output back to a boolean array
        output._data = output.data.astype(bool)  # type: ignore

        # Transform self back to boolean array
        self._data = self.data.astype(bool)  # type: ignore

        if inplace:
            self._transform = output._transform  # type: ignore
            self._crs = output._crs  # type: ignore
            # Little trick to force the shape, same as in Raster.reproject()
            self._data = output._data  # type: ignore
            self.data = output._data  # type: ignore
            return None
        else:
            return output

    # Note the star is needed because of the default argument 'mode' preceding non default arg 'inplace'
    # Then the final overload must be duplicated
    @overload
    def crop(
        self: Mask,
        crop_geom: Mask | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        *,
        inplace: Literal[False] = False,
    ) -> Mask: ...

    @overload
    def crop(
        self: Mask,
        crop_geom: Mask | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        *,
        inplace: Literal[True],
    ) -> None: ...

    @overload
    def crop(
        self: Mask,
        crop_geom: Mask | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        *,
        inplace: bool = False,
    ) -> Mask | None: ...

    def crop(
        self: Mask,
        crop_geom: Mask | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        *,
        inplace: bool = False,
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
            return super().crop(crop_geom=crop_geom, mode=mode, inplace=inplace)

    def polygonize(
        self,
        target_values: Number | tuple[Number, Number] | list[Number] | NDArrayNum | Literal["all"] = 1,
        data_column_name: str = "id",
    ) -> Vector:
        # If target values is passed but does not correspond to 0 or 1, raise a warning
        if not isinstance(target_values, (int, np.integer, float, np.floating)) or target_values not in [0, 1]:
            warnings.warn("In-value converted to 1 for polygonizing boolean mask.")
            target_values = 1

        # Convert to unsigned integer and pass to parent method
        self._data = self.data.astype("uint8")  # type: ignore

        # Get output from parent method
        output = super().polygonize(target_values=target_values)

        # Convert array back to boolean
        self._data = self.data.astype(bool)  # type: ignore

        return output

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

        # Need to cast output to Raster before computing proximity, as output will not be boolean
        # (super() would instantiate Mask() again)
        raster = Raster(
            {
                "data": self.data.astype("uint8"),
                "transform": self.transform,
                "crs": self.crs,
                "nodata": self.nodata,
                "area_or_point": self.area_or_point,
            }
        )
        return raster.proximity(
            vector=vector,
            target_values=target_values,
            geometry_type=geometry_type,
            in_or_out=in_or_out,
            distance_unit=distance_unit,
        )

    def __and__(self: Mask, other: Mask | NDArrayBool) -> Mask:
        """Bitwise and between masks, or a mask and an array."""
        self_data, other_data = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"  # type: ignore
        )[0:2]

        return self.copy(self_data & other_data)  # type: ignore

    def __rand__(self: Mask, other: Mask | NDArrayBool) -> Mask:
        """Bitwise and between masks, or a mask and an array."""

        return self.__and__(other)

    def __or__(self: Mask, other: Mask | NDArrayBool) -> Mask:
        """Bitwise or between masks, or a mask and an array."""

        self_data, other_data = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"  # type: ignore
        )[0:2]

        return self.copy(self_data | other_data)  # type: ignore

    def __ror__(self: Mask, other: Mask | NDArrayBool) -> Mask:
        """Bitwise or between masks, or a mask and an array."""

        return self.__or__(other)

    def __xor__(self: Mask, other: Mask | NDArrayBool) -> Mask:
        """Bitwise xor between masks, or a mask and an array."""

        self_data, other_data = _cast_numeric_array_raster(
            self, other, operation_name="an arithmetic operation"  # type: ignore
        )[0:2]

        return self.copy(self_data ^ other_data)  # type: ignore

    def __rxor__(self: Mask, other: Mask | NDArrayBool) -> Mask:
        """Bitwise xor between masks, or a mask and an array."""

        return self.__xor__(other)

    def __invert__(self: Mask) -> Mask:
        """Bitwise inversion of a mask."""

        return self.copy(~self.data)
