# Copyright (c) 2026 GeoUtils developers
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
Module for the Xarray accessor "rst" mirroring the API of the Raster class.
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import rasterio
import rioxarray as rioxr
import xarray as xr
from affine import Affine
from rasterio.crs import CRS
from rioxarray.exceptions import MissingSpatialDimensionError
from rioxarray.rioxarray import affine_to_coords

from geoutils._dispatch import is_dask_array
from geoutils._misc import _validate_downsample, import_optional
from geoutils._typing import DTypeLike, MArrayNum, NDArrayBool, NDArrayNum, Number
from geoutils._xarray import _rebuild_dataset, _select_dataset_variables
from geoutils.raster.base import RasterBase
from geoutils.raster.transformation import (
    _open_downsampled_raster,
    _overview_level_for_downsample,
)

if TYPE_CHECKING:
    from geoutils.stats.variography import Variogram


# Opening raster files


def open_raster(
    filename: str,
    is_mask: bool = False,
    downsample: Number = 1,
    *,
    as_type: Literal["dataarray", "dataset"] = "dataarray",
    data_name: str | None = None,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Open a raster as a Xarray DataArray or Dataset.

    This function relies directly on rioxarray.open_rasterio(), to which keyword arguments are passed.

    The raster array is always masked (i.e. forced to floating with NaNs) and squeezed (single-band dimension
    removed).

    Use ``downsample`` to open a smaller raster by sampling rows and columns at regular intervals. For example,
    ``downsample=2`` keeps one row and one column out of every two, returning about one quarter of the source pixels.

    To open an overview (reduced copy) already stored in the file, pass ``overview_level`` through ``kwargs`` passed
    to Rioxarray. Level 0 selects the first stored overview, level 1 the second, and so on. The returned array uses
    that overview's shape and resolution. See Rioxarray's `Cloud Optimized GeoTIFF example
    <https://corteva.github.io/rioxarray/stable/examples/COG.html>`_.

    Overviews are also used automatically with ``downsample``: GeoUtils reads the closest suitable overview and
    resamples it to the requested factor, or reads the original raster if no suitable overview exists. ``downsample``
    and ``overview_level`` cannot be combined because one selects an overview automatically and the other explicitly.
    See `Rasterio's overview documentation <https://rasterio.readthedocs.io/en/stable/topics/overviews.html>`_ for how
    reduced reads use stored overviews.

    :param filename: Path to the raster file to open.
    :param is_mask: Whether to return the raster values as booleans.
    :param downsample: Downsampling factor (e.g., 2 selects one out of two pixels for every row/column). Rows or
        columns that do not fill a complete interval are omitted. Default 1 keeps the native resolution.
    :param as_type: Returned representation: dataarray (the default) or dataset. A Dataset contains one raster
        variable, with multiple bands on its band dimension.
    :param data_name: Name of the DataArray or Dataset variable. Defaults to the source array name, or raster for
        an unnamed array returned as a Dataset.
    :param kwargs: Keyword arguments passed to :func:`rioxarray.open_rasterio`.
    :returns: The opened raster as a DataArray or Dataset with Rioxarray georeferencing.
    """

    downsample = _validate_downsample(downsample)
    if as_type not in ("dataarray", "dataset"):
        raise ValueError("Argument 'as_type' must be 'dataarray' or 'dataset'.")
    if downsample > 1 and kwargs.get("overview_level") is not None:
        raise ValueError("downsample and overview_level cannot be used together.")

    # Open the native grid directly, or expose an exact reduced grid through a lazy GDAL virtual raster
    if downsample == 1:
        ds = rioxr.open_rasterio(filename, masked=True, **kwargs)
    else:
        with rasterio.open(filename) as source:
            overview_level = _overview_level_for_downsample(source, downsample)
            with _open_downsampled_raster(source, downsample) as vrt:
                if overview_level is not None:
                    kwargs["overview_level"] = overview_level
                ds = rioxr.open_rasterio(vrt, masked=True, **kwargs)

    # Remove the band dimension if there is only one
    if ds.sizes.get("band") == 1:
        ds = ds.squeeze("band")  # Delete band coordinate (only one dimension)

    # If input needs to be interpreted as a boolean mask
    if is_mask:
        ds = ds.astype(bool)

    # Name the raster and wrap its coordinates and lazy data in one Dataset variable when requested
    if data_name is not None:
        ds = ds.rename(data_name)
    if as_type == "dataset":
        return ds.to_dataset(name=ds.name or "raster")
    return ds


# Writing raster files


def _write_raster(xarray_obj: xr.DataArray | xr.Dataset, *args: Any, **kwargs: Any) -> Any:
    """Write Xarray rasters with a scheduler lock when their values are chunked."""

    # Serialize chunk writes across threads or worker processes without gathering the complete raster
    # Dataset variables can have different chunks, so inspect each array independently
    variables = xarray_obj.data_vars.values() if isinstance(xarray_obj, xr.Dataset) else (xarray_obj,)
    if "lock" not in kwargs and any(variable.chunks for variable in variables):
        import_optional("dask")
        from dask.utils import get_scheduler_lock

        kwargs["lock"] = get_scheduler_lock(xarray_obj)

    return xarray_obj.rio.to_raster(*args, **kwargs)


# DataArray accessor


@xr.register_dataarray_accessor("rst")
class DataArrayRasterAccessor(RasterBase):
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
            encoded_nodata = self._obj.rio.encoded_nodata

            # Move encoded nodata to the attributes without retaining a duplicate value
            encoding = dict(self._obj.encoding)
            encoding.pop("_FillValue", None)
            self._obj.rio.set_encoding(encoding, inplace=True)
            self._obj.rio.write_nodata(encoded_nodata, inplace=True)

    @property
    def data(self) -> xr.DataArray:
        # Overloads abstract method in RasterBase
        return self._obj.data

    @data.setter
    def data(self, new_data: xr.DataArray) -> None:
        self._obj.data = new_data

    @property
    def _chunks(self) -> tuple[tuple[int, ...], ...] | None:
        return self._obj.chunks

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
        # Derive coordinate from new transform
        coords = affine_to_coords(affine=new_transform, width=self._obj.sizes["x"], height=self._obj.sizes["y"])
        # This is lazy (doesn't load data), while calling the arrays directly is
        self._obj = self._obj.assign_coords({"x": coords["x"], "y": coords["y"]})

        # Need to call write transform now
        self._obj.rio.write_transform(new_transform, inplace=True)

    def _set_crs(self, new_crs: CRS) -> None:
        # Overloads abstract method in RasterBase
        self._obj.rio.write_crs(new_crs, inplace=True)

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
        :param cast_nodata: Unused; accepted for compatibility with Raster.copy().


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

        # Xarray converts NumPy masked arrays to floating arrays with NaN, even when no values are masked. Preserve the
        # original dtype when possible, and only materialize masked values when they exist
        if np.ma.isMaskedArray(data):
            masked = np.ma.asarray(data)
            mask = np.ma.getmaskarray(masked)
            if mask.any():
                if np.issubdtype(masked.dtype, np.floating):
                    data = masked.filled(np.nan)
                else:
                    data = masked.astype(np.float32).filled(np.nan)
            else:
                data = np.ma.getdata(masked)

        # Remove only a singleton band axis so one row or column still defines a raster grid
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]

        # Rotated grids need two-dimensional map coordinates alongside their row and column axes
        if data.ndim in (2, 3):
            spatial_coords: dict[str, Any] = affine_to_coords(
                affine=transform, width=data.shape[-1], height=data.shape[-2]
            )
            if np.ndim(spatial_coords["x"]) == 2:
                spatial_coords = {
                    "x": np.arange(data.shape[-1]),
                    "y": np.arange(data.shape[-2]),
                    "xc": (("y", "x"), spatial_coords["x"]),
                    "yc": (("y", "x"), spatial_coords["y"]),
                }

        # For a 2-d array
        if data.ndim == 2:
            out_ds = xr.DataArray(
                data=data,
                dims=("y", "x"),
                coords=spatial_coords,
                attrs=tags,
            )
        elif data.ndim == 3:
            out_ds = xr.DataArray(
                data=data,
                dims=("band", "y", "x"),
                coords={"band": np.arange(1, data.shape[0] + 1), **spatial_coords},
                attrs=tags,
            )

        # Set other attributes
        out_ds.rio.write_transform(transform, inplace=True)
        if crs is not None:
            out_ds.rio.write_crs(crs, inplace=True)
        out_ds.rio.write_nodata(nodata, inplace=True)

        return out_ds

    def to_geoutils(self) -> RasterBase:
        """
        Convert the DataArray to an in-memory GeoUtils Raster.

        :returns: A Raster with identical values and georeferencing. Dask inputs retain their lazy source graph;
            ordinary file-backed DataArrays load their values during conversion.
        """

        from geoutils.raster import Raster  # Runtime import to avoid circularity issues

        # Materialize a separate Dask result so conversion never replaces the source's lazy array
        ds = self._obj.compute() if self._chunks is not None else self._obj
        return Raster.from_array(
            data=ds.data,
            crs=self.crs,
            transform=self.transform,
            nodata=self.nodata,
            tags=self.tags,
            area_or_point=self.area_or_point,
        )

    def to_file(self, *args: Any, **kwargs: Any) -> Any:
        """
        Write raster to file.

        Wrapper around rioxarray.to_raster(), with additional logic to support Dask writing per default.

        Dask rasters write one chunk at a time using a lock chosen from the active scheduler. The call completes the
        file by default, while compute=False returns a Dask object whose compute() method writes later.
        A user lock is passed through, including lock=False to disable chunked writing.

        :param args: Positional arguments passed to rioxarray.to_raster().
        :param kwargs: Keyword arguments passed to rioxarray.to_raster().

        :returns: A deferred Dask write when compute=False with chunked writing; otherwise None.
        """

        return _write_raster(self._obj, *args, **kwargs)


# Dataset accessor

_RASTER_ARRAY_OPERATION_LOCK = threading.Lock()


def _raster_variables(dataset: xr.Dataset) -> list[str]:
    """Identify numeric raster variables with two spatial dimensions and a CRS."""

    variables = []
    for name, variable in dataset.data_vars.items():
        if variable.ndim not in (2, 3) or variable.dtype.kind not in "biufc":
            continue
        try:
            x_dimension, y_dimension = variable.rio.x_dim, variable.rio.y_dim
        except MissingSpatialDimensionError:
            continue
        if x_dimension in variable.dims and y_dimension in variable.dims and variable.rio.crs is not None:
            variables.append(str(name))
    return variables


def _prepare_raster_variable(variable: xr.DataArray) -> tuple[xr.DataArray, dict[str, str]]:
    """Copy raster metadata and arrange CF dimensions as band/y/x for DataArray calculations."""

    # DataArray initialization can normalize nodata encoding, so isolate its metadata from the Dataset
    prepared = variable.copy(deep=False)
    x_dimension, y_dimension = prepared.rio.x_dim, prepared.rio.y_dim
    dimensions = {name: axis for name, axis in ((x_dimension, "x"), (y_dimension, "y")) if name != axis}
    extra = [dimension for dimension in prepared.dims if dimension not in (x_dimension, y_dimension)]
    if extra and extra[0] != "band":
        dimensions[extra[0]] = "band"

    # Array algorithms interpret the last two axes as rows/columns, regardless of the stored dimension order
    prepared = prepared.rename(dimensions) if dimensions else prepared
    order = ("band", "y", "x") if extra else ("y", "x")
    return prepared.transpose(*order), dimensions


def _prepare_raster_input(value: Any) -> Any:
    """Arrange native raster arguments on band/y/x axes while leaving other input types unchanged."""

    if not isinstance(value, xr.DataArray) or value.ndim not in (2, 3):
        return value
    try:
        prepared, _ = _prepare_raster_variable(value)
    except MissingSpatialDimensionError:
        return value
    return prepared


def _run_raster_array_operation(
    variable: xr.DataArray, operation: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Run a whole-grid DataArray operation in a Dask task and return its array result."""

    with _RASTER_ARRAY_OPERATION_LOCK:
        with warnings.catch_warnings():
            # We ignore Rasterio/GDAL warning within each worker process
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            return getattr(variable.rst, operation)(*args, **kwargs).data


@xr.register_dataset_accessor("rst")
class DatasetRasterAccessor:
    """
    This class defines the Xarray accessor 'rst' for rasters in a Dataset.

    Dataset methods delegate calculations to the DataArray accessor for all recognized raster variables (default) or
    for selected ones (by passing ``variables`` to an operation), then combine the results to return a Dataset.
    Other variables and independent coordinates are preserved. Operations raise an error if their new grid would
    affect an unselected variable.

    A raster variable is recognized as a numeric 2D or 3D variable with two spatial dimensions and a CRS.
    The ``variables`` property of the accessor lists all variables recognized as rasters.
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        """Initialization of the accessor."""
        self._obj = xarray_obj

    @property
    def variables(self) -> list[str]:
        """Names of compatible raster value variables, in Dataset order."""
        return _raster_variables(self._obj)

    def to_file(self, *args: Any, **kwargs: Any) -> Any:
        """
        Write Dataset rasters to a file.

        Dask rasters write one chunk at a time using a lock chosen from the active scheduler. The call completes the
        file by default, while compute=False returns a Dask object whose compute() method writes later.

        An explicitly supplied lock is passed through, including lock=False to disable chunked writing.

        :param args: Positional arguments passed to rioxarray.to_raster().
        :param kwargs: Keyword arguments passed to rioxarray.to_raster().
        :returns: A deferred Dask write when compute=False with chunked writing; otherwise None.
        """
        return _write_raster(self._obj, *args, **kwargs)

    def _wrap_output_raster(
        self, operation: str, variables: Sequence[str] | None, /, *args: Any, **kwargs: Any
    ) -> xr.Dataset:
        """
        Function to wrap operations yielding a raster output for all variables in the Dataset.

        We delegate a raster operation to each selected DataArray, then reconstruct the complete Dataset.

        Internal logic:
        - _select_dataset_variables() validates the selection,
        - each variable's rst accessor performs the calculation,
        - _rebuild_dataset() checks shared coordinates before combining results with independent variables.
        """

        # 1/ Checks to validate selection/arguments
        selected = _select_dataset_variables(self._obj, self.variables, variables)
        if kwargs.get("inplace"):
            raise ValueError("Dataset raster operations return a new Dataset; inplace=True is unsupported.")
        if kwargs.get("as_array"):
            raise ValueError("Dataset raster operations return a Dataset; omit as_array=True.")

        # 2/ We perform the operation for every variable
        # Some per-operation behaviour is covered below

        # Arguments passing a reference or masks need the same axis order as the raster values
        args = tuple(_prepare_raster_input(value) for value in args)
        for parameter in ("ref", "mask"):
            if parameter in kwargs:
                kwargs[parameter] = _prepare_raster_input(kwargs[parameter])
        transformed = {}

        # Looping through each variable
        for name in selected:

            # Normalize CF spatial names and the optional band/time dimension for DataArray methods
            original = self._obj[name]
            x_dimension, y_dimension = original.rio.x_dim, original.rio.y_dim
            extra = [dimension for dimension in original.dims if dimension not in (x_dimension, y_dimension)]
            variable, dimensions = _prepare_raster_variable(original)

            # Filter each spatial slice without mixing dates or bands
            if operation == "filter" and extra:
                slices = [
                    getattr(variable.isel(band=index, drop=True).rst, operation)(*args, **kwargs)
                    for index in range(variable.sizes["band"])
                ]
                # Identify slices by their band/time labels, or by position when labels are absent
                labels = variable.coords.get("band", xr.IndexVariable("band", np.arange(variable.sizes["band"])))
                result = xr.concat(slices, dim=labels)

            # GDAL needs the complete grid, defer that eager DataArray call to one task
            elif variable.chunks and operation in ("fill_nodata", "sieve"):
                dask = import_optional("dask")
                import dask.array as da

                # Filling missing pixels can produce fractional values from integer input
                dtype = np.result_type(variable.dtype, np.float32) if operation == "fill_nodata" else variable.dtype
                task = dask.delayed(_run_raster_array_operation)(variable, operation, args, kwargs)
                array = da.from_delayed(task, shape=variable.shape, dtype=dtype)

                # Copy through the accessor so encoded nodata is normalized as it is for eager results
                result = variable.rst.copy(new_array=array, deep=False)
            else:
                result = getattr(variable.rst, operation)(*args, **kwargs)

            # Require coordinates and raster metadata for the Dataset reconstruction
            if not isinstance(result, xr.DataArray):
                raise TypeError(f"Dataset {operation}() requires a DataArray result; omit array-only output options.")

            # 3/ Restore names and coordinate metadata for each raster

            # Separate grids can use different named mappings in a NetCDF Dataset
            mapping = original.rio.grid_mapping
            if mapping != result.rio.grid_mapping:
                result = result.rename({result.rio.grid_mapping: mapping})
                result = result.rio.write_grid_mapping(grid_mapping_name=mapping)
            if dimensions:
                # Restore native dimension names and nonspatial labels after the DataArray calculation
                result = result.rename({axis: name for name, axis in dimensions.items() if axis in result.dims})
                if extra and extra[0] in original.coords:
                    result = result.assign_coords({extra[0]: original.coords[extra[0]].variable})

                # CF axis attributes let Rioxarray recognize the restored spatial names
                for dimension, axis in ((x_dimension, "X"), (y_dimension, "Y")):
                    if dimension not in ("x", "y"):
                        result.coords[dimension].attrs.setdefault("axis", axis)

            # Return axes to their stored order after calculations on band/y/x arrays
            result = result.transpose(*original.dims)
            if operation in ("filter", "fill_nodata", "sieve"):
                # Value operations leave every original coordinate and grid mapping valid
                result = xr.DataArray(result.variable, coords=original.coords, name=name)
                result.encoding["grid_mapping"] = original.rio.grid_mapping
            transformed[name] = result.rename(name)

        # 4/ Combine transformed rasters into Dataset
        return _rebuild_dataset(self._obj, transformed)

    def reproject(self, *args: Any, variables: Sequence[str] | None = None, **kwargs: Any) -> xr.Dataset:
        """
        Reproject rasters in the dataset.

        See :meth:`DataArray.rst.reproject() <geoutils.DataArrayRasterAccessor.reproject>` for shared arguments.

        :param args: Positional reprojection arguments.
        :param variables: Raster variable names to process. All recognized raster variables are used by default.
        :param kwargs: Keyword reprojection arguments.
        :returns: Dataset with the transformed raster variables.
        """
        return self._wrap_output_raster("reproject", variables, *args, **kwargs)

    def crop(self, *args: Any, variables: Sequence[str] | None = None, **kwargs: Any) -> xr.Dataset:
        """
        Crop rasters in the dataset.

        See :meth:`DataArray.rst.crop() <geoutils.DataArrayRasterAccessor.crop>` for shared arguments.

        :param args: Positional crop arguments.
        :param variables: Raster variable names to process. All recognized raster variables are used by default.
        :param kwargs: Keyword crop options.
        :returns: Dataset containing the cropped raster variables.
        """
        return self._wrap_output_raster("crop", variables, *args, **kwargs)

    def icrop(self, *args: Any, variables: Sequence[str] | None = None, **kwargs: Any) -> xr.Dataset:
        """
        Crop raster rows and columns in the dataset.

        See :meth:`DataArray.rst.icrop() <geoutils.DataArrayRasterAccessor.icrop>` for shared arguments.

        :param args: Pixel bounds passed to icrop().
        :param variables: Raster variable names to process. All recognized raster variables are used by default.
        :param kwargs: Keyword crop options.
        :returns: Dataset containing the cropped raster variables.
        """
        return self._wrap_output_raster("icrop", variables, *args, **kwargs)

    def clip(self, *args: Any, variables: Sequence[str] | None = None, **kwargs: Any) -> xr.Dataset:
        """
        Clip rasters in the dataset.

        See :meth:`DataArray.rst.clip() <geoutils.DataArrayRasterAccessor.clip>` for shared arguments.

        :param args: Mask passed to clip().
        :param variables: Raster variable names to process. All recognized raster variables are used by default.
        :param kwargs: Keyword clipping options.
        :returns: Dataset containing the clipped raster variables.
        """
        return self._wrap_output_raster("clip", variables, *args, **kwargs)

    def filter(self, *args: Any, variables: Sequence[str] | None = None, **kwargs: Any) -> xr.Dataset:
        """
        Filter raster values in the dataset.

        See :meth:`DataArray.rst.filter() <geoutils.DataArrayRasterAccessor.filter>` for shared arguments.

        :param args: Filter method and positional options.
        :param variables: Raster variable names to process. All recognized raster variables are used by default.
        :param kwargs: Keyword filter options.
        :returns: Dataset with filtered values and unchanged coordinates.
        """
        return self._wrap_output_raster("filter", variables, *args, **kwargs)

    def fill_nodata(self, *args: Any, variables: Sequence[str] | None = None, **kwargs: Any) -> xr.Dataset:
        """
        Fill raster gaps in the dataset.

        See :meth:`DataArray.rst.fill_nodata() <geoutils.DataArrayRasterAccessor.fill_nodata>` for shared arguments.

        :param args: Positional gap-filling options.
        :param variables: Raster variable names to process. All recognized raster variables are used by default.
        :param kwargs: Keyword gap-filling options.
        :returns: Dataset with filled raster values.
        """
        return self._wrap_output_raster("fill_nodata", variables, *args, **kwargs)

    def sieve(self, *args: Any, variables: Sequence[str] | None = None, **kwargs: Any) -> xr.Dataset:
        """
        Remove small regions from rasters in the dataset.

        See :meth:`DataArray.rst.sieve() <geoutils.DataArrayRasterAccessor.sieve>` for shared arguments.

        :param args: Positional sieve options.
        :param variables: Raster variable names to process. All recognized raster variables are used by default.
        :param kwargs: Keyword sieve options.
        :returns: Dataset containing the sieved rasters.
        """
        return self._wrap_output_raster("sieve", variables, *args, **kwargs)

    def translate(self, *args: Any, variables: Sequence[str] | None = None, **kwargs: Any) -> xr.Dataset:
        """
        Translate raster grids in the dataset.

        See :meth:`DataArray.rst.translate() <geoutils.DataArrayRasterAccessor.translate>` for shared arguments.

        :param args: Coordinate offsets passed to translate().
        :param variables: Raster variable names to process. All recognized raster variables are used by default.
        :param kwargs: Keyword translation options.
        :returns: Dataset containing the translated rasters.
        """
        return self._wrap_output_raster("translate", variables, *args, **kwargs)

    def _wrap_output_point(
        self, operation: str, points: Any, variables: Sequence[str] | None, *args: Any, **kwargs: Any
    ) -> xr.Dataset:
        """
        Function to wrap operations yielding a point output for all variables in the Dataset.

        We delegate a raster operation to each selected DataArray, then reconstruct the complete Dataset given that
        the output is now point data.
        """

        from geoutils._dispatch import _check_match_points, _get_pointcloud_interface
        from geoutils.pointcloud.referencing import _point_coordinates
        from geoutils.pointcloud.xr_accessor import DataArrayPointCloudAccessor

        # 1/ Checks to validate selection/arguments
        selected = _select_dataset_variables(self._obj, self.variables, variables)
        if kwargs.pop("as_array", False):
            raise ValueError("Dataset point sampling returns a Dataset; omit as_array=True.")

        # 2/ We prepare point inputs for DataArray methods

        # Adapt the point support without constructing geometry or loading lazy point values
        # The first selected raster provides the CRS reference for inputs without CRS metadata
        reference, _ = _prepare_raster_variable(self._obj[selected[0]])
        if isinstance(points, xr.Dataset):
            from geoutils.pointcloud.xr_accessor import _point_dataset_support

            # Represent shared point coordinates, CRS and auxiliary values on one DataArray
            points = _point_dataset_support(points, points.pc.variables)
        elif not isinstance(points, xr.DataArray):
            interface = _get_pointcloud_interface(points)
            if interface is not None:
                # Require an explicit conversion for partitioned point rows
                if interface._is_dask:
                    raise ValueError("Convert lazy GeoDataFrame points explicitly with to_xarray() before sampling.")
                points = interface.to_xarray().rename({"x": "x_point", "y": "y_point"})
            else:
                # Read Dask X/Y directly so eager coordinate validation does not compute them
                if isinstance(points, tuple) and len(points) == 2 and any(is_dask_array(axis) for axis in points):
                    x, y = points
                else:
                    # Validate plain coordinates or reproject point geometry into the reference raster CRS
                    x, y = _check_match_points(reference.rst, points)[0]

                # Attach a CRS and placeholder values; sampling uses the point coordinates
                x, y = np.atleast_1d(x), np.atleast_1d(y)
                crs = 4326 if kwargs.pop("input_latlon", False) else reference.rst.crs
                points = DataArrayPointCloudAccessor.from_xyz(x, y, np.empty(x.shape[0]), crs)
                points = points.rename({"x": "x_point", "y": "y_point"})

        # 3/ We perform the operation on selected rasters

        # We keep the original names of point coordinates
        _, x_name, y_name = _point_coordinates(points)
        names = {axis: name for axis, name in (("x", x_name), ("y", y_name)) if axis != name}
        transformed = {}
        for name in selected:
            # We arrange raster axes as band/y/x before sampling the shared points
            raster, _ = _prepare_raster_variable(self._obj[name])
            result = getattr(raster.rst, operation)(points, *args, **kwargs)
            if not isinstance(result, xr.DataArray):
                raise TypeError("Dataset point sampling requires DataArray results.")
            transformed[name] = result.rename(names).rename(name)

        # 4/ We combine sampled values with untouched Dataset variables
        return _rebuild_dataset(self._obj, transformed)

    def interp_at_points(
        self, points: Any, *args: Any, variables: Sequence[str] | None = None, **kwargs: Any
    ) -> xr.Dataset:
        """
        Interpolate rasters in the dataset at points.

        See :meth:`DataArray.rst.interp_at_points() <geoutils.DataArrayRasterAccessor.interp_at_points>` for shared
        arguments.

        :param points: Point DataArray, Dataset, point cloud, or X/Y arrays.
        :param args: Positional interpolation options.
        :param variables: Raster variable names to process. All recognized raster variables are used by default.
        :param kwargs: Keyword interpolation options.
        :returns: Dataset with selected raster variables replaced by values on the point dimension.
        """
        return self._wrap_output_point("interp_at_points", points, variables, *args, **kwargs)

    def interp_points(
        self, points: Any, *args: Any, variables: Sequence[str] | None = None, **kwargs: Any
    ) -> xr.Dataset:
        """
        Interpolate rasters in the dataset at points.

        Alias of interp_at_points(). See
        :meth:`DataArray.rst.interp_points() <geoutils.DataArrayRasterAccessor.interp_points>` for shared arguments.

        :param points: Native compatible point input.
        :param args: Positional interpolation options.
        :param variables: Raster variable names to process. All recognized raster variables are used by default.
        :param kwargs: Keyword interpolation options.
        :returns: Dataset containing interpolated point values.
        """
        return self.interp_at_points(points, *args, variables=variables, **kwargs)

    def resample_at_points(
        self, points: Any, *args: Any, variables: Sequence[str] | None = None, **kwargs: Any
    ) -> xr.Dataset:
        """
        Resample rasters in the dataset at points.

        See :meth:`DataArray.rst.resample_at_points() <geoutils.DataArrayRasterAccessor.resample_at_points>` for shared
        arguments.

        :param points: Native point input defining shared output locations.
        :param args: Resampling operator and positional options.
        :param variables: Raster variable names to process. All recognized raster variables are used by default.
        :param kwargs: Keyword resampling options.
        :returns: Dataset with selected raster values sampled on the point dimension.
        """
        return self._wrap_output_point("resample_at_points", points, variables, *args, **kwargs)

    def reduce_at_points(
        self, points: Any, *args: Any, variables: Sequence[str] | None = None, **kwargs: Any
    ) -> xr.Dataset:
        """
        Reduce raster windows in the dataset at points.

        See :meth:`DataArray.rst.reduce_at_points() <geoutils.DataArrayRasterAccessor.reduce_at_points>` for shared
        arguments.

        :param points: Native point input defining output locations.
        :param args: Reducer and positional window options.
        :param variables: Raster variable names to process. All recognized raster variables are used by default.
        :param kwargs: Keyword reduction options.
        :returns: Dataset with reduced raster values on the point dimension.
        """
        return self._wrap_output_point("reduce_at_points", points, variables, *args, **kwargs)

    def _wrap_output_dict(
        self, operation: str, variables: Sequence[str] | None, /, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Function to wrap operations yielding a dictionary output for all variables in the Dataset.

        We delegate a raster operation to each selected DataArray, then collect the results in a dictionary with one
        key per variable name.
        """

        # 1/ Checks to validate selection/arguments
        selected = _select_dataset_variables(self._obj, self.variables, variables)

        # 2/ We prepare raster arguments and calculation options

        # Arrange native raster arguments without reading values
        args = tuple(_prepare_raster_input(value) for value in args)
        if "mask" in kwargs:
            kwargs["mask"] = _prepare_raster_input(kwargs["mask"])

        # Reuse explicit distance bins even when their edges can be iterated only once
        bins = kwargs.get("bins")
        if bins is not None and not isinstance(bins, str):
            kwargs["bins"] = tuple(bins)

        # 3/ We collect a result for every selected raster variable

        # Independent calculations preserve the requested variable order and source metadata
        results = {}
        for name in selected:
            raster, _ = _prepare_raster_variable(self._obj[name])
            results[name] = getattr(raster.rst, operation)(*args, **kwargs)
        return results

    def variogram(self, *, variables: Sequence[str] | None = None, **kwargs: Any) -> dict[str, Variogram]:
        """
        Estimate an empirical variogram for each selected raster variable.

        Each variable is sampled independently after excluding its missing values. Dask rasters are read in chunks;
        the returned variograms contain eager NumPy arrays. An integer random_state is reused for each variable,
        while a supplied generator advances in the selected variable order.

        See :meth:`DataArray.rst.variogram() <geoutils.DataArrayRasterAccessor.variogram>` for shared arguments.

        :param variables: Raster variable names to analyze. All recognized raster variables are used by default.
        :param kwargs: Pair sampling, estimation and fitting options passed to DataArray.rst.variogram().
        :returns: Dictionary mapping each selected variable name to its Variogram.
        """
        return self._wrap_output_dict("variogram", variables, **kwargs)

    def stats(self, *args: Any, variables: Sequence[str] | None = None, **kwargs: Any) -> Any:
        """
        Calculate statistics for rasters in the dataset.

        Grouping or vector fractional coverage is calculated once for all variables, and statistics are computed at
        once for all variables with same dimensions.

        See :meth:`DataArray.rst.stats() <geoutils.DataArrayRasterAccessor.stats>` for shared arguments.

        :param args: Statistics to calculate.
        :param variables: Raster variable names to process. All recognized raster variables are used by default.
        :param kwargs: Grouping, sampling and reduction options. Use variables instead of values.
        :returns: Named statistics, a grouped table, or a grouped table and masks.
        """

        selected = _select_dataset_variables(self._obj, self.variables, variables)
        if "values" in kwargs:
            raise ValueError("Use 'variables' to select Dataset raster values instead of 'values'.")
        if "by" in kwargs and kwargs["by"] is not None:
            kwargs["by"] = {name: _prepare_raster_input(value) for name, value in kwargs["by"].items()}
        for parameter in ("at", "mask"):
            if parameter in kwargs:
                kwargs[parameter] = _prepare_raster_input(kwargs[parameter])
        values = {name: _prepare_raster_variable(self._obj[name])[0] for name in selected}
        return values[selected[0]].rst.stats(*args, values=values, **kwargs)
