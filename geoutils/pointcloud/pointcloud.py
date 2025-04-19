# Copyright (c) 2025 GeoUtils developers
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
"""Module for PointCloud class."""

from __future__ import annotations

import os.path
import pathlib
import warnings
from typing import Any, Iterable, Literal, Callable, overload

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from rasterio.coords import BoundingBox
from rasterio.transform import from_origin
from shapely.geometry.base import BaseGeometry

import geoutils as gu
from geoutils._typing import ArrayLike, NDArrayNum, NDArrayBool, DTypeLike, Number
from geoutils.interface.gridding import _grid_pointcloud
from geoutils.raster.georeferencing import _coords
from geoutils.stats.stats import _get_single_stat, _statistics, _STATS_ALIASES
from geoutils.stats.sampling import subsample_array

try:
    import laspy

    has_laspy = True
except ImportError:
    has_laspy = False

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



def _load_laspy_data(filename: str, columns: list[str]) -> gpd.GeoDataFrame:
    """Load point cloud data from LAS/LAZ/COPC file."""

    # Read file
    las = laspy.read(filename)

    # Get data from requested columns
    data = np.vstack([las[n] for n in columns]).T

    # Build geodataframe
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=las.x, y=las.y, crs=las.header.parse_crs(prefer_wkt=False)),
        data=data,
        columns=columns,
    )

    return gdf


def _load_laspy_metadata(
    filename: str,
) -> tuple[CRS, int, BoundingBox, pd.Index]:
    """Load point cloud metadata from LAS/LAZ/COPC file."""

    with laspy.open(filename) as f:

        crs = f.header.parse_crs(prefer_wkt=False)
        nb_points = f.header.point_count
        bounds = BoundingBox(left=f.header.x_min, right=f.header.x_max, bottom=f.header.y_min, top=f.header.y_max)
        columns_names = pd.Index(list(f.header.point_format.dimension_names))

    return crs, nb_points, bounds, columns_names


# def _write_laspy(filename: str, pc: gpd.GeoDataFrame):
#     """Write a point cloud dataset as LAS/LAZ/COPC."""
#
#     with laspy.open(filename) as f:
#         new_hdr = laspy.LasHeader(version="1.4", point_format=6)
#         # You can set the scales and offsets to values that suits your data
#         new_hdr.scales = np.array([1.0, 0.5, 0.1])
#         new_las = laspy.LasData(header = new_hdr, points=)
#
#     return

def _cast_numeric_array_pointcloud(
    pc: PointCloud, other: PointCloud | NDArrayNum | Number, operation_name: str
) -> tuple[Number, NDArrayNum]:
    """
    Cast a point cloud and another point cloud or array or number to arrays with proper metadata, or raise an error message.

    :param pc: Pointcloud.
    :param other: Point cloud or array or number.
    :param operation_name: Name of operation to raise in the error message.
    """

    # Check first input is a point cloud
    if not isinstance(pc, PointCloud):
        raise ValueError("Developer error: Only a point cloud should be passed as first argument to this function.")

    # Check that other is of correct type
    # If not, a NotImplementedError should be raised, in case other's class has a method implemented.
    # See https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
    if not isinstance(other, (PointCloud, np.ndarray, float, int, np.floating, np.integer)):
        raise NotImplementedError(
            f"Operation between an object of type {type(other)} and a point cloud impossible. Must be a point cloud, "
            f"np.ndarray or single number."
        )

    # If other is a point cloud
    if isinstance(other, PointCloud):

        other_data = other.data
        # Check that both point clouds have the same shape and georeferences
        if pc.georeferenced_coords_equal(other):  # type: ignore
            pass
        else:
            raise ValueError(
                "Both point clouds must have the same points X/Y coordinates and CRS for " + operation_name + "."
            )

    # If other is an array
    elif isinstance(other, np.ndarray):

        other_data = other.squeeze()
        if other.squeeze().ndim == 1 and other.squeeze().shape[0] == pc.nb_points:
            pass
        else:
            raise ValueError(
                "The array must be 1-dimensional with the same number of points as the point cloud for " + operation_name + "."
            )

    else:
        other_data = other

    return other_data

class PointCloud(gu.Vector):  # type: ignore[misc]
    """
    The georeferenced point cloud.

    A point cloud is a vector of 2D point geometries associated to numeric values from a data column, and potentially
    auxiliary data columns.

     Main attributes:
        ds: :class:`geopandas.GeoDataFrame`
            Geodataframe of the point cloud.
        data_column: str
            Name of point cloud data column.
        crs: :class:`pyproj.crs.CRS`
            Coordinate reference system of the point cloud.
        bounds: :class:`rio.coords.BoundingBox`
            Coordinate bounds of the point cloud.


    All other attributes are derivatives of those attributes, or read from the file on disk.
    See the API for more details.
    """

    def __init__(
        self,
        filename_or_dataset: str | pathlib.Path | gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry,
        data_column: str,
    ):
        """
        Instantiate a point cloud from either a data column name and a vector (filename, GeoPandas dataframe or series,
        or a Shapely geometry), or only with a point cloud file type.

        :param filename_or_dataset: Path to vector file, or GeoPandas dataframe or series, or Shapely geometry.
        :param data_column: Name of main data column defining the point cloud.
        """

        self._ds: gpd.GeoDataFrame | None = None
        self._name: str | None = None
        self._crs: CRS | None = None
        self._bounds: BoundingBox
        self._data_column: str
        self._data: NDArrayNum
        self._nb_points: int
        self._all_columns: pd.Index

        # If PointCloud is passed, simply point back to PointCloud
        if isinstance(filename_or_dataset, PointCloud):
            for key in filename_or_dataset.__dict__:
                setattr(self, key, filename_or_dataset.__dict__[key])
            return
        # For filename, rely on parent Vector class or LAS file reader
        else:
            if isinstance(filename_or_dataset, (str, pathlib.Path)) and os.path.splitext(filename_or_dataset)[-1] in [
                ".las",
                ".laz",
            ]:
                # Load only metadata, and not the data
                fn = filename_or_dataset if isinstance(filename_or_dataset, str) else filename_or_dataset.name
                crs, nb_points, bounds, columns = _load_laspy_metadata(fn)
                self._name = fn
                self._crs = crs
                self._nb_points = nb_points
                self._all_columns = columns
                self._bounds = bounds
                self._ds = None
            # Check on filename are done with Vector.__init__
            else:
                super().__init__(filename_or_dataset)
                # Verify that the vector can be cast as a point cloud
                if not all(p == "Point" for p in self.ds.geom_type):
                    raise ValueError(
                        "This vector file contains non-point geometries, " "cannot be instantiated as a point cloud."
                    )

        # Set data column following user input
        self.set_data_column(new_data_column=data_column)

    # TODO: Could also move to Vector directly?
    ##############################################
    # OVERRIDDEN VECTOR METHODS TO SUPPORT LOADING
    ##############################################

    @property
    def ds(self) -> gpd.GeoDataFrame:
        """Geodataframe of the point cloud."""
        # We need to override the Vector method to introduce the is_loaded dynamic for LAS files
        if not self.is_loaded:
            self.load()
        return self._ds  # type: ignore

    @ds.setter
    def ds(self, new_ds: gpd.GeoDataFrame | gpd.GeoSeries) -> None:
        """Set a new geodataframe for the point cloud."""
        # We need to override the setter Vector method because we have overridden the property method
        # (even if the code below is the same)
        if isinstance(new_ds, gpd.GeoDataFrame):
            self._ds = new_ds
        elif isinstance(new_ds, gpd.GeoSeries):
            self._ds = gpd.GeoDataFrame(geometry=new_ds)
        else:
            raise ValueError("The dataset of a vector must be set with a GeoSeries or a GeoDataFrame.")

    @property
    def crs(self) -> CRS:
        """Coordinate reference system of the vector."""
        # Overriding method in Vector
        if self.is_loaded:
            return super().crs
        else:
            return self._crs

    @property
    def bounds(self) -> BoundingBox:
        # Overriding method in Vector
        if self.is_loaded:
            return super().bounds
        else:
            return self._bounds

    #####################################
    # NEW METHODS SPECIFIC TO POINT CLOUD
    #####################################

    @property
    def data(self) -> NDArrayNum:
        """
        Data of the point cloud.

        Points to the data column of the geodataframe, equivalent to calling self.ds[self.data_column].
        """
        # Triggers the loading mechanism through self.ds
        return self.ds[self.data_column].values

    @data.setter
    def data(self, new_data: NDArrayNum) -> None:
        """Set new data for the point cloud."""

        self.ds[self.data_column] = new_data

    @property
    def all_columns(self) -> pd.Index:
        """Index of all columns of the point cloud, excluding the column of 2D point geometries."""
        # Overriding method in Vector
        if self.is_loaded:
            all_columns = super().columns
            all_columns_nongeom = all_columns[all_columns != "geometry"]
            return all_columns_nongeom
        else:
            return self._all_columns

    @property
    def data_column(self) -> str:
        """Name of data column of the point cloud."""
        return self._data_column

    @data_column.setter
    def data_column(self, new_data_column: str) -> None:
        self.set_data_column(new_data_column=new_data_column)

    def set_data_column(self, new_data_column: str) -> None:
        """Set new column as point cloud data column."""

        if new_data_column not in self.all_columns:
            raise ValueError(
                f"Data column {new_data_column} not found among columns, available columns "
                f"are: {', '.join(self.all_columns)}."
            )
        self._data_column = new_data_column

    @property
    def is_loaded(self) -> bool:
        """Whether the point cloud data is loaded"""
        return self._ds is not None

    @property
    def nb_points(self) -> int:
        """Number of points in the point cloud."""
        # New method for point cloud
        if self.is_loaded:
            return len(self.ds)
        else:
            return self._nb_points

    def load(self, columns: Literal["all", "main"] | list[str] = "main") -> None:
        """
        Load point cloud from disk (only supported for LAS files).

        :param columns: Columns to load. Defaults to main data column only.
        """

        if self.is_loaded:
            raise ValueError("Data are already loaded.")

        if self.name is None:
            raise AttributeError(
                "Cannot load as filename is not set anymore. Did you manually update the filename attribute?"
            )

        if columns == "all":
            columns_to_load = self.all_columns
        elif columns == "main":
            columns_to_load = [self.data_column]
        else:
            columns_to_load = columns

        ds = _load_laspy_data(filename=self.name, columns=columns_to_load)
        self._ds = ds

    @overload
    def astype(
            self: PointCloud, dtype: DTypeLike, convert_nodata: bool = True, *, inplace: Literal[False] = False
    ) -> PointCloud:
        ...

    @overload
    def astype(self: PointCloud, dtype: DTypeLike, convert_nodata: bool = True, *, inplace: Literal[True]) -> None:
        ...

    @overload
    def astype(
            self: PointCloud, dtype: DTypeLike, convert_nodata: bool = True, *, inplace: bool = False
    ) -> PointCloud | None:
        ...

    def astype(
            self: PointCloud, dtype: DTypeLike, convert_coords: bool = False, inplace: bool = False
    ) -> PointCloud | None:
        """
        Convert data type of the point cloud data column.

        :param dtype: Any numpy dtype or string accepted by numpy.astype.
        :param convert_coords: Whether to convert the data type of coordinates values as well.
        :param inplace: Whether to modify the point cloud in-place.

        :returns: Point cloud with updated dtype (or None if inplace).
        """

        out_data = self.data.astype(dtype)

        if inplace:
            self._data = out_data  # type: ignore
            if convert_coords:
                self.ds.geometry.x = self.ds.geometry.x.values.astype(dtype)
                self.ds.geometry.y = self.ds.geometry.y.values.astype(dtype)
            return None
        else:
            if convert_coords:
                x = self.ds.geometry.x.values.astype(dtype)
                y = self.ds.geometry.y.values.astype(dtype)
                return gu.PointCloud.from_xyz(x=x, y=y, z=out_data, crs=self.crs, data_column=self.data_column)
            else:
                return self.copy(new_array=out_data)


    def copy(self: PointCloud, new_array: NDArrayNum | None = None) -> PointCloud:
        """
        Copy the point cloud in-memory.

        :param new_array: New data array to use in the copied point cloud's data column.

        :return: Copy of the point cloud.
        """


        # Define new array
        if new_array is not None:
            if not isinstance(new_array, np.ndarray):
                raise ValueError("New data must be an array.")
            new_array = new_array.squeeze()
            if not (new_array.ndim == 1 and new_array.shape[0] == self.nb_points):
                raise ValueError("New data array must be 1-dimensional with the same number of points as the point "
                                 "cloud being copied.")
            data = new_array
        else:
            data = self.data.copy()

        # Send to from_array
        cp = self.from_xyz(
            x=self.geometry.x.values,
            y=self.geometry.y.values,
            z=data,
            crs=self.crs,
            data_column=self.data_column
        )

        return cp

    @classmethod
    def from_xyz(cls, x: ArrayLike, y: ArrayLike, z: ArrayLike, crs: CRS, data_column: str = "z") -> PointCloud:
        """
        Create point cloud from three 1D array-like coordinates for X/Y/Z.

        Note that this is the most modular method to create a point cloud, as it allows to specify different data
        types for the different coordinates or columns.

        :param x: X coordinates of point cloud.
        :param y: Y coordinates of point cloud.
        :param z: Z values of point cloud.
        :param crs: Coordinate reference system.
        :param data_column: Data column of point cloud.

        :return Point cloud.
        """

        # Build geodataframe
        gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(x=x, y=y, crs=crs), data={data_column: z}
        )

        # If the data was transformed into boolean, re-initialize as a Mask subclass
        # Typing: we can specify this behaviour in @overload once we add the NumPy plugin of MyPy
        if z.dtype == bool:
            return PointCloudMask(filename_or_dataset=gdf, data_column=data_column)  # type: ignore
        # Otherwise, keep as a given PointCloudType subclass
        else:
            return cls(filename_or_dataset=gdf, data_column=data_column)


    @classmethod
    def from_array(cls, data: NDArrayNum, crs: CRS, data_column: str = "z") -> PointCloud:
        """
        Create point cloud from a 3 x N or N x 3 array of X coordinates, Y coordinates and Z values.

        :param data: Point cloud coordinates and data values as 3 x N or N x 3 array.
        :param crs: Coordinate reference system of point cloud.
        :param data_column: Data column of point cloud.

        :return Point cloud.
        """

        # Check shape
        if data.ndim != 2 or (data.shape[0] != 3 and data.shape[1] != 3):
            raise ValueError("Array must be of shape 3xN or Nx3.")

        # Make the first axis the one with size 3
        if data.shape[0] != 3:
            data = data.T

        return cls.from_xyz(x=data[0, :], y=data[1, :], z=data[2, :], crs=crs, data_column=data_column)

    @classmethod
    def from_tuples(
        cls, tuples_xyz: Iterable[tuple[Number, Number, Number]], crs: CRS, data_column: str = "z"
    ) -> PointCloud:
        """
        Create point cloud from an iterable of 3-tuples (X coordinate, Y coordinate, Z value).

        :param tuples_xyz: Point cloud coordinates and data as an iterable of 3-tuples.
        :param crs: Coordinate reference system of point cloud.
        :param data_column: Data column of point cloud.

        :return Point cloud.
        """

        return cls.from_array(np.array(tuples_xyz), crs=crs, data_column=data_column)

    def to_xyz(self) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum]:
        """Convert point cloud to three 1D arrays of coordinates for X/Y/Z."""

        return self.geometry.x.values, self.geometry.y.values, self.ds[self.data_column].values

    def to_array(self) -> NDArrayNum:
        """Convert point cloud to a 3 x N array of X coordinates, Y coordinates and Z values."""

        return np.stack((self.geometry.x.values, self.geometry.y.values, self.ds[self.data_column].values), axis=0)

    def to_tuples(self) -> Iterable[tuple[Number, Number, Number]]:
        """Convert point cloud to a list of 3-tuples (X coordinate, Y coordinate, Z value)."""

        return list(zip(self.geometry.x.values, self.geometry.y.values, self.ds[self.data_column].values))

    def __array_ufunc__(
            self,
            ufunc: Callable[[NDArrayNum | tuple[NDArrayNum, NDArrayNum]], NDArrayNum | tuple[NDArrayNum, NDArrayNum]],
            method: str,
            *inputs: PointCloud | tuple[PointCloud, PointCloud] | tuple[NDArrayNum, PointCloud] | tuple[PointCloud, NDArrayNum],
            **kwargs: Any,
    ) -> PointCloud | tuple[PointCloud, PointCloud]:
        """
        Method to cast NumPy universal functions directly on PointCloud classes, by passing to the masked array.
        This function basically applies the ufunc (with its method and kwargs) to .data, and rebuilds the PointCloud from
        self.__class__. The cases separate the number of input nin and output nout, to properly feed .data and return
        PointCloud objects.
        See more details in NumPy doc, e.g., https://numpy.org/doc/stable/user/basics.dispatch.html#basics-dispatch.
        """

        # In addition to running ufuncs, this function takes over arithmetic operations (__add__, __multiply__, etc...)
        # when the first input provided is a NumPy array and second input a PointCloud.
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

            # Check the casting between Point cloud and array inputs, and return error messages if not consistent

            # Raise errors if necessary
            if isinstance(inputs[0], PointCloud):
                pc = inputs[0]
                other = inputs[1]
            else:
                pc = inputs[1]
                other = inputs[0]
            _ = _cast_numeric_array_pointcloud(pc, other, "an arithmetic operation")  # type: ignore

            # Get data depending on argument order
            if isinstance(inputs[0], PointCloud):
                first_arg = inputs[0].data
            else:
                first_arg = inputs[0]

            if isinstance(inputs[1], PointCloud):
                second_arg = inputs[1].data
            else:
                second_arg = inputs[1]

            # For one output
            if ufunc.nout == 1:
                return self.copy(new_array=final_ufunc(first_arg, second_arg, **kwargs))

            # If the universal function has two outputs (Note: no ufunc exists that has three outputs or more)
            else:
                output = final_ufunc(first_arg, second_arg, **kwargs)  # type: ignore
                return self.copy(new_array=output[0]), self.copy(new_array=output[1])

    def __array_function__(
            self, func: Callable[[NDArrayNum, Any], Any], types: tuple[type], args: Any, kwargs: Any
    ) -> Any:
        """
        Method to cast NumPy array function directly on a Point cloud object by applying it to the masked array.
        A limited number of function is supported, listed in point cloud.handled_array_funcs.
        """

        # If function is not implemented
        if func.__name__ not in _HANDLED_FUNCTIONS_1NIN + _HANDLED_FUNCTIONS_2NIN:
            return NotImplemented

        # For subclassing
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented

        # Get first argument
        first_arg = args[0].data

        # Separate one and two input functions
        if func.__name__ in _HANDLED_FUNCTIONS_1NIN:
            outputs = func(first_arg, *args[1:], **kwargs)  # type: ignore
        # Two input functions require casting
        else:
            # Check the casting between point cloud and array inputs, and return error messages if not consistent
            if isinstance(args[0], PointCloud):
                pc = args[0]
                other = args[1]
            else:
                pc = args[1]
                other = args[0]
            _ = _cast_numeric_array_pointcloud(pc, other, operation_name="an arithmetic operation")
            second_arg = args[1].data
            outputs = func(first_arg, second_arg, *args[2:], **kwargs)  # type: ignore

        # Below, we recast to PointCloud if the shape was preserved, otherwise return an array
        # First, if there are several outputs in a tuple which are arrays
        if isinstance(outputs, tuple) and isinstance(outputs[0], np.ndarray):
            if all(output.shape == args[0].data.shape for output in outputs):
                return tuple(self.copy(new_array=output) for output in outputs)
            else:
                return outputs
        # Second, if there is a single output which is an array
        elif isinstance(outputs, np.ndarray):
            if outputs.shape == args[0].data.shape:
                return self.copy(new_array=outputs)
            else:
                return outputs
        # Else, return outputs directly
        else:
            return outputs

    def __add__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        Sum two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data + other_data
        return self.copy(new_array=out_data)

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __radd__(self: PointCloud, other: NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Sum two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        For when other is first item in the operation (e.g. 1 + rst).
        """
        return self.__add__(other)  # type: ignore

    def __neg__(self: PointCloud) -> PointCloud:
        """
        Take the point cloud negation.

        Returns a point cloud with -self.data.
        """
        return self.copy(-self.data)

    def __sub__(self, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        Subtract two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data - other_data
        return self.copy(new_array=out_data)

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __rsub__(self: PointCloud, other: NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Subtract two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        For when other is first item in the operation (e.g. 1 - rst).
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = other_data - self.data
        return self.copy(new_array=out_data)

    def __mul__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        Multiply two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data * other_data
        return self.copy(new_array=out_data)

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __rmul__(self: PointCloud, other: NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Multiply two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        For when other is first item in the operation (e.g. 2 * rst).
        """
        return self.__mul__(other)

    def __truediv__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        True division of two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data / other_data
        return self.copy(new_array=out_data)

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __rtruediv__(self: PointCloud, other: NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        True division of two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        For when other is first item in the operation (e.g. 1/rst).
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = other_data / self.data
        return self.copy(new_array=out_data)

    def __floordiv__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        Floor division of two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data // other_data
        return self.copy(new_array=out_data)

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __rfloordiv__(self: PointCloud, other: NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Floor division of two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        For when other is first item in the operation (e.g. 1/rst).
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = other_data // self.data
        return self.copy(new_array=out_data)

    def __mod__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        Modulo of two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data % other_data
        return self.copy(new_array=out_data)

    def __pow__(self: PointCloud, power: int | float) -> PointCloud:
        """
        Power of a point cloud to a number.
        """
        # Check that input is a number
        if not isinstance(power, (float, int, np.floating, np.integer)):
            raise ValueError("Power needs to be a number.")

        # Calculate the product of arrays and save to new point cloud
        out_data = self.data**power
        return self.copy(new_array=out_data)

    def __eq__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Element-wise equality of two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        This operation casts the result into a Mask.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data == other_data
        return self.copy(new_array=out_data)

    def __ne__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Element-wise negation of two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        This operation casts the result into a Mask.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data != other_data
        return self.copy(new_array=out_data)

    def __lt__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        Element-wise lower than comparison of two point clouds, or a point cloud and a numpy array,
        or a point cloud and single number.

        This operation casts the result into a Mask.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data < other_data
        return self.copy(new_array=out_data)

    def __le__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        Element-wise lower or equal comparison of two point clouds, or a point cloud and a numpy array,
        or a point cloud and single number.

        This operation casts the result into a Mask.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data <= other_data
        return self.copy(new_array=out_data)

    def __gt__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        Element-wise greater than comparison of two point clouds, or a point cloud and a numpy array,
        or a point cloud and single number.

        This operation casts the result into a Mask.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data > other_data
        return self.copy(new_array=out_data)

    def __ge__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        Element-wise greater or equal comparison of two point clouds, or a point cloud and a numpy array,
        or a point cloud and single number.

        This operation casts the result into a Mask.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data >= other_data
        return self.copy(new_array=out_data)


    def pointcloud_equal(self, other: PointCloud, **kwargs: Any) -> bool:
        """
        Check if two point clouds are equal.

        This means that:
        - The two vectors (geodataframes) are equal.
        - The data column is the same for both point clouds.

        Keyword arguments are passed to geopandas.assert_geodataframe_equal.
        """

        # Vector equality
        vector_eq = super().vector_equal(other, **kwargs)
        # Data column equality
        data_column_eq = self.data_column == other.data_column

        return all([vector_eq, data_column_eq])

    def georeferenced_coords_equal(self: PointCloud, pc: PointCloud) -> bool:
        """
        Check that point cloud X/Y coordinates and CRS are equal.

        :param pc: Another pointcloud.

        :return: Whether the two objects have the same georeferenced points.
        """

        return all([self.crs == pc.crs,
                    np.array_equal(self.geometry.x.values, pc.geometry.x.values),
                    np.array_equal(self.geometry.y.values, pc.geometry.y.values)])

    @overload
    def get_stats(
            self,
            stats_name: str | Callable[[NDArrayNum], np.floating[Any]],
    ) -> np.floating[Any]:
        ...

    @overload
    def get_stats(
            self,
            stats_name: list[str | Callable[[NDArrayNum], np.floating[Any]]] | None = None,
    ) -> dict[str, np.floating[Any]]:
        ...

    def get_stats(
            self,
            stats_name: (
                    str | Callable[[NDArrayNum], np.floating[Any]] | list[
                str | Callable[[NDArrayNum], np.floating[Any]]] | None
            ) = None,
    ) -> np.floating[Any] | dict[str, np.floating[Any]]:
        """
        Retrieve specified statistics or all available statistics for the point cloud data. Allows passing custom callables
        to calculate custom stats.

        :param stats_name: Name or list of names of the statistics to retrieve. If None, all statistics are returned.
            Accepted names include:
            `mean`, `median`, `max`, `min`, `sum`, `sum of squares`, `90th percentile`, `LE90`, `nmad`, `rmse`,
            `std`, `valid count`, `total count`, `percentage valid points` and if an inlier mask is passed :
            `valid inlier count`, `total inlier count`, `percentage inlier point`, `percentage valid inlier points`.
            Custom callables can also be provided.
        :returns: The requested statistic or a dictionary of statistics if multiple or all are requested.
        """

        # Force load if not loaded
        if not self.is_loaded:
            self.load()

        data = self.data
        stats_dict = _statistics(data=data)
        if stats_name is None:
            return stats_dict

        stats_aliases = _STATS_ALIASES

        if isinstance(stats_name, list):
            result = {}
            for name in stats_name:
                if callable(name):
                    result[name.__name__] = name(self.data)
                else:
                    result[name] = _get_single_stat(stats_dict, stats_aliases, name)
            return result
        else:
            if callable(stats_name):
                return stats_name(self.data)
            else:
                return _get_single_stat(stats_dict, stats_aliases, stats_name)

    @overload
    def subsample(
        self,
        subsample: int | float,
        return_indices: Literal[False] = False,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> NDArrayNum:
        ...

    @overload
    def subsample(
        self,
        subsample: int | float,
        return_indices: Literal[True],
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> tuple[NDArrayNum, ...]:
        ...

    @overload
    def subsample(
        self,
        subsample: float | int,
        return_indices: bool = False,
        random_state: int | np.random.Generator | None = None,
    ) -> NDArrayNum | tuple[NDArrayNum, ...]:
        ...

    def subsample(
        self,
        subsample: float | int,
        return_indices: bool = False,
        random_state: int | np.random.Generator | None = None,
    ) -> NDArrayNum | tuple[NDArrayNum, ...]:
        """
        Randomly sample the point cloud. Only valid values are considered.

        :param subsample: Subsample size. If <= 1, a fraction of the total pixels to extract.
            If > 1, the number of pixels.
        :param return_indices: Whether to return the extracted indices only.
        :param random_state: Random state or seed number.

        :return: Array of sampled valid values, or array of sampled indices.
        """

        return subsample_array(
            array=self.data, subsample=subsample, return_indices=return_indices, random_state=random_state
        )


    def grid(
        self,
        ref: gu.Raster | None = None,
        grid_coords: tuple[NDArrayNum, NDArrayNum] | None = None,
        res: float | tuple[float, float] | None = None,
        resampling: Literal["nearest", "linear", "cubic"] = "linear",
        dist_nodata_pixel: float = 1.0,
    ) -> gu.Raster:
        """Grid point cloud into a point cloud."""

        if isinstance(ref, gu.Raster):
            if grid_coords is None:
                warnings.warn(
                    "Both reference point cloud and grid coordinates were passed for gridding, "
                    "using only the reference point cloud."
                )
            grid_coords = ref.coords(grid=False)
        else:
            if res is not None:
                xsize = (self.bounds.right - self.bounds.left) / res
                ysize = (self.bounds.top - self.bounds.bottom) / res
                transform = from_origin(west=self.bounds.left, north=self.bounds.top, xsize=xsize, ysize=ysize)
                grid_coords = _coords(transform=transform, shape=(ysize, xsize), grid=False, area_or_point=None)
            else:
                grid_coords = grid_coords

        array, transform = _grid_pointcloud(
            self.ds,
            grid_coords=grid_coords,
            data_column_name=self.data_column,
            resampling=resampling,
            dist_nodata_pixel=dist_nodata_pixel,
        )

        return gu.Raster.from_array(data=array, transform=transform, crs=self.crs, nodata=None)

    def subsample(self, subsample: float | int, random_state: int | np.random.Generator | None = None) -> PointCloud:

        indices = subsample_array(
            array=self.data, subsample=subsample, return_indices=True, random_state=random_state
        )

        return PointCloud(self.ds[indices])

    # @classmethod
    # def from_point cloud(cls, point cloud: gu.Point cloud) -> PointCloud:
    #     """Create a point cloud from a point cloud. Equivalent with Point cloud.to_pointcloud."""
    #
    #    pc = _point cloud_to_pointcloud(source_point cloud=point cloud)


class PointCloudMask(PointCloud):

    """
   The georeferenced point cloud mask.

   A point cloud mask is a point cloud with a boolean data columns (True or False), that can serve to index or assign
   values to other point clouds of the same georeferenced points.

   Subclasses :class:`geoutils.PointCloud`.

    Main attributes:
        ds: :class:`geopandas.GeoDataFrame`
            Geodataframe of the point cloud.
        data_column: str
            Name of point cloud data column.
        crs: :class:`pyproj.crs.CRS`
            Coordinate reference system of the point cloud.
        bounds: :class:`rio.coords.BoundingBox`
            Coordinate bounds of the point cloud.

   All other attributes are derivatives of those attributes, or read from the file on disk.
   See the API for more details.
   """

    def __init__(
            self,
            filename_or_dataset: PointCloud | str | pathlib.Path | gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry,
            **kwargs: Any,
    ) -> None:

        self._data: MArrayNum | MArrayBool | None = None  # type: ignore

        # If a Mask is passed, simply point back to Mask
        if isinstance(filename_or_dataset, PointCloudMask):
            for key in filename_or_dataset.__dict__:
                setattr(self, key, filename_or_dataset.__dict__[key])
            return
        # Else rely on parent Raster class options (including raised errors)
        else:
            super().__init__(filename_or_dataset, **kwargs)

            # Convert masked array to boolean
            self._data = self.data.astype(bool)  # type: ignore


    def __and__(self: PointCloudMask, other: PointCloudMask | NDArrayBool) -> PointCloudMask:
        """Bitwise and between masks, or a mask and an array."""
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")

        return self.copy(self.data & other_data)  # type: ignore

    def __rand__(self: PointCloudMask, other: PointCloudMask | NDArrayBool) -> PointCloudMask:
        """Bitwise and between masks, or a mask and an array."""

        return self.__and__(other)

    def __or__(self: PointCloudMask, other: PointCloudMask | NDArrayBool) -> PointCloudMask:
        """Bitwise or between masks, or a mask and an array."""

        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")

        return self.copy(self.data | other_data)  # type: ignore

    def __ror__(self: PointCloudMask, other: PointCloudMask | NDArrayBool) -> PointCloudMask:
        """Bitwise or between masks, or a mask and an array."""

        return self.__or__(other)

    def __xor__(self: PointCloudMask, other: PointCloudMask | NDArrayBool) -> PointCloudMask:
        """Bitwise xor between masks, or a mask and an array."""

        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")

        return self.copy(self.data ^ other_data)  # type: ignore

    def __rxor__(self: PointCloudMask, other: PointCloudMask | NDArrayBool) -> PointCloudMask:
        """Bitwise xor between masks, or a mask and an array."""

        return self.__xor__(other)

    def __invert__(self: PointCloudMask) -> PointCloudMask:
        """Bitwise inversion of a mask."""

        return self.copy(~self.data)