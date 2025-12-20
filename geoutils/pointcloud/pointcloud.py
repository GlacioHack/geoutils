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

import logging
import os.path
import pathlib
import warnings
from typing import Any, Callable, Iterable, Literal, TypeVar, overload

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import CRS
from rasterio.coords import BoundingBox
from rasterio.transform import from_origin
from shapely.geometry.base import BaseGeometry

import geoutils as gu
from geoutils import profiler
from geoutils._typing import ArrayLike, DTypeLike, NDArrayBool, NDArrayNum, Number
from geoutils.interface.gridding import _grid_pointcloud
from geoutils.raster.georeferencing import _coords
from geoutils.stats.sampling import subsample_array
from geoutils.stats.stats import _statistics

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

# This is a generic Vector-type (if subclasses are made, this will change appropriately)
PointCloudType = TypeVar("PointCloudType", bound="PointCloud")


def _load_laspy_data(filename: str, columns: list[str]) -> gpd.GeoDataFrame:
    """Load point cloud data from LAS/LAZ/COPC file as a geodataframe."""

    # Read file
    las = laspy.read(filename)

    # Get data from main Z column and other requested columns
    columns_no_z = [c for c in columns if c != "Z"]
    data = np.vstack([las.z] + [las[n] for n in columns_no_z]).T
    column_names = ["Z"] + columns_no_z

    # Build geodataframe
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=las.x, y=las.y, crs=las.header.parse_crs(prefer_wkt=False)),
        data=data,
        columns=column_names,
    )

    return gdf


def _load_laspy_metadata(
    filename: str,
) -> tuple[CRS, int, BoundingBox, pd.Index]:
    """Load point cloud metadata from LAS/LAZ/COPC file."""

    with laspy.open(filename) as f:

        # Parse CRS, point count and bounds
        crs = f.header.parse_crs(prefer_wkt=False)
        nb_points = f.header.point_count
        bounds = BoundingBox(left=f.header.x_min, right=f.header.x_max, bottom=f.header.y_min, top=f.header.y_max)

        # Parse column names as pandas index, removing X/Y that will be transformed into a geometry
        columns_names = list(f.header.point_format.dimension_names)
        columns_names = [c for c in columns_names if c not in ["X", "Y"]]
        columns_names = pd.Index(columns_names)

    return crs, nb_points, bounds, columns_names


def _write_laspy(
    filename: str | pathlib.Path,
    pc: gpd.GeoDataFrame,
    data_column: str | None,
    version: Any = None,
    point_format: Any = None,
    offsets: tuple[float, float, float] = None,
    scales: tuple[float, float, float] = None,
    **kwargs: Any,
) -> None:
    """Write a point cloud geodataframe to a LAS/LAZ/COPC file."""

    # Initiate header with user arguments
    header = laspy.LasHeader(version=version, point_format=point_format)
    if scales is not None:
        header.scales = np.array(scales)
    if offsets is not None:
        header.offsets = np.array(offsets)
    for k, v in kwargs.items():
        setattr(header, k, v)

    # Adding extra dimensions for auxiliary variables
    aux_columns = [c for c in pc.columns if c not in [data_column, "geometry"]]
    for c in aux_columns:
        header.add_extra_dim(laspy.ExtraBytesParams(name=c, type=pc[c].dtype))

    las = laspy.LasData(header)

    # The las x,y,z will be automatically scaled into X,Y,Z based on the header scales
    las.x = pc.geometry.x.values
    las.y = pc.geometry.y.values
    if data_column is not None:
        las.z = pc[data_column].values
    else:
        las.z = pc.geometry.z.values

    # Add auxiliary columns
    for c in aux_columns:
        setattr(las, c, pc[c].values)

    las.write(filename)

    return


def _cast_numeric_array_pointcloud(
    pc: PointCloudType, other: PointCloudType | NDArrayNum | Number, operation_name: str
) -> NDArrayNum | Number:
    """
    Cast a point cloud and another point cloud or array or number to arrays with proper metadata, or raise an error
    message.

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
        if other.squeeze().ndim == 1 and other.squeeze().shape[0] == pc.point_count:
            pass
        else:
            raise ValueError(
                "The array must be 1-dimensional with the same number of points as the point cloud for "
                + operation_name
                + "."
            )

    else:
        other_data = other  # type: ignore

    return other_data


class PointCloud(gu.Vector):  # type: ignore[misc]
    """
    The georeferenced point cloud.

    A point cloud is a vector of 2D point geometries associated to numeric values from a main data column, and can
    also contain auxiliary data columns.

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

    @profiler.profile("geoutils.pointcloud.pointcloud.__init__", memprof=True)  # type: ignore
    def __init__(
        self,
        filename_or_dataset: str | pathlib.Path | gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry,
        data_column: str | None = None,
    ):
        """
        Instantiate a point cloud from either a data column name and a vector (filename, GeoPandas dataframe or series,
        or a Shapely geometry), or only with a point cloud file type.

        :param filename_or_dataset: Path to vector file, or GeoPandas dataframe or series, or Shapely geometry.
        :param data_column: Name of main data column defining the point cloud (not required for LAS/LAZ formats).
        """

        self._ds: gpd.GeoDataFrame | None = None
        self._name: str | None = None
        self._crs: CRS | None = None
        self._data_column: str | None = None
        self._bounds: BoundingBox
        self._data: NDArrayNum
        self._nb_points: int
        self.__nongeo_columns: pd.Index

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
                # No need to pass a data column for LAS/LAZ file, as Z is the logical default
                if data_column is None:
                    data_column = "Z"
                # Load only metadata, and not the data
                fn = filename_or_dataset if isinstance(filename_or_dataset, str) else filename_or_dataset.name
                crs, nb_points, bounds, columns = _load_laspy_metadata(fn)
                self._name = fn
                self._crs = crs
                self._nb_points = nb_points
                self.__nongeo_columns = columns
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

        # Set data column name based on user input
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

        # Overriding method in Vector in case dataset is not loaded
        if self.is_loaded:
            return super().crs
        # Return CRS on disk
        else:
            return self._crs

    @property
    def bounds(self) -> BoundingBox:
        # Overriding method in Vector in case dataset is not loaded
        if self.is_loaded:
            return super().bounds
        # Return bounds on disk
        else:
            return self._bounds

    def columns(self) -> pd.Index:
        # Overriding method in Vector in case dataset is not loaded
        if self.is_loaded:
            return super().columns
        # Return columns on disk (adding a placeholder geometry to replace X/Y)
        else:
            return pd.Index(list(self._nongeo_columns) + ["geometry"])

    #####################################
    # NEW METHODS SPECIFIC TO POINT CLOUD
    #####################################

    @property
    def _has_z(self) -> bool:
        """Whether the point geometries all have a Z coordinate or not."""

        return all(p.has_z for p in self.ds.geometry) if len(self.ds.geometry) > 0 else False

    @property
    def data(self) -> NDArrayNum:
        """
        Data of the point cloud.

        Points to either the Z axis of the point geometries, or the associated data column of the geodataframe.
        """
        # Triggers the loading mechanism through self.ds
        if self.data_column is not None:
            return self.ds[self.data_column].values
        else:
            return self.geometry.z.values

    @data.setter
    def data(self, new_data: NDArrayNum) -> None:
        """Set new data for the point cloud."""

        if self.data_column is not None:
            self.ds[self.data_column] = new_data
        else:
            self.ds.geometry = gpd.points_from_xy(x=self.geometry.x, y=self.geometry.y, z=new_data, crs=self.crs)

    @property
    def _nongeo_columns(self) -> pd.Index:
        """Columns of the point cloud excluding the column of 2D point geometries."""
        # Overriding method in Vector
        if self.is_loaded:
            nongeo_columns = super().columns
            nongeo_columns = nongeo_columns[nongeo_columns != "geometry"]
            return nongeo_columns
        else:
            return self.__nongeo_columns

    @property
    def data_column(self) -> str | None:
        """
        Name of data column of the point cloud.

        Can be None if point geometries are 3D.
        """
        return self._data_column

    @data_column.setter
    def data_column(self, new_data_column: str) -> None:
        self.set_data_column(new_data_column=new_data_column)

    def set_data_column(self, new_data_column: str | None) -> None:
        """Set new column as point cloud data column."""

        # If point geometries are 3D, only for loaded data (otherwise _has_z would load data)
        if self.is_loaded:
            if self._has_z:
                if new_data_column is None:
                    self._data_column = None
                    return
                else:
                    warnings.warn(
                        f"Overriding 3D points with with data column '{new_data_column}'. Set data_column "
                        f"to None to use the 3D point geometries instead."
                    )

        # If point geometries are 2D and the data column is undefined
        if new_data_column is None:
            raise ValueError("A data column name must be passed for a point cloud with 2D point geometries.")

        # If 2D and data column is defined, check that it exists
        if new_data_column not in self._nongeo_columns:
            raise ValueError(
                f"Data column {new_data_column} not found among columns. Available columns "
                f"are: {', '.join(self._nongeo_columns)}."
            )

        # Set data column name
        self._data_column = new_data_column

    @property
    def is_loaded(self) -> bool:
        """Whether the point cloud data is loaded."""
        return self._ds is not None

    @property
    def point_count(self) -> int:
        """Number of points in the point cloud."""
        # New method for point cloud
        if self.is_loaded:
            return len(self.ds)
        else:
            return self._nb_points

    @property
    def is_mask(self) -> bool:
        """Whether the point cloud mask is a mask (boolean type)."""

        return np.dtype(self.data.dtype) == np.bool_

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
            columns_to_load = self._nongeo_columns
        elif columns == "main":
            columns_to_load = [self.data_column]
        else:
            columns_to_load = columns

        ds = _load_laspy_data(filename=self.name, columns=columns_to_load)
        self._ds = ds

    @overload
    def astype(
        self: PointCloud, dtype: DTypeLike, convert_coords: bool = False, *, inplace: Literal[False] = False
    ) -> PointCloud: ...

    @overload
    def astype(self: PointCloud, dtype: DTypeLike, convert_coords: bool = False, *, inplace: Literal[True]) -> None: ...

    @overload
    def astype(
        self: PointCloud, dtype: DTypeLike, convert_coords: bool = False, *, inplace: bool = False
    ) -> PointCloud | None: ...

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

    def copy(self: PointCloud, new_array: NDArrayNum | NDArrayBool | None = None) -> PointCloud:
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
            if not (new_array.ndim == 1 and new_array.shape[0] == self.point_count):
                raise ValueError(
                    "New data array must be 1-dimensional with the same number of points as the point "
                    "cloud being copied."
                )
            data = new_array
        else:
            data = self.data.copy()

        # Send to from_xyz
        cp = self.from_xyz(
            x=self.geometry.x.values,
            y=self.geometry.y.values,
            z=data,
            crs=self.crs,
            data_column=self.data_column,
            use_z=self._has_z and self.data_column is None,
        )

        return cp

    def to_las(
        self,
        filename: str | pathlib.Path,
        version: Any = None,
        point_format: Any = None,
        offsets: tuple[float, float, float] = None,
        scales: tuple[float, float, float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Write the point cloud to LAS/LAZ/COPC file.

        :param filename: Name of output file.
        :param version: LAS/LAZ/COPC version.
        :param point_format: Point format.
        :param offsets: Offsets for X/Y/Z.
        :param scales: Scales for X/Y/Z.
        :param kwargs: Other keyword arguments to set the LAS file header (e.g., "offsets", "scales").
        """

        _write_laspy(
            filename=filename,
            pc=self.ds,
            data_column=self.data_column,
            version=version,
            point_format=point_format,
            offsets=offsets,
            scales=scales,
            **kwargs,
        )

    @classmethod
    def from_xyz(
        cls, x: ArrayLike, y: ArrayLike, z: ArrayLike, crs: CRS, data_column: str | None = None, use_z: bool = False
    ) -> PointCloud:
        """
        Create point cloud from three 1D array-like coordinates for X/Y/Z.

        Note that this is the most modular method to create a point cloud, as it allows to specify different data
        types for the different coordinates or columns.

        :param x: X coordinates of point cloud.
        :param y: Y coordinates of point cloud.
        :param z: Z values of point cloud.
        :param crs: Coordinate reference system.
        :param data_column: Data column name to associate to 2D point geometries (defaults to "z" if none is passed).
        :param use_z: Use 3D point geometries with Z coordinates instead of a data column.

        :return Point cloud.
        """

        # Build geodataframe
        if not use_z:
            data_column = data_column if data_column is not None else "z"
            gdf = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(x=np.atleast_1d(x), y=np.atleast_1d(y), crs=crs),
                data={data_column: np.atleast_1d(z)},
            )
        else:
            data_column = None
            gdf = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(x=np.atleast_1d(x), y=np.atleast_1d(y), z=np.atleast_1d(z), crs=crs),
            )

        # If the data was transformed into boolean, re-initialize as a Mask subclass
        # Typing: we can specify this behaviour in @overload once we add the NumPy plugin of MyPy
        if np.atleast_1d(z)[0].dtype == bool:
            return PointCloud(filename_or_dataset=gdf, data_column=data_column)  # type: ignore
        # Otherwise, keep as a given PointCloudType subclass
        else:
            return cls(filename_or_dataset=gdf, data_column=data_column)

    @classmethod
    def from_array(cls, data: NDArrayNum, crs: CRS, data_column: str | None = None, use_z: bool = False) -> PointCloud:
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

        return cls.from_xyz(x=data[0, :], y=data[1, :], z=data[2, :], crs=crs, data_column=data_column, use_z=use_z)

    @classmethod
    def from_tuples(
        cls,
        tuples_xyz: Iterable[tuple[Number, Number, Number]],
        crs: CRS,
        data_column: str | None = None,
        use_z: bool = False,
    ) -> PointCloud:
        """
        Create point cloud from an iterable of 3-tuples (X coordinate, Y coordinate, Z value).

        :param tuples_xyz: Point cloud coordinates and data as an iterable of 3-tuples.
        :param crs: Coordinate reference system of point cloud.
        :param data_column: Data column of point cloud.

        :return Point cloud.
        """

        return cls.from_array(np.array(tuples_xyz), crs=crs, data_column=data_column, use_z=use_z)

    def to_xyz(self) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum]:
        """Convert point cloud to three 1D arrays of coordinates for X/Y/Z."""

        return self.geometry.x.values, self.geometry.y.values, self.data

    def to_array(self) -> NDArrayNum:
        """Convert point cloud to a 3 x N array of X coordinates, Y coordinates and Z values."""

        return np.stack((self.geometry.x.values, self.geometry.y.values, self.data), axis=0)

    def to_tuples(self) -> Iterable[tuple[Number, Number, Number]]:
        """Convert point cloud to a list of 3-tuples (X coordinate, Y coordinate, Z value)."""

        return list(zip(self.geometry.x.values, self.geometry.y.values, self.data))

    def __getitem__(self, index: PointCloud | NDArrayBool | Any) -> PointCloud | Any:
        """
        Index the point cloud.

        In addition to all index types supported by GeoPandas, also supports a point cloud mask of same georeferencing.
        """

        if isinstance(index, PointCloud) or isinstance(index, np.ndarray):
            # If input is mask with the same shape and georeferencing, convert to ndarray
            _cast_numeric_array_pointcloud(self, index, operation_name="an indexing operation")  # type: ignore
            if isinstance(index, PointCloud):
                ind = index.data
            else:
                ind = index  # type: ignore
            return PointCloud(super().__getitem__(ind), data_column=self.data_column)

        # Otherwise, use index and leave it to GeoPandas
        else:
            ind = index  # type: ignore
            return super().__getitem__(ind)

    def __setitem__(self, index: Any, assign: Any) -> None:
        """
        Perform index assignment on the point cloud.
        """

        # Let the vector class do the job
        super().__setitem__(index, assign)

        return None

    def __array_ufunc__(
        self,
        ufunc: Callable[[NDArrayNum | tuple[NDArrayNum, NDArrayNum]], NDArrayNum | tuple[NDArrayNum, NDArrayNum]],
        method: str,
        *inputs: tuple[PointCloud]
        | tuple[PointCloud, PointCloud]
        | tuple[NDArrayNum, PointCloud]
        | tuple[PointCloud, NDArrayNum],
        **kwargs: Any,
    ) -> PointCloud | tuple[PointCloud, PointCloud]:
        """
        Method to cast NumPy universal functions directly on PointCloud classes, by passing to the masked array.
        This function basically applies the ufunc (with its method and kwargs) to .data, and rebuilds the PointCloud
        from self.__class__. The cases separate the number of input nin and output nout, to properly feed .data and
        return PointCloud objects.
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
                pc = inputs[1]  # type: ignore
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

    def plot(
        self,
        column: str | None = None,
        ref_crs: gu.Raster | gu.Vector | gpd.GeoDataFrame | str | CRS | int | None = None,
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
        Plot the point cloud.

        This method is a wrapper to geopandas.GeoDataFrame.plot. Any \*\*kwargs which
        you give this method will be passed to it.

        :param column: Column to plot. Default is the data column of the point cloud.
        :param ref_crs: Coordinate reference system to match when plotting.
        :param cmap: Colormap to use. Default is plt.rcParams['image.cmap'].
        :param vmin: Colorbar minimum value. Default is data min.
        :param vmax: Colorbar maximum value. Default is data max.
        :param alpha: Transparency of raster and colorbar.
        :param cbar_title: Colorbar label. Default is None.
        :param add_cbar: Set to True to display a colorbar. Default is True if a "column" argument is passed.
        :param ax: A figure ax to be used for plotting. If None, will plot on current axes. If "new",
            will create a new axis.
        :param return_axes: Whether to return axes.

        :returns: None, or (ax, caxes) if return_axes is True
        """

        # Ensure that the vector is in the same crs as a reference
        if isinstance(ref_crs, (gu.Raster, gu.Vector, gpd.GeoDataFrame, str)):
            vect_reproj = self.reproject(ref=ref_crs)
        elif isinstance(ref_crs, (CRS, int)):
            vect_reproj = self.reproject(crs=ref_crs)
        else:
            vect_reproj = self

        if column is None:
            column = self.data_column

        # Create axes, or get current ones by default (like in matplotlib)
        if ax is None:
            ax0 = plt.gca()
        elif isinstance(ax, str) and ax.lower() == "new":
            _, ax0 = plt.subplots()
        elif isinstance(ax, matplotlib.axes.Axes):
            ax0 = ax
        else:
            raise ValueError("ax must be a matplotlib.axes.Axes instance, 'new' or None.")

        # Set add_cbar depending on column argument
        if add_cbar:
            add_cbar = True
        else:
            add_cbar = False

        # Update with this function's arguments
        if add_cbar:
            legend = True
        else:
            legend = False

        if "legend" in list(kwargs.keys()):
            legend = kwargs.pop("legend")

        # Get colormap arguments that might have been passed in the keyword args
        if "legend_kwds" in list(kwargs.keys()) and legend:
            legend_kwds = kwargs.pop("legend_kwds")
            if cbar_title is not None:
                legend_kwds.update({"label": cbar_title})  # Pad updates depending on figsize during plot,
        else:
            if cbar_title is not None:
                legend_kwds = {"label": cbar_title}
            else:
                legend_kwds = None

        # Add colorbar
        if add_cbar or cbar_title:
            divider = make_axes_locatable(ax0)
            cax = divider.append_axes("right", size="5%", pad="2%")
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cbar = matplotlib.colorbar.ColorbarBase(
                cax, cmap=cmap, norm=norm
            )  # , orientation="horizontal", ticklocation="top")
            cbar.solids.set_alpha(alpha)
        else:
            cax = None
            cbar = None

        # Plot
        vect_reproj.ds.plot(
            ax=ax0,
            cax=cax,
            column=column,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            legend=legend,
            legend_kwds=legend_kwds,
            **kwargs,
        )
        plt.sca(ax0)

        # If returning axes
        if return_axes:
            return ax0, cax
        else:
            return None

    def __add__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        Sum two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data + other_data
        return self.copy(new_array=out_data)  # type: ignore

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
        return self.copy(new_array=out_data)  # type: ignore

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __rsub__(self: PointCloud, other: NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Subtract two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        For when other is first item in the operation (e.g. 1 - rst).
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = other_data - self.data
        return self.copy(new_array=out_data)  # type: ignore

    def __mul__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        Multiply two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data * other_data
        return self.copy(new_array=out_data)  # type: ignore

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __rmul__(self: PointCloud, other: NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Multiply two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        For when other is first item in the operation (e.g. 2 * rst).
        """
        return self.__mul__(other)  # type: ignore

    def __truediv__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        True division of two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data / other_data
        return self.copy(new_array=out_data)  # type: ignore

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __rtruediv__(self: PointCloud, other: NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        True division of two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        For when other is first item in the operation (e.g. 1/rst).
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = other_data / self.data
        return self.copy(new_array=out_data)  # type: ignore

    def __floordiv__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        Floor division of two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data // other_data  # type: ignore
        return self.copy(new_array=out_data)

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __rfloordiv__(self: PointCloud, other: NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Floor division of two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        For when other is first item in the operation (e.g. 1/rst).
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = other_data // self.data  # type: ignore
        return self.copy(new_array=out_data)

    def __mod__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        Modulo of two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data % other_data  # type: ignore
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
        Element-wise equality of two point clouds, or a point cloud and a numpy array, or a point cloud and single
        number.

        This operation casts the result into a mask (boolean Raster).

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data == other_data
        return self.copy(new_array=out_data)

    def __ne__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Element-wise negation of two point clouds, or a point cloud and a numpy array, or a point cloud and single
        number.

        This operation casts the result into a mask (boolean Raster).

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data != other_data
        return self.copy(new_array=out_data)

    def __lt__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Element-wise lower than comparison of two point clouds, or a point cloud and a numpy array,
        or a point cloud and single number.

        This operation casts the result into a mask (boolean Raster).

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data < other_data
        return self.copy(new_array=out_data)

    def __le__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Element-wise lower or equal comparison of two point clouds, or a point cloud and a numpy array,
        or a point cloud and single number.

        This operation casts the result into a mask (boolean Raster).

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data <= other_data
        return self.copy(new_array=out_data)

    def __gt__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Element-wise greater than comparison of two point clouds, or a point cloud and a numpy array,
        or a point cloud and single number.

        This operation casts the result into a mask (boolean Raster).

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data > other_data
        return self.copy(new_array=out_data)

    def __ge__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Element-wise greater or equal comparison of two point clouds, or a point cloud and a numpy array,
        or a point cloud and single number.

        This operation casts the result into a mask (boolean Raster).

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data >= other_data
        return self.copy(new_array=out_data)

    def __and__(self: PointCloud, other: PointCloud | NDArrayBool) -> PointCloud:
        """Bitwise and between masks, or a mask and an array."""
        other_data = _cast_numeric_array_pointcloud(
            self, other, operation_name="an arithmetic operation"  # type: ignore
        )

        return self.copy(self.data & other_data)  # type: ignore

    def __rand__(self: PointCloud, other: PointCloud | NDArrayBool) -> PointCloud:
        """Bitwise and between masks, or a mask and an array."""

        return self.__and__(other)

    def __or__(self: PointCloud, other: PointCloud | NDArrayBool) -> PointCloud:
        """Bitwise or between masks, or a mask and an array."""

        other_data = _cast_numeric_array_pointcloud(
            self, other, operation_name="an arithmetic operation"  # type: ignore
        )

        return self.copy(self.data | other_data)  # type: ignore

    def __ror__(self: PointCloud, other: PointCloud | NDArrayBool) -> PointCloud:
        """Bitwise or between masks, or a mask and an array."""

        return self.__or__(other)

    def __xor__(self: PointCloud, other: PointCloud | NDArrayBool) -> PointCloud:
        """Bitwise xor between masks, or a mask and an array."""

        other_data = _cast_numeric_array_pointcloud(
            self, other, operation_name="an arithmetic operation"  # type: ignore
        )

        return self.copy(self.data ^ other_data)  # type: ignore

    def __rxor__(self: PointCloud, other: PointCloud | NDArrayBool) -> PointCloud:
        """Bitwise xor between masks, or a mask and an array."""

        return self.__xor__(other)

    def __invert__(self: PointCloud) -> PointCloud:
        """Bitwise inversion of a mask."""

        return self.copy(~self.data)

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

        return all(
            [
                self.crs == pc.crs,
                np.array_equal(self.geometry.x.values, pc.geometry.x.values),
                np.array_equal(self.geometry.y.values, pc.geometry.y.values),
            ]
        )

    @overload
    def get_stats(
        self,
        stats_name: str | Callable[[NDArrayNum], np.floating[Any]],
    ) -> np.floating[Any]: ...

    @overload
    def get_stats(
        self,
        stats_name: list[str | Callable[[NDArrayNum], np.floating[Any]]] | None = None,
    ) -> dict[str, np.floating[Any]]: ...

    @profiler.profile("geoutils.pointcloud.pointcloud.get_stats", memprof=True)  # type: ignore
    def get_stats(
        self,
        stats_name: (
            str | Callable[[NDArrayNum], np.floating[Any]] | list[str | Callable[[NDArrayNum], np.floating[Any]]] | None
        ) = None,
    ) -> np.floating[Any] | dict[str, np.floating[Any]]:
        """
        Retrieve specified statistics or all available statistics for the point cloud data. Allows passing custom
        callables to calculate custom stats.

        Common statistics for an N-D array :

        - Mean: arithmetic mean of the data, ignoring masked values.
        - Median: middle value when the valid data points are sorted in increasing order, ignoring masked values.
        - Max: maximum value among the data, ignoring masked values.
        - Min: minimum value among the data, ignoring masked values.
        - Sum: sum of all data, ignoring masked values.
        - Sum of squares: sum of the squares of all data, ignoring masked values.
        - 90th percentile: point below which 90% of the data falls, ignoring masked values.
        - IQR (Interquartile Range): difference between the 75th and 25th percentile of a dataset,
        ignoring masked values.
        - LE90 (Linear Error with 90% confidence): difference between the 95th and 5th percentiles of a dataset,
        representing the range within which 90% of the data points lie. Ignore masked values.
        - NMAD (Normalized Median Absolute Deviation): robust measure of variability in the data, less sensitive to
        outliers compared to standard deviation. Ignore masked values.
        - RMSE (Root Mean Square Error): commonly used to express the magnitude of errors or variability and can give
        insight into the spread of the data. Only relevant when the raster represents a difference of two objects.
        Ignore masked values.
        - Std (Standard deviation): measures the spread or dispersion of the data around the mean,
        ignoring masked values.
        - Valid count: number of finite data points in the array. It counts the non-masked elements.
        - Total count: total size of the raster.
        - Percentage valid points: ratio between Valid count and Total count.

        For all statistics up to "Std", functions from numpy.ma module are used (directly or in the calculation) in case
        of a masked array, numpy module otherwise.

        "Valid count" represents all non zero and not masked pixels in the input data (final_count_nonzero).
        Numpy Masked functions is used is this case or if the Point Cloud was already a masked array.
        Percentage valid points is calculated accordingly.

        If an inlier mask is passed:
        - Total inlier count: number of data points in the inlier mask.
        - Valid inlier count: number of unmasked data points in the array after applying the inlier mask.
        - Percentage inlier points: ratio between Valid inlier count and Valid count. Useful for classification
        statistics.
        - Percentage valid inlier points: ratio between Valid inlier count and Total inlier count.

        They are all computed based on the previously stated final_count_nonzero.

        Callable functions are supported as well.

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

        # Given list or all attributes to compute if None
        if isinstance(stats_name, list) or stats_name is None:
            return _statistics(data, stats_name)  # type: ignore
        else:
            # Single attribute to compute
            if isinstance(stats_name, str):
                return _statistics(data, [stats_name])[stats_name]  # type: ignore
            elif callable(stats_name):
                return stats_name(data)  # type: ignore
            else:
                logging.warning("Statistic name '%s' is a not recognized string", stats_name)

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

    @profiler.profile("geoutils.pointcloud.pointcloud.subsample", memprof=True)  # type: ignore
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

    @profiler.profile("geoutils.pointcloud.pointcloud.grid", memprof=True)  # type: ignore
    def grid(
        self,
        ref: gu.Raster | None = None,
        grid_coords: tuple[NDArrayNum, NDArrayNum] | None = None,
        res: float | tuple[float, float] | None = None,
        resampling: Literal["nearest", "linear", "cubic"] = "linear",
        dist_nodata_pixel: float = 1.0,
        nodata: int | float = -9999,
    ) -> gu.Raster:
        """
        Grid point cloud into a raster.

        Output grid can be defined either by passing a reference raster to match, or by passing output grid coordinates
        for X/Y (must be regular in each dimension), or by specifying an output grid resolution (which uses the
        upper-left bounds of point cloud to define the start of the grid).

        :param ref: Reference raster to match (if output grid coordinates or output resolution undefined).
        :param grid_coords: Output grid coordinates in X and Y (if reference raster or output resolution undefined).
        :param res: Output resolution (if reference raster or output grid coordinates undefined).
        :param resampling: Resampling method within delauney triangles (defaults to linear).
        :param dist_nodata_pixel: Distance from the point cloud after which grid cells are filled by nodata values,
            expressed in number of pixels.
        :param nodata: Nodata value of output raster (defaults to -9999).

        :return: Raster from gridded point cloud.
        """

        if isinstance(ref, gu.Raster):
            if grid_coords is not None:
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

        return gu.Raster.from_array(data=array, transform=transform, crs=self.crs, nodata=nodata)
