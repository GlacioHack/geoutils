"""Module for PointCloud class."""

from __future__ import annotations

import os.path
import pathlib
import warnings
from typing import Any, Iterable, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from rasterio.coords import BoundingBox
from shapely.geometry.base import BaseGeometry

import geoutils as gu
from geoutils._typing import ArrayLike, NDArrayNum, Number
from geoutils.interface.gridding import _grid_pointcloud
from geoutils.interface.raster_point import _raster_to_pointcloud
from geoutils.raster.sampling import subsample_array

try:
    import laspy

    has_laspy = True
except ImportError:
    has_laspy = False


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


class PointCloud(gu.Vector):
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
        data_column: str | None = "z",
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
        self._bounds: BoundingBox | None = None
        self._data_column: str | None = None
        self._nb_points: int | None = None
        self._columns: pd.Index | None = None

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
                crs, nb_points, bounds, columns = _load_laspy_metadata(filename_or_dataset)
                self._name = filename_or_dataset
                self._crs = crs
                self._nb_points = nb_points
                self._columns = columns
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

    @property
    def all_columns(self) -> pd.Index:
        """Index of all columns of the point cloud, excluding the column of 2D point geometries."""
        # Overriding method in Vector
        if self.is_loaded:
            all_columns = super().columns
            all_columns_nongeom = all_columns[all_columns != "geometry"]
            return all_columns_nongeom
        else:
            return self._columns

    #####################################
    # NEW METHODS SPECIFIC TO POINT CLOUD
    #####################################

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

    def load(self, columns: Literal["all", "main"] | list[str] = "main"):
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
            columns = self.all_columns
        elif columns == "main":
            columns = [self.data_column]

        ds = _load_laspy_data(filename=self.name, columns=columns)
        self._ds = ds

    @classmethod
    def from_array(cls, array: NDArrayNum, crs: CRS, data_column: str | None = "z") -> PointCloud:
        """Create point cloud from a 3 x N or N x 3 array of X coordinate, Y coordinates and Z values."""

        # Check shape
        if array.shape[0] != 3 and array.shape[1] != 3:
            raise ValueError("Array must be of shape 3xN or Nx3.")

        # Make the first axis the one with size 3
        if array.shape[0] != 3:
            array_in = array.T
        else:
            array_in = array

        # Build geodataframe
        gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(x=array_in[0, :], y=array_in[1, :], crs=crs), data={data_column: array_in[2, :]}
        )

        return cls(filename_or_dataset=gdf, data_column=data_column)

    @classmethod
    def from_tuples(
        cls, tuples_xyz: Iterable[tuple[Number, Number, Number]], crs: CRS, data_column: str | None = "z"
    ) -> PointCloud:
        """Create point cloud from an iterable of 3-tuples (X coordinate, Y coordinate, Z value)."""

        return cls.from_array(np.array(tuples_xyz), crs=crs, data_column=data_column)

    @classmethod
    def from_xyz(cls, x: ArrayLike, y: ArrayLike, z: ArrayLike, crs: CRS, data_column: str | None = "z") -> PointCloud:
        """Create point cloud from three 1D array-like coordinates for X/Y/Z."""

        return cls.from_array(np.stack((x, y, z)), crs=crs, data_column=data_column)

    def to_array(self) -> NDArrayNum:
        """Convert point cloud to a 3 x N array of X coordinates, Y coordinates and Z values."""

        return np.stack((self.geometry.x.values, self.geometry.y.values, self.ds[self.data_column].values), axis=0)

    def to_tuples(self) -> Iterable[tuple[Number, Number, Number]]:
        """Convert point cloud to a list of 3-tuples (X coordinate, Y coordinate, Z value)."""

        return list(zip(self.geometry.x.values, self.geometry.y.values, self.ds[self.data_column].values))

    def to_xyz(self) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum]:
        """Convert point cloud to three 1D arrays of coordinates for X/Y/Z."""

        return self.geometry.x.values, self.geometry.y.values, self.ds[self.data_column].values

    def pointcloud_equal(self, other: PointCloud, **kwargs: Any):
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

    def grid(
        self,
        ref: gu.Raster | None,
        grid_coords: tuple[np.ndarray, np.ndarray] | None,
        resampling: Literal["nearest", "linear", "cubic"],
        dist_nodata_pixel: float = 1.0,
    ) -> gu.Raster:
        """Grid point cloud into a raster."""

        if isinstance(ref, gu.Raster):
            if grid_coords is None:
                warnings.warn(
                    "Both reference raster and grid coordinates were passed for gridding, "
                    "using only the reference raster."
                )
            grid_coords = ref.coords(grid=False)
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

        subsample = subsample_array(
            array=self.ds[self.data_column].values, subsample=subsample, return_indices=True, random_state=random_state
        )

        return PointCloud(self.ds[subsample])

    # @classmethod
    # def from_raster(cls, raster: gu.Raster) -> PointCloud:
    #     """Create a point cloud from a raster. Equivalent with Raster.to_pointcloud."""
    #
    #    pc = _raster_to_pointcloud(source_raster=raster)
