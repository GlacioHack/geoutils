"""Module for PointCloud class."""
from __future__ import annotations

import os.path
import warnings
from typing import Iterable, Literal
import pathlib

from pyproj import CRS
import numpy as np
import geopandas as gpd
from rasterio.coords import BoundingBox
from shapely.geometry.base import BaseGeometry

import geoutils as gu
from geoutils.interface.gridding import _grid_pointcloud
from geoutils._typing import Number

try:
    import laspy
    has_laspy = True
except ImportError:
    has_laspy = False


def _load_laspy_data(filename: str, main_data_column: str, auxiliary_data_columns: list[str] | None = None) -> gpd.GeoDataFrame:
    """Load point cloud data from LAS/LAZ/COPC file."""

    las = laspy.read(filename)

    # Get data from requested columns
    if auxiliary_data_columns is not None:
        data = [las[n] for n in auxiliary_data_columns]
    else:
        data = []
    data.insert(0, las[main_data_column])
    data = np.vstack(data).transpose()

    # Build geodataframe
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=las.x, y=las.y,
                                                       crs=las.header.parse_crs(prefer_wkt=False)),
                           data=data)

    return gdf



def _load_laspy_metadata(filename: str, ) -> tuple[CRS, int, BoundingBox, list[str]]:
    """Load point cloud metadata from LAS/LAZ/COPC file."""

    with laspy.open(filename) as f:

        crs = f.header.parse_crs(prefer_wkt=False)
        nb_points = f.header.point_count
        bounds = BoundingBox(left=f.header.x_min, right=f.header.x_max, bottom=f.header.y_min, top=f.header.y_max)
        columns_names = list(f.header.point_format.dimension_names)

    return crs, nb_points, bounds, columns_names


# def _write_laspy(filename: str, pc: gpd.GeoDataFrame):
#     """Write a point cloud dataset as LAS/LAZ/COPC."""
#
#     with laspy.open(filename) as f:
#         new_hdr = laspy.LasHeader(version="1.4", point_format=6)
#         # You can set the scales and offsets to values tha suits your data
#         new_hdr.scales = np.array([1.0, 0.5, 0.1])
#         new_las = laspy.LasData(header = new_hdr, points=)
#
#     return


class PointCloud(gu.Vector):
    """
    The georeferenced point cloud.

    A point cloud is a vector of 2D point geometries associated to values from a main data column, optionally with
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

    def __init__(self,
                 filename_or_dataset: str | pathlib.Path | gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry,
                 data_column: str | None = None):
        """
        Instantiate a point cloud from either a data column name and a vector (filename, GeoPandas dataframe or series,
        or a Shapely geometry), or only with a point cloud file type.

        :param filename_or_dataset: Path to vector file, or GeoPandas dataframe or series, or Shapely geometry.
        :param data_column: Name of data column defining the point cloud.
        """

        self._ds: gpd.GeoDataFrame | None = None
        self._name: str | None = None
        self._crs: CRS | None = None
        self._bounds: BoundingBox | None = None
        self._data_column: str | None = None
        self._nb_points: int | None = None
        self._columns: list[str] | None = None

        # If PointCloud is passed, simply point back to PointCloud
        if isinstance(filename_or_dataset, PointCloud):
            for key in filename_or_dataset.__dict__:
                setattr(self, key, filename_or_dataset.__dict__[key])
            return
        # For filename, rely on parent Vector class or LAS file reader
        else:
            if isinstance(filename_or_dataset, (str, pathlib.Path)) and \
                    os.path.splitext(filename_or_dataset)[-1] in [".las", ".laz"]:
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
                    raise ValueError("This vector file contains non-point geometries, "
                                     "cannot be instantiated as a point cloud.")

        if data_column not in self.columns:
            raise ValueError(f"Data column {data_column} not found among columns, available columns "
                             f"are: {', '.join(self.columns)}.")
        self._data_column = data_column

    @property
    def ds(self) -> gpd.GeoDataFrame:
        if not self.is_loaded:
            self.load()
        return self._ds  # type: ignore

    @ds.setter
    def ds(self, new_ds: gpd.GeoDataFrame | gpd.GeoSeries) -> None:
        """Set a new geodataframe."""

        if isinstance(new_ds, gpd.GeoDataFrame):
            self._ds = new_ds
        elif isinstance(new_ds, gpd.GeoSeries):
            self._ds = gpd.GeoDataFrame(geometry=new_ds)
        else:
            raise ValueError("The dataset of a vector must be set with a GeoSeries or a GeoDataFrame.")

    @property
    def crs(self) -> CRS:
        """Coordinate reference system of the vector."""
        if self._ds is not None:
            return self.ds.crs
        else:
            return self._crs

    def bounds(self) -> BoundingBox:
        if self._ds is not None:
            return super().bounds
        else:
            return self._bounds

    def load(self):
        """Load point cloud from disk (only supported for LAS files)."""

        ds = _load_laspy_data(filename=self.name, main_data_column=self.data_column)
        self._ds = ds

    @property
    def is_loaded(self) -> bool:
        """Whether the point cloud data is loaded"""
        return self._ds is not None

    @property
    def data_column(self) -> str:
        """Name of main data column of the point cloud."""
        return self._data_column

    @property
    def columns(self) -> list[str]:
        """Name of auxiliary data columns of the point cloud."""
        if self.is_loaded:
            return self.ds.columns
        else:
            return self._columns

    @classmethod
    def from_array(cls, array: np.ndarray, crs: CRS, data_column: str | None = "z") -> PointCloud:
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
        gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=array_in[0, :], y=array_in[1, :], crs=crs),
                               data={data_column: array[2, :]})

        return cls(filename_or_dataset=gdf, data_column=data_column)


    @classmethod
    def from_tuples(cls, tuples: Iterable[tuple[Number, Number, Number]], crs: CRS, data_column: str | None = None):
        """Create point cloud from a N-size list of tuples (X coordinate, Y coordinate, Z value)."""

        cls.from_array(np.array(zip(*tuples)), crs=crs, data_column=data_column)


    def to_array(self):
        """Convert point cloud to a 3 x N array of X coordinates, Y coordinates and Z values."""

        return np.stack((self.geometry.x.values, self.geometry.y.values, self.ds[self.data_column].values), axis=0)

    def to_tuples(self):
        """Convert point cloud to a N-size list of tuples (X coordinate, Y coordinate, Z value)."""

        return self.geometry.x.values, self.geometry.y.values, self.ds[self.data_column].values

    def grid(self,
             ref: gu.Raster | None,
             grid_coords: tuple[np.ndarray, np.ndarray] | None,
             resampling: Literal["nearest", "linear", "cubic"],
             dist_nodata_pixel: float = 1.) -> gu.Raster:
        """Grid point cloud into a raster."""

        if isinstance(ref, gu.Raster):
            if grid_coords is None:
                warnings.warn("Both reference raster and grid coordinates were passed for gridding, "
                              "using only the reference raster.")
            grid_coords = ref.coords(grid=False)
        else:
            grid_coords = grid_coords

        array, transform = _grid_pointcloud(self.ds, grid_coords=grid_coords, data_column_name=self.data_column,
                                 resampling=resampling, dist_nodata_pixel=dist_nodata_pixel)

        return gu.Raster.from_array(data=array, transform=transform, crs=self.crs, nodata=None)

    # def subsample(self) -> PointCloud | NDArrayf:
    #
    # @classmethod
    # def from_raster(cls, raster: gu.Raster) -> PointCloud:
    #     """Create a point cloud from a raster. Equivalent with Raster.to_pointcloud."""
    #
    #    pc = _raster_to_pointcloud(source_raster=raster)
