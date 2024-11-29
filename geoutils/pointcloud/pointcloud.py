"""Module for PointCloud class."""
from __future__ import annotations

import warnings
from typing import Any, Iterable, Literal
import pathlib

from pyproj import CRS
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry.base import BaseGeometry

import geoutils as gu
from geoutils.interface.gridding import _grid_pointcloud
from geoutils._typing import Number

try:
    import laspy
    has_laspy = True
except ImportError:
    has_laspy = False


def _read_pdal(filename: str, **kwargs: Any) -> gpd.GeoDataFrame:
    """Read a point cloud dataset with PDAL."""

    # Basic json command to read an entire file
    json_string = f"""
    [
        "{filename}"
    ]
    """

    # Run and extract array as dataframe
    pipeline = pdal.Pipeline(json_string)
    pipeline.execute()
    df = pd.DataFrame(pipeline.arrays[0])

    # Build geodataframe from 2D points and data table
    crs = CRS.from_wkt(pipeline.srswkt2)
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=df["X"], y=df["Y"], crs=crs), data=df.iloc[:, 2:])

    return gdf

def _write_pdal(filename: str, **kwargs):
    """Write a point cloud dataset with PDAL."""

    return


class PointCloud(gu.Vector):
    """
    The georeferenced point cloud.

    A point cloud is a vector of 2D point geometries associated to values from a main data column, with access
    to values from auxiliary data columns.

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

    _data_column: str | None

    def __init__(self,
                 filename_or_dataset: str | pathlib.Path | gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry,
                 data_column: str | None = None,):
        """
        Instantiate a point cloud from either a data column name and a vector (filename, GeoPandas dataframe or series,
        or a Shapely geometry), or only with a point cloud file type.

        :param filename_or_dataset: Path to file, or GeoPandas dataframe or series, or Shapely geometry.
        :param data_column: Name of data column defining the point cloud.
        """

        # If PointCloud is passed, simply point back to PointCloud
        if isinstance(filename_or_dataset, PointCloud):
            for key in filename_or_dataset.__dict__:
                setattr(self, key, filename_or_dataset.__dict__[key])
            return
        # Else rely on parent Vector class options (including raised errors)
        else:
            super().__init__(filename_or_dataset)

        if data_column not in self.ds.columns():
            raise ValueError(f"Data column {data_column} not found in vector file, available columns "
                             f"are: {', '.join(self.ds.columns)}.")
        self._data_column = data_column

    @property
    def data(self):
        return self.ds[self.data_column].values

    @property
    def data_column(self) -> str:
        """Name of main data column of the point cloud."""
        return self._data_column

    @property
    def auxiliary_columns(self) -> list[str]:
        """Name of auxiliary data columns of the point cloud."""
        return self._auxiliary_columns

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
    def from_tuples(self, tuples: Iterable[tuple[Number, Number, Number]], crs: CRS, data_column: str | None = None):
        """Create point cloud from a N-size list of tuples (X coordinate, Y coordinate, Z value)."""

        self.from_array(np.array(zip(*tuples)), crs=crs, data_column=data_column)


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
