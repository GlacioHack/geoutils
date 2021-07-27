"""
geoutils.vectortools provides a toolset for working with vector data.
"""
from __future__ import annotations

import warnings
from collections import abc
from numbers import Number
from typing import TypeVar

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features, warp
from rasterio.crs import CRS
import shapely
from scipy.spatial import Voronoi

import geoutils as gu

# This is a generic Vector-type (if subclasses are made, this will change appropriately)
VectorType = TypeVar("VectorType", bound="Vector")


class Vector:
    """
    Create a Vector object from a fiona-supported vector dataset.
    """

    def __init__(self, filename: str | gpd.GeoDataFrame):
        """
        Load a fiona-supported dataset, given a filename.

        :param filename: The filename or GeoDataFrame of the dataset.

        :return: A Vector object
        """

        if isinstance(filename, str):
            ds = gpd.read_file(filename)
            self.ds = ds
            self.name: str | gpd.GeoDataFrame | None = filename
        elif isinstance(filename, gpd.GeoDataFrame):
            self.ds = filename
            self.name = None
        else:
            raise ValueError("filename argument not recognised.")

        self.crs = self.ds.crs

    def __repr__(self) -> str:
        return str(self.ds.__repr__())

    def __str__(self) -> str:
        """Provide string of information about Raster."""
        return self.info()

    def info(self) -> str:
        """
        Returns string of information about the vector (filename, coordinate system, number of layers, features, etc.).

        :returns: text information about Vector attributes.
        :rtype: str
        """
        as_str = [  # 'Driver:             {} \n'.format(self.driver),
            f"Filename:           {self.name} \n",
            f"Coordinate System:  EPSG:{self.ds.crs.to_epsg()}\n",
            f"Number of features: {len(self.ds)} \n",
            f"Extent:             {self.ds.total_bounds.tolist()} \n",
            f"Attributes:         {self.ds.columns.tolist()} \n",
            self.ds.__repr__(),
        ]

        return "".join(as_str)

    @property
    def bounds(self) -> rio.coords.BoundingBox:
        """Get a bounding box of the total bounds of the Vector."""
        return rio.coords.BoundingBox(*self.ds.total_bounds)

    def copy(self: VectorType) -> VectorType:
        """Return a copy of the Vector."""
        # Utilise the copy method of GeoPandas
        new_vector = self.__new__(type(self))
        new_vector.__init__(self.ds.copy())
        return new_vector  # type: ignore

    def crop2raster(self, rst: gu.Raster) -> None:
        """
        Update self so that features outside the extent of a raster file are cropped.

        Reprojection is done on the fly if both data set have different projections.

        :param rst: A Raster object or string to filename
        """
        # If input is string, open as Raster
        if isinstance(rst, str):
            rst = gu.Raster(rst)

        # Convert raster extent into self CRS
        # Note: could skip this if we could test if rojections are same
        # Note: should include a method in Raster to get extent in other projections, not only using corners
        left, bottom, right, top = rst.bounds
        x1, y1, x2, y2 = warp.transform_bounds(rst.crs, self.ds.crs, left, bottom, right, top)
        self.ds = self.ds.cx[x1:x2, y1:y2]

    def create_mask(
        self,
        rst: str | gu.georaster.RasterType | None = None,
        crs: CRS | None = None,
        xres: float | None = None,
        yres: float | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        buffer: int | float | np.number = 0,
    ) -> np.ndarray:
        """
        Rasterize the vector features into a boolean raster which has the extent/dimensions of \
the provided raster file.

        Alternatively, user can specify a grid to rasterize on using xres, yres, bounds and crs.
        Only xres is mandatory, by default yres=xres and bounds/crs are set to self's.

        Vector features which fall outside the bounds of the raster file are not written to the new mask file.

        :param rst: A Raster object or string to filename
        :param crs: A pyproj or rasterio CRS object (Default to rst.crs if not None then self.crs)
        :param xres: Output raster spatial resolution in x. Only is rst is None.
        :param yres: Output raster spatial resolution in y. Only if rst is None. (Default to xres)
        :param bounds: Output raster bounds (left, bottom, right, top). Only if rst is None (Default to self bounds)
        :param buffer: Size of buffer to be added around the features, in the raster's projection units.
        If a negative value is set, will erode the features.

        :returns: array containing the mask
        """
        # If input rst is string, open as Raster
        if isinstance(rst, str):
            rst = gu.Raster(rst)  # type: ignore

        # If no rst given, use provided dimensions
        if rst is None:

            # At minimum, xres must be set
            if xres is None:
                raise ValueError("at least rst or xres must be set")
            if yres is None:
                yres = xres

            # By default, use self's CRS and bounds
            if crs is None:
                crs = self.ds.crs
            if bounds is None:
                bounds = self.ds.total_bounds

            # Calculate raster shape
            left, bottom, right, top = bounds
            height = abs((right - left) / xres)
            width = abs((top - bottom) / yres)

            if width % 1 != 0 or height % 1 != 0:
                warnings.warn("Bounds not a multiple of xres/yres, use rounded bounds")

            width = int(np.round(width))
            height = int(np.round(height))
            out_shape = (height, width)

            # Calculate raster transform
            transform = rio.transform.from_bounds(left, bottom, right, top, width, height)

        # otherwise use directly rst's dimensions
        elif isinstance(rst, gu.Raster):
            out_shape = rst.shape
            transform = rst.transform
            crs = rst.crs
            bounds = rst.bounds
        else:
            raise ValueError("`rst` must be either a str, geoutils.Raster or None")

        # Copying GeoPandas dataframe before applying changes
        gdf = self.ds.copy()

        # Crop vector geometries to avoid issues when reprojecting
        left, bottom, right, top = bounds  # type: ignore
        x1, y1, x2, y2 = warp.transform_bounds(crs, gdf.crs, left, bottom, right, top)
        gdf = gdf.cx[x1:x2, y1:y2]

        # Reproject vector into rst CRS
        gdf = gdf.to_crs(crs)

        # Create a buffer around the features
        if not isinstance(buffer, (int, float, np.number)):
            raise ValueError(f"`buffer` must be a number, currently set to {type(buffer)}")
        if buffer != 0:
            gdf.geometry = [geom.buffer(buffer) for geom in gdf.geometry]
        elif buffer == 0:
            pass

        # Rasterize geometry
        mask = features.rasterize(
            shapes=gdf.geometry, fill=0, out_shape=out_shape, transform=transform, default_value=1, dtype="uint8"
        ).astype("bool")

        # Force output mask to be of same dimension as input rst
        if rst is not None:
            mask = mask.reshape((rst.count, rst.height, rst.width))  # type: ignore

        return mask

    def rasterize(
        self,
        rst: str | gu.georaster.RasterType | None = None,
        crs: CRS | None = None,
        xres: float | None = None,
        yres: float | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        in_value: int | float | abc.Iterable[int | float] | None = None,
        out_value: int | float = 0,
    ) -> np.ndarray:
        """
        Return an array with input geometries burned in.

        By default, output raster has the extent/dimensions of the provided raster file.
        Alternatively, user can specify a grid to rasterize on using xres, yres, bounds and crs.
        Only xres is mandatory, by default yres=xres and bounds/crs are set to self's.

        Burn value is set by user and can be either a single number, or an iterable of same length as self.ds.
        Default is an index from 1 to len(self.ds).

        :param rst: A raster to be used as reference for the output grid
        :param crs: A pyproj or rasterio CRS object (Default to rst.crs if not None then self.crs)
        :param xres: Output raster spatial resolution in x. Only is rst is None.
        :param yres: Output raster spatial resolution in y. Only if rst is None. (Default to xres)
        :param bounds: Output raster bounds (left, bottom, right, top). Only if rst is None (Default to self bounds)
        :param in_value: Value(s) to be burned inside the polygons (Default is self.ds.index + 1)
        :param out_value: Value to be burned outside the polygons (Default is 0)

        :returns: array containing the burned geometries
        """
        # If input rst is string, open as Raster
        if isinstance(rst, str):
            rst = gu.Raster(rst)  # type: ignore

        # If no rst given, use provided dimensions
        if rst is None:

            # At minimum, xres must be set
            if xres is None:
                raise ValueError("at least rst or xres must be set")
            if yres is None:
                yres = xres

            # By default, use self's CRS and bounds
            if crs is None:
                crs = self.ds.crs
            if bounds is None:
                bounds = self.ds.total_bounds

            # Calculate raster shape
            left, bottom, right, top = bounds
            height = abs((right - left) / xres)
            width = abs((top - bottom) / yres)

            if width % 1 != 0 or height % 1 != 0:
                warnings.warn("Bounds not a multiple of xres/yres, use rounded bounds")

            width = int(np.round(width))
            height = int(np.round(height))
            out_shape = (height, width)

            # Calculate raster transform
            transform = rio.transform.from_bounds(left, bottom, right, top, width, height)

        # otherwise use directly rst's dimensions
        else:
            out_shape = rst.shape  # type: ignore
            transform = rst.transform  # type: ignore
            crs = rst.crs  # type: ignore

        # Reproject vector into rst CRS
        vect = self.ds.to_crs(crs)

        # Set default burn value, index from 1 to len(self.ds)
        if in_value is None:
            in_value = self.ds.index + 1

        # Rasterize geometry
        if isinstance(in_value, abc.Iterable):
            if len(in_value) != len(vect.geometry):  # type: ignore
                raise ValueError(
                    "in_value must have same length as self.ds.geometry, currently {} != {}".format(
                        len(in_value), len(vect.geometry)  # type: ignore
                    )
                )

            out_geom = ((geom, value) for geom, value in zip(vect.geometry, in_value))

            mask = features.rasterize(shapes=out_geom, fill=out_value, out_shape=out_shape, transform=transform)

        elif isinstance(in_value, Number):
            mask = features.rasterize(
                shapes=vect.geometry, fill=out_value, out_shape=out_shape, transform=transform, default_value=in_value
            )
        else:
            raise ValueError("in_value must be a single number or an iterable with same length as self.ds.geometry")

        return mask

    def query(self: VectorType, expression: str, inplace: bool = False) -> VectorType:
        """
        Query the Vector dataset with a valid Pandas expression.

        :param expression: A python-like expression to evaluate. Example: "col1 > col2"
        :param inplace: Whether the query should modify the data in place or return a modified copy.

        :returns: Vector resulting from the provided query expression or itself if inplace=True.
        """
        # Modify inplace if wanted and return the self instance.
        if inplace:
            self.ds.query(expression, inplace=True)
            return self

        # Otherwise, create a new Vector from the queried dataset.
        new_vector = self.__new__(type(self))
        new_vector.__init__(self.ds.query(expression))
        return new_vector  # type: ignore


############################################
# Additional stand-alone utility functions #
############################################

def extract_vertices(gdf: gpd.GeoDataFrame) -> list[list[tuple[float, float]]]:
    """
    Function to extract the exterior vertices of all shapes within a gpd.GeoDataFrame.

    :param gdf: The GeoDataFrame from which the vertices need to be extracted.

    :returns: A list containing a list of (x, y) positions of the vertices. The length of the primary list is equal \ to the number of geometries inside gdf, and length of each sublist is the number of vertices in the geometry.
    """
    vertices = []
    # Loop on all geometries within gdf
    for geom in gdf.geometry:
        # Extract geometry exterior(s)
        if geom.geom_type == 'MultiPolygon':
            exteriors = [p.exterior for p in geom]
        elif geom.geom_type == 'Polygon':
            exteriors = [geom.exterior,]
        elif geom.geom_type == 'LineString':
            exteriors = [geom,]
        elif geom.geom_type == 'MultiLineString':
            exteriors = geom
        else:
            raise NotImplementedError(f"Geometry type {geom.geom_type} not implemented.")

        vertices.extend([list(ext.coords) for ext in exteriors])

    return vertices


def generate_voronoi_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Function to generate the Voronoi polygons (tesselation) from the vertices of all geometries in a GeoDataFrame.

    Uses scipy.spatial.voronoi.

    :param: The GeoDataFrame from whose vertices are used for the Voronoi polygons.

    :returns: A GeoDataFrame containing the Voronoi polygons.
    """
    # Extract the coordinates of the vertices of all geometries in gdf
    vertices = extract_vertices(gdf)
    coords = np.concatenate(vertices)

    # Create the Voronoi diagram and extract ridges
    vor = Voronoi(coords)
    lines = [shapely.geometry.LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]
    polys = list(shapely.ops.polygonize(lines))
    if len(polys) == 0:
        raise ValueError("Invalid geometry, cannot generate finite Voronoi polygons")

    # Convert into GeoDataFrame
    voronoi = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys))
    voronoi.crs = gdf.crs

    return voronoi
