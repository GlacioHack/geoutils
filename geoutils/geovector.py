"""
geoutils.vectortools provides a toolset for working with vector data.
"""
import collections
import warnings
from numbers import Number

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features, warp


class Vector:
    """
    Create a Vector object from a fiona-supported vector dataset.
    """

    def __init__(self, filename):
        """
        Load a fiona-supported dataset, given a filename.

        :param filename: The filename or GeoDataFrame of the dataset.
        :type filename: str or gpd.GeoDataFrame

        :return: A Vector object
        """

        if isinstance(filename, str):
            ds = gpd.read_file(filename)
            self.ds = ds
            self.name = filename
        elif isinstance(filename, gpd.GeoDataFrame):
            self.ds = filename
            self.name = None
        else:
            raise ValueError("filename argument not recognised.")

        self.crs = self.ds.crs

    def __repr__(self):
        return self.ds.__repr__()

    def __str__(self):
        """ Provide string of information about Raster. """
        return self.info()

    def info(self):
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

    def copy(self):
        """Return a copy of the Vector."""
        # Utilise the copy method of GeoPandas
        return Vector(self.ds.copy())

    def crop2raster(self, rst):
        """
        Update self so that features outside the extent of a raster file are cropped.

        Reprojection is done on the fly if both data set have different projections.

        :param rst: A Raster object or string to filename
        :type rst: Raster object or str
        """
        # If input is string, open as Raster
        if isinstance(rst, str):
            from geoutils.georaster import Raster

            rst = Raster(rst)

        # Convert raster extent into self CRS
        # Note: could skip this if we could test if rojections are same
        # Note: should include a method in Raster to get extent in other projections, not only using corners
        left, bottom, right, top = rst.bounds
        x1, y1, x2, y2 = warp.transform_bounds(rst.crs, self.ds.crs, left, bottom, right, top)
        self.ds = self.ds.cx[x1:x2, y1:y2]

    def create_mask(self, rst=None, crs=None, xres=None, yres=None, bounds=None):
        """
        Rasterize the vector features into a boolean raster which has the extent/dimensions of \
the provided raster file.

        Alternatively, user can specify a grid to rasterize on using xres, yres, bounds and crs.
        Only xres is mandatory, by default yres=xres and bounds/crs are set to self's.

        Vector features which fall outside the bounds of the raster file are not written to the new mask file.

        :param rst: A Raster object or string to filename
        :type rst: Raster object or str
        :param crs: A pyproj or rasterio CRS object (Default to rst.crs if not None then self.crs)
        :type crs: pyproj.crs.crs.CRS, rasterio.crs.CRS
        :param xres: Output raster spatial resolution in x. Only is rst is None.
        :type xres: float
        :param yres: Output raster spatial resolution in y. Only if rst is None. (Default to xres)
        :type yres: float
        :param bounds: Output raster bounds (left, bottom, right, top). Only if rst is None (Default to self bounds)
        :type bounds: tuple

        :returns: array containing the mask
        :rtype: numpy.array
        """
        # If input rst is string, open as Raster
        if isinstance(rst, str):
            from geoutils.georaster import Raster

            rst = Raster(rst)

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
            out_shape = rst.shape
            transform = rst.transform
            crs = rst.crs

        # Reproject vector into rst CRS
        # Note: would need to check if CRS are different
        vect = self.ds.to_crs(crs)

        # Rasterize geomtry
        mask = features.rasterize(
            shapes=vect.geometry, fill=0, out_shape=out_shape, transform=transform, default_value=1, dtype="uint8"
        ).astype("bool")

        # Force output mask to be of same dimension as input rst
        if rst is not None:
            mask = mask.reshape((rst.count, rst.height, rst.width))

        return mask

    def rasterize(self, rst=None, crs=None, xres=None, yres=None, bounds=None, in_value=None, out_value=0):
        """
        Return an array with input geometries burned in.

        By default, output raster has the extent/dimensions of the provided raster file.
        Alternatively, user can specify a grid to rasterize on using xres, yres, bounds and crs.
        Only xres is mandatory, by default yres=xres and bounds/crs are set to self's.

        Burn value is set by user and can be either a single number, or an iterable of same length as self.ds.
        Default is an index from 1 to len(self.ds).

        :param rst: A raster to be used as reference for the output grid
        :type rst: Raster object or str
        :param crs: A pyproj or rasterio CRS object (Default to rst.crs if not None then self.crs)
        :type crs: pyproj.crs.crs.CRS, rasterio.crs.CRS
        :param xres: Output raster spatial resolution in x. Only is rst is None.
        :type xres: float
        :param yres: Output raster spatial resolution in y. Only if rst is None. (Default to xres)
        :type yres: float
        :param bounds: Output raster bounds (left, bottom, right, top). Only if rst is None (Default to self bounds)
        :type bounds: tuple
        :param in_value: Value(s) to be burned inside the polygons (Default is self.ds.index + 1)
        :type in_value: int, float, iterable
        :param out_value: Value to be burned outside the polygons (Default is 0)
        :type out_value: int, float

        :returns: array containing the burned geometries
        :rtype: numpy.array
        """
        # If input rst is string, open as Raster
        if isinstance(rst, str):
            from geoutils.georaster import Raster

            rst = Raster(rst)

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
            out_shape = rst.shape
            transform = rst.transform
            crs = rst.crs

        # Reproject vector into rst CRS
        # Note: would need to check if CRS are different
        vect = self.ds.to_crs(crs)

        # Set default burn value, index from 1 to len(self.ds)
        if in_value is None:
            in_value = self.ds.index + 1

        # Rasterize geometry
        if isinstance(in_value, collections.abc.Iterable):
            if len(in_value) != len(vect.geometry):
                raise ValueError(
                    "in_value must have same length as self.ds.geometry, currently {} != {}".format(
                        len(in_value), len(vect.geometry)
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

    def query(self, expression: str, inplace=False):
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
        return Vector(self.ds.query(expression))
