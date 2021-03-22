"""
geoutils.vectortools provides a toolset for working with vector data.
"""
import warnings

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features, warp


class Vector(object):
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
            raise ValueError('filename argument not recognised.')

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
            'Filename:           {} \n'.format(self.name),
            'Coordinate System:  EPSG:{}\n'.format(
                self.ds.crs.to_epsg()),
            'Number of features: {} \n'.format(len(self.ds)),
            'Extent:             {} \n'.format(self.ds.total_bounds.tolist()),
            'Attributes:         {} \n'.format(self.ds.columns.tolist()),
            self.ds.__repr__()]

        return "".join(as_str)

    def crop2raster(self, rst):
        """
        Update self so that features outside the extent of a raster file are cropped. Reprojection is done on the fly if both data set have different projections.

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
        x1, x2, y1, y2 = warp.transform_bounds(
            rst.crs, self.ds.crs, left, bottom, right, top)
        self.ds = self.ds.cx[x1:x2, y1:y2]

    def create_mask(self, rst=None, crs=None, xres=None, yres=None, bounds=None, in_value=255, out_value=0):
        """
        Rasterize the vector features into a raster which has the extent/dimensions of the provided raster file.

        Alternatively, user can specify a grid to rasterize on using xres, yres, bounds and crs. Only xres is mandatory, by default yres=xres and bounds/crs are set to self's.
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
        :param in_value: Value to be burnt inside the polygons (Default 255)
        :type in_value: float
        :param out_value: Value to be burnt outside the polygons (Default 0)
        :type out_value: float

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
                raise ValueError('at least rst or xres must be set')
            if yres is None:
                yres = xres

            # By default, use self's CRS and bounds
            if crs is None:
                crs = self.ds.crs
            if bounds is None:
                bounds = self.ds.total_bounds

            # Calculate raster shape
            left, bottom, right, top = bounds
            height = abs((right-left)/xres)
            width = abs((top-bottom)/yres)

            if width % 1 != 0 or height % 1 != 0:
                warnings.warn(
                    "Bounds not a multiple of xres/yres, use rounded bounds")

            width = int(np.round(width))
            height = int(np.round(height))
            out_shape = (height, width)

            # Calculate raster transform
            transform = rio.transform.from_bounds(
                left, bottom, right, top, width, height)

        # otherwise use directly rst's dimensions
        else:
            out_shape = rst.shape
            transform = rst.transform
            crs = rst.crs

        # Reproject vector into rst CRS
        # Note: would need to check if CRS are different
        vect = self.ds.to_crs(crs)

        # Rasterize geomtry
        mask = features.rasterize(shapes=vect.geometry,
                                  fill=out_value, out_shape=out_shape,
                                  transform=transform, default_value=in_value)

        return mask
