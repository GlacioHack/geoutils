"""
GeoUtils.vector_tools provides a toolset for working with vector data.
"""
import warnings
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio import warp, features

from GeoUtils.raster_tools import Raster


class Vector():
    """
    Create a Vector object from a fiona-supported vector dataset.
    """

    def __init__(self, filename):
        """
        Load a fiona-supported dataset, given a filename.

        :param filename: The filename of the dataset.
        :type filename: str

        :return: A Vector object
        """

        ds = gpd.read_file(filename)
        self.ds = ds
        self.name = filename

    def crop2raster(self, rst):
        """
        Update self so that features outside the extent of a raster file are cropped. Reprojection is done on the fly if both data set have different projections.

        :param rst: A Raster object or string to filename
        :type rst: Raster object or str
        """
        # If input is string, open as Raster
        if isinstance(rst, str):
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
        Crop a vector file to the extent of a raster file. Reprojection is done on the fly if both data set have different projections.

        :param rst: A Raster object or string to filename
        :type rst: Raster object or str
        :param crs: A pyproj or rasterio CRS object
        :type crs: pyproj.crs.crs.CRS, rasterio.crs.CRS
        :param xres: Output raster spatial resolution in x
        :type xres: float
        :param yres: Output raster spatial resolution in x
        :type yres: float
        :param bounds: Output raster bounds (left, bottom, right, top)
        :type bounds: tuple
        :param in_value: Value to be burnt inside the polygons
        :type in_value: float
        :param out_value: Value to be burnt outside the polygons
        :type out_value: float

        :returns: array containing the mask
        :rtype: numpy.array
        """
        # If input rst is string, open as Raster
        if isinstance(rst, str):
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
