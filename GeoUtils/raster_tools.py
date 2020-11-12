"""
GeoUtils.raster_tools provides a toolset for working with raster data.
"""
import os
import numpy as np
import rasterio as rio
import rasterio.mask as riomask
from rasterio.io import MemoryFile
from shapely.geometry.polygon import Polygon
import GeoUtils.vector_tools as vt


# Attributes from rasterio's DatasetReader object to be kept by default
default_attrs = ['bounds', 'count', 'crs', 'dataset_mask', 'driver', 'dtypes', 'height', 'indexes', 'name', 'nodata',
                 'res', 'shape', 'transform', 'width']


class Raster(object):
    """
    Create a Raster object from a rasterio-supported raster dataset.
    """

    # This only gets set if a disk-based file is read in.
    # If the Raster is created with from_array, from_mem etc, this stays as None.
    filename = None

    def __init__(self, filename: str, attrs=None, load_data=False, bands=None):
        """
        Load a rasterio-supported dataset, given a filename.

        :param filename: The filename of the dataset.
        :type filename: str
        :param attrs: A list of attributes from rasterio's DataReader class to add to the Raster object.
            Default list is ['bounds', 'count', 'crs', 'dataset_mask', 'driver', 'dtypes', 'height', 'indexes',
             'name', 'nodata', 'res', 'shape', 'transform', 'width']
        :type attrs: list of strings
        :param load_data: Load the raster data into the object. Default is False.
        :type load_data: bool
        :param bands: The band(s) to load into the object. Default is to load all bands.
        :type bands: int, or list of ints

        :return: A Raster object
        """

        # Save the absolute on-disk filename
        self.filename = os.path.abspath(filename)

        # open the file in memory
        self.memfile = MemoryFile(open(filename, 'rb'))

        # read the file as a rasterio dataset
        self.ds = self.memfile.open()

        self._read_attrs(list(attrs))  # cast as a list to prevent iteration weirdness

        if load_data:
            self.load(bands)
            self.isLoaded = True
        else:
            self.data = None
            self.nbands = None
            self.isLoaded = False

    def __repr__(self):
        """ Convert object to formal string representation. """
        L = [getattr(self, item) for item in self._saved_attrs]
        s = "%s.%s(%s)" % (self.__class__.__module__,
                           self.__class__.__qualname__,
                           ", ".join(map(str, L)))

        return s

    def __str__(self):
        """ Provide string of information about Raster. """
        return self.info()

    def _read_attrs(self, attrs=None):
        # Copy most used attributes/methods
        if attrs is None:
            self._saved_attrs = default_attrs
            attrs = default_attrs
        else:
            for attr in default_attrs:
                if attr not in attrs:
                    attrs.append(attr)
            self._saved_attrs = attrs

        for attr in attrs:
            setattr(self, attr, getattr(self.ds, attr))

    def _update(self, imgdata, metadata):
        """
        update the object with a new image or coordinates.
        """
        memfile = MemoryFile()
        with memfile.open(**metadata) as ds:
            ds.write(imgdata)

        self.memfile = memfile
        self.ds = memfile.open()
        self._read_attrs()
        if self.isLoaded:
            self.load()

    def info(self, stats=False):
        """ 
        Returns string of information about the raster (filename, coordinate system, number of columns/rows, etc.).

        :param stats: Add statistics for each band of the dataset (max, min, median, mean, std. dev.). Default is to
            not calculate statistics.
        :type stats: bool

        :returns: text information about Raster attributes.
        :rtype: str
        """
        as_str = ['Driver:             {} \n'.format(self.driver),
                  'File on disk:       {} \n'.format(self.filename),
                  'RIO MemoryFile:     {}\n'.format(self.name),
                  'Size:               {}, {}\n'.format(self.width, self.height),
                  'Coordinate System:  EPSG:{}\n'.format(self.crs.to_epsg()),
                  'NoData Value:       {}\n'.format(self.nodata),
                  'Pixel Size:         {}, {}\n'.format(*self.res),
                  'Upper Left Corner:  {}, {}\n'.format(*self.bounds[:2]),
                  'Lower Right Corner: {}, {}\n'.format(*self.bounds[2:])]

        if stats:
            if self.data is not None:
                if self.nbands == 1:
                    as_str.append('[MAXIMUM]:          {:.2f}\n'.format(np.nanmax(self.data)))
                    as_str.append('[MINIMUM]:          {:.2f}\n'.format(np.nanmin(self.data)))
                    as_str.append('[MEDIAN]:           {:.2f}\n'.format(np.nanmedian(self.data)))
                    as_str.append('[MEAN]:             {:.2f}\n'.format(np.nanmean(self.data)))
                    as_str.append('[STD DEV]:          {:.2f}\n'.format(np.nanstd(self.data)))
                else:
                    for b in range(self.nbands):
                        as_str.append('Band {}:'.format(b + 1))  # try to keep with rasterio convention.
                        as_str.append('[MAXIMUM]:          {:.2f}\n'.format(np.nanmax(self.data[b, :, :])))
                        as_str.append('[MINIMUM]:          {:.2f}\n'.format(np.nanmin(self.data[b, :, :])))
                        as_str.append('[MEDIAN]:           {:.2f}\n'.format(np.nanmedian(self.data[b, :, :])))
                        as_str.append('[MEAN]:             {:.2f}\n'.format(np.nanmean(self.data[b, :, :])))
                        as_str.append('[STD DEV]:          {:.2f}\n'.format(np.nanstd(self.data[b, :, :])))

        return "".join(as_str)

    def load(self, bands=None):
        """
        Load specific bands of the dataset, using rasterio.read()

        :param bands: The band(s) to load. Note that rasterio begins counting at 1, not 0.
        :type bands: int, or list of ints
        """
        if bands is None:
            self.data = self.ds.read()
        else:
            self.data = self.ds.read(bands)

        if self.data.ndim == 3:
            self.nbands = self.data.shape[0]
        else:
            self.nbands = 1

    def crop(self, cropGeom):
        """
        Crop the Raster to a given extent.

        :param cropGeom: Geometry to crop raster to, as either a Raster object, a Vector object, or a list of
            coordinates. If cropGeom is a Raster, crop() will crop to the boundary of the raster as returned by
            Raster.ds.bounds. If cropGeom is a Vector, crop() will crop to the bounding geometry. If cropGeom is a
            list of coordinates, the order is assumed to be [xmin, ymin, xmax, ymax].

        """
        if isinstance(cropGeom, Raster):
            xmin, ymin, xmax, ymax = cropGeom.bounds
        elif isinstance(cropGeom, vt.Vector):
            pass
        elif isinstance(cropGeom, list):
            xmin, ymin, xmax, ymax = cropGeom
        else:
            raise ValueError("cropGeom must be a Raster, Vector, or list of coordinates.")

        crop_bbox = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
        meta = self.ds.meta

        crop_img, tfm = riomask.mask(self.ds, [crop_bbox], crop=True)
        meta.update({'height': crop_img.shape[1],
                     'width': crop_img.shape[2],
                     'transform': tfm})
        self._update(crop_img, meta)

    def clip(self):
        pass


class SatelliteImage(Raster):
    pass
