"""
GeoUtils.raster_tools provides a toolset for working with raster data.
"""
import numpy as np
import rasterio as rio
from rasterio.io import MemoryFile

from rasterio.crs import CRS
from affine import Affine

import os

# Attributes from rasterio's DatasetReader object to be kept by default
saved_attrs = ['bounds', 'count', 'crs', 'dataset_mask', 'driver', 'dtypes', 'height', 'indexes', 'name', 'nodata',
               'res', 'shape', 'transform', 'width']


class Raster(object):
    """
    Create a Raster object from a rasterio-supported raster dataset.
    """

    # This only gets set if a disk-based file is read in. If the Raster is created with from_array, from_mem etc, this stays as None.
    filename = None

    def __init__(self, filename, saved_attrs=saved_attrs, load_data=True, bands=None):
        """
        Load a rasterio-supported dataset, given a filename.

        :param filename: The filename of the dataset, or a rasterio MemoryFile object.
        :type filename: str or rio.io.MemoryFile.
        :param saved_attrs: A list of attributes from rasterio's DataReader class to add to the Raster object.
            Default list is ['bounds', 'count', 'crs', 'dataset_mask', 'driver', 'dtypes', 'height', 'indexes',
             'name', 'nodata', 'res', 'shape', 'transform', 'width']
        :type saved_attrs: list of strings
        :param load_data: Load the raster data into the object. Default is True.
        :type load_data: bool
        :param bands: The band(s) to load into the object. Default is to load all bands.
        :type bands: int, or list of ints

        :return: A Raster object
        """

        # Image is a file on disk.
        if isinstance(filename, str):
            # Save the absolute on-disk filename
            self.filename = os.path.abspath(filename)
            # open the file in memory
            self.memfile = MemoryFile(open(filename, 'rb'))

        # Or, image is already a Memory File.
        elif isinstance(filename, rio.io.MemoryFile):
            self.filename = None
            self.memfile = filename

        # Provide a catch in case trying to load from data array
        elif isinstance(filename, np.array):
            raise ValueError('np.array provided as filename. Did you mean to call Raster.from_array(...) instead? ')
        
        # Don't recognise the input, so stop here.
        else:
            raise ValueError('filename argument not recognised.')

        # Read the file as a rasterio dataset
        self.ds = self.memfile.open()

        # Copy most used attributes/methods
        self._saved_attrs = saved_attrs
        for attr in saved_attrs:
            setattr(self, attr, getattr(self.ds, attr))

        if load_data:
            self.load(bands)
        else:
            self.data = None
            self.nbands = None


    @classmethod
    def from_array(cls, data, transform, crs, nodata=None):
        """ Create a Raster from a numpy array and some geo-referencing information.

        :param data:
        :dtype data:
        :param transform: the 2-D affine transform for the image mapping. 
            Either a tuple(x_res, 0.0, top_left_x, 0.0, y_res, top_left_y) or 
            an affine.Affine object.
        :dtype transform: tuple, affine.Affine.
        :param crs: Coordinate Reference System for image. Either a rasterio CRS, 
            or the EPSG integer.
        :dtype crs: rasterio.crs.CRS or int
        :param nodata:
        :dtype nodata:

        :returns: A Raster object containing the provided data.
        :rtype: Raster.

        Example:
        You have a data array in EPSG:32645. It has a spatial resolution of
        30 m in x and y, and its top left corner is X=478000, Y=3108140.
        >>> transform = (30.0, 0.0, 478000.0, 0.0, -30.0, 3108140.0)
        >>> myim = Raster.from_array(data, transform, 32645)

        """

        if not isinstance(transform, Affine):
            if isinstance(transform, tuple):
                transform = Affine(*transform)
            else:
                raise ValueError('transform argument needs to be Affine or tuple.')

        # Enable shortcut to create CRS from an EPSG ID.
        if isinstance(crs, int):
            crs = _create_crs_from_epsg(crs)

        # If a 2-D ('single-band') array is passed in, give it a band dimension.
        if len(data.shape) < 3:
            data = np.expand_dims(data, 0)

        # Open handle to new memory file
        mfh = MemoryFile()

        # Create the memory file
        with rio.open(mfh, 'w',
            height=data.shape[1],
            width=data.shape[2],
            count=data.shape[0],
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata, 
            driver='GTiff') as ds:

            ds.write(data)

        # Initialise a Raster object created with MemoryFile.
        # (i.e., __init__ will now be run.)
        return cls(mfh)

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
                        as_str.append('Band {}:'.format(b + 1))  # \ntry to keep with rasterio convention.
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

    def crop(self):
        pass

    def clip(self):
        pass


class SatelliteImage(Raster):
    pass


def _create_crs_from_epsg(epsg):
    """ Given an EPSG code, generate a rasterio CRS object.

    :param epsg: the EPSG code for which to generate a CRS.
    :dtype epsg: int
    :returns: the CRS object
    :rtype: rasterio.crs.CRS
    """
    if not isinstance(epsg, int):
        raise ValueError('EPSG code must be provided as int.')
    return CRS.from_epsg(epsg)
