"""
GeoUtils.raster_tools provides a toolset for working with raster data.
"""
import numpy as np
import rasterio as rio
from rasterio.io import MemoryFile
import os


try:
    import rioxarray
except ImportError:
    _has_rioxarray = False
else:
    _has_rioxarray = True


# Attributes from rasterio's DatasetReader object to be kept by default
saved_attrs = ['bounds', 'count', 'crs', 'dataset_mask', 'driver', 'dtypes', 'height', 'indexes', 'name', 'nodata',
               'res', 'shape', 'transform', 'width']


class Raster(object):
    """
    Create a Raster object from a rasterio-supported raster dataset.
    """

    # This only gets set if a disk-based file is read in. If the Raster is created with from_array, from_mem etc, this stays as None.
    filename = None

    def __init__(self, filename: str, saved_attrs=saved_attrs, load_data=False, bands=None):
        """
        Load a rasterio-supported dataset, given a filename.

        :param filename: The filename of the dataset.
        :type filename: str
        :param saved_attrs: A list of attributes from rasterio's DataReader class to add to the Raster object.
            Default list is ['bounds', 'count', 'crs', 'dataset_mask', 'driver', 'dtypes', 'height', 'indexes',
             'name', 'nodata', 'res', 'shape', 'transform', 'width']
        :type saved_attrs: list of strings
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

        # Copy most used attributes/methods
        self._saved_attrs = saved_attrs
        for attr in saved_attrs:
            setattr(self, attr, getattr(self.ds, attr))

        if load_data:
            self.load(bands)
        else:
            self.data = None
            self.nbands = None

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


    def save(self, filename, driver='GTiff', dtype=None,
        blank_value=None):
        """ Write the Raster to a geo-referenced file. 

        Given a filename to save the Raster to, create a geo-referenced file
        on disk which contains the contents of self.data.

        If blank_value is set to an integer or float, then instead of writing 
        the contents of self.data to disk, write this provided value to every
        pixel instead.

        :param filename: Filename to write the file to.
        :type filename: str
        :param driver: the 'GDAL' driver to use to write the file as.
        :type driver: str
        :param dtype: Data Type to write the image as (defaults to dtype of image data)
        :type dtype: np.dtype
        :param blank_value: Use to write an image out with every pixel's value
        corresponding to this value, instead of writing the image data to disk.
        :type blank_value: None, int, float.

        :returns: None.

        """

        dtype = self.data.dtype if dtype is None else dtype

        if (self.data is None) & (blank_value is None):
            return AttributeError('No data loaded, and alterative blank_value not set.')
        elif blank_value is not None:
            if isinstance(blank_value, int) | isinstance(blank_value, float):
                save_data = np.zeros((self.ds.count, self.ds.height, self.ds.width))
                save_data[:,:,:] = blank_value
            else:
                raise ValueError('blank_values must be one of int, float (or None).')
        else:
            save_data = self.data

        with rio.open(filename, 'w', 
            driver=driver, 
            height=self.ds.height, 
            width=self.ds.width, 
            count=self.ds.count,
            dtype=save_data.dtype, 
            crs=self.ds.crs, 
            transform=self.ds.transform,
            nodata=self.ds.nodata) as dst:

            dst.write(save_data)

        return



    def to_xarray(self, name=None):
        """ Convert this Raster into an xarray DataArray using rioxarray.

        This method uses rioxarray to generate a DataArray with associated
        geo-referencing information.

        See the documentation of rioxarray and xarray for more information on 
        the methods and attributes of the resulting DataArray.
        
        :param name: Set the name of the DataArray.
        :type name: str
        :returns: xarray DataArray
        :rtype: xr.DataArray

        """

        if not _has_rioxarray:
            raise ImportError('rioxarray is required for this functionality.')

        xr = rioxarray.open_rasterio(self.ds)
        if name is not None:
            xr.name = name

        return xr


        



class SatelliteImage(Raster):
    pass
