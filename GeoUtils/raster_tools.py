"""
GeoUtils.raster_tools provides a toolset for working with raster data.
"""
import numpy as np
import rasterio as rio


# Attributes from rasterio's DatasetReader object to be kept by default
saved_attrs = ['bounds', 'count', 'crs', 'dataset_mask', 'driver', 'dtypes', 'height', 'indexes', 'name', 'nodata', 'res', 'shape', 'transform', 'width']


class Raster():
    """
    Create a Raster object from a rasterio-supported raster dataset.
    """
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
        # Read file's metadata
        ds = rio.open(filename)
        self.ds = ds
        
        # Copy most used attributes/methods
        for attr in saved_attrs:
            setattr(self, attr, getattr(ds, attr))

        if load_data:
            self.load(bands)
        else:
            self.data = None
            self.nbands = None

    def info(self, stats=False):
        """ 
        Prints information about the raster (filename, coordinate system, number of columns/rows, etc.).

        :param stats: Print statistics for each band of the dataset (max, min, median, mean, std. dev.). Default is to
            not calculate statistics.
        :type stats: bool
        """
        print('Driver:             {}'.format(self.driver))
        #        if self.intype != 'MEM':
        print('File:               {}'.format(self.name))
        # else:
        #    print('File:               {}'.format('in memory'))
        print('Size:               {}, {}'.format(self.width, self.height))
        print('Coordinate System:  EPSG:{}'.format(self.crs.to_epsg()))
        print('NoData Value:       {}'.format(self.nodata))
        print('Pixel Size:         {}, {}'.format(*self.res))
        print('Upper Left Corner:  {}, {}'.format(*self.bounds[:2]))
        print('Lower Right Corner: {}, {}'.format(*self.bounds[2:]))
        if stats:
            if self.data is not None:
                if self.nbands == 1:
                    print('[MAXIMUM]:          {:.2f}'.format(np.nanmax(self.data)))
                    print('[MINIMUM]:          {:.2f}'.format(np.nanmin(self.data)))
                    print('[MEDIAN]:           {:.2f}'.format(np.nanmedian(self.data)))
                    print('[MEAN]:             {:.2f}'.format(np.nanmean(self.data)))
                    print('[STD DEV]:          {:.2f}'.format(np.nanstd(self.data)))
                else:
                    for b in range(self.nbands):
                        print('Band {}:'.format(b+1))  # try to keep with rasterio convention.
                        print('[MAXIMUM]:          {:.2f}'.format(np.nanmax(self.data[b, :, :])))
                        print('[MINIMUM]:          {:.2f}'.format(np.nanmin(self.data[b, :, :])))
                        print('[MEDIAN]:           {:.2f}'.format(np.nanmedian(self.data[b, :, :])))
                        print('[MEAN]:             {:.2f}'.format(np.nanmean(self.data[b, :, :])))
                        print('[STD DEV]:          {:.2f}'.format(np.nanstd(self.data[b, :, :])))

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
