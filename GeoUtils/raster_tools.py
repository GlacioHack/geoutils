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
    def __init__(self, filename: str, saved_attrs=saved_attrs, load_data=True, bands=None):

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
        Load specific bands of the dataset.
        """
        if bands is None:
            self.data = self.ds.read()
        else:
            self.data = self.ds.read(bands)

        if self.data.ndim == 3:
            self.nbands = self.data.shape[0]
        else:
            self.nbands = 1


class SatelliteImage(Raster):
    pass
