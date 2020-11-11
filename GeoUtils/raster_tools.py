import numpy as np
import rasterio as rio


# Attributes from rasterio's DatasetReader object to be kept by default
saved_attrs = ['bounds', 'count', 'crs', 'dataset_mask', 'driver', 'dtypes', 'height', 'indexes', 'name', 'nodata', 'res', 'shape', 'transform', 'width']

class Raster():
    """
    Create a Raster object from a rasterio-supported raster dataset.
    """
    def __init__(self, filename: str, saved_attrs=saved_attrs):

        # Read file's metadata
        ds = rio.open(filename)
        self.ds = ds
        
        # Copy most used attributes/methods
        for attr in saved_attrs:
            setattr(self, attr, getattr(ds,attr)) 
        
    def info(self):
        """ 
        Prints information about the raster (filename, coordinate system, number of columns/rows, etc.).
        """
        print('Driver:             {}'.format(self.driver))
        #        if self.intype != 'MEM':
        print('File:               {}'.format(self.name))
        #else:
        #    print('File:               {}'.format('in memory'))
        print('Size:               {}, {}'.format(self.width, self.height))
        print('Coordinate System:  EPSG:{}'.format(self.crs.to_epsg()))
        print('NoData Value:       {}'.format(self.nodata))
        print('Pixel Size:         {}, {}'.format(*self.res))
        print('Upper Left Corner:  {}, {}'.format(*self.bounds[:2]))
        print('Lower Right Corner: {}, {}'.format(*self.bounds[2:]))
        #print('[MAXIMUM]:          {}'.format(np.nanmax(self.img)))
        #print('[MINIMUM]:          {}'.format(np.nanmin(self.img)))

