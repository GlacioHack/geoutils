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
        self._saved_attrs = saved_attrs
        for attr in saved_attrs:
            setattr(self, attr, getattr(ds,attr)) 


    def __repr__(self):
    	""" Convert object to formal string representation. """
    	L = [getattr(self, item) for item in self._saved_attrs]
    	s = "%s.%s(%s)" % (self.__class__.__module__,
    		self.__class__.__qualname__,
    		", ".join(map(str, L)))

    	return s


    def __str__(self):
    	return self.info()

        
    def info(self):
        """ 
        Returns string of information about the raster (filename, coordinate system, number of columns/rows, etc.).
        """
        as_str = []
        as_str.append('Driver:             {} \n'.format(self.driver))
        as_str.append('File:               {}\n'.format(self.name))
        as_str.append('Size:               {}, {}\n'.format(self.width, self.height))
        as_str.append('Coordinate System:  EPSG:{}\n'.format(self.crs.to_epsg()))
        as_str.append('NoData Value:       {}\n'.format(self.nodata))
        as_str.append('Pixel Size:         {}, {}\n'.format(*self.res))
        as_str.append('Upper Left Corner:  {}, {}\n'.format(*self.bounds[:2]))
        as_str.append('Lower Right Corner: {}, {}\n'.format(*self.bounds[2:]))

        return "".join(as_str)
        #print('[MAXIMUM]:          {}'.format(np.nanmax(self.img)))
        #print('[MINIMUM]:          {}'.format(np.nanmin(self.img)))

