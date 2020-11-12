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
        self._saved_attrs = saved_attrs
        for attr in saved_attrs:
        	setattr(self, attr, getattr(ds, attr))

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
        as_str = []
        as_str.append('Driver:             {} \n'.format(self.driver))
        as_str.append('File:               {}\n'.format(self.name))
        as_str.append('Size:               {}, {}\n'.format(self.width, self.height))
        as_str.append('Coordinate System:  EPSG:{}\n'.format(self.crs.to_epsg()))
        as_str.append('NoData Value:       {}\n'.format(self.nodata))
        as_str.append('Pixel Size:         {}, {}\n'.format(*self.res))
        as_str.append('Upper Left Corner:  {}, {}\n'.format(*self.bounds[:2]))
        as_str.append('Lower Right Corner: {}, {}\n'.format(*self.bounds[2:]))

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
                        as_str.append('Band {}:'.format(b+1))  # \ntry to keep with rasterio convention.
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



    def save(self, filename, driver='GTiff', dtype=None):
    	""" Write the Raster to a geo-referenced file. 

    	Only works if data have been loaded into memory.

    	:param filename: Filename to write the file to.
    	:type filename: str
    	:param driver: the 'GDAL' driver to use to write the file as.
    	:type driver: str
    	:param dtype: Data Type to write the image as (defaults to dtype of image data)
    	:type dtype: np.dtype

    	"""

    	"""
    	NOTE: The code in this function could become deprecated very quickly.
    	At the moment we're transiently creating a new rio object, but if 
    	we proceed with in-memory representation plans then it would be sufficient
    	to simply call .write() on that instead.
    	"""

    	dtype = self.data.dtype if dtype is None else dtype

    	if self.data is None:
    		raise AttributeError('No raster data loaded into memory.')


    	with rio.open(filename, 'w', 
    		driver=driver, 
    		height=self.ds.height, 
    		width=self.ds.width, 
    		count=self.ds.count,
    		dtype=dtype, 
    		crs=self.ds.crs, 
    		transform=self.ds.transform) as dst:

    		dst.write(self.data, 1)



class SatelliteImage(Raster):
    pass
