"""
GeoUtils.raster_tools provides a toolset for working with raster data.
"""
import os
import numpy as np
import rasterio as rio
import rasterio.mask as riomask
import rasterio.warp as riowarp
import rasterio.windows as riowindows
import rasterio.transform as riotransform
from rasterio.io import MemoryFile
from rasterio.crs import CRS
from affine import Affine
from shapely.geometry.polygon import Polygon
import GeoUtils.vector_tools as vt

try:
    import rioxarray
except ImportError:
    _has_rioxarray = False
else:
    _has_rioxarray = True


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
        :param attrs: Additional attributes from rasterio's DataReader class to add to the Raster object.
            Default list is ['bounds', 'count', 'crs', 'dataset_mask', 'driver', 'dtypes', 'height', 'indexes',
            'name', 'nodata', 'res', 'shape', 'transform', 'width'] - if no attrs are specified, these will be added.
        :type attrs: list of strings
        :param load_data: Load the raster data into the object. Default is False.
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

        self._read_attrs(attrs)

        if load_data:
            self.load(bands)
            self.isLoaded = True
        else:
            self.data = None
            self.nbands = None
            self.isLoaded = False


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

    def _read_attrs(self, attrs=None):
        # Copy most used attributes/methods
        if attrs is None:
            self._saved_attrs = default_attrs
            attrs = default_attrs
        else:
            if isinstance(attrs, str):
                attrs = [attrs]
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

    def crop(self, cropGeom, mode='match_pixel'):
        """
        Crop the Raster to a given extent.

        :param cropGeom: Geometry to crop raster to, as either a Raster object, a Vector object, or a list of
            coordinates. If cropGeom is a Raster, crop() will crop to the boundary of the raster as returned by
            Raster.ds.bounds. If cropGeom is a Vector, crop() will crop to the bounding geometry. If cropGeom is a
            list of coordinates, the order is assumed to be [xmin, ymin, xmax, ymax].
        :param mode: one of 'match_pixel' (default) or 'match_extent'. 'match_pixel' will preserve the original pixel
            resolution, cropping to the extent that most closely aligns with the current coordinates. 'match_extent'
            will match the extent exactly, adjusting the pixel resolution to fit the extent.
        :type mode: str

        """
        assert mode in ['match_extent', 'match_pixel'], "mode must be one of 'match_pixel', 'match_extent'"
        if isinstance(cropGeom, Raster):
            xmin, ymin, xmax, ymax = cropGeom.bounds
        elif isinstance(cropGeom, vt.Vector):
            raise NotImplementedError
        elif isinstance(cropGeom, (list, tuple)):
            xmin, ymin, xmax, ymax = cropGeom
        else:
            raise ValueError("cropGeom must be a Raster, Vector, or list of coordinates.")

        meta = self.ds.meta

        if mode == 'match_pixel':
            crop_bbox = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

            crop_img, tfm = riomask.mask(self.ds, [crop_bbox], crop=True, all_touched=True)
            meta.update({'height': crop_img.shape[1],
                         'width': crop_img.shape[2],
                         'transform': tfm})

        else:
            window = riowindows.from_bounds(xmin, ymin, xmax, ymax, transform=self.transform)
            new_height = int(window.height)
            new_width = int(window.width)
            new_tfm = riotransform.from_bounds(xmin, ymin, xmax, ymax, width=new_width, height=new_height)

            if self.isLoaded:
                new_img = np.zeros((self.nbands, new_height, new_width), dtype=self.data.dtype)
            else:
                new_img = np.zeros((self.count, new_height, new_width), dtype=self.data.dtype)

            crop_img, tfm = riowarp.reproject(self.data, new_img,
                                              src_transform=self.transform,
                                              dst_transform=new_tfm,
                                              src_crs=self.crs,
                                              dst_crs=self.crs)
            meta.update({'height': new_height,
                         'width': new_width,
                         'transform': tfm})

        self._update(crop_img, meta)

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
