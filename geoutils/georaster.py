"""
GeoUtils.raster_tools provides a toolset for working with raster data.
"""
import os
import warnings
import numpy as np

import rasterio as rio
import rasterio.mask
import rasterio.warp
import rasterio.windows
import rasterio.transform
from rasterio.io import MemoryFile
from rasterio.crs import CRS
from rasterio.warp import Resampling
from rasterio.plot import show as rshow

from affine import Affine
from shapely.geometry.polygon import Polygon

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
    matches_disk = None

    def __init__(self, filename, attrs=None, load_data=True, bands=None, 
        as_memfile=False):

        """
        Load a rasterio-supported dataset, given a filename.

        :param filename: The filename of the dataset.
        :type filename: str
        :param attrs: Additional attributes from rasterio's DataReader class to add to the Raster object.
            Default list is ['bounds', 'count', 'crs', 'dataset_mask', 'driver', 'dtypes', 'height', 'indexes',
            'name', 'nodata', 'res', 'shape', 'transform', 'width'] - if no attrs are specified, these will be added.
        :type attrs: list of strings
        :param load_data: Load the raster data into the object. Default is True.
        :type load_data: bool
        :param bands: The band(s) to load into the object. Default is to load all bands.
        :type bands: int, or list of ints
        :param as_memfile: open the dataset via a rio.MemoryFile.
        :type as_memfile: bool

        :return: A Raster object
        """

        # Image is a file on disk.
        if isinstance(filename, str):
            # Save the absolute on-disk filename
            self.filename = os.path.abspath(filename)
            if as_memfile:
                # open the file in memory
                self.memfile = MemoryFile(open(filename, 'rb'))
                # Read the file as a rasterio dataset
                self.ds = self.memfile.open()
            else:
                self.memfile = None
                self.ds = rio.open(filename, 'r')

        # Or, image is already a Memory File.
        elif isinstance(filename, rio.io.MemoryFile):
            self.filename = None
            self.memfile = filename
            self.ds = self.memfile.open()

        # Provide a catch in case trying to load from data array
        elif isinstance(filename, np.array):
            raise ValueError(
                'np.array provided as filename. Did you mean to call Raster.from_array(...) instead? ')

        # Don't recognise the input, so stop here.
        else:
            raise ValueError('filename argument not recognised.')



        self._read_attrs(attrs)

        if load_data:
            self.load(bands)
            self.isLoaded = True
            if isinstance(filename, str):
                self.matches_disk = True
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
                raise ValueError(
                    'transform argument needs to be Affine or tuple.')

        # Enable shortcut to create CRS from an EPSG ID.
        if isinstance(crs, int):
            crs = CRS.from_epsg(crs)

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

    def _update(self, imgdata=None, metadata=None, vrt_to_driver='GTiff'):
        """
        Update the object with a new image or metadata.

        :param imgdata: image data to update with.
        :type imgdata: None or np.array
        :param metadata: metadata to update with.
        :type metadata: dict
        :param vrt_to_driver: name of driver to coerce a VRT to. This is required
        because rasterio does not support writing to to a VRTSourcedRasterBand.
        :type vrt_to_driver: str
        """
        memfile = MemoryFile()
        if imgdata is None:
            imgdata = self.data
        if metadata is None:
            metadata = self.metadata

        if metadata['driver'] == 'VRT':
            metadata['driver'] = vrt_to_driver

        with memfile.open(**metadata) as ds:
            ds.write(imgdata)

        self.memfile = memfile
        self.ds = memfile.open()
        self._read_attrs()
        self.matches_disk = False
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
        as_str = ['Driver:               {} \n'.format(self.driver),
                  'File on disk:         {} \n'.format(self.filename),
                  'RIO MemoryFile:       {} \n'.format(self.name),
                  'Matches file on disk? {} \n'.format(self.matches_disk),
                  'Size:                 {}, {}\n'.format(
                      self.width, self.height),
                  'Coordinate System:    EPSG:{}\n'.format(self.crs.to_epsg()),
                  'NoData Value:         {}\n'.format(self.nodata),
                  'Pixel Size:           {}, {}\n'.format(*self.res),
                  'Upper Left Corner:    {}, {}\n'.format(*self.bounds[:2]),
                  'Lower Right Corner:   {}, {}\n'.format(*self.bounds[2:])]

        if stats:
            if self.data is not None:
                if self.nbands == 1:
                    as_str.append('[MAXIMUM]:          {:.2f}\n'.format(
                        np.nanmax(self.data)))
                    as_str.append('[MINIMUM]:          {:.2f}\n'.format(
                        np.nanmin(self.data)))
                    as_str.append('[MEDIAN]:           {:.2f}\n'.format(
                        np.nanmedian(self.data)))
                    as_str.append('[MEAN]:             {:.2f}\n'.format(
                        np.nanmean(self.data)))
                    as_str.append('[STD DEV]:          {:.2f}\n'.format(
                        np.nanstd(self.data)))
                else:
                    for b in range(self.nbands):
                        # try to keep with rasterio convention.
                        as_str.append('Band {}:'.format(b + 1))
                        as_str.append('[MAXIMUM]:          {:.2f}\n'.format(
                            np.nanmax(self.data[b, :, :])))
                        as_str.append('[MINIMUM]:          {:.2f}\n'.format(
                            np.nanmin(self.data[b, :, :])))
                        as_str.append('[MEDIAN]:           {:.2f}\n'.format(
                            np.nanmedian(self.data[b, :, :])))
                        as_str.append('[MEAN]:             {:.2f}\n'.format(
                            np.nanmean(self.data[b, :, :])))
                        as_str.append('[STD DEV]:          {:.2f}\n'.format(
                            np.nanstd(self.data[b, :, :])))

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
        import geoutils.geovector as vt

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

            crop_img, tfm = rio.mask.mask(self.ds, [crop_bbox], crop=True, all_touched=True)
            meta.update({'height': crop_img.shape[1],
                         'width': crop_img.shape[2],
                         'transform': tfm})

        else:
            window = rio.windows.from_bounds(xmin, ymin, xmax, ymax, transform=self.transform)
            new_height = int(window.height)
            new_width = int(window.width)
            new_tfm = rio.transform.from_bounds(xmin, ymin, xmax, ymax, width=new_width, height=new_height)

            if self.isLoaded:
                new_img = np.zeros((self.nbands, new_height, new_width), dtype=self.data.dtype)
            else:
                new_img = np.zeros((self.count, new_height, new_width), dtype=self.data.dtype)

            crop_img, tfm = rio.warp.reproject(self.data, new_img,
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

    def reproject(self, dst_ref=None, dst_crs=None, dst_size=None, dst_bounds=None, dst_res=None,
                  nodata=None, dtype=None, resampling=Resampling.nearest,
                  **kwargs):
        """ 
        Reproject raster to a specified grid.

        The output grid can either be given by a reference Raster (using `dst_ref`),
        or by manually providing the output CRS (`dst_crs`), dimensions (`dst_size`),
        resolution (with `dst_size`) and/or bounds (`dst_bounds`).
        Any resampling algorithm implemented in rasterio can be used.

        Currently: requires image data to have been loaded into memory.
        NOT SUITABLE for large datasets yet! This requires work...

        To reproject a Raster with different source bounds, first run Raster.crop.

        :param dst_ref: a reference raster. If set will use the attributes of this raster for the output grid.
        Can be provided as Raster/rasterio data set or as path to the file.
        :type dst_ref: Raster object, rasterio data set or a str.
        :param crs: Specify the Coordinate Reference System to reproject to.
        :dtype crs: int, dict, str, CRS      
        :param dst_size: Raster size to write to (x, y). Do not use with dst_res.
        :dtype dst_size: tuple(int, int)
        :param dst_bounds: a BoundingBox object or a dictionary containing left, bottom, right, top bounds in the source CRS.
        :dtype dst_bounds: dict or rio.coords.BoundingBox
        :param dst_res: Pixel size in units of target CRS. Either 1 value or (xres, yres). Do not use with dst_size.
        :dtype dst_res: float or tuple(float, float)
        :param nodata: nodata value in reprojected data.
        :dtype nodata: int, float, None
        :param resampling: A rasterio Resampling method
        :dtype resample: rio.warp.Resampling object
        :param **kwargs: additional keywords are passed to rasterio.warp.reproject. Use with caution.

        :returns: Raster
        :rtype: Raster
        """

        # Check that either dst_ref or dst_crs is provided
        if dst_ref is not None:
            if dst_crs is not None:
                raise ValueError("Either of `dst_ref` or `dst_crs` must be set. Not both.")
        else:
            if dst_crs is None:
                raise ValueError("One of `dst_ref` or `dst_crs` must be set.")
            
        # Case a raster is provided as reference
        if dst_ref is not None:

            # Check that dst_ref type is either str, Raster or rasterio data set
            # Preferably use Raster instance to avoid rasterio data set to remain open. See PR #45
            if isinstance(dst_ref, Raster):
                ds_ref = dst_ref
            elif isinstance(dst_ref, rio.io.MemoryFile) or isinstance(dst_ref, rasterio.io.DatasetReader):
                ds_ref = dst_ref
            elif isinstance(dst_ref, str):
                assert os.path.exists(
                    dst_ref), "Reference raster does not exist"
                ds_ref = Raster(dst_ref, load_data=False)
            else:
                raise ValueError(
                    "Type of dst_ref not understood, must be path to file (str), Raster or rasterio data set")

            # Read reprojecting params from ref raster
            dst_crs = ds_ref.crs
            dst_size = (ds_ref.width, ds_ref.height)
            dst_res = None
            dst_bounds = ds_ref.bounds
        else:
            # Determine target CRS
            dst_crs = CRS.from_user_input(dst_crs)

        # If dst_ref is None, check other input arguments
        if dst_size is not None and dst_res is not None:
            raise ValueError(
                'dst_size and dst_res both specified. Specify only one.')

        if dtype is None:
            # CHECK CORRECT IMPLEMENTATION! (rasterio dtypes seems to be on a per-band basis)
            dtype = self.dtypes[0]

        # Basic reprojection options, needed in all cases.
        reproj_kwargs = {
            'src_transform': self.transform,
            'src_crs': self.crs,
            'dst_crs': dst_crs,
            'resampling': resampling,
            'dst_nodata': self.nodata
        }

        # Create a BoundingBox if required
        if dst_bounds is not None:
            if not isinstance(dst_bounds, rio.coords.BoundingBox):
                dst_bounds = rio.coords.BoundingBox(dst_bounds['left'], dst_bounds['bottom'],
                                                    dst_bounds['right'], dst_bounds['top'])

        # Determine target raster size/resolution
        dst_transform = None
        if dst_res is not None:
            if dst_bounds is None:
                # Let rasterio determine the maximum bounds of the new raster.
                reproj_kwargs.update({'dst_resolution': dst_res})
            else:
                
                # Bounds specified. First check if xres and yres are different.
                if isinstance(dst_res, tuple):
                    xres = dst_res[0]
                    yres = dst_res[1]
                else:
                    xres = dst_res
                    yres = dst_res

                # Calculate new raster size which ensures that pixels have 
                # precisely the resolution specified.
                dst_width = np.ceil((dst_bounds.right - dst_bounds.left) / xres)
                dst_height = np.ceil(np.abs(dst_bounds.bottom - dst_bounds.top) / yres)
                dst_size = (int(dst_width), int(dst_height))
                
                # As a result of precise pixel size, the destination bounds may
                # have to be adjusted.
                x1 = dst_bounds.left + (xres*dst_width)
                y1 = dst_bounds.top - (yres*dst_height)
                dst_bounds = rio.coords.BoundingBox(top=dst_bounds.top, 
                    left=dst_bounds.left, bottom=y1, right=x1)
                

        if dst_size is not None:
            # Fix raster size at nx, ny.
            dst_shape = (self.count, dst_size[1], dst_size[0])

            # Fix nx,ny with destination bounds requested.
            if dst_bounds is not None:
                dst_transform = rio.transform.from_bounds(*dst_bounds,
                                                          width=dst_shape[2], height=dst_shape[1])
                reproj_kwargs.update({'dst_transform': dst_transform})

            dst_data = np.ones(dst_shape)
            reproj_kwargs.update({'destination': dst_data})

        # Currently reprojects all in-memory bands at once.
        # This may need to be improved to allow reprojecting from-disk.
        # See rio.warp.reproject docstring for more info.
        dst_data, dst_transformed = rio.warp.reproject(self.data, **reproj_kwargs)

        # Check for funny business.
        if dst_transform is not None:
            assert dst_transform == dst_transformed

        # Write results to a new Raster.
        dst_r = Raster.from_array(dst_data, dst_transformed, dst_crs, nodata)

        return dst_r

    def shift(self, xoff, yoff):
        """
        Translate the Raster by a given x,y offset.

        :param xoff: Translation x offset.
        :type xoff: float
        :param yoff: Translation y offset.
        :type yoff: float

        """
        meta = self.ds.meta
        dx, b, xmin, d, dy, ymax = list(self.transform)[:6]

        meta.update({'transform': rio.transform.Affine(dx, b, xmin + xoff,
                                                       d, dy, ymax + yoff)})
        self._update(metadata=meta)

    def save(self, filename, driver='GTiff', dtype=None, blank_value=None):
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
                save_data[:, :, :] = blank_value
            else:
                raise ValueError(
                    'blank_values must be one of int, float (or None).')
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

    def get_bounds_projected(self, out_crs, densify_pts_max:int=5000):
        """
        Return self's bounds in the given CRS.

        :param out_crs: Output CRS
        :type out_crs: rasterio.crs.CRS
        :param densify_pts_max: Maximum points to be added between image corners to account for non linear edges (Default 5000)
        Reduce if time computation is really critical (ms) or increase if extent is not accurate enough.
        :type densify_pts_max: int
        """
        # Max points to be added between image corners to account for non linear edges
        # rasterio's default is a bit low for very large images
        # instead, use image dimensions, with a maximum of 50000
        densify_pts = min( max(self.width, self.height), densify_pts_max)

        # Calculate new bounds
        left, bottom, right, top = self.bounds
        new_bounds = rio.warp.transform_bounds(self.crs, out_crs, left, bottom, right, top, densify_pts)

        return new_bounds
    
    
    def intersection(self, rst):
        """ 
        Returns the bounding box of intersection between this image and another.

        If the rasters have different projections, the intersection extent is given in self's projection system.
        :param rst : path to the second image (or another Raster instance)
        :type rst: str, Raster

        :returns: extent of the intersection between the 2 images \
        (xmin, ymin, xmax, ymax) in self's coordinate system.
        :rtype: tuple
        """
        from geoutils import projtools
        # If input rst is string, open as Raster
        if isinstance(rst, str):
            rst = Raster(rst, load_data=False)

        # Check if both files have the same projection
        # To be implemented
        same_proj = True

        # Find envelope of rasters' intersections
        poly1 = projtools.bounds2poly(self.bounds)
        # poly1.AssignSpatialReference(self.crs)

        # Create a polygon of the envelope of the second image
        poly2 = projtools.bounds2poly(rst.bounds)
        # poly2.AssignSpatialReference(rst.srs)

        # If coordinate system is different, reproject poly2 into poly1
        if not same_proj:
            raise NotImplementedError()

        # Compute intersection envelope
        intersect = poly1.intersection(poly2)
        extent = intersect.envelope.bounds

        # check that intersection is not void
        if intersect.area == 0:
            warnings.warn('Warning: Intersection is void')
            return 0
        else:
            return extent

    def show(self, **kwargs):
        """ Show/display the image, with axes in projection of image.

        This method is a wrapper to rasterio.plot.show. Any **kwargs which you give
        this method will be passed to rasterio.plot.show.

        You can also pass in **kwargs to be used by the underlying imshow or 
        contour methods of matplotlib. The example below shows provision of
        a kwarg for rasterio.plot.show, and a kwarg for matplotlib as well::

            import matplotlib.pyplot as plt
            ax1 = plt.subplot(111)
            mpl_kws = {'cmap':'seismic'}
            myimage.show(ax=ax1, mpl_kws)

        :returns: None
        :rtype: None

        """
        rshow(self.ds, **kwargs)

    def value_at_coords(self, x, y, latlon=False, band=None, masked=False,
                        window=None, return_window=False, boundless=True,
                        reducer_function=np.ma.mean):
        """ Extract the pixel value(s) at the specified coordinates.

        Extract pixel value of each band in dataset at the specified
        coordinates. Alternatively, if band is specified, return only that
        band's pixel value.

        Optionally, return mean of pixels within a square window.

        :param x: x (or longitude) coordinate.
        :type x: float
        :param y: y (or latitude) coordinate.
        :type y: float
        :param latlon: Set to True if coordinates provided as longitude/latitude.
        :type latlon: boolean
        :param band: the band number to extract from.
        :type band: int
        :param masked: If `masked` is `True` the return value will be a masked
        array. Otherwise (the default) the return value will be a
        regular array.
        :type masked: bool, optional (default False)
        :param window: expand area around coordinate to dimensions \
                  window * window. window must be odd.
        :type window: None, int
        :param return_window: If True when window=int, returns (mean,array) \
        where array is the dataset extracted via the specified window size.
        :type return_window: boolean
        :param boundless: If `True`, windows that extend beyond the dataset's extent
        are permitted and partially or completely filled arrays (with self.nodata) will
        be returned as appropriate.
        :type boundless: bool, optional (default False)
        :param reducer_function: a function to apply to the values in window.
        :type reducer_function: function, optional (Default is np.ma.mean)

        :returns: When called on a Raster or with a specific band \
        set, return value of pixel.
        :rtype: float
        :returns: If mutiple band Raster and the band is not specified, a \
        dictionary containing the value of the pixel in each band.
        :rtype: dict
        :returns: In addition, if return_window=True, return tuple of \
        (values, arrays)
        :rtype: tuple

        :examples:

        >>> self.value_at_coords(-48.125,67.8901,window=3)
        Returns mean of a 3*3 window:
            v v v \
            v c v  | = float(mean)
            v v v /
        (c = provided coordinate, v= value of surrounding coordinate)

        """

        if window is not None:
            if window % 2 != 1:
                raise ValueError('Window must be an odd number.')

        def format_value(value):
            """ Check if valid value has been extracted """
            if type(value) in [np.ndarray, np.ma.core.MaskedArray]:
                if window != None:
                    value = reducer_function(value.flatten())
                else:
                    value = value[0, 0]
            else:
                value = None
            return value

        # Need to implement latlon option later
        if latlon:
            raise NotImplementedError()

        # Convert coordinates to pixel space
        row, col = self.ds.index(x, y)

        # Decide what pixel coordinates to read:
        if window != None:
            half_win = (window - 1) / 2
            # Subtract start coordinates back to top left of window
            col = col - half_win
            row = row - half_win
            # Offset to read to == window
            width = window
            height = window
        else:
            # Start reading at col,row and read 1px each way
            width = 1
            height = 1

        # Make sure coordinates are int
        col = int(col)
        row = int(row)

        # Create rasterio's window for reading
        window = rio.windows.Window(col, row, width, height)

        # Get values for all bands
        if band is None:

            # Deal with single band case
            if self.nbands == 1:
                data = self.ds.read(
                    window=window, fill_value=self.nodata, boundless=boundless, masked=masked)
                value = format_value(data)
                win = data

            # Deal with multiband case
            else:
                value = {}
                win = {}

                for b in self.indexes:
                    data = self.ds.read(
                        window=window, fill_value=self.nodata, boundless=boundless, indexes=b, masked=masked)
                    val = format_value(data)
                    # Store according to GDAL band numbers
                    value[b] = val
                    win[b] = data

        # Or just for specified band in multiband case
        elif isinstance(band, int):
            data = self.ds.read(
                window=window, fill_value=self.nodata, boundless=boundless, indexes=band, masked=masked)
            value = format_value(data)
        else:
            raise ValueError(
                'Value provided for band was not int or None.')

        if return_window == True:
            return (value, win)
        else:
            return value
    
class SatelliteImage(Raster):
    pass
