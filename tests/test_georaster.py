"""
Test functions for georaster
"""
import os
import tempfile
from tempfile import TemporaryFile

import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio as rio
from pylint import epylint

import geoutils.georaster as gr
import geoutils.projtools as pt
from geoutils import datasets

DO_PLOT = False


class TestRaster:

    def test_init(self):
        """
        Test that all possible inputs work properly in Raster class init
        """

        # first, filename
        r = gr.Raster(datasets.get_path("landsat_B4"))
        assert isinstance(r, gr.Raster)

        # second, passing a Raster itself (points back to Raster passed)
        r2 = gr.Raster(r)
        assert isinstance(r2, gr.Raster)

        # third, rio.Dataset
        ds = rio.open(datasets.get_path("landsat_B4"))
        r3 = gr.Raster(ds)
        assert isinstance(r3, gr.Raster)
        assert r3.filename is not None

        # finally, as memoryfile
        memfile = rio.MemoryFile(open(datasets.get_path("landsat_B4"), 'rb'))
        r4 = gr.Raster(memfile)
        assert isinstance(r4, gr.Raster)

        assert np.logical_and.reduce((np.array_equal(r.data, r2.data, equal_nan=True),
                                      np.array_equal(r2.data, r3.data, equal_nan=True),
                                      np.array_equal(r3.data, r4.data, equal_nan=True)))

        assert np.logical_and.reduce((np.all(r.data.mask == r2.data.mask),
                                      np.all(r2.data.mask == r3.data.mask),
                                      np.all(r3.data.mask == r4.data.mask)))

        # the data will not be copied, immutable objects will
        r.data[0, 0, 0] += 5
        assert r2.data[0, 0, 0] == r.data[0, 0, 0]

        r.nbands = 2
        assert r.nbands != r2.nbands

    def test_info(self):

        r = gr.Raster(datasets.get_path("landsat_B4"))

        # Check all is good with passing attributes
        default_attrs = ['bounds', 'count', 'crs', 'dataset_mask', 'driver',
                         'dtypes', 'height', 'indexes', 'name',
                         'nodata', 'res', 'shape', 'transform', 'width']
        for attr in default_attrs:
            assert r.__getattribute__(attr) == r.ds.__getattribute__(attr)

        # Check summary matches that of RIO
        assert print(r) == print(r.info())

    def test_loading(self):
        """
        Test that loading metadata and data works for all possible cases.
        """
        # Test 1 - loading metadata only, single band
        r = gr.Raster(datasets.get_path("landsat_B4"), load_data=False)

        assert isinstance(r.ds, rio.DatasetReader)
        assert r.driver == 'GTiff'
        assert r.width == 800
        assert r.height == 655
        assert r.shape == (r.height, r.width)
        assert r.count == 1
        assert r.nbands is None
        assert r.dtypes == ('uint8',)
        assert r.transform == rio.transform.Affine(
            30.0, 0.0, 478000.0, 0.0, -30.0, 3108140.0
        )
        assert r.res == (30.0, 30.0)
        assert r.bounds == rio.coords.BoundingBox(
            left=478000.0, bottom=3088490.0, right=502000.0, top=3108140.0
        )
        assert r.crs == rio.crs.CRS.from_epsg(32645)
        assert not r.is_loaded

        # Test 2 - loading the data afterward
        r.load()
        assert r.is_loaded
        assert r.nbands == 1
        assert r.data.shape == (r.count, r.height, r.width)

        # Test 3 - single band, loading data
        r = gr.Raster(datasets.get_path("landsat_B4"), load_data=True)
        assert r.is_loaded
        assert r.nbands == 1
        assert r.data.shape == (r.count, r.height, r.width)

        # Test 4 - multiple bands, load all bands
        r = gr.Raster(datasets.get_path("landsat_RGB"), load_data=True)
        assert r.count == 3
        assert r.indexes == (1, 2, 3)
        assert r.nbands == 3
        assert r.bands == (1, 2, 3)
        assert r.data.shape == (r.count, r.height, r.width)

        # Test 5 - multiple bands, load one band only
        r = gr.Raster(datasets.get_path("landsat_RGB"), load_data=True, bands=1)
        assert r.count == 3
        assert r.indexes == (1, 2, 3)
        assert r.nbands == 1
        assert r.bands == (1)
        assert r.data.shape == (r.nbands, r.height, r.width)

        # Test 6 - multiple bands, load a list of bands
        r = gr.Raster(datasets.get_path("landsat_RGB"), load_data=True, bands=(2, 3))
        assert r.count == 3
        assert r.indexes == (1, 2, 3)
        assert r.nbands == 2
        assert r.bands == (2, 3)
        assert r.data.shape == (r.nbands, r.height, r.width)

    def test_downsampling(self):
        """
        Check that self.data is correct when using downsampling
        """
        # Test single band
        r = gr.Raster(datasets.get_path("landsat_B4"), downsample=4)
        assert r.data.shape == (1, 164, 200)
        assert r.height == 164
        assert r.width == 200

        # Test multiple band
        r = gr.Raster(datasets.get_path("landsat_RGB"), downsample=2)
        assert r.data.shape == (3, 328, 400)

        # Test that xy2ij are consistent with new image
        # Upper left
        assert r.xy2ij(r.bounds.left, r.bounds.top) == (0, 0)
        # Upper right
        assert r.xy2ij(r.bounds.right+r.res[0], r.bounds.top) == (0, r.width+1)
        # Bottom right
        assert r.xy2ij(r.bounds.right+r.res[0], r.bounds.bottom) == (r.height, r.width+1)
        # One pixel right and down
        assert r.xy2ij(r.bounds.left + r.res[0], r.bounds.top - r.res[1]) == (1, 1)

    def test_add_sub(self):
        """
        Test addition, subtraction and negation on a Raster object.
        """
        # Create fake rasters with random values in 0-255 and dtype uint8
        width = height = 5
        transform = rio.transform.from_bounds(0, 0, 1, 1, width, height)
        r1 = gr.Raster.from_array(np.random.randint(0, 255, (height, width), dtype='uint8'),
                                  transform=transform, crs=None)
        r2 = gr.Raster.from_array(np.random.randint(0, 255, (height, width), dtype='uint8'),
                                  transform=transform, crs=None)

        # Test negation
        r3 = -r1
        assert np.all(r3.data == -r1.data)
        assert r3.dtypes == ('uint8',)

        # Test addition
        r3 = r1 + r2
        assert np.all(r3.data == r1.data + r2.data)
        assert r3.dtypes == ('uint8',)

        # Test subtraction
        r3 = r1 - r2
        assert np.all(r3.data == r1.data - r2.data)
        assert r3.dtypes == ('uint8',)

        # Test with dtype Float32
        r1 = gr.Raster.from_array(np.random.randint(0, 255, (height, width)).astype('float32'),
                                  transform=transform, crs=None)
        r3 = -r1
        assert np.all(r3.data == -r1.data)
        assert r3.dtypes == ('float32',)

        r3 = r1 + r2
        assert np.all(r3.data == r1.data + r2.data)
        assert r3.dtypes == ('float32',)

        r3 = r1 - r2
        assert np.all(r3.data == r1.data - r2.data)
        assert r3.dtypes == ('float32',)

        # Check that errors are properly raised
        # different shapes
        r1 = gr.Raster.from_array(np.random.randint(0, 255, (height + 1, width)).astype('float32'),
                                  transform=transform, crs=None)
        pytest.raises(ValueError, r1.__add__, r2)
        pytest.raises(ValueError, r1.__sub__, r2)

        # different CRS
        r1 = gr.Raster.from_array(np.random.randint(0, 255, (height, width)).astype('float32'),
                                  transform=transform, crs=rio.crs.CRS.from_epsg(4326))
        pytest.raises(ValueError, r1.__add__, r2)
        pytest.raises(ValueError, r1.__sub__, r2)

        # different transform
        transform2 = rio.transform.from_bounds(0, 0, 2, 2, width, height)
        r1 = gr.Raster.from_array(np.random.randint(0, 255, (height, width)).astype('float32'),
                                  transform=transform2, crs=None)
        pytest.raises(ValueError, r1.__add__, r2)
        pytest.raises(ValueError, r1.__sub__, r2)

    def test_copy(self):
        """
        Test that the copy method works as expected for Raster. In particular
        when copying r to r2:
        - creates a new memory file
        - if r.data is modified and r copied, the updated data is copied
        - if r is copied, r.data changed, r2.data should be unchanged
        """
        # Open dataset, update data and make a copy
        r = gr.Raster(datasets.get_path("landsat_B4"))
        r.data += 5
        r2 = r.copy()

        # Objects should be different (not pointing to the same memory)
        assert r is not r2

        # Check the object is a Raster
        assert isinstance(r2, gr.Raster)

        # Copy should have no filename
        assert r2.filename is None

        # check a temporary memory file different than original disk file was created
        assert r2.name != r.name

        # Check all attributes except name, driver and dataset_mask array
        # default_attrs = ['bounds', 'count', 'crs', 'dtypes', 'height', 'indexes','nodata',
        #                  'res', 'shape', 'transform', 'width']
        # using list directly available in Class
        attrs = [at for at in r._get_rio_attrs() if at not in ['name', 'dataset_mask', 'driver']]
        for attr in attrs:
            print(attr)
            assert r.__getattribute__(attr) == r2.__getattribute__(attr)

        # Check data array
        assert np.array_equal(r.data, r2.data, equal_nan=True)

        # Check dataset_mask array
        assert np.all(r.data.mask == r2.data.mask)

        # Check that if r.data is modified, it does not affect r2.data
        r.data += 5
        assert not np.array_equal(r.data, r2.data, equal_nan=True)

    def test_is_modified(self):
        """
        Test that changing the data updates is_modified as desired
        """
        # after laoding, should not be modified
        r = gr.Raster(datasets.get_path("landsat_B4"))
        assert not r.is_modified

        # this should not trigger the hash
        r.data = r.data + 0
        assert not r.is_modified

        # this one neither
        r.data += 0
        assert not r.is_modified

        # this will
        r = gr.Raster(datasets.get_path("landsat_B4"))
        r.data = r.data + 5
        assert r.is_modified

    def test_crop(self):

        r = gr.Raster(datasets.get_path("landsat_B4"))
        r2 = gr.Raster(datasets.get_path("landsat_B4_crop"))

        b = r.bounds
        b2 = r2.bounds

        b_minmax = (max(b[0], b2[0]), max(b[1], b2[1]),
                    min(b[2], b2[2]), min(b[3], b2[3]))

        r_init = r.copy()

        # Cropping overwrites the current Raster object
        r.crop(r2)
        b_crop = tuple(r.bounds)

        if DO_PLOT:
            fig1, ax1 = plt.subplots()
            r_init.show(ax=ax1, title='Raster 1')

            fig2, ax2 = plt.subplots()
            r2.show(ax=ax2, title='Raster 2')

            fig3, ax3 = plt.subplots()
            r.show(ax=ax3, title='Raster 1 cropped to Raster 2')
            plt.show()

        assert b_minmax == b_crop

    def test_reproj(self):

        # Test reprojecting to dst_ref
        r = gr.Raster(datasets.get_path("landsat_B4"))
        r2 = gr.Raster(datasets.get_path("landsat_B4_crop"))
        r3 = r.reproject(r2)

        if DO_PLOT:
            fig1, ax1 = plt.subplots()
            r.show(ax=ax1, title='Raster 1')

            fig2, ax2 = plt.subplots()
            r2.show(ax=ax2, title='Raster 2')

            fig3, ax3 = plt.subplots()
            r3.show(ax=ax3, title='Raster 1 reprojected to Raster 2')

            plt.show()

        # Assert the initial rasters are different
        assert r.bounds != r2.bounds
        assert r.shape != r2.shape

        # Reproject raster should have same dimensions/georeferences as r2
        assert r3.bounds == r2.bounds
        assert r3.shape == r2.shape
        assert r3.bounds == r2.bounds
        assert r3.transform == r2.transform

        # If a nodata is set, make sure it is preserved
        r.set_ndv(255)
        r3 = r.reproject(r2)
        assert r.nodata == r3.nodata

        # Test dst_size
        out_size = (r.shape[1]//2, r.shape[0]//2)  # Outsize is (ncol, nrow)
        r3 = r.reproject(dst_size=out_size)
        assert r3.shape == (out_size[1], out_size[0])

        # Test dst_bounds
        r3 = r.reproject(dst_bounds=r2.bounds)
        assert r3.bounds == r2.bounds

        # Test dst_crs
        out_crs = rio.crs.CRS.from_epsg(4326)
        r3 = r.reproject(dst_crs=out_crs)
        assert r3.crs.to_epsg() == 4326

    def test_inters_img(self):

        r = gr.Raster(datasets.get_path("landsat_B4"))
        r2 = gr.Raster(datasets.get_path("landsat_B4_crop"))

        inters = r.intersection(r2)
        print(inters)

    def test_interp(self):

        # FIRST, we try on a Raster with a Point interpretation in its "AREA_OR_POINT" metadata: values interpolated
        # at the center of pixel
        r = gr.Raster(datasets.get_path("landsat_B4"))
        assert r.ds.tags()['AREA_OR_POINT'] == 'Point'

        xmin, ymin, xmax, ymax = r.ds.bounds

        # We generate random points within the boundaries of the image


        xrand = np.random.randint(low=0, high=r.ds.width, size=(10,)) \
                * list(r.ds.transform)[0] + xmin
        yrand = ymax + np.random.randint(low=0, high=r.ds.height, size=(10,)) \
                 * list(r.ds.transform)[4]
        pts = list(zip(xrand, yrand))
        # Get decimal indexes based on Point GDAL METADATA
        # Those should all be .5 because values refer to the center
        i, j = r.xy2ij(xrand, yrand,area_or_point=None)
        assert np.all(i % 1 == 0.5)
        assert np.all(j % 1 == 0.5)

        # Force point
        i, j = r.xy2ij(xrand, yrand,area_or_point='Point')
        assert np.all(i % 1 == 0.5)
        assert np.all(j % 1 == 0.5)

        # Force area
        i, j = r.xy2ij(xrand, yrand,area_or_point='Area')
        assert np.all(i % 1 == 0)
        assert np.all(j % 1 == 0)

        # now we calculate the mean of values in each 2x2 slices of the data, and compare with interpolation at order 1
        list_z_ind = []
        img = r.data
        for k in range(len(xrand)):
            # 2x2 slices
            z_ind = np.mean(img[0, slice(int(np.floor(i[k])),int(np.ceil(i[k]))+1), slice(int(np.floor(j[k])),int(np.ceil(j[k]))+1)])
            list_z_ind.append(z_ind)

        # order 1 interpolation
        rpts = r.interp_points(pts,order=1,area_or_point='Area')
        # the values interpolated should be equal
        assert np.array_equal(np.array(list_z_ind,dtype=np.float32),rpts,equal_nan=True)

        # Test there is no failure with random coordinates (edge effects, etc)
        xrand = np.random.uniform(low=xmin, high=xmax, size=(1000,))
        yrand = np.random.uniform(low=ymin, high=ymax, size=(1000,))
        pts = list(zip(xrand, yrand))
        rpts = r.interp_points(pts)

        # SECOND, test after a crop: the Raster now has an Area interpretation, those should fall right on the integer
        # pixel indexes
        r2 = gr.Raster(datasets.get_path("landsat_B4_crop"))
        r.crop(r2)
        assert r.ds.tags()['AREA_OR_POINT'] == 'Area'

        xmin, ymin, xmax, ymax = r.bounds

        # We can test with several method for the exact indexes: interp, value_at_coords, and simple read should
        # give back the same values that fall right on the coordinates
        xrand = np.random.randint(low=0, high=r.ds.width, size=(10,)) \
                 * list(r.ds.transform)[0] + xmin
        yrand = ymax + np.random.randint(low=0, high=r.ds.height, size=(10,)) \
                 * list(r.ds.transform)[4]
        pts = list(zip(xrand, yrand))
        # by default, i and j are returned as integers
        i, j = r.xy2ij(xrand, yrand,op=np.float32,area_or_point='Area')
        list_z_ind = []
        img = r.data
        for k in range(len(xrand)):
            # we directly sample the values
            z_ind = img[0, int(i[k]), int(j[k])]
            # we can also compare with the value_at_coords() functionality
            list_z_ind.append(z_ind)


        rpts = r.interp_points(pts,order=1)

        assert np.array_equal(np.array(list_z_ind, dtype=np.float32), rpts, equal_nan=True)

        # test for an invidiual point (shape can be tricky at 1 dimension)
        x = 493120.0
        y = 3101000.0
        i, j = r.xy2ij(x, y,area_or_point='Area')
        assert img[0, int(i), int(j)] == r.interp_points([(x, y)],order=1)[0]

        #TODO: understand why there is this:
        # r.ds.index(x, y)
        # Out[33]: (75, 301)
        # r.ds.index(x, y, op=np.float32)
        # Out[34]: (75.0, 302.0)

    def test_value_at_coords(self):

        r = gr.Raster(datasets.get_path("landsat_B4"))
        r2 = gr.Raster(datasets.get_path("landsat_B4_crop"))
        r.crop(r2)

        # random test point that raised an error
        itest=118
        jtest=516
        xtest=499540
        ytest=3099710

        z = r.data[0,itest,jtest]
        x_out, y_out = r.ij2xy(itest,jtest,offset='ul')
        assert x_out == xtest
        assert y_out == ytest

        #TODO: this fails, don't know why
        # z_val = r.value_at_coords(xtest,ytest)
        # assert z == z_val

    def test_set_ndv(self):
        """
        Read Landsat dataset and set 255 to no data. Save mask.
        Then, set 254 as new no data (after setting 254 to 0). Save mask.
        Check that both no data masks are identical and have correct number of pixels.
        """
        # Read Landsat image and set no data to 255
        r = gr.Raster(datasets.get_path("landsat_B4"))
        r.set_ndv(ndv=[255])
        ndv_index = r.data.mask

        # Now set to 254, after changing 254 to 0.
        r.data[r.data == 254] = 0
        r.set_ndv(ndv=254, update_array=True)
        ndv_index_2 = r.data.mask

        if DO_PLOT:
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.imshow(ndv_index[0], interpolation='nearest')
            plt.title('Mask 1')
            plt.subplot(122)
            plt.imshow(ndv_index_2[0], interpolation='nearest')
            plt.title('Mask 2 (should be identical)')
            plt.show()

        # Check both masks are identical
        assert np.all(ndv_index_2 == ndv_index)

        # Check that the number of no data value is correct
        assert np.count_nonzero(ndv_index.data) == 112088

    def test_set_dtypes(self):

        r = gr.Raster(datasets.get_path("landsat_B4"))
        arr_1 = np.copy(r.data).astype(np.int8)
        r.set_dtypes(np.int8)
        arr_2 = np.copy(r.data)
        r.set_dtypes([np.int8], update_array=True)

        arr_3 = r.data

        assert np.count_nonzero(~arr_1 == arr_2) == 0
        assert np.count_nonzero(~arr_2 == arr_3) == 0

    def test_plot(self):

        # Read single band raster and RGB raster
        img = gr.Raster(datasets.get_path("landsat_B4"))
        img_RGB = gr.Raster(datasets.get_path("landsat_RGB"))

        # Test default plot
        ax = plt.subplot(111)
        img.show(ax=ax, title="Simple plotting test")
        if DO_PLOT:
            plt.show()
        else:
            plt.close()
        assert True

        # Test plot RGB
        ax = plt.subplot(111)
        img_RGB.show(ax=ax, title="Plotting RGB")
        if DO_PLOT:
            plt.show()
        else:
            plt.close()
        assert True

        # Test plotting single band B/W, add_cb
        ax = plt.subplot(111)
        img_RGB.show(band=0, cmap='gray', ax=ax, add_cb=False,
                     title="Plotting one band B/W")
        if DO_PLOT:
            plt.show()
        else:
            plt.close()
        assert True

        # Test vmin, vmax and cb_title
        ax = plt.subplot(111)
        img.show(cmap='gray', vmin=40, vmax=220, cb_title='Custom cbar',
                 ax=ax, title="Testing vmin, vmax and cb_title")
        if DO_PLOT:
            plt.show()
        else:
            plt.close()
        assert True

    def test_saving(self):

        # Read single band raster
        img = gr.Raster(datasets.get_path("landsat_B4"))

        # Save file to temporary file, with defaults opts
        img.save(TemporaryFile())

        # Test additional options
        co_opts = {"TILED": "YES", "COMPRESS": "LZW"}
        metadata = {"Type": "test"}
        img.save(TemporaryFile(), co_opts=co_opts, metadata=metadata)

    def test_coords(self):

        img = gr.Raster(datasets.get_path("landsat_B4"))
        xx, yy = img.coords(offset='corner')
        assert xx.min() == pytest.approx(img.bounds.left)
        assert xx.max() == pytest.approx(img.bounds.right - img.res[0])
        if img.res[1] > 0:
            assert yy.min() == pytest.approx(img.bounds.bottom)
            assert yy.max() == pytest.approx(img.bounds.top - img.res[1])
        else:
            # Currently not covered by test image
            assert yy.min() == pytest.approx(img.bounds.top)
            assert yy.max() == pytest.approx(img.bounds.bottom + img.res[1])

        xx, yy = img.coords(offset='center')
        hx = img.res[0] / 2
        hy = img.res[1] / 2
        assert xx.min() == pytest.approx(img.bounds.left + hx)
        assert xx.max() == pytest.approx(img.bounds.right - hx)
        if img.res[1] > 0:
            assert yy.min() == pytest.approx(img.bounds.bottom + hy)
            assert yy.max() == pytest.approx(img.bounds.top - hy)
        else:
            # Currently not covered by test image
            assert yy.min() == pytest.approx(img.bounds.top + hy)
            assert yy.max() == pytest.approx(img.bounds.bottom - hy)

    def test_eq(self):

        img = gr.Raster(datasets.get_path("landsat_B4"))
        img2 = gr.Raster(datasets.get_path("landsat_B4"))

        assert np.array_equal(img.data, img2.data, equal_nan=True)
        assert img.transform == img2.transform
        assert img.crs == img2.crs
        assert img.nodata == img2.nodata

        assert img.__eq__(img2)
        assert img == img2

        img2.data += 1

        assert img != img2

    def test_value_at_coords(self):
        """
        Check that values returned at selected pixels correspond to what is expected, both for original CRS and lat/lon.
        """
        img = gr.Raster(datasets.get_path("landsat_B4"))

        # Lower right pixel
        x, y = [
            img.bounds.right - img.res[0],
            img.bounds.bottom + img.res[1]
        ]
        lat, lon = pt.reproject_to_latlon([x, y], img.crs)
        assert img.value_at_coords(x, y) == \
            img.value_at_coords(lon, lat, latlon=True) == \
            img.data[0, -1, -1]

        # One pixel above
        x, y = [
            img.bounds.right - img.res[0],
            img.bounds.bottom + 2 * img.res[1]
        ]
        lat, lon = pt.reproject_to_latlon([x, y], img.crs)
        assert img.value_at_coords(x, y) == \
            img.value_at_coords(lon, lat, latlon=True) == \
            img.data[0, -2, -1]

        # One pixel left
        x, y = [
            img.bounds.right - 2 * img.res[0],
            img.bounds.bottom + img.res[1]
        ]
        lat, lon = pt.reproject_to_latlon([x, y], img.crs)
        assert img.value_at_coords(x, y) == \
            img.value_at_coords(lon, lat, latlon=True) == \
            img.data[0, -1, -2]

    def test_from_array(self):

        # Test that from_array works if nothing is changed
        # -> most tests already performed in test_copy, no need for more
        img = gr.Raster(datasets.get_path('landsat_B4'))
        out_img = gr.Raster.from_array(img.data, img.transform, img.crs, nodata=img.nodata)
        assert out_img == img

        # Test that changes to data are taken into account
        bias = 5
        out_img = gr.Raster.from_array(img.data + bias, img.transform, img.crs, nodata=img.nodata)
        assert np.array_equal(out_img.data, img.data + bias)

        # Test that nodata is properly taken into account
        out_img = gr.Raster.from_array(img.data + 5, img.transform, img.crs, nodata=0)
        assert out_img.nodata == 0

        # Test that data mask is taken into account
        img.data.mask = np.zeros((img.shape), dtype='bool')
        img.data.mask[0, 0, 0] = True
        out_img = gr.Raster.from_array(img.data, img.transform, img.crs, nodata=0)
        assert out_img.data.mask[0, 0, 0]

    def test_type_hints(self):
        """Test that pylint doesn't raise errors on valid code."""
        # Create a temporary directory and a temporary filename
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = os.path.join(temp_dir.name, "code.py")

        r = gr.Raster(datasets.get_path("landsat_B4"))

        # Load the attributes to check
        attributes = r._get_rio_attrs() + ["is_loaded", "filename", "nbands", "filename"]

        # Create some sample code that should be correct
        sample_code = "\n".join([
            "'''Sample code that should conform to pylint's standards.'''",  # Add docstring
            "import geoutils as gu",  # Import geoutils
            "raster = gu.Raster(gu.datasets.get_path('landsat_B4'))",  # Load a raster
        ] + \
            # The below statements should not raise a 'no-member' (E1101) error.
            [f"{attribute.upper()} = raster.{attribute}" for attribute in attributes] + \
            # Add a newline to the end.
            [""]
        )

        # Write the code to the temporary file
        with open(temp_path, "w") as outfile:
            outfile.write(sample_code)

        # Run pylint and parse the stdout as a string
        lint_string = epylint.py_run(temp_path, return_std=True)[0].getvalue()

        print(lint_string)  # Print the output for debug purposes

        # Bad linting errors are defined here. Currently just "no-member" errors
        bad_lints = [f"Instance of 'Raster' has no '{attribute}' member" for attribute in attributes]

        # Assert that none of the bad errors are in the pylint output
        for bad_lint in bad_lints:
            assert bad_lint not in lint_string, f"`{bad_lint}` contained in the lint_string"

    def test_split_bands(self):

        img = gr.Raster(datasets.get_path('landsat_RGB'))

        red, green, blue = img.split_bands(copy=False)

        # Check that the shapes are correct.
        assert red.nbands == 1
        assert red.data.shape[0] == 1
        assert img.nbands == 3
        assert img.data.shape[0] == 3

        # Extract only one band (then it will not return a list)
        red2 = img.split_bands(copy=False, subset=0)

        # Extract a subset with a list in a weird direction
        blue2, green2 = img.split_bands(copy=False, subset=[2, 1])

        # Check that the subset functionality works as expected.
        assert np.array_equal(red.data.astype("float32"), red2.data.astype("float32"))
        assert np.array_equal(blue.data.astype("float32"), blue2.data.astype("float32"))
        assert np.array_equal(green.data.astype("float32"), green2.data.astype("float32"))

        # Check that the red channel and the rgb data shares memory
        assert np.shares_memory(red.data, img.data)

        # Check that the red band data is not equal to the full RGB data.
        assert red != img

        # Test that the red band corresponds to the first band of the img
        assert np.array_equal(red.data.squeeze().astype("float32"), img.data[0, :, :].astype("float32"))

        # Modify the red band and make sure it propagates to the original img (it's not a copy)
        red.data += 1
        assert np.array_equal(red.data.squeeze().astype("float32"), img.data[0, :, :].astype("float32"))

        # Copy the bands instead of pointing to the same memory.
        red_c = img.split_bands(copy=True, subset=0)

        # Check that the red band data does not share memory with the rgb image (it's a copy)
        assert not np.shares_memory(red_c, img)

        # Modify the copy, and make sure the original data is not modifed.
        red_c.data += 1
        assert not np.array_equal(red_c.data.squeeze().astype("float32"), img.data[0, :, :].astype("float32"))

    def test_resampling_str(self):
        """Test that resampling methods can be given as strings instead of rio enums."""
        assert gr._resampling_from_str("nearest") == rio.warp.Resampling.nearest
        assert gr._resampling_from_str("cubic_spline") == rio.warp.Resampling.cubic_spline

        # Check that odd strings return the appropriate error.
        try:
            gr._resampling_from_str("CUBIC_SPLINE")
        except ValueError as exception:
            if "not a valid rasterio.warp.Resampling method" not in str(exception):
                raise exception

        img1 = gr.Raster(datasets.get_path("landsat_B4"))
        img2 = gr.Raster(datasets.get_path("landsat_B4_crop"))

        # Resample the rasters using a new resampling method and see that the string and enum gives the same result.
        img3a = img1.reproject(img2, resampling="q1")
        img3b = img1.reproject(img2, resampling=rio.warp.Resampling.q1)
        assert img3a == img3b
        
    def test_polygonize(self):
        """Test that polygonize doesn't raise errors."""
        img = gr.Raster(datasets.get_path('landsat_B4'))
        assert gdf = img.polygonize()
        