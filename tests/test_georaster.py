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
        assert r.xy2ij(r.bounds.right+r.res[0], r.bounds.top) == (0, r.width)
        # Bottom right
        assert r.xy2ij(r.bounds.right+r.res[0], r.bounds.bottom) == (r.height, r.width)
        # One pixel right and down
        assert r.xy2ij(r.bounds.left + r.res[0], r.bounds.top - r.res[1]) == (1, 1)

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

        r = gr.Raster(datasets.get_path("landsat_B4"))

        xmin, ymin, xmax, ymax = r.ds.bounds

        # testing interp, find_value, and read when it falls right on the coordinates
        xrand = (np.random.randint(low=0, high=r.ds.width, size=(10,))
                 * list(r.ds.transform)[0] + xmin + list(r.ds.transform)[0]/2)
        yrand = (ymax + np.random.randint(low=0, high=r.ds.height, size=(10,))
                 * list(r.ds.transform)[4] - list(r.ds.transform)[4]/2)
        pts = list(zip(xrand, yrand))
        i, j = r.xy2ij(xrand, yrand)
        list_z = []
        list_z_ind = []
        r.load()
        img = r.data
        for k in range(len(xrand)):
            z_ind = img[0, i[k], j[k]]
            z = r.value_at_coords(xrand[k], yrand[k])
            list_z_ind.append(z_ind)
            list_z.append(z)

        rpts = r.interp_points(pts)
        print(list_z_ind)
        print(list_z)
        print(rpts)

        # Individual tests
        x = 493135.0
        y = 3104015.0
        print(r.value_at_coords(x, y))
        i, j = r.xy2ij(x, y)
        print(img[0, i, j])
        print(r.interp_points([(x, y)]))

        # random float
        xrand = np.random.uniform(low=xmin, high=xmax, size=(1000,))
        yrand = np.random.uniform(low=ymin, high=ymax, size=(1000,))
        pts = list(zip(xrand, yrand))
        rpts = r.interp_points(pts)
        # print(rpts)

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
