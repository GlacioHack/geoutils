"""
Test functions for georaster
"""
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import numpy as np
import pytest

import rasterio as rio
from rasterio.io import MemoryFile

import geoutils.georaster as gr
from geoutils import datasets


DO_PLOT = False


class TestRaster:

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
        assert not r.isLoaded

        # Test 2 - loading the data afterward
        r.load()
        assert r.isLoaded
        assert r.nbands == 1
        assert r.data.shape == (r.count, r.height, r.width)

        # Test 3 - single band, loading data
        r = gr.Raster(datasets.get_path("landsat_B4"), load_data=True)
        assert r.isLoaded
        assert r.nbands == 1
        assert r.data.shape == (r.count, r.height, r.width)

        # Test 4 - multiple bands, load all bands
        r = gr.Raster(datasets.get_path("landsat_RGB"), load_data=True)
        assert r.count == 3
        assert r.indexes == (1, 2, 3)
        assert r.nbands == 3
        assert r.data.shape == (r.count, r.height, r.width)

        # Test 5 - multiple bands, load one band only
        r = gr.Raster(datasets.get_path("landsat_RGB"), load_data=True, bands=1)
        assert r.count == 3
        assert r.indexes == (1, 2, 3)
        assert r.nbands == 1
        assert r.data.shape == (r.nbands, r.height, r.width)

        # Test 6 - multiple bands, load a list of bands
        r = gr.Raster(datasets.get_path("landsat_RGB"), load_data=True, bands=(1,2))
        assert r.count == 3
        assert r.indexes == (1, 2, 3)
        assert r.nbands == 2
        assert r.data.shape == (r.nbands, r.height, r.width)

    def test_downsampling(self):
        """
        Check that self.data is correct when using downsampling
        """
        # Test single band
        r = gr.Raster(datasets.get_path("landsat_B4"), downsampl=4)
        assert r.data.shape == (1, 164, 200)
        assert r.height == 655  # this should not have changed
        assert r.width == 800

        # Test multiple band
        r = gr.Raster(datasets.get_path("landsat_RGB"), downsampl=2)
        assert r.data.shape == (3, 328, 400)

    def test_copy(self):

        r = gr.Raster(datasets.get_path("landsat_B4"))
        r2 = r.copy()

        # Should have no filename
        assert r2.filename is None
        #check a temporary memory file different than original disk file was created
        assert r2.name != r.name
        # Check all attributes except name and dataset_mask array
        default_attrs = ['bounds', 'count', 'crs', 'dtypes', 'height', 'indexes','nodata',
                         'res', 'shape', 'transform', 'width']
        for attr in default_attrs:
            print(attr)
            assert r.__getattribute__(attr) == r2.__getattribute__(attr)

        # Check data array
        assert np.count_nonzero(~r.data == r2.data) == 0
        # Check dataset_mask array
        assert np.count_nonzero(~r.dataset_mask() == r2.dataset_mask()) == 0

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

        # TODO: not sure what to assert here

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
