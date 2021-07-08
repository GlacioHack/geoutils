"""
Test functions for georaster
"""
import os
import tempfile
import warnings
from tempfile import TemporaryFile

import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio as rio
from pylint import epylint

import geoutils as gu
import geoutils.georaster as gr
import geoutils.geovector as gv
import geoutils.projtools as pt
from geoutils import datasets

DO_PLOT = False


class TestRaster:
    def test_init(self) -> None:
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
        memfile = rio.MemoryFile(open(datasets.get_path("landsat_B4"), "rb"))
        r4 = gr.Raster(memfile)
        assert isinstance(r4, gr.Raster)

        assert np.logical_and.reduce(
            (
                np.array_equal(r.data, r2.data, equal_nan=True),
                np.array_equal(r2.data, r3.data, equal_nan=True),
                np.array_equal(r3.data, r4.data, equal_nan=True),
            )
        )

        assert np.logical_and.reduce(
            (
                np.all(r.data.mask == r2.data.mask),
                np.all(r2.data.mask == r3.data.mask),
                np.all(r3.data.mask == r4.data.mask),
            )
        )

        # the data will not be copied, immutable objects will
        r.data[0, 0, 0] += 5
        assert r2.data[0, 0, 0] == r.data[0, 0, 0]

        r.nbands = 2
        assert r.nbands != r2.nbands

    def test_info(self) -> None:

        r = gr.Raster(datasets.get_path("landsat_B4"))

        # Check all is good with passing attributes
        default_attrs = [
            "bounds",
            "count",
            "crs",
            "dataset_mask",
            "driver",
            "dtypes",
            "height",
            "indexes",
            "name",
            "nodata",
            "res",
            "shape",
            "transform",
            "width",
        ]
        for attr in default_attrs:
            assert r.__getattribute__(attr) == r.ds.__getattribute__(attr)

        # Check summary matches that of RIO
        assert str(r) == r.info()

        # Check that the stats=True flag doesn't trigger a warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            stats = r.info(stats=True)

        # Validate that the mask is respected by adding 0 values (there are none to begin with.)
        r.data.ravel()[:1000] = 0
        # Set the nodata value to 0, then validate that they are excluded from the new minimum
        r.set_ndv(0)
        new_stats = r.info(stats=True)
        for i, line in enumerate(stats.splitlines()):
            if "MINIMUM" not in line:
                continue
            assert line == new_stats.splitlines()[i]

    def test_loading(self) -> None:
        """
        Test that loading metadata and data works for all possible cases.
        """
        # Test 1 - loading metadata only, single band
        r = gr.Raster(datasets.get_path("landsat_B4"), load_data=False)

        assert isinstance(r.ds, rio.DatasetReader)
        assert r.driver == "GTiff"
        assert r.width == 800
        assert r.height == 655
        assert r.shape == (r.height, r.width)
        assert r.count == 1
        assert r.nbands is None
        assert np.array_equal(r.dtypes, ["uint8"])
        assert r.transform == rio.transform.Affine(30.0, 0.0, 478000.0, 0.0, -30.0, 3108140.0)
        assert np.array_equal(r.res, [30.0, 30.0])
        assert r.bounds == rio.coords.BoundingBox(left=478000.0, bottom=3088490.0, right=502000.0, top=3108140.0)
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
        assert np.array_equal(r.indexes, [1, 2, 3])
        assert r.nbands == 3
        assert np.array_equal(r.bands, [1, 2, 3])
        assert r.data.shape == (r.count, r.height, r.width)

        # Test 5 - multiple bands, load one band only
        r = gr.Raster(datasets.get_path("landsat_RGB"), load_data=True, bands=1)
        assert r.count == 3
        assert np.array_equal(r.indexes, [1, 2, 3])
        assert r.nbands == 1
        assert r.bands == (1)
        assert r.data.shape == (r.nbands, r.height, r.width)

        # Test 6 - multiple bands, load a list of bands
        r = gr.Raster(datasets.get_path("landsat_RGB"), load_data=True, bands=[2, 3])
        assert r.count == 3
        assert np.array_equal(r.indexes, [1, 2, 3])
        assert r.nbands == 2
        assert np.array_equal(r.bands, (2, 3))
        assert r.data.shape == (r.nbands, r.height, r.width)

    def test_downsampling(self) -> None:
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
        assert r.xy2ij(r.bounds.right + r.res[0], r.bounds.top) == (0, r.width + 1)
        # Bottom right
        assert r.xy2ij(r.bounds.right + r.res[0], r.bounds.bottom) == (r.height, r.width + 1)
        # One pixel right and down
        assert r.xy2ij(r.bounds.left + r.res[0], r.bounds.top - r.res[1]) == (1, 1)

    def test_add_sub(self) -> None:
        """
        Test addition, subtraction and negation on a Raster object.
        """
        # Create fake rasters with random values in 0-255 and dtype uint8
        width = height = 5
        transform = rio.transform.from_bounds(0, 0, 1, 1, width, height)
        r1 = gr.Raster.from_array(
            np.random.randint(0, 255, (height, width), dtype="uint8"), transform=transform, crs=None
        )
        r2 = gr.Raster.from_array(
            np.random.randint(0, 255, (height, width), dtype="uint8"), transform=transform, crs=None
        )

        # Test negation
        r3 = -r1
        assert np.all(r3.data == -r1.data)
        assert np.array_equal(r3.dtypes, ["uint8"])

        # Test addition
        r3 = r1 + r2
        assert np.all(r3.data == r1.data + r2.data)
        assert np.array_equal(r3.dtypes, ["uint8"])

        # Test subtraction
        r3 = r1 - r2
        assert np.all(r3.data == r1.data - r2.data)
        assert np.array_equal(r3.dtypes, ["uint8"])

        # Test with dtype Float32
        r1 = gr.Raster.from_array(
            np.random.randint(0, 255, (height, width)).astype("float32"), transform=transform, crs=None
        )
        r3 = -r1
        assert np.all(r3.data == -r1.data)
        assert np.array_equal(r3.dtypes, ["float32"])

        r3 = r1 + r2
        assert np.all(r3.data == r1.data + r2.data)
        assert np.array_equal(r3.dtypes, ["float32"])

        r3 = r1 - r2
        assert np.all(r3.data == r1.data - r2.data)
        assert np.array_equal(r3.dtypes, ["float32"])

        # Check that errors are properly raised
        # different shapes
        r1 = gr.Raster.from_array(
            np.random.randint(0, 255, (height + 1, width)).astype("float32"), transform=transform, crs=None
        )
        expected_message = "Both rasters must have the same shape, transform and CRS."
        with pytest.raises(ValueError, match=expected_message):
            r1.__add__(r2)

        with pytest.raises(ValueError, match=expected_message):
            r1.__sub__(r2)

        # different CRS
        r1 = gr.Raster.from_array(
            np.random.randint(0, 255, (height, width)).astype("float32"),
            transform=transform,
            crs=rio.crs.CRS.from_epsg(4326),
        )

        with pytest.raises(ValueError, match=expected_message):
            r1.__add__(r2)

        with pytest.raises(ValueError, match=expected_message):
            r1.__sub__(r2)

        # different transform
        transform2 = rio.transform.from_bounds(0, 0, 2, 2, width, height)
        r1 = gr.Raster.from_array(
            np.random.randint(0, 255, (height, width)).astype("float32"), transform=transform2, crs=None
        )

        with pytest.raises(ValueError, match=expected_message):
            r1.__add__(r2)

        with pytest.raises(ValueError, match=expected_message):
            r1.__sub__(r2)

    def test_copy(self) -> None:
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
        attrs = [at for at in r._get_rio_attrs() if at not in ["name", "dataset_mask", "driver"]]
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

    def test_is_modified(self) -> None:
        """
        Test that changing the data updates is_modified as desired
        """
        # after loading, should not be modified
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

    def test_crop(self) -> None:

        r = gr.Raster(datasets.get_path("landsat_B4"))
        r2 = gr.Raster(datasets.get_path("landsat_B4_crop"))

        # Read a vector and extract only the largest outline within the extent of r
        outlines = gu.Vector(datasets.get_path("glacier_outlines"))
        outlines.ds = outlines.ds.to_crs(r.crs)
        outlines.crop2raster(r)
        outlines = outlines.query(f"index == {np.argmax(outlines.ds.geometry.area)}")

        # Crop the raster to the outline and validate that it got smaller
        r_outline_cropped = r.crop(outlines, inplace=False)
        assert r.data.size > r_outline_cropped.data.size  # type: ignore

        b = r.bounds
        b2 = r2.bounds

        b_minmax = (max(b[0], b2[0]), max(b[1], b2[1]), min(b[2], b2[2]), min(b[3], b2[3]))

        r_init = r.copy()

        # Cropping overwrites the current Raster object
        r.crop(r2)
        b_crop = tuple(r.bounds)

        if DO_PLOT:
            fig1, ax1 = plt.subplots()
            r_init.show(ax=ax1, title="Raster 1")

            fig2, ax2 = plt.subplots()
            r2.show(ax=ax2, title="Raster 2")

            fig3, ax3 = plt.subplots()
            r.show(ax=ax3, title="Raster 1 cropped to Raster 2")
            plt.show()

        assert b_minmax == b_crop

    def test_reproj(self) -> None:

        # Reference raster to be used
        r = gr.Raster(datasets.get_path("landsat_B4"))
        r.set_ndv(0)  # to avoid warnings - will be used when reprojecting outside bounds

        # A second raster with different bounds, shape and resolution
        r2 = gr.Raster(datasets.get_path("landsat_B4_crop"))
        r2 = r2.reproject(dst_res=20)
        assert r2.res == (20, 20)

        # Assert the initial rasters are different
        assert r.bounds != r2.bounds
        assert r.shape != r2.shape
        assert r.res != r2.res

        # Test reprojecting to dst_ref
        # Reproject raster should have same dimensions/georeferences as r2
        r3 = r.reproject(r2)
        assert r3.bounds == r2.bounds
        assert r3.shape == r2.shape
        assert r3.bounds == r2.bounds
        assert r3.transform == r2.transform

        if DO_PLOT:
            fig1, ax1 = plt.subplots()
            r.show(ax=ax1, title="Raster 1")

            fig2, ax2 = plt.subplots()
            r2.show(ax=ax2, title="Raster 2")

            fig3, ax3 = plt.subplots()
            r3.show(ax=ax3, title="Raster 1 reprojected to Raster 2")

            plt.show()

        # If a nodata is set, make sure it is preserved
        r.set_ndv(255)
        r3 = r.reproject(r2)
        assert r.nodata == r3.nodata

        # Test dst_size - this should modify the shape, and hence resolution, but not the bounds
        out_size = (r.shape[1] // 2, r.shape[0] // 2)  # Outsize is (ncol, nrow)
        r3 = r.reproject(dst_size=out_size)
        assert r3.shape == (out_size[1], out_size[0])
        assert r3.bounds == r.bounds

        # Test dst_bounds
        # if bounds is a multiple of res, outptut res should be preserved
        bounds = np.copy(r.bounds)
        dst_bounds = rio.coords.BoundingBox(
            left=bounds[0], bottom=bounds[1] + r.res[0], right=bounds[2] - 2 * r.res[1], top=bounds[3]
        )
        r3 = r.reproject(dst_bounds=dst_bounds)
        assert r3.bounds == dst_bounds
        assert r3.res == r.res

        # Create bounds with 1/2 and 1/3 pixel extra on the right/bottom.
        bounds = np.copy(r.bounds)
        dst_bounds = rio.coords.BoundingBox(
            left=bounds[0], bottom=bounds[1] - r.res[0] / 3.0, right=bounds[2] + r.res[1] / 2.0, top=bounds[3]
        )

        # if bounds are not a multiple of res, the latter will be updated accordingly
        r3 = r.reproject(dst_bounds=dst_bounds)
        assert r3.bounds == dst_bounds
        assert r3.res != r.res

        # If dst_res is set, the resolution will be enforced
        # Bounds will be enforced for upper-left pixel, but adjusted by up to one pixel for the lower right bound.
        r3 = r.reproject(dst_bounds=dst_bounds, dst_res=r.res)
        assert r3.res == r.res
        assert r3.bounds.left == dst_bounds.left
        assert r3.bounds.top == dst_bounds.top
        assert np.abs(r3.bounds.right - dst_bounds.right) < r3.res[1]
        assert np.abs(r3.bounds.bottom - dst_bounds.bottom) < r3.res[0]

        # Test dst_crs
        out_crs = rio.crs.CRS.from_epsg(4326)
        r3 = r.reproject(dst_crs=out_crs)
        assert r3.crs.to_epsg() == 4326

    def test_inters_img(self) -> None:

        r = gr.Raster(datasets.get_path("landsat_B4"))
        r2 = gr.Raster(datasets.get_path("landsat_B4_crop"))

        inters = r.intersection(r2)
        print(inters)

    def test_interp(self) -> None:

        # FIRST, we try on a Raster with a Point interpretation in its "AREA_OR_POINT" metadata: values interpolated
        # at the center of pixel
        r = gr.Raster(datasets.get_path("landsat_B4"))
        assert r.ds.tags()["AREA_OR_POINT"] == "Point"

        xmin, ymin, xmax, ymax = r.ds.bounds

        # We generate random points within the boundaries of the image

        xrand = np.random.randint(low=0, high=r.ds.width, size=(10,)) * list(r.ds.transform)[0] + xmin
        yrand = ymax + np.random.randint(low=0, high=r.ds.height, size=(10,)) * list(r.ds.transform)[4]
        pts = list(zip(xrand, yrand))
        # Get decimal indexes based on Point GDAL METADATA
        # Those should all be .5 because values refer to the center
        i, j = r.xy2ij(xrand, yrand, area_or_point=None)
        assert np.all(i % 1 == 0.5)
        assert np.all(j % 1 == 0.5)

        # Force point
        i, j = r.xy2ij(xrand, yrand, area_or_point="Point")
        assert np.all(i % 1 == 0.5)
        assert np.all(j % 1 == 0.5)

        # Force area
        i, j = r.xy2ij(xrand, yrand, area_or_point="Area")
        assert np.all(i % 1 == 0)
        assert np.all(j % 1 == 0)

        # now we calculate the mean of values in each 2x2 slices of the data, and compare with interpolation at order 1
        list_z_ind = []
        img = r.data
        for k in range(len(xrand)):
            # 2x2 slices
            z_ind = np.mean(
                img[
                    0,
                    slice(int(np.floor(i[k])), int(np.ceil(i[k])) + 1),
                    slice(int(np.floor(j[k])), int(np.ceil(j[k])) + 1),
                ]
            )
            list_z_ind.append(z_ind)

        # order 1 interpolation
        rpts = r.interp_points(pts, order=1, area_or_point="Area")
        # the values interpolated should be equal
        assert np.array_equal(np.array(list_z_ind, dtype=np.float32), rpts, equal_nan=True)

        # Test there is no failure with random coordinates (edge effects, etc)
        xrand = np.random.uniform(low=xmin, high=xmax, size=(1000,))
        yrand = np.random.uniform(low=ymin, high=ymax, size=(1000,))
        pts = list(zip(xrand, yrand))
        rpts = r.interp_points(pts)

        # SECOND, test after a crop: the Raster now has an Area interpretation, those should fall right on the integer
        # pixel indexes
        r2 = gr.Raster(datasets.get_path("landsat_B4_crop"))
        r.crop(r2)
        assert r.ds.tags()["AREA_OR_POINT"] == "Area"

        xmin, ymin, xmax, ymax = r.bounds

        # We can test with several method for the exact indexes: interp, value_at_coords, and simple read should
        # give back the same values that fall right on the coordinates
        xrand = np.random.randint(low=0, high=r.ds.width, size=(10,)) * list(r.ds.transform)[0] + xmin
        yrand = ymax + np.random.randint(low=0, high=r.ds.height, size=(10,)) * list(r.ds.transform)[4]
        pts = list(zip(xrand, yrand))
        # by default, i and j are returned as integers
        i, j = r.xy2ij(xrand, yrand, op=np.float32, area_or_point="Area")
        list_z_ind = []
        img = r.data
        for k in range(len(xrand)):
            # we directly sample the values
            z_ind = img[0, int(i[k]), int(j[k])]
            # we can also compare with the value_at_coords() functionality
            list_z_ind.append(z_ind)

        rpts = r.interp_points(pts, order=1)

        assert np.array_equal(np.array(list_z_ind, dtype=np.float32), rpts, equal_nan=True)

        # test for an invidiual point (shape can be tricky at 1 dimension)
        x = 493120.0
        y = 3101000.0
        i, j = r.xy2ij(x, y, area_or_point="Area")
        assert img[0, int(i), int(j)] == r.interp_points([(x, y)], order=1)[0]

        # TODO: understand why there is this:
        # r.ds.index(x, y)
        # Out[33]: (75, 301)
        # r.ds.index(x, y, op=np.float32)
        # Out[34]: (75.0, 302.0)

    def test_value_at_coords(self) -> None:

        r = gr.Raster(datasets.get_path("landsat_B4"))
        r2 = gr.Raster(datasets.get_path("landsat_B4_crop"))
        r.crop(r2)

        # random test point that raised an error
        itest = 118
        jtest = 516
        xtest = 499540
        ytest = 3099710

        # z = r.data[0, itest, jtest]
        x_out, y_out = r.ij2xy(itest, jtest, offset="ul")
        assert x_out == xtest
        assert y_out == ytest

        # TODO: this fails, don't know why
        # z_val = r.value_at_coords(xtest,ytest)
        # assert z == z_val

    def test_set_ndv(self) -> None:
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
            plt.imshow(ndv_index[0], interpolation="nearest")
            plt.title("Mask 1")
            plt.subplot(122)
            plt.imshow(ndv_index_2[0], interpolation="nearest")
            plt.title("Mask 2 (should be identical)")
            plt.show()

        # Check both masks are identical
        assert np.all(ndv_index_2 == ndv_index)

        # Check that the number of no data value is correct
        assert np.count_nonzero(ndv_index.data) == 112088

    def test_set_dtypes(self) -> None:

        r = gr.Raster(datasets.get_path("landsat_B4"))
        arr_1 = np.copy(r.data).astype(np.int8)
        r.set_dtypes(np.int8)
        arr_2 = np.copy(r.data)
        r.set_dtypes([np.int8], update_array=True)

        arr_3 = r.data

        assert np.count_nonzero(~arr_1 == arr_2) == 0
        assert np.count_nonzero(~arr_2 == arr_3) == 0

    def test_plot(self) -> None:

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
        img_RGB.show(band=0, cmap="gray", ax=ax, add_cb=False, title="Plotting one band B/W")
        if DO_PLOT:
            plt.show()
        else:
            plt.close()
        assert True

        # Test vmin, vmax and cb_title
        ax = plt.subplot(111)
        img.show(cmap="gray", vmin=40, vmax=220, cb_title="Custom cbar", ax=ax, title="Testing vmin, vmax and cb_title")
        if DO_PLOT:
            plt.show()
        else:
            plt.close()
        assert True

    def test_saving(self) -> None:

        # Read single band raster
        img = gr.Raster(datasets.get_path("landsat_B4"))

        # Save file to temporary file, with defaults opts
        img.save(TemporaryFile())

        # Test additional options
        co_opts = {"TILED": "YES", "COMPRESS": "LZW"}
        metadata = {"Type": "test"}
        img.save(TemporaryFile(), co_opts=co_opts, metadata=metadata)

    def test_coords(self) -> None:

        img = gr.Raster(datasets.get_path("landsat_B4"))
        xx, yy = img.coords(offset="corner")
        assert xx.min() == pytest.approx(img.bounds.left)
        assert xx.max() == pytest.approx(img.bounds.right - img.res[0])
        if img.res[1] > 0:
            assert yy.min() == pytest.approx(img.bounds.bottom)
            assert yy.max() == pytest.approx(img.bounds.top - img.res[1])
        else:
            # Currently not covered by test image
            assert yy.min() == pytest.approx(img.bounds.top)
            assert yy.max() == pytest.approx(img.bounds.bottom + img.res[1])

        xx, yy = img.coords(offset="center")
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

    def test_eq(self) -> None:

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

    def test_value_at_coords2(self) -> None:
        """
        Check that values returned at selected pixels correspond to what is expected, both for original CRS and lat/lon.
        """
        img = gr.Raster(datasets.get_path("landsat_B4"))

        # Lower right pixel
        x, y = [img.bounds.right - img.res[0], img.bounds.bottom + img.res[1]]
        lat, lon = pt.reproject_to_latlon([x, y], img.crs)
        assert img.value_at_coords(x, y) == img.value_at_coords(lon, lat, latlon=True) == img.data[0, -1, -1]

        # One pixel above
        x, y = [img.bounds.right - img.res[0], img.bounds.bottom + 2 * img.res[1]]
        lat, lon = pt.reproject_to_latlon([x, y], img.crs)
        assert img.value_at_coords(x, y) == img.value_at_coords(lon, lat, latlon=True) == img.data[0, -2, -1]

        # One pixel left
        x, y = [img.bounds.right - 2 * img.res[0], img.bounds.bottom + img.res[1]]
        lat, lon = pt.reproject_to_latlon([x, y], img.crs)
        assert img.value_at_coords(x, y) == img.value_at_coords(lon, lat, latlon=True) == img.data[0, -1, -2]

    def test_from_array(self) -> None:

        # Test that from_array works if nothing is changed
        # -> most tests already performed in test_copy, no need for more
        img = gr.Raster(datasets.get_path("landsat_B4"))
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
        img.data.mask = np.zeros((img.shape), dtype="bool")
        img.data.mask[0, 0, 0] = True
        out_img = gr.Raster.from_array(img.data, img.transform, img.crs, nodata=0)
        assert out_img.data.mask[0, 0, 0]

    def test_type_hints(self) -> None:
        """Test that pylint doesn't raise errors on valid code."""
        # Create a temporary directory and a temporary filename
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = os.path.join(temp_dir.name, "code.py")

        r = gr.Raster(datasets.get_path("landsat_B4"))

        # Load the attributes to check
        attributes = r._get_rio_attrs() + ["is_loaded", "filename", "nbands", "filename"]

        # Create some sample code that should be correct
        sample_code = "\n".join(
            [
                "'''Sample code that should conform to pylint's standards.'''",  # Add docstring
                "import geoutils as gu",  # Import geoutils
                "raster = gu.Raster(gu.datasets.get_path('landsat_B4'))",  # Load a raster
            ]
            + [  # The below statements should not raise a 'no-member' (E1101) error.
                f"{attribute.upper()} = raster.{attribute}" for attribute in attributes
            ]
            + [""]  # Add a newline to the end.
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

    def test_split_bands(self) -> None:

        img = gr.Raster(datasets.get_path("landsat_RGB"))

        red, green, blue = img.split_bands(copy=False)

        # Check that the shapes are correct.
        assert red.nbands == 1
        assert red.data.shape[0] == 1
        assert img.nbands == 3
        assert img.data.shape[0] == 3

        # Extract only one band (then it will not return a list)
        red2 = img.split_bands(copy=False, subset=0)[0]

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
        red_c = img.split_bands(copy=True, subset=0)[0]

        # Check that the red band data does not share memory with the rgb image (it's a copy)
        assert not np.shares_memory(red_c, img)

        # Modify the copy, and make sure the original data is not modified.
        red_c.data += 1
        assert not np.array_equal(red_c.data.squeeze().astype("float32"), img.data[0, :, :].astype("float32"))

    def test_resampling_str(self) -> None:
        """Test that resampling methods can be given as strings instead of rio enums."""
        assert gr._resampling_from_str("nearest") == rio.warp.Resampling.nearest  # noqa
        assert gr._resampling_from_str("cubic_spline") == rio.warp.Resampling.cubic_spline  # noqa

        # Check that odd strings return the appropriate error.
        try:
            gr._resampling_from_str("CUBIC_SPLINE")  # noqa
        except ValueError as exception:
            if "not a valid rasterio.warp.Resampling method" not in str(exception):
                raise exception

        img1 = gr.Raster(datasets.get_path("landsat_B4"))
        img2 = gr.Raster(datasets.get_path("landsat_B4_crop"))

        # Resample the rasters using a new resampling method and see that the string and enum gives the same result.
        img3a = img1.reproject(img2, resampling="q1")
        img3b = img1.reproject(img2, resampling=rio.warp.Resampling.q1)
        assert img3a == img3b

    def test_polygonize(self) -> None:
        """Test that polygonize doesn't raise errors."""
        img = gr.Raster(datasets.get_path("landsat_B4"))

        value = np.unique(img)[0]

        pixel_area = np.sum(img == value) * img.res[0] * img.res[1]

        polygonized = img.polygonize(value)

        polygon_area = polygonized.ds.area.sum()

        assert polygon_area == pytest.approx(pixel_area)
        assert isinstance(polygonized, gv.Vector)

    def test_to_points(self) -> None:
        """Test the outputs of the to_points method and that it doesn't load if not needed."""
        # Create a small raster to test point sampling on
        img1 = gu.Raster.from_array(
            np.arange(25, dtype="int32").reshape(5, 5), transform=rio.transform.from_origin(0, 5, 1, 1), crs=4326
        )

        # Sample the whole raster (fraction==1)
        points = img1.to_points(1)

        # Validate that 25 points were sampled (equating to img1.height * img1.width) with x, y, and band0 values.
        assert isinstance(points, np.ndarray)
        assert points.shape == (25, 3)
        assert np.array_equal(np.asarray(points[:, 0]), np.tile(np.linspace(0.5, 4.5, 5), 5))

        assert img1.to_points(0.2).shape == (5, 3)

        img2 = gu.Raster(datasets.get_path("landsat_RGB"), load_data=False)

        points = img2.to_points(10)

        assert points.shape == (10, 5)
        assert not img2.is_loaded

        points_frame = img2.to_points(10, as_frame=True)

        assert np.array_equal(points_frame.columns, ["b1", "b2", "b3", "geometry"])
        assert points_frame.crs == img2.crs


@pytest.mark.parametrize("dtype", ["float32", "uint8", "int32"])  # type: ignore
def test_numpy_functions(dtype: str) -> None:
    """Test how rasters can be used as/with numpy arrays."""
    warnings.simplefilter("error")

    # Create an array of unique values starting at 0 and ending at 24
    array = np.arange(25, dtype=dtype).reshape((1, 5, 5))
    # Create an associated dummy transform
    transform = rio.transform.from_origin(0, 5, 1, 1)

    # Create a raster from the array
    raster = gu.Raster.from_array(array, transform=transform, crs=4326)

    # Test some ufuncs
    assert np.median(raster) == 12.0
    assert np.mean(raster) == 12.0

    # Check that rasters don't  become arrays when using simple arithmetic.
    assert isinstance(raster + 1, gr.Raster)

    # Test that array_equal works
    assert np.array_equal(array, raster)

    # Test the data setter method by creating a new array
    raster.data = array + 2

    # Check that the median updated accordingly.
    assert np.median(raster) == 14.0

    # Test
    raster += array

    assert isinstance(raster, gr.Raster)
    assert np.median(raster) == 26.0
