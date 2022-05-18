"""
Test functions for georaster
"""
from __future__ import annotations

import os
import tempfile
import warnings
from tempfile import NamedTemporaryFile, TemporaryFile

import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio as rio
from pylint import epylint

import geoutils as gu
import geoutils.georaster as gr
import geoutils.geovector as gv
import geoutils.misc
import geoutils.projtools as pt
from geoutils import datasets
from geoutils.georaster.raster import _default_ndv
from geoutils.misc import resampling_method_from_str

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
                geoutils.misc.array_equal(r.data, r2.data, equal_nan=True),
                geoutils.misc.array_equal(r2.data, r3.data, equal_nan=True),
                geoutils.misc.array_equal(r3.data, r4.data, equal_nan=True),
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
        assert geoutils.misc.array_equal(r.dtypes, ["uint8"])
        assert r.transform == rio.transform.Affine(30.0, 0.0, 478000.0, 0.0, -30.0, 3108140.0)
        assert geoutils.misc.array_equal(r.res, [30.0, 30.0])
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
        assert geoutils.misc.array_equal(r.indexes, [1, 2, 3])
        assert r.nbands == 3
        assert geoutils.misc.array_equal(r.bands, [1, 2, 3])
        assert r.data.shape == (r.count, r.height, r.width)

        # Test 5 - multiple bands, load one band only
        r = gr.Raster(datasets.get_path("landsat_RGB"), load_data=True, bands=1)
        assert r.count == 3
        assert geoutils.misc.array_equal(r.indexes, [1, 2, 3])
        assert r.nbands == 1
        assert r.bands == (1)
        assert r.data.shape == (r.nbands, r.height, r.width)

        # Test 6 - multiple bands, load a list of bands
        r = gr.Raster(datasets.get_path("landsat_RGB"), load_data=True, bands=[2, 3])
        assert r.count == 3
        assert geoutils.misc.array_equal(r.indexes, [1, 2, 3])
        assert r.nbands == 2
        assert geoutils.misc.array_equal(r.bands, (2, 3))
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
        assert geoutils.misc.array_equal(r3.dtypes, ["uint8"])

        # Test addition
        r3 = r1 + r2
        assert np.all(r3.data == r1.data + r2.data)
        assert geoutils.misc.array_equal(r3.dtypes, ["uint8"])

        # Test subtraction
        r3 = r1 - r2
        assert np.all(r3.data == r1.data - r2.data)
        assert geoutils.misc.array_equal(r3.dtypes, ["uint8"])

        # Test with dtype Float32
        r1 = gr.Raster.from_array(
            np.random.randint(0, 255, (height, width)).astype("float32"), transform=transform, crs=None
        )
        r3 = -r1
        assert np.all(r3.data == -r1.data)
        assert geoutils.misc.array_equal(r3.dtypes, ["float32"])

        r3 = r1 + r2
        assert np.all(r3.data == r1.data + r2.data)
        assert geoutils.misc.array_equal(r3.dtypes, ["float32"])

        r3 = r1 - r2
        assert np.all(r3.data == r1.data - r2.data)
        assert geoutils.misc.array_equal(r3.dtypes, ["float32"])

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
        assert geoutils.misc.array_equal(r.data, r2.data, equal_nan=True)

        # Check dataset_mask array
        assert np.all(r.data.mask == r2.data.mask)

        # Check that if r.data is modified, it does not affect r2.data
        r.data += 5
        assert not geoutils.misc.array_equal(r.data, r2.data, equal_nan=True)

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

    @pytest.mark.parametrize("dataset", ["landsat_B4", "landsat_RGB"])  # type: ignore
    def test_masking(self, dataset: str) -> None:
        """
        Test self.set_mask
        """
        # Test boolean mask
        r = gr.Raster(datasets.get_path(dataset))
        mask = r.data == np.min(r.data)
        r.set_mask(mask)
        assert (np.count_nonzero(mask) > 0) & np.array_equal(mask > 0, r.data.mask)

        # Test non boolean mask with values > 0
        r = gr.Raster(datasets.get_path(dataset))
        mask = np.where(r.data == np.min(r.data), 32, 0)
        r.set_mask(mask)
        assert (np.count_nonzero(mask) > 0) & np.array_equal(mask > 0, r.data.mask)

        # Test that previous mask is also preserved
        mask2 = r.data == np.max(r.data)
        assert np.count_nonzero(mask2) > 0
        r.set_mask(mask2)
        assert np.array_equal((mask > 0) | (mask2 > 0), r.data.mask)
        assert np.count_nonzero(~r.data.mask[mask > 0]) == 0

        # Test that shape of first dimension is ignored if equal to 1
        r = gr.Raster(datasets.get_path(dataset))
        if r.data.shape[0] == 1:
            mask = (r.data == np.min(r.data)).squeeze()
            r.set_mask(mask)
            assert (np.count_nonzero(mask) > 0) & np.array_equal(mask > 0, r.data.mask.squeeze())

        # Test that proper issue is raised if shape is incorrect
        r = gr.Raster(datasets.get_path(dataset))
        wrong_shape = np.array(r.data.shape) + 1
        mask = np.zeros(wrong_shape)
        with pytest.raises(ValueError, match="mask must be of the same shape as existing data*"):
            r.set_mask(mask)

        # Test that proper issue is raised if mask is not a numpy array
        with pytest.raises(ValueError, match="mask must be a numpy array"):
            r.set_mask(1)

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
        warnings.simplefilter("error")

        # Reference raster to be used
        r = gr.Raster(datasets.get_path("landsat_B4"))

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
        r_ndv = r.copy()
        r_ndv.set_ndv(255)
        r3 = r_ndv.reproject(r2)
        assert r_ndv.nodata == r3.nodata

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

        # Assert that when reprojection creates nodata (voids), if no nodata is set, a default value is set
        r3 = r.reproject(dst_bounds=dst_bounds)
        assert r.nodata is None
        assert r3.nodata == 255

        # Particularly crucial if nodata falls outside the original image range -> check range is preserved
        r_float = r.astype("float32")  # type: ignore
        assert r_float.nodata is None
        r3 = r_float.reproject(dst_bounds=dst_bounds)
        assert r3.nodata == -99999
        assert np.min(r3.data.data) == r3.nodata
        assert np.min(r3.data) == np.min(r_float.data)
        assert np.max(r3.data) == np.max(r_float.data)

        # Check that dst_nodata works as expected
        r3 = r_float.reproject(dst_bounds=dst_bounds, dst_nodata=999)
        assert r3.nodata == 999
        assert np.max(r3.data.data) == r3.nodata

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
        assert geoutils.misc.array_equal(np.array(list_z_ind, dtype=np.float32), rpts, equal_nan=True)

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

        assert geoutils.misc.array_equal(np.array(list_z_ind, dtype=np.float32), rpts, equal_nan=True)

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

        # Check that nodata can also be set upon loading
        r = gr.Raster(datasets.get_path("landsat_B4"), nodata=5)
        assert r.nodata == 5

        # Check that an error is raised if nodata value is incompatible with dtype
        expected_message = r"ndv value .* incompatible with self.dtype .*"
        with pytest.raises(ValueError, match=expected_message):
            r.set_ndv(0.5)

    def test_default_ndv(self) -> None:
        """
        Test that the default nodata values are as expected.
        """
        assert _default_ndv("uint8") == np.iinfo("uint8").max
        assert _default_ndv("int8") == np.iinfo("int8").min
        assert _default_ndv("uint16") == np.iinfo("uint16").max
        assert _default_ndv("int16") == np.iinfo("int16").min
        assert _default_ndv("uint32") == 99999
        for dtype in ["int32", "float32", "float64", "float128"]:
            assert _default_ndv(dtype) == -99999

        # Check it works with most frequent np.dtypes too
        assert _default_ndv(np.dtype("uint8")) == np.iinfo("uint8").max
        for dtype in [np.dtype("int32"), np.dtype("float32"), np.dtype("float64")]:
            assert _default_ndv(dtype) == -99999

        # Check it works with most frequent types too
        assert _default_ndv(np.uint8) == np.iinfo("uint8").max
        for dtype in [np.int32, np.float32, np.float64]:
            assert _default_ndv(dtype) == -99999

        # Check that an error is raised for other types
        expected_message = "No default nodata value set for dtype"
        with pytest.raises(NotImplementedError, match=expected_message):
            _default_ndv("bla")

    def test_astype(self) -> None:

        r = gr.Raster(datasets.get_path("landsat_B4"))

        # Test changing dtypes that does not modify the data
        for dtype in [np.uint8, np.uint16, np.float32, np.float64, "float32"]:
            rout = r.astype(dtype)  # type: ignore
            assert rout == r
            assert np.dtype(rout.dtypes[0]) == dtype
            assert rout.data.dtype == dtype

        # Test a dtype that will modify the data
        dtype = np.int8
        rout = r.astype(dtype)  # type: ignore
        assert rout != r
        assert np.dtype(rout.dtypes[0]) == dtype
        assert rout.data.dtype == dtype
        pytest.warns(UserWarning, r.astype, dtype)  # check a warning is raised

        # Test modify in place
        for dtype in [np.uint8, np.uint16, np.float32, np.float64, "float32"]:
            r2 = r.copy()
            out = r2.astype(dtype, inplace=True)
            assert out is None
            assert r2 == r
            assert np.dtype(r2.dtypes[0]) == dtype
            assert r2.data.dtype == dtype

        # Test with masked values
        # First line is set to 0 and 0 set to nodata - check that 0 not used
        # Note that nodata must be set or astype will raise an error
        assert not np.any(r2.data == 0)
        r2 = r.copy()
        r2.data[0, 0] = 0
        r2.set_ndv(0)
        for dtype in [np.uint8, np.uint16, np.float32, np.float64, "float32"]:
            rout = r2.astype(dtype)  # type: ignore
            assert rout == r2
            assert np.dtype(rout.dtypes[0]) == dtype
            assert rout.data.dtype == dtype

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

        # Temporary folder
        temp_dir = tempfile.TemporaryDirectory()

        # Save file to temporary file, with defaults opts
        temp_file = NamedTemporaryFile(mode="w", delete=False, dir=temp_dir.name)
        img.save(temp_file.name)
        saved = gr.Raster(temp_file.name)
        assert gu.misc.array_equal(img.data, saved.data)

        # Test additional options
        co_opts = {"TILED": "YES", "COMPRESS": "LZW"}
        metadata = {"Type": "test"}
        temp_file = NamedTemporaryFile(mode="w", delete=False, dir=temp_dir.name)
        img.save(temp_file.name, co_opts=co_opts, metadata=metadata)
        saved = gr.Raster(temp_file.name)
        assert gu.misc.array_equal(img.data, saved.data)
        assert saved.ds.tags()["Type"] == "test"

        # Test that nodata value is enforced when masking - since value 0 is not used, data should be unchanged
        temp_file = NamedTemporaryFile(mode="w", delete=False, dir=temp_dir.name)
        img.save(temp_file.name, nodata=0)
        saved = gr.Raster(temp_file.name)
        assert gu.misc.array_equal(img.data, saved.data)
        assert saved.nodata == 0

        # Test that mask is preserved
        mask = img.data == np.min(img.data)
        img.set_mask(mask)
        temp_file = NamedTemporaryFile(mode="w", delete=False, dir=temp_dir.name)
        img.save(temp_file.name, nodata=0)
        saved = gr.Raster(temp_file.name)
        assert gu.misc.array_equal(img.data, saved.data)

        # Test that a warning is raised if nodata is not set
        with pytest.warns(UserWarning):
            img.save(TemporaryFile())

        # Clean up teporary folder - fails on Windows
        try:
            temp_dir.cleanup()
        except (NotADirectoryError, PermissionError):
            pass

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

        assert geoutils.misc.array_equal(img.data, img2.data, equal_nan=True)
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
        assert geoutils.misc.array_equal(out_img.data, img.data + bias)

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
        assert geoutils.misc.array_equal(red.data.astype("float32"), red2.data.astype("float32"))
        assert geoutils.misc.array_equal(blue.data.astype("float32"), blue2.data.astype("float32"))
        assert geoutils.misc.array_equal(green.data.astype("float32"), green2.data.astype("float32"))

        # Check that the red channel and the rgb data shares memory
        assert np.shares_memory(red.data, img.data)

        # Check that the red band data is not equal to the full RGB data.
        assert red != img

        # Test that the red band corresponds to the first band of the img
        assert geoutils.misc.array_equal(red.data.squeeze().astype("float32"), img.data[0, :, :].astype("float32"))

        # Modify the red band and make sure it propagates to the original img (it's not a copy)
        red.data += 1
        assert geoutils.misc.array_equal(red.data.squeeze().astype("float32"), img.data[0, :, :].astype("float32"))

        # Copy the bands instead of pointing to the same memory.
        red_c = img.split_bands(copy=True, subset=0)[0]

        # Check that the red band data does not share memory with the rgb image (it's a copy)
        assert not np.shares_memory(red_c, img)

        # Modify the copy, and make sure the original data is not modified.
        red_c.data += 1
        assert not geoutils.misc.array_equal(
            red_c.data.squeeze().astype("float32"), img.data[0, :, :].astype("float32")
        )

    def test_resampling_str(self) -> None:
        """Test that resampling methods can be given as strings instead of rio enums."""
        warnings.simplefilter("error")
        assert resampling_method_from_str("nearest") == rio.warp.Resampling.nearest  # noqa
        assert resampling_method_from_str("cubic_spline") == rio.warp.Resampling.cubic_spline  # noqa

        # Check that odd strings return the appropriate error.
        try:
            resampling_method_from_str("CUBIC_SPLINE")  # noqa
        except ValueError as exception:
            if "not a valid rasterio.warp.Resampling method" not in str(exception):
                raise exception

        img1 = gr.Raster(datasets.get_path("landsat_B4"))
        img2 = gr.Raster(datasets.get_path("landsat_B4_crop"))
        img1.set_ndv(0)
        img2.set_ndv(0)

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
        assert polygonized.crs == img.crs

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
        assert geoutils.misc.array_equal(np.asarray(points[:, 0]), np.tile(np.linspace(0.5, 4.5, 5), 5))

        assert img1.to_points(0.2).shape == (5, 3)

        img2 = gu.Raster(datasets.get_path("landsat_RGB"), load_data=False)

        points = img2.to_points(10)

        assert points.shape == (10, 5)
        assert not img2.is_loaded

        points_frame = img2.to_points(10, as_frame=True)

        assert geoutils.misc.array_equal(points_frame.columns, ["b1", "b2", "b3", "geometry"])
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
    assert geoutils.misc.array_equal(array, raster)

    # Test the data setter method by creating a new array
    raster.data = array + 2

    # Check that the median updated accordingly.
    assert np.median(raster) == 14.0

    # Test
    raster += array

    assert isinstance(raster, gr.Raster)
    assert np.median(raster) == 26.0


class TestsArithmetic:
    """
    Test that all arithmetic overloading functions work as expected.
    """

    # Create fake rasters with random values in 0-255 and dtype uint8
    width = height = 5
    transform = rio.transform.from_bounds(0, 0, 1, 1, width, height)
    r1 = gr.Raster.from_array(np.random.randint(1, 255, (height, width), dtype="uint8"), transform=transform, crs=None)
    r2 = gr.Raster.from_array(np.random.randint(1, 255, (height, width), dtype="uint8"), transform=transform, crs=None)

    # Tests with different dtype
    r1_f32 = gr.Raster.from_array(
        np.random.randint(1, 255, (height, width)).astype("float32"), transform=transform, crs=None
    )

    # Test with ndv value set
    r1_ndv = gr.Raster.from_array(
        np.random.randint(1, 255, (height, width)).astype("float32"),
        transform=transform,
        crs=None,
        nodata=_default_ndv("float32"),
    )

    # Test with 0 values
    r2_zero = gr.Raster.from_array(
        np.random.randint(1, 255, (height, width)).astype("float32"),
        transform=transform,
        crs=None,
        nodata=_default_ndv("float32"),
    )
    r2_zero.data[0, 0, 0] = 0

    # Create rasters with different shape, crs or transforms for testing errors
    r1_wrong_shape = gr.Raster.from_array(
        np.random.randint(0, 255, (height + 1, width)).astype("float32"), transform=transform, crs=None
    )

    r1_wrong_crs = gr.Raster.from_array(
        np.random.randint(0, 255, (height, width)).astype("float32"),
        transform=transform,
        crs=rio.crs.CRS.from_epsg(4326),
    )

    transform2 = rio.transform.from_bounds(0, 0, 2, 2, width, height)
    r1_wrong_transform = gr.Raster.from_array(
        np.random.randint(0, 255, (height, width)).astype("float32"), transform=transform2, crs=None
    )

    # Tests with child class
    satimg = gu.SatelliteImage.from_array(
        np.random.randint(1, 255, (height, width)).astype("float32"), transform=transform, crs=None
    )

    def test_equal(self) -> None:
        """
        Test that __eq__ and __ne__ work as expected
        """
        r1 = self.r1
        r2 = r1.copy()
        assert r1 == r2

        # Change data
        r2.data += 1
        assert r1 != r2

        # Change transform
        r2 = r1.copy()
        r2.transform = rio.transform.from_bounds(0, 0, 1, 1, self.width + 1, self.height)
        assert r1 != r2

        # Change CRS
        r2 = r1.copy()
        r2.crs = rio.crs.CRS.from_epsg(4326)
        assert r1 != r2

        # Change ndv
        r2 = r1.copy()
        r2.set_ndv(34)
        assert r1 != r2

    # List of operations with two operands
    ops_2args = [
        "__add__",
        "__radd__",
        "__sub__",
        "__rsub__",
        "__mul__",
        "__rmul__",
        "__truediv__",
        "__rtruediv__",
        "__floordiv__",
        "__rfloordiv__",
        "__mod__",
    ]

    @pytest.mark.parametrize("op", ops_2args)  # type: ignore
    def test_ops_2args_expl(self, op: str) -> None:
        """
        Check that arithmetic overloading functions, with two operands, work as expected when called explicitly.
        """
        warnings.filterwarnings("ignore", message="invalid value encountered")

        # Test various inputs: Raster with different dtypes, np.ndarray, single number
        r1 = self.r1
        r1_f32 = self.r1_f32
        r1_ndv = self.r1_ndv
        r2 = self.r2
        r2_zero = self.r2_zero
        satimg = self.satimg
        array = np.random.randint(1, 255, (1, self.height, self.width)).astype("float64")
        floatval = 3.14
        intval = 1

        # Test with 2 uint8 rasters
        r1 = self.r1
        r2 = self.r2
        r3 = getattr(r1, op)(r2)
        ctype = np.find_common_type([r1.data.dtype, r2.data.dtype], [])
        numpy_output = getattr(r1.data.astype(ctype), op)(r2.data.astype(ctype))
        assert isinstance(r3, gr.Raster)
        assert np.all(r3.data == numpy_output)
        assert r3.data.dtype == numpy_output.dtype
        if np.sum(r3.data.mask) == 0:
            assert r3.nodata is None
        else:
            assert r3.nodata == _default_ndv(ctype)
        assert r3.crs == r1.crs
        assert r3.transform == r1.transform

        # Test original data are not modified
        r1_copy = r1.copy()
        r2_copy = r2.copy()
        r3 = getattr(r1, op)(r2)
        assert isinstance(r3, gr.Raster)
        assert r1 == r1_copy
        assert r2 == r2_copy

        # Test with different dtypes
        r1 = self.r1_f32
        r2 = self.r2
        r3 = getattr(r1_f32, op)(r2)
        assert r3.data.dtype == np.dtype("float32")
        assert np.all(r3.data == getattr(r1.data, op)(r2.data))
        if np.sum(r3.data.mask) == 0:
            assert r3.nodata is None
        else:
            assert r3.nodata == _default_ndv("float32")

        # Test with ndv set
        r1 = self.r1
        r3 = getattr(r1_ndv, op)(r2)
        assert np.all(r3.data == getattr(r1_ndv.data, op)(r2.data))
        if np.sum(r3.data.mask) == 0:
            assert r3.nodata == r1_ndv.nodata
        else:
            assert r3.nodata == _default_ndv(r1_ndv.data.dtype)

        # Test with zeros values (e.g. division)
        r1 = self.r1
        r3 = getattr(r1, op)(r2_zero)
        assert np.all(r3.data == getattr(r1.data, op)(r2_zero.data))
        if np.sum(r3.data.mask) == 0:
            assert r3.nodata == r2_zero.nodata
        else:
            assert r3.nodata == _default_ndv(r1_ndv.data.dtype)

        # Test with a numpy array
        r1 = self.r1_f32
        r3 = getattr(r1, op)(array)
        assert isinstance(r3, gr.Raster)
        assert np.all(r3.data == getattr(r1.data, op)(array))
        if np.sum(r3.data.mask) == 0:
            assert r3.nodata is None
        else:
            assert r3.nodata == _default_ndv("float32")

        # Test with an integer
        r3 = getattr(r1, op)(intval)
        assert isinstance(r3, gr.Raster)
        assert np.all(r3.data == getattr(r1.data, op)(intval))
        if np.sum(r3.data.mask) == 0:
            assert r3.nodata is None
        else:
            assert r3.nodata == _default_ndv("uint8")

        # Test with a float value
        r3 = getattr(r1, op)(floatval)
        dtype = np.dtype(rio.dtypes.get_minimum_dtype(floatval))
        assert isinstance(r3, gr.Raster)
        assert r3.data.dtype == dtype
        assert np.all(r3.data == getattr(r1.data, op)(np.array(floatval).astype(dtype)))
        if np.sum(r3.data.mask) == 0:
            assert r3.nodata is None
        else:
            assert r3.nodata == _default_ndv(dtype)

        # Test with child class
        r3 = getattr(satimg, op)(intval)
        assert isinstance(r3, gu.satimg.SatelliteImage)

    reflective_ops = [["__add__", "__radd__"], ["__mul__", "__rmul__"]]

    @pytest.mark.parametrize("ops", reflective_ops)  # type: ignore
    def test_reflectivity(self, ops: list[str]) -> None:
        """
        Check reflective operations
        """
        warnings.filterwarnings("ignore", message="invalid value encountered")

        # Test various inputs: Raster with different dtypes, np.ndarray, single number
        array = np.random.randint(1, 255, (1, self.height, self.width)).astype("float64")
        floatval = 3.14
        intval = 1

        # Get reflective operations
        op1, op2 = ops

        # Test with uint8 rasters
        r3 = getattr(self.r1, op1)(self.r2)
        r4 = getattr(self.r1, op2)(self.r2)
        assert r3 == r4

        # Test with different dtypes
        r3 = getattr(self.r1_f32, op1)(self.r2)
        r4 = getattr(self.r1_f32, op2)(self.r2)
        assert r3 == r4

        # Test with ndv set
        r3 = getattr(self.r1_ndv, op1)(self.r2)
        r4 = getattr(self.r1_ndv, op2)(self.r2)
        assert r3 == r4

        # Test with zeros values (e.g. division)
        r3 = getattr(self.r1, op1)(self.r2_zero)
        r4 = getattr(self.r1, op2)(self.r2_zero)
        assert r3 == r4

        # Test with a numpy array
        r3 = getattr(self.r1, op1)(array)
        r4 = getattr(self.r1, op2)(array)
        assert r3 == r4

        # Test with an integer
        r3 = getattr(self.r1, op1)(intval)
        r4 = getattr(self.r1, op2)(intval)
        assert r3 == r4

        # Test with a float value
        r3 = getattr(self.r1, op1)(floatval)
        r4 = getattr(self.r1, op2)(floatval)
        assert r3 == r4

    @classmethod
    def from_array(
        cls: type[TestsArithmetic],
        data: np.ndarray | np.ma.masked_array,
        rst_ref: gr.RasterType,
        nodata: int | float | None = None,
    ) -> gr.Raster:
        """
        Generate a Raster from numpy array, with set georeferencing. Used for testing only.
        """
        if nodata is None:
            nodata = rst_ref.nodata

        return gr.Raster.from_array(data, crs=rst_ref.crs, transform=rst_ref.transform, nodata=nodata)

    def test_ops_2args_implicit(self) -> None:
        """
        Test certain arithmetic overloading when called with symbols (+, -, *, /, //, %)
        """
        warnings.filterwarnings("ignore", message="invalid value encountered")

        # Test various inputs: Raster with different dtypes, np.ndarray, single number
        r1 = self.r1
        r1_f32 = self.r1_f32
        r2 = self.r2
        array = np.random.randint(1, 255, (1, self.height, self.width)).astype("uint8")
        floatval = 3.14

        # Addition
        assert r1 + r2 == self.from_array(r1.data + r2.data, rst_ref=r1)
        assert r1_f32 + r2 == self.from_array(r1_f32.data + r2.data, rst_ref=r1)
        # assert array + r2 == self.from_array(array + r2.data, rst_ref=r1)  # this case fails as using numpy's add...
        assert r2 + array == self.from_array(r2.data + array, rst_ref=r1)
        assert r1 + floatval == self.from_array(r1.data.astype("float32") + floatval, rst_ref=r1)
        assert floatval + r1 == self.from_array(floatval + r1.data.astype("float32"), rst_ref=r1)
        assert r1 + r2 == r2 + r1

        # Multiplication
        assert r1 * r2 == self.from_array(r1.data * r2.data, rst_ref=r1)
        assert r1_f32 * r2 == self.from_array(r1_f32.data * r2.data, rst_ref=r1)
        # assert array * r2 == self.from_array(array * r2.data, rst_ref=r1)  # this case fails as using numpy's mul...
        assert r2 * array == self.from_array(r2.data * array, rst_ref=r1)
        assert r1 * floatval == self.from_array(r1.data.astype("float32") * floatval, rst_ref=r1)
        assert floatval * r1 == self.from_array(floatval * r1.data.astype("float32"), rst_ref=r1)
        assert r1 * r2 == r2 * r1

        # Subtraction
        assert r1 - r2 == self.from_array(r1.data - r2.data, rst_ref=r1)
        assert r1_f32 - r2 == self.from_array(r1_f32.data - r2.data, rst_ref=r1)
        # assert array - r2 == self.from_array(array - r2.data, rst_ref=r1)  # this case fails
        assert r2 - array == self.from_array(r2.data - array, rst_ref=r1)
        assert r1 - floatval == self.from_array(r1.data.astype("float32") - floatval, rst_ref=r1)
        assert floatval - r1 == self.from_array(floatval - r1.data.astype("float32"), rst_ref=r1)

        # True division
        assert r1 / r2 == self.from_array(r1.data / r2.data, rst_ref=r1)
        assert r1_f32 / r2 == self.from_array(r1_f32.data / r2.data, rst_ref=r1)
        # assert array / r2 == self.from_array(array / r2.data, rst_ref=r1)  # this case fails
        assert r2 / array == self.from_array(r2.data / array, rst_ref=r2)
        assert r1 / floatval == self.from_array(r1.data.astype("float32") / floatval, rst_ref=r1)
        assert floatval / r1 == self.from_array(floatval / r1.data.astype("float32"), rst_ref=r1)

        # Floor division
        assert r1 // r2 == self.from_array(r1.data // r2.data, rst_ref=r1)
        assert r1_f32 // r2 == self.from_array(r1_f32.data // r2.data, rst_ref=r1)
        # assert array // r2 == self.from_array(array // r2.data, rst_ref=r1)  # this case fails
        assert r2 // array == self.from_array(r2.data // array, rst_ref=r1)
        assert r1 // floatval == self.from_array(r1.data // floatval, rst_ref=r1)
        assert floatval // r1 == self.from_array(floatval // r1.data, rst_ref=r1)

        # Modulo
        assert r1 % r2 == self.from_array(r1.data % r2.data, rst_ref=r1)
        assert r1_f32 % r2 == self.from_array(r1_f32.data % r2.data, rst_ref=r1)
        # assert array % r2 == self.from_array(array % r2.data, rst_ref=r1)  # this case fails
        assert r2 % array == self.from_array(r2.data % array, rst_ref=r1)
        assert r1 % floatval == self.from_array(r1.data.astype("float32") % floatval, rst_ref=r1)

    @pytest.mark.parametrize("op", ops_2args)  # type: ignore
    def test_raise_errors(self, op: str) -> None:
        """
        Test that errors are properly raised in certain situations.
        """
        # different shapes
        expected_message = "Both rasters must have the same shape, transform and CRS."
        with pytest.raises(ValueError, match=expected_message):
            getattr(self.r1_wrong_shape, op)(self.r2)

        # different CRS
        with pytest.raises(ValueError, match=expected_message):
            getattr(self.r1_wrong_crs, op)(self.r2)

        # different transform
        with pytest.raises(ValueError, match=expected_message):
            getattr(self.r1_wrong_transform, op)(self.r2)

        # Wrong type of "other"
        expected_message = "Operation between an object of type .* and a Raster impossible."
        with pytest.raises(NotImplementedError, match=expected_message):
            getattr(self.r1, op)("some_string")

    @pytest.mark.parametrize("power", [2, 3.14, -1])  # type: ignore
    def test_power(self, power: float | int) -> None:

        if power > 0:  # Integers to negative integer powers are not allowed.
            assert self.r1**power == self.from_array(self.r1.data**power, rst_ref=self.r1)
        assert self.r1_f32**power == self.from_array(self.r1_f32.data**power, rst_ref=self.r1_f32)
