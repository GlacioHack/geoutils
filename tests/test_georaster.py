"""
Test functions for georaster
"""
from __future__ import annotations

import os
import re
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
from geoutils import examples
from geoutils.georaster.raster import _default_nodata, _default_rio_attrs
from geoutils.misc import resampling_method_from_str

DO_PLOT = False


class TestRaster:

    landsat_b4_path = examples.get_path("everest_landsat_b4")
    landsat_b4_crop_path = examples.get_path("everest_landsat_b4_cropped")
    landsat_rgb_path = examples.get_path("everest_landsat_rgb")
    everest_outlines_path = examples.get_path("everest_rgi_outlines")
    aster_dem_path = examples.get_path("exploradores_aster_dem")
    aster_outlines_path = examples.get_path("exploradores_rgi_outlines")

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_init(self, example: str) -> None:
        """Test that all possible inputs work properly in Raster class init"""

        # First, filename
        r = gr.Raster(example)
        assert isinstance(r, gr.Raster)

        # Second, passing a Raster itself (points back to Raster passed)
        r2 = gr.Raster(r)
        assert isinstance(r2, gr.Raster)

        # Third, rio.Dataset
        ds = rio.open(example)
        r3 = gr.Raster(ds)
        assert isinstance(r3, gr.Raster)
        assert r3.filename is not None

        # Finally, as memoryfile
        memfile = rio.MemoryFile(open(example, "rb"))
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

        # The data will not be copied, immutable objects will
        r.data[0, 0, 0] += 5
        assert r2.data[0, 0, 0] == r.data[0, 0, 0]

        # With r.nbands = 2
        r._data = np.repeat(r.data, 2).reshape((2,) + r.shape)
        assert r.nbands != r2.nbands

        # Test that loaded data are always masked_arrays (but the mask may be empty, i.e. 'False')
        assert np.ma.isMaskedArray(gr.Raster(example, masked=True).data)
        assert np.ma.isMaskedArray(gr.Raster(example, masked=False).data)

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_info(self, example: str) -> None:
        """Test that the information summary is consistent with that of rasterio"""

        r = gr.Raster(example)

        # Check all is good with passing attributes
        with rio.open(example) as dataset:
            for attr in _default_rio_attrs:
                assert r.__getattribute__(attr) == dataset.__getattribute__(attr)

        # Check summary matches that of RIO
        assert str(r) == r.info()

        # Check that the stats=True flag doesn't trigger a warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            stats = r.info(stats=True)

        # Check the stats adapt to nodata values
        if r.dtypes[0] == "uint8":
            # Validate that the mask is respected by adding 0 values (there are none to begin with.)
            r.data.ravel()[:1000] = 0
            # Set the nodata value to 0, then validate that they are excluded from the new minimum
            r.set_nodata(0)
        elif r.dtypes[0] == "float32":
            # We do the same with -99999 here
            r.data.ravel()[:1000] = -99999
            # And replace the nodata value
            r.set_nodata(-99999)

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
        # For the first example with Landsat B4
        r = gr.Raster(self.landsat_b4_path, load_data=False)

        assert r.driver == "GTiff"
        assert r.width == 800
        assert r.height == 655
        assert r.shape == (r.height, r.width)
        assert r.count == 1
        assert geoutils.misc.array_equal(r.dtypes, ["uint8"])
        assert r.transform == rio.transform.Affine(30.0, 0.0, 478000.0, 0.0, -30.0, 3108140.0)
        assert geoutils.misc.array_equal(r.res, [30.0, 30.0])
        assert r.bounds == rio.coords.BoundingBox(left=478000.0, bottom=3088490.0, right=502000.0, top=3108140.0)
        assert r.crs == rio.crs.CRS.from_epsg(32645)
        assert not r.is_loaded

        # And the second example with ASTER DEM
        r2 = gr.Raster(self.aster_dem_path, load_data=False)

        assert r2.driver == "GTiff"
        assert r2.width == 539
        assert r2.height == 618
        assert r2.shape == (r2.height, r2.width)
        assert r2.count == 1
        assert geoutils.misc.array_equal(r2.dtypes, ["float32"])
        assert r2.transform == rio.transform.Affine(30.0, 0.0, 627175.0, 0.0, -30.0, 4852085.0)
        assert geoutils.misc.array_equal(r2.res, [30.0, 30.0])
        assert r2.bounds == rio.coords.BoundingBox(left=627175.0, bottom=4833545.0, right=643345.0, top=4852085.0)
        assert r2.crs == rio.crs.CRS.from_epsg(32718)
        assert not r2.is_loaded

        # Test 2 - loading the data afterward
        r.load()
        assert r.is_loaded
        assert r.nbands == 1
        assert r.data.shape == (r.count, r.height, r.width)

        # Test 3 - single band, loading data
        r = gr.Raster(self.landsat_b4_path, load_data=True)
        assert r.is_loaded
        assert r.nbands == 1
        assert r.data.shape == (r.count, r.height, r.width)

        # Test 4 - multiple bands, load all bands
        r = gr.Raster(self.landsat_rgb_path, load_data=True)
        assert r.count == 3
        assert geoutils.misc.array_equal(r.indexes, [1, 2, 3])
        assert r.nbands == 3
        assert geoutils.misc.array_equal(r.bands, [1, 2, 3])
        assert r.data.shape == (r.count, r.height, r.width)

        # Test 5 - multiple bands, load one band only
        r = gr.Raster(self.landsat_rgb_path, load_data=True, bands=1)
        assert r.count == 3
        assert geoutils.misc.array_equal(r.indexes, [1, 2, 3])
        assert r.nbands == 1
        # assert r.bands == (1)
        assert r.data.shape == (r.nbands, r.height, r.width)

        # Test 6 - multiple bands, load a list of bands
        r = gr.Raster(self.landsat_rgb_path, load_data=True, bands=[2, 3])
        assert r.count == 3
        assert geoutils.misc.array_equal(r.indexes, [1, 2, 3])
        assert r.nbands == 2
        assert geoutils.misc.array_equal(r.bands, (2, 3))
        assert r.data.shape == (r.nbands, r.height, r.width)

    @pytest.mark.parametrize("nodata_init", [None, "type_default"])  # type: ignore
    @pytest.mark.parametrize(
        "dtype", ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64", "longdouble"]
    )  # type: ignore
    def test_data_setter(self, dtype: str, nodata_init: str | None) -> None:
        """
        Test that the behaviour of data setter, which is triggered directly using from_array, is as expected.

        In details, we check that the data setter:

        1. Writes the data in a masked array, whether the input is a classic array or a masked_array,
        2. Reshapes the data in a 3D array if it is 2D,
        3. Sets a new nodata value only if the provided array has non-finite values that are unmasked (including if
            there is no mask defined at all, e.g. for classic array with NaNs),
        4. Masks non-finite values that are unmasked, whether the input is a classic array or a masked_array,
        5. Raises an error if the new data does not have the right shape,
        6. Raises an error if the new data does not have the dtype of the Raster.
        """

        # Initiate a random array for testing
        width = height = 5
        transform = rio.transform.from_bounds(0, 0, 1, 1, width, height)

        # Create random values between the lower and upper limit of the data type, max absolute 99999 for floats
        if "int" in dtype:
            val_min = np.iinfo(int_type=dtype).min
            val_max = np.iinfo(int_type=dtype).max
            randint_dtype = dtype
        else:
            val_min = -99999
            val_max = 99999
            randint_dtype = "int32"

        # Fix the random seed
        np.random.seed(42)
        arr = np.random.randint(low=val_min, high=val_max, size=(1, width, height), dtype=randint_dtype).astype(dtype)
        mask = np.random.randint(0, 2, size=(1, width, height), dtype=bool)

        # Check that we are actually masking stuff
        assert np.count_nonzero(mask) > 0

        # Add a random floating point value if the data type is float
        if "float" in dtype:
            arr += np.random.normal(size=(1, width, height))

        # Use either the default nodata or None
        if nodata_init == "type_default":
            nodata: int | None = _default_nodata(dtype)
        else:
            nodata = None

        # -- First test: consistency with input array --

        # 3 cases: classic array without mask, masked_array without mask and masked_array with mask
        r1 = gr.Raster.from_array(data=arr, transform=transform, crs=None, nodata=nodata)
        r2 = gr.Raster.from_array(data=np.ma.masked_array(arr), transform=transform, crs=None, nodata=nodata)
        r3 = gr.Raster.from_array(data=np.ma.masked_array(arr, mask=mask), transform=transform, crs=None, nodata=nodata)

        # Check nodata is correct
        assert r1.nodata == nodata
        assert r2.nodata == nodata
        assert r3.nodata == nodata

        # Compare the consistency of the data setter whether it is passed a masked_array or an unmasked one
        assert np.array_equal(r1.data.data, arr)
        assert not r1.data.mask
        assert np.array_equal(r2.data.data, arr)
        assert not r2.data.mask
        assert np.array_equal(r3.data.data, arr)
        assert np.array_equal(r3.data.mask, mask)

        # -- Second test: passing a 2D array --

        # 3 cases: classic array without mask, masked_array without mask and masked_array with mask
        r1 = gr.Raster.from_array(data=arr.squeeze(), transform=transform, crs=None, nodata=nodata)
        r2 = gr.Raster.from_array(data=np.ma.masked_array(arr).squeeze(), transform=transform, crs=None, nodata=nodata)
        r3 = gr.Raster.from_array(
            data=np.ma.masked_array(arr, mask=mask).squeeze(), transform=transform, crs=None, nodata=nodata
        )

        # Check nodata is correct
        assert r1.nodata == nodata
        assert r2.nodata == nodata
        assert r3.nodata == nodata

        # Check the shape has been adjusted back to 3D
        assert np.array_equal(r1.data.data, arr)
        assert not r1.data.mask
        assert np.array_equal(r2.data.data, arr)
        assert not r2.data.mask
        assert np.array_equal(r3.data.data, arr)
        assert np.array_equal(r3.data.mask, mask)

        # -- Third and fourth test: the function sets a new nodata/mask only with unmasked non-finite values --
        arr_with_unmasked_nodata = np.copy(arr)
        if "float" in dtype:
            # We set one random unmasked value to NaN
            indices = np.indices(np.shape(arr))
            ind_nm = indices[:, ~mask]
            rand_ind = np.random.randint(low=0, high=ind_nm.shape[1], size=1)[0]
            arr_with_unmasked_nodata[ind_nm[0, rand_ind], ind_nm[1, rand_ind], ind_nm[2, rand_ind]] = np.nan

            if nodata is None:
                with pytest.warns(
                    UserWarning,
                    match="Setting default nodata {:.0f} to mask non-finite values found in the array, as "
                    "no nodata value was defined.".format(_default_nodata(dtype)),
                ):
                    r1 = gr.Raster.from_array(
                        data=arr_with_unmasked_nodata, transform=transform, crs=None, nodata=nodata
                    )
                    r2 = gr.Raster.from_array(
                        data=np.ma.masked_array(arr_with_unmasked_nodata), transform=transform, crs=None, nodata=nodata
                    )
                    r3 = gr.Raster.from_array(
                        data=np.ma.masked_array(arr_with_unmasked_nodata, mask=mask),
                        transform=transform,
                        crs=None,
                        nodata=nodata,
                    )
            else:
                r1 = gr.Raster.from_array(data=arr_with_unmasked_nodata, transform=transform, crs=None, nodata=nodata)
                r2 = gr.Raster.from_array(
                    data=np.ma.masked_array(arr_with_unmasked_nodata), transform=transform, crs=None, nodata=nodata
                )
                r3 = gr.Raster.from_array(
                    data=np.ma.masked_array(arr_with_unmasked_nodata, mask=mask),
                    transform=transform,
                    crs=None,
                    nodata=nodata,
                )

            # Check nodata is correct
            if nodata is None:
                new_nodata = _default_nodata(dtype)
            else:
                new_nodata = nodata
            assert r1.nodata == new_nodata
            assert r2.nodata == new_nodata
            assert r3.nodata == new_nodata

            # Check that masks have changed to adapt to the non-finite value
            assert np.array_equal(r1.data.data, arr_with_unmasked_nodata, equal_nan=True)
            assert np.array_equal(r1.data.mask, ~np.isfinite(arr_with_unmasked_nodata))
            assert np.array_equal(r2.data.data, arr_with_unmasked_nodata, equal_nan=True)
            assert np.array_equal(r2.data.mask, ~np.isfinite(arr_with_unmasked_nodata))
            assert np.array_equal(r3.data.data, arr_with_unmasked_nodata, equal_nan=True)
            assert np.array_equal(r3.data.mask, np.logical_or(mask, ~np.isfinite(arr_with_unmasked_nodata)))

        # Check that setting data with a different data type results in an error
        rst = gr.Raster.from_array(data=arr, transform=transform, crs=None, nodata=nodata)
        if "int" in dtype:
            new_dtype = "float32"
        else:
            new_dtype = "uint8"

        with pytest.raises(
            ValueError,
            match=re.escape(
                "New data must be of the same type as existing data: {}. Use copy() to set a new array "
                "with different dtype, or astype() to change type.".format(str(np.dtype(dtype)))
            ),
        ):
            rst.data = rst.data.astype(new_dtype)

        # Check that setting data with a different shape results in an error
        new_shape = (1, 25)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "New data must be of the same shape as existing data: ({}, {}). Given: "
                "{}.".format(str(width), str(height), str(new_shape))
            ),
        ):
            rst.data = rst.data.reshape(new_shape)

    @pytest.mark.parametrize("example", [aster_dem_path, landsat_b4_path, landsat_rgb_path])  # type: ignore
    def test_get_nanarray(self, example: str) -> None:
        """
        Check that self.get_nanarray behaves as expected for examples with invalid data or not, and with several bands
        or a single one.
        """

        # -- First, we test without returning a mask --

        # Get nanarray
        rst = gr.Raster(example)
        rst_copy = rst.copy()
        rst_arr = rst.get_nanarray()

        # If there is no mask in the masked array, the array should not have NaNs and be equal to that of data.data
        if not np.ma.is_masked(rst.data):
            assert np.count_nonzero(np.isnan(rst_arr)) == 0
            assert np.array_equal(rst.data.data.squeeze(), rst_arr)

        # Otherwise, the arrays should be equal with a fill_value of NaN
        else:
            assert np.count_nonzero(np.isnan(rst_arr)) > 0
            assert np.ma.allequal(rst.data.squeeze(), rst_arr, fill_value=np.nan)

        # Check that modifying the NaN array does not back-propagate to the original array (np.ma.filled returns a view
        # when there is no invalid data, but in this case get_nanarray should copy the data).
        rst_arr += 5
        assert rst == rst_copy

        # -- Then, we test with a mask returned --
        rst_arr, mask = rst.get_nanarray(return_mask=True)

        assert np.array_equal(mask, np.ma.getmaskarray(rst.data).squeeze())

        # Also check for back-propagation here with the mask and array
        rst_arr += 5
        mask = ~mask
        assert rst == rst_copy

    def test_downsampling(self) -> None:
        """
        Check that self.data is correct when using downsampling
        """
        # Test single band
        r = gr.Raster(self.landsat_b4_path, downsample=4)
        assert r.data.shape == (1, 164, 200)
        assert r.height == 164
        assert r.width == 200

        # Test multiple band
        r = gr.Raster(self.landsat_rgb_path, downsample=2)
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

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_copy(self, example: str) -> None:
        """
        Test that the copy method works as expected for Raster.
        We check that:
        1. Copying creates a new memory file
        2. If r.data is modified and r copied, the updated data is copied
        3. If r is copied, r.data changed, r2.data should be unchanged
        Then, we check the new_array argument of copy():
        4. Check that output Rasters are equal whether the input array is a NaN np.ndarray or a masked_array
        5. Check that the new_array argument works when providing a different data type
        """

        # -- First and second test: copying create a new memory file that has all the same attributes --

        # Open data, modify, and copy
        r = gr.Raster(example)
        r.data += 5
        r2 = r.copy()

        # Objects should be different (not pointing to the same memory)
        assert r is not r2

        # Check the object is a Raster
        assert isinstance(r2, gr.Raster)

        # Copy should have no filename
        assert r2.filename is None

        # Check a temporary memory file different than original disk file was created
        assert r2.name != r.name

        # Check all attributes except name and driver
        default_attrs = _default_rio_attrs.copy()
        for attr in ["name", "driver"]:
            default_attrs.remove(attr)
        attrs = default_attrs
        for attr in attrs:
            assert r.__getattribute__(attr) == r2.__getattribute__(attr)

        # Check data array
        assert geoutils.misc.array_equal(r.data, r2.data, equal_nan=True)

        # Check dataset_mask array
        assert np.all(r.data.mask == r2.data.mask)

        # -- Third test: if r.data is modified, it does not affect r2.data --
        r.data += 5
        assert not geoutils.misc.array_equal(r.data, r2.data, equal_nan=True)

        # -- Fourth test: check the new array parameter works with either ndarray filled with NaNs, or masked arrays --

        # First, we pass the new array as the masked array, mask and data of the new Raster object should be identical
        r2 = r.copy(new_array=r.data)
        assert r == r2

        # Same when passing the new array as a NaN ndarray
        r_arr = gu.spatial_tools.get_array_and_mask(r)[0]
        r2 = r.copy(new_array=r_arr)
        assert r == r2

        # -- Fifth test: check that the new_array argument works when providing a new dtype ##
        if "int" in r.dtypes[0]:
            new_dtype = "float32"
        else:
            new_dtype = "uint8"
        r2 = r.copy(new_array=r_arr.astype(dtype=new_dtype))

        assert r2.dtypes[0] == new_dtype

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_is_modified(self, example: str) -> None:
        """
        Test that changing the data updates is_modified as desired
        """
        # After loading, should not be modified
        r = gr.Raster(example)
        assert not r.is_modified

        # This should not trigger the hash
        r.data = r.data + 0
        assert not r.is_modified

        # This one neither
        r.data += 0
        assert not r.is_modified

        # This will
        r = gr.Raster(example)
        r.data = r.data + 5
        assert r.is_modified

    @pytest.mark.parametrize("example", [landsat_b4_path, landsat_rgb_path, aster_dem_path])  # type: ignore
    def test_masking(self, example: str) -> None:
        """
        Test self.set_mask
        """
        # Test boolean mask
        r = gr.Raster(example)
        mask = r.data.data == np.min(r.data.data)
        r.set_mask(mask)
        assert (np.count_nonzero(mask) > 0) & np.array_equal(mask > 0, r.data.mask)

        # Test non-boolean mask with values > 0
        r = gr.Raster(example)
        mask = np.where(r.data.data == np.min(r.data.data), 32, 0)
        r.set_mask(mask)
        assert (np.count_nonzero(mask) > 0) & np.array_equal(mask > 0, r.data.mask)

        # Test that previous mask is also preserved
        mask2 = r.data.data == np.max(r.data.data)
        assert np.count_nonzero(mask2) > 0
        r.set_mask(mask2)
        assert np.array_equal((mask > 0) | (mask2 > 0), r.data.mask)
        assert np.count_nonzero(~r.data.mask[mask > 0]) == 0

        # Test that shape of first dimension is ignored if equal to 1
        r = gr.Raster(example)
        if r.data.shape[0] == 1:
            mask = (r.data.data == np.min(r.data.data)).squeeze()
            r.set_mask(mask)
            assert (np.count_nonzero(mask) > 0) & np.array_equal(mask > 0, r.data.mask.squeeze())

        # Test that proper issue is raised if shape is incorrect
        r = gr.Raster(example)
        wrong_shape = np.array(r.data.shape) + 1
        mask = np.zeros(wrong_shape)
        with pytest.raises(ValueError, match="mask must be of the same shape as existing data*"):
            r.set_mask(mask)

        # Test that proper issue is raised if mask is not a numpy array
        with pytest.raises(ValueError, match="mask must be a numpy array"):
            r.set_mask(1)

    test_data = [[landsat_b4_path, everest_outlines_path], [aster_dem_path, aster_outlines_path]]

    @pytest.mark.parametrize("data", test_data)  # type: ignore
    def test_crop(self, data: list[str]) -> None:

        raster_path, outlines_path = data
        r = gr.Raster(raster_path)

        # -- Test with cropGeom being a list/tuple -- ##
        cropGeom: list[float] = list(r.bounds)

        # Test with same bounds -> should be the same #
        cropGeom2 = [cropGeom[0], cropGeom[1], cropGeom[2], cropGeom[3]]
        r_cropped = r.crop(cropGeom2, inplace=False)
        assert r_cropped.bounds == r.bounds
        assert gu.misc.array_equal(r.data, r_cropped.data)

        # - Test cropping each side by a random integer of pixels - #
        rand_int = np.random.randint(1, min(r.shape) - 1)

        # left
        cropGeom2 = [cropGeom[0] + rand_int * r.res[0], cropGeom[1], cropGeom[2], cropGeom[3]]
        r_cropped = r.crop(cropGeom2, inplace=False)
        assert list(r_cropped.bounds) == cropGeom2
        assert gu.misc.array_equal(r.data[:, :, rand_int:], r_cropped.data)

        # right
        cropGeom2 = [cropGeom[0], cropGeom[1], cropGeom[2] - rand_int * r.res[0], cropGeom[3]]
        r_cropped = r.crop(cropGeom2, inplace=False)
        assert list(r_cropped.bounds) == cropGeom2
        assert gu.misc.array_equal(r.data[:, :, :-rand_int], r_cropped.data)

        # bottom
        cropGeom2 = [cropGeom[0], cropGeom[1] + rand_int * abs(r.res[1]), cropGeom[2], cropGeom[3]]
        r_cropped = r.crop(cropGeom2, inplace=False)
        assert list(r_cropped.bounds) == cropGeom2
        assert gu.misc.array_equal(r.data[:, :-rand_int, :], r_cropped.data)

        # top
        cropGeom2 = [cropGeom[0], cropGeom[1], cropGeom[2], cropGeom[3] - rand_int * abs(r.res[1])]
        r_cropped = r.crop(cropGeom2, inplace=False)
        assert list(r_cropped.bounds) == cropGeom2
        assert gu.misc.array_equal(r.data[:, rand_int:, :], r_cropped.data)

        # same but tuple
        cropGeom3: tuple[float, float, float, float] = (
            cropGeom[0],
            cropGeom[1],
            cropGeom[2],
            cropGeom[3] - rand_int * r.res[0],
        )
        r_cropped = r.crop(cropGeom3, inplace=False)
        assert list(r_cropped.bounds) == list(cropGeom3)
        assert gu.misc.array_equal(r.data[:, rand_int:, :], r_cropped.data)

        # -- Test with CropGeom being a Raster -- #
        r_cropped2 = r.crop(r_cropped, inplace=False)
        assert r_cropped2.bounds == r_cropped.bounds
        assert gu.misc.array_equal(r_cropped2.data, r_cropped)

        # -- Test with inplace=True (Default) -- #
        r_copy = r.copy()
        r_copy.crop(r_cropped)
        assert r_copy.bounds == r_cropped.bounds
        assert gu.misc.array_equal(r_copy.data, r_cropped)

        # - Test cropping each side with a non integer pixel, mode='match_pixel' - #
        rand_float = np.random.randint(1, min(r.shape) - 1) + 0.25

        # left
        cropGeom2 = [cropGeom[0] + rand_float * r.res[0], cropGeom[1], cropGeom[2], cropGeom[3]]
        r_cropped = r.crop(cropGeom2, inplace=False)
        assert r.shape[1] - (r_cropped.bounds.right - r_cropped.bounds.left) / r.res[0] == int(rand_float)
        assert gu.misc.array_equal(r.data[:, :, int(rand_float) :], r_cropped.data)

        # right
        cropGeom2 = [cropGeom[0], cropGeom[1], cropGeom[2] - rand_float * r.res[0], cropGeom[3]]
        r_cropped = r.crop(cropGeom2, inplace=False)
        assert r.shape[1] - (r_cropped.bounds.right - r_cropped.bounds.left) / r.res[0] == int(rand_float)
        assert gu.misc.array_equal(r.data[:, :, : -int(rand_float)], r_cropped.data)

        # bottom
        cropGeom2 = [cropGeom[0], cropGeom[1] + rand_float * abs(r.res[1]), cropGeom[2], cropGeom[3]]
        r_cropped = r.crop(cropGeom2, inplace=False)
        assert r.shape[0] - (r_cropped.bounds.top - r_cropped.bounds.bottom) / r.res[1] == int(rand_float)
        assert gu.misc.array_equal(r.data[:, : -int(rand_float), :], r_cropped.data)

        # top
        cropGeom2 = [cropGeom[0], cropGeom[1], cropGeom[2], cropGeom[3] - rand_float * abs(r.res[1])]
        r_cropped = r.crop(cropGeom2, inplace=False)
        assert r.shape[0] - (r_cropped.bounds.top - r_cropped.bounds.bottom) / r.res[1] == int(rand_float)
        assert gu.misc.array_equal(r.data[:, int(rand_float) :, :], r_cropped.data)

        # -- Test with mode='match_extent' -- #
        # Test all sides at once, with rand_float less than half the smallest extent
        # The cropped extent should exactly match the requested extent, res will be changed accordingly
        rand_float = np.random.randint(1, min(r.shape) / 2 - 1) + 0.25
        cropGeom2 = [
            cropGeom[0] + rand_float * r.res[0],
            cropGeom[1] + rand_float * abs(r.res[1]),
            cropGeom[2] - rand_float * r.res[0],
            cropGeom[3] - rand_float * abs(r.res[1]),
        ]
        r_cropped = r.crop(cropGeom2, inplace=False, mode="match_extent")
        assert list(r_cropped.bounds) == cropGeom2
        # The change in resolution should be less than what would occur with +/- 1 pixel
        assert np.all(
            abs(np.array(r.res) - np.array(r_cropped.res)) < np.array(r.res) / np.array(r_cropped.shape)[::-1]
        )

        r_cropped2 = r.crop(r_cropped, inplace=False, mode="match_extent")
        assert r_cropped2.bounds == r_cropped.bounds
        assert gu.misc.array_equal(r_cropped2.data, r_cropped.data)

        # -- Test with CropGeom being a Vector -- #
        outlines = gu.Vector(outlines_path)
        outlines.ds = outlines.ds.to_crs(r.crs)
        r_cropped = r.crop(outlines, inplace=False)

        # Calculate intersection of the two bounding boxes and make sure crop has same bounds
        win_outlines = rio.windows.from_bounds(*outlines.bounds, transform=r.transform)
        win_raster = rio.windows.from_bounds(*r.bounds, transform=r.transform)
        final_window = win_outlines.intersection(win_raster).round_lengths().round_offsets()
        new_bounds = rio.windows.bounds(final_window, transform=r.transform)
        assert list(r_cropped.bounds) == list(new_bounds)

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_reproject(self, example: str) -> None:
        warnings.simplefilter("error")

        # Reference raster to be used
        r = gr.Raster(example)

        # -- Check proper errors are raised if nodata are not set -- #
        r_nodata = r.copy()
        r_nodata.set_nodata(None)

        # Make sure at least one pixel is masked for test 1
        rand_indices = gu.spatial_tools.subsample_raster(r_nodata.data, 10, return_indices=True)
        r_nodata.data[rand_indices] = np.ma.masked
        assert np.count_nonzero(r_nodata.data.mask) > 0

        # make sure at least one pixel is set at default nodata for test
        default_nodata = _default_nodata(r_nodata.dtypes[0])
        rand_indices = gu.spatial_tools.subsample_raster(r_nodata.data, 10, return_indices=True)
        r_nodata.data[rand_indices] = default_nodata
        assert np.count_nonzero(r_nodata.data == default_nodata) > 0

        # 1 - if no src_nodata is set and masked values exist, raises an error
        with pytest.raises(ValueError, match="No nodata set, use `src_nodata`"):
            _ = r_nodata.reproject(dst_res=r_nodata.res[0] / 2, dst_nodata=0)

        # 2 - if no dst_nodata is set and default value conflicts with existing value, a warning is raised
        with pytest.warns(
            UserWarning,
            match=re.escape(
                f"For reprojection, dst_nodata must be set. Default chosen value {_default_nodata(r_nodata.dtypes[0])} exists in \
self.data. This may have unexpected consequences. Consider setting a different nodata with \
self.set_nodata()."
            ),
        ):
            _ = r_nodata.reproject(dst_res=r_nodata.res[0] / 2, src_nodata=default_nodata)

        # 3 - if default nodata does not conflict, should not raise a warning
        r_nodata.data[r_nodata.data == default_nodata] = 3
        _ = r_nodata.reproject(dst_res=r_nodata.res[0] / 2, src_nodata=default_nodata)

        # -- Additional tests -- #

        # specific for the landsat test case, default nodata 255 cannot be used (see above), so use 0
        if r.nodata is None:
            r.set_nodata(0)

        # - Create 2 artificial rasters -
        # for r2b, bounds are cropped to the upper left by an integer number of pixels (i.e. crop)
        # for r2, resolution is also set to 2/3 the input res
        min_size = min(r.shape)
        rand_int = np.random.randint(min_size / 10, min(r.shape) - min_size / 10)
        new_transform = rio.transform.from_origin(
            r.bounds.left + rand_int * r.res[0], r.bounds.top - rand_int * abs(r.res[1]), r.res[0], r.res[1]
        )

        # data is cropped to the same extent
        new_data = r.data[:, rand_int::, rand_int::]
        r2b = gr.Raster.from_array(data=new_data, transform=new_transform, crs=r.crs, nodata=r.nodata)

        # Create a raster with different resolution
        dst_res = r.res[0] * 2 / 3
        r2 = r2b.reproject(dst_res=dst_res)
        assert r2.res == (dst_res, dst_res)

        # Assert the initial rasters are different
        assert r.bounds != r2b.bounds
        assert r.shape != r2b.shape
        assert r.bounds != r2.bounds
        assert r.shape != r2.shape
        assert r.res != r2.res

        # Test reprojecting with dst_ref=r2b (i.e. crop) -> output should have same shape, bounds and data
        r3 = r.reproject(r2b)
        assert r3.bounds == r2b.bounds
        assert r3.shape == r2b.shape
        assert r3.bounds == r2b.bounds
        assert r3.transform == r2b.transform
        assert gu.misc.array_equal(r3.data, r2b.data)

        if DO_PLOT:
            fig1, ax1 = plt.subplots()
            r.show(ax=ax1, title="Raster 1")

            fig2, ax2 = plt.subplots()
            r2b.show(ax=ax2, title="Raster 2")

            fig3, ax3 = plt.subplots()
            r3.show(ax=ax3, title="Raster 1 reprojected to Raster 2")

            plt.show()

        # Test reprojecting with dst_ref=r2 -> output should have same shape, bounds and transform
        # Data should be slightly different due to difference in input resolution
        r3 = r.reproject(r2)
        assert r3.bounds == r2.bounds
        assert r3.shape == r2.shape
        assert r3.bounds == r2.bounds
        assert r3.transform == r2.transform
        assert not gu.misc.array_equal(r3.data, r2.data)

        if DO_PLOT:
            fig1, ax1 = plt.subplots()
            r.show(ax=ax1, title="Raster 1")

            fig2, ax2 = plt.subplots()
            r2.show(ax=ax2, title="Raster 2")

            fig3, ax3 = plt.subplots()
            r3.show(ax=ax3, title="Raster 1 reprojected to Raster 2")

            plt.show()

        # - Check that if mask is modified afterwards, it is taken into account during reproject - #
        # Create a raster with (additional) random gaps
        r_gaps = r.copy()
        nsamples = 200
        rand_indices = gu.spatial_tools.subsample_raster(r_gaps.data, nsamples, return_indices=True)
        r_gaps.data[rand_indices] = np.ma.masked
        assert np.sum(r_gaps.data.mask) - np.sum(r.data.mask) == nsamples  # sanity check

        # reproject raster, and reproject mask. Check that both have same number of masked pixels
        # TODO: should test other resampling algo
        r_gaps_reproj = r_gaps.reproject(dst_res=dst_res, resampling="nearest")
        mask = gu.Raster.from_array(
            r_gaps.data.mask.astype("uint8"), crs=r_gaps.crs, transform=r_gaps.transform, nodata=None
        )
        mask_reproj = mask.reproject(dst_res=dst_res, dst_nodata=255, resampling="nearest")
        # Final masked pixels are those originally masked (=1) and the values masked during reproject, e.g. edges
        tot_masked_true = np.count_nonzero(mask_reproj.data.mask) + np.count_nonzero(mask_reproj.data == 1)
        assert np.count_nonzero(r_gaps_reproj.data.mask) == tot_masked_true

        # If a nodata is set, make sure it is preserved
        r_nodata = r.copy()

        r_nodata.set_nodata(0)

        r3 = r_nodata.reproject(r2)
        assert r_nodata.nodata == r3.nodata

        # Test dst_size - this should modify the shape, and hence resolution, but not the bounds
        out_size = (r.shape[1] // 2, r.shape[0] // 2)  # Outsize is (ncol, nrow)
        r3 = r.reproject(dst_size=out_size)
        assert r3.shape == (out_size[1], out_size[0])
        assert r3.res != r.res
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

        # If bounds are not a multiple of res, the latter will be updated accordingly
        r3 = r.reproject(dst_bounds=dst_bounds)
        assert r3.bounds == dst_bounds
        assert r3.res != r.res

        # Assert that when reprojection creates nodata (voids), if no nodata is set, a default value is set
        r3 = r.reproject(dst_bounds=dst_bounds)
        if r.nodata is None:
            assert r3.nodata == _default_nodata(r.dtypes[0])

        # Particularly crucial if nodata falls outside the original image range
        # -> check range is preserved (with nearest interpolation)
        r_float = r.astype("float32")  # type: ignore
        if r_float.nodata is None:
            r3 = r_float.reproject(dst_bounds=dst_bounds, resampling="nearest")
            assert r3.nodata == -99999
            assert np.min(r3.data.data) == r3.nodata
            assert np.min(r3.data) == np.min(r_float.data)
            assert np.max(r3.data) == np.max(r_float.data)

        # Check that dst_nodata works as expected
        r3 = r_float.reproject(dst_bounds=dst_bounds, dst_nodata=9999)
        assert r3.nodata == 9999
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

        # Test that reproject works from self.ds and yield same result as from in-memory array
        # TO DO: fix issue that default behavior sets nodata to 255 and masks valid values
        r3 = r.reproject(dst_crs=out_crs, dst_nodata=0)
        r = gr.Raster(example, load_data=False)
        r4 = r.reproject(dst_crs=out_crs, dst_nodata=0)
        assert gu.misc.array_equal(r3.data, r4.data)

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_intersection(self, example: list[str]) -> None:
        """Check the behaviour of the intersection function"""
        # Load input raster
        r = gr.Raster(example)
        bounds_orig: list[float] = list(r.bounds)

        # For the Landsat dataset, to avoid warnings with crop
        if r.nodata is None:
            r.set_nodata(0)

        # -- Test with same bounds -> should be the same -- #
        r_cropped = r.copy()
        intersection = r.intersection(r_cropped)
        assert intersection == r.bounds

        # -- Test with a second raster cropped to a smaller extent -- #
        # First with integer pixel cropped -> intersection should match smaller raster
        rand_int = np.random.randint(1, min(r.shape) / 2 - 1)
        bounds_new = [
            bounds_orig[0] + rand_int * r.res[0],
            bounds_orig[1] + rand_int * abs(r.res[1]),
            bounds_orig[2] - rand_int * r.res[0],
            bounds_orig[3] - rand_int * abs(r.res[1]),
        ]
        r_cropped = r.crop(bounds_new, inplace=False, mode="match_pixel")
        intersection = r.intersection(r_cropped, match_ref=False)
        assert intersection == r_cropped.bounds

        # Second with non-matching resolution, two cases
        rand_float = np.random.randint(1, min(r.shape) / 2 - 1) + 0.25
        bounds_new = [
            bounds_orig[0] + rand_float * r.res[0],
            bounds_orig[1] + rand_float * abs(r.res[1]),
            bounds_orig[2] - rand_float * r.res[0],
            bounds_orig[3] - rand_float * abs(r.res[1]),
        ]
        r_cropped = r.crop(bounds_new, inplace=False, mode="match_extent")

        # Case 1 - with match_ref = False -> intersection should match smaller raster bounds
        intersection = r.intersection(r_cropped, match_ref=False)
        assert intersection == r_cropped.bounds

        # Case 2 - with match_ref = True, 3 checks are made
        intersection_match = r.intersection(r_cropped, match_ref=True)

        # A - intersection should be larger than without match_ref
        poly1 = gu.projtools.bounds2poly(intersection_match)
        poly2 = gu.projtools.bounds2poly(intersection)
        assert poly1.contains(poly2)

        # B - the difference between both should be less than self's resolution
        diff = np.array(intersection) - np.array(intersection_match)
        assert max(abs(diff[0]), abs(diff[2])) < r.res[0]  # along x direction
        assert max(abs(diff[1]), abs(diff[3])) < r.res[1]  # along y direction

        # C - intersection bounds are a multiple of the reference pixel positions
        assert (intersection_match[0] - r.bounds[0]) % r.res[0] == 0
        assert (intersection_match[2] - r.bounds[0]) % r.res[0] == 0
        assert (intersection_match[1] - r.bounds[3]) % r.res[0] == 0
        assert (intersection_match[3] - r.bounds[3]) % r.res[1] == 0

        # -- Test with a second raster shifted right and up, integer shift -- #
        transform_shifted = (
            r.transform.a,
            r.transform.b,
            r.transform.c + rand_int * r.res[0],
            r.transform.d,
            r.transform.e,
            r.transform.f + rand_int * abs(r.res[1]),
        )
        r_shifted = gu.Raster.from_array(r.data, crs=r.crs, transform=transform_shifted, nodata=r.nodata)
        intersection = r.intersection(r_shifted)

        # left and bottom bounds should correspond to shifted raster
        assert intersection[0] == r_shifted.bounds[0]
        assert intersection[1] == r_shifted.bounds[1]

        # right and top bounds should correspond to original raster
        assert intersection[2] == r.bounds[2]
        assert intersection[3] == r.bounds[3]

        # -- Test with a second raster shifted right and up, float shift -- #
        transform_shifted = (
            r.transform.a,
            r.transform.b,
            r.transform.c + rand_float * r.res[0],
            r.transform.d,
            r.transform.e,
            r.transform.f + rand_float * abs(r.res[1]),
        )
        r_shifted = gu.Raster.from_array(r.data, crs=r.crs, transform=transform_shifted, nodata=r.nodata)

        # With match_ref = False, same as with integer
        intersection = r.intersection(r_shifted, match_ref=False)
        assert intersection[0] == r_shifted.bounds[0]
        assert intersection[1] == r_shifted.bounds[1]
        assert intersection[2] == r.bounds[2]
        assert intersection[3] == r.bounds[3]

        # With match_ref = True, right and top should match ref, left and bottom should match shifted +/- res
        intersection = r.intersection(r_shifted, match_ref=True)
        assert abs(intersection[0] - r_shifted.bounds[0]) < r.res[0]
        assert abs(intersection[1] - r_shifted.bounds[1]) < r.res[1]
        assert intersection[2] == r.bounds[2]
        assert intersection[3] == r.bounds[3]

        # -- Test with a non overlapping raster -- #
        warnings.simplefilter("error")
        transform_shifted = (
            r.transform.a,
            r.transform.b,
            r.transform.c + (r.width + 1) * r.res[0],
            r.transform.d,
            r.transform.e,
            r.transform.f,
        )
        r_nonoverlap = gu.Raster.from_array(r.data, crs=r.crs, transform=transform_shifted, nodata=r.nodata)

        with pytest.warns(UserWarning, match="Intersection is void"):
            intersection = r.intersection(r_nonoverlap)
            assert intersection == (0.0, 0.0, 0.0, 0.0)

    def test_interp(self) -> None:

        # First, we try on a Raster with a Point interpretation in its "AREA_OR_POINT" metadata: values interpolated
        # at the center of pixel
        r = gr.Raster(self.landsat_b4_path)
        assert r.tags["AREA_OR_POINT"] == "Point"

        xmin, ymin, xmax, ymax = r.bounds

        # We generate random points within the boundaries of the image

        xrand = np.random.randint(low=0, high=r.width, size=(10,)) * list(r.transform)[0] + xmin
        yrand = ymax + np.random.randint(low=0, high=r.height, size=(10,)) * list(r.transform)[4]
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

        # Now, we calculate the mean of values in each 2x2 slices of the data, and compare with interpolation at order 1
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

        # First order interpolation
        rpts = r.interp_points(pts, order=1, area_or_point="Area")
        # The values interpolated should be equal
        assert geoutils.misc.array_equal(np.array(list_z_ind, dtype=np.float32), rpts, equal_nan=True)

        # Test there is no failure with random coordinates (edge effects, etc)
        xrand = np.random.uniform(low=xmin, high=xmax, size=(1000,))
        yrand = np.random.uniform(low=ymin, high=ymax, size=(1000,))
        pts = list(zip(xrand, yrand))
        rpts = r.interp_points(pts)

        # Second, test after a crop: the Raster now has an Area interpretation, those should fall right on the integer
        # pixel indexes
        r2 = gr.Raster(self.landsat_b4_crop_path)
        r.crop(r2)
        assert r.tags["AREA_OR_POINT"] == "Area"

        xmin, ymin, xmax, ymax = r.bounds

        # We can test with several method for the exact indexes: interp, value_at_coords, and simple read should
        # give back the same values that fall right on the coordinates
        xrand = np.random.randint(low=0, high=r.width, size=(10,)) * list(r.transform)[0] + xmin
        yrand = ymax + np.random.randint(low=0, high=r.height, size=(10,)) * list(r.transform)[4]
        pts = list(zip(xrand, yrand))
        # By default, i and j are returned as integers
        i, j = r.xy2ij(xrand, yrand, op=np.float32, area_or_point="Area")
        list_z_ind = []
        img = r.data
        for k in range(len(xrand)):
            # We directly sample the values
            z_ind = img[0, int(i[k]), int(j[k])]
            # We can also compare with the value_at_coords() functionality
            list_z_ind.append(z_ind)

        rpts = r.interp_points(pts, order=1)

        assert geoutils.misc.array_equal(np.array(list_z_ind, dtype=np.float32), rpts, equal_nan=True)

        # Test for an invidiual point (shape can be tricky at 1 dimension)
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

        r = gr.Raster(self.landsat_b4_path)
        r2 = gr.Raster(self.landsat_b4_crop_path)
        r.crop(r2)

        # Random test point that raised an error
        itest = 118
        jtest = 450
        xtest = 496930
        ytest = 3099170

        # z = r.data[0, itest, jtest]
        x_out, y_out = r.ij2xy(itest, jtest, offset="ul")
        assert x_out == xtest
        assert y_out == ytest

        z_val = r.value_at_coords(xtest, ytest)
        z = r.data.data[0, itest, jtest]
        assert z == z_val

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_set_nodata(self, example: str) -> None:
        """
        We test set_nodata() with all possible input parameters, check expected behaviour for updating array and mask,
        and that errors and warnings are raised when they should be.
        """

        # Read raster and save a copy to compare later
        r = gr.Raster(example)
        r_copy = r.copy()
        old_nodata = r.nodata
        # We chose nodata that doesn't exist in the raster yet for both our examples (for uint8, the default value of
        # 255 exist, so we replace by 0)
        new_nodata = _default_nodata(r.dtypes[0]) if not r.dtypes[0] == "uint8" else 0

        # -- First, test set_nodata() with default parameters --

        # Check that the new_nodata does not exist in the raster yet, and set it
        assert np.count_nonzero(r_copy.data.data == new_nodata) == 0
        r.set_nodata(nodata=new_nodata)

        # The nodata value should have been set in the metadata
        assert r.nodata == new_nodata

        # By default, the array should have been updated
        if old_nodata is not None:
            index_old_nodata = r_copy.data.data == old_nodata
            assert all(r.data.data[index_old_nodata] == new_nodata)
        else:
            index_old_nodata = np.zeros(np.shape(r.data), dtype=bool)

        # The rest of the array and mask should be unchanged
        assert np.array_equal(r.data.data[~index_old_nodata], r_copy.data.data[~index_old_nodata])
        # We call np.ma.getmaskarray to always compare a full boolean array
        assert np.array_equal(np.ma.getmaskarray(r.data), np.ma.getmaskarray(r_copy.data))

        # Then, we repeat for a nodata that already existed, we artificially modify the value on an unmasked pixel
        r = r_copy.copy()
        mask_pixel_artificially_set = np.zeros(np.shape(r.data), dtype=bool)
        mask_pixel_artificially_set[0, 0, 0] = True
        r.data.data[mask_pixel_artificially_set] = new_nodata
        # We set the value as masked before unmasking to create the mask if it does not exist yet
        r.data[mask_pixel_artificially_set] = np.ma.masked
        r.data.mask[mask_pixel_artificially_set] = False

        # Check the value is valid in the masked array
        assert np.count_nonzero(r_copy.data == new_nodata) >= 0
        # A warning should be raised when setting the nodata
        with pytest.warns(
            UserWarning,
            match=re.escape(
                "New nodata value found in the data array. Those will be masked, and the old "
                "nodata cells will now take the same value. Use set_nodata() with update_array=False "
                "and/or update_mask=False to change this behaviour."
            ),
        ):
            r.set_nodata(nodata=new_nodata)

        # The nodata value should have been set in the metadata
        assert r.nodata == new_nodata

        # By default, the array should have been updated similarly for the old nodata
        if old_nodata is not None:
            index_old_nodata = r_copy.data.data == old_nodata
            assert all(r.data.data[index_old_nodata] == new_nodata)
        else:
            index_old_nodata = np.zeros(np.shape(r.data.data), dtype=bool)

        # The rest of the array is similarly unchanged
        index_unchanged = np.logical_and(~index_old_nodata, ~mask_pixel_artificially_set)
        assert np.array_equal(r.data.data[index_unchanged], r_copy.data.data[index_unchanged])
        # But, this time, the mask is only unchanged for the array excluding the pixel artificially modified
        assert np.array_equal(
            np.ma.getmaskarray(r.data)[~mask_pixel_artificially_set],
            np.ma.getmaskarray(r_copy.data)[~mask_pixel_artificially_set],
        )

        # More specifically, it has changed for that pixel
        assert (
            np.ma.getmaskarray(r.data)[mask_pixel_artificially_set]
            == ~np.ma.getmaskarray(r_copy.data)[mask_pixel_artificially_set]
        )

        # -- Second, test set_nodata() with update_array=False --
        r = r_copy.copy()

        r.data.data[mask_pixel_artificially_set] = new_nodata
        # We set the value as masked before unmasking to create the mask if it does not exist yet
        r.data[mask_pixel_artificially_set] = np.ma.masked
        r.data.mask[mask_pixel_artificially_set] = False

        with pytest.warns(
            UserWarning,
            match=re.escape(
                "New nodata value found in the data array. Those will be masked. Use set_nodata() "
                "with update_mask=False to change this behaviour."
            ),
        ):
            r.set_nodata(nodata=new_nodata, update_array=False)

        # The nodata value should have been set in the metadata
        assert r.nodata == new_nodata

        # Now, the array should not have been updated, so the entire array should be unchanged except for the pixel
        assert np.array_equal(r.data.data[~mask_pixel_artificially_set], r_copy.data.data[~mask_pixel_artificially_set])
        # But the mask should have been updated on the pixel
        assert (
            np.ma.getmaskarray(r.data)[mask_pixel_artificially_set]
            == ~np.ma.getmaskarray(r_copy.data)[mask_pixel_artificially_set]
        )

        # -- Third, test set_nodata() with update_mask=False --
        r = r_copy.copy()

        r.data.data[mask_pixel_artificially_set] = new_nodata
        # We set the value as masked before unmasking to create the mask if it does not exist yet
        r.data[mask_pixel_artificially_set] = np.ma.masked
        r.data.mask[mask_pixel_artificially_set] = False

        with pytest.warns(
            UserWarning,
            match=re.escape(
                "New nodata value found in the data array. The old nodata cells will now take the same "
                "value. Use set_nodata() with update_array=False to change this behaviour."
            ),
        ):
            r.set_nodata(nodata=new_nodata, update_mask=False)

        # The nodata value should have been set in the metadata
        assert r.nodata == new_nodata

        # The array should have been updated
        if old_nodata is not None:
            index_old_nodata = r_copy.data.data == old_nodata
            assert all(r.data.data[index_old_nodata] == new_nodata)
        else:
            index_old_nodata = np.zeros(np.shape(r.data.data), dtype=bool)

        index_unchanged = np.logical_and(~index_old_nodata, ~mask_pixel_artificially_set)
        # The rest of the array should be similarly unchanged
        assert np.array_equal(r.data.data[index_unchanged], r_copy.data.data[index_unchanged])

        # But the mask should still be the same
        assert np.array_equal(np.ma.getmaskarray(r.data), np.ma.getmaskarray(r_copy.data))

        # -- Fourth, test set_nodata() with both update_array=False and update_mask=False --
        r = r_copy.copy()

        r.set_nodata(nodata=new_nodata, update_array=False, update_mask=False)
        r.data.data[mask_pixel_artificially_set] = new_nodata
        # We set the value as masked before unmasking to create the mask if it does not exist yet
        r.data[mask_pixel_artificially_set] = np.ma.masked
        r.data.mask[mask_pixel_artificially_set] = False

        # The nodata value should have been set in the metadata
        assert r.nodata == new_nodata

        # The array should not have been updated except for the pixel
        assert np.array_equal(r.data.data[~mask_pixel_artificially_set], r_copy.data.data[~mask_pixel_artificially_set])
        # And the mask neither
        assert np.array_equal(np.ma.getmaskarray(r.data), np.ma.getmaskarray(r_copy.data))

        # -- Fifth, let's check that errors are raised when they should --

        # A ValueError if input nodata is neither a list, tuple, integer, floating
        with pytest.raises(ValueError, match="Type of nodata not understood, must be list or float or int"):
            r.set_nodata(nodata="this_should_not_work")  # type: ignore

        # A ValueError if nodata value is incompatible with dtype
        expected_message = r"nodata value .* incompatible with self.dtype .*"
        if "int" in r.dtypes[0]:
            with pytest.raises(ValueError, match=expected_message):
                # Feed a floating numeric to an integer type
                r.set_nodata(0.5)
        elif "float" in r.dtypes[0]:
            # Feed a floating value not supported by our example data
            with pytest.raises(ValueError, match=expected_message):
                r.set_nodata(np.finfo("longdouble").min)

        # -- Sixth, check the special behaviour with None
        r = r_copy.copy()
        r.set_nodata(None)

        # The metadata should be updated to None
        assert r.nodata is None

        # The array cannot be updated, so it is left as is
        assert np.array_equal(r.data.data, r_copy.data.data)
        # However, the old nodata values are unset by default, let's check this
        if old_nodata is not None:
            index_old_nodata = r_copy.data.data == old_nodata
            # The arrays on this index should be booleans opposites
            assert np.array_equal(
                np.ma.getmaskarray(r.data)[index_old_nodata], ~np.ma.getmaskarray(r_copy.data)[index_old_nodata]
            )
        else:
            index_old_nodata = np.zeros(np.shape(r.data.data), dtype=bool)
        # The rest should be equal
        assert np.array_equal(
            np.ma.getmaskarray(r.data)[~index_old_nodata], np.ma.getmaskarray(r_copy.data)[~index_old_nodata]
        )

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_nodata_setter(self, example: str) -> None:
        """Check that the nodata setter gives the same result as set_nodata with default parameters"""

        r = gu.Raster(example)
        r_copy = r.copy()

        r.set_nodata(_default_nodata(r.dtypes[0]))
        r_copy.nodata = _default_nodata(r.dtypes[0])

        assert r == r_copy

    def test_default_nodata(self) -> None:
        """
        Test that the default nodata values are as expected.
        """
        assert _default_nodata("uint8") == np.iinfo("uint8").max
        assert _default_nodata("int8") == np.iinfo("int8").min
        assert _default_nodata("uint16") == np.iinfo("uint16").max
        assert _default_nodata("int16") == np.iinfo("int16").min
        assert _default_nodata("uint32") == 99999
        for dtype in ["int32", "float32", "float64", "longdouble"]:
            assert _default_nodata(dtype) == -99999

        # Check it works with most frequent np.dtypes too
        assert _default_nodata(np.dtype("uint8")) == np.iinfo("uint8").max
        for dtype in [np.dtype("int32"), np.dtype("float32"), np.dtype("float64")]:
            assert _default_nodata(dtype) == -99999

        # Check it works with most frequent types too
        assert _default_nodata(np.uint8) == np.iinfo("uint8").max
        for dtype in [np.int32, np.float32, np.float64]:
            assert _default_nodata(dtype) == -99999

        # Check that an error is raised for other types
        expected_message = "No default nodata value set for dtype"
        with pytest.raises(NotImplementedError, match=expected_message):
            _default_nodata("bla")

    def test_astype(self) -> None:
        warnings.simplefilter("error")

        r = gr.Raster(self.landsat_b4_path)

        # Test changing dtypes that does not modify the data
        for dtype in [np.uint8, np.uint16, np.float32, np.float64, "float32"]:
            rout = r.astype(dtype)  # type: ignore
            assert rout == r
            assert np.dtype(rout.dtypes[0]) == dtype
            assert rout.data.dtype == dtype

        # Test a dtype that will modify the data
        with pytest.warns(UserWarning, match="dtype conversion will result in a loss"):
            dtype = np.int8
            rout = r.astype(dtype)  # type: ignore
            assert rout != r
            assert np.dtype(rout.dtypes[0]) == dtype
            assert rout.data.dtype == dtype

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

        with pytest.warns(UserWarning):
            r2.set_nodata(0)

        for dtype in [np.uint8, np.uint16, np.float32, np.float64, "float32"]:
            rout = r2.astype(dtype)  # type: ignore
            assert rout == r2
            assert np.dtype(rout.dtypes[0]) == dtype
            assert rout.data.dtype == dtype

    def test_plot(self) -> None:

        # Read single band raster and RGB raster
        img = gr.Raster(self.landsat_b4_path)
        img_RGB = gr.Raster(self.landsat_rgb_path)

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
        img = gr.Raster(self.landsat_b4_path)

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
        assert saved.tags["Type"] == "test"

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

        img = gr.Raster(self.landsat_b4_path)
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

        img = gr.Raster(self.landsat_b4_path)
        img2 = gr.Raster(self.landsat_b4_path)

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
        img = gr.Raster(self.landsat_b4_path)

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
        img = gr.Raster(self.landsat_b4_path)
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

        # r = gr.Raster(self.landsat_b4_path)

        # Load the attributes to check
        attributes = ["transform", "crs", "nodata", "name", "driver", "is_loaded", "filename", "nbands", "filename"]
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

        img = gr.Raster(self.landsat_rgb_path)

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
        assert not np.shares_memory(red_c.data, img.data)

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

        img1 = gr.Raster(self.landsat_b4_path)
        img2 = gr.Raster(self.landsat_b4_crop_path)
        img1.set_nodata(0)
        img2.set_nodata(0)

        # Resample the rasters using a new resampling method and see that the string and enum gives the same result.
        img3a = img1.reproject(img2, resampling="q1")
        img3b = img1.reproject(img2, resampling=rio.warp.Resampling.q1)
        assert img3a == img3b

    def test_polygonize(self) -> None:
        """Test that polygonize doesn't raise errors."""
        img = gr.Raster(self.landsat_b4_path)

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

        img2 = gu.Raster(self.landsat_rgb_path, load_data=False)

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


class TestArithmetic:
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

    # Test with nodata value set
    r1_nodata = gr.Raster.from_array(
        np.random.randint(1, 255, (height, width)).astype("float32"),
        transform=transform,
        crs=None,
        nodata=_default_nodata("float32"),
    )

    # Test with 0 values
    r2_zero = gr.Raster.from_array(
        np.random.randint(1, 255, (height, width)).astype("float32"),
        transform=transform,
        crs=None,
        nodata=_default_nodata("float32"),
    )
    r2_zero.data[0, 0, 0] = 0

    # Create rasters with different shape, crs or transforms for testing errors
    r1_wrong_shape = gr.Raster.from_array(
        np.random.randint(0, 255, (height + 1, width)).astype("float32"),
        transform=transform,
        crs=None,
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

        # Change nodata
        r2 = r1.copy()
        r2.set_nodata(34)
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
        r1_nodata = self.r1_nodata
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
            assert r3.nodata == _default_nodata(ctype)
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
            assert r3.nodata == _default_nodata("float32")

        # Test with nodata set
        r1 = self.r1
        r3 = getattr(r1_nodata, op)(r2)
        assert np.all(r3.data == getattr(r1_nodata.data, op)(r2.data))
        if np.sum(r3.data.mask) == 0:
            assert r3.nodata == r1_nodata.nodata
        else:
            assert r3.nodata == _default_nodata(r1_nodata.data.dtype)

        # Test with zeros values (e.g. division)
        r1 = self.r1
        r3 = getattr(r1, op)(r2_zero)
        assert np.all(r3.data == getattr(r1.data, op)(r2_zero.data))
        if np.sum(r3.data.mask) == 0:
            assert r3.nodata == r2_zero.nodata
        else:
            assert r3.nodata == _default_nodata(r1_nodata.data.dtype)

        # Test with a numpy array
        r1 = self.r1_f32
        r3 = getattr(r1, op)(array)
        assert isinstance(r3, gr.Raster)
        assert np.all(r3.data == getattr(r1.data, op)(array))
        if np.sum(r3.data.mask) == 0:
            assert r3.nodata is None
        else:
            assert r3.nodata == _default_nodata("float32")

        # Test with an integer
        r3 = getattr(r1, op)(intval)
        assert isinstance(r3, gr.Raster)
        assert np.all(r3.data == getattr(r1.data, op)(intval))
        if np.sum(r3.data.mask) == 0:
            assert r3.nodata is None
        else:
            assert r3.nodata == _default_nodata("uint8")

        # Test with a float value
        r3 = getattr(r1, op)(floatval)
        dtype = np.dtype(rio.dtypes.get_minimum_dtype(floatval))
        assert isinstance(r3, gr.Raster)
        assert r3.data.dtype == dtype
        assert np.all(r3.data == getattr(r1.data, op)(np.array(floatval).astype(dtype)))
        if np.sum(r3.data.mask) == 0:
            assert r3.nodata is None
        else:
            assert r3.nodata == _default_nodata(dtype)

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

        # Test with nodata set
        r3 = getattr(self.r1_nodata, op1)(self.r2)
        r4 = getattr(self.r1_nodata, op2)(self.r2)
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
        cls: type[TestArithmetic],
        data: np.ndarray | np.ma.masked_array,
        rst_ref: gr.RasterType,
        nodata: int | float | list[int] | list[float] | None = None,
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

        # Test various inputs: Raster with different dtypes, np.ndarray with 2D or 3D shape, single number
        r1 = self.r1
        r1_f32 = self.r1_f32
        r2 = self.r2
        array_3d = np.random.randint(1, 255, (1, self.height, self.width)).astype("uint8")
        array_2d = np.random.randint(1, 255, (self.height, self.width)).astype("uint8")
        floatval = 3.14

        # Addition
        assert r1 + r2 == self.from_array(r1.data + r2.data, rst_ref=r1)
        assert r1_f32 + r2 == self.from_array(r1_f32.data + r2.data, rst_ref=r1)
        assert array_3d + r2 == self.from_array(array_3d + r2.data, rst_ref=r1)
        assert r2 + array_3d == self.from_array(r2.data + array_3d, rst_ref=r1)
        assert array_2d + r2 == self.from_array(array_2d[np.newaxis, :, :] + r2.data, rst_ref=r1)
        assert r2 + array_2d == self.from_array(r2.data + array_2d[np.newaxis, :, :], rst_ref=r1)
        assert r1 + floatval == self.from_array(r1.data.astype("float32") + floatval, rst_ref=r1)
        assert floatval + r1 == self.from_array(floatval + r1.data.astype("float32"), rst_ref=r1)
        assert r1 + r2 == r2 + r1

        # Multiplication
        assert r1 * r2 == self.from_array(r1.data * r2.data, rst_ref=r1)
        assert r1_f32 * r2 == self.from_array(r1_f32.data * r2.data, rst_ref=r1)
        assert array_3d * r2 == self.from_array(array_3d * r2.data, rst_ref=r1)
        assert r2 * array_3d == self.from_array(r2.data * array_3d, rst_ref=r1)
        assert array_2d * r2 == self.from_array(array_2d[np.newaxis, :, :] * r2.data, rst_ref=r1)
        assert r2 * array_2d == self.from_array(r2.data * array_2d[np.newaxis, :, :], rst_ref=r1)
        assert r1 * floatval == self.from_array(r1.data.astype("float32") * floatval, rst_ref=r1)
        assert floatval * r1 == self.from_array(floatval * r1.data.astype("float32"), rst_ref=r1)
        assert r1 * r2 == r2 * r1

        # Subtraction
        assert r1 - r2 == self.from_array(r1.data - r2.data, rst_ref=r1)
        assert r1_f32 - r2 == self.from_array(r1_f32.data - r2.data, rst_ref=r1)
        assert array_3d - r2 == self.from_array(array_3d - r2.data, rst_ref=r1)
        assert r2 - array_3d == self.from_array(r2.data - array_3d, rst_ref=r1)
        assert array_2d - r2 == self.from_array(array_2d[np.newaxis, :, :] - r2.data, rst_ref=r1)
        assert r2 - array_2d == self.from_array(r2.data - array_2d[np.newaxis, :, :], rst_ref=r1)
        assert r1 - floatval == self.from_array(r1.data.astype("float32") - floatval, rst_ref=r1)
        assert floatval - r1 == self.from_array(floatval - r1.data.astype("float32"), rst_ref=r1)

        # True division
        assert r1 / r2 == self.from_array(r1.data / r2.data, rst_ref=r1)
        assert r1_f32 / r2 == self.from_array(r1_f32.data / r2.data, rst_ref=r1)
        assert array_3d / r2 == self.from_array(array_3d / r2.data, rst_ref=r1)
        assert r2 / array_3d == self.from_array(r2.data / array_3d, rst_ref=r2)
        assert array_2d / r2 == self.from_array(array_2d[np.newaxis, :, :] / r2.data, rst_ref=r1)
        assert r2 / array_2d == self.from_array(r2.data / array_2d[np.newaxis, :, :], rst_ref=r2)
        assert r1 / floatval == self.from_array(r1.data.astype("float32") / floatval, rst_ref=r1)
        assert floatval / r1 == self.from_array(floatval / r1.data.astype("float32"), rst_ref=r1)

        # Floor division
        assert r1 // r2 == self.from_array(r1.data // r2.data, rst_ref=r1)
        assert r1_f32 // r2 == self.from_array(r1_f32.data // r2.data, rst_ref=r1)
        assert array_3d // r2 == self.from_array(array_3d // r2.data, rst_ref=r1)
        assert r2 // array_3d == self.from_array(r2.data // array_3d, rst_ref=r1)
        assert array_2d // r2 == self.from_array(array_2d[np.newaxis, :, :] // r2.data, rst_ref=r1)
        assert r2 // array_2d == self.from_array(r2.data // array_2d[np.newaxis, :, :], rst_ref=r1)
        assert r1 // floatval == self.from_array(r1.data // floatval, rst_ref=r1)
        assert floatval // r1 == self.from_array(floatval // r1.data, rst_ref=r1)

        # Modulo
        assert r1 % r2 == self.from_array(r1.data % r2.data, rst_ref=r1)
        assert r1_f32 % r2 == self.from_array(r1_f32.data % r2.data, rst_ref=r1)
        assert array_3d % r2 == self.from_array(array_3d % r2.data, rst_ref=r1)
        assert r2 % array_3d == self.from_array(r2.data % array_3d, rst_ref=r1)
        assert array_2d % r2 == self.from_array(array_2d[np.newaxis, :, :] % r2.data, rst_ref=r1)
        assert r2 % array_2d == self.from_array(r2.data % array_2d[np.newaxis, :, :], rst_ref=r1)
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


class TestArrayInterface:
    """Test that the array interface of Raster works as expected for ufuncs and array functions"""

    # -- First, we list all universal NumPy functions, or "ufuncs" --

    # All universal functions of NumPy, about 90 in 2022. See list: https://numpy.org/doc/stable/reference/ufuncs.html
    ufuncs_str = [
        ufunc
        for ufunc in np.core.umath.__all__
        if (
            ufunc[0] != "_"
            and ufunc.islower()
            and "err" not in ufunc
            and ufunc not in ["e", "pi", "frompyfunc", "euler_gamma"]
        )
    ]

    # Universal functions with one input argument and one output, corresponding to (in NumPy 1.22.4):
    # ['absolute', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctanh', 'cbrt', 'ceil', 'conj', 'conjugate',
    # 'cos', 'cosh', 'deg2rad', 'degrees', 'exp', 'exp2', 'expm1', 'fabs', 'floor', 'invert', 'isfinite', 'isinf',
    # 'isnan', 'isnat', 'log', 'log10', 'log1p', 'log2', 'logical_not', 'negative', 'positive', 'rad2deg', 'radians',
    # 'reciprocal', 'rint', 'sign', 'signbit', 'sin', 'sinh', 'spacing', 'sqrt', 'square', 'tan', 'tanh', 'trunc']
    ufuncs_str_1nin_1nout = [
        ufunc for ufunc in ufuncs_str if (getattr(np, ufunc).nin == 1 and getattr(np, ufunc).nout == 1)
    ]

    # Universal functions with one input argument and two output (Note: none exist for three outputs or above)
    # Those correspond to: ['frexp', 'modf']
    ufuncs_str_1nin_2nout = [
        ufunc for ufunc in ufuncs_str if (getattr(np, ufunc).nin == 1 and getattr(np, ufunc).nout == 2)
    ]

    # Universal functions with two input arguments and one output, corresponding to:
    # ['add', 'arctan2', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'copysign', 'divide', 'equal', 'floor_divide',
    #  'float_power', 'fmax', 'fmin', 'fmod', 'gcd', 'greater', 'greater_equal', 'heaviside', 'hypot', 'lcm', 'ldexp',
    #  'left_shift', 'less', 'less_equal', 'logaddexp', 'logaddexp2', 'logical_and', 'logical_or', 'logical_xor',
    #  'maximum', 'minimum', 'mod', 'multiply', 'nextafter', 'not_equal', 'power', 'remainder', 'right_shift',
    #  'subtract', 'true_divide']
    ufuncs_str_2nin_1nout = [
        ufunc for ufunc in ufuncs_str if (getattr(np, ufunc).nin == 2 and getattr(np, ufunc).nout == 1)
    ]

    # Universal functions with two input arguments and two outputs (Note: none exist for three outputs or above)
    # These correspond to: ['divmod']
    ufuncs_str_2nin_2nout = [
        ufunc for ufunc in ufuncs_str if (getattr(np, ufunc).nin == 2 and getattr(np, ufunc).nout == 2)
    ]

    # -- Second, we list array functions we intend to support in the array interface --

    # To my knowledge, there is no list that includes all numpy functions (and we probably don't want to test them all)
    # Let's include manually the important ones:
    # - statistics: normal and for NaNs;
    # - sorting and counting;
    # Most other math functions are already universal functions

    # The full list exists in Raster
    handled_functions = gu.georaster.raster._HANDLED_FUNCTIONS

    # Details below:
    # NaN functions: [f for f in np.lib.nanfunctions.__all__]
    # nanstatfuncs = ['nansum', 'nanmax', 'nanmin', 'nanargmax', 'nanargmin', 'nanmean', 'nanmedian', 'nanpercentile',
    #             'nanvar', 'nanstd', 'nanprod', 'nancumsum', 'nancumprod', 'nanquantile']

    # Statistics and sorting matching NaN functions: https://numpy.org/doc/stable/reference/routines.statistics.html
    # and https://numpy.org/doc/stable/reference/routines.sort.html
    # statfuncs = ['sum', 'max', 'min', 'argmax', 'argmin', 'mean', 'median', 'percentile', 'var', 'std', 'prod',
    #              'cumsum', 'cumprod', 'quantile']

    # Sorting and counting ounting with single array input:
    # sortfuncs = ['sort', 'count_nonzero', 'unique]

    # --  Third, we define the test data --

    # We create two random array of varying dtype
    width = height = 5
    min_val = np.iinfo("int32").min
    max_val = np.iinfo("int32").max
    transform = rio.transform.from_bounds(0, 0, 1, 1, width, height)
    np.random.seed(42)
    arr1 = np.random.randint(min_val, max_val, (height, width), dtype="int32") + np.random.normal(size=(height, width))
    arr2 = np.random.randint(min_val, max_val, (height, width), dtype="int32") + np.random.normal(size=(height, width))

    # Create two random masks
    mask1 = np.random.randint(0, 2, size=(width, height), dtype=bool)
    mask2 = np.random.randint(0, 2, size=(width, height), dtype=bool)

    # Assert that there is at least one unmasked value
    assert np.count_nonzero(~mask1) > 0
    assert np.count_nonzero(~mask2) > 0

    @pytest.mark.parametrize("ufunc_str", ufuncs_str_1nin_1nout + ufuncs_str_1nin_2nout)  # type: ignore
    @pytest.mark.parametrize(
        "dtype", ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64", "longdouble"]
    )  # type: ignore
    @pytest.mark.parametrize("nodata_init", [None, "type_default"])  # type: ignore
    def test_array_ufunc_1nin_1nout(self, ufunc_str: str, nodata_init: None | str, dtype: str) -> None:
        """Test that ufuncs with one input and one output consistently return the same result as for masked arrays."""

        # We set the default nodata
        if nodata_init == "type_default":
            nodata: int | None = _default_nodata(dtype)
        else:
            nodata = None

        # Create Raster
        ma1 = np.ma.masked_array(data=self.arr1.astype(dtype), mask=self.mask1)
        rst = gr.Raster.from_array(ma1, transform=self.transform, crs=None, nodata=nodata)

        # Get ufunc
        ufunc = getattr(np, ufunc_str)

        # Find the common dtype between the Raster and the most constrained input type (first character is the input)
        com_dtype = np.find_common_type([dtype] + [ufunc.types[0][0]], [])

        # Catch warnings
        with warnings.catch_warnings():

            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            # Check if our input dtype is possible on this ufunc, if yes check that outputs are identical
            if com_dtype in (str(np.dtype(t[0])) for t in ufunc.types):

                # For a single output
                if ufunc.nout == 1:
                    assert np.ma.allequal(ufunc(rst.data), ufunc(rst).data)

                # For two outputs
                elif ufunc.nout == 2:
                    outputs_rst = ufunc(rst)
                    outputs_ma = ufunc(rst.data)
                    assert np.ma.allequal(outputs_ma[0], outputs_rst[0].data) and np.ma.allequal(
                        outputs_ma[1], outputs_rst[1].data
                    )

            # If the input dtype is not possible, check that NumPy raises a TypeError
            else:
                with pytest.raises(TypeError):
                    ufunc(rst.data)
                with pytest.raises(TypeError):
                    ufunc(rst)

    @pytest.mark.parametrize("ufunc_str", ufuncs_str_2nin_1nout + ufuncs_str_2nin_2nout)  # type: ignore
    @pytest.mark.parametrize(
        "dtype1", ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64", "longdouble"]
    )  # type: ignore
    @pytest.mark.parametrize(
        "dtype2", ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64", "longdouble"]
    )  # type: ignore
    @pytest.mark.parametrize("nodata1_init", [None, "type_default"])  # type: ignore
    @pytest.mark.parametrize("nodata2_init", [None, "type_default"])  # type: ignore
    def test_array_ufunc_2nin_1nout(
        self, ufunc_str: str, nodata1_init: None | str, nodata2_init: str, dtype1: str, dtype2: str
    ) -> None:
        """Test that ufuncs with two input arguments consistently return the same result as for masked arrays."""

        # We set the default nodatas
        if nodata1_init == "type_default":
            nodata1: int | None = _default_nodata(dtype1)
        else:
            nodata1 = None
        if nodata2_init == "type_default":
            nodata2: int | None = _default_nodata(dtype2)
        else:
            nodata2 = None

        ma1 = np.ma.masked_array(data=self.arr1.astype(dtype1), mask=self.mask1)
        ma2 = np.ma.masked_array(data=self.arr2.astype(dtype2), mask=self.mask2)
        rst1 = gr.Raster.from_array(ma1, transform=self.transform, crs=None, nodata=nodata1)
        rst2 = gr.Raster.from_array(ma2, transform=self.transform, crs=None, nodata=nodata2)

        ufunc = getattr(np, ufunc_str)

        # Find the common dtype between the Raster and the most constrained input type (first character is the input)
        com_dtype1 = np.find_common_type([dtype1] + [ufunc.types[0][0]], [])
        com_dtype2 = np.find_common_type([dtype2] + [ufunc.types[0][1]], [])

        # If the two input types can be the same type, pass a tuple with the common type of both
        # Below we ignore datetime and timedelta types "m" and "M"
        if all(t[0] == t[1] for t in ufunc.types if not ("m" or "M") in t[0:2]):
            com_dtype_both = np.find_common_type([com_dtype1, com_dtype2], [])
            com_dtype_tuple = (com_dtype_both, com_dtype_both)

        # Otherwise, pass the tuple with each common type
        else:
            com_dtype_tuple = (com_dtype1, com_dtype2)

        # Catch warnings
        with warnings.catch_warnings():

            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            # Check if both our input dtypes are possible on this ufunc, if yes check that outputs are identical
            if com_dtype_tuple in ((str(np.dtype(t[0])), str(np.dtype(t[1]))) for t in ufunc.types):

                # For a single output
                if ufunc.nout == 1:

                    # There exists a single exception due to negative integers as exponent of integers in "power"
                    if ufunc_str == "power" and "int" in dtype1 and "int" in dtype2 and np.min(rst2.data) < 0:
                        with pytest.raises(ValueError, match="Integers to negative integer powers are not allowed."):
                            ufunc(rst1, rst2)
                        with pytest.raises(ValueError, match="Integers to negative integer powers are not allowed."):
                            ufunc(rst1.data, rst2.data)

                    # Otherwise, run the normal assertion for a single output
                    else:
                        assert np.ma.allequal(ufunc(rst1.data, rst2.data), ufunc(rst1, rst2).data)

                # For two outputs
                elif ufunc.nout == 2:
                    outputs_rst = ufunc(rst1, rst2)
                    outputs_ma = ufunc(rst1.data, rst2.data)
                    assert np.ma.allequal(outputs_ma[0], outputs_rst[0].data) and np.ma.allequal(
                        outputs_ma[1], outputs_rst[1].data
                    )

            # If the input dtype is not possible, check that NumPy raises a TypeError
            else:
                with pytest.raises(TypeError):
                    ufunc(rst1.data, rst2.data)
                with pytest.raises(TypeError):
                    ufunc(rst1, rst2)

    @pytest.mark.parametrize("arrfunc_str", handled_functions)  # type: ignore
    @pytest.mark.parametrize(
        "dtype", ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64", "longdouble"]
    )  # type: ignore
    @pytest.mark.parametrize("nodata_init", [None, "type_default"])  # type: ignore
    def test_array_functions(self, arrfunc_str: str, dtype: str, nodata_init: None | str) -> None:
        """Test that array functions that we support give the same output as they would on the masked array"""

        # We set the default nodata
        if nodata_init == "type_default":
            nodata: int | None = _default_nodata(dtype)
        else:
            nodata = None

        # Create Raster
        ma1 = np.ma.masked_array(data=self.arr1.astype(dtype), mask=self.mask1)
        rst = gr.Raster.from_array(ma1, transform=self.transform, crs=None, nodata=nodata)

        # Get array func
        arrfunc = getattr(np, arrfunc_str)

        # Find the common dtype between the Raster and the most constrained input type (first character is the input)
        # com_dtype = np.find_common_type([dtype] + [arrfunc.types[0][0]], [])

        # Catch warnings
        with warnings.catch_warnings():

            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # Pass an argument for functions that require it (nanpercentile, percentile, quantile and nanquantile) and
            # define specific behaviour
            if "percentile" in arrfunc_str:
                arg = 80.0
                # For percentiles and quantiles, the statistic is computed after removing the masked values
                output_rst = arrfunc(rst, arg)
                output_ma = arrfunc(rst.data.compressed(), arg)
            elif "quantile" in arrfunc_str:
                arg = 0.8
                output_rst = arrfunc(rst, arg)
                output_ma = arrfunc(rst.data.compressed(), arg)
            elif "median" in arrfunc_str:
                # For the median, the statistic is computed by masking the values through np.ma.median
                output_rst = arrfunc(rst)
                output_ma = np.ma.median(rst.data)
            else:
                output_rst = arrfunc(rst)
                output_ma = arrfunc(rst.data)

            if isinstance(output_rst, np.ndarray):
                assert np.ma.allequal(output_rst, output_ma)
            else:
                assert output_rst == output_ma
