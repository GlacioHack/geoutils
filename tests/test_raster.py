"""
Test functions for raster
"""
from __future__ import annotations

import os
import pathlib
import re
import tempfile
import warnings
from io import StringIO
from tempfile import TemporaryFile

import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio as rio
import xarray as xr
from pylint.lint import Run
from pylint.reporters.text import TextReporter

import geoutils as gu
import geoutils.projtools as pt
from geoutils import examples
from geoutils._typing import MArrayNum, NDArrayNum
from geoutils.misc import resampling_method_from_str
from geoutils.projtools import reproject_to_latlon
from geoutils.raster.raster import _default_nodata, _default_rio_attrs

DO_PLOT = False


def run_gdal_proximity(
    input_raster: gu.Raster, target_values: list[float] | None, distunits: str = "GEO"
) -> NDArrayNum:
    """Run GDAL's ComputeProximity and return the read numpy array."""
    # Rasterio strongly recommends against importing gdal along rio, so this is done here instead.
    from osgeo import gdal, gdalconst

    gdal.UseExceptions()

    # Initiate empty GDAL raster for proximity output
    drv = gdal.GetDriverByName("MEM")
    proxy_ds = drv.Create("", input_raster.shape[1], input_raster.shape[0], 1, gdal.GetDataTypeByName("Float32"))
    proxy_ds.GetRasterBand(1).SetNoDataValue(-9999)

    # Save input in temporary file to read with GDAL
    # (avoids the nightmare of setting nodata, transform, crs in GDAL format...)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, "input.tif")
        input_raster.save(temp_path)
        ds_raster_in = gdal.Open(temp_path, gdalconst.GA_ReadOnly)

        # Define GDAL options
        proximity_options = ["DISTUNITS=" + distunits]
        if target_values is not None:
            proximity_options.insert(0, "VALUES=" + ",".join([str(tgt) for tgt in target_values]))

        # Compute proximity
        gdal.ComputeProximity(ds_raster_in.GetRasterBand(1), proxy_ds.GetRasterBand(1), proximity_options)
        # Save array
        proxy_array = proxy_ds.GetRasterBand(1).ReadAsArray().astype("float32")
        proxy_array[proxy_array == -9999] = np.nan

        # Close GDAL datasets
        proxy_ds = None
        ds_raster_in = None

    return proxy_array


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

        # First, string filename
        r0 = gu.Raster(example)
        assert isinstance(r0, gu.Raster)

        # Second, filename in a pathlib object
        path = pathlib.Path(example)
        r1 = gu.Raster(path)
        assert isinstance(r1, gu.Raster)

        # Third, passing a Raster itself (points back to Raster passed)
        r2 = gu.Raster(r0)
        assert isinstance(r2, gu.Raster)

        # Fourth, rio.Dataset
        ds = rio.open(example)
        r3 = gu.Raster(ds)
        assert isinstance(r3, gu.Raster)
        assert r3.filename is not None

        # Finally, as memoryfile
        memfile = rio.MemoryFile(open(example, "rb"))
        r4 = gu.Raster(memfile)
        assert isinstance(r4, gu.Raster)

        # All rasters should be equal
        assert all(
            [r0.raster_equal(r1), r0.raster_equal(r1), r0.raster_equal(r2), r0.raster_equal(r3), r0.raster_equal(r4)]
        )

        # For re-instantiation via Raster (r2 above), we check the behaviour:
        # By default, raster were unloaded, and were loaded during raster_equal() independently
        # So the instances should not be pointing to the same data and not mirror modifs
        r0.data[0, 0] += 5
        assert r2.data[0, 0] != r0.data[0, 0]

        # However, if we reinstantiate now that the data is loaded, it should point to the same
        r5 = gu.Raster(r0)
        r0.data[0, 0] += 5
        assert r5.data[0, 0] == r0.data[0, 0]

        # With r.count = 2
        r0._data = np.repeat(r0.data, 2).reshape((2,) + r0.shape)
        assert r0.count != r2.count

        # Test that loaded data are always masked_arrays (but the mask may be empty, i.e. 'False')
        assert np.ma.isMaskedArray(gu.Raster(example).data)

        # Check that an error is raised when instantiating with an array
        with pytest.raises(
            TypeError, match=re.escape("The filename is an array, did you mean to call Raster.from_array(...) instead?")
        ):
            gu.Raster(np.ones(shape=(1, 1)))  # type: ignore
        with pytest.raises(
            TypeError, match="The filename argument is not recognised, should be a path or a Rasterio dataset."
        ):
            gu.Raster(1)  # type: ignore

    @pytest.mark.parametrize("example", [landsat_b4_path, landsat_rgb_path, aster_dem_path])  # type: ignore
    def test_repr_str(self, example: str) -> None:
        """Test the representation of a raster works"""

        # For data not loaded by default
        r = gu.Raster(example)

        r_repr = r.__repr__()
        r_str = r.__str__()

        # Check that the class is printed correctly
        assert r_repr[0:6] == "Raster"

        # Check that all main attribute names are printed
        attrs_shown = ["data", "transform", "crs", "nodata"]
        assert all(attr + "=" in r_repr for attr in attrs_shown)

        assert r_str == "not_loaded"
        assert r_repr.split("data=")[1][:10] == "not_loaded"

        # With data loaded
        r.load()

        r_repr = r.__repr__()
        r_str = r.__str__()

        assert r_str == r.data.__str__()
        assert r_repr.split("data=")[1][:10] != "not_loaded"

    @pytest.mark.parametrize("example", [landsat_b4_path, landsat_rgb_path, aster_dem_path])  # type: ignore
    def test_info(self, example: str) -> None:
        """Test that the information summary is consistent with that of rasterio"""

        r = gu.Raster(example)

        # Check default runs without error (prints to screen)
        output = r.info()
        assert output is None

        # Check all is good with passing attributes
        with rio.open(example) as dataset:
            for attr in _default_rio_attrs:
                assert r.__getattribute__(attr) == dataset.__getattribute__(attr)

        # Check that the stats=True flag doesn't trigger a warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="New nodata.*", category=UserWarning)
            stats = r.info(stats=True, verbose=False)

        # Check the stats adapt to nodata values
        if r.dtypes[0] == "uint8":
            # Validate that the mask is respected by adding 0 values (there are none to begin with.)
            r.data.ravel()[:1000] = 0
            # Set the nodata value to 0, then validate that they are excluded from the new minimum, and warning raised
            with pytest.warns(UserWarning):
                r.set_nodata(0)
        elif r.dtypes[0] == "float32":
            # We do the same with -99999 here
            r.data.ravel()[:1000] = -99999
            # And replace the nodata value, and warning raised
            with pytest.warns(UserWarning):
                r.set_nodata(-99999)

        new_stats = r.info(stats=True, verbose=False)
        for i, line in enumerate(stats.splitlines()):
            if "MINIMUM" not in line:
                continue
            assert line == new_stats.splitlines()[i]

    def test_load(self) -> None:
        """
        Test that loading metadata and data works for all possible cases.
        """
        # Test 0 - test that loading explicitly via load() or implicitly via .data is similar
        r_explicit = gu.Raster(self.landsat_b4_path, load_data=True)
        r_implicit = gu.Raster(self.landsat_b4_path)

        assert r_explicit.raster_equal(r_implicit)

        # Same for a multi-band raster
        r_explicit_rgb = gu.Raster(self.landsat_rgb_path, load_data=True)
        r_implicit_rgb = gu.Raster(self.landsat_rgb_path)

        assert r_explicit_rgb.raster_equal(r_implicit_rgb)

        # Test 1 - loading metadata only, single band
        # For the first example with Landsat B4
        r = gu.Raster(self.landsat_b4_path)

        assert not r.is_loaded
        assert r.driver == "GTiff"
        assert r.width == 800
        assert r.height == 655
        assert r.shape == (r.height, r.width)
        assert r.count == 1
        assert r.count_on_disk == 1
        assert r.bands == (1,)
        assert r.bands_on_disk == (1,)
        assert np.array_equal(r.dtypes, ["uint8"])
        assert r.transform == rio.transform.Affine(30.0, 0.0, 478000.0, 0.0, -30.0, 3108140.0)
        assert np.array_equal(r.res, [30.0, 30.0])
        assert r.bounds == rio.coords.BoundingBox(left=478000.0, bottom=3088490.0, right=502000.0, top=3108140.0)
        assert r.crs == rio.crs.CRS.from_epsg(32645)

        # And the second example with ASTER DEM
        r2 = gu.Raster(self.aster_dem_path)

        assert not r2.is_loaded
        assert r2.driver == "GTiff"
        assert r2.width == 539
        assert r2.height == 618
        assert r2.shape == (r2.height, r2.width)
        assert r2.count == 1
        assert r.count_on_disk == 1
        assert r.bands == (1,)
        assert r.bands_on_disk == (1,)
        assert np.array_equal(r2.dtypes, ["float32"])
        assert r2.transform == rio.transform.Affine(30.0, 0.0, 627175.0, 0.0, -30.0, 4852085.0)
        assert np.array_equal(r2.res, [30.0, 30.0])
        assert r2.bounds == rio.coords.BoundingBox(left=627175.0, bottom=4833545.0, right=643345.0, top=4852085.0)
        assert r2.crs == rio.crs.CRS.from_epsg(32718)

        # Test 2 - loading the data afterward
        r.load()
        assert r.is_loaded
        assert r.count == 1
        assert r.count_on_disk == 1
        assert r.bands == (1,)
        assert r.bands_on_disk == (1,)
        assert r.data.shape == (r.height, r.width)

        # Test 3 - single band, loading data
        r = gu.Raster(self.landsat_b4_path, load_data=True)
        assert r.is_loaded
        assert r.count == 1
        assert r.count_on_disk == 1
        assert r.bands == (1,)
        assert r.bands_on_disk == (1,)
        assert r.data.shape == (r.height, r.width)

        # Test 4 - multiple bands, load all bands
        r = gu.Raster(self.landsat_rgb_path, load_data=True)
        assert r.count == 3
        assert r.count_on_disk == 3
        assert r.bands == (
            1,
            2,
            3,
        )
        assert r.bands_on_disk == (
            1,
            2,
            3,
        )
        assert r.data.shape == (r.count, r.height, r.width)

        # Test 5 - multiple bands, load one band only
        r = gu.Raster(self.landsat_rgb_path, load_data=True, bands=1)
        assert r.count == 1
        assert r.count_on_disk == 3
        assert r.bands == (1,)
        assert r.bands_on_disk == (1, 2, 3)
        assert r.data.shape == (r.height, r.width)

        # Test 6 - multiple bands, load a list of bands
        r = gu.Raster(self.landsat_rgb_path, load_data=True, bands=[2, 3])
        assert r.count == 2
        assert r.count_on_disk == 3
        assert r.bands == (1, 2)
        assert r.bands_on_disk == (1, 2, 3)
        assert r.data.shape == (r.count, r.height, r.width)

        # Test 7 - load a single band a posteriori calling load()
        r = gu.Raster(self.landsat_rgb_path)
        r.load(bands=1)
        assert r.count == 1
        assert r.count_on_disk == 3
        assert r.bands == (1,)
        assert r.bands_on_disk == (1, 2, 3)
        assert r.data.shape == (r.height, r.width)

        # Test 8 - load a list of band a posteriori calling load()
        r = gu.Raster(self.landsat_rgb_path)
        r.load(bands=[2, 3])
        assert r.count == 2
        assert r.count_on_disk == 3
        assert r.bands == (1, 2)
        assert r.bands_on_disk == (1, 2, 3)
        assert r.data.shape == (r.count, r.height, r.width)

        # Check that errors are raised when appropriate
        with pytest.raises(ValueError, match="Data are already loaded."):
            r.load()
        with pytest.raises(
            AttributeError,
            match="Cannot load as filename is not set anymore. " "Did you manually update the filename attribute?",
        ):
            r = gu.Raster(self.landsat_b4_path)
            r.filename = None
            r.load()

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_to_rio_dataset(self, example: str):
        """Test the export to a rasterio dataset"""

        # Open raster and export to rio dataset
        rst = gu.Raster(example)
        rio_ds = rst.to_rio_dataset()

        # Check that the output is indeed a MemoryFile
        assert isinstance(rio_ds, rio.io.DatasetReader)

        # Check that all attributes are equal
        rio_attrs_conserved = [attr for attr in _default_rio_attrs if attr not in ["name", "driver"]]
        for attr in rio_attrs_conserved:
            assert rst.__getattribute__(attr) == rio_ds.__getattribute__(attr)

        # Check that the masked arrays are equal
        assert np.array_equal(rst.data.data, rio_ds.read().squeeze())
        assert np.array_equal(rst.data.mask, rio_ds.read(masked=True).mask.squeeze())

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path, landsat_rgb_path])  # type: ignore
    def test_to_xarray(self, example: str):
        """Test the export to a xarray dataset"""

        # Open raster and export to rio dataset
        rst = gu.Raster(example)
        ds = rst.to_xarray()

        # Check that the output is indeed a xarray Dataset
        assert isinstance(ds, xr.DataArray)

        # Check that all attributes are equal
        if rst.count > 1:
            assert ds.band.size == rst.count
        assert ds.x.size == rst.width
        assert ds.y.size == rst.height

        # Check that coordinates are shifted by a half pixel
        assert ds.x.values[0] == rst.bounds.left + rst.res[0] / 2
        assert ds.y.values[0] == rst.bounds.top - rst.res[1] / 2
        assert ds.x.values[-1] == rst.bounds.right - rst.res[0] / 2
        assert ds.y.values[-1] == rst.bounds.bottom + rst.res[1] / 2

        # Check that georeferencing attribute are equal
        new_trans_order = ["c", "a", "b", "f", "d", "e"]
        assert ds.spatial_ref.GeoTransform == " ".join([str(getattr(rst.transform, attr)) for attr in new_trans_order])

        # Check that CRS are equal
        assert ds.spatial_ref.crs_wkt == rst.crs.to_wkt()

        # Check that the arrays are equal in NaN type
        if rst.count > 1:
            assert np.array_equal(rst.data.data, ds.data)
        else:
            assert np.array_equal(rst.data.data, ds.data.squeeze())

    @pytest.mark.parametrize("nodata_init", [None, "type_default"])  # type: ignore
    @pytest.mark.parametrize(
        "dtype", ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"]
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
        7. Raises a warning if the new data has the nodata value in the masked array, but unmasked.
        """

        # Initiate a random array for testing
        width = height = 5
        transform = rio.transform.from_bounds(0, 0, 1, 1, width, height)

        # Create random values between the lower and upper limit of the data type, max absolute 99999 for floats
        if "int" in dtype:
            val_min: int = np.iinfo(int_type=dtype).min  # type: ignore
            val_max: int = np.iinfo(int_type=dtype).max  # type: ignore
            randint_dtype = dtype
        else:
            val_min = -99999
            val_max = 99999
            randint_dtype = "int32"

        # Fix the random seed
        np.random.seed(42)
        arr = np.random.randint(
            low=val_min, high=val_max, size=(width, height), dtype=randint_dtype  # type: ignore
        ).astype(dtype)
        mask = np.random.randint(0, 2, size=(width, height), dtype=bool)

        # Check that we are actually masking stuff
        assert np.count_nonzero(mask) > 0

        # Add a random floating point value if the data type is float
        if "float" in dtype:
            arr += np.random.normal(size=(width, height))

        # Use either the default nodata or None
        if nodata_init == "type_default":
            nodata: int | None = _default_nodata(dtype)
        else:
            nodata = None

        # -- First test: consistency with input array --

        # 3 cases: classic array without mask, masked_array without mask and masked_array with mask
        r1 = gu.Raster.from_array(data=arr, transform=transform, crs=None, nodata=nodata)
        r2 = gu.Raster.from_array(data=np.ma.masked_array(arr), transform=transform, crs=None, nodata=nodata)
        r3 = gu.Raster.from_array(data=np.ma.masked_array(arr, mask=mask), transform=transform, crs=None, nodata=nodata)

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
        r1 = gu.Raster.from_array(data=arr.squeeze(), transform=transform, crs=None, nodata=nodata)
        r2 = gu.Raster.from_array(data=np.ma.masked_array(arr).squeeze(), transform=transform, crs=None, nodata=nodata)
        r3 = gu.Raster.from_array(
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
            arr_with_unmasked_nodata[ind_nm[0, rand_ind], ind_nm[1, rand_ind]] = np.nan

            if nodata is None:
                with pytest.warns(
                    UserWarning,
                    match="Setting default nodata {:.0f} to mask non-finite values found in the array, as "
                    "no nodata value was defined.".format(_default_nodata(dtype)),
                ):
                    r1 = gu.Raster.from_array(
                        data=arr_with_unmasked_nodata, transform=transform, crs=None, nodata=nodata
                    )
                    r2 = gu.Raster.from_array(
                        data=np.ma.masked_array(arr_with_unmasked_nodata), transform=transform, crs=None, nodata=nodata
                    )
                    r3 = gu.Raster.from_array(
                        data=np.ma.masked_array(arr_with_unmasked_nodata, mask=mask),
                        transform=transform,
                        crs=None,
                        nodata=nodata,
                    )
            else:
                r1 = gu.Raster.from_array(data=arr_with_unmasked_nodata, transform=transform, crs=None, nodata=nodata)
                r2 = gu.Raster.from_array(
                    data=np.ma.masked_array(arr_with_unmasked_nodata), transform=transform, crs=None, nodata=nodata
                )
                r3 = gu.Raster.from_array(
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
        rst = gu.Raster.from_array(data=arr, transform=transform, crs=None, nodata=nodata)
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

        # Last part: Check the replacing of nodata values that were unmasked

        # Check that feeding an array with a nodata value unmasked raises the warning and corrects it
        nodata = 126
        ma = np.ma.masked_array(arr, mask=mask)
        ma.data[0, 0] = nodata
        ma.mask[0, 0] = False
        with pytest.warns(UserWarning, match="Unmasked values equal to the nodata value found in data array.*"):
            # Issue from user when creating the array
            raster = gu.Raster.from_array(data=ma, transform=transform, crs=None, nodata=nodata)
        assert raster.data.mask[0, 0]

        # Check that it can happen during a numerical operation
        ma = np.ma.masked_array(arr, mask=mask)
        ma.data[0, 0] = 125
        ma.mask[0, 0] = False
        raster = gu.Raster.from_array(data=ma, transform=transform, crs=None, nodata=126)
        with pytest.warns(UserWarning, match="Unmasked values equal to the nodata value found in data array.*"):
            # Issue during numerical operation
            new_raster = raster + 1
        assert new_raster.data.mask[0, 0]

    @pytest.mark.parametrize("example", [aster_dem_path, landsat_b4_path, landsat_rgb_path])  # type: ignore
    def test_get_nanarray(self, example: str) -> None:
        """
        Check that self.get_nanarray behaves as expected for examples with invalid data or not, and with several bands
        or a single one.
        """

        # -- First, we test without returning a mask --

        # Get nanarray
        rst = gu.Raster(example)
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
        assert rst.raster_equal(rst_copy)

        # -- Then, we test with a mask returned --
        rst_arr, mask = rst.get_nanarray(return_mask=True)

        assert np.array_equal(mask, np.ma.getmaskarray(rst.data).squeeze())

        # Also check for back-propagation here with the mask and array
        rst_arr += 5
        mask = ~mask
        assert rst.raster_equal(rst_copy)

    @pytest.mark.parametrize("example", [aster_dem_path, landsat_b4_path, landsat_rgb_path])  # type: ignore
    def test_downsampling(self, example: str) -> None:
        """
        Check that self metadata are correctly updated when using downsampling
        """
        # Load raster at full resolution
        rst_orig = gu.Raster(example)
        bounds_orig = rst_orig.bounds

        # -- Tries various downsampling factors to ensure rounding is needed in at least one case --
        for down_fact in np.arange(2, 8):
            rst_down = gu.Raster(example, downsample=int(down_fact))

            # - Check that output resolution is as intended -
            assert rst_down.res[0] == rst_orig.res[0] * down_fact
            assert rst_down.res[1] == rst_orig.res[1] * down_fact

            # - Check that downsampled width and height are as intended -
            # Due to rounding, width/height can be up to 1 pixel larger than unrounded value
            assert abs(rst_down.width - rst_orig.width / down_fact) < 1
            assert abs(rst_down.height - rst_orig.height / down_fact) < 1
            assert rst_down.shape == (rst_down.height, rst_down.width)

            # - Check that bounds are updated accordingly -
            # left/top bounds should be the same, right/bottom should be rounded to nearest pixel
            bounds_down = rst_down.bounds
            assert bounds_down.left == bounds_orig.left
            assert bounds_down.top == bounds_orig.top
            assert abs(bounds_down.right - bounds_orig.right) < rst_down.res[0]
            assert abs(bounds_down.bottom - bounds_orig.bottom) < rst_down.res[1]

            # - Check that metadata are consistent, with/out loading data -
            assert not rst_down.is_loaded
            width_unload = rst_down.width
            height_unload = rst_down.height
            bounds_unload = rst_down.bounds
            rst_down.load()
            width_load = rst_down.width
            height_load = rst_down.height
            bounds_load = rst_down.bounds
            assert width_load == width_unload
            assert height_load == height_unload
            assert bounds_load == bounds_unload

            # - Test that xy2ij are consistent with new image -
            # Upper left
            assert rst_down.xy2ij(rst_down.bounds.left, rst_down.bounds.top) == (0, 0)
            # Upper right
            assert rst_down.xy2ij(rst_down.bounds.right + rst_down.res[0], rst_down.bounds.top) == (
                0,
                rst_down.width + 1,
            )
            # Bottom right
            assert rst_down.xy2ij(rst_down.bounds.right + rst_down.res[0], rst_down.bounds.bottom) == (
                rst_down.height,
                rst_down.width + 1,
            )
            # One pixel right and down
            assert rst_down.xy2ij(rst_down.bounds.left + rst_down.res[0], rst_down.bounds.top - rst_down.res[1]) == (
                1,
                1,
            )

        # -- Check that error is raised when downsampling value is not valid --
        with pytest.raises(TypeError, match="downsample must be of type int or float."):
            gu.Raster(example, downsample=[1, 1])  # type: ignore

    def test_add_sub(self) -> None:
        """
        Test addition, subtraction and negation on a Raster object.
        """
        # Create fake rasters with random values in 0-255 and dtype uint8
        width = height = 5
        transform = rio.transform.from_bounds(0, 0, 1, 1, width, height)
        r1 = gu.Raster.from_array(
            np.random.randint(0, 255, (height, width), dtype="uint8"), transform=transform, crs=None
        )
        r2 = gu.Raster.from_array(
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
        r1 = gu.Raster.from_array(
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
        r1 = gu.Raster.from_array(
            np.random.randint(0, 255, (height + 1, width)).astype("float32"), transform=transform, crs=None
        )
        expected_message = "Both rasters must have the same shape, transform and CRS."
        with pytest.raises(ValueError, match=expected_message):
            r1.__add__(r2)

        with pytest.raises(ValueError, match=expected_message):
            r1.__sub__(r2)

        # different CRS
        r1 = gu.Raster.from_array(
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
        r1 = gu.Raster.from_array(
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
        r = gu.Raster(example)
        r.data += 5
        r2 = r.copy()

        # Objects should be different (not pointing to the same memory)
        assert r is not r2

        # Check the object is a Raster
        assert isinstance(r2, gu.Raster)

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
        assert np.array_equal(r.data, r2.data, equal_nan=True)

        # Check dataset_mask array
        assert np.array_equal(r.data.mask, r2.data.mask)

        # -- Third test: if r.data is modified, it does not affect r2.data --
        r.data += 5
        assert not np.array_equal(r.data.data, r2.data.data, equal_nan=True)

        # -- Fourth test: check the new array parameter works with either ndarray filled with NaNs, or masked arrays --

        # First, we pass the new array as the masked array, mask and data of the new Raster object should be identical
        r2 = r.copy(new_array=r.data)
        assert r.raster_equal(r2)

        # When passing the new array as a NaN ndarray, only the valid data is equal, because masked data is NaN in one
        # case, and -9999 in the other
        r_arr = gu.raster.get_array_and_mask(r)[0]
        r2 = r.copy(new_array=r_arr)
        assert np.ma.allequal(r.data, r2.data)
        # If a nodata value exists, and we update the NaN pixels to be that nodata value, then the two Rasters should
        # be perfectly equal
        if r2.nodata is not None:
            r2.data.data[np.isnan(r2.data.data)] = r2.nodata
        assert r.raster_equal(r2)

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
        r = gu.Raster(example)
        assert not r.is_modified

        # This should not trigger the hash
        r.data = r.data + 0
        assert not r.is_modified

        # This one neither
        r.data += 0
        assert not r.is_modified

        # This will
        r = gu.Raster(example)
        r.data = r.data + 5
        assert r.is_modified

    @pytest.mark.parametrize("example", [landsat_b4_path, landsat_rgb_path, aster_dem_path])  # type: ignore
    def test_masking(self, example: str) -> None:
        """
        Test self.set_mask
        """
        # Test boolean mask
        r = gu.Raster(example)
        # We need to know the existing nodata in case they exist, as set_mask only masks new values
        orig_mask = r.data.mask.copy()
        mask = r.data.data == np.nanmin(r.data)
        r.set_mask(mask)
        assert (np.count_nonzero(mask) > 0) & np.array_equal(orig_mask | mask > 0, r.data.mask)

        #  Test mask object
        r2 = gu.Raster(example)
        mask2 = r2.data == np.nanmin(r2)
        r2.set_mask(mask2)
        # Indexing at 0 for the mask in case the data has multiple bands
        assert (np.count_nonzero(mask2) > 0) & np.array_equal(orig_mask | mask2.filled(False), r2.data.mask)
        # The two last masking (array or Mask) should yield the same result when the data is only 2D
        if r.count == 1:
            assert np.array_equal(r.data.mask, r2.data.mask)

        # Test non-boolean mask with values > 0
        r = gu.Raster(example)
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
        r = gu.Raster(example)
        if r.data.shape[0] == 1:
            mask = (r.data.data == np.min(r.data.data)).squeeze()
            r.set_mask(mask)
            assert (np.count_nonzero(mask) > 0) & np.array_equal(mask > 0, r.data.mask.squeeze())

        # Test that proper issue is raised if shape is incorrect
        r = gu.Raster(example)
        wrong_shape = np.array(r.data.shape) + 1
        mask = np.zeros(wrong_shape)
        with pytest.raises(ValueError, match="mask must be of the same shape as existing data*"):
            r.set_mask(mask)

        # Test that proper issue is raised if mask is not a numpy array
        with pytest.raises(ValueError, match="mask must be a numpy array"):
            r.set_mask(1)

    @pytest.mark.parametrize("example", [landsat_b4_path, landsat_rgb_path, aster_dem_path])  # type: ignore
    def test_getitem_setitem(self, example: str) -> None:
        """Test the __getitem__ method ([]) for indexing and __setitem__ for index assignment."""

        # -- First, we test mask or boolean array indexing and assignment, specific to rasters --

        # Open a Raster
        rst = gu.Raster(example)

        # Create a boolean array of the same shape, and a mask of the same transform/crs
        arr = np.random.randint(low=0, high=2, size=rst.shape, dtype=bool)
        mask = gu.Mask.from_array(data=arr, transform=rst.transform, crs=rst.crs)

        # Check that indexing works with both of those
        vals_arr = rst[arr]
        vals_mask = rst[mask]

        # Those indexing operations should yield the same 1D array of values
        assert np.array_equal(vals_mask, vals_arr, equal_nan=True)

        # Now, we test index assignment
        rst2 = rst.copy()

        # It should work with a number, or a 1D array of the same length as the indexed one
        rst[mask] = 1.0
        rst2[mask] = np.ones(rst2.shape)[arr]

        # The rasters should be the same
        assert rst2.raster_equal(rst)

        # -- Second, we test NumPy indexes (slices, integers, ellipses, new axes) --

        # Indexing
        assert np.ma.allequal(rst[0], rst.data[0])  # Test an integer
        assert np.ma.allequal(rst[0:10], rst.data[0:10])  # Test a slice
        # New axis adds a dimension, but the 0 index reduces one, so we still get a 2D raster
        assert np.ma.allequal(rst[np.newaxis, 0], rst.data[np.newaxis, 0])  # Test a new axis
        assert np.ma.allequal(rst[...], rst.data[...])  # Test an ellipsis

        # Index assignment
        rst[0] = 1
        assert np.ma.allequal(rst.data[0], np.ones(np.shape(rst.data[0])))  # Test an integer
        rst[0:10] = 1
        assert np.ma.allequal(rst.data[0:10], np.ones(np.shape(rst.data[0:10])))  # Test a slice
        # Same as above for the new axis
        rst[0, np.newaxis] = 1
        assert np.ma.allequal(rst.data[0, np.newaxis], np.ones(np.shape(rst.data[0, np.newaxis])))  # Test a new axis
        rst[...] = 1
        assert np.ma.allequal(rst.data[...], np.ones(np.shape(rst.data[...])))  # Test an ellipsis

        # -- Finally, we check that errors are raised for both indexing and index assignment --

        # For indexing
        op_name_index = "an indexing operation"
        op_name_assign = "an index assignment operation"
        message_raster = (
            "Both rasters must have the same shape, transform and CRS for {}. "
            "For example, use raster1 = raster1.reproject(raster2) to reproject raster1 on the "
            "same grid and CRS than raster2."
        )
        message_array = (
            "The raster and array must have the same shape for {}. "
            "For example, if the array comes from another raster, use raster1 = "
            "raster1.reproject(raster2) beforehand to reproject raster1 on the same grid and CRS "
            "than raster2. Or, if the array does not come from a raster, define one with raster = "
            "Raster.from_array(array, array_transform, array_crs, array_nodata) then reproject."
        )

        # An error when the shape is wrong
        with pytest.raises(ValueError, match=re.escape(message_array.format(op_name_index))):
            rst[arr[:-1, :-1]]

        # An error when the georeferencing of the Mask does not match
        mask.shift(1, 1, inplace=True)
        with pytest.raises(ValueError, match=re.escape(message_raster.format(op_name_index))):
            rst[mask]

        # A warning when the array type is not boolean
        with pytest.warns(UserWarning, match="Input array was cast to boolean for indexing."):
            rst[arr.astype("uint8")]
            rst[arr.astype("uint8")] = 1

        # For index assignment
        # An error when the shape is wrong
        with pytest.raises(ValueError, match=re.escape(message_array.format(op_name_assign))):
            rst[arr[:-1, :-1]] = 1

        with pytest.raises(ValueError, match=re.escape(message_raster.format(op_name_assign))):
            rst[mask] = 1

    test_data = [[landsat_b4_path, everest_outlines_path], [aster_dem_path, aster_outlines_path]]

    @pytest.mark.parametrize("data", test_data)  # type: ignore
    def test_crop(self, data: list[str]) -> None:
        """Test for crop method, also called by square brackets through __getitem__"""

        raster_path, outlines_path = data
        r = gu.Raster(raster_path)

        # -- Test with crop_geom being a list/tuple -- ##
        crop_geom: list[float] = list(r.bounds)

        # Test unloaded inplace cropping conserves the shape
        r.crop(crop_geom=[crop_geom[0] + r.res[0], crop_geom[1], crop_geom[2], crop_geom[3]], inplace=True)
        assert len(r.data.shape) == 2

        r = gu.Raster(raster_path)

        # Test with same bounds -> should be the same #
        crop_geom2 = [crop_geom[0], crop_geom[1], crop_geom[2], crop_geom[3]]
        r_cropped = r.crop(crop_geom2)
        assert r_cropped.raster_equal(r)

        # - Test cropping each side by a random integer of pixels - #
        rand_int = np.random.randint(1, min(r.shape) - 1)

        # Left
        crop_geom2 = [crop_geom[0] + rand_int * r.res[0], crop_geom[1], crop_geom[2], crop_geom[3]]
        r_cropped = r.crop(crop_geom2)
        assert list(r_cropped.bounds) == crop_geom2
        assert np.array_equal(r.data[:, rand_int:].data, r_cropped.data.data, equal_nan=True)
        assert np.array_equal(r.data[:, rand_int:].mask, r_cropped.data.mask)

        # Right
        crop_geom2 = [crop_geom[0], crop_geom[1], crop_geom[2] - rand_int * r.res[0], crop_geom[3]]
        r_cropped = r.crop(crop_geom2)
        assert list(r_cropped.bounds) == crop_geom2
        assert np.array_equal(r.data[:, :-rand_int].data, r_cropped.data.data, equal_nan=True)
        assert np.array_equal(r.data[:, :-rand_int].mask, r_cropped.data.mask)

        # Bottom
        crop_geom2 = [crop_geom[0], crop_geom[1] + rand_int * abs(r.res[1]), crop_geom[2], crop_geom[3]]
        r_cropped = r.crop(crop_geom2)
        assert list(r_cropped.bounds) == crop_geom2
        assert np.array_equal(r.data[:-rand_int, :].data, r_cropped.data.data, equal_nan=True)
        assert np.array_equal(r.data[:-rand_int, :].mask, r_cropped.data.mask)

        # Top
        crop_geom2 = [crop_geom[0], crop_geom[1], crop_geom[2], crop_geom[3] - rand_int * abs(r.res[1])]
        r_cropped = r.crop(crop_geom2)
        assert list(r_cropped.bounds) == crop_geom2
        assert np.array_equal(r.data[rand_int:, :].data, r_cropped.data, equal_nan=True)
        assert np.array_equal(r.data[rand_int:, :].mask, r_cropped.data.mask)

        # Same but tuple
        crop_geom3: tuple[float, float, float, float] = (
            crop_geom[0],
            crop_geom[1],
            crop_geom[2],
            crop_geom[3] - rand_int * r.res[0],
        )
        r_cropped = r.crop(crop_geom3)
        assert list(r_cropped.bounds) == list(crop_geom3)
        assert np.array_equal(r.data[rand_int:, :].data, r_cropped.data.data, equal_nan=True)
        assert np.array_equal(r.data[rand_int:, :].mask, r_cropped.data.mask)

        # -- Test with crop_geom being a Raster -- #
        r_cropped2 = r.crop(r_cropped)
        assert r_cropped2.raster_equal(r_cropped)

        # Check that bound reprojection is done automatically if the CRS differ
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="For reprojection, nodata must be set.*")

            r_cropped_reproj = r_cropped.reproject(crs=3857)
            r_cropped3 = r.crop(r_cropped_reproj)

        # Original CRS bounds can be deformed during transformation, but result should be equivalent to this
        r_cropped4 = r.crop(crop_geom=r_cropped_reproj.get_bounds_projected(out_crs=r.crs))
        assert r_cropped3.raster_equal(r_cropped4)

        # -- Test with inplace=True -- #
        r_copy = r.copy()
        r_copy.crop(r_cropped, inplace=True)
        assert r_copy.raster_equal(r_cropped)

        # - Test cropping each side with a non integer pixel, mode='match_pixel' - #
        rand_float = np.random.randint(1, min(r.shape) - 1) + 0.25

        # left
        crop_geom2 = [crop_geom[0] + rand_float * r.res[0], crop_geom[1], crop_geom[2], crop_geom[3]]
        r_cropped = r.crop(crop_geom2)
        assert r.shape[1] - (r_cropped.bounds.right - r_cropped.bounds.left) / r.res[0] == int(rand_float)
        assert np.array_equal(r.data[:, int(rand_float) :].data, r_cropped.data.data, equal_nan=True)
        assert np.array_equal(r.data[:, int(rand_float) :].mask, r_cropped.data.mask)

        # right
        crop_geom2 = [crop_geom[0], crop_geom[1], crop_geom[2] - rand_float * r.res[0], crop_geom[3]]
        r_cropped = r.crop(crop_geom2)
        assert r.shape[1] - (r_cropped.bounds.right - r_cropped.bounds.left) / r.res[0] == int(rand_float)
        assert np.array_equal(r.data[:, : -int(rand_float)].data, r_cropped.data.data, equal_nan=True)
        assert np.array_equal(r.data[:, : -int(rand_float)].mask, r_cropped.data.mask)

        # bottom
        crop_geom2 = [crop_geom[0], crop_geom[1] + rand_float * abs(r.res[1]), crop_geom[2], crop_geom[3]]
        r_cropped = r.crop(crop_geom2)
        assert r.shape[0] - (r_cropped.bounds.top - r_cropped.bounds.bottom) / r.res[1] == int(rand_float)
        assert np.array_equal(r.data[: -int(rand_float), :].data, r_cropped.data.data, equal_nan=True)
        assert np.array_equal(r.data[: -int(rand_float), :].mask, r_cropped.data.mask)

        # top
        crop_geom2 = [crop_geom[0], crop_geom[1], crop_geom[2], crop_geom[3] - rand_float * abs(r.res[1])]
        r_cropped = r.crop(crop_geom2)
        assert r.shape[0] - (r_cropped.bounds.top - r_cropped.bounds.bottom) / r.res[1] == int(rand_float)
        assert np.array_equal(r.data[int(rand_float) :, :].data, r_cropped.data.data, equal_nan=True)
        assert np.array_equal(r.data[int(rand_float) :, :].mask, r_cropped.data.mask)

        # -- Test with mode='match_extent' -- #
        # Test all sides at once, with rand_float less than half the smallest extent
        # The cropped extent should exactly match the requested extent, res will be changed accordingly
        rand_float = np.random.randint(1, min(r.shape) / 2 - 1) + 0.25
        crop_geom2 = [
            crop_geom[0] + rand_float * r.res[0],
            crop_geom[1] + rand_float * abs(r.res[1]),
            crop_geom[2] - rand_float * r.res[0],
            crop_geom[3] - rand_float * abs(r.res[1]),
        ]

        # Filter warning about nodata not set in reprojection (because match_extent triggers reproject)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="For reprojection, nodata must be set.*")
            r_cropped = r.crop(crop_geom2, mode="match_extent")

        assert list(r_cropped.bounds) == crop_geom2
        # The change in resolution should be less than what would occur with +/- 1 pixel
        assert np.all(
            abs(np.array(r.res) - np.array(r_cropped.res)) < np.array(r.res) / np.array(r_cropped.shape)[::-1]
        )

        # Filter warning about nodata not set in reprojection (because match_extent triggers reproject)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="For reprojection, nodata must be set.*")
            r_cropped2 = r.crop(r_cropped, mode="match_extent")
        assert r_cropped2.raster_equal(r_cropped)

        # -- Test with crop_geom being a Vector -- #
        outlines = gu.Vector(outlines_path)

        # First, we reproject manually the outline
        outlines_reproj = gu.Vector(outlines.ds.to_crs(r.crs))
        r_cropped = r.crop(outlines_reproj)

        # Calculate intersection of the two bounding boxes and make sure crop has same bounds
        win_outlines = rio.windows.from_bounds(*outlines_reproj.bounds, transform=r.transform)
        win_raster = rio.windows.from_bounds(*r.bounds, transform=r.transform)
        final_window = win_outlines.intersection(win_raster).round_lengths().round_offsets()
        new_bounds = rio.windows.bounds(final_window, transform=r.transform)
        assert list(r_cropped.bounds) == list(new_bounds)

        # Second, we check that bound reprojection is done automatically if the CRS differ
        r_cropped2 = r.crop(outlines)
        assert list(r_cropped2.bounds) == list(new_bounds)

        # -- Test crop works as expected even if transform has been modified, e.g. through downsampling -- #
        # Test that with downsampling, cropping to same bounds result in same raster
        r = gu.Raster(raster_path, downsample=5)
        r_test = r.crop(r.bounds)
        assert r_test.raster_equal(r)

        # - Test that cropping yields the same results whether data is loaded or not -
        # With integer cropping (left)
        rand_int = np.random.randint(1, min(r.shape) - 1)
        crop_geom2 = [crop_geom[0] + rand_int * r.res[0], crop_geom[1], crop_geom[2], crop_geom[3]]
        r = gu.Raster(raster_path, downsample=5, load_data=False)
        assert not r.is_loaded
        r_crop_unloaded = r.crop(crop_geom2)
        r.load()
        r_crop_loaded = r.crop(crop_geom2)
        # TODO: the following condition should be met once issue #447 is solved
        # assert r_crop_unloaded.raster_equal(r_crop_loaded)
        assert r_crop_unloaded.shape == r_crop_loaded.shape
        assert r_crop_unloaded.transform == r_crop_loaded.transform

        # With a float number of pixels added to the right, mode 'match_pixel'
        rand_float = np.random.randint(1, min(r.shape) - 1) + 0.25
        crop_geom2 = [crop_geom[0], crop_geom[1], crop_geom[2] + rand_float * r.res[0], crop_geom[3]]
        r = gu.Raster(raster_path, downsample=5, load_data=False)
        assert not r.is_loaded
        r_crop_unloaded = r.crop(crop_geom2, mode="match_pixel")
        r.load()
        r_crop_loaded = r.crop(crop_geom2, mode="match_pixel")
        # TODO: the following condition should be met once issue #447 is solved
        # assert r_crop_unloaded.raster_equal(r_crop_loaded)
        assert r_crop_unloaded.shape == r_crop_loaded.shape
        assert r_crop_unloaded.transform == r_crop_loaded.transform

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path, landsat_rgb_path])  # type: ignore
    def test_shift(self, example: str) -> None:
        """Tests shift works as intended"""

        r = gu.Raster(example)

        # Get original transform
        orig_transform = r.transform
        orig_bounds = r.bounds

        # Shift raster by georeferenced units (default)
        # Check the default behaviour is not inplace
        r_notinplace = r.shift(xoff=1, yoff=1)
        assert isinstance(r_notinplace, gu.Raster)

        # Check inplace
        r.shift(xoff=1, yoff=1, inplace=True)
        # Both shifts should have yielded the same transform
        assert r.transform == r_notinplace.transform

        # Only bounds should change
        assert orig_transform.c + 1 == r.transform.c
        assert orig_transform.f + 1 == r.transform.f
        for attr in ["a", "b", "d", "e"]:
            assert getattr(orig_transform, attr) == getattr(r.transform, attr)

        assert orig_bounds.left + 1 == r.bounds.left
        assert orig_bounds.right + 1 == r.bounds.right
        assert orig_bounds.bottom + 1 == r.bounds.bottom
        assert orig_bounds.top + 1 == r.bounds.top

        # Shift raster using pixel units
        orig_transform = r.transform
        orig_bounds = r.bounds
        orig_res = r.res
        r.shift(xoff=1, yoff=1, distance_unit="pixel", inplace=True)

        # Only bounds should change
        assert orig_transform.c + 1 * orig_res[0] == r.transform.c
        assert orig_transform.f + 1 * orig_res[1] == r.transform.f
        for attr in ["a", "b", "d", "e"]:
            assert getattr(orig_transform, attr) == getattr(r.transform, attr)

        assert orig_bounds.left + 1 * orig_res[0] == r.bounds.left
        assert orig_bounds.right + 1 * orig_res[0] == r.bounds.right
        assert orig_bounds.bottom + 1 * orig_res[1] == r.bounds.bottom
        assert orig_bounds.top + 1 * orig_res[1] == r.bounds.top

        # Check that an error is raised for a wrong distance_unit
        with pytest.raises(ValueError, match="Argument 'distance_unit' should be either 'pixel' or 'georeferenced'."):
            r.shift(xoff=1, yoff=1, distance_unit="wrong_value")  # type: ignore

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_reproject(self, example: str) -> None:
        warnings.simplefilter("error")

        # Reference raster to be used
        r = gu.Raster(example)

        # -- Check proper errors are raised if nodata are not set -- #
        r_nodata = r.copy()
        r_nodata.set_nodata(None)

        # Make sure at least one pixel is masked for test 1
        rand_indices = gu.raster.subsample_array(r_nodata.data, 10, return_indices=True)
        r_nodata.data[rand_indices] = np.ma.masked
        assert np.count_nonzero(r_nodata.data.mask) > 0

        # make sure at least one pixel is set at default nodata for test
        default_nodata = _default_nodata(r_nodata.dtypes[0])
        rand_indices = gu.raster.subsample_array(r_nodata.data, 10, return_indices=True)
        r_nodata.data[rand_indices] = default_nodata
        assert np.count_nonzero(r_nodata.data == default_nodata) > 0

        # 1 - if no force_source_nodata is set and masked values exist, raises an error
        with pytest.raises(
            ValueError,
            match=re.escape(
                "No nodata set, set one for the raster with self.set_nodata() or use a "
                "temporary one with `force_source_nodata`."
            ),
        ):
            _ = r_nodata.reproject(res=r_nodata.res[0] / 2, nodata=0)

        # 2 - if no nodata is set and default value conflicts with existing value, a warning is raised
        with pytest.warns(
            UserWarning,
            match=re.escape(
                f"For reprojection, nodata must be set. Default chosen value "
                f"{_default_nodata(r_nodata.dtypes[0])} exists in self.data. This may have unexpected "
                f"consequences. Consider setting a different nodata with self.set_nodata()."
            ),
        ):
            r_test = r_nodata.reproject(res=r_nodata.res[0] / 2, force_source_nodata=default_nodata)
        assert r_test.nodata == default_nodata

        # 3 - if default nodata does not conflict, should not raise a warning
        r_nodata.data[r_nodata.data == default_nodata] = 3
        r_test = r_nodata.reproject(res=r_nodata.res[0] / 2, force_source_nodata=default_nodata)
        assert r_test.nodata == default_nodata

        # -- Test setting each combination of georeferences bounds, res and size -- #

        # specific for the landsat test case, default nodata 255 cannot be used (see above), so use 0
        if r.nodata is None:
            r.set_nodata(0)

        # - Test size - this should modify the shape, and hence resolution, but not the bounds -
        out_size = (r.shape[1] // 2, r.shape[0] // 2)  # Outsize is (ncol, nrow)
        r_test = r.reproject(grid_size=out_size)
        assert r_test.shape == (out_size[1], out_size[0])
        assert r_test.res != r.res
        assert r_test.bounds == r.bounds

        # - Test bounds -
        # if bounds is a multiple of res, outptut res should be preserved
        bounds = np.copy(r.bounds)
        dst_bounds = rio.coords.BoundingBox(
            left=bounds[0], bottom=bounds[1] + r.res[0], right=bounds[2] - 2 * r.res[1], top=bounds[3]
        )
        r_test = r.reproject(bounds=dst_bounds)
        assert r_test.bounds == dst_bounds
        assert r_test.res == r.res

        # Create bounds with 1/2 and 1/3 pixel extra on the right/bottom.
        bounds = np.copy(r.bounds)
        dst_bounds = rio.coords.BoundingBox(
            left=bounds[0], bottom=bounds[1] - r.res[0] / 3.0, right=bounds[2] + r.res[1] / 2.0, top=bounds[3]
        )

        # If bounds are not a multiple of res, the latter will be updated accordingly
        r_test = r.reproject(bounds=dst_bounds)
        assert r_test.bounds == dst_bounds
        assert r_test.res != r.res

        # - Test size and bounds -
        r_test = r.reproject(grid_size=out_size, bounds=dst_bounds)
        assert r_test.shape == (out_size[1], out_size[0])
        assert r_test.bounds == dst_bounds

        # - Test res -
        # Using a single value, output res will be enforced, resolution will be different
        res_single = r.res[0] * 2
        r_test = r.reproject(res=res_single)
        assert r_test.res == (res_single, res_single)
        assert r_test.shape != r.shape

        # Using a tuple
        res_tuple = (r.res[0] * 0.5, r.res[1] * 4)
        r_test = r.reproject(res=res_tuple)
        assert r_test.res == res_tuple
        assert r_test.shape != r.shape

        # - Test res and bounds -
        # Bounds will be enforced for upper-left pixel, but adjusted by up to one pixel for the lower right bound.
        # for single res value
        r_test = r.reproject(bounds=dst_bounds, res=res_single)
        assert r_test.res == (res_single, res_single)
        assert r_test.bounds.left == dst_bounds.left
        assert r_test.bounds.top == dst_bounds.top
        assert np.abs(r_test.bounds.right - dst_bounds.right) < res_single
        assert np.abs(r_test.bounds.bottom - dst_bounds.bottom) < res_single

        # For tuple
        r_test = r.reproject(bounds=dst_bounds, res=res_tuple)
        assert r_test.res == res_tuple
        assert r_test.bounds.left == dst_bounds.left
        assert r_test.bounds.top == dst_bounds.top
        assert np.abs(r_test.bounds.right - dst_bounds.right) < res_tuple[0]
        assert np.abs(r_test.bounds.bottom - dst_bounds.bottom) < res_tuple[1]

        # - Test crs -
        out_crs = rio.crs.CRS.from_epsg(4326)
        r_test = r.reproject(crs=out_crs)
        assert r_test.crs.to_epsg() == 4326

        # -- Additional tests --
        # First, make sure dst_bounds extend beyond current extent to create nodata
        dst_bounds = rio.coords.BoundingBox(
            left=bounds[0], bottom=bounds[1] - r.res[0], right=bounds[2] + 2 * r.res[1], top=bounds[3]
        )
        r_test = r.reproject(bounds=dst_bounds)
        assert np.count_nonzero(r_test.data.mask) > 0

        # If nodata falls outside the original image range, check range is preserved (with nearest interpolation)
        r_float = r.astype("float32")  # type: ignore
        if (r_float.nodata < np.min(r_float)) or (r_float.nodata > np.max(r_float)):
            r_test = r_float.reproject(bounds=dst_bounds, resampling="nearest")
            assert r_test.nodata == r_float.nodata
            assert np.count_nonzero(r_test.data.data == r_test.nodata) > 0  # Some values should be set to nodata
            assert np.min(r_test.data) == np.min(r_float.data)  # But min and max should not be affected
            assert np.max(r_test.data) == np.max(r_float.data)

        # Check that nodata works as expected
        r_test = r_float.reproject(bounds=dst_bounds, nodata=9999)
        assert r_test.nodata == 9999
        assert np.count_nonzero(r_test.data.data == r_test.nodata) > 0

        # Test that reproject works the same whether data is already loaded or not
        assert r.is_loaded
        r_test1 = r.reproject(crs=out_crs, nodata=0)
        r_unload = gu.Raster(example, load_data=False)
        assert not r_unload.is_loaded
        r_test2 = r_unload.reproject(crs=out_crs, nodata=0)
        assert r_test1.raster_equal(r_test2)

        # Test that reproject does not fail with resolution as np.integer or np.float types, single value or tuple
        astype_funcs = [int, np.int32, float, np.float64]
        for astype_func in astype_funcs:
            r.reproject(res=astype_func(20.5), nodata=0)
        for i in range(len(astype_funcs)):
            for j in range(len(astype_funcs)):
                r.reproject(res=(astype_funcs[i](20.5), astype_funcs[j](10.5)), nodata=0)

        # Test that reprojection works for several bands
        for n in [2, 3, 4]:
            img1 = gu.Raster.from_array(
                np.ones((n, 500, 500), dtype="uint8"), transform=rio.transform.from_origin(0, 500, 1, 1), crs=4326
            )

            img2 = gu.Raster.from_array(
                np.ones((n, 500, 500), dtype="uint8"), transform=rio.transform.from_origin(50, 500, 1, 1), crs=4326
            )

            out_img = img2.reproject(img1)
            assert np.shape(out_img.data) == (n, 500, 500)
            assert (out_img.count, *out_img.shape) == (n, 500, 500)

        # Test that the rounding of resolution is correct for large decimal numbers
        # (we take an example that used to fail, see issue #354 and #357)
        data = np.ones((4759, 2453))
        transform = rio.transform.Affine(
            24.12423878332849, 0.0, 238286.29553975424, 0.0, -24.12423878332849, 6995453.456051373
        )
        crs = rio.CRS.from_epsg(32633)
        nodata = -9999.0
        rst = gu.Raster.from_array(data=data, transform=transform, crs=crs, nodata=nodata)

        rst_reproj = rst.reproject(bounds=rst.bounds, res=(20.0, 20.0))
        # This used to be 19.999999999999999 due to floating point precision
        assert rst_reproj.res == (20.0, 20.0)

        # -- Test match reference functionalities --

        # - Create 2 artificial rasters -
        # for r2b, bounds are cropped to the upper left by an integer number of pixels (i.e. crop)
        # for r2, resolution is also set to 2/3 the input res
        min_size = min(r.shape)
        rand_int = np.random.randint(min_size / 10, min(r.shape) - min_size / 10)
        new_transform = rio.transform.from_origin(
            r.bounds.left + rand_int * r.res[0], r.bounds.top - rand_int * abs(r.res[1]), r.res[0], r.res[1]
        )

        # data is cropped to the same extent
        new_data = r.data[rand_int::, rand_int::]
        r2b = gu.Raster.from_array(data=new_data, transform=new_transform, crs=r.crs, nodata=r.nodata)

        # Create a raster with different resolution
        dst_res = r.res[0] * 2 / 3
        r2 = r2b.reproject(res=dst_res)
        assert r2.res == (dst_res, dst_res)

        # Assert the initial rasters are different
        assert r.bounds != r2b.bounds
        assert r.shape != r2b.shape
        assert r.bounds != r2.bounds
        assert r.shape != r2.shape
        assert r.res != r2.res

        # Test reprojecting with ref=r2b (i.e. crop) -> output should have same shape, bounds and data, i.e. be the
        # same object
        r3 = r.reproject(r2b)
        assert r3.bounds == r2b.bounds
        assert r3.shape == r2b.shape
        assert r3.bounds == r2b.bounds
        assert r3.transform == r2b.transform
        assert np.array_equal(r3.data.data, r2b.data.data, equal_nan=True)
        assert np.array_equal(r3.data.mask, r2b.data.mask)

        if DO_PLOT:
            fig1, ax1 = plt.subplots()
            r.plot(ax=ax1, title="Raster 1")

            fig2, ax2 = plt.subplots()
            r2b.plot(ax=ax2, title="Raster 2")

            fig3, ax3 = plt.subplots()
            r3.plot(ax=ax3, title="Raster 1 reprojected to Raster 2")

            plt.show()

        # Test reprojecting with ref=r2 -> output should have same shape, bounds and transform
        # Data should be slightly different due to difference in input resolution
        r3 = r.reproject(r2)
        assert r3.bounds == r2.bounds
        assert r3.shape == r2.shape
        assert r3.bounds == r2.bounds
        assert r3.transform == r2.transform
        assert not np.array_equal(r3.data.data, r2.data.data, equal_nan=True)

        if DO_PLOT:
            fig1, ax1 = plt.subplots()
            r.plot(ax=ax1, title="Raster 1")

            fig2, ax2 = plt.subplots()
            r2.plot(ax=ax2, title="Raster 2")

            fig3, ax3 = plt.subplots()
            r3.plot(ax=ax3, title="Raster 1 reprojected to Raster 2")

            plt.show()

        # -- Check that if mask is modified afterwards, it is taken into account during reproject -- #
        # Create a raster with (additional) random gaps
        r_gaps = r.copy()
        nsamples = 200
        rand_indices = gu.raster.subsample_array(r_gaps.data, nsamples, return_indices=True)
        r_gaps.data[rand_indices] = np.ma.masked
        assert np.sum(r_gaps.data.mask) - np.sum(r.data.mask) == nsamples  # sanity check

        # reproject raster, and reproject mask. Check that both have same number of masked pixels
        # TODO: should test other resampling algo
        r_gaps_reproj = r_gaps.reproject(res=dst_res, resampling="nearest")
        mask = gu.Raster.from_array(
            r_gaps.data.mask.astype("uint8"), crs=r_gaps.crs, transform=r_gaps.transform, nodata=None
        )
        mask_reproj = mask.reproject(res=dst_res, nodata=255, resampling="nearest")
        # Final masked pixels are those originally masked (=1) and the values masked during reproject, e.g. edges
        tot_masked_true = np.count_nonzero(mask_reproj.data.mask) + np.count_nonzero(mask_reproj.data == 1)
        assert np.count_nonzero(r_gaps_reproj.data.mask) == tot_masked_true

        # If a nodata is set, make sure it is preserved
        r_nodata = r.copy()

        r_nodata.set_nodata(0)

        r3 = r_nodata.reproject(r2)
        assert r_nodata.nodata == r3.nodata

        # -- Check inplace behaviour works -- #

        # Check when transform is updated (via res)
        r_tmp_res = r.copy()
        r_res = r_tmp_res.reproject(res=r.res[0] / 2)
        r_tmp_res.reproject(res=r.res[0] / 2, inplace=True)

        assert r_res.raster_equal(r_tmp_res)

        # Check when CRS is updated
        r_tmp_crs = r.copy()
        r_crs = r_tmp_crs.reproject(crs=out_crs)
        r_tmp_crs.reproject(crs=out_crs, inplace=True)

        assert r_crs.raster_equal(r_tmp_crs)

        # -- Test additional errors raised for argument combinations -- #

        # If both ref and crs are set
        with pytest.raises(ValueError, match=re.escape("Either of `ref` or `crs` must be set. Not both.")):
            _ = r.reproject(ref=r2, crs=r.crs)

        # Size and res are mutually exclusive
        with pytest.raises(ValueError, match=re.escape("size and res both specified. Specify only one.")):
            _ = r.reproject(grid_size=(10, 10), res=50)

        # If wrong type for `ref`
        with pytest.raises(
            TypeError, match=re.escape("Type of ref not understood, must be path to file (str), Raster.")
        ):
            _ = r.reproject(ref=3)

        # If input reference is string and file and does not exist
        with pytest.raises(ValueError, match=re.escape("Reference raster does not exist.")):
            _ = r.reproject(ref="no_file.tif")

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_intersection(self, example: list[str]) -> None:
        """Check the behaviour of the intersection function"""
        # Load input raster
        r = gu.Raster(example)
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

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path, landsat_rgb_path])  # type: ignore
    def test_ij2xy_xy2ij(self, example: str) -> None:
        """Test ij2xy and that the two functions are reversible."""

        # Open raster
        rst = gu.Raster(example)
        xmin, ymin, xmax, ymax = rst.bounds

        # Check ij2xy manually for the four corners

        # With offset="center", should be pixel center
        xmin_center = xmin + rst.res[0] / 2
        ymin_center = ymin + rst.res[1] / 2
        xmax_center = xmax - rst.res[0] / 2
        ymax_center = ymax - rst.res[1] / 2
        assert rst.ij2xy([0], [0], offset="center") == ([xmin_center], [ymax_center])
        assert rst.ij2xy([rst.shape[0] - 1], [0], offset="center") == ([xmin_center], [ymin_center])
        assert rst.ij2xy([0], [rst.shape[1] - 1], offset="center") == ([xmax_center], [ymax_center])
        assert rst.ij2xy([rst.shape[0] - 1], [rst.shape[1] - 1], offset="center") == ([xmax_center], [ymin_center])

        # With offset="ll", lower-left
        xmin_center = xmin
        ymin_center = ymin
        xmax_center = xmax - rst.res[0]
        ymax_center = ymax - rst.res[1]
        assert rst.ij2xy([0], [0], offset="ll") == ([xmin_center], [ymax_center])
        assert rst.ij2xy([rst.shape[0] - 1], [0], offset="ll") == ([xmin_center], [ymin_center])
        assert rst.ij2xy([0], [rst.shape[1] - 1], offset="ll") == ([xmax_center], [ymax_center])
        assert rst.ij2xy([rst.shape[0] - 1], [rst.shape[1] - 1], offset="ll") == ([xmax_center], [ymin_center])

        # We generate random points within the boundaries of the image
        xrand = np.random.randint(low=0, high=rst.width, size=(10,)) * list(rst.transform)[0] + xmin
        yrand = ymax + np.random.randint(low=0, high=rst.height, size=(10,)) * list(rst.transform)[4]

        # Test reversibility (only works with default upper-left offset)
        i, j = rst.xy2ij(xrand, yrand)
        xnew, ynew = rst.ij2xy(i, j)
        assert all(xnew == xrand)
        assert all(ynew == yrand)

        # TODO: clarify this weird behaviour of rasterio.index with floats?
        # r.ds.index(x, y)
        # Out[33]: (75, 301)
        # r.ds.index(x, y, op=np.float32)
        # Out[34]: (75.0, 302.0)

    def test_xy2ij_and_interp(self) -> None:
        """Test xy2ij with shift_area_or_point argument, and related interp function"""

        # First, we try on a Raster with a Point interpretation in its "AREA_OR_POINT" metadata: values interpolated
        # at the center of pixel
        r = gu.Raster(self.landsat_b4_path)
        assert r.tags["AREA_OR_POINT"] == "Point"
        xmin, ymin, xmax, ymax = r.bounds

        # We generate random points within the boundaries of the image
        xrand = np.random.randint(low=0, high=r.width, size=(10,)) * list(r.transform)[0] + xmin
        yrand = ymax + np.random.randint(low=0, high=r.height, size=(10,)) * list(r.transform)[4]
        pts = list(zip(xrand, yrand))

        # Get decimal indexes based on "Point", should refer to the corner still (shift False by default)
        i, j = r.xy2ij(xrand, yrand)
        assert np.all(i % 1 == 0)
        assert np.all(j % 1 == 0)

        # Those should all be .5 because values refer to the center and are shifted
        i, j = r.xy2ij(xrand, yrand, shift_area_or_point=True)
        assert np.all(i % 1 == 0.5)
        assert np.all(j % 1 == 0.5)

        # Force "Area", should refer to corner
        r.tags.update({"AREA_OR_POINT": "Area"})
        i, j = r.xy2ij(xrand, yrand, shift_area_or_point=True)
        assert np.all(i % 1 == 0)
        assert np.all(j % 1 == 0)

        # Check errors are raised when type of tag is incorrect
        r0 = r.copy()
        # When the tag is not a string
        r0.tags.update({"AREA_OR_POINT": 1})
        with pytest.raises(TypeError, match=re.escape('Attribute self.tags["AREA_OR_POINT"] must be a string.')):
            r0.xy2ij(xrand, yrand, shift_area_or_point=True)
        # When the tag is not "Area" or "Point"
        r0.tags.update({"AREA_OR_POINT": "Pt"})
        with pytest.raises(
            ValueError, match=re.escape('Attribute self.tags["AREA_OR_POINT"] must be one of "Area" or "Point".')
        ):
            r0.xy2ij(xrand, yrand, shift_area_or_point=True)

        # Check that the function warns when no tag is defined
        r0.tags.pop("AREA_OR_POINT")
        with pytest.warns(
            UserWarning,
            match=re.escape('Attribute AREA_OR_POINT undefined in self.tags, using "Area" as default (no shift).'),
        ):
            i0, j0 = r0.xy2ij(xrand, yrand, shift_area_or_point=True)
        # And that it defaults to "Area"
        assert all(i == i0)
        assert all(j == j0)

        # Now, we calculate the mean of values in each 2x2 slices of the data, and compare with interpolation at order 1
        list_z_ind = []
        img = r.data
        for k in range(len(xrand)):
            # 2x2 slices
            z_ind = np.mean(
                img[
                    slice(int(np.floor(i[k])), int(np.ceil(i[k])) + 1),
                    slice(int(np.floor(j[k])), int(np.ceil(j[k])) + 1),
                ]
            )
            list_z_ind.append(z_ind)

        # First order interpolation
        rpts = r.interp_points(pts, order=1)
        # The values interpolated should be equal
        assert np.array_equal(np.array(list_z_ind, dtype=np.float32), rpts, equal_nan=True)

        # Test there is no failure with random coordinates (edge effects, etc)
        xrand = np.random.uniform(low=xmin, high=xmax, size=(1000,))
        yrand = np.random.uniform(low=ymin, high=ymax, size=(1000,))
        pts = list(zip(xrand, yrand))
        rpts = r.interp_points(pts)

        # Second, test after a crop: the Raster now has an Area interpretation, those should fall right on the integer
        # pixel indexes
        r2 = gu.Raster(self.landsat_b4_crop_path)
        r.crop(r2)
        assert r.tags["AREA_OR_POINT"] == "Area"

        xmin, ymin, xmax, ymax = r.bounds

        # We can test with several method for the exact indexes: interp, and simple read should
        # give back the same values that fall right on the coordinates
        xrand = np.random.randint(low=0, high=r.width, size=(10,)) * list(r.transform)[0] + xmin
        yrand = ymax + np.random.randint(low=0, high=r.height, size=(10,)) * list(r.transform)[4]
        pts = list(zip(xrand, yrand))
        # By default, i and j are returned as integers
        i, j = r.xy2ij(xrand, yrand, op=np.float32)
        list_z_ind = []
        img = r.data
        for k in range(len(xrand)):
            # We directly sample the values
            z_ind = img[int(i[k]), int(j[k])]
            list_z_ind.append(z_ind)

        rpts = r.interp_points(pts, order=1)

        assert np.array_equal(np.array(list_z_ind, dtype=np.float32), rpts, equal_nan=True)

        # Test for an invidiual point (shape can be tricky at 1 dimension)
        x = 493120.0
        y = 3101000.0
        i, j = r.xy2ij(x, y)
        val = r.interp_points([(x, y)], order=1)[0]
        val_img = img[int(i[0]), int(j[0])]
        assert val_img == val

        # Finally, check that interp convert to latlon
        lat, lon = gu.projtools.reproject_to_latlon([x, y], in_crs=r.crs)
        val_latlon = r.interp_points([(lat, lon)], order=1, input_latlon=True)[0]
        assert val == pytest.approx(val_latlon, abs=0.0001)

    def test_value_at_coords(self) -> None:
        """
        Test that value at coords works as intended
        """

        # -- Tests 1: check based on indexed values --

        # Open raster
        r = gu.Raster(self.landsat_b4_crop_path)

        # A pixel center where all neighbouring coordinates are different:
        # array([[[237, 194, 239],
        #          [250, 173, 164],
        #          [255, 192, 128]]]
        itest0 = 120
        jtest0 = 451
        # This is the center of the pixel
        xtest0 = 496975.0
        ytest0 = 3099095.0

        # Verify coordinates match indexes
        x_out, y_out = r.ij2xy(itest0, jtest0, offset="center")
        assert x_out == xtest0
        assert y_out == ytest0

        # Check that the value at this coordinate is the same as when indexing
        z_val = r.value_at_coords(xtest0, ytest0)
        z = r.data.data[itest0, jtest0]
        assert z == z_val

        # Check that the value is the same the other 4 corners of the pixel
        assert z == r.value_at_coords(xtest0 + 0.49 * r.res[0], ytest0 - 0.49 * r.res[1])
        assert z == r.value_at_coords(xtest0 - 0.49 * r.res[0], ytest0 + 0.49 * r.res[1])
        assert z == r.value_at_coords(xtest0 - 0.49 * r.res[0], ytest0 - 0.49 * r.res[1])
        assert z == r.value_at_coords(xtest0 + 0.49 * r.res[0], ytest0 + 0.49 * r.res[1])

        # -- Tests 2: check arguments work as intended --

        # 1/ Lat-lon argument check by getting the coordinates of our last test point
        lat, lon = reproject_to_latlon(points=[xtest0, ytest0], in_crs=r.crs)
        z_val_2 = r.value_at_coords(lon, lat, latlon=True)
        assert z_val == z_val_2

        # 2/ Band argument
        # Get the band indexes for the multi-band Raster
        r_multi = gu.Raster(self.landsat_rgb_path)
        itest, jtest = r_multi.xy2ij(xtest0, ytest0)
        itest = int(itest[0])
        jtest = int(jtest[0])
        # Extract the values
        z_band1 = r_multi.value_at_coords(xtest0, ytest0, band=1)
        z_band2 = r_multi.value_at_coords(xtest0, ytest0, band=2)
        z_band3 = r_multi.value_at_coords(xtest0, ytest0, band=3)
        # Compare to the Raster array slice
        assert list(r_multi.data[:, itest, jtest]) == [z_band1, z_band2, z_band3]

        # 3/ Masked argument
        r_multi.data[:, itest, jtest] = np.ma.masked
        z_not_ma = r_multi.value_at_coords(xtest0, ytest0, band=1)
        assert not np.ma.is_masked(z_not_ma)
        z_ma = r_multi.value_at_coords(xtest0, ytest0, band=1, masked=True)
        assert np.ma.is_masked(z_ma)

        # 4/ Window argument
        val_window, z_window = r_multi.value_at_coords(
            xtest0, ytest0, band=1, window=3, masked=True, return_window=True
        )
        assert (
            val_window
            == np.ma.mean(r_multi.data[0, itest - 1 : itest + 2, jtest - 1 : jtest + 2])
            == np.ma.mean(z_window)
        )
        assert np.array_equal(z_window, r_multi.data[0, itest - 1 : itest + 2, jtest - 1 : jtest + 2])

        # 5/ Reducer function argument
        val_window2 = r_multi.value_at_coords(
            xtest0, ytest0, band=1, window=3, masked=True, reducer_function=np.ma.median
        )
        assert val_window2 == np.ma.median(r_multi.data[0, itest - 1 : itest + 2, jtest - 1 : jtest + 2])

        # -- Tests 3: check that errors are raised when supposed for non-boolean arguments --

        # Verify that passing a window that is not a whole number fails
        with pytest.raises(ValueError, match=re.escape("Window must be a whole number.")):
            r.value_at_coords(xtest0, ytest0, window=3.5)  # type: ignore
        # Same for an odd number
        with pytest.raises(ValueError, match=re.escape("Window must be an odd number.")):
            r.value_at_coords(xtest0, ytest0, window=4)
        # But a window that is a whole number as a float works
        r.value_at_coords(xtest0, ytest0, window=3.0)  # type: ignore

        # -- Tests 4: check that passing an array-like object works

        # For simple coordinates
        x_coords = [xtest0, xtest0 + 100]
        y_coords = [ytest0, ytest0 - 100]
        vals = r_multi.value_at_coords(x=x_coords, y=y_coords)
        val0, win0 = r_multi.value_at_coords(x=x_coords[0], y=y_coords[0], return_window=True)
        val1, win1 = r_multi.value_at_coords(x=x_coords[1], y=y_coords[1], return_window=True)

        assert len(vals) == len(x_coords)
        assert vals[0] == val0
        assert vals[1] == val1

        # With a return window argument
        vals, windows = r_multi.value_at_coords(x=x_coords, y=y_coords, return_window=True)
        assert len(windows) == len(x_coords)
        assert np.array_equal(windows[0], win0, equal_nan=True)
        assert np.array_equal(windows[1], win1, equal_nan=True)

        # -- Tests 5 -- Check image corners and latlon argument

        # Lower right pixel
        x, y = [r.bounds.right - r.res[0] / 2, r.bounds.bottom + r.res[1] / 2]
        lat, lon = pt.reproject_to_latlon([x, y], r.crs)
        assert r.value_at_coords(x, y) == r.value_at_coords(lon, lat, latlon=True) == r.data[-1, -1]

        # One pixel above
        x, y = [r.bounds.right - r.res[0] / 2, r.bounds.bottom + 3 * r.res[1] / 2]
        lat, lon = pt.reproject_to_latlon([x, y], r.crs)
        assert r.value_at_coords(x, y) == r.value_at_coords(lon, lat, latlon=True) == r.data[-2, -1]

        # One pixel left
        x, y = [r.bounds.right - 3 * r.res[0] / 2, r.bounds.bottom + r.res[1] / 2]
        lat, lon = pt.reproject_to_latlon([x, y], r.crs)
        assert r.value_at_coords(x, y) == r.value_at_coords(lon, lat, latlon=True) == r.data[-1, -2]

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_set_nodata(self, example: str) -> None:
        """
        We test set_nodata() with all possible input parameters, check expected behaviour for updating array and mask,
        and that errors and warnings are raised when they should be.
        """

        # Read raster and save a copy to compare later
        r = gu.Raster(example)
        r_copy = r.copy()
        old_nodata = r.nodata
        # We chose nodata that doesn't exist in the raster yet for both our examples (for uint8, the default value of
        # 255 exist, so we replace by 0)
        new_nodata = _default_nodata(r.dtypes[0]) if not r.dtypes[0] == "uint8" else 0

        # -- First, test set_nodata() with default parameters --

        # Check that the new_nodata does not exist in the raster yet, and set it
        assert np.count_nonzero(r_copy.data.data == new_nodata) == 0
        r.set_nodata(new_nodata=new_nodata)

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
        mask_pixel_artificially_set[0, 0] = True
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
                "New nodata value cells already exist in the data array. These cells will now be "
                "masked, and the old nodata value cells will update to the same new value. "
                "Use set_nodata() with update_array=False or update_mask=False to change "
                "this behaviour."
            ),
        ):
            r.set_nodata(new_nodata=new_nodata)

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
                "New nodata value cells already exist in the data array. These cells will now be masked. "
                "Use set_nodata() with update_mask=False to change this behaviour."
            ),
        ):
            r.set_nodata(new_nodata=new_nodata, update_array=False)

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
                "New nodata value cells already exist in the data array. The old nodata cells will update to "
                "the same new value. Use set_nodata() with update_array=False to change this behaviour."
            ),
        ):
            r.set_nodata(new_nodata=new_nodata, update_mask=False)

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

        r.set_nodata(new_nodata=new_nodata, update_array=False, update_mask=False)
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
        with pytest.raises(ValueError, match="Type of nodata not understood, must be float or int."):
            r.set_nodata(new_nodata="this_should_not_work")  # type: ignore

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

        with warnings.catch_warnings():
            # Ignore warning that nodata value is already used in the raster data
            warnings.filterwarnings(
                "ignore", category=UserWarning, message="New nodata value cells already exist in the data array.*"
            )
            r.set_nodata(_default_nodata(r.dtypes[0]))
            r_copy.nodata = _default_nodata(r.dtypes[0])

        assert r.raster_equal(r_copy)

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
        for dtype_obj in [np.dtype("int32"), np.dtype("float32"), np.dtype("float64")]:
            assert _default_nodata(dtype_obj) == -99999  # type: ignore

        # Check it works with most frequent types too
        assert _default_nodata(np.uint8) == np.iinfo("uint8").max
        for dtype_obj in [np.int32, np.float32, np.float64]:
            assert _default_nodata(dtype_obj) == -99999

        # Check that an error is raised for other types
        expected_message = "No default nodata value set for dtype."
        with pytest.raises(NotImplementedError, match=expected_message):
            _default_nodata("bla")

        # Check that an error is raised for a wrong type
        with pytest.raises(TypeError, match="dtype 1 not understood."):
            _default_nodata(1)  # type: ignore

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path, landsat_rgb_path])  # type: ignore
    def test_astype(self, example: str) -> None:

        warnings.filterwarnings(
            "ignore", category=UserWarning, message="Unmasked values equal to the nodata value found in data array.*"
        )

        # Load raster
        r = gu.Raster(example)

        all_dtypes = ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"]

        dtypes_preserving = list({np.promote_types(r.dtypes[0], dtype) for dtype in all_dtypes})
        dtypes_nonpreserving = [dtype for dtype in all_dtypes if dtype not in dtypes_preserving]

        # Test changing dtypes that does not modify the data
        for target_dtype in dtypes_preserving:
            rout = r.astype(target_dtype)  # type: ignore

            if "LE71" in os.path.basename(example):
                assert np.ma.allequal(r.data, rout.data)  # Only the same array ignoring nodata values
            else:
                assert np.array_equal(r.data.data, rout.data.data)  # Not the same array anymore with conversion
                assert np.array_equal(r.data.mask, rout.data.mask)

            assert np.dtype(rout.dtypes[0]) == target_dtype
            assert rout.data.dtype == target_dtype
            # For any data type, data should be recast to the new type
            assert rout.nodata == _default_nodata(target_dtype)

        # Test dtypes that will modify the data
        for target_dtype2 in dtypes_nonpreserving:

            with pytest.warns(UserWarning, match="dtype conversion will result in a loss of information.*"):
                rout = r.astype(target_dtype2)  # type: ignore

            assert np.array_equal(
                r.data.data.astype(target_dtype2), rout.data.data
            )  # Not the same array anymore with conversion

            assert np.dtype(rout.dtypes[0]) == target_dtype2
            assert rout.data.dtype == target_dtype2
            assert rout.nodata == _default_nodata(target_dtype2)

        # Test modify in place
        dtype = np.float64
        r2 = r.copy()
        out = r2.astype(dtype, inplace=True)
        assert out is None
        assert np.ma.allequal(r.data, r2.data)
        assert np.dtype(r2.dtypes[0]) == dtype
        assert r2.data.dtype == dtype
        assert r2.nodata == _default_nodata(dtype)

        # Test without converting nodata
        dtype = np.float64
        r3 = r.copy()
        out = r3.astype(dtype, inplace=True, convert_nodata=False)
        assert out is None
        assert np.ma.allequal(r.data, r3.data)
        assert np.dtype(r3.dtypes[0]) == dtype
        assert r3.data.dtype == dtype
        assert r3.nodata == r.nodata

    # The multi-band example will not have a colorbar, so not used in tests
    @pytest.mark.parametrize("example", [landsat_b4_path, landsat_b4_crop_path, aster_dem_path])  # type: ignore
    @pytest.mark.parametrize("figsize", np.arange(2, 20, 2))  # type: ignore
    def test_show_cbar(self, example, figsize) -> None:
        """
        Test cbar matches plot height.
        """
        # Plot raster with cbar
        r0 = gu.Raster(example)
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        r0.plot(
            ax=ax,
            add_cbar=True,
        )
        fig.axes[0].set_axis_off()
        fig.axes[1].set_axis_off()

        # Get size of main plot
        ax0_bbox = fig.axes[0].get_tightbbox()
        xmin, ymin, xmax, ymax = ax0_bbox.bounds
        h = ymax - ymin

        # Get size of cbar
        ax_cbar_bbox = fig.axes[1].get_tightbbox()
        xmin, ymin, xmax, ymax = ax_cbar_bbox.bounds
        h_cbar = ymax - ymin
        plt.close("all")

        # Assert height is the same
        assert h == pytest.approx(h_cbar)

    def test_show(self) -> None:
        # Read single band raster and RGB raster
        img = gu.Raster(self.landsat_b4_path)
        img_RGB = gu.Raster(self.landsat_rgb_path)

        # Test default plot
        img.plot()
        if DO_PLOT:
            plt.show()
        else:
            plt.close()
        assert True

        # Test with new figure
        plt.figure()
        img.plot()
        if DO_PLOT:
            plt.show()
        else:
            plt.close()
        assert True

        # Test with provided ax
        ax = plt.subplot(111)
        img.plot(ax=ax, title="Simple plotting test")
        if DO_PLOT:
            plt.show()
        else:
            plt.close()
        assert True

        # Test plot RGB
        ax = plt.subplot(111)
        img_RGB.plot(ax=ax, title="Plotting RGB")
        if DO_PLOT:
            plt.show()
        else:
            plt.close()
        assert True

        # Test plotting single band B/W, add_cbar
        ax = plt.subplot(111)
        img_RGB.plot(bands=1, cmap="gray", ax=ax, add_cbar=False, title="Plotting one band B/W")
        if DO_PLOT:
            plt.show()
        else:
            plt.close()
        assert True

        # Test vmin, vmax and cbar_title
        ax = plt.subplot(111)
        img.plot(
            cmap="gray", vmin=40, vmax=220, cbar_title="Custom cbar", ax=ax, title="Testing vmin, vmax and cbar_title"
        )
        if DO_PLOT:
            plt.show()
        else:
            plt.close()
        assert True

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_save(self, example: str) -> None:
        # Read single band raster
        img = gu.Raster(example)

        # Temporary folder
        temp_dir = tempfile.TemporaryDirectory()

        # Save file to temporary file, with defaults opts
        temp_file = os.path.join(temp_dir.name, "test.tif")
        img.save(temp_file)
        saved = gu.Raster(temp_file)
        assert img.raster_equal(saved)

        # Try to save with a pathlib path (create a new temp file for Windows)
        path = pathlib.Path(temp_file)
        img.save(path)

        # Test additional options
        co_opts = {"TILED": "YES", "COMPRESS": "LZW"}
        metadata = {"Type": "test"}
        img.save(temp_file, co_opts=co_opts, metadata=metadata)
        saved = gu.Raster(temp_file)
        assert img.raster_equal(saved)
        assert saved.tags["Type"] == "test"

        # Test that nodata value is enforced when masking - since value 0 is not used, data should be unchanged
        img.save(temp_file, nodata=0)
        saved = gu.Raster(temp_file)
        assert np.ma.allequal(img.data, saved.data)
        assert saved.nodata == 0

        # Test that mask is preserved if nodata value is valid
        mask = img.data == np.min(img.data)
        img.set_mask(mask)
        if img.nodata is not None:
            img.save(temp_file)
            saved = gu.Raster(temp_file)
            assert np.array_equal(img.data.mask, saved.data.mask)

        # Test that a warning is raised if nodata is not set and a mask exists (defined above)
        if img.nodata is None:
            with pytest.warns(UserWarning):
                img.save(TemporaryFile())

        # Test with blank argument
        img.save(temp_file, blank_value=0)
        saved = gu.Raster(temp_file)

        assert np.array_equal(saved.data.data, np.zeros(np.shape(saved.data)))

        # Clean up temporary folder - fails on Windows
        try:
            temp_dir.cleanup()
        except (NotADirectoryError, PermissionError):
            pass

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path, landsat_rgb_path])  # type: ignore
    def test_coords(self, example: str) -> None:
        img = gu.Raster(self.landsat_b4_path)

        # With corner argument
        xx0, yy0 = img.coords(offset="corner", grid=False)

        assert xx0[0] == pytest.approx(img.bounds.left)
        assert xx0[-1] == pytest.approx(img.bounds.right - img.res[0])
        if img.res[1] > 0:
            assert yy0[0] == pytest.approx(img.bounds.bottom)
            assert yy0[-1] == pytest.approx(img.bounds.top - img.res[1])
        else:
            # Currently not covered by test image
            assert yy0[0] == pytest.approx(img.bounds.top)
            assert yy0[-1] == pytest.approx(img.bounds.bottom + img.res[1])

        # With center argument
        xx, yy = img.coords(offset="center", grid=False)
        hx = img.res[0] / 2
        hy = img.res[1] / 2
        assert xx[0] == pytest.approx(img.bounds.left + hx)
        assert xx[-1] == pytest.approx(img.bounds.right - hx)
        if img.res[1] > 0:
            assert yy[0] == pytest.approx(img.bounds.bottom + hy)
            assert yy[-1] == pytest.approx(img.bounds.top - hy)
        else:
            # Currently not covered by test image
            assert yy[-1] == pytest.approx(img.bounds.top + hy)
            assert yy[0] == pytest.approx(img.bounds.bottom - hy)

        # With grid argument (default argument, repeated here for clarity)
        xxgrid, yygrid = img.coords(offset="corner", grid=True)
        assert np.array_equal(xxgrid, np.repeat(xx0[np.newaxis, :], img.height, axis=0))
        assert np.array_equal(yygrid, np.flipud(np.repeat(yy0[:, np.newaxis], img.width, axis=1)))

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path, landsat_rgb_path])  # type: ignore
    def test_from_array(self, example: str) -> None:

        if "LE71" in os.path.basename(example):
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Unmasked values equal to the nodata value found in data array.*",
            )

        # Test that from_array works if nothing is changed
        # -> most tests already performed in test_copy and data.setter, no need for many more
        img = gu.Raster(self.aster_dem_path)
        out_img = gu.Raster.from_array(img.data, img.transform, img.crs, nodata=img.nodata)
        assert out_img.raster_equal(img)

        # Test that changes to data are taken into account
        bias = 5
        out_img = gu.Raster.from_array(img.data + bias, img.transform, img.crs, nodata=img.nodata)
        assert np.ma.allequal(out_img.data, img.data + bias)

        # Test that nodata is properly taken into account
        out_img = gu.Raster.from_array(img.data + 5, img.transform, img.crs, nodata=0)
        assert out_img.nodata == 0

        # Test that data mask is taken into account
        img.data.mask = np.zeros((img.shape), dtype="bool")
        img.data.mask[0, 0] = True
        out_img = gu.Raster.from_array(img.data, img.transform, img.crs, nodata=0)
        assert out_img.data.mask[0, 0]

        # Check that error is raised if the transform is not affine
        with pytest.raises(TypeError, match="The transform argument needs to be Affine or tuple."):
            gu.Raster.from_array(data=img.data, transform="lol", crs=None, nodata=None)  # type: ignore

    def test_type_hints(self) -> None:
        """Test that pylint doesn't raise errors on valid code."""
        # Create a temporary directory and a temporary filename
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = os.path.join(temp_dir.name, "code.py")

        # r = gu.Raster(self.landsat_b4_path)

        # Load the attributes to check
        attributes = ["transform", "crs", "nodata", "name", "driver", "is_loaded", "filename"]
        # Create some sample code that should be correct
        sample_code = "\n".join(
            [
                "'''Sample code that should conform to pylint's standards.'''",  # Add docstring
                "import geoutils as gu",  # Import geoutils
                "raster = gu.Raster(gu.examples.get_path('landsat_B4'))",  # Load a raster
            ]
            + [  # The below statements should not raise a 'no-member' (E1101) error.
                f"{attribute.upper()} = raster.{attribute}" for attribute in attributes
            ]
            + [""]  # Add a newline to the end.
        )

        # Write the code to the temporary file
        with open(temp_path, "w") as outfile:
            outfile.write(sample_code)

        # Run pylint and parse the stdout as a string, only test
        pylint_output = StringIO()
        Run([temp_path], reporter=TextReporter(pylint_output), exit=False)

        lint_string = pylint_output.getvalue()
        print(lint_string)  # Print the output for debug purposes

        # Bad linting errors are defined here. Currently just "no-member" errors
        bad_lints = [f"Instance of 'Raster' has no '{attribute}' member" for attribute in attributes]

        # Assert that none of the bad errors are in the pylint output
        for bad_lint in bad_lints:
            assert bad_lint not in lint_string, f"`{bad_lint}` contained in the lint_string"

    def test_split_bands(self) -> None:
        img = gu.Raster(self.landsat_rgb_path)

        red, green, blue = img.split_bands(copy=False)

        # Check that the shapes are correct.
        assert red.count == 1
        assert img.count == 3

        # Extract only one band (then it will not return a list)
        red2 = img.split_bands(copy=False, bands=1)[0]

        # Extract a subset with a list in a weird direction
        blue2, green2 = img.split_bands(copy=False, bands=[3, 2])

        # Check that the subset functionality works as expected.
        assert red.raster_equal(red2)
        assert green.raster_equal(green2)
        assert blue.raster_equal(blue2)

        # Check that the red channel and the rgb data shares memory
        assert np.shares_memory(red.data, img.data)

        # Check that the red band data is not equal to the full RGB data.
        assert not red.raster_equal(img)

        # Test that the red band corresponds to the first band of the img
        assert np.array_equal(
            red.data.data.squeeze().astype("float32"), img.data.data[0, :, :].astype("float32"), equal_nan=True
        )

        # Modify the red band and make sure it propagates to the original img (it's not a copy)
        red.data += 1
        assert np.array_equal(
            red.data.data.squeeze().astype("float32"), img.data.data[0, :, :].astype("float32"), equal_nan=True
        )

        # Copy the bands instead of pointing to the same memory.
        red_c = img.split_bands(copy=True, bands=1)[0]

        # Check that the red band data does not share memory with the rgb image (it's a copy)
        assert not np.shares_memory(red_c.data, img.data)

        # Modify the copy, and make sure the original data is not modified.
        red_c.data += 1
        assert not np.array_equal(
            red_c.data.data.squeeze().astype("float32"), img.data.data[0, :, :].astype("float32"), equal_nan=True
        )

    def test_resampling_str(self) -> None:
        """Test that resampling methods can be given as strings instead of rio enums."""
        warnings.simplefilter("error")
        assert resampling_method_from_str("nearest") == rio.enums.Resampling.nearest  # noqa
        assert resampling_method_from_str("cubic_spline") == rio.enums.Resampling.cubic_spline  # noqa

        # Check that odd strings return the appropriate error.
        try:
            resampling_method_from_str("CUBIC_SPLINE")  # noqa
        except ValueError as exception:
            if "not a valid rasterio.enums.Resampling method" not in str(exception):
                raise exception

        img1 = gu.Raster(self.landsat_b4_path)
        img2 = gu.Raster(self.landsat_b4_crop_path)
        img1.set_nodata(0)
        img2.set_nodata(0)

        # Resample the rasters using a new resampling method and see that the string and enum gives the same result.
        img3a = img1.reproject(img2, resampling="q1")
        img3b = img1.reproject(img2, resampling=rio.enums.Resampling.q1)
        assert img3a.raster_equal(img3b)

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_polygonize(self, example: str) -> None:
        """Test that polygonize doesn't raise errors."""

        img = gu.Raster(example)

        # -- Test 1: basic functioning of polygonize --

        # Get unique value for image and the corresponding area
        value = np.unique(img)[0]
        pixel_area = np.count_nonzero(img.data == value) * img.res[0] * img.res[1]

        # Polygonize the raster for this value, and compute the total area
        polygonized = img.polygonize(target_values=value)
        polygon_area = polygonized.ds.area.sum()

        # Check that these two areas are approximately equal
        assert polygon_area == pytest.approx(pixel_area)
        assert isinstance(polygonized, gu.Vector)
        assert polygonized.crs == img.crs

        # -- Test 2: data types --

        # Check that polygonize works as expected for any input dtype (e.g. float64 being not supported by GeoPandas)
        for dtype in ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"]:
            img_dtype = img.copy()
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, message="dtype conversion will result in a " "loss of information.*"
                )
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="Unmasked values equal to the nodata value found in data array.*",
                )
                img_dtype = img_dtype.astype(dtype)
            value = np.unique(img_dtype)[0]
            img_dtype.polygonize(target_values=value)

        # And for a boolean object, such as a mask
        mask = img > value
        mask.polygonize(target_values=1)

    # Test all options, with both an artificial Raster (that has all target values) and a real Raster
    @pytest.mark.parametrize("distunits", ["GEO", "PIXEL"])  # type: ignore
    # 0 and 1,2,3 are especially useful for the artificial Raster, and 112 for the real Raster
    @pytest.mark.parametrize("target_values", [[1, 2, 3], [0], [112], None])  # type: ignore
    @pytest.mark.parametrize(
        "raster",
        [
            gu.Raster(landsat_b4_path),
            gu.Raster.from_array(
                np.arange(25, dtype="int32").reshape(5, 5), transform=rio.transform.from_origin(0, 5, 1, 1), crs=4326
            ),
        ],
    )  # type: ignore
    def test_proximity_against_gdal(self, distunits: str, target_values: list[float] | None, raster: gu.Raster) -> None:
        """Test that proximity matches the results of GDAL for any parameter."""

        # TODO: When adding new rasters for tests, specify warning only for Landsat
        warnings.filterwarnings("ignore", message="Setting default nodata -99999 to mask non-finite values *")

        # We generate proximity with GDAL and GeoUtils
        gdal_proximity = run_gdal_proximity(raster, target_values=target_values, distunits=distunits)
        # We translate distunits GDAL option into its GeoUtils equivalent
        if distunits == "GEO":
            distance_unit = "georeferenced"
        else:
            distance_unit = "pixel"
        geoutils_proximity = (
            raster.proximity(distance_unit=distance_unit, target_values=target_values)
            .data.data.squeeze()
            .astype("float32")
        )

        # The results should be the same in all cases
        try:
            # In some cases, the proximity differs slightly (generally <1%) for complex settings
            # (Landsat Raster with target of 112)
            # It looks like GDAL might not have the right value,
            # so this particular case is treated differently in tests
            if target_values is not None and target_values[0] == 112 and raster.filename is not None:
                # Get index and number of not almost equal point (tolerance of 10-4)
                ind_not_almost_equal = np.abs(gdal_proximity - geoutils_proximity) > 1e-04
                nb_not_almost_equal = np.count_nonzero(ind_not_almost_equal)
                # Check that this is a minority of points (less than 0.5%)
                assert nb_not_almost_equal < 0.005 * raster.width * raster.height

                # Replace these exceptions by zero in both
                gdal_proximity[ind_not_almost_equal] = 0.0
                geoutils_proximity[ind_not_almost_equal] = 0.0
                # Check that all the rest is almost equal
                assert np.allclose(gdal_proximity, geoutils_proximity, atol=1e-04, equal_nan=True)

            # Otherwise, results are exactly equal
            else:
                assert np.array_equal(gdal_proximity, geoutils_proximity, equal_nan=True)

        # For debugging
        except Exception as exception:
            import matplotlib.pyplot as plt

            # Plotting the xdem and GDAL attributes for comparison (plotting "diff" can also help debug)
            plt.subplot(121)
            plt.imshow(gdal_proximity)
            # plt.imshow(np.abs(gdal_proximity - geoutils_proximity)>0.1)
            plt.colorbar()
            plt.subplot(122)
            plt.imshow(geoutils_proximity)
            # plt.imshow(raster.data.data == 112)
            plt.colorbar()
            plt.show()

            # ind_not_equal = np.abs(gdal_proximity - geoutils_proximity)>0.1
            # print(gdal_proximity[ind_not_equal])
            # print(geoutils_proximity[ind_not_equal])

            raise exception

    def test_proximity_parameters(self) -> None:
        """
        Test that new (different to GDAL's) proximity parameters run.
        No need to test the results specifically, as those rely entirely on the previous test with GDAL,
        and tests in rasterize and shapely.
        #TODO: Maybe add one test with an artificial vector to check it works as intended
        """

        # -- Test 1: with self's Raster alone --
        raster1 = gu.Raster(self.landsat_b4_path)
        prox1 = raster1.proximity()

        # The raster should have the same extent, resolution and CRS
        assert raster1.georeferenced_grid_equal(prox1)

        # It should change with target values specified
        prox2 = raster1.proximity(target_values=[255])
        assert not np.array_equal(prox1.data, prox2.data)

        # -- Test 2: with a vector provided --
        vector = gu.Vector(self.everest_outlines_path)

        # With default options (boundary geometry)
        raster1.proximity(vector=vector)

        # With the base geometry
        raster1.proximity(vector=vector, geometry_type="geometry")

        # With another geometry option
        raster1.proximity(vector=vector, geometry_type="centroid")

        # With only inside proximity
        raster1.proximity(vector=vector, in_or_out="in")

    def test_to_points(self) -> None:
        """Test the outputs of the to_points method and that it doesn't load if not needed."""

        # Create a small raster to test point sampling on
        img0 = gu.Raster.from_array(
            np.arange(25, dtype="int32").reshape(5, 5), transform=rio.transform.from_origin(0, 5, 1, 1), crs=4326
        )

        # Sample the whole raster (fraction==1)
        points = img0.to_points(1, as_array=True)

        # Validate that 25 points were sampled (equating to img1.height * img1.width) with x, y, and band0 values.
        assert isinstance(points, np.ndarray)
        assert points.shape == (25, 3)
        assert np.array_equal(np.asarray(points[:, 0]), np.tile(np.linspace(0.5, 4.5, 5), 5))

        assert img0.to_points(0.2, as_array=True).shape == (5, 3)

        # Try with a single-band raster
        img1 = gu.Raster(self.aster_dem_path)

        points = img1.to_points(10, as_array=True)

        assert points.shape == (10, 3)
        assert not img1.is_loaded

        points_frame = img1.to_points(10)

        assert isinstance(points_frame, gu.Vector)
        assert np.array_equal(points_frame.ds.columns, ["b1", "geometry"])
        assert points_frame.crs == img1.crs

        # Try with a multi-band raster
        img2 = gu.Raster(self.landsat_rgb_path)

        points = img2.to_points(10, as_array=True)

        assert points.shape == (10, 5)
        assert not img2.is_loaded

        points_frame = img2.to_points(10)

        assert isinstance(points_frame, gu.Vector)
        assert np.array_equal(points_frame.ds.columns, ["b1", "b2", "b3", "geometry"])
        assert points_frame.crs == img2.crs


class TestMask:
    # Paths to example data
    landsat_b4_path = examples.get_path("everest_landsat_b4")
    landsat_b4_crop_path = examples.get_path("everest_landsat_b4_cropped")
    landsat_rgb_path = examples.get_path("everest_landsat_rgb")
    everest_outlines_path = examples.get_path("everest_rgi_outlines")
    aster_dem_path = examples.get_path("exploradores_aster_dem")
    aster_outlines_path = examples.get_path("exploradores_rgi_outlines")

    # Synthetic data
    width = height = 5
    transform = rio.transform.from_bounds(0, 0, 1, 1, width, height)
    np.random.seed(42)
    arr = np.random.randint(low=0, high=2, size=(1, width, height), dtype=bool)
    arr_mask = np.random.randint(0, 2, size=(1, width, height), dtype=bool)
    mask_ma = np.ma.masked_array(data=arr, mask=arr_mask)

    # Mask without nodata
    mask_landsat_b4 = gu.Raster(landsat_b4_path) > 125
    # Mask with nodata
    mask_aster_dem = gu.Raster(aster_dem_path) > 2000
    # Mask from an outline
    mask_everest = gu.Vector(everest_outlines_path).create_mask(gu.Raster(landsat_b4_path))

    @pytest.mark.parametrize("example", [landsat_b4_path, landsat_rgb_path, aster_dem_path])  # type: ignore
    def test_init(self, example: str) -> None:
        """Test that Mask subclass initialization function as intended."""

        # A warning should be raised when the raster is a multi-band
        if "RGB" not in os.path.basename(example):
            mask = gu.Mask(example)
        else:
            with pytest.warns(
                UserWarning, match="Multi-band raster provided to create a Mask, only the first band will be used."
            ):
                mask = gu.Mask(example)

        # Check the masked array type
        assert mask.data.dtype == "bool"
        # Check output is the correct instance
        assert isinstance(mask, gu.Mask)
        # Check the dtypes metadata
        assert mask.dtypes[0] == "bool"
        # Check the nodata
        assert mask.nodata is None
        # Check the nbands metadata
        assert mask.count == 1

        # Check that a mask object is sent back from its own init
        mask2 = gu.Mask(mask)
        assert mask.raster_equal(mask2)

    @pytest.mark.parametrize("example", [landsat_b4_path, landsat_rgb_path, aster_dem_path])  # type: ignore
    def test_repr_str(self, example: str) -> None:
        """Test the representation of a raster works"""

        warnings.filterwarnings("ignore", message="Multi-band raster provided to create a Mask*")

        # For data not loaded by default
        r = gu.Mask(example)

        r_repr = r.__repr__()
        r_str = r.__str__()

        # Check that the class is printed correctly
        assert r_repr[0:4] == "Mask"

        # Check that all main attribute names are printed
        attrs_shown = ["data", "transform", "crs"]
        assert all(attr + "=" in r_repr for attr in attrs_shown)

        # Check nodata is removed
        assert "nodata=" not in r_repr

        assert r_str == r.data.__str__()

    def test_from_array(self) -> None:
        """Test that Raster.__init__ casts to Mask with dict input of from_array() and a boolean data array."""

        mask_rst = gu.Raster.from_array(data=self.mask_ma, transform=self.transform, crs=None, nodata=None)

        assert isinstance(mask_rst, gu.Mask)
        assert mask_rst.transform == self.transform
        assert mask_rst.crs is None
        assert mask_rst.nodata is None

    # List all logical operators which will cast Rasters into Masks
    ops_logical = [
        "__lt__",
        "__le__",
        "__eq__",
        "__ne__",
        "__gt__",
        "__ge__",
    ]

    @pytest.mark.parametrize("op", ops_logical)  # type: ignore
    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_logical_casting_real(self, example: str, op: str) -> None:
        """
        Test that logical operations cast Raster object to Mask on real data
        (synthetic done in TestArithmetic).
        """

        rst = gu.Raster(example)

        # Logical operations should cast to a Mask object, preserving the mask
        mask = getattr(rst, op)(1)
        assert isinstance(mask, gu.Mask)
        assert np.array_equal(mask.data.data, getattr(rst.data.data, op)(1))
        assert np.array_equal(mask.data.mask, rst.data.mask)

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_implicit_logical_casting_real(self, example: str) -> None:
        """
        Test that implicit logical operations on real data
        (synthetic done in TestArithmetic).
        """

        rst = gu.Raster(example)

        # Equality
        mask = rst == 1
        assert isinstance(mask, gu.Mask)
        assert np.array_equal(mask.data.data, rst.data.data == 1)
        assert np.array_equal(mask.data.mask, rst.data.mask)

        # Non-equality
        mask = rst != 1
        assert isinstance(mask, gu.Mask)
        assert np.array_equal(mask.data.data, rst.data.data != 1)
        assert np.array_equal(mask.data.mask, rst.data.mask)

        # Lower than
        mask = rst < 1
        assert isinstance(mask, gu.Mask)
        assert np.array_equal(mask.data.data, rst.data.data < 1)
        assert np.array_equal(mask.data.mask, rst.data.mask)

        # Lower equal
        mask = rst <= 1
        assert isinstance(mask, gu.Mask)
        assert np.array_equal(mask.data.data, rst.data.data <= 1)
        assert np.array_equal(mask.data.mask, rst.data.mask)

        # Greater than
        mask = rst > 1
        assert isinstance(mask, gu.Mask)
        assert np.array_equal(mask.data.data, rst.data.data > 1)
        assert np.array_equal(mask.data.mask, rst.data.mask)

        # Greater equal
        mask = rst >= 1
        assert isinstance(mask, gu.Mask)
        assert np.array_equal(mask.data.data, rst.data.data >= 1)
        assert np.array_equal(mask.data.mask, rst.data.mask)

    @pytest.mark.parametrize("mask", [mask_landsat_b4, mask_aster_dem, mask_everest])  # type: ignore
    def test_reproject(self, mask: gu.Mask) -> None:
        # Test 1: with a classic resampling (bilinear)

        # Reproject mask - resample to 100 x 100 grid
        mask_orig = mask.copy()
        mask_reproj = mask.reproject(grid_size=(100, 100), force_source_nodata=2)

        # Check instance is respected
        assert isinstance(mask_reproj, gu.Mask)
        # Check the dtype of the original mask was properly reconverted
        assert mask.data.dtype == bool
        # Check the original mask was not modified during reprojection
        assert mask_orig.raster_equal(mask)

        # Check inplace behaviour works
        mask_tmp = mask.copy()
        mask_tmp.reproject(grid_size=(100, 100), force_source_nodata=2, inplace=True)
        assert mask_tmp.raster_equal(mask_reproj)

        # This should be equivalent to converting the array to uint8, reprojecting, converting back
        mask_uint8 = mask.astype("uint8")
        mask_uint8_reproj = mask_uint8.reproject(grid_size=(100, 100), force_source_nodata=2)
        mask_uint8_reproj.data = mask_uint8_reproj.data.astype("bool")

        assert mask_reproj.raster_equal(mask_uint8_reproj)

        # Test 2: should raise a warning when the resampling differs from nearest

        with pytest.warns(
            UserWarning,
            match="Reprojecting a mask with a resampling method other than 'nearest', "
            "the boolean array will be converted to float during interpolation.",
        ):
            mask.reproject(resampling="bilinear")

    @pytest.mark.parametrize("mask", [mask_landsat_b4, mask_aster_dem, mask_everest])  # type: ignore
    def test_crop(self, mask: gu.Mask) -> None:
        # Test with same bounds -> should be the same #

        mask_orig = mask.copy()
        crop_geom = mask.bounds
        mask_cropped = mask.crop(crop_geom)
        assert mask_cropped.raster_equal(mask)

        # Check if instance is respected
        assert isinstance(mask_cropped, gu.Mask)
        # Check the dtype of the original mask was properly reconverted
        assert mask.data.dtype == bool
        # Check the original mask was not modified during cropping
        assert mask_orig.raster_equal(mask)

        # Check inplace behaviour works
        mask_tmp = mask.copy()
        mask_tmp.crop(crop_geom, inplace=True)
        assert mask_tmp.raster_equal(mask_cropped)

        # - Test cropping each side by a random integer of pixels - #
        rand_int = np.random.randint(1, min(mask.shape) - 1)

        # Left
        crop_geom2 = [crop_geom[0] + rand_int * mask.res[0], crop_geom[1], crop_geom[2], crop_geom[3]]
        mask_cropped = mask.crop(crop_geom2)
        assert list(mask_cropped.bounds) == crop_geom2
        assert np.array_equal(mask.data[:, rand_int:].data, mask_cropped.data.data, equal_nan=True)
        assert np.array_equal(mask.data[:, rand_int:].mask, mask_cropped.data.mask)

        # Right
        crop_geom2 = [crop_geom[0], crop_geom[1], crop_geom[2] - rand_int * mask.res[0], crop_geom[3]]
        mask_cropped = mask.crop(crop_geom2)
        assert list(mask_cropped.bounds) == crop_geom2
        assert np.array_equal(mask.data[:, :-rand_int].data, mask_cropped.data.data, equal_nan=True)
        assert np.array_equal(mask.data[:, :-rand_int].mask, mask_cropped.data.mask)

        # Bottom
        crop_geom2 = [crop_geom[0], crop_geom[1] + rand_int * abs(mask.res[1]), crop_geom[2], crop_geom[3]]
        mask_cropped = mask.crop(crop_geom2)
        assert list(mask_cropped.bounds) == crop_geom2
        assert np.array_equal(mask.data[:-rand_int, :].data, mask_cropped.data.data, equal_nan=True)
        assert np.array_equal(mask.data[:-rand_int, :].mask, mask_cropped.data.mask)

        # Top
        crop_geom2 = [crop_geom[0], crop_geom[1], crop_geom[2], crop_geom[3] - rand_int * abs(mask.res[1])]
        mask_cropped = mask.crop(crop_geom2)
        assert list(mask_cropped.bounds) == crop_geom2
        assert np.array_equal(mask.data[rand_int:, :].data, mask_cropped.data, equal_nan=True)
        assert np.array_equal(mask.data[rand_int:, :].mask, mask_cropped.data.mask)

        # Test inplace
        mask_orig = mask.copy()
        mask_orig.crop(crop_geom2, inplace=True)
        assert list(mask_orig.bounds) == crop_geom2
        assert np.array_equal(mask.data[rand_int:, :].data, mask_orig.data, equal_nan=True)
        assert np.array_equal(mask.data[rand_int:, :].mask, mask_orig.data.mask)

        # Run with match_extent, check that inplace or not yields the same result

        # TODO: Pretty sketchy with the current functioning of "match_extent",
        #  should we just remove it from Raster.crop() ?

        # mask_cropped = mask.crop(crop_geom2, inplace=False, mode="match_extent")
        # mask_orig.crop(crop_geom2, mode="match_extent")
        # assert mask_cropped.raster_equal(mask_orig)

    @pytest.mark.parametrize("mask", [mask_landsat_b4, mask_aster_dem, mask_everest])  # type: ignore
    def test_polygonize(self, mask: gu.Mask) -> None:

        mask_orig = mask.copy()
        # Run default
        vect = mask.polygonize()
        # Check the dtype of the original mask was properly reconverted
        assert mask.data.dtype == bool
        # Check the original mask was not modified during polygonizing
        assert mask_orig.raster_equal(mask)

        # Check the output is cast into a vector
        assert isinstance(vect, gu.Vector)

        # Run with zero as target
        vect = mask.polygonize(target_values=0)
        assert isinstance(vect, gu.Vector)

        # Check a warning is raised when using a non-boolean value
        with pytest.warns(UserWarning, match="In-value converted to 1 for polygonizing boolean mask."):
            mask.polygonize(target_values=2)

    @pytest.mark.parametrize("mask", [mask_landsat_b4, mask_aster_dem, mask_everest])  # type: ignore
    def test_proximity(self, mask: gu.Mask) -> None:

        mask_orig = mask.copy()
        # Run default
        rast = mask.proximity()
        # Check the dtype of the original mask was properly reconverted
        assert mask.data.dtype == bool
        # Check the original mask was not modified during reprojection
        assert mask_orig.raster_equal(mask)

        # Check that output is cast back into a raster
        assert isinstance(rast, gu.Raster)
        # A mask is a raster, so also need to check this
        assert not isinstance(rast, gu.Mask)

    @pytest.mark.parametrize("mask", [mask_landsat_b4, mask_aster_dem, mask_everest])  # type: ignore
    def test_save(self, mask: gu.Mask) -> None:
        """Test saving for masks"""

        # Temporary folder
        temp_dir = tempfile.TemporaryDirectory()

        # Save file to temporary file, with defaults opts
        temp_file = os.path.join(temp_dir.name, "test.tif")
        mask.save(temp_file)
        saved = gu.Mask(temp_file)

        # A raster (or mask) in-memory has more information than on disk, we need to update it before checking equality
        # The values in its .data.data that are masked in .data.mask are not necessarily equal to the nodata value
        mask.data.data[mask.data.mask] = True  # The default nodata 255 is converted to boolean True on masked values

        # Check all attributes are equal
        assert mask.raster_equal(saved)

        # Clean up temporary folder - fails on Windows
        try:
            temp_dir.cleanup()
        except (NotADirectoryError, PermissionError):
            pass


class TestArithmetic:
    """
    Test that all arithmetic overloading functions work as expected.
    """

    # Create fake rasters with random values in 0-255 and dtype uint8
    # TODO: Add the case where a mask exists in the array, as in test_data_setter
    width = height = 5
    transform = rio.transform.from_bounds(0, 0, 1, 1, width, height)
    r1 = gu.Raster.from_array(np.random.randint(1, 255, (height, width), dtype="uint8"), transform=transform, crs=None)
    r2 = gu.Raster.from_array(np.random.randint(1, 255, (height, width), dtype="uint8"), transform=transform, crs=None)

    # Tests with different dtype
    r1_f32 = gu.Raster.from_array(
        np.random.randint(1, 255, (height, width)).astype("float32"), transform=transform, crs=None
    )

    # Test with nodata value set
    r1_nodata = gu.Raster.from_array(
        np.random.randint(1, 255, (height, width)).astype("float32"),
        transform=transform,
        crs=None,
        nodata=_default_nodata("float32"),
    )

    # Test with 0 values
    r2_zero = gu.Raster.from_array(
        np.random.randint(1, 255, (height, width)).astype("float32"),
        transform=transform,
        crs=None,
        nodata=_default_nodata("float32"),
    )
    r2_zero.data[0, 0] = 0

    # Create rasters with different shape, crs or transforms for testing errors
    r1_wrong_shape = gu.Raster.from_array(
        np.random.randint(0, 255, (height + 1, width)).astype("float32"),
        transform=transform,
        crs=None,
    )

    r1_wrong_crs = gu.Raster.from_array(
        np.random.randint(0, 255, (height, width)).astype("float32"),
        transform=transform,
        crs=rio.crs.CRS.from_epsg(4326),
    )

    transform2 = rio.transform.from_bounds(0, 0, 2, 2, width, height)
    r1_wrong_transform = gu.Raster.from_array(
        np.random.randint(0, 255, (height, width)).astype("float32"), transform=transform2, crs=None
    )

    # Tests with child class
    satimg = gu.SatelliteImage.from_array(
        np.random.randint(1, 255, (height, width)).astype("float32"), transform=transform, crs=None
    )

    def test_raster_equal(self) -> None:
        """
        Test that raster_equal() works as expected.
        """
        r1 = self.r1
        r2 = r1.copy()
        assert r1.raster_equal(r2)

        # Change data
        r2.data += 1
        assert not r1.raster_equal(r2)

        # Change mask (False by default)
        r2 = r1.copy()
        r2.data[0, 0] = np.ma.masked
        assert not r1.raster_equal(r2)

        # Change fill_value (999999 by default)
        r2 = r1.copy()
        r2.data.fill_value = 0
        assert not r1.raster_equal(r2)

        # Change dtype
        r2 = r1.copy()
        r2 = r2.astype("float32")
        assert not r1.raster_equal(r2)

        # Change transform
        r2 = r1.copy()
        r2.transform = rio.transform.from_bounds(0, 0, 1, 1, self.width + 1, self.height)
        assert not r1.raster_equal(r2)

        # Change CRS
        r2 = r1.copy()
        r2.crs = rio.crs.CRS.from_epsg(4326)
        assert not r1.raster_equal(r2)

        # Change nodata
        r2 = r1.copy()
        r2.set_nodata(34)
        assert not r1.raster_equal(r2)

    def test_equal_georeferenced_grid(self) -> None:
        """
        Test that equal for shape, crs and transform work as expected
        """

        # -- Test 1: based on a copy --
        r1 = self.r1
        r2 = r1.copy()
        assert r1.georeferenced_grid_equal(r2)

        # Change data
        r2.data += 1
        assert r1.georeferenced_grid_equal(r2)

        # Change mask (False by default)
        r2 = r1.copy()
        r2.data[0, 0] = np.ma.masked
        assert r1.georeferenced_grid_equal(r2)

        # Change fill_value (999999 by default)
        r2 = r1.copy()
        r2.data.fill_value = 0
        assert r1.georeferenced_grid_equal(r2)

        # Change dtype
        r2 = r1.copy()
        r2 = r2.astype("float32")
        assert r1.georeferenced_grid_equal(r2)

        # Change transform
        r2 = r1.copy()
        r2.transform = rio.transform.from_bounds(0, 0, 1, 1, self.width + 1, self.height)
        assert not r1.georeferenced_grid_equal(r2)

        # Change CRS
        r2 = r1.copy()
        r2.crs = rio.crs.CRS.from_epsg(4326)
        assert not r1.georeferenced_grid_equal(r2)

        # Change nodata
        r2 = r1.copy()
        r2.set_nodata(34)
        assert r1.georeferenced_grid_equal(r2)

        # -- Test 2: based on another Raster with one different georeferenced grid attribute --

        assert not r1.georeferenced_grid_equal(self.r1_wrong_crs)

        assert not r1.georeferenced_grid_equal(self.r1_wrong_shape)

        assert not r1.georeferenced_grid_equal(self.r1_wrong_transform)

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
        array = np.random.randint(1, 255, (self.height, self.width)).astype("float64")
        floatval = 3.14
        intval = 1

        # Test with 2 uint8 rasters
        r1 = self.r1
        r2 = self.r2
        r3 = getattr(r1, op)(r2)
        ctype = np.promote_types(r1.data.dtype, r2.data.dtype)
        numpy_output = getattr(r1.data.astype(ctype), op)(r2.data.astype(ctype))
        assert isinstance(r3, gu.Raster)
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
        assert isinstance(r3, gu.Raster)
        assert r1.raster_equal(r1_copy)
        assert r2.raster_equal(r2_copy)

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
        assert isinstance(r3, gu.Raster)
        assert np.all(r3.data == getattr(r1.data, op)(array))
        if np.sum(r3.data.mask) == 0:
            assert r3.nodata is None
        else:
            assert r3.nodata == _default_nodata("float32")

        # Test with an integer
        r3 = getattr(r1, op)(intval)
        assert isinstance(r3, gu.Raster)
        assert np.all(r3.data == getattr(r1.data, op)(intval))
        if np.sum(r3.data.mask) == 0:
            assert r3.nodata is None
        else:
            assert r3.nodata == _default_nodata("uint8")

        # Test with a float value
        r3 = getattr(r1, op)(floatval)
        dtype = np.dtype(rio.dtypes.get_minimum_dtype(floatval))
        assert isinstance(r3, gu.Raster)
        assert r3.data.dtype == dtype
        assert np.all(r3.data == getattr(r1.data, op)(np.array(floatval).astype(dtype)))
        if np.sum(r3.data.mask) == 0:
            assert r3.nodata is None
        else:
            assert r3.nodata == _default_nodata(dtype)

        # Test with child class
        r3 = getattr(satimg, op)(intval)
        assert isinstance(r3, gu.SatelliteImage)

    reflective_ops = [["__add__", "__radd__"], ["__mul__", "__rmul__"]]

    @pytest.mark.parametrize("ops", reflective_ops)  # type: ignore
    def test_reflectivity(self, ops: list[str]) -> None:
        """
        Check reflective operations
        """
        warnings.filterwarnings("ignore", message="invalid value encountered")

        # Test various inputs: Raster with different dtypes, np.ndarray, single number
        array = np.random.randint(1, 255, (self.height, self.width)).astype("float64")
        floatval = 3.14
        intval = 1

        # Get reflective operations
        op1, op2 = ops

        # Test with uint8 rasters
        r3 = getattr(self.r1, op1)(self.r2)
        r4 = getattr(self.r1, op2)(self.r2)
        assert r3.raster_equal(r4)

        # Test with different dtypes
        r3 = getattr(self.r1_f32, op1)(self.r2)
        r4 = getattr(self.r1_f32, op2)(self.r2)
        assert r3.raster_equal(r4)

        # Test with nodata set
        r3 = getattr(self.r1_nodata, op1)(self.r2)
        r4 = getattr(self.r1_nodata, op2)(self.r2)
        assert r3.raster_equal(r4)

        # Test with zeros values (e.g. division)
        r3 = getattr(self.r1, op1)(self.r2_zero)
        r4 = getattr(self.r1, op2)(self.r2_zero)
        assert r3.raster_equal(r4)

        # Test with a numpy array
        r3 = getattr(self.r1, op1)(array)
        r4 = getattr(self.r1, op2)(array)
        assert r3.raster_equal(r4)

        # Test with an integer
        r3 = getattr(self.r1, op1)(intval)
        r4 = getattr(self.r1, op2)(intval)
        assert r3.raster_equal(r4)

        # Test with a float value
        r3 = getattr(self.r1, op1)(floatval)
        r4 = getattr(self.r1, op2)(floatval)
        assert r3.raster_equal(r4)

    @classmethod
    def from_array(
        cls: type[TestArithmetic],
        data: NDArrayNum | MArrayNum,
        rst_ref: gu.RasterType,
        nodata: int | float | list[int] | list[float] | None = None,
    ) -> gu.Raster:
        """
        Generate a Raster from numpy array, with set georeferencing. Used for testing only.
        """
        if nodata is None:
            nodata = rst_ref.nodata

        return gu.Raster.from_array(data, crs=rst_ref.crs, transform=rst_ref.transform, nodata=nodata)

    def test_ops_2args_implicit(self) -> None:
        """
        Test certain arithmetic overloading when called with symbols (+, -, *, /, //, %).
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
        assert (r1 + r2).raster_equal(self.from_array(r1.data + r2.data, rst_ref=r1))
        assert (r1_f32 + r2).raster_equal(self.from_array(r1_f32.data + r2.data, rst_ref=r1))
        assert (array_3d + r2).raster_equal(self.from_array(array_3d + r2.data, rst_ref=r2))
        assert (r2 + array_3d).raster_equal(self.from_array(r2.data + array_3d, rst_ref=r2))
        assert (array_2d + r2).raster_equal(self.from_array(array_2d[np.newaxis, :, :] + r2.data, rst_ref=r2))
        assert (r2 + array_2d).raster_equal(self.from_array(r2.data + array_2d[np.newaxis, :, :], rst_ref=r2))
        assert (r1 + floatval).raster_equal(self.from_array(r1.data + floatval, rst_ref=r1))
        assert (floatval + r1).raster_equal(self.from_array(floatval + r1.data, rst_ref=r1))
        assert (r1 + r2).raster_equal(r2 + r1)

        # Multiplication
        assert (r1 * r2).raster_equal(self.from_array(r1.data * r2.data, rst_ref=r1))
        assert (r1_f32 * r2).raster_equal(self.from_array(r1_f32.data * r2.data, rst_ref=r1))
        assert (array_3d * r2).raster_equal(self.from_array(array_3d * r2.data, rst_ref=r2))
        assert (r2 * array_3d).raster_equal(self.from_array(r2.data * array_3d, rst_ref=r2))
        assert (array_2d * r2).raster_equal(self.from_array(array_2d[np.newaxis, :, :] * r2.data, rst_ref=r2))
        assert (r2 * array_2d).raster_equal(self.from_array(r2.data * array_2d[np.newaxis, :, :], rst_ref=r2))
        assert (r1 * floatval).raster_equal(self.from_array(r1.data * floatval, rst_ref=r1))
        assert (floatval * r1).raster_equal(self.from_array(floatval * r1.data, rst_ref=r1))
        assert (r1 * r2).raster_equal(r2 * r1)

        # Subtraction
        assert (r1 - r2).raster_equal(self.from_array(r1.data - r2.data, rst_ref=r1))
        assert (r1_f32 - r2).raster_equal(self.from_array(r1_f32.data - r2.data, rst_ref=r1))
        assert (array_3d - r2).raster_equal(self.from_array(array_3d - r2.data, rst_ref=r2))
        assert (r2 - array_3d).raster_equal(self.from_array(r2.data - array_3d, rst_ref=r2))
        assert (array_2d - r2).raster_equal(self.from_array(array_2d[np.newaxis, :, :] - r2.data, rst_ref=r2))
        assert (r2 - array_2d).raster_equal(self.from_array(r2.data - array_2d[np.newaxis, :, :], rst_ref=r2))
        assert (r1 - floatval).raster_equal(self.from_array(r1.data - floatval, rst_ref=r1))
        assert (floatval - r1).raster_equal(self.from_array(floatval - r1.data, rst_ref=r1))

        # True division
        assert (r1 / r2).raster_equal(self.from_array(r1.data / r2.data, rst_ref=r1))
        assert (r1_f32 / r2).raster_equal(self.from_array(r1_f32.data / r2.data, rst_ref=r1))
        assert (array_3d / r2).raster_equal(self.from_array(array_3d / r2.data, rst_ref=r2))
        assert (r2 / array_3d).raster_equal(self.from_array(r2.data / array_3d, rst_ref=r2))
        assert (array_2d / r2).raster_equal(self.from_array(array_2d[np.newaxis, :, :] / r2.data, rst_ref=r1))
        assert (r2 / array_2d).raster_equal(self.from_array(r2.data / array_2d[np.newaxis, :, :], rst_ref=r2))
        assert (r1 / floatval).raster_equal(self.from_array(r1.data / floatval, rst_ref=r1))
        assert (floatval / r1).raster_equal(self.from_array(floatval / r1.data, rst_ref=r1))

        # Floor division
        assert (r1 // r2).raster_equal(self.from_array(r1.data // r2.data, rst_ref=r1))
        assert (r1_f32 // r2).raster_equal(self.from_array(r1_f32.data // r2.data, rst_ref=r1))
        assert (array_3d // r2).raster_equal(self.from_array(array_3d // r2.data, rst_ref=r1))
        assert (r2 // array_3d).raster_equal(self.from_array(r2.data // array_3d, rst_ref=r1))
        assert (array_2d // r2).raster_equal(self.from_array(array_2d[np.newaxis, :, :] // r2.data, rst_ref=r1))
        assert (r2 // array_2d).raster_equal(self.from_array(r2.data // array_2d[np.newaxis, :, :], rst_ref=r1))
        assert (r1 // floatval).raster_equal(self.from_array(r1.data // floatval, rst_ref=r1))
        assert (floatval // r1).raster_equal(self.from_array(floatval // r1.data, rst_ref=r1))

        # Modulo
        assert (r1 % r2).raster_equal(self.from_array(r1.data % r2.data, rst_ref=r1))
        assert (r1_f32 % r2).raster_equal(self.from_array(r1_f32.data % r2.data, rst_ref=r1))
        assert (array_3d % r2).raster_equal(self.from_array(array_3d % r2.data, rst_ref=r1))
        assert (r2 % array_3d).raster_equal(self.from_array(r2.data % array_3d, rst_ref=r1))
        assert (array_2d % r2).raster_equal(self.from_array(array_2d[np.newaxis, :, :] % r2.data, rst_ref=r1))
        assert (r2 % array_2d).raster_equal(self.from_array(r2.data % array_2d[np.newaxis, :, :], rst_ref=r1))
        assert (r1 % floatval).raster_equal(self.from_array(r1.data % floatval, rst_ref=r1))

    def test_ops_logical_implicit(self) -> None:
        """
        Test logical arithmetic overloading when called with symbols (==, !=, <, <=, >, >=).
        """
        warnings.filterwarnings("ignore", message="invalid value encountered")

        # Test various inputs: Raster with different dtypes, np.ndarray with 2D or 3D shape, single number
        r1 = self.r1
        r1_f32 = self.r1_f32
        r2 = self.r2
        array_3d = np.random.randint(1, 255, (1, self.height, self.width)).astype("uint8")
        array_2d = np.random.randint(1, 255, (self.height, self.width)).astype("uint8")
        floatval = 3.14

        # Equality
        assert (r1 == r2).raster_equal(self.from_array(r1.data == r2.data, rst_ref=r1))
        assert (r1_f32 == r2).raster_equal(self.from_array(r1_f32.data == r2.data, rst_ref=r1))
        assert (array_3d == r2).raster_equal(self.from_array(array_3d == r2.data, rst_ref=r2))
        assert (r2 == array_3d).raster_equal(self.from_array(r2.data == array_3d, rst_ref=r2))
        assert (array_2d == r2).raster_equal(self.from_array(array_2d[np.newaxis, :, :] == r2.data, rst_ref=r2))
        assert (r2 == array_2d).raster_equal(self.from_array(r2.data == array_2d[np.newaxis, :, :], rst_ref=r2))
        assert (r1 == floatval).raster_equal(self.from_array(r1.data == floatval, rst_ref=r1))
        assert (floatval == r1).raster_equal(self.from_array(floatval == r1.data, rst_ref=r1))
        assert (r1 == r2).raster_equal(r2 == r1)

        # Non-equality
        assert (r1 != r2).raster_equal(self.from_array(r1.data != r2.data, rst_ref=r1))
        assert (r1_f32 != r2).raster_equal(self.from_array(r1_f32.data != r2.data, rst_ref=r1))
        assert (array_3d != r2).raster_equal(self.from_array(array_3d != r2.data, rst_ref=r2))
        assert (r2 != array_3d).raster_equal(self.from_array(r2.data != array_3d, rst_ref=r2))
        assert (array_2d != r2).raster_equal(self.from_array(array_2d[np.newaxis, :, :] != r2.data, rst_ref=r2))
        assert (r2 != array_2d).raster_equal(self.from_array(r2.data != array_2d[np.newaxis, :, :], rst_ref=r2))
        assert (r1 != floatval).raster_equal(self.from_array(r1.data != floatval, rst_ref=r1))
        assert (floatval != r1).raster_equal(self.from_array(floatval != r1.data, rst_ref=r1))
        assert (r1 != r2).raster_equal(r2 != r1)

        # Lower than
        assert (r1 < r2).raster_equal(self.from_array(r1.data < r2.data, rst_ref=r1))
        assert (r1_f32 < r2).raster_equal(self.from_array(r1_f32.data < r2.data, rst_ref=r1))
        assert (array_3d < r2).raster_equal(self.from_array(array_3d < r2.data, rst_ref=r2))
        assert (r2 < array_3d).raster_equal(self.from_array(r2.data < array_3d, rst_ref=r2))
        assert (array_2d < r2).raster_equal(self.from_array(array_2d[np.newaxis, :, :] < r2.data, rst_ref=r2))
        assert (r2 < array_2d).raster_equal(self.from_array(r2.data < array_2d[np.newaxis, :, :], rst_ref=r2))
        assert (r1 < floatval).raster_equal(self.from_array(r1.data < floatval, rst_ref=r1))
        assert (floatval < r1).raster_equal(self.from_array(floatval < r1.data, rst_ref=r1))

        # Lower equal
        assert (r1 <= r2).raster_equal(self.from_array(r1.data <= r2.data, rst_ref=r1))
        assert (r1_f32 <= r2).raster_equal(self.from_array(r1_f32.data <= r2.data, rst_ref=r1))
        assert (array_3d <= r2).raster_equal(self.from_array(array_3d <= r2.data, rst_ref=r2))
        assert (r2 <= array_3d).raster_equal(self.from_array(r2.data <= array_3d, rst_ref=r2))
        assert (array_2d <= r2).raster_equal(self.from_array(array_2d[np.newaxis, :, :] <= r2.data, rst_ref=r1))
        assert (r2 <= array_2d).raster_equal(self.from_array(r2.data <= array_2d[np.newaxis, :, :], rst_ref=r2))
        assert (r1 <= floatval).raster_equal(self.from_array(r1.data <= floatval, rst_ref=r1))
        assert (floatval <= r1).raster_equal(self.from_array(floatval <= r1.data, rst_ref=r1))

        # Greater than
        assert (r1 > r2).raster_equal(self.from_array(r1.data > r2.data, rst_ref=r1))
        assert (r1_f32 > r2).raster_equal(self.from_array(r1_f32.data > r2.data, rst_ref=r1))
        assert (array_3d > r2).raster_equal(self.from_array(array_3d > r2.data, rst_ref=r1))
        assert (r2 > array_3d).raster_equal(self.from_array(r2.data > array_3d, rst_ref=r1))
        assert (array_2d > r2).raster_equal(self.from_array(array_2d[np.newaxis, :, :] > r2.data, rst_ref=r1))
        assert (r2 > array_2d).raster_equal(self.from_array(r2.data > array_2d[np.newaxis, :, :], rst_ref=r1))
        assert (r1 > floatval).raster_equal(self.from_array(r1.data > floatval, rst_ref=r1))
        assert (floatval > r1).raster_equal(self.from_array(floatval > r1.data, rst_ref=r1))

        # Greater equal
        assert (r1 >= r2).raster_equal(self.from_array(r1.data >= r2.data, rst_ref=r1))
        assert (r1_f32 >= r2).raster_equal(self.from_array(r1_f32.data >= r2.data, rst_ref=r1))
        assert (array_3d >= r2).raster_equal(self.from_array(array_3d >= r2.data, rst_ref=r1))
        assert (r2 >= array_3d).raster_equal(self.from_array(r2.data >= array_3d, rst_ref=r1))
        assert (array_2d >= r2).raster_equal(self.from_array(array_2d[np.newaxis, :, :] >= r2.data, rst_ref=r1))
        assert (r2 >= array_2d).raster_equal(self.from_array(r2.data >= array_2d[np.newaxis, :, :], rst_ref=r1))
        assert (r1 >= floatval).raster_equal(self.from_array(r1.data >= floatval, rst_ref=r1))

    def test_ops_logical_bitwise_implicit(self) -> None:
        # Create two masks
        r1 = self.r1
        m1 = self.r1 > 128
        m2 = self.r2 > 128
        array_2d = np.random.randint(1, 255, (self.height, self.width)).astype("uint8") > 128

        # Bitwise or
        assert (m1 | m2).raster_equal(self.from_array(m1.data | m2.data, rst_ref=r1))
        assert (m1 | array_2d).raster_equal(self.from_array(m1.data | array_2d, rst_ref=r1))
        assert (array_2d | m1).raster_equal(self.from_array(array_2d | m1.data, rst_ref=r1))

        # Bitwise and
        assert (m1 & m2).raster_equal(self.from_array(m1.data & m2.data, rst_ref=r1))
        assert (m1 & array_2d).raster_equal(self.from_array(m1.data & array_2d, rst_ref=r1))
        assert (array_2d & m1).raster_equal(self.from_array(array_2d & m1.data, rst_ref=r1))

        # Bitwise xor
        assert (m1 ^ m2).raster_equal(self.from_array(m1.data ^ m2.data, rst_ref=r1))
        assert (m1 ^ array_2d).raster_equal(self.from_array(m1.data ^ array_2d, rst_ref=r1))
        assert (array_2d ^ m1).raster_equal(self.from_array(array_2d ^ m1.data, rst_ref=r1))

        # Bitwise invert
        assert (~m1).raster_equal(self.from_array(~m1.data, rst_ref=r1))

    @pytest.mark.parametrize("op", ops_2args)  # type: ignore
    def test_raise_errors(self, op: str) -> None:
        """
        Test that errors are properly raised in certain situations.

        !! Important !! Here we test errors with the operator on the raster only (arithmetic overloading),
        calling with array first is supported with the NumPy interface and tested in ArrayInterface.
        """
        # Rasters with different CRS, transform, or shape
        # Different shape
        expected_message = (
            "Both rasters must have the same shape, transform and CRS for an arithmetic operation. "
            "For example, use raster1 = raster1.reproject(raster2) to reproject raster1 on the "
            "same grid and CRS than raster2."
        )
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            getattr(self.r2, op)(self.r1_wrong_shape)

        # Different CRS
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            getattr(self.r2, op)(self.r1_wrong_crs)

        # Different transform
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            getattr(self.r2, op)(self.r1_wrong_transform)

        # Array with different shape
        expected_message = (
            "The raster and array must have the same shape for an arithmetic operation. "
            "For example, if the array comes from another raster, use raster1 = "
            "raster1.reproject(raster2) beforehand to reproject raster1 on the same grid and CRS "
            "than raster2. Or, if the array does not come from a raster, define one with raster = "
            "Raster.from_array(array, array_transform, array_crs, array_nodata) then reproject."
        )
        # Different shape, masked array
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            getattr(self.r2, op)(self.r1_wrong_shape.data)

        # Different shape, normal array with NaNs
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            getattr(self.r2, op)(self.r1_wrong_shape.data.filled(np.nan))

        # Wrong type of "other"
        expected_message = "Operation between an object of type .* and a Raster impossible."
        with pytest.raises(NotImplementedError, match=expected_message):
            getattr(self.r1, op)("some_string")

    @pytest.mark.parametrize("power", [2, 3.14, -1])  # type: ignore
    def test_power(self, power: float | int) -> None:
        if power > 0:  # Integers to negative integer powers are not allowed.
            assert self.r1**power == self.from_array(self.r1.data**power, rst_ref=self.r1)
        assert self.r1_f32**power == self.from_array(self.r1_f32.data**power, rst_ref=self.r1_f32)

    @pytest.mark.parametrize("dtype", ["float32", "uint8", "int32"])  # type: ignore
    def test_numpy_functions(self, dtype: str) -> None:
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

        # Check that rasters don't become arrays when using simple arithmetic.
        assert isinstance(raster + 1, gu.Raster)

        # Test the data setter method by creating a new array
        raster.data = array + 2

        # Check that the median updated accordingly.
        assert np.median(raster) == 14.0

        # Test
        raster += array

        assert isinstance(raster, gu.Raster)
        assert np.median(raster) == 26.0


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

    # Separate between two lists (single input and double input) for testing
    handled_functions_2in = gu.raster.raster._HANDLED_FUNCTIONS_2NIN
    handled_functions_1in = gu.raster.raster._HANDLED_FUNCTIONS_1NIN

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
    # This third one is to try ufunc methods like reduce()
    arr3 = np.random.randint(min_val, max_val, (height, width), dtype="int32") + np.random.normal(size=(height, width))

    # Create two random masks
    mask1 = np.random.randint(0, 2, size=(width, height), dtype=bool)
    mask2 = np.random.randint(0, 2, size=(width, height), dtype=bool)
    mask3 = np.random.randint(0, 2, size=(width, height), dtype=bool)

    # Assert that there is at least one unmasked value
    assert np.count_nonzero(~mask1) > 0
    assert np.count_nonzero(~mask2) > 0
    assert np.count_nonzero(~mask3) > 0

    # Wrong shaped arrays to check errors are raised
    arr_wrong_shape = np.random.randint(min_val, max_val, (height - 1, width - 1), dtype="int32") + np.random.normal(
        size=(height - 1, width - 1)
    )
    wrong_transform = rio.transform.from_bounds(0, 0, 1, 1, width - 1, height - 1)
    mask_wrong_shape = np.random.randint(0, 2, size=(width - 1, height - 1), dtype=bool)

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
            warnings.filterwarnings(
                "ignore", category=UserWarning, message="Setting default nodata -99999 to mask non-finite values*"
            )

        # Create Raster
        with warnings.catch_warnings():
            # For integer data types, unmasked nodata values can be created
            if "int" in str(dtype):
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="Unmasked values equal to the nodata value found in data array.*",
                )
            ma1 = np.ma.masked_array(data=self.arr1.astype(dtype), mask=self.mask1)
            rst = gu.Raster.from_array(ma1, transform=self.transform, crs=None, nodata=nodata)

        # Get ufunc
        ufunc = getattr(np, ufunc_str)

        # Find the common dtype between the Raster and the most constrained input type (first character is the input)
        try:
            com_dtype = np.promote_types(dtype, ufunc.types[0][0])
        # The promote_types function raises an error for object dtypes (previously returned by find_common_dtypes)
        # (TypeError needed for backwards compatibility; also exceptions.DTypePromotionError for NumPy 1.25 and above)
        except TypeError:
            com_dtype = np.dtype("O")

        # Catch warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # Check if our input dtype is possible on this ufunc, if yes check that outputs are identical
            if com_dtype in [str(np.dtype(t[0])) for t in ufunc.types]:  # noqa
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

        with warnings.catch_warnings():
            if any("int" in dtype for dtype in [str(dtype1), str(dtype2)]):
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="Unmasked values equal to the nodata value found in data array.*",
                )
            ma1 = np.ma.masked_array(data=self.arr1.astype(dtype1), mask=self.mask1)
            ma2 = np.ma.masked_array(data=self.arr2.astype(dtype2), mask=self.mask2)
            rst1 = gu.Raster.from_array(ma1, transform=self.transform, crs=None, nodata=nodata1)
            rst2 = gu.Raster.from_array(ma2, transform=self.transform, crs=None, nodata=nodata2)

        ufunc = getattr(np, ufunc_str)

        # Find the common dtype between the Raster and the most constrained input type (first character is the input)
        try:
            com_dtype1 = np.promote_types(dtype1, ufunc.types[0][0])
        # The promote_types function raises an error for object dtypes (previously returned by find_common_dtypes)
        # (TypeError needed for backwards compatibility; also exceptions.DTypePromotionError for NumPy 1.25 and above)
        except TypeError:
            com_dtype1 = np.dtype("O")

        try:
            com_dtype2 = np.promote_types(dtype2, ufunc.types[0][1])
        # The promote_types function raises an error for object dtypes (previously returned by find_common_dtypes)
        # (TypeError needed for backwards compatibility; also exceptions.DTypePromotionError for NumPy 1.25 and above)
        except TypeError:
            com_dtype2 = np.dtype("O")

        # If the two input types can be the same type, pass a tuple with the common type of both
        # Below we ignore datetime and timedelta types "m" and "M", and int64 types "q" and "Q"
        if all(t[0] == t[1] for t in ufunc.types if not any(x in t[0:2] for x in ["m", "M", "q", "Q"])):
            try:
                com_dtype_both = np.promote_types(com_dtype1, com_dtype2)
            except TypeError:
                com_dtype_both = np.dtype("O")
            com_dtype_tuple = (com_dtype_both, com_dtype_both)

        # Otherwise, pass the tuple with each common type
        else:
            com_dtype_tuple = (com_dtype1, com_dtype2)

        # Catch warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings(
                "ignore", category=UserWarning, message="Setting default nodata -99999 to mask non-finite values*"
            )

            # Check if both our input dtypes are possible on this ufunc, if yes check that outputs are identical
            if com_dtype_tuple in [(np.dtype(t[0]), np.dtype(t[1])) for t in ufunc.types]:  # noqa
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

    @pytest.mark.parametrize("arrfunc_str", handled_functions_1in)  # type: ignore
    @pytest.mark.parametrize(
        "dtype", ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64", "longdouble"]
    )  # type: ignore
    @pytest.mark.parametrize("nodata_init", [None, "type_default"])  # type: ignore
    def test_array_functions_1nin(self, arrfunc_str: str, dtype: str, nodata_init: None | str) -> None:
        """
        Test that single-input array functions that we support give the same output as they would on the masked array.
        """

        # We set the default nodata
        if nodata_init == "type_default":
            nodata: int | None = _default_nodata(dtype)
        else:
            nodata = None

        # Create Raster
        with warnings.catch_warnings():
            # For integer data types, unmasked nodata values can be created
            if "int" in str(dtype):
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="Unmasked values equal to the nodata value found in data array.*",
                )
            ma1 = np.ma.masked_array(data=self.arr1.astype(dtype), mask=self.mask1)
            rst = gu.Raster.from_array(ma1, transform=self.transform, crs=None, nodata=nodata)

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
            elif "gradient" in arrfunc_str:
                # For the gradient, we need to take a single band
                output_rst = arrfunc(rst)
                output_ma = np.gradient(rst.data)
            else:
                output_rst = arrfunc(rst)
                output_ma = arrfunc(rst.data)

            # Gradient is the only supported array function returning two arguments for now
            if "gradient" in arrfunc_str:
                assert np.ma.allequal(output_rst[0], output_ma[0]) and np.ma.allequal(output_rst[1], output_ma[1])
            # This test is for when the NumPy function reduces the dimension of the array but not completely
            elif isinstance(output_ma, np.ndarray):
                # When the NumPy function preserves the shape, it returns a Raster
                if output_ma.shape == rst.data.shape:
                    assert isinstance(output_rst, gu.Raster)
                    assert np.ma.allequal(output_rst.data, output_ma)
                # Otherwise, it returns an array
                else:
                    assert np.ma.allequal(output_rst, output_ma)
            # This test is for when the NumPy function reduces the dimension to a single number
            else:
                assert output_rst == output_ma

    @pytest.mark.parametrize("arrfunc_str", handled_functions_2in)  # type: ignore
    @pytest.mark.parametrize(
        "dtype1", ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64", "longdouble"]
    )  # type: ignore
    @pytest.mark.parametrize(
        "dtype2", ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64", "longdouble"]
    )  # type: ignore
    @pytest.mark.parametrize("nodata1_init", [None, "type_default"])  # type: ignore
    @pytest.mark.parametrize("nodata2_init", [None, "type_default"])  # type: ignore
    def test_array_functions_2nin(
        self, arrfunc_str: str, nodata1_init: None | str, nodata2_init: str, dtype1: str, dtype2: str
    ) -> None:
        """
        Test that double-input array functions that we support give the same output as they would on the masked array.
        """

        # We set the default nodatas
        if nodata1_init == "type_default":
            nodata1: int | None = _default_nodata(dtype1)
        else:
            nodata1 = None
        if nodata2_init == "type_default":
            nodata2: int | None = _default_nodata(dtype2)
        else:
            nodata2 = None

        with warnings.catch_warnings():
            # For integer data types, unmasked nodata values can be created
            if any("int" in dtype for dtype in [str(dtype1), str(dtype2)]):
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="Unmasked values equal to the nodata value found in data array.*",
                )
            ma1 = np.ma.masked_array(data=self.arr1.astype(dtype1), mask=self.mask1)
            ma2 = np.ma.masked_array(data=self.arr2.astype(dtype2), mask=self.mask2)
            rst1 = gu.Raster.from_array(ma1, transform=self.transform, crs=None, nodata=nodata1)
            rst2 = gu.Raster.from_array(ma2, transform=self.transform, crs=None, nodata=nodata2)

        # Get array func
        arrfunc = getattr(np, arrfunc_str)

        # Catch warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # Compute outputs
            output_rst = arrfunc(rst1, rst2)
            output_ma = arrfunc(rst1.data, rst2.data)

            # When the NumPy function preserves the shape, it returns a Raster
            if isinstance(output_ma, np.ndarray) and output_ma.shape == rst1.data.shape:
                assert isinstance(output_rst, gu.Raster)
                assert np.ma.allequal(output_rst.data, output_ma)
            # Otherwise, it returns an array
            else:
                assert np.ma.allequal(output_rst, output_ma)

    @pytest.mark.parametrize("method_str", ["reduce"])  # type: ignore
    def test_ufunc_methods(self, method_str):
        """
        Test that universal function methods all behave properly, don't need to test all
        nodatas and dtypes as this was done above.
        """

        ma1 = np.ma.masked_array(data=self.arr1.astype("float32"), mask=self.mask1)
        ma2 = np.ma.masked_array(data=self.arr2.astype("float32"), mask=self.mask2)
        ma3 = np.ma.masked_array(data=self.arr3.astype("float32"), mask=self.mask3)

        rst1 = gu.Raster.from_array(ma1, transform=self.transform, crs=None, nodata=_default_nodata("float32"))
        rst2 = gu.Raster.from_array(ma2, transform=self.transform, crs=None, nodata=_default_nodata("float32"))
        rst3 = gu.Raster.from_array(ma3, transform=self.transform, crs=None, nodata=_default_nodata("float32"))

        # Methods reduce, accumulate, reduceat and at only supported for binary function (2nin)
        # -- Test 1: -- Try a ufunc with 2nin, 1nout like np.add
        ufunc_2nin_1nout = getattr(np.add, method_str)
        output_rst = ufunc_2nin_1nout((rst1, rst2, rst3))
        output_ma = ufunc_2nin_1nout((ma1, ma2, ma3))

        assert np.ma.allequal(output_rst.data, output_ma)

        # Methods reduce only supports function that output a single value
        # -- Test 2: -- Try a ufunc with 2nin, 2nout: there's only divmod
        # ufunc_2nin_2nout = getattr(np.divmod, method_str)
        # outputs_rst = ufunc_2nin_2nout((rst1, rst2, rst3))
        # outputs_ma = ufunc_2nin_2nout((ma1, ma2, ma3))
        #
        # assert np.ma.allequal(outputs_ma[0], outputs_rst[0].data) and np.ma.allequal(
        #             outputs_ma[1], outputs_rst[1].data)

    @pytest.mark.parametrize(
        "np_func_name", ufuncs_str_2nin_1nout + ufuncs_str_2nin_2nout + handled_functions_2in
    )  # type: ignore
    def test_raise_errors_2nin(self, np_func_name: str) -> None:
        """Check that proper errors are raised when input raster/array don't match (only 2-input functions)."""

        # Create Rasters
        ma = np.ma.masked_array(data=self.arr1, mask=self.mask1)
        ma_wrong_shape = np.ma.masked_array(data=self.arr_wrong_shape, mask=self.mask_wrong_shape)
        rst = gu.Raster.from_array(ma, transform=self.transform, crs=4326, nodata=_default_nodata(ma.dtype))
        rst_wrong_shape = gu.Raster.from_array(
            ma_wrong_shape, transform=self.transform, crs=4326, nodata=_default_nodata(ma_wrong_shape.dtype)
        )
        rst_wrong_crs = gu.Raster.from_array(ma, transform=self.transform, crs=32610, nodata=_default_nodata(ma.dtype))
        rst_wrong_transform = gu.Raster.from_array(
            ma, transform=self.wrong_transform, crs=4326, nodata=_default_nodata(ma_wrong_shape.dtype)
        )

        # Get ufunc
        np_func = getattr(np, np_func_name)

        # Strange errors happening only for these 4 functions...
        # See issue #457
        if np_func_name not in ["allclose", "isclose", "array_equal", "array_equiv"]:

            # Rasters with different CRS, transform, or shape
            # Different shape
            expected_message = (
                "Both rasters must have the same shape, transform and CRS for an arithmetic operation. "
                "For example, use raster1 = raster1.reproject(raster2) to reproject raster1 on the "
                "same grid and CRS than raster2."
            )

            with pytest.raises(ValueError, match=re.escape(expected_message)):
                np_func(rst, rst_wrong_shape)

            # Different CRS
            with pytest.raises(ValueError, match=re.escape(expected_message)):
                np_func(rst, rst_wrong_crs)

            # Different transform
            with pytest.raises(ValueError, match=re.escape(expected_message)):
                np_func(rst, rst_wrong_transform)

            # Array with different shape
            expected_message = (
                "The raster and array must have the same shape for an arithmetic operation. "
                "For example, if the array comes from another raster, use raster1 = "
                "raster1.reproject(raster2) beforehand to reproject raster1 on the same grid and CRS "
                "than raster2. Or, if the array does not come from a raster, define one with raster = "
                "Raster.from_array(array, array_transform, array_crs, array_nodata) then reproject."
            )
            # Different shape, masked array
            # Check reflectivity just in case (just here, not later)
            with pytest.raises(ValueError, match=re.escape(expected_message)):
                np_func(ma_wrong_shape, rst)
            with pytest.raises(ValueError, match=re.escape(expected_message)):
                np_func(rst, ma_wrong_shape)

            # Different shape, normal array with NaNs
            with pytest.raises(ValueError, match=re.escape(expected_message)):
                np_func(ma_wrong_shape.filled(np.nan), rst)
            with pytest.raises(ValueError, match=re.escape(expected_message)):
                np_func(rst, ma_wrong_shape.filled(np.nan))
