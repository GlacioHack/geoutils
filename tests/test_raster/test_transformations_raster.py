"""Test for geotransformations of raster objects."""

from __future__ import annotations

import re
import warnings
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio as rio
from pyproj import CRS
from packaging.version import Version

import geoutils as gu
from geoutils import examples
from geoutils.exceptions import InvalidGridError
from geoutils.raster.transformations import _resampling_method_from_str, _dask_reproject, _rio_reproject
from geoutils.raster.raster import _default_nodata
from geoutils.multiproc import ClusterGenerator, MultiprocConfig
from geoutils.stats.sampling import _subsample_numpy

DO_PLOT = False


class TestRasterTransformations:

    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    landsat_b4_crop_path = examples.get_path_test("everest_landsat_b4_cropped")
    landsat_rgb_path = examples.get_path_test("everest_landsat_rgb")
    everest_outlines_path = examples.get_path_test("everest_rgi_outlines")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")
    aster_outlines_path = examples.get_path_test("exploradores_rgi_outlines")

    def test_resampling_str(self) -> None:
        """Test that resampling methods can be given as strings instead of rio enums."""
        warnings.simplefilter("error")
        assert _resampling_method_from_str("nearest") == rio.enums.Resampling.nearest  # noqa
        assert _resampling_method_from_str("cubic_spline") == rio.enums.Resampling.cubic_spline  # noqa

        # Check that odd strings return the appropriate error.
        try:
            _resampling_method_from_str("CUBIC_SPLINE")  # noqa
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

    test_data = [[landsat_b4_path, everest_outlines_path], [aster_dem_path, aster_outlines_path]]

    @pytest.mark.parametrize("data", test_data)
    def test_crop(self, data: list[str]) -> None:
        """Test for crop method, also called by square brackets through __getitem__"""

        raster_path, outlines_path = data
        r = gu.Raster(raster_path)

        # -- Test with bbox being a list/tuple -- ##
        bbox: list[float] = list(r.bounds)

        # Test unloaded inplace cropping conserves the shape
        r.crop(bbox=[bbox[0] + r.res[0], bbox[1], bbox[2], bbox[3]], inplace=True)
        assert len(r.data.shape) == 2

        r = gu.Raster(raster_path)

        # Test with same bounds -> should be the same #
        bbox2 = [bbox[0], bbox[1], bbox[2], bbox[3]]
        r_cropped = r.crop(bbox2)
        assert r_cropped.raster_equal(r)

        # - Test cropping each side by a random integer of pixels - #
        rng = np.random.default_rng(42)
        rand_int = rng.integers(1, min(r.shape) - 1)

        # Left
        bbox2 = [bbox[0] + rand_int * r.res[0], bbox[1], bbox[2], bbox[3]]
        r_cropped = r.crop(bbox2)
        assert list(r_cropped.bounds) == bbox2
        assert np.array_equal(r.data[:, rand_int:].data, r_cropped.data.data, equal_nan=True)
        assert np.array_equal(r.data[:, rand_int:].mask, r_cropped.data.mask)

        # Right
        bbox2 = [bbox[0], bbox[1], bbox[2] - rand_int * r.res[0], bbox[3]]
        r_cropped = r.crop(bbox2)
        assert list(r_cropped.bounds) == bbox2
        assert np.array_equal(r.data[:, :-rand_int].data, r_cropped.data.data, equal_nan=True)
        assert np.array_equal(r.data[:, :-rand_int].mask, r_cropped.data.mask)

        # Bottom
        bbox2 = [bbox[0], bbox[1] + rand_int * abs(r.res[1]), bbox[2], bbox[3]]
        r_cropped = r.crop(bbox2)
        assert list(r_cropped.bounds) == bbox2
        assert np.array_equal(r.data[:-rand_int, :].data, r_cropped.data.data, equal_nan=True)
        assert np.array_equal(r.data[:-rand_int, :].mask, r_cropped.data.mask)

        # Top
        bbox2 = [bbox[0], bbox[1], bbox[2], bbox[3] - rand_int * abs(r.res[1])]
        r_cropped = r.crop(bbox2)
        assert list(r_cropped.bounds) == bbox2
        assert np.array_equal(r.data[rand_int:, :].data, r_cropped.data, equal_nan=True)
        assert np.array_equal(r.data[rand_int:, :].mask, r_cropped.data.mask)

        # Same but tuple
        bbox3: tuple[float, float, float, float] = (
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3] - rand_int * r.res[0],
        )
        r_cropped = r.crop(bbox3)
        assert list(r_cropped.bounds) == list(bbox3)
        assert np.array_equal(r.data[rand_int:, :].data, r_cropped.data.data, equal_nan=True)
        assert np.array_equal(r.data[rand_int:, :].mask, r_cropped.data.mask)

        # -- Test with bbox being a Raster -- #
        r_cropped2 = r.crop(r_cropped)
        assert r_cropped2.raster_equal(r_cropped)

        # Check that bound reprojection is done automatically if the CRS differ
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="For reprojection, nodata must be set.*")

            r_cropped_reproj = r_cropped.reproject(crs=3857)
            r_cropped3 = r.crop(r_cropped_reproj)

        # Original CRS bounds can be deformed during transformation, but result should be equivalent to this
        r_cropped4 = r.crop(bbox=r_cropped_reproj.get_bounds_projected(out_crs=r.crs))
        assert r_cropped3.raster_equal(r_cropped4)

        # -- Test with inplace=True -- #
        r_copy = r.copy()
        r_copy.crop(r_cropped, inplace=True)
        assert r_copy.raster_equal(r_cropped)

        # - Test cropping each side with a non integer pixel, mode='match_pixel' - #
        rand_float = rng.integers(1, min(r.shape) - 1) + 0.25

        # left
        bbox2 = [bbox[0] + rand_float * r.res[0], bbox[1], bbox[2], bbox[3]]
        r_cropped = r.crop(bbox2)
        assert r.shape[1] - (r_cropped.bounds.right - r_cropped.bounds.left) / r.res[0] == int(rand_float)
        assert np.array_equal(r.data[:, int(rand_float) :].data, r_cropped.data.data, equal_nan=True)
        assert np.array_equal(r.data[:, int(rand_float) :].mask, r_cropped.data.mask)

        # right
        bbox2 = [bbox[0], bbox[1], bbox[2] - rand_float * r.res[0], bbox[3]]
        r_cropped = r.crop(bbox2)
        assert r.shape[1] - (r_cropped.bounds.right - r_cropped.bounds.left) / r.res[0] == int(rand_float)
        assert np.array_equal(r.data[:, : -int(rand_float)].data, r_cropped.data.data, equal_nan=True)
        assert np.array_equal(r.data[:, : -int(rand_float)].mask, r_cropped.data.mask)

        # bottom
        bbox2 = [bbox[0], bbox[1] + rand_float * abs(r.res[1]), bbox[2], bbox[3]]
        r_cropped = r.crop(bbox2)
        assert r.shape[0] - (r_cropped.bounds.top - r_cropped.bounds.bottom) / r.res[1] == int(rand_float)
        assert np.array_equal(r.data[: -int(rand_float), :].data, r_cropped.data.data, equal_nan=True)
        assert np.array_equal(r.data[: -int(rand_float), :].mask, r_cropped.data.mask)

        # top
        bbox2 = [bbox[0], bbox[1], bbox[2], bbox[3] - rand_float * abs(r.res[1])]
        r_cropped = r.crop(bbox2)
        assert r.shape[0] - (r_cropped.bounds.top - r_cropped.bounds.bottom) / r.res[1] == int(rand_float)
        assert np.array_equal(r.data[int(rand_float) :, :].data, r_cropped.data.data, equal_nan=True)
        assert np.array_equal(r.data[int(rand_float) :, :].mask, r_cropped.data.mask)

        # -- Test with bbox being a Vector -- #
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
        r_cropped2_bbox_reproj = r.crop(bbox=outlines.get_bounds_projected(out_crs=r.crs))
        assert list(r_cropped2.bounds) == list(r_cropped2_bbox_reproj.bounds)

        # -- Test crop works as expected even if transform has been modified, e.g. through downsampling -- #
        # Test that with downsampling, cropping to same bounds result in same raster
        r = gu.Raster(raster_path, downsample=5)
        r_test = r.crop(r.bounds)
        assert r_test.raster_equal(r)

        # - Test that cropping yields the same results whether data is loaded or not -
        # With integer cropping (left)
        rand_int = rng.integers(1, min(r.shape) - 1)
        bbox2 = [bbox[0] + rand_int * r.res[0], bbox[1], bbox[2], bbox[3]]
        r = gu.Raster(raster_path, downsample=5, load_data=False)
        assert not r.is_loaded
        r_crop_unloaded = r.crop(bbox2)
        r.load()
        r_crop_loaded = r.crop(bbox2)
        # TODO: the following condition should be met once issue #447 is solved
        # assert r_crop_unloaded.raster_equal(r_crop_loaded)
        assert r_crop_unloaded.shape == r_crop_loaded.shape
        assert r_crop_unloaded.transform == r_crop_loaded.transform

        # With a float number of pixels added to the right, mode 'match_pixel'
        rand_float = rng.integers(1, min(r.shape) - 1) + 0.25
        bbox2 = [bbox[0], bbox[1], bbox[2] + rand_float * r.res[0], bbox[3]]
        r = gu.Raster(raster_path, downsample=5, load_data=False)
        assert not r.is_loaded
        r_crop_unloaded = r.crop(bbox2)
        r.load()
        r_crop_loaded = r.crop(bbox2)
        # TODO: the following condition should be met once issue #447 is solved
        # assert r_crop_unloaded.raster_equal(r_crop_loaded)
        assert r_crop_unloaded.shape == r_crop_loaded.shape
        assert r_crop_unloaded.transform == r_crop_loaded.transform

        # - Check related to pixel interpretation -

        # Check warning for a different area_or_point for the match-reference geometry works
        r.set_area_or_point("Area", shift_area_or_point=False)
        r2 = r.copy()
        r2.set_area_or_point("Point", shift_area_or_point=False)

        with pytest.warns(UserWarning, match='One raster has a pixel interpretation "Area" and the other "Point".*'):
            r.crop(r2)

        # Check that cropping preserves the interpretation
        bbox = [bbox[0] + r.res[0], bbox[1], bbox[2], bbox[3]]
        r_crop = r.crop(bbox)
        assert r_crop.area_or_point == "Area"
        r2_crop = r2.crop(bbox)
        assert r2_crop.area_or_point == "Point"

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path, landsat_rgb_path])
    def test_translate(self, example: str) -> None:
        """Test translation works as intended"""

        r = gu.Raster(example)

        # Get original transform
        orig_transform = r.transform
        orig_bounds = r.bounds

        # Shift raster by georeferenced units (default)
        # Check the default behaviour is not inplace
        r_notinplace = r.translate(xoff=1, yoff=1)
        assert isinstance(r_notinplace, gu.Raster)

        # Check inplace
        r.translate(xoff=1, yoff=1, inplace=True)
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
        r.translate(xoff=1, yoff=1, distance_unit="pixel", inplace=True)

        # Only bounds should change
        assert orig_transform.c + 1 * orig_res[0] == r.transform.c
        assert orig_transform.f - 1 * orig_res[1] == r.transform.f
        for attr in ["a", "b", "d", "e"]:
            assert getattr(orig_transform, attr) == getattr(r.transform, attr)

        assert orig_bounds.left + 1 * orig_res[0] == r.bounds.left
        assert orig_bounds.right + 1 * orig_res[0] == r.bounds.right
        assert orig_bounds.bottom - 1 * orig_res[1] == r.bounds.bottom
        assert orig_bounds.top - 1 * orig_res[1] == r.bounds.top

        # Check that an error is raised for a wrong distance_unit
        with pytest.raises(ValueError, match="Argument 'distance_unit' should be either 'pixel' or 'georeferenced'."):
            r.translate(xoff=1, yoff=1, distance_unit="wrong_value")  # type: ignore

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])
    def test_reproject(self, example: str) -> None:

        # Reference raster to be used
        r = gu.Raster(example)

        # -- Check proper errors are raised if nodata are not set -- #
        r_nodata = r.copy()
        r_nodata.set_nodata(None)

        # Make sure at least one pixel is masked for test 1
        rand_indices = _subsample_numpy(r_nodata.data, 10, return_indices=True)
        r_nodata.data[rand_indices] = np.ma.masked
        assert np.count_nonzero(r_nodata.data.mask) > 0

        # Make sure at least one pixel is set at default nodata for test
        default_nodata = _default_nodata(r_nodata.dtype)
        rand_indices = _subsample_numpy(r_nodata.data, 10, return_indices=True)
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
                f"{_default_nodata(r_nodata.dtype)} exists in self.data. This may have unexpected "
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
        assert r_test.shape == (out_size[0], out_size[1])
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
        assert r_test.shape == (out_size[0], out_size[1])
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
        rng = np.random.default_rng(42)
        rand_int = rng.integers(min_size / 10, min(r.shape) - min_size / 10)
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
        rand_indices = _subsample_numpy(r_gaps.data, nsamples, return_indices=True)
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
        with pytest.raises(InvalidGridError, match="Either 'ref' or 'crs' must be provided"):
            _ = r.reproject(ref=r2, crs=r.crs)

        # Size and res are mutually exclusive
        with pytest.raises(InvalidGridError, match="Both output grid resolution 'res' and shape"):
            _ = r.reproject(grid_size=(10, 10), res=50)

        # If wrong type for `ref`
        with pytest.raises(InvalidGridError, match="Cannot interpret reference grid from"):
            _ = r.reproject(ref=3)

        # -- Check warning for area_or_point works -- #
        r.set_area_or_point("Area", shift_area_or_point=False)
        r2 = r.copy()
        r2.set_area_or_point("Point", shift_area_or_point=False)

        with (pytest.warns(UserWarning, match="One raster has a pixel"),):
            r.reproject(r2)

        # Check that reprojecting preserves interpretation
        r_reproj = r.reproject(res=r.res[0] * 2)
        assert r_reproj.area_or_point == "Area"
        r2_reproj = r2.reproject(res=r2.res[0] * 2)
        assert r2_reproj.area_or_point == "Point"


class TestMaskGeotransformations:
    # Paths to example data
    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    landsat_rgb_path = examples.get_path_test("everest_landsat_rgb")
    everest_outlines_path = examples.get_path_test("everest_rgi_outlines")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")

    # Mask without nodata
    mask_landsat_b4 = gu.Raster(landsat_b4_path) > 125
    # Mask with nodata
    mask_aster_dem = gu.Raster(aster_dem_path) > 2000
    # Mask from an outline
    mask_everest = gu.Vector(everest_outlines_path).create_mask(gu.Raster(landsat_b4_path))

    @pytest.mark.parametrize("mask", [mask_landsat_b4, mask_aster_dem, mask_everest])
    def test_crop(self, mask: gu.Raster) -> None:
        # Test with same bounds -> should be the same #

        mask_orig = mask.copy()
        bbox = mask.bounds
        mask_cropped = mask.crop(bbox)
        assert mask_cropped.raster_equal(mask)

        # Check if instance is respected
        assert isinstance(mask_cropped, gu.Raster)
        # Check the dtype of the original mask was properly reconverted
        assert mask.data.dtype == bool
        # Check the original mask was not modified during cropping
        assert mask_orig.raster_equal(mask)

        # Check inplace behaviour works
        mask_tmp = mask.copy()
        mask_tmp.crop(bbox, inplace=True)
        assert mask_tmp.raster_equal(mask_cropped)

        # - Test cropping each side by a random integer of pixels - #
        rng = np.random.default_rng(42)
        rand_int = rng.integers(1, min(mask.shape) - 1)

        # Left
        bbox2 = [bbox[0] + rand_int * mask.res[0], bbox[1], bbox[2], bbox[3]]
        mask_cropped = mask.crop(bbox2)
        assert list(mask_cropped.bounds) == bbox2
        assert np.array_equal(mask.data[:, rand_int:].data, mask_cropped.data.data, equal_nan=True)
        assert np.array_equal(mask.data[:, rand_int:].mask, mask_cropped.data.mask)

        #  With icrop
        bbox2_pixel = [rand_int, 0, mask.width, mask.height]
        mask_cropped_pix = mask.icrop(bbox2_pixel)
        assert mask_cropped.raster_equal(mask_cropped_pix)

        # Right
        bbox2 = [bbox[0], bbox[1], bbox[2] - rand_int * mask.res[0], bbox[3]]
        mask_cropped = mask.crop(bbox2)
        assert list(mask_cropped.bounds) == bbox2
        assert np.array_equal(mask.data[:, :-rand_int].data, mask_cropped.data.data, equal_nan=True)
        assert np.array_equal(mask.data[:, :-rand_int].mask, mask_cropped.data.mask)

        #  With icrop
        bbox2_pixel = [0, 0, mask.width - rand_int, mask.height]
        mask_cropped_pix = mask.icrop(bbox2_pixel)
        assert mask_cropped.raster_equal(mask_cropped_pix)

        # Bottom
        bbox2 = [bbox[0], bbox[1] + rand_int * abs(mask.res[1]), bbox[2], bbox[3]]
        mask_cropped = mask.crop(bbox2)
        assert list(mask_cropped.bounds) == bbox2
        assert np.array_equal(mask.data[:-rand_int, :].data, mask_cropped.data.data, equal_nan=True)
        assert np.array_equal(mask.data[:-rand_int, :].mask, mask_cropped.data.mask)

        #  With icrop
        bbox2_pixel = [0, 0, mask.width, mask.height - rand_int]
        mask_cropped_pix = mask.icrop(bbox2_pixel)
        assert mask_cropped.raster_equal(mask_cropped_pix)

        # Top
        bbox2 = [bbox[0], bbox[1], bbox[2], bbox[3] - rand_int * abs(mask.res[1])]
        mask_cropped = mask.crop(bbox2)
        assert list(mask_cropped.bounds) == bbox2
        assert np.array_equal(mask.data[rand_int:, :].data, mask_cropped.data, equal_nan=True)
        assert np.array_equal(mask.data[rand_int:, :].mask, mask_cropped.data.mask)

        #  With icrop
        bbox2_pixel = [0, rand_int, mask.width, mask.height]
        mask_cropped_pix = mask.icrop(bbox2_pixel)
        assert mask_cropped.raster_equal(mask_cropped_pix)

        # Test inplace
        mask_orig = mask.copy()
        mask_orig.crop(bbox2, inplace=True)
        assert list(mask_orig.bounds) == bbox2
        assert np.array_equal(mask.data[rand_int:, :].data, mask_orig.data, equal_nan=True)
        assert np.array_equal(mask.data[rand_int:, :].mask, mask_orig.data.mask)

        # With icrop
        mask_orig_pix = mask.copy()
        mask_orig_pix.icrop(bbox2_pixel, inplace=True)
        assert mask_orig.raster_equal(mask_orig_pix)

    @pytest.mark.parametrize("mask", [mask_landsat_b4, mask_aster_dem, mask_everest])
    def test_reproject(self, mask: gu.Raster) -> None:
        # Test 1: with a classic resampling (bilinear)

        # Reproject mask - resample to 100 x 100 grid
        mask_orig = mask.copy()
        mask_reproj = mask.reproject(grid_size=(100, 100), resampling="nearest")

        # Check instance is respected
        assert isinstance(mask_reproj, gu.Raster) and mask_reproj.is_mask
        # Check the dtype of the original mask was properly reconverted
        assert mask.data.dtype == bool
        # Check the original mask was not modified during reprojection
        assert mask_orig.raster_equal(mask)

        # Check inplace behaviour works
        mask_tmp = mask.copy()
        mask_tmp.reproject(grid_size=(100, 100), inplace=True, resampling="nearest")
        assert mask_tmp.raster_equal(mask_reproj)

        # This should be equivalent to converting the array to uint8, reprojecting, converting back
        mask_uint8 = mask.astype("uint8")
        mask_uint8_reproj = mask_uint8.reproject(grid_size=(100, 100), resampling="nearest")
        mask_uint8_reproj = mask_uint8_reproj.astype("bool")
        # The strict comparison ensures masked data are propagated exactly the same
        assert mask_reproj.raster_equal(mask_uint8_reproj, strict_masked=True)

        # Test 2: Should raise a warning when the resampling differs from nearest
        with pytest.raises(
            UserWarning, match=re.escape("Reprojecting a raster mask (boolean type) with a resampling method")
        ):
            mask.reproject(res=50, resampling="bilinear")

    def test_reproject__no_inters(self) -> None:
        """Test reprojection behaviour without intersection of inputs."""

        # Create two raster, one boolean
        dem_bool = gu.Raster.from_array(
            np.random.randint(2, size=(100, 100), dtype=bool),
            transform=rio.transform.from_origin(0, 100, 1, 1),
            crs=4326,
        )
        ref_dem = gu.Raster.from_array(
            np.random.randint(100, size=(100, 100), dtype="uint8"),
            transform=rio.transform.from_origin(0, 100, 1, 1),
            crs=4326,
        )

        # With no intersection
        dem_bool_crop = dem_bool.icrop((0, 0, 20, 20))
        ref_dem_crop = ref_dem.icrop((40, 40, 60, 60))

        assert not dem_bool_crop.get_footprint_projected(ref_dem_crop.crs).intersects(
            ref_dem_crop.get_footprint_projected(ref_dem_crop.crs)
        )[0]

        res = dem_bool_crop.reproject(ref_dem_crop, resampling="nearest")
        assert isinstance(res, gu.raster.raster.Raster)
        # All output points should be masked
        assert np.all(res.data.mask)
        # Default data value (behind the mask) is True
        assert np.all(res.data.data)

class TestMultiproc:

    aster_dem_path = examples.get_path_test("exploradores_aster_dem")
    landsat_rgb_path = examples.get_path_test("everest_landsat_rgb")
    num_workers = min(2, cpu_count())  # Safer limit for CI
    cluster = ClusterGenerator("test", nb_workers=num_workers)

    @pytest.mark.parametrize("example", [aster_dem_path])
    @pytest.mark.parametrize("tile_size", [20])
    @pytest.mark.parametrize("cluster", [None, cluster])
    def test_multiproc_reproject(self, example: str, tile_size: int, cluster: None | AbstractCluster) -> None:
        """Test for multiproc_reproject"""

        r = gu.Raster(example)
        config = MultiprocConfig(tile_size, cluster=cluster)

        # - Test reprojection with bounds and resolution -
        dst_bounds = rio.coords.BoundingBox(
            left=r.bounds.left, bottom=r.bounds.bottom + r.res[0], right=r.bounds.right - 2 * r.res[1], top=r.bounds.top
        )
        res_tuple = (r.res[0] * 0.5, r.res[1] * 3)

        # Multiprocessing reprojection
        r_multi = r.reproject(bounds=dst_bounds, res=res_tuple, mp_config=config)

        # Assert that the raster has not been loaded during reprojection
        assert not r.is_loaded

        # Single-process reprojection
        r_single = r.reproject(bounds=dst_bounds, res=res_tuple)

        # Assert the results are the same
        assert r_single.raster_allclose(r_multi, warn_failure_reason=True, strict_masked=False)

        # - Test reprojection with CRS change -
        for out_crs in [rio.crs.CRS.from_epsg(4326)]:

            # Single-process reprojection
            r_single = r.reproject(crs=out_crs)

            # Multiprocessing reprojection
            r_multi = r.reproject(crs=out_crs, mp_config=config)

            # Assert the results are the same
            assert r_single.raster_allclose(r_multi, warn_failure_reason=True, strict_masked=False)

        # Check that reprojection works for several bands in multiproc as well
        for n in [3]:
            img1 = gu.Raster.from_array(
                np.ones((n, 50, 50), dtype="uint8"),
                transform=rio.transform.from_origin(0, 50, 1, 1),
                crs=4326,
                nodata=0,
            )
            img2 = gu.Raster.from_array(
                np.ones((n, 50, 50), dtype="uint8"),
                transform=rio.transform.from_origin(50, 50, 1, 1),
                crs=4326,
                nodata=0,
            )

            out_img_single = img2.reproject(img1)
            out_img_multi = img2.reproject(ref=img1, mp_config=config)

            assert out_img_multi.count == n
            assert out_img_multi.shape == (50, 50)
            assert out_img_single.raster_allclose(out_img_multi, warn_failure_reason=True, strict_masked=False)


def _build_dst_transform_shifted_newres(
    src_transform: rio.transform.Affine,
    src_shape: tuple[int, int],
    src_crs: CRS,
    dst_crs: CRS,
    bounds_rel_shift: tuple[float, float],
    res_rel_fac: tuple[float, float],
) -> rio.transform.Affine:
    """
    Build a destination transform intersecting the source transform given source/destination shapes,
    and possibly introducing a relative shift in upper-left bound and multiplicative change in resolution.
    """

    # Get bounding box in source CRS
    bounds = rio.coords.BoundingBox(*rio.transform.array_bounds(src_shape[0], src_shape[1], src_transform))

    # Construct an aligned transform in the destination CRS assuming the same resolution
    tmp_transform = rio.warp.calculate_default_transform(
        src_crs,
        dst_crs,
        src_shape[1],
        src_shape[0],
        left=bounds.left,
        right=bounds.right,
        top=bounds.top,
        bottom=bounds.bottom,
        dst_width=src_shape[1],
        dst_height=src_shape[0],
    )[0]
    # This allows us to get bounds and resolution in the units of the new CRS
    tmp_res = (tmp_transform[0], abs(tmp_transform[4]))
    tmp_bounds = rio.coords.BoundingBox(*rio.transform.array_bounds(src_shape[0], src_shape[1], tmp_transform))
    # Now we can create a shifted/different-res destination grid
    dst_transform = rio.transform.from_origin(
        west=tmp_bounds.left + bounds_rel_shift[0] * tmp_res[0] * src_shape[1],
        north=tmp_bounds.top + 150 * bounds_rel_shift[0] * tmp_res[1] * src_shape[0],
        xsize=tmp_res[0] / res_rel_fac[0],
        ysize=tmp_res[1] / res_rel_fac[1],
    )

    return dst_transform


class TestDask:
    """
    Testing class for delayed functions.

    We test on a first set of rasters big enough to clearly monitor the memory usage, and a second set small enough
    to run fast to check a wide range of input parameters.

    We compare outputs with the in-memory function specifically for input variables that influence the delayed
    algorithm and might lead to new errors (for example: array shape to get subsample/points locations for
    subsample and interp_points, or destination chunksizes to map output of reproject).
    """

    # Skip the whole module if Dask is not installed
    pytest.importorskip("dask")
    import dask.array as da

    # Define random seed for generating test data
    rng = da.random.default_rng(seed=42)

    # Smaller test files for fast checks, with various shapes and with/without nodata
    list_small_shapes = [(51, 47)]
    with_nodata = [False, True]
    list_small_darr = []
    for small_shape in list_small_shapes:
        for w in with_nodata:
            small_darr = rng.normal(size=small_shape[0] * small_shape[1])
            # Add about 5% of nodata values
            if w:
                ind_nodata = rng.choice(small_darr.size, size=int(0.05 * small_darr.size), replace=False)
                small_darr[list(ind_nodata)] = np.nan
            small_darr = small_darr.reshape(small_shape[0], small_shape[1])
            list_small_darr.append(small_darr)

    # List of in-memory chunksize for small tests
    list_small_chunksizes_in_mem = [(20, 20), (17, 39)]

    # Create a corresponding boolean array for each numerical dask array
    # Every finite numerical value (valid numerical value) corresponds to True (valid boolean value).
    darr_bool = []
    for small_darr in list_small_darr:
        darr_bool.append(da.where(da.isfinite(small_darr), True, False))

    @pytest.mark.parametrize("darr", list_small_darr)
    @pytest.mark.parametrize("chunksizes_in_mem", list_small_chunksizes_in_mem)
    @pytest.mark.parametrize("dst_chunksizes", list_small_chunksizes_in_mem)
    # Shift upper left corner of output bounds (relative to projected input bounds) by fractions of the raster size
    @pytest.mark.parametrize("dst_bounds_rel_shift", [(0, 0), (-0.2, 0.5)])
    # Modify output resolution (relative to projected input resolution) by a factor
    @pytest.mark.parametrize("dst_res_rel_fac", [(1, 1), (2.1, 0.54)])
    # Same for shape
    @pytest.mark.parametrize("dst_shape_diff", [(0, 0), (-28, 117)])
    def test_dask_reproject__output(
        self,
        darr: da.Array,
        chunksizes_in_mem: tuple[int, int],
        dst_chunksizes: tuple[int, int],
        dst_bounds_rel_shift: tuple[float, float],
        dst_res_rel_fac: tuple[float, float],
        dst_shape_diff: tuple[int, int],
    ) -> None:
        """
        Checks for the delayed reproject function.
        Variables that influence specifically the delayed function are:
        - Input/output chunksizes,
        - Input array shape,
        - Output geotransform relative to projected input geotransform,
        - Output array shape relative to input.
        """

        # 0/ Define input parameters

        # Get input and output shape
        darr = darr.rechunk(chunksizes_in_mem)
        src_shape = darr.shape
        dst_shape = (src_shape[0] + dst_shape_diff[0], src_shape[1] + dst_shape_diff[1])
        arr = np.array(darr)

        # Define arbitrary input transform, as we only care about the relative difference with the output transform
        src_transform = rio.transform.from_bounds(10, 10, 15, 15, src_shape[0], src_shape[1])

        # Other arguments having (normally) no influence
        src_crs = CRS(4326)
        dst_crs = CRS(32630)
        src_nodata = -9999
        dst_nodata = 99999
        resampling = rio.enums.Resampling.bilinear

        # Get shifted dst_transform with new resolution
        dst_transform = _build_dst_transform_shifted_newres(
            src_transform=src_transform,
            src_crs=src_crs,
            dst_crs=dst_crs,
            src_shape=src_shape,
            bounds_rel_shift=dst_bounds_rel_shift,
            res_rel_fac=dst_res_rel_fac,
        )

        # 2/ Run delayed reproject with memory monitoring
        reproj_kwargs = {"src_transform": src_transform, "src_crs": src_crs, "dst_crs": dst_crs,
                         "resampling": resampling, "dst_transform": dst_transform, "src_nodata": src_nodata,
                         "dst_nodata": dst_nodata, "dst_shape": dst_shape, "dtype": darr.dtype, "num_threads": 1}

        reproj_arr = _dask_reproject(
            darr,
            dst_chunksizes=dst_chunksizes,
            **reproj_kwargs,
        )
        # Load in memory
        reproj_arr = np.array(reproj_arr)

        # 3/ Outputs check: load in memory and compare with a direct Rasterio reproject

        dst_arr = _rio_reproject(
            arr,
            reproj_kwargs,
        )

        # For recent version (tolerance argument added and set to zero), results are the same
        if Version(rio.__version__) >= Version("1.5.0"):
            assert np.allclose(reproj_arr, dst_arr, equal_nan=True)
