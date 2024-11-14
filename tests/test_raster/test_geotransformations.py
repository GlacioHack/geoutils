"""Test for geotransformations of raster objects."""

from __future__ import annotations

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio as rio

import geoutils as gu
from geoutils import examples
from geoutils.raster.geotransformations import _resampling_method_from_str
from geoutils.raster.raster import _default_nodata

DO_PLOT = False


class TestRasterGeotransformations:

    landsat_b4_path = examples.get_path("everest_landsat_b4")
    landsat_b4_crop_path = examples.get_path("everest_landsat_b4_cropped")
    landsat_rgb_path = examples.get_path("everest_landsat_rgb")
    everest_outlines_path = examples.get_path("everest_rgi_outlines")
    aster_dem_path = examples.get_path("exploradores_aster_dem")
    aster_outlines_path = examples.get_path("exploradores_rgi_outlines")

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
        # Set img2 pixel interpretation as "Point" to match "img1" and avoid any warnings
        img2.set_area_or_point("Point", shift_area_or_point=False)
        img1.set_nodata(0)
        img2.set_nodata(0)

        # Resample the rasters using a new resampling method and see that the string and enum gives the same result.
        img3a = img1.reproject(img2, resampling="q1")
        img3b = img1.reproject(img2, resampling=rio.enums.Resampling.q1)
        assert img3a.raster_equal(img3b)

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
        rng = np.random.default_rng(42)
        rand_int = rng.integers(1, min(r.shape) - 1)

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
        rand_float = rng.integers(1, min(r.shape) - 1) + 0.25

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
        rand_float = rng.integers(1, min(r.shape) / 2 - 1) + 0.25
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
        rand_int = rng.integers(1, min(r.shape) - 1)
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
        rand_float = rng.integers(1, min(r.shape) - 1) + 0.25
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

        # - Check related to pixel interpretation -

        # Check warning for a different area_or_point for the match-reference geometry works
        r.set_area_or_point("Area", shift_area_or_point=False)
        r2 = r.copy()
        r2.set_area_or_point("Point", shift_area_or_point=False)

        with pytest.warns(UserWarning, match='One raster has a pixel interpretation "Area" and the other "Point".*'):
            r.crop(r2)

        # Check that cropping preserves the interpretation
        crop_geom = [crop_geom[0] + r.res[0], crop_geom[1], crop_geom[2], crop_geom[3]]
        r_crop = r.crop(crop_geom)
        assert r_crop.area_or_point == "Area"
        r2_crop = r2.crop(crop_geom)
        assert r2_crop.area_or_point == "Point"

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path, landsat_rgb_path])  # type: ignore
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
        assert orig_transform.f + 1 * orig_res[1] == r.transform.f
        for attr in ["a", "b", "d", "e"]:
            assert getattr(orig_transform, attr) == getattr(r.transform, attr)

        assert orig_bounds.left + 1 * orig_res[0] == r.bounds.left
        assert orig_bounds.right + 1 * orig_res[0] == r.bounds.right
        assert orig_bounds.bottom + 1 * orig_res[1] == r.bounds.bottom
        assert orig_bounds.top + 1 * orig_res[1] == r.bounds.top

        # Check that an error is raised for a wrong distance_unit
        with pytest.raises(ValueError, match="Argument 'distance_unit' should be either 'pixel' or 'georeferenced'."):
            r.translate(xoff=1, yoff=1, distance_unit="wrong_value")  # type: ignore

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
        default_nodata = _default_nodata(r_nodata.dtype)
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

        # -- Check warning for area_or_point works -- #
        r.set_area_or_point("Area", shift_area_or_point=False)
        r2 = r.copy()
        r2.set_area_or_point("Point", shift_area_or_point=False)

        with pytest.warns(UserWarning, match='One raster has a pixel interpretation "Area" and the other "Point".*'):
            r.reproject(r2)

        # Check that reprojecting preserves interpretation
        r_reproj = r.reproject(res=r.res[0] * 2)
        assert r_reproj.area_or_point == "Area"
        r2_reproj = r2.reproject(res=r2.res[0] * 2)
        assert r2_reproj.area_or_point == "Point"


class TestMaskGeotransformations:
    # Paths to example data
    landsat_b4_path = examples.get_path("everest_landsat_b4")
    landsat_rgb_path = examples.get_path("everest_landsat_rgb")
    everest_outlines_path = examples.get_path("everest_rgi_outlines")
    aster_dem_path = examples.get_path("exploradores_aster_dem")

    # Mask without nodata
    mask_landsat_b4 = gu.Raster(landsat_b4_path) > 125
    # Mask with nodata
    mask_aster_dem = gu.Raster(aster_dem_path) > 2000
    # Mask from an outline
    mask_everest = gu.Vector(everest_outlines_path).create_mask(gu.Raster(landsat_b4_path))

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
        rng = np.random.default_rng(42)
        rand_int = rng.integers(1, min(mask.shape) - 1)

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
            mask.reproject(res=50, resampling="bilinear", force_source_nodata=2)
