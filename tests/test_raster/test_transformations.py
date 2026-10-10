"""Test for geotransformations of raster objects."""

from __future__ import annotations

import re
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio as rio
from affine import Affine
from packaging.version import Version
from pyproj import CRS
from shapely.geometry import Polygon

import geoutils as gu
from geoutils import examples, open_raster
from geoutils._misc import import_optional, silence_rasterio_message
from geoutils.exceptions import InvalidCRSError, InvalidGridError
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.cluster import MpCluster
from geoutils.operators import GridNeighbours, Interpolator, PointNeighbours, Reducer
from geoutils.operators.interpolator import Linear, Nearest, RasterConvolution
from geoutils.operators.reducer import (
    Maximum,
    Mean,
    Median,
    Minimum,
    Mode,
    Quantile,
    RootMeanSquare,
    Sum,
)
from geoutils.projtools import _get_bounds_projected
from geoutils.raster.raster import _default_nodata
from geoutils.raster.transformation import (
    _resampling_method_from_str,
)
from geoutils.sampling.subsampling import _subsample_numpy
from tests.operator_helpers import (
    LocalMeanInterpolator,
    NoSupportReducer,
    PropagatingMeanReducer,
    SupportMassReducer,
    WindowRangeInterpolator,
)

DO_PLOT = False


class TestTransformation:
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
    @pytest.mark.filterwarnings("ignore:Argument 'inplace' is deprecated:DeprecationWarning")
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
        assert r_crop_unloaded.raster_equal(r_crop_loaded)
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
        assert r_crop_unloaded.raster_equal(r_crop_loaded)
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

    def test_crop__deferred(self) -> None:
        """Checks that crop() does not read raster values until they are needed."""

        # Open one unloaded raster and one loaded raster for comparison
        source = gu.Raster(self.landsat_b4_path)
        eager_source = gu.Raster(self.landsat_b4_path, load_data=True)
        left, bottom, right, top = source.bounds
        first_bbox = (left + 2 * source.res[0], bottom, right, top - 2 * source.res[1])
        second_bbox = (left + 4 * source.res[0], bottom, right - 3 * source.res[0], top - 4 * source.res[1])

        # Apply two crops without reading the source or result
        cropped = source.crop(first_bbox).crop(second_bbox)
        assert not source.is_loaded
        assert not cropped.is_loaded

        # Compare the result with the same crops on loaded data
        expected = eager_source.crop(first_bbox).crop(second_bbox)
        assert cropped.raster_equal(expected, strict_masked=True)
        assert not source.is_loaded

    def test_icrop__rotated_grid(self, tmp_path: Any) -> None:
        """Checks that pixel cropping preserves the values and affine transform of a rotated raster."""

        # Write a rotated raster so both loaded and deferred pixel crops use the same source values
        values = np.arange(72, dtype=np.int16).reshape(8, 9)
        transform = Affine.translation(500_000, 5_100_000) * Affine.rotation(10) * Affine.scale(100, -100)
        raster = gu.Raster.from_array(values, transform=transform, crs=32632, nodata=-9999)
        source_path = tmp_path / "rotated_crop.tif"
        raster.to_file(source_path)
        unloaded = gu.Raster(source_path)

        # The crop starts at source column 2 and row 1, with a shorter output in both axes
        loaded_crop = raster.icrop((2, 1, 7, 6))
        deferred_crop = unloaded.icrop((2, 1, 7, 6))
        assert loaded_crop is not None and deferred_crop is not None
        assert not unloaded.is_loaded
        assert not deferred_crop.is_loaded

        # Check both crops use the exact source pixel slice and its translated rotated grid
        expected_transform = transform * Affine.translation(2, 1)
        assert loaded_crop.transform == expected_transform
        assert deferred_crop.transform == expected_transform
        np.testing.assert_array_equal(loaded_crop.to_nanarray(), values[1:6, 2:7])
        np.testing.assert_array_equal(deferred_crop.to_nanarray(), values[1:6, 2:7])

    def test_clip(self) -> None:
        """Checks that clip() is masking cells outside the given geometry."""

        # Create a 4 x 4 raster and a triangle polygon vector crossing its cell grid
        values = np.arange(16, dtype=np.int16).reshape(4, 4)
        transform = rio.transform.from_origin(0, 4, 1, 1)
        raster = gu.Raster.from_array(values, transform=transform, crs=32610, nodata=-9999)
        triangle = Polygon([(0, 0), (4, 0), (0, 4)])

        # Clip to geometry
        clipped = raster.clip(triangle)
        expected_inside = rio.features.geometry_mask(
            [triangle], out_shape=raster.shape, transform=raster.transform, invert=True
        )

        # Check expected inside/outside geometries
        assert clipped.shape == raster.shape
        assert clipped.transform == raster.transform
        np.testing.assert_array_equal(np.ma.getmaskarray(clipped.data), ~expected_inside)
        np.testing.assert_array_equal(clipped.data.data[expected_inside], values[expected_inside])

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

    def test_reproject__unloaded_multiband_to_single_band_reference(self) -> None:
        """Checks that a multi-band raster can use a match reference for reprojection."""

        # Open multiband and single band rasters, eager and unloaded
        source = gu.Raster(self.landsat_rgb_path)
        eager_source = gu.Raster(self.landsat_rgb_path, load_data=True)
        reference = gu.Raster(self.landsat_b4_crop_path)
        source.set_nodata(0, update_array=False, update_mask=False)
        eager_source.set_nodata(0, update_array=False, update_mask=False)
        assert not source.is_loaded

        # Reproject both sources onto the single-band reference grid
        result = source.reproject(reference)
        expected = eager_source.reproject(reference)

        # Check that all three bands use the reference shape and match the eager result
        assert result.count == source.count == 3
        assert result.shape == reference.shape
        assert result.raster_equal(expected, strict_masked=True)

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])
    @pytest.mark.filterwarnings("ignore:Argument 'inplace' is deprecated:DeprecationWarning")
    def test_reproject(self, example: str) -> None:

        # Reference raster to be used
        r = gu.Raster(example)

        # -- Test setting each combination of georeferences bounds, res and size -- #

        # Landsat data contain the default nodata value 255, so use 0
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

        # -- Check warning for area_or_point works -- #
        r.set_area_or_point("Area", shift_area_or_point=False)
        r2 = r.copy()
        r2.set_area_or_point("Point", shift_area_or_point=False)

        with pytest.warns(UserWarning) as raised:
            r.reproject(r2)
        assert any("One raster has a pixel" in str(warning.message) for warning in raised)
        assert any(
            "Output projection, bounds and grid size are identical" in str(warning.message) for warning in raised
        )

        # Check that reprojecting preserves interpretation
        r_reproj = r.reproject(res=r.res[0] * 2)
        assert r_reproj.area_or_point == "Area"
        r2_reproj = r2.reproject(res=r2.res[0] * 2)
        assert r2_reproj.area_or_point == "Point"

    @pytest.mark.parametrize("loaded", [False, True])
    def test_reproject__gcp_rcp_match_rasterio(self, raster_gcp_rpc: gu.Raster, loaded: bool) -> None:
        """Checks that reprojecting with GCPs/RPCs matches exactly Rasterio."""

        # Use the same source loaded/unloaded
        source = raster_gcp_rpc
        if loaded:
            source.load()
        options = {"RPC_HEIGHT": 80} if source.rpcs else {}
        georeferencing = {"gcps": source.gcps[0]} if source.gcps[0] else {"rpcs": source.rpcs}
        src_crs = source.gcps[1] if source.gcps[0] else rio.CRS.from_epsg(4326)

        # Calculate GDAL warp grid, and reproject
        transform, width, height = rio.warp.calculate_default_transform(
            src_crs,
            3857,
            source.width,
            source.height,
            dst_width=source.width,
            dst_height=source.height,
            **georeferencing,
            **options,
        )
        with rio.open(source.name) as dataset:
            values = dataset.read()
        expected = np.empty((source.count, height, width), dtype=source.dtype)
        with silence_rasterio_message(param_name="RPC_HEIGHT"):
            rio.warp.reproject(
                values,
                expected,
                src_crs=src_crs,
                src_nodata=source.nodata,
                dst_crs=3857,
                dst_transform=transform,
                dst_nodata=source.nodata,
                resampling=rio.enums.Resampling.nearest,
                tolerance=0,
                XSCALE=1,
                YSCALE=1,
                **georeferencing,
                **options,
            )

        # Check that the affine output clears GCPs/RPCs, and that we have exactly equality
        result = source.reproject(crs=3857, resampling="nearest", transformer_options=options)
        assert source.is_loaded
        assert result.shape == (height, width)
        assert result.transform == transform
        assert result.crs == rio.CRS.from_epsg(3857)
        np.testing.assert_array_equal(result.data.filled(source.nodata), expected)
        assert result.gcps == ([], None)
        assert result.rpcs is None

    def test_reproject__assigned_gcp_rcp(self, raster_gcp_rpc: gu.Raster) -> None:
        """Checks that GCPs/RPCs assigned manually before reprojection also work."""

        # We recreate the array without GCPs/RCPs, then assign them manually
        source = raster_gcp_rpc
        image = gu.Raster.from_array(source.data, rio.Affine.identity(), None, nodata=source.nodata)
        image.gcps, image.rpcs = source.gcps, source.rpcs
        expected = source.reproject(crs=3857, resampling="nearest")
        result = image.reproject(crs=3857, resampling="nearest")

        # Check exact equality
        assert result.raster_equal(expected)

    @pytest.mark.parametrize("raster_gcp_rpc", ["gcp_projected", "gcp_nonlinear"], indirect=True)
    @pytest.mark.parametrize("gcp_crs", ["supplied", "source"])
    def test_reproject__gcp_crs(self, raster_gcp_rpc: gu.Raster, gcp_crs: str) -> None:
        """Checks that stored GCPs use their own CRS or default to the raster CRS."""

        # We recreate the image with a different raster CRS (with GCPs that also store their own CRS)
        source = raster_gcp_rpc
        points, point_crs = source.gcps
        image_crs = 3857 if gcp_crs == "supplied" else point_crs
        image = gu.Raster.from_array(source.data, rio.Affine.identity(), image_crs, nodata=source.nodata)
        gcps = (points, point_crs if gcp_crs == "supplied" else None)
        image.gcps = gcps

        # Reproject both with own GCP CRS or raster CRS only
        expected = source.reproject(crs=3857, resampling="nearest")
        result = image.reproject(crs=3857, resampling="nearest")

        # Check exact equality
        assert result.raster_equal(expected)

    @pytest.mark.parametrize("raster_gcp_rpc", ["gcp_rpc_polynomial", "gcp_rpc_rational"], indirect=True)
    @pytest.mark.parametrize("source_method", ["RPC", "GCP_POLYNOMIAL", "GCP_TPS"])
    @pytest.mark.parametrize("raster_type", ["raster", "dataarray"])
    def test_reproject__select_gcp_rcp(self, raster_gcp_rpc: gu.Raster, source_method: str, raster_type: str) -> None:
        """Checks that SRC_METHOD selects stored GCPs/RPCs properly for reprojection."""

        # Open an image with both RPCs and GCPs defined at once
        source = raster_gcp_rpc
        image = source if raster_type == "raster" else gu.open_raster(source.name).rst
        points = [point.asdict() for point in image.gcps[0]]
        gcp_crs = image.gcps[1]
        rpcs = image.rpcs.to_gdal()
        options: dict[str, Any] = {"SRC_METHOD": source_method}
        if source_method == "RPC":
            options["RPC_HEIGHT"] = 80
        saved_options = options.copy()

        # Reproject with both at once and a transformer option selecting, or with the other GCP/RCP set to None
        expected_source = gu.Raster(source.name)
        if source_method == "RPC":
            expected_source.gcps = ([], None)
        else:
            expected_source.rpcs = None
        expected = expected_source.reproject(crs=3857, resampling="nearest", transformer_options=options)
        result = image.reproject(crs=3857, resampling="nearest", transformer_options=options)

        # Compare exact equality
        result_raster = result if raster_type == "raster" else result.rst
        assert result_raster.transform == expected.transform
        assert result_raster.crs == expected.crs
        np.testing.assert_array_equal(result_raster.to_nanarray(), expected.to_nanarray())
        assert result_raster.gcps == ([], None)
        assert result_raster.rpcs is None

        # Both GCPs/RCPs remain available in the source raster
        assert [point.asdict() for point in image.gcps[0]] == points
        assert image.gcps[1] == gcp_crs
        assert image.rpcs.to_gdal() == rpcs
        assert options == saved_options

    def test_reproject__affine_transformations(self, raster_gcp_rpc: gu.Raster) -> None:
        """
        Checks that running reproject() on GCP/RPC rasters does enable later crop/clip/etc by giving an affine
        transform (the hint we use a lot in GCP/RCP errors, so really needs to work properly!).
        """

        # We reproject a raster with stored GCPs/RPCs
        raster = raster_gcp_rpc.reproject(crs=4326, grid_size=(40, 48), resampling="nearest")
        assert raster.gcps == ([], None)
        assert raster.rpcs is None

        # Now we can try to crop/clip
        left, top = raster.transform * (5, 6)
        right, bottom = raster.transform * (30, 28)
        bounds = (left, bottom, right, top)
        cropped = raster.crop(bounds)
        clipped = raster.clip(bounds)

        # We check cropping does select the bounds, while clipping create appropriate NaNs
        np.testing.assert_array_equal(cropped.to_nanarray(), raster.to_nanarray()[:, 6:28, 5:30])
        expected_mask = np.ones(raster.shape, dtype=bool)
        expected_mask[6:28, 5:30] = False
        expected_mask = np.ma.getmaskarray(raster.data) | expected_mask
        np.testing.assert_array_equal(np.ma.getmaskarray(clipped.data), expected_mask)
        np.testing.assert_array_equal(clipped.to_nanarray()[:, 6:28, 5:30], cropped.to_nanarray())

        # We can now also translate the affine grid
        translated = raster.translate(0.01, -0.02)
        assert translated.transform.c == raster.transform.c + 0.01
        assert translated.transform.f == raster.transform.f - 0.02
        np.testing.assert_array_equal(translated.to_nanarray(), raster.to_nanarray())
        assert raster.res == (abs(raster.transform.a), abs(raster.transform.e))
        assert tuple(raster.bbox) == rio.transform.array_bounds(*raster.shape, raster.transform)


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
    @pytest.mark.filterwarnings("ignore:Argument 'inplace' is deprecated:DeprecationWarning")
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

    @pytest.mark.parametrize("method", ["crop", "icrop"])
    def test_crop__unloaded_mask_boolean_values(self, tmp_path: Any, method: str) -> None:
        """
        Checks that crop() and icrop() return exact boolean values and nodata pixels without loading the source mask.
        """

        # Write integer mask values with one nodata pixel inside the window being cropped
        values = (np.arange(30).reshape(5, 6) % 2).astype("uint8")
        values[2, 2] = 255
        masked_values = np.ma.masked_equal(values, 255)
        transform = rio.transform.from_origin(100, 200, 10, 10)
        path = tmp_path / "mask.tif"
        gu.Raster.from_array(masked_values, transform, 32633, nodata=255).to_file(path)
        unloaded = gu.Raster(path, is_mask=True)
        loaded = gu.Raster(path, is_mask=True, load_data=True)

        # Select rows 1:4 and columns 1:5 through both public coordinate conventions
        if method == "crop":
            output = unloaded.crop((110, 160, 150, 190))
            expected = loaded.crop((110, 160, 150, 190))
        else:
            output = unloaded.icrop((1, 1, 5, 4))
            expected = loaded.icrop((1, 1, 5, 4))

        # Match the full-load path and verify logical values and nodata pixels independently
        assert not unloaded.is_loaded
        assert not output.is_loaded
        assert output.is_mask
        assert output.data.dtype == bool
        assert output.raster_equal(expected, strict_masked=True)
        np.testing.assert_array_equal(output.data.data, values[1:4, 1:5].astype(bool))
        np.testing.assert_array_equal(np.ma.getmaskarray(output.data), values[1:4, 1:5] == 255)

    @pytest.mark.parametrize("mask", [mask_landsat_b4, mask_aster_dem, mask_everest])
    @pytest.mark.filterwarnings("ignore:Argument 'inplace' is deprecated:DeprecationWarning")
    def test_reproject(self, mask: gu.Raster) -> None:
        # Reproject with nearest neighbor resampling

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


@pytest.mark.skipif(find_spec("dask") is None, reason="Only runs if dask is installed.")
class TestTransformationChunked:
    """Test module for raster transformations run with Dask or multiprocessing."""

    def test_clip__chunked_backends_equal(self, tmp_path: Any) -> None:
        """Checks that clip with Dask and multiprocessing give the same result as in-memory."""

        import dask.array as da

        # Write three bands in 2 x 3 blocks, with one cell outside the geometry
        values = np.arange(3 * 7 * 8, dtype=np.float32).reshape(3, 7, 8)
        existing_mask = np.zeros(values.shape, dtype=bool)
        existing_mask[:, 5, 1] = True
        masked_values = np.ma.masked_array(values, mask=existing_mask)
        transform = rio.transform.from_origin(0, 7, 1, 1)
        path = tmp_path / "clip_source.tif"
        source = gu.Raster.from_array(masked_values, transform=transform, crs=32610, nodata=-9999)
        source.to_file(path)
        geometry = Polygon([(0, 0), (8, 0), (0, 7)])

        # Clip the same file in memory, with Dask + MP
        expected = source.clip(geometry)
        dask_source = open_raster(path, chunks={"band": 1, "y": 2, "x": 3})
        raster_source = gu.Raster(path)
        dask_result = dask_source.rst.clip(geometry)
        with MpCluster({"nb_workers": 2}) as cluster:
            mp_config = MultiprocConfig(chunks=(2, 3), outfile=str(tmp_path / "clip_multiproc.tif"), cluster=cluster)
            multiproc_result = raster_source.clip(geometry, mp_config=mp_config)

        # Check Dask laziness and loading behaviour
        assert isinstance(dask_result.data, da.Array)
        assert dask_result.chunks == dask_source.chunks
        assert not raster_source.is_loaded
        assert not multiproc_result.is_loaded
        with pytest.raises(ValueError, match="Cannot use Multiprocessing and Dask simultaneously"):
            dask_source.rst.clip(geometry, mp_config=mp_config)

        # Read all results and check they exactly match
        expected_values = expected.to_nanarray()
        np.testing.assert_allclose(dask_result.compute().data, expected_values, equal_nan=True)
        np.testing.assert_allclose(multiproc_result.to_nanarray(), expected_values, equal_nan=True)
        assert multiproc_result.transform == expected.transform
        assert multiproc_result.crs == expected.crs

    def test_clip__multiproc_default_nodata(self, tmp_path: Any) -> None:
        """Checks that multiprocessing clip() adds a nodata value when the source has none."""

        # Write a raster without nodata so clipping creates the first missing cells
        values = np.arange(30, dtype=np.float32).reshape(5, 6)
        transform = rio.transform.from_origin(0, 5, 1, 1)
        path = tmp_path / "clip_without_nodata.tif"
        gu.Raster.from_array(values, transform=transform, crs=32610).to_file(path)
        source = gu.Raster(path)
        geometry = Polygon([(0, 0), (6, 0), (0, 5)])

        # Verify the warning about the new nodata value
        config = MultiprocConfig(chunks=(2, 3), outfile=str(tmp_path / "clip_with_default_nodata.tif"))
        with pytest.warns(UserWarning, match="multiprocessing clip.*will use the default"):
            result = source.clip(geometry, mp_config=config)

        # Compare result with in memory clipping
        expected = gu.Raster(path, load_data=True).clip(geometry)
        assert not source.is_loaded
        assert not result.is_loaded
        assert result.nodata == _default_nodata(values.dtype)
        np.testing.assert_allclose(result.to_nanarray(), expected.to_nanarray(), equal_nan=True)

    @pytest.mark.parametrize("load_source", [False, True])
    def test_reproject__multiprocessing_logical_mask(self, tmp_path: Any, load_source: bool) -> None:
        """
        Checks that multiprocessing reprojection returns exact mask values and nodata pixels on a shifted grid.
        """

        # Write both boolean states and one nodata pixel, then shift the target to leave an uncovered column
        rows, columns = np.indices((6, 7))
        values = ((rows + columns) % 2).astype("uint8")
        values[2, 3] = 255
        transform = rio.transform.from_origin(0, 6, 1, 1)
        source_path = tmp_path / "mask_source.tif"
        data = np.ma.masked_equal(values, 255)
        gu.Raster.from_array(data, transform, 32633, nodata=255).to_file(source_path)
        reference = gu.Raster.from_array(np.zeros((6, 7)), rio.transform.from_origin(1, 6, 1, 1), 32633)

        # Compare a complete in-memory reprojection with windows read through the multiprocessing backend
        loaded = gu.Raster(source_path, is_mask=True, load_data=True)
        expected = loaded.reproject(ref=reference, resampling="nearest")
        source = gu.Raster(source_path, is_mask=True, load_data=load_source)
        config = MultiprocConfig(chunks=(3, 4), outfile=str(tmp_path / "mask_reprojected.tif"))
        output = source.reproject(ref=reference, resampling="nearest", mp_config=config)

        # Check that the input loading state is unchanged and restore boolean interpretation before reading the output
        assert source.is_loaded == load_source
        assert not output.is_loaded
        assert output.is_mask
        assert output.data.dtype == bool
        assert output.transform == reference.transform

        # Check the internal hole and uncovered column independently, then compare the known true/false values
        expected_missing = np.zeros((6, 7), dtype=bool)
        expected_missing[2, 2] = True
        expected_missing[:, -1] = True
        np.testing.assert_array_equal(np.ma.getmaskarray(output.data), expected_missing)
        np.testing.assert_array_equal(output.data.compressed(), expected.data.compressed())

    def test_reproject__small_chunked_grid_matches_base(self, tmp_path: Any) -> None:
        """Regression test for small-grid Dask/Multiprocessing block placement during reprojection."""

        import dask.array as da

        # Write a small gradient raster whose shifted output exposes misplaced blocks
        src_arr = np.linspace(0, 99, 100, dtype="float32").reshape(10, 10)
        transform = rio.transform.from_origin(0, 5, 1, 1)
        raster = gu.Raster.from_array(src_arr, transform=transform, crs=4326, nodata=200)
        source_path = tmp_path / "reproject_source.tif"
        raster.to_file(source_path)

        # Extend every edge and establish the eager reference grid
        dst_bounds = rio.coords.BoundingBox(left=-1, bottom=-6, right=11, top=6)
        base = raster.reproject(bounds=dst_bounds, res=(1, 1), resampling="nearest")

        # Reproject an unloaded Raster through multiprocessing chunks
        raster_mp = gu.Raster(source_path)
        mp_config = MultiprocConfig(chunks=5, outfile=str(tmp_path / "reproject_mp.tif"))
        mp = raster_mp.reproject(bounds=dst_bounds, res=(1, 1), resampling="nearest", mp_config=mp_config)

        # Reproject the same file through a lazy Dask-backed accessor
        ds = open_raster(source_path, chunks={"band": 1, "x": 5, "y": 5})
        dask_r = ds.rst.reproject(bounds=dst_bounds, res=(1, 1), resampling="nearest")

        # Both chunked sources stay lazy and match the eager pixel placement
        assert not raster_mp.is_loaded
        assert not ds._in_memory
        assert isinstance(ds.data, da.Array)
        assert np.allclose(base.to_nanarray(), mp.to_nanarray(), equal_nan=True)
        assert np.allclose(base.to_nanarray(), dask_r.compute().data, equal_nan=True)

    @pytest.mark.parametrize("path_index", [0, 2])
    @pytest.mark.parametrize("tile_size", [20])
    @pytest.mark.parametrize("crs_mode", ["epsg4326", "metric_utm"])
    @pytest.mark.parametrize("bounds_mode", ["shrink", "extend"])
    @pytest.mark.parametrize("res_mode", ["upsample", "downsample"])
    @pytest.mark.parametrize("resampling", [rio.enums.Resampling.nearest, rio.enums.Resampling.bilinear])
    def test_reproject__chunked_backends_equal(
        self,
        path_index: int,
        tile_size: int,
        crs_mode: Literal["epsg4326", "metric_utm"],
        bounds_mode: Literal["shrink", "extend"],
        res_mode: Literal["upsample", "downsample"],
        resampling: rio.enums.Resampling,
        lazy_test_files_tiny: list[str],
    ) -> None:
        """
        Test that reproject yields identical output for:
         - In-memory base function through Raster,
         - In-memory base function through Xarray DataArray,
         - Dask backend through Xarray accessor (lazy input),
         - Multiprocessing backend through Raster class (lazy input).

        We vary grid definition through:
         - CRS choice (EPSG:4326 vs local metric CRS),
         - Bounds (shrink vs extend),
         - Resolution (upsample vs downsample).

        Additionally, both Dask and Multiprocessing inputs remain lazy (unloaded).
        """

        import dask.array as da

        warnings.filterwarnings("ignore", category=UserWarning, message="Output projection, bounds and grid size*")
        warnings.filterwarnings("ignore", category=UserWarning, message="Only nodata is different*")

        # 1/ Prepare backend inputs
        # Get filepath of on-disk (for laziness) test file
        path_raster = lazy_test_files_tiny[path_index]

        # Base raster input (in-memory)
        raster_base = gu.Raster(path_raster)
        raster_base.load()
        assert raster_base.is_loaded

        # Base data array input (in-memory)
        ds_base = open_raster(path_raster)
        ds_base.load()
        assert ds_base._in_memory

        # Multiprocessing input (lazy)
        raster_mp = gu.Raster(path_raster)
        assert not raster_mp.is_loaded

        # Dask input (lazy)
        ds_dask = open_raster(path_raster, chunks={"x": tile_size, "y": tile_size})
        assert not ds_dask._in_memory
        assert isinstance(ds_dask.data, da.Array)
        assert ds_dask.data.chunks is not None

        # 2/ Parameterize CRS, bounds and resolution
        if crs_mode == "epsg4326":
            out_crs: CRS | None = CRS.from_epsg(4326)
        elif crs_mode == "metric_utm":
            # Local metric CRS (typically UTM) derived from raster extent
            out_crs = raster_base.get_metric_crs()
        else:
            raise ValueError

        # Bounds modifications: shrink or extend relative to original bounds
        b = raster_base.bounds
        xpad = float(raster_base.res[0]) * 10.0
        ypad = float(raster_base.res[1]) * 10.0

        if bounds_mode == "shrink":
            dst_bounds = rio.coords.BoundingBox(
                left=b.left + xpad,
                bottom=b.bottom + ypad,
                right=b.right - xpad,
                top=b.top - ypad,
            )
        elif bounds_mode == "extend":
            dst_bounds = rio.coords.BoundingBox(
                left=b.left - xpad,
                bottom=b.bottom - ypad,
                right=b.right + xpad,
                top=b.top + ypad,
            )
        else:
            raise ValueError
        dst_bounds = _get_bounds_projected(dst_bounds, in_crs=raster_base.crs, out_crs=out_crs)

        # Resolution modifications: upsample or downsample relative to original

        # Get native resolution in destination CRS units
        # (width / ncols, height / nrows)
        src_bounds_proj = _get_bounds_projected(raster_base.bounds, in_crs=raster_base.crs, out_crs=out_crs)
        dst_width = src_bounds_proj.right - src_bounds_proj.left
        dst_height = src_bounds_proj.top - src_bounds_proj.bottom
        ny, nx = raster_base.shape
        base_res_x = float(dst_width) / float(nx)
        base_res_y = float(dst_height) / float(ny)
        if res_mode == "upsample":
            res = (base_res_x * 0.5, base_res_y * 0.5)
        elif res_mode == "downsample":
            res = (base_res_x * 2.0, base_res_y * 2.0)
        else:
            raise ValueError

        # Multiprocessing config
        mp_config = MultiprocConfig(chunks=(tile_size, tile_size + 5))

        # 3/ Run reproject for each backend
        base = raster_base.reproject(
            crs=out_crs,
            bounds=dst_bounds,
            res=res,
            resampling=resampling,
        )
        xr_base = ds_base.rst.reproject(
            crs=out_crs,
            bounds=dst_bounds,
            res=res,
            resampling=resampling,
        )
        dask_r = ds_dask.rst.reproject(
            crs=out_crs,
            bounds=dst_bounds,
            res=res,
            resampling=resampling,
        )
        mp_r = raster_mp.reproject(
            crs=out_crs,
            bounds=dst_bounds,
            res=res,
            resampling=resampling,
            mp_config=mp_config,
        )

        # 4/ Laziness checks
        assert not ds_dask._in_memory
        assert isinstance(ds_dask.data, da.Array)
        assert not raster_mp.is_loaded

        # 5/ Output checks: all backends must match base
        # (For reproject, no artefacts only since we added "tolerance" argument in Rasterio,
        # which officially came out in 1.5; so we skip the test for earlier versions)
        if Version(rio.__version__) < Version("1.5.0"):
            return
        assert base.raster_allclose(xr_base, warn_failure_reason=True, strict_masked=False)
        assert base.raster_allclose(dask_r, warn_failure_reason=True, strict_masked=False)
        assert base.raster_allclose(mp_r, warn_failure_reason=True, strict_masked=False)

    def test_icrop__gcp_rcp_pixel_offsets(self, raster_gcp_rpc: gu.Raster) -> None:
        """Checks that lazy pixel cropping shifts GCPs/RPCs and matches eager exactly."""

        # We open a GCP/RCPs raster (with chunking defined so that crop crosses several chunks)
        import_optional("dask")
        source = raster_gcp_rpc
        lazy = gu.open_raster(source.name, chunks={"band": 1, "y": 7, "x": 9})
        expected = gu.Raster(source.name, load_data=True).icrop((9, 7, 50, 45))
        assert not source.is_loaded
        assert not lazy.rst.is_loaded

        # We crop to the same pixel window lazily and eagerly
        result = lazy.rst.icrop((9, 7, 50, 45))
        assert not result.rst.is_loaded
        assert result.rst._chunks is not None
        assert not lazy.rst.is_loaded

        # We check the GCP/RPC image offsets, and that we exactly match the eager call
        assert [point.asdict() for point in result.rst.gcps[0]] == [point.asdict() for point in expected.gcps[0]]
        assert result.rst.gcps[1] == expected.gcps[1]
        assert result.rst.rpcs == expected.rpcs
        np.testing.assert_array_equal(result.compute().values, expected.to_nanarray())
        assert not source.is_loaded
        assert not lazy.rst.is_loaded

    @pytest.mark.parametrize("resampling", ["nearest", "bilinear"])
    @pytest.mark.parametrize("backend", ["dask", "mp"])
    def test_reproject__chunk_invariance(
        self, raster_gcp_rpc: gu.Raster, tmp_path: Path, resampling: str, backend: str
    ) -> None:
        """Checks that uneven GCP/RPC chunks reproduce eager values exactly, and respect laziness."""

        # We use the GCP/RCP raster, and reproject with eager, also using a transformer option
        source = raster_gcp_rpc
        options = {"RPC_HEIGHT": 80} if source.rpcs else {}
        expected = gu.Raster(source.name).reproject(crs=3857, resampling=resampling, transformer_options=options)
        assert not source.is_loaded

        # We open in chunks (uneven size to leave a shorter final chunk)
        # Then we run with Dask/MP, which should stay lazy/unloaded
        if backend == "dask":
            import_optional("dask")
            lazy = gu.open_raster(source.name, chunks={"band": 1, "y": 7, "x": 9})
            result = lazy.rst.reproject(crs=3857, resampling=resampling, transformer_options=options)
            assert lazy.rst._chunks is not None
            assert result.rst._chunks is not None
            assert not lazy.rst.is_loaded
            assert not result.rst.is_loaded
            values = result.compute().values
            assert not lazy.rst.is_loaded
            assert result.rst.rpcs is None
            assert result.rst.gcps == ([], None)
        else:
            with MpCluster({"nb_workers": 2}) as cluster:
                config = MultiprocConfig(cluster=cluster, chunks=(7, 9), outfile=str(tmp_path / "warped.tif"))
                result = source.reproject(
                    crs=3857, resampling=resampling, transformer_options=options, mp_config=config
                )
            assert not result.is_loaded
            assert not source.is_loaded
            assert result.rpcs is None
            assert result.gcps == ([], None)
            values = result.data.filled(np.nan)

        # Finally we compare for exact equality
        np.testing.assert_array_equal(values, expected.data.filled(np.nan))
        assert not source.is_loaded

    def test_reproject__rpc_transformer_option(self, raster_gcp_rpc: gu.Raster, tmp_path: Path) -> None:
        """Checks that RPC_HEIGHT transformer option behaves properly with eager/lazy reprojection."""

        # We create a DEM with a height of 80 m everywhere across the image
        # RPCs use ground height to locate image pixels, so supplying this DEM through RPC_DEM should give
        # the same reprojection as RPC_HEIGHT=80, which sets the height directly without an elevation file
        source = raster_gcp_rpc
        if source.rpcs is None:
            pytest.skip("RPC elevation options do not apply to GCPs")
        dem_file = tmp_path / "elevation.tif"
        dem = gu.Raster.from_array(
            np.full((200, 200), 80, dtype="float32"), rio.transform.from_origin(9, 51, 0.01, 0.01), 4326
        )
        dem.to_file(dem_file)
        expected = gu.Raster(source.name).reproject(
            crs=3857, resampling="nearest", transformer_options={"RPC_HEIGHT": 80}
        )
        options = {"RPC_DEM": str(dem_file)}

        # We run reproject, and check exact equality
        eager = gu.Raster(source.name).reproject(crs=3857, resampling="nearest", transformer_options=options)
        lazy = gu.open_raster(source.name, chunks={"y": 7, "x": 9})
        result = lazy.rst.reproject(ref=eager, resampling="nearest", transformer_options=options)
        assert result.rst._chunks is not None
        assert not lazy.rst.is_loaded
        assert not result.rst.is_loaded
        np.testing.assert_array_equal(result.compute().values, eager.data.filled(np.nan))
        np.testing.assert_array_equal(eager.data.filled(np.nan), expected.data.filled(np.nan))
        assert not lazy.rst.is_loaded
        assert not source.is_loaded

    def test_reproject__gcp_transformer_option(self, raster_gcp_rpc: gu.Raster) -> None:
        """Checks that TPS transformer option for GCP behaves properly, and matches lazy/eager."""

        # We use thin plate splines (GCP_TPS) to map image pixels to ground coordinates
        # This bends the image smoothly through the control points
        # Dask must use this mapping both to find the source chunks and to reproject their pixels
        source = raster_gcp_rpc
        if not source.gcps[0]:
            pytest.skip("Thin plate splines do not apply to RPCs")
        options = {"SRC_METHOD": "GCP_TPS"}
        expected = gu.Raster(source.name).reproject(crs=3857, resampling="nearest", transformer_options=options)
        lazy = gu.open_raster(source.name, chunks={"y": 7, "x": 9})
        result = lazy.rst.reproject(crs=3857, resampling="nearest", transformer_options=options)

        # Check exact equality and laziness
        assert result.rst._chunks is not None
        assert not lazy.rst.is_loaded
        assert not result.rst.is_loaded
        np.testing.assert_array_equal(result.compute().values, expected.data.filled(np.nan))
        assert not lazy.rst.is_loaded
        assert not source.is_loaded

    @pytest.mark.parametrize("raster_gcp_rpc", ["gcp_rpc_polynomial", "gcp_rpc_rational"], indirect=True)
    @pytest.mark.parametrize("source_method", ["RPC", "GCP_POLYNOMIAL", "GCP_TPS"])
    @pytest.mark.parametrize("backend", ["dask", "mp"])
    def test_reproject__selected_gcp_rcp_chunk_invariance(
        self, raster_gcp_rpc: gu.Raster, tmp_path: Path, source_method: str, backend: str
    ) -> None:
        """Checks that Dask/MP uses the same selected GCPs/RPCs as eager reprojection and preserves laziness."""

        # Open an image with both GCPs and RPCs, and select one through transformers options
        source = raster_gcp_rpc
        points = [point.asdict() for point in source.gcps[0]]
        gcp_crs = source.gcps[1]
        rpcs = source.rpcs.to_gdal()
        options: dict[str, Any] = {"SRC_METHOD": source_method}
        if source_method == "RPC":
            options["RPC_HEIGHT"] = 80
        saved_options = options.copy()
        assert not source.is_loaded

        # Run reproject eagerly, then in chunks with Dask/MP
        expected_source = gu.Raster(source.name)
        expected = expected_source.reproject(crs=3857, resampling="nearest", transformer_options=options)
        assert expected_source.is_loaded
        assert expected.is_loaded
        if backend == "dask":
            import_optional("dask")
            lazy = gu.open_raster(source.name, chunks={"band": 1, "y": 7, "x": 9})
            assert not lazy.rst.is_loaded
            assert lazy.rst._chunks is not None
            result = lazy.rst.reproject(crs=3857, resampling="nearest", transformer_options=options)
            assert not lazy.rst.is_loaded
            assert not result.rst.is_loaded
            assert result.rst._chunks is not None
            values = result.compute().values
            assert not lazy.rst.is_loaded
            assert [point.asdict() for point in lazy.rst.gcps[0]] == points
            assert lazy.rst.gcps[1] == gcp_crs
            assert lazy.rst.rpcs.to_gdal() == rpcs
            result_raster = result.rst
        else:
            with MpCluster({"nb_workers": 2}) as cluster:
                config = MultiprocConfig(cluster=cluster, chunks=(7, 9), outfile=str(tmp_path / "selected.tif"))
                result = source.reproject(crs=3857, resampling="nearest", transformer_options=options, mp_config=config)
            assert not source.is_loaded
            assert not result.is_loaded
            values = result.to_nanarray()
            result_raster = result

        # Check exact equality and laziness, and that GCPs/RCPs are still present in the source
        assert result_raster.transform == expected.transform
        assert result_raster.crs == expected.crs
        np.testing.assert_array_equal(values, expected.to_nanarray())
        assert result_raster.gcps == ([], None)
        assert result_raster.rpcs is None
        assert not source.is_loaded
        assert [point.asdict() for point in source.gcps[0]] == points
        assert source.gcps[1] == gcp_crs
        assert source.rpcs.to_gdal() == rpcs
        assert options == saved_options

    @pytest.mark.parametrize("backend", ["dask", "mp"])
    @pytest.mark.parametrize("georeferencing", ["gcp_rcp", "affine"])
    def test_reproject__empty_multiband_chunks(
        self, raster_gcp_rpc: gu.Raster, tmp_path: Path, backend: str, georeferencing: str
    ) -> None:
        """ Checks that empty chunks keep the right number of bands during reprojection. """

        # We use several bands and request a reprojected extent larger than source to create empty output chunks
        filename = tmp_path / "integer_image.tif"
        raster_gcp_rpc.load()
        integer_image = raster_gcp_rpc.astype("int16", convert_nodata=False)
        if georeferencing == "affine":
            integer_image.gcps, integer_image.rpcs = ([], None), None
            integer_image.crs = 4326
            integer_image.transform = rio.transform.from_origin(10, 50, 0.001, 0.001)
        integer_image.to_file(filename)
        source = gu.Raster(filename)
        options = {"crs": 4326, "bounds": (9.95, 49.85, 10.15, 50.1), "grid_size": (53, 67), "resampling": "nearest"}

        # Reproject eagerly, and check we have nodata on the area outside the original image
        expected = gu.Raster(filename).reproject(**options)
        assert np.ma.getmaskarray(expected.data)[:, :7, :9].all()

        # Do the same with Dask/MP in chunks, check laziness/loading
        if backend == "dask":
            lazy = gu.open_raster(str(filename), chunks={"band": 1, "y": 7, "x": 9})
            result = lazy.rst.reproject(**options)
            assert not lazy.rst.is_loaded
            assert not result.rst.is_loaded
            assert result.rst._chunks is not None
            values = result.compute().values
            assert not lazy.rst.is_loaded
        else:
            config = MultiprocConfig(chunks=(7, 9), outfile=str(tmp_path / "empty_chunks.tif"))
            result = source.reproject(mp_config=config, **options)
            assert not source.is_loaded
            assert not result.is_loaded
            assert result.dtype == np.dtype("int16")
            values = result.to_nanarray()

        # Check exact equality with eager
        np.testing.assert_array_equal(values, expected.to_nanarray())
        assert not source.is_loaded


class TestReprojectionOperators:
    """
    Test module for reprojection with custom operators, neighborhoods, and uncertainty.

    Tests on interpolators are covered in test_operators/test_interpolator.py.
    Tests on fractional area and uncertainty are covered in test_operators/test_reducer.py.
    """

    def test_reproject__custom_interpolator(self) -> None:
        """Checks a custom interpolator for reprojection."""

        # Use same input/output transform
        values = np.arange(25, dtype=np.float64).reshape(5, 5)
        # Make neighboring windows have different ranges from the centered window
        values[2, 2] = 100.0
        transform = rio.transform.from_origin(0, 5, 1, 1)
        raster = gu.Raster.from_array(values, transform=transform, crs=4326, nodata=-9999)
        reference = gu.Raster.from_array(np.zeros((5, 5)), transform=transform, crs=4326, nodata=-9999)

        # We use a custom operator that replace each center value by the range (max - min) in 3x3 window
        operator = WindowRangeInterpolator()
        result = raster.reproject(reference, resampling=operator)
        assert result is not None
        # We check we match the range computed manually with NumPy
        assert result.data[2, 2] == np.max(values[1:4, 1:4]) - np.min(values[1:4, 1:4])
        assert operator.batch_calls == 1

    def test_reproject__custom_reducer(self) -> None:
        """Checks a custom reducer for reprojection."""

        # Downsample from 4x4 to 2x2, matching each destination pixel by a 2x2 source block exactly
        values = np.arange(16, dtype=np.float64).reshape(4, 4)
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 4, 1, 1), crs=4326, nodata=-9999)
        reference = gu.Raster.from_array(
            np.zeros((2, 2)),
            rio.transform.from_origin(0, 4, 2, 2),
            crs=4326,
            nodata=-9999,
        )

        # Reproject with custom mean
        result = raster.reproject(reference, resampling=Mean())
        assert result is not None

        # We check we match exactly the mean of the corresponding 2 x 2 source block
        expected = np.empty(reference.shape, dtype=np.float64)
        for row in range(reference.shape[0]):
            for column in range(reference.shape[1]):
                source_block = values[2 * row : 2 * row + 2, 2 * column : 2 * column + 2]
                expected[row, column] = np.mean(source_block)
        np.testing.assert_array_equal(result.data, expected)

    @pytest.mark.parametrize("operator_type", [Mean, Sum, NoSupportReducer])
    def test_reproject__neighborhood(self, operator_type: type[Reducer]) -> None:
        """Checks a custom neighborhood for reprojection (has to be grid neighborhood!)."""

        # Create synthetic raster, and neighborhood with window of size 3 on custom operator
        values = np.arange(25, dtype=np.float64).reshape(5, 5) ** 2
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 5, 1, 1), crs=32632, nodata=-9999)
        neighborhood = GridNeighbours(size=3)
        operator = operator_type(neighborhood=neighborhood)

        # Reproject
        result = raster.reproject(raster, resampling=operator)
        assert result is not None
        assert operator.default_neighborhood is neighborhood

        # Compare with NumPy
        calculation = np.mean if operator_type is Mean else np.sum
        # The center will use 3x3 = 9 cells, the upper-left has only 4 cells inside
        expected = [calculation(values[1:4, 1:4]), calculation(values[:2, :2])]
        np.testing.assert_allclose([result.data[2, 2], result.data[0, 0]], expected, rtol=0, atol=1e-13)

    @pytest.mark.parametrize("operator_type", [Mean, Sum, Minimum, Maximum, Median, NoSupportReducer])
    @pytest.mark.parametrize(
        "neighborhood",
        [
            GridNeighbours(size=3),
            GridNeighbours(size=5, shape="circular"),
            GridNeighbours(((-2, 1), (0, 0), (1, -1))),
        ],
        ids=["square", "circular", "offsets"],
    )
    @pytest.mark.parametrize("nodata_propagation", ["ignore", "propagate"])
    def test_reproject__matches_resample_at_points(
        self,
        operator_type: type[Reducer],
        neighborhood: GridNeighbours,
        nodata_propagation: Literal["ignore", "propagate"],
    ) -> None:
        """
        Checks that reducer reprojection matches resampling at points on complex (rotated) grids,
        including nodata and edges behaviour.
        """

        # Two bands with different values so neighborhood/nodata affect the result
        values = np.arange(42, dtype=np.float64).reshape(6, 7) ** 2
        values[2, 3] = np.nan
        transform = Affine.translation(500000, 4640000) * Affine.rotation(23) * Affine.scale(2, -3)
        raster = gu.Raster.from_array(np.stack((values, values + 100)), transform, crs=32632, nodata=-9999)
        reference = gu.Raster.from_array(
            np.zeros((8, 9)), transform * Affine.translation(-0.75, -0.875), crs=32632, nodata=-9999
        )

        # Reproject with operator, neighborhood
        operator = operator_type(neighborhood=neighborhood)
        result = raster.reproject(reference, resampling=operator, nodata_propagation=nodata_propagation)
        assert result is not None

        # Extract coordinates, and apply the same calculation per point through resample_at_points()
        rows, columns = np.indices(reference.shape)
        x, y = reference.ij2xy(rows.reshape(-1), columns.reshape(-1), force_offset="center")
        for band in (1, 2):
            expected = raster.resample_at_points(
                (x, y), method=operator, band=band, as_array=True, nodata_handling=nodata_propagation
            )
            # Exact equality for any operator/neighborhood
            np.testing.assert_array_equal(result.to_nanarray()[band - 1], expected.reshape(reference.shape))

    def test_reproject__automatic_neighbours(self) -> None:
        """Checks that neighborhood default uses only nearby raster."""

        # Use an interpolator without explicit neighborhood
        values = np.arange(25, dtype=float).reshape(5, 5)
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 5, 1, 1), crs=4326)
        operator = LocalMeanInterpolator()

        # Reproject
        result = raster.reproject(raster, resampling=operator)
        assert result is not None

        # Check equality
        assert result.to_nanarray()[2, 2] == np.mean(values[1:4, 1:4])  # Should be using 3x3
        assert operator.default_neighborhood is None

    @pytest.mark.parametrize("operator", [Nearest(), Linear()])
    @pytest.mark.parametrize("rotation", [0, 23])
    def test_reproject__subpixel_precision(self, operator: Interpolator, rotation: int) -> None:
        """Checks that reprojection with interpolators uses subpixel centers on complex grids (shifted, rotated)."""

        # We use a plane to know interpolated value at any fractional row/column
        rows, columns = np.indices((6, 7))
        values = (10 * rows + columns).astype(np.float64)
        transform = Affine.translation(500000, 4640000) * Affine.rotation(rotation) * Affine.scale(0.02, -0.03)
        raster = gu.Raster.from_array(values, transform, crs=32632, nodata=-9999)
        # Destination is shifted 1.375 rows and 1.25 columns
        reference = gu.Raster.from_array(
            np.zeros((3, 4)), transform * Affine.translation(1.25, 1.375), crs=32632, nodata=-9999
        )

        # Reproject
        result = raster.reproject(reference, resampling=operator)
        assert result is not None

        # The affine translation shifts target centers in source pixel coordinates, even when the grid is rotated
        # Target (row, column) maps to source (row + 1.375, column + 1.25)
        target_rows, target_columns = np.indices(reference.shape)
        expected = 10 * target_rows + target_columns
        if isinstance(operator, Nearest):
            # Nearest rounds row + 1.375 and column + 1.25 to row + 1 and column + 1: 10 * 1 + 1 = 11
            np.testing.assert_array_equal(result.to_nanarray(), expected + 11)
        else:
            # Linear samples the plane at the fractional shift: 10 * 1.375 + 1.25 = 15
            np.testing.assert_allclose(result.to_nanarray(), expected + 15, rtol=0, atol=1e-6)

    @pytest.mark.parametrize("operator", [Nearest(), Linear(), Mean(), Mean(neighborhood=GridNeighbours(size=3))])
    def test_reproject__centers_and_footprints_across_crs(self, operator: Interpolator | Reducer) -> None:
        """
        Checks that resampling methods behave properly when reprojecting in a different CRS.

        We write independent calculations to verify nearest cell, linear interpolation, 3x3 mean, and a mean
        weighted by overlap with the source cells.
        The test also checks that only the 4th call to overlap-weighted mean triggers internally the function to
        reproject the target cell corners.
        """

        # Place one geographic destination cell inside a 4x4 Web Mercator raster
        # The values are a plane with formula 4 * row + column, that we can easily use to verify interpolation
        values = np.arange(16, dtype=np.float64).reshape(4, 4)
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 400, 100, 100), crs=3857, nodata=-9999)
        reference = gu.Raster.from_array(
            np.zeros((1, 1)), rio.transform.from_origin(0.0005, 0.0031, 0.0024, 0.0024), crs=4326, nodata=-9999
        )
        with patch("geoutils.projtools.reproject_points", wraps=gu.projtools.reproject_points) as transform_points:
            result = raster.reproject(
                reference,
                resampling=operator,
                coverage="fractional"
                if isinstance(operator, Reducer) and operator.default_neighborhood is None
                else None,
            )
        assert result is not None

        # We check projtools was called the right number of times internally: only fractional area needs to project
        # the four corners, other operators only need to project the center
        projected_counts = [len(call.args[0][0]) for call in transform_points.call_args_list]
        uses_footprint = isinstance(operator, Reducer) and operator.default_neighborhood is None
        assert projected_counts == ([1, 4] if uses_footprint else [1])

        # We back-project left/center/right longitudes and bottom/center/top latitudes into the source CRS
        target_bounds = reference.bounds
        longitudes = np.linspace(target_bounds.left, target_bounds.right, 3)
        latitudes = np.linspace(target_bounds.bottom, target_bounds.top, 3)
        projected_x, projected_y = gu.projtools.reproject_points(
            (longitudes, latitudes), in_crs=reference.crs, out_crs=raster.crs
        )
        x = np.asarray(projected_x)
        y = np.asarray(projected_y)

        # We can locate the projected center within the source grid
        source_bounds = raster.bounds
        source_width, source_height = raster.res
        center_row = (source_bounds.top - y[1]) / source_height - 0.5
        center_column = (x[1] - source_bounds.left) / source_width - 0.5

        # For nearest, expected is the closest cell
        if isinstance(operator, Nearest):
            expected = values[int(np.floor(center_row + 0.5)), int(np.floor(center_column + 0.5))]
        # For linear, we can use the plane 4 * row + column with projected center values
        elif isinstance(operator, Linear):
            expected = 4 * center_row + center_column
        # The projected center lies in source row 1, column 1, so its window contains these nine complete cells
        elif operator.default_neighborhood is not None:
            expected = np.mean(values[:3, :3])
        else:
            # The projected output cell is a rectangle inside the source raster
            # Width = smaller right edge - larger left edge, clipped at zero
            left_edges = source_bounds.left + np.arange(values.shape[1]) * source_width
            widths = np.maximum(0, np.minimum(left_edges + source_width, x[2]) - np.maximum(left_edges, x[0]))

            # Same for height
            top_edges = source_bounds.top - np.arange(values.shape[0]) * source_height
            heights = np.maximum(0, np.minimum(top_edges, y[2]) - np.maximum(top_edges - source_height, y[0]))

            # Fractional area of cells is width times height
            areas = heights[:, None] * widths[None, :]
            # Finally, we divide the area-weighted values by the total covered area
            expected = np.sum(areas * values) / np.sum(areas)

        # Check equality of the one output pixel with expected
        np.testing.assert_allclose(result.to_nanarray(), [[expected]], rtol=0, atol=1e-9)

    def test_reproject__nodata_gdal(self) -> None:
        """Checks that area reduction omits nodata even when a custom reducer normally propagates it."""

        # We create a shifted output with overlap on one nodata cell
        values = np.full((2, 2), 2.0)
        values[0, 0] = np.nan
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 2, 1, 1), crs=32631, nodata=-9999)
        reference = gu.Raster.from_array(np.zeros((1, 1)), rio.transform.from_origin(0.5, 1.5, 1, 1), crs=32631)

        # Default area reduction only uses finite values to derive the mean
        result = raster.reproject(reference, resampling=PropagatingMeanReducer())
        assert result is not None
        assert result.to_nanarray()[0, 0] == 2.0


class TestReprojectionGDALOperators:
    """Test module for GDAL agreement with our own interpolators/reducers."""

    def test_reproject__default_interpolator_nodata(self) -> None:
        """Checks that interpolation masks the nearest NaN cell by default, like GDAL."""

        # Synthetic raster with a missing cell
        values = np.array([[np.nan, 2.0], [3.0, 4.0]])
        source = gu.Raster.from_array(values, rio.transform.from_origin(0, 2, 1, 1), crs=32632, nodata=-9999)
        reference = gu.Raster.from_array(
            np.zeros((1, 1)), rio.transform.from_origin(0.25, 1.75, 1, 1), crs=32632, nodata=-9999
        )

        # GDAL "bilinear" should equal default/"nearest" with Linear(), but not "ignore"
        default = source.reproject(reference, resampling=Linear())
        nearest = source.reproject(reference, resampling=Linear(), nodata_propagation="nearest")
        ignored = source.reproject(reference, resampling=Linear(), nodata_propagation="ignore")
        expected = source.reproject(reference, resampling="bilinear")
        assert default is not None and nearest is not None and ignored is not None and expected is not None
        assert np.isnan(default.to_nanarray()[0, 0])
        np.testing.assert_array_equal(default.to_nanarray(), nearest.to_nanarray())
        np.testing.assert_array_equal(default.to_nanarray(), expected.to_nanarray())
        assert np.isfinite(ignored.to_nanarray()[0, 0])

    def test_reproject__gdal_mean_weights_outside_source(self) -> None:
        """Checks that GDAL and fractional means use their distinct source-edge weights."""

        # The destination covers both rows and columns but starts half a cell beyond the source
        values = np.array([[1.0, 3.0], [10.0, 30.0]])
        source = gu.Raster.from_array(values, rio.transform.from_origin(0, 2, 1, 1), crs=32632, nodata=-9999)
        reference = gu.Raster.from_array(
            np.zeros((1, 1)), rio.transform.from_origin(-0.5, 2.5, 2, 2), crs=32632, nodata=-9999
        )

        # GDAL weights the outer row/column by 1.5, while true covered area clips them to 1
        gdal = source.reproject(reference, resampling="average")
        matched = source.reproject(reference, resampling=Mean())
        fractional = source.reproject(reference, resampling=Mean(), area_weighting="intersection")

        # Weighted means are 4.875 with GDAL's extended edge and 6.667 with clipped area
        assert gdal is not None and matched is not None and fractional is not None
        np.testing.assert_allclose(matched.to_nanarray(), gdal.to_nanarray(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(matched.to_nanarray(), [[4.875]], rtol=0, atol=1e-12)
        np.testing.assert_allclose(fractional.to_nanarray(), [[20 / 3]], rtol=0, atol=1e-12)

    def test_reproject__interior_diagonal_bounds_differ_from_polygon(self) -> None:
        """Checks that rotated interior cells use different diagonal bounds and polygon area weights."""

        # Rotate destination cells inside the source raster, away from its outer boundary
        rows, columns = np.indices((20, 20))
        values = (rows**2 + 3 * columns**2).astype(float)
        source_transform = rio.transform.from_origin(0, 20, 1, 1)
        source = gu.Raster.from_array(values, source_transform, crs=32632, nodata=-9999)
        target_transform = source_transform * Affine.translation(6, 5) * Affine.rotation(12) * Affine.scale(1.1)
        reference = gu.Raster.from_array(np.zeros((6, 6)), target_transform, crs=32632, nodata=-9999)

        # Compare the diagonal rectangle with the actual four-corner polygon
        expected = source.reproject(reference, resampling="average")
        bounds = source.reproject(reference, resampling=Mean())
        polygon = source.reproject(reference, resampling=Mean(), area_weighting="intersection")
        assert expected is not None and bounds is not None and polygon is not None
        np.testing.assert_allclose(bounds.to_nanarray(), expected.to_nanarray(), rtol=0, atol=1e-10)
        assert np.max(np.abs(bounds.to_nanarray()[1:-1, 1:-1] - polygon.to_nanarray()[1:-1, 1:-1])) > 0.01

    @pytest.mark.parametrize(
        ("method", "operator"),
        [
            ("nearest", Nearest()),
            ("bilinear", Linear()),
            ("cubic", RasterConvolution("cubic")),
            ("cubic_spline", RasterConvolution("cubic_spline")),
            ("lanczos", RasterConvolution("lanczos")),
            ("average", Mean()),
            ("sum", Sum()),
            ("min", Minimum()),
            ("max", Maximum()),
            ("rms", RootMeanSquare()),
            ("mode", Mode(weighted=False, tie_break="first_to_mode")),
            ("med", Median(method="inverted_cdf")),
            ("q1", Quantile(0.25, method="inverted_cdf")),
            ("q3", Quantile(0.75, method="inverted_cdf")),
        ],
    )
    @pytest.mark.parametrize("geometry", ["shifted", "different_crs", "rotated", "rotated_different_crs"])
    def test_reproject__operator_matches_gdal(
        self, method: str, operator: Interpolator | Reducer, geometry: str
    ) -> None:
        """Checks that GeoUtils operators agree with GDAL on shifted, projected, and rotated grids."""

        # Rasterio before 1.5 cannot disable GDAL's CRS approximation for an exact comparison
        if geometry in ("different_crs", "rotated_different_crs") and Version(rio.__version__) < Version("1.5.0"):
            pytest.skip("Exact GDAL CRS transforms require Rasterio 1.5 or newer.")

        # We create a synthetic raster with varied values and one NaN
        rows, columns = np.indices((30, 30))
        values = (columns * 0.7 + rows * 0.2 + (columns * rows % 7) * 1.3).astype(np.float64)
        values[11, 13] = np.nan
        source_transform = rio.transform.from_origin(500_000, 5_100_000, 100, 100)
        if geometry in ("rotated", "rotated_different_crs"):
            source_transform = Affine.translation(500_000, 5_100_000) * Affine.rotation(10) * Affine.scale(100, -100)
        source = gu.Raster.from_array(values, transform=source_transform, crs=32632, nodata=-9999)

        # We define different "references" for destination: same CRS, different CRS, rotated
        if geometry in ("different_crs", "rotated_different_crs"):
            reference = source.reproject(crs=4326, grid_size=(30, 30), resampling="nearest")
        elif geometry == "rotated":
            shift, scale = (2.17, 1.37) if isinstance(operator, Interpolator) else (2.0, 2.0)
            transform = source_transform * Affine.translation(shift, shift) * Affine.scale(scale, scale)
            reference = gu.Raster.from_array(np.zeros((12, 12)), transform=transform, crs=32632, nodata=-9999)
        else:
            transform = rio.transform.from_origin(500_350, 5_099_650, 120, 130)
            reference = gu.Raster.from_array(np.zeros((12, 12)), transform=transform, crs=32632, nodata=-9999)
        assert reference is not None

        # We reproject with both GDAL method and operator
        expected = source.reproject(reference, resampling=method)
        actual = source.reproject(reference, resampling=operator)

        # We check almost equality
        assert expected is not None and actual is not None
        expected_values = expected.to_nanarray()
        actual_values = actual.to_nanarray()
        assert np.array_equal(np.isnan(actual_values), np.isnan(expected_values))
        np.testing.assert_allclose(actual_values, expected_values, rtol=0, atol=1e-4)

    def test_reproject__error_intersection_area_with_fixed_window(self) -> None:
        """Checks an error is raised for polygon area weighting with a fixed reducer window."""

        # A fixed window selects source cells independently of the destination footprint
        source = gu.Raster.from_array(np.ones((5, 5)), rio.transform.from_origin(0, 5, 1, 1), crs=32632)
        reference = gu.Raster.from_array(np.zeros((2, 2)), rio.transform.from_origin(0, 5, 2, 2), crs=32632)

        # Polygon area weighting only applies to a reducer's destination-cell neighborhood
        with pytest.raises(ValueError, match="Intersection area weighting requires"):
            source.reproject(reference, resampling=Mean(), area_weighting="intersection", window=3)

    def test_reproject__error_intersection_area_without_fractional_coverage(self) -> None:
        """Checks an error is raised for polygon area weighting without fractional coverage."""

        # Polygon area weights need a destination footprint
        source = gu.Raster.from_array(np.ones((5, 5)), rio.transform.from_origin(0, 5, 1, 1), crs=32632)
        reference = gu.Raster.from_array(np.zeros((2, 2)), rio.transform.from_origin(0, 5, 2, 2), crs=32632)

        # Polygon intersection cannot be combined with whole-cell selection
        with pytest.raises(ValueError, match="requires coverage='fractional'"):
            source.reproject(reference, resampling=Mean(), area_weighting="intersection", coverage="all_touched")


class TestReprojectionOverlapBackends:
    """Test module for matching raster-cell coverage across Numba, ExactExtract, and Shapely."""

    @pytest.mark.parametrize("geometry", ["shifted", "rotated", "sheared", "different_crs"])
    @pytest.mark.parametrize("backend", ["auto", "numba", "exactextract"])
    @pytest.mark.parametrize(
        ("coverage", "area_weighting"),
        [
            ("fractional", "intersection"),
            ("fractional", "diagonal_bounds"),
            ("center", "diagonal_bounds"),
            ("all_touched", "diagonal_bounds"),
        ],
    )
    @pytest.mark.parametrize("propagation", ["ignore", "propagate"])
    @pytest.mark.parametrize("operator", [Mean(), Quantile(0.25, method="inverted_cdf")])
    def test_reproject__overlap_backends_agree(
        self,
        geometry: str,
        backend: str,
        coverage: Literal["fractional", "center", "all_touched"],
        area_weighting: Literal["diagonal_bounds", "intersection"],
        propagation: Literal["ignore", "propagate"],
        operator: Reducer,
    ) -> None:
        """Checks that raster overlap methods agree for coverage, nodata, and source grid transforms."""

        if backend in ("numba", "exactextract"):
            pytest.importorskip(backend)

        # Vary source values and include nodata so coverage and valid-area weighting both matter
        rows, columns = np.indices((24, 24))
        values = (rows * 0.4 + columns * 0.7 + (rows * columns % 5)).astype(float)
        values[8, 9] = np.nan
        source_transform = rio.transform.from_origin(500_000, 5_100_000, 100, 100)
        if geometry == "rotated":
            source_transform = Affine.translation(500_000, 5_100_000) * Affine.rotation(13) * Affine.scale(100, -100)
        elif geometry == "sheared":
            source_transform = Affine(100, 17, 500_000, 7, -100, 5_100_000)
        source = gu.Raster.from_array(values, transform=source_transform, crs=32632, nodata=-9999)

        # Place unaligned destination cells in the source, or transform their corners from another CRS
        if geometry == "different_crs":
            reference = source.reproject(crs=4326, grid_size=(18, 18), resampling="nearest")
        else:
            target_transform = source_transform * Affine.translation(1.3, 2.1) * Affine.rotation(4) * Affine.scale(1.25)
            reference = gu.Raster.from_array(np.zeros((12, 12)), transform=target_transform, crs=32632, nodata=-9999)
        assert reference is not None

        # Shapely is the geometry reference for direct and fraction-based reductions
        expected = source.reproject(
            reference,
            resampling=operator,
            coverage=coverage,
            area_weighting=area_weighting,
            nodata_propagation=propagation,
            overlap_backend="shapely",
        )
        actual = source.reproject(
            reference,
            resampling=operator,
            coverage=coverage,
            area_weighting=area_weighting,
            nodata_propagation=propagation,
            overlap_backend=backend,
        )
        assert expected is not None and actual is not None
        expected_values = expected.to_nanarray()
        actual_values = actual.to_nanarray()
        assert np.array_equal(np.isnan(actual_values), np.isnan(expected_values))
        np.testing.assert_allclose(actual_values, expected_values, rtol=0, atol=1e-5)


@pytest.mark.skipif(find_spec("dask") is None, reason="Requires Dask")
class TestReprojectionOperatorsChunked:
    """
    Test module for eager and lazy agreement when reprojecting operator neighborhoods.

    Tests comparing SciPy/Numba backends are covered in test_operators/.
    """

    def test_reproject__default_neighb_chunk_invariance(self) -> None:
        """Checks that default neighborhoods agree between eager and chunked for default neighborhood."""

        # Synthetic raster with local reducer
        values = np.arange(42, dtype=float).reshape(6, 7)
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 6, 1, 1), crs=4326, nodata=-9999)
        operator = LocalMeanInterpolator()

        # Reproject onto the same grid, reducing
        expected = raster.reproject(raster, resampling=operator)
        lazy_source = raster.to_xarray().chunk({"y": 2, "x": 3})
        lazy_result = lazy_source.rst.reproject(ref=raster, resampling=operator)
        assert expected is not None
        assert hasattr(lazy_source.data, "compute")
        assert hasattr(lazy_result.data, "compute")

        # Check exact quality
        np.testing.assert_array_equal(np.asarray(lazy_result.compute()).squeeze(), expected.to_nanarray())

    @pytest.mark.parametrize(
        "operator",
        [
            Nearest(),
            Linear(),
            RasterConvolution("cubic"),
            RasterConvolution("cubic_spline"),
            RasterConvolution("lanczos"),
            Mean(),
            Mean(neighborhood=GridNeighbours(size=3)),
        ],
    )
    def test_reproject__crs_chunk_invariance(self, tmp_path: Any, operator: Interpolator | Reducer) -> None:
        """Checks that reprojection across CRS with operators is consistent across eager, Dask and MP execution."""

        # Synthetic 4x4 with chunks of 2x3 to get uneven blocks
        values = np.arange(16, dtype=np.float64).reshape(4, 4)
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 400, 100, 100), crs=3857, nodata=-9999)
        source_path = tmp_path / "operator_source.tif"
        raster.to_file(source_path)
        reference = gu.Raster.from_array(
            np.zeros((4, 5)), rio.transform.from_origin(0.0005, 0.0031, 0.0004, 0.0004), crs=4326, nodata=-9999
        )

        # Reproject all
        expected = raster.reproject(reference, resampling=operator)
        lazy_source = open_raster(source_path, chunks={"y": 2, "x": 3})
        lazy_result = lazy_source.rst.reproject(ref=reference, resampling=operator)
        unloaded_source = gu.Raster(source_path)
        mp_config = MultiprocConfig(chunks=(2, 3), outfile=str(tmp_path / "operator_result.tif"))
        mp_result = unloaded_source.reproject(reference, resampling=operator, mp_config=mp_config)

        # Dask output should be lazy, MP without loading either raster in memory
        assert expected is not None and mp_result is not None
        assert not lazy_source._in_memory
        assert not lazy_result._in_memory
        assert hasattr(lazy_result.data, "compute")
        assert not unloaded_source.is_loaded
        assert not mp_result.is_loaded

        # Check exact equality
        dask_values = np.asarray(lazy_result.compute()).squeeze()
        mp_values = mp_result.to_nanarray()
        if isinstance(operator, Mean) and operator.default_neighborhood is None:
            # Allow tiny numerical differences for fractional area cases
            np.testing.assert_allclose(dask_values, expected.to_nanarray(), rtol=0, atol=1e-14)
            np.testing.assert_allclose(mp_values, expected.to_nanarray(), rtol=0, atol=1e-14)
        else:
            np.testing.assert_array_equal(dask_values, expected.to_nanarray())
            np.testing.assert_array_equal(mp_values, expected.to_nanarray())

    @pytest.mark.parametrize("backend", ["numba", "exactextract", "shapely"])
    @pytest.mark.parametrize("source_angle", [0, 10, 60])
    @pytest.mark.parametrize("area_weighting", ["intersection", "diagonal_bounds"])
    def test_reproject__overlap_backend_chunk_invariance(
        self, tmp_path: Any, backend: str, source_angle: int, area_weighting: Literal["diagonal_bounds", "intersection"]
    ) -> None:
        """Checks weighted reprojection across eager, Dask, and MP with north-up or rotated sources."""

        pytest.importorskip(backend)

        # Split an 8x9 source into chunks of 3x4, leaving shorter final row and column chunks
        rows, columns = np.indices((8, 9))
        values = (rows * 0.4 + columns * 0.7).astype(float)
        values[3, 4] = np.nan
        transform = Affine.translation(500_000, 5_100_000) * Affine.rotation(source_angle) * Affine.scale(100, -100)
        raster = gu.Raster.from_array(values, transform=transform, crs=32632, nodata=-9999)
        source_path = tmp_path / "overlap_source.tif"
        raster.to_file(source_path)
        target_transform = transform * Affine.translation(0.8, 1.2) * Affine.rotation(3) * Affine.scale(1.35)
        reference = gu.Raster.from_array(np.zeros((5, 6)), target_transform, crs=32632, nodata=-9999)

        # Calculate the same weighted mean without chunks, with Dask, and with multiprocessing
        options = {
            "resampling": Mean(),
            "coverage": "fractional",
            "area_weighting": area_weighting,
            "overlap_backend": backend,
        }
        expected = raster.reproject(reference, **options)
        lazy_source = open_raster(source_path, chunks={"y": 3, "x": 4})
        lazy_result = lazy_source.rst.reproject(ref=reference, **options)
        unloaded_source = gu.Raster(source_path)
        mp_config = MultiprocConfig(chunks=(3, 4), outfile=str(tmp_path / f"{backend}-overlap.tif"))
        mp_result = unloaded_source.reproject(reference, mp_config=mp_config, **options)

        # Both chunked inputs and outputs stay unloaded until values are requested
        assert expected is not None and mp_result is not None
        assert not lazy_source._in_memory
        assert not lazy_result._in_memory
        assert hasattr(lazy_result.data, "compute")
        assert not unloaded_source.is_loaded
        assert not mp_result.is_loaded

        # Weight sums can change slightly when the source is split across chunks
        expected_values = expected.to_nanarray()
        dask_values = np.asarray(lazy_result.compute()).squeeze()
        mp_values = mp_result.to_nanarray()
        np.testing.assert_allclose(dask_values, expected_values, rtol=0, atol=2e-7)
        np.testing.assert_allclose(mp_values, expected_values, rtol=0, atol=2e-7)

    @pytest.mark.parametrize(
        "neighborhood",
        [
            GridNeighbours(size=13),
            GridNeighbours(size=13, shape="circular"),
            GridNeighbours(((-6, 0), (0, 0), (0, 7))),
        ],
        ids=["square", "circular", "offsets"],
    )
    def test_reproject__reducer_neighb_chunk_invariance(self, tmp_path: Any, neighborhood: GridNeighbours) -> None:
        """Checks Dask and MP give same results as eager for various neighborhoods."""

        # 1/ Write raster

        # We pick chunks of 3x2 to leave shorter blocks in 11x13 raster
        values = np.arange(143, dtype=np.float64).reshape(11, 13) ** 2
        values[4, 6] = np.nan
        raster = gu.Raster.from_array(
            np.stack((values, values + 1000)), rio.transform.from_origin(0, 11, 1, 1), crs=32632, nodata=-9999
        )
        source_path = tmp_path / "window_source.tif"
        raster.to_file(source_path)
        reference = gu.Raster.from_array(
            np.zeros((12, 14)), rio.transform.from_origin(-0.25, 11.375, 1, 1), crs=32632, nodata=-9999
        )
        operator = Mean(neighborhood=neighborhood)

        # 2/ Run eager, Dask and MP
        expected = raster.reproject(reference, resampling=operator)
        lazy_source = open_raster(source_path, chunks={"y": 3, "x": 2})
        lazy_result = lazy_source.rst.reproject(ref=reference, resampling=operator)
        unloaded_source = gu.Raster(source_path)
        mp_config = MultiprocConfig(chunks=(3, 2), outfile=str(tmp_path / "window_result.tif"))
        mp_result = unloaded_source.reproject(reference, resampling=operator, mp_config=mp_config)

        # Dask lazy, and MP unloaded for input/output
        assert expected is not None and mp_result is not None
        assert not lazy_source._in_memory
        assert not lazy_result._in_memory
        assert hasattr(lazy_result.data, "compute")
        assert not unloaded_source.is_loaded
        assert not mp_result.is_loaded

        # Check for almost equality
        np.testing.assert_allclose(np.asarray(lazy_result.compute()), expected.to_nanarray(), rtol=1e-15, atol=1e-12)
        np.testing.assert_allclose(mp_result.to_nanarray(), expected.to_nanarray(), rtol=1e-15, atol=1e-12)


class TestReprojectionErrors:
    """Test module reprojection errors."""

    def test_reproject__error_removed_nodata_rule(self) -> None:
        """Checks an error is raised for the removed gdal nodata rule."""

        # Request a custom interpolator so reproject() validates its nodata rule
        raster = gu.Raster.from_array(np.ones((3, 3)), rio.transform.from_origin(0, 3, 1, 1), crs=32631)
        reference = gu.Raster.from_array(np.zeros((2, 2)), rio.transform.from_origin(0, 3, 1.5, 1.5), crs=32631)
        options: dict[str, Any] = {"nodata_propagation": "gdal"}

        # The former spelling is not a nodata behavior
        with pytest.raises(ValueError, match="nodata_propagation must be one of"):
            raster.reproject(reference, resampling=Linear(), **options)

    def test_reproject__error_nearest_with_reducer(self) -> None:
        """Checks an error is raised for nearest-source nodata masking with a reducer."""

        # A reducer calculates from intersected source cells
        raster = gu.Raster.from_array(np.ones((3, 3)), rio.transform.from_origin(0, 3, 1, 1), crs=32631)
        reference = gu.Raster.from_array(np.zeros((2, 2)), rio.transform.from_origin(0, 3, 1.5, 1.5), crs=32631)

        # The nearest nodata rule applies to interpolators
        with pytest.raises(ValueError, match="requires an Interpolator"):
            raster.reproject(reference, resampling=Mean(), nodata_propagation="nearest")

    @pytest.mark.parametrize(
        "example", [examples.get_path_test("everest_landsat_b4"), examples.get_path_test("exploradores_aster_dem")]
    )
    def test_reproject__error_nodata(self, example: str) -> None:
        """Checks that reproject() rejects masked data and warns when its fallback nodata conflicts."""

        # Check handling of missing nodata
        r = gu.Raster(example)
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

    def test_reproject__error_argcombinations(self) -> None:
        """Checks that reproject() rejects conflicting settings and an invalid reference."""

        # -- Test additional errors raised for argument combinations -- #
        r = gu.Raster.from_array(np.ones((4, 4)), rio.transform.from_origin(0, 4, 1, 1), crs=32632)
        r2 = r.copy()

        # If both ref and crs are set
        with pytest.raises(InvalidGridError, match="Either 'ref' or 'crs' must be provided"):
            _ = r.reproject(ref=r2, crs=r.crs)

        # Size and res are mutually exclusive
        with pytest.raises(InvalidGridError, match="Both output grid resolution 'res' and shape"):
            _ = r.reproject(grid_size=(10, 10), res=50)

        # If wrong type for `ref`
        with pytest.raises(InvalidGridError, match="Cannot interpret reference grid from"):
            _ = r.reproject(ref=3)

    def test_reproject__error_missing_crs(self) -> None:
        """Checks that we raise an error during reprojection if source raster has no CRS."""

        transform = rio.transform.from_origin(0, 4, 1, 1)
        raster = gu.Raster.from_array(np.ones((4, 4)), transform, crs=None)
        reference = gu.Raster.from_array(np.zeros((4, 4)), transform, crs=32632)
        with pytest.raises(InvalidCRSError, match="Projection not recognized"):
            raster.reproject(reference)

    @pytest.mark.parametrize(
        "mask",
        [
            TestMaskGeotransformations.mask_landsat_b4,
            TestMaskGeotransformations.mask_aster_dem,
            TestMaskGeotransformations.mask_everest,
        ],
    )
    def test_reproject__error_mask_resampling(self, mask: gu.Raster) -> None:
        """Checks that mask reprojection reports a warning for bilinear resampling."""

        # Check warning for resampling other than nearest
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with pytest.raises(
                UserWarning, match=re.escape("Reprojecting a raster mask (boolean type) with a resampling method")
            ):
                mask.reproject(res=50, resampling="bilinear")

    def test_reproject__reducer_fractional_error(self) -> None:
        """Checks that reducers not supporting fractional area raised an error during reprojection."""

        source = gu.Raster.from_array(
            np.arange(4, dtype=np.float64).reshape(2, 2),
            rio.transform.from_origin(0, 2, 1, 1),
            crs=32632,
            nodata=-9999,
        )
        reference = gu.Raster.from_array(
            np.zeros((1, 1)),
            rio.transform.from_origin(0.5, 1.5, 1, 1),
            crs=32632,
            nodata=-9999,
        )

        # A reducer without support_weights=True raises an error
        with pytest.raises(ValueError, match="does not accept support_weights"):
            source.reproject(reference, resampling=NoSupportReducer(), coverage="fractional")
        result = source.reproject(reference, resampling=SupportMassReducer(), coverage="fractional")
        assert result is not None
        assert result.data[0, 0] == 1

    def test_reproject__error_pointneighb(self) -> None:
        """Checks that reproject() rejects PointNeighbours because the source values are raster cells."""

        # Identical grids still validate the reducer configuration before any early return
        raster = gu.Raster.from_array(np.ones((4, 4)), rio.transform.from_origin(0, 4, 1, 1), crs=32631)
        with pytest.raises(ValueError, match="PointNeighbours applies to point sources"):
            raster.reproject(raster, resampling=Mean(neighborhood=PointNeighbours(k=3)))

    @pytest.mark.parametrize("operator_type", [Nearest, Linear])
    def test_reproject__warn_regular_offsets(self, operator_type: type[Interpolator]) -> None:
        """Checks that a larger raster window warns and preserves regular interpolation during reprojection."""

        # A shifted target grid forces interpolation of a nonlinear source surface
        rows, cols = np.indices((6, 6), dtype=float)
        raster = gu.Raster.from_array(rows * cols, rio.transform.from_origin(0, 6, 1, 1), crs=32631)
        reference = gu.Raster.from_array(np.zeros((4, 4)), rio.transform.from_origin(0.2, 5.7, 1, 1), crs=32631)
        expected = raster.reproject(reference, resampling=operator_type())
        with pytest.warns(UserWarning, match="extra GridNeighbours offsets are ignored"):
            result = raster.reproject(reference, resampling=operator_type(neighborhood=GridNeighbours(size=3)))
        np.testing.assert_array_equal(result.to_nanarray(), expected.to_nanarray())

    @pytest.mark.parametrize("raster_gcp_rpc", ["rpc_polynomial", "rpc_rational"], indirect=True)
    @pytest.mark.parametrize("source_method", [None, "GEOTRANSFORM", "GEOLOC_ARRAY", "unknown"])
    def test_reproject__error_conflicting_gcp_rcp(self, raster_gcp_rpc: gu.Raster, source_method: str | None) -> None:
        """Checks an error is raised when a raster stores both GCPs and RPCs at once."""

        # Add GCPs to a raster that already stores RPCs
        source = raster_gcp_rpc
        points = [
            rio.control.GroundControlPoint(row=row, col=col, x=10 + col * 0.001, y=50 - row * 0.001)
            for row in (0, source.height)
            for col in (0, source.width)
        ]
        source.gcps = points, rio.CRS.from_epsg(4326)
        options = {"SRC_METHOD": source_method} if source_method is not None else None
        # We raise an error
        with pytest.raises(ValueError, match="Source has both GCPs and RPCs.*SRC_METHOD"):
            source.reproject(crs=3857, transformer_options=options)
        assert not source.is_loaded
        assert source.gcps == (points, rio.CRS.from_epsg(4326))
        assert source.rpcs is not None

    def test_reproject__error_operator(self, raster_gcp_rpc: gu.Raster) -> None:
        """Checks an error is raised for a generic GeoUtils interpolator on a GCP/RPC source."""
        with pytest.raises(ValueError, match="requires a Rasterio resampling method"):
            raster_gcp_rpc.reproject(crs=3857, resampling=Linear())

    @pytest.mark.parametrize("reference_type", ["raster", "dataarray"])
    def test_reproject__error_gcp_rcp_reference(self, raster_gcp_rpc: gu.Raster, reference_type: str) -> None:
        """Checks an error is raised when GCP/RPC referencing is used as an affine destination grid."""

        reference = raster_gcp_rpc
        reference.crs = 4326
        destination = reference if reference_type == "raster" else gu.open_raster(reference.name)
        if reference_type == "dataarray":
            destination.rst.crs = 4326
        source = gu.Raster.from_array(np.ones((10, 10)), rio.transform.from_origin(10, 50, 0.01, 0.01), crs=4326)

        with pytest.raises(ValueError, match="Reference grid requires an affine grid"):
            source.reproject(ref=destination)
        assert not reference.is_loaded


class TestTransformationErrors:
    """Test module for errors/warnings in transformation functions."""

    @pytest.mark.parametrize("raster_type", ["raster", "dataarray"])
    @pytest.mark.parametrize("loaded", [False, True])
    @pytest.mark.parametrize(
        "method,options",
        [
            pytest.param("crop", {}, id="crop"),
            pytest.param("clip", {}, id="clip"),
            pytest.param("translate", {}, id="translate_georeferenced"),
            pytest.param("translate", {"distance_unit": "pixel", "inplace": True}, id="translate_pixel_inplace"),
        ],
    )
    def test_methods__error_gcp_rcp_georeferencing(
        self, raster_gcp_rpc: gu.Raster, raster_type: str, loaded: bool, method: str, options: dict[str, Any]
    ) -> None:
        """Checks an error is raised for affine transformations on GCP/RPC rasters without changing loading."""

        # Open image referenced only by GCPs/RPCs
        raster = raster_gcp_rpc if raster_type == "raster" else gu.open_raster(raster_gcp_rpc.name).rst
        if loaded:
            raster.load()
        arguments = (1, 2) if method == "translate" else ((10, 49.95, 10.05, 50),)
        assert raster.is_loaded is loaded

        # We save the GCPs/RPCs to check the error is raised BEFORE modifying the metadata
        transform = raster.transform
        points = [point.asdict() for point in raster.gcps[0]]
        gcp_crs = raster.gcps[1]
        rpcs = raster.rpcs.to_gdal() if raster.rpcs is not None else None

        # We raise an error, saying to use reproject() before to define an affine grid
        with pytest.raises(ValueError, match=r"requires an affine grid.*Call reproject\(\) first"):
            getattr(raster, method)(*arguments, **options)
        assert raster.is_loaded is loaded

        # Then, we check metadata did not change
        assert raster.transform == transform
        assert [point.asdict() for point in raster.gcps[0]] == points
        assert raster.gcps[1] == gcp_crs
        assert (raster.rpcs.to_gdal() if raster.rpcs is not None else None) == rpcs

    @pytest.mark.parametrize("method", ["crop", "clip"])
    def test_methods__error_gcp_rcp_reference(self, raster_gcp_rpc: gu.Raster, method: str) -> None:
        """Checks an error is raised for crop/clip with a GCP/RPC raster."""

        reference = raster_gcp_rpc
        reference.crs = 4326
        source = gu.Raster.from_array(np.ones((10, 10)), rio.transform.from_origin(10, 50, 0.01, 0.01), crs=4326)
        with pytest.raises(ValueError, match="requires an affine grid"):
            getattr(source, method)(reference)
        assert not reference.is_loaded


class TestTransformationErrorsChunked:
    """Test module for crop(), clip() and translate() errors before Dask/MP loads inputs or writes files."""

    @pytest.mark.parametrize(
        "backend,method", [("dask", "crop"), ("dask", "clip"), ("dask", "translate"), ("mp", "clip")]
    )
    def test_methods__error_gcp_rcp_georeferencing(
        self, raster_gcp_rpc: gu.Raster, tmp_path: Path, backend: str, method: str
    ) -> None:
        """Checks an error is raised for crop(), clip() or translate() before Dask/MP loads inputs or writes files."""

        # Open both loaded/unloaded rasters with GCPs/RCPs
        source = raster_gcp_rpc
        eager = gu.Raster(source.name, load_data=True)
        arguments = (1, 2) if method == "translate" else ((10, 49.95, 10.05, 50),)
        assert eager.is_loaded
        assert not source.is_loaded

        # We open in chunks with Dask/MP, check loading/laziness
        chunked: gu.Raster | gu.RasterAccessor
        options: dict[str, Any] = {}
        if backend == "dask":
            import_optional("dask")
            lazy = gu.open_raster(source.name, chunks={"band": 1, "y": 7, "x": 9})
            chunked = lazy.rst
            original_data = lazy.data
            assert chunked._chunks is not None
        else:
            outfile = tmp_path / "clipped.tif"
            config = MultiprocConfig(chunks=(7, 9), outfile=str(outfile))
            chunked = source
            options["mp_config"] = config
            assert not outfile.exists()
        assert not chunked.is_loaded

        # We check all raise the same error correctly, and laziness is maintained (error happens early enough that we
        # don't update the source data or create a result from the method)
        with pytest.raises(ValueError, match="requires an affine grid") as eager_error:
            getattr(eager, method)(*arguments)
        with pytest.raises(ValueError, match="requires an affine grid") as chunked_error:
            getattr(chunked, method)(*arguments, **options)
        assert str(chunked_error.value) == str(eager_error.value)
        assert eager.is_loaded
        assert not chunked.is_loaded
        assert not source.is_loaded
        if backend == "dask":
            assert lazy.data is original_data
            assert lazy.rst._chunks is not None
        else:
            assert not outfile.exists()
