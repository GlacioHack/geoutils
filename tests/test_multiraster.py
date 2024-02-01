"""
Test tools involving multiple rasters.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pyproj
import pytest
import rasterio as rio

import geoutils as gu
from geoutils import examples
from geoutils.raster import RasterType


class StackMergeImages:
    """
    Test cases for stacking and merging images
    Split an image with some overlap, then stack/merge it, and validate bounds and shape.
    Param `cls` is used to set the type of the output, e.g. gu.Raster (default).
    """

    def __init__(
        self, image: str, cls: Callable[[str], RasterType] = gu.Raster, different_crs: pyproj.CRS | None = None
    ) -> None:

        warnings.filterwarnings("ignore", category=UserWarning, message="For reprojection, nodata must be set.*")

        img = cls(examples.get_path(image))
        self.img = img

        # Find the easting midpoint of the img
        x_midpoint = np.mean([img.bounds.right, img.bounds.left])
        x_midpoint -= (x_midpoint - img.bounds.left) % img.res[0]

        # Cut the img into two imgs that slightly overlap each other.
        self.img1 = img.copy()
        self.img1.crop(
            rio.coords.BoundingBox(
                right=x_midpoint + img.res[0] * 3, left=img.bounds.left, top=img.bounds.top, bottom=img.bounds.bottom
            ),
            inplace=True,
        )
        self.img2 = img.copy()
        self.img2.crop(
            rio.coords.BoundingBox(
                left=x_midpoint - img.res[0] * 3, right=img.bounds.right, top=img.bounds.top, bottom=img.bounds.bottom
            ),
            inplace=True,
        )
        if different_crs:
            self.img2 = self.img2.reproject(crs=different_crs)

        # To check that use_ref_bounds work - create a img that do not cover the whole extent
        self.img3 = img.copy()
        self.img3.crop(
            rio.coords.BoundingBox(
                left=x_midpoint - img.res[0] * 3,
                right=img.bounds.right - img.res[0] * 2,
                top=img.bounds.top,
                bottom=img.bounds.bottom,
            ),
            inplace=True,
        )


@pytest.fixture
def images_1d():  # type: ignore
    return StackMergeImages("everest_landsat_b4")


@pytest.fixture
def images_different_crs():  # type: ignore
    return StackMergeImages("everest_landsat_b4", different_crs=4326)


@pytest.fixture
def sat_images():  # type: ignore
    return StackMergeImages("everest_landsat_b4", cls=gu.SatelliteImage)


@pytest.fixture
def images_3d():  # type: ignore
    return StackMergeImages("everest_landsat_rgb")


class TestMultiRaster:
    @pytest.mark.parametrize(
        "rasters",
        [
            pytest.lazy_fixture("images_1d"),
            pytest.lazy_fixture("sat_images"),
            pytest.lazy_fixture("images_different_crs"),
            pytest.lazy_fixture("images_3d"),
        ],
    )  # type: ignore
    def test_stack_rasters(self, rasters) -> None:  # type: ignore
        """Test stack_rasters"""

        # Silence the reprojection warning for default nodata value
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="New nodata value cells already exist in the data array.*"
        )
        warnings.filterwarnings("ignore", category=UserWarning, message="For reprojection, nodata must be set.*")

        # Merge the two overlapping DEMs and check that output bounds and shape is correct
        if rasters.img1.count > 1:
            # Check warning is raised once
            with pytest.warns(
                expected_warning=UserWarning,
                match="Some input Rasters have multiple bands, only their first band will be used.",
            ):
                stacked_img = gu.raster.stack_rasters([rasters.img1, rasters.img2])
            # Then ignore the other ones
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Some input Rasters have multiple bands, only their first band will be used.",
            )

        else:
            stacked_img = gu.raster.stack_rasters([rasters.img1, rasters.img2])

        assert stacked_img.count == 2
        # If the rasters were in a different projection, the final shape can vary by 1 pixel
        if not all(rast.crs == rasters.img.crs for rast in [rasters.img1, rasters.img2]):
            assert rasters.img.height == pytest.approx(stacked_img.height, abs=1)
            assert rasters.img.width == pytest.approx(stacked_img.width, abs=1)
        else:
            assert rasters.img.shape == stacked_img.shape
        assert type(stacked_img) == gu.Raster  # Check output object is always Raster, whatever input was given
        assert np.count_nonzero(np.isnan(stacked_img.data)) == 0  # Check no NaNs introduced

        merged_bounds = gu.projtools.merge_bounds(
            [rasters.img1.bounds, rasters.img2.get_bounds_projected(rasters.img1.crs)], resolution=rasters.img1.res[0]
        )
        assert merged_bounds == stacked_img.bounds

        # Check that reference works with input Raster
        stacked_img = gu.raster.stack_rasters([rasters.img1, rasters.img2], reference=rasters.img, use_ref_bounds=True)
        assert rasters.img.bounds == stacked_img.bounds

        # Others than int or gu.Raster should raise a ValueError
        try:
            stacked_img = gu.raster.stack_rasters([rasters.img1, rasters.img2], reference="a string")
        except ValueError as exception:
            if "reference should be" not in str(exception):
                raise exception

        # Check that use_ref_bounds works - use a img that do not cover the whole extent

        # This case should not preserve original extent
        stacked_img = gu.raster.stack_rasters([rasters.img1, rasters.img3])
        assert stacked_img.bounds != rasters.img.bounds

        # This case should preserve original extent
        stacked_img2 = gu.raster.stack_rasters([rasters.img1, rasters.img3], reference=rasters.img, use_ref_bounds=True)
        assert stacked_img2.bounds == rasters.img.bounds

    @pytest.mark.parametrize(
        "rasters",
        [
            pytest.lazy_fixture("images_1d"),
            pytest.lazy_fixture("images_3d"),
            pytest.lazy_fixture("images_different_crs"),
        ],
    )  # type: ignore
    def test_merge_rasters(self, rasters) -> None:  # type: ignore
        """Test merge_rasters"""
        # Merge the two overlapping DEMs and check that it closely resembles the initial DEM

        # Silence the reprojection warning for default nodata value
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="New nodata value cells already exist in the data array.*"
        )
        warnings.filterwarnings("ignore", category=UserWarning, message="For reprojection, nodata must be set.*")

        # Ignore warning already checked in test_stack_rasters
        if rasters.img1.count > 1:
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Some input Rasters have multiple bands, only their first band will be used.",
            )

        merged_img = gu.raster.merge_rasters([rasters.img1, rasters.img2], merge_algorithm=np.nanmean)

        if not all(rast.crs == rasters.img.crs for rast in [rasters.img1, rasters.img2]):
            assert rasters.img.height == pytest.approx(merged_img.height, abs=1)
            assert rasters.img.width == pytest.approx(merged_img.width, abs=1)
        else:
            assert rasters.img.shape == merged_img.shape

        merged_bounds = gu.projtools.merge_bounds(
            [rasters.img1.bounds, rasters.img2.get_bounds_projected(rasters.img1.crs)], resolution=rasters.img1.res[0]
        )
        assert merged_bounds == merged_img.bounds

        assert np.count_nonzero(np.isnan(merged_img.data)) == 0  # Check no NaNs introduced

        # Check that only works if CRS were the same
        if all(rast.crs == rasters.img.crs for rast in [rasters.img1, rasters.img2]):
            diff = rasters.img.data - merged_img.data
            assert np.abs(np.nanmean(diff)) < 1

        # Check that reference works
        merged_img2 = gu.raster.merge_rasters([rasters.img1, rasters.img2], reference=rasters.img, use_ref_bounds=True)
        # Check that only works if CRS were the same
        if all(rast.crs == rasters.img.crs for rast in [rasters.img1, rasters.img2]):
            assert merged_img2 == merged_img

    # Group rasters for for testing `load_multiple_rasters`
    # two overlapping, single band rasters
    # two overlapping, 1 and 3 band rasters
    # three overlapping rasters
    # TODO: add a case with different CRS - issue raised #310
    raster_groups = [
        [gu.examples.get_path("everest_landsat_b4"), gu.examples.get_path("everest_landsat_b4_cropped")],
        [gu.examples.get_path("everest_landsat_rgb"), gu.examples.get_path("everest_landsat_b4_cropped")],
        [
            gu.examples.get_path("everest_landsat_b4"),
            gu.examples.get_path("everest_landsat_rgb"),
            gu.examples.get_path("everest_landsat_b4_cropped"),
        ],
    ]

    @pytest.mark.parametrize("raster_paths", raster_groups)  # type: ignore
    def test_load_multiple_overlap(self, raster_paths: list[str]) -> None:
        """
        Test load_multiple_rasters functionalities, when rasters overlap -> no warning is raised
        """
        # - Test that with crop=False and ref_grid=None, rasters are simply loaded - #
        output_rst: list[gu.Raster] = gu.raster.load_multiple_rasters(raster_paths, crop=False, ref_grid=None)
        for k, rst in enumerate(output_rst):
            assert rst.is_loaded
            rst2 = gu.Raster(raster_paths[k])
            assert rst.raster_equal(rst2)

        # - Test that with crop=True and ref_grid=None, rasters are cropped only in area of overlap - #
        output_rst = gu.raster.load_multiple_rasters(raster_paths, crop=True, ref_grid=None)
        ref_crs = gu.Raster(raster_paths[0], load_data=False).crs

        # Save original and new bounds (as polygons) in the reference CRS
        orig_poly_bounds = []
        new_poly_bounds = []
        for k, rst in enumerate(output_rst):
            assert rst.is_loaded
            rst2 = gu.Raster(raster_paths[k], load_data=False)
            assert rst.crs == rst2.crs  # CRS should not have been changed
            orig_poly_bounds.append(gu.projtools.bounds2poly(rst2.bounds, rst.crs, ref_crs))
            new_poly_bounds.append(gu.projtools.bounds2poly(rst.bounds, rst.crs, ref_crs))

        # Check that the new bounds are contained in the original bounds of all rasters
        for poly1 in new_poly_bounds:
            for poly2 in orig_poly_bounds:
                assert poly2.contains(poly1)

        # - Test that with crop=False and ref_grid=0, rasters all have the same grid as the reference - #
        # For the landsat test case, a warning will be raised because nodata is None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            output_rst = gu.raster.load_multiple_rasters(raster_paths, crop=False, ref_grid=0)

        ref_rst = gu.Raster(raster_paths[0], load_data=False)
        for k, rst in enumerate(output_rst):
            rst2 = gu.Raster(raster_paths[k], load_data=False)
            assert rst.is_loaded
            # Georeferences are forced to ref
            assert rst.crs == ref_rst.crs
            assert rst.shape == ref_rst.shape
            assert rst.transform == ref_rst.transform
            # Original number of bounds and nodata must be preserved
            assert rst.count == rst2.count
            if rst2.nodata is not None:  # if none, reprojection with set a default value
                assert rst.nodata == rst2.nodata

    raster_groups = [
        [gu.examples.get_path("everest_landsat_b4"), gu.examples.get_path("exploradores_aster_dem")],
    ]

    @pytest.mark.parametrize("raster_paths", raster_groups)  # type: ignore
    def test_load_multiple_no_overlap(self, raster_paths: list[str]) -> None:
        """
        Test load_multiple_rasters functionalities with rasters that do not overlap -> raises warning in certain cases
        """
        # - With crop=False and ref_grid=None, rasters are simply loaded - #
        output_rst: list[gu.Raster] = gu.raster.load_multiple_rasters(raster_paths, crop=False, ref_grid=None)
        for k, rst in enumerate(output_rst):
            assert rst.is_loaded
            rst2 = gu.Raster(raster_paths[k])
            assert rst.raster_equal(rst2)

        # - With crop=True -> should raise a warning - #
        with pytest.warns(UserWarning, match="Intersection is void, returning unloaded rasters."):
            output_rst = gu.raster.load_multiple_rasters(raster_paths, crop=True, ref_grid=None)

        # - Should work with crop=False and ref_grid=0 - #
        # For the landsat test case, a warning will be raised because nodata is None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            output_rst = gu.raster.load_multiple_rasters(raster_paths, crop=False, ref_grid=0)

        ref_rst = gu.Raster(raster_paths[0], load_data=False)
        for k, rst in enumerate(output_rst):
            rst2 = gu.Raster(raster_paths[k], load_data=False)
            assert rst.is_loaded
            # Georeferences are forced to ref
            assert rst.crs == ref_rst.crs
            assert rst.shape == ref_rst.shape
            assert rst.transform == ref_rst.transform
            # Original number of bounds and nodata must be preserved
            assert rst.count == rst2.count
            if rst2.nodata is not None:  # if none, reprojection with set a default value
                assert rst.nodata == rst2.nodata
            # Logically, the non overlapping raster should have only masked values
            if k != 0:
                assert np.count_nonzero(~rst.data.mask) == 0
