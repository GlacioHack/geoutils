"""
Functions to test the spatial tools.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pytest
import rasterio as rio

import geoutils as gu
from geoutils import examples
from geoutils.georaster import RasterType

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
def test_load_multiple_overlap(raster_paths: list[str]) -> None:
    """
    Test load_multiple_rasters functionalities, when rasters overlap -> no warning is raised
    """
    # - Test that with crop=False and ref_grid=None, rasters are simply loaded - #
    output_rst: list[gu.Raster] = gu.spatial_tools.load_multiple_rasters(raster_paths, crop=False, ref_grid=None)
    for k, rst in enumerate(output_rst):
        assert rst.is_loaded
        rst2 = gu.Raster(raster_paths[k])
        assert rst == rst2

    # - Test that with crop=True and ref_grid=None, rasters are cropped only in area of overlap - #
    output_rst = gu.spatial_tools.load_multiple_rasters(raster_paths, crop=True, ref_grid=None)
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
        output_rst = gu.spatial_tools.load_multiple_rasters(raster_paths, crop=False, ref_grid=0)

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
def test_load_multiple_no_overlap(raster_paths: list[str]) -> None:
    """
    Test load_multiple_rasters functionalities with rasters that do not overlap -> raises warning in certain cases
    """
    # - With crop=False and ref_grid=None, rasters are simply loaded - #
    output_rst: list[gu.Raster] = gu.spatial_tools.load_multiple_rasters(raster_paths, crop=False, ref_grid=None)
    for k, rst in enumerate(output_rst):
        assert rst.is_loaded
        rst2 = gu.Raster(raster_paths[k])
        assert rst == rst2

    # - With crop=True -> should raise a warning - #
    with pytest.warns(UserWarning, match="Intersection is void, returning unloaded rasters."):
        output_rst = gu.spatial_tools.load_multiple_rasters(raster_paths, crop=True, ref_grid=None)

    # - Should work with crop=False and ref_grid=0 - #
    # For the landsat test case, a warning will be raised because nodata is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        output_rst = gu.spatial_tools.load_multiple_rasters(raster_paths, crop=False, ref_grid=0)

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


class stack_merge_images:
    """
    Test cases for stacking and merging images
    Split an image with some overlap, then stack/merge it, and validate bounds and shape.
    Param `cls` is used to set the type of the output, e.g. gu.Raster (default).
    """

    def __init__(self, image: str, cls: Callable[[str], RasterType] = gu.Raster) -> None:
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
            )
        )
        self.img2 = img.copy()
        self.img2.crop(
            rio.coords.BoundingBox(
                left=x_midpoint - img.res[0] * 3, right=img.bounds.right, top=img.bounds.top, bottom=img.bounds.bottom
            )
        )

        # To check that use_ref_bounds work - create a img that do not cover the whole extent
        self.img3 = img.copy()
        self.img3.crop(
            rio.coords.BoundingBox(
                left=x_midpoint - img.res[0] * 3,
                right=img.bounds.right - img.res[0] * 2,
                top=img.bounds.top,
                bottom=img.bounds.bottom,
            )
        )


@pytest.fixture
def images_1d():  # type: ignore
    return stack_merge_images("everest_landsat_b4")


@pytest.fixture
def sat_images():  # type: ignore
    return stack_merge_images("everest_landsat_b4", cls=gu.SatelliteImage)


@pytest.fixture
def images_3d():  # type: ignore
    return stack_merge_images("everest_landsat_rgb")


@pytest.mark.parametrize(
    "rasters", [pytest.lazy_fixture("images_1d"), pytest.lazy_fixture("sat_images"), pytest.lazy_fixture("images_3d")]
)  # type: ignore
def test_stack_rasters(rasters) -> None:  # type: ignore
    """Test stack_rasters"""

    # Silence the reprojection warning for default nodata value
    warnings.filterwarnings("ignore", category=UserWarning, message="New nodata value found in the data array.*")
    warnings.filterwarnings("ignore", category=UserWarning, message="For reprojection, dst_nodata must be set.*")

    # Merge the two overlapping DEMs and check that output bounds and shape is correct
    if rasters.img1.count > 1:
        # Check warning is raised once
        with pytest.warns(
            expected_warning=UserWarning,
            match="Some input Rasters have multiple bands, only their first band will be used.",
        ):
            stacked_img = gu.spatial_tools.stack_rasters([rasters.img1, rasters.img2])
        # Then ignore the other ones
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Some input Rasters have multiple bands, only their first band will be used.",
        )

    else:
        stacked_img = gu.spatial_tools.stack_rasters([rasters.img1, rasters.img2])

    assert stacked_img.count == 2
    assert rasters.img.shape == stacked_img.shape
    assert type(stacked_img) == gu.Raster  # Check output object is always Raster, whatever input was given
    assert np.count_nonzero(np.isnan(stacked_img.data)) == 0  # Check no NaNs introduced

    merged_bounds = gu.spatial_tools.merge_bounding_boxes(
        [rasters.img1.bounds, rasters.img2.bounds], resolution=rasters.img1.res[0]
    )
    assert merged_bounds == stacked_img.bounds

    # Check that reference works with input Raster
    stacked_img = gu.spatial_tools.stack_rasters([rasters.img1, rasters.img2], reference=rasters.img)
    assert rasters.img.bounds == stacked_img.bounds

    # Others than int or gu.Raster should raise a ValueError
    try:
        stacked_img = gu.spatial_tools.stack_rasters([rasters.img1, rasters.img2], reference="a string")
    except ValueError as exception:
        if "reference should be" not in str(exception):
            raise exception

    # Check that use_ref_bounds works - use a img that do not cover the whole extent

    # This case should not preserve original extent
    stacked_img = gu.spatial_tools.stack_rasters([rasters.img1, rasters.img3])
    assert stacked_img.bounds != rasters.img.bounds

    # This case should preserve original extent
    stacked_img2 = gu.spatial_tools.stack_rasters(
        [rasters.img1, rasters.img3], reference=rasters.img, use_ref_bounds=True
    )
    assert stacked_img2.bounds == rasters.img.bounds


@pytest.mark.parametrize(
    "rasters", [pytest.lazy_fixture("images_1d"), pytest.lazy_fixture("images_3d")]
)  # type: ignore
def test_merge_rasters(rasters) -> None:  # type: ignore
    """Test merge_rasters"""
    # Merge the two overlapping DEMs and check that it closely resembles the initial DEM

    # Silence the reprojection warning for default nodata value
    warnings.filterwarnings("ignore", category=UserWarning, message="New nodata value found in the data array.*")
    warnings.filterwarnings("ignore", category=UserWarning, message="For reprojection, dst_nodata must be set.*")

    # Ignore warning already checked in test_stack_rasters
    if rasters.img1.count > 1:
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Some input Rasters have multiple bands, only their first band will be used.",
        )

    merged_img = gu.spatial_tools.merge_rasters([rasters.img1, rasters.img2], merge_algorithm=np.nanmean)

    assert rasters.img.shape == merged_img.shape
    assert rasters.img.bounds == merged_img.bounds
    assert np.count_nonzero(np.isnan(merged_img.data)) == 0  # Check no NaNs introduced

    diff = rasters.img.data - merged_img.data

    assert np.abs(np.nanmean(diff)) < 1

    # Check that reference works
    merged_img2 = gu.spatial_tools.merge_rasters([rasters.img1, rasters.img2], reference=rasters.img)
    assert merged_img2 == merged_img


def test_subdivide_array() -> None:

    test_shape = (6, 4)
    test_count = 4
    subdivision_grid = gu.spatial_tools.subdivide_array(test_shape, test_count)

    assert subdivision_grid.shape == test_shape
    assert np.unique(subdivision_grid).size == test_count

    assert np.unique(gu.spatial_tools.subdivide_array((3, 3), 3)).size == 3

    with pytest.raises(ValueError, match=r"Expected a 2D shape, got 1D shape.*"):
        gu.spatial_tools.subdivide_array((5,), 2)

    with pytest.raises(ValueError, match=r"Shape.*smaller than.*"):
        gu.spatial_tools.subdivide_array((5, 2), 15)


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int32", "float32", "float16"])  # type: ignore
@pytest.mark.parametrize(
    "mask_and_viewable",
    [
        (None, True),  # An ndarray with no mask should support views
        (False, True),  # A masked array with an empty mask should support views
        ([True, False, False, False], False),  # A masked array with an occupied mask should not support views.
        ([False, False, False, False], True),  # A masked array with an empty occupied mask should support views.
    ],
)  # type: ignore
@pytest.mark.parametrize(
    "shape_and_check_passes",
    [
        ((1, 2, 2), True),  # A 3D array with a shape[0] == 1 is okay.
        ((2, 1, 2), False),  # A 3D array with a shape[0] != 1 is not okay.
        ((2, 2), True),  # A 2D array is okay.
        ((4,), True),  # A 1D array is okay.
    ],
)  # type: ignore
def test_get_array_and_mask(
    dtype: str,
    mask_and_viewable: tuple[None | bool | list[bool], bool],
    shape_and_check_passes: tuple[tuple[int, ...], bool],
) -> None:
    """Validate that the function returns views when expected, and copies otherwise."""
    warnings.simplefilter("error")

    masked_values, view_should_be_possible = mask_and_viewable
    shape, check_should_pass = shape_and_check_passes

    # Create an array of the specified dtype
    array = np.ones(shape, dtype=dtype)
    if masked_values is not None:
        if masked_values is False:
            array = np.ma.masked_array(array)
        else:
            array = np.ma.masked_array(array, mask=np.reshape(masked_values, array.shape))

    # Validate that incorrect shapes raise the correct error.
    if not check_should_pass:
        with pytest.raises(ValueError, match="Invalid array shape given"):
            gu.spatial_tools.get_array_and_mask(array, check_shape=True)

        # Stop the test here as the failure is now validated.
        return

    # Get a copy of the array and check its shape (it should always pass at this point)
    arr, _ = gu.spatial_tools.get_array_and_mask(array, copy=True, check_shape=True)

    # Validate that the array is a copy
    assert not np.shares_memory(arr, array)

    # If it was an integer dtype and it had a mask, validate that the array is now "float32"
    if np.issubdtype(dtype, np.integer) and np.any(masked_values or False):
        assert arr.dtype == "float32"

    # If there was no mask or the mask was empty, validate that arr and array are equivalent
    if not np.any(masked_values or False):
        assert np.sum(np.abs(array - arr)) == 0.0

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        # Try to create a view.
        arr_view, mask = gu.spatial_tools.get_array_and_mask(array, copy=False)

        # If it should be possible, validate that there were no warnings.
        if view_should_be_possible:
            assert len(caught_warnings) == 0, (caught_warnings[0].message, array)
        # Otherwise, validate that one warning was raised with the correct text.
        else:
            assert len(caught_warnings) == 1
            assert "Copying is required" in str(caught_warnings[0].message)

    # Validate that the view shares memory if it was possible, or otherwise that it is a copy.
    if view_should_be_possible:
        assert np.shares_memory(array, arr_view)
    else:
        assert not np.shares_memory(array, arr_view)


class TestSubsample:
    """
    Different examples of 1D to 3D arrays with masked values for testing.
    """

    # Case 1 - 1D array, 1 masked value
    array1D = np.ma.masked_array(np.arange(10), mask=np.zeros(10))
    array1D.mask[3] = True
    assert np.ndim(array1D) == 1
    assert np.count_nonzero(array1D.mask) > 0

    # Case 2 - 2D array, 1 masked value
    array2D = np.ma.masked_array(np.arange(9).reshape((3, 3)), mask=np.zeros((3, 3)))
    array2D.mask[0, 1] = True
    assert np.ndim(array2D) == 2
    assert np.count_nonzero(array2D.mask) > 0

    # Case 3 - 3D array, 1 masked value
    array3D = np.ma.masked_array(np.arange(9).reshape((1, 3, 3)), mask=np.zeros((1, 3, 3)))
    array3D = np.ma.vstack((array3D, array3D + 10))
    array3D.mask[0, 0, 1] = True
    assert np.ndim(array3D) == 3
    assert np.count_nonzero(array3D.mask) > 0

    @pytest.mark.parametrize("array", [array1D, array2D, array3D])  # type: ignore
    def test_subsample(self, array: np.ndarray) -> None:
        """
        Test gu.spatial_tools.subsample_raster.
        """
        # Test that subsample > 1 works as expected, i.e. output 1D array, with no masked values, or selected size
        for npts in np.arange(2, np.size(array)):
            random_values = gu.spatial_tools.subsample_raster(array, subsample=npts)
            assert np.ndim(random_values) == 1
            assert np.size(random_values) == npts
            assert np.count_nonzero(random_values.mask) == 0

        # Test if subsample > number of valid values => return all
        random_values = gu.spatial_tools.subsample_raster(array, subsample=np.size(array) + 3)
        assert np.all(np.sort(random_values) == array[~array.mask])

        # Test if subsample = 1 => return all valid values
        random_values = gu.spatial_tools.subsample_raster(array, subsample=1)
        assert np.all(np.sort(random_values) == array[~array.mask])

        # Test if subsample < 1
        random_values = gu.spatial_tools.subsample_raster(array, subsample=0.5)
        assert np.size(random_values) == int(np.size(array) * 0.5)

        # Test with optional argument return_indices
        indices = gu.spatial_tools.subsample_raster(array, subsample=0.3, return_indices=True)
        assert np.ndim(indices) == 2
        assert len(indices) == np.ndim(array)
        assert np.ndim(array[indices]) == 1
        assert np.size(array[indices]) == int(np.size(array) * 0.3)
