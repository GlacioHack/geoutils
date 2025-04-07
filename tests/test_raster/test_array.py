"""Test array tools."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import rasterio as rio

import geoutils as gu
from geoutils.raster.array import get_array_and_mask, get_valid_extent, get_xy_rotated


class TestArray:
    @pytest.mark.parametrize("dtype", ["uint8", "uint16", "int32", "float32", "float16"])  # type: ignore
    @pytest.mark.parametrize(
        "mask_and_viewable",
        [
            (None, True),  # A ndarray with no mask should support views
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
        self,
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
                get_array_and_mask(array, check_shape=True)

            # Stop the test here as the failure is now validated.
            return

        # Get a copy of the array and check its shape (it should always pass at this point)
        arr, _ = get_array_and_mask(array, copy=True, check_shape=True)

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
            arr_view, mask = get_array_and_mask(array, copy=False)

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

    def test_get_valid_extent(self) -> None:
        """Check the function to get valid extent."""

        # Create an artificial array and masked array
        arr = np.ones(shape=(5, 5))
        arr_mask = np.zeros(shape=(5, 5), dtype=bool)
        mask_ma = np.ma.masked_array(data=arr, mask=arr_mask)

        # For no invalid values, the function should return the edges
        # For the array
        assert (0, 4, 0, 4) == get_valid_extent(arr)
        # For the masked-array
        assert (0, 4, 0, 4) == get_valid_extent(mask_ma)

        # 1/ First column:
        # If we mask it in the masked array
        mask_ma[0, :] = np.ma.masked
        assert (1, 4, 0, 4) == get_valid_extent(mask_ma)

        # If we changed the array to NaNs
        arr[0, :] = np.nan
        assert (1, 4, 0, 4) == get_valid_extent(arr)
        mask_ma.data[0, :] = np.nan
        mask_ma.mask = False
        assert (1, 4, 0, 4) == get_valid_extent(mask_ma)

        # 2/ First row:
        arr = np.ones(shape=(5, 5))
        arr_mask = np.zeros(shape=(5, 5), dtype=bool)
        mask_ma = np.ma.masked_array(data=arr, mask=arr_mask)
        # If we mask it in the masked array
        mask_ma[:, 0] = np.ma.masked
        assert (0, 4, 1, 4) == get_valid_extent(mask_ma)

        # If we changed the array to NaNs
        arr[:, 0] = np.nan
        assert (0, 4, 1, 4) == get_valid_extent(arr)
        mask_ma.data[:, 0] = np.nan
        mask_ma.mask = False
        assert (0, 4, 1, 4) == get_valid_extent(mask_ma)

        # 3/ Last column:
        arr = np.ones(shape=(5, 5))
        arr_mask = np.zeros(shape=(5, 5), dtype=bool)
        mask_ma = np.ma.masked_array(data=arr, mask=arr_mask)

        # If we mask it in the masked array
        mask_ma[-1, :] = np.ma.masked
        assert (0, 3, 0, 4) == get_valid_extent(mask_ma)

        # If we changed the array to NaNs
        arr[-1, :] = np.nan
        assert (0, 3, 0, 4) == get_valid_extent(arr)
        mask_ma.data[-1, :] = np.nan
        mask_ma.mask = False
        assert (0, 3, 0, 4) == get_valid_extent(mask_ma)

        # 4/ Last row:
        arr = np.ones(shape=(5, 5))
        arr_mask = np.zeros(shape=(5, 5), dtype=bool)
        mask_ma = np.ma.masked_array(data=arr, mask=arr_mask)

        # If we mask it in the masked array
        mask_ma[:, -1] = np.ma.masked
        assert (0, 4, 0, 3) == get_valid_extent(mask_ma)

        # If we changed the array to NaNs
        arr[:, -1] = np.nan
        assert (0, 4, 0, 3) == get_valid_extent(arr)
        mask_ma.data[:, -1] = np.nan
        mask_ma.mask = False
        assert (0, 4, 0, 3) == get_valid_extent(mask_ma)

    def test_get_xy_rotated(self) -> None:
        """Check the function to rotate array."""

        # Create an artificial raster
        rng = np.random.default_rng(42)
        width = height = 5
        transform = rio.transform.from_bounds(0, 0, 1, 1, width, height)
        r1 = gu.Raster.from_array(rng.integers(1, 255, (height, width), dtype="uint8"), transform=transform, crs=None)

        # First, we get initial coords
        xx, yy = r1.coords(grid=True, force_offset="ll")

        # Rotating the coordinates 90 degrees should be the same as rotating the array
        xx90, yy90 = get_xy_rotated(r1, along_track_angle=90)
        assert np.allclose(np.rot90(xx90), xx)
        assert np.allclose(np.rot90(yy90), yy)

        # Same for 180 degrees
        xx180, yy180 = get_xy_rotated(r1, along_track_angle=180)
        assert np.allclose(np.rot90(xx180, k=2), xx)
        assert np.allclose(np.rot90(yy180, k=2), yy)

        # Same for 270 degrees
        xx270, yy270 = get_xy_rotated(r1, along_track_angle=270)
        assert np.allclose(np.rot90(xx270, k=3), xx)
        assert np.allclose(np.rot90(yy270, k=3), yy)

        # 360 degrees should get us back on our feet
        xx360, yy360 = get_xy_rotated(r1, along_track_angle=360)
        assert np.allclose(xx360, xx)
        assert np.allclose(yy360, yy)

        # Test that the values make sense for 45 degrees
        xx45, yy45 = get_xy_rotated(r1, along_track_angle=45)
        # Should have zero on the upper left corner for xx
        assert xx45[0, 0] == pytest.approx(0)
        # Then a multiple of sqrt2 along each dimension
        assert xx45[1, 0] == pytest.approx(xx45[0, 1]) == pytest.approx(0.1 * np.sqrt(2))
        # The lower right corner should have the highest coordinate (0.8) times sqrt(2)
        assert xx45[-1, -1] == pytest.approx(np.max(xx) * np.sqrt(2))

        # Finally, yy should be rotated by 90
        assert np.allclose(np.rot90(xx45), yy45)

        xx, yy = get_xy_rotated(r1, along_track_angle=90)
