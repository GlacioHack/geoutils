"""Array tools related to rasters."""

from __future__ import annotations

import warnings

import numpy as np

import geoutils as gu
from geoutils._typing import MArrayNum, NDArrayNum


def get_mask(array: NDArrayNum | MArrayNum) -> NDArrayNum:
    """
    Return the mask of invalid values, whether array is a ndarray with NaNs or a np.ma.masked_array.

    :param array: Input array.

    :returns invalid_mask: boolean array, True where array is masked or Nan.
    """
    mask = (array.mask | ~np.isfinite(array.data)) if isinstance(array, np.ma.masked_array) else ~np.isfinite(array)
    return mask.squeeze()


def get_array_and_mask(
    array: NDArrayNum | MArrayNum, check_shape: bool = True, copy: bool = True
) -> tuple[NDArrayNum, NDArrayNum]:
    """
    Return array with masked values set to NaN and the associated mask.
    Works whether array is a ndarray with NaNs or a np.ma.masked_array.

    :param array: Input array.
    :param check_shape: Validate that the array is either a 1D array, a 2D array or a 3D array of shape (1, rows, cols).
    :param copy: Return a copy of 'array'. If False, a view will be attempted (and warn if not possible)

    :returns array_data, invalid_mask: a tuple of ndarrays. First is array with invalid pixels converted to NaN, \
    second is mask of invalid pixels (True if invalid).
    """
    #
    if isinstance(array, gu.Raster):
        array = array.data

    if check_shape:
        if len(array.shape) > 2 and array.shape[0] > 1:
            raise ValueError(
                f"Invalid array shape given: {array.shape}." "Expected 2D array or 3D array where arr.shape[0] == 1"
            )

    # If an occupied mask exists and a view was requested, trigger a warning.
    if not copy and np.any(getattr(array, "mask", False)):
        warnings.warn("Copying is required to respect the mask. Returning copy. Set 'copy=True' to hide this message.")
        copy = True

    # If array is of type integer and has a mask, it needs to be converted to float (to assign nans)
    if np.any(getattr(array, "mask", False)) and np.issubdtype(array.dtype, np.integer):  # type: ignore
        array = array.astype(np.float32)  # type: ignore

    # Convert into a regular ndarray (a view or copy depending on the 'copy' argument)
    array_data = np.array(array).squeeze() if copy else np.asarray(array).squeeze()

    # Get the mask of invalid pixels and set nans if it is occupied.
    invalid_mask = get_mask(array)
    if np.any(invalid_mask):
        array_data[invalid_mask] = np.nan

    return array_data, invalid_mask


def get_valid_extent(array: NDArrayNum | MArrayNum) -> tuple[int, ...]:
    """
    Return (rowmin, rowmax, colmin, colmax), the first/last row/column of array with valid pixels
    """
    if not array.dtype == "bool":
        valid_mask = ~get_mask(array)
    else:
        valid_mask = array
    cols_nonzero = np.where(np.count_nonzero(valid_mask, axis=0) > 0)[0]
    rows_nonzero = np.where(np.count_nonzero(valid_mask, axis=1) > 0)[0]
    return rows_nonzero[0], rows_nonzero[-1], cols_nonzero[0], cols_nonzero[-1]


def get_xy_rotated(raster: gu.Raster, along_track_angle: float) -> tuple[NDArrayNum, NDArrayNum]:
    """
    Rotate x, y axes of image to get along- and cross-track distances.
    :param raster: Raster to get x,y positions from.
    :param along_track_angle: Angle by which to rotate axes (degrees)

    :returns xxr, yyr: Arrays corresponding to along (x) and cross (y) track distances.
    """

    myang = np.deg2rad(along_track_angle)

    # Get grid coordinates
    xx, yy = raster.coords(grid=True)
    xx -= np.min(xx)
    yy -= np.min(yy)

    # Get rotated coordinates

    # For along-track
    xxr = xx * np.cos(myang) - yy * np.sin(myang)
    # For cross-track
    yyr = xx * np.sin(myang) + yy * np.cos(myang)

    # Re-initialize coordinate at zero
    xxr -= np.nanmin(xxr)
    yyr -= np.nanmin(yyr)

    return xxr, yyr
