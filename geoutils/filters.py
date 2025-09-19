# Copyright (c) 2025 GeoUtils developers
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES)
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Filters to remove outliers and reduce noise in rasters."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import scipy
from scipy.ndimage import generic_filter as scipy_generic_filter

from geoutils._typing import NDArrayNum

try:
    from numba import jit, prange

    _has_numba = True
except ImportError:
    _has_numba = False

    def jit(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Fake jit decorator if numba is not installed
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


def _filter(array: NDArrayNum, method: str | Callable[..., NDArrayNum], **kwargs: dict[str, Any]) -> NDArrayNum:
    """
    Apply a filter to the array. See description of raster.filter.
    """

    if isinstance(method, str):
        filter_map: dict[str, Callable[..., NDArrayNum]] = {
            "gaussian": gaussian_filter,
            "median": median_filter,
            "mean": mean_filter,
            "max": max_filter,
            "min": min_filter,
            "distance": distance_filter,
        }

        if method not in filter_map:
            raise ValueError(f"Unsupported filter method '{method}'. " f"Available methods: {list(filter_map.keys())}")

        filter_func = filter_map[method]

    elif callable(method):
        filter_func = method
    else:
        raise TypeError("`method` must be a string or a callable.")

    # Apply filter
    filtered_data = filter_func(array, **kwargs)
    return filtered_data


def gaussian_filter(array: NDArrayNum, sigma: float, **kwargs: Any) -> NDArrayNum:
    """
    Apply a Gaussian filter to a raster that may contain NaNs.
    N.B: kernel_size is set automatically based on sigma.

    :param array: The input array to be filtered.
    :param sigma: The sigma of the Gaussian kernel

    :returns: The filtered array (same shape as input)
    """

    if array.ndim == 1:
        raise ValueError("Gaussian filter can't be applied to 1D arrays.")

    # Boolean mask: True where NaN
    mask = np.isnan(array)
    mask = np.asarray(mask, dtype=bool)
    # Replace NaNs with 0 for convolution
    arr_filled = np.where(mask, 0, array)

    # Apply gaussian filter to values and to mask
    filtered = scipy.ndimage.gaussian_filter(arr_filled, sigma, **kwargs)

    filtered[mask] = np.nan

    return filtered


@jit(nopython=True, parallel=True)  # type: ignore
def median_filter_numba(array: NDArrayNum, window_size: int) -> NDArrayNum:
    """
    Apply a median filter to a raster that may contain NaNs, using numbas's implementation.

    :param array: The input array to be filtered.
    :param window_size: The size of the window to use (must be odd).

    :returns: The filtered array (same shape as input).
    """

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    # Get input shapes
    N1, N2 = array.shape

    # Define ranges to loop through given padding
    row_range = N1 - window_size + 1
    col_range = N2 - window_size + 1

    # Allocate an output array
    outputs = np.full((row_range, col_range), fill_value=np.nan, dtype=np.float32)

    # Loop over every pixel concurrently by using prange
    for row in prange(row_range):
        for col in prange(col_range):

            outputs[row, col] = np.nanmedian(array[row : row + window_size, col : col + window_size])

    return outputs


def median_filter(array: NDArrayNum, window_size: int, engine: Literal["scipy", "numba"] = "numba") -> NDArrayNum:
    """
    Apply a median filter to a raster that may contain NaNs, using scipy's implementation.

    :param array: The input array to be filtered.
    :param window_size: The size of the window to use (must be odd).
    :param engine: Filtering engine to use, either "scipy" or "numba".

    :returns: The filtered array (same shape as input).
    """

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    if array.ndim == 2:
        return _apply_median_filter_2d(array, window_size, engine)
    elif array.ndim == 3:
        return np.stack([_apply_median_filter_2d(slice_, window_size, engine) for slice_ in array])
    else:
        raise ValueError("Input array must be 2D or 3D.")


def _apply_median_filter_2d(array: NDArrayNum, window_size: int, engine: Literal["scipy", "numba"]) -> NDArrayNum:
    """
    Apply a 2D median filter on an array that may contain NaNs.

    :param array: 2D input array to filter, may contain NaNs.
    :param window_size: Size of the median filter window (must be odd).
    :param engine: Filtering engine to use, either "scipy" or "numba".
    :returns: Filtered array of the same shape as input.
    """

    if engine == "scipy":
        return scipy_generic_filter(array, np.nanmedian, size=window_size, mode="constant", cval=np.nan)
    else:
        if not _has_numba:
            raise ValueError("Optional dependency needed. Install 'numba'.")
        hw = int((window_size - 1) / 2)
        array = np.pad(array, pad_width=((hw, hw), (hw, hw)), constant_values=np.nan)
        return median_filter_numba(array, window_size)


def mean_filter(array: NDArrayNum, kernel_size: int = 5, **kwargs: Any) -> NDArrayNum:
    """
    Apply a mean filter to a raster that may contain NaNs.

    :param array: The input array to be filtered.
    :param kernel_size: The size of the kernel.

    :returns: The filtered array (same shape as input).
    """

    if np.ndim(array) not in [2]:
        raise ValueError(f"Invalid array shape given: {array.shape}. Expected 2D array.")
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
    return generic_filter(array, scipy.ndimage.convolve, **{"weights": kernel, **kwargs})  # type: ignore


def min_filter(array: NDArrayNum, size: int = 5, **kwargs: Any) -> NDArrayNum:
    """
    Apply a minimum filter to a raster that may contain NaNs, using scipy's implementation.

    :param array: The input array to be filtered.
    :param size:  the shape that is taken from the input array, at every element position,
    to define the input to the filter function

    :returns: The filtered array (same shape as input).
    """
    # Check that array dimension is 2 or 3
    if np.ndim(array) not in [2, 3]:
        raise ValueError(f"Invalid array shape given: {array.shape}. Expected 2D or 3D array.")

    nans = np.isnan(array)
    # We replace temporarily NaNs by infinite values during filtering to avoid spreading NaNs
    array_nans_replaced = np.where(nans, np.inf, array)
    array_nans_replaced_f = scipy.ndimage.minimum_filter(array_nans_replaced, size=size, **kwargs)
    # In the end, we want the filtered array without infinite values, so we put back NaNs
    return np.where(nans, array, array_nans_replaced_f)


def max_filter(array: NDArrayNum, size: int = 5, **kwargs: Any) -> NDArrayNum:
    """
    Apply a maximum filter to a raster that may contain NaNs, using scipy's implementation.

    :param array: the input array to be filtered.
    :param size:  the shape that is taken from the input array, at every element position,
    to define the input to the filter function

    :returns: the filtered array (same shape as input).
    """
    # Check that array dimension is 2 or 3
    if np.ndim(array) not in [2, 3]:
        raise ValueError(f"Invalid array shape given: {array.shape}. Expected 2D or 3D array.")

    nans = np.isnan(array)
    # We replace temporarily NaNs by negative infinite values during filtering to avoid spreading NaNs
    array_nans_replaced = np.where(nans, -np.inf, array)
    array_nans_replaced_f = scipy.ndimage.maximum_filter(array_nans_replaced, size=size, **kwargs)
    # In the end we want the filtered array without negative infinite values, so we put back NaNs
    return np.where(nans, array, array_nans_replaced_f)


def distance_filter(array: NDArrayNum, radius: float, outlier_threshold: float) -> NDArrayNum:
    """
    Filter out pixels whose value is distantly more than a set threshold from the average value of all neighbor \
pixels within a given radius.
    Filtered pixels are set to NaN.

    TO DO: Add an option on how the "average" value should be calculated, i.e. using a Gaussian, median etc filter.

    :param array: the input array to be filtered.
    :param radius: the radius in which the average value is calculated (for Gaussian filter, this is sigma).
    :param outlier_threshold: the minimum difference abs(array - mean) for a pixel to be considered an outlier.

    :returns: the filtered array (same shape as input)
    """
    # Create mask of valid (finite) values
    valid_mask = np.isfinite(array)

    # Smooth both the data and the valid mask
    smoothed = gaussian_filter(np.nan_to_num(array, nan=0.0), sigma=radius)
    normalization = gaussian_filter(valid_mask.astype(float), sigma=radius)

    # Avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        local_mean = smoothed / normalization

    # Compute the outliers
    diff = np.abs(array - local_mean)
    outliers = (diff > outlier_threshold) & valid_mask

    # Create output with outliers set to NaN
    out_array = array.copy()
    out_array[outliers] = np.nan

    return out_array


def generic_filter(
    array: NDArrayNum,
    filter_function: Callable[..., NDArrayNum],
    **kwargs: dict[Any, Any] | int,
) -> NDArrayNum:
    """
    Apply a filter from a function.

    :param array: the input array to be filtered.
    :param filter_function: the function of the filter.

    :returns: the filtered array (same shape as input).
    """
    # Check that array dimension is 2 or 3
    if np.ndim(array) not in [2, 3]:
        raise ValueError(f"Invalid array shape given: {array.shape}. Expected 2D or 3D array.")

    return filter_function(array, **kwargs)
