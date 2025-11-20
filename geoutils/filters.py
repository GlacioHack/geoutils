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

"""
Filters class to remove outliers and reduce noise in rasters.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import scipy
from packaging.version import Version
from scipy.ndimage import uniform_filter

from geoutils._typing import NDArrayNum

if Version(scipy.__version__) > Version("1.16.0"):
    generic_filter_scipy = scipy.ndimage.vectorized_filter
    _has_vectorized_filter = True
else:
    generic_filter_scipy = scipy.ndimage.generic_filter
    _has_vectorized_filter = False

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


def _filter(array: NDArrayNum, method: str | Callable[..., NDArrayNum], size: int = 3, **kwargs: Any) -> NDArrayNum:
    """
    Dispatch filter application by method name or custom callable.
    :param array: array to filter
    :param method: filter method name or callable
    """
    filter_map: dict[str, Callable[..., NDArrayNum]] = {
        "gaussian": gaussian_filter,
        "median": lambda arr, size=size, **_: generic_filter_scipy(arr, np.nanmedian, size=size),
        "mean": lambda arr, size=size, **_: generic_filter_scipy(arr, np.nanmean, size=size),
        "max": lambda arr, size=size, **_: generic_filter_scipy(arr, np.nanmax, size=size),
        "min": lambda arr, size=size, **_: generic_filter_scipy(arr, np.nanmin, size=size),
        "distance": distance_filter,
    }

    if isinstance(method, str):
        if method not in filter_map:
            raise ValueError(f"Unsupported filter method '{method}'. Available: {list(filter_map)}")
        func = filter_map[method]
    elif callable(method):
        func = method
    else:
        raise TypeError("`method` must be a string or a callable.")

    return func(array, **kwargs)


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
    mask_f = (~mask).astype(float)

    # Replace NaNs with 0
    arr_filled = np.where(mask, 0, array)

    # Apply gaussian filter to values and mask
    filtered = scipy.ndimage.gaussian_filter(arr_filled, sigma, **kwargs)
    normalization = scipy.ndimage.gaussian_filter(mask_f, sigma, **kwargs)

    # Avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        filtered /= normalization

    # Where normalization is zero, set result to NaN
    filtered[normalization == 0] = np.nan

    return filtered


@jit(nopython=True, parallel=True)  # type: ignore
def median_filter_numba(array: NDArrayNum, size: int) -> NDArrayNum:
    """
    Apply a median filter to a raster that may contain NaNs, using numbas's implementation.

    :param array: The input array to be filtered.
    :param size: The size of the window to use (must be odd).

    :returns: The filtered array (same shape as input).
    """

    if size % 2 == 0:
        raise ValueError("`size` must be odd.")

    N1, N2 = array.shape
    pad = size // 2

    padded = np.full((N1 + 2 * pad, N2 + 2 * pad), np.nan, dtype=array.dtype)

    for row in range(N1):
        for col in range(N2):
            padded[row + pad, col + pad] = array[row, col]

    outputs = np.full((N1, N2), np.nan, dtype=array.dtype)

    for row in prange(N1):
        for col in prange(N2):
            window = padded[row : row + size, col : col + size]
            outputs[row, col] = np.nanmedian(window)

    return outputs


def median_filter(array: NDArrayNum, size: int, engine: Literal["scipy", "numba"] = "scipy") -> NDArrayNum:
    """
    Apply a median filter to a raster that may contain NaNs.

    For 2D arrays, the filter is applied over both dimensions.
    For 3D arrays, the filter is applied independently to each 2D slice
    (i.e., only along the horizontal dimensions, not across the third dimension).

    This differs from scipy's built-in median_filter, which applies
    the filter across all dimensions by default.

    :param array: The input array to be filtered.
    :param size: The size of the filtering window (must be odd).
    :param engine: Filtering engine to use, either "scipy" or "numba".
    :returns: The filtered array (same shape as input).
    """

    if size % 2 == 0:
        raise ValueError("`size` must be odd.")

    if array.ndim == 2:
        return _apply_median_filter_2d(array, size, engine)
    elif array.ndim == 3:
        return np.stack([_apply_median_filter_2d(slice_, size, engine) for slice_ in array])
    raise ValueError("Input array must be 2D or 3D.")


def _apply_median_filter_2d(array: NDArrayNum, size: int, engine: Literal["scipy", "numba"]) -> NDArrayNum:
    """
    Apply a 2D median filter on an array that may contain NaNs.

    :param array: 2D input array to filter, may contain NaNs.
    :param size: Size of the median filter window (must be odd).
    :param engine: Filtering engine to use, either "scipy" or "numba".
    :returns: Filtered array of the same shape as input.
    """

    nans = np.isnan(array)

    if engine == "scipy":
        median_vals = generic_filter_scipy(array, np.nanmedian, size=size, mode="constant", cval=np.nan)
        return np.where(nans, array, median_vals)
    if not _has_numba:
        raise ValueError("Install 'numba' for accelerated filtering.")
    median_vals = median_filter_numba(array, size)
    return np.where(nans, array, median_vals)


def mean_filter(array: NDArrayNum, size: int) -> NDArrayNum:
    """
    Apply a mean filter to a 2D array that may contain NaNs.

    :param array: 2D input array
    :param size: size of the square kernel
    :no_data: no data value
    :return: filtered array with same shape
    """
    if array.ndim != 2:
        raise ValueError(f"Invalid array shape {array.shape}, expected 2D.")

    # Mask nodata values
    nans = np.isnan(array)
    mask = ~np.isnan(array)
    array_filled = np.where(mask, array, 0)
    # Compute sum over the kernel
    sum_vals = uniform_filter(array_filled, size=size, mode="constant", cval=0.0)
    # Count of valid (non-nodata) pixels in the kernel
    count_vals = uniform_filter(mask.astype(float), size=size, mode="constant", cval=0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_vals = sum_vals / count_vals

    return np.where(nans, array, mean_vals)


def min_filter(array: NDArrayNum, size: int = 5, **kwargs: Any) -> NDArrayNum:
    """
    Apply a minimum filter to a raster that may contain NaNs, using scipy's implementation.

    :param array: The input array to be filtered.
    :param size:  the shape that is taken from the input array, at every element position,
    to define the input to the filter function

    :returns: The filtered array (same shape as input).
    """
    # Check that array dimension is 2 or 3
    if array.ndim not in [2, 3]:
        raise ValueError(f"Invalid array shape given: {array.shape}. Expected 2D or 3D array.")

    nans = np.isnan(array)
    # We replace temporarily NaNs by infinite values during filtering to avoid spreading NaNs
    array_nans_replaced = np.where(nans, np.inf, array)
    array_nans_replaced_f = scipy.ndimage.minimum_filter(
        array_nans_replaced, size=size, mode="constant", cval=np.inf, **kwargs
    )
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
    if array.ndim not in [2, 3]:
        raise ValueError(f"Invalid array shape given: {array.shape}. Expected 2D or 3D array.")

    nans = np.isnan(array)
    # We replace temporarily NaNs by negative infinite values during filtering to avoid spreading NaNs
    array_nans_replaced = np.where(nans, -np.inf, array)
    array_nans_replaced_f = scipy.ndimage.maximum_filter(
        array_nans_replaced, size=size, mode="constant", cval=-np.inf, **kwargs
    )
    # In the end we want the filtered array without negative infinite values, so we put back NaNs
    return np.where(nans, array, array_nans_replaced_f)


def distance_filter(array: NDArrayNum, radius: float, outlier_threshold: float) -> NDArrayNum:
    """
    Filter out pixels whose value is distant more than a set threshold from the average value of all neighbor \
    pixels within a given radius.
    Filtered pixels are set to NaN.
    For npw, we use the gaussian filter for calculated the average value

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
    **kwargs: Any,
) -> NDArrayNum:
    """
    Apply a filter from a function.

    :param array: the input array to be filtered.
    :param filter_function: the function of the filter.

    :returns: the filtered array (same shape as input).
    """
    # Check that array dimension is 2 or 3
    if array.ndim not in [2, 3]:
        raise ValueError(f"Invalid array shape given: {array.shape}. Expected 2D or 3D array.")
    return filter_function(array, **kwargs)
