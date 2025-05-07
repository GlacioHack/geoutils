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

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy

from geoutils._typing import NDArrayNum


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


def _nan_safe_filter(array: NDArrayNum, filter_func: Callable[..., NDArrayNum], **kwargs: Any) -> NDArrayNum:
    """Apply a NaN-safe filter using a weighting trick."""
    # Check that array dimension is 2 or 3
    if np.ndim(array) not in [2, 3]:
        raise ValueError(f"Invalid array shape given: {array.shape}. Expected 2D or 3D array.")

    # In case array does not contain NaNs, use scipy's gaussian filter directly
    if not np.isnan(array).any():
        return filter_func(array, **kwargs)

    # If array contain NaNs, need a more sophisticated approach
    # Inspired by https://stackoverflow.com/a/36307291
    # Run filter on the array with NaNs set to 0
    array_filled = np.nan_to_num(array, nan=0.0)
    weights = (~np.isnan(array)).astype(array_filled.dtype)

    filtered_array_filled = filter_func(array_filled, **kwargs)
    del array_filled

    weight_sum = filter_func(weights, **kwargs)
    del weights

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered")
        filtered_array = filtered_array_filled / weight_sum

    return filtered_array


def gaussian_filter(array: NDArrayNum, sigma: float) -> NDArrayNum:
    """
    Apply a Gaussian filter to a raster that may contain NaNs, using scipy's implementation.
    gaussian_filter_cv is recommended as it is usually faster, but this depends on the value of sigma.

    N.B: kernel_size is set automatically based on sigma.

    :param array: the input array to be filtered.
    :param sigma: the sigma of the Gaussian kernel

    :returns: the filtered array (same shape as input)
    """
    return _nan_safe_filter(array, scipy.ndimage.gaussian_filter, sigma=sigma)


def median_filter(array: NDArrayNum, **kwargs: Any) -> NDArrayNum:
    """
    Apply a median filter to a raster that may contain NaNs, using scipy's implementation.

    :param array: the input array to be filtered.

    :returns: the filtered array (same shape as input).
    """
    return _nan_safe_filter(array, scipy.ndimage.median_filter, **kwargs)


def mean_filter(array: NDArrayNum, kernel_size: int, **kwargs: Any) -> NDArrayNum:
    """
    Apply a mean filter to a raster that may contain NaNs.

    :param array: the input array to be filtered.
    :param kernel_size: the size of the kernel.

    :returns: the filtered array (same shape as input).
    """
    # Check that array dimension is 2
    if np.ndim(array) not in [2]:
        raise ValueError(f"Invalid array shape given: {array.shape}. Expected 2D array.")
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
    return _nan_safe_filter(array, scipy.ndimage.convolve, weights=kernel, **kwargs)


def min_filter(array: NDArrayNum, **kwargs: Any) -> NDArrayNum:
    """
    Apply a minimum filter to a raster that may contain NaNs, using scipy's implementation.

    :param array: the input array to be filtered.

    :returns: the filtered array (same shape as input).
    """
    # Check that array dimension is 2 or 3
    if np.ndim(array) not in [2, 3]:
        raise ValueError(f"Invalid array shape given: {array.shape}. Expected 2D or 3D array.")

    nans = np.isnan(array)
    # We replace temporarily NaNs by infinite values during filtering to avoid spreading NaNs
    array_nans_replaced = np.where(nans, np.inf, array)
    array_nans_replaced_f = scipy.ndimage.minimum_filter(array_nans_replaced, **kwargs)
    # In the end we want the filtered array without infinite values, so we put back NaNs
    return np.where(nans, array, array_nans_replaced_f)


def max_filter(array: NDArrayNum, **kwargs: Any) -> NDArrayNum:
    """
    Apply a maximum filter to a raster that may contain NaNs, using scipy's implementation.

    :param array: the input array to be filtered.

    :returns: the filtered array (same shape as input).
    """
    # Check that array dimension is 2 or 3
    if np.ndim(array) not in [2, 3]:
        raise ValueError(f"Invalid array shape given: {array.shape}. Expected 2D or 3D array.")

    nans = np.isnan(array)
    # We replace temporarily NaNs by negative infinite values during filtering to avoid spreading NaNs
    array_nans_replaced = np.where(nans, -np.inf, array)
    array_nans_replaced_f = scipy.ndimage.maximum_filter(array_nans_replaced, **kwargs)
    # In the end we want the filtered array without negative infinite values, so we put back NaNs
    return np.where(nans, array, array_nans_replaced_f)


def distance_filter(array: NDArrayNum, radius: float, outlier_threshold: float) -> NDArrayNum:
    """
    Filter out pixels whose value is distant more than a set threshold from the average value of all neighbor \
pixels within a given radius.
    Filtered pixels are set to NaN.

    TO DO: Add an option on how the "average" value should be calculated, i.e. using a Gaussian, median etc filter.

    :param array: the input array to be filtered.
    :param radius: the radius in which the average value is calculated (for Gaussian filter, this is sigma).
    :param outlier_threshold: the minimum difference abs(array - mean) for a pixel to be considered an outlier.

    :returns: the filtered array (same shape as input)
    """
    # Calculate the average value within the radius
    smooth = gaussian_filter(array, sigma=radius)

    # Filter outliers
    outliers = (np.abs(array - smooth)) > outlier_threshold
    out_array = np.copy(array)
    out_array[outliers] = np.nan

    return out_array


def generic_filter(
    array: NDArrayNum, filter_function: Callable[..., NDArrayNum], **kwargs: dict[Any, Any]
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

    return scipy.ndimage.generic_filter(array, filter_function, **kwargs)
