# Copyright (c) 2026 GeoUtils developers
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
Filters to remove outliers and reduce noise in rasters.

Raster backend dispatch precedes the individual filters. Stacked convolution and NaN-aware patch filtering
provide shared kernels and valid counts for terrain and uncertainty calculations in xDEM.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scipy
import scipy.ndimage
from packaging.version import Version
from rasterio.features import sieve as rio_sieve

from geoutils._misc import import_optional
from geoutils._typing import MArrayNum, NDArrayBool, NDArrayNum
from geoutils.multiproc import MultiprocConfig, map_overlap
from geoutils.raster.array import _as_bands, _masked_raster_data, _processing_mask

if TYPE_CHECKING:
    from geoutils.raster.base import RasterBase, RasterLike
    from geoutils.raster.raster import Raster

if Version(scipy.__version__) > Version("1.16.0"):
    generic_filter_scipy = scipy.ndimage.vectorized_filter
    _has_vectorized_filter = True
else:
    generic_filter_scipy = scipy.ndimage.generic_filter
    _has_vectorized_filter = False

try:
    from numba import jit, prange

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def jit(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Fake jit decorator if numba is not installed
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


try:
    import dask.array as da
except Exception:  # keep optional at import time
    da = None  # type: ignore


def _sieve(
    source_raster: RasterLike,
    size: int,
    connectivity: Literal[4, 8] = 4,
    mask: RasterLike | NDArrayBool | None = None,
) -> NDArrayNum | MArrayNum:
    """Remove connected integer regions smaller than a pixel count from every band."""

    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("Argument 'size' must be a strictly positive integer.")
    if connectivity not in (4, 8):
        raise ValueError("Argument 'connectivity' must be 4 or 8.")

    # Rasterio delegates connected-region filtering to GDAL and accepts integer values only
    source = _masked_raster_data(source_raster)
    if not (np.issubdtype(source.dtype, np.integer) or np.issubdtype(source.dtype, np.bool_)):
        raise ValueError("Sieve requires an integer or Boolean raster.")
    bands, squeeze = _as_bands(source)
    requested_mask = _processing_mask(mask, source.shape)
    output = np.ma.empty(bands.shape, dtype=source.dtype)

    # Each band retains its own nodata mask while using the same optional spatial mask
    for band_index, band in enumerate(bands):
        source_valid = ~np.ma.getmaskarray(band)
        valid = source_valid & requested_mask
        values = np.asarray(band.data)
        sieved = rio_sieve(values, size=int(size), mask=valid.astype(np.uint8), connectivity=connectivity)
        output[band_index] = np.ma.array(sieved, mask=~source_valid)

    result = output[0] if squeeze else output
    if np.ma.getmaskarray(result).any():
        # NaN keeps masked integer output consistent between Raster and Xarray representations
        return result.astype(np.result_type(result.dtype, np.float32)).filled(np.nan)
    return np.asarray(result)


def _overlap_depth_for_filter(method: str | Callable[..., NDArrayNum], size: int, **kwargs: Any) -> int:
    """
    Compute halo depth (pixels) needed for map_overlap so the filter is correct at block edges.
    """

    # Default for windowed filters: depth = radius
    # The size is assumed odd
    def _window_radius(sz: int) -> int:
        return max(0, int(sz) // 2)

    # For callables
    if not isinstance(method, str):
        # Unknown callable => assume window-like behavior using `size`
        return _window_radius(size)

    # For median, mean, max, min and distance filters
    if method in {"median", "mean", "max", "min", "distance"}:
        return _window_radius(size)

    # For gaussian filter
    if method == "gaussian":
        # Uses radius = truncate * sigma
        sigma = kwargs.get("sigma", 1)  # Needs to be the default defined in gaussian filter
        truncate = float(kwargs.get("truncate", 4.0))

        # Sigma can be scalar or sequence; take max to ensure enough depth for both axes
        if np.isscalar(sigma):
            sig = float(sigma)  # type: ignore
        else:
            sig = float(np.max(np.asarray(sigma, dtype=float)))

        radius = int(math.ceil(truncate * sig))
        return max(0, radius)

    # Fallback
    return _window_radius(size)


def _filter_base(
    array: NDArrayNum, method: str | Callable[..., NDArrayNum], size: int = 3, **kwargs: Any
) -> NDArrayNum:
    """
    Dispatch filter application by method name or custom callable.

    :param array: Array to filter.
    :param method: Filter method name or callable.
    """

    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float32)
    if np.ma.isMaskedArray(array):
        array = array.filled(np.nan)

    # With new SciPy, just use vectorized version
    filter_map: dict[str, Callable[..., NDArrayNum]]
    if Version(scipy.__version__) > Version("1.16.0"):
        filter_map = {
            "gaussian": gaussian_filter,
            "median": lambda arr, size=size, **_: generic_filter_scipy(
                arr, np.nanmedian, size=size, mode="constant", cval=np.nan
            ),
            "mean": lambda arr, size=size, **_: generic_filter_scipy(
                arr, np.nanmean, size=size, mode="constant", cval=np.nan
            ),
            "max": lambda arr, size=size, **_: generic_filter_scipy(
                arr, np.nanmax, size=size, mode="constant", cval=np.nan
            ),
            "min": lambda arr, size=size, **_: generic_filter_scipy(
                arr, np.nanmin, size=size, mode="constant", cval=np.nan
            ),
            "distance": distance_filter,
        }
    # With old SciPy, maintain speed with tricks from older custom filters
    else:
        filter_map = {
            "gaussian": gaussian_filter,
            "median": lambda arr, size=size, **_: median_filter(arr, size=size),
            "mean": lambda arr, size=size, **_: mean_filter(arr, size=size),
            "max": lambda arr, size=size, **_: max_filter(arr, size=size),
            "min": lambda arr, size=size, **_: min_filter(arr, size=size),
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


def _dask_filter(
    array: da.Array,
    method: str | Callable[..., NDArrayNum],
    size: int = 3,
    **kwargs: Any,
) -> da.Array:
    """
    Apply filter to a Dask array.

    Wrapper around map_overlap.
    """
    import_optional("dask")

    # Get depth of overlap
    depth = _overlap_depth_for_filter(method, size=size, **kwargs)

    # Block function to pass
    def _block_func(block: NDArrayNum) -> NDArrayNum:
        # Block already includes the halo from map_overlap
        return _filter_base(block, method=method, size=size, **kwargs)

    # Call map_overlap
    return da.map_overlap(
        _block_func,
        array,
        depth=depth,
        boundary=np.nan,
        dtype=array.dtype,
    )


def _multiproc_filter(
    rst: Raster,
    mp_config: MultiprocConfig,
    method: str | Callable[..., NDArrayNum],
    size: int = 3,
    **kwargs: Any,
) -> Raster:
    # Get depth of overlap
    depth = _overlap_depth_for_filter(method, size=size, **kwargs)

    # Call Multiprocessing map_overlap
    return map_overlap(_multiproc_filter_block, rst, mp_config, method, size, kwargs, depth=depth)


def _multiproc_filter_block(
    block: Raster,
    method: str | Callable[..., NDArrayNum],
    size: int,
    kwargs: dict[str, Any],
) -> Raster:
    """Filter one raster block in a serializable multiprocessing task."""

    # Convert masked values to NaNs before applying the common filter implementation
    nan_block = block.get_nanarray()
    filtered_block = _filter_base(nan_block, method=method, size=size, **kwargs)
    return block.copy(new_array=filtered_block)


def _filter(
    source_raster: RasterBase,
    method: str | Callable[..., NDArrayNum],
    size: int,
    sigma: int = 1,
    engine: Literal["scipy", "numba"] = "scipy",
    outlier_threshold: float = 2.0,
    mp_config: MultiprocConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Parent function to filter raster, dispatching to in-memory, Dask or Multiprocessing backend."""

    # Cannot use Multiprocessing backend and Dask backend simultaneously
    mp_backend = mp_config is not None
    dask_backend = da is not None and source_raster._chunks is not None
    if mp_backend and dask_backend:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove mp_config parameter "
            "from filter(). To use Multiprocessing, open the file without chunks."
        )

    # Pass positional argument as kwargs only if relevant
    if kwargs is None:
        kwargs = {}
    if method == "gaussian":
        kwargs.update({"sigma": sigma})
    if method == "median":
        kwargs.update({"engine": engine})
    if method == "distance":
        kwargs.update({"outlier_threshold": outlier_threshold})

    # Dispatch based on backend
    if mp_backend:
        assert mp_config is not None
        return _multiproc_filter(source_raster, mp_config=mp_config, method=method, size=size, **kwargs)  # type: ignore
    elif dask_backend:
        array = _dask_filter(source_raster.data, method=method, size=size, **kwargs)
    else:
        array = _filter_base(source_raster.data, method=method, size=size, **kwargs)
    return source_raster.copy(new_array=array)


def gaussian_filter(array: NDArrayNum, sigma: float = 1, **kwargs: Any) -> NDArrayNum:
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
    filtered = scipy.ndimage.gaussian_filter(arr_filled, sigma, mode="constant", cval=0, **kwargs)
    normalization = scipy.ndimage.gaussian_filter(mask_f, sigma, mode="constant", cval=0, **kwargs)

    # Avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        filtered /= normalization

    # Where normalization is zero, set result to NaN
    filtered[normalization == 0] = np.nan

    return filtered


@jit(nopython=True, parallel=True)
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


def _apply_median_filter_2d(
    array: NDArrayNum, size: int = 5, engine: Literal["scipy", "numba"] = "scipy"
) -> NDArrayNum:
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

    else:
        import_optional("numba")
        median_vals = median_filter_numba(array, size)
        return np.where(nans, array, median_vals)


def mean_filter(array: NDArrayNum, size: int = 5) -> NDArrayNum:
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
    sum_vals = scipy.ndimage.uniform_filter(array_filled, size=size, mode="constant", cval=0.0)
    # Count of valid (non-nodata) pixels in the kernel
    count_vals = scipy.ndimage.uniform_filter(mask.astype(float), size=size, mode="constant", cval=0.0)

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


def distance_filter(array: NDArrayNum, sigma: float = 5, outlier_threshold: float = 2) -> NDArrayNum:
    """
    Filter out pixels whose value is distant more than a set threshold from the average value of all neighbor \
    pixels within a given radius.
    Filtered pixels are set to NaN.
    For npw, we use the gaussian filter for calculated the average value

    :param array: Input array to be filtered.
    :param sigma: Radius in which the average value is calculated (for Gaussian filter, this is sigma).
    :param outlier_threshold: the minimum difference abs(array - mean) for a pixel to be considered an outlier.

    :returns: the filtered array (same shape as input)
    """
    # Create mask of valid (finite) values
    valid_mask = np.isfinite(array)

    # Smooth both the data and the valid mask
    smoothed = gaussian_filter(np.nan_to_num(array, nan=0.0), sigma=sigma)
    normalization = gaussian_filter(valid_mask.astype(float), sigma=sigma)

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


#########################################
# STACKED CONVOLUTION AND PATCH FILTERS
#########################################


def _create_circular_mask(
    shape: tuple[int, int], center: tuple[int, int] | None = None, radius: float | None = None
) -> NDArrayBool:
    """Create a circular kernel, using the array centre and half width by default."""

    # Use the array center and its nearest edge to choose a fully contained default circle
    w, h = shape

    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    # Select cells strictly inside the radius to preserve the patch kernel boundary convention
    Y, X = np.ogrid[:w, :h]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center < radius

    return mask


def convolution(imgs: NDArrayNum, filters: NDArrayNum, method: str = "scipy") -> NDArrayNum:
    """
    Convolution on a number n_N of 2D images of size N1 x N2 using a number of kernels n_M of sizes M1 x M2, using
    either scipy.ndimage.convolve or accelerated numba loops.
    Note that the indexes on n_M and n_N correspond to first axes on the array to speed up computations (prefetching).
    Inspired by: https://laurentperrinet.github.io/sciblog/posts/2017-09-20-the-fastest-2d-convolution-in-the-world.html

    :param imgs: Input array of size (n_N, N1, N2) with n_N images of size N1 x N2
    :param filters: Input array of filters of size (n_M, M1, M2) with n_M filters of size M1 x M2
    :param method: Method to perform the convolution: "scipy" or "numba"

    :return: Filled array of outputs of size (n_N, n_M, N1, N2)
    """

    # Validate image and kernel stacks before allocating the output
    imgs = np.asarray(imgs, dtype=float)
    filters = np.asarray(filters, dtype=float)
    if imgs.ndim != 3 or filters.ndim != 3 or any(size < 1 for size in filters.shape):
        raise ValueError("Images and filters must be 3D stacks with non-empty kernels.")

    # Initialize output array according to input shapes
    n_N, N1, N2 = imgs.shape
    n_M, M1, M2 = filters.shape
    output = np.zeros((n_N, n_M, N1, N2))

    # Apply each kernel to each image, preserving the existing NaN padding outside the image
    if method.lower() == "scipy":
        for image_index in range(n_N):
            for filter_index in range(n_M):
                output[image_index, filter_index] = scipy.ndimage.convolve(
                    imgs[image_index], filters[filter_index], mode="constant", cval=np.nan
                )
    elif "numba" in method.lower():
        # Load compilation support only when the caller selects the optional Numba implementation
        numba = import_optional("numba")

        @numba.njit(parallel=True)
        def _numba_convolution(imgs: NDArrayNum, filters: NDArrayNum, output: NDArrayNum) -> NDArrayNum:
            """Accumulate convolution over image and kernel stacks using compiled loops."""
            # Read image and kernel dimensions for the output loops
            n_N, N1, N2 = imgs.shape
            n_M, M1, M2 = filters.shape

            # Restrict windows to complete footprints within the padded input
            row_range = N1 - M1 + 1
            col_range = N2 - M2 + 1

            # Accumulate each output pixel from its complete input window
            for ii in range(n_N):
                for rr in numba.prange(row_range):
                    for cc in numba.prange(col_range):
                        for m1 in range(M1):
                            for m2 in range(M2):
                                for ff in range(n_M):
                                    imgval = imgs[ii, rr + m1, cc + m2]

                                    # Reverse both kernel axes to compute convolution rather than correlation
                                    filterval = filters[ff, M1 - 1 - m1, M2 - 1 - m2]
                                    output[ii, ff, rr, cc] += imgval * filterval

            return output

        # Pad asymmetrically for even kernel widths so compiled loops match SciPy's kernel origin
        half_M1 = int((M1 - 1) / 2)
        half_M2 = int((M2 - 1) / 2)
        imgs_pad = np.pad(imgs, pad_width=((0, 0), (half_M1, M1 // 2), (half_M2, M2 // 2)), constant_values=np.nan)
        output = _numba_convolution(
            imgs=imgs_pad,
            filters=filters,
            output=output,
        )
    else:
        raise ValueError('Method must be "scipy" or "numba".')

    return output


def mean_filter_nan(
    img: NDArrayNum, kernel_size: int, kernel_shape: str = "circular", method: str = "scipy"
) -> tuple[NDArrayNum, NDArrayNum, int]:
    """
    Apply a mean filter to an image with a square or circular kernel of size p and with NaN values ignored.

    :param img: Input array of size (N1, N2)
    :param kernel_size: Size M of kernel, which will be a symmetrical (M, M) kernel
    :param kernel_shape: Shape of kernel, either "square" or "circular"
    :param method: Method to perform the convolution: "scipy" or "numba"

    :return: Array of size (N1, N2) with mean values, Array of size (N1, N2) with number of valid pixels, Number of
        pixels in the kernel
    """

    # Use one width for both axes of the square or circular kernel
    p = kernel_size

    # Copy the array and replace NaNs by zeros before summing them in the convolution
    img_zeroed = img.copy()
    img_zeroed[~np.isfinite(img_zeroed)] = 0

    # Define the cells belonging to the requested kernel shape
    if kernel_shape.lower() == "square":
        kernel = np.ones((p, p), dtype="uint8")

    # Use the same circle boundary convention as the empirical patch method
    elif kernel_shape.lower() == "circular":
        kernel = _create_circular_mask((p, p)).astype("uint8")
    else:
        raise ValueError('Kernel shape should be "square" or "circular".')

    # Run convolution to compute the sum of img values
    summed_img = convolution(
        imgs=img_zeroed.reshape((1, img_zeroed.shape[0], img_zeroed.shape[1])),
        filters=kernel.reshape((1, kernel.shape[0], kernel.shape[1])),
        method=method,
    ).squeeze()

    # Count only finite observations when normalizing each window sum
    nodata_img = np.ones(np.shape(img), dtype=np.int8)
    nodata_img[~np.isfinite(img)] = 0

    # Count the number of valid pixels in the kernel with a convolution
    nb_valid_img = convolution(
        imgs=nodata_img.reshape((1, nodata_img.shape[0], nodata_img.shape[1])),  # type: ignore
        filters=kernel.reshape((1, kernel.shape[0], kernel.shape[1])),
        method=method,
    ).squeeze()

    # Divide by finite counts and leave windows without valid data undefined
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_img = summed_img / nb_valid_img

    # Retain the full kernel count so callers can reject incompletely observed windows
    nb_pixel_per_kernel = np.count_nonzero(kernel)

    return mean_img, nb_valid_img, nb_pixel_per_kernel
