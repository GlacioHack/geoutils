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

"""Filters raster in grid windows."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import scipy.ndimage
from affine import Affine
from packaging.version import Version
from rasterio.features import sieve as rio_sieve

from geoutils._misc import import_optional
from geoutils._typing import MArrayNum, NDArrayBool, NDArrayNum
from geoutils.multiproc import MultiprocConfig, map_overlap
from geoutils.operators.neighbours import GridNeighbours, _grid_window_kernel
from geoutils.operators.neighbours import _create_circular_mask as _create_circular_mask
from geoutils.operators.nodata import NodataHandling
from geoutils.operators.reducer import (
    Count,
    Maximum,
    Mean,
    Median,
    Minimum,
    Range,
    Reducer,
    RootMeanSquare,
    Sum,
    _can_reduce_arrays,
    _jit,
)
from geoutils.raster.array import _as_bands, _masked_raster_data, _processing_mask

if TYPE_CHECKING:
    import rasterio as rio

    from geoutils.raster.base import RasterBase, RasterLike
    from geoutils.raster.raster import Raster

try:
    import dask.array as da
except Exception:  # keep optional at import time
    da = None  # type: ignore

try:
    from numba import prange
except ImportError:
    prange = range


if Version(scipy.__version__) > Version("1.16.0"):
    generic_filter_scipy = scipy.ndimage.vectorized_filter
    _has_vectorized_filter = True
else:
    generic_filter_scipy = scipy.ndimage.generic_filter
    _has_vectorized_filter = False


def _window_sum(array: NDArrayNum, kernel: NDArrayNum, engine: Literal["scipy", "numba"]) -> NDArrayNum:
    """Add weighted neighboring values, using sliding sums for rectangular windows."""

    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float64)
    if engine == "scipy" and np.all(kernel == 1):
        return scipy.ndimage.uniform_filter(array, size=kernel.shape, mode="constant", cval=0) * kernel.size
    # Convolution reverses its kernel; neighborhood offsets select values in their original direction
    return convolution(array[None], kernel[None, ::-1, ::-1], engine=engine, cval=0)[0, 0]


def _reduce_grid_array(
    array: NDArrayNum,
    operator: Reducer,
    kernel: NDArrayNum,
    *,
    nodata_propagation: NodataHandling | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayNum | None:
    """Reduce every raster cell with a fixed window, or return None for a custom calculation.

    _window_sum() shares sliding sums and convolution with the array filters. Extrema and ordinary medians use
    SciPy's corresponding filters. Missing cells are excluded from normalization, while cells outside the array
    are absent from the neighborhood, including when nodata must propagate.
    """

    supported = (Mean, Sum, Count, RootMeanSquare, Minimum, Maximum, Range, Median)
    if type(operator) not in supported or not _can_reduce_arrays(operator):
        return None
    handling = operator.default_nodata_propagation if nodata_propagation is None else nodata_propagation
    if handling not in ("ignore", "propagate"):
        return None
    _validate_filter_engine(engine)

    # Exclude masked and nonfinite observations from both values and counts
    source = np.asarray(np.ma.getdata(array), dtype=np.float64)
    valid = np.isfinite(source) & ~np.ma.getmaskarray(array)
    footprint = kernel > 0
    counts = np.rint(_window_sum(valid.astype(float), footprint.astype(float), engine))
    weighted = np.any(kernel[footprint] != 1)
    weight_sum = _window_sum(valid.astype(float), kernel, engine) if weighted else counts
    enough = counts >= operator.minimum_inputs
    if engine == "numba":
        # The existing Numba loops use square buffers; pad rectangular footprints with excluded cells
        side = max(footprint.shape)
        row_pad, col_pad = (side - footprint.shape[0]) // 2, (side - footprint.shape[1]) // 2
        numba_footprint = np.pad(footprint, ((row_pad, row_pad), (col_pad, col_pad)))

    # Linear statistics share the same weighted sums and normalization
    if type(operator) is Count:
        output = weight_sum
    elif type(operator) in (Mean, Sum, RootMeanSquare):
        values = np.where(valid, source, 0)
        if type(operator) is RootMeanSquare:
            values = np.square(values)
        output = _window_sum(values, kernel, engine)
        if type(operator) is not Sum:
            np.divide(output, weight_sum, out=output, where=weight_sum > 0)
            enough &= weight_sum > 0
        if type(operator) is RootMeanSquare:
            np.sqrt(np.maximum(output, 0), out=output)
    elif type(operator) is Median:
        missing_values = np.where(valid, source, np.nan)
        if engine == "numba":
            output = _median_filter_numba(missing_values, side, numba_footprint)
        else:
            output = generic_filter_scipy(
                missing_values, np.nanmedian, footprint=footprint, mode="constant", cval=np.nan
            )
    else:
        extrema = []
        for maximum in (False, True) if type(operator) is Range else (type(operator) is Maximum,):
            fill = -np.inf if maximum else np.inf
            filled = np.where(valid, source, fill)
            if engine == "numba":
                reduced = _minmax_filter_numba(filled[None], side, fill, maximum, numba_footprint)[0]
            else:
                calculation = scipy.ndimage.maximum_filter if maximum else scipy.ndimage.minimum_filter
                reduced = calculation(filled, footprint=footprint, mode="constant", cval=fill)
            extrema.append(reduced)
        output = extrema[-1]
        if type(operator) is Range:
            output = output - extrema[0]

    # Propagate missing source cells, without counting the padding outside the raster
    if handling == "propagate":
        missing = _window_sum((~valid).astype(float), footprint.astype(float), engine)
        enough &= np.rint(missing) == 0
    output[~enough] = np.nan
    return output


def _validate_filter_engine(engine: str) -> None:
    """Check that a filter engine is supported and load Numba only when requested."""

    if engine not in {"scipy", "numba"}:
        raise ValueError('Engine must be "scipy" or "numba".')
    if engine == "numba":
        import_optional("numba")


def gaussian_filter(
    array: NDArrayNum, sigma: float = 1, engine: Literal["scipy", "numba"] = "scipy", **kwargs: Any
) -> NDArrayNum:
    """
    Apply a Gaussian filter to a raster that may contain NaNs.
    N.B: kernel_size is set automatically based on sigma.

    For 3D arrays, each 2D band is filtered independently.

    :param array: The input array to be filtered.
    :param sigma: The sigma of the Gaussian kernel
    :param engine: Filtering engine to use, either "scipy" or "numba".

    :returns: The filtered array (same shape as input)
    """

    if array.ndim not in [2, 3]:
        raise ValueError("Gaussian filter can't be applied to 1D arrays.")
    _validate_filter_engine(engine)
    if sigma < 0:
        raise ValueError("Sigma must be non-negative.")

    # Add a band dimension to 2D inputs so both shapes follow the same filtering path
    squeeze = array.ndim == 2
    bands = array[np.newaxis] if squeeze else array

    # Boolean mask: True where NaN
    mask = np.isnan(bands)
    mask_f = (~mask).astype(float)

    # Replace NaNs with 0
    arr_filled = np.where(mask, 0, bands)

    # Apply gaussian filter to values and mask
    if engine == "scipy":
        if squeeze:
            filtered = scipy.ndimage.gaussian_filter(arr_filled[0], sigma, mode="constant", cval=0, **kwargs)[
                np.newaxis
            ]
            normalization = scipy.ndimage.gaussian_filter(mask_f[0], sigma, mode="constant", cval=0, **kwargs)[
                np.newaxis
            ]
        else:
            filtered = np.stack(
                [scipy.ndimage.gaussian_filter(band, sigma, mode="constant", cval=0, **kwargs) for band in arr_filled]
            )
            normalization = np.stack(
                [scipy.ndimage.gaussian_filter(band, sigma, mode="constant", cval=0, **kwargs) for band in mask_f]
            )
    else:
        truncate = float(kwargs.pop("truncate", 4.0))
        if kwargs:
            raise ValueError("The Numba engine only supports the 'truncate' Gaussian filter option.")

        # Build SciPy's one-dimensional Gaussian kernel and convolve along each spatial axis
        radius = int(truncate * sigma + 0.5)
        coordinates = np.arange(-radius, radius + 1, dtype=float)
        kernel_1d = np.exp(-0.5 / sigma**2 * coordinates**2) if sigma > 0 else np.ones(1)
        kernel_1d /= kernel_1d.sum()
        horizontal_kernel = kernel_1d.reshape(1, 1, -1)
        vertical_kernel = kernel_1d.reshape(1, -1, 1)
        filtered = convolution(arr_filled, horizontal_kernel, engine="numba", cval=0)[:, 0]
        filtered = convolution(filtered, vertical_kernel, engine="numba", cval=0)[:, 0]
        normalization = convolution(mask_f, horizontal_kernel, engine="numba", cval=0)[:, 0]
        normalization = convolution(normalization, vertical_kernel, engine="numba", cval=0)[:, 0]

    # Avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        filtered /= normalization

    # Where normalization is zero, set result to NaN
    filtered[normalization == 0] = np.nan

    return filtered[0] if squeeze else filtered


@_jit(nopython=True, parallel=True)
def _median_filter_numba(array: NDArrayNum, size: int, footprint: NDArrayBool | None = None) -> NDArrayNum:
    """
    Apply a median filter to a raster that may contain NaNs, using numbas's implementation.

    :param array: The input array to be filtered.
    :param size: The size of the window to use (must be odd).
    :param footprint: Optional selected cells within the square window.

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
            if footprint is None:
                outputs[row, col] = np.nanmedian(window)
            else:
                outputs[row, col] = np.nanmedian(window.flatten()[footprint.flatten()])

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
    _validate_filter_engine(engine)

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
        median_vals = _median_filter_numba(array, size)
        return np.where(nans, array, median_vals)


@overload
def mean_filter(
    array: NDArrayNum,
    size: int = 5,
    *,
    kernel_shape: Literal["square", "circular"] = "square",
    engine: Literal["scipy", "numba"] = "scipy",
    preserve_nodata: bool = True,
    boundless: bool = True,
    return_counts: Literal[False] = False,
) -> NDArrayNum: ...


@overload
def mean_filter(
    array: NDArrayNum,
    size: int = 5,
    *,
    kernel_shape: Literal["square", "circular"] = "square",
    engine: Literal["scipy", "numba"] = "scipy",
    preserve_nodata: bool = True,
    boundless: bool = True,
    return_counts: Literal[True],
) -> tuple[NDArrayNum, NDArrayNum, int]: ...


def mean_filter(
    array: NDArrayNum,
    size: int = 5,
    *,
    kernel_shape: Literal["square", "circular"] = "square",
    engine: Literal["scipy", "numba"] = "scipy",
    preserve_nodata: bool = True,
    boundless: bool = True,
    return_counts: bool = False,
) -> NDArrayNum | tuple[NDArrayNum, NDArrayNum, int]:
    """
    Apply a mean filter to a 2D array that may contain NaNs.

    :param array: 2D input array.
    :param size: Size of the square or circular kernel.
    :param kernel_shape: Shape of the kernel, either "square" or "circular".
    :param engine: Filtering engine to use, either "scipy" or "numba".
    :param preserve_nodata: Whether missing input cells remain missing in the output.
    :param boundless: Whether to compute partial windows along array edges.
    :param return_counts: Whether to also return finite cell counts and the total kernel cell count.
    :returns: Filtered array, optionally with the finite counts and total kernel cell count.
    """
    if array.ndim != 2:
        raise ValueError(f"Invalid array shape {array.shape}, expected 2D.")
    _validate_filter_engine(engine)

    # Mask nodata values
    valid = np.isfinite(array)
    array_filled = np.where(valid, array, 0)

    # Define the cells included in the requested kernel
    if kernel_shape == "square":
        kernel = np.ones((size, size), dtype=float)
        kernel_count = size**2
    elif kernel_shape == "circular":
        kernel = _create_circular_mask((size, size)).astype(float)
        kernel_count = int(np.count_nonzero(kernel))
    else:
        raise ValueError('Kernel shape should be "square" or "circular".')

    # Keep the optimized SciPy implementation used by the existing square mean filter
    sum_vals = _window_sum(array_filled, kernel[::-1, ::-1], engine)
    count_vals = np.rint(_window_sum(valid.astype(float), kernel[::-1, ::-1], engine))
    finite_counts = np.rint(count_vals) if return_counts else None

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_vals = sum_vals / count_vals

    # Exclude windows that extend beyond the image when complete patches are required
    if not boundless:
        before = (size - 1) // 2
        after = size // 2
        if before > 0:
            mean_vals[:before, :] = np.nan
            mean_vals[:, :before] = np.nan
            if finite_counts is not None:
                finite_counts[:before, :] = np.nan
                finite_counts[:, :before] = np.nan
        if after > 0:
            mean_vals[-after:, :] = np.nan
            mean_vals[:, -after:] = np.nan
            if finite_counts is not None:
                finite_counts[-after:, :] = np.nan
                finite_counts[:, -after:] = np.nan

    if preserve_nodata:
        mean_vals = np.where(valid, mean_vals, array)

    if return_counts:
        assert finite_counts is not None
        return mean_vals, finite_counts, kernel_count
    return mean_vals


@_jit(nopython=True, parallel=True, cache=True)
def _minmax_filter_numba(
    array: NDArrayNum, size: int, fill_value: float, find_maximum: bool, footprint: NDArrayBool | None = None
) -> NDArrayNum:
    """Apply a compiled minimum or maximum filter independently to stacked 2D arrays."""

    before = size // 2
    padded = np.full(
        (array.shape[0], array.shape[1] + size - 1, array.shape[2] + size - 1),
        fill_value,
        dtype=array.dtype,
    )
    padded[
        :,
        before : before + array.shape[1],
        before : before + array.shape[2],
    ] = array
    output = np.empty_like(array)

    for band in prange(array.shape[0]):
        for row in range(array.shape[1]):
            for col in range(array.shape[2]):
                result = fill_value
                for window_row in range(size):
                    for window_col in range(size):
                        if footprint is not None and not footprint[window_row, window_col]:
                            continue
                        value = padded[band, row + window_row, col + window_col]
                        if (find_maximum and value > result) or (not find_maximum and value < result):
                            result = value
                output[band, row, col] = result

    return output


def min_filter(
    array: NDArrayNum, size: int = 5, engine: Literal["scipy", "numba"] = "scipy", **kwargs: Any
) -> NDArrayNum:
    """
    Apply a minimum filter to a raster that may contain NaNs.

    For 3D arrays, each 2D band is filtered independently.

    :param array: The input array to be filtered.
    :param size:  the shape that is taken from the input array, at every element position,
    to define the input to the filter function
    :param engine: Filtering engine to use, either "scipy" or "numba".

    :returns: The filtered array (same shape as input).
    """
    return _extrema_filter(array, size, engine, False, **kwargs)


def max_filter(
    array: NDArrayNum, size: int = 5, engine: Literal["scipy", "numba"] = "scipy", **kwargs: Any
) -> NDArrayNum:
    """
    Apply a maximum filter to a raster that may contain NaNs.

    For 3D arrays, each 2D band is filtered independently.

    :param array: the input array to be filtered.
    :param size:  the shape that is taken from the input array, at every element position,
    to define the input to the filter function
    :param engine: Filtering engine to use, either "scipy" or "numba".

    :returns: the filtered array (same shape as input).
    """
    return _extrema_filter(array, size, engine, True, **kwargs)


def _extrema_filter(
    array: NDArrayNum, size: int, engine: Literal["scipy", "numba"], find_maximum: bool, **kwargs: Any
) -> NDArrayNum:
    """Apply minimum or maximum filtering independently to each raster band."""

    # Check that array dimension is 2 or 3
    if array.ndim not in [2, 3]:
        raise ValueError(f"Invalid array shape given: {array.shape}. Expected 2D or 3D array.")
    _validate_filter_engine(engine)

    fill_value = -np.inf if find_maximum else np.inf
    calculation = scipy.ndimage.maximum_filter if find_maximum else scipy.ndimage.minimum_filter
    nans = np.isnan(array)
    # We replace temporarily NaNs by infinite values during filtering to avoid spreading NaNs
    array_nans_replaced = np.where(nans, fill_value, array)
    if engine == "scipy":
        if array.ndim == 2:
            array_nans_replaced_f = calculation(
                array_nans_replaced, size=size, mode="constant", cval=fill_value, **kwargs
            )
        else:
            array_nans_replaced_f = np.stack(
                [
                    calculation(band, size=size, mode="constant", cval=fill_value, **kwargs)
                    for band in array_nans_replaced
                ]
            )
    else:
        if kwargs:
            raise ValueError("The Numba engine does not support additional extrema filter options.")
        bands = array_nans_replaced[np.newaxis] if array.ndim == 2 else array_nans_replaced
        array_nans_replaced_f = _minmax_filter_numba(bands, size, fill_value, find_maximum)
        if array.ndim == 2:
            array_nans_replaced_f = array_nans_replaced_f[0]
    # In the end, we want the filtered array without infinite values, so we put back NaNs
    return np.where(nans, array, array_nans_replaced_f)


@_jit(nopython=True, parallel=True, cache=True)
def _convolution_numba(imgs: NDArrayNum, filters: NDArrayNum, output: NDArrayNum) -> NDArrayNum:
    """Accumulate convolution over image and kernel stacks using compiled loops."""

    # Read image and kernel dimensions for the output loops
    n_N, N1, N2 = imgs.shape
    n_M, M1, M2 = filters.shape

    # Restrict windows to complete footprints within the padded input
    row_range = N1 - M1 + 1
    col_range = N2 - M2 + 1

    # Accumulate each output pixel from its complete input window
    for ii in range(n_N):
        for rr in prange(row_range):
            for cc in range(col_range):
                for m1 in range(M1):
                    for m2 in range(M2):
                        for ff in range(n_M):
                            imgval = imgs[ii, rr + m1, cc + m2]

                            # Reverse both kernel axes to compute convolution
                            filterval = filters[ff, M1 - 1 - m1, M2 - 1 - m2]
                            output[ii, ff, rr, cc] += imgval * filterval

    return output


def convolution(
    imgs: NDArrayNum,
    filters: NDArrayNum,
    engine: Literal["scipy", "numba"] = "scipy",
    cval: float = np.nan,
) -> NDArrayNum:
    """
    Convolution on a number n_N of 2D images of size N1 x N2 using a number of kernels n_M of sizes M1 x M2, using
    either scipy.ndimage.convolve or accelerated numba loops.
    Note that the indexes on n_M and n_N correspond to first axes on the array to speed up computations (prefetching).
    Inspired by: https://laurentperrinet.github.io/sciblog/posts/2017-09-20-the-fastest-2d-convolution-in-the-world.html

    :param imgs: Input array of size (n_N, N1, N2) with n_N images of size N1 x N2
    :param filters: Input array of filters of size (n_M, M1, M2) with n_M filters of size M1 x M2
    :param engine: Filtering engine to use, either "scipy" or "numba".
    :param cval: Value used outside the image boundaries.

    :return: Filled array of outputs of size (n_N, n_M, N1, N2)
    """

    # Validate image and kernel stacks before allocating the output
    imgs = np.asarray(imgs, dtype=float)
    filters = np.asarray(filters, dtype=float)
    if imgs.ndim != 3 or filters.ndim != 3 or any(size < 1 for size in filters.shape):
        raise ValueError("Images and filters must be 3D stacks with non-empty kernels.")
    _validate_filter_engine(engine)

    # Initialize output array according to input shapes
    n_N, N1, N2 = imgs.shape
    n_M, M1, M2 = filters.shape
    output = np.zeros((n_N, n_M, N1, N2))

    # Apply each kernel to each image, preserving the existing NaN padding outside the image
    if engine == "scipy":
        for image_index in range(n_N):
            for filter_index in range(n_M):
                output[image_index, filter_index] = scipy.ndimage.convolve(
                    imgs[image_index], filters[filter_index], mode="constant", cval=cval
                )
    else:
        # Pad asymmetrically for even kernel widths so compiled loops match SciPy's kernel origin
        half_M1 = int((M1 - 1) / 2)
        half_M2 = int((M2 - 1) / 2)
        imgs_pad = np.pad(
            imgs,
            pad_width=((0, 0), (half_M1, M1 // 2), (half_M2, M2 // 2)),
            constant_values=cval,
        )
        output = _convolution_numba(
            imgs=imgs_pad,
            filters=filters,
            output=output,
        )

    return output


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


def _filter_grid(
    array: NDArrayNum,
    operator: Reducer,
    *,
    transform: rio.transform.Affine,
    neighborhood: GridNeighbours,
    kernel: NDArrayNum | None,
    fractional: bool = False,
    preserve_nodata: bool = True,
    boundless: bool = True,
    nodata_handling: NodataHandling | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
    source_index_offset: tuple[int, int] = (0, 0),
    source_shape: tuple[int, int] | None = None,
    band: int = 1,
) -> NDArrayNum:
    """Reduce windows across each raster band and apply the requested output mask.

    _reduce_grid_array() uses array filters for built-in reducers. Other reducers receive bounded batches through
    _resample_at_points(), with coordinates and cell IDs referring to the complete source raster.
    """

    from geoutils.operators.execution import _resample_at_points

    bands = array[None] if array.ndim == 2 else array
    if bands.ndim != 3:
        raise ValueError("Raster filtering requires a 2D array or a stack of bands.")
    global_shape = (array.shape[-2], array.shape[-1]) if source_shape is None else source_shape
    output = np.empty(bands.shape, dtype=float)
    handling = operator.default_nodata_propagation if nodata_handling is None else nodata_handling

    # Apply the same neighborhood independently to each band
    for index, values in enumerate(bands):
        reduced = (
            None
            if kernel is None
            else _reduce_grid_array(values, operator, kernel, nodata_propagation=handling, engine=engine)
        )
        if reduced is None:
            reduced = np.empty(values.shape, dtype=float)
            for start in range(0, values.size, 4096):
                positions = np.arange(start, min(start + 4096, values.size))
                rows, cols = np.unravel_index(positions, values.shape)
                x = transform.a * (cols + 0.5) + transform.b * (rows + 0.5) + transform.c
                y = transform.d * (cols + 0.5) + transform.e * (rows + 0.5) + transform.f
                reduced.ravel()[positions] = _resample_at_points(
                    values,
                    transform,
                    (x, y),
                    operator,
                    area_or_point=None,
                    shift_area_or_point=False,
                    nodata_propagation=handling,
                    dist_nodata_spread=None,
                    neighborhood=neighborhood,
                    fractional_window=2 * max(neighborhood.overlap) + 1 if fractional else None,
                    fractional_shape=neighborhood.window_shape or "square",
                    band=band + index,
                    source_index_offset=source_index_offset,
                    source_shape=global_shape,
                )
        output[index] = reduced

    # Complete windows must fit inside the source raster, including across chunk boundaries
    if not boundless:
        if kernel is None:
            row_offsets, col_offsets = np.asarray(neighborhood.offsets).T
        else:
            selected_rows, selected_cols = np.nonzero(kernel)
            row_offsets = selected_rows - kernel.shape[0] // 2
            col_offsets = selected_cols - kernel.shape[1] // 2
        rows = np.arange(bands.shape[1]) + source_index_offset[0]
        cols = np.arange(bands.shape[2]) + source_index_offset[1]
        row_inside = (rows + row_offsets.min() >= 0) & (rows + row_offsets.max() < global_shape[0])
        col_inside = (cols + col_offsets.min() >= 0) & (cols + col_offsets.max() < global_shape[1])
        output[:, ~row_inside, :] = np.nan
        output[:, :, ~col_inside] = np.nan
    if preserve_nodata:
        valid = np.isfinite(np.ma.getdata(bands)) & ~np.ma.getmaskarray(bands)
        output[~valid] = np.nan
    return output[0] if array.ndim == 2 else output


def _overlap_depth_for_filter(method: str | Reducer | Callable[..., NDArrayNum], size: int, **kwargs: Any) -> int:
    """Return the number of neighboring pixels each chunk needs to filter its edges."""

    # A window of odd size needs half its width from neighboring chunks
    if isinstance(method, Reducer):
        kernel = kwargs["kernel"]
        return (
            max(kwargs["neighborhood"].overlap) if kernel is None else max(dimension // 2 for dimension in kernel.shape)
        )

    # For a custom callable, assume the requested size describes its window
    if not isinstance(method, str):
        return max(0, int(size) // 2)

    # These filters use the requested window size along both spatial axes
    if method in {"median", "mean", "max", "min"}:
        return max(0, int(size) // 2)

    # A Gaussian kernel reaches truncate × sigma pixels from the center
    if method in {"gaussian", "distance"}:
        sigma = kwargs.get("sigma", 1 if method == "gaussian" else 5)  # Match the filter's default sigma
        truncate = float(kwargs.get("truncate", 4.0))

        # Sigma can be scalar or sequence; take max to ensure enough depth for both axes
        if np.isscalar(sigma):
            sig = float(sigma)  # type: ignore
        else:
            sig = float(np.max(np.asarray(sigma, dtype=float)))

        radius = int(math.ceil(truncate * sig))
        return max(0, radius)

    # Use the ordinary window radius for other method names
    return max(0, int(size) // 2)


def _filter_base(
    array: NDArrayNum, method: str | Reducer | Callable[..., NDArrayNum], size: int = 3, **kwargs: Any
) -> NDArrayNum:
    """
    Dispatch filter application by method name or custom callable.

    :param array: Array to filter.
    :param method: Filter method name or callable.
    """

    if isinstance(method, Reducer):
        return _filter_grid(array, method, **kwargs)
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float32)
    if np.ma.isMaskedArray(array):
        array = array.filled(np.nan)

    # Use the named implementations so optimized filters and their engines remain available on every SciPy version
    filter_map: dict[str, Callable[..., Any]] = {
        "gaussian": gaussian_filter,
        "median": median_filter,
        "mean": mean_filter,
        "max": max_filter,
        "min": min_filter,
        "distance": distance_filter,
    }

    if isinstance(method, str):
        if method not in filter_map:
            raise ValueError(f"Unsupported filter method '{method}'. Available: {list(filter_map)}")
        func = filter_map[method]
        if method in {"median", "mean", "max", "min"}:
            kwargs["size"] = size
    elif callable(method):
        func = method
    else:
        raise TypeError("`method` must be a string or a callable.")

    return func(array, **kwargs)


def _dask_filter(
    array: da.Array,
    method: str | Reducer | Callable[..., NDArrayNum],
    size: int = 3,
    **kwargs: Any,
) -> da.Array:
    """Filter Dask chunks after reading enough neighboring pixels at each edge."""
    import_optional("dask")

    # Match the overlap to the filter's window or Gaussian radius
    depth = _overlap_depth_for_filter(method, size=size, **kwargs)
    depths = {axis: min(depth, array.shape[axis]) if axis >= array.ndim - 2 else 0 for axis in range(array.ndim)}
    if isinstance(method, Reducer):
        # Explicit rechunking makes each block's original position available to custom reducers
        rechunk = {
            axis: max(depths[axis], max(array.chunks[axis]))
            for axis in range(array.ndim)
            if min(array.chunks[axis]) < depths[axis]
        }
        if rechunk:
            array = array.rechunk(rechunk)

    # Dask passes each block with its neighboring pixels already attached
    def _block_func(block: NDArrayNum, block_info: Any = None) -> NDArrayNum:
        """Filter a block using its position in the complete source raster."""

        options = kwargs.copy()
        if isinstance(method, Reducer) and block_info is not None:
            location = block_info[None]["chunk-location"]
            starts = [
                sum(array.chunks[axis][:part]) - (depths[axis] if part > 0 else 0) for axis, part in enumerate(location)
            ]
            row, col = starts[-2:]
            options["source_index_offset"] = (row, col)
            options["transform"] = kwargs["transform"] * Affine.translation(col, row)
            options["band"] = starts[0] + 1 if array.ndim == 3 else 1
        return _filter_base(block, method=method, size=size, **options)

    # Fill pixels outside the full array with NaN, as the in-memory filter does
    return da.map_overlap(
        _block_func,
        array,
        depth=depths,
        boundary="none" if isinstance(method, Reducer) else np.nan,
        dtype=np.float64 if isinstance(method, Reducer) else array.dtype,
        meta=np.array((), dtype=np.float64 if isinstance(method, Reducer) else array.dtype),
    )


def _multiproc_filter(
    rst: Raster,
    mp_config: MultiprocConfig,
    method: str | Reducer | Callable[..., NDArrayNum],
    size: int = 3,
    **kwargs: Any,
) -> Raster:
    """Filter raster tiles in workers after including pixels from adjacent tiles."""

    # Give workers enough neighboring pixels to calculate values at tile edges
    depth = _overlap_depth_for_filter(method, size=size, **kwargs)

    # Use the same block function and window depth as Dask filtering
    return map_overlap(_multiproc_filter_block, rst, mp_config, method, size, kwargs, depth=depth)


def _multiproc_filter_block(
    block: Raster,
    method: str | Reducer | Callable[..., NDArrayNum],
    size: int,
    kwargs: dict[str, Any],
) -> Raster:
    """Filter one raster block in a serializable multiprocessing task."""

    # Convert masked values to NaNs before applying the common filter implementation
    nan_block = block.to_nanarray()
    if isinstance(method, Reducer):
        kwargs = kwargs.copy()
        col, row = ~kwargs["transform"] * (block.transform.c, block.transform.f)
        kwargs["source_index_offset"] = (int(round(row)), int(round(col)))
        kwargs["transform"] = block.transform
    filtered_block = _filter_base(nan_block, method=method, size=size, **kwargs)
    return block.copy(new_array=filtered_block)


def _filter(
    source_raster: RasterBase,
    method: str | Reducer | Callable[..., NDArrayNum],
    size: int | None,
    sigma: int = 1,
    engine: Literal["scipy", "numba"] = "scipy",
    outlier_threshold: float = 2.0,
    mp_config: MultiprocConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Filter a raster in memory, through Dask chunks, or in worker processes."""

    # Cannot use Multiprocessing backend and Dask backend simultaneously
    mp_backend = mp_config is not None
    dask_backend = da is not None and source_raster._chunks is not None
    if mp_backend and dask_backend:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove mp_config parameter "
            "from filter(). To use Multiprocessing, open the file without chunks."
        )

    # Supply only the options used by the selected filter
    if kwargs is None:
        kwargs = {}
    if method == "gaussian":
        kwargs.update({"sigma": sigma})
    if isinstance(method, str) and method in {"gaussian", "median", "mean", "max", "min", "distance"}:
        kwargs.update({"engine": engine})
    if method == "distance":
        kwargs.update({"outlier_threshold": outlier_threshold})

    if isinstance(method, Reducer):
        # Explicit filter options override the neighborhood for this call without changing the reducer
        neighborhood = method.default_neighborhood
        if neighborhood is not None and not isinstance(neighborhood, GridNeighbours):
            raise TypeError("Raster filtering requires GridNeighbours.")
        shape = kwargs.pop("kernel_shape", None)
        if size is not None or shape is not None or neighborhood is None:
            window_size = 3 if neighborhood is None else 2 * max(neighborhood.overlap) + 1
            if size is not None:
                window_size = size
            window_shape = shape or (neighborhood.window_shape if neighborhood is not None else None) or "square"
            neighborhood = GridNeighbours(size=window_size, shape=window_shape)
        if kwargs.get("fractional", False) and neighborhood.window_shape is None:
            # Record the geometric shape for custom reducers that need fractional neighborhoods
            window_size = 2 * max(neighborhood.overlap) + 1
            window_shape = "square" if len(neighborhood.offsets) == window_size**2 else "circular"
            neighborhood = GridNeighbours(neighborhood.offsets, window_shape=window_shape)
        kernel = _grid_window_kernel(
            neighborhood,
            fractional=kwargs.get("fractional", False),
            transform=source_raster.transform,
            max_cells=max(4096, 4 * int(np.prod(source_raster.shape))),
        )
        kwargs.update(
            neighborhood=neighborhood,
            kernel=kernel,
            engine=engine,
            transform=source_raster.transform,
            source_shape=source_raster.shape,
        )
    size = 3 if size is None else size

    # Send file tiles to workers, build a lazy Dask result, or filter the complete array
    if mp_backend:
        assert mp_config is not None
        return _multiproc_filter(source_raster, mp_config=mp_config, method=method, size=size, **kwargs)  # type: ignore
    elif dask_backend:
        array = _dask_filter(source_raster.data, method=method, size=size, **kwargs)
    else:
        array = _filter_base(source_raster.data, method=method, size=size, **kwargs)
    return source_raster.copy(new_array=array)


def distance_filter(
    array: NDArrayNum,
    sigma: float = 5,
    outlier_threshold: float = 2,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayNum:
    """
    Filter out pixels whose value is distant more than a set threshold from the average value of all neighbor \
    pixels within a given radius.
    Filtered pixels are set to NaN.
    For npw, we use the gaussian filter for calculated the average value

    :param array: Input array to be filtered.
    :param sigma: Radius in which the average value is calculated (for Gaussian filter, this is sigma).
    :param outlier_threshold: the minimum difference abs(array - mean) for a pixel to be considered an outlier.
    :param engine: Filtering engine to use, either "scipy" or "numba".

    :returns: the filtered array (same shape as input)
    """
    # Create mask of valid (finite) values
    valid_mask = np.isfinite(array)

    # Smooth both the data and the valid mask
    smoothed = gaussian_filter(np.nan_to_num(array, nan=0.0), sigma=sigma, engine=engine)
    normalization = gaussian_filter(valid_mask.astype(float), sigma=sigma, engine=engine)

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
