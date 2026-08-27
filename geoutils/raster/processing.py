# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Apply GDAL-backed connected-region and nodata processing to raster arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr
from rasterio.features import sieve as rio_sieve
from rasterio.fill import fillnodata as rio_fillnodata

from geoutils._typing import MArrayNum, NDArrayBool, NDArrayNum

if TYPE_CHECKING:
    from geoutils.raster.base import RasterLike


def _masked_raster_data(source_raster: RasterLike) -> MArrayNum:
    """Return raster values and their mask as an in-memory masked array."""

    data = source_raster.data
    if isinstance(data, xr.DataArray):
        array = data.to_masked_array(copy=True)
    else:
        array = np.ma.array(data, copy=True)

    # Combine masked, non-finite and explicit nodata representations across both raster classes
    invalid = np.ma.getmaskarray(array) | ~np.isfinite(array.data)
    nodata = getattr(source_raster, "nodata", None)
    if nodata is not None:
        invalid |= array.data == nodata
    return np.ma.array(array.data, mask=invalid, copy=False)


def _processing_mask(mask: RasterLike | NDArrayBool | None, shape: tuple[int, ...]) -> NDArrayBool:
    """Return a Boolean processing mask matching one raster band."""

    if mask is None:
        return np.ones(shape[-2:], dtype=bool)

    # Raster masks may have a singleton band dimension that has no spatial meaning
    mask_data: Any = mask.data if hasattr(mask, "data") and not isinstance(mask, np.ndarray) else mask
    if isinstance(mask_data, xr.DataArray):
        mask_data = mask_data.to_numpy()
    mask_array = np.asarray(mask_data).squeeze()
    if mask_array.shape != shape[-2:]:
        raise ValueError(f"Mask shape {mask_array.shape} does not match raster shape {shape[-2:]}.")
    return mask_array.astype(bool)


def _as_bands(array: MArrayNum) -> tuple[MArrayNum, bool]:
    """Expose two-dimensional and multiband arrays through one band dimension."""

    if array.ndim == 2:
        return array[np.newaxis, ...], True
    if array.ndim == 3:
        return array, False
    raise ValueError("Raster processing expects a two-dimensional or multiband array.")


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

    # Rasterio delegates connected-region processing to GDAL and accepts integer values only
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


def _fill_nodata(
    source_raster: RasterLike,
    max_search_distance: float = 100.0,
    smoothing_iterations: int = 0,
    interpolation: Literal["inv_dist", "nearest"] = "inv_dist",
    mask: RasterLike | NDArrayBool | None = None,
) -> MArrayNum:
    """Fill nodata cells from nearby finite values in every raster band."""

    if not np.isfinite(max_search_distance) or max_search_distance <= 0:
        raise ValueError("Argument 'max_search_distance' must be finite and strictly positive.")
    if (
        isinstance(smoothing_iterations, bool)
        or not isinstance(smoothing_iterations, (int, np.integer))
        or smoothing_iterations < 0
    ):
        raise ValueError("Argument 'smoothing_iterations' must be a non-negative integer.")
    if interpolation not in ("inv_dist", "nearest"):
        raise ValueError("Argument 'interpolation' must be 'inv_dist' or 'nearest'.")

    # Floating-point output can represent cells that remain unfilled as NaN
    source = _masked_raster_data(source_raster)
    bands, squeeze = _as_bands(source)
    requested_mask = _processing_mask(mask, source.shape)
    output_dtype = np.result_type(source.dtype, np.float32)
    output = np.ma.empty(bands.shape, dtype=output_dtype)

    for band_index, band in enumerate(bands):
        values = np.asarray(band.data, dtype=output_dtype)
        valid = (~np.ma.getmaskarray(band)) & np.isfinite(values) & requested_mask

        # GDAL leaves cells beyond the search distance as NaN when the input starts with NaN
        working = values.copy()
        working[~valid] = np.nan
        filled = rio_fillnodata(
            working,
            mask=valid.astype(np.uint8),
            max_search_distance=float(max_search_distance),
            smoothing_iterations=int(smoothing_iterations),
            interpolation=interpolation,
        )
        output[band_index] = np.ma.masked_invalid(filled)
    return output[0] if squeeze else output
