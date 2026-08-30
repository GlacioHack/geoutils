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

"""Fill missing raster values from neighboring valid data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from rasterio.fill import fillnodata as rio_fillnodata

from geoutils._typing import MArrayNum, NDArrayBool
from geoutils.raster.array import _as_bands, _masked_raster_data, _processing_mask

if TYPE_CHECKING:
    from geoutils.raster.base import RasterLike


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
