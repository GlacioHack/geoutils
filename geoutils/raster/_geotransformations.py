# Copyright (c) 2025 GeoUtils developers
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
Module for geotransformations functionalities required across several other modules (e.g., "chunk" module for Dask or
Multiprocessing) and later imported in geotransformations. Required to have separate to avoid import conflicts.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio as rio
from packaging.version import Version
from rasterio.enums import Resampling

from geoutils._misc import silence_rasterio_message
from geoutils._typing import DTypeLike, NDArrayNum
from geoutils.raster.georeferencing import (
    _default_nodata,
    _res,
)

if TYPE_CHECKING:
    from geoutils.raster.base import RasterType


###########################
# 1/ REPROJECT SUBFUNCTIONS
###########################


def _resampling_method_from_str(method_str: str) -> rio.enums.Resampling:
    """Get a rasterio resampling method from a string representation, e.g. "cubic_spline"."""
    # Try to match the string version of the resampling method with a rio Resampling enum name
    for method in rio.enums.Resampling:
        if method.name == method_str:
            resampling_method = method
            break
    # If no match was found, raise an error.
    else:
        raise ValueError(
            f"'{method_str}' is not a valid rasterio.enums.Resampling method. "
            f"Valid methods: {[method.name for method in rio.enums.Resampling]}"
        )
    return resampling_method


def _check_reproj_nodata_dtype(
    source_raster: RasterType,
    nodata: int | float | None,
    dtype: DTypeLike | None,
    force_source_nodata: int | float | None,
) -> tuple[DTypeLike, int | float | None, int | float | None]:
    """Check user inputs of reproject regarding nodata and data type."""

    # Set output dtype
    if dtype is None:
        # Warning: this will not work for multiple bands with different dtypes
        dtype = source_raster.dtype

    # --- Set source nodata if provided -- #
    if force_source_nodata is None:
        src_nodata = source_raster.nodata
    else:
        src_nodata = force_source_nodata
        # Raise warning if a different nodata value exists for this raster than the forced one (not None)
        if source_raster.nodata is not None:
            warnings.warn(
                "Forcing source nodata value of {} despite an existing nodata value of {} in the raster. "
                "To silence this warning, use self.set_nodata() before reprojection instead of forcing.".format(
                    force_source_nodata, source_raster.nodata
                )
            )

    # --- Set destination nodata if provided -- #
    # This is needed in areas not covered by the input data.
    # If None, will use GeoUtils' default, as rasterio's default is unknown, hence cannot be handled properly.
    if nodata is None:
        nodata = source_raster.nodata
        if nodata is None:
            nodata = _default_nodata(dtype)
            # If nodata is already being used, raise a warning.
            if not source_raster.is_loaded:
                warnings.warn(
                    f"For reprojection, nodata must be set. Setting default nodata to {nodata}. You may "
                    f"set a different nodata with `nodata`."
                )

            elif nodata in source_raster.data:
                warnings.warn(
                    f"For reprojection, nodata must be set. Default chosen value {nodata} exists in "
                    f"self.data. This may have unexpected consequences. Consider setting a different nodata with "
                    f"self.set_nodata()."
                )

    return dtype, src_nodata, nodata


def _is_reproj_needed(src_shape: tuple[int, int], reproj_kwargs: dict[str, Any]) -> bool:
    """Check if reprojection is actually needed based on transformation parameters."""

    src_transform = reproj_kwargs["src_transform"]
    transform = reproj_kwargs["dst_transform"]
    src_crs = reproj_kwargs["src_crs"]
    crs = reproj_kwargs["dst_crs"]
    grid_size = reproj_kwargs["dst_shape"][::-1]
    src_res = _res(src_transform)
    res = _res(transform)

    # Caution, grid_size is (width, height) while shape is (height, width)
    return all(
        [
            (transform == src_transform) or (transform is None),
            (crs == src_crs) or (crs is None),
            (grid_size == src_shape[::-1]) or (grid_size is None),
            np.all(np.array(res) == src_res) or (res is None),
        ]
    )


def _rio_reproject(src_arr: NDArrayNum, reproj_kwargs: dict[str, Any]) -> NDArrayNum:
    """Rasterio reprojection wrapper.

    :param src_arr: Source array for data.
    :param reproj_kwargs: Reprojection parameter dictionary.
    """

    # All masked values must be set to a nodata value for rasterio's reproject to work properly
    if np.ma.isMaskedArray(src_arr):
        is_input_masked = True
        src_mask = np.ma.getmaskarray(src_arr)
        src_arr = src_arr.data  # type: ignore
    else:
        is_input_masked = False
        src_mask = ~np.isfinite(src_arr)

    # Check reprojection is possible with nodata (boolean raster will be converted, so no need to check)
    if np.dtype(src_arr.dtype) != bool and (reproj_kwargs["src_nodata"] is None and np.sum(src_mask) > 0):
        raise ValueError(
            "No nodata set, set one for the raster with self.set_nodata() or use a temporary one "
            "with `force_source_nodata`."
        )

    # For a boolean type
    convert_bool = False
    if np.dtype(src_arr.dtype) == np.bool_:
        # To convert back later
        convert_bool = True
        # Convert to uint8 for nearest, float otherwise
        if reproj_kwargs["resampling"] in [Resampling.nearest, "nearest"]:
            src_arr = src_arr.astype("uint8")  # type: ignore
        else:
            warnings.warn(
                "Reprojecting a raster mask (boolean type) with a resampling method other than 'nearest', "
                "results in the boolean array being converted to float during reprojection."
            )
            src_arr = src_arr.astype("float32")  # type: ignore

        # Convert automated output dtype to the input dtype
        if np.dtype(reproj_kwargs["dtype"]) == np.bool_:
            reproj_kwargs["dtype"] = src_arr.dtype

        # Update nodata value, which won't exist
        reproj_kwargs["src_nodata"] = _default_nodata(src_arr.dtype)

    # Fill with nodata values on mask
    if reproj_kwargs["src_nodata"] is not None:
        src_arr[src_mask] = reproj_kwargs["src_nodata"]

    # Check if multiband
    is_multiband = len(src_arr.shape) > 2

    # Prepare destination array
    shape = (src_arr.shape[0], *reproj_kwargs["dst_shape"]) if is_multiband else reproj_kwargs["dst_shape"]
    dst_arr = np.zeros(shape, dtype=reproj_kwargs["dtype"])

    # Performance keywords
    if reproj_kwargs["num_threads"] == 0:
        # Default to cpu count minus one. If the cpu count is undefined, num_threads will be 1
        cpu_count = os.cpu_count() or 2
        num_threads = cpu_count - 1
    else:
        num_threads = reproj_kwargs["num_threads"]

    # We force XSCALE=1 and YSCALE=1 passed to GDAL.Warp to avoid resampling deformations depending on extent/shape,
    # which leads to different results on chunks or a full array
    # See: https://gdal.org/en/stable/api/gdalwarp_cpp.html#_CPPv415GDALWarpOptions
    # And: https://github.com/rasterio/rasterio/issues/2995
    reproj_kwargs.update(
        {
            "num_threads": num_threads,
            "XSCALE": 1,
            "YSCALE": 1,
        }
    )
    # If Rasterio is recent enough version, force tolerance to 0 to avoid deformations on chunks
    # See: https://github.com/rasterio/rasterio/issues/2433#issuecomment-2786157846
    if Version(rio.__version__) >= Version("1.5.0"):
        reproj_kwargs.update({"tolerance": 0})

    # Pop dtype and dst_shape arguments that don't exist in Rasterio, and are only used above
    reproj_kwargs.pop("dtype")
    reproj_kwargs.pop("dst_shape")

    # Rasterio raises a warning that src_transform are not defined when multiple ones are passed during chunked ops
    # (Dask/Multiproc), although this is not the case, maybe a bug upstream?
    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)

    # XSCALE/YSCALE have been supported for a while, but not officially exposed in the API until Rasterio 1.5,
    # so we need to silence them in warnings to avoid noise for users
    with silence_rasterio_message(param_name="SCALE"):
        # Run reprojection
        _ = rio.warp.reproject(src_arr, dst_arr, **reproj_kwargs)

    # Get output mask
    if reproj_kwargs["dst_nodata"] is not None:
        dst_mask = dst_arr == reproj_kwargs["dst_nodata"]
    else:
        dst_mask = np.zeros(src_arr.shape, dtype=bool)

    # If output needs to be converted back to boolean
    if convert_bool:
        dst_arr = dst_arr.astype(bool)

    # Set mask
    if is_input_masked:
        dst_arr = np.ma.masked_array(data=dst_arr, mask=dst_mask, fill_value=reproj_kwargs["dst_nodata"])
    else:
        dst_arr = dst_arr
        dst_arr[dst_mask] = np.nan

    return dst_arr
