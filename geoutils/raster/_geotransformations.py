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
from typing import Any, Iterable

import affine
import numpy as np
import rasterio as rio
from packaging.version import Version
from rasterio.crs import CRS
from rasterio.enums import Resampling

import geoutils as gu
from geoutils._typing import DTypeLike, NDArrayBool, NDArrayNum
from geoutils.raster.georeferencing import (
    _cast_pixel_interpretation,
    _default_nodata,
    _res,
)

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


def _user_input_reproject(
    source_raster: gu.Raster,
    ref: gu.Raster,
    crs: CRS | str | int | None,
    res: float | Iterable[float] | None,
    bounds: dict[str, float] | rio.coords.BoundingBox | None,
    nodata: int | float | None,
    dtype: DTypeLike | None,
    force_source_nodata: int | float | None,
) -> tuple[
    CRS, DTypeLike, int | float | None, int | float | None, float | Iterable[float] | None, rio.coords.BoundingBox
]:
    """Check all user inputs of reproject."""

    # --- Sanity checks on inputs and defaults -- #
    # Check that either ref or crs is provided
    if ref is not None and crs is not None:
        raise ValueError("Either of `ref` or `crs` must be set. Not both.")
    # If none are provided, simply preserve the CRS
    elif ref is None and crs is None:
        crs = source_raster.crs

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
            # TODO: for uint8, if all values are used, apply rio.warp to mask to identify invalid values
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

    # Create a BoundingBox if required
    if bounds is not None:
        if not isinstance(bounds, rio.coords.BoundingBox):
            bounds = rio.coords.BoundingBox(
                bounds["left"],
                bounds["bottom"],
                bounds["right"],
                bounds["top"],
            )

    # Case a raster is provided as reference
    if ref is not None:
        # Check that ref type is either str, Raster or rasterio data set
        # Preferably use Raster instance to avoid rasterio data set to remain open. See PR #45
        if isinstance(ref, gu.Raster):
            # Raise a warning if the reference is a raster that has a different pixel interpretation
            _cast_pixel_interpretation(source_raster.area_or_point, ref.area_or_point)
            ds_ref = ref
        elif isinstance(ref, str):
            if not os.path.exists(ref):
                raise ValueError("Reference raster does not exist.")
            ds_ref = gu.Raster(ref, load_data=False)
        else:
            raise TypeError("Type of ref not understood, must be path to file (str), Raster.")

        # Read reprojecting params from ref raster
        crs = ds_ref.crs
        res = ds_ref.res
        bounds = ds_ref.bounds
    else:
        # Determine target CRS
        crs = CRS.from_user_input(crs)
        res = res

    return crs, dtype, src_nodata, nodata, res, bounds


def _get_target_georeferenced_grid(
    raster: gu.Raster,
    crs: CRS | str | int | None = None,
    grid_size: tuple[int, int] | None = None,
    res: int | float | Iterable[float] | None = None,
    bounds: dict[str, float] | rio.coords.BoundingBox | None = None,
) -> tuple[affine.Affine, tuple[int, int]]:
    """
    Derive the georeferencing parameters (transform, size) for the target grid.

    Needed to reproject a raster to a different grid (resolution or size, bounds) and/or
    coordinate reference system (CRS).

    If requested bounds are incompatible with output resolution (would result in non integer number of pixels),
    the bounds are rounded up to the nearest compatible value.

    :param crs: Destination coordinate reference system as a string or EPSG. Defaults to this raster's CRS.
    :param grid_size: Destination size as (ncol, nrow). Mutually exclusive with ``res``.
    :param res: Destination resolution (pixel size) in units of destination CRS. Single value or (xres, yres).
        Mutually exclusive with ``size``.
    :param bounds: Destination bounds as a Rasterio bounding box, or a dictionary containing left, bottom,
        right, top bounds in the destination CRS.

    :returns: Calculated transform and size.
    """
    # --- Input sanity checks --- #
    # check size and res are not both set
    if (grid_size is not None) and (res is not None):
        raise ValueError("size and res both specified. Specify only one.")

    # Set CRS to input CRS by default
    if crs is None:
        crs = raster.crs

    if grid_size is None:
        width, height = None, None
    else:
        width, height = grid_size

    # Convert bounds to BoundingBox
    if bounds is not None:
        if not isinstance(bounds, rio.coords.BoundingBox):
            bounds = rio.coords.BoundingBox(
                bounds["left"],
                bounds["bottom"],
                bounds["right"],
                bounds["top"],
            )

    # If all georeferences are the same as input, skip calculating because of issue in
    # rio.warp.calculate_default_transform (https://github.com/rasterio/rasterio/issues/3010)
    if (
        (crs == raster.crs)
        & ((grid_size is None) | ((height == raster.shape[0]) & (width == raster.shape[1])))
        & ((res is None) | np.all(np.array(res) == raster.res))
        & ((bounds is None) | (bounds == raster.bounds))
    ):
        return raster.transform, raster.shape[::-1]

    # --- First, calculate default transform ignoring any change in bounds --- #
    tmp_transform, tmp_width, tmp_height = rio.warp.calculate_default_transform(
        raster.crs,
        crs,
        raster.width,
        raster.height,
        left=raster.bounds.left,
        right=raster.bounds.right,
        top=raster.bounds.top,
        bottom=raster.bounds.bottom,
        resolution=res,
        dst_width=width,
        dst_height=height,
    )

    # If no bounds specified, can directly use output of rio.warp.calculate_default_transform
    if bounds is None:
        dst_size = (tmp_width, tmp_height)
        dst_transform = tmp_transform

    # --- Second, crop to requested bounds --- #
    else:
        # If output size and bounds are known, can use rio.transform.from_bounds to get dst_transform
        if grid_size is not None:
            dst_transform = rio.transform.from_bounds(
                bounds.left, bounds.bottom, bounds.right, bounds.top, grid_size[0], grid_size[1]
            )
            dst_size = grid_size

        else:
            # Otherwise, need to calculate the new output size, rounded to nearest integer
            ref_win = rio.windows.from_bounds(*list(bounds), tmp_transform).round_lengths()
            dst_size = (int(ref_win.width), int(ref_win.height))

            if res is not None:
                # In this case, we force output resolution
                if isinstance(res, tuple):
                    dst_transform = rio.transform.from_origin(bounds.left, bounds.top, res[0], res[1])
                else:
                    dst_transform = rio.transform.from_origin(bounds.left, bounds.top, res, res)
            else:
                # In this case, we force output bounds
                dst_transform = rio.transform.from_bounds(
                    bounds.left, bounds.bottom, bounds.right, bounds.top, dst_size[0], dst_size[1]
                )

    return dst_transform, dst_size


def _get_reproj_params(
    source_raster: gu.Raster,
    crs: CRS,
    res: float | Iterable[float] | None,
    grid_size: tuple[int, int] | None,
    bounds: dict[str, float] | rio.coords.BoundingBox | None,
    dtype: DTypeLike,
    src_nodata: int | float | None,
    nodata: int | float | None,
    resampling: Resampling | str,
) -> dict[str, Any]:
    """Get all reprojection parameters."""

    # First, set basic reprojection options
    reproj_kwargs = {
        "src_transform": source_raster.transform,
        "src_crs": source_raster.crs,
        "resampling": resampling if isinstance(resampling, Resampling) else _resampling_method_from_str(resampling),
        "src_nodata": src_nodata,
        "dst_nodata": nodata,
        "dst_crs": crs,
        "dtype": dtype,
    }

    # Second, determine target transform and grid size
    transform, grid_size = _get_target_georeferenced_grid(
        source_raster, crs=crs, grid_size=grid_size, res=res, bounds=bounds
    )

    # Finally, update reprojection options accordingly
    reproj_kwargs.update({"dst_transform": transform})
    reproj_kwargs.update({"dst_shape": grid_size[::-1]})

    return reproj_kwargs


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


def _rio_reproject(
    src_arr: NDArrayNum | NDArrayBool, src_mask: NDArrayBool, reproj_kwargs: dict[str, Any]
) -> tuple[NDArrayNum | NDArrayBool, NDArrayBool]:
    """Rasterio reprojection wrapper.

    :param src_arr: Source array for data.
    :param src_mask: Source array for mask, only required if array is not float.
    :param reproj_kwargs: Reprojection parameter dictionary.
    """

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
    if reproj_kwargs["n_threads"] == 0:
        # Default to cpu count minus one. If the cpu count is undefined, num_threads will be 1
        cpu_count = os.cpu_count() or 2
        num_threads = cpu_count - 1
    else:
        num_threads = reproj_kwargs["n_threads"]

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
    # If Rasterio has old enough version, force tolerance to 0 to avoid deformations on chunks
    # See: https://github.com/rasterio/rasterio/issues/2433#issuecomment-2786157846
    if Version(rio.__version__) > Version("1.4.3"):
        reproj_kwargs.update({"tolerance": 0})

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

    return dst_arr, dst_mask
