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
Functionalities for geotransformations of raster objects.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Literal

import affine
import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.enums import Resampling

import geoutils as gu
from geoutils import profiler
from geoutils._typing import DTypeLike, MArrayNum
from geoutils.raster._geotransformations import (
    _get_reproj_params,
    _is_reproj_needed,
    _rio_reproject,
    _user_input_reproject,
)
from geoutils.raster.distributed_computing.multiproc import _multiproc_reproject
from geoutils.raster.georeferencing import _cast_pixel_interpretation

##############
# 1/ REPROJECT
##############


@profiler.profile("geoutils.raster.geotransformations._reproject", memprof=True)  # type: ignore
def _reproject(
    source_raster: gu.Raster,
    ref: gu.Raster,
    crs: CRS | str | int | None = None,
    res: float | Iterable[float] | None = None,
    grid_size: tuple[int, int] | None = None,
    bounds: dict[str, float] | rio.coords.BoundingBox | None = None,
    nodata: int | float | None = None,
    dtype: DTypeLike | None = None,
    resampling: Resampling | str = Resampling.bilinear,
    force_source_nodata: int | float | None = None,
    silent: bool = False,
    n_threads: int = 0,
    memory_limit: int = 64,
    multiproc_config: gu.raster.MultiprocConfig | None = None,
) -> tuple[bool, MArrayNum | None, affine.Affine | None, CRS | None, int | float | None]:
    """
    Reproject raster. See Raster.reproject() for details.
    """

    # 1/ Process user input
    crs, dtype, src_nodata, nodata, res, bounds = _user_input_reproject(
        source_raster=source_raster,
        ref=ref,
        crs=crs,
        bounds=bounds,
        res=res,
        nodata=nodata,
        dtype=dtype,
        force_source_nodata=force_source_nodata,
    )

    # 2/ Derive georeferencing parameters for reprojection (transform, grid size)
    reproj_kwargs = _get_reproj_params(
        source_raster=source_raster,
        crs=crs,
        res=res,
        grid_size=grid_size,
        bounds=bounds,
        dtype=dtype,
        src_nodata=src_nodata,
        nodata=nodata,
        resampling=resampling,
    )

    # 3/ Check if reprojection is needed, otherwise return source raster with warning
    if _is_reproj_needed(src_shape=source_raster.shape, reproj_kwargs=reproj_kwargs):
        if (nodata == src_nodata) or (nodata is None):
            if not silent:
                warnings.warn("Output projection, bounds and grid size are identical -> returning self (not a copy!)")
            return True, None, None, None, None

        elif nodata is not None:
            if not silent:
                warnings.warn(
                    "Only nodata is different, consider using the 'set_nodata()' method instead'\
                ' -> returning self (not a copy!)"
                )
            return True, None, None, None, None

    # 4/ Check reprojection is possible (boolean raster will be converted, so no need to check)
    if np.dtype(source_raster.dtype) != bool and (src_nodata is None and np.sum(source_raster.data.mask) > 0):
        raise ValueError(
            "No nodata set, set one for the raster with self.set_nodata() or use a temporary one "
            "with `force_source_nodata`."
        )

    # 5/ Perform reprojection
    reproj_kwargs.update({"n_threads": n_threads, "warp_mem_limit": memory_limit})
    if multiproc_config is not None:
        _multiproc_reproject(source_raster, config=multiproc_config, **reproj_kwargs)
        return False, None, None, None, None

    else:
        # All masked values must be set to a nodata value for rasterio's reproject to work properly
        src_arr = source_raster.data.data
        src_mask = np.ma.getmaskarray(source_raster.data)
        dst_arr, dst_mask = _rio_reproject(src_arr=src_arr, src_mask=src_mask, reproj_kwargs=reproj_kwargs)
        # Set mask
        dst_arr = np.ma.masked_array(data=dst_arr, mask=dst_mask, fill_value=nodata)

        return False, dst_arr, reproj_kwargs["dst_transform"], reproj_kwargs["dst_crs"], reproj_kwargs["dst_nodata"]


#########
# 2/ CROP
#########


@profiler.profile("geoutils.raster.geotransformations._crop", memprof=True)  # type: ignore
def _crop(
    source_raster: gu.Raster,
    bbox: gu.Raster | gu.Vector | list[float] | tuple[float, ...],
    mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
    distance_unit: Literal["georeferenced", "pixel"] = "georeferenced",
) -> tuple[MArrayNum, affine.Affine]:
    """Crop raster. See details in Raster.crop()."""

    assert mode in ["match_extent", "match_pixel"], "mode must be one of 'match_pixel', 'match_extent'"
    assert distance_unit in ["georeferenced", "pixel"], "distance_unit must be 'georeferenced' or 'pixel'"

    if isinstance(bbox, (gu.Raster, gu.Vector)):
        # For another Vector or Raster, we reproject the bounding box in the same CRS as self
        xmin, ymin, xmax, ymax = bbox.get_bounds_projected(out_crs=source_raster.crs)
        if isinstance(bbox, gu.Raster):
            # Raise a warning if the reference is a raster that has a different pixel interpretation
            _cast_pixel_interpretation(source_raster.area_or_point, bbox.area_or_point)
    elif isinstance(bbox, (list, tuple)):
        if distance_unit == "georeferenced":
            xmin, ymin, xmax, ymax = bbox
        else:
            colmin, rowmin, colmax, rowmax = bbox
            xmin, ymax = rio.transform.xy(source_raster.transform, rowmin, colmin, offset="ul")
            xmax, ymin = rio.transform.xy(source_raster.transform, rowmax, colmax, offset="ul")
    else:
        raise ValueError("cropGeom must be a Raster, Vector, or list of coordinates.")

    if mode == "match_pixel":
        # Finding the intersection of requested bounds and original bounds, cropped to image shape
        ref_win = rio.windows.from_bounds(xmin, ymin, xmax, ymax, transform=source_raster.transform)
        self_win = rio.windows.from_bounds(*source_raster.bounds, transform=source_raster.transform).crop(
            *source_raster.shape
        )
        final_window = ref_win.intersection(self_win).round_lengths().round_offsets()

        # Update bounds and transform accordingly
        new_xmin, new_ymin, new_xmax, new_ymax = rio.windows.bounds(final_window, transform=source_raster.transform)
        tfm = rio.transform.from_origin(new_xmin, new_ymax, *source_raster.res)

        if source_raster.is_loaded:
            # In case data is loaded on disk, can extract directly from np array
            (rowmin, rowmax), (colmin, colmax) = final_window.toranges()
            crop_img = source_raster.data[..., rowmin:rowmax, colmin:colmax]

        else:

            assert source_raster._disk_shape is not None  # This should not be the case, sanity check to make mypy happy

            # If data was not loaded, and self's transform was updated (e.g. due to downsampling) need to
            # get the Window corresponding to on disk data
            ref_win_disk = rio.windows.from_bounds(
                new_xmin, new_ymin, new_xmax, new_ymax, transform=source_raster._disk_transform
            )
            self_win_disk = rio.windows.from_bounds(
                *source_raster.bounds, transform=source_raster._disk_transform
            ).crop(*source_raster._disk_shape[1:])
            final_window_disk = ref_win_disk.intersection(self_win_disk).round_lengths().round_offsets()

            # Round up to downsampling size, to match __init__
            final_window_disk = rio.windows.round_window_to_full_blocks(
                final_window_disk, ((source_raster._downsample, source_raster._downsample),)
            )

            # Load data for "on_disk" window but out_shape matching in-memory transform -> enforce downsampling
            # AD (24/04/24): Note that the same issue as #447 occurs here when final_window_disk extends beyond
            # self's bounds. Using option `boundless=True` solves the issue but causes other tests to fail
            # This should be fixed with #447 and previous line would be obsolete.
            with rio.open(source_raster.filename) as raster:
                crop_img = raster.read(
                    indexes=source_raster._bands,
                    masked=source_raster._masked,
                    window=final_window_disk,
                    out_shape=(final_window.height, final_window.width),
                )

            # Squeeze first axis for single-band
            if crop_img.ndim == 3 and crop_img.shape[0] == 1:
                crop_img = crop_img.squeeze(axis=0)

    else:
        bbox = rio.coords.BoundingBox(left=xmin, bottom=ymin, right=xmax, top=ymax)
        out_rst = source_raster.reproject(bounds=bbox)  # should we instead raise an issue and point to reproject?
        crop_img = out_rst.data
        tfm = out_rst.transform

    return crop_img, tfm


##############
# 3/ TRANSLATE
##############


@profiler.profile("geoutils.raster.geotransformations._translate", memprof=True)  # type: ignore
def _translate(
    transform: affine.Affine,
    xoff: float,
    yoff: float,
    distance_unit: Literal["georeferenced", "pixel"] = "georeferenced",
) -> affine.Affine:
    """
    Translate geotransform horizontally, either in pixels or georeferenced units.

    :param transform: Input geotransform.
    :param xoff: Translation x offset.
    :param yoff: Translation y offset.
    :param distance_unit: Distance unit, either 'georeferenced' (default) or 'pixel'.

    :return: Translated transform.
    """

    if distance_unit not in ["georeferenced", "pixel"]:
        raise ValueError("Argument 'distance_unit' should be either 'pixel' or 'georeferenced'.")

    # Get transform
    dx, b, xmin, d, dy, ymax = list(transform)[:6]

    # Convert pixel offsets to georeferenced units
    if distance_unit == "pixel":
        xoff *= dx
        yoff *= dy

    return rio.transform.Affine(dx, b, xmin + xoff, d, dy, ymax + yoff)
