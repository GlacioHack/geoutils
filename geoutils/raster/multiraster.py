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

"""Multiple rasters tools."""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
import rasterio as rio
import rasterio.warp
from tqdm import tqdm

import geoutils as gu
from geoutils._typing import NDArrayNum
from geoutils.raster._geotransformations import _resampling_method_from_str
from geoutils.raster.array import get_array_and_mask
from geoutils.raster.raster import RasterType, _default_nodata


def load_multiple_rasters(
    raster_paths: list[str], crop: bool = True, ref_grid: int | None = None, **kwargs: Any
) -> list[RasterType]:
    """
    Function to load multiple rasters at once in a memory efficient way.
    First load metadata only.
    Optionally, crop all rasters to their intersection (default).
    Optionally, reproject all rasters to the grid of one raster set as reference (after optional crop).
    Otherwise, simply load the full rasters.

    :param raster_paths: List of paths to the rasters to be loaded.
    :param crop: If set to True, will only load rasters in the area they intersect.
    :param ref_grid: If set to an integer value, the raster with that index will be considered as the reference
        and all other rasters will be reprojected on the same grid (after optional crop).
    :param kwargs: Optional arguments to be passed to Raster.reproject, e.g. the resampling method.

    :returns: List of loaded Raster instances.
    """
    # If ref_grid is provided, need to reproject
    if isinstance(ref_grid, int):
        reproject = True
    # if no ref_grid provided, still need a reference CRS, use first by default
    elif ref_grid is None:
        ref_grid = 0
        reproject = False
    else:
        raise ValueError("`ref_grid` must be None or an integer")

    # Need to define a reference CRS for calculating intersection
    ref_crs = gu.Raster(raster_paths[ref_grid], load_data=False).crs

    # First load all rasters metadata
    output_rst = []
    bounds = []
    for path in raster_paths:
        # Initialize raster
        rst = gu.Raster(path, load_data=False)
        output_rst.append(rst)

        # Get bound in reference CRS
        bound = rst.get_bounds_projected(ref_crs)
        bounds.append(bound)

    # Second get the intersection of all raster bounds
    intersection = gu.projtools.merge_bounds(bounds, merging_algorithm="intersection")

    # Optionally, crop the rasters
    if crop:
        # Check that intersection is not void (changed to NaN instead of empty tuple end 2022)
        if intersection == () or all(np.isnan(i) for i in intersection):
            warnings.warn("Intersection is void, returning unloaded rasters.")
            return output_rst

        for rst in output_rst:
            # Calculate bounds in rst's CRS
            # rasterio's default for densify_pts is too low for very large images, set a default of 5000
            new_bounds = rio.warp.transform_bounds(
                ref_crs, rst.crs, intersection[0], intersection[1], intersection[2], intersection[3], densify_pts=5000
            )
            # Ensure bounds align with the original ones, to avoid resampling at this stage
            new_bounds = gu.projtools.align_bounds(rst.transform, new_bounds)
            rst.crop(new_bounds, mode="match_pixel", inplace=True)

    # Optionally, reproject all rasters to the reference grid
    if reproject:
        ref_rst = output_rst[ref_grid]

        # Set output bounds - intersection if crop is True, otherwise use that of ref_grid
        if crop:
            # make sure new bounds align with reference's bounds (to avoid resampling ref)
            new_bounds = intersection
            new_bounds = gu.projtools.align_bounds(ref_rst.transform, intersection)
            new_bounds = {"left": new_bounds[0], "bottom": new_bounds[1], "right": new_bounds[2], "top": new_bounds[3]}
        else:
            new_bounds = ref_rst.bounds

        # Reproject all rasters
        for index, rst in enumerate(output_rst):
            out_rst = rst.reproject(crs=ref_rst.crs, bounds=new_bounds, res=ref_rst.res, silent=True, **kwargs)
            if not out_rst.is_loaded:
                out_rst.load()
            output_rst[index] = out_rst

    # if no crop or reproject option, simply load the rasters
    if (not crop) & (not reproject):
        for rst in output_rst:
            rst.load()

    return output_rst


def stack_rasters(
    rasters: list[RasterType],
    reference: int | gu.Raster = 0,
    resampling_method: str | rio.enums.Resampling = "bilinear",
    use_ref_bounds: bool = False,
    diff: bool = False,
    progress: bool = True,
) -> gu.Raster:
    """
    Stack a list of rasters on their maximum extent into a multi-band raster.

    The input rasters can have any transform or CRS, and will be reprojected to the
    reference raster's CRS and resolution.
    The output multi-band raster has an extent that is the union of all raster extents,
    except if `use_ref_bounds` is used,
    and the number of band equal to the number of input rasters.

    Use diff=True to return directly the difference to the reference raster.

    Note that all rasters will be loaded once in memory. The data is only loaded for
    reprojection then deleted to optimize memory usage.

    :param rasters: List of rasters to be stacked.
    :param reference: Index of reference raster in the list or separate reference raster.
        Defaults to the first raster in the list.
    :param resampling_method: Resampling method for reprojection.
    :param use_ref_bounds: If True, will use reference bounds, otherwise will use maximum bounds of all rasters.
    :param diff: If True, will return the difference to the reference raster.
    :param progress: If True, will display a progress bar. Default is True.

    :returns: The merged raster with same CRS and resolution (and optionally bounds) as the reference.
    """
    # Check resampling method
    if isinstance(resampling_method, str):
        resampling_method = _resampling_method_from_str(resampling_method)

    # Check raster has a single band
    if any(r.count > 1 for r in rasters):
        warnings.warn("Some input Rasters have multiple bands, only their first band will be used.")

    # Select reference raster
    if isinstance(reference, int):
        reference_raster = rasters[reference]
    elif isinstance(reference, gu.Raster):
        reference_raster = reference
    else:
        raise ValueError("reference should be either an integer or geoutils.Raster object")

    # Set output bounds
    if use_ref_bounds:
        dst_bounds = reference_raster.bounds
    else:
        dst_bounds = gu.projtools.merge_bounds(
            [raster.get_bounds_projected(out_crs=reference_raster.crs) for raster in rasters],
            resolution=reference_raster.res[0],
            return_rio_bbox=True,
        )

    # Make a data list and add all the reprojected rasters into it.
    data: list[NDArrayNum] = []

    for raster in tqdm(rasters, disable=not progress):
        # Check that data is loaded, otherwise temporarily load it
        if not raster.is_loaded:
            raster.load()

        nodata = (
            reference_raster.nodata
            if reference_raster.nodata is not None
            else gu.raster.raster._default_nodata(reference_raster.data.dtype)
        )
        # Reproject to reference grid
        reprojected_raster = raster.reproject(
            bounds=dst_bounds,
            res=reference_raster.res,
            crs=reference_raster.crs,
            dtype=reference_raster.data.dtype,
            nodata=nodata,
            resampling=resampling_method,
            silent=True,
        )
        # If the georeferenced grid was the same, reproject() will have returned self with a warning (silenced here),
        # and we want to copy the raster and just modify its nodata (or would modify raster inputs of this function)
        if reprojected_raster.georeferenced_grid_equal(raster):
            reprojected_raster = reprojected_raster.copy()
            reprojected_raster.set_nodata(nodata)

        # Optionally calculate difference
        if diff:
            diff_to_ref = (reference_raster.data - reprojected_raster.data).squeeze()
            diff_to_ref, _ = get_array_and_mask(diff_to_ref)
            data.append(diff_to_ref)
        else:
            # img_data, _ = get_array_and_mask(reprojected_raster.data.squeeze())
            # Use only first band
            if reprojected_raster.count == 1:
                data.append(reprojected_raster.data[:])
            else:
                data.append(reprojected_raster.data[0, :])

        # Remove unloaded rasters
        if not raster.is_loaded:
            raster._data = None

    # Convert to masked array
    data = np.ma.asarray(data)
    if reference_raster.nodata is not None:
        nodata = reference_raster.nodata
    else:
        nodata = _default_nodata(data.dtype)
    data[np.isnan(data)] = nodata  # type: ignore

    # Save as gu.Raster - needed as some child classes may not accept multiple bands
    r = gu.Raster.from_array(
        data=data,
        transform=rio.transform.from_bounds(*dst_bounds, width=data[0].shape[1], height=data[0].shape[0]),
        crs=reference_raster.crs,
        nodata=nodata,
    )

    return r


def merge_rasters(
    rasters: list[RasterType],
    reference: int | RasterType = 0,
    merge_algorithm: Callable | list[Callable] = np.nanmean,  # type: ignore
    resampling_method: str | rio.enums.Resampling = "bilinear",
    use_ref_bounds: bool = False,
    progress: bool = True,
) -> RasterType:
    """
    Spatially merge a list of rasters into one larger raster of their maximum extent.

    The input rasters can have any transform or CRS, and will be reprojected to the
    reference raster's CRS and resolution.
    The output merged raster has an extent that is the union of all raster extents,
    except if `use_ref_bounds` is used.

    Note that all rasters will be loaded once in memory. The data is only loaded for
    reprojection then deleted to optimize memory usage.

    :param rasters: List of rasters to be merged.
    :param reference: Index of reference raster in the list or separate reference raster.
        Defaults to the first raster in the list.
    :param merge_algorithm: Reductor function (or list of functions) to merge the rasters with. Defaults to the mean.
        If several algorithms are provided, each result is returned as a separate band.
    :param resampling_method: Resampling method for reprojection.
    :param use_ref_bounds: If True, will use reference bounds, otherwise will use maximum bounds of all rasters.
    :param progress: If True, will display a progress bar. Default is True.

    :returns: The merged raster with same CRS and resolution (and optionally bounds) as the reference.
    """
    # Make sure merge_algorithm is a list
    if not isinstance(merge_algorithm, (list, tuple)):
        merge_algorithm = [
            merge_algorithm,
        ]

    # Try to run the merge_algorithm with an arbitrary list. Raise an error if the algorithm is incompatible.
    for algo in merge_algorithm:
        try:
            algo([1, 2])
        except TypeError as exception:
            raise TypeError(f"merge_algorithm must be able to take a list as its first argument.\n\n{exception}")

    # Select reference raster
    if isinstance(reference, int):
        reference_raster = rasters[reference]
    elif isinstance(reference, gu.Raster):
        reference_raster = reference
    else:
        raise ValueError("reference should be either an integer or geoutils.Raster object")

    # Reproject and stack all rasters
    raster_stack = stack_rasters(
        rasters,
        reference=reference,
        resampling_method=resampling_method,
        use_ref_bounds=use_ref_bounds,
        progress=progress,
    )

    # Try to use the keyword axis=0 for the merging algorithm (if it's a numpy ufunc).
    merged_data = []
    for algo in merge_algorithm:
        try:
            merged_data.append(algo(raster_stack.data, axis=0))
        # If that doesn't work, use the slower np.apply_along_axis approach.
        except TypeError as exception:
            if not (
                "'axis' is an invalid keyword" in str(exception)
                or "got an unexpected keyword argument 'axis'" in str(exception)
            ):
                raise exception
            merged_data.append(np.apply_along_axis(algo, axis=0, arr=raster_stack.data))

    # Convert to masked array, and set all Nans to nodata
    merged_data = np.ma.asarray(merged_data)
    if reference_raster.nodata is not None:
        nodata = reference_raster.nodata
    else:
        nodata = _default_nodata(merged_data.dtype)
    merged_data[np.isnan(merged_data)] = nodata

    # Save as gu.Raster
    merged_raster = reference_raster.from_array(
        data=np.reshape(merged_data, (len(merged_data),) + merged_data[0].shape),
        transform=rio.transform.from_bounds(
            *raster_stack.bounds, width=merged_data[0].shape[1], height=merged_data[0].shape[0]
        ),
        crs=reference_raster.crs,
        nodata=nodata,
    )

    return merged_raster
