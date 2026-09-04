# Copyright (c) 2026 GeoUtils developers
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

"""Shared support handling and conversions between rasters and point clouds.

Spatial input normalization and raster support masking come first. Regular point to raster conversion follows, then
raster to point conversion completes the module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Literal

import affine
import geopandas as gpd
import numpy as np
import rasterio as rio
import xarray as xr
from rasterio.crs import CRS

from geoutils._dispatch import get_geo_attr, has_geo_attr, is_dask_array
from geoutils._typing import NDArrayNum
from geoutils.raster.array import _selected_raster_data, get_mask_from_array
from geoutils.raster.referencing import _default_nodata, _xy2ij

if TYPE_CHECKING:
    from geoutils.pointcloud.pointcloud import PointCloud, PointCloudLike
    from geoutils.raster.base import RasterType


###########################
# 1/ SHARED SPATIAL SUPPORT
###########################


def _raster_from_input(value: Any, owner: Any, name: str) -> Any:
    """Return a raster input or attach an owner's metadata to a raw array."""

    # Reuse raster objects and accessors so their georeferencing stays authoritative
    raster = value if hasattr(value, "ij2xy") else getattr(value, "rst", None)
    if raster is not None:
        return raster

    # Require a raster owner before interpreting a raw array as gridded data
    owner_raster = owner if hasattr(owner, "ij2xy") else getattr(owner, "rst", None)
    if owner_raster is None:
        raise ValueError(f"Two-dimensional value {name!r} must be tied to a raster input.")

    # Unwrap Xarray and accept the common singleton band representation
    array = value.data if isinstance(value, xr.DataArray) else value
    array = array if hasattr(array, "ndim") else np.asarray(array)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or tuple(array.shape) != tuple(owner_raster.shape):
        raise ValueError(f"Array {name!r} must match the shape of its native raster input.")

    # Attach the owner's grid so later alignment follows the raster API
    return owner_raster.from_array(
        data=array if is_dask_array(array) else np.ma.masked_invalid(array),
        transform=owner_raster.transform,
        crs=owner_raster.crs,
        nodata=owner_raster.nodata,
        area_or_point=owner_raster.area_or_point,
    )


def _aligned_raster(value: Any, owner: Any, support: Any, name: str, align: str) -> Any:
    """Return a raster aligned to raster or point support."""

    # Normalize raw arrays before comparing their owner's spatial reference
    raster = _raster_from_input(value, owner, name)
    if hasattr(support, "georeferenced_grid_equal"):
        if support.georeferenced_grid_equal(raster):
            return raster

        # Reproject grid inputs only when the caller permits spatial alignment
        if align == "reproject":
            return raster.reproject(ref=support, silent=True)
        raise ValueError(f"Raster value {name!r} does not share the selected support grid.")

    # Match a point support CRS without imposing a raster grid
    if raster.crs != support.crs:
        if align != "reproject":
            raise ValueError(f"Raster value {name!r} does not share the point support CRS.")
        raster = raster.reproject(crs=support.crs, silent=True)

    # Normalize reprojection outputs that expose the raster API through an accessor
    normalized = raster if hasattr(raster, "ij2xy") else getattr(raster, "rst", None)
    if normalized is None:
        raise TypeError(f"Raster value {name!r} could not be normalized after reprojection.")
    return normalized


def _mask_on_raster(mask: Any | None, support: Any, mask_mode: str, align: str) -> Any:
    """Evaluate a user mask on raster support."""

    # Keep every cell eligible when no additional mask was requested
    if mask is None:
        return np.ones(support.shape, dtype=bool)

    # Normalize vector inputs only after excluding raster objects and accessors
    mask_raster = mask if hasattr(mask, "ij2xy") else getattr(mask, "rst", None)
    from geoutils.vector.base import _as_vector

    vector = _as_vector(mask) if mask_raster is None else None
    if vector is not None:
        values = np.asarray(vector.create_mask(ref=support, as_array=True), dtype=bool)
        return values if mask_mode == "inside" else ~values

    # Align raster masks while accepting raw Boolean arrays on the support grid
    if mask_raster is not None:
        mask_raster = _aligned_raster(mask_raster, mask_raster, support, "mask", align)
        values = _selected_raster_data(mask_raster, fill_value=False)
    else:
        values = mask if hasattr(mask, "ndim") else np.asarray(mask)
        if np.ma.isMaskedArray(values):
            values = np.ma.asarray(values).filled(False)

    # Drop only a singleton band so one row or one column remains a spatial dimension
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]

    # Reject ambiguous numeric masks and arrays on a different grid shape
    if tuple(values.shape) != tuple(support.shape) or not np.issubdtype(values.dtype, np.bool_):
        raise ValueError("A raster support mask must be Boolean and match the support grid.")
    return values


#########################
# 2/ POINT CLOUD TO RASTER
#########################


def _regular_pointcloud_to_raster(
    pointcloud: PointCloudLike,
    grid_coords: tuple[NDArrayNum, NDArrayNum] = None,
    transform: rio.transform.Affine = None,
    shape: tuple[int, int] = None,
    nodata: int | float | None = None,
    data_column_name: str | None = "b1",
    area_or_point: Literal["Area", "Point"] = "Point",
) -> tuple[NDArrayNum, affine.Affine, CRS, int | float | None, Literal["Area", "Point"]]:
    """
    Convert a regular point cloud to a raster. See Raster.from_pointcloud_regular() for details.
    """

    # Extract geodataframe and data column name depending on input
    if has_geo_attr(pointcloud, "ds", accessors=("pc",)):
        gdf_pc = get_geo_attr(pointcloud, "ds", accessors=("pc",))
        pc_data_column_name = get_geo_attr(pointcloud, "data_column", accessors=("pc",))
        if pc_data_column_name is not None:
            data_column_name = pc_data_column_name
    else:
        gdf_pc = pointcloud

    # Get transform and shape from input
    if grid_coords is not None:

        # Input checks
        if (
            not isinstance(grid_coords, tuple)
            or not (isinstance(grid_coords[0], np.ndarray) and grid_coords[0].ndim == 1)
            or not (isinstance(grid_coords[1], np.ndarray) and grid_coords[1].ndim == 1)
        ):
            raise TypeError("Input grid coordinates must be 1D arrays.")

        diff_x = np.diff(grid_coords[0])
        diff_y = np.diff(grid_coords[1])

        if not all(diff_x == diff_x[0]) and all(diff_y == diff_y[0]):
            raise ValueError("Grid coordinates must be regular (equally spaced, independently along X and Y).")

        # Build transform from min X, max Y and step in both
        out_transform = rio.transform.from_origin(np.min(grid_coords[0]), np.max(grid_coords[1]), diff_x[0], diff_y[0])
        # Y is first axis, X is second axis
        out_shape = (len(grid_coords[1]), len(grid_coords[0]))

    elif transform is not None and shape is not None:

        out_transform = transform
        out_shape = shape

    else:
        raise ValueError("Either grid coordinates or both geotransform and shape must be provided.")

    # Create raster from inputs, with placeholder data for now
    dtype = gdf_pc[data_column_name].dtype
    out_nodata = nodata if nodata is not None else _default_nodata(dtype)
    arr = np.ones(out_shape, dtype=dtype)

    # Get indexes of point cloud coordinates in the raster, forcing no shift
    i, j = _xy2ij(
        x=gdf_pc.geometry.x.values,
        y=gdf_pc.geometry.y.values,
        shift_area_or_point=False,
        transform=out_transform,
        area_or_point=area_or_point,
    )

    # If coordinates are not integer type (forced in xy2ij), then some points are not falling on exact coordinates
    if not np.issubdtype(i.dtype, np.integer) or not np.issubdtype(i.dtype, np.integer):
        raise ValueError("Some point cloud coordinates differ from the grid coordinates.")

    # Set values
    mask = np.ones(np.shape(arr), dtype=bool)
    mask[i, j] = False
    arr[i, j] = gdf_pc[data_column_name].values

    # Set output values
    raster_arr = np.ma.masked_array(data=arr, mask=mask)

    return raster_arr, out_transform, gdf_pc.crs, out_nodata, area_or_point


#########################
# 3/ RASTER TO POINT CLOUD
#########################


def _raster_to_pointcloud(
    source_raster: RasterType,
    data_column_name: str = "b1",
    data_band: int = 1,
    auxiliary_data_bands: list[int] | None = None,
    auxiliary_column_names: list[str] | None = None,
    subsample: float | int = 1,
    skip_nodata: bool = True,
    as_array: bool = False,
    random_state: int | np.random.Generator | None = None,
    force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"] = "ul",
) -> NDArrayNum | PointCloud:
    """
    Convert a raster to a point cloud. See Raster.to_pointcloud() for details.
    """

    # Input checks

    # Main data column checks
    if not isinstance(data_column_name, str):
        raise ValueError("Data column name must be a string.")
    if not (isinstance(data_band, int) and data_band >= 1 and data_band <= source_raster.count):
        raise ValueError(
            f"Data band number must be an integer between 1 and the total number of bands ({source_raster.count})."
        )

    # Rename data column if a different band is selected but the name is still default
    if data_band != 1 and data_column_name == "b1":
        data_column_name = "b" + str(data_band)

    # Auxiliary data columns checks
    if auxiliary_column_names is not None and auxiliary_data_bands is None:
        raise ValueError("Passing auxiliary column names requires passing auxiliary data band numbers as well.")
    if auxiliary_data_bands is not None:
        if not (isinstance(auxiliary_data_bands, Iterable) and all(isinstance(b, int) for b in auxiliary_data_bands)):
            raise ValueError("Auxiliary data band number must be an iterable containing only integers.")
        if any((1 > b or source_raster.count < b) for b in auxiliary_data_bands):
            raise ValueError(
                f"Auxiliary data band numbers must be between 1 and the total number of bands ({source_raster.count})."
            )
        if data_band in auxiliary_data_bands:
            raise ValueError(
                f"Main data band {data_band} should not be listed in auxiliary data bands {auxiliary_data_bands}."
            )

        # Ensure auxiliary column name is defined if auxiliary data bands is not None
        if auxiliary_column_names is not None:
            if not (
                isinstance(auxiliary_column_names, Iterable) and all(isinstance(b, str) for b in auxiliary_column_names)
            ):
                raise ValueError("Auxiliary column names must be an iterable containing only strings.")
            if not len(auxiliary_column_names) == len(auxiliary_data_bands):
                raise ValueError(
                    f"Length of auxiliary column name and data band numbers should be the same, "
                    f"found {len(auxiliary_column_names)} and {len(auxiliary_data_bands)} respectively."
                )

        else:
            auxiliary_column_names = [f"b{i}" for i in auxiliary_data_bands]

        # Define bigger list with all bands and names
        all_bands = [data_band] + auxiliary_data_bands
        all_column_names = [data_column_name] + auxiliary_column_names

    else:
        all_bands = [data_band]
        all_column_names = [data_column_name]

    # If subsample is the entire array, load it to optimize speed
    if subsample == 1 and not source_raster.is_loaded:
        source_raster.load()

    # Band indexes in the array are band number minus one
    all_indexes = [b - 1 for b in all_bands]

    # We do 2D subsampling on the data band only, regardless of valid masks on other bands
    if skip_nodata:
        if source_raster.is_loaded or source_raster._is_xr:
            if source_raster.count == 1:
                self_mask = get_mask_from_array(
                    source_raster.data
                )  # This is to avoid the case where the mask is just "False"
            else:
                self_mask = get_mask_from_array(
                    source_raster.data[data_band - 1, :, :]
                )  # This is to avoid the case where the mask is just "False"
            valid_mask = ~self_mask

        # Load only mask of valid data from disk if array not loaded
        else:
            valid_mask = ~source_raster._load_only_mask(bands=data_band)
    # If we are not skipping nodata values, valid mask is everywhere
    else:
        if source_raster.count == 1:
            valid_mask = np.ones(source_raster.data.shape, dtype=bool)
        else:
            valid_mask = np.ones(source_raster.data[0, :].shape, dtype=bool)

    # Get subsample on valid mask
    from geoutils.sampling.subsampling import _subsample_numpy

    # Build a low memory boolean masked array with invalid values masked to pass to subsampling
    ma_valid = np.ma.masked_array(data=np.ones(np.shape(valid_mask), dtype=bool), mask=~valid_mask)
    # Take a subsample within the valid values
    indices = _subsample_numpy(array=ma_valid, subsample=subsample, random_state=random_state, return_indices=True)

    # If the Raster is loaded, pick from the data while ignoring the mask
    if source_raster.is_loaded or source_raster._is_xr:
        if source_raster.count == 1:
            pixel_data = source_raster.data[indices[0], indices[1]]
        else:
            pixel_data = source_raster.data[all_indexes, :][:, indices[0], indices[1]]

    # Otherwise use rasterio.sample to load only requested pixels
    else:
        # Extract the coordinates at subsampled pixels with valid data
        # To extract data, we always use "upper left" which rasterio interprets as the exact raster coordinates
        # Further below we redefine output coordinates based on point interpretation
        x_coords, y_coords = (np.array(a) for a in source_raster.ij2xy(indices[0], indices[1], force_offset="ul"))

        with rio.open(source_raster.name) as raster:
            # Rasterio uses indexes (starts at 1)
            pixel_data = np.array(list(raster.sample(zip(x_coords, y_coords), indexes=all_bands))).T

    # At this point there should not be any nodata anymore, so we can transform everything to normal array
    if np.ma.isMaskedArray(pixel_data):
        pixel_data = pixel_data.data

    # If nodata values were not skipped, convert them to NaNs and change data type
    if skip_nodata is False:
        pixel_data = pixel_data.astype("float32")
        pixel_data[pixel_data == source_raster.nodata] = np.nan

    # Now we force the coordinates we define for the point cloud, according to pixel interpretation
    x_coords_2, y_coords_2 = (
        np.array(a) for a in source_raster.ij2xy(indices[0], indices[1], force_offset=force_pixel_offset)
    )

    if not as_array:
        from geoutils.pointcloud import (
            PointCloud,  # Runtime import to avoid circularity issues
        )

        pc = PointCloud(
            gpd.GeoDataFrame(
                pixel_data.T,
                columns=all_column_names,
                geometry=gpd.points_from_xy(x_coords_2, y_coords_2),
                crs=source_raster.crs,
            ),
            data_column=data_column_name,
        )
        return pc
    else:
        # Merge the coordinates and pixel data an array of N x K
        # This has the downside of converting all the data to the same data type
        points_arr = np.vstack((x_coords_2.reshape(1, -1), y_coords_2.reshape(1, -1), pixel_data)).T
        return points_arr
