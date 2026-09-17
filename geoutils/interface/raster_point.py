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

"""Exact conversions between rasters and regular point clouds, without gridding or interpolation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

import affine
import numpy as np
import rasterio as rio
from rasterio.crs import CRS

from geoutils._dispatch import get_geo_attr, has_geo_attr
from geoutils._typing import NDArrayNum
from geoutils.raster.referencing import _default_nodata, _xy2ij
from geoutils.sampling.subsampling import _subsample_raster

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterType


##################################
# 1/ REGULAR POINT CLOUD TO RASTER
##################################


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
            or len(grid_coords) != 2
            or not (isinstance(grid_coords[0], np.ndarray) and grid_coords[0].ndim == 1)
            or not (isinstance(grid_coords[1], np.ndarray) and grid_coords[1].ndim == 1)
        ):
            raise TypeError("Input grid coordinates must be 1D arrays.")
        if len(grid_coords[0]) < 2 or len(grid_coords[1]) < 2:
            raise ValueError("Grid coordinates must contain at least two values along X and Y.")

        diff_x = np.diff(grid_coords[0])
        diff_y = np.diff(grid_coords[1])

        if not np.allclose(diff_x, diff_x[0]) or not np.allclose(diff_y, diff_y[0]):
            raise ValueError("Grid coordinates must be regular (equally spaced, independently along X and Y).")
        if diff_x[0] <= 0 or diff_y[0] <= 0:
            raise ValueError("Grid coordinates must increase along X and Y.")

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
    if not np.issubdtype(i.dtype, np.integer) or not np.issubdtype(j.dtype, np.integer):
        raise ValueError("Some point cloud coordinates differ from the grid coordinates.")

    # Reject positions outside the requested grid before NumPy can wrap negative indexes around an array edge
    if np.any(i < 0) or np.any(i >= out_shape[0]) or np.any(j < 0) or np.any(j >= out_shape[1]):
        raise ValueError("Some point cloud coordinates fall outside the grid.")

    # Set values
    mask = np.ones(np.shape(arr), dtype=bool)
    mask[i, j] = False
    arr[i, j] = gdf_pc[data_column_name].values

    # Set output values
    raster_arr = np.ma.masked_array(data=arr, mask=mask)

    return raster_arr, out_transform, gdf_pc.crs, out_nodata, area_or_point


###################################
# 2/ RASTER TO REGULAR POINT CLOUD
###################################


def _raster_to_pointcloud(
    source_raster: RasterType,
    data_column_name: str = "b1",
    data_band: int = 1,
    auxiliary_data_bands: Iterable[int] | None = None,
    auxiliary_column_names: Iterable[str] | None = None,
    subsample: float | int = 1,
    skip_nodata: bool = True,
    as_array: bool = False,
    random_state: int | np.random.Generator | None = None,
    force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"] = "ul",
    mp_config: MultiprocConfig | None = None,
    force_output_to_memory: bool = False,
) -> Any:
    """
    Convert raster to a point cloud or coordinate array.

    We delegate the work to the shared raster subsampling implementation (that already contains the fast path for
    converting the whole array at once, with subsample == 1; and the subsampling logic for other cases).
    """

    return _subsample_raster(
        source_raster=source_raster,
        data_column_name=data_column_name,
        data_band=data_band,
        auxiliary_data_bands=auxiliary_data_bands,
        auxiliary_column_names=auxiliary_column_names,
        subsample=subsample,
        skip_nodata=skip_nodata,
        as_array=as_array,
        random_state=random_state,
        force_pixel_offset=force_pixel_offset,
        mp_config=mp_config,
        force_output_to_memory=force_output_to_memory,
    )
