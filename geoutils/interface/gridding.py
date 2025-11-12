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

"""Functionalities for gridding points (point cloud to raster)."""

import warnings
from typing import Literal

import affine
import geopandas as gpd
import numpy as np
import rasterio as rio
from scipy.interpolate import griddata

from geoutils._typing import NDArrayNum


def _grid_pointcloud(
    pc: gpd.GeoDataFrame,
    grid_coords: tuple[NDArrayNum, NDArrayNum] = None,
    data_column_name: str | None = None,
    resampling: Literal["nearest", "linear", "cubic"] = "linear",
    dist_nodata_pixel: float = 1.0,
) -> tuple[NDArrayNum, affine.Affine]:
    """
    Grid point cloud (possibly irregular coordinates) to raster (regular grid) using delaunay triangles interpolation.

    Based on scipy.interpolate.griddata combined to a nearest point search to replace values of grid cells further than
    a certain distance (in number of pixels) by nodata values (as griddata interpolates all values in convex hull, no
    matter the distance).

    :param pc: Point cloud.
    :param grid_coords: Regular raster grid coordinates in X and Y (i.e. equally spaced, independently for each axis).
    :param data_column_name: Name of data column for point cloud (if 2D point geometries are used).
    :param resampling: Resampling method within delauney triangles (defaults to linear).
    :param dist_nodata_pixel: Distance from the point cloud after which grid cells are filled by nodata values,
        expressed in number of pixels.
    """

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

    # 1/ Interpolate irregular point cloud on a regular grid

    # Get meshgrid coordinates
    xx, yy = np.meshgrid(grid_coords[0], grid_coords[1])

    # Use griddata on all points
    aligned_dem = griddata(
        points=(pc.geometry.x.values, pc.geometry.y.values),
        values=pc[data_column_name].values if data_column_name is not None else pc.geometry.z.values,
        xi=(xx, yy),
        method=resampling,
        rescale=True,  # Rescale inputs to unit cube to avoid precision issues
    )

    # 2/ Identify which grid points are more than X pixels away from the point cloud, and convert to NaNs
    # (otherwise all grid points in the convex hull of the irregular triangulation are filled, no matter the distance)

    # Get the nearest point for each grid point
    grid_pc = gpd.GeoDataFrame(
        data={"placeholder": np.ones(len(xx.ravel()))},
        geometry=gpd.points_from_xy(x=xx.ravel(), y=yy.ravel()),
        crs=pc.crs,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Geometry is in a geographic CRS.*")
        near = gpd.sjoin_nearest(grid_pc, pc)
        # In case there are several points at the same distance, it doesn't matter which one is used to compute the
        # distance, so we keep the first index of closest point
        index_right = near.groupby(by=near.index)["index_right"].min()

    # Compute distance between points as a function of the pixel sizes in X and Y
    res_x = np.abs(grid_coords[0][1] - grid_coords[0][0])
    res_y = np.abs(grid_coords[1][1] - grid_coords[1][0])
    dist = np.sqrt(
        ((pc.geometry.x.values[index_right] - grid_pc.geometry.x.values) / res_x) ** 2
        + ((pc.geometry.y.values[index_right] - grid_pc.geometry.y.values) / res_y) ** 2
    )

    # Replace all points further away than the distance of nodata by NaNs
    aligned_dem[dist.reshape(aligned_dem.shape) > dist_nodata_pixel] = np.nan

    # Flip Y axis of grid
    aligned_dem = np.flip(aligned_dem, axis=0)

    # 3/ Derive output transform from input grid
    transform_from_coords = rio.transform.from_origin(min(grid_coords[0]), max(grid_coords[1]), res_x, res_y)

    return aligned_dem, transform_from_coords
