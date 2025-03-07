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

"""
Module for multiprocessing-delayed functions for out-of-memory raster operations.
"""
from collections import abc
from typing import Literal

import numpy as np
import rasterio as rio
from rasterio import CRS
from rasterio._io import Resampling

from geoutils import Raster
from geoutils._typing import DTypeLike, NDArrayNum
from geoutils.projtools import reproject_from_latlon
from geoutils.raster import RasterType, compute_tiling
from geoutils.raster.distributed_computing.cluster import (
    AbstractCluster,
    ClusterGenerator,
)
from geoutils.raster.geotransformations import (
    _get_target_georeferenced_grid,
    _user_input_reproject,
)


def get_raster_tile(raster_unload: Raster, tile: NDArrayNum) -> Raster:
    xmin, xmax, ymin, ymax = tile
    raster_tile = raster_unload.icrop(bbox=(xmin, ymin, xmax, ymax))
    return raster_tile


# 2/ POINT INTERPOLATION ON REGULAR OR EQUAL GRID
def multiproc_interp_points_block(
    raster_unload: Raster,
    interp_coords: tuple[float, float],
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
    input_latlon: bool = False,
) -> float:
    """
    Interpolates the value of a point from a raster dataset using a specified interpolation method.
    The function optimizes memory usage by only opening the smallest possible raster window
    based on the interpolation method and point's coordinates.

    :param raster_unload: raster data source to be interpolated.
    :param interp_coords: Tuple of floats representing the coordinates (x, y) of the point to interpolate. If the
        coordinates are in latitude/longitude, set `input_latlon` to `True` to reproject them into the raster's CRS.
    :param method: Interpolation method, one of 'nearest', 'linear', 'cubic', 'quintic', 'slinear', 'pchip' or
            'splinef2d'. For more information, see scipy.ndimage.map_coordinates and scipy.interpolate.interpn.
            Default is linear.
    :param input_latlon: Whether the input is in latlon, unregarding of Raster CRS. Default is False.

    :return: A float representing the interpolated value at the given point. If the point is outside the raster's
        bounds, the function returns `np.nan`.

    :raises ValueError:
    If the interpolation method is not valid (i.e., not in the allowed methods).
    """

    # Convert coordinates if input in latlon
    if input_latlon:
        interp_coords = reproject_from_latlon(interp_coords, out_crs=raster_unload.crs)  # type: ignore

    # Check if the point is outside the raster bounds
    if raster_unload.outside_image([interp_coords[0]], [interp_coords[1]], index=False):
        return np.nan

    # Define the interpolation radius based on method
    margin = {
        "nearest": 1,
        "linear": 1,  # Linear interpolation needs immediate neighbors
        "cubic": 2,  # Cubic interpolation needs more neighbors
        "quintic": 3,  # Quintic would need even more neighbors
        "slinear": 1,
        "pchip": 1,
        "splinef2d": 2,
    }

    # Validate the interpolation method
    if method not in margin:
        raise ValueError(
            f"Invalid interpolation method '{method}'. "
            "Supported methods are: 'nearest', 'linear', 'cubic', 'quintic', 'slinear', 'pchip', 'splinef2d'."
        )
    interp_margin = margin[method] + 1  # Add margin to handle the crop

    # Calculate bounding box around the point based on the interpolation margin
    x, y = interp_coords
    pixel_size_y, pixel_size_x = raster_unload.res  # Get the raster pixel size
    xmin = x - (interp_margin * pixel_size_x)
    xmax = x + (interp_margin * pixel_size_x)
    ymin = y - (interp_margin * pixel_size_y)
    ymax = y + (interp_margin * pixel_size_y)
    bbox = [xmin, ymin, xmax, ymax]

    # Crop the raster to the bounding box (open a window of the raster)
    raster_window = raster_unload.crop(bbox)

    # Perform interpolation on the window
    interp_value = raster_window.interp_points(
        points=interp_coords, method=method, input_latlon=False  # Already converted the coords to CRS if necessary
    )

    return interp_value[0]


def multiproc_interp_points(
    raster_unload: Raster,
    points: list[tuple[float, float]],
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
    input_latlon: bool = False,
    cluster: AbstractCluster | None = None,
) -> list[float]:
    """
    Interpolate raster at point coordinates on out-of-memory chunks.

    :param raster_unload: raster data source to be interpolated.
    :param points: Point(s) at which to interpolate raster value. If points fall outside of image, value
            returned is nan. Shape should be (N,2).
    :param method: Interpolation method, one of 'nearest', 'linear', 'cubic', 'quintic', 'slinear', 'pchip' or
            'splinef2d'. For more information, see scipy.ndimage.map_coordinates and scipy.interpolate.interpn.
            Default is linear.
    :param input_latlon: Whether the input is in latlon, unregarding of Raster CRS.
    :param cluster: An `AbstractCluster` object that handles multiprocessing. This object is responsible for
            distributing the tasks across multiple processes and retrieving the results. If `None`, the function will
            execute without parallelism.

    :return: A list of interpolated raster values corresponding to the input points. If a point falls outside
            the raster bounds, the corresponding value in the output array will be `NaN`. The output array will have
            the same length as the number of input points.
    """
    if cluster is None:
        cluster = ClusterGenerator("basic")  # type: ignore
    assert cluster is not None  # for mypy

    tasks = []
    for point in points:
        tasks.append(
            cluster.launch_task(fun=multiproc_interp_points_block, args=[raster_unload, point, method, input_latlon])
        )

    # Collect results
    try:
        results = [cluster.get_res(chunk) for chunk in tasks]
    except Exception as e:
        raise RuntimeError(f"Error retrieving subsampled data from multiprocessing tasks: {e}")

    return results


# 3/ REPROJECT
def bbox_intersection(bbox1: rio.coords.BoundingBox, bbox2: rio.coords.BoundingBox) -> rio.coords.BoundingBox | None:
    """
    Compute the intersection of two bounding boxes.

    :param bbox1: The first bounding box (left, bottom, right, top).
    :param bbox2: The second bounding box (left, bottom, right, top).
    :return: The intersection as a BoundingBox object, or None if there is no intersection.
    """
    # Calculate the intersecting coordinates
    left = max(bbox1.left, bbox2.left)
    bottom = max(bbox1.bottom, bbox2.bottom)
    right = min(bbox1.right, bbox2.right)
    top = min(bbox1.top, bbox2.top)

    # Check if there is a valid intersection
    if left < right and bottom < top:
        return rio.coords.BoundingBox(left=left, bottom=bottom, right=right, top=top)
    else:
        return None  # No intersection


def snap_bounds_to_grid(bounds: rio.coords.BoundingBox, transform: rio.transform.Affine) -> rio.coords.BoundingBox:
    """
    Snap bounding box coordinates to the nearest pixel grid based on the transform.

    :param bounds: Bounds of the tile.
    :param transform: transform of the full reprojected raster.
    :return: The snapped bounds
    """
    left, top = transform * (map(round, ~transform * (bounds.left, bounds.top)))
    right, bottom = transform * map(round, ~transform * (bounds.right, bounds.bottom))

    return rio.coords.BoundingBox(left, bottom, right, top)


def bounds_no_overlap(
    bounds: rio.coords.BoundingBox, res: tuple[float, float], overlap: int, border: tuple[bool, bool, bool, bool]
) -> rio.coords.BoundingBox:
    """
    Adjusts the given bounding box to remove overlap from neighboring tiles based on the specified overlap size.
    The overlap is only removed from the sides of the tile that are not on the edge of the raster.

    :param bounds: Bounds of the tile.
    :param res: Resolution of the pixels in the tile (width pixel res, height pixel res).
    :param overlap: Size of overlap between tiles (in pixels).
    :param border: Whether the tile is on an edge of the raster (four booleans for the four edges, same order as
        rasterio bounding box : left, bottom, right, top).
    :return: A new BoundingBox with the adjusted bounds after removing overlaps on non-border sides.
    """
    left = bounds.left + overlap * res[0] * (not border[0])
    bottom = bounds.bottom + overlap * res[1] * (not border[1])
    right = bounds.right - overlap * res[0] * (not border[2])
    top = bounds.top - overlap * res[1] * (not border[3])

    return rio.coords.BoundingBox(left, bottom, right, top)


def reproject_block(
    raster_unload: Raster,
    tile: NDArrayNum,
    overlap: int,
    border: tuple[bool, bool, bool, bool],
    transform: rio.transform.Affine,
    crs: CRS | str | int | None = None,
    res: float | abc.Iterable[float] | None = None,
    bounds: rio.coords.BoundingBox | None = None,
    nodata: int | float | None = None,
    dtype: DTypeLike | None = None,
    resampling: Resampling | str = Resampling.bilinear,
    force_source_nodata: int | float | None = None,
    silent: bool = False,
) -> Raster | None:
    """
    Reproject tile to a different geotransform (resolution, bounds) and/or coordinate reference system (CRS).

    :param raster_unload: raster data source to be interpolated.
    :param tile: specific tile ([row_start, row_end, col_start, col_end]) of the raster being processed. This is a
            subset of the raster data from which interpolation values are calculated.
    :param overlap: Size of overlap between tiles (in pixels).
    :param border: Whether the tile is on an edge of the raster (four booleans for the four edges, same order as
        rasterio bounding box : left, bottom, right, top).
    :param transform: transform of the full reprojected raster.
    :param crs: Destination coordinate reference system as a string or EPSG. If ``ref`` not set,
        defaults to this raster's CRS.
    :param res: Destination resolution (pixel size) in units of destination CRS. Single value or (xres, yres).
        Do not use with ``grid_size``.
    :param bounds: Destination bounds as a Rasterio bounding box.
    :param nodata: Destination nodata value. If set to ``None``, will use the same as source. If source does
        not exist, will use GDAL's default.
    :param dtype: Destination data type of array.
    :param resampling: A Rasterio resampling method, can be passed as a string.
        See https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling
        for the full list.
    :param force_source_nodata: Force a source nodata value (read from the metadata by default).
    :param silent: Whether to print warning statements.

    :return: The reprojected tile, or None if the tile is outside the reprojection window
    """
    # load tile
    raster_tile = get_raster_tile(raster_unload, tile)

    # Remove overlap from the repojected tile bounds
    reprojected_tile_bounds = bounds_no_overlap(raster_tile.bounds, raster_tile.res, overlap, border)

    if crs != raster_tile.crs:
        # Transform the bounds to the target CRS
        reprojected_tile_bounds = rio.warp.transform_bounds(raster_tile.crs, crs, *reprojected_tile_bounds)
        reprojected_tile_bounds = rio.coords.BoundingBox(*reprojected_tile_bounds)

    # Ensure that the reprojected tile bounds fit within the overall raster bounds
    reprojected_tile_bounds = bbox_intersection(bounds, reprojected_tile_bounds)

    # If intersection is empty, the tile is outside the reprojection window.
    if reprojected_tile_bounds is None:
        return None

    # Snap the bounds to the grid to make sure they align correctly with the raster's grid
    reprojected_tile_bounds = snap_bounds_to_grid(reprojected_tile_bounds, transform)

    # Reproject the raster tile
    return raster_tile.reproject(
        ref=None,
        crs=crs,
        res=res,
        bounds=reprojected_tile_bounds,
        nodata=nodata,
        dtype=dtype,
        resampling=resampling,
        force_source_nodata=force_source_nodata,
        silent=silent,
    )


def multiproc_reproject(
    raster_unload: Raster | str,
    output_file: str,
    tile_size: int,
    overlap: int,
    ref: RasterType | str | None = None,
    crs: CRS | str | int | None = None,
    res: float | abc.Iterable[float] | None = None,
    grid_size: tuple[int, int] | None = None,
    bounds: rio.coords.BoundingBox | None = None,
    nodata: int | float | None = None,
    dtype: DTypeLike | None = None,
    resampling: Resampling | str = Resampling.bilinear,
    force_source_nodata: int | float | None = None,
    silent: bool = False,
    cluster: AbstractCluster | None = None,
) -> None:
    """
    Reproject raster to a different geotransform (resolution, bounds) and/or coordinate reference system (CRS).
    Compute tiling and use multiprocessing to reproject each tile.

    :param raster_unload: raster data source to be reprojected.
    :param output_file: Path to the full output raster file.
    :param tile_size: Size of each tile in pixels (square tiles).
    :param overlap: Size of overlap between tiles (in pixels).
    :param ref: Reference raster to match resolution, bounds and CRS.
    :param crs: Destination coordinate reference system as a string or EPSG. If ``ref`` not set,
        defaults to this raster's CRS.
    :param res: Destination resolution (pixel size) in units of destination CRS. Single value or (xres, yres).
            Do not use with ``grid_size``.
    :param grid_size: Destination grid size as (x, y). Do not use with ``res``.
    :param bounds: Destination bounds as a Rasterio bounding box.
    :param nodata: Destination nodata value. If set to ``None``, will use the same as source. If source does
        not exist, will use GDAL's default.
    :param dtype: Destination data type of array.
    :param resampling: A Rasterio resampling method, can be passed as a string.
        See https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling
        for the full list.
    :param force_source_nodata: Force a source nodata value (read from the metadata by default).
    :param silent: Whether to print warning statements.
    :param cluster: An `AbstractCluster` object that handles multiprocessing. This object is responsible for
            distributing the tasks across multiple processes and retrieving the results. If `None`, the function will
            execute without parallelism.
    """
    if cluster is None:
        cluster = ClusterGenerator("basic")  # type: ignore
    assert cluster is not None  # for mypy

    if isinstance(raster_unload, str):
        raster_unload = Raster(raster_unload)

    # Process user inputs
    crs, dtype, src_nodata, nodata, res, bounds = _user_input_reproject(
        source_raster=raster_unload,
        ref=ref,
        crs=crs,
        bounds=bounds,
        res=res,
        nodata=nodata,
        dtype=dtype,
        force_source_nodata=force_source_nodata,
    )

    # Retrieve transform and grid_size
    transform, grid_size = _get_target_georeferenced_grid(
        raster_unload, crs=crs, grid_size=grid_size, res=res, bounds=bounds
    )
    width, height = grid_size

    # Retrieve bounds
    if bounds is None:
        bounds = rio.coords.BoundingBox(
            left=transform.c,
            top=transform.f,
            right=transform.c + width * transform.a,
            bottom=transform.f + height * transform.e,
        )

    # Retrieve res
    if res is None:
        res = (abs(transform.a), abs(transform.e))

    # Compute tiling grid
    tiling_grid = compute_tiling(tile_size, raster_unload.shape, raster_unload.shape, overlap)

    # Create an empty task array for multiprocessing
    task = []

    # Open file on disk to write tile by tile
    with rio.open(
        output_file,
        "w+",
        driver="GTiff",
        height=height,
        width=width,
        count=raster_unload.count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:

        # Launch reprojection for each tile
        for row in range(tiling_grid.shape[0]):
            for col in range(tiling_grid.shape[1]):
                tile = tiling_grid[row, col]
                border = (col == 0, row == tiling_grid.shape[0] - 1, col == tiling_grid.shape[1] - 1, row == 0)
                task.append(
                    cluster.launch_task(
                        fun=reproject_block,
                        args=[
                            raster_unload,
                            tile,
                            overlap,
                            border,
                            transform,
                            crs,
                            res,
                            bounds,
                            nodata,
                            dtype,
                            resampling,
                            force_source_nodata,
                            silent,
                        ],
                    )
                )

        try:
            # Retrieve reprojection results
            for result in task:
                reprojected_tile = cluster.get_res(result)

                # if tile reprojection is None, nothing to write on disk
                if reprojected_tile is None:
                    continue

                # Calculate the position (row_offset, col_offset) of the top-left corner of this tile in the raster
                col_offset, row_offset = map(
                    round, ~transform * (reprojected_tile.bounds.left, reprojected_tile.bounds.top)
                )

                # Compute writing window
                dst_window = rio.windows.Window(
                    col_offset, row_offset, width=reprojected_tile.width, height=reprojected_tile.height
                )

                # Cast to 3D before saving if single band
                if reprojected_tile.count == 1:
                    data = reprojected_tile[np.newaxis, :, :]
                else:
                    data = reprojected_tile.data

                # Avoid overwriting already existing data in the window
                dst_data = dst.read(window=dst_window)
                data = np.where(dst_data != nodata, dst_data, data.data)

                # Write the reprojected tile to the correct location in the full raster
                dst.write(data, window=dst_window)
        except Exception as e:
            raise RuntimeError(f"Error retrieving reprojected data from multiprocessing tasks: {e}")
