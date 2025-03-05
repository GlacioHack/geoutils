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
from geoutils.raster import RasterType
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


# # 1/ SUBSAMPLING
# def multiproc_nb_valids(
#         raster_unload: Raster,
#         tile: NDArrayNum,
# ) -> NDArrayNum:
#     raster_tile = get_raster_tile(raster_unload, tile)
#     return _nb_valids(raster_tile.data)
#
# def multiproc_subsample_block(
#         raster_unload: Raster,
#         tile: NDArrayNum,
#         subsample_indices: NDArrayNum,
# ) -> NDArrayNum:
#     raster_tile = get_raster_tile(raster_unload, tile)
#     return _subsample_block(raster_tile.data, subsample_indices)
#
# def multiproc_subsample_indices_block(
#         raster_unload: Raster,
#         tile: NDArrayNum,
#         subsample_indices: NDArrayNum,
#         block_id: dict[str, Any],
# ) -> NDArrayNum:
#     raster_tile = get_raster_tile(raster_unload, tile)
#     return _subsample_indices_block(raster_tile.data, subsample_indices, block_id)
#
#
# def multiproc_subsample(
#         raster_unload: Raster,
#         tiling_grid: NDArrayNum,
#         subsample: int | float = 1,
#         return_indices: bool = False,
#         random_state: int | np.random.Generator | None = None,
#         silence_max_subsample: bool = False,
#         cluster: AbstractCluster | None = None,
# ) -> NDArrayNum | tuple[NDArrayNum, NDArrayNum]:
#     if cluster is None:
#         cluster = ClusterGenerator("basic")
#
#     # Get random state
#     rng = np.random.default_rng(random_state)
#
#     # Step 1: Compute number of valid points per block
#     nb_valid_per_tile = []
#     for row in range(tiling_grid.shape[0]):
#         for col in range(tiling_grid.shape[1]):
#             tile = tiling_grid[row, col]
#             nb_valid_per_tile.append(cluster.launch_task(fun=multiproc_nb_valids, args=[raster_unload, tile]))
#
#     # Retrieve results
#     try:
#         nb_valid_per_tile = [cluster.get_res(nb_valid) for nb_valid in nb_valid_per_tile]
#     except Exception as e:
#         raise RuntimeError(f"Error retrieving valid point counts from multiprocessing tasks: {e}")
#
#     total_nb_valid = np.sum(np.array(nb_valid_per_tile))
#
#     # Step 2: Calculate subsample size
#     subsample_size = _get_subsample_size_from_user_input(
#         subsample=subsample, total_nb_valids=total_nb_valid, silence_max_subsample=silence_max_subsample
#     )
#
#     # Step 3: Generate random 1D indices for subsampling
#     indices_1d = rng.choice(total_nb_valid, subsample_size, replace=False)
#
#     # Step 4: Distribute subsample indices across tiles
#     ind_per_block = _get_indices_block_per_subsample(
#         indices_1d, num_chunks=tiling_grid.shape[:2], nb_valids_per_block=nb_valid_per_tile
#     )
#
#     if return_indices:
#         # Step 5A: Return indices (2D)
#         list_subsample_indices = []
#         for row in range(tiling_grid.shape[0]):
#             for col in range(tiling_grid.shape[1]):
#                 tile = tiling_grid[row, col]
#                 block_indices = ind_per_block[row * tiling_grid.shape[1] + col]
#                 block_id = {"xstart": tile[0], "ystart": tile[2]}
#                 list_subsample_indices.append(
#                     cluster.launch_task(fun=multiproc_subsample_indices_block,
#                                         args=[raster_unload, tile, block_indices, block_id]))
#         # Retrieve results
#         try:
#             list_subsample_indices = [cluster.get_res(subsample) for subsample in list_subsample_indices]
#         except Exception as e:
#             raise RuntimeError(f"Error retrieving subsampled data from multiprocessing tasks: {e}")
#
#         return list_subsample_indices
#
#     else:
#         # Step 5B: Return subsamples
#         list_subsamples = []
#         for row in range(tiling_grid.shape[0]):
#             for col in range(tiling_grid.shape[1]):
#                 tile = tiling_grid[row, col]
#                 block_indices = ind_per_block[row * tiling_grid.shape[1] + col]
#                 list_subsamples.append(cluster.launch_task(fun=multiproc_subsample_block,
#                                                            args=[raster_unload, tile, block_indices]))
#         # Retrieve results
#         try:
#             list_subsamples = [cluster.get_res(subsample) for subsample in list_subsamples]
#         except Exception as e:
#             raise RuntimeError(f"Error retrieving subsampled data from multiprocessing tasks: {e}")
#
#         return list_subsamples
#
#
# def subsample_tile(
#         raster_unload: Raster,
#         tile: NDArrayNum,
#         subsample: int | float = 1,
#         return_indices: bool = False,
#         random_state: int | np.random.Generator | None = None,
# ):
#     raster_tile = get_raster_tile(raster_unload, tile)
#     return raster_tile.subsample(subsample, return_indices, random_state)
#
#
# def multiproc_subsample2(
#         raster_unload: Raster,
#         tiling_grid: NDArrayNum,
#         subsample: int | float = 1,
#         return_indices: bool = False,
#         random_state: int | np.random.Generator | None = None,
#         cluster: AbstractCluster | None = None,
# ) -> NDArrayNum | tuple[NDArrayNum, NDArrayNum]:
#     if cluster is None:
#         cluster = ClusterGenerator("basic")
#
#     list_subsamples = []
#     for row in range(tiling_grid.shape[0]):
#         for col in range(tiling_grid.shape[1]):
#             tile = tiling_grid[row, col]
#             list_subsamples.append(cluster.launch_task(
#                   fun=subsample_tile,
#                   args=[raster_unload, tile, subsample, return_indices, random_state]
#                   ))
#
#     try:
#         list_subsamples = [cluster.get_res(subsample) for subsample in list_subsamples]
#     except Exception as e:
#         raise RuntimeError(f"Error retrieving subsampled data from multiprocessing tasks: {e}")
#
#     return list_subsamples


# 2/ POINT INTERPOLATION ON REGULAR OR EQUAL GRID
def multiproc_interp_points_block(
    raster_unload: Raster,
    tile: NDArrayNum,
    interp_coords: NDArrayNum,
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
    input_latlon: bool = False,
) -> tuple[NDArrayNum, NDArrayNum]:
    """
    Interpolates raster values for a block of points within a given tile in parallel.

    This function is designed to process a block of interpolation coordinates and compute the corresponding
    raster values for points that fall within the image bounds of the specified tile. It operates on an out-of-memory
    tile, which is a subset of a larger raster, and applies the specified interpolation method. Only points
    that fall within the tile are processed, and their original indices are returned for re-ordering after
    multiprocessing.

    :param raster_unload: raster data source to be interpolated.
    :param tile: specific tile ([row_start, row_end, col_start, col_end]) of the raster being processed. This is a
            subset of the raster data from which interpolation values are calculated.
    :param interp_coords: A 2D numpy array of shape (2, N), where N is the number of points to interpolate.
            The first row contains the x-coordinates and the second row contains the y-coordinates of the points at
            which raster values will be interpolated.
    :param method: Interpolation method, one of 'nearest', 'linear', 'cubic', 'quintic', 'slinear', 'pchip' or
            'splinef2d'. For more information, see scipy.ndimage.map_coordinates and scipy.interpolate.interpn.
            Default is linear.
    :param input_latlon: Whether the input is in latlon, unregarding of Raster CRS.

    :return: A tuple containing:
        - `interp_points`: interpolated raster values for the valid points in the tile.
        - `valid_indices`: indices corresponding to the original points that fall within
          the bounds of the tile.
    """

    raster_tile = get_raster_tile(raster_unload, tile)

    # Filter points that are within the image bounds
    interp_block_x, interp_block_y, valid_indices = [], [], []
    for i, (x, y) in enumerate(interp_coords):
        if not raster_tile.outside_image([x], [y], index=False):
            interp_block_x.append(x)
            interp_block_y.append(y)
            valid_indices.append(i)  # Track valid point's original index

    if not valid_indices:  # No points in the tile
        return np.array([]), np.array([])

    interp_coords_block = (np.array(interp_block_x), np.array(interp_block_y))

    # interpolate points
    interp_points = raster_tile.interp_points(
        points=interp_coords_block,
        method=method,
        input_latlon=input_latlon,
    )
    return interp_points, np.array(valid_indices)


def multiproc_interp_points(
    raster_unload: Raster,
    tiling_grid: NDArrayNum,
    points: list[tuple[float, float]],
    method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
    input_latlon: bool = False,
    cluster: AbstractCluster | None = None,
) -> NDArrayNum:
    """
    Interpolate raster at point coordinates on out-of-memory chunks.

    :param raster_unload: raster data source to be interpolated.
    :param tiling_grid: A 2D numpy array representing the tiling grid that divides the raster into smaller chunks.
            Each element of the grid represents a tile ([row_start, row_end, col_start, col_end]) that will be processed
            independently.
    :param points: Point(s) at which to interpolate raster value. If points fall outside of image, value
            returned is nan. Shape should be (N,2).
    :param method: Interpolation method, one of 'nearest', 'linear', 'cubic', 'quintic', 'slinear', 'pchip' or
            'splinef2d'. For more information, see scipy.ndimage.map_coordinates and scipy.interpolate.interpn.
            Default is linear.
    :param input_latlon: Whether the input is in latlon, unregarding of Raster CRS.
    :param cluster: An `AbstractCluster` object that handles multiprocessing. This object is responsible for
            distributing the tasks across multiple processes and retrieving the results. If `None`, the function will
            execute without parallelism.

    :return: A 1D numpy array of interpolated raster values corresponding to the input points. If a point falls outside
            the raster bounds, the corresponding value in the output array will be `NaN`. The output array will have
            the same length as the number of input points.
    """
    if cluster is None:
        cluster = ClusterGenerator("basic")  # type: ignore
    assert cluster is not None  # for mypy

    points_arr = np.array(points)

    # Submit interpolation by block as tasks in parallel
    task = []
    for row in range(tiling_grid.shape[0]):
        for col in range(tiling_grid.shape[1]):
            tile = tiling_grid[row, col]
            task.append(
                cluster.launch_task(
                    fun=multiproc_interp_points_block, args=[raster_unload, tile, points_arr, method, input_latlon]
                )
            )

    # Collect results
    try:
        results = [cluster.get_res(chunk) for chunk in task]
    except Exception as e:
        raise RuntimeError(f"Error retrieving subsampled data from multiprocessing tasks: {e}")

    # Initialize arrays for gathering results
    interp_points = []
    indices = []

    # Collect valid results
    for interp_points_block, indices_block in results:
        if len(indices_block) > 0:
            interp_points.append(interp_points_block)
            indices.append(indices_block)

    if len(interp_points) == 0:  # No valid interpolated points
        return np.full(len(points), np.nan)

    # Concatenate points and indices
    interp_points = np.concatenate(interp_points)
    indices = np.concatenate(indices)

    # Sort the output array, fill NaN for missing indices
    sorted_interp_points = np.full(len(points), np.nan)
    sorted_interp_points[indices] = interp_points

    return sorted_interp_points


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
    """
    left, bottom, right, top = bounds

    # Snap the bounds to the nearest pixel edges
    left, bottom = transform * map(round, ~transform * (left, bottom))
    right, top = transform * map(round, ~transform * (right, top))

    return rio.coords.BoundingBox(left, bottom, right, top)


def reproject_block(
    raster_unload: Raster,
    tile: NDArrayNum,
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
    :param transform: transform of the reprojection.
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
    """
    raster_tile = get_raster_tile(raster_unload, tile)

    if crs == raster_tile.crs:
        tile_bounds = raster_tile.bounds
    else:
        tile_bounds = rio.warp.transform_bounds(raster_tile.crs, crs, *raster_tile.bounds)
        tile_bounds = rio.coords.BoundingBox(*tile_bounds)
    intersect_bounds = bbox_intersection(bounds, tile_bounds)

    # If intersection is empty, the tile is out of reprojection window.
    if intersect_bounds is None:
        return None

    intersect_bounds = snap_bounds_to_grid(intersect_bounds, transform)

    return raster_tile.reproject(
        ref=None,
        crs=crs,
        res=res,
        bounds=intersect_bounds,
        nodata=nodata,
        dtype=dtype,
        resampling=resampling,
        force_source_nodata=force_source_nodata,
        silent=silent,
    )


def multiproc_reproject(
    raster_unload: Raster | str,
    tiling_grid: NDArrayNum,
    output_file: str,
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
    :param tiling_grid: A 2D numpy array representing the tiling grid that divides the raster into smaller chunks.
    :param output_file: Path to the full output raster file.
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
                task.append(
                    cluster.launch_task(
                        fun=reproject_block,
                        args=[
                            raster_unload,
                            tile,
                            transform,
                            crs,
                            res,
                            bounds,
                            nodata,
                            dtype,
                            resampling,
                            src_nodata,
                            silent,
                        ],
                    )
                )

        try:
            # Retrieve reprojection results
            for chunk in task:
                reprojected_tile = cluster.get_res(chunk)

                # if tile reprojection is None, nothing to write on disk
                if reprojected_tile is None:
                    continue

                # Calculate the position (row_offset, col_offset) of the top-left corner of this tile in the full raster
                col_offset, row_offset = map(
                    int, ~transform * (reprojected_tile.bounds.left, reprojected_tile.bounds.top)
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
