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

"""Process out-of-memory calculations"""
from typing import Any, Callable

import numpy as np
import rasterio as rio

from geoutils import Raster
from geoutils._typing import NDArrayNum
from geoutils.raster import RasterType, compute_tiling
from geoutils.raster.distributed_computing.cluster import (
    AbstractCluster,
    ClusterGenerator,
)


def load_raster_tile(raster_unload: RasterType, tile: NDArrayNum) -> RasterType:
    """
    Extracts a specific tile (spatial subset) from the raster based on the provided tile coordinates.

    :param raster_unload: The input raster from which the tile is to be extracted.
    :param tile: The bounding box of the tile as [xmin, xmax, ymin, ymax].
    :return: The extracted raster tile.
    """
    xmin, xmax, ymin, ymax = tile
    # Crop the raster to extract the tile based on the bounding box coordinates
    raster_tile = raster_unload.icrop(bbox=(xmin, ymin, xmax, ymax))
    return raster_tile


def remove_tile_padding(raster: RasterType, raster_tile: RasterType, tile: NDArrayNum, padding: int) -> None:
    """
    Removes the padding added around tiles during terrain attribute computation to prevent edge effects.

    :param raster: The full DEM object from which tiles are extracted.
    :param raster_tile: The raster tile with possible padding that needs removal.
    :param tile: The bounding box of the tile as [xmin, xmax, ymin, ymax].
    :param padding: The padding size to be removed from each side of the tile.
    """
    # New bounding box dimensions after removing padding
    xmin, xmax, ymin, ymax = 0, raster_tile.height, 0, raster_tile.width

    # Remove padding only if the tile is not at the DEM's edges
    if tile[0] != 0:
        tile[0] += padding
        xmin += padding
    if tile[1] != raster.height:
        tile[1] -= padding
        xmax -= padding
    if tile[2] != 0:
        tile[2] += padding
        ymin += padding
    if tile[3] != raster.width:
        tile[3] -= padding
        ymax -= padding

    # Apply the new bounding box to crop the tile and remove the padding
    raster_tile.icrop(bbox=(xmin, ymin, xmax, ymax), inplace=True)


def map_block(
    func: Callable[..., Any],
    raster: RasterType,
    tile: NDArrayNum,
    depth: int,
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> tuple[Any, NDArrayNum]:
    """
    Apply a function to a specific tile of a raster, handling loading and padding.

    :param func: The function to apply to each tile.
    :param raster: The input raster.
    :param tile: The bounding box of the tile as [xmin, xmax, ymin, ymax].
    :param depth: The padding size used to overlap tiles.
    :param args: Additional arguments to pass to the function being applied.

    :return: The processed tile and its bounding box.
    """
    # Load raster tile
    raster_tile = load_raster_tile(raster, tile)

    # Apply user-defined function to the tile
    result_tile = func(raster_tile, *args, **kwargs)

    # Remove padding
    remove_tile_padding(raster, result_tile, tile, depth)

    return result_tile, tile


def map_multiproc(
    func: Callable[..., Any],
    raster_path: str | RasterType,
    tile_size: int,
    outfile: str,
    *args: tuple[Any, ...],
    depth: int = 0,
    cluster: AbstractCluster | None = None,
    **kwargs: dict[str, Any],
) -> None:
    """
    Multiprocessing function for applying out_of_memory operations on raster.

    This function divides the input raster into tiles, processes them in parallel using multiprocessing, and
    writes the output to a new raster file. It handles the overlap between tiles (depth) to avoid edge effects.

    :param func: The function to apply to each tile.
    :param raster_path: Path to the input raster or the Raster object itself.
    :param tile_size: The size of each tile to divide the raster into.
    :param outfile: The path where the resulting output raster will be saved.
    :param args: Additional arguments to pass to the function being applied.
    :param depth: The padding size for overlapping tiles (default is 0).
    :param cluster: An optional multiprocessing cluster to use for task distribution.
    """
    if cluster is None:
        # Initialize a basic multiprocessing cluster if none is provided
        cluster = ClusterGenerator("basic")  # type: ignore
    assert cluster is not None  # for mypy

    # Load DEM metadata if raster_path is a filepath, otherwise use the Raster object
    if not isinstance(raster_path, Raster):
        raster = Raster(raster_path)
    else:
        raster = raster_path

    # Generate tiling grid
    tiling_grid = compute_tiling(tile_size, raster.shape, raster.shape, overlap=depth)

    # Create tasks for multiprocessing
    tasks = []
    for row in range(tiling_grid.shape[0]):
        for col in range(tiling_grid.shape[1]):
            tile = tiling_grid[row, col]
            # Launch the task on the cluster to process each tile
            tasks.append(cluster.launch_task(fun=map_block, args=[func, raster, tile, depth, *args], **kwargs))

    # get first tile to retrieve dtype and nodata
    attr_tile0, _ = cluster.get_res(tasks[0])

    if isinstance(attr_tile0, Raster):
        # Create a new raster file to save the processed results
        with rio.open(
            outfile,
            "w",
            driver="GTiff",
            height=raster.height,
            width=raster.width,
            count=raster.count,
            dtype=attr_tile0.dtype,
            crs=raster.crs,
            transform=raster.transform,
            nodata=attr_tile0.nodata,
        ) as dst:
            try:
                # Iterate over the tasks and retrieve the processed tiles
                for results in tasks:
                    attr_tile, dst_tile = cluster.get_res(results)

                    # Define the window in the output file where the tile should be written
                    dst_window = rio.windows.Window(
                        col_off=dst_tile[2],
                        row_off=dst_tile[0],
                        width=dst_tile[3] - dst_tile[2],
                        height=dst_tile[1] - dst_tile[0],
                    )

                    # Cast to 3D before saving if single band
                    if attr_tile.count == 1:
                        data = attr_tile[np.newaxis, :, :]
                    else:
                        data = attr_tile.data

                    # Write the processed tile to the appropriate location in the output file
                    dst.write(data, window=dst_window)
            except Exception as e:
                raise RuntimeError(f"Error retrieving terrain attribute from multiprocessing tasks: {e}")
