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


class MultiprocConfig:
    """
    Configuration class for handling multiprocessing parameters in raster processing.

    This class encapsulates settings related to multiprocessing, allowing users to specify
    chunk size, depth, output file, and an optional cluster for parallel processing.
    It is designed to be passed into functions that require multiprocessing capabilities.
    """

    def __init__(self, chunk_size: int, outfile: str, depth: int = 0, cluster: AbstractCluster | None = None):
        """
        Initialize the MultiprocConfig instance with multiprocessing settings.

        :param chunk_size: The size of the chunks for splitting raster data.
        :param outfile: The file path where the output will be written.
        :param depth: The overlap size between chunks to prevent edge effects (default is 0).
        :param cluster: A cluster object for distributed computing, or None for sequential processing.
        """
        self.chunk_size = chunk_size
        self.outfile = outfile
        self.depth = depth
        if cluster is None:
            # Initialize a basic multiprocessing cluster if none is provided
            cluster = ClusterGenerator("basic")  # type: ignore
        assert isinstance(cluster, AbstractCluster)  # for mypy
        self.cluster = cluster


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


def remove_tile_padding(raster_shape: tuple[int, int], raster_tile: RasterType, tile: NDArrayNum, padding: int) -> None:
    """
    Removes the padding added around tiles during terrain attribute computation to prevent edge effects.

    :param raster_shape: The shape (height, width) of the raster from which tiles are extracted.
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
    if tile[1] != raster_shape[0]:
        tile[1] -= padding
        xmax -= padding
    if tile[2] != 0:
        tile[2] += padding
        ymin += padding
    if tile[3] != raster_shape[1]:
        tile[3] -= padding
        ymax -= padding

    # Apply the new bounding box to crop the tile and remove the padding
    raster_tile.icrop(bbox=(xmin, ymin, xmax, ymax), inplace=True)


def apply_func_block(
    func: Callable[..., Any],
    raster: RasterType,
    tile: NDArrayNum,
    depth: int,
    *args: Any,
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
    if isinstance(result_tile, Raster):
        remove_tile_padding((raster.height, raster.width), result_tile, tile, depth)

    return result_tile, tile


def map_overlap_multiproc(
    func: Callable[..., Any],
    raster_path: str | RasterType,
    config: MultiprocConfig,
    *args: Any,
    **kwargs: dict[str, Any],
) -> None:
    """
    Multiprocessing function for applying out_of_memory operations on raster.

    This function divides the input raster into tiles, processes them in parallel using multiprocessing, and
    writes the output to a new raster file. It handles the overlap between tiles (depth) to avoid edge effects.

    :param func: The function to apply to each tile.
    :param raster_path: Path to the input raster or the Raster object itself.
    :param config: Configuration object containing chunk size, output file, depth, and optional cluster.
    :param args: Additional arguments to pass to the function being applied.
    """
    # Load DEM metadata if raster_path is a filepath, otherwise use the Raster object
    if not isinstance(raster_path, Raster):
        raster = Raster(raster_path)
    else:
        raster = raster_path

    # Generate tiling grid
    tiling_grid = compute_tiling(config.chunk_size, raster.shape, raster.shape, overlap=config.depth)

    # Create tasks for multiprocessing
    tasks = []
    for row in range(tiling_grid.shape[0]):
        for col in range(tiling_grid.shape[1]):
            tile = tiling_grid[row, col]
            # Launch the task on the cluster to process each tile
            tasks.append(
                config.cluster.launch_task(
                    fun=apply_func_block, args=[func, raster, tile, config.depth, *args], **kwargs
                )
            )

    # get first tile to retrieve dtype and nodata
    attr_tile0, _ = config.cluster.get_res(tasks[0])

    if isinstance(attr_tile0, Raster):
        # Create a new raster file to save the processed results
        with rio.open(
            config.outfile,
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
                    attr_tile, dst_tile = config.cluster.get_res(results)

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
