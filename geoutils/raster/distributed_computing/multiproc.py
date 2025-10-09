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
from __future__ import annotations

import logging
import tempfile
from typing import Any, Callable, Literal, overload

import numpy as np
import rasterio as rio
from rasterio._io import Resampling

import geoutils as gu
from geoutils._typing import DTypeLike, NDArrayNum
from geoutils.raster.distributed_computing.chunked import (
    _build_geotiling_and_meta,
    _chunks2d_from_chunksizes_shape,
    _reproject_per_block,
)
from geoutils.raster.distributed_computing.cluster import (
    AbstractCluster,
    ClusterGenerator,
)
from geoutils.raster.tiling import compute_tiling


class MultiprocConfig:
    """
    Configuration class for handling multiprocessing parameters in raster processing.

    This class encapsulates settings related to multiprocessing, allowing users to specify
    chunk size, output file, and an optional cluster for parallel processing.
    It is designed to be passed into functions that require multiprocessing capabilities.
    """

    def __init__(
        self, chunk_size: int, outfile: str | None = None, driver: str = "GTiff", cluster: AbstractCluster | None = None
    ):
        """
        Initialize the MultiprocConfig instance with multiprocessing settings.

        :param chunk_size: The size of the chunks for splitting raster data.
        :param outfile: The file path where the output will be written.
        :param driver: Driver to write file with.
        :param cluster: A cluster object for distributed computing, or None for sequential processing.
        """
        self.chunk_size = chunk_size
        if outfile is None:
            with tempfile.NamedTemporaryFile() as tmp:
                self.outfile = tmp.name
        else:
            self.outfile = outfile
        self.driver = driver
        if cluster is None:
            # Initialize a basic multiprocessing cluster if none is provided
            cluster = ClusterGenerator("basic")  # type: ignore
        assert isinstance(cluster, AbstractCluster)  # for mypy
        self.cluster = cluster

    def copy(self) -> MultiprocConfig:
        return MultiprocConfig(chunk_size=self.chunk_size, outfile=self.outfile, cluster=self.cluster)


def _load_raster_tile(raster_unload: gu.Raster, tile: NDArrayNum) -> gu.Raster:
    """
    Extracts a specific tile (spatial subset) from the raster based on the provided tile coordinates.

    :param raster_unload: The input raster from which the tile is to be extracted.
    :param tile: The bounding box of the tile as [xmin, xmax, ymin, ymax].
    :return: The extracted raster tile.
    """
    rowmin, rowmax, colmin, colmax = tile
    # Crop the raster to extract the tile based on the bounding box coordinates
    raster_tile = raster_unload.icrop(bbox=(colmin, rowmin, colmax, rowmax))
    return raster_tile


def _remove_tile_padding(raster_shape: tuple[int, int], raster_tile: gu.Raster, tile: NDArrayNum, padding: int) -> None:
    """
    Removes the padding added around tiles during terrain attribute computation to prevent edge effects.

    :param raster_shape: The shape (height, width) of the raster from which tiles are extracted.
    :param raster_tile: The raster tile with possible padding that needs removal.
    :param tile: The bounding box of the tile as [rowmin, rowmax, colmin, colmax].
    :param padding: The padding size to be removed from each side of the tile.
    """
    # New bounding box dimensions after removing padding
    colmin, rowmin, colmax, rowmax = 0, 0, raster_tile.width, raster_tile.height

    # Remove padding only if the tile is not at the DEM's edges
    if tile[0] != 0:
        tile[0] += padding
        rowmin += padding
    if tile[1] != raster_shape[0]:
        tile[1] -= padding
        rowmax -= padding
    if tile[2] != 0:
        tile[2] += padding
        colmin += padding
    if tile[3] != raster_shape[1]:
        tile[3] -= padding
        colmax -= padding

    # Apply the new bounding box to crop the tile and remove the padding
    raster_tile.icrop(bbox=(colmin, rowmin, colmax, rowmax), inplace=True)


def _apply_func_block(
    func: Callable[..., Any],
    raster: Any,
    tile: NDArrayNum,
    depth: int,
    *args: Any,
    **kwargs: Any,
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
    raster_tile = _load_raster_tile(raster, tile)

    # Apply user-defined function to the tile
    result_tile = func(raster_tile, *args, **kwargs)

    # Remove padding
    if isinstance(result_tile, gu.Raster):
        _remove_tile_padding((raster.height, raster.width), result_tile, tile, depth)
    elif isinstance(result_tile, tuple) and isinstance(result_tile[0], gu.Raster):
        _remove_tile_padding((raster.height, raster.width), result_tile[0], tile, depth)

    # If the raster is a mask, convert to uint8 before saving and force nodata to 255
    if isinstance(result_tile, gu.Raster) and result_tile.is_mask:
        result_tile.astype("uint8", inplace=True)
        result_tile.set_nodata(255)

    return result_tile, tile


def map_overlap_multiproc_save(
    func: Callable[..., gu.Raster],
    raster_path: str | gu.Raster,
    config: MultiprocConfig,
    *args: Any,
    depth: int = 0,
    **kwargs: Any,
) -> gu.Raster:
    """
    Applies a function to raster tiles in parallel and saves the result to a file.

    This function divides the input raster into overlapping tiles, processes them in parallel using
    multiprocessing, and writes the processed results to an output raster file.

    Use this function when `func` returns a :class:`geoutils.Raster`,
    as it ensures the processed data is written to `config.outfile`.

    :param func: A function to apply to each raster tile. It must return a :class:`geoutils.Raster` object.
    :param raster_path: Path to the input raster file or an existing :class:`geoutils.Raster` object.
    :param config: Configuration object containing chunk size, output file path, and an optional cluster.
        The `outfile` parameter in `config` must be provided.
    :param args: Additional positional arguments to pass to `func`.
    :param depth: The overlap size between tiles to avoid edge effects, default is 0.
    :param kwargs: Additional keyword arguments to pass to `func`.

    :raises ValueError: If `config.outfile` is not provided.
    :raises RuntimeError: If an error occurs while processing the raster tiles.
    """
    # Load DEM metadata if raster_path is a filepath, otherwise use the Raster object
    if not isinstance(raster_path, gu.Raster):
        raster = gu.Raster(raster_path)
    else:
        raster = raster_path

    # Generate tiling grid
    tiling_grid = compute_tiling(config.chunk_size, raster.shape, raster.shape, overlap=depth)

    # Create tasks for multiprocessing
    tasks = []
    for row in range(tiling_grid.shape[0]):
        for col in range(tiling_grid.shape[1]):
            tile = tiling_grid[row, col]
            # Launch the task on the cluster to process each tile
            tasks.append(
                config.cluster.launch_task(fun=_apply_func_block, args=[func, raster, tile, depth, *args], **kwargs)
            )

    # get first tile to retrieve dtype and nodata
    result_tile0, _ = config.cluster.get_res(tasks[0])
    file_metadata = {
        "width": raster.width,
        "height": raster.height,
        "count": raster.count,
        "crs": raster.crs,
        "transform": raster.transform,
        "dtype": result_tile0.dtype,
        "nodata": result_tile0.nodata,
    }

    return _write_multiproc_result(tasks, config, file_metadata)


def _write_multiproc_result(
    tasks: list[Any],
    config: MultiprocConfig,
    file_metadata: dict[str, Any],
) -> gu.Raster:

    # Create a new raster file to save the processed results
    with rio.open(config.outfile, "w", driver=config.driver, **file_metadata) as dst:
        try:
            # Iterate over the tasks and retrieve the processed tiles
            for results in tasks:
                result_tile, dst_tile = config.cluster.get_res(results)
                is_mask = isinstance(result_tile, gu.Raster) and result_tile.is_mask

                # Define the window in the output file where the tile should be written
                dst_window = rio.windows.Window(
                    col_off=dst_tile[2],
                    row_off=dst_tile[0],
                    width=dst_tile[3] - dst_tile[2],
                    height=dst_tile[1] - dst_tile[0],
                )

                # Cast to 3D before saving if single band
                if isinstance(result_tile, gu.Raster):
                    data = result_tile.data if result_tile.count > 1 else result_tile[np.newaxis, :, :]
                else:
                    data = result_tile if len(result_tile.shape) > 2 else result_tile[np.newaxis, :, :]

                # Write the processed tile to the appropriate location in the output file
                dst.write(data, window=dst_window)
            logging.info(f"Raster saved under {config.outfile}")
        except Exception as e:
            raise RuntimeError(f"Error retrieving raster tiles from multiprocessing tasks: {e}")
    if is_mask:
        return gu.Raster(config.outfile, as_mask=True)
    return gu.Raster(config.outfile)


@overload
def map_multiproc_collect(
    func: Callable[..., Any],
    raster_path: str | Any,
    config: MultiprocConfig,
    *args: Any,
    depth: int = 0,
    return_tile: Literal[True],
    **kwargs: Any,
) -> list[tuple[Any, NDArrayNum]]: ...


@overload
def map_multiproc_collect(
    func: Callable[..., Any],
    raster_path: str | Any,
    config: MultiprocConfig,
    *args: Any,
    depth: int = 0,
    return_tile: Literal[False] = False,
    **kwargs: Any,
) -> list[Any]: ...


def map_multiproc_collect(
    func: Callable[..., Any],
    raster_path: str | gu.Raster,
    config: MultiprocConfig,
    *args: Any,
    depth: int = 0,
    return_tile: bool = False,
    **kwargs: Any,
) -> list[Any] | list[tuple[Any, NDArrayNum]]:
    """
    Applies a function to raster tiles in parallel and collects the results into a list.

    This function splits an input raster into overlapping tiles, processes them in parallel,
    and returns the results as a list. It is intended for cases where `func` does *not* return
    a :class:`geoutils.Raster`, but instead returns arbitrary values (e.g., numerical statistics, feature
    extractions, etc.).

    If `return_tile=True`, the function returns a list of tuples, where each tuple contains
    the computed result and the corresponding tile indices.

    :param func: A function to apply to each raster tile. It should return any type *except* :class:`geoutils.Raster`.
    :param raster_path: Path to the input raster file or an existing :class:`geoutils.Raster` object.
    :param config: Configuration object containing chunk size, output file path, and an optional cluster.
    :param args: Additional positional arguments to pass to `func`.
    :param depth: The overlap size between tiles to avoid edge effects.
    :param return_tile: If `True`, the output includes the tile indices in addition to the results.
    :param kwargs: Additional keyword arguments to pass to `func`.

    :returns:
        - `list[Any]` if `return_tile=False` (default).
        - `list[tuple[Any, NDArrayNum]]` if `return_tile=True`.

    :raises RuntimeError: If an error occurs while processing the raster tiles.
    """

    # Load DEM metadata if raster_path is a filepath, otherwise use the Raster object
    if not isinstance(raster_path, gu.Raster):
        raster = gu.Raster(raster_path)
    else:
        raster = raster_path

    # Generate tiling grid
    tiling_grid = compute_tiling(config.chunk_size, raster.shape, raster.shape, overlap=depth)

    # Create tasks for multiprocessing
    tasks = []
    for row in range(tiling_grid.shape[0]):
        for col in range(tiling_grid.shape[1]):
            tile = tiling_grid[row, col]
            # Launch the task on the cluster to process each tile
            tasks.append(
                config.cluster.launch_task(fun=_apply_func_block, args=[func, raster, tile, depth, *args], **kwargs)
            )

    try:
        list_results = []
        # Iterate over the tasks and retrieve the processed tiles
        for results in tasks:
            result, dst_tile = config.cluster.get_res(results)
            if return_tile:
                list_results.append((result, dst_tile))
            else:
                list_results.append(result)
        return list_results

    except Exception as e:
        raise RuntimeError(f"Error retrieving terrain attribute from multiprocessing tasks: {e}")


def _wrapper_multiproc_reproject_per_block(
    rst: gu.Raster,
    src_block_ids: list[dict[str, int]],
    dst_block_id: dict[str, int],
    idx_d2s: list[int],
    block_ids: list[dict[str, int]],
    combined_meta: dict[str, Any],
    **kwargs: Any,
) -> tuple[NDArrayNum, tuple[int, int, int, int]]:
    """Wrapper to use Delayed reprojection per destination block
    (also rebuilds a square array combined from intersecting source blocks)."""

    # Get source array block for each destination block
    s = src_block_ids
    src_arrs = (rst.icrop(bbox=(s[idx]["xs"], s[idx]["ys"], s[idx]["xe"], s[idx]["ye"])).data for idx in idx_d2s)

    # Call reproject per block
    dst_block_arr = _reproject_per_block(*src_arrs, block_ids=block_ids, combined_meta=combined_meta, **kwargs)

    return dst_block_arr, (dst_block_id["ys"], dst_block_id["ye"], dst_block_id["xs"], dst_block_id["xe"])


def _multiproc_reproject(
    rst: gu.Raster,
    config: MultiprocConfig,
    src_crs: rio.CRS,
    src_nodata: int | float | None,
    dst_shape: tuple[int, int],
    dst_transform: rio.Affine,
    dst_crs: rio.CRS,
    dst_nodata: int | float | None,
    dtype: DTypeLike,
    resampling: Resampling,
    **kwargs: Any,
) -> None:
    """
    Reproject georeferenced raster on out-of-memory chunks with multiprocessing.
    See Raster.reproject() for details.
    """

    # Prepare geotiling and reprojection metadata for source and destination grids
    src_chunks = _chunks2d_from_chunksizes_shape(chunksizes=(config.chunk_size, config.chunk_size), shape=rst.shape)
    src_geotiling, dst_geotiling, dst_chunks, dest2source, src_block_ids, meta_params, dst_block_geogrids = (
        _build_geotiling_and_meta(
            src_shape=rst.shape,
            src_transform=rst.transform,
            src_crs=rst.crs,
            dst_shape=dst_shape,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_chunks=src_chunks,
            dst_chunksizes=(config.chunk_size, config.chunk_size),
        )
    )

    # 4/ Call a delayed function that uses rio.warp to reproject the combined source block(s) to each destination block
    kwargs.update(
        {
            "src_nodata": src_nodata,
            "dst_nodata": dst_nodata,
            "resampling": resampling,
            "src_crs": src_crs,
            "dst_crs": dst_crs,
        }
    )
    # Get location of destination blocks to write file
    dst_block_ids = np.array(dst_geotiling.get_block_locations())

    # Create tasks for multiprocessing
    tasks = []
    for i in range(len(dest2source)):
        tasks.append(
            config.cluster.launch_task(
                fun=_wrapper_multiproc_reproject_per_block,
                args=[
                    rst,
                    src_block_ids,
                    dst_block_ids[i],
                    dest2source[i],
                    meta_params[i][1],
                    meta_params[i][0],
                ],
                kwargs=kwargs,
            )
        )

    # Retrieve metadata for saving file
    file_metadata = {
        "width": dst_shape[1],
        "height": dst_shape[0],
        "count": rst.count,
        "crs": dst_crs,
        "transform": dst_transform,
        "dtype": dtype,
        "nodata": dst_nodata,
    }

    # Create a new raster file to save the processed results
    _write_multiproc_result(tasks, config, file_metadata)
