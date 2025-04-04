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

from collections import abc
from typing import Any, Callable, Literal, overload

import numpy as np
import rasterio as rio
from rasterio import CRS
from rasterio._io import Resampling

import geoutils as gu
from geoutils._typing import DTypeLike, NDArrayNum
from geoutils.raster.distributed_computing.cluster import (
    AbstractCluster,
    ClusterGenerator,
)
from geoutils.raster.distributed_computing.delayed_dask import (
    ChunkedGeoGrid,
    GeoGrid,
    _chunks2d_from_chunksizes_shape,
    _combined_blocks_shape_transform,
    _reproject_per_block,
)
from geoutils.raster.geotransformations import (
    _get_target_georeferenced_grid,
    _user_input_reproject,
)
from geoutils.raster.tiling import compute_tiling


class MultiprocConfig:
    """
    Configuration class for handling multiprocessing parameters in raster processing.

    This class encapsulates settings related to multiprocessing, allowing users to specify
    chunk size, output file, and an optional cluster for parallel processing.
    It is designed to be passed into functions that require multiprocessing capabilities.
    """

    def __init__(self, chunk_size: int, outfile: str | None = None, cluster: AbstractCluster | None = None):
        """
        Initialize the MultiprocConfig instance with multiprocessing settings.

        :param chunk_size: The size of the chunks for splitting raster data.
        :param outfile: The file path where the output will be written.
        :param cluster: A cluster object for distributed computing, or None for sequential processing.
        """
        self.chunk_size = chunk_size
        self.outfile = outfile
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

    return result_tile, tile


def map_overlap_multiproc_save(
    func: Callable[..., gu.Raster],
    raster_path: str | gu.Raster,
    config: MultiprocConfig,
    *args: Any,
    depth: int = 0,
    **kwargs: Any,
) -> None:
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
        "driver": "GTIFF",
        "width": raster.width,
        "height": raster.height,
        "count": raster.count,
        "crs": raster.crs,
        "transform": raster.transform,
        "dtype": result_tile0.dtype,
        "nodata": result_tile0.nodata,
    }

    _write_multiproc_result(tasks, config, file_metadata)


def _write_multiproc_result(
    tasks: list[Any],
    config: MultiprocConfig,
    file_metadata: dict[str, Any] | None = None,
) -> None:

    # Ensure that an output file is provided
    if config.outfile is None:
        raise ValueError("Output file must be provided when the function returns a Raster.")

    # Create a new raster file to save the processed results
    if file_metadata is None:
        file_metadata = {}
    with rio.open(config.outfile, "w", **file_metadata) as dst:
        try:
            # Iterate over the tasks and retrieve the processed tiles
            for results in tasks:
                result_tile, dst_tile = config.cluster.get_res(results)

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
            print(f"Raster saved under {config.outfile}")
        except Exception as e:
            raise RuntimeError(f"Error retrieving terrain attribute from multiprocessing tasks: {e}")


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


def _multiproc_reproject_per_block(
    *src_arrs: tuple[NDArrayNum], block_ids: list[dict[str, int]], combined_meta: dict[str, Any], **kwargs: Any
) -> NDArrayNum:
    """
    Delayed reprojection per destination block (also rebuilds a square array combined from intersecting source blocks).
    """
    return _reproject_per_block(*src_arrs, block_ids=block_ids, combined_meta=combined_meta, **kwargs)


def _wrapper_multiproc_reproject_per_block(
    rst: gu.Raster,
    src_block_ids: list[dict[str, int]],
    dst_block_id: dict[str, int],
    idx_d2s: list[int],
    block_ids: list[dict[str, int]],
    combined_meta: dict[str, Any],
    **kwargs: Any,
) -> tuple[NDArrayNum, tuple[int, int, int, int]]:
    """Wrapper to use reproject_per_block for multiprocessing."""

    # Get source array block for each destination block
    s = src_block_ids
    src_arrs = (rst.icrop(bbox=(s[idx]["xs"], s[idx]["ys"], s[idx]["xe"], s[idx]["ye"])).data for idx in idx_d2s)

    # Call reproject per block
    dst_block_arr = _multiproc_reproject_per_block(
        *src_arrs, block_ids=block_ids, combined_meta=combined_meta, **kwargs
    )

    return dst_block_arr, (dst_block_id["ys"], dst_block_id["ye"], dst_block_id["xs"], dst_block_id["xe"])


def _multiproc_reproject(
    rst: gu.Raster,
    config: MultiprocConfig,
    ref: gu.Raster | str | None = None,
    crs: CRS | str | int | None = None,
    res: float | abc.Iterable[float] | None = None,
    grid_size: tuple[int, int] | None = None,
    bounds: rio.coords.BoundingBox | None = None,
    nodata: int | float | None = None,
    dtype: DTypeLike | None = None,
    resampling: Resampling | str = Resampling.bilinear,
    force_source_nodata: int | float | None = None,
    **kwargs: Any,
) -> None:
    """
    Reproject georeferenced raster on out-of-memory chunks with multiprocessing.

    :param rst: raster data source to be reprojected.
    :param config: Configuration object containing chunk size, output file path, and an optional cluster.
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
    """
    # Process user inputs
    dst_crs, dst_dtype, src_nodata, dst_nodata, dst_res, dst_bounds = _user_input_reproject(
        source_raster=rst,
        ref=ref,
        crs=crs,
        bounds=bounds,
        res=res,
        nodata=nodata,
        dtype=dtype,
        force_source_nodata=force_source_nodata,
    )

    # Retrieve transform and grid_size
    dst_transform, dst_grid_size = _get_target_georeferenced_grid(
        rst, crs=dst_crs, grid_size=grid_size, res=dst_res, bounds=dst_bounds
    )
    dst_width, dst_height = dst_grid_size
    dst_shape = (dst_height, dst_width)

    # 1/ Define source and destination chunked georeferenced grid through simple classes storing CRS/transform/shape,
    # which allow to consistently derive shape/transform for each block and their CRS-projected footprints

    # Define georeferenced grids for source/destination array
    src_geogrid = GeoGrid(transform=rst.transform, shape=rst.shape, crs=rst.crs)
    dst_geogrid = GeoGrid(transform=dst_transform, shape=dst_shape, crs=dst_crs)

    # Add the chunking
    chunks_x = tuple(
        (config.chunk_size if i <= rst.shape[0] else rst.shape[0] % config.chunk_size)
        for i in np.arange(config.chunk_size, rst.shape[0] + config.chunk_size, config.chunk_size)
    )
    chunks_y = tuple(
        (config.chunk_size if i <= rst.shape[1] else rst.shape[1] % config.chunk_size)
        for i in np.arange(config.chunk_size, rst.shape[1] + config.chunk_size, config.chunk_size)
    )
    src_chunks = (chunks_x, chunks_y)

    src_geotiling = ChunkedGeoGrid(grid=src_geogrid, chunks=src_chunks)

    # For destination, we need to create the chunks based on destination chunksizes
    dst_chunks = _chunks2d_from_chunksizes_shape(chunksizes=(config.chunk_size, config.chunk_size), shape=dst_shape)
    dst_geotiling = ChunkedGeoGrid(grid=dst_geogrid, chunks=dst_chunks)

    # 2/ Get footprints of tiles in CRS of destination array, with a buffer of 2 pixels for destination ones to ensure
    # overlap, then map indexes of source blocks that intersect a given destination block
    src_footprints = src_geotiling.get_block_footprints(crs=dst_crs)
    dst_footprints = dst_geotiling.get_block_footprints().buffer(2 * max(dst_geogrid.res))
    dest2source = [np.where(dst.intersects(src_footprints).values)[0] for dst in dst_footprints]

    # 3/ To reconstruct a square source array during chunked reprojection, we need to derive the combined shape and
    # transform of each tuples of source blocks
    src_block_ids = np.array(src_geotiling.get_block_locations())
    meta_params = [
        (
            _combined_blocks_shape_transform(sub_block_ids=src_block_ids[sbid], src_geogrid=src_geogrid)  # type: ignore
            if len(sbid) > 0
            else ({}, [])
        )
        for sbid in dest2source
    ]
    # We also add the output transform/shape for this destination chunk in the combined meta
    # (those are the only two that are chunk-specific)
    dst_block_geogrids = dst_geotiling.get_blocks_as_geogrids()
    for i, (c, _) in enumerate(meta_params):
        c.update({"dst_shape": dst_block_geogrids[i].shape, "dst_transform": tuple(dst_block_geogrids[i].transform)})

    # 4/ Call a delayed function that uses rio.warp to reproject the combined source block(s) to each destination block

    # Add fixed arguments to keywords
    kwargs.update(
        {
            "src_nodata": src_nodata,
            "dst_nodata": dst_nodata,
            "resampling": resampling,
            "src_crs": rst.crs,
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
        "driver": "GTIFF",
        "width": dst_width,
        "height": dst_height,
        "count": rst.count,
        "crs": dst_crs,
        "transform": dst_transform,
        "dtype": rst.dtype,
        "nodata": dst_nodata,
    }

    # Create a new raster file to save the processed results
    _write_multiproc_result(tasks, config, file_metadata)
