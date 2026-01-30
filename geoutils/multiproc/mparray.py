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
"""Module defining array and configuration routines for chunked operations with Multiprocessing."""
from __future__ import annotations

import logging
import tempfile
import warnings

import numpy as np
from typing import Any, Callable, Literal, overload, TYPE_CHECKING

import rasterio as rio
from geoutils.multiproc.cluster import AbstractCluster, ClusterGenerator
from geoutils._typing import NDArrayNum
from geoutils._dispatch import get_geo_attr, has_geo_attr
from geoutils._misc import import_optional

if TYPE_CHECKING:
    from geoutils.raster.raster import Raster

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


def _generate_tiling_grid(
    row_min: int,
    col_min: int,
    row_max: int,
    col_max: int,
    row_split: int,
    col_split: int,
    overlap: int = 0,
) -> NDArrayNum:
    """
    Generate a grid of positions by splitting [row_min, row_max] x
    [col_min, col_max] into tiles of size row_split x col_split with optional overlap.

    :param row_min: Minimum row index of the bounding box to split.
    :param col_min: Minimum column index of the bounding box to split.
    :param row_max: Maximum row index of the bounding box to split.
    :param col_max: Maximum column index of the bounding box to split.
    :param row_split: Height of each tile.
    :param col_split: Width of each tile.
    :param overlap: size of overlapping between tiles (both vertically and horizontally).
    :return: A numpy array grid with splits in two dimensions (0: row, 1: column),
             where each cell contains [row_min, row_max, col_min, col_max].
    """
    if overlap < 0:
        raise ValueError(f"Overlap negative : {overlap}, must be positive")
    if not isinstance(overlap, int):
        raise TypeError(f"Overlap : {overlap}, must be an integer")

    # Calculate the total range of rows and columns
    col_range = col_max - col_min
    row_range = row_max - row_min

    # Calculate the number of splits considering overlap
    nb_col_split = int(np.ceil(col_range / col_split))
    nb_row_split = int(np.ceil(row_range / row_split))

    # If the leftover part after full split is smaller than or equal to the overlap, reduce the number of splits by 1
    # This ensures we do not generate unnecessary additional tiles that overlap too much.
    if 0 < col_range % col_split <= overlap:
        nb_col_split = max(nb_col_split - 1, 1)
    if 0 < row_range % row_split <= overlap:
        nb_row_split = max(nb_row_split - 1, 1)

    # Initialize the output grid
    tiling_grid = np.zeros(shape=(nb_row_split, nb_col_split, 4), dtype=int)

    for row in range(nb_row_split):
        for col in range(nb_col_split):
            # Calculate the start of the tile
            row_start = max(row_min + row * row_split - overlap, 0)
            col_start = max(col_min + col * col_split - overlap, 0)

            # Calculate the end of the tile ensuring it doesn't exceed the bounds
            row_end = min(row_max, (row + 1) * row_split + overlap)
            col_end = min(col_max, (col + 1) * col_split + overlap)

            # Populate the grid with the tile boundaries
            tiling_grid[row, col] = [row_start, row_end, col_start, col_end]

    return tiling_grid


def compute_tiling(
    tile_size: int,
    raster_shape: tuple[int, int],
    ref_shape: tuple[int, int],
    overlap: int = 0,
) -> NDArrayNum:
    """
    Compute the raster tiling grid to coregister raster by block.

    :param tile_size: Size of each tile (square tiles).
    :param raster_shape: Shape of the raster to determine tiling parameters.
    :param ref_shape: The shape of another raster to coregister, use to validate the shape.
    :param overlap: Size of overlap between tiles (optional).
    :return: tiling_grid (array of tile boundaries).

    :raises ValueError: if overlap is negative.
    :raises TypeError: if overlap is not an integer.
    """
    if raster_shape != ref_shape:
        raise Exception("Reference and secondary rasters do not have the same shape")
    row_max, col_max = raster_shape

    # Generate tiling
    tiling_grid = _generate_tiling_grid(0, 0, row_max, col_max, tile_size, tile_size, overlap=overlap)
    return tiling_grid


def plot_tiling(raster: Raster, tiling_grid: NDArrayNum) -> None:
    """
    Plot raster with its tiling.

    :param raster: The raster to plot with its tiling.
    :param tiling_grid: tiling given by compute_tiling.
    """
    mpl = import_optional("matplotlib")

    ax, caxes = raster.plot(return_axes=True)
    for tile in tiling_grid.reshape(-1, 4):
        row_min, row_max, col_min, col_max = tile
        x_min, y_min = raster.transform * (col_min, row_min)  # Bottom-left corner
        x_max, y_max = raster.transform * (col_max, row_max)  # Top-right corne
        rect = mpl.patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min, edgecolor="red", facecolor="none", linewidth=1.5
        )
        ax.add_patch(rect)

def _load_raster_tile(raster_unload: Raster, tile: NDArrayNum) -> Raster:
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


def _remove_tile_padding(raster_shape: tuple[int, int], raster_tile: Raster, tile: NDArrayNum, padding: int) -> None:
    """
    Removes the padding added around tiles during map_overlap computation to prevent edge effects.

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
    # If raster
    if has_geo_attr(result_tile, "transform"):
        _remove_tile_padding((raster.height, raster.width), result_tile, tile, depth)
    # Else
    elif isinstance(result_tile, tuple) and has_geo_attr(result_tile[0], "transform"):
        _remove_tile_padding((raster.height, raster.width), result_tile[0], tile, depth)

    # If the raster is a mask, convert to uint8 before saving and force nodata to 255
    if has_geo_attr(result_tile, "transform") and result_tile.is_mask:
        result_tile.astype("uint8", inplace=True)
        result_tile.set_nodata(255)

    return result_tile, tile


def map_overlap_multiproc_save(
    func: Callable[..., Raster],
    raster_path: str | Raster,
    config: MultiprocConfig,
    *args: Any,
    depth: int = 0,
    **kwargs: Any,
) -> Raster:
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

    # To avoid circular import, runtime here
    from geoutils.raster import Raster

    # Load DEM metadata if raster_path is a filepath, otherwise use the Raster object
    if isinstance(raster_path, str):
        raster = Raster(raster_path)
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

    raster_output = _write_multiproc_result(tasks, config, file_metadata)

    # Warns user if output file is a BigTIFF
    if raster_output._is_bigtiff():
        warnings.warn(
            "Due to the size of the output raster, it has been saved with a BigTIFF format.",
            category=UserWarning,
        )

    return raster_output


def _write_multiproc_result(
    tasks: list[Any],
    config: MultiprocConfig,
    file_metadata: dict[str, Any],
) -> Raster:

    # To avoid circular import, runtime here
    from geoutils.raster import Raster

    # Create a new raster file to save the processed results
    with rio.open(config.outfile, "w", driver=config.driver, **file_metadata, BIGTIFF="IF_NEEDED") as dst:
        try:
            # Iterate over the tasks and retrieve the processed tiles
            for results in tasks:
                result_tile, dst_tile = config.cluster.get_res(results)
                is_mask = has_geo_attr(result_tile, "is_mask") and get_geo_attr(result_tile, "is_mask")

                # Define the window in the output file where the tile should be written
                dst_window = rio.windows.Window(
                    col_off=dst_tile[2],
                    row_off=dst_tile[0],
                    width=dst_tile[3] - dst_tile[2],
                    height=dst_tile[1] - dst_tile[0],
                )

                # Cast to 3D before saving if single band
                if isinstance(result_tile, np.ndarray):
                    data = result_tile if len(result_tile.shape) > 2 else result_tile[np.newaxis, :, :]
                # If raster
                else:
                    data = result_tile.data if result_tile.count > 1 else result_tile[np.newaxis, :, :]

                # Write the processed tile to the appropriate location in the output file
                dst.write(data, window=dst_window)
            logging.info(f"Raster saved under {config.outfile}")
        except Exception as e:
            raise RuntimeError(f"Error retrieving raster tiles from multiprocessing tasks: {e}")

    if is_mask:
        return Raster(config.outfile, as_mask=True)
    return Raster(config.outfile)


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
    raster_path: str | Raster,
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
    # To avoid circular import, runtime here
    from geoutils.raster import Raster

    # Load DEM metadata if raster_path is a filepath, otherwise use the Raster object
    if isinstance(raster_path, str):
        raster = Raster(raster_path)
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
        raise RuntimeError(f"Error retrieving raster from multiprocessing tasks: {e}")
