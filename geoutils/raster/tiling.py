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

"""Tiling tools for arrays and rasters."""
from __future__ import annotations

import math
import sys

import numpy as np
from matplotlib.patches import Rectangle

import geoutils as gu
from geoutils._typing import NDArrayNum


def _get_closest_rectangle(size: int) -> tuple[int, int]:
    """
    Given a 1D array size, return a rectangular shape that is closest to a cube which the size fits in.

    If 'size' does not have an integer root, a rectangle is returned that is slightly larger than 'size'.

    :examples:
        >>> _get_closest_rectangle(4)  # size will be 4
        (2, 2)
        >>> _get_closest_rectangle(9)  # size will be 9
        (3, 3)
        >>> _get_closest_rectangle(3)  # size will be 4; needs padding afterward.
        (2, 2)
        >>> _get_closest_rectangle(55) # size will be 56; needs padding afterward.
        (7, 8)
        >>> _get_closest_rectangle(24)  # size will be 25; needs padding afterward
        (5, 5)
        >>> _get_closest_rectangle(85620)  # size will be 85849; needs padding afterward
        (293, 293)
        >>> _get_closest_rectangle(52011)  # size will be 52212; needs padding afterward
        (228, 229)
    """
    close_cube = int(np.sqrt(size))

    # If size has an integer root, return the respective cube.
    if close_cube**2 == size:
        return close_cube, close_cube

    # One of these rectangles/cubes will cover all cells, so return the first that does.
    potential_rectangles = [(close_cube, close_cube + 1), (close_cube + 1, close_cube + 1)]

    for rectangle in potential_rectangles:
        if np.prod(rectangle) >= size:
            return rectangle

    # Default return, should never reach here as one of the potential_rectangles will cover all cells.
    return potential_rectangles[-1]


def subdivide_array(shape: tuple[int, ...], count: int) -> NDArrayNum:
    """
    Create indices for subdivison of an array in a number of blocks.

    If 'count' is divisible by the product of 'shape', the amount of cells in each block will be equal.
    If 'count' is not divisible, the amount of cells in each block will be very close to equal.

    :param shape: The shape of a array to be subdivided.
    :param count: The amount of subdivisions to make.

    :examples:
        >>> subdivide_array((4, 4), 4)  # doctest: +SKIP
        array([[0, 0, 1, 1],
               [0, 0, 1, 1],
               [2, 2, 3, 3],
               [2, 2, 3, 3]])

        >>> subdivide_array((6, 4), 4)  # doctest: +SKIP
        array([[0, 0, 1, 1],
               [0, 0, 1, 1],
               [0, 0, 1, 1],
               [2, 2, 3, 3],
               [2, 2, 3, 3],
               [2, 2, 3, 3]])

        >>> subdivide_array((5, 4), 3)  # doctest: +SKIP
        array([[0, 0, 0, 0],
               [0, 0, 0, 0],
               [1, 1, 2, 2],
               [1, 1, 2, 2],
               [1, 1, 2, 2]])

    :raises ValueError: If the 'shape' size (`np.prod(shape)`) is smallern than 'count'
                        If the shape is not a 2D shape.

    :returns: An array of shape 'shape' with 'count' unique indices.
    """
    try:
        import skimage.transform
    except ImportError:
        raise ImportError("Missing optional dependency, skimage.transform, required by this function.")

    # Check if system is 64bit or 32bit and catch potential numpy overflow because MSVC `long` is int32_t
    if sys.maxsize > 2**32:
        size = np.prod(shape, dtype=np.int64)
    else:
        size = np.prod(shape)

    if count > size:
        raise ValueError(f"Shape '{shape}' size ({size}) is smaller than 'count' ({count}).")

    if len(shape) != 2:
        raise ValueError(f"Expected a 2D shape, got {len(shape)}D shape: {shape}")

    # Generate a small grid of indices, with the same unique count as 'count'
    rect = _get_closest_rectangle(count)
    small_indices = np.pad(np.arange(count), np.prod(rect) - count, mode="edge")[: int(np.prod(rect))].reshape(rect)

    # Upscale the grid to fit the output shape using nearest neighbour scaling.
    indices = skimage.transform.resize(small_indices, shape, order=0, preserve_range=True).astype(int)

    return indices.reshape(shape)


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
    nb_col_split = math.ceil(col_range / col_split)
    nb_row_split = math.ceil(row_range / row_split)

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


def plot_tiling(raster: gu.Raster, tiling_grid: NDArrayNum) -> None:
    """
    Plot raster with its tiling.

    :param raster: The raster to plot with its tiling.
    :param tiling_grid: tiling given by compute_tiling.
    """
    ax, caxes = raster.plot(return_axes=True)
    for tile in tiling_grid.reshape(-1, 4):
        row_min, row_max, col_min, col_max = tile
        x_min, y_min = raster.transform * (col_min, row_min)  # Bottom-left corner
        x_max, y_max = raster.transform * (col_max, row_max)  # Top-right corne
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor="red", facecolor="none", linewidth=1.5)
        ax.add_patch(rect)
