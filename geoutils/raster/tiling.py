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

import sys

import numpy as np
from scipy.ndimage import zoom

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

    :param shape: The shape of array to be subdivided.
    :param count: The amount of subdivisions to make.

    :returns: An array of shape 'shape' with 'count' unique indices.
    """

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

    # Compute zoom factors
    zoom_factors = np.array(shape) / np.array(small_indices.shape)

    # Upscale the grid to fit the output shape using nearest neighbour scaling.
    indices = zoom(small_indices, zoom=zoom_factors, order=0).astype(int)

    return indices.reshape(shape)
