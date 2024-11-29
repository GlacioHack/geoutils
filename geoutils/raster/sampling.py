"""Sampling tools for arrays and rasters."""

from __future__ import annotations

from typing import Literal, overload

import numpy as np

from geoutils._typing import MArrayNum, NDArrayNum
from geoutils.raster.array import get_mask_from_array


@overload
def subsample_array(
    array: NDArrayNum | MArrayNum,
    subsample: float | int,
    return_indices: Literal[False] = False,
    *,
    random_state: int | np.random.Generator | None = None,
) -> NDArrayNum: ...


@overload
def subsample_array(
    array: NDArrayNum | MArrayNum,
    subsample: float | int,
    return_indices: Literal[True],
    *,
    random_state: int | np.random.Generator | None = None,
) -> tuple[NDArrayNum, ...]: ...


@overload
def subsample_array(
    array: NDArrayNum | MArrayNum,
    subsample: float | int,
    return_indices: bool = False,
    random_state: int | np.random.Generator | None = None,
) -> NDArrayNum | tuple[NDArrayNum, ...]: ...


def subsample_array(
    array: NDArrayNum | MArrayNum,
    subsample: float | int,
    return_indices: bool = False,
    random_state: int | np.random.Generator | None = None,
) -> NDArrayNum | tuple[NDArrayNum, ...]:
    """
    Randomly subsample a 1D or 2D array by a sampling factor, taking only non NaN/masked values.

    :param array: Input array.
    :param subsample: Subsample size. If <= 1, will be considered a fraction of valid pixels to extract.
        If > 1 will be considered the number of pixels to extract.
    :param return_indices: If set to True, will return the extracted indices only.
    :param random_state: Random state, or seed number to use for random calculations (for testing)

    :returns: The subsampled array (1D) or the indices to extract (same shape as input array)
    """
    # Define state for random sampling (to fix results during testing)
    rng = np.random.default_rng(random_state)

    # Remove invalid values and flatten array
    mask = get_mask_from_array(array)  # -> need to remove .squeeze in get_mask
    valids = np.argwhere(~mask.flatten()).squeeze()

    # Get number of points to extract
    # If subsample is one, we don't perform any subsampling operation, we return the valid array or indices directly
    if subsample == 1:
        unraveled_indices = np.unravel_index(valids, array.shape)
        if return_indices:
            return unraveled_indices
        else:
            return array[unraveled_indices]
    if (subsample <= 1) & (subsample > 0):
        npoints = int(subsample * np.count_nonzero(~mask))
    elif subsample > 1:
        npoints = int(subsample)
    else:
        raise ValueError("`subsample` must be > 0")

    # Checks that array and npoints are correct
    assert np.ndim(valids) == 1, "Something is wrong with array dimension, check input data and shape"
    if npoints > np.size(valids):
        npoints = np.size(valids)

    # Randomly extract npoints without replacement
    indices = rng.choice(valids, npoints, replace=False)
    unraveled_indices = np.unravel_index(indices, array.shape)

    if return_indices:
        return unraveled_indices
    else:
        return array[unraveled_indices]


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
        return (close_cube, close_cube)

    # One of these rectangles/cubes will cover all cells, so return the first that does.
    potential_rectangles = [(close_cube, close_cube + 1), (close_cube + 1, close_cube + 1)]

    for rectangle in potential_rectangles:
        if np.prod(rectangle) >= size:
            return rectangle

    raise NotImplementedError(f"Function criteria not met for rectangle of size: {size}")


def subdivide_array(shape: tuple[int, ...], count: int) -> NDArrayNum:
    """
    Create indices for subdivison of an array in a number of blocks.

    If 'count' is divisible by the product of 'shape', the amount of cells in each block will be equal.
    If 'count' is not divisible, the amount of cells in each block will be very close to equal.

    :param shape: The shape of a array to be subdivided.
    :param count: The amount of subdivisions to make.

    :examples:
        >>> subdivide_array((4, 4), 4)
        array([[0, 0, 1, 1],
               [0, 0, 1, 1],
               [2, 2, 3, 3],
               [2, 2, 3, 3]])

        >>> subdivide_array((6, 4), 4)
        array([[0, 0, 1, 1],
               [0, 0, 1, 1],
               [0, 0, 1, 1],
               [2, 2, 3, 3],
               [2, 2, 3, 3],
               [2, 2, 3, 3]])

        >>> subdivide_array((5, 4), 3)
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

    if count > np.prod(shape):
        raise ValueError(f"Shape '{shape}' size ({np.prod(shape)}) is smaller than 'count' ({count}).")

    if len(shape) != 2:
        raise ValueError(f"Expected a 2D shape, got {len(shape)}D shape: {shape}")

    # Generate a small grid of indices, with the same unique count as 'count'
    rect = _get_closest_rectangle(count)
    small_indices = np.pad(np.arange(count), np.prod(rect) - count, mode="edge")[: int(np.prod(rect))].reshape(rect)

    # Upscale the grid to fit the output shape using nearest neighbour scaling.
    indices = skimage.transform.resize(small_indices, shape, order=0, preserve_range=True).astype(int)

    return indices.reshape(shape)
