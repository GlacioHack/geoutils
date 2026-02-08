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

"""Module for array sampling statistics."""

from __future__ import annotations

import warnings
from typing import Literal, overload, Any, Callable

import numpy as np

from geoutils._typing import MArrayNum, NDArrayNum, NDArrayBool
from geoutils.raster.array import get_mask_from_array
from geoutils._misc import import_optional

# Dask as optional dependency
try:
    import dask
    import dask.array as da
    from dask import delayed
    from dask.utils import cached_cumsum
except ImportError:

    def delayed(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Fake delayed decorator if dask is not installed
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

##############################################################
# 1/ SUBSAMPLE AT FINITE (NON-NODATA) RANDOM POINT COORDINATES
##############################################################

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


# 1/ SUBSAMPLING
# At the date of April 2024:
# Getting an exact subsample size out-of-memory only for valid values is not supported directly by Dask/Xarray

# It is not trivial because we don't know where valid values will be in advance, and because of ragged output (varying
# output length considerations), which prevents from using high-level functions with good efficiency
# We thus follow https://blog.dask.org/2021/07/02/ragged-output (the dask.array.map_blocks solution has a larger RAM
# usage by having to drop an axis and re-chunk along 1D of the 2D array, so we use the delayed solution instead)


def _get_subsample_size_from_user_input(
    subsample: int | float, total_nb_valids: int, silence_max_subsample: bool
) -> int:
    """Get subsample size based on a user input of either integer size or fraction of the number of valid points."""

    # If value is between 0 and 1, use a fraction
    if (subsample <= 1) & (subsample > 0):
        npoints = int(subsample * total_nb_valids)
    # Otherwise use the value directly
    elif subsample > 1:
        # Use the number of valid points if larger than subsample asked by user
        npoints = min(int(subsample), total_nb_valids)
        if subsample > total_nb_valids:
            if not silence_max_subsample:
                warnings.warn(
                    f"Subsample value of {subsample} is larger than the number of valid pixels of {total_nb_valids},"
                    f" using all valid pixels as a subsample.",
                    category=UserWarning,
                )
    else:
        raise ValueError("Subsample must be > 0.")

    return npoints


def _get_indices_block_per_subsample(
    indices_1d: NDArrayNum, num_chunks: tuple[int, int], nb_valids_per_block: list[int]
) -> list[list[int]]:
    """
    Get list of 1D valid subsample indices relative to the block for each block.

    The 1D valid subsample indices correspond to the subsample index to apply for a flattened array of valid values.
    Relative to the block means converted so that the block indexes for valid values starts at 0 up to the number of
    valid values in that block (while the input indices go from zero to the total number of valid values in the full
    array).

    :param indices_1d: Subsample 1D indexes among a total number of valid values.
    :param num_chunks: Number of chunks in X and Y.
    :param nb_valids_per_block: Number of valid pixels per block.

    :returns: Relative 1D valid subsample index per block.
    """

    # Apply a cumulative sum to get the first 1D total index of each block
    valids_cumsum = np.cumsum(nb_valids_per_block)

    # We can write a faster algorithm by sorting
    indices_1d = np.sort(indices_1d)

    # Could we write nested lists into array format to further save RAM?
    # We define a list of indices per block
    relative_index_per_block = [[] for _ in range(num_chunks[0] * num_chunks[1])]
    k = 0  # K is the block number
    for i in indices_1d:

        # Move to the next block K where current 1D subsample index is, if not in this one
        while i >= valids_cumsum[k]:
            k += 1

        # Add 1D subsample index  relative to first subsample index of this block
        first_index_block = valids_cumsum[k - 1] if k >= 1 else 0  # The first 1D valid subsample index of the block
        relative_index = i - first_index_block
        relative_index_per_block[k].append(relative_index)

    return relative_index_per_block


@delayed
def _delayed_nb_valids(arr_chunk: NDArrayNum | NDArrayBool) -> NDArrayNum:
    """Count number of valid values per block."""
    if arr_chunk.dtype == "bool":
        return np.array([np.count_nonzero(arr_chunk)]).reshape((1, 1))
    return np.array([np.count_nonzero(np.isfinite(arr_chunk))]).reshape((1, 1))


@delayed
def _delayed_subsample_block(
    arr_chunk: NDArrayNum | NDArrayBool, subsample_indices: NDArrayNum
) -> NDArrayNum | NDArrayBool:
    """Subsample the valid values at the corresponding 1D valid indices per block."""

    if arr_chunk.dtype == "bool":
        return arr_chunk[arr_chunk][subsample_indices]
    return arr_chunk[np.isfinite(arr_chunk)][subsample_indices]


@delayed
def _delayed_subsample_indices_block(
    arr_chunk: NDArrayNum | NDArrayBool, subsample_indices: NDArrayNum, block_id: dict[str, Any]
) -> NDArrayNum:
    """Return 2D indices from the subsampled 1D valid indices per block."""

    if arr_chunk.dtype == "bool":
        ix, iy = np.unravel_index(np.argwhere(arr_chunk.flatten())[subsample_indices], shape=arr_chunk.shape)
    else:
        #  Unravel indices of valid data to the shape of the block
        ix, iy = np.unravel_index(
            np.argwhere(np.isfinite(arr_chunk.flatten()))[subsample_indices], shape=arr_chunk.shape
        )

    # Convert to full-array indexes by adding the row and column starting indexes for this block
    ix += block_id["xstart"]
    iy += block_id["ystart"]

    return np.hstack((ix, iy))


def delayed_subsample(
    darr: da.Array,
    subsample: int | float = 1,
    return_indices: bool = False,
    random_state: int | np.random.Generator | None = None,
    silence_max_subsample: bool = False,
) -> NDArrayNum | tuple[NDArrayNum, NDArrayNum]:
    """
    Subsample a raster at valid values on out-of-memory chunks.

    Optionally, this function can return the 2D indices of the subsample of valid values instead.

    The random subsample is distributed evenly across valid values, no matter which chunk they belong to.
    First, the number of valid values in each chunk are computed out-of-memory. Then, a subsample is defined among
    the total number of valid values, which are then indexed sequentially along the chunk valid values out-of-memory.

    A random state will give a fixed subsample for a delayed array with a fixed chunksize. However, the subsample
    will vary with changing chunksize because the 1D delayed indexing depends on it (indexing per valid value per
    flattened chunk). For this reason, a loaded array will also have a different subsample due to its direct 1D
    indexing (per valid value for the entire flattened array).

    To ensure you reuse a similar subsample of valid values for several arrays, call this function with
    return_indices=True, then sample your arrays out-of-memory with .vindex[indices[0], indices[1]]
    (this assumes that these arrays have valid values at the same locations).

    Only valid values are sampled. If passing a numerical array, then only finite values are considered valid values.
    If passing a boolean array, then only True values are considered valid values.

    :param darr: Input dask array. This can be a boolean or a numerical array.
    :param subsample: Subsample size. If <= 1, will be considered a fraction of valid pixels to extract.
        If > 1 will be considered the number of valid pixels to extract.
    :param return_indices: If set to True, will return the extracted indices only.
    :param random_state: Random state, or seed number to use for random calculations.
    :param silence_max_subsample: Whether to silence the warning for the subsample size being larger than the total
        number of valid points (warns by default).

    :return: Subsample of values from the array (optionally, their indexes).
    """

    # To raise appropriate error on missing optional dependency
    import_optional("dask")

    # Get random state
    rng = np.random.default_rng(random_state)

    # Compute number of valid points for each block out-of-memory
    blocks = darr.to_delayed().ravel()
    list_delayed_valids = [
        da.from_delayed(_delayed_nb_valids(b), shape=(1, 1), dtype=np.dtype("int32")) for b in blocks
    ]
    nb_valids_per_block = np.concatenate([dask.compute(*list_delayed_valids)]).squeeze()

    # Sum to get total number of valid points
    total_nb_valids = np.sum(nb_valids_per_block)

    # Get subsample size (depending on user input)
    subsample_size = _get_subsample_size_from_user_input(
        subsample=subsample, total_nb_valids=total_nb_valids, silence_max_subsample=silence_max_subsample
    )

    # Get random 1D indexes for the subsample size
    indices_1d = rng.choice(total_nb_valids, subsample_size, replace=False)

    # Sort which indexes belong to which chunk
    ind_per_block = _get_indices_block_per_subsample(
        indices_1d, num_chunks=darr.numblocks, nb_valids_per_block=nb_valids_per_block
    )

    # To just get the subsample without indices
    if not return_indices:
        # Task a delayed subsample to be computed for each block, skipping blocks with no values to sample
        list_subsamples = [
            _delayed_subsample_block(b, ind)
            for i, (b, ind) in enumerate(zip(blocks, ind_per_block))
            if len(ind_per_block[i]) > 0
        ]
        # Cast output to the right expected dtype and length, then compute and concatenate
        list_subsamples_delayed = [
            da.from_delayed(s, shape=(nb_valids_per_block[i],), dtype=darr.dtype) for i, s in enumerate(list_subsamples)
        ]
        subsamples = np.concatenate(dask.compute(*list_subsamples_delayed), axis=0)

        return subsamples

    # To return indices
    else:
        # Get starting 2D index for each chunk of the full array
        # (mirroring what is done in block_id of dask.array.map_blocks)
        # https://github.com/dask/dask/blob/24493f58660cb933855ba7629848881a6e2458c1/dask/array/core.py#L908
        # This list also includes the last index as well (not used here)
        starts = [cached_cumsum(c, initial_zero=True) for c in darr.chunks]
        num_chunks = darr.numblocks
        # Get the starts per 1D block ID by unravelling starting indexes for each block
        indexes_xi, indexes_yi = np.unravel_index(np.arange(len(blocks)), shape=(num_chunks[0], num_chunks[1]))
        block_ids = [
            {"xstart": starts[0][indexes_xi[i]], "ystart": starts[1][indexes_yi[i]]} for i in range(len(blocks))
        ]

        # Task delayed subsample indices to be computed for each block, skipping blocks with no values to sample
        list_subsample_indices = [
            _delayed_subsample_indices_block(b, ind, block_id=block_ids[i])
            for i, (b, ind) in enumerate(zip(blocks, ind_per_block))
            if len(ind_per_block[i]) > 0
        ]
        # Cast output to the right expected dtype and length, then compute and concatenate
        list_subsamples_indices_delayed = [
            da.from_delayed(s, shape=(2, len(ind_per_block[i])), dtype=np.dtype("int32"))
            for i, s in enumerate(list_subsample_indices)
        ]
        indices = np.concatenate(dask.compute(*list_subsamples_indices_delayed), axis=0)

        return indices[:, 0], indices[:, 1]



