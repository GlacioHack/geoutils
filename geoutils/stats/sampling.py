# Copyright (c) 2026 GeoUtils developers
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
import operator
from typing import Literal, overload, Any, Callable, TYPE_CHECKING

import numpy as np

from geoutils._typing import MArrayNum, NDArrayNum, NDArrayBool
from geoutils.raster.array import get_mask_from_array
from geoutils._misc import import_optional
from geoutils.multiproc import compute_tiling, MultiprocConfig

if TYPE_CHECKING:
    from geoutils.raster.raster import Raster, RasterBase

# Import Dask as optional dependency
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

###################################################
# 1/ SUBSAMPLING AT FINITE RANDOM POINT COORDINATES
###################################################

# Common input check

def _get_subsample_size_from_user_input(
    subsample: int | float, total_nb_valids: int,
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
            warnings.warn(
                f"Subsample value of {subsample} is larger than the number of valid pixels of {total_nb_valids},"
                f" using all valid pixels as a subsample.",
                category=UserWarning,
            )
    else:
        raise ValueError("Subsample must be > 0.")

    return npoints


######################
# NumPy implementation
######################

def _splitmix64(x: NDArrayNum) -> NDArrayNum:
    """
    Vectorized SplitMix64 mixer from uint64 to uint64.

    This function performs a fast deterministic mapping from integer IDs to "random-looking" 64-bit keys,
    that we use further below for reproducible subsampling based on global linear indices (the chunk-independent method
    "topk").
    We cannot use NumPy directly here because they don't expose their mixers used under-the-hood.

    References
    ----------
    - Steele et al., "Fast Splittable Pseudorandom Number Generators", OOPSLA 2014
      https://doi.org/10.1145/2660193.2660195
    - Sebastiano Vigna, SplitMix64 reference implementation, https://prng.di.unimi.it/splitmix64.c
    """
    # Add a large odd constant derived from the golden ratio
    # This ensures that consecutive inputs do not map to related outputs
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)

    # First mixing step: XOR-shift to spread high bits into low bits, then multiply by a chosen odd constant
    z = x
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z &= np.uint64(0xFFFFFFFFFFFFFFFF)

    # Second mixing step: Another XOR-shift followed by multiplication with a different constant
    # The constants were empirically chosen to achieve strong avalanche properties (each input bit affects
    # many output bits)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z &= np.uint64(0xFFFFFFFFFFFFFFFF)

    # Final XOR-shift to finish diffusion
    return z ^ (z >> np.uint64(31))

@overload
def _subsample_numpy(
    array: NDArrayNum | MArrayNum,
    subsample: float | int,
    return_indices: Literal[False] = False,
    *,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "sequential",
) -> NDArrayNum: ...


@overload
def _subsample_numpy(
    array: NDArrayNum | MArrayNum,
    subsample: float | int,
    return_indices: Literal[True],
    *,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "sequential",
) -> tuple[NDArrayNum, ...]: ...


def _subsample_numpy(
    array: NDArrayNum | MArrayNum,
    subsample: float | int,
    return_indices: bool = False,
    *,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "sequential",
) -> NDArrayNum | tuple[NDArrayNum, ...]:
    """
    Subsample valid values of a 1D or 2D array.

    :param array: Input array.
    :param subsample: Subsample size. If <= 1, considered a fraction of valid pixels to extract.
        If > 1 considered the number of pixels to extract.
    :param return_indices: If True, return extracted indices (same shape semantics as np.unravel_index).
    :param random_state: Random state, or seed number to use for random calculations (for testing).
    :param strategy: Sampling strategy:
        - "sequential": Random draw from valid indices (chunk-dependent, different output than chunked implementation).
        - "topk": Deterministic key-per-pixel draw (chunk-invariant, same output in chunked implementation).

    :returns: The subsampled array (1D) or the indices to extract (same shape as input array).
    """

    # Determine valid pixels and their global linear indices (row * nx + col)
    mask = get_mask_from_array(array)
    valids = np.flatnonzero(~mask.ravel())  # Robust 1D index list (global linear indices)
    total_nb_valids = int(valids.size)

    # If no valid values, early return
    if total_nb_valids == 0:
        if return_indices:
            return tuple(np.array([], dtype=int) for _ in range(array.ndim))
        return np.array([], dtype=array.dtype)

    # Get subsample size (depending on user input) using the helper
    subsample_size = _get_subsample_size_from_user_input(subsample=subsample, total_nb_valids=total_nb_valids)

    # If subsample is exactly 1, we don't subsample: return all valid values/indices
    if subsample == 1:
        unraveled = np.unravel_index(valids, array.shape)
        return unraveled if return_indices else array[unraveled]

    # If requested size is 0, we return empty
    if subsample_size <= 0:
        if return_indices:
            return tuple(np.array([], dtype=int) for _ in range(array.ndim))
        return np.array([], dtype=array.dtype)

    # STRATEGY 1: "sequential", we use a random order for the index of valid values
    if strategy == "sequential":

        rng = np.random.default_rng(random_state
                                    )
        # Choose random indexes among all valids
        chosen = rng.choice(valids, subsample_size, replace=False)

        # Unravel indexes, and return values or indexes
        unraveled = np.unravel_index(chosen, array.shape)
        return unraveled if return_indices else array[unraveled]

    # STRATEGY 2: "topk", deterministic by global linear index (gaves the same result with Dask/Multiprocessing)
    elif strategy == "topk":
        # Convert random_state into a stable integer seed used in key generation
        if isinstance(random_state, np.random.Generator):
            seed = int(random_state.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        elif random_state is None:
            seed = 0
        else:
            seed = int(random_state)

        # Get global indexes and their keys
        gids = valids.astype(np.uint64)
        keys = _splitmix64(np.uint64(seed) ^ gids)

        # Global linear indices of chosen valid pixels
        sel = np.argpartition(keys, subsample_size - 1)[:subsample_size]
        chosen = valids[sel]

        # Unravel indexes, and return values or indexes
        unraveled = np.unravel_index(chosen, array.shape)
        return unraveled if return_indices else array[unraveled]

    else:
        raise ValueError(f"Unknown strategy {strategy!r}. Choose 'sequential' or 'topk'.")


#####################
# Dask implementation
#####################

# At the date of April 2024:
# Getting an exact subsample size out-of-memory only for valid values is not supported directly by Dask/Xarray

# It is not trivial because we don't know where valid values will be in advance, and because of ragged output (varying
# output length considerations), which prevents from using high-level functions with good efficiency
# We thus follow https://blog.dask.org/2021/07/02/ragged-output (the dask.array.map_blocks solution has a larger RAM
# usage by having to drop an axis and re-chunk along 1D of the 2D array, so we use the delayed solution instead)

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
    if arr_chunk.dtype == np.bool_:
        return np.array([np.count_nonzero(arr_chunk)]).reshape((1, 1))
    return np.array([np.count_nonzero(np.isfinite(arr_chunk))]).reshape((1, 1))

@delayed
def _delayed_topk_candidates_block(
    arr_chunk: NDArrayNum | NDArrayBool,
    block_id: dict[str, Any],
    *,
    seed: int,
    k: int,
    nx_full: int,  # Width of full array
    return_indices_local: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return up to k valid samples from one block as (keys, payload).

    Those are:
    - keys: uint64 keys for selected valid pixels in this block
    - payload:
        * if return_indices_local=True: global linear indices (gid) of selected pixels (int64)
        * else: selected values from the array (dtype of arr_chunk, but typically float)
    """

    # If no samples, return empty
    if k <= 0:
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=np.int64)

    # Only valid values are sampled (finite for numerical arrays, True for boolean arrays)
    if np.issubdtype(arr_chunk.dtype, np.bool_):
        valid = arr_chunk
    else:
        valid = np.isfinite(arr_chunk)

    # Get nonzero indices for flattened array, and number of valid values
    flat = np.flatnonzero(valid.ravel())
    nvalid = int(flat.size)

    # If no valid
    if nvalid == 0:
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=np.int64)

    # Convert flat relative indices to local (row, col) within the chunk
    ncols = int(arr_chunk.shape[1])
    r = flat // ncols
    c = flat - r * ncols

    # Get absolute indices by adding  metadata passed to this chunk
    row0 = int(block_id["row_start"])
    col0 = int(block_id["col_start"])
    gid = (row0 + r) * nx_full + (col0 + c)

    # Get deterministic key per pixel based only on (seed, gid)
    key = _splitmix64(np.uint64(seed) ^ gid.astype(np.uint64))

    # Keep only the smallest m keys in this block (m <= k)
    m = min(int(k), nvalid)
    sel = np.argpartition(key, m - 1)[:m]
    key_sel = key[sel]

    # If return indices
    if return_indices_local:
        gid_sel = gid[sel]
        return key_sel, gid_sel
    # Otherwise, returning values
    else:
        # Extract values for selected valid pixels
        if np.issubdtype(arr_chunk.dtype, np.bool_):
            vals = np.ones(m, dtype=np.bool_)
        else:
            vals = arr_chunk.ravel()[flat[sel]]
        return key_sel, vals

@delayed
def _delayed_merge_topk(
    keys_list: list[np.ndarray],
    payload_list: list[np.ndarray],
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge per-block candidates and return the global top-k by key.

    This global reduction steps allows to make the strategy chunk-invariant.
    """

    # If list of all keys is empty, return empty results
    if len(keys_list) == 0:
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=np.int64)

    # Flatten and concatenate all per-block candidate outputs into one global list
    keys = np.concatenate([np.asarray(x, dtype=np.uint64).ravel() for x in keys_list], axis=0)
    payload = np.concatenate([np.asarray(x).ravel() for x in payload_list], axis=0)

    # Handle the case of no valid pixels anywhere
    n = int(keys.size)
    if n == 0:
        return keys, payload

    # We only need the global top-k smallest keys (k may exceed n if there are fewer candidates than requested)
    m = min(int(k), n)

    # Select indices of the m smallest keys efficiently
    #  (np.argpartition is O(n) average and avoids sorting the full array which would be O(n log n))
    sel = np.argpartition(keys, m - 1)[:m]

    # Sort the selected indices by their key values to produce a stable, deterministic ordering
    sel = sel[np.argsort(keys[sel])]

    # Return the m smallest keys and their associated payload entries
    return keys[sel], payload[sel]

@delayed
def _delayed_gid_to_rc(gid: np.ndarray, nx_full: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert global linear indices back to (row, col) indices.
    """
    gid = np.asarray(gid, dtype=np.int64).ravel()
    r = gid // np.int64(nx_full)
    c = gid - r * np.int64(nx_full)
    return r.astype(np.int64), c.astype(np.int64)

@delayed
def _delayed_subsample_block(
    arr_chunk: NDArrayNum | NDArrayBool, subsample_indices: NDArrayNum
) -> NDArrayNum | NDArrayBool:
    """Subsample the valid values at the corresponding 1D valid indices per block."""

    if arr_chunk.dtype == np.bool_:
        return arr_chunk[arr_chunk][subsample_indices]
    return arr_chunk[np.isfinite(arr_chunk)][subsample_indices]

@delayed
def _delayed_subsample_indices_block(
    arr_chunk: NDArrayNum | NDArrayBool, subsample_indices: NDArrayNum, block_id: dict[str, Any]
) -> NDArrayNum:
    """Return 2D indices from the subsampled 1D valid indices per block."""

    if arr_chunk.dtype == np.bool_:
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

def _dask_subsample(
    darr: da.Array,
    subsample: int | float = 1,
    return_indices: bool = False,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "sequential",
) -> da.Array | tuple[da.Array, da.Array]:
    """
    Subsample valid values out-of-memory from a 2D Dask array.

    Strategy "topk" is chunk-invariant (same sample no matter chunk size, and same as the NumPy implementation), while
    "sequential" is chunk-dependent but slightly faster.

    Returns a delayed subsampled Dask array of the output (either values or indices).
    """

    # To raise appropriate error on missing optional dependency
    import_optional("dask")

    # Get random state
    # For method="sequential", we use the RNG stream based on valid orders (chunk-dependent)
    # For method="topk", we convert random_state into an integer seed used in the deterministic key function
    rng = np.random.default_rng(random_state)

    # Create a delayed object for each block, and flatten the blocks into a 1d shape
    blocks = darr.to_delayed().ravel()

    # Compute number of valid points for each block out-of-memory
    list_delayed_valids = [da.from_delayed(_delayed_nb_valids(b), shape=(1, 1),
                                           dtype=np.dtype("int32")) for b in blocks]
    # Compute once, then flatten
    nb_valids_per_block = np.concatenate([x.ravel() for x in
                                          dask.compute(*list_delayed_valids)], axis=0).astype(np.int64)

    # Sum to get total number of valid points
    total_nb_valids = int(np.sum(nb_valids_per_block))

    # Get subsample size (depending on user input)
    subsample_size = _get_subsample_size_from_user_input(subsample=subsample, total_nb_valids=total_nb_valids)

    # Quick exit if there are no valid pixels or subsample_size is 0
    if subsample_size <= 0 or total_nb_valids <= 0:
        if return_indices:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
        else:
            return np.empty((0,), dtype=darr.dtype)

    # 1/ Build block IDs (starting indices for each block in the full array)

    # We get starting 2D index for each chunk of the full array (mirroring what is done in dask.array.map_blocks)
    # https://github.com/dask/dask/blob/24493f58660cb933855ba7629848881a6e2458c1/dask/array/core.py#L908
    # This list also includes the last index as well (not used here)
    starts = [cached_cumsum(c, initial_zero=True) for c in darr.chunks]
    num_chunks = darr.numblocks

    # Get the starts per 1D block ID by unravelling starting indexes for each block
    indexes_yb, indexes_xb = np.unravel_index(np.arange(len(blocks)), shape=(num_chunks[0], num_chunks[1]))

    block_ids = [
        {"row_start": starts[0][indexes_yb[i]], "col_start": starts[1][indexes_xb[i]]}
         for i in range(len(blocks))
    ]

    # STRATEGY 1: "sequential" (chunk-dependent)
    if strategy == "sequential":

        # Get random 1D indexes for the subsample size
        indices_1d = rng.choice(total_nb_valids, subsample_size, replace=False)

        # Sort which indexes belong to which chunk
        ind_per_block = _get_indices_block_per_subsample(
            indices_1d, num_chunks=darr.numblocks, nb_valids_per_block=nb_valids_per_block.tolist()
        )

        # To just get the subsample without indices
        if not return_indices:
            # Task a delayed subsample to be computed for each block, skipping blocks with no values to sample
            used = [i for i in range(len(blocks)) if len(ind_per_block[i]) > 0]
            list_subsamples = [_delayed_subsample_block(blocks[i], np.asarray(ind_per_block[i], dtype=np.int64)) for i in used]

            # Cast output to the right expected dtype and length, then compute and concatenate
            list_subsamples_da = [
                da.from_delayed(s, shape=(len(ind_per_block[i]),), dtype=darr.dtype)
                for s, i in zip(list_subsamples, used)
            ]
            return da.concatenate(list_subsamples_da, axis=0)

        # To return indices
        else:
            # Task delayed subsample indices to be computed for each block, skipping blocks with no values to sample
            used = [i for i in range(len(blocks)) if len(ind_per_block[i]) > 0]
            list_subsample_indices = [
                _delayed_subsample_indices_block(blocks[i], np.asarray(ind_per_block[i], dtype=np.int64),
                                                 block_id=block_ids[i])
                for i in used
            ]

            # Cast output to the right expected dtype and length, then compute and concatenate
            list_indices_da = [
                da.from_delayed(s, shape=(len(ind_per_block[i]), 2), dtype=np.int32)
                for s, i in zip(list_subsample_indices, used)
            ]
            indices = da.concatenate(list_indices_da, axis=0)
            return indices[:, 0], indices[:, 1]

    # STRATEGY 2: "topk" (chunk-invariant; deterministic by (seed, global linear index))
    elif strategy == "topk":

        # Convert random_state to an integer seed for deterministic key generation
        if isinstance(random_state, np.random.Generator):
            seed = int(random_state.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        elif random_state is None:
            seed = 0
        else:
            seed = int(random_state)

        # Full-array width for global linear indexing
        nx_full = int(darr.shape[1])

        # One candidate extraction task per block: Each block returns up to subsample_size candidates (keys + payload)
        cands = [
            _delayed_topk_candidates_block(
                blocks[i],
                block_ids[i],
                seed=seed,
                k=subsample_size,
                nx_full=nx_full,
                return_indices_local=return_indices,
            )
            for i in range(len(blocks))
        ]

        # Separate keys and payload lists (payload are either values or global linear indices)
        keys_list = [c[0] for c in cands]
        payload_list = [c[1] for c in cands]

        # Global merge to get the top-k across all blocks
        merged = _delayed_merge_topk(keys_list, payload_list, k=subsample_size)

        # Lazily extract tuple elements
        payload_delayed = dask.delayed(operator.getitem)(merged, 1)

        if not return_indices:
            # The payload is a Delayed object that returns a 1D numpy array of length smaller than subsample_size
            # We know the final size is exactly subsample_size
            return da.from_delayed(payload_delayed, shape=(subsample_size,), dtype=darr.dtype)

        else:
            # The payload is global linear indices (gid) that we convert lazily to (row, col)
            rr_cc = _delayed_gid_to_rc(payload_delayed, nx_full)
            rr_delayed = dask.delayed(operator.getitem)(rr_cc, 0)
            cc_delayed = dask.delayed(operator.getitem)(rr_cc, 1)

            rr = da.from_delayed(rr_delayed, shape=(subsample_size,), dtype=np.int64)
            cc = da.from_delayed(cc_delayed, shape=(subsample_size,), dtype=np.int64)
            return rr, cc

    else:
        raise ValueError(f"Unknown strategy {strategy!r}, available strategies are 'sequential' or 'topk'.")

################################
# Multiprocessing implementation
################################

def _wrapper_multiproc_nb_valids_per_block(rst: Raster, tile_idx: NDArrayNum) -> int:
    """Count valid values in one tile out-of-memory."""
    rst_block = rst.icrop((tile_idx[2], tile_idx[0], tile_idx[3], tile_idx[1]))
    arr = rst_block.data

    if np.issubdtype(arr.dtype, np.bool_):
        return int(np.count_nonzero(arr))
    return int(np.count_nonzero(np.isfinite(arr)))


def _wrapper_multiproc_subsample_values_block(
    rst: Raster,
    tile_idx: NDArrayNum,
    subsample_indices_rel: NDArrayNum,
) -> NDArrayNum:
    """
    Subsample values in one tile using 1D indices relative to the tile's valid-value list.
    """

    # Get tile out-of-memory
    rst_block = rst.icrop((tile_idx[2], tile_idx[0], tile_idx[3], tile_idx[1]))
    arr = rst_block.data

    # Return subsample of finite values (or True values for boolean input)
    if np.issubdtype(arr.dtype, np.bool_):
        return arr[arr].ravel()[subsample_indices_rel]
    return arr[np.isfinite(arr)].ravel()[subsample_indices_rel]


def _wrapper_multiproc_subsample_indices_block(
    rst: Raster,
    tile_idx: NDArrayNum,
    subsample_indices_rel: NDArrayNum,
) -> NDArrayNum:
    """
    Return indices of the sampled valid pixels in one tile.

    Output shape: (n, 2) with columns [row, col] in full-array coordinates.
    """

    # Get tile out-of-memory
    rst_block = rst.icrop((tile_idx[2], tile_idx[0], tile_idx[3], tile_idx[1]))
    arr = rst_block.data

    # Get starting row/col of the tile
    row0 = int(tile_idx[0])
    col0 = int(tile_idx[2])

    # Get relative indices of finite values (or True for boolean)
    if np.issubdtype(arr.dtype, np.bool_):
        flat_valid = np.flatnonzero(arr.ravel())
    else:
        flat_valid = np.flatnonzero(np.isfinite(arr).ravel())

    # Use input to draw them
    flat_sel = flat_valid[subsample_indices_rel.astype(np.int64)]

    # Transform back into absolute indices
    ncols = int(arr.shape[1])
    r = (flat_sel // ncols).astype(np.int64) + row0
    c = (flat_sel - (flat_sel // ncols) * ncols).astype(np.int64) + col0

    return np.stack((r, c), axis=1)


def _wrapper_multiproc_topk_candidates_block(
    rst: Raster,
    tile_idx: NDArrayNum,
    *,
    seed: int,
    k: int,
    nx_full: int,
    return_indices: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return up to k candidates from one tile as (keys, payload).

    keys: uint64 keys for selected valid pixels in this tile.
    payload:
      - if return_indices=True: global linear indices (gid = row*nx_full + col) (int64)
      - else: sampled values (array dtype)
    """
    # If no subsample, early return
    if k <= 0:
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=np.int64)

    # Get tile out-of-memory
    rst_block = rst.icrop((tile_idx[2], tile_idx[0], tile_idx[3], tile_idx[1]))
    arr = rst_block.data

    # Tile offsets in full-array indices
    row0 = int(tile_idx[0])
    col0 = int(tile_idx[2])

    # Get valids indices
    if np.issubdtype(arr.dtype, np.bool_):
        valid = arr
    else:
        valid = np.isfinite(arr)
    flat = np.flatnonzero(valid.ravel())
    nvalid = int(flat.size)

    # If no valid, early return
    if nvalid == 0:
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=np.int64)

    # Get relative row and columns
    ncols = int(arr.shape[1])
    r = flat // ncols
    c = flat - r * ncols

    # Global linear index from absolute row and columns: gid = (row0 + r) * nx_full + (col0 + c)
    gid = (np.int64(row0) + r.astype(np.int64)) * np.int64(nx_full) + (np.int64(col0) + c.astype(np.int64))
    # Derive key from gid
    key = _splitmix64(np.uint64(seed) ^ gid.astype(np.uint64))

    # Select the appropriate number of keys
    m = min(int(k), nvalid)
    sel = np.argpartition(key, m - 1)[:m]
    key_sel = key[sel]

    # If we return indices
    if return_indices:
        return key_sel, gid[sel]

    # If we return values
    if np.issubdtype(arr.dtype, np.bool_):
        vals = np.ones(m, dtype=np.bool_)
    else:
        vals = arr.ravel()[flat[sel]]
    return key_sel, vals

def _multiproc_subsample(
    rst: Raster,
    config: MultiprocConfig,
    subsample: int | float = 1,
    return_indices: bool = False,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "sequential",
) -> NDArrayNum | tuple[NDArrayNum, NDArrayNum]:
    """
    Subsample valid values out-of-memory from a 2D raster array using Multiprocessing tasks.

    Strategy "topk" is chunk-invariant (same sample no matter chunk size, and same as the NumPy implementation), while
    "sequential" is chunk-dependent but slightly faster.

    Returns a concatenated subsampled NumPy array collected from all tasks (either values or indices).
    """

    # Get tiling
    tiling = compute_tiling(tile_size=config.chunk_size, raster_shape=rst.shape, overlap=0)

    # Get number of chunks and blocks
    num_chunks = (tiling.shape[0], tiling.shape[1])
    num_blocks = int(np.prod(num_chunks))

    # Flatten tile_idx list in row-major block order
    indexes_row, indexes_col = np.unravel_index(np.arange(num_blocks), shape=num_chunks)
    tile_ids = [tiling[indexes_row[i], indexes_col[i], :] for i in range(num_blocks)]

    # Count valid values per tile in parallel
    tasks = [
        config.cluster.launch_task(fun=_wrapper_multiproc_nb_valids_per_block, args=[rst, tile_ids[i]], kwargs={})
        for i in range(num_blocks)
    ]
    try:
        nb_valids_per_block = np.array([config.cluster.get_res(t) for t in tasks], dtype=np.int64)
    except Exception as e:
        raise RuntimeError(f"Error retrieving valid-count results from multiprocessing tasks: {e}")

    total_nb_valids = int(nb_valids_per_block.sum())

    # Get subsample size (depending on user input)
    subsample_size = _get_subsample_size_from_user_input(subsample=subsample, total_nb_valids=total_nb_valids)

    # Early exit if too few samples or valids
    if subsample_size <= 0 or total_nb_valids <= 0:
        if return_indices:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
        return np.empty((0,), dtype=rst.dtype)

    # METHOD 1: sequential (chunk-dependent)
    if strategy == "sequential":
        rng = np.random.default_rng(random_state)

        # Sample indices among the valids
        indices_1d = rng.choice(total_nb_valids, subsample_size, replace=False)

        # Map the sampled indices to per-tile relative indices
        ind_per_block = _get_indices_block_per_subsample(
            indices_1d=indices_1d,
            num_chunks=num_chunks,
            nb_valids_per_block=nb_valids_per_block.tolist(),
        )

        used = [i for i in range(num_blocks) if len(ind_per_block[i]) > 0]

        # Sample them through multiprocessing, either for indices or values
        if not return_indices:
            tasks = [
                config.cluster.launch_task(
                    fun=_wrapper_multiproc_subsample_values_block,
                    args=[rst, tile_ids[i], np.asarray(ind_per_block[i], dtype=np.int64)],
                    kwargs={},
                )
                for i in used
            ]

            try:
                list_vals = [config.cluster.get_res(t) for t in tasks]
            except Exception as e:
                raise RuntimeError(f"Error retrieving subsampled values from multiprocessing tasks: {e}")

            # Concatenate in tile order (this yields deterministic order given tiling; not random order)
            return np.concatenate(list_vals, axis=0)

        else:
            tasks = [
                config.cluster.launch_task(
                    fun=_wrapper_multiproc_subsample_indices_block,
                    args=[rst, tile_ids[i], np.asarray(ind_per_block[i], dtype=np.int64)],
                    kwargs={},
                )
                for i in used
            ]

            try:
                list_rc = [config.cluster.get_res(t) for t in tasks]  # each (n_i, 2)
            except Exception as e:
                raise RuntimeError(f"Error retrieving subsampled indices from multiprocessing tasks: {e}")

            rc = np.concatenate(list_rc, axis=0)
            rows = rc[:, 0].astype(np.int64)
            cols = rc[:, 1].astype(np.int64)
            return rows, cols

    # METHOD 2: topk (chunk-invariant)
    elif strategy == "topk":

        # Convert random_state to an integer seed used in deterministic keys
        if isinstance(random_state, np.random.Generator):
            seed = int(random_state.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        elif random_state is None:
            seed = 0
        else:
            seed = int(random_state)

        # Get full-array width
        nx_full = int(rst.shape[1])

        tasks = [
            config.cluster.launch_task(
                fun=_wrapper_multiproc_topk_candidates_block,
                args=[rst, tile_ids[i]],
                kwargs=dict(seed=seed, k=subsample_size, nx_full=nx_full, return_indices=return_indices),
            )
            for i in range(num_blocks)
        ]

        try:
            cand = [config.cluster.get_res(t) for t in tasks]  # list of (keys, payload)
        except Exception as e:
            raise RuntimeError(f"Error retrieving topk candidates from multiprocessing tasks: {e}")

        keys_list = [np.asarray(k, dtype=np.uint64).ravel() for k, _ in cand]
        payload_list = [np.asarray(p).ravel() for _, p in cand]

        keys = np.concatenate(keys_list) if keys_list else np.empty((0,), dtype=np.uint64)
        payload = np.concatenate(payload_list) if payload_list else np.empty((0,), dtype=np.int64)

        if keys.size == 0:
            if return_indices:
                return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
            return np.empty((0,), dtype=rst.dtype)

        m = min(int(subsample_size), int(keys.size))
        sel = np.argpartition(keys, m - 1)[:m]
        sel = sel[np.argsort(keys[sel])]

        payload_sel = payload[sel]

        if not return_indices:
            return payload_sel

        # payload is gid -> (row, col)
        gid = payload_sel.astype(np.int64)
        rows = gid // np.int64(nx_full)
        cols = gid - rows * np.int64(nx_full)
        return rows.astype(np.int64), cols.astype(np.int64)

    else:
        raise ValueError(f"Unknown strategy {method!r}. Choose 'sequential' or 'topk'.")

######################################################
# Wrapper dispatching to NumPy or Dask/Multiprocessing
######################################################

def _subsample(
    source_raster: RasterBase,
    subsample: float | int = 1,
    return_indices: bool = False,
    *,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "sequential",
    mp_config: MultiprocConfig | None = None,
) -> Any:
    """
    Subsample an array at valid values, dispatching automatically to NumPy, Dask or Multiprocessing implementation.

    :param source_raster: Input array (NumPy/masked or Dask).
    :param subsample: Subsample size or fraction.
    :param return_indices: If True, return (rows, cols) indices instead of values.
    :param random_state: Seed or Generator.
    :param strategy: Either "sequential" (chunk/order dependent) or "topk" (chunk-invariant).

    :returns:
      - values: 1D array of sampled values
      - indices: (rows, cols) (axis order)
      - for Dask input: returns lazy `da.Array` unless compute=True
    """

    # Cannot use Multiprocessing backend and Dask backend simultaneously
    mp_backend = mp_config is not None
    # The check below can only run on Xarray
    dask_backend = da is not None and source_raster._chunks is not None

    if mp_backend and dask_backend:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove mp_config parameter "
            "from reproject(). To use Multiprocessing, open the file without chunks."
        )

    subsample_kwargs = {"subsample": subsample, "return_indices": return_indices, "random_state": random_state,
                        "strategy": strategy}

    # Multiprocessing (out-of-memory)
    if mp_backend:
        return _multiproc_subsample(source_raster, config=mp_config, **subsample_kwargs)
    # Dask (out-of-memory)
    elif dask_backend:
        return _dask_subsample(source_raster.data, **subsample_kwargs)
    # NumPy
    else:
        return _subsample_numpy(source_raster.data, **subsample_kwargs)
