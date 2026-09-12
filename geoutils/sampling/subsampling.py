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

"""Module for subsampling: selecting a random subset of valid points in a N-D array."""

from __future__ import annotations

import operator
import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict, cast, overload

import numpy as np

from geoutils._dispatch import is_dask_array
from geoutils._misc import import_optional
from geoutils._typing import ArrayLike, DTypeLike, MArrayNum, NDArrayBool, NDArrayNum
from geoutils.multiproc import MultiprocConfig, compute_tiling
from geoutils.raster.array import get_mask_from_array

if TYPE_CHECKING:
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterBase, RasterLike
    from geoutils.vector.base import VectorLike

# Import Dask as optional dependency
try:
    import dask
    import dask.array as da
    from dask import delayed
    from dask.utils import cached_cumsum
except ImportError:

    da = None

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
    subsample: int | float,
    total_nb_valids: int,
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
                f"Argument ``subsample`` with value {subsample} is larger than the number of valid pixels of "
                f"{total_nb_valids},"
                f" using all valid pixels as a subsample.",
                category=UserWarning,
            )
    else:
        raise ValueError("Argument ``subsample`` must be > 0.")

    return npoints


######################
# NumPy implementation
######################


def _splitmix64(x: np.typing.NDArray[np.uint64]) -> NDArrayNum:
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

    # Force input to uint64 to avoid accidental casting
    x = np.asarray(x, dtype=np.uint64)
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)

    # Add a large odd constant derived from the golden ratio
    # This ensures that consecutive inputs do not map to related outputs
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & mask

    # First mixing step: XOR-shift to spread high bits into low bits, then multiply by a chosen odd constant
    z = x
    z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9)  # type: ignore[assignment]
    z &= mask

    # Second mixing step: Another XOR-shift followed by multiplication with a different constant
    # The constants were empirically chosen to achieve strong avalanche properties (each input bit affects
    # many output bits)
    z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB)  # type: ignore[assignment]
    z &= mask

    # Final XOR-shift to finish diffusion
    z = z ^ (z >> 31)  # type: ignore[assignment]

    return z.astype(np.uint64, copy=False)


@overload
def _subsample_numpy(
    array: NDArrayNum | MArrayNum,
    subsample: float | int,
    return_indices: Literal[False] = False,
    *,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "sequential",
    mask: NDArrayBool | None = None,
) -> NDArrayNum: ...


@overload
def _subsample_numpy(
    array: NDArrayNum | MArrayNum,
    subsample: float | int,
    return_indices: Literal[True],
    *,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "sequential",
    mask: NDArrayBool | None = None,
) -> tuple[NDArrayNum, ...]: ...


def _subsample_numpy(
    array: NDArrayNum | MArrayNum,
    subsample: float | int,
    return_indices: bool = False,
    *,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "sequential",
    mask: NDArrayBool | None = None,
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
    :param mask: Prepared boolean eligibility mask with the same shape as array.

    :returns: The subsampled array (1D) or the indices to extract (same shape as input array).
    """

    # Determine valid pixels and their global linear indices (row * nx + col)
    valid = ~get_mask_from_array(array).reshape(array.shape)
    if mask is not None:
        valid &= mask
    valids = np.flatnonzero(valid.ravel())  # Robust 1D index list (global linear indices)
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

        rng = np.random.default_rng(random_state)
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
        sel = sel[np.lexsort((gids[sel], keys[sel]))]  # Stable: key then gid
        chosen = valids[sel]

        # Unravel indexes, and return values or indexes
        unraveled = np.unravel_index(chosen, array.shape)
        return unraveled if return_indices else array[unraveled]

    else:
        raise ValueError(f"Unknown ``strategy`` {strategy!r}. Choose 'sequential' or 'topk'.")


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


def _valid_subsample_mask(arr_chunk: NDArrayNum | NDArrayBool, mask_chunk: NDArrayBool | None = None) -> NDArrayBool:
    """Find finite, unmasked values eligible for sampling, or eligible True values for boolean data."""

    valid = ~get_mask_from_array(arr_chunk).reshape(arr_chunk.shape)
    if arr_chunk.dtype == np.bool_:
        valid &= np.ma.getdata(arr_chunk)
    if mask_chunk is not None:
        valid &= mask_chunk
    return valid


@delayed
def _delayed_nb_valids(arr_chunk: NDArrayNum | NDArrayBool, *, mask_chunk: NDArrayBool | None = None) -> NDArrayNum:
    """Count number of valid values per block."""
    valid = _valid_subsample_mask(arr_chunk, mask_chunk)
    return np.array([np.count_nonzero(valid)]).reshape((1, 1))


@delayed
def _delayed_topk_candidates_block(
    arr_chunk: NDArrayNum | NDArrayBool,
    block_id: dict[str, Any],
    *,
    seed: int,
    k: int,
    nx_full: int,  # Width of full array
    return_indices_local: bool,
    mask_chunk: NDArrayBool | None = None,
) -> tuple[NDArrayNum, NDArrayNum | NDArrayBool]:
    """
    Return up to k valid samples from one block as (keys, payload).

    Those are:
    - keys: uint64 keys for selected valid pixels in this block
    - payload:
        * if return_indices_local=True: global linear indices (gid) of selected pixels (int64)
        * else: selected values from the array (dtype of arr_chunk, but typically float)
    """

    # Keep empty payloads in the same dtype as sampled values or original indices
    payload_dtype: DTypeLike = np.int64 if return_indices_local else arr_chunk.dtype

    # If no samples, return empty
    if k <= 0:
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=payload_dtype)

    # Only valid values are sampled (finite for numerical arrays, True for boolean arrays)
    valid = _valid_subsample_mask(arr_chunk, mask_chunk)

    # Get nonzero indices for flattened array, and number of valid values
    flat = np.flatnonzero(valid.ravel())
    nvalid = int(flat.size)

    # If no valid
    if nvalid == 0:
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=payload_dtype)

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
    keys_list: list[NDArrayNum],
    payload_list: list[NDArrayNum],
    *,
    k: int,
) -> tuple[NDArrayNum, NDArrayNum]:
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
def _delayed_gid_to_rc(gid: NDArrayNum, nx_full: int) -> tuple[NDArrayNum, NDArrayNum]:
    """
    Convert global linear indices back to (row, col) indices.
    """
    gid = np.asarray(gid, dtype=np.int64).ravel()
    r = gid // np.int64(nx_full)
    c = gid - r * np.int64(nx_full)
    return r.astype(np.int64), c.astype(np.int64)


@delayed
def _delayed_subsample_block(
    arr_chunk: NDArrayNum | NDArrayBool, subsample_indices: NDArrayNum, *, mask_chunk: NDArrayBool | None = None
) -> NDArrayNum | NDArrayBool:
    """Subsample the valid values at the corresponding 1D valid indices per block."""

    valid = _valid_subsample_mask(arr_chunk, mask_chunk)
    return arr_chunk[valid][subsample_indices]


@delayed
def _delayed_subsample_indices_block(
    arr_chunk: NDArrayNum | NDArrayBool,
    subsample_indices: NDArrayNum,
    block_id: dict[str, Any],
    *,
    mask_chunk: NDArrayBool | None = None,
) -> NDArrayNum:
    """Return 2D indices from the subsampled 1D valid indices per block."""

    #  Unravel indices of valid data to the shape of the block
    valid = _valid_subsample_mask(arr_chunk, mask_chunk)
    ix, iy = np.unravel_index(np.argwhere(valid.flatten())[subsample_indices], shape=arr_chunk.shape)

    # Convert to full-array indexes by adding the row and column starting indexes for this block
    ix += block_id["row_start"]
    iy += block_id["col_start"]

    return np.hstack((ix, iy))


def _dask_subsample(
    darr: da.Array,
    subsample: int | float = 1,
    return_indices: bool = False,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "sequential",
    preserve_order: bool = False,
    *,
    mask: da.Array | None = None,
) -> da.Array | tuple[da.Array, da.Array]:
    """
    Subsample valid values out-of-memory from a 2D Dask array.

    Strategy "topk" is chunk-invariant (same sample no matter chunk size, and same as the NumPy implementation), while
    "sequential" is chunk-dependent but slightly faster.

    Returns a delayed subsampled Dask array of the output (either values or indices).

    :param preserve_order: Restore sequential random-draw order after collecting block results. The one-dimensional
        sampling adapter uses this to match NumPy point sampling; existing raster calls keep block order.
    :param mask: Boolean array marking values eligible for sampling, with the same shape as darr.
    """

    # To raise appropriate error on missing optional dependency
    import_optional("dask")

    # Get random state
    # For method="sequential", we use the RNG stream based on valid orders (chunk-dependent)
    # For method="topk", we convert random_state into an integer seed used in the deterministic key function
    rng = np.random.default_rng(random_state)

    # Create a delayed object for each block, and flatten the blocks into a 1d shape
    blocks = darr.to_delayed().ravel()

    # Give each data block the matching mask cells without loading either array
    mask_blocks = [None] * len(blocks)
    if mask is not None:
        mask_blocks = da.asarray(mask).rechunk(darr.chunks).to_delayed().ravel().tolist()

    # Compute number of valid points for each block out-of-memory
    list_delayed_valids = [
        da.from_delayed(_delayed_nb_valids(b, mask_chunk=m), shape=(1, 1), dtype=np.dtype("int32"))
        for b, m in zip(blocks, mask_blocks)
    ]
    # Compute once, then flatten
    nb_valids_per_block = np.concatenate([x.ravel() for x in dask.compute(*list_delayed_valids)], axis=0).astype(
        np.int64
    )

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
        {"row_start": starts[0][indexes_yb[i]], "col_start": starts[1][indexes_xb[i]]} for i in range(len(blocks))
    ]

    # STRATEGY 1: "sequential" (chunk-dependent)
    if strategy == "sequential" or (preserve_order and subsample == 1):

        # Get random 1D indexes for the subsample size
        indices_1d = (
            np.arange(total_nb_valids)
            if preserve_order and subsample == 1
            else rng.choice(total_nb_valids, subsample_size, replace=False)
        )
        # Block selection sorts valid positions; recover the original draw order only when requested
        draw_order = np.argsort(np.argsort(indices_1d)) if preserve_order else slice(None)

        # Sort which indexes belong to which chunk
        ind_per_block = _get_indices_block_per_subsample(
            indices_1d, num_chunks=darr.numblocks, nb_valids_per_block=nb_valids_per_block.tolist()
        )

        # To just get the subsample without indices
        if not return_indices:
            # Task a delayed subsample to be computed for each block, skipping blocks with no values to sample
            used = [i for i in range(len(blocks)) if len(ind_per_block[i]) > 0]
            list_subsamples = [
                _delayed_subsample_block(
                    blocks[i], np.asarray(ind_per_block[i], dtype=np.int64), mask_chunk=mask_blocks[i]
                )
                for i in used
            ]

            # Cast output to the right expected dtype and length, then compute and concatenate
            list_subsamples_da = [
                da.from_delayed(s, shape=(len(ind_per_block[i]),), dtype=darr.dtype)
                for s, i in zip(list_subsamples, used)
            ]
            return da.concatenate(list_subsamples_da, axis=0)[draw_order]

        # To return indices
        else:
            # Task delayed subsample indices to be computed for each block, skipping blocks with no values to sample
            used = [i for i in range(len(blocks)) if len(ind_per_block[i]) > 0]
            list_subsample_indices = [
                _delayed_subsample_indices_block(
                    blocks[i],
                    np.asarray(ind_per_block[i], dtype=np.int64),
                    block_id=block_ids[i],
                    mask_chunk=mask_blocks[i],
                )
                for i in used
            ]

            # Cast output to the right expected dtype and length, then compute and concatenate
            list_indices_da = [
                da.from_delayed(s, shape=(len(ind_per_block[i]), 2), dtype=np.int32)
                for s, i in zip(list_subsample_indices, used)
            ]
            indices = da.concatenate(list_indices_da, axis=0)[draw_order]
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
                mask_chunk=mask_blocks[i],
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
        raise ValueError(f"Unknown ``strategy`` {strategy!r}, available strategies are 'sequential' or 'topk'.")


################################
# Multiprocessing implementation
################################


def _read_subsample_raster_block(
    rst: RasterBase,
    tile_idx: NDArrayNum,
    *,
    band: int = 1,
    mask: RasterLike | VectorLike | ArrayLike | None = None,
) -> tuple[NDArrayNum | NDArrayBool | MArrayNum, NDArrayBool]:
    """
    Read one band and its eligible cells from a raster tile without changing the source.

    Crop raster masks and slice array masks to the same tile; evaluate vector masks from its coordinates.
    Both sampling passes use this helper so their finite counts and selected values use the same mask.
    Keep the original values and dtype separate from eligibility so integer data need no NaN conversion.
    """

    from geoutils._dispatch import _get_raster_interface, _is_raster, has_geo_attr
    from geoutils.sampling.support import _as_array, _mask_at_support

    # Read the tile and select its requested band without replacing the source raster's band selection
    window = (tile_idx[2], tile_idx[0], tile_idx[3], tile_idx[1])
    rst_block = _get_raster_interface(rst.icrop(window))
    data = _as_array(rst_block.data)
    arr = data if data.ndim == 2 else data[band - 1]

    # Exclude nodata and nonfinite values; boolean rasters sample only their True cells
    valid = _valid_subsample_mask(arr)

    # Read only the corresponding mask window, then combine it with the finite cells in this tile
    if mask is not None:
        if _is_raster(mask):
            mask = _get_raster_interface(mask).icrop(window)
        elif not has_geo_attr(mask, "create_mask", accessors=("vct",)):
            mask = _as_array(mask)[tile_idx[0] : tile_idx[1], tile_idx[2] : tile_idx[3]]
        valid &= cast(NDArrayBool, _mask_at_support(mask, rst_block))
    return arr, valid


def _wrapper_multiproc_nb_valids_per_block(
    rst: RasterBase,
    tile_idx: NDArrayNum,
    *,
    band: int = 1,
    mask: RasterLike | VectorLike | ArrayLike | None = None,
) -> int:
    """Count valid values in one tile out-of-memory."""

    _, valid = _read_subsample_raster_block(rst, tile_idx, band=band, mask=mask)
    return int(np.count_nonzero(valid))


def _wrapper_multiproc_subsample_values_block(
    rst: RasterBase,
    tile_idx: NDArrayNum,
    subsample_indices_rel: NDArrayNum,
    *,
    band: int = 1,
    mask: RasterLike | VectorLike | ArrayLike | None = None,
) -> NDArrayNum | NDArrayBool | MArrayNum:
    """
    Subsample values in one tile using 1D indices relative to the tile's valid-value list.
    """

    # Get tile out-of-memory
    arr, valid = _read_subsample_raster_block(rst, tile_idx, band=band, mask=mask)

    # Return subsample of finite values (or True values for boolean input)
    return arr[valid].ravel()[subsample_indices_rel]


def _wrapper_multiproc_subsample_indices_block(
    rst: RasterBase,
    tile_idx: NDArrayNum,
    subsample_indices_rel: NDArrayNum,
    *,
    band: int = 1,
    mask: RasterLike | VectorLike | ArrayLike | None = None,
) -> NDArrayNum:
    """
    Return indices of the sampled valid pixels in one tile.

    Output shape: (n, 2) with columns [row, col] in full-array coordinates.
    """

    # Get tile out-of-memory
    arr, valid = _read_subsample_raster_block(rst, tile_idx, band=band, mask=mask)

    # Get starting row/col of the tile
    row0 = int(tile_idx[0])
    col0 = int(tile_idx[2])

    # Get relative indices of finite values (or True for boolean)
    flat_valid = np.flatnonzero(valid.ravel())

    # Use input to draw them
    flat_sel = flat_valid[subsample_indices_rel.astype(np.int64)]

    # Transform back into absolute indices
    ncols = int(arr.shape[1])
    r = (flat_sel // ncols).astype(np.int64) + row0
    c = (flat_sel - (flat_sel // ncols) * ncols).astype(np.int64) + col0

    return np.stack((r, c), axis=1)


def _wrapper_multiproc_topk_candidates_block(
    rst: RasterBase,
    tile_idx: NDArrayNum,
    *,
    seed: int,
    k: int,
    nx_full: int,
    return_indices: bool,
    band: int = 1,
    mask: RasterLike | VectorLike | ArrayLike | None = None,
) -> tuple[NDArrayNum, NDArrayNum | NDArrayBool | MArrayNum]:
    """
    Return up to k candidates from one tile as (keys, payload).

    keys: uint64 keys for selected valid pixels in this tile.
    payload:
      - if return_indices=True: global linear indices (gid = row*nx_full + col) (int64)
      - else: sampled values (array dtype)
    """
    # If no subsample, early return
    if k <= 0:
        payload_dtype = np.int64 if return_indices else rst.dtype
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=payload_dtype)

    # Get tile out-of-memory
    arr, valid = _read_subsample_raster_block(rst, tile_idx, band=band, mask=mask)

    # Tile offsets in full-array indices
    row0 = int(tile_idx[0])
    col0 = int(tile_idx[2])

    # Get valids indices
    flat = np.flatnonzero(valid.ravel())
    nvalid = int(flat.size)

    # If no valid, early return
    if nvalid == 0:
        payload_dtype = np.int64 if return_indices else arr.dtype
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=payload_dtype)

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
    vals: NDArrayNum | NDArrayBool | MArrayNum
    if np.issubdtype(arr.dtype, np.bool_):
        vals = np.ones(m, dtype=np.bool_)
    else:
        vals = arr.ravel()[flat[sel]]
    return key_sel, vals


def _multiproc_subsample(
    rst: RasterBase,
    config: MultiprocConfig,
    subsample: int | float = 1,
    return_indices: bool = False,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "sequential",
    *,
    band: int = 1,
    mask: RasterLike | VectorLike | ArrayLike | None = None,
) -> NDArrayNum | tuple[NDArrayNum, NDArrayNum]:
    """
    Subsample valid values out-of-memory from a 2D raster array using Multiprocessing tasks.

    Strategy "topk" is chunk-invariant (same sample no matter chunk size, and same as the NumPy implementation), while
    "sequential" is chunk-dependent but slightly faster.

    Returns a concatenated subsampled NumPy array collected from all tasks (either values or indices).
    The mask must already match the source grid; each worker reads only its corresponding window.
    """

    # Get tiling
    tiling = compute_tiling(tile_size=config.chunks, raster_shape=rst.shape, overlap=0)

    # Get number of chunks and blocks
    num_chunks = (tiling.shape[0], tiling.shape[1])
    num_blocks = int(np.prod(num_chunks))

    # Flatten tile_idx list in row-major block order
    indexes_row, indexes_col = np.unravel_index(np.arange(num_blocks), shape=num_chunks)
    tile_ids = [tiling[indexes_row[i], indexes_col[i], :] for i in range(num_blocks)]

    # Count valid values per tile in parallel
    tasks = [
        config.cluster.submit(_wrapper_multiproc_nb_valids_per_block, rst, tile_ids[i], band=band, mask=mask)
        for i in range(num_blocks)
    ]
    try:
        nb_valids_per_block = np.array(config.cluster.gather(tasks), dtype=np.int64)
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
                config.cluster.submit(
                    _wrapper_multiproc_subsample_values_block,
                    rst,
                    tile_ids[i],
                    np.asarray(ind_per_block[i], dtype=np.int64),
                    band=band,
                    mask=mask,
                )
                for i in used
            ]

            try:
                list_vals = config.cluster.gather(tasks)
            except Exception as e:
                raise RuntimeError(f"Error retrieving subsampled values from multiprocessing tasks: {e}")

            # Concatenate in tile order (this yields deterministic order given tiling; not random order)
            return np.concatenate(list_vals, axis=0)

        else:
            tasks = [
                config.cluster.submit(
                    _wrapper_multiproc_subsample_indices_block,
                    rst,
                    tile_ids[i],
                    np.asarray(ind_per_block[i], dtype=np.int64),
                    band=band,
                    mask=mask,
                )
                for i in used
            ]

            try:
                list_rc = config.cluster.gather(tasks)  # each (n_i, 2)
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
            config.cluster.submit(
                _wrapper_multiproc_topk_candidates_block,
                rst,
                tile_ids[i],
                seed=seed,
                k=subsample_size,
                nx_full=nx_full,
                return_indices=return_indices,
                band=band,
                mask=mask,
            )
            for i in range(num_blocks)
        ]

        try:
            cand = config.cluster.gather(tasks)  # list of (keys, payload)
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
        raise ValueError(f"Unknown ``strategy`` {strategy!r}. Choose 'sequential' or 'topk'.")


######################################################
# Wrapper dispatching to NumPy or Dask/Multiprocessing
######################################################


def _subsample(
    source_raster: RasterBase,
    subsample: float | int = 1,
    band: int = 1,
    return_indices: bool = False,
    *,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "sequential",
    mp_config: MultiprocConfig | None = None,
    mask: RasterLike | VectorLike | ArrayLike | None = None,
) -> Any:
    """
    Subsample an array at valid values, dispatching automatically to NumPy, Dask or Multiprocessing implementation.

    _mask_at_support() places masks on the source grid before the NumPy and Dask samplers count eligible values.
    _multiproc_subsample() instead places masks within each tile to keep unloaded inputs out of memory.
    All paths keep eligibility separate from data values so masking preserves the sampled dtype and indices.

    :param source_raster: Raster or raster accessor providing band values and their optional Dask chunks.
    :param subsample: Positive fraction of finite values at most one, or maximum number of values above one.
    :param band: Raster band to subsample, counting from one.
    :param return_indices: If True, return (rows, cols) indices instead of values.
    :param random_state: Seed or Generator.
    :param strategy: Either "sequential" (chunk/order dependent) or "topk" (chunk-invariant).
    :param mp_config: Tile sizes and worker cluster for multiprocessing. Cannot be combined with a Dask source.
    :param mask: Boolean array, aligned mask raster, or vector geometries restricting eligible cells.

    :returns: One-dimensional sampled values, or a tuple of row and column index arrays. Dask generally returns
        lazy arrays after computing finite counts; an empty Dask sample returns NumPy arrays.
    """

    from geoutils._dispatch import _get_raster_interface, _is_raster, has_geo_attr
    from geoutils.sampling.support import (
        _as_array,
        _mask_at_support,
        _normalize_sampling_input,
    )

    # Check the selected band before reading data or submitting worker tasks
    if not 1 <= band <= source_raster.count:
        raise ValueError("Argument ``band`` must be between one and the raster band count.")

    # Cannot use Multiprocessing backend and Dask backend simultaneously
    mp_backend = mp_config is not None
    # The check below can only run on Xarray
    dask_backend = da is not None and source_raster._chunks is not None

    if mp_backend and dask_backend:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove ``mp_config`` parameter "
            "from subsample(). To use Multiprocessing, open the file without ``chunks``."
        )

    class _SubsampleKwargs(TypedDict):
        subsample: int | float
        return_indices: bool
        random_state: int | np.random.Generator | None
        strategy: Literal["sequential", "topk"]

    subsample_kwargs: _SubsampleKwargs = {
        "subsample": subsample,
        "return_indices": return_indices,
        "random_state": random_state,
        "strategy": strategy,
    }

    # Validate masks before worker dispatch, keeping spatial masks available for reading one tile at a time
    if mp_backend:
        assert mp_config is not None
        mask = _normalize_sampling_input(mask)
        if _is_raster(mask):
            mask_raster = _get_raster_interface(mask)
            if mask_raster._chunks is not None:
                raise ValueError("Cannot use Multiprocessing and Dask masks simultaneously.")
            if not source_raster.georeferenced_grid_equal(mask_raster):
                raise ValueError("Raster value ``mask`` does not share the selected support grid.")
            if not mask_raster.is_mask and not np.issubdtype(mask_raster.dtype, np.bool_):
                raise ValueError("Argument ``mask`` must be boolean and contain one value per input location.")
            mask = mask_raster
        elif mask is not None and not has_geo_attr(mask, "create_mask", accessors=("vct",)):
            mask = _mask_at_support(mask, source_raster)
            if is_dask_array(mask):
                raise ValueError("Cannot use Multiprocessing and Dask masks simultaneously.")
        # Restrict disk reads to the selected band through a shallow view, leaving the caller's band selection intact
        sampling_raster = source_raster
        sampling_band = band
        if not source_raster._is_xr and not source_raster.is_loaded:
            sampling_raster = source_raster.copy(deep=False)
            sampling_raster._bands = (source_raster.bands[band - 1],)
            sampling_band = 1
        return _multiproc_subsample(
            sampling_raster, config=mp_config, band=sampling_band, mask=mask, **subsample_kwargs
        )

    # Read one band without converting masked integer values to floating point
    data = _as_array(source_raster.data)
    arr = data if data.ndim == 2 else data[band - 1]
    mask_array = _mask_at_support(mask, source_raster)

    # Keep Dask data and mask blocks lazy; an eager source collects only a supplied lazy mask
    if dask_backend:
        return _dask_subsample(arr, mask=mask_array, **subsample_kwargs)
    if mask_array is not None and is_dask_array(mask_array):
        mask_array = mask_array.compute()
    return _subsample_numpy(arr, mask=mask_array, **subsample_kwargs)  # type: ignore


def _sample_valid_indices(
    valid: Any,
    *,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    strategy: Literal["sequential", "topk"],
) -> tuple[np.typing.NDArray[np.int64], ...]:
    """
    Choose eligible array positions without collecting a complete lazy validity mask.

    Sampling options follow _subsample(). One-dimensional inputs use a single-column grid so Dask's existing block
    sampler can work on point rows. This preserves global row positions, topk keys and NumPy's sequential draw order.
    Two-dimensional sequential sampling keeps the existing Dask block order.

    :param valid: One- or two-dimensional boolean NumPy or Dask array marking eligible locations.
    :returns: Computed integer position arrays, one per input dimension.
    """

    # Accept point rows or grid cells, the dimensions supported by the shared samplers
    if valid.ndim not in (1, 2):
        raise ValueError("Valid sampling locations must form a one- or two-dimensional array.")

    # Dask's sampler already selects True cells; avoid expanding boolean blocks into floating-point arrays
    if is_dask_array(valid):
        grid = valid[:, None] if valid.ndim == 1 else valid
        indexes = _dask_subsample(
            grid,
            subsample=subsample,
            return_indices=True,
            random_state=random_state,
            strategy=strategy,
            preserve_order=valid.ndim == 1,
        )
        # Compute both coordinate arrays together so they share the same sampling tasks
        indexes = dask.compute(*indexes)
        if valid.ndim == 1:
            indexes = indexes[:1]
    else:
        # NumPy's sampler selects finite values, so represent excluded cells by NaN
        sampling_values = np.where(valid, 1.0, np.nan)
        indexes = _subsample_numpy(
            sampling_values,
            subsample=subsample,
            return_indices=True,
            random_state=random_state,
            strategy=strategy,
        )
    return tuple(np.asarray(index, dtype=np.int64) for index in indexes)


def _subsample_pointcloud(
    source_pointcloud: PointCloudBase,
    subsample: float | int,
    return_indices: bool = False,
    random_state: int | np.random.Generator | None = None,
    *,
    mask: RasterLike | PointCloudLike | VectorLike | ArrayLike | None = None,
) -> NDArrayNum | tuple[NDArrayNum, ...]:
    """
    Subsample finite point cloud values, gathering only selected values from a lazy data column.

    _dask_subsample() treats point rows as a single-column grid and preserves NumPy's random-draw order.
    _mask_at_support() places the optional mask on these rows before either sampler counts eligible values.
    Eager columns use _subsample_numpy() directly. Both paths return computed results without changing the value dtype.

    :param source_pointcloud: Point cloud or accessor whose selected data column supplies the sampled values.
    :param subsample: Positive fraction of finite values at most one, or maximum number of values above one.
    :param return_indices: Return a one-element tuple of row-position indices instead of sampled values.
    :param random_state: Integer seed or NumPy Generator used for the random draw.
    :param mask: Boolean array or spatial mask restricting eligible point rows.
    :returns: Computed one-dimensional NumPy values, or a one-element tuple of indices into the original row order.
    """

    from geoutils.sampling.support import _as_array, _mask_at_support

    # Read native values and reuse any known partition lengths when placing the mask on the same point rows
    data = _as_array(source_pointcloud.data)
    partition_lengths = tuple(int(length) for length in data.chunks[0]) if source_pointcloud._is_dask else None
    mask_array = _mask_at_support(mask, source_pointcloud, point_partition_lengths=partition_lengths)

    # Let the existing Dask sampler collect only the requested values or row positions from point partitions
    if source_pointcloud._is_dask:
        sampled = _dask_subsample(
            data[:, None],
            subsample=subsample,
            return_indices=return_indices,
            random_state=random_state,
            preserve_order=True,
            mask=None if mask_array is None else mask_array[:, None],
        )
        if return_indices:
            rows = sampled[0]
            return (rows.compute() if is_dask_array(rows) else rows,)
        assert not isinstance(sampled, tuple)
        return sampled.compute() if is_dask_array(sampled) else sampled

    # Preserve the established NumPy sampling rules for eager point data
    if mask_array is not None and is_dask_array(mask_array):
        mask_array = mask_array.compute()
    if return_indices:
        return _subsample_numpy(data, subsample, return_indices=True, random_state=random_state, mask=mask_array)
    return _subsample_numpy(data, subsample, return_indices=False, random_state=random_state, mask=mask_array)
