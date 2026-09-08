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

"""Subsampling tools shared by raster and point cloud data."""

from __future__ import annotations

import operator
import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict, overload

import numpy as np

from geoutils._misc import import_optional
from geoutils._typing import MArrayNum, NDArrayBool, NDArrayNum
from geoutils.multiproc import MultiprocConfig, compute_tiling
from geoutils.raster.array import get_mask_from_array

if TYPE_CHECKING:
    from geoutils.raster.raster import Raster, RasterBase

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


####################################
# 1/ SHARED SIZE AND NUMPY SAMPLING
####################################


def _get_subsample_size_from_user_input(
    subsample: int | float,
    total_nb_valids: int,
) -> int:
    """Turn a requested count or fraction into the number of available values to sample."""

    # Use values up to one as a fraction of all available values
    if (subsample <= 1) & (subsample > 0):
        npoints = int(subsample * total_nb_valids)
    # Use larger values as a requested count
    elif subsample > 1:
        # Return every available value when the requested count is larger
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


def _splitmix64(x: np.typing.NDArray[np.uint64]) -> NDArrayNum:
    """Give each unsigned integer a repeatable, well distributed sampling score.

    The `topk` strategy scores global cell numbers with this function. It therefore selects the same cells for any
    Dask chunk layout. We keep the small algorithm here because NumPy does not expose the equivalent internal step.

    References
    ----------
    - Steele et al., "Fast Splittable Pseudorandom Number Generators", OOPSLA 2014
      https://doi.org/10.1145/2660193.2660195
    - Sebastiano Vigna, SplitMix64 reference implementation, https://prng.di.unimi.it/splitmix64.c
    """

    # Use unsigned 64-bit arithmetic required by the published algorithm
    x = np.asarray(x, dtype=np.uint64)
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)

    # Shift consecutive cell numbers before mixing their bits
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & mask

    # Mix high and low bits so nearby cell numbers receive unrelated scores
    z = x
    z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9)  # type: ignore[assignment]
    z &= mask

    # Mix a second time so every input bit can affect the final score
    z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB)  # type: ignore[assignment]
    z &= mask

    # Apply the final bit shift from the reference implementation
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

    # Find available values and number their positions across the flattened array
    mask = get_mask_from_array(array)
    valids = np.flatnonzero(~mask.ravel())  # Robust 1D index list (global linear indices)
    total_nb_valids = int(valids.size)

    # Return the requested empty result when the array has no available value
    if total_nb_valids == 0:
        if return_indices:
            return tuple(np.array([], dtype=int) for _ in range(array.ndim))
        return np.array([], dtype=array.dtype)

    # Turn the caller's count or fraction into the final sample size
    subsample_size = _get_subsample_size_from_user_input(subsample=subsample, total_nb_valids=total_nb_valids)

    # Preserve the established meaning of one: return every available value
    if subsample == 1:
        unraveled = np.unravel_index(valids, array.shape)
        return unraveled if return_indices else array[unraveled]

    # Return an empty result when a small fraction rounds down to zero
    if subsample_size <= 0:
        if return_indices:
            return tuple(np.array([], dtype=int) for _ in range(array.ndim))
        return np.array([], dtype=array.dtype)

    # Draw directly from the available positions for the faster `sequential` strategy
    if strategy == "sequential":

        rng = np.random.default_rng(random_state)
        # Choose positions without replacement
        chosen = rng.choice(valids, subsample_size, replace=False)

        # Convert flat positions back to array indexes when requested
        unraveled = np.unravel_index(chosen, array.shape)
        return unraveled if return_indices else array[unraveled]

    # Score global positions for the `topk` strategy so chunk layout does not change the sample
    elif strategy == "topk":
        # Convert either random-state form to one integer seed
        if isinstance(random_state, np.random.Generator):
            seed = int(random_state.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        elif random_state is None:
            seed = 0
        else:
            seed = int(random_state)

        # Give every available position a repeatable random score
        gids = valids.astype(np.uint64)
        keys = _splitmix64(np.uint64(seed) ^ gids)

        # Keep the positions with the smallest scores in a stable order
        sel = np.argpartition(keys, subsample_size - 1)[:subsample_size]
        sel = sel[np.lexsort((gids[sel], keys[sel]))]  # Stable: key then gid
        chosen = valids[sel]

        # Convert flat positions back to array indexes when requested
        unraveled = np.unravel_index(chosen, array.shape)
        return unraveled if return_indices else array[unraveled]

    else:
        raise ValueError(f"Unknown strategy {strategy!r}. Choose 'sequential' or 'topk'.")


######################
# 2/ DASK SAMPLING
######################

# Dask cannot know how many available values each chunk will return before running it. We therefore use delayed
# tasks for results with different lengths: https://blog.dask.org/2021/07/02/ragged-output


def _get_indices_block_per_subsample(
    indices_1d: NDArrayNum, num_chunks: tuple[int, int], nb_valids_per_block: list[int]
) -> list[list[int]]:
    """Map selected positions in the full list of available values to positions within each Dask chunk.

    :param indices_1d: Subsample 1D indexes among a total number of valid values.
    :param num_chunks: Number of chunks in X and Y.
    :param nb_valids_per_block: Number of valid pixels per block.

    :returns: Relative 1D valid subsample index per block.
    """

    # Find where each chunk ends in the full list of available values
    valids_cumsum = np.cumsum(nb_valids_per_block)

    # Sort once so we can walk through chunks in order
    indices_1d = np.sort(indices_1d)

    # Create one list of selected positions for each chunk
    relative_index_per_block = [[] for _ in range(num_chunks[0] * num_chunks[1])]
    k = 0  # K is the block number
    for i in indices_1d:

        # Move to the chunk that contains this position in the full list
        while i >= valids_cumsum[k]:
            k += 1

        # Store the position relative to the start of that chunk's list of available values
        first_index_block = valids_cumsum[k - 1] if k >= 1 else 0  # The first 1D valid subsample index of the block
        relative_index = i - first_index_block
        relative_index_per_block[k].append(relative_index)

    return relative_index_per_block


@delayed
def _delayed_nb_valids(arr_chunk: NDArrayNum | NDArrayBool) -> NDArrayNum:
    """Count available values in one Dask chunk."""
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
) -> tuple[NDArrayNum, NDArrayNum | NDArrayBool]:
    """Return up to `k` available values or cell numbers with their sampling scores from one chunk."""

    # Return empty arrays when this task does not need a sample
    if k <= 0:
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=np.int64)

    # Treat finite numbers and true boolean cells as available values
    if np.issubdtype(arr_chunk.dtype, np.bool_):
        valid = arr_chunk
    else:
        valid = np.isfinite(arr_chunk)

    # Find available positions in the flattened chunk
    flat = np.flatnonzero(valid.ravel())
    nvalid = int(flat.size)

    # Return empty arrays when the chunk has no available value
    if nvalid == 0:
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=np.int64)

    # Convert flat chunk positions to rows and columns within the chunk
    ncols = int(arr_chunk.shape[1])
    r = flat // ncols
    c = flat - r * ncols

    # Add the chunk start to recover full array cell numbers
    row0 = int(block_id["row_start"])
    col0 = int(block_id["col_start"])
    gid = (row0 + r) * nx_full + (col0 + c)

    # Score each cell from only the seed and its full array position
    key = _splitmix64(np.uint64(seed) ^ gid.astype(np.uint64))

    # Keep the smallest scores needed from this chunk
    m = min(int(k), nvalid)
    sel = np.argpartition(key, m - 1)[:m]
    key_sel = key[sel]

    # Return full array cell numbers when requested
    if return_indices_local:
        gid_sel = gid[sel]
        return key_sel, gid_sel
    # Otherwise return the selected values
    else:
        # Boolean samples contain true values by definition
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
    """Combine chunk results and keep the `k` smallest scores across the full array."""

    # Return empty arrays when there are no chunk results
    if len(keys_list) == 0:
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=np.int64)

    # Join the scores and matching values or cell numbers from every chunk
    keys = np.concatenate([np.asarray(x, dtype=np.uint64).ravel() for x in keys_list], axis=0)
    payload = np.concatenate([np.asarray(x).ravel() for x in payload_list], axis=0)

    # Return the joined empty arrays when no chunk found an available value
    n = int(keys.size)
    if n == 0:
        return keys, payload

    # Keep at most the requested number of scores
    m = min(int(k), n)

    # Find the smallest scores without sorting values that will be discarded
    sel = np.argpartition(keys, m - 1)[:m]

    # Sort the kept scores so repeated calls return the same order
    sel = sel[np.argsort(keys[sel])]

    # Return the kept scores with their matching values or cell numbers
    return keys[sel], payload[sel]


@delayed
def _delayed_gid_to_rc(gid: NDArrayNum, nx_full: int) -> tuple[NDArrayNum, NDArrayNum]:
    """Convert flattened full array positions back to row and column numbers."""
    gid = np.asarray(gid, dtype=np.int64).ravel()
    r = gid // np.int64(nx_full)
    c = gid - r * np.int64(nx_full)
    return r.astype(np.int64), c.astype(np.int64)


@delayed
def _delayed_subsample_block(
    arr_chunk: NDArrayNum | NDArrayBool, subsample_indices: NDArrayNum
) -> NDArrayNum | NDArrayBool:
    """Read selected positions from one chunk's list of available values."""

    if arr_chunk.dtype == np.bool_:
        return arr_chunk[arr_chunk][subsample_indices]
    return arr_chunk[np.isfinite(arr_chunk)][subsample_indices]


@delayed
def _delayed_subsample_indices_block(
    arr_chunk: NDArrayNum | NDArrayBool, subsample_indices: NDArrayNum, block_id: dict[str, Any]
) -> NDArrayNum:
    """Convert selected positions in one chunk's available values to full array rows and columns."""

    if arr_chunk.dtype == np.bool_:
        ix, iy = np.unravel_index(np.argwhere(arr_chunk.flatten())[subsample_indices], shape=arr_chunk.shape)
    else:
        # Convert selected flat positions to rows and columns within the chunk
        ix, iy = np.unravel_index(
            np.argwhere(np.isfinite(arr_chunk.flatten()))[subsample_indices], shape=arr_chunk.shape
        )

    # Add the chunk start to recover full array rows and columns
    ix += block_id["row_start"]
    iy += block_id["col_start"]

    return np.hstack((ix, iy))


def _dask_subsample(
    darr: da.Array,
    subsample: int | float = 1,
    return_indices: bool = False,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "sequential",
) -> da.Array | tuple[da.Array, da.Array]:
    """Sample available values from a 2D Dask array without loading the full array.

    `topk` matches NumPy for any chunk layout. `sequential` is slightly faster, but changing chunks can change its
    sample. The result stays lazy and contains either values or row and column numbers.
    """

    # Raise the standard optional package error before building Dask tasks
    import_optional("dask")

    # Prepare the random generator used by the faster sequential strategy
    rng = np.random.default_rng(random_state)

    # Get one delayed object per chunk in row-major order
    blocks = darr.to_delayed().ravel()

    # Count available values in each chunk without loading the full array
    list_delayed_valids = [
        da.from_delayed(_delayed_nb_valids(b), shape=(1, 1), dtype=np.dtype("int32")) for b in blocks
    ]
    # Run all count tasks once and store one count per chunk
    nb_valids_per_block = np.concatenate([x.ravel() for x in dask.compute(*list_delayed_valids)], axis=0).astype(
        np.int64
    )

    # Add chunk counts to get the full number of available values
    total_nb_valids = int(np.sum(nb_valids_per_block))

    # Turn the caller's count or fraction into the final sample size
    subsample_size = _get_subsample_size_from_user_input(subsample=subsample, total_nb_valids=total_nb_valids)

    # Return the requested empty result when no value can be sampled
    if subsample_size <= 0 or total_nb_valids <= 0:
        if return_indices:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
        else:
            return np.empty((0,), dtype=darr.dtype)

    # 1/ Find the first full array row and column of every chunk
    # This follows the chunk-location data passed by dask.array.map_blocks():
    # https://github.com/dask/dask/blob/24493f58660cb933855ba7629848881a6e2458c1/dask/array/core.py#L908
    # cached_cumsum() also returns the unused array end after the final chunk
    starts = [cached_cumsum(c, initial_zero=True) for c in darr.chunks]
    num_chunks = darr.numblocks

    # Match each flattened delayed chunk with its row and column in the chunk grid
    indexes_yb, indexes_xb = np.unravel_index(np.arange(len(blocks)), shape=(num_chunks[0], num_chunks[1]))

    block_ids = [
        {"row_start": starts[0][indexes_yb[i]], "col_start": starts[1][indexes_xb[i]]} for i in range(len(blocks))
    ]

    # 2a/ Draw positions from the combined list of available values for `sequential`
    if strategy == "sequential":

        # Draw positions without replacement
        indices_1d = rng.choice(total_nb_valids, subsample_size, replace=False)

        # Map each selected position to its chunk and its position within that chunk
        ind_per_block = _get_indices_block_per_subsample(
            indices_1d, num_chunks=darr.numblocks, nb_valids_per_block=nb_valids_per_block.tolist()
        )

        # Read selected values when the caller does not request their locations
        if not return_indices:
            # Create tasks only for chunks that contain a selected value
            used = [i for i in range(len(blocks)) if len(ind_per_block[i]) > 0]
            list_subsamples = [
                _delayed_subsample_block(blocks[i], np.asarray(ind_per_block[i], dtype=np.int64)) for i in used
            ]

            # Give Dask each task's known size and join the lazy results
            list_subsamples_da = [
                da.from_delayed(s, shape=(len(ind_per_block[i]),), dtype=darr.dtype)
                for s, i in zip(list_subsamples, used)
            ]
            return da.concatenate(list_subsamples_da, axis=0)

        # Convert selected positions to full array rows and columns when requested
        else:
            # Create tasks only for chunks that contain a selected position
            used = [i for i in range(len(blocks)) if len(ind_per_block[i]) > 0]
            list_subsample_indices = [
                _delayed_subsample_indices_block(
                    blocks[i], np.asarray(ind_per_block[i], dtype=np.int64), block_id=block_ids[i]
                )
                for i in used
            ]

            # Give Dask each task's known size and join the lazy row and column pairs
            list_indices_da = [
                da.from_delayed(s, shape=(len(ind_per_block[i]), 2), dtype=np.int32)
                for s, i in zip(list_subsample_indices, used)
            ]
            indices = da.concatenate(list_indices_da, axis=0)
            return indices[:, 0], indices[:, 1]

    # 2b/ Score full array cell numbers for `topk` so chunk layout does not change the sample
    elif strategy == "topk":

        # Convert either random-state form to one integer seed
        if isinstance(random_state, np.random.Generator):
            seed = int(random_state.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        elif random_state is None:
            seed = 0
        else:
            seed = int(random_state)

        # Keep the full array width needed to flatten row and column numbers
        nx_full = int(darr.shape[1])

        # Ask each chunk for up to the requested number of smallest scores
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

        # Separate scores from their matching values or cell numbers
        keys_list = [c[0] for c in cands]
        payload_list = [c[1] for c in cands]

        # Keep the smallest scores across all chunks
        merged = _delayed_merge_topk(keys_list, payload_list, k=subsample_size)

        # Keep the merged values or cell numbers as a delayed result
        payload_delayed = dask.delayed(operator.getitem)(merged, 1)

        if not return_indices:
            # Tell Dask the final value count, which was fixed after counting available cells
            return da.from_delayed(payload_delayed, shape=(subsample_size,), dtype=darr.dtype)

        else:
            # Convert flattened full array positions to lazy row and column arrays
            rr_cc = _delayed_gid_to_rc(payload_delayed, nx_full)
            rr_delayed = dask.delayed(operator.getitem)(rr_cc, 0)
            cc_delayed = dask.delayed(operator.getitem)(rr_cc, 1)

            rr = da.from_delayed(rr_delayed, shape=(subsample_size,), dtype=np.int64)
            cc = da.from_delayed(cc_delayed, shape=(subsample_size,), dtype=np.int64)
            return rr, cc

    else:
        raise ValueError(f"Unknown strategy {strategy!r}, available strategies are 'sequential' or 'topk'.")


###############################
# 3/ MULTIPROCESSING SAMPLING
###############################


def _wrapper_multiproc_nb_valids_per_block(rst: Raster, tile_idx: NDArrayNum) -> int:
    """Read one raster tile and count its available values."""
    rst_block = rst.icrop((tile_idx[2], tile_idx[0], tile_idx[3], tile_idx[1]))
    arr = rst_block.data

    if np.issubdtype(arr.dtype, np.bool_):
        return int(np.count_nonzero(arr))
    return int(np.count_nonzero(~get_mask_from_array(arr)))


def _wrapper_multiproc_subsample_values_block(
    rst: Raster,
    tile_idx: NDArrayNum,
    subsample_indices_rel: NDArrayNum,
) -> NDArrayNum:
    """Read selected positions from one tile's list of available values."""

    # Read only this tile from the raster
    rst_block = rst.icrop((tile_idx[2], tile_idx[0], tile_idx[3], tile_idx[1]))
    arr = rst_block.data

    # Return finite numbers or true boolean cells at the selected positions
    if np.issubdtype(arr.dtype, np.bool_):
        return arr[arr].ravel()[subsample_indices_rel]
    return arr[np.isfinite(arr)].ravel()[subsample_indices_rel]


def _wrapper_multiproc_subsample_indices_block(
    rst: Raster,
    tile_idx: NDArrayNum,
    subsample_indices_rel: NDArrayNum,
) -> NDArrayNum:
    """Return full raster rows and columns for selected available values in one tile."""

    # Read only this tile from the raster
    rst_block = rst.icrop((tile_idx[2], tile_idx[0], tile_idx[3], tile_idx[1]))
    arr = rst_block.data

    # Record where this tile starts in the full raster
    row0 = int(tile_idx[0])
    col0 = int(tile_idx[2])

    # Find finite numbers or true boolean cells inside the tile
    if np.issubdtype(arr.dtype, np.bool_):
        flat_valid = np.flatnonzero(arr.ravel())
    else:
        flat_valid = np.flatnonzero(np.isfinite(arr).ravel())

    # Select the requested positions in the tile's list of available values
    flat_sel = flat_valid[subsample_indices_rel.astype(np.int64)]

    # Convert tile positions to full raster rows and columns
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
) -> tuple[NDArrayNum, NDArrayNum | NDArrayBool]:
    """Return up to `k` available values or cell numbers with their sampling scores from one raster tile."""

    # Return empty arrays when this task does not need a sample
    if k <= 0:
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=np.int64)

    # Read only this tile from the raster
    rst_block = rst.icrop((tile_idx[2], tile_idx[0], tile_idx[3], tile_idx[1]))
    arr = rst_block.data

    # Record where this tile starts in the full raster
    row0 = int(tile_idx[0])
    col0 = int(tile_idx[2])

    # Find finite numbers or true boolean cells inside the tile
    if np.issubdtype(arr.dtype, np.bool_):
        valid = arr
    else:
        valid = np.isfinite(arr)
    flat = np.flatnonzero(valid.ravel())
    nvalid = int(flat.size)

    # Return empty arrays when the tile has no available value
    if nvalid == 0:
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=np.int64)

    # Convert flat tile positions to rows and columns within the tile
    ncols = int(arr.shape[1])
    r = flat // ncols
    c = flat - r * ncols

    # Convert full raster rows and columns to one flat cell number
    gid = (np.int64(row0) + r.astype(np.int64)) * np.int64(nx_full) + (np.int64(col0) + c.astype(np.int64))
    # Score each cell from only the seed and its full raster position
    key = _splitmix64(np.uint64(seed) ^ gid.astype(np.uint64))

    # Keep the smallest scores needed from this tile
    m = min(int(k), nvalid)
    sel = np.argpartition(key, m - 1)[:m]
    key_sel = key[sel]

    # Return full raster cell numbers when requested
    if return_indices:
        return key_sel, gid[sel]

    # Otherwise return the selected values
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
) -> NDArrayNum | tuple[NDArrayNum, NDArrayNum]:
    """Sample available raster values by reading separate tiles in worker processes.

    `topk` matches NumPy for any tile size. `sequential` is slightly faster, but changing tile sizes can change its
    sample. The function joins worker results into values or full raster row and column numbers.
    """

    # Split the raster into the requested worker tiles
    tiling = compute_tiling(tile_size=config.chunks, raster_shape=rst.shape, overlap=0)

    # Count tiles along each raster direction
    num_chunks = (tiling.shape[0], tiling.shape[1])
    num_blocks = int(np.prod(num_chunks))

    # List tiles from left to right and top to bottom
    indexes_row, indexes_col = np.unravel_index(np.arange(num_blocks), shape=num_chunks)
    tile_ids = [tiling[indexes_row[i], indexes_col[i], :] for i in range(num_blocks)]

    # Count available values in every tile using worker tasks
    tasks = [config.cluster.submit(_wrapper_multiproc_nb_valids_per_block, rst, tile_ids[i]) for i in range(num_blocks)]
    try:
        nb_valids_per_block = np.array(config.cluster.gather(tasks), dtype=np.int64)
    except Exception as e:
        raise RuntimeError(f"Error retrieving valid-count results from multiprocessing tasks: {e}")

    total_nb_valids = int(nb_valids_per_block.sum())

    # Turn the caller's count or fraction into the final sample size
    subsample_size = _get_subsample_size_from_user_input(subsample=subsample, total_nb_valids=total_nb_valids)

    # Return the requested empty result when no value can be sampled
    if subsample_size <= 0 or total_nb_valids <= 0:
        if return_indices:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
        return np.empty((0,), dtype=rst.dtype)

    # Draw positions from the combined list of available values for `sequential`
    if strategy == "sequential":
        rng = np.random.default_rng(random_state)

        # Draw positions without replacement
        indices_1d = rng.choice(total_nb_valids, subsample_size, replace=False)

        # Map each selected position to its tile and its position within that tile
        ind_per_block = _get_indices_block_per_subsample(
            indices_1d=indices_1d,
            num_chunks=num_chunks,
            nb_valids_per_block=nb_valids_per_block.tolist(),
        )

        used = [i for i in range(num_blocks) if len(ind_per_block[i]) > 0]

        # Read selected values in worker processes when locations are not requested
        if not return_indices:
            tasks = [
                config.cluster.submit(
                    _wrapper_multiproc_subsample_values_block,
                    rst,
                    tile_ids[i],
                    np.asarray(ind_per_block[i], dtype=np.int64),
                )
                for i in used
            ]

            try:
                list_vals = config.cluster.gather(tasks)
            except Exception as e:
                raise RuntimeError(f"Error retrieving subsampled values from multiprocessing tasks: {e}")

            # Join results in tile order so the same tile layout returns the same order
            return np.concatenate(list_vals, axis=0)

        else:
            tasks = [
                config.cluster.submit(
                    _wrapper_multiproc_subsample_indices_block,
                    rst,
                    tile_ids[i],
                    np.asarray(ind_per_block[i], dtype=np.int64),
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

    # Score full raster cell numbers for `topk` so tile size does not change the sample
    elif strategy == "topk":

        # Convert either random-state form to one integer seed
        if isinstance(random_state, np.random.Generator):
            seed = int(random_state.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        elif random_state is None:
            seed = 0
        else:
            seed = int(random_state)

        # Keep the raster width needed to flatten row and column numbers
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

        # Convert flattened full raster positions back to rows and columns
        gid = payload_sel.astype(np.int64)
        rows = gid // np.int64(nx_full)
        cols = gid - rows * np.int64(nx_full)
        return rows.astype(np.int64), cols.astype(np.int64)

    else:
        raise ValueError(f"Unknown strategy {strategy!r}. Choose 'sequential' or 'topk'.")


##########################
# 4/ PUBLIC METHOD ROUTING
##########################


def _subsample(
    source_raster: RasterBase,
    subsample: float | int = 1,
    band: int = 1,
    return_indices: bool = False,
    *,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "sequential",
    mp_config: MultiprocConfig | None = None,
) -> Any:
    """Run a raster's NumPy, Dask, or multiprocessing sampling path.

    _multiproc_subsample() reads separate raster tiles in workers. _dask_subsample() keeps chunked input lazy, and
    _subsample_numpy() handles data already held in memory.

    :param source_raster: Input array (NumPy/masked or Dask).
    :param subsample: Subsample size or fraction.
    :param band: Band to subsample.
    :param return_indices: If True, return (rows, cols) indices instead of values.
    :param random_state: Seed or Generator.
    :param strategy: Either "sequential" (depends on chunk order) or "topk" (same for every chunk layout).

    :returns:
      - values: 1D array of sampled values
      - indices: (rows, cols) (axis order)
      - for Dask input: lazy `da.Array` values or indexes
    """

    # Detect the one storage path that should perform the sample
    mp_backend = mp_config is not None
    dask_backend = da is not None and source_raster._chunks is not None

    if mp_backend and dask_backend:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove mp_config parameter "
            "from subsample(). To use Multiprocessing, open the file without chunks."
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

    # Read raster tiles in worker processes when requested
    if mp_backend:
        assert mp_config is not None
        # Expose only the selected band while worker tasks read raster tiles
        orig_bands = source_raster.bands
        source_raster._bands = (band,)
        try:
            return _multiproc_subsample(source_raster, config=mp_config, **subsample_kwargs)
        finally:
            source_raster._bands = orig_bands
    else:
        if source_raster.data.ndim != 2:
            arr = source_raster.data[band - 1, :, :]
        else:
            arr = source_raster.data
        # Keep Dask input lazy through the Dask sampling path
        if dask_backend:
            return _dask_subsample(arr, **subsample_kwargs)
        # Sample an in-memory array directly with NumPy
        else:
            return _subsample_numpy(arr, **subsample_kwargs)  # type: ignore


def _subsample_pointcloud(
    source_pointcloud: Any,
    subsample: float | int,
    return_indices: bool = False,
    random_state: int | np.random.Generator | None = None,
) -> NDArrayNum | tuple[NDArrayNum, ...]:
    """Load one point cloud value column and sample it with the shared NumPy path."""

    data = source_pointcloud.data.compute().values if source_pointcloud._is_dask else np.asarray(source_pointcloud.data)
    if return_indices:
        return _subsample_numpy(
            array=data,
            subsample=subsample,
            return_indices=True,
            random_state=random_state,
        )
    return _subsample_numpy(
        array=data,
        subsample=subsample,
        return_indices=False,
        random_state=random_state,
    )
