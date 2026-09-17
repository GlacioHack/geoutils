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

"""Subsample arrays, rasters and point clouds."""

from __future__ import annotations

import operator
import pathlib
import tempfile
import warnings
from collections.abc import Callable, Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast, overload

import affine
import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rasterio.crs import CRS

from geoutils._dispatch import is_dask_array
from geoutils._misc import import_optional
from geoutils._typing import ArrayLike, DTypeLike, MArrayNum, NDArrayBool, NDArrayNum
from geoutils.multiproc import MultiprocConfig, compute_tiling
from geoutils.raster.array import get_mask_from_array
from geoutils.raster.referencing import _ij2xy

if TYPE_CHECKING:
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterBase, RasterLike, RasterType
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


########################
# 1/ ARRAY SUBSAMPLING
########################

#########################
# 1A/ COMMON SELECTION
#########################


@dataclass(frozen=True)
class SubsampleMeta:
    """Store the size, deterministic key seed and optional cutoff for a chunked subsample."""

    sample_size: int
    seed: int
    cutoff: np.uint64 | None


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


def _valid_subsample_mask(
    arr_chunk: NDArrayNum | NDArrayBool,
    mask_chunk: NDArrayBool | None = None,
    *,
    skip_nodata: bool = True,
) -> NDArrayBool:
    """Find values eligible for sampling, or eligible True values for boolean data."""

    # Include every source cell when nodata values and False boolean values are requested
    if skip_nodata:
        valid = ~get_mask_from_array(arr_chunk).reshape(arr_chunk.shape)
        if arr_chunk.dtype == np.bool_:
            valid &= np.ma.getdata(arr_chunk)
    else:
        valid = np.ones(arr_chunk.shape, dtype=bool)

    # Apply the explicit sampling mask independently of source value validity
    if mask_chunk is not None:
        valid &= mask_chunk
    return valid


def _get_indices_block_per_subsample(indices_1d: NDArrayNum, nb_valids_per_block: NDArray[np.int64]) -> list[list[int]]:
    """
    Get list of 1D valid subsample indices relative to each block.

    The 1D valid subsample indices correspond to the subsample index to apply for a flattened array of valid values.
    Relative to the block means converted so that the block indexes for valid values starts at 0 up to the number of
    valid values in that block (while the input indices go from zero to the total number of valid values in the full
    array).

    :param indices_1d: Subsample 1D indexes among a total number of valid values.
    :param nb_valids_per_block: Number of valid pixels per block.

    :returns: Relative 1D valid subsample index per block.
    """

    # Apply a cumulative sum to get the first 1D total index of each block
    valids_cumsum = np.cumsum(nb_valids_per_block)

    # We can write a faster algorithm by sorting
    indices_1d = np.sort(indices_1d)

    # We define a list of indices per block
    relative_index_per_block = [[] for _ in nb_valids_per_block]
    k = 0  # K is the block number
    for i in indices_1d:

        # Move to the next block K where current 1D subsample index is, if not in this one
        while i >= valids_cumsum[k]:
            k += 1

        # Add 1D subsample index relative to first subsample index of this block
        first_index_block = valids_cumsum[k - 1] if k >= 1 else 0  # The first 1D valid subsample index of the block
        relative_index = i - first_index_block
        relative_index_per_block[k].append(relative_index)

    return relative_index_per_block


def _prepare_sequential_subsample(
    nb_valids_per_block: NDArray[np.int64],
    sample_size: int,
    random_state: int | np.random.Generator | None,
    *,
    select_all: bool,
    preserve_order: bool = False,
) -> tuple[Sequence[list[int] | slice], NDArray[np.int64], NDArrayNum | slice]:
    """Choose valid positions per block for a sequential sample and record their output order."""

    # Keep complete blocks compact; partial samples need only their selected positions within each block
    if select_all:
        indices_per_block: Sequence[list[int] | slice] = [slice(None)] * len(nb_valids_per_block)
        selected_counts = nb_valids_per_block
        output_order: NDArrayNum | slice = slice(None)
    else:
        rng = np.random.default_rng(random_state)
        indices_1d = rng.choice(int(nb_valids_per_block.sum()), sample_size, replace=False)

        # Block selection sorts valid positions, we recover the original draw order
        output_order = np.argsort(np.argsort(indices_1d)) if preserve_order else slice(None)
        indices_per_block = _get_indices_block_per_subsample(indices_1d, nb_valids_per_block)
        selected_counts = np.asarray([len(indices) for indices in indices_per_block], dtype=np.int64)

    return indices_per_block, selected_counts, output_order


def _subsample_exceeds_largest_chunk(sample_size: int, largest_chunk: int) -> bool:
    """
    Check whether the requested subsample contains more points than the largest input chunk.

    If the subsample size exceeds the size of an input chunk, it cannot concatenate all keys in memory to find the
    subsampled point indices directly, and instead uses the algorithm described in _iterative_topk_cutoff().
    """

    return sample_size > largest_chunk


def _resolve_topk_seed(random_state: int | np.random.Generator | None) -> int:
    """Convert a random state to the integer seed used for deterministic top-k keys."""

    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    if random_state is None:
        return 0
    return int(random_state)


def _splitmix64(x: np.typing.NDArray[np.uint64]) -> NDArrayNum:
    """
    Vectorized SplitMix64 mixer from uint64 to uint64.

    This function performs a fast deterministic mapping from integer IDs to "random-looking" 64-bit keys,
    that we use further below for reproducible subsampling based on global linear indices (the chunk-independent method
    "topk").
    We cannot use a NumPy function directly here because they don't expose their mixers used under-the-hood.

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


def _select_topk_keys(
    global_indices: NDArray[np.int64],
    seed: int,
    *,
    sample_size: int | None = None,
    cutoff: np.uint64 | None = None,
    sort: bool = False,
    keys_only: bool = False,
) -> tuple[NDArray[np.uint64], NDArray[np.intp] | NDArrayBool | None]:
    """Select deterministic keys by sample size or cutoff and return their positions when needed."""

    if (sample_size is None) == (cutoff is None):
        raise ValueError("Exactly one of ``sample_size`` or ``cutoff`` must be provided.")

    # Calculate one deterministic key for every eligible global index
    global_indices = np.asarray(global_indices, dtype=np.int64)
    keys = np.asarray(_splitmix64(np.uint64(seed) ^ global_indices.astype(np.uint64)), dtype=np.uint64)
    if cutoff is not None:
        selected_cutoff = keys <= cutoff
        return keys[selected_cutoff], selected_cutoff

    # Keep only the requested number of smallest keys
    assert sample_size is not None
    selected_size = min(sample_size, len(keys))
    if selected_size == 0:
        return keys[:0], None if keys_only else np.empty(0, dtype=np.intp)
    if keys_only:
        if selected_size < len(keys):
            keys.partition(selected_size - 1)
            return keys[:selected_size].copy(), None
        return keys, None

    selected = np.argpartition(keys, selected_size - 1)[:selected_size]
    if sort:
        selected = selected[np.lexsort((global_indices[selected], keys[selected]))]
    return keys[selected], selected


def _merge_topk_candidates(
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


def _recover_splitmix64_indices(keys: np.typing.NDArray[np.uint64], seed: int) -> np.typing.NDArray[np.int64]:
    """Recover global cell indexes from SplitMix64 keys without allocating another complete array."""

    # Prepare the constants that undo the two multiplications in _splitmix64()
    if len(keys) == 0:
        return keys.view(np.int64)
    second_multiplier_inverse = np.uint64(0x319642B2D24D8EC3)
    first_multiplier_inverse = np.uint64(0x96DE1B173F119089)
    initial_increment = np.uint64(0x9E3779B97F4A7C15)

    # Reverse each mixing operation in small groups so temporary arrays stay independent of the sample size
    temporary = np.empty(min(65_536, len(keys)), dtype=np.uint64)
    for start in range(0, len(keys), len(temporary)):
        values = keys[start : start + len(temporary)]
        work = temporary[: len(values)]

        # Undo the final shift and the second multiplication
        np.right_shift(values, np.uint64(31), out=work)
        np.bitwise_xor(values, work, out=values)
        np.right_shift(values, np.uint64(62), out=work)
        np.bitwise_xor(values, work, out=values)
        np.multiply(values, second_multiplier_inverse, out=values)

        # Undo the middle shift and the first multiplication
        np.right_shift(values, np.uint64(27), out=work)
        np.bitwise_xor(values, work, out=values)
        np.right_shift(values, np.uint64(54), out=work)
        np.bitwise_xor(values, work, out=values)
        np.multiply(values, first_multiplier_inverse, out=values)

        # Undo the first shift, initial increment and seed combination
        np.right_shift(values, np.uint64(30), out=work)
        np.bitwise_xor(values, work, out=values)
        np.right_shift(values, np.uint64(60), out=work)
        np.bitwise_xor(values, work, out=values)
        np.subtract(values, initial_increment, out=values)
        np.bitwise_xor(values, np.uint64(seed), out=values)

    return keys.view(np.int64)


def _topk_key_histogram(keys: NDArray[np.uint64], prefix: int, prefix_bits: int, digit_bits: int) -> NDArray[np.int64]:
    """
    Count keys in the next set of ranges for one chunk of a large requested subsample.

    See description of _iterative_topk_cutoff() for details on the implementation logic.
    """

    # Keep only keys in the range chosen by earlier passes
    if prefix_bits:
        prefix_matches = keys >> np.uint64(64 - prefix_bits) == np.uint64(prefix)
        keys = keys[prefix_matches]

    # Split the remaining keys into smaller ranges and count each range
    shift = 64 - prefix_bits - digit_bits
    digit_mask = np.uint64((1 << digit_bits) - 1)
    digits = ((keys >> np.uint64(shift)) & digit_mask).astype(np.intp)
    return np.bincount(digits, minlength=1 << digit_bits).astype(np.int64, copy=False)


def _topk_prefix_keys(keys: NDArray[np.uint64], prefix: int, prefix_bits: int) -> NDArray[np.uint64]:
    """
    Return keys from one chunk in the range chosen for the requested subsample.

    See description of _iterative_topk_cutoff() for details on the implementation logic.
    """

    prefix_matches = keys >> np.uint64(64 - prefix_bits) == np.uint64(prefix)
    return keys[prefix_matches]


def _topk_histogram_candidates(
    keys: NDArray[np.uint64], digit_bits: int, candidate_prefixes: tuple[int, int]
) -> tuple[NDArray[np.int64], NDArray[np.uint64]]:
    """Count the first key ranges and keep keys from an interval likely to contain the cutoff."""

    histogram = _topk_key_histogram(keys, 0, 0, digit_bits)
    prefix_start, prefix_stop = candidate_prefixes
    prefixes = keys >> np.uint64(64 - digit_bits)
    candidate_mask = (prefixes >= prefix_start) & (prefixes < prefix_stop)
    return histogram, keys[candidate_mask]


def _merge_topk_histogram_candidates(
    parts: list[tuple[NDArray[np.int64], NDArray[np.uint64] | None]], candidate_limit: int
) -> tuple[NDArray[np.int64], NDArray[np.uint64] | None]:
    """Add histograms while keeping no more than one raster chunk of possible cutoff keys."""

    histogram = np.sum([part[0] for part in parts], axis=0, dtype=np.int64)
    candidate_parts = [part[1] for part in parts]
    if any(part is None for part in candidate_parts):
        return histogram, None

    candidates = cast(list[NDArray[np.uint64]], candidate_parts)
    if sum(len(part) for part in candidates) > candidate_limit:
        return histogram, None
    return histogram, np.concatenate(candidates)


def _topk_candidate_prefixes(
    subsample: float | int, total_cells: int, largest_chunk: int, digit_bits: int
) -> tuple[int, int]:
    """Choose a bounded key interval likely to contain the requested sample cutoff."""

    prefix_count = 1 << digit_bits

    # Cache ranges expected to hold three quarters of one chunk, leaving room for uneven key counts
    cached_prefix_count = max(1, min(prefix_count, largest_chunk * prefix_count * 3 // (total_cells * 4)))
    if 0 < subsample <= 1:
        # Center fractional samples on their expected key quantile
        predicted_prefix = min(prefix_count - 1, int(subsample * prefix_count))
        lower_prefixes = cached_prefix_count // 2
    else:
        # Reserve most prefixes above the all-valid estimate because missing cells can move the cutoff upward
        requested_size = max(1, min(int(subsample), total_cells))
        predicted_prefix = (requested_size - 1) * prefix_count // total_cells
        lower_prefixes = max(1, cached_prefix_count // 8)

    prefix_start = max(0, predicted_prefix - lower_prefixes)
    prefix_stop = min(prefix_count, prefix_start + cached_prefix_count)
    return max(0, prefix_stop - cached_prefix_count), prefix_stop


def _sum_topk_histograms(histograms: list[NDArray[np.int64]]) -> NDArray[np.int64]:
    """Add key counts for the requested subsample from a group of up to eight Dask chunks."""

    return np.sum(histograms, axis=0, dtype=np.int64)


def _iterative_topk_cutoff(
    subsample: float | int,
    subsample_meta: SubsampleMeta,
    largest_chunk: int,
    total_cells: int,
    number_chunks: int,
    histogram_for_prefix: Callable[
        [int, int, int, int, tuple[int, int] | None],
        tuple[NDArray[np.int64], NDArray[np.uint64] | None],
    ],
    keys_for_prefix: Callable[[int, int, int, int], NDArray[np.uint64]],
) -> SubsampleMeta:
    """
    Find the cutoff separating cells in/out of the subsample without loading all indexes in memory.

    Dask and multiprocessing use this method only when the sample size exceeds the largest raster chunk.

    This function finds the "k" value that separates the "topk" samples kept for the subsample,
    but in a chunk-by-chunk manner for cases where the subsample itself is very large (e.g. 80% of the raster).
    This requires several iterations to converge towards the right value.

    The cutoff algorithm follows these steps:
    1. Start with the full unsigned 64-bit key interval (0 through 2**64 - 1), split it into subranges, and count,
       across all chunks, how many keys fall in each subrange.
    2. Use the cumulative counts and requested sample size to identify the subrange containing the cutoff, and discard
       the other subranges.
    3. Repeat the count within that range until it contains no more keys than the largest raster chunk.
    4. Collect the remaining keys and select the exact cutoff value.

    References
    ----------
    - NIST Dictionary of Algorithms and Data Structures, "Selection problem"
      https://xlinux.nist.gov/dads/HTML/selectkth.html
    - Alabi et al., "Fast k-selection algorithms for graphics processing units", Journal of Experimental
      Algorithmics 17, 2012. https://doi.org/10.1145/2133803.2345676
    """

    # Use the sample size and key seed resolved before the backend starts its cutoff passes
    sample_size = subsample_meta.sample_size
    seed = subsample_meta.seed
    if sample_size == 0:
        return subsample_meta

    # Keep count arrays small while using more ranges when the raster contains many chunks
    digit_bits = min(12, max(8, (number_chunks - 1).bit_length()))
    prefix = 0
    prefix_bits = 0
    rank = sample_size - 1
    candidate_cache: tuple[int, int, int, NDArray[np.uint64]] | None = None
    while prefix_bits < 64:
        current_digit_bits = min(digit_bits, 64 - prefix_bits)
        candidate_prefixes = None
        if prefix_bits == 0:
            candidate_prefixes = _topk_candidate_prefixes(subsample, total_cells, largest_chunk, current_digit_bits)
        histogram, candidate_keys = histogram_for_prefix(
            seed, prefix, prefix_bits, current_digit_bits, candidate_prefixes
        )
        if candidate_prefixes is not None and candidate_keys is not None:
            candidate_cache = (*candidate_prefixes, current_digit_bits, candidate_keys)

        # Find the key range containing the last selected point (subsample size)
        cumulative_counts = np.cumsum(histogram)
        digit = int(np.searchsorted(cumulative_counts, rank, side="right"))
        preceding_count = 0 if digit == 0 else int(cumulative_counts[digit - 1])
        rank -= preceding_count
        prefix = (prefix << current_digit_bits) | digit
        prefix_bits += current_digit_bits
        group_size = int(histogram[digit])

        # Find the exact cutoff from no more than one chunk of keys
        if group_size <= largest_chunk:
            if prefix_bits == 64:
                return SubsampleMeta(sample_size=sample_size, seed=seed, cutoff=np.uint64(prefix))
            group_keys = None
            if candidate_cache is not None:
                cached_prefix_start, cached_prefix_stop, cached_prefix_bits, cached_keys = candidate_cache
                if cached_prefix_bits == prefix_bits and cached_prefix_start <= prefix < cached_prefix_stop:
                    cached_group_keys = _topk_prefix_keys(cached_keys, prefix, prefix_bits)
                    if len(cached_group_keys) == group_size:
                        group_keys = cached_group_keys
            if group_keys is None:
                group_keys = keys_for_prefix(seed, prefix, prefix_bits, group_size)
            if len(group_keys) != group_size:
                raise RuntimeError("The number of random keys changed while finding the subsample cutoff.")
            group_keys.partition(rank)
            return SubsampleMeta(sample_size=sample_size, seed=seed, cutoff=group_keys[rank])

    raise RuntimeError("Could not find the subsample cutoff.")


#########################
# 1B/ NUMPY SUBSAMPLING
#########################


@overload
def _subsample_numpy(
    array: NDArrayNum | MArrayNum,
    subsample: float | int,
    return_indices: Literal[False] = False,
    *,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "topk",
    skip_nodata: bool = True,
    mask: NDArrayBool | None = None,
) -> NDArrayNum: ...


@overload
def _subsample_numpy(
    array: NDArrayNum | MArrayNum,
    subsample: float | int,
    return_indices: Literal[True],
    *,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "topk",
    skip_nodata: bool = True,
    mask: NDArrayBool | None = None,
) -> tuple[NDArrayNum, ...]: ...


def _subsample_numpy(
    array: NDArrayNum | MArrayNum,
    subsample: float | int,
    return_indices: bool = False,
    *,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "topk",
    skip_nodata: bool = True,
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
    :param skip_nodata: Whether to exclude nodata values, and False cells in a boolean array.
    :param mask: Prepared boolean eligibility mask with the same shape as array.

    :returns: The subsampled array (1D) or the indices to extract (same shape as input array).
    """

    # Return the complete grid directly when neither nodata nor a mask restricts it
    if subsample == 1 and not skip_nodata and mask is None:
        if return_indices:
            return np.unravel_index(np.arange(array.size), array.shape)
        return array.reshape(-1)

    # Determine valid pixels according to skip_nodata and their global linear indices (row * nx + col)
    valid = _valid_subsample_mask(array, mask, skip_nodata=skip_nodata)
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
        seed = _resolve_topk_seed(random_state)

        # Select global indexes with the smallest deterministic keys
        _, sel = _select_topk_keys(valids.astype(np.int64, copy=False), seed, sample_size=subsample_size, sort=True)
        assert sel is not None
        chosen = valids[sel]

        # Unravel indexes, and return values or indexes
        unraveled = np.unravel_index(chosen, array.shape)
        return unraveled if return_indices else array[unraveled]

    else:
        raise ValueError(f"Unknown ``strategy`` {strategy!r}. Choose 'sequential' or 'topk'.")


########################
# 1C/ DASK SUBSAMPLING
########################

# At the date of April 2024:
# Getting an exact subsample size out-of-memory only for valid values is not supported directly by Dask/Xarray

# It is not trivial because we don't know where valid values will be in advance, and because of ragged output (varying
# output length considerations), which prevents from using high-level functions with good efficiency
# We thus follow https://blog.dask.org/2021/07/02/ragged-output (the dask.array.map_blocks solution has a larger RAM
# usage by having to drop an axis and re-chunk along 1D of the 2D array, so we use the delayed solution instead)


@delayed
def _delayed_nb_valids(
    arr_chunk: NDArrayNum | NDArrayBool,
    *,
    skip_nodata: bool = True,
    mask_chunk: NDArrayBool | None = None,
) -> NDArrayNum:
    """Count number of valid values per block."""
    valid = _valid_subsample_mask(arr_chunk, mask_chunk, skip_nodata=skip_nodata)
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
    skip_nodata: bool = True,
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

    # Only valid values are sampled by default (finite for numerical arrays, True for boolean arrays)
    valid = _valid_subsample_mask(arr_chunk, mask_chunk, skip_nodata=skip_nodata)

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

    # Keep the smallest deterministic keys and their positions within this block
    key_sel, sel = _select_topk_keys(gid.astype(np.int64, copy=False), seed, sample_size=k)
    assert sel is not None

    # If return indices
    if return_indices_local:
        gid_sel = gid[sel]
        return key_sel, gid_sel
    # Otherwise, returning values
    else:
        # Extract values for selected pixels, including nodata when requested
        vals = arr_chunk.ravel()[flat[sel]]
        return key_sel, vals


@delayed
def _delayed_merge_topk(
    keys_list: list[NDArrayNum],
    payload_list: list[NDArrayNum],
    *,
    k: int,
) -> tuple[NDArrayNum, NDArrayNum]:
    """Combine one bounded group of delayed top-k candidates."""

    return _merge_topk_candidates(keys_list, payload_list, k=k)


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
    arr_chunk: NDArrayNum | NDArrayBool,
    subsample_indices: NDArrayNum | slice,
    *,
    skip_nodata: bool = True,
    mask_chunk: NDArrayBool | None = None,
) -> NDArrayNum | NDArrayBool:
    """Subsample the valid values at the corresponding 1D valid indices per block."""

    valid = _valid_subsample_mask(arr_chunk, mask_chunk, skip_nodata=skip_nodata)
    return arr_chunk[valid][subsample_indices]


@delayed
def _delayed_subsample_indices_block(
    arr_chunk: NDArrayNum | NDArrayBool,
    subsample_indices: NDArrayNum | slice,
    block_id: dict[str, Any],
    *,
    skip_nodata: bool = True,
    mask_chunk: NDArrayBool | None = None,
) -> NDArrayNum:
    """Return 2D indices from the subsampled 1D valid indices per block."""

    #  Unravel indices of valid data to the shape of the block
    valid = _valid_subsample_mask(arr_chunk, mask_chunk, skip_nodata=skip_nodata)
    flat_valid = np.flatnonzero(valid.ravel())
    ix, iy = np.unravel_index(flat_valid[subsample_indices], shape=arr_chunk.shape)

    # Convert to full-array indexes by adding the row and column starting indexes for this block
    ix += block_id["row_start"]
    iy += block_id["col_start"]

    return np.column_stack((ix, iy))


def _array_chunk_topk_keys(
    array: Any | None,
    block_id: dict[str, Any],
    raster_width: int,
    seed: int,
    skip_nodata: bool,
    mask: Any | None,
) -> NDArray[np.uint64]:
    """Calculate deterministic random keys for eligible cells in one array chunk or bounded raster tile."""

    # Build global linear indexes for this chunk from its full-array offsets
    row_start = int(block_id["row_start"])
    column_start = int(block_id["col_start"])
    if array is None:
        if skip_nodata or mask is not None:
            raise RuntimeError("Array values are required when nodata or a mask restricts eligible cells.")
        row_stop = int(block_id["row_stop"])
        column_stop = int(block_id["col_stop"])
        chunk_shape = (row_stop - row_start, column_stop - column_start)
    else:
        chunk_shape = array.shape

    row_offsets = np.arange(row_start, row_start + chunk_shape[0], dtype=np.int64) * np.int64(raster_width)
    row_offsets += column_start
    global_indices = (row_offsets[:, None] + np.arange(chunk_shape[1], dtype=np.int64)).reshape(-1)

    # Keep only indexes whose values and optional mask make them eligible
    if array is not None:
        valid = _valid_subsample_mask(array, mask, skip_nodata=skip_nodata)
        global_indices = global_indices[valid.reshape(-1)]
    return np.asarray(_splitmix64(np.uint64(seed) ^ global_indices.astype(np.uint64)), dtype=np.uint64)


def _delayed_array_topk_histogram(
    array: Any | None,
    block_id: dict[str, Any],
    raster_width: int,
    seed: int,
    prefix: int,
    prefix_bits: int,
    digit_bits: int,
    skip_nodata: bool,
    mask: Any | None,
) -> NDArray[np.int64]:
    """Count deterministic keys in one Dask array chunk for a cutoff pass."""

    keys = _array_chunk_topk_keys(array, block_id, raster_width, seed, skip_nodata, mask)
    return _topk_key_histogram(keys, prefix, prefix_bits, digit_bits)


def _delayed_array_topk_histogram_candidates(
    array: Any | None,
    block_id: dict[str, Any],
    raster_width: int,
    seed: int,
    digit_bits: int,
    candidate_prefixes: tuple[int, int],
    skip_nodata: bool,
    mask: Any | None,
) -> tuple[NDArray[np.int64], NDArray[np.uint64]]:
    """Count key ranges and keep possible cutoff keys from one Dask array chunk."""

    keys = _array_chunk_topk_keys(array, block_id, raster_width, seed, skip_nodata, mask)
    return _topk_histogram_candidates(keys, digit_bits, candidate_prefixes)


def _delayed_array_topk_prefix_keys(
    array: Any | None,
    block_id: dict[str, Any],
    raster_width: int,
    seed: int,
    prefix: int,
    prefix_bits: int,
    skip_nodata: bool,
    mask: Any | None,
) -> NDArray[np.uint64]:
    """Return keys in the final cutoff range from one Dask array chunk."""

    keys = _array_chunk_topk_keys(array, block_id, raster_width, seed, skip_nodata, mask)
    return _topk_prefix_keys(keys, prefix, prefix_bits)


def _dask_array_topk_cutoff(
    blocks: list[Any | None],
    mask_blocks: list[Any | None],
    block_ids: list[dict[str, Any]],
    array_shape: tuple[int, int],
    largest_chunk: int,
    subsample: float | int,
    subsample_meta: SubsampleMeta,
    skip_nodata: bool,
) -> SubsampleMeta:
    """Find the deterministic key cutoff for a Dask array without collecting the selected indexes."""

    dask = import_optional("dask")
    delayed = dask.delayed

    def histogram_for_prefix(
        seed: int,
        prefix: int,
        prefix_bits: int,
        digit_bits: int,
        candidate_prefixes: tuple[int, int] | None,
    ) -> tuple[NDArray[np.int64], NDArray[np.uint64] | None]:
        """Count one key range pass across all Dask chunks."""

        # Keep a bounded interval around the expected cutoff during the first array pass
        if candidate_prefixes is not None:
            histogram_candidates = [
                delayed(_delayed_array_topk_histogram_candidates)(
                    block,
                    block_id,
                    array_shape[1],
                    seed,
                    digit_bits,
                    candidate_prefixes,
                    skip_nodata,
                    mask_block,
                )
                for block, mask_block, block_id in zip(blocks, mask_blocks, block_ids)
            ]
            while len(histogram_candidates) > 1:
                histogram_candidates = [
                    delayed(_merge_topk_histogram_candidates)(histogram_candidates[start : start + 2], largest_chunk)
                    for start in range(0, len(histogram_candidates), 2)
                ]
            return cast(
                tuple[NDArray[np.int64], NDArray[np.uint64] | None],
                dask.compute(histogram_candidates[0])[0],
            )

        histograms = [
            delayed(_delayed_array_topk_histogram)(
                block,
                block_id,
                array_shape[1],
                seed,
                prefix,
                prefix_bits,
                digit_bits,
                skip_nodata,
                mask_block,
            )
            for block, mask_block, block_id in zip(blocks, mask_blocks, block_ids)
        ]
        while len(histograms) > 1:
            histograms = [
                delayed(_sum_topk_histograms)(histograms[start : start + 8]) for start in range(0, len(histograms), 8)
            ]
        return cast(NDArray[np.int64], dask.compute(histograms[0])[0]), None

    def keys_for_prefix(seed: int, prefix: int, prefix_bits: int, group_size: int) -> NDArray[np.uint64]:
        """Collect the last bounded key range from every Dask chunk."""

        key_parts = [
            delayed(_delayed_array_topk_prefix_keys)(
                block,
                block_id,
                array_shape[1],
                seed,
                prefix,
                prefix_bits,
                skip_nodata,
                mask_block,
            )
            for block, mask_block, block_id in zip(blocks, mask_blocks, block_ids)
        ]
        group_keys = np.empty(group_size, dtype=np.uint64)
        offset = 0
        for chunk_keys in dask.compute(*key_parts):
            stop = offset + len(chunk_keys)
            group_keys[offset:stop] = chunk_keys
            offset = stop
        return group_keys[:offset]

    return _iterative_topk_cutoff(
        subsample,
        subsample_meta,
        largest_chunk,
        int(np.prod(array_shape)),
        len(blocks),
        histogram_for_prefix,
        keys_for_prefix,
    )


@delayed
def _delayed_subsample_cutoff_block(
    array: Any,
    block_id: dict[str, Any],
    seed: int,
    cutoff: np.uint64,
    raster_width: int,
    return_indices: bool,
    skip_nodata: bool,
    mask: Any | None,
) -> pd.DataFrame:
    """Select values or global row/column indexes below a deterministic key cutoff in one Dask chunk."""

    # Calculate eligible local and global positions once for both output forms
    valid = _valid_subsample_mask(array, mask, skip_nodata=skip_nodata)
    local_indices = np.flatnonzero(valid.ravel())
    rows, columns = np.unravel_index(local_indices, array.shape)
    rows = rows.astype(np.int64, copy=False) + int(block_id["row_start"])
    columns = columns.astype(np.int64, copy=False) + int(block_id["col_start"])
    global_indices = rows * np.int64(raster_width) + columns

    # Keep the exact top-k set determined by the cutoff pass
    selected_keys, selected = _select_topk_keys(global_indices, seed, cutoff=cutoff)
    assert selected is not None
    order_keys = np.bitwise_xor(selected_keys, np.uint64(1 << 63)).view(np.int64)
    if return_indices:
        return pd.DataFrame({"_key": order_keys, "row": rows[selected], "column": columns[selected]})
    return pd.DataFrame({"_key": order_keys, "value": np.asarray(array).ravel()[local_indices[selected]]})


def _dask_subsample(
    darr: da.Array,
    subsample: int | float = 1,
    return_indices: bool = False,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "topk",
    preserve_order: bool = False,
    *,
    skip_nodata: bool = True,
    mask: da.Array | None = None,
    force_output_to_memory: bool = False,
) -> da.Array | tuple[da.Array, da.Array]:
    """
    Subsample valid values out-of-memory from a 2D Dask array.

    Strategy "topk" is chunk-invariant (same sample no matter chunk size, and same as the NumPy implementation), while
    "sequential" is chunk-dependent but slightly faster.

    Returns a delayed subsampled Dask array of the output (either values or indices).

    :param preserve_order: Restore sequential random-draw order after collecting block results. The one-dimensional
        sampling adapter uses this to match NumPy point sampling; existing raster calls keep block order.
    :param skip_nodata: Whether to exclude nodata values, and False cells in a boolean array.
    :param mask: Boolean array marking values eligible for sampling, with the same shape as darr.
    """

    # To raise appropriate error on missing optional dependency
    import_optional("dask")

    # Flatten an unrestricted full grid lazily without reading its values to build indexes
    if subsample == 1 and not skip_nodata and mask is None:
        flat_values = darr.reshape(-1)
        if not return_indices:
            return flat_values
        flat_indices = da.arange(int(darr.size), chunks=flat_values.chunks)
        columns = int(darr.shape[1])
        return flat_indices // columns, flat_indices % columns

    # Use full-width row blocks so the complete result keeps the raster's row-by-row order
    if subsample == 1 and darr.ndim == 2:
        darr = darr.rechunk({1: darr.shape[1]})

    # Create a delayed object for each block, and flatten the blocks into a 1d shape
    blocks = darr.to_delayed().ravel()

    # Give each data block the matching mask cells without loading either array
    mask_blocks = [None] * len(blocks)
    if mask is not None:
        mask_blocks = da.asarray(mask).rechunk(darr.chunks).to_delayed().ravel().tolist()

    # Compute number of valid points for each block out-of-memory
    list_delayed_valids = [
        da.from_delayed(
            _delayed_nb_valids(b, skip_nodata=skip_nodata, mask_chunk=m),
            shape=(1, 1),
            dtype=np.dtype("int32"),
        )
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
    # This list also includes the last index used to define each block stop
    starts = [cached_cumsum(c, initial_zero=True) for c in darr.chunks]
    num_chunks = darr.numblocks

    # Get the starts per 1D block ID by unravelling starting indexes for each block
    indexes_yb, indexes_xb = np.unravel_index(np.arange(len(blocks)), shape=(num_chunks[0], num_chunks[1]))

    block_ids = [
        {
            "row_start": starts[0][indexes_yb[i]],
            "row_stop": starts[0][indexes_yb[i] + 1],
            "col_start": starts[1][indexes_xb[i]],
            "col_stop": starts[1][indexes_xb[i] + 1],
        }
        for i in range(len(blocks))
    ]

    # 2/ Build the requested sample

    # 2A/ Select large "topk" samples with a key cutoff
    largest_chunk = max(int(rows * columns) for rows in darr.chunks[0] for columns in darr.chunks[1])
    sample_exceeds_largest_chunk = _subsample_exceeds_largest_chunk(subsample_size, largest_chunk)
    if strategy == "topk" and subsample != 1 and sample_exceeds_largest_chunk and not force_output_to_memory:
        subsample_meta = SubsampleMeta(
            sample_size=subsample_size,
            seed=_resolve_topk_seed(random_state),
            cutoff=None,
        )
        subsample_meta = _dask_array_topk_cutoff(
            blocks=blocks.tolist(),
            mask_blocks=mask_blocks,
            block_ids=block_ids,
            array_shape=(int(darr.shape[0]), int(darr.shape[1])),
            largest_chunk=largest_chunk,
            subsample=subsample,
            subsample_meta=subsample_meta,
            skip_nodata=skip_nodata,
        )
        assert subsample_meta.cutoff is not None
        selected_parts = [
            _delayed_subsample_cutoff_block(
                block,
                block_id,
                subsample_meta.seed,
                subsample_meta.cutoff,
                int(darr.shape[1]),
                return_indices,
                skip_nodata,
                mask_block,
            )
            for block, mask_block, block_id in zip(blocks, mask_blocks, block_ids)
        ]
        from geoutils.pointcloud.dataframe import _import_dask_dataframe

        dask_dataframe = _import_dask_dataframe()
        if return_indices:
            meta = pd.DataFrame(
                {
                    "_key": pd.Series(dtype=np.int64),
                    "row": pd.Series(dtype=np.int64),
                    "column": pd.Series(dtype=np.int64),
                }
            )
            selected_frame = dask_dataframe.from_delayed(selected_parts, meta=meta).sort_values("_key")
            rows = selected_frame["row"].to_dask_array(lengths=True)
            columns = selected_frame["column"].to_dask_array(lengths=True)
            return rows, columns
        meta = pd.DataFrame({"_key": pd.Series(dtype=np.int64), "value": pd.Series(dtype=np.dtype(darr.dtype))})
        selected_frame = dask_dataframe.from_delayed(selected_parts, meta=meta).sort_values("_key")
        return selected_frame["value"].to_dask_array(lengths=True)

    # 2B/ Select "sequential" samples by position within each block
    if strategy == "sequential" or subsample == 1:
        ind_per_block, selected_counts, draw_order = _prepare_sequential_subsample(
            nb_valids_per_block,
            subsample_size,
            random_state,
            select_all=subsample == 1,
            preserve_order=preserve_order,
        )

        # To just get the subsample without indices
        if not return_indices:
            # Task a delayed subsample to be computed for each block, skipping blocks with no values to sample
            used = np.flatnonzero(selected_counts).tolist()
            list_subsamples = [
                _delayed_subsample_block(
                    blocks[i],
                    (
                        ind_per_block[i]
                        if isinstance(ind_per_block[i], slice)
                        else np.asarray(ind_per_block[i], dtype=np.int64)
                    ),
                    skip_nodata=skip_nodata,
                    mask_chunk=mask_blocks[i],
                )
                for i in used
            ]

            # Cast output to the right expected dtype and length, then compute and concatenate
            list_subsamples_da = [
                da.from_delayed(s, shape=(int(selected_counts[i]),), dtype=darr.dtype)
                for s, i in zip(list_subsamples, used)
            ]
            return da.concatenate(list_subsamples_da, axis=0)[draw_order]

        # To return indices
        else:
            # Task delayed subsample indices to be computed for each block, skipping blocks with no values to sample
            used = np.flatnonzero(selected_counts).tolist()
            list_subsample_indices = [
                _delayed_subsample_indices_block(
                    blocks[i],
                    (
                        ind_per_block[i]
                        if isinstance(ind_per_block[i], slice)
                        else np.asarray(ind_per_block[i], dtype=np.int64)
                    ),
                    block_id=block_ids[i],
                    skip_nodata=skip_nodata,
                    mask_chunk=mask_blocks[i],
                )
                for i in used
            ]

            # Cast output to the right expected dtype and length, then compute and concatenate
            list_indices_da = [
                da.from_delayed(s, shape=(int(selected_counts[i]), 2), dtype=np.int64)
                for s, i in zip(list_subsample_indices, used)
            ]
            indices = da.concatenate(list_indices_da, axis=0)[draw_order]
            return indices[:, 0], indices[:, 1]

    # 2C/ Select bounded "topk" candidates from every block
    elif strategy == "topk":

        # Convert random_state to an integer seed for deterministic key generation
        seed = _resolve_topk_seed(random_state)

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
                skip_nodata=skip_nodata,
                mask_chunk=mask_blocks[i],
            )
            for i in range(len(blocks))
        ]

        # Separate keys and payload lists (payload are either values or global linear indices)
        keys_list = [c[0] for c in cands]
        payload_list = [c[1] for c in cands]

        # Global merge to get the top-k across all blocks
        # Combine at most eight candidate arrays at a time so the final task never receives every block's candidates
        while len(keys_list) > 1:
            merged_groups = [
                _delayed_merge_topk(keys_list[start : start + 8], payload_list[start : start + 8], k=subsample_size)
                for start in range(0, len(keys_list), 8)
            ]
            keys_list = [merged[0] for merged in merged_groups]
            payload_list = [merged[1] for merged in merged_groups]
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


###################################
# 1D/ MULTIPROCESSING SUBSAMPLING
###################################


def _read_subsample_raster_block(
    rst: RasterBase,
    tile_idx: NDArrayNum,
    *,
    band: int = 1,
    skip_nodata: bool = True,
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

    # Exclude nodata and nonfinite values when requested; boolean rasters sample only their True cells by default
    valid = _valid_subsample_mask(arr, skip_nodata=skip_nodata)

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
    skip_nodata: bool = True,
    mask: RasterLike | VectorLike | ArrayLike | None = None,
) -> int:
    """Count valid values in one tile out-of-memory."""

    _, valid = _read_subsample_raster_block(rst, tile_idx, band=band, skip_nodata=skip_nodata, mask=mask)
    return int(np.count_nonzero(valid))


def _wrapper_multiproc_nb_valids_per_block_positional(
    rst: RasterBase,
    tile_idx: NDArrayNum,
    band: int,
    skip_nodata: bool,
    mask: RasterLike | VectorLike | ArrayLike | None,
) -> int:
    """Pass positional multiprocessing task arguments to the valid-value count."""

    return _wrapper_multiproc_nb_valids_per_block(
        rst,
        tile_idx,
        band=band,
        skip_nodata=skip_nodata,
        mask=mask,
    )


def _wrapper_multiproc_subsample_values_block(
    rst: RasterBase,
    tile_idx: NDArrayNum,
    subsample_indices_rel: NDArrayNum | slice,
    *,
    band: int = 1,
    skip_nodata: bool = True,
    return_global_indices: bool = False,
    mask: RasterLike | VectorLike | ArrayLike | None = None,
) -> NDArrayNum | NDArrayBool | MArrayNum | tuple[NDArrayNum, NDArrayNum | NDArrayBool | MArrayNum]:
    """
    Return values sampled from one raster tile, with global indexes when needed for ordering.
    """

    # Get tile out-of-memory
    arr, valid = _read_subsample_raster_block(rst, tile_idx, band=band, skip_nodata=skip_nodata, mask=mask)

    # Return subsample of finite values (or True values for boolean input), including nodata when requested
    flat_valid = np.flatnonzero(valid.ravel())
    flat_selected = flat_valid[subsample_indices_rel]
    values = arr.ravel()[flat_selected]
    if not return_global_indices:
        return values

    # Pair complete samples with their positions so the original row-by-row order can be restored
    ncols = int(arr.shape[1])
    rows = flat_selected // ncols + int(tile_idx[0])
    columns = flat_selected % ncols + int(tile_idx[2])
    global_indices = rows * int(rst.shape[1]) + columns
    return global_indices.astype(np.int64), values


def _wrapper_multiproc_subsample_indices_block(
    rst: RasterBase,
    tile_idx: NDArrayNum,
    subsample_indices_rel: NDArrayNum | slice,
    *,
    band: int = 1,
    skip_nodata: bool = True,
    mask: RasterLike | VectorLike | ArrayLike | None = None,
) -> NDArrayNum:
    """
    Return indices of the sampled valid pixels in one tile.

    Output shape: (n, 2) with columns [row, col] in full-array coordinates.
    """

    # Get tile out-of-memory
    arr, valid = _read_subsample_raster_block(rst, tile_idx, band=band, skip_nodata=skip_nodata, mask=mask)

    # Get starting row/col of the tile
    row0 = int(tile_idx[0])
    col0 = int(tile_idx[2])

    # Get relative indices of finite values (or True for boolean)
    flat_valid = np.flatnonzero(valid.ravel())

    # Use input to draw them
    indexes = (
        subsample_indices_rel
        if isinstance(subsample_indices_rel, slice)
        else np.asarray(subsample_indices_rel, dtype=np.int64)
    )
    flat_sel = flat_valid[indexes]

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
    return_keys_only: bool = False,
    band: int = 1,
    skip_nodata: bool = True,
    mask: RasterLike | VectorLike | ArrayLike | None = None,
) -> NDArrayNum | tuple[NDArrayNum, NDArrayNum | NDArrayBool | MArrayNum]:
    """
    Return up to k candidates from one tile as keys alone or as (keys, payload).

    keys: uint64 keys for selected valid pixels in this tile.
    The key-only form avoids keeping a same-sized cell index array during selection. Otherwise, payload contains:
      - if return_indices=True: global linear indices (gid = row*nx_full + col) (int64)
      - else: sampled values (array dtype)
    """
    # If no subsample, early return
    if k <= 0:
        if return_keys_only:
            return np.empty((0,), dtype=np.uint64)
        payload_dtype = np.int64 if return_indices else rst.dtype
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=payload_dtype)

    # Tile offsets in full-array indices
    row0 = int(tile_idx[0])
    col0 = int(tile_idx[2])
    row_stop = int(tile_idx[1])
    col_stop = int(tile_idx[3])

    # We don't need the raster values for sampling only indices
    if return_indices and not skip_nodata and mask is None:
        rows = np.arange(row0, row_stop, dtype=np.int64)[:, None]
        columns = np.arange(col0, col_stop, dtype=np.int64)[None, :]
        gid = (rows * np.int64(nx_full) + columns).ravel()
        key, sel = _select_topk_keys(gid, seed, sample_size=k, keys_only=return_keys_only)
        if return_keys_only:
            return key
        assert sel is not None
        return key, gid[sel]

    # Get tile out-of-memory
    arr, valid = _read_subsample_raster_block(rst, tile_idx, band=band, skip_nodata=skip_nodata, mask=mask)

    # Get valids indices
    flat = np.flatnonzero(valid.ravel())
    nvalid = int(flat.size)

    # If no valid, early return
    if nvalid == 0:
        if return_keys_only:
            return np.empty((0,), dtype=np.uint64)
        payload_dtype = np.int64 if return_indices else arr.dtype
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=payload_dtype)

    # Get relative row and columns
    ncols = int(arr.shape[1])
    r = flat // ncols
    c = flat - r * ncols

    # Global linear index from absolute row and columns: gid = (row0 + r) * nx_full + (col0 + c)
    gid = (np.int64(row0) + r.astype(np.int64)) * np.int64(nx_full) + (np.int64(col0) + c.astype(np.int64))
    # Select the appropriate number of deterministic keys
    key_sel, sel = _select_topk_keys(gid, seed, sample_size=k, keys_only=return_keys_only)
    if return_keys_only:
        return key_sel
    assert sel is not None

    # If we return indices
    if return_indices:
        return key_sel, gid[sel]

    # If we return values
    vals: NDArrayNum | NDArrayBool | MArrayNum = arr.ravel()[flat[sel]]
    return key_sel, vals


def _wrapper_multiproc_topk_candidates_block_positional(
    rst: RasterBase,
    tile_idx: NDArrayNum,
    seed: int,
    k: int,
    nx_full: int,
    return_indices: bool,
    return_keys_only: bool,
    band: int,
    skip_nodata: bool,
    mask: RasterLike | VectorLike | ArrayLike | None,
) -> NDArrayNum | tuple[NDArrayNum, NDArrayNum | NDArrayBool | MArrayNum]:
    """Pass positional multiprocessing task arguments to the top-k tile calculation."""

    return _wrapper_multiproc_topk_candidates_block(
        rst,
        tile_idx,
        seed=seed,
        k=k,
        nx_full=nx_full,
        return_indices=return_indices,
        return_keys_only=return_keys_only,
        band=band,
        skip_nodata=skip_nodata,
        mask=mask,
    )


def _multiproc_array_topk_keys(
    source_raster: RasterType,
    tile_idx: NDArrayNum,
    seed: int,
    band: int,
    skip_nodata: bool,
    mask: RasterLike | VectorLike | ArrayLike | None = None,
) -> NDArray[np.uint64]:
    """Calculate random keys for usable cells in one multiprocessing tile of a large requested subsample."""

    # Read the main band only when its values determine which cells may be selected
    if skip_nodata or mask is not None:
        array, valid = _read_subsample_raster_block(
            source_raster,
            tile_idx,
            band=band,
            skip_nodata=skip_nodata,
            mask=mask,
        )
        block_id = {"row_start": int(tile_idx[0]), "col_start": int(tile_idx[2])}
        return _array_chunk_topk_keys(array, block_id, source_raster.shape[1], seed, False, valid)

    row_start, row_stop, column_start, column_stop = (int(value) for value in tile_idx)
    row_offsets = np.arange(row_start, row_stop, dtype=np.int64) * np.int64(source_raster.shape[1]) + column_start
    global_indices = (row_offsets[:, None] + np.arange(column_stop - column_start, dtype=np.int64)).reshape(-1)
    return np.asarray(_splitmix64(np.uint64(seed) ^ global_indices.astype(np.uint64)), dtype=np.uint64)


def _wrapper_multiproc_array_topk_histogram(
    source_raster: RasterType,
    tile_idx: NDArrayNum,
    seed: int,
    prefix: int,
    prefix_bits: int,
    digit_bits: int,
    band: int,
    skip_nodata: bool,
    mask: RasterLike | VectorLike | ArrayLike | None,
) -> NDArray[np.int64]:
    """Count random keys in one tile during one pass over the requested subsample."""

    # Calculate keys only for cells that may appear in the output
    keys = _multiproc_array_topk_keys(source_raster, tile_idx, seed, band, skip_nodata, mask)
    return _topk_key_histogram(keys, prefix, prefix_bits, digit_bits)


def _wrapper_multiproc_array_topk_histogram_candidates(
    source_raster: RasterType,
    tile_idx: NDArrayNum,
    seed: int,
    digit_bits: int,
    candidate_prefixes: tuple[int, int],
    band: int,
    skip_nodata: bool,
    mask: RasterLike | VectorLike | ArrayLike | None,
) -> tuple[NDArray[np.int64], NDArray[np.uint64]]:
    """Count first-pass key ranges and return possible cutoff keys from one multiprocessing tile."""

    keys = _multiproc_array_topk_keys(source_raster, tile_idx, seed, band, skip_nodata, mask)
    return _topk_histogram_candidates(keys, digit_bits, candidate_prefixes)


def _wrapper_multiproc_array_topk_prefix_keys(
    source_raster: RasterType,
    tile_idx: NDArrayNum,
    seed: int,
    prefix: int,
    prefix_bits: int,
    band: int,
    skip_nodata: bool,
    mask: RasterLike | VectorLike | ArrayLike | None,
) -> NDArray[np.uint64]:
    """Return keys in the range chosen for the requested subsample from one multiprocessing tile."""

    # Calculate keys only for cells that may appear in the output
    keys = _multiproc_array_topk_keys(source_raster, tile_idx, seed, band, skip_nodata, mask)
    return _topk_prefix_keys(keys, prefix, prefix_bits)


def _multiproc_array_topk_cutoff(
    source_raster: RasterType,
    tiles: NDArrayNum,
    largest_chunk: int,
    subsample: float | int,
    subsample_meta: SubsampleMeta,
    band: int,
    skip_nodata: bool,
    mp_config: MultiprocConfig,
    mask: RasterLike | VectorLike | ArrayLike | None = None,
) -> SubsampleMeta:
    """
    Find the key cutoff for the subsample, with Multiproc per-chunk operations.

    Only used for a subsample size larger than a raster chunk.
    """

    from geoutils.multiproc.cluster import _map_bounded

    def histogram_for_prefix(
        seed: int,
        prefix: int,
        prefix_bits: int,
        digit_bits: int,
        candidate_prefixes: tuple[int, int] | None,
    ) -> tuple[NDArray[np.int64], NDArray[np.uint64] | None]:
        """Count keys for the requested subsample in every multiprocessing tile and add the results."""

        # Collect the bounded first-pass cutoff candidates selected by the shared algorithm
        if candidate_prefixes is not None:
            candidate_arguments = (
                (
                    source_raster,
                    tile,
                    seed,
                    digit_bits,
                    candidate_prefixes,
                    band,
                    skip_nodata,
                    mask,
                )
                for tile in tiles
            )
            histogram_candidates = [
                result
                for _, result in _map_bounded(
                    mp_config.cluster, _wrapper_multiproc_array_topk_histogram_candidates, candidate_arguments
                )
            ]
            while len(histogram_candidates) > 1:
                histogram_candidates = [
                    _merge_topk_histogram_candidates(histogram_candidates[start : start + 2], largest_chunk)
                    for start in range(0, len(histogram_candidates), 2)
                ]
            return histogram_candidates[0]

        histogram_arguments = (
            (
                source_raster,
                tile,
                seed,
                prefix,
                prefix_bits,
                digit_bits,
                band,
                skip_nodata,
                mask,
            )
            for tile in tiles
        )
        histogram = np.zeros(1 << digit_bits, dtype=np.int64)
        for _, tile_histogram in _map_bounded(
            mp_config.cluster, _wrapper_multiproc_array_topk_histogram, histogram_arguments
        ):
            histogram += tile_histogram
        return histogram, None

    def keys_for_prefix(seed: int, prefix: int, prefix_bits: int, group_size: int) -> NDArray[np.uint64]:
        """Collect keys in the last range chosen for the requested subsample from every multiprocessing tile."""

        arguments = ((source_raster, tile, seed, prefix, prefix_bits, band, skip_nodata, mask) for tile in tiles)
        group_keys = np.empty(group_size, dtype=np.uint64)
        offset = 0
        for _, tile_keys in _map_bounded(mp_config.cluster, _wrapper_multiproc_array_topk_prefix_keys, arguments):
            stop = offset + len(tile_keys)
            group_keys[offset:stop] = tile_keys
            offset = stop
        return group_keys[:offset]

    return _iterative_topk_cutoff(
        subsample,
        subsample_meta,
        largest_chunk,
        int(np.prod(source_raster.shape)),
        len(tiles),
        histogram_for_prefix,
        keys_for_prefix,
    )


def _wrapper_multiproc_cutoff_block(
    rst: RasterBase,
    tile_idx: NDArrayNum,
    seed: int,
    cutoff: np.uint64 | None,
    return_indices: bool,
    band: int,
    skip_nodata: bool,
    mask: RasterLike | VectorLike | ArrayLike | None,
) -> tuple[NDArray[np.uint64], NDArrayNum]:
    """Select random keys and their value or index payload below a cutoff in one raster tile."""

    # Read the tile once and build the global positions of its eligible cells
    array, valid = _read_subsample_raster_block(
        rst,
        tile_idx,
        band=band,
        skip_nodata=skip_nodata,
        mask=mask,
    )
    local_indices = np.flatnonzero(valid.ravel())
    local_rows, local_columns = np.unravel_index(local_indices, array.shape)
    rows = local_rows.astype(np.int64, copy=False) + int(tile_idx[0])
    columns = local_columns.astype(np.int64, copy=False) + int(tile_idx[2])
    global_indices = rows * np.int64(rst.shape[1]) + columns

    # Keep the exact selected set, using source indexes to order a complete sample
    if cutoff is None:
        keys = global_indices.astype(np.uint64)
        selected: slice | NDArrayBool = slice(None)
    else:
        keys, selected_by_cutoff = _select_topk_keys(global_indices, seed, cutoff=cutoff)
        assert selected_by_cutoff is not None
        selected = cast(NDArrayBool, selected_by_cutoff)
    if return_indices:
        return keys, np.column_stack((rows[selected], columns[selected]))
    return keys, np.asarray(array).ravel()[local_indices[selected]]


def _write_multiproc_subsample_npy(
    rst: RasterBase,
    tile_ids: list[NDArrayNum],
    subsample_meta: SubsampleMeta,
    config: MultiprocConfig,
    return_indices: bool,
    band: int,
    skip_nodata: bool,
    mask: RasterLike | VectorLike | ArrayLike | None,
) -> NDArrayNum | tuple[NDArrayNum, NDArrayNum]:
    """Write a large value or index subsample to a NumPy memory-mapped file one tile at a time."""

    from geoutils.multiproc.cluster import _map_bounded

    output_path = pathlib.Path(config.outfile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_shape = (2, subsample_meta.sample_size) if return_indices else (subsample_meta.sample_size,)
    output_dtype = np.dtype(np.int64) if return_indices else np.dtype(rst.dtype)

    # Store keys beside their compact payload so an in-place bounded-memory sort preserves top-k order
    record_fields: list[tuple[str, Any]] = [("key", np.uint64)]
    record_fields.extend([("row", np.int64), ("column", np.int64)] if return_indices else [("value", output_dtype)])
    with tempfile.TemporaryDirectory(prefix=".geoutils-subsample-", dir=output_path.parent) as directory:
        temporary_output = pathlib.Path(directory) / "output.npy"
        with ExitStack() as memory_maps:
            records = np.lib.format.open_memmap(
                pathlib.Path(directory) / "selected.npy",
                mode="w+",
                dtype=np.dtype(record_fields),
                shape=(subsample_meta.sample_size,),
            )
            memory_maps.callback(records._mmap.close)

            # Keep worker results bounded while collecting every selected key and payload
            arguments = (
                (
                    rst,
                    tile,
                    subsample_meta.seed,
                    subsample_meta.cutoff,
                    return_indices,
                    band,
                    skip_nodata,
                    mask,
                )
                for tile in tile_ids
            )
            offset = 0
            for _, (keys, selected) in _map_bounded(config.cluster, _wrapper_multiproc_cutoff_block, arguments):
                selected = np.asarray(selected)
                count = len(selected)
                records["key"][offset : offset + count] = keys
                if return_indices:
                    records["row"][offset : offset + count] = selected[:, 0]
                    records["column"][offset : offset + count] = selected[:, 1]
                else:
                    records["value"][offset : offset + count] = selected
                offset += count

            if offset != subsample_meta.sample_size:
                raise RuntimeError("The number of selected values changed while writing the subsample.")
            records.sort(order="key", kind="heapsort")

            # Copy only the requested 1D values or 2D indexes into the public NumPy file
            output = np.lib.format.open_memmap(temporary_output, mode="w+", dtype=output_dtype, shape=output_shape)
            memory_maps.callback(output._mmap.close)
            chunk_size = max(int((tile[1] - tile[0]) * (tile[3] - tile[2])) for tile in tile_ids)
            for start in range(0, subsample_meta.sample_size, chunk_size):
                stop = min(subsample_meta.sample_size, start + chunk_size)
                if return_indices:
                    output[0, start:stop] = records["row"][start:stop]
                    output[1, start:stop] = records["column"][start:stop]
                else:
                    output[start:stop] = records["value"][start:stop]
            output.flush()

        # Close every map before moving files, which Windows does not allow while a file is mapped
        temporary_output.replace(output_path)

    # Reopen read-only so the returned arrays remain file-backed
    stored = np.load(output_path, mmap_mode="r")
    if return_indices:
        return stored[0], stored[1]
    return stored


def _write_multiproc_point_subsample_npy(
    sampled: Any,
    filename: str,
    return_indices: bool,
) -> NDArrayNum | tuple[NDArrayNum, ...]:
    """Write point value or index partitions to the NumPy file requested by the multiprocessing backend."""

    dask = import_optional("dask")
    arrays = list(sampled) if return_indices else [sampled]
    sample_size = int(arrays[0].shape[0])
    output_shape = (len(arrays), sample_size) if return_indices else (sample_size,)
    output_dtype = np.dtype(np.int64) if return_indices else np.dtype(arrays[0].dtype)
    output_path = pathlib.Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write a replacement file so existing memory maps of the same configured path remain valid
    with tempfile.TemporaryDirectory(prefix=".geoutils-subsample-", dir=output_path.parent) as directory:
        temporary_output = pathlib.Path(directory) / "output.npy"
        with ExitStack() as memory_maps:
            output = np.lib.format.open_memmap(temporary_output, mode="w+", dtype=output_dtype, shape=output_shape)
            memory_maps.callback(output._mmap.close)
            for axis, array in enumerate(arrays):
                offset = 0
                for delayed_part in array.to_delayed().ravel():
                    part = np.asarray(dask.compute(delayed_part)[0]).ravel()
                    stop = offset + len(part)
                    if return_indices:
                        output[axis, offset:stop] = part
                    else:
                        output[offset:stop] = part
                    offset = stop
                if offset != sample_size:
                    raise RuntimeError("The number of selected values changed while writing the subsample.")
            output.flush()

        # Close the map before moving its file, which Windows does not allow while the file is mapped
        temporary_output.replace(output_path)

    stored = np.load(output_path, mmap_mode="r")
    if return_indices:
        return tuple(stored[axis] for axis in range(len(arrays)))
    return stored


def _multiproc_subsample(
    rst: RasterBase,
    config: MultiprocConfig,
    subsample: int | float = 1,
    return_indices: bool = False,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "topk",
    *,
    band: int = 1,
    skip_nodata: bool = True,
    mask: RasterLike | VectorLike | ArrayLike | None = None,
    return_linear_indices: bool = False,
    force_output_to_memory: bool = False,
) -> NDArrayNum | tuple[NDArrayNum, NDArrayNum]:
    """
    Subsample valid values out-of-memory from a 2D raster array using Multiprocessing tasks.

    Strategy "topk" is chunk-invariant (same sample no matter chunk size, and same as the NumPy implementation), while
    "sequential" is chunk-dependent but slightly faster.

    Returns a concatenated subsampled NumPy array collected from all tasks (either values or indices).
    """

    if return_linear_indices and (strategy != "topk" or subsample == 1):
        raise ValueError("Linear index output is only available for partial top-k sampling.")

    # Return complete unrestricted grid indexes without opening the source file
    if subsample == 1 and not skip_nodata and mask is None and return_indices:
        flat_indices = np.arange(int(np.prod(rst.shape)), dtype=np.int64)
        rows, columns = np.unravel_index(flat_indices, rst.shape)
        return rows.astype(np.int64), columns.astype(np.int64)

    # 1/ Prepare raster tiles and valid counts

    # Get tiling
    tiling = compute_tiling(tile_size=config.chunks, raster_shape=rst.shape, overlap=0)

    # Get number of chunks and blocks
    num_chunks = (tiling.shape[0], tiling.shape[1])
    num_blocks = int(np.prod(num_chunks))

    # Flatten tile_idx list in row-major block order
    indexes_row, indexes_col = np.unravel_index(np.arange(num_blocks), shape=num_chunks)
    tile_ids = [tiling[indexes_row[i], indexes_col[i], :] for i in range(num_blocks)]

    # Count valid values per tile in parallel, unless skip_nodata=False makes every cell valid
    if not skip_nodata and mask is None:
        nb_valids_per_block = np.array([(tile[1] - tile[0]) * (tile[3] - tile[2]) for tile in tile_ids], dtype=np.int64)
    else:
        tasks = [
            config.cluster.submit(
                _wrapper_multiproc_nb_valids_per_block,
                rst,
                tile_ids[i],
                band=band,
                skip_nodata=skip_nodata,
                mask=mask,
            )
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
        if return_linear_indices:
            return np.empty((0,), dtype=np.int64)
        if return_indices:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
        return np.empty((0,), dtype=rst.dtype)

    # 2/ Build the requested sample

    # 2A/ Write large complete or "topk" samples to a NumPy file
    largest_chunk = max(int((tile[1] - tile[0]) * (tile[3] - tile[2])) for tile in tile_ids)
    sample_exceeds_largest_chunk = _subsample_exceeds_largest_chunk(subsample_size, largest_chunk)
    write_chunked = sample_exceeds_largest_chunk and not force_output_to_memory and not return_linear_indices
    if write_chunked and (subsample == 1 or strategy == "topk"):
        if subsample == 1:
            subsample_meta = SubsampleMeta(sample_size=subsample_size, seed=0, cutoff=None)
        else:
            subsample_meta = SubsampleMeta(
                sample_size=subsample_size,
                seed=_resolve_topk_seed(random_state),
                cutoff=None,
            )
            subsample_meta = _multiproc_array_topk_cutoff(
                source_raster=rst,
                tiles=np.asarray(tile_ids),
                largest_chunk=largest_chunk,
                subsample=subsample,
                subsample_meta=subsample_meta,
                band=band,
                skip_nodata=skip_nodata,
                mp_config=config,
                mask=mask,
            )
        return _write_multiproc_subsample_npy(
            rst,
            tile_ids,
            subsample_meta,
            config,
            return_indices,
            band,
            skip_nodata,
            mask,
        )

    # 2B/ Select "sequential" samples by position within each tile
    if strategy == "sequential" or subsample == 1:
        ind_per_block, selected_counts, _ = _prepare_sequential_subsample(
            nb_valids_per_block,
            subsample_size,
            random_state,
            select_all=subsample == 1,
        )

        used = np.flatnonzero(selected_counts).tolist()

        # Sample them through multiprocessing, either for indices or values
        if not return_indices:
            tasks = [
                config.cluster.submit(
                    _wrapper_multiproc_subsample_values_block,
                    rst,
                    tile_ids[i],
                    (
                        ind_per_block[i]
                        if isinstance(ind_per_block[i], slice)
                        else np.asarray(ind_per_block[i], dtype=np.int64)
                    ),
                    band=band,
                    skip_nodata=skip_nodata,
                    return_global_indices=subsample == 1,
                    mask=mask,
                )
                for i in used
            ]

            try:
                sampled_values = config.cluster.gather(tasks)
            except Exception as e:
                raise RuntimeError(f"Error retrieving subsampled values from multiprocessing tasks: {e}")

            # Restore a complete raster to its original row-by-row order
            if subsample == 1:
                global_indices = np.concatenate([indexes for indexes, _ in sampled_values])
                values = np.concatenate([values for _, values in sampled_values])
                return values[np.argsort(global_indices)]

            # Concatenate in tile order (this yields deterministic order given tiling; not random order)
            return np.concatenate(sampled_values, axis=0)

        else:
            tasks = [
                config.cluster.submit(
                    _wrapper_multiproc_subsample_indices_block,
                    rst,
                    tile_ids[i],
                    (
                        ind_per_block[i]
                        if isinstance(ind_per_block[i], slice)
                        else np.asarray(ind_per_block[i], dtype=np.int64)
                    ),
                    band=band,
                    skip_nodata=skip_nodata,
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
            if subsample == 1:
                order = np.argsort(rows * np.int64(rst.shape[1]) + cols)
                rows = rows[order]
                cols = cols[order]
            return rows, cols

    # 2C/ Select bounded "topk" candidates from every tile
    elif strategy == "topk":

        # Convert random_state to an integer seed used in deterministic keys
        seed = _resolve_topk_seed(random_state)

        # Get full-array width
        nx_full = int(rst.shape[1])
        return_keys_only = return_indices or return_linear_indices

        from geoutils.multiproc.cluster import _map_bounded

        arguments = (
            (
                rst,
                tile_ids[i],
                seed,
                subsample_size,
                nx_full,
                return_keys_only,
                return_keys_only,
                band,
                skip_nodata,
                mask,
            )
            for i in range(num_blocks)
        )

        # Keep only keys that can be changed back to indexes, avoiding a second full-size array during each merge
        if return_keys_only:
            max_tile_size = max(int((tile[1] - tile[0]) * (tile[3] - tile[2])) for tile in tile_ids)
            max_candidates_per_tile = min(subsample_size, max_tile_size)
            merge_group_size = 8
            key_buffer = np.empty(subsample_size + merge_group_size * max_candidates_per_tile, dtype=np.uint64)
            retained = 0
            pending = 0
            pending_tiles = 0

            try:
                for _, candidate_keys in _map_bounded(
                    config.cluster, _wrapper_multiproc_topk_candidates_block_positional, arguments
                ):
                    candidates = np.asarray(candidate_keys, dtype=np.uint64).ravel()
                    key_buffer[retained + pending : retained + pending + len(candidates)] = candidates
                    pending += len(candidates)
                    pending_tiles += 1

                    # Reduce several tile results in the same reusable array to avoid repeated scans of the sample
                    if pending_tiles == merge_group_size:
                        merged_size = retained + pending
                        if merged_size > subsample_size:
                            key_buffer[:merged_size].partition(subsample_size - 1)
                        retained = min(subsample_size, merged_size)
                        pending = 0
                        pending_tiles = 0
            except Exception as e:
                raise RuntimeError(f"Error retrieving topk candidates from multiprocessing tasks: {e}")

            # Reduce the final shorter group, then restore cell indexes in their deterministic key order
            merged_size = retained + pending
            if merged_size > subsample_size:
                key_buffer[:merged_size].partition(subsample_size - 1)
            retained = min(subsample_size, merged_size)
            key_buffer.resize(retained, refcheck=False)
            key_buffer.sort()
            gid = _recover_splitmix64_indices(key_buffer, seed)

            if return_linear_indices:
                return gid

            # Reuse the recovered index array for columns so only the two public result arrays remain
            rows = np.empty_like(gid)
            np.floor_divide(gid, np.int64(nx_full), out=rows)
            np.remainder(gid, np.int64(nx_full), out=gid)
            return rows, gid

        # Merge bounded groups of completed tiles so memory depends on the sample size, not the number of tiles
        keys: NDArrayNum = np.empty((0,), dtype=np.uint64)
        payload = np.empty((0,), dtype=np.int64 if return_indices or return_linear_indices else rst.dtype)
        pending_keys: list[NDArrayNum] = []
        pending_payloads: list[NDArrayNum] = []
        try:
            for _, (candidate_keys, candidate_payload) in _map_bounded(
                config.cluster, _wrapper_multiproc_topk_candidates_block_positional, arguments
            ):
                pending_keys.append(np.asarray(candidate_keys, dtype=np.uint64).ravel())
                pending_payloads.append(np.asarray(candidate_payload).ravel())
                if len(pending_keys) == 8:
                    keys, payload = _merge_topk_candidates(
                        [keys, *pending_keys], [payload, *pending_payloads], k=subsample_size
                    )
                    pending_keys.clear()
                    pending_payloads.clear()
        except Exception as e:
            raise RuntimeError(f"Error retrieving topk candidates from multiprocessing tasks: {e}")

        if pending_keys:
            keys, payload = _merge_topk_candidates(
                [keys, *pending_keys], [payload, *pending_payloads], k=subsample_size
            )

        if keys.size == 0:
            if return_linear_indices:
                return np.empty((0,), dtype=np.int64)
            if return_indices:
                return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
            return np.empty((0,), dtype=rst.dtype)

        payload_sel = payload

        if not (return_indices or return_linear_indices):
            return payload_sel

        if return_linear_indices:
            return payload_sel.astype(np.int64, copy=False)

        # payload is gid -> (row, col)
        gid = payload_sel.astype(np.int64, copy=False)
        rows = gid // np.int64(nx_full)
        cols = gid % np.int64(nx_full)
        return rows, cols

    else:
        raise ValueError(f"Unknown ``strategy`` {strategy!r}. Choose 'sequential' or 'topk'.")


#################################
# 1E/ ARRAY SUBSAMPLING PARENT
#################################


def _subsample(
    source_raster: RasterBase,
    subsample: float | int = 1,
    band: int = 1,
    return_indices: bool = False,
    *,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "topk",
    mp_config: MultiprocConfig | None = None,
    skip_nodata: bool = True,
    mask: RasterLike | VectorLike | ArrayLike | None = None,
    return_linear_indices: bool = False,
    force_output_to_memory: bool = False,
) -> Any:
    """
    Subsample an array at either all or only at valid values, dispatching automatically to NumPy,
    Dask or Multiprocessing implementation.

    :param source_raster: Raster or raster accessor providing band values and their optional Dask chunks.
    :param subsample: Subsample size, either as a fraction of values (0 to 1), or maximum number of values (above 1).
        Use 1 to return all samples.
    :param band: Raster band to subsample, counting from one.
    :param return_indices: If True, return (rows, cols) indices instead of values.
    :param random_state: Seed or Generator.
    :param strategy: Either "sequential" (chunk/order dependent) or "topk" (chunk-invariant).
    :param mp_config: Tile sizes and worker cluster for multiprocessing. Cannot be combined with a Dask source.
    :param skip_nodata: Whether to exclude nodata values, and False cells in a boolean raster.
    :param mask: Boolean array, aligned mask raster, or vector geometries restricting eligible cells.
    :param return_linear_indices: Return the multiprocessing sample as one compact index array.

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
        skip_nodata: bool

    subsample_kwargs: _SubsampleKwargs = {
        "subsample": subsample,
        "return_indices": return_indices,
        "random_state": random_state,
        "strategy": strategy,
        "skip_nodata": skip_nodata,
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
            sampling_raster,
            config=mp_config,
            band=sampling_band,
            mask=mask,
            return_linear_indices=return_linear_indices,
            force_output_to_memory=force_output_to_memory,
            **subsample_kwargs,
        )

    if return_linear_indices:
        raise ValueError("Linear index output is only available for internal multiprocessing calls.")

    # Read one band without converting masked integer values to floating point
    data = _as_array(source_raster.data)
    arr = data if data.ndim == 2 else data[band - 1]
    mask_array = _mask_at_support(mask, source_raster)

    # Keep Dask data and mask blocks lazy; an eager source collects only a supplied lazy mask
    if dask_backend:
        return _dask_subsample(
            arr,
            mask=mask_array,
            force_output_to_memory=force_output_to_memory,
            **subsample_kwargs,
        )
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
            force_output_to_memory=True,
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


########################
# 2/ POINT SUBSAMPLING
########################


def _read_point_subsample_values(reader: Any, rows: slice) -> NDArrayNum:
    """Read one consecutive point value partition from a multiprocessing file reader."""

    from geoutils.multiproc.readers import _read_values

    return np.asarray(_read_values(reader.block(rows)))


def _subsample_pointcloud_array(
    source_pointcloud: PointCloudBase,
    subsample: float | int,
    return_indices: bool = False,
    random_state: int | np.random.Generator | None = None,
    *,
    strategy: Literal["sequential", "topk"] = "topk",
    mp_config: MultiprocConfig | None = None,
    force_output_to_memory: bool = False,
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
    :param strategy: Sampling strategy passed to the shared array subsampler.
    :param mp_config: Point partition size and NumPy output path for a large multiprocessing result. Cannot be
        combined with a Dask source.
    :param force_output_to_memory: Bypass cutoff selection when the sample is larger than one partition.
    :param mask: Boolean array or spatial mask restricting eligible point rows.
    :returns: Computed one-dimensional NumPy values, or a one-element tuple of indices into the original row order.
    """

    from geoutils.sampling.support import _as_array, _mask_at_support

    if mp_config is not None and source_pointcloud._is_dask:
        raise ValueError("Cannot use Multiprocessing and Dask simultaneously. Remove ``mp_config`` or use eager data.")

    point_partition_size = None
    if mp_config is not None:
        from geoutils.pointcloud.las import _point_partition_size

        point_partition_size = _point_partition_size(mp_config)

    # Keep an unloaded point file in bounded row partitions when no spatial mask requires its coordinates
    data = None
    partition_lengths = None
    mask_array = None
    if mp_config is not None and not source_pointcloud.is_loaded and not source_pointcloud._is_pd and mask is None:
        from geoutils.multiproc.readers import _reader_from_source

        reader = _reader_from_source(source_pointcloud, None, source_pointcloud, mp_config)
        if reader is not None:
            import dask.array as da

            assert point_partition_size is not None
            row_slices = [
                slice(start, min(start + point_partition_size, reader.shape[0]))
                for start in range(0, reader.shape[0], point_partition_size)
            ]
            partition_lengths = tuple(rows.stop - rows.start for rows in row_slices)
            delayed_parts = [delayed(_read_point_subsample_values)(reader, rows) for rows in row_slices]
            parts = [
                da.from_delayed(part, shape=(length,), dtype=reader.dtype)
                for part, length in zip(delayed_parts, partition_lengths)
            ]
            data = da.concatenate(parts) if parts else np.empty(0, dtype=reader.dtype)

    # Read native values and place an optional mask on the same point rows
    if data is None:
        data = _as_array(source_pointcloud.data)
        partition_lengths = tuple(int(length) for length in data.chunks[0]) if source_pointcloud._is_dask else None
        mask_array = _mask_at_support(mask, source_pointcloud, point_partition_lengths=partition_lengths)

    # Let the existing Dask sampler collect only the requested values or row positions from point partitions
    if is_dask_array(data):
        dask_data: Any = data
        sampled = _dask_subsample(
            dask_data[:, None],
            subsample=subsample,
            return_indices=return_indices,
            random_state=random_state,
            strategy=strategy,
            preserve_order=True,
            mask=None if mask_array is None else mask_array[:, None],
            force_output_to_memory=force_output_to_memory,
        )
        if return_indices:
            assert isinstance(sampled, tuple)
            first_array = sampled[0]
            point_sample: Any = (first_array,)
        else:
            assert not isinstance(sampled, tuple)
            first_array = sampled
            point_sample = first_array
        if not is_dask_array(first_array):
            return point_sample if return_indices else first_array
        sample_size = int(first_array.shape[0])
        largest_partition = max(int(length) for length in dask_data.chunks[0])
        if mp_config is not None and sample_size > largest_partition and not force_output_to_memory:
            return _write_multiproc_point_subsample_npy(point_sample, mp_config.outfile, return_indices)
        if return_indices:
            return (first_array.compute(),)
        return first_array.compute()

    # Preserve the established NumPy sampling rules for eager point data
    if mask_array is not None and is_dask_array(mask_array):
        mask_array = mask_array.compute()
    if mp_config is not None and not force_output_to_memory:
        assert point_partition_size is not None
        valid = _valid_subsample_mask(data, mask_array, skip_nodata=True)
        sample_size = _get_subsample_size_from_user_input(subsample, int(np.count_nonzero(valid)))
        if sample_size > point_partition_size:
            import dask.array as da

            lazy_data = da.from_array(data, chunks=point_partition_size)
            lazy_mask = None if mask_array is None else da.from_array(mask_array, chunks=point_partition_size)
            sampled = _dask_subsample(
                lazy_data[:, None],
                subsample=subsample,
                return_indices=return_indices,
                random_state=random_state,
                strategy=strategy,
                preserve_order=True,
                mask=None if lazy_mask is None else lazy_mask[:, None],
            )
            point_sample = (sampled[0],) if return_indices else sampled
            return _write_multiproc_point_subsample_npy(point_sample, mp_config.outfile, return_indices)
    if return_indices:
        return _subsample_numpy(
            data,
            subsample,
            return_indices=True,
            random_state=random_state,
            strategy=strategy,
            mask=mask_array,
        )
    return _subsample_numpy(
        data,
        subsample,
        return_indices=False,
        random_state=random_state,
        strategy=strategy,
        mask=mask_array,
    )


def _select_point_subsample_partition(
    dataframe: gpd.GeoDataFrame,
    positions: NDArray[np.int64],
    sample_order: NDArray[np.int64],
) -> gpd.GeoDataFrame:
    """Select rows from one point partition and record their order in the complete sample."""

    selected = dataframe.iloc[positions].copy()
    selected["_geoutils_subsample_order"] = sample_order
    return selected


def _pointcloud_from_subsample_indices(
    source_pointcloud: PointCloudBase,
    indices: NDArrayNum,
) -> Any:
    """Build an eager or lazy point cloud containing the sampled source rows."""

    positions = np.asarray(indices, dtype=np.int64)
    if not source_pointcloud._is_dask:
        selected = source_pointcloud.ds.iloc[positions].copy()
        return source_pointcloud._cast_pointcloud_output(selected)

    # Split global row positions between the existing Dask dataframe partitions
    dask = import_optional("dask")
    partition_lengths = np.asarray(source_pointcloud.ds.map_partitions(len).compute(), dtype=np.int64)
    partition_starts = np.cumsum(np.concatenate(([0], partition_lengths[:-1])))
    selected_parts = []
    for dataframe, start, length in zip(source_pointcloud.ds.to_delayed(), partition_starts, partition_lengths):
        sample_positions = np.flatnonzero((positions >= start) & (positions < start + length)).astype(np.int64)
        if len(sample_positions) == 0:
            continue
        local_positions = positions[sample_positions] - start
        selected_parts.append(
            dask.delayed(_select_point_subsample_partition)(dataframe, local_positions, sample_positions)
        )

    from geoutils.pointcloud.dataframe import _import_dask_dataframe
    from geoutils.vector.pd_accessor import _import_dask_geopandas

    # Restore random-draw or key order after rows from the same source partition have been selected together
    meta = source_pointcloud.ds._meta.copy()
    meta["_geoutils_subsample_order"] = pd.Series(dtype=np.int64)
    dask_dataframe = _import_dask_dataframe()
    if selected_parts:
        selected = dask_dataframe.from_delayed(selected_parts, meta=meta).sort_values("_geoutils_subsample_order")
        selected = selected.drop(columns="_geoutils_subsample_order")
    else:
        selected = dask_dataframe.from_pandas(meta.drop(columns="_geoutils_subsample_order"), npartitions=1)
    selected = _import_dask_geopandas().from_dask_dataframe(
        selected,
        geometry=source_pointcloud.ds._meta.geometry.name,
    )
    return source_pointcloud._cast_pointcloud_output(selected)


def _stage_point_subsample_partition(
    source: gpd.GeoDataFrame | pathlib.Path,
    columns: list[str],
    start: int,
    count: int,
    positions: NDArray[np.int64],
    filename: pathlib.Path,
    las_output: bool,
    las_elevation_column: str | None,
) -> tuple[pathlib.Path, np.ndarray[Any, Any] | None]:
    """Read and stage selected point rows, returning their coordinate bounds for LAS/LAZ output."""

    # Read only this source partition when the point cloud remains file-backed
    if isinstance(source, pathlib.Path):
        from geoutils.pointcloud.las import _is_laspy_supported, _load_laspy_data_slice

        if _is_laspy_supported(source):
            dataframe = _load_laspy_data_slice(source, columns=columns, start=start, count=count)
        else:
            import pyogrio

            dataframe = pyogrio.read_dataframe(source, skip_features=start, max_features=max(1, count))
            if count == 0:
                dataframe = dataframe.iloc[:0]
    else:
        dataframe = source

    # Keep all geometry and attribute columns for the selected point rows
    from geoutils.pointcloud.writing import _stage_pointcloud_partition

    selected = dataframe.iloc[positions].copy()
    bounds = None
    if las_output:
        from geoutils.pointcloud.las import _las_coordinate_bounds

        bounds = _las_coordinate_bounds(selected, las_elevation_column)
    return _stage_pointcloud_partition(selected, filename), bounds


def _multiproc_subsample_pointcloud(
    source_pointcloud: PointCloudBase,
    subsample: float | int,
    random_state: int | np.random.Generator | None,
    strategy: Literal["sequential", "topk"],
    mp_config: MultiprocConfig,
    mask: RasterLike | PointCloudLike | VectorLike | ArrayLike | None,
) -> Any:
    """Write sampled point rows in source partitions and return the file-backed point cloud."""

    from geoutils.multiproc.cluster import _map_bounded
    from geoutils.pointcloud.las import _is_laspy_supported, _point_partition_size
    from geoutils.pointcloud.writing import (
        _resolve_pointcloud_output,
        _write_pointcloud_partitions,
    )

    if source_pointcloud._is_dask:
        raise ValueError("Cannot use Multiprocessing and Dask simultaneously. Remove ``mp_config`` or use eager data.")
    partition_size = _point_partition_size(mp_config)
    output_filename, driver = _resolve_pointcloud_output(
        mp_config.outfile,
        mp_config.driver,
        supported_drivers=("GPKG", "LAS", "LAZ"),
        operation_name="point cloud subsampling",
    )
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    # Keep unopened GPKG/LAS sources on disk and split eager dataframes before submitting them
    source_filename = pathlib.Path(source_pointcloud.name) if source_pointcloud.name is not None else None
    if not source_pointcloud.is_loaded:
        if source_filename is None:
            raise ValueError("Unloaded point cloud subsampling requires a file-backed source.")
        if not _is_laspy_supported(source_filename):
            import pyogrio

            if pyogrio.read_info(source_filename)["driver"] != "GPKG":
                raise ValueError("Unloaded point cloud subsampling supports LAS, LAZ and GPKG sources.")
        dataframe = None
    else:
        dataframe = source_pointcloud.ds

    # Select row positions through the array path, using a temporary NumPy file for a large sample
    with mp_config.temporary() as index_config, ExitStack() as memory_maps:
        index_config.outfile += ".npy"
        sampled_indices = _subsample_pointcloud_array(
            source_pointcloud,
            subsample,
            return_indices=True,
            random_state=random_state,
            strategy=strategy,
            mp_config=index_config,
            mask=mask,
        )[0]
        if isinstance(sampled_indices, np.memmap):
            sampled_indices._mmap.close()
            sampled_indices = np.load(index_config.outfile, mmap_mode="r+")[0]
            memory_maps.callback(sampled_indices._mmap.close)
        else:
            sampled_indices = np.asarray(sampled_indices, dtype=np.int64)
        sampled_indices.sort(kind="heapsort")
        if isinstance(sampled_indices, np.memmap):
            sampled_indices.flush()

        # Stage every nonempty source partition in row order without holding the sampled point cloud in memory
        with tempfile.TemporaryDirectory(prefix=".geoutils-point-subsample-", dir=output_filename.parent) as directory:
            temporary_directory = pathlib.Path(directory)
            arguments = []
            point_count = source_pointcloud.point_count
            for start in range(0, max(point_count, 1), partition_size):
                count = min(partition_size, point_count - start)
                left = int(np.searchsorted(sampled_indices, start, side="left"))
                right = int(np.searchsorted(sampled_indices, start + count, side="left"))
                if left == right and len(sampled_indices) > 0:
                    continue
                local_positions = sampled_indices[left:right] - start
                partition_source = source_filename if dataframe is None else dataframe.iloc[start : start + count]
                arguments.append(
                    (
                        partition_source,
                        list(source_pointcloud._nongeo_columns),
                        start,
                        count,
                        local_positions,
                        temporary_directory / f"partition_{start}.pkl",
                        driver != "GPKG",
                        source_pointcloud.data_column,
                    )
                )

            partition_results = [
                result
                for _, result in _map_bounded(
                    mp_config.cluster,
                    _stage_point_subsample_partition,
                    arguments,
                )
            ]
            partition_filenames = [filename for filename, _ in partition_results]
            pointcloud = _write_pointcloud_partitions(
                output_filename,
                partition_filenames,
                driver=driver,
                data_column=source_pointcloud.data_column if driver == "GPKG" else None,
                geometry_type="Point Z" if source_pointcloud._has_z else "Point",
                las_elevation_column=source_pointcloud.data_column,
                las_bounds=[bounds for _, bounds in partition_results] if driver != "GPKG" else None,
            )

    # Accessors return an eager GeoDataFrame while PointCloud inputs keep the output file unloaded
    if source_pointcloud._is_pd:
        pointcloud.load(columns="all")
        return source_pointcloud._cast_pointcloud_output(pointcloud.ds)
    return pointcloud


def _subsample_pointcloud(
    source_pointcloud: PointCloudBase,
    subsample: float | int,
    return_indices: bool = False,
    random_state: int | np.random.Generator | None = None,
    *,
    as_array: bool = False,
    strategy: Literal["sequential", "topk"] = "topk",
    mp_config: MultiprocConfig | None = None,
    force_output_to_memory: bool = False,
    mask: RasterLike | PointCloudLike | VectorLike | ArrayLike | None = None,
) -> Any:
    """
    Subsample point cloud, or return their values or positions as arrays.

    _subsample_pointcloud_array() selects values or source row positions for every backend. Array requests return
    those results directly. Point requests pass the selected positions to _pointcloud_from_subsample_indices(), or
    to _multiproc_subsample_pointcloud() when complete rows should be written to a file in bounded partitions.
    """

    # Keep the existing value/index path for explicit array output
    if as_array:
        return _subsample_pointcloud_array(
            source_pointcloud,
            subsample,
            return_indices=return_indices,
            random_state=random_state,
            strategy=strategy,
            mp_config=mp_config,
            force_output_to_memory=force_output_to_memory,
            mask=mask,
        )
    if return_indices:
        raise ValueError("Argument ``return_indices=True`` requires ``as_array=True``.")

    # Multiprocessing writes point rows to file unless we force to memory
    if mp_config is not None and not force_output_to_memory:
        return _multiproc_subsample_pointcloud(
            source_pointcloud,
            subsample,
            random_state,
            strategy,
            mp_config,
            mask,
        )

    sampled_indices = _subsample_pointcloud_array(
        source_pointcloud,
        subsample,
        return_indices=True,
        random_state=random_state,
        strategy=strategy,
        force_output_to_memory=force_output_to_memory,
        mask=mask,
    )[0]
    pointcloud = _pointcloud_from_subsample_indices(source_pointcloud, sampled_indices)
    if force_output_to_memory and source_pointcloud._is_dask:
        return source_pointcloud._cast_pointcloud_output(pointcloud.compute())
    return pointcloud


#########################
# 3/ RASTER SUBSAMPLING
#########################


######################
# 3A/ SHARED HELPERS
######################


def _sample_raster_cell_indices(
    source_raster: RasterType,
    data_band: int,
    subsample: float | int,
    skip_nodata: bool,
    random_state: int | np.random.Generator | None,
    mp_config: MultiprocConfig | None,
    force_output_to_memory: bool = False,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Get row and column indexes of the requested subsample.

    See description of _subsample_raster_from_indices() for details on the overall logic.
    """

    # Use the index selector with "topk" so every output band follows one chunk-independent sample
    indices = _subsample(
        source_raster,
        subsample=subsample,
        band=data_band,
        return_indices=True,
        random_state=random_state,
        strategy="topk",
        mp_config=mp_config,
        skip_nodata=skip_nodata,
        force_output_to_memory=force_output_to_memory,
    )
    if any(is_dask_array(index) for index in indices):
        indices = import_optional("dask").compute(*indices)
    rows, columns = indices
    return np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)


def _extract_raster_cell_values(
    source_raster: RasterType,
    bands: list[int],
    rows: NDArray[np.int64],
    columns: NDArray[np.int64],
    mp_config: MultiprocConfig | None,
) -> Any:
    """
    Read every band at the raster row/columns of the subsample.

    See description of _subsample_raster_from_indices() for details on the overall logic.
    """

    # Choose how to read the requested subsample from a lazy array, memory, or the raster file
    data = source_raster.data if source_raster.is_loaded or source_raster._is_xr else None

    # Dask
    if data is not None and is_dask_array(data):
        import dask.array as da

        # Select each requested band lazily at the row/column of the subsample
        dask_data: Any = data
        band_values = [
            dask_data.vindex[rows, columns] if dask_data.ndim == 2 else dask_data[band - 1].vindex[rows, columns]
            for band in bands
        ]

        # Join the selected bands while keeping their values lazy
        pixel_data = da.stack(band_values, axis=0)

    # Eager
    elif data is not None:
        # Select each requested band directly from the raster array in memory
        band_values = [data[rows, columns] if data.ndim == 2 else data[band - 1, rows, columns] for band in bands]

        # Respect nodata masks when a band has one
        pixel_data = (
            np.ma.stack(band_values, axis=0)
            if any(np.ma.isMaskedArray(values) for values in band_values)
            else np.stack(band_values, axis=0)
        )

    # Multiproc
    else:
        from geoutils.multiproc import MultiprocConfig
        from geoutils.multiproc.cluster import _map_bounded
        from geoutils.multiproc.readers import _read_selected_raster_bands

        # Prepare small raster tiles for reading the requested cells from the file
        read_config = mp_config if mp_config is not None else MultiprocConfig(chunks=512)

        # Convert row/column to a flat index so that every band reads the same raster cells
        flat_indices = rows * source_raster.shape[1] + columns
        chunk_rows, chunk_columns = (
            (read_config.chunks, read_config.chunks) if isinstance(read_config.chunks, int) else read_config.chunks
        )
        tile_columns = (source_raster.shape[1] + chunk_columns - 1) // chunk_columns
        tile_ids = (rows // chunk_rows) * tile_columns + columns // chunk_columns

        # Group the sample positions once so each selected tile reads every requested band in one worker call
        order = np.argsort(tile_ids, kind="stable")
        boundaries = np.flatnonzero(np.diff(tile_ids[order])) + 1
        positions_by_tile = np.split(order, boundaries) if len(order) > 0 else []
        arguments = (
            (source_raster, flat_indices[positions], bands, read_config.chunks) for positions in positions_by_tile
        )
        dtype = np.dtype(bool if source_raster.is_mask else source_raster.dtype)
        pixel_data = np.ma.masked_all((len(bands), len(flat_indices)), dtype=dtype)
        for positions, (_, values) in zip(
            positions_by_tile,
            _map_bounded(read_config.cluster, _read_selected_raster_bands, arguments),
        ):
            pixel_data.data[:, positions] = np.ma.getdata(values)
            pixel_data.mask[:, positions] = np.ma.getmaskarray(values)

    return pixel_data


def _subsample_raster_from_indices(
    source_raster: RasterType,
    bands: list[int],
    column_names: list[str],
    data_column_name: str,
    subsample: float | int,
    skip_nodata: bool,
    random_state: int | np.random.Generator | None,
    force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"],
    as_array: bool,
    read_config: MultiprocConfig | None,
    force_output_to_memory: bool = False,
) -> Any:
    """
    Build the complete point cloud or X/Y/data array for the subsample.

    All subsample indices and values will at some point concatenate in memory at once.

    Thus, this function is used by:
    - Eager,
    - Dask only for sample sizes smaller than raster chunk size, otherwise it requires different logic explained in
        chunked helpers below),
    - Multiproc for small sample sizes too, but also when as_array=True, because Multiproc cannot stream chunk-by-chunk
        to an array like Dask (only to a point cloud file).
    """

    # 1/ Select indices through subsample()
    rows, columns = _sample_raster_cell_indices(
        source_raster=source_raster,
        data_band=bands[0],  # We can use the same indices for all bands
        subsample=subsample,
        skip_nodata=skip_nodata,
        random_state=random_state,
        mp_config=read_config,
        force_output_to_memory=force_output_to_memory,
    )

    # 2/ Read every output band at the positions selected from the main band
    pixel_data = _extract_raster_cell_values(
        source_raster=source_raster,
        bands=bands,
        rows=rows,
        columns=columns,
        mp_config=read_config,
    )

    # 3/ Normalize nodata values after cell selection
    if is_dask_array(pixel_data) and skip_nodata:
        import dask.array as da

        pixel_data = da.ma.getdata(pixel_data)
    elif np.ma.isMaskedArray(pixel_data):
        pixel_data = pixel_data.data

    # Convert retained nodata values to NaN in a floating output array
    if not skip_nodata:
        pixel_data = pixel_data.astype(np.result_type(pixel_data.dtype, np.float32))
        if is_dask_array(pixel_data):
            import dask.array as da

            pixel_data = da.ma.filled(pixel_data, np.nan)
            if source_raster.nodata is not None:
                pixel_data = da.where(pixel_data == source_raster.nodata, np.nan, pixel_data)
        elif source_raster.nodata is not None:
            pixel_data[pixel_data == source_raster.nodata] = np.nan

    # 4/ Calculate coordinates from affine transform and pixel offset
    x_coords, y_coords = _ij2xy(
        i=rows,
        j=columns,
        transform=source_raster.transform,
        area_or_point=source_raster.area_or_point,
        shift_area_or_point=False,
        force_offset=force_pixel_offset,
    )

    # 5/ Build output, lazy for Dask, otherwise eager
    if is_dask_array(pixel_data):
        import dask.array as da

        if not is_dask_array(x_coords):
            x_coords = da.from_array(np.asarray(x_coords), chunks=pixel_data.chunks[1])
            y_coords = da.from_array(np.asarray(y_coords), chunks=pixel_data.chunks[1])
        if as_array:
            return da.stack((x_coords, y_coords, *[pixel_data[index] for index in range(len(bands))]), axis=1)

        from geoutils.pointcloud.dataframe import (
            _build_pointcloud_output,
            _import_dask_dataframe,
        )
        from geoutils.vector.pd_accessor import _import_dask_geopandas

        # Build matching Dask series so dataframe values keep their raster dtype and continuous point index
        point_chunks = pixel_data.chunks[1]
        x_coords = x_coords.rechunk(point_chunks)
        y_coords = y_coords.rechunk(point_chunks)
        dask_dataframe = _import_dask_dataframe()
        dask_geopandas = _import_dask_geopandas()
        dataframe = dask_dataframe.from_dask_array(pixel_data.T, columns=column_names)
        coordinate_frame = dask_dataframe.concat(
            [
                dask_dataframe.from_dask_array(x_coords, columns="x"),
                dask_dataframe.from_dask_array(y_coords, columns="y"),
            ],
            axis=1,
        )
        geometry = dask_geopandas.points_from_xy(coordinate_frame, x="x", y="y", crs=source_raster.crs)
        dataframe = dataframe.assign(geometry=geometry)
        dataframe = dask_geopandas.from_dask_dataframe(dataframe, geometry="geometry")

        # Finalize point metadata; the shared builder also adds ``.pc`` and ``.vct`` to this Dask frame
        return _build_pointcloud_output(dataframe, data_column=data_column_name, as_dataframe=True)

    # Build an eager array or PointCloud result
    if as_array:
        return np.vstack((np.asarray(x_coords), np.asarray(y_coords), pixel_data)).T

    from geoutils.pointcloud import PointCloud

    dataframe = gpd.GeoDataFrame(
        pixel_data.T,
        columns=column_names,
        geometry=gpd.points_from_xy(np.asarray(x_coords), np.asarray(y_coords)),
        crs=source_raster.crs,
    )
    return PointCloud(dataframe, data_column=data_column_name)


##########################
# 3B/ CHUNKED HELPERS
##########################


def _raster_values_to_point_partition(
    band_values: Any,
    selected_indices: NDArray[np.int64],
    raster_shape: tuple[int, int],
    transform: affine.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    crs: CRS | None,
    nodata: int | float | None,
    column_names: list[str],
    skip_nodata: bool,
    force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"],
    topk_selection: tuple[int, np.uint64] | None,
    as_array: bool,
) -> NDArrayNum | gpd.GeoDataFrame:
    """
    Convert the values and global indexes of cells selected in one chunk into point rows.

    Used only when the requested subsample is larger than one chunk.

    The objective is eventually to write each raster chunk subsamples out-of-memory into a point partition of the point
    file output.
    """

    band_values = band_values.reshape((len(column_names), -1))

    # Keep cells with random key at or below the cutoff
    if topk_selection is not None:
        seed, cutoff = topk_selection
        if skip_nodata:
            eligible = _valid_subsample_mask(band_values[0], skip_nodata=True).reshape(-1)
            selected_indices = selected_indices[eligible]
            band_values = band_values[:, eligible]
        _, selected = _select_topk_keys(selected_indices, seed, cutoff=cutoff)
        assert selected is not None
        selected_indices = selected_indices[selected]
        band_values = band_values[:, selected]

    # Remove cells with nodata in the main band, or replace nodata values with NaN
    elif skip_nodata:
        keep = _valid_subsample_mask(band_values[0], skip_nodata=True).reshape(-1)
        selected_indices = selected_indices[keep]
        band_values = band_values[:, keep]

    if skip_nodata:
        band_values = np.ma.getdata(band_values)
    else:
        band_values = np.ma.filled(band_values.astype(np.result_type(band_values.dtype, np.float32)), np.nan)
        if nodata is not None:
            band_values[band_values == nodata] = np.nan

    # Derive point coordinates from the geotransform and pixel interpretation
    rows, columns = np.unravel_index(selected_indices, raster_shape)
    x_coords, y_coords = _ij2xy(
        i=rows,
        j=columns,
        transform=transform,
        area_or_point=area_or_point,
        shift_area_or_point=False,
        force_offset=force_pixel_offset,
    )

    # Return array rows for Dask or a point dataframe for Dask/multiprocessing
    if as_array:
        return np.column_stack((x_coords, y_coords, *band_values))
    return gpd.GeoDataFrame(
        {name: band_values[index] for index, name in enumerate(column_names)},
        geometry=gpd.points_from_xy(x_coords, y_coords),
        crs=crs,
    )


###########################
# 3C/ EAGER SUBSAMPLING
###########################


def _eager_subsample_raster(
    source_raster: RasterType,
    bands: list[int],
    column_names: list[str],
    data_column_name: str,
    subsample: float | int,
    skip_nodata: bool,
    random_state: int | np.random.Generator | None,
    force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"],
    as_array: bool,
) -> Any:
    """
    Build an eager point output, reading an unloaded partial raster in bounded tiles.

    If the entire raster should be converted (subsample == 1), it is loaded.
    For a smaller requested subsample, values are read in small parts without loading the source raster.
     (mirroring Xarray.isel() default behaviour)
    """

    read_config = None
    if not source_raster.is_loaded and not source_raster._is_xr:
        if subsample == 1:
            source_raster.load()
        else:
            from geoutils.multiproc import MultiprocConfig

            read_config = MultiprocConfig(chunks=512)

    return _subsample_raster_from_indices(
        source_raster,
        bands,
        column_names,
        data_column_name,
        subsample,
        skip_nodata,
        random_state,
        force_pixel_offset,
        as_array,
        read_config,
    )


##########################
# 3D/ DASK SUBSAMPLING
##########################


def _wrapper_subsample_raster_partition_dask(
    band_values: Any,
    tile_idx: NDArrayNum,
    raster_shape: tuple[int, int],
    transform: affine.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    crs: CRS | None,
    nodata: int | float | None,
    column_names: list[str],
    skip_nodata: bool,
    force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"],
    topk_selection: tuple[int, np.uint64] | None,
    as_array: bool,
) -> NDArrayNum | gpd.GeoDataFrame:
    """
    Convert one Dask raster chunk into one lazy part of the point result.

    Only used for a subsample size larger than a raster chunk.
    """

    # Build the global cell indexes covered by this raster chunk
    row_slice = slice(int(tile_idx[0]), int(tile_idx[1]))
    column_slice = slice(int(tile_idx[2]), int(tile_idx[3]))
    rows = np.arange(row_slice.start, row_slice.stop, dtype=np.int64)[:, None]
    columns = np.arange(column_slice.start, column_slice.stop, dtype=np.int64)[None, :]
    selected_indices = (rows * raster_shape[1] + columns).ravel()

    return _raster_values_to_point_partition(
        band_values,
        selected_indices,
        raster_shape,
        transform,
        area_or_point,
        crs,
        nodata,
        column_names,
        skip_nodata,
        force_pixel_offset,
        topk_selection,
        as_array,
    )


def _build_dask_pointcloud_partitions(
    parts: list[Any],
    column_names: list[str],
    column_dtype: DTypeLike,
    crs: CRS | None,
    data_column_name: str,
) -> Any:
    """
    Build one lazy point dataframe from the point rows produced for each Dask chunk.

    Only used for a subsample size larger than a raster chunk.
    """

    from geoutils.pointcloud.dataframe import (
        _build_pointcloud_output,
        _import_dask_dataframe,
    )
    from geoutils.vector.pd_accessor import _import_dask_geopandas

    empty_frame = gpd.GeoDataFrame(
        {name: np.empty(0, dtype=column_dtype) for name in column_names},
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )
    dask_dataframe = _import_dask_dataframe()
    dataframe = (
        dask_dataframe.from_delayed(parts, meta=empty_frame)
        if parts
        else dask_dataframe.from_pandas(empty_frame, npartitions=1)
    )
    dataframe = _import_dask_geopandas().from_dask_dataframe(dataframe, geometry="geometry")
    return _build_pointcloud_output(dataframe, data_column=data_column_name, as_dataframe=True)


def _dask_subsample_raster(
    source_raster: RasterType,
    bands: list[int],
    column_names: list[str],
    data_column_name: str,
    subsample: float | int,
    skip_nodata: bool,
    random_state: int | np.random.Generator | None,
    force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"],
    as_array: bool,
    force_output_to_memory: bool,
) -> Any:
    """
    Build a lazy Dask point output from raster chunks.

    If subsample size is smaller than a raster chunk, we use subsample() directly through
    _subsample_raster_from_indices().

    If subsample size is larger than a raster chunk, we use _dask_array_topk_cutoff() to find the subsample
    cutoff without loading more than a single raster chunk, then build the output with
    _build_dask_pointcloud_partitions().
    """

    dask = import_optional("dask")
    import dask.array as da

    data: Any = source_raster.data
    main_data = data if data.ndim == 2 else data[bands[0] - 1]
    row_chunks, column_chunks = main_data.chunks
    row_starts = np.cumsum((0, *row_chunks))
    column_starts = np.cumsum((0, *column_chunks))
    tiles = np.array(
        [
            (row_starts[row], row_starts[row + 1], column_starts[column], column_starts[column + 1])
            for row in range(len(row_chunks))
            for column in range(len(column_chunks))
        ],
        dtype=np.int64,
    )
    largest_chunk = max(int(rows * columns) for rows in row_chunks for columns in column_chunks)
    main_blocks = main_data.to_delayed().ravel().tolist()

    # Materialize the complete requested sample when explicitly asked, bypassing cutoff selection and partition loops
    if force_output_to_memory:
        output = _subsample_raster_from_indices(
            source_raster,
            bands,
            column_names,
            data_column_name,
            subsample,
            skip_nodata,
            random_state,
            force_pixel_offset,
            as_array,
            read_config=None,
            force_output_to_memory=True,
        )
        output = output.compute() if hasattr(output, "compute") else output
        if as_array:
            return output
        from geoutils.pointcloud import PointCloud

        return PointCloud(output, data_column=data_column_name)

    # Count eligible cells once so the automatic path follows the exact output size
    sample_size = int(np.prod(source_raster.shape))
    if subsample != 1:
        if skip_nodata:
            valid_counts = dask.compute(
                *[_delayed_nb_valids(block, skip_nodata=True, mask_chunk=None) for block in main_blocks]
            )
            total_nb_valids = sum(int(np.asarray(count).sum()) for count in valid_counts)
        else:
            total_nb_valids = sample_size
        sample_size = _get_subsample_size_from_user_input(subsample, total_nb_valids)
        if not _subsample_exceeds_largest_chunk(sample_size, largest_chunk):
            return _subsample_raster_from_indices(
                source_raster,
                bands,
                column_names,
                data_column_name,
                subsample,
                skip_nodata,
                random_state,
                force_pixel_offset,
                as_array,
                read_config=None,
            )

    # Otherwise, find the key cutoff for the subsample, and write partition by partition to a point cloud file
    topk_selection = None
    if subsample != 1:
        block_ids = [
            {
                "row_start": int(tile[0]),
                "row_stop": int(tile[1]),
                "col_start": int(tile[2]),
                "col_stop": int(tile[3]),
            }
            for tile in tiles
        ]
        cutoff_blocks = main_blocks if skip_nodata else [None] * len(main_blocks)
        subsample_meta = SubsampleMeta(
            sample_size=sample_size,
            seed=_resolve_topk_seed(random_state),
            cutoff=None,
        )
        subsample_meta = _dask_array_topk_cutoff(
            blocks=cutoff_blocks,
            mask_blocks=[None] * len(main_blocks),
            block_ids=block_ids,
            array_shape=source_raster.shape,
            largest_chunk=largest_chunk,
            subsample=subsample,
            subsample_meta=subsample_meta,
            skip_nodata=skip_nodata,
        )
        if subsample_meta.sample_size == 0:
            empty = np.empty((0, 2 + len(bands)), dtype=np.result_type(np.float64, source_raster.dtype))
            if as_array:
                return da.from_array(empty, chunks=empty.shape)
            return _build_dask_pointcloud_partitions(
                [],
                column_names,
                source_raster.dtype,
                source_raster.crs,
                data_column_name,
            )
        assert subsample_meta.cutoff is not None
        topk_selection = (subsample_meta.seed, subsample_meta.cutoff)

    # Select input bands in one block per raster chunk
    band_data = data[None, ...] if data.ndim == 2 else data[[band - 1 for band in bands]]
    band_data = band_data.rechunk({0: len(bands)})
    band_blocks = band_data.to_delayed()[0].ravel().tolist()
    parts = [
        dask.delayed(_wrapper_subsample_raster_partition_dask)(
            block,
            tile,
            source_raster.shape,
            source_raster.transform,
            source_raster.area_or_point,
            source_raster.crs,
            source_raster.nodata,
            column_names,
            skip_nodata,
            force_pixel_offset,
            topk_selection,
            as_array,
        )
        for block, tile in zip(band_blocks, tiles)
    ]

    # Assemble rows lazily
    if as_array:
        dtype = np.result_type(np.float64, source_raster.dtype)
        arrays = [da.from_delayed(part, shape=(np.nan, 2 + len(bands)), dtype=dtype) for part in parts]
        return da.concatenate(arrays, axis=0)

    column_dtype = np.result_type(source_raster.dtype, np.float32) if not skip_nodata else source_raster.dtype
    return _build_dask_pointcloud_partitions(
        parts,
        column_names,
        column_dtype,
        source_raster.crs,
        data_column_name,
    )


####################################
# 3E/ MULTIPROCESSING SUBSAMPLING
####################################


def _wrapper_subsample_raster_partition_mp(
    source_raster: RasterType,
    flat_indices: NDArray[np.int64] | tuple[slice, slice],
    bands: list[int],
    column_names: list[str],
    skip_nodata: bool,
    force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"],
    chunks: int | tuple[int, int],
    filename: pathlib.Path,
    topk_selection: tuple[int, np.uint64] | None = None,
    las_output: bool = False,
    las_elevation_column: str | None = None,
) -> tuple[pathlib.Path, np.ndarray[Any, Any] | None]:
    """
    Convert cells from one raster tile in a multiprocessing worker and save its point rows.

    Every multiprocessing point cloud saved to a file uses this function.

    A complete conversion passes a tile slice, and a subsample no larger than the biggest tile passes the indexes of
    its selected raster cells. A larger subsample passes a tile slice with the key separating selected and unselected
    cells. Processing one tile per call limits raster reads and point construction to that tile. Multiprocessing array
    output collects the indexes of all raster cells selected for the requested subsample instead.
    LAS/LAZ output also returns each partition's coordinate bounds so the parent can build one shared header without
    rereading every row.
    """

    # 1/ Build indexes for the complete tile, or keep its sampled cell indexes
    if isinstance(flat_indices, tuple):
        row_slice, column_slice = flat_indices
        rows = np.arange(row_slice.start, row_slice.stop, dtype=np.int64)[:, None]
        columns = np.arange(column_slice.start, column_slice.stop, dtype=np.int64)[None, :]
        selected_indices = (rows * source_raster.shape[1] + columns).ravel()
    else:
        selected_indices = np.asarray(flat_indices, dtype=np.int64)

    from geoutils.multiproc.readers import _read_selected_raster_bands

    selected_main_values = None
    if topk_selection is not None:
        seed, cutoff = topk_selection
        if skip_nodata:
            selected_main_values = _read_selected_raster_bands(source_raster, selected_indices, bands[:1], chunks)
            eligible = _valid_subsample_mask(selected_main_values[0], skip_nodata=True).reshape(-1)
            selected_indices = selected_indices[eligible]
            selected_main_values = selected_main_values[:, eligible]

        _, selected = _select_topk_keys(selected_indices, seed, cutoff=cutoff)
        assert selected is not None
        selected_indices = selected_indices[selected]
        if selected_main_values is not None:
            selected_main_values = selected_main_values[:, selected]
        topk_selection = None

    # 2/ Read every requested band from small raster tiles
    if selected_main_values is None:
        band_values = _read_selected_raster_bands(source_raster, selected_indices, bands, chunks)
    elif len(bands) == 1:
        band_values = selected_main_values
    else:
        auxiliary_values = _read_selected_raster_bands(source_raster, selected_indices, bands[1:], chunks)
        band_values = np.ma.concatenate((selected_main_values, auxiliary_values), axis=0)

    # 3/ Convert the selected raster values and positions to one point partition
    dataframe = cast(
        gpd.GeoDataFrame,
        _raster_values_to_point_partition(
            band_values,
            selected_indices,
            source_raster.shape,
            source_raster.transform,
            source_raster.area_or_point,
            source_raster.crs,
            source_raster.nodata,
            column_names,
            skip_nodata,
            force_pixel_offset,
            topk_selection,
            as_array=False,
        ),
    )

    # 4/ Save the point partition to a temporary file
    from geoutils.pointcloud.writing import _stage_pointcloud_partition

    bounds = None
    if las_output:
        from geoutils.pointcloud.las import _las_coordinate_bounds

        bounds = _las_coordinate_bounds(dataframe, las_elevation_column)
    return _stage_pointcloud_partition(dataframe, filename), bounds


def _multiproc_subsample_raster(
    source_raster: RasterType,
    bands: list[int],
    column_names: list[str],
    data_column_name: str,
    subsample: float | int,
    skip_nodata: bool,
    random_state: int | np.random.Generator | None,
    force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"],
    mp_config: MultiprocConfig,
    as_array: bool,
    force_output_to_memory: bool,
) -> Any:
    """
    Build a Multiproc point output from raster chunks.

    If subsample size is smaller than a raster chunk or as_array=True, we use subsample() directly through
    _subsample_raster_from_indices().

    If subsample size is larger than a raster chunk, we use _multiproc_array_topk_cutoff() to find the subsample
    indices without loading more than a single raster chunk, then build the output chunk by chunk with
    _write_pointcloud_partitions().
    """

    if as_array or force_output_to_memory:
        # Use loaded values directly instead of sending the complete in-memory raster to worker tasks
        read_config = None if source_raster.is_loaded else mp_config
        return _subsample_raster_from_indices(
            source_raster,
            bands,
            column_names,
            data_column_name,
            subsample,
            skip_nodata,
            random_state,
            force_pixel_offset,
            as_array=as_array,
            read_config=read_config,
            force_output_to_memory=True,
        )

    from geoutils.multiproc.cluster import _map_bounded
    from geoutils.pointcloud.writing import (
        _resolve_pointcloud_output,
        _stage_pointcloud_partition,
        _write_pointcloud_partitions,
    )

    # Preserve point geometry and named band columns through GeoPackage
    output_filename, driver = _resolve_pointcloud_output(
        mp_config.outfile,
        mp_config.driver,
        supported_drivers=("GPKG", "LAS", "LAZ"),
        operation_name="raster subsampling",
    )
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".geoutils-raster-points-", dir=output_filename.parent) as directory:
        temporary_directory = pathlib.Path(directory)

        # Give workers an unloaded file even when the caller supplied an in-memory raster or Xarray accessor
        worker_source = source_raster
        if source_raster.is_loaded or source_raster.name is None:
            from geoutils.raster import Raster

            source_filename = temporary_directory / "source.tif"
            source_raster.to_file(source_filename)
            worker_source = Raster(source_filename, load_data=False)

        from geoutils.multiproc import compute_tiling

        tiling = compute_tiling(mp_config.chunks, worker_source.shape, overlap=0)
        tiles = tiling.reshape((-1, 4))
        largest_chunk = max(int((tile[1] - tile[0]) * (tile[3] - tile[2])) for tile in tiles)

        # Count eligible cells once so the automatic path follows the exact output size
        sample_size = int(np.prod(worker_source.shape))
        if subsample != 1:
            if skip_nodata:
                count_arguments = ((worker_source, tile, bands[0], True, None) for tile in tiles)
                total_nb_valids = sum(
                    count
                    for _, count in _map_bounded(
                        mp_config.cluster,
                        _wrapper_multiproc_nb_valids_per_block_positional,
                        count_arguments,
                    )
                )
            else:
                total_nb_valids = sample_size
            sample_size = _get_subsample_size_from_user_input(subsample, total_nb_valids)
        sample_exceeds_largest_chunk = _subsample_exceeds_largest_chunk(sample_size, largest_chunk)

        # Materialize a bounded sample, save it once, and return an unloaded wrapper for the requested output
        if not sample_exceeds_largest_chunk:
            pointcloud = _subsample_raster_from_indices(
                worker_source,
                bands,
                column_names,
                data_column_name,
                subsample,
                skip_nodata,
                random_state,
                force_pixel_offset,
                as_array=False,
                read_config=mp_config,
            )
            partition_filename = _stage_pointcloud_partition(pointcloud.ds, temporary_directory / "partition.pkl")
            partition_filenames: Iterable[pathlib.Path] = (partition_filename,)
            if driver == "GPKG":
                las_bounds = None
            else:
                from geoutils.pointcloud.las import _las_coordinate_bounds

                las_bounds = [_las_coordinate_bounds(pointcloud.ds, data_column_name)]

        # Split complete conversions by raster tile
        elif subsample == 1:
            selected_parts: Iterable[
                tuple[int, NDArray[np.int64] | tuple[slice, slice], tuple[int, np.uint64] | None]
            ] = (
                (
                    tile_id,
                    (slice(int(tile[0]), int(tile[1])), slice(int(tile[2]), int(tile[3]))),
                    None,
                )
                for tile_id, tile in enumerate(tiles)
            )

        # Find cutoff key out-of-memory when subsample size exceeds one raster chunk size
        else:
            subsample_meta = SubsampleMeta(
                sample_size=sample_size,
                seed=_resolve_topk_seed(random_state),
                cutoff=None,
            )
            subsample_meta = _multiproc_array_topk_cutoff(
                source_raster=worker_source,
                tiles=tiles,
                largest_chunk=largest_chunk,
                subsample=subsample,
                subsample_meta=subsample_meta,
                band=bands[0],
                skip_nodata=skip_nodata,
                mp_config=mp_config,
            )
            if subsample_meta.sample_size == 0:
                selected_parts = ((0, np.empty(0, dtype=np.int64), None),)
            else:
                assert subsample_meta.cutoff is not None
                selected_parts = (
                    (
                        tile_id,
                        (slice(int(tile[0]), int(tile[1])), slice(int(tile[2]), int(tile[3]))),
                        (subsample_meta.seed, subsample_meta.cutoff),
                    )
                    for tile_id, tile in enumerate(tiles)
                )

        if sample_exceeds_largest_chunk:
            arguments = (
                (
                    worker_source,
                    selected,
                    bands,
                    column_names,
                    skip_nodata,
                    force_pixel_offset,
                    mp_config.chunks,
                    temporary_directory / f"partition_{tile_id}.pkl",
                    topk_selection,
                    driver != "GPKG",
                    data_column_name,
                )
                for tile_id, selected, topk_selection in selected_parts
            )
            partition_results = [
                result
                for _, result in _map_bounded(mp_config.cluster, _wrapper_subsample_raster_partition_mp, arguments)
            ]
            partition_filenames = [filename for filename, _ in partition_results]
            las_bounds = [bounds for _, bounds in partition_results] if driver != "GPKG" else None

        # Finally, we assemble the final file and return it as an unloaded PointCloud!
        return _write_pointcloud_partitions(
            output_filename,
            partition_filenames,
            driver=driver,
            data_column=data_column_name if driver == "GPKG" else None,
            geometry_type="Point",
            las_elevation_column=data_column_name,
            las_bounds=las_bounds,
        )


##############################
# 3F/ SUBSAMPLING PARENT
##############################


def _subsample_raster(
    source_raster: RasterType,
    subsample: float | int = 1,
    data_column_name: str = "b1",
    data_band: int = 1,
    auxiliary_data_bands: Iterable[int] | None = None,
    auxiliary_column_names: Iterable[str] | None = None,
    skip_nodata: bool = True,
    as_array: bool = False,
    random_state: int | np.random.Generator | None = None,
    force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"] = "ul",
    mp_config: MultiprocConfig | None = None,
    force_output_to_memory: bool = False,
    rename_default_data_column: bool = True,
) -> Any:
    """
    Subsample raster cells into point output with eager, Dask, or multiprocessing execution.

    See RasterBase.subsample() and to_pointcloud() for details on the arguments.

    Internally, this function checks user inputs, then passes on to:
    - _eager_subsample_raster() for an in-memory input,
    - _dask_subsample_raster() for a Dask input, reading chunk by chunk and returning a lazy Dask array or
        GeoDataFrame, optionally written chunk-by-chunk as well,
    - _multiproc_subsample_raster() for a multiprocessing input, reading the raster chunk by chunk and writing to a
        point cloud file, optionally chunk-by-chunk too.

    Without subsampling, the array is reshaped chunk-by-chunk to a lazy Dask object or point cloud file.

    With subsampling, the deterministic "topk" method selects the same raster cells across execution backends.
    Then, depending on subsampling size, two scenarios are triggered:
    - For a subsample size smaller than one raster chunk, the input reading happens chunk-by-chunk, but the output
        subsample is computed at once in memory, eagerly for multiprocessing or lazily for Dask.
    - For a subsample size larger than one raster chunk, both chunked implementations call an iterative algorithm to
        find the k cutoff of the "topk" algorithm without loading the equivalent of the subsample size in memory. The
        output points are then written chunk-by-chunk to a file or lazy Dask object.

    This ensures that no more than a multiple of the raster input chunk size is loaded or returned at once.
    """

    # 1/ Input checks

    # Main data column checks
    if not isinstance(data_column_name, str):
        raise ValueError("Data column name must be a string.")
    if not (isinstance(data_band, int) and data_band >= 1 and data_band <= source_raster.count):
        raise ValueError(
            f"Data band number must be an integer between 1 and the total number of bands ({source_raster.count})."
        )

    # Rename data column if a different band is selected but the name is still default
    if rename_default_data_column and data_band != 1 and data_column_name == "b1":
        data_column_name = "b" + str(data_band)

    # Auxiliary data columns checks
    if auxiliary_column_names is not None and auxiliary_data_bands is None:
        raise ValueError("Passing auxiliary column names requires passing auxiliary data band numbers as well.")
    if auxiliary_data_bands is None:
        auxiliary_data_bands = [band for band in range(1, source_raster.count + 1) if band != data_band]

    if not isinstance(auxiliary_data_bands, Iterable):
        raise ValueError("Auxiliary data band number must be an iterable containing only integers.")
    auxiliary_data_bands = list(auxiliary_data_bands)
    if not all(isinstance(b, int) for b in auxiliary_data_bands):
        raise ValueError("Auxiliary data band number must be an iterable containing only integers.")
    if any((1 > b or source_raster.count < b) for b in auxiliary_data_bands):
        raise ValueError(
            f"Auxiliary data band numbers must be between 1 and the total number of bands ({source_raster.count})."
        )
    if data_band in auxiliary_data_bands:
        raise ValueError(
            f"Main data band {data_band} should not be listed in auxiliary data bands {auxiliary_data_bands}."
        )

    # Define and validate one name for every auxiliary band
    if auxiliary_column_names is not None:
        if not isinstance(auxiliary_column_names, Iterable) or isinstance(auxiliary_column_names, (str, bytes)):
            raise ValueError("Auxiliary column names must be an iterable containing only strings.")
        auxiliary_column_names = list(auxiliary_column_names)
        if not all(isinstance(b, str) for b in auxiliary_column_names):
            raise ValueError("Auxiliary column names must be an iterable containing only strings.")
        if not len(auxiliary_column_names) == len(auxiliary_data_bands):
            raise ValueError(
                f"Length of auxiliary column name and data band numbers should be the same, "
                f"found {len(auxiliary_column_names)} and {len(auxiliary_data_bands)} respectively."
            )
    else:
        auxiliary_column_names = [f"b{i}" for i in auxiliary_data_bands]

    # Build the ordered band and column lists used by every execution backend
    all_bands = [data_band] + auxiliary_data_bands
    all_column_names = [data_column_name] + auxiliary_column_names

    if len(set(all_column_names)) != len(all_column_names) or "geometry" in all_column_names:
        raise ValueError("Point cloud data column names must be unique and cannot be 'geometry'.")

    # 2/ Validate execution backend and point coordinate convention

    # One operation cannot be scheduled by Dask and multiprocessing at the same time
    dask_backend = source_raster._chunks is not None
    if dask_backend and mp_config is not None:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove ``mp_config`` from "
            "subsample(). To use Multiprocessing, open the raster without ``chunks``."
        )
    if source_raster._is_xr and mp_config is not None:
        raise ValueError("Argument ``mp_config`` requires a Raster input rather than an Xarray accessor.")

    # Validate the coordinate convention before launching lazy or multiprocessing work
    if force_pixel_offset not in ("center", "ul", "ur", "ll", "lr"):
        raise ValueError(f"Unknown pixel offset {force_pixel_offset!r}.")

    # Use one seed so every tile participates in the same random subsample
    if random_state is None and subsample != 1:
        random_state = int(np.random.default_rng().integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))

    # 3/ Call the relevant backend depending on input type (eager, Dask, Multiproc)
    if dask_backend:
        return _dask_subsample_raster(
            source_raster=source_raster,
            bands=all_bands,
            column_names=all_column_names,
            data_column_name=data_column_name,
            subsample=subsample,
            skip_nodata=skip_nodata,
            random_state=random_state,
            force_pixel_offset=force_pixel_offset,
            as_array=as_array,
            force_output_to_memory=force_output_to_memory,
        )

    if mp_config is not None:
        return _multiproc_subsample_raster(
            source_raster=source_raster,
            bands=all_bands,
            column_names=all_column_names,
            data_column_name=data_column_name,
            subsample=subsample,
            skip_nodata=skip_nodata,
            random_state=random_state,
            force_pixel_offset=force_pixel_offset,
            mp_config=mp_config,
            as_array=as_array,
            force_output_to_memory=force_output_to_memory,
        )

    return _eager_subsample_raster(
        source_raster=source_raster,
        bands=all_bands,
        column_names=all_column_names,
        data_column_name=data_column_name,
        subsample=subsample,
        skip_nodata=skip_nodata,
        random_state=random_state,
        force_pixel_offset=force_pixel_offset,
        as_array=as_array,
    )
