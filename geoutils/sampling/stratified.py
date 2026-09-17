# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Subsample positions independently within encoded integer strata."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from geoutils._dispatch import is_dask_array
from geoutils._misc import import_optional
from geoutils._typing import NDArrayNum
from geoutils.multiproc.chunked import iter_chunk_slices
from geoutils.multiproc.cluster import _map_bounded
from geoutils.sampling.subsampling import _splitmix64

if TYPE_CHECKING:
    from geoutils.multiproc.mparray import MultiprocConfig


def _prune_stratified_candidates(
    candidates: tuple[NDArrayNum, NDArrayNum, NDArrayNum], quotas: dict[int, int] | int | float
) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum]:
    """
    Keep each stratum's smallest random keys from one block or a batch of block candidates.

    Sort group IDs once so each selection inspects only its group's contiguous slice. Quotas follow
    _sample_strata_block(): fractions are evaluated here only when the candidates contain the whole input;
    chunked fractions require global counts computed before block selection.

    :param candidates: Three matching one-dimensional arrays of group IDs, random keys and original flat positions.
        Their order need not be sorted until the final sample is returned.
    :returns: The same three arrays restricted to each group's smallest keys, keeping their input dtypes.
    """

    # Place members of the same group together without scanning the full input separately for every group
    labels, keys, positions = candidates
    if labels.size == 0:
        return candidates
    order = np.argsort(labels, kind="stable")
    labels, keys, positions = labels[order], keys[order], positions[order]
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    groups = labels[starts]
    counts = np.diff(np.r_[starts, labels.size])

    # Keep whole small groups and partially select only slices that exceed their global quota
    chosen = []
    for group, start, count in zip(groups, starts, counts):
        # A shared fixed cap needs no separate count pass; fractions can also be resolved here for eager input
        if isinstance(quotas, dict):
            quota = min(quotas[int(group)], int(count))
        else:
            quota = int(quotas * count) if quotas <= 1 else min(int(quotas), int(count))
        if quota == 0:
            continue
        if quota == count:
            selected = np.arange(start, start + count)
        else:
            selected = start + np.argpartition(keys[start : start + count], quota - 1)[:quota]
        chosen.append(selected)

    # Preserve the array dtypes even when rounding leaves every group with an empty sample
    selection = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
    return labels[selection], keys[selection], positions[selection]


def _sample_strata_block(
    group_ids: NDArrayNum,
    origin: tuple[int, ...],
    full_shape: tuple[int, ...],
    quotas: dict[int, int] | int | float,
    seed: int | None,
    ranks: dict[int, NDArrayNum] | None,
) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum]:
    """
    Sample a block's strata while keeping positions relative to the full input.

    Topk keeps each group's best global-key candidates for later merging. Sequential sampling receives already
    drawn member ranks relative to this block, so only those members need to leave the worker.

    :param group_ids: One NumPy block of the encoded group array passed to _stratified_subsample_indices().
    :param origin: Starting position of this block along each axis of the full input.
    :param full_shape: Dimensions of the full input, used to convert block positions to original flat indices.
    :param quotas: Mapping from group IDs to target sample counts across the full input, or one common cap above
        one. A fraction at most one is accepted only when this block contains the whole input.
    :param seed: Integer mixed with original flat indices to rank topk candidates. None selects by ranks instead.
    :param ranks: For sequential sampling, a mapping from each group ID to zero-based ranks among that group's
        members in this block's flat order. These are not indices into the block or the full input. None for topk.
    :returns: Matching one-dimensional arrays of group IDs, random keys and flat indices into the full input.
        Sequential results use zero keys because their members have already been selected.
    """

    # Ignore negative group IDs, which represent missing groupers and locations excluded by a user mask
    flat = np.flatnonzero(group_ids.ravel() >= 0)
    labels = group_ids.ravel()[flat]
    if group_ids.shape == full_shape:
        positions = flat
    else:
        # Convert tile-local positions only when the block covers part of the original input
        coordinates = np.unravel_index(flat, group_ids.shape)
        absolute = tuple(coordinate + start for coordinate, start in zip(coordinates, origin))
        positions = np.ravel_multi_index(absolute, full_shape)

    # Derive topk keys from original positions, never from a group's label or the block's local order
    if seed is not None:
        keys = _splitmix64(np.uint64(seed) ^ positions.astype(np.uint64))
        return _prune_stratified_candidates((labels, keys, positions), quotas)

    # Group sequential members once and translate their local ranks into original flat positions
    assert ranks is not None
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    starts = np.r_[0, np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1] if labels.size else []
    groups = sorted_labels[starts]
    selected = [order[start + ranks[int(group)]] for group, start in zip(groups, starts)]
    selection = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
    return labels[selection], np.zeros(selection.size, dtype=np.uint64), positions[selection]


def _merge_stratified_candidates(
    blocks: list[tuple[NDArrayNum, NDArrayNum, NDArrayNum]], quotas: dict[int, int] | int | float
) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum]:
    """
    Combine a small batch of block candidates and prune each stratum back to its global quota.

    :param blocks: Candidate tuples returned by _sample_strata_block() or an earlier merge.
    :param quotas: Target sample counts per group across the full input, or a common cap above one, as in
        _sample_strata_block(). Fractions must already have been converted to full-group target counts.
    :returns: One candidate tuple containing each group's smallest keys across the supplied blocks.
    """

    labels = np.concatenate([block[0] for block in blocks])
    keys = np.concatenate([block[1] for block in blocks])
    positions = np.concatenate([block[2] for block in blocks])
    return _prune_stratified_candidates((labels, keys, positions), quotas)


def _stratified_subsample_indices(
    group_ids: Any,
    subsample: int | float,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "topk",
    mp_config: MultiprocConfig | None = None,
) -> NDArrayNum:
    """
    Choose original flat positions independently within each nonnegative integer group ID.

    Fractions select floor(fraction * group size); amounts above one cap the count per group without warnings for
    smaller groups. One keeps all eligible locations. Values are not inspected, so callers can reuse the same
    positions across several value columns with different missing observations.

    Count memberships first when chunked fractions or sequential ranks need group sizes, then let
    _sample_strata_block() select from each block. Fixed-cap topk needs no separate count pass.
    For topk, _merge_stratified_candidates() prunes bounded batches of candidates using keys from _splitmix64().
    This gives identical samples for NumPy, Dask and multiprocessing regardless of chunk boundaries. Sequential
    sampling draws member ranks in group order, then maps them to blocks; its sample can depend on block layout.

    Dask builds an eight-way candidate merge tree. Multiprocessing handles up to eight tiles at a time and merges
    their candidates in the caller. Both return computed NumPy indices; neither gathers the full group-ID array.
    Count metadata and selected positions must fit in memory. NumPy inputs are already resident, including when
    their tiles are sent to worker processes.

    :param group_ids: NumPy or Dask integer array with one encoded group per location. Nonnegative IDs identify
        groups; negative IDs exclude locations. Grouping arrays and any user mask must already be combined here.
    :param subsample: Positive finite fraction at most one, or maximum number of sampled locations per group above
        one. Fractions round down separately in each group; one keeps all eligible locations.
    :param random_state: Integer seed or NumPy Generator. Topk draws one seed from a supplied generator and uses zero
        when omitted; sequential sampling uses the generator directly. A subsample of one consumes no draws.
    :param strategy: Topk selects the smallest deterministic keys; sequential draws random member ranks per group.
    :param mp_config: Tile sizes and worker cluster for a resident NumPy input. Omit for eager or Dask execution.
    :returns: Computed one-dimensional NumPy array of flat indices into the original input, ordered by random key
        for topk or by original position for sequential sampling and for a subsample of one.
    """

    # 1/ Check user input before starting any sampling tasks
    if not np.issubdtype(group_ids.dtype, np.integer):
        raise TypeError("Group IDs must be integers.")
    if not np.isfinite(subsample) or subsample <= 0:
        raise ValueError("Argument ``subsample`` must be a positive finite number.")
    if strategy not in ("sequential", "topk"):
        raise ValueError("Argument ``subsampling_strategy`` must be 'sequential' or 'topk'.")
    use_dask = is_dask_array(group_ids)
    if use_dask and mp_config is not None:
        raise ValueError("Cannot use Multiprocessing and Dask simultaneously.")
    if group_ids.size == 0:
        return np.empty(0, dtype=np.int64)

    # Draw one seed for the whole topk call; sequential sampling uses the generator directly below
    seed = None
    if strategy == "topk" and subsample != 1:
        if isinstance(random_state, np.random.Generator):
            seed = int(random_state.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        else:
            seed = 0 if random_state is None else int(random_state)

    # 2/ Divide the group IDs into blocks and record their original coordinate offsets
    shape = tuple(int(length) for length in group_ids.shape)
    if use_dask:
        import_optional("dask")
        import dask

        blocks = list(group_ids.to_delayed().ravel())
        starts = [np.r_[0, np.cumsum(lengths[:-1])] for lengths in group_ids.chunks]
        origins = list(product(*starts))
    else:
        # Reuse views of resident NumPy tiles; without multiprocessing, the input is one block
        chunks = shape if mp_config is None else mp_config.chunks
        tiles = list(iter_chunk_slices(shape, chunks))
        origins = [tuple(part.start for part in tile) for tile in tiles]
        blocks = [group_ids[tile] for tile in tiles]

    # 3/ Count full group memberships only when fractions or sequential ranks require them in advance
    quotas: dict[int, int] | int | float = subsample
    if seed is None or (subsample < 1 and (use_dask or mp_config is not None)):
        if use_dask:
            memberships = list(dask.compute(*[dask.delayed(np.unique)(block, return_counts=True) for block in blocks]))
        else:
            memberships = [np.unique(block, return_counts=True) for block in blocks]

        # Round each group's fraction once, after counts from all its blocks have been combined
        totals: dict[int, int] = {}
        for groups, counts in memberships:
            for group, count in zip(groups, counts):
                if group >= 0:
                    totals[int(group)] = totals.get(int(group), 0) + int(count)
        quotas = {
            group: int(subsample * count) if subsample <= 1 else min(int(subsample), count)
            for group, count in totals.items()
        }
        if not any(quotas.values()):
            return np.empty(0, dtype=np.int64)

    # Draw sorted member ranks once for each sequential group, following group and then block order
    ranks_per_block: list[dict[int, NDArrayNum] | None] = [None] * len(blocks)
    if seed is None:
        assert isinstance(quotas, dict)
        rng = np.random.default_rng(random_state)
        ranks = {}
        for group in sorted(totals):
            # Keep complete groups without consuming random numbers or constructing a discarded permutation
            if quotas[group] == totals[group]:
                ranks[group] = np.arange(totals[group])
            else:
                ranks[group] = np.sort(rng.choice(totals[group], quotas[group], replace=False))

        # Map each group's global member ranks onto its successive blocks without rescanning the input
        offsets = dict.fromkeys(totals, 0)
        for index, (groups, counts) in enumerate(memberships):
            block_ranks: dict[int, NDArrayNum] = {}
            for group, count in zip(groups, counts):
                if group < 0:
                    continue
                start = offsets[int(group)]
                low, high = np.searchsorted(ranks[int(group)], [start, start + count])
                block_ranks[int(group)] = ranks[int(group)][low:high] - start
                offsets[int(group)] += int(count)
            ranks_per_block[index] = block_ranks

    # 4/ Select within blocks, then combine only their sampled positions or bounded topk candidates
    if use_dask:
        candidates = [
            dask.delayed(_sample_strata_block)(block, origin, shape, quotas, seed, ranks)
            for block, origin, ranks in zip(blocks, origins, ranks_per_block)
        ]
        if seed is not None:
            while len(candidates) > 1:
                candidates = [
                    dask.delayed(_merge_stratified_candidates)(candidates[start : start + 8], quotas)
                    for start in range(0, len(candidates), 8)
                ]
        results = list(dask.compute(*candidates))
    elif mp_config is not None:
        # Bound pending tile payloads and intermediate candidates, including for process-based clusters
        arguments = zip(
            blocks, origins, [shape] * len(blocks), [quotas] * len(blocks), [seed] * len(blocks), ranks_per_block
        )
        results = []
        for index, result in _map_bounded(mp_config.cluster, _sample_strata_block, arguments):
            results.append(result)
            if seed is not None and ((index + 1) % 8 == 0 or index + 1 == len(blocks)):
                results = [_merge_stratified_candidates(results, quotas)]
    else:
        results = [_sample_strata_block(blocks[0], origins[0], shape, quotas, seed, ranks_per_block[0])]

    # Return a stable ordering by random key for topk and by original position for sequential or unsampled input
    positions = np.concatenate([result[2] for result in results])
    if seed is None:
        return np.sort(positions)
    keys = np.concatenate([result[1] for result in results])
    return positions[np.lexsort((positions, keys))]
