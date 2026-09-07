# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Sample pairs of raster cells or point cloud rows."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from geoutils._dispatch import (
    get_geo_attr,
    is_dask_array,
    is_dask_dataframe,
)
from geoutils._typing import NDArrayNum
from geoutils.interface.raster_point import _mask_on_raster
from geoutils.raster.array import _selected_raster_data
from geoutils.vector.base import _as_vector

#############################
# 1/ SHARED PAIR OPERATIONS
#############################


def _take_raster_values(array: Any, indices: NDArrayNum) -> NDArrayNum:
    """Read selected raster cells with one array indexing call."""

    # Convert flat pair indexes back to their raster row and column positions
    rows, columns = np.divmod(np.asarray(indices, dtype=np.int64), int(array.shape[1]))

    # Load only the requested cells so Dask reads grow with the sample size
    values = array.vindex[rows, columns].compute() if is_dask_array(array) else array[rows, columns]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values)


def _deduplicate_pairs(first: NDArrayNum, second: NDArrayNum, *, n_observations: int) -> tuple[NDArrayNum, NDArrayNum]:
    """Remove repeated pairs while treating A-B and B-A as the same pair."""

    # Put the smaller row number first so A-B and B-A have the same key
    low = np.minimum(first, second).astype(np.int64, copy=False)
    high = np.maximum(first, second).astype(np.int64, copy=False)
    keys = low * np.int64(n_observations) + high

    # Restore input order after selecting the first occurrence of each key
    _, positions = np.unique(keys, return_index=True)
    positions.sort()
    return low[positions], high[positions]


############################
# 2/ REGULAR RASTER SAMPLING
############################


class _RegularPairSampler:
    """Draw pairs of available raster cells while limiting temporary array sizes."""

    #################
    # CONFIGURATION
    #################

    def __init__(
        self,
        array: Any,
        *,
        dx: float,
        dy: float,
        n_pairs: int,
        min_distance: float,
        max_distance: float,
        strategy: Literal["independent", "anchors", "chunk_anchors", "anchor_batched"],
        deduplicate: Literal["none", "per_anchor", "global"],
        random_state: int | np.random.Generator | None,
        batch_pairs: int,
        max_rounds: int,
        max_oversample: float,
        chunks_per_round: int,
        anchors_per_round: int,
        distances_per_anchor: int,
        angles_per_distance: int,
        hybrid_local_fraction: float,
        max_local_distance: float | None,
        index_dtype: Any,
        distance_dtype: Any,
    ) -> None:
        # Store the grid and sampling options with consistent numeric types
        self.array = array
        self.shape = (int(array.shape[0]), int(array.shape[1]))
        self.size = int(np.prod(self.shape))
        self.dx, self.dy = float(abs(dx)), float(abs(dy))
        self.n_pairs = int(n_pairs)
        self.min_distance, self.max_distance = float(min_distance), float(max_distance)
        self.strategy, self.deduplicate = strategy, deduplicate
        self.rng = (
            random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        )
        self.batch_pairs, self.max_rounds = int(batch_pairs), int(max_rounds)
        self.max_oversample = float(max_oversample)
        self.chunks_per_round, self.anchors_per_round = int(chunks_per_round), int(anchors_per_round)
        self.distances_per_anchor, self.angles_per_distance = int(distances_per_anchor), int(angles_per_distance)
        self.hybrid_local_fraction = float(hybrid_local_fraction)
        self.index_dtype, self.distance_dtype = np.dtype(index_dtype), np.dtype(distance_dtype)

        # Check sampling options before creating any temporary arrays
        if self.n_pairs < 1:
            raise ValueError("n_pairs must be a positive integer.")
        if not 0 < self.min_distance < self.max_distance:
            raise ValueError("Require 0 < min_distance < max_distance.")
        if strategy not in {"independent", "anchors", "chunk_anchors", "anchor_batched"}:
            raise ValueError("Unknown regular grid pair sampling strategy.")
        if deduplicate not in {"none", "per_anchor", "global"}:
            raise ValueError("deduplicate must be 'none', 'per_anchor' or 'global'.")
        if not 0 <= self.hybrid_local_fraction <= 1:
            raise ValueError("hybrid_local_fraction must be between 0 and 1.")

        # Use Dask chunks as local areas, or split an in-memory raster into similarly sized areas
        if is_dask_array(array):
            self.chunk_shape = (int(array.chunks[0][0]), int(array.chunks[1][0]))
        else:
            self.chunk_shape = (min(2048, self.shape[0]), min(2048, self.shape[1]))
        chunk_rows, chunk_columns = self.chunk_shape
        self.max_local_distance = (
            float(np.hypot((chunk_columns - 1) * self.dx, (chunk_rows - 1) * self.dy))
            if max_local_distance is None
            else float(max_local_distance)
        )

    ###################
    # POSSIBLE PAIRS
    ###################

    def _offsets(self, count: int, maximum: float) -> tuple[NDArrayNum, NDArrayNum]:
        """Draw random directions with distance ranges represented evenly on a log scale."""

        # Limit local distances to the configured nearby area
        upper = min(self.max_distance, maximum)
        if upper <= self.min_distance:
            upper = self.max_distance

        # Draw distances and directions, then round them to row and column offsets
        radius = np.exp(self.rng.uniform(np.log(self.min_distance), np.log(upper), count))
        angle = self.rng.uniform(0, 2 * np.pi, count)
        column_offset = np.rint(radius * np.cos(angle) / self.dx).astype(np.int64)
        row_offset = np.rint(radius * np.sin(angle) / self.dy).astype(np.int64)

        # Remove rounded offsets whose exact grid distance falls outside the requested range
        exact_distance = np.hypot(column_offset * self.dx, row_offset * self.dy)
        in_range = (
            ((row_offset != 0) | (column_offset != 0))
            & (exact_distance >= self.min_distance)
            & (exact_distance <= upper)
        )
        return row_offset[in_range], column_offset[in_range]

    def _sample_anchors(self, count: int, *, chunk_aligned: bool) -> NDArrayNum:
        """Draw first endpoints across the grid or from a small set of chunks."""

        # Draw directly from the full grid when pairs do not need to stay near selected chunks
        if not chunk_aligned:
            return self.rng.integers(0, self.size, count, dtype=np.int64)

        # Select only a few source chunks before drawing cell numbers
        n_chunk_rows = int(np.ceil(self.shape[0] / self.chunk_shape[0]))
        n_chunk_columns = int(np.ceil(self.shape[1] / self.chunk_shape[1]))
        chunk_count = min(self.chunks_per_round, n_chunk_rows * n_chunk_columns)
        chosen = self.rng.choice(n_chunk_rows * n_chunk_columns, chunk_count, replace=False)

        # Split first endpoints between those chunks to limit Dask reads
        anchors: list[NDArrayNum] = []
        per_chunk = int(np.ceil(count / chunk_count))
        for flat_chunk in chosen:
            # Use each edge chunk's true size so every drawn cell exists
            chunk_row, chunk_column = divmod(int(flat_chunk), n_chunk_columns)
            row_start, column_start = chunk_row * self.chunk_shape[0], chunk_column * self.chunk_shape[1]
            row_stop = min(row_start + self.chunk_shape[0], self.shape[0])
            column_stop = min(column_start + self.chunk_shape[1], self.shape[1])
            take = min(per_chunk, count - sum(part.size for part in anchors))
            rows = self.rng.integers(row_start, row_stop, take, dtype=np.int64)
            columns = self.rng.integers(column_start, column_stop, take, dtype=np.int64)
            anchors.append(rows * self.shape[1] + columns)
            if sum(part.size for part in anchors) == count:
                break

        # Join the first endpoints after all selected chunks have contributed
        return np.concatenate(anchors) if anchors else np.empty(0, dtype=np.int64)

    def _from_anchors(self, anchors: NDArrayNum, count: int, *, local: bool) -> tuple[NDArrayNum, NDArrayNum]:
        """Reuse first endpoints and draw a separate offset for each pair."""

        # Return empty integer arrays before NumPy can try to repeat an empty input
        if anchors.size == 0 or count == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        # Repeat first endpoints and draw an offset for every requested pair
        repeated = np.resize(anchors, count)
        rows, columns = np.divmod(repeated, self.shape[1])
        row_offset, column_offset = self._offsets(count, self.max_local_distance if local else self.max_distance)
        length = min(count, row_offset.size)
        repeated, rows, columns = repeated[:length], rows[:length], columns[:length]
        target_rows, target_columns = rows + row_offset[:length], columns + column_offset[:length]

        # Remove offsets that leave the raster before converting endpoints to flat cell numbers
        inside = (
            (target_rows >= 0)
            & (target_rows < self.shape[0])
            & (target_columns >= 0)
            & (target_columns < self.shape[1])
        )
        if local:
            # Keep nearby pairs in the first endpoint's chunk to limit Dask reads
            inside &= (rows // self.chunk_shape[0] == target_rows // self.chunk_shape[0]) & (
                columns // self.chunk_shape[1] == target_columns // self.chunk_shape[1]
            )
        first = repeated[inside]
        second = target_rows[inside] * self.shape[1] + target_columns[inside]

        # Optionally keep each second endpoint only once for a given first endpoint
        if self.deduplicate == "per_anchor" and first.size:
            order = np.argsort(first, kind="stable")
            keys = first[order].astype(np.int64) * np.int64(self.size) + second[order]
            _, keep = np.unique(keys, return_index=True)
            positions = order[np.sort(keep)]
            first, second = first[positions], second[positions]
        return first, second

    def _anchor_batched(self, anchors: NDArrayNum, *, local: bool) -> tuple[NDArrayNum, NDArrayNum]:
        """Draw several distances and directions from every first endpoint."""

        # Return empty integer arrays when no first endpoint was supplied
        if anchors.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        # Draw several directions for each distance so one first endpoint yields many pairs
        upper = min(self.max_distance, self.max_local_distance) if local else self.max_distance
        radii = np.exp(
            self.rng.uniform(np.log(self.min_distance), np.log(upper), (anchors.size, self.distances_per_anchor))
        )
        angles = self.rng.uniform(
            0,
            2 * np.pi,
            (anchors.size, self.distances_per_anchor, self.angles_per_distance),
        )
        column_offset = np.rint(radii[:, :, None] * np.cos(angles) / self.dx).astype(np.int64).ravel()
        row_offset = np.rint(radii[:, :, None] * np.sin(angles) / self.dy).astype(np.int64).ravel()
        repeated = np.repeat(anchors, self.distances_per_anchor * self.angles_per_distance)

        # Remove zero and out-of-range offsets after rounding them to grid cells
        exact_distance = np.hypot(column_offset * self.dx, row_offset * self.dy)
        in_range = (
            ((row_offset != 0) | (column_offset != 0))
            & (exact_distance >= self.min_distance)
            & (exact_distance <= upper)
        )
        repeated, row_offset, column_offset = repeated[in_range], row_offset[in_range], column_offset[in_range]
        rows, columns = np.divmod(repeated, self.shape[1])
        target_rows, target_columns = rows + row_offset, columns + column_offset

        # Exclude second endpoints outside the raster or, for nearby pairs, outside the selected chunk
        inside = (
            (target_rows >= 0)
            & (target_rows < self.shape[0])
            & (target_columns >= 0)
            & (target_columns < self.shape[1])
        )
        if local:
            inside &= (rows // self.chunk_shape[0] == target_rows // self.chunk_shape[0]) & (
                columns // self.chunk_shape[1] == target_columns // self.chunk_shape[1]
            )
        first = repeated[inside]
        second = target_rows[inside] * self.shape[1] + target_columns[inside]

        # Optionally remove repeated second endpoints for each first endpoint
        if self.deduplicate == "per_anchor" and first.size:
            order = np.argsort(first, kind="stable")
            keys = first[order].astype(np.int64) * np.int64(self.size) + second[order]
            _, keep = np.unique(keys, return_index=True)
            positions = order[np.sort(keep)]
            first, second = first[positions], second[positions]
        return first, second

    def _independent(self, count: int) -> tuple[NDArrayNum, NDArrayNum]:
        """Draw each first endpoint and offset independently across the full grid."""

        # Draw every first endpoint independently
        row_offset, column_offset = self._offsets(count, self.max_distance)
        rows = self.rng.integers(0, self.shape[0], row_offset.size, dtype=np.int64)
        columns = self.rng.integers(0, self.shape[1], column_offset.size, dtype=np.int64)
        target_rows, target_columns = rows + row_offset, columns + column_offset

        # Keep only second endpoints that remain inside the raster after applying offsets
        inside = (
            (target_rows >= 0)
            & (target_rows < self.shape[0])
            & (target_columns >= 0)
            & (target_columns < self.shape[1])
        )
        return rows[inside] * self.shape[1] + columns[inside], (
            target_rows[inside] * self.shape[1] + target_columns[inside]
        )

    ####################
    # STRATEGY CHOICE
    ####################

    def _candidates(self, count: int) -> tuple[NDArrayNum, NDArrayNum]:
        """Draw one limited batch of possible pairs with the selected strategy."""

        # Split the batch between nearby and full-range pairs in the requested proportion
        local_count = int(round(count * self.hybrid_local_fraction))
        global_count = count - local_count
        first_parts: list[NDArrayNum] = []
        second_parts: list[NDArrayNum] = []

        # Draw full-range pairs independently and reuse first endpoints only for nearby pairs
        if self.strategy == "independent":
            if local_count:
                anchors = self._sample_anchors(min(self.anchors_per_round, local_count), chunk_aligned=True)
                first, second = self._from_anchors(anchors, local_count, local=True)
                first_parts.append(first)
                second_parts.append(second)
            if global_count:
                first, second = self._independent(global_count)
                first_parts.append(first)
                second_parts.append(second)

        # Reuse one set of first endpoints for both nearby and full-range pairs
        elif self.strategy in {"anchors", "chunk_anchors"}:
            anchors = self._sample_anchors(
                min(self.anchors_per_round, count), chunk_aligned=self.strategy == "chunk_anchors"
            )
            for part_count, local in ((local_count, True), (global_count, False)):
                if part_count:
                    first, second = self._from_anchors(anchors, part_count, local=local)
                    first_parts.append(first)
                    second_parts.append(second)

        # Draw several distances and directions from each first endpoint
        else:
            pairs_per_anchor = self.distances_per_anchor * self.angles_per_distance
            for part_count, local in ((local_count, True), (global_count, False)):
                if part_count:
                    anchors = self._sample_anchors(int(np.ceil(part_count / pairs_per_anchor)), chunk_aligned=local)
                    first, second = self._anchor_batched(anchors, local=local)
                    first_parts.append(first[:part_count])
                    second_parts.append(second[:part_count])

        # Return empty integer arrays when neither part requested a pair
        if not first_parts:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return np.concatenate(first_parts)[:count], np.concatenate(second_parts)[:count]

    #################
    # PAIR COLLECTION
    #################

    def sample(self) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum]:
        """Collect the requested number of pairs whose two raster values are available."""

        # Count available cells without loading a complete Dask mask
        if is_dask_array(self.array):
            dask_array = __import__("dask.array", fromlist=["array"])
            n_valid = int(dask_array.count_nonzero(dask_array.isfinite(self.array)).compute())
        else:
            n_valid = int(np.count_nonzero(np.isfinite(self.array)))
        finite_fraction = max(n_valid / max(self.size, 1), 1e-12)

        # Limit a unique sample to the number of different pairs that can exist
        maximum_unique = n_valid * (n_valid - 1) // 2
        target = min(self.n_pairs, maximum_unique)
        if target < self.n_pairs:
            warnings.warn(
                f"n_pairs exceeds the {maximum_unique} possible finite pairs; using that maximum.", UserWarning
            )
        if target == 0:
            raise ValueError("At least two finite raster cells are required to sample pairs.")

        # Use the share of available cells to estimate how many possible pairs each round needs
        pair_acceptance = max(finite_fraction**2, 1e-12)
        first_parts: list[NDArrayNum] = []
        second_parts: list[NDArrayNum] = []
        remaining, stalled = target, 0
        for _ in range(self.max_rounds):
            if remaining == 0:
                break

            # Limit each round so a raster with little data cannot create a very large temporary array
            estimated = int(np.ceil(remaining / pair_acceptance))
            maximum_batch = max(1, int(np.ceil(target * self.max_oversample)))
            count = min(max(estimated, min(self.batch_pairs, maximum_batch)), maximum_batch)
            first, second = self._candidates(count)

            # Read only proposed endpoints, then keep pairs where both values are available
            if first.size:
                finite = np.isfinite(_take_raster_values(self.array, first)) & np.isfinite(
                    _take_raster_values(self.array, second)
                )
                first, second = first[finite], second[finite]

            # Keep only the remaining number of pairs and detect rounds that find nothing
            if first.size:
                take = min(remaining, first.size)
                first_parts.append(first[:take])
                second_parts.append(second[:take])
                remaining -= take
                stalled = 0
            else:
                stalled += 1
            if stalled >= 5:
                break

        # Fail clearly when no round found one pair with two available values
        if not first_parts:
            raise ValueError("No finite raster pairs could be sampled.")

        # Remove duplicates across rounds when the caller requests global uniqueness
        first, second = np.concatenate(first_parts), np.concatenate(second_parts)
        if self.deduplicate == "global":
            first, second = _deduplicate_pairs(first, second, n_observations=self.size)
        if first.size < target:
            warnings.warn(
                f"Sampled {first.size} finite pairs out of {target} requested after {self.max_rounds} rounds.",
                UserWarning,
            )

        # Use the requested integer type and calculate exact map distances for the result
        first, second = first.astype(self.index_dtype, copy=False), second.astype(self.index_dtype, copy=False)
        first_rows, first_columns = np.divmod(first.astype(np.int64), self.shape[1])
        second_rows, second_columns = np.divmod(second.astype(np.int64), self.shape[1])
        distances = np.hypot(
            (second_columns - first_columns) * self.dx,
            (second_rows - first_rows) * self.dy,
        ).astype(self.distance_dtype, copy=False)
        return first, second, distances


##############################
# 3/ IRREGULAR POINT SAMPLING
##############################


@dataclass(frozen=True)
class _GridSpec:
    """Grid cells used to find nearby point rows without comparing every point."""

    cell_size: float
    x_min: float
    y_min: float
    n_columns: int
    n_rows: int


class _IrregularPairSampler:
    """Draw pairs from point locations using exact ring searches or nearby matches."""

    #################
    # CONFIGURATION
    #################

    def __init__(
        self,
        coordinates: NDArrayNum,
        *,
        n_pairs: int,
        min_distance: float,
        max_distance: float,
        n_bins: int,
        strategy: Literal["kdtree", "hashgrid", "nn_logvector"],
        anchors_per_round: int,
        attempts_per_anchor: int,
        max_rounds: int,
        cell_size: float | None,
        nn_tolerance: float,
        nn_batch_size: int,
        nn_oversample: float,
        nn_max_batches: int,
        random_state: int | np.random.Generator | None,
        index_dtype: Any,
        distance_dtype: Any,
    ) -> None:
        # Store coordinates and sampling options with consistent numeric types
        self.coordinates = np.asarray(coordinates, dtype=np.float64)
        self.size = len(self.coordinates)
        self.n_pairs = int(n_pairs)
        self.min_distance, self.max_distance = float(min_distance), float(max_distance)
        self.n_bins, self.strategy = int(n_bins), strategy
        self.anchors_per_round, self.attempts_per_anchor = int(anchors_per_round), int(attempts_per_anchor)
        self.max_rounds = int(max_rounds)
        self.cell_size = self.max_distance / 8 if cell_size is None else float(cell_size)
        self.nn_tolerance, self.nn_batch_size = float(nn_tolerance), int(nn_batch_size)
        self.nn_oversample, self.nn_max_batches = float(nn_oversample), int(nn_max_batches)
        self.rng = (
            random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        )
        self.index_dtype, self.distance_dtype = np.dtype(index_dtype), np.dtype(distance_dtype)

        # Prepare distance rings now and build search helpers only if the chosen strategy needs them
        self.edges = np.geomspace(self.min_distance, self.max_distance, self.n_bins + 1)
        self._tree: cKDTree | None = None
        self.grid: dict[tuple[int, int], NDArrayNum] | None = None
        self.grid_spec: _GridSpec | None = None

        # Check coordinates and batch options before starting repeated searches
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2 or self.size < 2:
            raise ValueError("coordinates must contain at least two X/Y points.")
        if self.n_pairs < 1 or self.n_bins < 1:
            raise ValueError("n_pairs and n_bins must be positive integers.")
        if not 0 < self.min_distance < self.max_distance:
            raise ValueError("Require 0 < min_distance < max_distance.")
        if strategy not in {"kdtree", "hashgrid", "nn_logvector"}:
            raise ValueError("Unknown irregular point pair sampling strategy.")
        if self.anchors_per_round < 1 or self.attempts_per_anchor < 1 or self.max_rounds < 1:
            raise ValueError("Exact search batch and round controls must be positive integers.")
        if self.cell_size <= 0 or self.nn_tolerance <= 0:
            raise ValueError("cell_size and nn_tolerance must be strictly positive.")
        if self.nn_batch_size < 1 or self.nn_oversample <= 0 or self.nn_max_batches < 1:
            raise ValueError("Nearest-neighbor batch controls must be strictly positive.")

    ########################
    # NEARBY POINT SEARCH
    ########################

    def _build_grid(self) -> None:
        """Group point rows into grid cells used for nearby searches."""

        # Convert coordinates to integer cells relative to the point cloud origin
        x, y = self.coordinates.T
        x_min, y_min = float(x.min()), float(y.min())
        columns = np.floor((x - x_min) / self.cell_size).astype(np.int32)
        rows = np.floor((y - y_min) / self.cell_size).astype(np.int32)
        n_columns, n_rows = int(columns.max()) + 1, int(rows.max()) + 1
        keys = columns.astype(np.int64) * np.int64(n_rows) + rows

        # Sort once so all point rows from one occupied cell sit together
        order = np.argsort(keys, kind="stable")
        boundaries = np.r_[0, np.flatnonzero(np.diff(keys[order])) + 1, self.size]
        self.grid = {}
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            column, row = divmod(int(keys[order[start]]), n_rows)
            self.grid[(column, row)] = order[start:stop].astype(np.int32, copy=False)

        # Keep the grid details needed to find cells around a distance ring
        self.grid_spec = _GridSpec(self.cell_size, x_min, y_min, n_columns, n_rows)

    @property
    def tree(self) -> cKDTree:
        """Build SciPy's nearby point search tree only when a strategy needs it."""

        # Avoid building the tree for grid-based strategies that never use it
        if self._tree is None:
            self._tree = cKDTree(self.coordinates)
        return self._tree

    def _hash_candidates(self, anchor: int, inner: float, outer: float) -> NDArrayNum:
        """Collect point rows from grid cells that may cross one distance ring."""

        # Build the search grid on first use so other strategies do not store it
        if self.grid is None or self.grid_spec is None:
            self._build_grid()
        assert self.grid is not None and self.grid_spec is not None
        x, y = self.coordinates[anchor]
        center_column = int(np.floor((x - self.grid_spec.x_min) / self.cell_size))
        center_row = int(np.floor((y - self.grid_spec.y_min) / self.cell_size))
        radius = int(np.ceil(outer / self.cell_size))

        # Visit only grid cells inside the square around the outer distance
        parts: list[NDArrayNum] = []
        for column in range(max(0, center_column - radius), min(self.grid_spec.n_columns, center_column + radius + 1)):
            for row in range(max(0, center_row - radius), min(self.grid_spec.n_rows, center_row + radius + 1)):
                # Skip cells that cannot touch the requested distance ring
                cell_x_min = self.grid_spec.x_min + column * self.cell_size
                cell_y_min = self.grid_spec.y_min + row * self.cell_size
                cell_x_max, cell_y_max = cell_x_min + self.cell_size, cell_y_min + self.cell_size
                nearest_x = max(cell_x_min - x, 0.0, x - cell_x_max)
                nearest_y = max(cell_y_min - y, 0.0, y - cell_y_max)
                farthest_x = max(abs(x - cell_x_min), abs(x - cell_x_max))
                farthest_y = max(abs(y - cell_y_min), abs(y - cell_y_max))
                if nearest_x**2 + nearest_y**2 >= outer**2 or farthest_x**2 + farthest_y**2 < inner**2:
                    continue
                indexes = self.grid.get((column, row))
                if indexes is not None:
                    parts.append(indexes)

        # Join point rows from matching cells before checking their exact distances
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)

    def _one_in_ring(self, anchor: int, inner: float, outer: float) -> tuple[int, float] | None:
        """Select one second point within an exact distance range of the first."""

        # Ask the selected search helper for points inside the outer distance
        candidates = (
            np.asarray(self.tree.query_ball_point(self.coordinates[anchor], outer), dtype=np.int64)
            if self.strategy == "kdtree"
            else self._hash_candidates(anchor, inner, outer).astype(np.int64, copy=False)
        )
        if candidates.size == 0:
            return None

        # Check exact distances because both helpers may also return points outside the inner distance
        differences = self.coordinates[candidates] - self.coordinates[anchor]
        squared = np.sum(differences**2, axis=1)
        in_ring = (squared >= inner**2) & (squared < outer**2) & (candidates != anchor)
        if not np.any(in_ring):
            return None

        # Choose one matching endpoint at random so stored row order does not bias the sample
        chosen = int(self.rng.choice(candidates[in_ring]))
        return chosen, float(np.linalg.norm(self.coordinates[chosen] - self.coordinates[anchor]))

    ###################
    # OFFSET MATCHING
    ###################

    def _nearest_vector(self) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum]:
        """Draw offsets across log-spaced distances and match them to nearby points."""

        # Collect limited batches until the requested number of pairs is reached
        first_parts: list[NDArrayNum] = []
        second_parts: list[NDArrayNum] = []
        distance_parts: list[NDArrayNum] = []
        remaining = self.n_pairs
        for _ in range(self.nn_max_batches):
            if remaining == 0:
                break

            # Draw random first points, log-spaced distances, and random directions
            count = min(self.nn_batch_size, int(np.ceil(self.nn_oversample * remaining)))
            anchors = self.rng.integers(0, self.size, count, dtype=np.int64)
            radii = np.exp(self.rng.uniform(np.log(self.min_distance), np.log(self.max_distance), count))
            angles = self.rng.uniform(0, 2 * np.pi, count)
            proposals = self.coordinates[anchors] + np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

            # Match each proposed endpoint to the nearest observed point within the allowed error
            proposal_distance, neighbors = self.tree.query(proposals, k=1)
            accepted = (neighbors != anchors) & (proposal_distance <= self.nn_tolerance * radii)
            first, second = anchors[accepted], neighbors[accepted].astype(np.int64, copy=False)
            distance = np.linalg.norm(self.coordinates[second] - self.coordinates[first], axis=1)
            in_range = (distance >= self.min_distance) & (distance <= self.max_distance)

            # Check exact distances and keep only the number of pairs still needed
            take = min(remaining, int(np.count_nonzero(in_range)))
            if take:
                first_parts.append(first[in_range][:take])
                second_parts.append(second[in_range][:take])
                distance_parts.append(distance[in_range][:take])
                remaining -= take

        # Distinguish finding no pair from finding fewer pairs than requested
        if not first_parts:
            raise ValueError("No point pairs could be sampled within the requested distances.")
        return np.concatenate(first_parts), np.concatenate(second_parts), np.concatenate(distance_parts)

    #################
    # PAIR COLLECTION
    #################

    def sample(self) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum]:
        """Collect point pairs with the selected search strategy."""

        # Use proposed offsets directly for the nearest-point strategy
        if self.strategy == "nn_logvector":
            first, second, distances = self._nearest_vector()
        else:
            # Search one first point and distance range at a time for the exact strategies
            first_values: list[int] = []
            second_values: list[int] = []
            distance_values: list[float] = []
            for _ in range(self.max_rounds):
                if len(first_values) >= self.n_pairs:
                    break

                # Reuse first points when one round requests more pairs than there are points
                replace = self.anchors_per_round > self.size
                anchors = self.rng.choice(self.size, self.anchors_per_round, replace=replace)
                for anchor in anchors:
                    for _ in range(self.attempts_per_anchor):
                        # Choose each log-spaced distance range equally often
                        bin_index = int(self.rng.integers(0, self.n_bins))
                        selected = self._one_in_ring(int(anchor), self.edges[bin_index], self.edges[bin_index + 1])
                        if selected is not None:
                            selected_index, distance = selected
                            first_values.append(int(anchor))
                            second_values.append(selected_index)
                            distance_values.append(distance)
                        if len(first_values) == self.n_pairs:
                            break
                    if len(first_values) == self.n_pairs:
                        break

            # Convert the collected Python lists to numeric arrays
            if not first_values:
                raise ValueError("No point pairs could be sampled within the requested distances.")
            first = np.asarray(first_values)
            second = np.asarray(second_values)
            distances = np.asarray(distance_values)

        # Warn when fewer pairs were found, but return every successful pair
        if first.size < self.n_pairs:
            warnings.warn(f"Sampled {first.size} point pairs out of {self.n_pairs} requested.", UserWarning)
        return (
            first.astype(self.index_dtype, copy=False),
            second.astype(self.index_dtype, copy=False),
            distances.astype(self.distance_dtype, copy=False),
        )


##################################
# 4/ OBJECT METHOD IMPLEMENTATIONS
##################################


def _random_raster_pairs(
    array: Any,
    *,
    dx: float,
    dy: float,
    n_pairs: int,
    min_distance: float,
    max_distance: float,
    random_state: int | np.random.Generator | None,
    max_rounds: int,
    batch_pairs: int,
) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum]:
    """Draw independent raster endpoints and keep pairs in the requested distance range."""

    # Start empty result arrays that each round extends after removing duplicates
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    size, n_columns = int(np.prod(array.shape)), int(array.shape[1])
    first: NDArrayNum = np.empty(0, dtype=np.int64)
    second: NDArrayNum = np.empty(0, dtype=np.int64)
    for _ in range(max_rounds):
        remaining = n_pairs - first.size
        if remaining <= 0:
            break

        # Draw a limited batch of independent endpoints to fill the remaining sample
        count = min(batch_pairs, max(10_000, remaining * 3))
        first_candidate = rng.integers(0, size, count, dtype=np.int64)
        second_candidate = rng.integers(0, size, count, dtype=np.int64)
        first_rows, first_columns = np.divmod(first_candidate, n_columns)
        second_rows, second_columns = np.divmod(second_candidate, n_columns)
        distances = np.hypot((second_columns - first_columns) * dx, (second_rows - first_rows) * dy)

        # Remove self-pairs, wrong distances, and pairs with a missing value
        keep = (
            (first_candidate != second_candidate)
            & (distances >= min_distance)
            & (distances <= max_distance)
            & np.isfinite(_take_raster_values(array, first_candidate))
            & np.isfinite(_take_raster_values(array, second_candidate))
        )
        if np.any(keep):
            # Remove duplicates across all rounds before keeping the requested count
            first, second = _deduplicate_pairs(
                np.concatenate((first, first_candidate[keep])),
                np.concatenate((second, second_candidate[keep])),
                n_observations=size,
            )
            first, second = first[:n_pairs], second[:n_pairs]

    # Distinguish finding no pair from finding fewer unique pairs than requested
    if first.size == 0:
        raise ValueError("No finite raster pairs could be sampled.")
    if first.size < n_pairs:
        warnings.warn(f"Sampled {first.size} unique raster pairs out of {n_pairs} requested.", UserWarning)

    # Calculate exact map distances after the final pair order is known
    first_rows, first_columns = np.divmod(first, n_columns)
    second_rows, second_columns = np.divmod(second, n_columns)
    distances = np.hypot((second_columns - first_columns) * dx, (second_rows - first_rows) * dy)
    return first, second, distances


def _pair_dataset(
    *,
    first: NDArrayNum,
    second: NDArrayNum,
    pair_values: NDArrayNum,
    distances: NDArrayNum,
    pair_coordinates: dict[str, NDArrayNum],
    attrs: dict[str, Any],
) -> xr.Dataset:
    """Build the Xarray Dataset returned by raster and point cloud pairsample() methods."""

    # Store the two endpoints along one labelled dimension shared by row numbers and values
    indexes = np.column_stack((first, second))
    data_vars: dict[str, Any] = {
        "index": (("pair", "endpoint"), indexes),
        "value": (("pair", "endpoint"), pair_values),
        "distance": ("pair", distances),
    }

    # Add raster or point coordinates while keeping the same core dataset layout
    for name, coordinate in pair_coordinates.items():
        data_vars[name] = (("pair", "endpoint"), coordinate)

    # Name the first and second endpoints so later code need not guess from array positions
    return xr.Dataset(
        data_vars=data_vars,
        coords={"pair": np.arange(len(first)), "endpoint": ["first", "second"]},
        attrs=attrs,
    )


def _sample_raster_pairs(
    raster: Any,
    *,
    band: int,
    n_pairs: int,
    sampling: Literal["loglag", "random_xy"],
    min_distance: float | None,
    max_distance: float | None,
    random_state: int | np.random.Generator | None,
    mask: Any | None,
    strategy: Literal["independent", "anchors", "chunk_anchors", "anchor_batched"],
    deduplicate: Literal["none", "per_anchor", "global"],
    batch_pairs: int,
    max_rounds: int,
    max_oversample: float,
    chunks_per_round: int,
    anchors_per_round: int,
    distances_per_anchor: int,
    angles_per_distance: int,
    hybrid_local_fraction: float,
    max_local_distance: float | None,
    index_dtype: Any,
    distance_dtype: Any,
) -> xr.Dataset:
    """Check raster pairsample() inputs, draw pairs, and build its Xarray result.

    _RegularPairSampler.sample() draws log-spaced pairs, while _random_raster_pairs() draws independent endpoints.
    _pair_dataset() gives either result the shared labelled layout.
    """

    # Select the raster band and check the requested output number types
    array = _selected_raster_data(raster, band)
    index_type, distance_type = np.dtype(index_dtype), np.dtype(distance_dtype)
    if not np.issubdtype(index_type, np.integer) or not np.issubdtype(distance_type, np.floating):
        raise TypeError("index_dtype must be integer and distance_dtype must be floating.")
    if int(np.prod(array.shape)) - 1 > np.iinfo(index_type).max:
        raise ValueError("index_dtype cannot represent every cell in this raster.")

    # Convert an array, raster, or vector mask to one Boolean grid
    if mask is not None:
        mask_array = _mask_on_raster(mask, raster, "inside", "raise")

        # Apply the mask without loading a Dask source array
        if is_dask_array(array):
            dask_array = __import__("dask.array", fromlist=["array"])
            array = dask_array.where(mask_array, array, np.nan)
        else:
            array = np.where(mask_array, array, np.nan)

    # Choose default map distances from the cell size and raster extent
    dx, dy = (float(abs(value)) for value in get_geo_attr(raster, "res"))
    diagonal = float(np.hypot((array.shape[1] - 1) * dx, (array.shape[0] - 1) * dy))
    minimum = min(dx, dy) if min_distance is None else float(min_distance)
    maximum = diagonal if max_distance is None else float(max_distance)
    if not 0 < minimum < maximum:
        raise ValueError("Require 0 < min_distance < max_distance.")

    # Call the log-spaced or independent endpoint sampling workflow
    if sampling == "loglag":
        first, second, distances = _RegularPairSampler(
            array,
            dx=dx,
            dy=dy,
            n_pairs=n_pairs,
            min_distance=minimum,
            max_distance=maximum,
            strategy=strategy,
            deduplicate=deduplicate,
            random_state=random_state,
            batch_pairs=batch_pairs,
            max_rounds=max_rounds,
            max_oversample=max_oversample,
            chunks_per_round=chunks_per_round,
            anchors_per_round=anchors_per_round,
            distances_per_anchor=distances_per_anchor,
            angles_per_distance=angles_per_distance,
            hybrid_local_fraction=hybrid_local_fraction,
            max_local_distance=max_local_distance,
            index_dtype=index_dtype,
            distance_dtype=distance_dtype,
        ).sample()
    elif sampling == "random_xy":
        first, second, distances = _random_raster_pairs(
            array,
            dx=dx,
            dy=dy,
            n_pairs=n_pairs,
            min_distance=minimum,
            max_distance=maximum,
            random_state=random_state,
            max_rounds=max_rounds,
            batch_pairs=batch_pairs,
        )
    else:
        raise ValueError("sampling must be 'loglag' or 'random_xy'.")

    # Use the requested number types and recover map coordinates for both endpoints
    first = first.astype(index_type, copy=False)
    second = second.astype(index_type, copy=False)
    distances = distances.astype(distance_type, copy=False)
    first_rows, first_columns = np.divmod(first.astype(np.int64), int(array.shape[1]))
    second_rows, second_columns = np.divmod(second.astype(np.int64), int(array.shape[1]))
    first_x, first_y = raster.ij2xy(first_rows, first_columns)
    second_x, second_y = raster.ij2xy(second_rows, second_columns)

    # Load only the selected raster values when building the Xarray result
    return _pair_dataset(
        first=first,
        second=second,
        pair_values=np.column_stack((_take_raster_values(array, first), _take_raster_values(array, second))),
        distances=distances,
        pair_coordinates={
            "row": np.column_stack((first_rows, second_rows)),
            "column": np.column_stack((first_columns, second_columns)),
            "x": np.column_stack((first_x, second_x)),
            "y": np.column_stack((first_y, second_y)),
        },
        attrs={
            "source": "raster",
            "crs": str(get_geo_attr(raster, "crs")),
            "sampling": sampling,
            "strategy": strategy if sampling == "loglag" else "random_xy",
            "deduplicate": deduplicate,
            "requested_pairs": int(n_pairs),
            "accepted_pairs": int(len(first)),
            "min_distance": minimum,
            "max_distance": maximum,
            "band": int(band),
        },
    )


def _sample_point_pairs(
    pointcloud: Any,
    *,
    n_pairs: int,
    sampling: Literal["loglag", "random_xy"],
    min_distance: float | None,
    max_distance: float | None,
    random_state: int | np.random.Generator | None,
    mask: Any | None,
    strategy: Literal["kdtree", "hashgrid", "nn_logvector"],
    n_bins: int,
    anchors_per_round: int,
    attempts_per_anchor: int,
    max_rounds: int,
    cell_size: float | None,
    nn_tolerance: float,
    nn_batch_size: int,
    nn_oversample: float,
    nn_max_batches: int,
    index_dtype: Any,
    distance_dtype: Any,
) -> xr.Dataset:
    """Check point cloud pairsample() inputs, draw pairs, and build its Xarray result.

    _IrregularPairSampler.sample() searches for log-spaced pairs; the independent path draws and filters endpoints
    directly. _pair_dataset() gives either result the shared labelled layout.
    """

    # Load the point table because pair searches need all coordinates
    dataframe = pointcloud.ds.compute() if is_dask_dataframe(pointcloud.ds) else pointcloud.ds
    values = np.asarray(
        dataframe[pointcloud.data_column] if pointcloud.data_column is not None else dataframe.geometry.z
    )
    coordinates = np.column_stack((dataframe.geometry.x.to_numpy(), dataframe.geometry.y.to_numpy()))

    # Keep rows with available coordinates and values, then apply the optional mask
    valid = np.isfinite(values) & np.all(np.isfinite(coordinates), axis=1)
    if mask is not None:
        # Accept vector objects and GeoDataFrames through the same mask path
        vector = _as_vector(mask)
        if vector is not None:
            valid &= np.asarray(vector.create_mask(ref=pointcloud, as_array=True), dtype=bool).squeeze()
        else:
            mask_array = np.asarray(mask).squeeze()
            if mask_array.ndim != 1 or len(mask_array) != len(values) or mask_array.dtype != bool:
                raise ValueError("mask must be Boolean with one value per point.")
            valid &= mask_array

    # Keep source row numbers so the result refers back to the original point table
    original_indexes = np.flatnonzero(valid)
    coordinates_valid, values_valid = coordinates[valid], values[valid]
    if len(values_valid) < 2:
        raise ValueError("At least two finite points are required to sample pairs.")

    # Check that the requested integer type can hold every original row number
    index_type, distance_type = np.dtype(index_dtype), np.dtype(distance_dtype)
    if not np.issubdtype(index_type, np.integer) or not np.issubdtype(distance_type, np.floating):
        raise TypeError("index_dtype must be integer and distance_dtype must be floating.")
    if len(values) - 1 > np.iinfo(index_type).max:
        raise ValueError("index_dtype cannot represent every point in this point cloud.")

    # Choose default distances from the point extent and typical point spacing
    bounds = np.ptp(coordinates_valid, axis=0)
    diagonal = float(np.hypot(*bounds))
    density_spacing = float(np.sqrt(max(bounds[0] * bounds[1], 0) / len(values_valid)))
    minimum = max(0.5 * density_spacing, float(np.finfo(float).eps)) if min_distance is None else float(min_distance)
    maximum = diagonal if max_distance is None else float(max_distance)
    if not 0 < minimum < maximum:
        raise ValueError("Require 0 < min_distance < max_distance.")
    if n_pairs < 1 or max_rounds < 1:
        raise ValueError("n_pairs and max_rounds must be positive integers.")

    # Use the selected point search for log-spaced distances
    if sampling == "loglag":
        first, second, distances = _IrregularPairSampler(
            coordinates_valid,
            n_pairs=n_pairs,
            min_distance=minimum,
            max_distance=maximum,
            n_bins=n_bins,
            strategy=strategy,
            anchors_per_round=anchors_per_round,
            attempts_per_anchor=attempts_per_anchor,
            max_rounds=max_rounds,
            cell_size=cell_size,
            nn_tolerance=nn_tolerance,
            nn_batch_size=nn_batch_size,
            nn_oversample=nn_oversample,
            nn_max_batches=nn_max_batches,
            random_state=random_state,
            index_dtype=index_dtype,
            distance_dtype=distance_dtype,
        ).sample()
    elif sampling == "random_xy":
        # Draw independent endpoints in limited rounds for uniform random sampling
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        first = np.empty(0, dtype=np.int64)
        second = np.empty(0, dtype=np.int64)
        for _ in range(max_rounds):
            remaining = n_pairs - first.size
            if remaining <= 0:
                break

            # Draw extra possible pairs before checking exact distances and removing duplicates
            count = min(nn_batch_size, max(10_000, remaining * 3))
            first_candidate = rng.integers(0, len(values_valid), count, dtype=np.int64)
            second_candidate = rng.integers(0, len(values_valid), count, dtype=np.int64)
            candidate_distances = np.linalg.norm(
                coordinates_valid[first_candidate] - coordinates_valid[second_candidate], axis=1
            )
            keep = (
                (first_candidate != second_candidate)
                & (candidate_distances >= minimum)
                & (candidate_distances <= maximum)
            )
            if np.any(keep):
                # Remove duplicates across rounds before keeping the requested number
                first, second = _deduplicate_pairs(
                    np.concatenate((first, first_candidate[keep])),
                    np.concatenate((second, second_candidate[keep])),
                    n_observations=len(values_valid),
                )
                first, second = first[:n_pairs], second[:n_pairs]

        # Return a smaller sample with a warning, but fail when no matching pair exists
        if first.size == 0:
            raise ValueError("No point pairs could be sampled within the requested distances.")
        if first.size < n_pairs:
            warnings.warn(f"Sampled {first.size} unique point pairs out of {n_pairs} requested.", UserWarning)
        distances = np.linalg.norm(coordinates_valid[first] - coordinates_valid[second], axis=1)
    else:
        raise ValueError("sampling must be 'loglag' or 'random_xy'.")

    # Map kept row numbers back to the original point table and requested number types
    original_first = original_indexes[first].astype(index_type, copy=False)
    original_second = original_indexes[second].astype(index_type, copy=False)
    distances = distances.astype(distance_type, copy=False)

    # Build the same labelled Xarray layout used for raster pairs
    return _pair_dataset(
        first=original_first,
        second=original_second,
        pair_values=np.column_stack((values_valid[first], values_valid[second])),
        distances=distances,
        pair_coordinates={
            "x": np.column_stack((coordinates_valid[first, 0], coordinates_valid[second, 0])),
            "y": np.column_stack((coordinates_valid[first, 1], coordinates_valid[second, 1])),
        },
        attrs={
            "source": "pointcloud",
            "crs": str(pointcloud.crs),
            "sampling": sampling,
            "strategy": strategy if sampling == "loglag" else "random_xy",
            "requested_pairs": int(n_pairs),
            "accepted_pairs": int(len(first)),
            "min_distance": minimum,
            "max_distance": maximum,
        },
    )
