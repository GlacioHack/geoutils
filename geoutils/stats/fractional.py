# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Calculate raster statistics from the area that each polygon covers."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray
from rasterio.transform import Affine

from geoutils._dispatch import _is_raster, _is_vector, is_dask_array
from geoutils._misc import import_optional
from geoutils.multiproc.chunked import iter_chunk_slices
from geoutils.multiproc.cluster import _map_bounded
from geoutils.multiproc.readers import _read_values, _ValueReader
from geoutils.operators.overlap import (
    GridIntersection,
    OverlapBackend,
    _grid_intersection_fractions,
    _grid_intersection_local_data,
    _union_geometries,
    _union_geometries_by_label,
)
from geoutils.operators.reducer import Count, Mean, Reducer, Sum, _can_reduce_arrays, _combined_weights
from geoutils.sampling.support import _sampling_specification
from geoutils.stats.reduction import _flox_groupby_reduce, _merge_blocks, _reducer_from_statistic
from geoutils.vector.base import _as_geodataframe

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.raster.base import RasterBase
    from geoutils.stats.grouping import _GroupDefinition
    from geoutils.stats.reduction import _Statistics


def _vector_fractional_groups(
    source: Any,
    by: Mapping[str, Any],
    support: RasterBase,
    definitions: Mapping[str, _GroupDefinition],
) -> tuple[str, pd.Index, NDArray[Any], bool]:
    """Combine polygons in each vector group and report whether an outside group is also needed."""

    if len(by) != 1:
        raise ValueError("Fractional statistics currently require exactly one vector grouper.")
    name, specification = next(iter(by.items()))
    group_source, selector = _sampling_specification(source, specification)
    if not _is_vector(group_source):
        raise TypeError("Fractional statistics require a vector object in argument by.")
    if selector is not None and not isinstance(selector, str):
        raise TypeError(f"Vector selector for {name!r} must be a column name.")

    # Reproject the polygons to the raster CRS before calculating their exact cell intersections
    dataframe = _as_geodataframe(group_source)
    if support.crs is not None and dataframe.crs != support.crs:
        dataframe = dataframe.to_crs(support.crs)
    geometries = np.asarray(dataframe.geometry.values, dtype=object)
    if selector is None:
        union = _union_geometries(geometries)
        labels = pd.CategoricalIndex([False, True], categories=[False, True], ordered=True, name=name)
        return name, labels, np.asarray([union], dtype=object), True

    if selector not in dataframe.columns:
        raise ValueError(f"Vector column {selector!r} does not exist.")
    selected_labels = dataframe[selector]
    if selected_labels.isna().any():
        raise ValueError("Fractional vector group labels cannot be missing.")
    definition = definitions.get(name)
    if definition is not None:
        assert definition.groups is not None
        labels = definition.groups
    elif isinstance(selected_labels.dtype, pd.CategoricalDtype):
        labels = pd.CategoricalIndex(
            selected_labels.dtype.categories,
            categories=selected_labels.dtype.categories,
            ordered=True,
            name=name,
        )
    else:
        unique = pd.Index(pd.unique(selected_labels), name=name)
        labels = pd.CategoricalIndex(unique, categories=unique, ordered=True, name=name)

    # Combine polygons with the same label so overlapping areas count only once for that group
    grouped_geometries = _union_geometries_by_label(geometries, selected_labels, labels)
    return name, labels, grouped_geometries, False


def _inside_outside_overlap(inside: GridIntersection, shape: tuple[int, int]) -> GridIntersection:
    """Create outside and inside groups from the cell areas covered by a polygon union."""

    coverage = np.zeros(shape[0] * shape[1], dtype=np.float64)
    inside_flat = inside.rows * shape[1] + inside.columns
    coverage[inside_flat] = inside.fractions
    outside_flat = np.flatnonzero(coverage < 1)
    outside_fractions = 1 - coverage[outside_flat]

    rows = np.concatenate((outside_flat // shape[1], inside.rows)).astype(np.int64, copy=False)
    columns = np.concatenate((outside_flat % shape[1], inside.columns)).astype(np.int64, copy=False)
    fractions = np.concatenate((outside_fractions, inside.fractions))
    offsets = np.asarray([0, len(outside_flat), len(outside_flat) + len(inside.rows)], dtype=np.int64)
    return GridIntersection(offsets=offsets, rows=rows, columns=columns, fractions=fractions)


def _fractional_statistic_reducers(statistics: _Statistics) -> list[Reducer | None]:
    """Convert statistic names to Reducers and pass through Reducers supplied directly."""

    reducers: list[Reducer | None] = []
    unsupported = []
    count_aliases = {"validcount", "totalcount", "percentagevalidpoints"}
    for statistic, name, alias in zip(statistics.requested, statistics.names, statistics.aliases):
        if isinstance(statistic, Reducer):
            reducers.append(statistic)
            continue
        reducer = None if alias in count_aliases else _reducer_from_statistic(alias or "", fractional=True)
        reducers.append(reducer)
        if reducer is None and alias not in count_aliases:
            unsupported.append(name)
    if unsupported:
        raise ValueError(f"Fractional statistics do not support these statistics: {unsupported!r}.")
    return reducers


def _fractional_chunk_state(
    values: list[Any],
    mask: Any | None,
    transform: Affine,
    geometries: NDArray[Any],
    group_numbers: NDArray[Any],
    include_outside: bool,
    overlap_backend: OverlapBackend,
) -> tuple[NDArray[Any], NDArray[Any], dict[str, NDArray[Any]]]:
    """Calculate fractional counts and sums for one raster chunk."""

    # Read the chunk and calculate fractional weights once for all raster values
    arrays = [_read_values(array) for array in values]
    eligibility = None if mask is None else np.asarray(_read_values(mask), dtype=bool)
    overlap = _grid_intersection_fractions(geometries, transform, arrays[0].shape, backend=overlap_backend)
    if include_outside:
        overlap = _inside_outside_overlap(overlap, arrays[0].shape)
    n_groups = overlap.geometry_count
    state = {name: np.zeros((len(arrays), n_groups), dtype=np.float64) for name in ("count", "total", "sum")}

    # Collect fractional counts and sums for each raster value
    for value_index, array in enumerate(arrays):
        local_inputs = _grid_intersection_local_data(array, transform, overlap, eligible=eligibility)
        for group_index, local in enumerate(local_inputs):
            eligible = np.ones(len(local.values), dtype=bool)
            if eligibility is not None:
                eligible = eligibility.reshape(-1)[local.source_ids]

            # Both fractional counts apply the mask, but only the valid count excludes nodata
            weights = _combined_weights(local)
            valid_weights = np.where(local.valid, weights, 0)
            support_weights = local.support_weights
            assert support_weights is not None
            total_weights = np.where(eligible, support_weights, 0)
            weighted_values = np.where(local.valid, local.values, 0).astype(np.float64) * valid_weights
            state["count"][value_index, group_index] = valid_weights.sum()
            state["total"][value_index, group_index] = total_weights.sum()
            state["sum"][value_index, group_index] = weighted_values.sum()

    # As _merge_blocks() also requires integer cell counts, separate from the fractional counts
    size = np.diff(overlap.offsets).astype(np.int64)
    return group_numbers, size, state


def _fractional_mergeable_stats(
    source: Any,
    by: Mapping[str, Any],
    values: Mapping[str, Any],
    support: RasterBase,
    statistics: _Statistics,
    mask: Any | None,
    definitions: Mapping[str, _GroupDefinition],
    observed: bool,
    overlap_backend: OverlapBackend,
    backend: str,
    engine: str | None,
    mp_config: MultiprocConfig | None,
) -> pd.DataFrame:
    """
    Calculate fractional statistics across raster chunks.

    _vector_fractional_groups() combines polygons with the same label. A spatial index selects each chunk's groups,
    and _fractional_chunk_state() calculates their fractional counts and sums. _merge_blocks() merges these
    before we calculate fractional means and percentages.
    """

    # 1/ Check inputs

    # Check supported fractional statistics and reducers
    supported = {"sum", "mean", "validcount", "totalcount", "percentagevalidpoints"}
    if any(alias not in supported for alias in statistics.aliases):
        raise ValueError("Mergeable fractional statistics support only sums, means, and counts.")
    for statistic in statistics.requested:
        if isinstance(statistic, Reducer) and (
            type(statistic) not in {Sum, Mean, Count}
            or not _can_reduce_arrays(statistic)
            or statistic.error_structure is not None
            or statistic.minimum_inputs != type(statistic).minimum_inputs
            or statistic.default_nodata_propagation != "ignore"
            or any(name in vars(statistic) for name in ("reduce", "evaluate", "coefficients"))
        ):
            raise ValueError("Mergeable fractional statistics require unmodified Sum(), Mean(), or Count() reducers.")

    # Check values and mask against the raster grid shape
    if any(array.shape != support.shape for array in values.values()) or (
        mask is not None and mask.shape != support.shape
    ):
        raise ValueError("Fractional statistic values and mask must match the raster support shape.")

    # 2/ Prepare group geometries and matching raster chunks

    # Index the polygons to select intersecting groups for each chunk
    _, labels, geometries, include_outside = _vector_fractional_groups(source, by, support, definitions)
    tree = shapely.STRtree(geometries)

    # Choose the chunk layout after checking which arrays use Dask
    use_dask = any(is_dask_array(array) for array in [*values.values(), mask])
    if use_dask and mp_config is not None:
        raise ValueError("Cannot use Multiprocessing and Dask simultaneously in fractional stats().")
    tiles: list[tuple[slice, ...]]
    if use_dask:
        dask = import_optional("dask")
        import dask.array as da

        # Align values and mask to the same chunks
        chunk_source = next(array for array in [*values.values(), mask] if is_dask_array(array))
        chunks = chunk_source.chunks
        blocks = [list(da.asarray(array).rechunk(chunks).to_delayed().ravel()) for array in values.values()]
        mask_blocks = (
            [None] * len(blocks[0]) if mask is None else list(da.asarray(mask).rechunk(chunks).to_delayed().ravel())
        )
        row_starts, column_starts = np.r_[0, np.cumsum(chunks[0])], np.r_[0, np.cumsum(chunks[1])]
        tiles = [
            (slice(r0, r1), slice(c0, c1))
            for r0, r1 in zip(row_starts[:-1], row_starts[1:])
            for c0, c1 in zip(column_starts[:-1], column_starts[1:])
        ]
    else:
        tiles = list(iter_chunk_slices(support.shape, 512 if mp_config is None else mp_config.chunks))

    # 3/ Pair each chunk values with its intersecting groups

    def arguments() -> Any:
        """Yield matching raster blocks and the small set of geometries intersecting their footprints."""

        for index, (rows, columns) in enumerate(tiles):
            # Calculate the chunk's footprint to find intersecting polygons
            transform = support.transform * Affine.translation(columns.start, rows.start)
            height, width = rows.stop - rows.start, columns.stop - columns.start
            footprint = shapely.Polygon(
                [transform * corner for corner in ((0, 0), (width, 0), (width, height), (0, height))]
            )

            # Skip chunks outside named groups, but include all chunks for inside/outside groups
            selected = np.sort(tree.query(footprint, predicate="intersects"))
            if not include_outside and len(selected) == 0:
                continue
            group_numbers = np.arange(2) if include_outside else selected

            # Select the chunk's values and mask
            if use_dask:
                arrays = [value_blocks[index] for value_blocks in blocks]
                chunk_mask = mask_blocks[index]
            else:
                tile = (rows, columns)
                arrays = [
                    array.block(tile) if isinstance(array, _ValueReader) else array[tile] for array in values.values()
                ]
                chunk_mask = (
                    mask.block(tile) if isinstance(mask, _ValueReader) else None if mask is None else mask[tile]
                )
            yield (
                arrays,
                chunk_mask,
                transform,
                geometries[selected],
                group_numbers,
                include_outside,
                overlap_backend,
            )

    # 4/ Calculate chunk summaries and combine them with the selected backend

    # Merge fractional counts and sums across chunks
    empty_state = {name: np.empty((len(values), 0), dtype=np.float64) for name in ("count", "total", "sum")}
    summary = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), empty_state)
    if backend == "flox":
        # Stack sparse fractional summaries and let Flox combine matching groups across chunks
        channels = []
        codes = []
        module = da if use_dask else np
        for args in arguments():
            chunk_state = dask.delayed(_fractional_chunk_state)(*args) if use_dask else _fractional_chunk_state(*args)
            group_numbers = args[4]
            arrays = []
            for name in empty_state:
                array = chunk_state[2][name]
                if use_dask:
                    array = da.from_delayed(array, shape=(len(values), len(group_numbers)), dtype=np.float64)
                arrays.append(array)
            channels.append(module.stack(arrays, axis=1))
            codes.append(group_numbers)

        # Sum matching groups across chunks for every value and statistic
        if channels:
            reduced = _flox_groupby_reduce(
                module.concatenate(channels, axis=-1),
                [np.concatenate(codes)],
                [np.arange(len(labels))],
                function="sum",
                use_dask=use_dask,
                fill_value=0,
                dtype=np.float64,
                engine=engine,
            )
            reduced = reduced.compute() if use_dask else reduced

            # Match the group summary format used by the other backends
            state = {name: reduced[:, index] for index, name in enumerate(empty_state)}
            summary = (np.arange(len(labels)), np.zeros(len(labels), dtype=np.int64), state)
    elif use_dask:
        # Build Dask tasks for the fractional counts and sums
        tasks = [dask.delayed(_fractional_chunk_state)(*args) for args in arguments()]

        # Merge up to eight summaries per task
        while len(tasks) > 1:
            tasks = [dask.delayed(_merge_blocks)(tasks[start : start + 8]) for start in range(0, len(tasks), 8)]
        if tasks:
            summary = tasks[0].compute()
    elif mp_config is not None:
        # Merge each returned worker summary so finished results do not accumulate
        for _, state in _map_bounded(mp_config.cluster, _fractional_chunk_state, arguments()):
            summary = _merge_blocks([summary, state])
    else:
        # Calculate and merge one chunk at a time in the current process
        for args in arguments():
            summary = _merge_blocks([summary, _fractional_chunk_state(*args)])

    # 5/ Calculate the requested statistics in the declared group order

    # Restore declared group labels
    group_numbers, _, state = summary
    columns = {}
    observed_groups = np.zeros(len(labels), dtype=bool)
    for value_index, value_name in enumerate(values):
        count, total, sums = (np.zeros(len(labels)) for _ in range(3))
        count[group_numbers], total[group_numbers], sums[group_numbers] = (
            state["count"][value_index],
            state["total"][value_index],
            state["sum"][value_index],
        )
        observed_groups |= total > 0
        columns[(value_name, "count")] = count

        # Calculate fractional means and percentages after merging all chunks
        for name, alias in zip(statistics.names, statistics.aliases):
            if alias == "sum":
                result = np.where(count > 0, sums, np.nan)
            elif alias == "mean":
                result = np.divide(sums, count, out=np.full(len(labels), np.nan), where=count > 0)
            elif alias == "validcount":
                result = count
            elif alias == "totalcount":
                result = total
            else:
                result = np.divide(100 * count, total, out=np.full(len(labels), np.nan), where=total > 0)
            columns[(value_name, name)] = result

    # 6/ Select output groups and build the result table

    # Select groups with a positive fractional total count when observed=True
    selected = np.flatnonzero(observed_groups) if observed else np.arange(len(labels))
    table = pd.DataFrame({name: array[selected] for name, array in columns.items()}, index=labels.take(selected))
    table.columns = pd.MultiIndex.from_tuples(columns, names=["value", "statistic"])

    # Store the grouping and calculation options with the result
    table.attrs["grouped_stats"] = {
        "observed": observed,
        "subsample": 1,
        "subsample_per_group": False,
        "strategy": "fractional",
        "subsampling_strategy": "topk",
        "mask_membership": "fractional vector coverage",
        "reduction_backend": backend,
        "flox_engine": engine,
    }
    return table


def _fractional_vector_stats(
    source: Any,
    by: Mapping[str, Any],
    *,
    values: Any,
    support: RasterBase,
    statistics: _Statistics,
    mask: Any | None,
    definitions: Mapping[str, _GroupDefinition],
    observed: bool,
    overlap_backend: OverlapBackend,
    return_local_data: bool,
    backend: str = "geoutils",
    flox_engine: str | None = None,
    mp_config: MultiprocConfig | None = None,
) -> Any:
    """
    Calculate raster statistics using the exact fraction of each cell covered by a vector group.

    Internal function logic:
    _vector_fractional_groups() creates one combined geometry per label.
    _grid_intersection_fractions() calculates the cells and covered area for each geometry, and
    _grid_intersection_local_data() pairs those areas with raster values.
    """

    if not _is_raster(support):
        raise TypeError("Fractional vector statistics require a raster support grid.")
    named_values = dict(values) if isinstance(values, Mapping) else {"value": values}
    use_dask = any(is_dask_array(array) for array in named_values.values()) or is_dask_array(mask)
    if backend == "flox" or use_dask or mp_config is not None:
        if return_local_data:
            raise ValueError("Fractional uncertainty propagation requires in-memory GeoUtils statistics.")
        return _fractional_mergeable_stats(
            source,
            by,
            named_values,
            support,
            statistics,
            mask,
            definitions,
            observed,
            overlap_backend,
            backend,
            flox_engine,
            mp_config,
        )
    eligibility = None if mask is None else np.asarray(mask, dtype=bool)
    if eligibility is not None and eligibility.shape != support.shape:
        raise ValueError("The statistics mask must match the raster support shape.")

    # Calculate the covered cells once, then reuse them for every selected raster band and statistic
    _, labels, geometries, include_outside = _vector_fractional_groups(source, by, support, definitions)
    overlap = _grid_intersection_fractions(
        geometries,
        support.transform,
        support.shape,
        backend=overlap_backend,
    )
    if include_outside:
        overlap = _inside_outside_overlap(overlap, support.shape)
    reducers = _fractional_statistic_reducers(statistics)

    columns: dict[tuple[str, str], NDArray[Any]] = {}
    kept_inputs = []
    output_labels: list[Hashable] = []
    nominal_values = []
    observed_groups = np.zeros(overlap.geometry_count, dtype=bool)
    for value_index, (value_name, array) in enumerate(named_values.items()):
        values_array = np.asanyarray(array)
        if values_array.shape != support.shape:
            raise ValueError(f"Fractional statistic value {value_name!r} must match the raster support shape.")
        local_inputs = _grid_intersection_local_data(
            values_array,
            support.transform,
            overlap,
            source_id_offset=value_index * values_array.size,
            eligible=eligibility,
        )

        # The effective count is the sum of covered fractions for cells with valid raster values
        counts = np.asarray([Count().evaluate(local) for local in local_inputs], dtype=np.float64)
        total_counts = []
        for local in local_inputs:
            support_valid = np.ones(len(local.values), dtype=bool)
            if eligibility is not None:
                positions = np.asarray(local.source_ids % values_array.size, dtype=np.int64)
                support_valid = eligibility.reshape(-1)[positions]
            total_count = (
                float(np.sum(local.support_weights[support_valid])) if local.support_weights is not None else 0
            )
            total_counts.append(total_count)
        total_counts_array = np.asarray(total_counts, dtype=np.float64)
        columns[(value_name, "count")] = counts
        observed_groups |= total_counts_array > 0
        for statistic, statistic_name, alias, reducer in zip(
            statistics.requested,
            statistics.names,
            statistics.aliases,
            reducers,
        ):
            results = []
            statistic_inputs = []
            for local, valid_count, total_count in zip(local_inputs, counts, total_counts_array):
                if alias == "validcount":
                    result = valid_count
                elif alias == "totalcount":
                    result = total_count
                elif alias == "percentagevalidpoints":
                    result = 100 * valid_count / total_count if total_count else np.nan
                else:
                    assert reducer is not None
                    result = reducer.evaluate(local)
                    statistic_inputs.append(local)
                results.append(result)
            result_array = np.asarray(results, dtype=np.float64)
            columns[(value_name, statistic_name)] = result_array

            # Save the exact input cells only for the single Reducer used to calculate uncertainty
            if return_local_data:
                assert isinstance(statistic, Reducer)
                kept_inputs.extend(statistic_inputs)
                output_labels.extend((value_name, label) for label in labels)
                nominal_values.extend(result_array)

    # Remove groups with no covered cells after combining validity information from every raster value
    selected = np.arange(len(labels)) if not observed else np.flatnonzero(observed_groups)
    column_index = pd.MultiIndex.from_tuples(columns, names=["value", "statistic"])
    table = pd.DataFrame({column: values[selected] for column, values in columns.items()}, index=labels.take(selected))
    table.columns = column_index
    table.attrs["grouped_stats"] = {
        "observed": observed,
        "subsample": 1,
        "subsample_per_group": False,
        "strategy": "fractional",
        "subsampling_strategy": "topk",
        "mask_membership": "fractional vector coverage",
    }
    if not return_local_data:
        return table

    selected_inputs = []
    selected_labels = []
    selected_nominal = []
    for value_index in range(len(named_values)):
        start = value_index * len(labels)
        for group_index in selected:
            position = start + int(group_index)
            selected_inputs.append(kept_inputs[position])
            selected_labels.append(output_labels[position])
            selected_nominal.append(nominal_values[position])
    return table, selected_inputs, selected_labels, np.asarray(selected_nominal)
