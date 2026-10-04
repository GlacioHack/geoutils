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
from numpy.typing import NDArray

from geoutils._dispatch import _is_raster, _is_vector, is_dask_array
from geoutils.operators.overlap import (
    GridIntersection,
    OverlapBackend,
    _grid_intersection_fractions,
    _grid_intersection_local_data,
    _union_geometries,
    _union_geometries_by_label,
)
from geoutils.operators.reducer import Count, Reducer
from geoutils.sampling.support import _sampling_specification
from geoutils.stats.reduction import _reducer_from_statistic
from geoutils.vector.base import _as_geodataframe

if TYPE_CHECKING:
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
) -> Any:
    """Calculate raster statistics using the exact fraction of each cell covered by a vector group.

    _vector_fractional_groups() creates one combined geometry per label. _grid_intersection_fractions() calculates
    the cells and covered area for each geometry, and _grid_intersection_local_data() pairs those areas with raster
    values. Named statistics use the same Reducers as ordinary grouped statistics.
    """

    if not _is_raster(support):
        raise TypeError("Fractional vector statistics require a raster support grid.")
    named_values = dict(values) if isinstance(values, Mapping) else {"value": values}
    if any(is_dask_array(array) for array in named_values.values()) or is_dask_array(mask):
        raise ValueError("Fractional vector statistics currently require eager raster values.")
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
