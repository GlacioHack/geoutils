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

"""Reduce arrays to descriptive statistics, globally or by group."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from sys import modules
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import iqr
from scipy.stats.mstats import mquantiles

from geoutils._dispatch import is_dask_array
from geoutils._misc import import_optional
from geoutils._typing import NDArrayNum
from geoutils.multiproc.readers import _ValueReader
from geoutils.operators.base import LocalData
from geoutils.operators.reducer import (
    Count,
    Maximum,
    Mean,
    Median,
    Minimum,
    Quantile,
    Reducer,
    RootMeanSquare,
    StandardDeviation,
    Sum,
    _can_reduce_arrays,
)
from geoutils.raster.array import get_mask_from_array
from geoutils.stats.estimators import linear_error, nmad, rmse, sum_square

if TYPE_CHECKING:
    from flox import Aggregation

    from geoutils.multiproc import MultiprocConfig
    from geoutils.stats.selection import _SelectionCounts

########################################
# 1/ STATISTIC DEFINITIONS AND VALIDATION
########################################

# We list the statistics that can be merged efficiently from aggregates computed in separate blocks
# For instance: a mean can be simply a sum divided by count, both that can be aggregated over many group-chunk
# intersections without requiring to load a whole group at once
# However, some statistics cannot aggregate like this and require complete groups, those are: median, percentiles, IQR,
# LE90, NMAD, and custom functions
MERGEABLE_STATISTICS = {
    "validcount",
    "totalcount",
    "percentagevalidpoints",
    "sum",
    "mean",
    "std",
    "min",
    "max",
    "sumofsquares",
    "rmse",
}
_TDIGEST_STATISTICS = {"median", "90thpercentile", "iqr", "le90", "nmad"}

# List aliases for statistics, total counts, and inlier counts
_STATS_ALIAS_CALLABLE = {
    "mean": "Mean",
    "median": "Median",
    "max": "Max",
    "min": "Min",
    "sum": "Sum",
    "sumofsquares": "Sum of squares",
    "90thpercentile": "90th percentile",
    "iqr": "IQR",
    "le90": "LE90",
    "nmad": "NMAD",
    "rmse": "RMSE",
    "std": "Standard deviation",
}
_STATS_ALIAS_COUNTS = {
    "validcount": "Valid count",
    "totalcount": "Total count",
    "percentagevalidpoints": "Percentage valid points",
}
_STATS_ALIAS_GEN = _STATS_ALIAS_CALLABLE | _STATS_ALIAS_COUNTS
_STATS_ALIAS_MASK = {
    "validinliercount": "Valid inlier count",
    "totalinliercount": "Total inlier count",
    "percentagevalidinlierpoints": "Percentage valid inlier points",
    "percentageinlierpoints": "Percentage inlier points",
}
# List of all statistics when user input is "all"
_STATS_ALIAS_ALL = _STATS_ALIAS_GEN | _STATS_ALIAS_MASK

# Allow synonyms
_SYNONYMS = {
    "maximum": "max",
    "minimum": "min",
    "sum2": "sumofsquares",
    "90percentile": "90thpercentile",
    "rms": "rmse",
    "standarddeviation": "std",
}

# List of defaults statistics used when no user input is passed
_STATS_LIST_MIN = [
    "min",
    "max",
    "mean",
    "std",
    "validcount",
    "totalcount",
    "percentagevalidpoints",
]


class _SumOfSquares(Reducer):
    """Compute a sum of squared values with optional data and area or length weights."""

    accepts_sample_weights = True
    accepts_support_weights = True

    def reduce(self, data: LocalData) -> float:
        """Multiply squared values by the same combined weights as Sum(), then add them."""

        weights = Sum().coefficients(data).weights
        return float(np.dot(weights, np.square(data.values)))


class _QuantileRange(Reducer):
    """Compute the difference between two ordinary or coverage-weighted quantiles."""

    def __init__(self, lower: float, upper: float, *, weighted: bool) -> None:
        """Store the lower and upper probabilities and whether the quantiles use weights."""

        self.lower = Quantile(lower, weighted=weighted)
        self.upper = Quantile(upper, weighted=weighted)
        self.accepts_sample_weights = weighted
        self.accepts_support_weights = weighted

    def reduce(self, data: LocalData) -> float:
        """Return the upper quantile minus the lower quantile."""

        return self.upper.reduce(data) - self.lower.reduce(data)


class _NormalizedMedianAbsoluteDeviation(Reducer):
    """Compute normalized median absolute deviation with optional coverage weights."""

    def __init__(self, *, weighted: bool, factor: float = 1.4826) -> None:
        """Store the median definition and conventional normal-distribution scale factor."""

        self.median = Median(weighted=weighted)
        self.factor = factor
        self.accepts_sample_weights = weighted
        self.accepts_support_weights = weighted

    def reduce(self, data: LocalData) -> float:
        """Return the scaled median of absolute deviations from the same weighted median."""

        center = self.median.reduce(data)
        deviations = replace(data, values=np.abs(data.values - center))
        return self.factor * self.median.reduce(deviations)


# Use the standard Reducers when they calculate the same result as an existing statistic
_REDUCER_STATISTIC_ALIASES: dict[type[Reducer], str] = {
    Mean: "mean",
    Median: "median",
    Maximum: "max",
    Minimum: "min",
    StandardDeviation: "std",
    Sum: "sum",
    Count: "validcount",
    RootMeanSquare: "rmse",
}


def _reducer_from_statistic(alias: str, *, fractional: bool = False) -> Reducer | None:
    """Return the Reducer that calculates a named internal statistic, when one exists."""

    factories: dict[str, Callable[[], Reducer]] = {
        "mean": Mean,
        "median": lambda: Median(weighted=fractional),
        "max": Maximum,
        "min": Minimum,
        "sum": Sum,
        "sumofsquares": _SumOfSquares,
        "std": StandardDeviation,
        "rmse": RootMeanSquare,
        "validcount": Count,
        "90thpercentile": lambda: Quantile(0.9, weighted=fractional),
        "iqr": lambda: _QuantileRange(0.25, 0.75, weighted=fractional),
        "le90": lambda: _QuantileRange(0.05, 0.95, weighted=fractional),
        "nmad": lambda: _NormalizedMedianAbsoluteDeviation(weighted=fractional),
    }
    factory = factories.get(alias)
    return None if factory is None else factory()


@dataclass(frozen=True)
class _Statistics:
    """
    Store requested statistics, output names and validated internal aliases.

    :param requested: Statistic names or callables.
    :param names: Matching output column names.
    :param aliases: Matching internal statistic names (None for callables).
    """

    requested: list[str | Callable[[Any], Any] | Reducer]
    names: list[str]
    aliases: list[str | None]
    grouped: bool = True
    single: bool = False

    @property
    def output_names(self) -> list[str]:
        """Return requested labels with the mandatory count for grouped output."""

        return ["count", *self.names] if self.grouped else self.names


def _get_stat_common_alias(stat_name: str) -> str | None:
    """Return the internal statistic name for a user alias."""

    # Ignore case, spaces and underscores in names such as "standard deviation"
    normalized_name = "".join(stat_name.lower().replace("_", "").split())
    normalized_name = _SYNONYMS.get(normalized_name, normalized_name)
    return normalized_name if normalized_name in _STATS_ALIAS_ALL else None


def _normalize_statistics(
    statistics: str | Callable[[Any], Any] | Reducer | Iterable[str | Callable[[Any], Any] | Reducer] | None,
    *,
    grouped: bool = True,
    masked: bool = False,
) -> _Statistics:
    """
    Validate statistics and choose their result names and internal aliases.

    See stats() for description of ``statistics`` argument.

    :param grouped: Include the grouped count column, reject unknown names and omit summary-only mask counts.
    :param masked: Include mask-specific counts in the complete global selection requested by ``"all"``.
    :returns: A _Statistics object separating requested statistics, output names and internal aliases.
    """

    # 1/ Expand the user request into the statistics to calculate
    # Keep the established default selection and add inlier counts only when a mask was used
    default_names = statistics is None or isinstance(statistics, str) and statistics == "all"
    if statistics is None:
        requested: list[str | Callable[[Any], Any] | Reducer] = list(_STATS_LIST_MIN)
    elif isinstance(statistics, str) and statistics == "all":
        requested = list(_STATS_ALIAS_GEN)
        if masked and not grouped:
            requested += list(_STATS_ALIAS_MASK)
    elif isinstance(statistics, (str, Reducer)) or callable(statistics):
        requested = [statistics]
    else:
        requested = list(statistics)

    # Grouped results always include a leading count column
    if grouped:
        requested = [statistic for statistic in requested if statistic != "count"]
        if default_names:
            requested = [statistic for statistic in requested if statistic != "validcount"]
        if requested == ["all"]:
            requested = [*_STATS_ALIAS_CALLABLE, "totalcount", "percentagevalidpoints"]
        elif "all" in requested:
            raise ValueError("Statistic 'all' cannot be combined with other statistics.")

    # 2/ Separate output labels from the internal names used by the estimators
    names = []
    for statistic in requested:
        if isinstance(statistic, str):
            names.append(statistic)
        elif isinstance(statistic, Reducer):
            names.append(type(statistic).__name__)
        elif callable(statistic):
            function = statistic
            while isinstance(function, partial):
                function = function.func
            names.append(getattr(function, "__name__", type(function).__name__))
        else:
            raise TypeError("Argument ``statistics`` must contain names or callable functions, including Reducers.")
    if default_names and not grouped:
        names = [_STATS_ALIAS_ALL[name] for name in names]
    aliases = [
        _get_stat_common_alias(statistic)
        if isinstance(statistic, str)
        else _REDUCER_STATISTIC_ALIASES.get(type(statistic))
        if isinstance(statistic, Reducer)
        else None
        for statistic in requested
    ]

    # 3/ Check user names before calculation, raising warning for unknown ones and skipping them
    if len(set(names)) != len(names):
        raise ValueError("Statistic names must be unique.")
    if grouped and "count" in names:
        raise ValueError("Statistic name 'count' is reserved for the mandatory grouped count.")
    allowed = _STATS_ALIAS_GEN if grouped else _STATS_ALIAS_ALL
    unknown = [
        statistic for statistic, alias in zip(requested, aliases) if isinstance(statistic, str) and alias not in allowed
    ]
    if unknown and grouped:
        raise ValueError(f"Unknown statistic names: {unknown!r}.")
    for statistic in unknown:
        warnings.warn(f"Statistic name {statistic} is not recognized", category=UserWarning)
    single = (
        isinstance(statistics, Reducer) or callable(statistics) or isinstance(statistics, str) and statistics != "all"
    )
    return _Statistics(requested=requested, names=names, aliases=aliases, grouped=grouped, single=single)


def _check_input_reduction_method(
    backend: str, method: str, reindex: bool | None, split_every: int | None = None
) -> None:
    """Check reduction argument inputs."""

    # Check method names first
    if method not in {"auto", "map-reduce", "blockwise", "cohorts", "groupwise"}:
        raise ValueError("Argument ``method`` must be 'auto', 'map-reduce', 'blockwise', 'cohorts' or 'groupwise'.")
    if reindex is not None and not isinstance(reindex, (bool, np.bool_)):
        raise TypeError("Argument ``reindex`` must be True, False or None.")
    if split_every is not None:
        if isinstance(split_every, (bool, np.bool_)) or not isinstance(split_every, (int, np.integer)):
            raise TypeError("Argument ``split_every`` must be an integer of at least 2 or None.")
        if split_every < 2:
            raise ValueError("Argument ``split_every`` must be at least 2.")

    # Then check support depending on backend
    if backend == "geoutils" and method in {"blockwise", "cohorts"}:
        raise ValueError(f"Argument method={method!r} requires backend='flox'.")
    if backend == "flox" and method == "groupwise":
        raise ValueError("Argument method='groupwise' requires backend='geoutils'.")
    if method == "groupwise" and reindex is not None:
        raise ValueError("Argument ``reindex`` applies to group summaries, not method='groupwise'.")
    if method == "groupwise" and split_every is not None:
        raise ValueError("Argument ``split_every`` applies to group summaries, not method='groupwise'.")
    if method == "cohorts" and reindex:
        raise ValueError("Argument method='cohorts' requires reindex=False or None.")


def _resolve_reduction_method(
    aliases: Sequence[str | None],
    total_groups: int,
    method: str = "auto",
    reindex: bool | None = None,
) -> tuple[str, bool | None, bool]:
    """
    Choose chunk method for the requested statistics.

    See _reduce_values() for ``method`` and ``total_groups`` arguments, and ``aliases`` are from _Statistics.

    :returns: The method, reindex setting and whether every requested statistic can be merged from block summaries.
    """

    _check_input_reduction_method("geoutils", method, reindex)

    # Get names of all mergeable stats
    mergeable_statistics = MERGEABLE_STATISTICS | _TDIGEST_STATISTICS
    mergeable = all(alias in mergeable_statistics or alias in _STATS_ALIAS_MASK for alias in aliases)

    # Resolve method and reindex (mostly depending on "auto", otherwise explicit setting from user input if possible)
    if method == "map-reduce" or reindex is not None:
        if not mergeable:
            raise ValueError("Custom statistics require method='groupwise' or 'auto'.")

    if method == "auto":
        method = "map-reduce" if mergeable else "groupwise"
    if method == "map-reduce":
        reindex = total_groups <= 4096 if reindex is None else bool(reindex)
    return method, reindex, mergeable


#####################################################
# 2/ T-DIGEST REDUCTION (FOR QUANTILE-TYPE REDUCERS)
#####################################################


def _check_input_tdigest(compression: float) -> None:
    """Validate t-digest user inputs."""

    if not np.isfinite(compression) or not 20 <= compression <= 1000:
        raise ValueError("Argument tdigest_compression must be finite and between 20 and 1000.")


def _digest_chunk(
    group_idx: NDArray[Any],
    array: NDArray[Any],
    *,
    size: int,
    dtype: Any,
    fill_value: Any,
    compression: float,
) -> NDArray[Any]:
    """Build a t-digest for each group."""

    crick = import_optional("crick", extra_name="num")
    result = np.full(size, None, dtype=object)

    # Sort samples by group
    order = np.argsort(group_idx, kind="stable")
    groups, starts = np.unique(group_idx[order], return_index=True)
    stops = np.r_[starts[1:], len(order)]

    # Build each t-digest from finite samples
    for group, start, stop in zip(groups, starts, stops):
        values = array[order[start:stop]]
        values = values[np.isfinite(values)]
        if len(values):
            digest = crick.TDigest(compression)
            digest.update(values.astype(np.float64, copy=False))
            result[group] = digest
    return result


def _digest_merge(
    group_idx: NDArray[Any],
    array: NDArray[Any] | None = None,
    *,
    size: int | None = None,
    dtype: Any = None,
    fill_value: Any = None,
    axis: int = -1,
    keepdims: bool = False,
) -> NDArray[Any]:
    """Merge t-digests for each group."""

    crick = import_optional("crick", extra_name="num")
    selections: Iterator[tuple[int | tuple[int, ...], Any]]
    if array is None:
        # Align chunks along the last axis
        values = np.moveaxis(group_idx, axis, -1)
        result = np.full(values.shape[:-1], None, dtype=object)
        selections = ((index, values[index]) for index in np.ndindex(result.shape))
    else:
        # Select t-digests by group code
        assert size is not None
        result = np.full(size, None, dtype=object)
        selections = ((int(group), array[group_idx == group]) for group in np.unique(group_idx))

    # Create new t-digests because Dask tasks may share the inputs
    for index, candidates in selections:
        digests = [item for item in candidates if isinstance(item, crick.TDigest)]
        if digests:
            digest = crick.TDigest(digests[0].compression)
            digest.merge(*digests)
            result[index] = digest
    return np.expand_dims(result, axis) if array is None and keepdims else result


def _digest_quantile(array: NDArray[Any], *, q: float) -> NDArray[Any]:
    """Calculate each group's quantile from its t-digest."""

    crick = import_optional("crick", extra_name="num")
    result = np.full(array.shape, np.nan, dtype=np.float64)
    for index in np.ndindex(array.shape):
        if isinstance(array[index], crick.TDigest):
            result[index] = array[index].quantile(q)
    return result


#########################
# 3/ COMPLETE VALUES AND GROUPS
#########################


def _reduce_complete_values(data: NDArrayNum, aliases: set[str]) -> tuple[dict[str, Any], int]:
    """
    Evaluate complete statistics with NumPy's masked or NaN-aware functions.

    We preserve input type when calling estimators: masked arrays use NumPy Masked functions, while
    ordinary arrays use functions that ignore NaNs.

    We also return the validity count, total count and percentages.

    See _reduce_values() for ``data`` argument description.

    :param aliases: Internal names of the numerical statistics to calculate, excluding count statistics.

    :returns: Estimates keyed by their internal names and the input validity count.
    """

    # Keep the existing distinction between masked values and NaNs in ordinary arrays
    masked = np.ma.isMaskedArray(data)
    if masked:
        # Treat NaNs as missing
        raw = np.ma.getdata(data)
        data = np.ma.array(raw, mask=np.ma.getmaskarray(data) | np.isnan(raw), copy=False)
    final_count = int(np.count_nonzero(~np.ma.getmaskarray(data) if masked else np.isfinite(data)))
    module = np.ma if masked else np
    prefix = "" if masked else "nan"
    functions = {name: getattr(module, prefix + name) for name in ("mean", "median", "max", "min", "sum", "std")}

    # Use the existing estimators and matching percentile definitions for masked arrays
    functions.update(
        sumofsquares=sum_square,
        **{
            "90thpercentile": (
                (lambda array: mquantiles(array, prob=0.9, alphap=1, betap=1)[0])
                if masked
                else partial(np.nanpercentile, q=90)
            )
        },
        le90=partial(linear_error, interval=90),
        iqr=partial(iqr, nan_policy="omit"),
        nmad=nmad,
        rmse=rmse,
    )

    # Report empty inputs consistently and avoid calling estimators that require at least one valid value
    if final_count == 0:
        warnings.warn("Empty raster, returns NaN for all stats", category=UserWarning)
    result = {alias: functions[alias](data) if final_count else np.nan for alias in aliases}
    return result, final_count


def _extrema_result(values: Sequence[Any] | NDArray[Any]) -> Any:
    """Keep integer extrema beyond floating-point precision exact, including missing group results."""

    present = [value for value in values if not pd.isna(value)]
    if present and all(isinstance(value, (int, np.integer)) for value in present):
        if any(abs(int(value)) > 2**53 for value in present):
            dtype = "UInt64" if any(value > np.iinfo(np.int64).max for value in present) else "Int64"
            return pd.array(values, dtype=dtype)
    return np.asarray(values, dtype=float)


def _aggregate_complete_groups(
    values: Sequence[NDArray[Any]],
    group_ids: NDArray[Any] | None,
    statistics: _Statistics,
) -> pd.DataFrame:
    """
    Calculate every statistic from complete in-memory groups after one stable sort.

    Locations are sorted once by group ID, placing all values from each group in one slice. Statistics that need the
    complete group, such as quantiles and user functions, can then use those slices without filtering the full array
    separately for every group. The stable sort keeps the original order within each group. With no group IDs,
    use the complete input directly and avoid allocating or sorting an artificial group array.

    See _reduce_values() for ``values`` and ``group_ids`` descriptions, and see
    _aggregate_complete_groups_chunked() for ``statistics``.

    :returns: A table with integer group rows and (value position, statistic name) columns.
    """

    # 1/ Locate complete groups while keeping the original order within each group

    # Use one direct slice for a full-array reduction, or sort grouped locations once by their group ID
    global_reduction = group_ids is None
    if global_reduction:
        order: slice | NDArray[Any] = slice(None)
        labels = np.array([0], dtype=np.int64)
        starts = np.array([0], dtype=np.int64)
        sizes = np.array([values[0].size], dtype=np.int64)
    else:
        ids = np.asarray(group_ids).ravel()
        selected = np.flatnonzero(ids >= 0)
        order = selected[np.argsort(ids[selected], kind="stable")]
        labels, starts, sizes = np.unique(ids[order], return_index=True, return_counts=True)

    # 2/ Calculate the requested estimators and counts for each selected value

    # Reuse the validated names for every group instead of parsing the same request again
    aliases = {alias for alias in statistics.aliases if alias in _STATS_ALIAS_CALLABLE}
    columns: dict[tuple[int, str], Any] = {}
    for value_index, array in enumerate(values):
        # Share the group ordering but handle missing observations independently for each selected value
        ordered = np.asanyarray(array) if global_reduction else np.asanyarray(array).ravel()[order]
        if not global_reduction:
            invalid = get_mask_from_array(ordered).reshape(ordered.shape)
            if np.any(invalid):
                if np.issubdtype(ordered.dtype, np.integer):
                    ordered = np.ma.array(np.ma.getdata(ordered), mask=invalid)
                else:
                    ordered = np.where(invalid, np.nan, np.ma.getdata(ordered))
        results: dict[str, list[Any]] = {"count": [], **{name: [] for name in statistics.names}}

        # Pass complete group values to statistics such as median and user functions
        for start, size in zip(starts, sizes):
            group = ordered if global_reduction else ordered[start : start + size]
            if global_reduction:
                source_ids = np.arange(group.size)
            else:
                assert isinstance(order, np.ndarray)
                source_ids = np.asarray(order[start : start + size])
            count = int(np.count_nonzero(~get_mask_from_array(group)))
            if global_reduction:
                # Preserve NumPy's established global behavior for unmasked infinities
                computed, count = _reduce_complete_values(group, aliases)
            else:
                computed, _ = _reduce_complete_values(group, aliases) if count and aliases else ({}, count)
            results["count"].append(count)

            # Distinguish finite values from all group locations when constructing counts and percentages
            for statistic, name, alias in zip(statistics.requested, statistics.names, statistics.aliases):
                result: Any
                if alias == "validcount":
                    result = count
                elif alias == "totalcount":
                    result = int(size)
                elif alias == "percentagevalidpoints":
                    result = 100 * count / size if size else np.nan
                elif isinstance(statistic, Reducer) and alias is None:
                    group_values = np.asanyarray(np.ma.getdata(group)).reshape(-1)
                    valid = ~get_mask_from_array(group).reshape(-1)
                    local = LocalData(values=group_values, valid=valid, source_ids=source_ids)
                    result = statistic.evaluate(local) if count else np.nan
                elif callable(statistic):
                    # Give user functions the complete group, with missing observations still present
                    callable_values = (
                        np.where(np.ma.getmaskarray(group), np.nan, np.ma.getdata(group))
                        if np.ma.isMaskedArray(group) and np.any(np.ma.getmaskarray(group))
                        else group
                    )
                    result = statistic(callable_values) if count else np.nan
                else:
                    result = computed.get(alias or "", np.nan)
                results[name].append(result)

        # Keep counts as integers and preserve extrema beyond floating-point precision when needed
        extrema_names = {name for name, alias in zip(statistics.names, statistics.aliases) if alias in {"min", "max"}}
        count_names = {
            name for name, alias in zip(statistics.names, statistics.aliases) if alias in {"validcount", "totalcount"}
        }
        for name, result_values in results.items():
            if name in extrema_names:
                columns[(value_index, name)] = _extrema_result(result_values)
            else:
                dtype = np.int64 if name == "count" or name in count_names else float
                columns[(value_index, name)] = np.asarray(result_values, dtype=dtype)

    # 3/ Assemble the shared table format used by in-memory and chunked reductions
    return pd.DataFrame(columns, index=labels)


################################################
# 4/ CHUNKED REDUCTION (DASK AND MULTIPROCESSING)
################################################


def _reader_block_requires_complete_values(block: Any) -> bool:
    """Identify values for which global NumPy semantics differ from finite grouped reductions."""

    from geoutils.multiproc.readers import _read_values

    values = _read_values(block)
    nonfinite = ~np.isfinite(values) if np.ma.isMaskedArray(values) else np.isinf(values)
    return bool(np.ma.filled(np.any(nonfinite), False))


def _reader_requires_complete_values(array: _ValueReader, mp_config: MultiprocConfig) -> bool:
    """Check exceptional nonfinite values in bounded reads before choosing global estimators."""

    from geoutils.multiproc.chunked import iter_chunk_slices
    from geoutils.multiproc.cluster import _map_bounded

    arguments = ((array.block(tile),) for tile in iter_chunk_slices(array.shape, mp_config.chunks))
    return any(
        result for _, result in _map_bounded(mp_config.cluster, _reader_block_requires_complete_values, arguments)
    )


def _reduce_block(
    values: Sequence[NDArray[Any]],
    group_ids: NDArray[Any] | None,
    total_groups: int,
    reindex: bool,
    statistics: set[str] | Sequence[set[str]],
    tdigest_compression: float = 100,
    centers: tuple[NDArray[Any], NDArray[Any]] | None = None,
) -> tuple[NDArray[Any], NDArray[Any], dict[str, NDArray[Any]]]:
    """
    Summarize groups in one array block so summaries from many blocks can be combined.

    A dense summary reserves one position for every declared group; a sparse summary stores only groups present in the
    block. Each selected value has its own finite count, while group sizes include missing selected values. For standard
    deviation, the summary stores the mean and sum of squared deviations needed by the pairwise merge. Stored quantities
    have shape (number of selected values, number of stored groups), and only requested quantities are allocated.

    Values and group_ids follow _reduce_values(), restricted to this block. Total_groups still includes every
    declared group across the full input.

    :param reindex: Reserve every declared group when True; otherwise store only groups present in the block.
    :param statistics: Internal names of the requested mergeable statistics, including any count statistics.
    :returns: Stored integer group labels, their total location counts, and named summary arrays with one row per
        selected value and one column per stored group. These are the summaries consumed by _merge_blocks().
    """

    from geoutils.multiproc.readers import _read_values

    # Read related Dask arrays together so worker slices share their source reads
    if any(is_dask_array(array) for array in [*values, group_ids]):
        *values, group_ids = import_optional("dask").compute(*values, group_ids, scheduler="synchronous")

    # Read file descriptors inside the worker while leaving supplied arrays unchanged
    values = [_read_values(value) for value in values]
    requested = [statistics] * len(values) if isinstance(statistics, set) else statistics
    statistics = set().union(*requested)

    # 1/ Number the groups represented in this block

    # Use one group for a full-array block, or store the requested dense or sparse grouped labels
    if group_ids is None:
        eligible = None
        labels = np.array([0], dtype=np.int64)
        codes = None
        size = np.array([values[0].size], dtype=np.int64)
    else:
        ids = np.asarray(group_ids).ravel()
        eligible = ids >= 0

        # Dense summaries use the declared numbers directly; sparse summaries give present groups consecutive slots
        if reindex:
            labels = np.arange(total_groups, dtype=np.int64)
            codes = ids[eligible].astype(np.int64)
        else:
            labels, codes = np.unique(ids[eligible], return_inverse=True)
        size = np.bincount(codes, minlength=len(labels))
    shape = (len(values), len(labels))
    state = {"count": np.zeros(shape, dtype=np.int64)}
    if "digest" in statistics:
        state["digest"] = np.full(shape, None, dtype=object)

    # 2/ Create only the summary arrays needed by the requested statistics
    needs_mean = bool(statistics & {"mean", "std"})
    if needs_mean:
        state["mean"] = np.zeros(shape, dtype=float)
    if "std" in statistics:
        state["m2"] = np.zeros(shape, dtype=float)
    summaries = statistics & {"sum", "sumofsquares", "min", "max"}
    if "rmse" in statistics:
        summaries.add("sumofsquares")

    # Empty groups start with neutral values so missing observations cannot change a sum, minimum, or maximum
    for name in summaries:
        fill = np.inf if name == "min" else -np.inf if name == "max" else 0.0
        integer_extrema = name in {"min", "max"} and any(
            np.issubdtype(array.dtype, np.integer) and array.dtype.itemsize >= 8 for array in values
        )
        state[name] = np.full(shape, fill, dtype=object if integer_extrema else float)

    # Skip blocks with no eligible groups
    if not len(labels):
        return labels, size, state

    # 3/ Reuse group IDs while handling missing data separately for each value array
    for index, array in enumerate(values):
        selected_statistics = requested[index]
        data = np.ma.getdata(array).ravel()
        finite = ~get_mask_from_array(array).ravel()
        if eligible is not None:
            data = data[eligible]
            finite = finite[eligible]
        valid_codes = None if codes is None else codes[finite]

        # Count finite values in each group, using a direct count for whole-array summaries
        count = (
            np.array([np.count_nonzero(finite)], dtype=np.int64)
            if valid_codes is None
            else np.bincount(valid_codes, minlength=len(labels))
        )
        state["count"][index] = count

        # Counts need only the finite-value mask; prepare numerical values only for requested estimates
        if not selected_statistics & {"digest", "mean", "std", "sum", "sumofsquares", "rmse", "min", "max"}:
            continue
        valid_data = data[finite]
        if "digest" in selected_statistics:
            digest_codes = np.zeros(len(valid_data), dtype=np.int64) if valid_codes is None else valid_codes
            digest_values = valid_data
            if centers is not None:
                # Use group medians from all blocks for NMAD
                positions = np.searchsorted(centers[0], labels[digest_codes])
                digest_values = np.abs(valid_data - centers[1][index, positions])
        if "digest" in selected_statistics:
            state["digest"][index] = _digest_chunk(
                digest_codes,
                digest_values,
                size=len(labels),
                dtype=object,
                fill_value=None,
                compression=tdigest_compression,
            )
        needs_float = bool(selected_statistics & {"mean", "std", "sum", "sumofsquares", "rmse"})
        data = valid_data.astype(float, copy=False) if needs_float else valid_data

        # Measure spread around each local mean to keep small variation accurate beside large values
        needs_mean = bool(selected_statistics & {"mean", "std"})
        if needs_mean or "sum" in selected_statistics:
            sums = (
                np.array([np.sum(data)], dtype=float)
                if valid_codes is None
                else np.bincount(valid_codes, weights=data, minlength=len(labels))
            )
            if "sum" in selected_statistics:
                state["sum"][index] = sums
            if needs_mean:
                mean = np.divide(sums, count, out=np.zeros(len(labels)), where=count > 0)
                state["mean"][index] = mean

            # Store squared deviations from the local mean; the merge later corrects for differences between means
            if "std" in selected_statistics:
                deviations = data - (mean[0] if valid_codes is None else mean[valid_codes])
                state["m2"][index] = (
                    np.array([np.sum(deviations**2)], dtype=float)
                    if valid_codes is None
                    else np.bincount(valid_codes, weights=deviations**2, minlength=len(labels))
                )

        # Keep uncentered squared values for sum of squares and RMSE, which measure magnitude rather than spread
        if selected_statistics & {"sumofsquares", "rmse"}:
            state["sumofsquares"][index] = (
                np.array([np.sum(data**2)], dtype=float)
                if valid_codes is None
                else np.bincount(valid_codes, weights=data**2, minlength=len(labels))
            )

        # Leave empty groups at their neutral extrema; finalization turns their estimates into NaN
        if data.size:
            if "min" in selected_statistics:
                if valid_codes is None:
                    state["min"][index, 0] = np.min(valid_data)
                else:
                    np.minimum.at(state["min"][index], valid_codes, valid_data)
            if "max" in selected_statistics:
                if valid_codes is None:
                    state["max"][index, 0] = np.max(valid_data)
                else:
                    np.maximum.at(state["max"][index], valid_codes, valid_data)
    return labels, size, state


def _merge_blocks(
    summaries: Sequence[tuple[NDArray[Any], NDArray[Any], dict[str, NDArray[Any]]]],
) -> tuple[NDArray[Any], NDArray[Any], dict[str, NDArray[Any]]]:
    """
    Combine dense or sparse block summaries with stable pairwise means and variances.

    Sparse group labels are aligned before counts, sums, minima, maxima, means and squared deviations are combined. The
    mean and variance update follows Chan, Golub and LeVeque (1983), avoiding subtraction of large squared sums.
    Group sizes include all eligible locations, while the counts used to weight means include only finite values.

    :param summaries: Nonempty sequence of block summaries in the format returned by _reduce_block().
    :returns: One summary in that same format, containing the union of the input groups.
    """

    # 1/ Align group IDs when each block stored only the groups it contained
    first_labels, _, first_state = summaries[0]
    same_labels = all(np.array_equal(summary[0], first_labels) for summary in summaries[1:])
    labels = first_labels if same_labels else np.unique(np.concatenate([summary[0] for summary in summaries]))
    size = np.zeros(len(labels), dtype=np.int64)
    shape = (first_state["count"].shape[0], len(labels))

    # 2/ Allocate outputs
    combined: dict[str, NDArray[Any]] = {}
    for name in first_state:
        fill = np.inf if name == "min" else -np.inf if name == "max" else None if name == "digest" else 0
        combined[name] = np.full(shape, fill, dtype=first_state[name].dtype)

    # 3/ Merge each block statistics
    for block_labels, block_size, state in summaries:
        # Avoid label indexing when all summaries already use the same group order
        positions = slice(None) if same_labels else np.searchsorted(labels, block_labels)
        size[positions] += block_size
        old_count = combined["count"][:, positions]
        new_count = old_count + state["count"]

        # Weight the mean shift by the fraction of finite values contributed by the incoming block
        if "mean" in combined:
            old_mean = combined["mean"][:, positions]
            delta = state["mean"] - old_mean
            fraction = np.divide(state["count"], new_count, out=np.zeros_like(delta), where=new_count > 0)
            combined["mean"][:, positions] = old_mean + delta * fraction

            # Correct the summed squared deviations for the difference between the two block means
            if "m2" in combined:
                correction = delta**2 * old_count * fraction
                combined["m2"][:, positions] += state["m2"] + correction
        combined["count"][:, positions] = new_count

        # Combine sums and ranges without keeping the original values
        for name in state.keys() - {"count", "mean", "m2"}:
            current = combined[name][:, positions]
            if name == "min":
                current = np.minimum(current, state[name])
            elif name == "max":
                current = np.maximum(current, state[name])
            elif name == "digest":
                current = _digest_merge(np.stack((current, state[name]), axis=-1))
            else:
                current = current + state[name]
            combined[name][:, positions] = current
    return labels, size, combined


def _finalize_blocks(
    summary: tuple[NDArray[Any], NDArray[Any], dict[str, NDArray[Any]]],
    statistics: _Statistics,
    *,
    global_reduction: bool = False,
) -> pd.DataFrame:
    """
    Turn combined block summaries into the same result columns as the in-memory path.

    Take the square root of the mean squared deviation for population standard deviation, or the mean squared value
    for RMSE. Groups with locations but no finite selected values keep their counts and receive NaN estimates.

    Statistics is the normalized request described in _aggregate_complete_groups_chunked().

    :param summary: Combined labels, total location counts and summary arrays returned by _merge_blocks().
    :returns: A table with integer group rows and (value position, statistic name) columns.
    """

    # Keep groups that contain locations even when every selected value is missing
    labels, size, state = summary
    observed = (size > 0) | global_reduction
    columns: dict[tuple[int, str], Any] = {}

    # Convert each selected value's summaries independently because their finite counts can differ
    for index in range(state["count"].shape[0]):
        count = state["count"][index]
        columns[(index, "count")] = count[observed]
        for name, alias in zip(statistics.names, statistics.aliases):
            if alias == "validcount":
                result = count
            elif alias == "totalcount":
                result = size
            elif alias == "percentagevalidpoints":
                result = np.divide(100 * count, size, out=np.full(len(size), np.nan), where=size > 0)

            # Normalize by the population count, keeping empty groups undefined instead of dividing by zero
            elif alias in {"std", "rmse"} and alias not in state:
                numerator = state["m2" if alias == "std" else "sumofsquares"][index]
                result = np.sqrt(np.divide(numerator, count, out=np.full(len(size), np.nan), where=count > 0))
            elif alias in _TDIGEST_STATISTICS and "digest" in state:
                digests = state["digest"][index]
                if alias == "nmad":
                    result = state["nmad"][index]
                elif alias == "iqr":
                    result = _digest_quantile(digests, q=0.75) - _digest_quantile(digests, q=0.25)
                elif alias == "le90":
                    result = _digest_quantile(digests, q=0.95) - _digest_quantile(digests, q=0.05)
                else:
                    result = _digest_quantile(digests, q=0.5 if alias == "median" else 0.9)
            elif alias in {"iqr", "le90"}:
                lower, upper = ("q25", "q75") if alias == "iqr" else ("q05", "q95")
                result = state[upper][index] - state[lower][index]
            else:
                assert alias is not None
                result = np.where(count > 0, state[alias][index], np.nan)
            selected_result = result[observed]
            columns[(index, name)] = _extrema_result(selected_result) if alias in {"min", "max"} else selected_result

    # Keep integer group IDs until the grouping parent restores category or interval labels
    return pd.DataFrame(columns, index=labels[observed])


def _collect_block(
    values: Sequence[NDArray[Any]], group_ids: NDArray[Any] | None, labels: Sequence[int]
) -> tuple[NDArray[Any] | None, list[NDArray[Any]]]:
    """
    Collect full group values from one block for statistics such as median or a user function.

    Apply the same group selection to every value array and retain missing values for total counts. With no group
    IDs, return the complete flattened arrays without constructing a group ID array.

    Values and group_ids follow _reduce_values(), restricted to this block.

    :param labels: Integer group IDs to collect from this block; ignored when group_ids is None.
    :returns: Selected flattened group IDs, or None without groups, followed by matching flattened value arrays.
    """

    from geoutils.multiproc.readers import _read_values

    # Read only this worker's file slices before selecting group members
    values = [_read_values(value) for value in values]

    # Return only group members, including missing values needed for total counts
    if group_ids is None:
        selected: slice | NDArray[Any] = slice(None)
        ids = None
    else:
        ids = np.asarray(group_ids).ravel()
        selected = ids == labels[0] if len(labels) == 1 else np.isin(ids, labels)
        ids = ids[selected]
    return ids, [np.asanyarray(array).ravel()[selected] for array in values]


def _aggregate_collected(
    blocks: Sequence[tuple[NDArray[Any] | None, list[NDArray[Any]]]],
    statistics: _Statistics,
) -> pd.DataFrame:
    """
    Calculate exact statistics after joining a group's values from every block it crosses.

    Concatenate corresponding value arrays and pass their shared group IDs to _aggregate_complete_groups().
    This uses the same estimators when a complete group spans several chunks.

    Statistics is the normalized request described in _aggregate_complete_groups_chunked().

    :param blocks: Nonempty sequence of matching group IDs and value arrays returned by _collect_block().
    :returns: The grouped table produced by _aggregate_complete_groups() for the joined values.
    """

    # Join each group's values without padding it to the full input shape
    ids = None if blocks[0][0] is None else np.concatenate([block[0] for block in blocks if block[0] is not None])
    arrays = []
    for index in range(len(blocks[0][1])):
        values = [block[1][index] for block in blocks]
        concatenate = np.ma.concatenate if any(np.ma.isMaskedArray(value) for value in values) else np.concatenate
        arrays.append(concatenate(values))
    return _aggregate_complete_groups(arrays, ids, statistics)


def _merge_block_tasks(
    value_blocks: Sequence[Sequence[Any]],
    id_blocks: Sequence[Any],
    total_groups: int,
    reindex: bool,
    reduction_statistics: set[str] | Sequence[set[str]],
    use_dask: bool,
    mp_config: MultiprocConfig | None,
    tdigest_compression: float,
    split_every: int,
    centers: tuple[NDArray[Any], NDArray[Any]] | None = None,
) -> tuple[NDArray[Any], NDArray[Any], dict[str, NDArray[Any]]]:
    """Schedule and merge bounded block summaries for one statistics or NMAD pass."""

    if not use_dask and mp_config is None:
        return _reduce_block(
            [blocks[0] for blocks in value_blocks],
            id_blocks[0],
            total_groups,
            reindex,
            reduction_statistics,
            tdigest_compression,
            centers,
        )

    if use_dask:
        submit = import_optional("dask").delayed

    # Limit submitted blocks so Dask cannot retain inputs from the whole raster while waiting for reductions
    levels: list[Any] = []
    for start in range(0, len(id_blocks), split_every):
        if use_dask:
            tasks = [
                submit(_reduce_block)(
                    [blocks[index] for blocks in value_blocks],
                    id_blocks[index],
                    total_groups,
                    reindex,
                    reduction_statistics,
                    tdigest_compression,
                    centers,
                )
                for index in range(start, min(start + split_every, len(id_blocks)))
            ]
            summary = submit(_merge_blocks)(tasks).compute()
        else:
            assert mp_config is not None
            handles = [
                mp_config.cluster.submit(
                    _reduce_block,
                    [blocks[index] for blocks in value_blocks],
                    id_blocks[index],
                    total_groups,
                    reindex,
                    reduction_statistics,
                    tdigest_compression,
                    centers,
                )
                for index in range(start, min(start + split_every, len(id_blocks)))
            ]
            summary = _merge_blocks(mp_config.cluster.gather(handles))
        depth = 0

        # Keep at most one accumulated summary per level, merging equally sized batches in submission order
        while depth < len(levels) and levels[depth] is not None:
            summary = _merge_blocks([levels[depth], summary])
            levels[depth] = None
            depth += 1
        if depth == len(levels):
            # Add a level when this batch has combined with every earlier level
            levels.append(summary)
        else:
            levels[depth] = summary

    # Combine the remaining occupied levels once all worker batches have finished
    remaining = [level for level in levels if level is not None]
    summary = remaining[0]
    for level in remaining[1:]:
        summary = _merge_blocks([summary, level])
    return summary


def _prepare_reduction_blocks(
    values: Sequence[Any], group_ids: Any | None, mp_config: MultiprocConfig | None
) -> tuple[list[list[Any]], list[Any], int, bool]:
    """Split values and group IDs into matching Dask chunks or worker tiles."""

    use_dask = mp_config is None and any(is_dask_array(array) for array in [group_ids, *values])
    if use_dask:
        import dask.array as da

        # Follow one input's chunks and optimize the column graphs together to share source reads
        chunk_source = next(array for array in [group_ids, *values] if is_dask_array(array))
        chunks = chunk_source.chunks
        value_blocks = [
            list(da.asarray(array).rechunk(chunks).to_delayed(optimize_graph=False).ravel()) for array in values
        ]
        id_blocks = (
            [None] * len(value_blocks[0])
            if group_ids is None
            else list(da.asarray(group_ids).rechunk(chunks).to_delayed(optimize_graph=False).ravel())
        )
        block_size = math.prod(max(lengths) for lengths in chunks)
    else:
        from geoutils.multiproc.chunked import iter_chunk_slices

        # Send lazy worker slices together, and leave file reads inside the workers
        tile_size = mp_config.chunks if mp_config is not None else values[0].shape
        lengths = (tile_size,) if isinstance(tile_size, int) else tile_size
        tiles = list(iter_chunk_slices(values[0].shape, tile_size))
        id_blocks = [None] * len(tiles) if group_ids is None else [group_ids[tile] for tile in tiles]
        value_blocks = [
            [array.block(tile) if isinstance(array, _ValueReader) else array[tile] for tile in tiles]
            for array in values
        ]
        block_size = math.prod(lengths[axis % len(lengths)] for axis in range(values[0].ndim))
    return value_blocks, id_blocks, block_size, use_dask


def _finalize_groupby_state(
    summary: tuple[NDArray[Any], NDArray[Any], dict[str, NDArray[Any]]],
    requested: Sequence[set[str]],
) -> tuple[NDArray[Any], NDArray[Any], dict[str, NDArray[Any]]]:
    """Return the same numerical summaries and missing groups from either reduction backend."""

    labels, size, state = summary
    count = state["count"]

    # Convert native centered deviations to the population standard deviation returned by Flox
    if "m2" in state:
        state["std"] = np.sqrt(np.divide(state.pop("m2"), count, out=np.full(count.shape, np.nan), where=count > 0))
    if "mean" in state and not any("mean" in aliases for aliases in requested):
        del state["mean"]

    # Unrequested estimates and empty groups have no numerical result, while counts stay integer
    for name in state.keys() - {"count", "digest"}:
        parent = {"sumofsquares": "rmse", "q25": "iqr", "q75": "iqr", "q05": "le90", "q95": "le90"}.get(name)
        selected = np.array([name in aliases or parent in aliases for aliases in requested])
        state[name] = np.where((count > 0) & selected[:, None], state[name], np.nan)
    return labels, size, state


def _groupby_reduce_geoutils(
    values: Sequence[Any],
    group_ids: Any | None,
    total_groups: int,
    aliases: set[str] | Sequence[set[str]],
    engine: str | None = None,
    compression: float = 100,
    method: str = "auto",
    reindex: bool | None = None,
    split_every: int | None = None,
    *,
    mp_config: MultiprocConfig | None = None,
) -> tuple[NDArray[Any], NDArray[Any], dict[str, NDArray[Any]]]:
    """
    Reduce multiple arrays by integer group IDs with GeoUtils own implementation.

    Both grouped reduction backends accept aligned values, negative IDs for excluded locations, and the number of
    declared groups. None uses one implicit group. Aliases can select the same summaries for every value, or a set
    per value. Results contain computed group labels, total location counts and named arrays of shape
    (number of values, number of groups). Counts use int64 and empty estimates are NaN.

    _prepare_reduction_blocks() aligns input chunks, _merge_block_tasks() combines their summaries, and
    _finalize_groupby_state() returns the same numerical results as _groupby_reduce_flox().
    """

    _check_input_flox_engine("geoutils", engine)
    _check_input_reduction_method("geoutils", method, reindex, split_every)
    requested = [aliases] * len(values) if isinstance(aliases, set) else aliases
    if not values or len(requested) != len(values):
        raise ValueError("Provide one set of reductions per value array.")
    supported = MERGEABLE_STATISTICS | _STATS_ALIAS_MASK.keys() | _TDIGEST_STATISTICS | {"digest"}
    if set().union(*requested) - supported:
        raise ValueError("The GeoUtils grouped reduction requires mergeable statistics.")
    if any(names & (_TDIGEST_STATISTICS | {"digest"}) for names in requested):
        _check_input_tdigest(compression)
        import_optional("crick", extra_name="num")
    reindex = total_groups <= 4096 if reindex is None else bool(reindex)
    if split_every is None:
        split_every = 8
        if mp_config is None and any(is_dask_array(array) for array in [group_ids, *values]):
            split_every = import_optional("dask").config.get("split_every", split_every)
        _check_input_reduction_method("geoutils", method, reindex, split_every)
    reductions = []
    for names in requested:
        selected = names - _TDIGEST_STATISTICS
        if names & _TDIGEST_STATISTICS:
            selected.add("digest")
        reductions.append(selected)

    # Build only the requested summaries and share the source reads across all values
    if values[0].size == 0:
        summary = _reduce_block(values, group_ids, total_groups, reindex, reductions, compression)
    else:
        value_blocks, id_blocks, _, use_dask = _prepare_reduction_blocks(values, group_ids, mp_config)
        summary = _merge_block_tasks(
            value_blocks,
            id_blocks,
            total_groups,
            reindex,
            reductions,
            use_dask,
            mp_config,
            compression,
            split_every=split_every,
        )

    # NMAD needs another pass after the group medians are known
    if any("nmad" in names for names in requested) and values[0].size:
        centers = summary[0], _digest_quantile(summary[2]["digest"], q=0.5)
        deviations = _merge_block_tasks(
            value_blocks,
            id_blocks,
            total_groups,
            reindex,
            [{"digest"} if "nmad" in names else set() for names in requested],
            use_dask,
            mp_config,
            compression,
            split_every=split_every,
            centers=centers,
        )
        summary[2]["nmad"] = 1.4826 * _digest_quantile(deviations[2]["digest"], q=0.5)
    elif any("nmad" in names for names in requested):
        summary[2]["nmad"] = np.full(summary[2]["count"].shape, np.nan)
    return _finalize_groupby_state(summary, requested)


def _aggregate_complete_groups_chunked(
    values: Sequence[Any],
    group_ids: Any | None,
    statistics: _Statistics,
    mp_config: MultiprocConfig | None,
) -> pd.DataFrame:
    """
    Calculate grouped statistics using Dask or multiprocesing.

    Internal logic:
    1. Split values and group labels into matching pieces, then choose how to calculate the requested statistics.
    2. Combine small summaries for mergeable statistics such as averages, or gather whole groups for exact medians and
        custom functions.
    3. Finish all calculations and return a table with one row per group and columns for each value and statistic.
    """

    # Return an empty table without starting tasks when the input is empty
    if values[0].size == 0:
        empty_ids = None if group_ids is None else np.empty(0, dtype=int)
        return _aggregate_complete_groups([np.empty(0) for _ in values], empty_ids, statistics)

    # Split values and group IDs into matching Dask chunks or NumPy tiles
    value_blocks, id_blocks, block_size, use_dask = _prepare_reduction_blocks(values, group_ids, mp_config)
    if use_dask:
        import dask

    # Find group memberships first, reading only group IDs until the required value blocks are known
    if group_ids is None:
        locations = {0: list(range(len(id_blocks)))}
        sizes = {0: int(values[0].size)}
    elif use_dask:
        memberships = list(dask.compute(*[dask.delayed(np.unique)(block, return_counts=True) for block in id_blocks]))
    else:
        memberships = [np.unique(block, return_counts=True) for block in id_blocks]
    if group_ids is not None:
        locations = {}
        sizes = {}
        for block_index, (labels, counts) in enumerate(memberships):
            # Ignore the missing marker and record every block contributing locations to each group
            for label, count in zip(labels[labels >= 0], counts[labels >= 0]):
                locations.setdefault(int(label), []).append(block_index)
                sizes[int(label)] = sizes.get(int(label), 0) + int(count)

    # Read several small groups together when they cross the same blocks
    cohorts: dict[tuple[int, ...], list[int]] = {}
    for label, block_indexes in locations.items():
        cohorts.setdefault(tuple(block_indexes), []).append(label)
    batches: list[tuple[list[int], tuple[int, ...]]] = []
    for cohort_blocks, labels in cohorts.items():
        batch: list[int] = []
        size = 0
        for label in labels:
            # Start another batch before adding a group would exceed the target size; never split one group
            if batch and size + sizes[label] > block_size:
                batches.append((batch, cohort_blocks))
                batch, size = [], 0
            batch.append(label)
            size += sizes[label]
        if batch:
            # Keep the last partly filled batch for this set of blocks
            batches.append((batch, cohort_blocks))

    # Keep each batch near one block's size unless one group is larger by itself
    tables = []
    for labels, cohort_blocks in batches:
        if use_dask:
            members = [
                dask.delayed(_collect_block)([blocks[index] for blocks in value_blocks], id_blocks[index], labels)
                for index in cohort_blocks
            ]
            table = dask.delayed(_aggregate_collected)(members, statistics)
        else:
            assert mp_config is not None

            # Workers select only the needed group members; the parent joins them for the exact estimators
            handles = [
                mp_config.cluster.submit(
                    _collect_block, [blocks[index] for blocks in value_blocks], id_blocks[index], labels
                )
                for index in cohort_blocks
            ]
            members = mp_config.cluster.gather(handles)
            table = _aggregate_collected(members, statistics)
        tables.append(table)

    if use_dask:
        from dask.base import get_scheduler

        # Submit one graph to a cluster so batches share source reads and graph scheduling
        scheduler = get_scheduler()
        distributed = modules.get("distributed")
        client = getattr(scheduler, "__self__", None)
        if distributed is not None and isinstance(client, distributed.Client):
            tables = list(dask.compute(*tables))
        else:
            # Local schedulers have no spilling, so finish one bounded batch at a time
            tables = [table.compute() for table in tables]

    # Keep exact integer columns when another batch contains only missing extrema
    if tables:
        for column in tables[0].columns:
            integer_dtypes = {str(table[column].dtype) for table in tables} & {"Int64", "UInt64"}
            if integer_dtypes:
                dtype = "UInt64" if "UInt64" in integer_dtypes else "Int64"
                for table in tables:
                    table[column] = table[column].astype(dtype)

    # Restore integer group order across batches, keeping the expected columns if no groups were found
    return (
        pd.concat(tables).sort_index()
        if tables
        else _aggregate_complete_groups([np.empty(0) for _ in values], np.empty(0, dtype=int), statistics)
    )


#############################
# 5/ FLOX REDUCTION
#############################

# We list here the Flox reductions functions that match GeoUtils definitions
_FLOX_REDUCTIONS = {"sum", "nansum", "count", "nanmean", "nanmin", "nanmax", "nanstd", "nanmedian", "nanquantile"}
_FLOX_EXACT_QUANTILES = {"nanmedian", "nanquantile"}


def _check_input_flox_engine(backend: str, engine: str | None) -> None:
    """Check the local numerical engine independently of the reduction and execution backends."""

    if engine is not None and backend != "flox":
        raise ValueError("Argument flox_engine requires the Flox reduction backend.")
    if engine not in {None, "numpy", "numba", "flox", "numbagg"}:
        raise ValueError("Argument flox_engine must be None, 'numpy', 'numba', 'flox' or 'numbagg'.")


def _check_input_flox_reducers(statistics: Sequence[Any]) -> None:
    """Reject reducer options that Flox doesn't support."""

    for statistic in statistics:
        if not isinstance(statistic, Reducer):
            continue
        ordinary = _can_reduce_arrays(statistic)
        if (
            not ordinary
            or statistic.minimum_inputs != type(statistic).minimum_inputs
            or statistic.default_nodata_propagation != "ignore"
            or statistic.error_structure is not None
            or any(name in vars(statistic) for name in ("reduce", "evaluate", "coefficients"))
        ):
            raise ValueError("The Flox backend does not support this Reducer or its configured options.")


def _check_flox_blockwise_groups(values: Sequence[Any], group_ids: NDArray[Any]) -> None:
    """Check that multidimensional groups fit within individual blocks before using Flox blockwise reductions."""

    # Flox can rearrange 1D groups itself, but requires complete groups in each multidimensional block
    if group_ids.ndim == 1:
        return
    for array in values:
        if not is_dask_array(array):
            continue
        starts = [np.r_[0, np.cumsum(chunks)] for chunks in array.chunks]
        seen: set[Any] = set()

        # Compare group labels across blocks without loading any raster values
        for index in np.ndindex(*array.numblocks):
            tile = tuple(slice(offsets[position], offsets[position + 1]) for offsets, position in zip(starts, index))
            labels = set(np.unique(group_ids[tile]))
            labels.discard(-1)
            if seen & labels:
                raise ValueError(
                    "Argument method='blockwise' requires each multidimensional group to fit in one chunk. "
                    "Rechunk the values or use method='map-reduce' for mergeable statistics, "
                    "or backend='geoutils' with method='groupwise' for complete groups."
                )
            seen.update(labels)


def _flox_groupby_reduce(
    values: Any,
    group_values: Sequence[Any],
    expected_groups: Sequence[pd.Index | NDArray[Any]],
    *,
    function: str | Aggregation,
    use_dask: bool,
    fill_value: float | int,
    dtype: Any,
    finalize_kwargs: Mapping[str, Any] | None = None,
    engine: str | None = None,
    sort: bool = False,
    reindex: bool | None = None,
    method: str | None = "map-reduce",
    split_every: int | None = None,
) -> Any:
    """Run one in-memory/Dask reduction over every grouping variable with Flox."""

    # Input checks
    if isinstance(function, str):
        if function not in _FLOX_REDUCTIONS:
            raise ValueError(f"The Flox adapter does not support reduction {function!r}.")
        if use_dask and function in _FLOX_EXACT_QUANTILES and method not in {None, "blockwise"}:
            raise ValueError("The Flox backend requires method='blockwise' or 'auto' for exact median or percentiles.")
    _check_input_flox_engine("flox", engine)
    if not group_values or len(group_values) != len(expected_groups):
        raise ValueError("Provide matching grouping arrays and expected groups.")

    # Reduce
    flox = import_optional("flox", extra_name="num")
    expected: Any = expected_groups[0] if len(expected_groups) == 1 else tuple(expected_groups)
    axes = tuple(range(-group_values[0].ndim, 0))
    # (Numbagg infers its reduction dtype from input and does not accept Flox dtype option)
    if engine == "numbagg" and dtype is not None:
        values = values.astype(dtype)
    reduce = partial(
        flox.groupby_reduce,
        values,
        *group_values,
        func=function,
        expected_groups=expected,
        sort=sort,
        axis=axes,
        fill_value=fill_value,
        dtype=None if engine == "numbagg" else dtype,
        method=method if use_dask else None,
        finalize_kwargs=finalize_kwargs,
        engine=engine,
        reindex=reindex,
    )
    if use_dask and split_every is not None:
        # Flox reads the merge size from Dask configuration while constructing its graph
        with import_optional("dask").config.set(split_every=split_every):
            result = reduce()[0]
    else:
        result = reduce()[0]
    return result.astype(dtype) if engine == "numbagg" and dtype is not None else result


def _tdigest_aggregation(compression: float) -> Aggregation:
    """Build a Flox aggregation that returns reusable t-digest states for each group."""

    _check_input_tdigest(compression)
    import_optional("crick", extra_name="num")
    flox = import_optional("flox", extra_name="num")
    return flox.Aggregation(
        "geoutils_tdigest",
        numpy=partial(_digest_chunk, compression=compression),
        chunk=partial(_digest_chunk, compression=compression),
        combine=_digest_merge,
        finalize=None,
        fill_value=None,
        final_fill_value=np.nan,
        dtypes=object,
        final_dtype=object,
    )


def _groupby_reduce_flox(
    values: Sequence[Any],
    group_ids: Any | None,
    total_groups: int,
    aliases: set[str] | Sequence[set[str]],
    engine: str | None = None,
    compression: float = 100,
    method: str = "auto",
    reindex: bool | None = None,
    split_every: int | None = None,
    *,
    mp_config: MultiprocConfig | None = None,
) -> tuple[NDArray[Any], NDArray[Any], dict[str, NDArray[Any]]]:
    """
    Calculate numerical estimates with Flox for aligned in-memory or Dask values and group IDs.

    Internal logic:
    _groupby_reduce_flox() performs the reductions and merges. Approximate quantiles use _tdigest_aggregation().
    The returned labels, group sizes and (value, group) arrays are formatted by _finalize_blocks().
    """

    _check_input_flox_engine("flox", engine)
    _check_input_reduction_method("flox", method, reindex, split_every)
    if mp_config is not None:
        raise ValueError("The Flox backend does not support multiprocessing.")
    requested = [aliases] * len(values) if isinstance(aliases, set) else aliases
    if not values or len(requested) != len(values):
        raise ValueError("Provide one set of reductions per value array.")
    all_aliases = set().union(*requested)
    use_dask = is_dask_array(group_ids) or any(is_dask_array(array) for array in values)
    if use_dask:
        import dask.array as module
    else:
        module = np

    # Use one implicit group for global statistics and preserve the prepared IDs for grouped statistics
    global_reduction = group_ids is None
    labels = np.arange(total_groups, dtype=np.int64)
    codes = module.zeros_like(values[0], dtype=np.int64) if global_reduction else group_ids
    assert codes is not None
    if global_reduction and method in {"blockwise", "cohorts"}:
        codes = np.broadcast_to(np.int64(0), values[0].shape)

    # Flox needs known group locations for cohorts and for checking complete groups within blocks
    if use_dask and method in {"blockwise", "cohorts"}:
        if is_dask_array(codes):
            raise ValueError(f"Argument method={method!r} requires in-memory group labels and mask.")
        if method == "blockwise":
            _check_flox_blockwise_groups(values, codes)

    # Flatten known labels to avoid incorrect multidimensional cohorts in Flox
    if use_dask and not is_dask_array(codes) and codes.ndim > 1 and method in {"auto", "cohorts"}:
        values = [array.reshape(-1) for array in values]
        codes = codes.reshape(-1)

    use_tdigest = method != "blockwise" and bool(all_aliases & (_TDIGEST_STATISTICS | {"digest"}))
    if use_tdigest:
        _check_input_tdigest(compression)
        import_optional("crick", extra_name="num")
    if use_tdigest and reindex:
        raise ValueError("Flox t-digest statistics require reindex=False or None.")

    reduce = partial(
        _flox_groupby_reduce,
        group_values=[codes],
        expected_groups=[labels],
        use_dask=use_dask,
        engine=engine,
        method=None if method == "auto" else method,
        reindex=reindex,
        split_every=split_every,
        sort=True,
    )
    size = reduce(module.ones_like(codes, dtype=np.int64), function="sum", fill_value=0, dtype=np.int64)

    functions = {"sum": "nansum", "mean": "nanmean", "min": "nanmin", "max": "nanmax", "std": "nanstd"}
    reduced_values = []
    deviation_inputs = []
    for array, names in zip(values, requested):
        # Collect only the probabilities needed for the requested quantiles and ranges
        probabilities = {}
        if names & {"median", "nmad"}:
            probabilities["median"] = 0.5
        if "90thpercentile" in names:
            probabilities["90thpercentile"] = 0.9
        if "iqr" in names:
            probabilities.update(q25=0.25, q75=0.75)
        if "le90" in names:
            probabilities.update(q05=0.05, q95=0.95)

        # Global estimates include unmasked infinities, while grouped estimates select finite observations
        raw = module.ma.getdata(array)
        mask = module.ma.getmaskarray(array)
        finite = ~mask & module.isfinite(raw)
        valid = ~mask & ~module.isnan(raw) if global_reduction else finite
        masked = np.ma.isMaskedArray(array._meta if is_dask_array(array) else array)
        count_valid = ~mask & ~module.isnan(raw) if global_reduction and masked else finite
        data = module.where(valid, raw.astype(np.float64), np.nan)
        scheduled = {
            "count": reduce(count_valid.astype(np.int64), function="sum", fill_value=0, dtype=np.int64),
            "size": size,
        }

        # Let Flox reduce numerical estimates across the complete array or its Dask chunks
        for name, function in functions.items():
            if name not in names:
                continue
            dtype: Any = np.float64
            fill_value: Any = np.nan
            selected = data
            if name in {"min", "max"} and np.issubdtype(raw.dtype, np.integer):
                # Integer bounds avoid rounding extrema above 2**53 when masking missing observations
                limits = np.iinfo(raw.dtype)
                fill_value = int(limits.max if name == "min" else limits.min)
                selected = module.where(valid, raw, fill_value)
                dtype = raw.dtype
            scheduled[name] = reduce(selected, function=function, fill_value=fill_value, dtype=dtype)
        if names & {"sumofsquares", "rmse"}:
            scheduled["sumofsquares"] = reduce(
                module.square(data), function="nansum", fill_value=np.nan, dtype=np.float64
            )

        # Share one digest per value and group for every approximate percentile
        if (probabilities or "digest" in names) and use_tdigest:
            scheduled["digest"] = reduce(
                data,
                function=_tdigest_aggregation(compression),
                fill_value=np.nan,
                dtype=object,
                sort=True,
                reindex=False,
            )
        elif probabilities and (use_dask or engine in {None, "flox"}):
            # Request quantiles together because Flox blockwise tasks can share names for different probabilities
            quantiles = reduce(
                data,
                function="nanquantile",
                fill_value=np.nan,
                dtype=np.float64,
                finalize_kwargs={"q": tuple(probabilities.values())},
            )
            for index, name in enumerate(probabilities):
                scheduled[name] = quantiles[index]
        else:
            # NumPy and Numba engines accept only one exact quantile per call
            for name, probability in probabilities.items():
                scheduled[name] = reduce(
                    data,
                    function="nanquantile",
                    fill_value=np.nan,
                    dtype=np.float64,
                    finalize_kwargs={"q": probability},
                )
        reduced_values.append(scheduled)
        deviation_inputs.append(data if "nmad" in names else None)

    # Compute the batch together so reductions can share Dask source reads
    if use_dask:
        reduced_values = import_optional("dask").compute(reduced_values)[0]

    # Reduce absolute deviations from the medians after every contributing chunk has been combined
    for names, data, results in zip(requested, deviation_inputs, reduced_values):
        if "nmad" in names:
            medians = _digest_quantile(results["digest"], q=0.5) if use_tdigest else results["median"]
            centers = module.take(medians, module.maximum(codes, 0))
            deviations = module.abs(data - centers)
            if use_tdigest:
                states = reduce(
                    deviations,
                    function=_tdigest_aggregation(compression),
                    fill_value=np.nan,
                    dtype=object,
                    sort=True,
                    reindex=False,
                )
                states = states.compute() if use_dask else states
                results["nmad"] = 1.4826 * _digest_quantile(states, q=0.5)
            else:
                results["nmad"] = 1.4826 * reduce(deviations, function="nanmedian", fill_value=np.nan, dtype=np.float64)

    # Add the value dimension expected by the shared output formatter
    combined = {}
    for name in set().union(*(result.keys() for result in reduced_values)) - {"size"}:
        arrays = [
            result.get(name, np.full(total_groups, None if name == "digest" else np.nan)) for result in reduced_values
        ]
        # Preserve integer extrema when selected values have different dtypes
        if name in {"min", "max"} and any(np.issubdtype(array.dtype, np.integer) for array in arrays):
            arrays = [array.astype(object) for array in arrays]
        combined[name] = np.stack(arrays)
    return _finalize_groupby_state((labels, reduced_values[0]["size"], combined), requested)


############################
# 6/ SHARED PARENT REDUCTION
############################


def _aggregate_summaries(
    values: Sequence[Any],
    group_ids: Any | None,
    statistics: _Statistics,
    total_groups: int,
    *,
    backend: str,
    flox_engine: str | None,
    tdigest_compression: float,
    method: str,
    reindex: bool | None,
    split_every: int | None,
    mp_config: MultiprocConfig | None,
) -> pd.DataFrame:
    """
    Calculate a statistics table from numerical summaries.

    _groupby_reduce_geoutils() or _groupby_reduce_flox() calculates the summaries, and _finalize_blocks() formats
    them as a table. Arguments follow _reduce_values().
    """

    # Request the same summaries and format their results through either numerical backend
    reduce = _groupby_reduce_geoutils if backend == "geoutils" else _groupby_reduce_flox
    summary = reduce(
        values,
        group_ids,
        total_groups,
        {alias for alias in statistics.aliases if alias is not None},
        flox_engine,
        tdigest_compression,
        method,
        reindex,
        split_every,
        mp_config=mp_config,
    )

    table = _finalize_blocks(summary, statistics, global_reduction=group_ids is None)

    # Preserve NumPy's global estimates when a request also calculates t-digest quantiles
    if (
        backend == "geoutils"
        and group_ids is None
        and mp_config is None
        and not any(is_dask_array(array) for array in values)
    ):
        exact_aliases = {
            alias for alias in statistics.aliases if alias in _STATS_ALIAS_CALLABLE and alias not in _TDIGEST_STATISTICS
        }
        if exact_aliases:
            for index, array in enumerate(values):
                estimates, _ = _reduce_complete_values(array, exact_aliases)
                for name, alias in zip(statistics.names, statistics.aliases):
                    if alias is not None and alias in estimates:
                        table[index, name] = estimates[alias]
    return table


def _reduce_values(
    values: Sequence[Any],
    statistics: _Statistics,
    *,
    group_ids: Any | None = None,
    total_groups: int = 1,
    mp_config: MultiprocConfig | None = None,
    backend: str = "geoutils",
    flox_engine: str | None = None,
    tdigest_compression: float = 100,
    method: str = "auto",
    reindex: bool | None = None,
    split_every: int | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Reduce one or more arrays globally or by integer group IDs.

    _resolve_reduction_method() chooses block summaries or complete groups.
    _aggregate_complete_groups_chunked() schedules the shared calculations for Dask or multiprocessing.
    In-memory grouped inputs use the same _reduce_block() and _finalize_blocks() summaries, while one global group
    or exact grouped estimates use _aggregate_complete_groups().

    The shared block helpers define the estimates, while _aggregate_complete_groups_chunked() documents the backend
    differences in scheduling, memory use and observation order. This parent returns a computed table for every backend.

    Statistics, method, reindex and mp_config follow stats().

    :param values: Nonempty sequence of numeric NumPy or Dask arrays with matching shapes and missing values
        represented by NaN. Dask inputs must have known shapes and compatible chunks.
    :param group_ids: Integer array with the same shape as the values. IDs from zero to total_groups - 1 identify
        groups, and negative IDs exclude locations. None treats all locations as one group.
    :param total_groups: Number of declared group combinations, including groups with no eligible locations.
    :returns: A computed table with integer group rows and (value position, statistic name) columns, and the resolved
        reduction method.
    """

    # 1/ Inputs check, for execution and per backend
    _check_input_reduction_method(backend, method, reindex, split_every)
    use_dask = is_dask_array(group_ids) or any(is_dask_array(array) for array in values)
    if use_dask and mp_config is not None:
        raise ValueError("Dask inputs cannot be combined with Multiprocessing statistics.")
    chunked = use_dask or mp_config is not None

    # Calculate in-memory quantiles exactly unless a reduction method was explicitly requested
    if not chunked and method == "auto" and any(alias in _TDIGEST_STATISTICS for alias in statistics.aliases):
        method = "groupwise" if backend == "geoutils" else "blockwise"

    if backend == "flox":
        if mp_config is not None:
            raise ValueError("The Flox backend does not support multiprocessing.")
        _check_input_flox_reducers(statistics.requested)
        if any(alias is None for alias in statistics.aliases):
            raise ValueError("The Flox backend does not support custom statistics.")
        resolved_method = method
    else:
        resolved_method, reindex, mergeable = _resolve_reduction_method(
            statistics.aliases, total_groups, method, reindex
        )

        _check_input_reduction_method("geoutils", resolved_method, reindex, split_every)

    # 2/ Reduction

    # Gather complete groups for exact statistics with the GeoUtils backend
    if backend == "geoutils":
        # Use complete groups for global in-memory statistics and "groupwise"
        if not chunked and (
            resolved_method == "groupwise"
            or group_ids is None
            and not any(alias in _TDIGEST_STATISTICS for alias in statistics.aliases)
        ):
            return _aggregate_complete_groups(values, group_ids, statistics), resolved_method

        # Use complete groups also for Dask
        if chunked and resolved_method == "groupwise":
            return _aggregate_complete_groups_chunked(values, group_ids, statistics, mp_config), resolved_method

    # Otherwise, reduce numerical summaries through GeoUtils or Flox
    return (
        _aggregate_summaries(
            values,
            group_ids,
            statistics,
            total_groups,
            backend=backend,
            flox_engine=flox_engine,
            tdigest_compression=tdigest_compression,
            method=resolved_method,
            reindex=reindex,
            split_every=split_every,
            mp_config=mp_config,
        ),
        resolved_method,
    )


############################
# 7/ GLOBAL OUTPUT
############################


def _global_reduction_statistics(statistics: _Statistics) -> _Statistics:
    """Keep numerical estimators in the shared reducer while leaving global-only output to its formatter."""

    selected = [
        (statistic, name, alias)
        for statistic, name, alias in zip(statistics.requested, statistics.names, statistics.aliases)
        if alias in _STATS_ALIAS_CALLABLE or isinstance(statistic, Reducer) and alias is None
    ]
    return replace(
        statistics,
        requested=[statistic for statistic, _, _ in selected],
        names=[name for _, name, _ in selected],
        aliases=[alias for _, _, alias in selected],
        single=False,
    )


def _global_reduction_requires_complete_values(
    values: Sequence[Any],
    statistics: _Statistics,
    method: str,
    mp_config: MultiprocConfig | None,
) -> bool:
    """Identify global inputs that need their complete array for established NumPy or callable behavior."""

    # User functions receive the complete selected array, including its original dimensions
    if any(callable(statistic) for statistic in statistics.requested):
        return True
    if method == "groupwise" or not any(alias in _STATS_ALIAS_CALLABLE for alias in statistics.aliases):
        return False

    # NumPy's global estimators include infinities while the grouped block reducer selects finite observations
    dask_checks = []
    for array in values:
        if not np.issubdtype(array.dtype, np.inexact):
            continue
        if is_dask_array(array):
            import_optional("dask")
            import dask.array as da

            dask_checks.append(da.ma.filled(da.isinf(array), False).any())
        else:
            if isinstance(array, _ValueReader):
                assert mp_config is not None
                if _reader_requires_complete_values(array, mp_config):
                    return True
            elif bool(np.ma.filled(np.any(np.isinf(array)), False)):
                return True
    if dask_checks:
        import dask

        return any(dask.compute(*dask_checks))
    return False


def _materialize_global_values(values: Sequence[Any]) -> list[Any]:
    """Read complete raster or point cloud values and compute Dask arrays together for global-only behavior."""

    materialized = list(values)
    dask_positions = []
    dask_values = []
    for index, array in enumerate(materialized):
        if isinstance(array, _ValueReader):
            window = tuple(slice(0, length) for length in array.shape)
            materialized[index] = array.read(window)
        elif is_dask_array(array):
            dask_positions.append(index)
            dask_values.append(array)
    if dask_values:
        import_optional("dask")
        import dask

        for index, array in zip(dask_positions, dask.compute(*dask_values)):
            materialized[index] = array
    return materialized


def _materialize_global_counts(
    counts: Sequence[_SelectionCounts | None],
) -> list[_SelectionCounts | None]:
    """Compute all small Dask mask counts together before formatting global summaries."""

    materialized = list(counts)
    positions = []
    values = []
    for index, selection_counts in enumerate(counts):
        if selection_counts is None:
            continue
        positions.append(index)
        values.extend((selection_counts.valid_before_mask, selection_counts.selected_locations))
    if any(is_dask_array(value) for value in values):
        import_optional("dask")
        import dask

        values = list(dask.compute(*values))
    for position, start in zip(positions, range(0, len(values), 2)):
        selection_counts = counts[position]
        assert selection_counts is not None
        materialized[position] = replace(
            selection_counts,
            valid_before_mask=values[start],
            selected_locations=values[start + 1],
        )
    return materialized


def _format_global_stats(
    table: pd.DataFrame,
    value_names: Sequence[str],
    values: Sequence[Any],
    counts: Sequence[_SelectionCounts | None],
    statistics: _Statistics,
) -> Any:
    """Restore scalar or dictionary global output from the one-group reduction table and selection counts."""

    results = {}
    for value_index, (value_name, array, selection_counts) in enumerate(zip(value_names, values, counts)):
        final_count = int(table.loc[0, (value_index, "count")])
        valid_count = final_count if selection_counts is None else int(selection_counts.valid_before_mask)
        total_count = int(array.size)
        selected_count = None if selection_counts is None else int(selection_counts.selected_locations)
        value_result = {}

        # Map the shared numerical columns and the global mask counts back to the requested output labels
        for statistic, name, alias in zip(statistics.requested, statistics.names, statistics.aliases):
            if callable(statistic):
                result = statistic(array)
            elif alias in _STATS_ALIAS_CALLABLE:
                result = table.loc[0, (value_index, name)]
            elif alias == "validcount":
                result = valid_count
            elif alias == "totalcount":
                result = total_count
            elif alias == "percentagevalidpoints":
                result = 100 * valid_count / total_count if total_count else np.nan
            elif alias == "validinliercount" and selection_counts is not None:
                result = final_count
            elif alias == "totalinliercount" and selection_counts is not None:
                result = selected_count
            elif alias == "percentageinlierpoints" and selection_counts is not None:
                result = 100 * final_count / valid_count if valid_count else np.nan
            elif alias == "percentagevalidinlierpoints" and selection_counts is not None:
                result = 100 * final_count / selected_count if selected_count else 0
            elif isinstance(statistic, Reducer):
                result = table.loc[0, (value_index, name)]
            else:
                result = np.nan
            if not callable(statistic) and isinstance(result, np.generic) and not np.ma.is_masked(result):
                result = result.item()
            value_result[name] = result
        results[value_name] = next(iter(value_result.values())) if statistics.single else value_result
    return next(iter(results.values())) if len(results) == 1 else results


def _reduce_global_values(
    selected_values: Mapping[str, tuple[Any, _SelectionCounts | None]],
    statistics: _Statistics,
    *,
    mp_config: MultiprocConfig | None,
    backend: str = "geoutils",
    flox_engine: str | None = None,
    tdigest_compression: float = 100,
    method: str = "auto",
    reindex: bool | None = None,
    split_every: int | None = None,
) -> Any:
    """
    Calculate global statistics through the same reducer as one complete grouped bin.

    _global_reduction_statistics() selects the numerical estimates shared with grouped statistics. _reduce_values()
    calculates them with one implicit group, and _format_global_stats() restores the established global counts,
    callable results and scalar or dictionary output.

    Selected_values is returned by _sample_and_mask_global_values(); statistics, method and mp_config follow stats().

    :returns: Statistics for one selected value, or a mapping from value names to their statistics.
    """

    # Resolve the requested backend before removing global-only counts and callable output from the shared table
    if backend == "flox" and any(alias is None for alias in statistics.aliases):
        raise ValueError("The Flox backend does not support custom statistics.")
    names = list(selected_values)
    values = [selected_values[name][0] for name in names]
    counts = [selected_values[name][1] for name in names]
    chunked = any(is_dask_array(array) for array in values) or mp_config is not None
    reduction_aliases = [
        alias
        for statistic, alias in zip(statistics.requested, statistics.aliases)
        if callable(statistic) or alias in _STATS_ALIAS_CALLABLE
    ]
    resolved_method = method
    if backend == "geoutils":
        resolved_method, _, _ = _resolve_reduction_method(reduction_aliases, 1, method, reindex)

    # Read complete arrays only for custom functions or the exceptional global handling of infinities
    use_complete_values = (
        backend == "geoutils"
        and chunked
        and _global_reduction_requires_complete_values(values, statistics, resolved_method, mp_config)
    )
    if use_complete_values:
        values = _materialize_global_values(values)
    reduction_statistics = _global_reduction_statistics(statistics)
    table, _ = _reduce_values(
        values,
        reduction_statistics,
        mp_config=None if use_complete_values else mp_config,
        backend=backend,
        flox_engine=flox_engine,
        tdigest_compression=tdigest_compression,
        method=method,
        reindex=reindex,
        split_every=split_every,
    )

    # Finish the small global count fields and custom functions after every numerical backend returns its table
    counts = _materialize_global_counts(counts)
    return _format_global_stats(table, names, values, counts, statistics)
