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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
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
from geoutils.raster.array import get_mask_from_array
from geoutils.stats.estimators import linear_error, nmad, rmse, sum_square

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.stats.selection import _SelectionCounts

#########################
# 1/ HELPERS
#########################

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


@dataclass(frozen=True)
class _Statistics:
    """
    Store requested statistics, output names and validated internal aliases.

    :param requested: Statistic names or callables.
    :param names: Matching output column names.
    :param aliases: Matching internal statistic names (None for callables).
    """

    requested: list[str | Callable[[Any], Any]]
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
    statistics: str | Callable[[Any], Any] | Iterable[str | Callable[[Any], Any]] | None,
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
        requested: list[str | Callable[[Any], Any]] = list(_STATS_LIST_MIN)
    elif isinstance(statistics, str) and statistics == "all":
        requested = list(_STATS_ALIAS_GEN)
        if masked and not grouped:
            requested += list(_STATS_ALIAS_MASK)
    elif isinstance(statistics, str) or callable(statistics):
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
        elif callable(statistic):
            function = statistic
            while isinstance(function, partial):
                function = function.func
            names.append(getattr(function, "__name__", type(function).__name__))
        else:
            raise TypeError("Argument ``statistics`` must contain names or callable functions.")
    if default_names and not grouped:
        names = [_STATS_ALIAS_ALL[name] for name in names]
    aliases = [_get_stat_common_alias(statistic) if isinstance(statistic, str) else None for statistic in requested]

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
    single = callable(statistics) or isinstance(statistics, str) and statistics != "all"
    return _Statistics(requested=requested, names=names, aliases=aliases, grouped=grouped, single=single)


def _resolve_strategy(
    aliases: Sequence[str | None], strategy: str, total_groups: int, chunked: bool
) -> tuple[str, bool]:
    """
    Choose chunk strategy for the requested statistics.

    See _reduce_values() for ``strategy`` and ``total_groups`` arguments, and ``aliases`` are from _Statistics.

    :param chunked: Whether Dask or multiprocessing will calculate separate input blocks.

    :returns: The strategy and whether every requested statistic can be merged from block summaries.
    """

    # Mask-related counts use the same count reductions as ordinary validity statistics
    mergeable = all(alias in MERGEABLE_STATISTICS or alias in _STATS_ALIAS_MASK for alias in aliases)
    if strategy == "auto":
        if not mergeable:
            strategy = "groupwise"
        else:
            strategy = "dense" if total_groups <= 4096 else "sparse"
    elif strategy not in {"dense", "sparse", "groupwise"}:
        raise ValueError("Argument ``strategy`` must be 'auto', 'dense', 'sparse' or 'groupwise'.")
    if chunked and not mergeable and strategy != "groupwise":
        raise ValueError("Exact quantiles and custom statistics require ``strategy``='groupwise' or 'auto'.")
    return strategy, mergeable


#########################
# 2/ EAGER REDUCTION
#########################


def _reduce_complete_values_eager(data: NDArrayNum, aliases: set[str]) -> tuple[dict[str, Any], int]:
    """
    Evaluate whole-array statistics with NumPy's masked or NaN-aware functions.

    We preserve input type when calling estimators: masked arrays use NumPy Masked functions, while
    ordinary arrays use functions that ignore NaNs.

    We also return the validity count, total count and percentages.

    See _reduce_values() for ``data`` argument description.

    :param aliases: Internal names of the numerical statistics to calculate, excluding count statistics.

    :returns: Estimates keyed by their internal names and the input validity count.
    """

    # Keep the existing distinction between masked values and NaNs in ordinary arrays
    masked = np.ma.isMaskedArray(data)
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


def _aggregate_eager(
    values: Sequence[NDArray[Any]], group_ids: NDArray[Any] | None, statistics: _Statistics
) -> pd.DataFrame:
    """
    Calculate every statistic from complete in-memory groups after one stable sort.

    Locations are sorted once by group ID, placing all values from each group in one slice. Statistics that need the
    complete group, such as quantiles and user functions, can then use those slices without filtering the full array
    separately for every group. The stable sort keeps the original order within each group. With no group IDs,
    use the complete input directly and avoid allocating or sorting an artificial group array.

    See _reduce_values() for ``values`` and ``group_ids`` descriptions, and see _aggregate_chunked() for
    ``statistics``.

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
            count = int(np.count_nonzero(~get_mask_from_array(group)))
            if global_reduction:
                computed, count = _reduce_complete_values_eager(group, aliases)
            else:
                computed, _ = _reduce_complete_values_eager(group, aliases) if count and aliases else ({}, count)
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

    # 3/ Assemble the shared table format used by eager and chunked reductions
    return pd.DataFrame(columns, index=labels)


#############################################
# 3/ CHUNKED REDUCTION (DASK AND MULTIPROCESSING)
#############################################


def _reader_block_requires_eager(block: Any) -> bool:
    """Identify values for which global NumPy semantics differ from finite grouped reductions."""

    from geoutils.multiproc.readers import _read_values

    values = _read_values(block)
    nonfinite = ~np.isfinite(values) if np.ma.isMaskedArray(values) else np.isinf(values)
    return bool(np.ma.filled(np.any(nonfinite), False))


def _reader_requires_eager(array: _ValueReader, mp_config: MultiprocConfig) -> bool:
    """Check exceptional nonfinite values in bounded reads before choosing global estimators."""

    from geoutils.multiproc.chunked import iter_chunk_slices
    from geoutils.multiproc.cluster import _map_bounded

    arguments = ((array.block(tile),) for tile in iter_chunk_slices(array.shape, mp_config.chunks))
    return any(result for _, result in _map_bounded(mp_config.cluster, _reader_block_requires_eager, arguments))


def _statistics_dask(data: Any, aliases: set[str]) -> tuple[dict[str, Any], Any]:
    """
    Build only the requested Dask reductions while returning lazy scalar results.

    Reuse the global median when both median and NMAD are requested. Counts and estimates remain lazy for the caller
    to compute together; exact quantiles may still need to bring values from several chunks into one worker.

    This optional native Dask calculation is kept as a reference for tests and benchmarks. Public stats() uses
    _reduce_values() so global and grouped reductions share the GeoUtils implementation.

    Data follows _reduce_values(), and aliases follow _reduce_complete_values_eager().

    :returns: Lazy estimates keyed by their internal names and a lazy finite-value count.
    """

    import_optional("dask")
    import dask.array as da

    # Dask requires explicit axes for full-array quantiles. This remains lazy, but exact global quantiles can still be
    # memory-intensive at execution time because Dask has to combine data across chunks
    axes = tuple(range(data.ndim))
    median = da.nanquantile(data, 0.50, axis=axes) if aliases & {"median", "nmad"} else None
    functions: dict[str, Callable[[], Any]] = {
        name: partial(getattr(da, "nan" + name), data) for name in ("mean", "max", "min", "sum", "std")
    }
    squared_values = data.astype(np.float64) if np.issubdtype(data.dtype, np.integer) else data

    # Defer graph construction for quantiles and squared values until their statistics are requested
    functions.update(
        median=lambda: median,
        sumofsquares=lambda: da.nansum(da.square(squared_values)),
        **{"90thpercentile": lambda: da.nanquantile(data, 0.90, axis=axes)},
        le90=lambda: da.nanquantile(data, 0.95, axis=axes) - da.nanquantile(data, 0.05, axis=axes),
        iqr=lambda: da.nanquantile(data, 0.75, axis=axes) - da.nanquantile(data, 0.25, axis=axes),
        nmad=lambda: 1.4826 * da.nanquantile(da.fabs(data - median), 0.50, axis=axes),
        rmse=lambda: da.sqrt(da.nanmean(da.square(squared_values))),
    )
    finite = da.isfinite(data)
    if np.ma.isMaskedArray(data._meta):
        finite = da.ma.filled(finite, False)
    final_count = finite.sum()
    results = {alias: functions[alias]() for alias in aliases}
    for alias in aliases & {"sum", "sumofsquares"}:
        results[alias] = da.where(final_count > 0, results[alias], np.nan)
    return results, final_count


def _reduce_block(
    values: Sequence[NDArray[Any]],
    group_ids: NDArray[Any] | None,
    total_groups: int,
    dense: bool,
    statistics: set[str],
) -> tuple[NDArray[Any], NDArray[Any], dict[str, NDArray[Any]]]:
    """
    Summarize groups in one array block so summaries from many blocks can be combined.

    A dense summary reserves one position for every declared group; a sparse summary stores only groups present in the
    block. Each selected value has its own finite count, while group sizes include missing selected values. For standard
    deviation, the summary stores the mean and sum of squared deviations needed by the pairwise merge. Stored quantities
    have shape (number of selected values, number of stored groups), and only requested quantities are allocated.

    Values and group_ids follow _reduce_values(), restricted to this block. Total_groups still includes every
    declared group across the full input.

    :param dense: Reserve every declared group when True; otherwise store only groups present in the block.
    :param statistics: Internal names of the requested mergeable statistics, including any count statistics.
    :returns: Stored integer group labels, their total location counts, and named summary arrays with one row per
        selected value and one column per stored group. These are the summaries consumed by _merge_blocks().
    """

    from geoutils.multiproc.readers import _read_values

    # Read file descriptors inside the worker while leaving supplied arrays unchanged
    values = [_read_values(value) for value in values]

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
        if dense:
            labels = np.arange(total_groups, dtype=np.int64)
            codes = ids[eligible].astype(np.int64)
        else:
            labels, codes = np.unique(ids[eligible], return_inverse=True)
        size = np.bincount(codes, minlength=len(labels))
    shape = (len(values), len(labels))
    state = {"count": np.zeros(shape, dtype=np.int64)}

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

    # 3/ Reuse group IDs while handling missing data separately for each value array
    for index, array in enumerate(values):
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
        if len(state) == 1:
            continue
        valid_data = data[finite]
        needs_float = bool(state.keys() & {"mean", "m2", "sum", "sumofsquares"})
        data = valid_data.astype(float, copy=False) if needs_float else valid_data

        # Measure spread around each local mean to keep small variation accurate beside large values
        if needs_mean or "sum" in state:
            sums = (
                np.array([np.sum(data)], dtype=float)
                if valid_codes is None
                else np.bincount(valid_codes, weights=data, minlength=len(labels))
            )
            if "sum" in state:
                state["sum"][index] = sums
            if needs_mean:
                mean = np.divide(sums, count, out=np.zeros(len(labels)), where=count > 0)
                state["mean"][index] = mean

            # Store squared deviations from the local mean; the merge later corrects for differences between means
            if "m2" in state:
                deviations = data - (mean[0] if valid_codes is None else mean[valid_codes])
                state["m2"][index] = (
                    np.array([np.sum(deviations**2)], dtype=float)
                    if valid_codes is None
                    else np.bincount(valid_codes, weights=deviations**2, minlength=len(labels))
                )

        # Keep uncentered squared values for sum of squares and RMSE, which measure magnitude rather than spread
        if "sumofsquares" in state:
            state["sumofsquares"][index] = (
                np.array([np.sum(data**2)], dtype=float)
                if valid_codes is None
                else np.bincount(valid_codes, weights=data**2, minlength=len(labels))
            )

        # Leave empty groups at their neutral extrema; finalization turns their estimates into NaN
        if data.size:
            if "min" in state:
                if valid_codes is None:
                    state["min"][index, 0] = np.min(valid_data)
                else:
                    np.minimum.at(state["min"][index], valid_codes, valid_data)
            if "max" in state:
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

    # 2/ Allocate the same quantities as the incoming summaries, using neutral starting values
    combined: dict[str, NDArray[Any]] = {}
    for name in first_state:
        fill = np.inf if name == "min" else -np.inf if name == "max" else 0
        combined[name] = np.full(shape, fill, dtype=first_state[name].dtype)

    # 3/ Merge each block's counts, means, spread, and other requested quantities
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
            else:
                current = current + state[name]
            combined[name][:, positions] = current
    return labels, size, combined


def _finalize_blocks(
    summary: tuple[NDArray[Any], NDArray[Any], dict[str, NDArray[Any]]],
    statistics: _Statistics,
) -> pd.DataFrame:
    """
    Turn combined block summaries into the same result columns as the in-memory path.

    Take the square root of the mean squared deviation for population standard deviation, or the mean squared value
    for RMSE. Groups with locations but no finite selected values keep their counts and receive NaN estimates.

    Statistics is the normalized request described in _aggregate_chunked().

    :param summary: Combined labels, total location counts and summary arrays returned by _merge_blocks().
    :returns: A table with integer group rows and (value position, statistic name) columns.
    """

    # Keep groups that contain locations even when every selected value is missing
    labels, size, state = summary
    observed = size > 0
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
            elif alias in {"std", "rmse"}:
                numerator = state["m2" if alias == "std" else "sumofsquares"][index]
                result = np.sqrt(np.divide(numerator, count, out=np.full(len(size), np.nan), where=count > 0))
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
    blocks: Sequence[tuple[NDArray[Any] | None, list[NDArray[Any]]]], statistics: _Statistics
) -> pd.DataFrame:
    """
    Calculate exact statistics after joining a group's values from every block it crosses.

    Concatenate corresponding value arrays and pass their shared group IDs to _aggregate_eager(). This keeps
    the eager estimator definitions while allowing one complete group to span several chunks.

    Statistics is the normalized request described in _aggregate_chunked().

    :param blocks: Nonempty sequence of matching group IDs and value arrays returned by _collect_block().
    :returns: The grouped table produced by _aggregate_eager() for the joined values.
    """

    # Join each group's values without padding it to the full input shape
    ids = None if blocks[0][0] is None else np.concatenate([block[0] for block in blocks if block[0] is not None])
    arrays = []
    for index in range(len(blocks[0][1])):
        values = [block[1][index] for block in blocks]
        concatenate = np.ma.concatenate if any(np.ma.isMaskedArray(value) for value in values) else np.concatenate
        arrays.append(concatenate(values))
    return _aggregate_eager(arrays, ids, statistics)


def _aggregate_chunked(
    values: Sequence[Any],
    group_ids: Any | None,
    statistics: _Statistics,
    total_groups: int,
    strategy: str,
    mp_config: MultiprocConfig | None,
) -> pd.DataFrame:
    """
    Calculate grouped statistics across Dask chunks or NumPy tiles handled by worker processes.

    ``dense`` reduces each input chunk to fixed-size arrays containing every declared group.
    ``sparse`` stores only groups present in each chunk and aligns their labels when chunks are combined.

    Both combine counts, sums, minima, maxima, means, standard deviations and RMSE without keeping the original values.

    Median, NMAD, other quantiles, and user functions cannot be aggregated across chunks from these summaries. For
    these statistics, ``groupwise`` finds the chunks containing each group and collects all its values before
    calculation. Groups found in the same chunks are read together in batches near one chunk's size.

    _collect_block() and _aggregate_collected() calculate complete groups. For mergeable statistics, _reduce_block()
    summarizes each block, _merge_blocks() combines small batches of summaries, and _finalize_blocks() builds the
    result columns. Both backends share these calculations, with the following execution differences:

    - Dask schedules block reductions and an eight-way merge tree. Multiprocessing submits up to eight reductions
      at a time, then merges their summaries in the caller using a bounded set of binary accumulation levels.
      Different merge trees and block layouts can produce small floating-point differences.
    - Both backends find group memberships before collecting values and process group batches one at a time.
      Dask schedules collection and complete-group calculation as tasks; multiprocessing collects members in
      workers and calculates their statistics in the caller. A complete group is never split, so its memory use
      can exceed the target batch size. Dask may recompute contributing blocks between successive batches.
    - Collected values follow block-grid order, then flattened order within each block. Equal block layouts give
      the same order across these backends, but tiling can change the order from an eager array's flattened order.
      Consequently, grouped user functions that depend on observation order can depend on the block layout.

    Both paths finish computation before returning a Pandas table.

    Values, group_ids, total_groups and mp_config follow _reduce_values(). Strategy is already resolved from auto
    to dense, sparse or groupwise by that parent.

    :param statistics: Validated _Statistics from _normalize_statistics(), reused by every block calculation.
    :returns: A computed table with integer group rows and (value position, statistic name) columns.
    """

    # 1/ Divide values and group IDs into matching blocks for the selected backend

    # Return an empty table without starting tasks when the input is empty
    if values[0].size == 0:
        empty_ids = None if group_ids is None else np.empty(0, dtype=int)
        return _aggregate_eager([np.empty(0) for _ in values], empty_ids, statistics)

    # Split values and group IDs into matching Dask chunks or NumPy tiles
    use_dask = is_dask_array(group_ids) or any(is_dask_array(array) for array in values)
    if use_dask:
        import_optional("dask")
        import dask
        import dask.array as da

        chunk_source = next(array for array in [group_ids, *values] if is_dask_array(array))
        chunks = chunk_source.chunks
        value_blocks = [list(da.asarray(array).rechunk(chunks).to_delayed().ravel()) for array in values]
        id_blocks: list[Any]

        # A whole-array reduction needs no group ID blocks, but still follows the same block ordering
        if group_ids is None:
            id_blocks = [None] * len(value_blocks[0])
        else:
            ids = da.asarray(group_ids).rechunk(chunks)
            id_blocks = list(ids.to_delayed().ravel())
        submit = dask.delayed
        block_size = math.prod(max(lengths) for lengths in chunks)
    else:
        # Follow the requested worker tile size in every array dimension
        if mp_config is None:
            raise ValueError("Chunked NumPy aggregation requires ``mp_config``.")
        from geoutils.multiproc.chunked import iter_chunk_slices

        lengths = (mp_config.chunks, mp_config.chunks) if isinstance(mp_config.chunks, int) else mp_config.chunks

        # Reuse identical tile views for group memberships and every selected value
        tiles = list(iter_chunk_slices(values[0].shape, mp_config.chunks))
        id_blocks = [None] * len(tiles) if group_ids is None else [group_ids[tile] for tile in tiles]
        from geoutils.multiproc.readers import _ValueReader

        value_blocks = [
            [array.block(tile) if isinstance(array, _ValueReader) else array[tile] for tile in tiles]
            for array in values
        ]
        block_size = math.prod(lengths[axis % 2] for axis in range(values[0].ndim))

    # 2/ Collect complete groups when the requested statistics need all their values
    if strategy == "groupwise":
        # Find group memberships first, reading only group IDs until the required value blocks are known
        if group_ids is None:
            locations = {0: list(range(len(id_blocks)))}
            sizes = {0: int(values[0].size)}
        elif use_dask:
            memberships = list(
                dask.compute(*[dask.delayed(np.unique)(block, return_counts=True) for block in id_blocks])
            )
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
                table = dask.delayed(_aggregate_collected)(members, statistics).compute()
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
            else _aggregate_eager([np.empty(0) for _ in values], np.empty(0, dtype=int), statistics)
        )

    # 3/ Combine small summaries when the requested statistics can be merged across blocks

    # Ask each block for only the counts, sums, means, or ranges needed by the request
    reduction_statistics = {alias for alias in statistics.aliases if alias is not None}
    dense = strategy == "dense"
    if use_dask:
        tasks = [
            submit(_reduce_block)(
                [blocks[index] for blocks in value_blocks], block, total_groups, dense, reduction_statistics
            )
            for index, block in enumerate(id_blocks)
        ]

        # Merge groups of eight tasks at each level so no single task receives every block summary
        while len(tasks) > 1:
            tasks = [submit(_merge_blocks)(tasks[start : start + 8]) for start in range(0, len(tasks), 8)]
        summary = tasks[0].compute()
    else:
        assert mp_config is not None
        # Combine worker results in small groups so finished summaries do not keep piling up in memory
        levels: list[Any] = []
        for start in range(0, len(id_blocks), 8):
            handles = [
                mp_config.cluster.submit(
                    _reduce_block,
                    [blocks[index] for blocks in value_blocks],
                    id_blocks[index],
                    total_groups,
                    dense,
                    reduction_statistics,
                )
                for index in range(start, min(start + 8, len(id_blocks)))
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
        summary = _merge_blocks([level for level in levels if level is not None])
    return _finalize_blocks(summary, statistics)


#########################
# 4/ SHARED PARENT REDUCTION
#########################


def _reduce_values(
    values: Sequence[Any],
    statistics: _Statistics,
    *,
    group_ids: Any | None = None,
    total_groups: int = 1,
    strategy: str = "auto",
    mp_config: MultiprocConfig | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Reduce one or more arrays globally or by integer group IDs.

    _resolve_strategy() chooses block summaries or complete groups.
    _aggregate_chunked() schedules the shared calculations for Dask or multiprocessing.
    Eager grouped inputs use the same _reduce_block() and _finalize_blocks() summaries, while one global group
    or exact grouped estimates use _aggregate_eager().

    The shared block helpers define the estimates, while _aggregate_chunked() documents the backend differences in
    scheduling, memory use and observation order. This parent returns a computed table for every backend.

    Statistics, strategy and mp_config follow stats().

    :param values: Nonempty sequence of numeric NumPy or Dask arrays with matching shapes and missing values
        represented by NaN. Dask inputs must have known shapes and compatible chunks.
    :param group_ids: Integer array with the same shape as the values. IDs from zero to total_groups - 1 identify
        groups, and negative IDs exclude locations. None treats all locations as one group.
    :param total_groups: Number of declared group combinations, including groups with no eligible locations.
    :returns: A computed table with integer group rows and (value position, statistic name) columns, and the resolved
        reduction strategy.
    """

    # Use matching block tasks for Dask and multiprocessing, with direct NumPy reduction otherwise
    use_dask = is_dask_array(group_ids) or any(is_dask_array(array) for array in values)
    if use_dask and mp_config is not None:
        raise ValueError("Dask inputs cannot be combined with Multiprocessing statistics.")
    chunked = use_dask or mp_config is not None
    resolved_strategy, mergeable = _resolve_strategy(statistics.aliases, strategy, total_groups, chunked)

    # Schedule blocks only when requested; eager arrays can use the same summaries without task overhead
    if chunked:
        table = _aggregate_chunked(values, group_ids, statistics, total_groups, resolved_strategy, mp_config)
    elif group_ids is None:
        table = _aggregate_eager(values, group_ids, statistics)
    elif mergeable:
        summary = _reduce_block(
            values,
            group_ids,
            total_groups,
            resolved_strategy == "dense",
            {alias for alias in statistics.aliases if alias is not None},
        )
        table = _finalize_blocks(summary, statistics)
    else:
        table = _aggregate_eager(values, group_ids, statistics)
    return table, resolved_strategy


############################
# 5/ GLOBAL OUTPUT
############################


def _global_reduction_statistics(statistics: _Statistics) -> _Statistics:
    """Keep numerical estimators in the shared reducer while leaving global-only output to its formatter."""

    selected = [
        (statistic, name, alias)
        for statistic, name, alias in zip(statistics.requested, statistics.names, statistics.aliases)
        if alias in _STATS_ALIAS_CALLABLE
    ]
    return replace(
        statistics,
        requested=[statistic for statistic, _, _ in selected],
        names=[name for _, name, _ in selected],
        aliases=[alias for _, _, alias in selected],
        single=False,
    )


def _global_values_require_eager(
    values: Sequence[Any],
    statistics: _Statistics,
    strategy: str,
    mp_config: MultiprocConfig | None,
) -> bool:
    """Identify global inputs that need their complete array for established NumPy or callable behavior."""

    # User functions receive the complete selected array, including its original dimensions
    if any(callable(statistic) for statistic in statistics.requested):
        return True
    if strategy == "groupwise" or not any(alias in _STATS_ALIAS_CALLABLE for alias in statistics.aliases):
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
                if _reader_requires_eager(array, mp_config):
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
    strategy: str,
    mp_config: MultiprocConfig | None,
) -> Any:
    """
    Calculate global statistics through the same reducer as one complete grouped bin.

    _global_reduction_statistics() selects the numerical estimates shared with grouped statistics. _reduce_values()
    calculates them with one implicit group, and _format_global_stats() restores the established global counts,
    callable results and scalar or dictionary output.

    Selected_values is returned by _sample_and_mask_global_values(); statistics, strategy and mp_config follow stats().

    :returns: Statistics for one selected value, or a mapping from value names to their statistics.
    """

    # Resolve the requested backend before removing global-only counts and callable output from the shared table
    names = list(selected_values)
    values = [selected_values[name][0] for name in names]
    counts = [selected_values[name][1] for name in names]
    chunked = any(is_dask_array(array) for array in values) or mp_config is not None
    strategy_aliases = [
        alias
        for statistic, alias in zip(statistics.requested, statistics.aliases)
        if callable(statistic) or alias in _STATS_ALIAS_CALLABLE
    ]
    resolved_strategy, _ = _resolve_strategy(strategy_aliases, strategy, 1, chunked)

    # Read complete arrays only for custom functions or the exceptional global handling of infinities
    use_eager = chunked and _global_values_require_eager(values, statistics, resolved_strategy, mp_config)
    if use_eager:
        values = _materialize_global_values(values)
    reduction_statistics = _global_reduction_statistics(statistics)
    table, _ = _reduce_values(
        values,
        reduction_statistics,
        strategy=resolved_strategy,
        mp_config=None if use_eager else mp_config,
    )

    # Finish the small global count fields and custom functions after every numerical backend returns its table
    counts = _materialize_global_counts(counts)
    return _format_global_stats(table, names, values, counts, statistics)
