# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Calculate statistics for values split into bins or categories."""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from geoutils._dispatch import is_dask_array, is_dask_dataframe
from geoutils._misc import import_optional
from geoutils.interface.raster_point import _mask_on_raster
from geoutils.sampling.cosampling import (
    _sample_vector_values,
    _sampling_specification,
    _sampling_support,
    _values_at_support,
)
from geoutils.sampling.subsampling import _dask_subsample, _subsample_numpy
from geoutils.stats.stats import (
    _STATS_ALIAS_ALL,
    _STATS_ALIAS_CALLABLE,
    _STATS_ALIAS_COUNTS,
    _get_stat_common_alias,
    _statistics,
)
from geoutils.vector.base import _as_vector

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig


__all__ = ["grouped_stats", "plot_grouped_stats"]


###################################
# 1/ GROUP MASKS CREATED ON REQUEST
###################################


class _GroupMasks(Mapping[Hashable, Any]):
    """Create Boolean masks on the result grid or points from one shared group number array."""

    def __init__(
        self,
        group_ids: Any,
        key_ids: Mapping[Hashable, int],
        shape: tuple[int, ...],
        support: Any | None,
    ) -> None:
        # Keep one group number array and follow the result table's row order
        self._group_ids = group_ids
        self._key_ids = dict(key_ids)
        self._shape = shape
        self._support = support

    def __getitem__(self, key: Hashable) -> Any:
        # Raise the usual dictionary error before creating a mask
        if key not in self._key_ids:
            raise KeyError(key)
        mask = (self._group_ids == self._key_ids[key]).reshape(self._shape)

        # Return a plain Boolean array when the input had no raster or point locations
        if self._support is None:
            return mask

        # Build a raster mask with the same grid and coordinate system as the input
        if hasattr(self._support, "ij2xy"):
            return self._support.from_array(
                data=mask,
                transform=self._support.transform,
                crs=self._support.crs,
                nodata=None,
                area_or_point=self._support.area_or_point,
                tags=self._support.tags.copy(),
            )

        # Keep point geometry and other columns while replacing the selected value with the mask
        if hasattr(self._support, "georeferenced_coords_equal") and hasattr(self._support, "data_column"):
            if self._support.data_column is not None:
                return self._support.copy(new_array=mask)

            # Add a Boolean column when the point values were stored in geometry Z coordinates
            dataframe = self._support.ds.copy()
            column = "group_mask"
            while column in dataframe.columns:
                column = f"_{column}"
            dataframe[column] = np.asarray(mask, dtype=bool)

            # Drop geometry Z values so the new Boolean column is the selected data
            dataframe.geometry = gpd.points_from_xy(
                self._support.geometry.x,
                self._support.geometry.y,
                crs=self._support.crs,
            )
            dataframe.attrs["data_column"] = column
            if getattr(self._support, "_ACCESSOR_OUTPUT", False):
                return dataframe

            # Rebuild the GeoUtils point object with the Boolean column selected
            from geoutils.pointcloud.pointcloud import PointCloud

            return PointCloud(dataframe, data_column=column)
        raise TypeError("Group masks require array, raster or point cloud support.")

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._key_ids)

    def __len__(self) -> int:
        return len(self._key_ids)


################################
# 2/ GROUP AND VALUE PREPARATION
################################


def _normalize_statistics(
    statistics: str | Callable[[Any], Any] | Iterable[str | Callable[[Any], Any]]
) -> tuple[list[str | Callable[[Any], Any]], list[str]]:
    """Check requested statistics and choose names for their result columns."""

    # Treat one name or function as a single statistic
    if isinstance(statistics, str) or callable(statistics):
        requested = [statistics]
    else:
        requested = list(statistics)

    # Expand `all` to the standard list while keeping one count column
    if requested == ["all"]:
        requested = [*_STATS_ALIAS_CALLABLE, "totalcount", "percentagevalidpoints"]
    elif "all" in requested:
        raise ValueError("Statistic 'all' cannot be combined with other statistics.")

    # Reject entries that are neither a known name nor a function
    requested = [statistic for statistic in requested if statistic != "count"]
    if any(not isinstance(statistic, str) and not callable(statistic) for statistic in requested):
        raise TypeError("statistics must contain names or callable functions.")

    # Put count first and remove a repeated count request
    names = [statistic if isinstance(statistic, str) else statistic.__name__ for statistic in requested]
    if len(set(names)) != len(names):
        raise ValueError("Statistic names must be unique.")
    return requested, ["count", *names]


def _encode_grouper(values: NDArray[Any], *, groups: pd.Index, edges: NDArray[Any] | None = None) -> NDArray[Any]:
    """Replace categories or intervals with their integer group numbers in one array block."""

    # Follow histogram edge rules and include values equal to the final upper edge
    array = np.asarray(values)
    if edges is not None:
        codes = np.searchsorted(edges, array, side="right") - 1
        codes[array == edges[-1]] = len(edges) - 2
        invalid = ~np.isfinite(array) | (codes < 0) | (codes >= len(edges) - 1)
        return np.where(invalid, -1, codes).astype(np.int64)

    # Let Pandas apply explicit interval boundaries or match category labels
    if isinstance(groups, pd.IntervalIndex):
        codes = groups.get_indexer(array.ravel())
    else:
        codes = pd.Categorical(array.ravel(), categories=groups, ordered=True).codes
    return np.asarray(codes, dtype=np.int64).reshape(array.shape)


def _prepare_groupers(
    by: Mapping[str, Any],
    *,
    bins: Mapping[str, int | Iterable[float] | pd.IntervalIndex],
    categories: Mapping[str, Iterable[Hashable]],
    mask: Any | None,
    shape: tuple[int, ...],
    use_dask: bool,
    chunks: Any | None,
) -> tuple[Any, list[pd.Index], int]:
    """Turn all grouping values into one array of combined integer group numbers."""

    # Check bin and category declarations before reading grouping values
    if not by:
        raise ValueError("by must contain at least one named grouper.")
    if any(not isinstance(name, str) or not name for name in by):
        raise ValueError("Grouper names must be non-empty strings.")
    unknown = (set(bins) | set(categories)).difference(by)
    if unknown:
        raise ValueError(f"Bin or category declarations do not match by: {sorted(unknown)!r}.")
    overlap = set(bins).intersection(categories)
    if overlap:
        raise ValueError(f"A grouper cannot define both bins and categories: {sorted(overlap)!r}.")

    # Convert all grouping arrays to Dask when any one of them uses Dask
    if use_dask:
        import_optional("dask")
        import dask.array as da

    # Start with the user mask because it applies to every group
    eligible: Any
    if mask is None:
        eligible = da.ones(shape, chunks=chunks, dtype=bool) if use_dask else np.ones(shape, dtype=bool)
    else:
        raw_mask: Any = mask.data if isinstance(mask, xr.DataArray) else mask
        if np.ma.isMaskedArray(raw_mask):
            raw_mask = np.ma.asarray(raw_mask).filled(False)
        raw_mask = da.asarray(raw_mask) if use_dask else np.asarray(raw_mask)
        if raw_mask.size != math.prod(shape) or not np.issubdtype(raw_mask.dtype, np.bool_):
            raise ValueError("mask must be Boolean and contain one value per input location.")
        eligible = raw_mask.reshape(shape)
        if use_dask:
            eligible = eligible.rechunk(chunks)
    user_eligible = eligible

    # Number each grouping variable separately in the caller's order
    encoded: list[Any] = []
    levels: list[pd.Index] = []
    for name, raw_values in by.items():
        categorical_values = raw_values if isinstance(raw_values, pd.Categorical) else None
        if isinstance(raw_values, xr.DataArray):
            raw_values = raw_values.data
        elif isinstance(raw_values, (pd.Series, pd.Index)):
            if isinstance(raw_values.dtype, pd.CategoricalDtype):
                categorical_values = pd.Categorical(raw_values)
            raw_values = raw_values.to_numpy()
        elif isinstance(raw_values, pd.Categorical):
            raw_values = np.asarray(raw_values)

        # Keep missing labels and make in-memory category values available to Pandas
        boolean_values = hasattr(raw_values, "dtype") and np.issubdtype(raw_values.dtype, np.bool_)
        if np.ma.isMaskedArray(raw_values):
            masked = np.ma.asarray(raw_values)
            fill_value = np.asarray(np.nan if np.issubdtype(masked.dtype, np.number) else None)
            raw_values = np.where(np.ma.getmaskarray(masked), fill_value, np.ma.getdata(masked))
        values: Any = raw_values if is_dask_array(raw_values) else np.asarray(raw_values)
        flattened_category = categorical_values is not None and values.ndim == 1 and values.size == math.prod(shape)
        if tuple(values.shape) != shape and not flattened_category:
            raise ValueError(f"Grouper {name!r} must contain one value per input location.")
        values = values.reshape(shape)
        if use_dask:
            values = values.rechunk(chunks) if is_dask_array(values) else da.from_array(values, chunks=chunks)

        # Use declared categories, Pandas categories, or the two Boolean values
        declared_categories: Iterable[Hashable] | None = categories.get(name)
        if declared_categories is None and categorical_values is not None:
            declared_categories = categorical_values.categories
        if declared_categories is None and (boolean_values or np.issubdtype(values.dtype, np.bool_)):
            declared_categories = (False, True)
        if declared_categories is not None:
            category_index = pd.Index(list(declared_categories))
            if category_index.empty or category_index.has_duplicates or category_index.hasnans:
                raise ValueError(f"Categories for {name!r} must be non-empty, unique and finite.")
            level = pd.CategoricalIndex(
                category_index,
                categories=category_index,
                ordered=True,
                name=name,
            )

            # Match categories in each Dask block with the same helper used for NumPy
            if is_dask_array(values):
                codes = values.map_blocks(_encode_grouper, groups=category_index, dtype=np.int64)
            else:
                codes = _encode_grouper(values, groups=category_index)
                if use_dask:
                    # Wrap integer codes in Dask when object labels cannot be split into Dask blocks directly
                    codes = da.asarray(codes)
            encoded.append(codes)
            levels.append(level)
            eligible = eligible & (codes >= 0)
            continue

        # Require bins when values are not categorical
        if name not in bins:
            raise ValueError(f"Grouper {name!r} requires an entry in bins or categories.")
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(f"Continuous grouper {name!r} must contain numeric values.")
        values = da.asarray(values) if use_dask else values
        specification = bins[name]

        # Build equal-width bins from available numeric group values allowed by the user mask
        if isinstance(specification, (int, np.integer)):
            if specification < 1:
                raise ValueError(f"The bin count for {name!r} must be positive.")
            finite = user_eligible & da.isfinite(values) if use_dask else user_eligible & np.isfinite(values)
            if use_dask:
                import dask

                lower, upper, finite_count = dask.compute(
                    da.min(da.where(finite, values, np.inf)),
                    da.max(da.where(finite, values, -np.inf)),
                    finite.sum(),
                )
            else:
                finite_values = np.asarray(values)[np.asarray(finite)]
                finite_count = finite_values.size
                lower = np.min(finite_values) if finite_count else np.nan
                upper = np.max(finite_values) if finite_count else np.nan
            if not finite_count:
                raise ValueError(f"Grouper {name!r} has no finite values inside mask.")
            if lower == upper:
                half_width = 0.5 * abs(float(lower)) if lower != 0 else 0.5
                lower, upper = lower - half_width, upper + half_width
            edges = np.linspace(float(lower), float(upper), int(specification) + 1)
            intervals = pd.IntervalIndex.from_breaks(edges, closed="left", name=name)

        # Keep an IntervalIndex's exact edges and open or closed sides
        elif isinstance(specification, pd.IntervalIndex):
            intervals = specification.rename(name)
            if intervals.empty or not intervals.is_non_overlapping_monotonic:
                raise ValueError(f"Intervals for {name!r} must be non-empty, ordered and non-overlapping.")
            if not all(np.isfinite(interval.left) and np.isfinite(interval.right) for interval in intervals):
                raise ValueError(f"Intervals for {name!r} must have finite bounds.")
            edges = None

        # Treat a numeric sequence as histogram edges and include the final upper edge
        else:
            edges = np.asarray(list(specification), dtype=float)
            if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
                raise ValueError(f"Bin edges for {name!r} must be finite and strictly increasing.")
            intervals = pd.IntervalIndex.from_breaks(edges, closed="left", name=name)

        # Number intervals in each array block and use Pandas only for custom open or closed sides
        if use_dask:
            codes = values.map_blocks(_encode_grouper, groups=intervals, edges=edges, dtype=np.int64)
        else:
            codes = _encode_grouper(values, groups=intervals, edges=edges)
        encoded.append(codes)
        levels.append(intervals)
        eligible = eligible & (codes >= 0)

    # Combine all group numbers into one array instead of storing one Boolean mask per group
    total_groups = math.prod(len(level) for level in levels)
    if total_groups > np.iinfo(np.int64).max:
        raise ValueError("The product of group counts exceeds the supported integer range.")
    group_ids = da.zeros(shape, dtype=np.int64) if use_dask else np.zeros(shape, dtype=np.int64)
    for codes, level in zip(encoded, levels):
        group_ids = group_ids * len(level) + codes
    group_ids = da.where(eligible, group_ids, -1) if use_dask else np.where(eligible, group_ids, -1)

    # Use the smallest signed integer type that can hold all groups plus the missing marker
    for dtype in (np.int8, np.int16, np.int32, np.int64):
        if total_groups - 1 <= np.iinfo(dtype).max:
            group_ids = group_ids.astype(dtype)
            break
    return group_ids, levels, total_groups


#################################
# 3/ IN-MEMORY GROUPED STATISTICS
#################################


# These statistics need only counts, sums, means, and ranges from each array block
_MERGEABLE_STATISTICS = {
    "count",
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


def _aggregate_eager(
    values: Sequence[NDArray[Any]], group_ids: NDArray[Any], statistics: Sequence[str | Callable[[Any], Any]]
) -> pd.DataFrame:
    """Sort in-memory values by group once and calculate every requested statistic."""

    # Remove values outside every group, then sort once instead of scanning the full array per group
    ids = np.asarray(group_ids).ravel()
    selected = np.flatnonzero(ids >= 0)
    order = selected[np.argsort(ids[selected], kind="stable")]
    labels, starts, sizes = np.unique(ids[order], return_index=True, return_counts=True)
    requested, names = _normalize_statistics(statistics)

    # Resolve count names once before looping over groups and value arrays
    aliases = [_get_stat_common_alias(stat, _STATS_ALIAS_ALL) if isinstance(stat, str) else None for stat in requested]
    ordinary = [stat for stat, alias in zip(requested, aliases) if alias not in _STATS_ALIAS_COUNTS]
    columns: dict[tuple[int, str], Any] = {}
    for value_index, array in enumerate(values):
        ordered = np.asarray(array).ravel()[order]
        ordered = np.where(np.isfinite(ordered), ordered, np.nan)
        results: dict[str, list[Any]] = {name: [] for name in names}

        # Pass complete group values to statistics such as median and user functions
        for start, size in zip(starts, sizes):
            group = ordered[start : start + size]
            count = int(np.count_nonzero(np.isfinite(group)))
            computed = _statistics(group, stats_name=ordinary) if count and ordinary else {}
            results["count"].append(count)
            for name, alias in zip(names[1:], aliases):
                result: Any
                if alias == "validcount":
                    result = count
                elif alias == "totalcount":
                    result = int(size)
                elif alias == "percentagevalidpoints":
                    result = 100 * count / size
                else:
                    result = computed.get(name, np.nan)
                results[name].append(result)

        # Keep numeric result columns and use integer group numbers as temporary row labels
        for name, result_values in results.items():
            columns[(value_index, name)] = np.asarray(result_values, dtype=np.int64 if name == "count" else float)
    return pd.DataFrame(columns, index=labels)


#########################################
# 4/ BLOCK STATISTICS AND COMBINATION
#########################################


def _reduce_grouped_block(
    values: Sequence[NDArray[Any]],
    group_ids: NDArray[Any],
    total_groups: int,
    dense: bool,
    statistics: set[str],
) -> tuple[NDArray[Any], NDArray[Any], dict[str, NDArray[Any]]]:
    """Summarize groups in one array block so summaries from many blocks can be combined.

    We count available values separately for every input. Group sizes also include missing values. For standard
    deviation, we keep the local mean and squared spread so large values do not hide small variation.
    """

    # Store only groups found in this block when the full result has many groups
    ids = np.asarray(group_ids).ravel()
    eligible = ids >= 0
    if dense:
        labels = np.arange(total_groups, dtype=np.int64)
        codes = ids[eligible].astype(np.int64)
    else:
        labels, codes = np.unique(ids[eligible], return_inverse=True)
    size = np.bincount(codes, minlength=len(labels))
    shape = (len(values), len(labels))
    state = {"count": np.zeros(shape, dtype=np.int64)}

    # Create only the summary arrays needed by the requested statistics
    needs_mean = bool(statistics & {"mean", "std"})
    if needs_mean:
        state["mean"] = np.zeros(shape, dtype=float)
    if "std" in statistics:
        state["m2"] = np.zeros(shape, dtype=float)
    for name in statistics & {"sum", "sumofsquares", "rmse", "min", "max"}:
        key = "sumofsquares" if name == "rmse" else name
        fill = np.inf if name == "min" else -np.inf if name == "max" else 0.0
        state[key] = np.full(shape, fill, dtype=float)

    # Reuse group numbers while handling missing data separately for each value array
    for index, array in enumerate(values):
        data = np.asarray(array).ravel()[eligible]
        finite = np.isfinite(data)
        data = data[finite].astype(float, copy=False)
        valid_codes = codes[finite]
        count = np.bincount(valid_codes, minlength=len(labels))
        state["count"][index] = count

        # Measure spread around each local mean to keep small variation accurate beside large values
        if needs_mean or "sum" in state:
            sums = np.bincount(valid_codes, weights=data, minlength=len(labels))
            if "sum" in state:
                state["sum"][index] = sums
            if needs_mean:
                mean = np.divide(sums, count, out=np.zeros(len(labels)), where=count > 0)
                state["mean"][index] = mean
            if "m2" in state:
                deviations = data - mean[valid_codes]
                state["m2"][index] = np.bincount(valid_codes, weights=deviations**2, minlength=len(labels))

        # Add counts, sums, and ranges by group with NumPy array operations
        if "sumofsquares" in state:
            state["sumofsquares"][index] = np.bincount(valid_codes, weights=data**2, minlength=len(labels))
        if "min" in state:
            np.minimum.at(state["min"][index], valid_codes, data)
        if "max" in state:
            np.maximum.at(state["max"][index], valid_codes, data)
    return labels, size, state


def _merge_grouped_blocks(
    summaries: Sequence[tuple[NDArray[Any], NDArray[Any], dict[str, NDArray[Any]]]],
) -> tuple[NDArray[Any], NDArray[Any], dict[str, NDArray[Any]]]:
    """Combine group summaries from many array blocks without losing small differences."""

    # Align group numbers when each block stored only the groups it contained
    first_labels, _, first_state = summaries[0]
    same_labels = all(np.array_equal(summary[0], first_labels) for summary in summaries[1:])
    labels = first_labels if same_labels else np.unique(np.concatenate([summary[0] for summary in summaries]))
    size = np.zeros(len(labels), dtype=np.int64)
    shape = (first_state["count"].shape[0], len(labels))
    combined: dict[str, NDArray[Any]] = {}
    for name in first_state:
        fill = np.inf if name == "min" else -np.inf if name == "max" else 0
        combined[name] = np.full(shape, fill, dtype=np.int64 if name == "count" else float)

    # Combine finite value counts before updating each mean and spread
    for block_labels, block_size, state in summaries:
        positions = slice(None) if same_labels else np.searchsorted(labels, block_labels)
        size[positions] += block_size
        old_count = combined["count"][:, positions]
        new_count = old_count + state["count"]
        if "mean" in combined:
            old_mean = combined["mean"][:, positions]
            delta = state["mean"] - old_mean
            fraction = np.divide(state["count"], new_count, out=np.zeros_like(delta), where=new_count > 0)
            combined["mean"][:, positions] = old_mean + delta * fraction
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


def _finalize_grouped_blocks(
    summary: tuple[NDArray[Any], NDArray[Any], dict[str, NDArray[Any]]],
    statistics: Sequence[str | Callable[[Any], Any]],
) -> pd.DataFrame:
    """Turn combined block summaries into the same result columns as the in-memory path."""

    # Keep groups that contain locations even when every selected value is missing
    labels, size, state = summary
    observed = size > 0
    requested, names = _normalize_statistics(statistics)
    aliases = [
        "count",
        *[_get_stat_common_alias(stat, _STATS_ALIAS_ALL) if isinstance(stat, str) else None for stat in requested],
    ]
    columns: dict[tuple[int, str], Any] = {}
    for index in range(state["count"].shape[0]):
        count = state["count"][index]
        for name, alias in zip(names, aliases):
            if alias in {"count", "validcount"}:
                result = count
            elif alias == "totalcount":
                result = size
            elif alias == "percentagevalidpoints":
                result = np.divide(100 * count, size, out=np.full(len(size), np.nan), where=size > 0)
            elif alias in {"std", "rmse"}:
                numerator = state["m2" if alias == "std" else "sumofsquares"][index]
                result = np.sqrt(np.divide(numerator, count, out=np.full(len(size), np.nan), where=count > 0))
            else:
                assert alias is not None
                result = np.where(count > 0, state[alias][index], np.nan)
            columns[(index, name)] = result[observed]
    return pd.DataFrame(columns, index=labels[observed])


def _collect_grouped_block(
    values: Sequence[NDArray[Any]], group_ids: NDArray[Any], labels: Sequence[int]
) -> tuple[NDArray[Any], list[NDArray[Any]]]:
    """Collect full group values from one block for statistics such as median or a user function."""

    # Return only group members, including missing values needed for total counts
    ids = np.asarray(group_ids).ravel()
    selected = ids == labels[0] if len(labels) == 1 else np.isin(ids, labels)
    return ids[selected], [np.asarray(array).ravel()[selected] for array in values]


def _aggregate_collected_groups(
    blocks: Sequence[tuple[NDArray[Any], list[NDArray[Any]]]], statistics: Sequence[str | Callable[[Any], Any]]
) -> pd.DataFrame:
    """Calculate exact statistics after joining a group's values from every block it crosses."""

    # Join each group's values without padding it to the full input shape
    ids = np.concatenate([block[0] for block in blocks])
    arrays = [np.concatenate([block[1][index] for block in blocks]) for index in range(len(blocks[0][1]))]
    return _aggregate_eager(arrays, ids, statistics)


########################################
# 5/ DASK AND MULTIPROCESSING EXECUTION
########################################


def _aggregate_chunked(
    values: Sequence[Any],
    group_ids: Any,
    statistics: Sequence[str | Callable[[Any], Any]],
    total_groups: int,
    strategy: str,
    mp_config: MultiprocConfig | None,
) -> pd.DataFrame:
    """Calculate grouped statistics from Dask chunks or raster tiles handled by worker processes.

    `dense` stores every declared group in each summary. `sparse` stores only groups found in that block. `groupwise`
    joins a group's full values when a statistic such as median cannot be calculated from smaller summaries.
    """

    # Return an empty table without starting tasks when the input is empty
    if group_ids.size == 0:
        return _aggregate_eager([np.empty(0) for _ in values], np.empty(0, dtype=int), statistics)

    # Split values and group numbers into matching Dask chunks or NumPy tiles
    use_dask = is_dask_array(group_ids)
    if use_dask:
        import_optional("dask")
        import dask
        import dask.array as da

        chunks = values[0].chunks
        ids = da.asarray(group_ids).rechunk(chunks)
        id_blocks = list(ids.to_delayed().ravel())
        value_blocks = [list(da.asarray(array).rechunk(chunks).to_delayed().ravel()) for array in values]
        submit = dask.delayed
        block_size = math.prod(max(lengths) for lengths in chunks)
    else:
        # Follow the requested worker tile size in every array dimension
        if mp_config is None:
            raise ValueError("Chunked NumPy aggregation requires mp_config.")
        from itertools import product

        lengths = (mp_config.chunks, mp_config.chunks) if isinstance(mp_config.chunks, int) else mp_config.chunks
        slices = [
            tuple(slice(start, min(start + lengths[axis % 2], length)) for start in range(0, length, lengths[axis % 2]))
            for axis, length in enumerate(group_ids.shape)
        ]
        tiles = list(product(*slices))
        id_blocks = [group_ids[tile] for tile in tiles]
        value_blocks = [[array[tile] for tile in tiles] for array in values]
        block_size = math.prod(lengths[axis % 2] for axis in range(group_ids.ndim))

    # Join full group values only for statistics such as median and user functions
    if strategy == "groupwise":
        if use_dask:
            memberships = list(
                dask.compute(*[dask.delayed(np.unique)(block, return_counts=True) for block in id_blocks])
            )
        else:
            memberships = [np.unique(block, return_counts=True) for block in id_blocks]
        locations: dict[int, list[int]] = {}
        sizes: dict[int, int] = {}
        for block_index, (labels, counts) in enumerate(memberships):
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
                if batch and size + sizes[label] > block_size:
                    batches.append((batch, cohort_blocks))
                    batch, size = [], 0
                batch.append(label)
                size += sizes[label]
            if batch:
                batches.append((batch, cohort_blocks))

        # Keep each batch near one block's size unless one group is larger by itself
        tables = []
        for labels, cohort_blocks in batches:
            if use_dask:
                members = [
                    dask.delayed(_collect_grouped_block)(
                        [blocks[index] for blocks in value_blocks], id_blocks[index], labels
                    )
                    for index in cohort_blocks
                ]
                table = dask.delayed(_aggregate_collected_groups)(members, statistics).compute()
            else:
                assert mp_config is not None
                handles = [
                    mp_config.cluster.submit(
                        _collect_grouped_block, [blocks[index] for blocks in value_blocks], id_blocks[index], labels
                    )
                    for index in cohort_blocks
                ]
                members = mp_config.cluster.gather(handles)
                table = _aggregate_collected_groups(members, statistics)
            tables.append(table)
        return (
            pd.concat(tables).sort_index()
            if tables
            else _aggregate_eager([np.empty(0) for _ in values], np.empty(0, dtype=int), statistics)
        )

    # Ask each block for only the counts, sums, means, or ranges needed by the request
    aliases = {_get_stat_common_alias(stat, _STATS_ALIAS_ALL) for stat in statistics if isinstance(stat, str)}
    reduction_statistics = {alias for alias in aliases if alias is not None}
    dense = strategy == "dense"
    if use_dask:
        tasks = [
            submit(_reduce_grouped_block)(
                [blocks[index] for blocks in value_blocks], block, total_groups, dense, reduction_statistics
            )
            for index, block in enumerate(id_blocks)
        ]
        while len(tasks) > 1:
            tasks = [submit(_merge_grouped_blocks)(tasks[start : start + 8]) for start in range(0, len(tasks), 8)]
        summary = tasks[0].compute()
    else:
        assert mp_config is not None
        # Combine worker results in small groups so finished summaries do not keep piling up in memory
        levels: list[Any] = []
        for start in range(0, len(id_blocks), 8):
            handles = [
                mp_config.cluster.submit(
                    _reduce_grouped_block,
                    [blocks[index] for blocks in value_blocks],
                    id_blocks[index],
                    total_groups,
                    dense,
                    reduction_statistics,
                )
                for index in range(start, min(start + 8, len(id_blocks)))
            ]
            summary = _merge_grouped_blocks(mp_config.cluster.gather(handles))
            depth = 0
            while depth < len(levels) and levels[depth] is not None:
                summary = _merge_grouped_blocks([levels[depth], summary])
                levels[depth] = None
                depth += 1
            if depth == len(levels):
                levels.append(summary)
            else:
                levels[depth] = summary
        summary = _merge_grouped_blocks([level for level in levels if level is not None])
    return _finalize_grouped_blocks(summary, statistics)


#####################################
# 6/ RESULT ASSEMBLY AND ARRAY API
#####################################


def _group_index(
    levels: Sequence[pd.Index], names: Sequence[str], group_numbers: Sequence[int] | NDArray[Any]
) -> tuple[pd.Index, dict[Hashable, int]]:
    """Build the ordered Pandas row index and map each row label to its integer group number."""

    # Split each combined number back into one number per grouping variable
    level_codes = [[] for _ in levels]
    for group_number in group_numbers:
        remainder = int(group_number)
        decoded = [0] * len(levels)
        for position in range(len(levels) - 1, -1, -1):
            decoded[position] = remainder % len(levels[position])
            remainder //= len(levels[position])
        for position, code in enumerate(decoded):
            level_codes[position].append(code)

    # Use a direct interval or category index when there is only one grouping variable
    if len(levels) == 1:
        selected = levels[0].take(level_codes[0])
        index = selected.rename(names[0])
        keys: list[Hashable] = list(index)
    else:
        index = pd.MultiIndex(levels=list(levels), codes=level_codes, names=list(names), verify_integrity=False)
        keys = list(index)
    return index, dict(zip(keys, (int(number) for number in group_numbers)))


def _compute_grouped_stats(
    values: Any | Mapping[str, Any],
    by: Mapping[str, Any],
    *,
    bins: Mapping[str, int | Iterable[float] | pd.IntervalIndex] | None,
    categories: Mapping[str, Iterable[Hashable]] | None,
    statistics: str | Callable[[Any], Any] | Iterable[str | Callable[[Any], Any]],
    mask: Any | None,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    strategy: Literal["auto", "dense", "sparse", "groupwise"],
    subsampling_strategy: Literal["sequential", "topk"],
    observed: bool,
    return_masks: bool,
    support: Any | None,
    mp_config: MultiprocConfig | None,
) -> pd.DataFrame | tuple[pd.DataFrame, Mapping[Hashable, Any]]:
    """Prepare arrays, calculate each group, and build the public Pandas result.

    _prepare_groupers() assigns every input to bins or categories. _aggregate_eager() or _aggregate_chunked() then
    calculates the requested statistics for the available storage backend. _group_index() restores the public row
    labels, and _GroupMasks creates each membership mask only if the caller reads it.
    """

    # Give a single value array its default name and keep any Dask arrays lazy
    named_values = dict(values) if isinstance(values, Mapping) else {"value": values}
    if not named_values or any(not isinstance(name, str) or not name for name in named_values):
        raise ValueError("values must contain at least one non-empty name.")
    if strategy not in {"auto", "dense", "sparse", "groupwise"}:
        raise ValueError("strategy must be 'auto', 'dense', 'sparse' or 'groupwise'.")
    if subsampling_strategy not in {"sequential", "topk"}:
        raise ValueError("subsampling_strategy must be 'sequential' or 'topk'.")
    if not isinstance(subsample, (int, float)) or subsample <= 0:
        raise ValueError("subsample must be a positive number.")
    requested_statistics, statistic_names = _normalize_statistics(statistics)

    # Check that all inputs share one shape and one Dask chunk layout for the calculation
    first_value = next(iter(named_values.values()))
    first_value = first_value.data if isinstance(first_value, xr.DataArray) else first_value
    first_value = first_value if hasattr(first_value, "shape") else np.asarray(first_value)
    shape = tuple(first_value.shape)
    if not shape:
        raise ValueError("values must contain at least one dimension.")
    all_inputs = [*named_values.values(), *by.values(), mask]
    raw_inputs = [value.data if isinstance(value, xr.DataArray) else value for value in all_inputs]
    lazy_inputs = [value for value in raw_inputs if is_dask_array(value)]
    use_dask = bool(lazy_inputs)
    chunks = lazy_inputs[0].reshape(shape).chunks if use_dask else None
    if use_dask and mp_config is not None:
        raise ValueError("Dask inputs cannot be combined with Multiprocessing grouped statistics.")
    if use_dask:
        import_optional("dask")
        import dask.array as da

    # Replace masked values with NaN and check each value array against the shared shape
    arrays: dict[str, Any] = {}
    for name, raw_values in named_values.items():
        raw_values = raw_values.data if isinstance(raw_values, xr.DataArray) else raw_values
        if np.ma.isMaskedArray(raw_values):
            raw_values = np.where(np.ma.getmaskarray(raw_values), np.nan, np.ma.getdata(raw_values))
        array = da.asarray(raw_values) if use_dask else np.asarray(raw_values)
        if tuple(array.shape) != shape:
            raise ValueError(f"Value {name!r} must match the shape of the other selected values.")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"Value {name!r} must contain numeric data.")
        arrays[name] = array.reshape(shape)
        if use_dask:
            arrays[name] = arrays[name].rechunk(chunks)

    # Assign every allowed input location to its combined group number
    group_ids, levels, total_groups = _prepare_groupers(
        by,
        bins={} if bins is None else dict(bins),
        categories={} if categories is None else dict(categories),
        mask=mask,
        shape=shape,
        use_dask=use_dask,
        chunks=chunks,
    )
    full_group_ids = group_ids

    # Select a smaller set of grouped locations before calculating statistics when requested
    if subsample != 1:
        if use_dask and len(shape) == 2:
            sampled_indices = _dask_subsample(
                da.where(group_ids >= 0, 1.0, np.nan),
                subsample=subsample,
                return_indices=True,
                random_state=random_state,
                strategy=subsampling_strategy,
            )
            rows, columns = (
                np.asarray(index.compute() if hasattr(index, "compute") else index, dtype=np.int64)
                for index in sampled_indices
            )
            flat_indices = rows * shape[1] + columns
        else:
            valid_groups = group_ids >= 0
            if use_dask:
                valid_groups = valid_groups.compute()
            (flat_indices,) = _subsample_numpy(
                np.where(np.asarray(valid_groups).ravel(), 1.0, np.nan),
                subsample=subsample,
                return_indices=True,
                random_state=random_state,
                strategy=subsampling_strategy,
            )
        selected = [
            group_ids.reshape(-1)[flat_indices],
            *(array.reshape(-1)[flat_indices] for array in arrays.values()),
        ]
        if use_dask:
            import dask

            selected = list(dask.compute(*selected))
        group_ids = np.asarray(selected[0])
        arrays = {name: np.asarray(array) for name, array in zip(arrays, selected[1:])}
        use_dask_for_aggregation = False
    else:
        use_dask_for_aggregation = use_dask

    # Choose how blocks store or gather groups from the requested statistics and group count
    aliases = [
        _get_stat_common_alias(stat, _STATS_ALIAS_ALL) if isinstance(stat, str) else None
        for stat in requested_statistics
    ]
    mergeable = all(alias in _MERGEABLE_STATISTICS for alias in aliases)
    resolved_strategy = strategy
    if strategy == "auto":
        # Store all groups for moderate counts, only seen groups for large counts, or full values when required
        if not mergeable:
            resolved_strategy = "groupwise"
        else:
            resolved_strategy = "dense" if total_groups <= 4096 else "sparse"
    chunked = use_dask_for_aggregation or (mp_config is not None and subsample == 1)
    if chunked and not mergeable and resolved_strategy != "groupwise":
        raise ValueError("Exact quantiles and custom statistics require strategy='groupwise' or 'auto'.")

    # Use the same block calculations for in-memory arrays, Dask, and worker processes
    if chunked:
        table = _aggregate_chunked(
            list(arrays.values()), group_ids, requested_statistics, total_groups, resolved_strategy, mp_config
        )
    elif mergeable:
        summary = _reduce_grouped_block(
            list(arrays.values()),
            group_ids,
            total_groups,
            resolved_strategy == "dense",
            {alias for alias in aliases if alias is not None},
        )
        table = _finalize_grouped_blocks(summary, requested_statistics)
    else:
        table = _aggregate_eager(list(arrays.values()), group_ids, requested_statistics)

    # Keep every declared group, or every group present before optional subsampling
    if not observed:
        group_numbers = np.arange(total_groups)
    elif subsample != 1:
        if is_dask_array(full_group_ids):
            import dask.array as da

            group_numbers = da.unique(full_group_ids).compute()
        else:
            group_numbers = np.unique(full_group_ids)
        group_numbers = group_numbers[group_numbers >= 0]
    else:
        group_numbers = table.index.to_numpy()
    table = table.reindex(group_numbers)

    # Restore public column and row labels; fill empty counts with zero and leave estimates as NaN
    columns = pd.MultiIndex.from_product([list(arrays), statistic_names], names=["value", "statistic"])
    table.columns = columns
    for name in arrays:
        table[(name, "count")] = table[(name, "count")].fillna(0).astype(np.int64)
        for statistic, alias in zip(statistic_names[1:], aliases):
            if alias in {"validcount", "totalcount"}:
                table[(name, statistic)] = table[(name, statistic)].fillna(0).astype(np.int64)
    index, key_ids = _group_index(levels, list(by), group_numbers)
    table.index = index
    table.attrs["grouped_stats"] = {
        "observed": observed,
        "subsample": subsample,
        "strategy": resolved_strategy,
        "subsampling_strategy": subsampling_strategy,
        "mask_membership": "groupers",
    }

    # Delay each Boolean group mask until the caller reads it from the returned mapping
    if return_masks:
        masks = _GroupMasks(full_group_ids, key_ids=key_ids, shape=shape, support=support)
        return table, masks
    return table


@overload
def grouped_stats(
    values: Any | Mapping[str, Any],
    by: Mapping[str, Any],
    *,
    bins: Mapping[str, int | Iterable[float] | pd.IntervalIndex] | None = None,
    categories: Mapping[str, Iterable[Hashable]] | None = None,
    statistics: str | Callable[[Any], Any] | Iterable[str | Callable[[Any], Any]] = ("median", "nmad"),
    mask: Any | None = None,
    subsample: int | float = 1,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["auto", "dense", "sparse", "groupwise"] = "auto",
    subsampling_strategy: Literal["sequential", "topk"] = "topk",
    observed: bool = True,
    return_masks: Literal[False] = False,
    mp_config: MultiprocConfig | None = None,
) -> pd.DataFrame: ...


@overload
def grouped_stats(
    values: Any | Mapping[str, Any],
    by: Mapping[str, Any],
    *,
    bins: Mapping[str, int | Iterable[float] | pd.IntervalIndex] | None = None,
    categories: Mapping[str, Iterable[Hashable]] | None = None,
    statistics: str | Callable[[Any], Any] | Iterable[str | Callable[[Any], Any]] = ("median", "nmad"),
    mask: Any | None = None,
    subsample: int | float = 1,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["auto", "dense", "sparse", "groupwise"] = "auto",
    subsampling_strategy: Literal["sequential", "topk"] = "topk",
    observed: bool = True,
    return_masks: Literal[True] = True,
    mp_config: MultiprocConfig | None = None,
) -> tuple[pd.DataFrame, Mapping[Hashable, Any]]: ...


def grouped_stats(
    values: Any | Mapping[str, Any],
    by: Mapping[str, Any],
    *,
    bins: Mapping[str, int | Iterable[float] | pd.IntervalIndex] | None = None,
    categories: Mapping[str, Iterable[Hashable]] | None = None,
    statistics: str | Callable[[Any], Any] | Iterable[str | Callable[[Any], Any]] = ("median", "nmad"),
    mask: Any | None = None,
    subsample: int | float = 1,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["auto", "dense", "sparse", "groupwise"] = "auto",
    subsampling_strategy: Literal["sequential", "topk"] = "topk",
    observed: bool = True,
    return_masks: bool = False,
    mp_config: MultiprocConfig | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, Mapping[Hashable, Any]]:
    """Calculate statistics for values grouped by continuous bins or discrete categories.

    Every grouper must have an entry in ``bins`` or ``categories`` unless it has a Boolean or Pandas categorical
    dtype. Numeric edge sequences use left-closed intervals and include the final right edge. Pass an
    :class:`pandas.IntervalIndex` to control edge closure explicitly. The result index follows the order of ``by``;
    columns have ``value`` and ``statistic`` levels, and a finite ``count`` is always included for each value.

    When ``return_masks`` is true, the second result behaves as a mapping from each dataframe index key to a Boolean
    array. Its masks describe complete eligible group membership after ``mask`` and valid groupers, before random
    subsampling and independently of missing selected values.

    ``dense`` stores every declared group in each chunk summary. ``sparse`` stores only groups found in that chunk.
    Both combine common statistics without keeping the original values. ``groupwise`` joins complete groups for
    statistics such as quantiles and user functions. ``auto`` selects ``groupwise`` for those cases. Otherwise it
    selects ``dense`` for up to 4096 group combinations and ``sparse`` above that. Dask and multiprocessing use the
    same NumPy calculations. ``observed`` controls which groups appear in the result. ``subsampling_strategy``
    controls which input locations are selected.

    :param values: Numeric array, or mapping of output names to arrays with matching shapes.
    :param by: Ordered mapping of grouper names to arrays with one value per input location.
    :param bins: Continuous group definitions as bin counts, numeric edges or IntervalIndexes.
    :param categories: Ordered categories for discrete groupers.
    :param statistics: Statistic name, callable or iterable accepted by geoutils.stats.get_stats() internals.
    :param mask: Boolean array defining locations eligible for grouping.
    :param subsample: Fraction when at most one, otherwise the maximum locations used for statistics.
    :param random_state: Random generator or seed used to reproduce subsampling.
    :param strategy: Group reduction strategy: ``"auto"``, ``"dense"``, ``"sparse"`` or ``"groupwise"``.
    :param subsampling_strategy: ``"topk"`` for chunk independent sampling or ``"sequential"`` for ordinary sampling.
    :param observed: Whether to omit declared group combinations with no eligible locations.
    :param return_masks: Whether to also return a lazy mapping of complete group membership masks.
    :param mp_config: Multiprocessing configuration for NumPy inputs; Dask uses its own scheduler.
    :returns: Grouped dataframe, optionally followed by its group mask mapping.
    """

    return _compute_grouped_stats(
        values,
        by,
        bins=bins,
        categories=categories,
        statistics=statistics,
        mask=mask,
        subsample=subsample,
        random_state=random_state,
        strategy=strategy,
        subsampling_strategy=subsampling_strategy,
        observed=observed,
        return_masks=return_masks,
        support=None,
        mp_config=mp_config,
    )


#####################################
# 7/ SPATIAL INPUT AND OUTPUT LOCATIONS
#####################################


def _vector_category_labels(codes: NDArray[Any], *, categories: Sequence[Hashable]) -> NDArray[Any]:
    """Restore vector feature labels from integer codes in one Dask block."""

    # Keep locations outside every feature marked as missing
    numeric = np.where(np.isfinite(codes), codes, -1).astype(np.int64)
    labels = pd.Categorical.from_codes(numeric.ravel(), categories=categories, ordered=True)
    return np.asarray(labels).reshape(codes.shape)


def _vector_group_values(
    vector: Any,
    selector: str | None,
    *,
    support: Any,
    support_dataframe: gpd.GeoDataFrame | None,
    declared_categories: Iterable[Hashable] | None,
) -> tuple[Any, list[Hashable] | None]:
    """Place vector coverage or one feature column on the chosen grid or point locations."""

    # Treat a vector without a selected column as a grouping variable for inside and outside
    if selector is None:
        count = len(vector.ds)
        values = _sample_vector_values(vector, np.ones(count), support, support_dataframe)
        return np.isfinite(values), None

    # Read feature labels once and keep the caller's declared category order
    dataframe = vector.ds
    dataframe = dataframe.compute() if is_dask_dataframe(dataframe) else dataframe
    if selector not in dataframe.columns:
        raise ValueError(f"Vector column {selector!r} does not exist.")
    if declared_categories is None:
        category_values = [value for value in pd.unique(dataframe[selector]) if not pd.isna(value)]
    else:
        category_values = list(declared_categories)
    if not category_values:
        raise ValueError(f"Vector column {selector!r} has no categories.")
    feature_codes = pd.Categorical(dataframe[selector], categories=category_values, ordered=True).codes

    # Use the same vector sampling as cosample() and keep Dask results lazy
    codes = _sample_vector_values(vector, feature_codes, support, support_dataframe)
    if is_dask_array(codes):
        categorical = codes.map_blocks(_vector_category_labels, categories=category_values, dtype=object)
    else:
        numeric_codes = np.where(np.isfinite(codes), codes, -1).astype(np.int64)
        categorical = pd.Categorical.from_codes(numeric_codes.ravel(), categories=category_values, ordered=True)
    return categorical, category_values


##########################
# 8/ PUBLIC METHOD ROUTING
##########################


def _grouped_stats(
    source: Any,
    by: Mapping[str, Any],
    *,
    values: int | str | Iterable[int | str] | Mapping[str, Any] | None,
    bins: Mapping[str, int | Iterable[float] | pd.IntervalIndex] | None,
    categories: Mapping[str, Iterable[Hashable]] | None,
    statistics: str | Callable[[Any], Any] | Iterable[str | Callable[[Any], Any]],
    at: Literal["self"] | Any | None,
    mask: Any | None,
    mask_mode: Literal["inside", "outside"],
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    strategy: Literal["auto", "dense", "sparse", "groupwise"],
    subsampling_strategy: Literal["sequential", "topk"],
    interpolation: str,
    align: Literal["raise", "reproject"],
    observed: bool,
    return_masks: bool,
    mp_config: MultiprocConfig | None,
) -> pd.DataFrame | tuple[pd.DataFrame, Mapping[Hashable, Any]]:
    """Place object values and groups at common locations before calling _compute_grouped_stats().

    _sampling_support() chooses the output grid or points, while _values_at_support() places each selected value on
    those locations. Vector groups use _vector_group_values() before the shared array calculation.
    """

    # Check spatial options before reading any possibly lazy data
    if mask_mode not in {"inside", "outside"}:
        raise ValueError("mask_mode must be 'inside' or 'outside'.")
    if align not in {"raise", "reproject"}:
        raise ValueError("align must be 'raise' or 'reproject'.")
    if isinstance(at, str) and at != "self":
        raise ValueError("at must be 'self' or a raster or point cloud support object.")
    # Choose the same output grid or point locations as cosample()
    inputs = [source, *by.values(), *(values.values() if isinstance(values, Mapping) else [])]
    support = _sampling_support(inputs, source if isinstance(at, str) else at)
    raster_support = support if hasattr(support, "ij2xy") else None
    point_support = None if raster_support is not None else support
    support_dataframe = None if point_support is None else point_support.ds

    # Turn requested raster bands or point columns into named output values
    source_raster = source if hasattr(source, "ij2xy") else getattr(source, "rst", None)
    source_pointcloud = (
        source
        if hasattr(source, "georeferenced_coords_equal") and hasattr(source, "data_column")
        else getattr(source, "pc", None)
    )
    value_specs: dict[str, Any]
    if source_raster is not None:
        if values is None:
            value_specs = {f"band_{band}": band for band in range(1, source_raster.count + 1)}
        elif isinstance(values, Mapping):
            value_specs = dict(values)
        elif isinstance(values, (int, np.integer)):
            value_specs = {f"band_{int(values)}": int(values)}
        elif isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
            value_specs = {f"band_{int(band)}": int(band) for band in values}
        else:
            raise TypeError("Raster values must select one or more integer band numbers.")
    elif source_pointcloud is not None:
        default_column = source_pointcloud.data_column
        if values is None:
            value_specs = {default_column if default_column is not None else "z": default_column}
        elif isinstance(values, Mapping):
            value_specs = dict(values)
        elif isinstance(values, str):
            value_specs = {values: values}
        elif isinstance(values, Iterable):
            value_specs = {}
            for column in values:
                if not isinstance(column, str):
                    raise TypeError("Point cloud values must select column names.")
                value_specs[column] = column
        else:
            raise TypeError("Point cloud values must select one or more column names.")
    else:
        raise TypeError("grouped_stats is only available on raster and point cloud objects.")

    # Read every selected value at the chosen locations and keep each value's missing data separate
    selected_values: dict[str, Any] = {}
    for name, value_selector in value_specs.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Selected value names must be non-empty strings.")
        value_source, value_selector = _sampling_specification(source, value_selector)
        selected_values[name] = _values_at_support(
            value_source,
            value_selector,
            owner=source,
            support=support,
            support_dataframe=support_dataframe,
            name=name,
            interpolation=interpolation,
            align=align,
            mp_config=mp_config,
            preserve_lazy=True,
        )

    # Turn every grouping request into values at the chosen locations
    selected_groupers: dict[str, Any] = {}
    resolved_categories = {} if categories is None else dict(categories)
    for name, specification in by.items():
        group_source, group_selector = _sampling_specification(source, specification)

        # Detect a PointCloud before its shared Vector behavior
        group_raster = group_source if hasattr(group_source, "ij2xy") else getattr(group_source, "rst", None)
        group_pointcloud = (
            group_source
            if hasattr(group_source, "georeferenced_coords_equal") and hasattr(group_source, "data_column")
            else getattr(group_source, "pc", None)
        )
        vector = _as_vector(group_source) if group_raster is None and group_pointcloud is None else None
        if vector is not None and name not in (bins or {}):
            if group_selector is not None and not isinstance(group_selector, str):
                raise TypeError(f"Vector selector for {name!r} must be a column name.")
            grouped_values, inferred_categories = _vector_group_values(
                vector,
                group_selector,
                support=support,
                support_dataframe=support_dataframe,
                declared_categories=resolved_categories.get(name),
            )
            selected_groupers[name] = grouped_values
            if inferred_categories is not None:
                resolved_categories[name] = inferred_categories
            continue

        selected_groupers[name] = _values_at_support(
            group_source,
            group_selector,
            owner=source,
            support=support,
            support_dataframe=support_dataframe,
            name=name,
            interpolation="nearest" if name in resolved_categories else interpolation,
            align=align,
            mp_config=mp_config,
            preserve_lazy=True,
        )

    # Read the user mask without removing a group merely because one selected value is missing
    if raster_support is not None:
        support_mask = _mask_on_raster(mask, support, mask_mode, align)
    elif mask is None:
        support_mask = None
    else:
        mask_raster = mask if hasattr(mask, "ij2xy") else getattr(mask, "rst", None)
        mask_pointcloud = (
            mask
            if hasattr(mask, "georeferenced_coords_equal") and hasattr(mask, "data_column")
            else getattr(mask, "pc", None)
        )
        vector = _as_vector(mask) if mask_raster is None and mask_pointcloud is None else None
        if vector is not None:
            support_mask = np.isfinite(
                _sample_vector_values(vector, np.ones(len(vector.ds)), support, support_dataframe)
            )
            if mask_mode == "outside":
                support_mask = ~support_mask
        elif mask_raster is not None or mask_pointcloud is not None:
            mask_values = _values_at_support(
                mask_raster if mask_raster is not None else mask_pointcloud,
                1 if mask_raster is not None else None,
                owner=mask,
                support=support,
                support_dataframe=support_dataframe,
                name="mask",
                interpolation="nearest",
                align=align,
                mp_config=mp_config,
                preserve_lazy=True,
            )
            support_mask = np.isfinite(mask_values) & (mask_values != 0)
        else:
            support_mask = mask.squeeze() if is_dask_array(mask) else np.atleast_1d(np.asanyarray(mask).squeeze())
            if np.ma.isMaskedArray(support_mask):
                support_mask = support_mask.filled(False)
            if support_dataframe is None:
                raise RuntimeError("Point support coordinates were not prepared.")
            if support_mask.dtype != bool or len(support_mask) != len(support_dataframe):
                raise ValueError("A point support mask must be Boolean with one value per point.")

    # Calculate the table and keep the grid or points available for returned group masks
    return _compute_grouped_stats(
        selected_values,
        selected_groupers,
        bins=bins,
        categories=resolved_categories,
        statistics=statistics,
        mask=support_mask,
        subsample=subsample,
        random_state=random_state,
        strategy=strategy,
        subsampling_strategy=subsampling_strategy,
        observed=observed,
        return_masks=return_masks,
        support=support,
        mp_config=mp_config,
    )


##########################
# 9/ GROUPED STAT PLOTTING
##########################


def _plot_axis(index: pd.Index) -> tuple[NDArray[Any], NDArray[Any], list[str] | None]:
    """Return plot positions and optional text labels for one grouping variable."""

    # Draw adjoining numeric intervals with their true widths
    if isinstance(index, pd.IntervalIndex) and len(index) > 0:
        adjacent = len(index) == 1 or np.all(np.asarray(index.right[:-1]) == np.asarray(index.left[1:]))
        if adjacent:
            edges = np.asarray([index[0].left, *index.right], dtype=float)
            return edges, np.asarray(index.mid, dtype=float), None

    # Give categories and separated intervals equal widths
    edges = np.arange(len(index) + 1, dtype=float)
    centers = edges[:-1] + 0.5
    return edges, centers, [str(value) for value in index]


def plot_grouped_stats(
    table: pd.DataFrame,
    *,
    value: str | None = None,
    statistic: str = "nmad",
    min_count: int = 0,
    cmap: Any = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    ax: Any | None = None,
    savefig_fname: str | None = None,
) -> Mapping[str, Any]:
    """Plot grouped statistics for one or two grouping variables with their sample counts.

    One grouping variable produces a statistic curve below its counts. Two grouping variables produce a colored
    grid, with counts above and to the right. Adjoining intervals keep their numeric widths. Categories use equal
    widths.

    :param table: Dataframe returned by grouped_stats() or an object grouped_stats() method.
    :param value: Selected value column. It may be omitted when the table contains one value.
    :param statistic: Statistic column to display.
    :param min_count: Hide statistic cells with fewer finite observations.
    :param cmap: Matplotlib colormap used for a two-dimensional statistic grid.
    :param vmin: Lower color limit for a two-dimensional statistic grid.
    :param vmax: Upper color limit for a two-dimensional statistic grid.
    :param ax: Optional Matplotlib axes whose area is divided into the plot panels.
    :param savefig_fname: Optional path used to save the completed figure.
    :returns: Mapping naming the Matplotlib axes created for each panel.
    """

    # Import Matplotlib only when the caller requests a plot
    matplotlib = import_optional("matplotlib")
    import matplotlib.pyplot as plt

    if not isinstance(table, pd.DataFrame) or not isinstance(table.columns, pd.MultiIndex):
        raise TypeError("table must be a grouped_stats dataframe with MultiIndex columns.")
    if list(table.columns.names) != ["value", "statistic"]:
        raise ValueError("table columns must have 'value' and 'statistic' levels.")
    available_values = list(dict.fromkeys(table.columns.get_level_values("value")))
    if value is None:
        if len(available_values) != 1:
            raise ValueError("value must be selected when the table contains multiple values.")
        value = available_values[0]
    if (value, statistic) not in table.columns or (value, "count") not in table.columns:
        raise ValueError(f"Value {value!r} must contain both {statistic!r} and 'count' statistics.")
    if table.index.nlevels not in {1, 2}:
        raise ValueError("plot_grouped_stats supports one or two group dimensions.")
    if min_count < 0:
        raise ValueError("min_count cannot be negative.")

    # Use the supplied axes as the plot area or create a new figure
    if ax is None:
        figure = plt.figure(figsize=(7, 6))
        frame = figure.add_axes((0.1, 0.1, 0.8, 0.8))
    elif isinstance(ax, matplotlib.axes.Axes):
        frame = ax
        figure = ax.figure
    else:
        raise TypeError("ax must be a Matplotlib Axes or None.")
    frame.set_axis_off()

    # Draw counts above the statistic for one grouping variable
    if table.index.nlevels == 1:
        count_axis = frame.inset_axes((0.0, 0.72, 1.0, 0.28))
        statistic_axis = frame.inset_axes((0.0, 0.0, 1.0, 0.64))
        edges, centers, labels = _plot_axis(table.index)
        counts = table[(value, "count")].to_numpy(dtype=float)
        values = table[(value, statistic)].where(table[(value, "count")] >= min_count).to_numpy(dtype=float)

        # Match count-bar widths to the numeric intervals or category positions
        count_axis.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="0.7", edgecolor="white")
        count_axis.set_xlim(edges[0], edges[-1])
        count_axis.set_ylabel("Count")
        count_axis.tick_params(axis="x", labelbottom=False)
        statistic_axis.plot(centers, values, marker="o")
        statistic_axis.set(xlim=(edges[0], edges[-1]), xlabel=table.index.name, ylabel=statistic)
        if labels is not None:
            statistic_axis.set_xticks(centers, labels, rotation=45, ha="right")
        axes = {"count": count_axis, "statistic": statistic_axis}

    # Draw a two-dimensional statistic grid with totals for each row and column
    else:
        statistic_axis = frame.inset_axes((0.0, 0.0, 0.68, 0.66))
        count_x_axis = frame.inset_axes((0.0, 0.72, 0.68, 0.28))
        count_y_axis = frame.inset_axes((0.74, 0.0, 0.26, 0.66))
        level_x, level_y = table.index.levels
        edges_x, centers_x, labels_x = _plot_axis(level_x)
        edges_y, centers_y, labels_y = _plot_axis(level_y)

        # Add every declared group combination so groups with no data appear as gaps
        full_index = pd.MultiIndex.from_product([level_x, level_y], names=table.index.names)
        counts = table[(value, "count")].reindex(full_index).unstack(level=1)
        plotted = table[(value, statistic)].where(table[(value, "count")] >= min_count)
        plotted = plotted.reindex(full_index).unstack(level=1)
        mesh = statistic_axis.pcolormesh(
            edges_x,
            edges_y,
            plotted.to_numpy(dtype=float).T,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )

        # Draw row and column totals with the same widths as the statistic grid
        counts_x = counts.sum(axis=1, skipna=True).to_numpy(dtype=float)
        counts_y = counts.sum(axis=0, skipna=True).to_numpy(dtype=float)
        count_x_axis.bar(edges_x[:-1], counts_x, width=np.diff(edges_x), align="edge", color="0.7", edgecolor="white")
        count_y_axis.barh(edges_y[:-1], counts_y, height=np.diff(edges_y), align="edge", color="0.7", edgecolor="white")
        count_x_axis.set(xlim=(edges_x[0], edges_x[-1]), ylabel="Count")
        count_y_axis.set(ylim=(edges_y[0], edges_y[-1]), xlabel="Count")
        count_x_axis.tick_params(axis="x", labelbottom=False)
        count_y_axis.tick_params(axis="y", labelleft=False)
        statistic_axis.set(
            xlim=(edges_x[0], edges_x[-1]),
            ylim=(edges_y[0], edges_y[-1]),
            xlabel=table.index.names[0],
            ylabel=table.index.names[1],
        )
        if labels_x is not None:
            statistic_axis.set_xticks(centers_x, labels_x, rotation=45, ha="right")
        if labels_y is not None:
            statistic_axis.set_yticks(centers_y, labels_y)
        colorbar = figure.colorbar(mesh, ax=statistic_axis, label=statistic)
        axes = {
            "count_x": count_x_axis,
            "count_y": count_y_axis,
            "statistic": statistic_axis,
            "colorbar": colorbar.ax,
        }

    # Save after every panel and label has been added
    if savefig_fname is not None:
        figure.savefig(savefig_fname, bbox_inches="tight")
    return axes
