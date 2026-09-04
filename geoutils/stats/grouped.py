# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Statistics grouped by continuous bins or discrete categories.

The public array function returns an ordinary Pandas dataframe whose index preserves interval and categorical
metadata. Raster and point cloud methods prepare values on a common spatial support before using the same engine.
Optional group masks are exposed through a lightweight mapping backed by one integer group layer.

The module first defines the mask view and prepares group membership, then computes array statistics. Spatial input
alignment follows before the object method dispatcher, with the plotting helper kept in the final section.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, overload

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from geoutils._dispatch import is_dask_array, is_dask_dataframe
from geoutils._misc import import_optional
from geoutils.interface.raster_point import _aligned_raster, _mask_on_raster
from geoutils.raster.array import _selected_raster_data
from geoutils.sampling.subsampling import _dask_subsample, _subsample_numpy
from geoutils.stats.stats import _STATS_ALIAS_CALLABLE, _statistics
from geoutils.vector.base import _as_vector

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig


Statistic: TypeAlias = str | Callable[[Any], Any]
BinSpec: TypeAlias = int | Iterable[float] | pd.IntervalIndex
GroupedStatsResult: TypeAlias = pd.DataFrame | tuple[pd.DataFrame, Mapping[Hashable, Any]]

__all__ = ["grouped_stats", "plot_grouped_stats"]


#########################
# 1/ LAZY GROUP MASK VIEW
#########################


class _GroupMasks(Mapping[Hashable, Any]):
    """Create support aligned Boolean masks from one shared group code layer."""

    def __init__(
        self,
        group_ids: Any,
        key_ids: Mapping[Hashable, int],
        shape: tuple[int, ...],
        support: Any | None,
    ) -> None:
        # Retain only one group layer while preserving the dataframe index order
        self._group_ids = group_ids
        self._key_ids = dict(key_ids)
        self._shape = shape
        self._support = support

    def __getitem__(self, key: Hashable) -> Any:
        # Raise the standard mapping error before allocating the requested mask
        if key not in self._key_ids:
            raise KeyError(key)
        mask = (self._group_ids == self._key_ids[key]).reshape(self._shape)

        # Return a plain Boolean array when no spatial support was supplied
        if self._support is None:
            return mask

        # Rebuild raster outputs through their native constructor to retain georeferencing
        if hasattr(self._support, "ij2xy"):
            return self._support.from_array(
                data=mask,
                transform=self._support.transform,
                crs=self._support.crs,
                nodata=None,
                area_or_point=self._support.area_or_point,
                tags=self._support.tags.copy(),
            )

        # Preserve point geometry and auxiliary columns while replacing the main data values
        if hasattr(self._support, "georeferenced_coords_equal") and hasattr(self._support, "data_column"):
            if self._support.data_column is not None:
                return self._support.copy(new_array=mask)

            # Add a Boolean data column when the source stores values as numeric geometry elevations
            dataframe = self._support.ds.copy()
            column = "group_mask"
            while column in dataframe.columns:
                column = f"_{column}"
            dataframe[column] = np.asarray(mask, dtype=bool)

            # Move values out of three dimensional geometry so the Boolean column is authoritative
            dataframe.geometry = gpd.points_from_xy(
                self._support.geometry.x,
                self._support.geometry.y,
                crs=self._support.crs,
            )
            dataframe.attrs["data_column"] = column
            if getattr(self._support, "_ACCESSOR_OUTPUT", False):
                return dataframe

            # Rebuild a GeoUtils object so its selected value column is recognized as a mask
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


def _normalize_statistics(statistics: Statistic | Iterable[Statistic]) -> tuple[list[Statistic], list[str]]:
    """Validate requested statistics and derive stable dataframe labels."""

    # Treat a single name or callable as one statistic rather than an iterable
    if isinstance(statistics, str) or callable(statistics):
        requested = [statistics]
    else:
        requested = list(statistics)

    # Expand the established complete selection without duplicating the mandatory count
    if requested == ["all"]:
        requested = [*_STATS_ALIAS_CALLABLE, "totalcount", "percentagevalidpoints"]
    elif "all" in requested:
        raise ValueError("Statistic 'all' cannot be combined with other statistics.")

    # Reject invalid entries before deriving callable names
    requested = [statistic for statistic in requested if statistic != "count"]
    if any(not isinstance(statistic, str) and not callable(statistic) for statistic in requested):
        raise TypeError("statistics must contain names or callable functions.")

    # Keep count first and ignore a duplicate explicitly requested by the caller
    names = [statistic if isinstance(statistic, str) else statistic.__name__ for statistic in requested]
    if len(set(names)) != len(names):
        raise ValueError("Statistic names must be unique.")
    return requested, ["count", *names]


def _prepare_groupers(
    by: Mapping[str, Any],
    *,
    bins: Mapping[str, BinSpec],
    categories: Mapping[str, Iterable[Hashable]],
    mask: Any | None,
    shape: tuple[int, ...],
    use_dask: bool,
) -> tuple[Any, list[pd.Index], int]:
    """Encode every grouper and combine its codes into one integer layer."""

    # Validate declarations before deriving categories or scanning continuous values
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

    # Use one backend for membership so mixed NumPy and Dask inputs remain aligned
    if use_dask:
        import_optional("dask")
        import dask.array as da

    # Start from the user mask because it limits every group consistently
    if mask is None:
        eligible = da.ones(shape, dtype=bool) if use_dask else np.ones(shape, dtype=bool)
    else:
        raw_mask = mask.data if isinstance(mask, xr.DataArray) else mask
        if np.ma.isMaskedArray(raw_mask):
            raw_mask = np.ma.asarray(raw_mask).filled(False)
        raw_mask = da.asarray(raw_mask) if use_dask else np.asarray(raw_mask)
        if raw_mask.size != math.prod(shape) or not np.issubdtype(raw_mask.dtype, np.bool_):
            raise ValueError("mask must be Boolean and contain one value per input location.")
        eligible = raw_mask.reshape(shape)
    user_eligible = eligible

    # Encode groupers separately so their declared order defines result ordering
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

        # Preserve masked cells as missing values before selecting the common backend
        boolean_values = hasattr(raw_values, "dtype") and np.issubdtype(raw_values.dtype, np.bool_)
        if np.ma.isMaskedArray(raw_values):
            masked = np.ma.asarray(raw_values)
            fill_value = np.asarray(np.nan if np.issubdtype(masked.dtype, np.number) else None)
            raw_values = np.where(np.ma.getmaskarray(masked), fill_value, np.ma.getdata(masked))
        values = da.asarray(raw_values) if use_dask else np.asarray(raw_values)
        flattened_category = categorical_values is not None and values.ndim == 1 and values.size == math.prod(shape)
        if tuple(values.shape) != shape and not flattened_category:
            raise ValueError(f"Grouper {name!r} must contain one value per input location.")
        values = values.reshape(shape)

        # Use declared categories, native Pandas categories or the unambiguous Boolean categories
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

            # Let Pandas encode eager data because it handles strings and missing objects consistently
            if use_dask:
                codes = da.full(shape, -1, dtype=np.int64)
                for code, category in enumerate(category_index):
                    codes = da.where(values == category, code, codes)
            else:
                codes = pd.Categorical(
                    np.asarray(values).ravel(),
                    categories=category_index,
                    ordered=True,
                ).codes.reshape(shape)
            encoded.append(codes)
            levels.append(level)
            eligible = eligible & (codes >= 0)
            continue

        # Require an explicit bin definition for every remaining grouper
        if name not in bins:
            raise ValueError(f"Grouper {name!r} requires an entry in bins or categories.")
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(f"Continuous grouper {name!r} must contain numeric values.")
        specification = bins[name]

        # Derive equal width edges from finite values retained by the user mask
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

        # Respect an IntervalIndex exactly, including its chosen edge closure
        elif isinstance(specification, pd.IntervalIndex):
            intervals = specification.rename(name)
            if intervals.empty or not intervals.is_non_overlapping_monotonic:
                raise ValueError(f"Intervals for {name!r} must be non-empty, ordered and non-overlapping.")
            if not all(np.isfinite(interval.left) and np.isfinite(interval.right) for interval in intervals):
                raise ValueError(f"Intervals for {name!r} must have finite bounds.")
            edges = None

        # Interpret numeric sequences as histogram edges with the final edge included
        else:
            edges = np.asarray(list(specification), dtype=float)
            if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
                raise ValueError(f"Bin edges for {name!r} must be finite and strictly increasing.")
            intervals = pd.IntervalIndex.from_breaks(edges, closed="left", name=name)

        # Compare against intervals directly to support every Pandas closure convention
        codes = da.full(shape, -1, dtype=np.int64) if use_dask else np.full(shape, -1, dtype=np.int64)
        for code, interval in enumerate(intervals):
            left = values >= interval.left if interval.closed_left else values > interval.left
            right = values <= interval.right if interval.closed_right else values < interval.right
            if edges is not None and code == len(intervals) - 1:
                right = values <= interval.right
            codes = da.where(left & right, code, codes) if use_dask else np.where(left & right, code, codes)
        encoded.append(codes)
        levels.append(intervals)
        eligible = eligible & (codes >= 0)

    # Combine ordered codes without storing a Boolean layer for every group
    total_groups = math.prod(len(level) for level in levels)
    if total_groups > np.iinfo(np.int64).max:
        raise ValueError("The product of group counts exceeds the supported integer range.")
    group_ids = da.zeros(shape, dtype=np.int64) if use_dask else np.zeros(shape, dtype=np.int64)
    for codes, level in zip(encoded, levels):
        group_ids = group_ids * len(level) + codes
    group_ids = da.where(eligible, group_ids, -1) if use_dask else np.where(eligible, group_ids, -1)

    # Retain the smallest signed integer layer that can represent every group and invalid membership
    for dtype in (np.int8, np.int16, np.int32, np.int64):
        if total_groups - 1 <= np.iinfo(dtype).max:
            group_ids = group_ids.astype(dtype)
            break
    return group_ids, levels, total_groups


###############################
# 3/ GROUPED ARRAY STATISTICS
###############################


def _aggregate_eager(
    values: Mapping[str, NDArray[Any]],
    group_ids: NDArray[Any],
    group_numbers: Sequence[int],
    statistics: Sequence[Statistic],
    statistic_names: Sequence[str],
) -> dict[int, dict[tuple[str, str], Any]]:
    """Compute all requested statistics from eager arrays by observed group."""

    records: dict[int, dict[tuple[str, str], Any]] = {}
    for group_number in group_numbers:
        membership = group_ids == group_number
        group_size = int(np.count_nonzero(membership))
        record: dict[tuple[str, str], Any] = {}
        for value_name, array in values.items():
            group_values = array[membership]
            valid_count = int(np.count_nonzero(np.isfinite(group_values)))
            record[(value_name, "count")] = valid_count

            # Replace every nonfinite value so numeric summaries consistently ignore infinities
            group_values = np.where(np.isfinite(group_values), group_values, np.nan)

            # Reuse GeoUtils statistic aliases while correcting group specific count metadata
            ordinary_statistics = [
                statistic
                for statistic in statistics
                if not isinstance(statistic, str)
                or "".join(statistic.lower().replace("_", "").split())
                not in {"validcount", "totalcount", "percentagevalidpoints"}
            ]
            computed = (
                _statistics(group_values, stats_name=ordinary_statistics)
                if valid_count > 0 and ordinary_statistics
                else {}
            )
            for statistic, statistic_name in zip(statistics, statistic_names[1:]):
                normalized = "".join(statistic.lower().replace("_", "").split()) if isinstance(statistic, str) else ""
                result: Any
                if normalized == "validcount":
                    result = valid_count
                elif normalized == "totalcount":
                    result = group_size
                elif normalized == "percentagevalidpoints":
                    result = 100 * valid_count / group_size if group_size else np.nan
                elif valid_count == 0:
                    result = np.nan
                else:
                    result = computed[statistic_name]
                record[(value_name, statistic_name)] = result
        records[int(group_number)] = record
    return records


def _aggregate_dask(
    values: Mapping[str, Any],
    group_ids: Any,
    group_numbers: Sequence[int],
    statistics: Sequence[Statistic],
    statistic_names: Sequence[str],
) -> dict[int, dict[tuple[str, str], Any]]:
    """Build grouped Dask reductions and compute their scalar results together."""

    import_optional("dask")
    import dask
    import dask.array as da

    # Collect every scalar in one graph execution to share group membership tasks
    tasks: list[Any] = []
    locations: list[tuple[int, str, str]] = []
    for group_number in group_numbers:
        membership = group_ids == group_number
        group_size = membership.sum()
        for value_name, array in values.items():
            finite_membership = membership & da.isfinite(array)
            grouped_values = da.where(finite_membership, array, np.nan)
            valid_count = finite_membership.sum()
            tasks.append(valid_count)
            locations.append((int(group_number), value_name, "count"))

            # Use direct count reductions because a full shaped masked array has a larger total size
            ordinary_statistics: list[Statistic] = []
            ordinary_names: list[str] = []
            for statistic, statistic_name in zip(statistics, statistic_names[1:]):
                normalized = "".join(statistic.lower().replace("_", "").split()) if isinstance(statistic, str) else ""
                if normalized == "validcount":
                    task = valid_count
                elif normalized == "totalcount":
                    task = group_size
                elif normalized == "percentagevalidpoints":
                    task = da.where(group_size > 0, 100 * valid_count / group_size, np.nan)
                else:
                    ordinary_statistics.append(statistic)
                    ordinary_names.append(statistic_name)
                    continue
                tasks.append(task)
                locations.append((int(group_number), value_name, statistic_name))

            # Delegate numeric summaries and custom callables to the established lazy statistic engine
            if ordinary_statistics:
                computed = _statistics(grouped_values, stats_name=list(ordinary_statistics))
                for statistic_name in ordinary_names:
                    tasks.append(computed[statistic_name])
                    locations.append((int(group_number), value_name, statistic_name))

    # Populate records in the same order as the dataframe columns
    results = dask.compute(*tasks)
    records: dict[int, dict[tuple[str, str], Any]] = {int(number): {} for number in group_numbers}
    for (group_number, value_name, statistic_name), result in zip(locations, results):
        records[group_number][(value_name, statistic_name)] = (
            result.item() if isinstance(result, np.generic) else result
        )
    return records


def _group_index(
    levels: Sequence[pd.Index], names: Sequence[str], group_numbers: Sequence[int]
) -> tuple[pd.Index, dict[Hashable, int]]:
    """Construct an ordered Pandas index and its corresponding group code lookup."""

    # Decode combined group numbers back to one code per declared grouper
    level_codes = [[] for _ in levels]
    for group_number in group_numbers:
        remainder = int(group_number)
        decoded = [0] * len(levels)
        for position in range(len(levels) - 1, -1, -1):
            decoded[position] = remainder % len(levels[position])
            remainder //= len(levels[position])
        for position, code in enumerate(decoded):
            level_codes[position].append(code)

    # Preserve a direct IntervalIndex or CategoricalIndex for one dimensional results
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
    bins: Mapping[str, BinSpec] | None,
    categories: Mapping[str, Iterable[Hashable]] | None,
    statistics: Statistic | Iterable[Statistic],
    mask: Any | None,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    strategy: Literal["sequential", "topk"],
    observed: bool,
    return_masks: bool,
    support: Any | None,
) -> GroupedStatsResult:
    """Prepare common arrays, aggregate groups and assemble the public result."""

    # Normalize named values while retaining their lazy array backends
    named_values = dict(values) if isinstance(values, Mapping) else {"value": values}
    if not named_values or any(not isinstance(name, str) or not name for name in named_values):
        raise ValueError("values must contain at least one non-empty name.")
    if strategy not in {"sequential", "topk"}:
        raise ValueError("strategy must be 'sequential' or 'topk'.")
    if not isinstance(subsample, (int, float)) or subsample <= 0:
        raise ValueError("subsample must be a positive number.")
    requested_statistics, statistic_names = _normalize_statistics(statistics)

    # Derive the common shape before flattening spatial dimensions for aggregation
    first_value = next(iter(named_values.values()))
    first_value = first_value.data if isinstance(first_value, xr.DataArray) else first_value
    first_value = first_value if hasattr(first_value, "shape") else np.asarray(first_value)
    shape = tuple(first_value.shape)
    if not shape:
        raise ValueError("values must contain at least one dimension.")
    all_inputs = [*named_values.values(), *by.values(), mask]
    use_dask = any(
        is_dask_array(value.data if isinstance(value, xr.DataArray) else value)
        for value in all_inputs
        if value is not None
    )
    if use_dask:
        import_optional("dask")
        import dask.array as da

    # Convert masked values to NaN and validate every selected value against the support
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

    # Encode full group membership before drawing any optional statistic subsample
    group_ids, levels, total_groups = _prepare_groupers(
        by,
        bins={} if bins is None else dict(bins),
        categories={} if categories is None else dict(categories),
        mask=mask,
        shape=shape,
        use_dask=use_dask,
    )
    full_group_ids = group_ids

    # Bound eager aggregation by selecting common group valid locations when requested
    if subsample != 1:
        if use_dask and len(shape) == 2:
            sampled_indices = _dask_subsample(
                da.where(group_ids >= 0, 1.0, np.nan),
                subsample=subsample,
                return_indices=True,
                random_state=random_state,
                strategy=strategy,
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
                strategy=strategy,
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

    # Base observed rows on complete support so sampling cannot silently remove a declared group
    if observed:
        if is_dask_array(full_group_ids):
            import dask.array as da

            observed_groups = np.asarray(
                da.unique(full_group_ids[full_group_ids >= 0]).compute(),
                dtype=np.int64,
            )
        else:
            complete_ids = np.asarray(full_group_ids)
            observed_groups = np.unique(complete_ids[complete_ids >= 0])
        group_numbers = [int(number) for number in observed_groups]
    else:
        group_numbers = list(range(total_groups))

    # Compute group summaries through the matching eager or lazy backend
    if use_dask_for_aggregation:
        records = _aggregate_dask(arrays, group_ids, group_numbers, requested_statistics, statistic_names)
    else:
        records = _aggregate_eager(
            {name: np.asarray(array) for name, array in arrays.items()},
            np.asarray(group_ids),
            group_numbers,
            requested_statistics,
            statistic_names,
        )

    # Assemble a stable two level column index for one or many selected values
    columns = pd.MultiIndex.from_product(
        [list(arrays), statistic_names],
        names=["value", "statistic"],
    )
    index, key_ids = _group_index(levels, list(by), group_numbers)
    table = pd.DataFrame(
        [[records[number].get(column, np.nan) for column in columns] for number in group_numbers],
        index=index,
        columns=columns,
    )
    table.attrs["grouped_stats"] = {
        "observed": observed,
        "subsample": subsample,
        "mask_membership": "groupers",
    }

    # Materialize no Boolean group layers until the caller accesses a mapping key
    if return_masks:
        masks = _GroupMasks(full_group_ids, key_ids=key_ids, shape=shape, support=support)
        return table, masks
    return table


@overload
def grouped_stats(
    values: Any | Mapping[str, Any],
    by: Mapping[str, Any],
    *,
    bins: Mapping[str, BinSpec] | None = None,
    categories: Mapping[str, Iterable[Hashable]] | None = None,
    statistics: Statistic | Iterable[Statistic] = ("median", "nmad"),
    mask: Any | None = None,
    subsample: int | float = 1,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "topk",
    observed: bool = True,
    return_masks: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def grouped_stats(
    values: Any | Mapping[str, Any],
    by: Mapping[str, Any],
    *,
    bins: Mapping[str, BinSpec] | None = None,
    categories: Mapping[str, Iterable[Hashable]] | None = None,
    statistics: Statistic | Iterable[Statistic] = ("median", "nmad"),
    mask: Any | None = None,
    subsample: int | float = 1,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "topk",
    observed: bool = True,
    return_masks: Literal[True] = True,
) -> tuple[pd.DataFrame, Mapping[Hashable, Any]]: ...


def grouped_stats(
    values: Any | Mapping[str, Any],
    by: Mapping[str, Any],
    *,
    bins: Mapping[str, BinSpec] | None = None,
    categories: Mapping[str, Iterable[Hashable]] | None = None,
    statistics: Statistic | Iterable[Statistic] = ("median", "nmad"),
    mask: Any | None = None,
    subsample: int | float = 1,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["sequential", "topk"] = "topk",
    observed: bool = True,
    return_masks: bool = False,
) -> GroupedStatsResult:
    """Calculate statistics for values grouped by continuous bins or discrete categories.

    Every grouper must have an entry in ``bins`` or ``categories`` unless it has a Boolean or Pandas categorical
    dtype. Numeric edge sequences use left-closed intervals and include the final right edge. Pass an
    :class:`pandas.IntervalIndex` to control edge closure explicitly. The result index follows the order of ``by``;
    columns have ``value`` and ``statistic`` levels, and a finite ``count`` is always included for each value.

    When ``return_masks`` is true, the second result behaves as a mapping from each dataframe index key to a Boolean
    array. Its masks describe complete eligible group membership after ``mask`` and valid groupers, before random
    subsampling and independently of missing selected values.

    :param values: Numeric array, or mapping of output names to arrays with matching shapes.
    :param by: Ordered mapping of grouper names to arrays with one value per input location.
    :param bins: Continuous group definitions as bin counts, numeric edges or IntervalIndexes.
    :param categories: Ordered categories for discrete groupers.
    :param statistics: Statistic name, callable or iterable accepted by :func:`geoutils.stats.get_stats` internals.
    :param mask: Boolean array defining locations eligible for grouping.
    :param subsample: Fraction when at most one, otherwise the maximum locations used for statistics.
    :param random_state: Random generator or seed used to reproduce subsampling.
    :param strategy: ``"topk"`` for chunk independent sampling or ``"sequential"`` for ordinary sampling.
    :param observed: Whether to omit declared group combinations with no eligible locations.
    :param return_masks: Whether to also return a lazy mapping of complete group membership masks.
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
        observed=observed,
        return_masks=return_masks,
        support=None,
    )


####################################
# 4/ SPATIAL INPUT AND SUPPORT SETUP
####################################


def _values_at_support(
    source: Any,
    selector: int | str | None,
    *,
    owner: Any,
    support: Any,
    support_dataframe: gpd.GeoDataFrame | None,
    name: str,
    interpolation: str,
    align: Literal["raise", "reproject"],
    mp_config: MultiprocConfig | None,
) -> Any:
    """Read one selected raster or point value on the requested spatial support."""

    # Normalize geospatial objects and accessors before interpreting raw arrays
    source_raster = source if hasattr(source, "ij2xy") else getattr(source, "rst", None)
    source_pointcloud = (
        source
        if hasattr(source, "georeferenced_coords_equal") and hasattr(source, "data_column")
        else getattr(source, "pc", None)
    )
    support_is_raster = hasattr(support, "ij2xy")

    # Use raw grids directly when their shape already identifies the selected support
    raw_values = source.data if isinstance(source, xr.DataArray) else source
    raw_ndim = raw_values.ndim if hasattr(raw_values, "ndim") else np.asarray(raw_values).ndim
    owner_raster = owner if hasattr(owner, "ij2xy") else getattr(owner, "rst", None)
    if source_raster is None and source_pointcloud is None and raw_ndim >= 2 and owner_raster is not None:
        support_shape = tuple(support.shape) if support_is_raster else None
        direct_values = raw_values.data if isinstance(raw_values, xr.DataArray) else raw_values
        if raw_ndim == 3 and direct_values.shape[0] == 1:
            direct_values = direct_values[0]
        if support_shape is not None and tuple(direct_values.shape) == support_shape:
            return direct_values

        # Attach other raw grids to their owner before applying an explicit reprojection
        source_raster = _aligned_raster(source, owner, support, name, align)

    if source_raster is not None:
        if selector is not None and not isinstance(selector, (int, np.integer)):
            raise TypeError(f"Raster selector for {name!r} must be a band number.")
        band = 1 if selector is None else int(selector)
        raster = _aligned_raster(source_raster, source_raster, support, name, align)
        if support_is_raster:
            return _selected_raster_data(raster, band)

        # Interpolate raster values only after the point support coordinates are known
        if support_dataframe is None:
            raise RuntimeError("Point support coordinates were not prepared.")
        points = (support_dataframe.geometry.x.to_numpy(), support_dataframe.geometry.y.to_numpy())
        return raster.interp_points(
            points=points,
            method=interpolation,
            band=band,
            as_array=True,
            mp_config=mp_config,
        )

    # Reject irregular point values on a regular grid because gridding requires an explicit method
    if source_pointcloud is not None:
        if support_is_raster:
            raise ValueError(f"Point value {name!r} cannot be evaluated on raster support without gridding.")
        if source_pointcloud.crs != support.crs:
            if align != "reproject":
                raise ValueError(f"Point value {name!r} does not share the support CRS.")
            source_pointcloud = source_pointcloud.reproject(crs=support.crs)
            source_pointcloud = (
                source_pointcloud if hasattr(source_pointcloud, "georeferenced_coords_equal") else source_pointcloud.pc
            )
        if not support.georeferenced_coords_equal(source_pointcloud):
            raise ValueError(f"Point value {name!r} does not share the ordered support coordinates.")

        # Materialize one point table because its rows define the exact output ordering
        dataframe = source_pointcloud.ds
        dataframe = dataframe.compute() if is_dask_dataframe(dataframe) else dataframe
        column = source_pointcloud.data_column if selector is None else selector
        if column is None:
            return np.asarray(dataframe.geometry.z)
        if not isinstance(column, str) or column not in dataframe.columns:
            raise ValueError(f"Point column {column!r} selected for {name!r} does not exist.")
        return np.asarray(dataframe[column])

    # Accept raw point values directly when the selected support supplies their complete ordering
    if support_is_raster:
        raise ValueError(f"Raw value {name!r} cannot be tied to the selected spatial support.")
    array = np.atleast_1d(np.asanyarray(source).squeeze())
    if support_dataframe is None or array.ndim != 1 or len(array) != len(support_dataframe):
        raise ValueError(f"Raw point value {name!r} must contain one value per support point.")
    return array


def _vector_group_values(
    vector: Any,
    selector: str | None,
    *,
    support: Any,
    support_dataframe: gpd.GeoDataFrame | None,
    declared_categories: Iterable[Hashable] | None,
) -> tuple[Any, list[Hashable] | None]:
    """Evaluate a vector union or feature category on raster or point support."""

    # A vector without a selected column is one Boolean inside/outside category variable
    if selector is None:
        return np.asarray(vector.create_mask(ref=support, as_array=True), dtype=bool).squeeze(), None

    # Read feature values once and preserve caller ordering when categories were declared
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

    # Burn compact category codes into a raster before restoring Pandas categorical metadata
    if hasattr(support, "ij2xy"):
        coded = vector.rasterize(
            ref=support,
            in_value=feature_codes.tolist(),
            out_value=-1,
            out_dtype=np.int64,
        )
        codes = np.asarray(_selected_raster_data(coded), dtype=np.int64)
    else:
        if support_dataframe is None:
            raise RuntimeError("Point support coordinates were not prepared.")
        codes = np.full(len(support_dataframe), -1, dtype=np.int64)
        from geoutils.vector.vector import Vector

        # Evaluate each category union so points retain one label despite multiple features
        for code, _category in enumerate(category_values):
            subset = Vector(dataframe.loc[feature_codes == code])
            inside = np.asarray(subset.create_mask(ref=support, as_array=True), dtype=bool).squeeze()
            codes[inside] = code

    categorical = pd.Categorical.from_codes(codes.ravel(), categories=category_values, ordered=True)
    return categorical, category_values


###########################
# 5/ OBJECT METHOD DISPATCH
###########################


def _grouped_stats(
    source: Any,
    by: Mapping[str, Any],
    *,
    values: int | str | Iterable[int | str] | Mapping[str, int | str] | None,
    bins: Mapping[str, BinSpec] | None,
    categories: Mapping[str, Iterable[Hashable]] | None,
    statistics: Statistic | Iterable[Statistic],
    at: Literal["self"] | Any | None,
    mask: Any | None,
    mask_mode: Literal["inside", "outside"],
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    strategy: Literal["sequential", "topk"],
    interpolation: str,
    align: Literal["raise", "reproject"],
    observed: bool,
    return_masks: bool,
    mp_config: MultiprocConfig | None,
) -> GroupedStatsResult:
    """Align object selections and groupers before calling the array engine."""

    # Validate spatial controls without accessing potentially lazy data
    if mask_mode not in {"inside", "outside"}:
        raise ValueError("mask_mode must be 'inside' or 'outside'.")
    if align not in {"raise", "reproject"}:
        raise ValueError("align must be 'raise' or 'reproject'.")
    if isinstance(at, str) and at != "self":
        raise ValueError("at must be 'self' or a raster or point cloud support object.")
    chosen_support = source if at is None or isinstance(at, str) else at
    raster_support = chosen_support if hasattr(chosen_support, "ij2xy") else getattr(chosen_support, "rst", None)
    point_support = (
        chosen_support
        if hasattr(chosen_support, "georeferenced_coords_equal") and hasattr(chosen_support, "data_column")
        else getattr(chosen_support, "pc", None)
    )
    if raster_support is None and point_support is None:
        raise TypeError("at must select raster or point cloud support.")
    support = raster_support if raster_support is not None else point_support

    # Materialize point coordinates once because all raster interpolation follows their order
    support_dataframe: gpd.GeoDataFrame | None = None
    if point_support is not None:
        if point_support._is_dask:
            support_dataframe = point_support.load()
            support = support_dataframe.pc
            point_support = support
        else:
            support_dataframe = point_support.ds

    # Normalize selected caller values into output names and source selectors
    source_raster = source if hasattr(source, "ij2xy") else getattr(source, "rst", None)
    source_pointcloud = (
        source
        if hasattr(source, "georeferenced_coords_equal") and hasattr(source, "data_column")
        else getattr(source, "pc", None)
    )
    value_specs: dict[str, int | str | None]
    if source_raster is not None:
        if values is None:
            value_specs = {f"band_{band}": band for band in range(1, source_raster.count + 1)}
        elif isinstance(values, Mapping):
            if any(not isinstance(band, (int, np.integer)) for band in values.values()):
                raise TypeError("Raster values must select integer band numbers.")
            value_specs = {name: int(band) for name, band in values.items()}
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
            if any(not isinstance(column, str) for column in values.values()):
                raise TypeError("Point cloud values must select column names.")
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

    # Evaluate every selected value on the same support without imposing common finite validity
    selected_values: dict[str, Any] = {}
    for name, value_selector in value_specs.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Selected value names must be non-empty strings.")
        selected_values[name] = _values_at_support(
            source,
            value_selector,
            owner=source,
            support=support,
            support_dataframe=support_dataframe,
            name=name,
            interpolation=interpolation,
            align=align,
            mp_config=mp_config,
        )

    # Resolve caller selectors, external objects and explicit source and selector pairs
    selected_groupers: dict[str, Any] = {}
    resolved_categories = {} if categories is None else dict(categories)
    for name, specification in by.items():
        group_source: Any
        group_selector: int | str | None
        if isinstance(specification, str):
            group_source, group_selector = source, specification
        elif isinstance(specification, (int, np.integer)):
            group_source, group_selector = source, int(specification)
        elif (
            isinstance(specification, tuple)
            and len(specification) == 2
            and (
                hasattr(specification[0], "ij2xy")
                or hasattr(specification[0], "georeferenced_coords_equal")
                or _as_vector(specification[0]) is not None
                or getattr(specification[0], "rst", None) is not None
                or getattr(specification[0], "pc", None) is not None
            )
        ):
            group_source, group_selector = specification
        else:
            group_source, group_selector = specification, None

        # Distinguish point clouds from their Vector parent before testing ordinary vectors
        group_raster = group_source if hasattr(group_source, "ij2xy") else getattr(group_source, "rst", None)
        group_pointcloud = (
            group_source
            if hasattr(group_source, "georeferenced_coords_equal") and hasattr(group_source, "data_column")
            else getattr(group_source, "pc", None)
        )
        vector = _as_vector(group_source) if group_raster is None and group_pointcloud is None else None
        if vector is not None:
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
            interpolation=interpolation,
            align=align,
            mp_config=mp_config,
        )

    # Evaluate the global mask without treating selected value gaps as group exclusions
    if raster_support is not None:
        support_mask = _mask_on_raster(mask, support, mask_mode, align)
    elif mask is None:
        if support_dataframe is None:
            raise RuntimeError("Point support coordinates were not prepared.")
        support_mask = np.ones(len(support_dataframe), dtype=bool)
    else:
        mask_raster = mask if hasattr(mask, "ij2xy") else getattr(mask, "rst", None)
        mask_pointcloud = (
            mask
            if hasattr(mask, "georeferenced_coords_equal") and hasattr(mask, "data_column")
            else getattr(mask, "pc", None)
        )
        vector = _as_vector(mask) if mask_raster is None and mask_pointcloud is None else None
        if vector is not None:
            support_mask = np.asarray(vector.create_mask(ref=support, as_array=True), dtype=bool).squeeze()
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
            )
            support_mask = np.isfinite(mask_values) & (np.asarray(mask_values) != 0)
        else:
            support_mask = np.atleast_1d(np.asanyarray(mask).squeeze())
            if np.ma.isMaskedArray(support_mask):
                support_mask = support_mask.filled(False)
            if support_dataframe is None:
                raise RuntimeError("Point support coordinates were not prepared.")
            if support_mask.dtype != bool or len(support_mask) != len(support_dataframe):
                raise ValueError("A point support mask must be Boolean with one value per point.")

    # Delegate binning and aggregation while retaining native support for returned masks
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
        observed=observed,
        return_masks=return_masks,
        support=support,
    )


##########################
# 6/ GROUPED STAT PLOTTING
##########################


def _plot_axis(index: pd.Index) -> tuple[NDArray[Any], NDArray[Any], list[str] | None]:
    """Return plot edges, centers and optional categorical labels for one group level."""

    # Preserve numeric interval widths when adjacent bins form a regular boundary sequence
    if isinstance(index, pd.IntervalIndex) and len(index) > 0:
        adjacent = len(index) == 1 or np.all(np.asarray(index.right[:-1]) == np.asarray(index.left[1:]))
        if adjacent:
            edges = np.asarray([index[0].left, *index.right], dtype=float)
            return edges, np.asarray(index.mid, dtype=float), None

    # Fall back to equal visual widths for categories and disjoint intervals
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
    """Plot one- or two-dimensional grouped statistics with marginal sample counts.

    One-dimensional groups are drawn as a statistic curve below their counts. Two-dimensional groups use a colored
    grid with counts above and to the right. Interval widths are retained when their boundaries are contiguous;
    categorical groups use equal visual widths.

    :param table: Dataframe returned by :func:`grouped_stats` or an object ``grouped_stats`` method.
    :param value: Selected value column. It may be omitted when the table contains one value.
    :param statistic: Statistic column to display.
    :param min_count: Hide statistic cells with fewer finite observations.
    :param cmap: Matplotlib colormap used for a two-dimensional statistic grid.
    :param vmin: Lower color limit for a two-dimensional statistic grid.
    :param vmax: Upper color limit for a two-dimensional statistic grid.
    :param ax: Optional Matplotlib axes whose area is divided into the diagnostic panels.
    :param savefig_fname: Optional path used to save the completed figure.
    :returns: Mapping naming the Matplotlib axes created for each panel.
    """

    # Import plotting only when a caller requests the optional visualization
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

    # Use an existing axes as a panel frame or create a clean figure frame
    if ax is None:
        figure = plt.figure(figsize=(7, 6))
        frame = figure.add_axes((0.1, 0.1, 0.8, 0.8))
    elif isinstance(ax, matplotlib.axes.Axes):
        frame = ax
        figure = ax.figure
    else:
        raise TypeError("ax must be a Matplotlib Axes or None.")
    frame.set_axis_off()

    # Draw the compact one-dimensional count and statistic layout
    if table.index.nlevels == 1:
        count_axis = frame.inset_axes((0.0, 0.72, 1.0, 0.28))
        statistic_axis = frame.inset_axes((0.0, 0.0, 1.0, 0.64))
        edges, centers, labels = _plot_axis(table.index)
        counts = table[(value, "count")].to_numpy(dtype=float)
        values = table[(value, statistic)].where(table[(value, "count")] >= min_count).to_numpy(dtype=float)

        # Align count bars with interval widths or equal categorical slots
        count_axis.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="0.7", edgecolor="white")
        count_axis.set_xlim(edges[0], edges[-1])
        count_axis.set_ylabel("Count")
        count_axis.tick_params(axis="x", labelbottom=False)
        statistic_axis.plot(centers, values, marker="o")
        statistic_axis.set(xlim=(edges[0], edges[-1]), xlabel=table.index.name, ylabel=statistic)
        if labels is not None:
            statistic_axis.set_xticks(centers, labels, rotation=45, ha="right")
        axes = {"count": count_axis, "statistic": statistic_axis}

    # Draw a two-dimensional grid and counts marginalized from the same exact grouping
    else:
        statistic_axis = frame.inset_axes((0.0, 0.0, 0.68, 0.66))
        count_x_axis = frame.inset_axes((0.0, 0.72, 0.68, 0.28))
        count_y_axis = frame.inset_axes((0.74, 0.0, 0.26, 0.66))
        level_x, level_y = table.index.levels
        edges_x, centers_x, labels_x = _plot_axis(level_x)
        edges_y, centers_y, labels_y = _plot_axis(level_y)

        # Restore the complete declared grid so unobserved combinations remain visible as gaps
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

        # Add marginal counts using the same physical or categorical widths as the statistic grid
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

    # Save only after every inset axes and label has been added
    if savefig_fname is not None:
        figure.savefig(savefig_fname, bbox_inches="tight")
    return axes
