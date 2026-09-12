# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Group raster and point cloud values into bins, categories, or vector zones."""

from __future__ import annotations

import copy
import math
import weakref
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from geoutils._dispatch import (
    _get_pointcloud_interface,
    _get_raster_interface,
    _is_pointcloud,
    _is_raster,
    _is_vector,
    is_dask_array,
    is_dask_dataframe,
)
from geoutils._misc import import_optional
from geoutils._typing import ArrayLike
from geoutils.multiproc.chunked import iter_chunk_slices
from geoutils.multiproc.cluster import _map_bounded
from geoutils.multiproc.readers import (
    _normalize_reader_mask,
    _read_selected_values,
    _read_values,
    _reader_from_source,
    _reader_from_vector,
    _ValueReader,
)
from geoutils.raster.array import get_mask_from_array
from geoutils.sampling.stratified import _stratified_subsample_indices
from geoutils.sampling.support import (
    _aligned_pointcloud,
    _as_array,
    _normalize_mask_array,
    _normalize_sampling_input,
    _sample_vector_values,
    _sampling_specification,
    _values_at_support,
)
from geoutils.stats.reduction import _reduce_values
from geoutils.stats.selection import _sample_eligible_indices
from geoutils.vector.base import _as_geodataframe

if TYPE_CHECKING:
    from geoutils.interface.interpolation import InterpolationMethod
    from geoutils.multiproc import MultiprocConfig
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterBase, RasterLike
    from geoutils.stats.reduction import _Statistics
    from geoutils.vector.base import VectorLike


__all__ = ["plot_grouped_stats"]


####################
# 1/ DEFINE GROUPS
####################


@dataclass(frozen=True)
class _GroupDefinition:
    """Store ordered labels, histogram edges or equal-width bin count to define a _PreparedGrouper."""

    groups: pd.Index | None = None
    edges: NDArray[Any] | None = None
    bin_count: int | None = None


@dataclass(frozen=True)
class _PreparedGrouper:
    """Store grouping values with their definition (see above), including already encoded vector categories."""

    values: Any
    definition: _GroupDefinition
    encoded: bool = False


def _resolve_group_definition(
    name: str,
    *,
    values: Any = None,
    bins: int | Iterable[float] | pd.IntervalIndex | None = None,
    categories: Iterable[Hashable] | None = None,
) -> _GroupDefinition:
    """Resolve bins or categories before inferring categories from the original value dtype."""

    # Keep explicit numeric declarations independent of categorical or boolean storage
    if bins is not None:
        if isinstance(bins, (int, np.integer, np.bool_)):
            if isinstance(bins, (bool, np.bool_)) or bins < 1:
                raise ValueError(f"The bin count for {name!r} must be a positive integer.")
            return _GroupDefinition(bin_count=int(bins))
        if isinstance(bins, pd.IntervalIndex):
            intervals = bins.rename(name)
            if intervals.empty or not intervals.is_non_overlapping_monotonic:
                raise ValueError(f"Intervals for {name!r} must be non-empty, ordered and non-overlapping.")
            if not all(np.isfinite(interval.left) and np.isfinite(interval.right) for interval in intervals):
                raise ValueError(f"Intervals for {name!r} must have finite bounds.")
            return _GroupDefinition(groups=intervals)

        # Treat a numeric sequence as histogram edges and include the final upper edge
        edges = np.asarray(list(bins), dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
            raise ValueError(f"Bin edges for {name!r} must be finite and strictly increasing.")
        return _GroupDefinition(groups=pd.IntervalIndex.from_breaks(edges, closed="left", name=name), edges=edges)

    # Use declared categories, Pandas categories, or the two boolean values
    dtype = getattr(values, "dtype", None)
    if categories is None and isinstance(dtype, pd.CategoricalDtype):
        if not is_dask_dataframe(values) or cast(Any, values).cat.known:
            categories = dtype.categories
    if categories is None and pd.api.types.is_bool_dtype(dtype):
        categories = (False, True)
    if categories is None:
        raise ValueError(f"Grouper {name!r} requires an entry in ``bins`` or ``categories``.")
    category_index = pd.Index(list(categories), tupleize_cols=False)
    if category_index.empty or category_index.has_duplicates or category_index.hasnans:
        raise ValueError(f"Categories for {name!r} must be non-empty, unique and finite.")

    # Keep the declaration order in the result, including categories absent from the input
    level = pd.CategoricalIndex(category_index, categories=category_index, ordered=True, name=name)
    return _GroupDefinition(groups=level)


def _validate_group_declarations(
    by: Mapping[str, Any],
    bins: Mapping[str, int | Iterable[float] | pd.IntervalIndex] | None,
    categories: Mapping[str, Iterable[Hashable]] | None,
) -> dict[str, _GroupDefinition]:
    """Validate grouper names and explicit declarations before spatial placement or array conversion."""

    # Check bin and category declarations before reading grouping values
    if not by:
        raise ValueError("Argument ``by`` must contain at least one named grouper.")
    if any(not isinstance(name, str) or not name for name in by):
        raise ValueError("Grouper names must be non-empty strings.")
    bins = {} if bins is None else bins
    categories = {} if categories is None else categories

    # Require one unambiguous interpretation for every declared grouper
    unknown = (set(bins) | set(categories)).difference(by)
    if unknown:
        raise ValueError(f"Bin or category declarations do not match ``by``: {sorted(unknown)!r}.")
    overlap = set(bins).intersection(categories)
    if overlap:
        raise ValueError(f"A grouper cannot define both ``bins`` and ``categories``: {sorted(overlap)!r}.")
    return {
        name: _resolve_group_definition(name, bins=bins.get(name), categories=categories.get(name))
        for name in by
        if name in bins or name in categories
    }


def _prepare_grouper(values: Any, definition: _GroupDefinition) -> _PreparedGrouper:
    """Convert one grouper to array values while keeping its categories or bins."""

    # Numeric bins read category values, while category groups can use exact integer category codes
    if isinstance(getattr(values, "dtype", None), pd.CategoricalDtype):
        if isinstance(definition.groups, pd.CategoricalIndex):
            categories = definition.groups.categories
            if isinstance(values, (pd.Categorical, pd.CategoricalIndex)):
                return _PreparedGrouper(values.set_categories(categories).codes, definition, encoded=True)
            categorical = values.cat.set_categories(categories)
            codes = _as_array(categorical.cat.codes)
            return _PreparedGrouper(codes, definition, encoded=True)
        values = values.astype(float)

    # Unwrap Xarray and dataframe containers without changing NumPy masks or Dask storage
    if isinstance(values, xr.DataArray):
        values = values.data
    elif is_dask_dataframe(values) or isinstance(values, (pd.Series, pd.Index)):
        values = _as_array(values)
    return _PreparedGrouper(values, definition)


#####################################
# 2/ NORMALIZE GROUPED INPUTS
#####################################


def _normalize_grouped_inputs_dask_eager(
    values: Mapping[str, Any],
    by: Mapping[str, _PreparedGrouper],
    mask: Any | None,
    mp_config: MultiprocConfig | None,
) -> tuple[dict[str, Any], dict[str, _PreparedGrouper], Any | None, tuple[int, ...], bool, Any | None]:
    """Prepare NumPy or Dask inputs with matching shapes and chunks."""

    # Unwrap value containers once, keeping native masks and Dask arrays
    named_values = {name: _as_array(array) for name, array in values.items()}
    first_value = next(iter(named_values.values()))
    shape = tuple(first_value.shape)
    if not shape:
        raise ValueError("Argument ``values`` must contain at least one dimension.")
    mask = _normalize_mask_array(mask, shape)

    # Use one Dask layout when any value, grouping variable or mask is lazy
    all_inputs = [*named_values.values(), *(grouper.values for grouper in by.values()), mask]
    raw_inputs = [value.data if isinstance(value, xr.DataArray) else value for value in all_inputs]
    lazy_inputs = [value for value in raw_inputs if is_dask_array(value)]
    use_dask = bool(lazy_inputs)
    chunks = lazy_inputs[0].reshape(shape).chunks if use_dask else None
    if use_dask and mp_config is not None:
        raise ValueError("Dask inputs cannot be combined with Multiprocessing grouped statistics.")
    if use_dask:
        import_optional("dask")
        import dask.array as da

    # Check every selected value against the shared shape and array type
    arrays: dict[str, Any] = {}
    for name, raw_values in named_values.items():
        array = da.asarray(raw_values) if use_dask else np.asanyarray(raw_values)
        if tuple(array.shape) != shape:
            raise ValueError(f"Value {name!r} must match the shape of the other selected values.")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"Value {name!r} must contain numeric data.")
        arrays[name] = array.reshape(shape)
        if use_dask:
            arrays[name] = arrays[name].rechunk(chunks)

    # Put every grouper on the same shape and Dask chunk layout as the selected values
    groupers: dict[str, _PreparedGrouper] = {}
    for name, prepared in by.items():
        raw_values = prepared.values
        group_values: Any = raw_values if is_dask_array(raw_values) else np.asanyarray(raw_values)
        flattened_category = prepared.encoded and group_values.ndim == 1 and group_values.size == math.prod(shape)
        if tuple(group_values.shape) != shape and not flattened_category:
            raise ValueError(f"Grouper {name!r} must contain one value per input location.")
        group_values = group_values.reshape(shape)
        if use_dask:
            group_values = (
                group_values.rechunk(chunks)
                if is_dask_array(group_values)
                else da.from_array(group_values, chunks=chunks)
            )
        if not isinstance(prepared.definition.groups, pd.CategoricalIndex) and not pd.api.types.is_numeric_dtype(
            group_values.dtype
        ):
            raise TypeError(f"Continuous grouper {name!r} must contain numeric values.")
        groupers[name] = _PreparedGrouper(group_values, prepared.definition, prepared.encoded)
    return arrays, groupers, mask, shape, use_dask, chunks


def _normalize_grouped_inputs_mp(
    values: Mapping[str, Any],
    by: Mapping[str, _PreparedGrouper],
    mask: Any | None,
    mp_config: MultiprocConfig,
) -> tuple[
    dict[str, Any],
    dict[str, _PreparedGrouper],
    Any | None,
    tuple[int, ...],
    list[tuple[slice, ...]],
]:
    """Validate file-backed inputs and define the tiles multiprocessing workers will read."""

    # Preserve readers for worker access while unwrapping only values already held in memory
    arrays = {name: value if isinstance(value, _ValueReader) else _as_array(value) for name, value in values.items()}
    shape = tuple(next(iter(arrays.values())).shape)
    if not shape:
        raise ValueError("Argument ``values`` must contain at least one dimension.")
    mask = _normalize_reader_mask(mask, shape)

    # Reject mixed Dask and multiprocessing execution before starting file reads
    raw_groupers = [grouper.values for grouper in by.values()]
    if any(is_dask_array(value) for value in [*arrays.values(), *raw_groupers, mask]):
        raise ValueError("Dask inputs cannot be combined with Multiprocessing grouped statistics.")
    if any(tuple(value.shape) != shape for value in arrays.values()):
        raise ValueError("Selected values must have matching shapes.")
    for name, value in arrays.items():
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError(f"Value {name!r} must contain numeric data.")

    # Check grouper shapes and types while keeping readers available for later worker passes
    groupers: dict[str, _PreparedGrouper] = {}
    for name, prepared in by.items():
        group_values = prepared.values
        if not isinstance(group_values, _ValueReader):
            group_values = _as_array(group_values)
            if prepared.encoded and group_values.ndim == 1 and group_values.size == math.prod(shape):
                group_values = group_values.reshape(shape)
        if tuple(group_values.shape) != shape:
            raise ValueError(f"Grouper {name!r} must match the shape of the selected values.")
        if not isinstance(prepared.definition.groups, pd.CategoricalIndex) and not pd.api.types.is_numeric_dtype(
            group_values.dtype
        ):
            raise TypeError(f"Continuous grouper {name!r} must contain numeric values.")
        groupers[name] = _PreparedGrouper(group_values, prepared.definition, prepared.encoded)

    # Reuse one ordered set of slices for every multiprocessing pass
    tiles = list(iter_chunk_slices(shape, mp_config.chunks))
    return arrays, groupers, mask, shape, tiles


######################################
# 3/ DERIVE BIN EDGES FROM COUNT
######################################


def _group_definition_from_limits(name: str, bin_count: int, lower: float, upper: float) -> _GroupDefinition:
    """Build equal-width histogram bins from min/mas in the group, including constant inputs."""

    # Give constant values a nonzero bin range so the edges stay strictly increasing
    if lower == upper:
        half_width = 0.5 * abs(float(lower)) if lower != 0 else 0.5
        lower, upper = lower - half_width, upper + half_width
    edges = np.linspace(float(lower), float(upper), bin_count + 1)
    return _GroupDefinition(groups=pd.IntervalIndex.from_breaks(edges, closed="left", name=name), edges=edges)


def _derive_bin_edges_from_count_dask_eager(
    by: Mapping[str, _PreparedGrouper],
    *,
    mask: Any | None,
    use_dask: bool,
    chunks: Any | None,
) -> dict[str, _PreparedGrouper]:
    """
    Derive equal-width bin edges for eager or Dask groupers declared by bin count.

    Only required for binning defined by a count requiring min/max knowledge of the variable, otherwise this function
    skips the grouping variable.

    :param use_dask: Whether to derive limits lazily with Dask or eagerly with NumPy.
    :param chunks: Common Dask chunk layout, or None for eager inputs.
    :returns: Prepared groupers with bin counts replaced by their derived interval labels and edges.
    """

    # Use the common Dask layout when a lazy input requires calculating its range
    if use_dask:
        import_optional("dask")
        import dask.array as da

    user_eligible = None if mask is None else da.asarray(mask).rechunk(chunks) if use_dask else mask
    resolved: dict[str, _PreparedGrouper] = {}
    for name, prepared in by.items():
        definition = prepared.definition
        values = prepared.values

        # Find the common finite range only for automatically generated equal-width bins
        if definition.bin_count is not None:
            if values.size == 0:
                raise ValueError(f"Grouper {name!r} has no finite values inside ``mask``.")
            if use_dask:
                import dask

                finite = da.ma.filled(da.isfinite(values), False)
                if user_eligible is not None:
                    finite &= user_eligible
                lower, upper, finite_count = dask.compute(
                    da.min(da.where(finite, values, np.inf)),
                    da.max(da.where(finite, values, -np.inf)),
                    finite.sum(),
                )
            else:
                finite = np.isfinite(np.ma.getdata(values)) & ~np.ma.getmaskarray(values)
                if user_eligible is not None:
                    finite &= user_eligible
                finite_values = np.asarray(values)[np.asarray(finite)]
                finite_count = finite_values.size
                lower = np.min(finite_values) if finite_count else np.nan
                upper = np.max(finite_values) if finite_count else np.nan
            if not finite_count:
                raise ValueError(f"Grouper {name!r} has no finite values inside ``mask``.")
            definition = _group_definition_from_limits(name, definition.bin_count, lower, upper)
        resolved[name] = _PreparedGrouper(values, definition, prepared.encoded)
    return resolved


def _wrapper_group_limits_block_mp(values: Any, mask: Any) -> tuple[float, float]:
    """Find finite grouper bounds after the user mask in one multiprocessing worker block."""

    array = _as_array(_read_values(values))
    eligible = ~get_mask_from_array(array).reshape(array.shape)
    if mask is not None:
        eligible &= _normalize_mask_array(_read_values(mask), array.shape)
    finite = np.ma.getdata(array)[eligible]
    if finite.size == 0:
        return np.inf, -np.inf
    return float(np.min(finite)), float(np.max(finite))


def _derive_bin_edges_from_count_mp(
    by: Mapping[str, _PreparedGrouper],
    mask: Any | None,
    tiles: Sequence[tuple[slice, ...]],
    mp_config: MultiprocConfig,
) -> dict[str, _PreparedGrouper]:
    """
    Derive equal-width bin edges for multiprocessing groupers declared by bin count.

    Same as eager/Dask logic.
    Only required for binning defined by a count requiring min/max knowledge of the variable, otherwise this function
    skips the grouping variable.

    _group_limits_mp() reads the finite minimum and maximum inside the user mask from each worker block.
    """

    resolved = {}
    for name, prepared in by.items():
        definition = prepared.definition
        group_values = prepared.values

        # If the bin count was used as a definition, we find the min/max to define the bin edges, otherwise we skip
        if definition.bin_count is not None:
            group_arguments = []
            for tile in tiles:
                block = group_values.block(tile) if isinstance(group_values, _ValueReader) else group_values[tile]
                block_mask = (
                    mask.block(tile) if isinstance(mask, _ValueReader) else None if mask is None else mask[tile]
                )
                group_arguments.append((block, block_mask))

            # We start at infinity and recursively accumulate min/max to get the global group min/max
            lower, upper = np.inf, -np.inf
            for _, (block_lower, block_upper) in _map_bounded(
                mp_config.cluster, _wrapper_group_limits_block_mp, group_arguments
            ):
                lower, upper = min(lower, block_lower), max(upper, block_upper)
            if not np.isfinite(lower):
                raise ValueError(f"Grouper {name!r} has no finite values inside ``mask``.")

            # Define bin edges
            definition = _group_definition_from_limits(name, definition.bin_count, lower, upper)
        resolved[name] = _PreparedGrouper(group_values, definition, prepared.encoded)
    return resolved


#################################
# 4/ ASSIGN GROUP IDs
#################################


def _encode_grouper(values: NDArray[Any], *, groups: pd.Index, edges: NDArray[Any] | None = None) -> NDArray[Any]:
    """
    Replace categories or intervals with their integer group IDs in one array block.

    Keep the input shape and mark missing or undeclared values with -1.
    Numeric edges follow histogram conventions (left close, right open), users can pass an IntervalIndex instead to
    have it explicitly defined.

    :param values: One block of a single grouping variable passed to _assign_group_ids_dask_eager().
    :param groups: Ordered category or interval labels.
    :param edges: Explicit numeric bin boundaries, or None to match the labels or intervals in groups directly.

    :returns: Integer group IDs with the input shape and -1 for excluded values.
    """

    # Follow histogram edge rules and include values equal to the final upper edge
    array = np.asarray(np.ma.getdata(values))
    missing = np.ma.getmaskarray(values)
    if edges is not None:
        codes = np.searchsorted(edges, array, side="right") - 1
        codes[array == edges[-1]] = len(edges) - 2
        invalid = ~np.isfinite(array) | (codes < 0) | (codes >= len(edges) - 1)
        return np.where(invalid | missing, -1, codes).astype(np.int64)

    # Let Pandas apply explicit interval boundaries or match category labels
    if isinstance(groups, pd.IntervalIndex):
        codes = groups.get_indexer(array.ravel())
    else:
        categories = groups.categories if isinstance(groups, pd.CategoricalIndex) else groups
        codes = pd.Categorical(array.ravel(), categories=categories, ordered=True).codes
    codes = np.asarray(codes, dtype=np.int64).reshape(array.shape)
    return np.where(missing, -1, codes)


def _group_layout(by: Mapping[str, _PreparedGrouper]) -> tuple[list[pd.Index], int]:
    """Return the ordered labels and total group count shared by every calculation backend."""

    levels = []
    for grouper in by.values():
        assert grouper.definition.groups is not None
        levels.append(grouper.definition.groups)
    total_groups = math.prod(len(level) for level in levels)
    if total_groups > np.iinfo(np.int64).max:
        raise ValueError("The product of group counts exceeds the supported integer range.")
    return levels, total_groups


@dataclass(frozen=True)
class _GroupAssignmentMP:
    """Keep multiprocessing group IDs and their temporary storage available to the parent calculation."""

    group_ids: Any
    levels: list[pd.Index]
    total_groups: int
    observed_ids: set[int]
    storage: ExitStack
    directory: str


def _assign_group_ids_dask_eager(
    by: Mapping[str, _PreparedGrouper],
    *,
    mask: Any | None,
    shape: tuple[int, ...],
    use_dask: bool,
    chunks: Any | None,
) -> tuple[Any, list[pd.Index], int]:
    """
    Assign combined group IDs to eager or Dask arrays.

    _encode_grouper() gives an ID to each variable's categories or intervals, following order of ``by``.
    A location receives -1 if the user mask excludes it or any grouper is missing or undeclared.

    :param by: Groupers returned by _derive_bin_edges_from_count_dask_eager().
    :param shape: Common shape of the values, grouping variables and user mask.
    :param use_dask: Whether to assign group IDs lazily with Dask or eagerly with NumPy.
    :param chunks: Common Dask chunk layout, or None for eager inputs.
    :returns: Combined group IDs with the input shape, ordered label indexes for each grouper, and the total number
        of declared combinations. Negative IDs mark excluded locations.
    """

    # Start with the user mask because it applies to every grouping variable
    if use_dask:
        import_optional("dask")
        import dask.array as da
    if mask is None:
        eligible = da.ones(shape, chunks=chunks, dtype=bool) if use_dask else np.ones(shape, dtype=bool)
    else:
        eligible = da.asarray(mask).rechunk(chunks) if use_dask else mask

    # Assign group IDs within each grouping variable in the order of "by"
    encoded: list[Any] = []
    levels, total_groups = _group_layout(by)
    for prepared, level in zip(by.values(), levels):
        definition = prepared.definition
        values = prepared.values.reshape(shape)

        # Number the intervals in each array block, use Pandas only for custom open or closed sides
        if prepared.encoded:
            codes = da.where(da.isfinite(values), values, -1) if use_dask else np.where(np.isfinite(values), values, -1)
            codes = codes.astype(np.int64)
        elif use_dask:
            codes = values.map_blocks(_encode_grouper, groups=level, edges=definition.edges, dtype=np.int64)
        else:
            codes = _encode_grouper(values, groups=level, edges=definition.edges)
        encoded.append(codes)
        eligible = eligible & (codes >= 0)

    # Combine the separate IDs into one compact group ID per input location
    group_ids = da.zeros(shape, chunks=chunks, dtype=np.int64) if use_dask else np.zeros(shape, dtype=np.int64)
    for codes, level in zip(encoded, levels):
        # Reserve one consecutive range for each earlier combination before adding this variable's number
        group_ids = group_ids * len(level) + codes
    group_ids = da.where(eligible, group_ids, -1) if use_dask else np.where(eligible, group_ids, -1)

    # Use the smallest signed integer type that can hold all groups plus the missing marker
    for dtype in (np.int8, np.int16, np.int32, np.int64):
        if total_groups - 1 <= np.iinfo(dtype).max:
            group_ids = group_ids.astype(dtype)
            break
    return group_ids, levels, total_groups


def _wrapper_assign_group_ids_block_mp(groupers: Mapping[str, Any], mask: Any, shape: tuple[int, ...]) -> Any:
    """Read and assign resolved group IDs within one multiprocessing worker block."""

    prepared = {
        name: _PreparedGrouper(_read_values(grouper.values), grouper.definition, grouper.encoded)
        for name, grouper in groupers.items()
    }
    block_mask = _normalize_mask_array(_read_values(mask), shape)
    ids, _, _ = _assign_group_ids_dask_eager(prepared, mask=block_mask, shape=shape, use_dask=False, chunks=None)
    return ids


def _assign_group_ids_mp(
    by: Mapping[str, _PreparedGrouper],
    mask: Any | None,
    shape: tuple[int, ...],
    tiles: Sequence[tuple[slice, ...]],
    mp_config: MultiprocConfig,
) -> _GroupAssignmentMP:
    """
    Assign group IDs from unloaded readers into temporary storage with multiprocessing.

    Follows the same logic as the eager/Dask implementation above.

    _assign_group_ids_block_mp() handles one block at a time, we keep the combined numbers on disk so sampling,
    reduction and optional returned masks can reuse them without loading all grouping arrays at once.
    """

    # Create the disk-backed array before workers begin returning group ID blocks
    levels, total_groups = _group_layout(by)
    storage = ExitStack()
    try:
        directory = storage.enter_context(TemporaryDirectory(prefix="geoutils-stats-"))
        ids = (
            np.memmap(f"{directory}/groups.dat", mode="w+", dtype=np.int64, shape=shape)
            if math.prod(shape)
            else np.empty(shape, dtype=np.int64)
        )
        if isinstance(ids, np.memmap):
            storage.callback(ids._mmap.close)
        observed_ids = set()

        # Send matching grouping and mask blocks to each worker in their original order
        id_arguments = []
        for tile in tiles:
            block_groupers = {
                name: replace(
                    grouper,
                    values=(
                        grouper.values.block(tile) if isinstance(grouper.values, _ValueReader) else grouper.values[tile]
                    ),
                )
                for name, grouper in by.items()
            }
            block_mask = mask.block(tile) if isinstance(mask, _ValueReader) else None if mask is None else mask[tile]
            block_shape = tuple(part.stop - part.start for part in tile)
            id_arguments.append((block_groupers, block_mask, block_shape))

        for index, block_ids in _map_bounded(mp_config.cluster, _wrapper_assign_group_ids_block_mp, id_arguments):
            ids[tiles[index]] = block_ids
            observed_ids.update(int(value) for value in np.unique(block_ids) if value >= 0)
        return _GroupAssignmentMP(
            group_ids=ids,
            levels=levels,
            total_groups=total_groups,
            observed_ids=observed_ids,
            storage=storage,
            directory=directory,
        )
    except Exception:
        storage.close()
        raise


################################
# 5/ SAMPLE GROUPED VALUES
################################


def _sample_grouped_values_dask_eager(
    values: Mapping[str, Any],
    group_ids: Any,
    *,
    subsample: int | float,
    subsample_per_group: bool,
    random_state: int | np.random.Generator | None,
    subsampling_strategy: Literal["sequential", "topk"],
    use_dask: bool,
) -> tuple[dict[str, Any], NDArray[Any]]:
    """Sample matching group IDs and selected values from eager or Dask arrays."""

    # If we subsample per group, we call stratified subsampling
    if subsample_per_group:
        flat_indices = _stratified_subsample_indices(
            group_ids,
            subsample=subsample,
            random_state=random_state,
            strategy=subsampling_strategy,
        )
    # Sample directly from all eligible locations when sampling is not per group
    else:
        flat_indices = _sample_eligible_indices(
            group_ids >= 0,
            subsample=subsample,
            random_state=random_state,
            strategy=subsampling_strategy,
        )

    # Read the same locations for every input value, computing Dask inputs together
    selected = [
        group_ids.reshape(-1)[flat_indices],
        *(array.reshape(-1)[flat_indices] for array in values.values()),
    ]
    if use_dask:
        import dask

        selected = list(dask.compute(*selected))
    selected_ids = np.asarray(selected[0])
    selected_values = {name: np.asanyarray(array) for name, array in zip(values, selected[1:])}
    return selected_values, selected_ids


def _sample_grouped_values_mp(
    values: Mapping[str, Any],
    group_ids: Any,
    *,
    shape: tuple[int, ...],
    subsample: int | float,
    subsample_per_group: bool,
    random_state: int | np.random.Generator | None,
    subsampling_strategy: Literal["sequential", "topk"],
    mp_config: MultiprocConfig,
    assignment: _GroupAssignmentMP | None,
) -> tuple[dict[str, Any], NDArray[Any]]:
    """
    Sample values with matching group IDs using multiprocessing backend.

    It mirrors the eager/Dask logic above.

    NumPy values are indexed directly. For unloaded readers, we use _read_selected_values() so
    workers read only the selected value locations.
    """

    # If we subsample per group, we call stratified subsampling
    if subsample_per_group:
        sample_ids = group_ids
        indexes = _stratified_subsample_indices(
            sample_ids,
            subsample,
            random_state=random_state,
            strategy=subsampling_strategy,
            mp_config=mp_config,
        )
    # Sample directly when all group IDs are in memory
    elif assignment is None:
        indexes = _sample_eligible_indices(
            group_ids >= 0,
            subsample=subsample,
            random_state=random_state,
            strategy=subsampling_strategy,
        )
    # Scan group IDs per chunk
    else:
        sample_ids = group_ids
        if math.prod(shape):
            sample_ids = np.memmap(f"{assignment.directory}/eligible.dat", mode="w+", dtype=np.int8, shape=shape)
            assignment.storage.callback(sample_ids._mmap.close)
            for tile in iter_chunk_slices(shape, mp_config.chunks):
                sample_ids[tile] = np.where(group_ids[tile] >= 0, 0, -1)
        indexes = _stratified_subsample_indices(
            sample_ids,
            subsample,
            random_state=random_state,
            strategy=subsampling_strategy,
            mp_config=mp_config,
        )

    # Read the same locations for every input value, leaving unloaded inputs to worker readers
    selected_ids = np.asarray(group_ids.reshape(-1)[indexes])
    selected_values = {
        name: (
            _read_selected_values(array, indexes, mp_config)
            if isinstance(array, _ValueReader)
            else array.reshape(-1)[indexes]
        )
        for name, array in values.items()
    }
    return selected_values, selected_ids


########################################
# 6/ BUILD GROUP LABELS AND MASKS
########################################


class _GroupMasks(Mapping[Hashable, Any]):
    """
    Create boolean masks on the result grid or points from one shared group ID array.

    This functionality is a helper for the ``return_masks`` option, to get a boolean mask of each group
    on the common support.
    We build each mask only when its result key is requested, avoiding one stored array per group.

    :param group_ids: Complete combined memberships from _assign_group_ids_dask_eager() or
        _assign_group_ids_mp(), before subsampling.
    :param key_ids: Mapping from result row labels to the integer group IDs represented by those rows.
    :param shape: Original value-array shape restored when a group mask is requested.
    :param support: Raster or point cloud defining the result locations, or None to return array masks.
    """

    def __init__(
        self,
        group_ids: Any,
        key_ids: Mapping[Hashable, int],
        shape: tuple[int, ...],
        support: RasterBase | PointCloudBase | None,
    ) -> None:
        """Keep the shared memberships and ordered result keys without creating individual masks."""

        # Keep one group ID array and follow the result table's row order
        self._group_ids = group_ids
        self._key_ids = dict(key_ids)
        self._shape = shape
        self._support = support

    def __getitem__(self, key: Hashable) -> RasterLike | PointCloudLike | ArrayLike:
        """Return one group's boolean mask as an array or an object on the original spatial support."""

        # 1/ Select the requested group full membership

        # Raise the usual dictionary error before creating a mask
        if key not in self._key_ids:
            raise KeyError(key)
        mask = (self._group_ids == self._key_ids[key]).reshape(self._shape)

        # Return a plain boolean array when the input had no raster or point locations
        if self._support is None:
            return mask

        # 2/ Build a raster mask with the same grid and coordinate system as the input
        if _is_raster(self._support):
            raster = cast("RasterBase", self._support)
            return raster.from_array(
                data=mask,
                transform=raster.transform,
                crs=raster.crs,
                nodata=None,
                area_or_point=raster.area_or_point,
                tags=raster.tags.copy(),
            )

        # 3/ Keep point geometry and other columns while replacing the selected value with the mask
        if _is_pointcloud(self._support):
            from geoutils.pointcloud.dataframe import (
                _assign_point_values,
                _build_pointcloud_output,
                _get_dataframe_attrs,
            )

            pointcloud = cast("PointCloudBase", self._support)
            if not pointcloud._is_pd and not pointcloud.is_loaded:
                # Read attributes for the requested mask without loading the caller's file-backed point object
                pointcloud = copy.copy(pointcloud)
                pointcloud.load(columns="all")
            dataframe = pointcloud.ds
            column = pointcloud.data_column
            if column is None:
                # Add a boolean column when the point values were stored in geometry Z coordinates
                column = "group_mask"
                while column in dataframe.columns:
                    # Avoid replacing an existing point attribute with the new membership column
                    column = f"_{column}"

            # Preserve every coordinate and assign masks by row position, including single-point inputs
            output = _assign_point_values(dataframe, {column: mask})
            return _build_pointcloud_output(
                output,
                data_column=column,
                as_dataframe=pointcloud._ACCESSOR_OUTPUT,
                attrs=_get_dataframe_attrs(dataframe),
                preserve_locations=True,
            )
        raise TypeError("Group masks require array, raster or point cloud support.")

    def __iter__(self) -> Iterator[Hashable]:
        """Iterate over group labels in the result table row order."""

        return iter(self._key_ids)

    def __len__(self) -> int:
        """Return the number of groups represented in the result table."""

        return len(self._key_ids)


def _group_index(
    levels: Sequence[pd.Index], names: Sequence[str], group_numbers: Sequence[int] | NDArray[Any]
) -> pd.Index:
    """
    Build the ordered Pandas row index from combined group IDs.

    We use a categorical or interval index for one grouper, and a MultiIndex for several groupers.

    :param levels: Ordered label indexes returned by the eager, Dask or multiprocessing group assignment.
    :param names: Grouper names in the same order as levels.
    :param group_numbers: Combined integer group IDs to include, in the requested row order.
    :returns: A Pandas index naming each requested group combination.
    """

    # Split each combined number back into one number per grouping variable
    level_codes = []
    remainders = np.asarray(group_numbers, dtype=np.int64)
    for level in reversed(levels):
        remainders, codes = np.divmod(remainders, len(level))
        level_codes.append(codes)
    level_codes.reverse()

    # Use a direct interval or category index when there is only one grouping variable
    if len(levels) == 1:
        selected = levels[0].take(level_codes[0])
        index = selected.rename(names[0])
    else:
        index = pd.MultiIndex(levels=list(levels), codes=level_codes, names=list(names), verify_integrity=False)
    return index


def _format_grouped_stats(
    table: pd.DataFrame,
    statistics: _Statistics,
    resolved_strategy: str,
    *,
    value_names: Sequence[str],
    by_names: Sequence[str],
    levels: Sequence[pd.Index],
    total_groups: int,
    full_group_ids: Any,
    shape: tuple[int, ...],
    support: RasterBase | PointCloudBase | None,
    observed: bool,
    subsample: int | float,
    subsample_per_group: bool,
    subsampling_strategy: Literal["sequential", "topk"],
    return_masks: bool,
    group_numbers: NDArray[Any] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, Mapping[Hashable, RasterLike | PointCloudLike | ArrayLike]]:
    """Format final reduced values, adding back empty categories with NaNs."""

    # Keep every group that was present before the optional subsampling
    if group_numbers is not None:
        group_numbers = np.asarray(group_numbers)
    elif not observed:
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

    # Fill empty counts/groups with zero and leave estimates as NaN
    columns = pd.MultiIndex.from_product([list(value_names), statistics.output_names], names=["value", "statistic"])
    table.columns = columns
    for name in value_names:
        table[(name, "count")] = table[(name, "count")].fillna(0).astype(np.int64)
        for statistic, alias in zip(statistics.names, statistics.aliases):
            if alias in {"validcount", "totalcount"}:
                table[(name, statistic)] = table[(name, statistic)].fillna(0).astype(np.int64)
    table.index = _group_index(levels, by_names, group_numbers)

    # Record sampling user inputs in the table attributes
    table.attrs["grouped_stats"] = {
        "observed": observed,
        "subsample": subsample,
        "subsample_per_group": subsample_per_group,
        "strategy": resolved_strategy,
        "subsampling_strategy": subsampling_strategy,
        "mask_membership": "groupers",
    }

    # Delay each boolean group mask until the caller reads it from the returned mapping
    if return_masks:
        key_ids = dict(zip(table.index, map(int, group_numbers)))
        masks = _GroupMasks(full_group_ids, key_ids=key_ids, shape=shape, support=support)
        return table, masks
    return table


##########################################
# 7/ CALCULATE GROUPED STATISTICS
##########################################


def _group_values_for_flox(
    by: Mapping[str, _PreparedGrouper], *, use_dask: bool
) -> tuple[list[Any], list[pd.Index | NDArray[Any]]]:
    """Prepare grouping values and their declared order for the eager or Dask Flox backend."""

    group_values = []
    expected_groups = []
    for prepared in by.values():
        values = prepared.values
        groups = prepared.definition.groups
        assert groups is not None

        # Keep ordinary category values for Flox to match, including their declared order
        masked = np.ma.isMaskedArray(values._meta) if use_dask else np.ma.isMaskedArray(values)
        if not prepared.encoded and not isinstance(groups, pd.IntervalIndex) and not masked:
            group_values.append(values)
            expected_groups.append(groups)
            continue

        # Keep existing category codes, or encode bins whose boundary rules differ from Flox's native bins
        if prepared.encoded:
            codes = values.astype(np.int64)
        elif use_dask:
            codes = values.map_blocks(
                _encode_grouper,
                groups=groups,
                edges=prepared.definition.edges,
                dtype=np.int64,
            )
        else:
            codes = _encode_grouper(values, groups=groups, edges=prepared.definition.edges)
        group_values.append(codes)
        expected_groups.append(np.arange(len(groups), dtype=np.int64))
    return group_values, expected_groups


def _groupby_reduce_flox(
    values: Any,
    group_values: Sequence[Any],
    expected_groups: Sequence[pd.Index | NDArray[Any]],
    *,
    function: str,
    use_dask: bool,
    fill_value: float | int,
    dtype: Any,
    finalize_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Run one eager or Dask Flox reduction over every grouping variable."""

    flox = import_optional("flox", extra_name="flox")
    expected: Any = expected_groups[0] if len(expected_groups) == 1 else tuple(expected_groups)
    axes = tuple(range(-group_values[0].ndim, 0))
    return flox.groupby_reduce(
        values,
        *group_values,
        func=function,
        expected_groups=expected,
        sort=False,
        axis=axes,
        fill_value=fill_value,
        dtype=dtype,
        method="map-reduce" if use_dask else None,
        finalize_kwargs=finalize_kwargs,
    )[0]


def _calculate_grouped_stats_flox(
    values: Mapping[str, Any],
    by: Mapping[str, _PreparedGrouper],
    *,
    statistics: _Statistics,
    mask: Any | None,
    full_group_ids: Any | None,
    shape: tuple[int, ...],
    support: RasterBase | PointCloudBase | None,
    observed: bool,
    subsample: int | float,
    subsampling_strategy: Literal["sequential", "topk"],
) -> pd.DataFrame:
    """
    Calculate grouped statistics with Flox from eager or Dask values and grouping variables.

    Flox matches the separate grouping values itself. GeoUtils only converts interval definitions to preserve their
    boundary rules, applies the common mask and finite-data rules, and builds the established result dataframe.
    """

    # Check statistics that Flox cannot reproduce through its grouped reductions
    unsupported = [
        name for name, alias in zip(statistics.names, statistics.aliases) if alias is None or alias == "nmad"
    ]
    if unsupported:
        raise ValueError(f"The Flox backend does not support these statistics: {unsupported!r}.")
    exact = {"median", "90thpercentile", "iqr", "le90"}
    use_dask = any(is_dask_array(array) for array in values.values())
    if use_dask and any(alias in exact for alias in statistics.aliases):
        raise ValueError("The Dask Flox backend does not support exact median or percentile statistics.")

    # Prepare Flox group inputs while keeping the categories and bins used for the final row index
    group_values, expected_groups = _group_values_for_flox(by, use_dask=use_dask)
    levels, total_groups = _group_layout(by)
    if use_dask:
        import_optional("dask")
        import dask.array as da

    # Apply the user mask and each value's missing data independently before stacking the selected values
    selected_values = []
    for array in values.values():
        if use_dask:
            data = da.ma.getdata(array)
            valid = ~da.ma.getmaskarray(array) & da.isfinite(data)
            if mask is not None:
                valid &= mask
            selected_values.append(da.where(valid, data, np.nan))
        else:
            data = np.ma.getdata(array)
            valid = ~get_mask_from_array(array)
            if mask is not None:
                valid &= mask
            selected_values.append(np.where(valid, data, np.nan))
    stacked = da.stack(selected_values) if use_dask else np.stack(selected_values)

    # Count all eligible group locations separately from each value's finite observations
    first = next(iter(values.values()))
    reduction_shape = tuple(first.shape)
    if use_dask:
        membership = da.ones(reduction_shape, chunks=first.chunks, dtype=float)
        if mask is not None:
            membership = da.where(mask, 1.0, np.nan)
    else:
        membership = np.ones(reduction_shape, dtype=float)
        if mask is not None:
            membership = np.where(mask, 1.0, np.nan)
    scheduled = {
        "membership": _groupby_reduce_flox(
            membership,
            group_values,
            expected_groups,
            function="count",
            use_dask=use_dask,
            fill_value=0,
            dtype=np.int64,
        ),
        "count": _groupby_reduce_flox(
            stacked,
            group_values,
            expected_groups,
            function="count",
            use_dask=use_dask,
            fill_value=0,
            dtype=np.int64,
        ),
    }

    # Ask Flox only for the reductions needed by the requested statistics
    aliases = {alias for alias in statistics.aliases if alias is not None}
    direct_functions = {
        "mean": "nanmean",
        "median": "nanmedian",
        "max": "nanmax",
        "min": "nanmin",
        "sum": "nansum",
        "std": "nanstd",
    }
    for alias, function in direct_functions.items():
        if alias in aliases:
            scheduled[alias] = _groupby_reduce_flox(
                stacked,
                group_values,
                expected_groups,
                function=function,
                use_dask=use_dask,
                fill_value=np.nan,
                dtype=np.float64,
                finalize_kwargs={"ddof": 0} if alias == "std" else None,
            )

    # Reduce squared values for sum of squares and RMSE without assigning combined group IDs
    squared = np.square(stacked)
    if "sumofsquares" in aliases:
        scheduled["sumofsquares"] = _groupby_reduce_flox(
            squared,
            group_values,
            expected_groups,
            function="nansum",
            use_dask=use_dask,
            fill_value=np.nan,
            dtype=np.float64,
        )
    if "rmse" in aliases:
        scheduled["mean_square"] = _groupby_reduce_flox(
            squared,
            group_values,
            expected_groups,
            function="nanmean",
            use_dask=use_dask,
            fill_value=np.nan,
            dtype=np.float64,
        )

    # Request the quantiles shared by the percentile, IQR and LE90 statistics
    quantiles = {}
    if "90thpercentile" in aliases:
        quantiles["q90"] = 0.90
    if "iqr" in aliases:
        quantiles.update(q75=0.75, q25=0.25)
    if "le90" in aliases:
        quantiles.update(q95=0.95, q05=0.05)
    for name, quantile in quantiles.items():
        scheduled[name] = _groupby_reduce_flox(
            stacked,
            group_values,
            expected_groups,
            function="nanquantile",
            use_dask=use_dask,
            fill_value=np.nan,
            dtype=np.float64,
            finalize_kwargs={"q": quantile},
        )

    # Compute all small Dask results together before constructing the Pandas output
    if use_dask:
        import dask

        keys = list(scheduled)
        scheduled = dict(zip(keys, dask.compute(*(scheduled[key] for key in keys))))
    membership_count = np.asarray(scheduled["membership"], dtype=np.int64).reshape(total_groups)
    finite_count = np.asarray(scheduled["count"], dtype=np.int64).reshape(len(values), total_groups)

    # Restore GeoUtils statistic names and count fields for each selected value
    columns = {}
    for value_index in range(len(values)):
        columns[(value_index, "count")] = finite_count[value_index]
        for name, statistic_alias in zip(statistics.names, statistics.aliases):
            if statistic_alias == "validcount":
                result = finite_count[value_index]
            elif statistic_alias == "totalcount":
                result = membership_count
            elif statistic_alias == "percentagevalidpoints":
                result = np.divide(
                    100 * finite_count[value_index],
                    membership_count,
                    out=np.full(total_groups, np.nan),
                    where=membership_count > 0,
                )
            elif statistic_alias == "rmse":
                result = np.sqrt(np.asarray(scheduled["mean_square"])[value_index].reshape(total_groups))
            elif statistic_alias == "90thpercentile":
                result = np.asarray(scheduled["q90"])[value_index].reshape(total_groups)
            elif statistic_alias == "iqr":
                result = np.asarray(scheduled["q75"])[value_index].reshape(total_groups) - np.asarray(scheduled["q25"])[
                    value_index
                ].reshape(total_groups)
            elif statistic_alias == "le90":
                result = np.asarray(scheduled["q95"])[value_index].reshape(total_groups) - np.asarray(scheduled["q05"])[
                    value_index
                ].reshape(total_groups)
            else:
                assert statistic_alias is not None
                result = np.asarray(scheduled[statistic_alias])[value_index].reshape(total_groups)
            columns[(value_index, name)] = result
    table = pd.DataFrame(columns, index=np.arange(total_groups))
    table = table.loc[membership_count > 0]

    # Use the common formatter to restore declared group labels, empty groups and output metadata
    return _format_grouped_stats(
        table,
        statistics,
        "flox",
        value_names=list(values),
        by_names=list(by),
        levels=levels,
        total_groups=total_groups,
        full_group_ids=np.empty(0, dtype=np.int64) if full_group_ids is None else full_group_ids,
        shape=shape,
        support=support,
        observed=observed,
        subsample=subsample,
        subsample_per_group=False,
        subsampling_strategy=subsampling_strategy,
        return_masks=False,
    )


def _calculate_grouped_stats(
    values: ArrayLike | Mapping[str, ArrayLike],
    by: Mapping[str, _PreparedGrouper],
    *,
    statistics: _Statistics,
    mask: Any | None,
    subsample: int | float,
    subsample_per_group: bool,
    random_state: int | np.random.Generator | None,
    strategy: Literal["auto", "dense", "sparse", "groupwise"],
    backend: Literal["geoutils", "flox"],
    subsampling_strategy: Literal["sequential", "topk"],
    observed: bool,
    return_masks: bool,
    support: RasterBase | PointCloudBase | None,
    mp_config: MultiprocConfig | None,
) -> pd.DataFrame | tuple[pd.DataFrame, Mapping[Hashable, RasterLike | PointCloudLike | ArrayLike]]:
    """
    Calculate statistics for values with prepared group definitions and aligned grouping values.

    Both backends first normalize their array inputs and derive bin edges when only a bin count was given. The Flox
    backend then lets Flox match group values and calculate the result. The GeoUtils backend assigns combined group
    IDs, optionally samples them, and passes them to _reduce_values(). Both paths finish with _format_grouped_stats().
    """

    # 1/ Normalize array inputs (eager vs Dask), chunk layout
    named_values: dict[str, Any] = dict(values) if isinstance(values, Mapping) else {"value": values}
    if not named_values or any(not isinstance(name, str) or not name for name in named_values):
        raise ValueError("Argument ``values`` must contain at least one non-empty name.")
    all_inputs = [*named_values.values(), *(grouper.values for grouper in by.values()), mask]
    use_reader_mp = mp_config is not None and any(isinstance(value, _ValueReader) for value in all_inputs)
    if use_reader_mp:
        assert mp_config is not None
        arrays, groupers, mask, shape, tiles = _normalize_grouped_inputs_mp(named_values, by, mask, mp_config)
        use_dask, chunks = False, None
    else:
        arrays, groupers, mask, shape, use_dask, chunks = _normalize_grouped_inputs_dask_eager(
            named_values, by, mask, mp_config
        )
        tiles = None

    # 2/ Derive bin edges from each grouper min/max when only a bin count was passed (e.g., 20) for a grouper
    if use_reader_mp:
        assert mp_config is not None and tiles is not None
        groupers = _derive_bin_edges_from_count_mp(groupers, mask, tiles, mp_config)
    else:
        groupers = _derive_bin_edges_from_count_dask_eager(
            groupers,
            mask=mask,
            use_dask=use_dask,
            chunks=chunks,
        )

    # Let Flox match the grouping values directly; combined group IDs are only needed for global subsampling
    if backend == "flox":
        full_group_ids = None
        if subsample != 1:
            full_group_ids, _, _ = _assign_group_ids_dask_eager(
                groupers,
                mask=mask,
                shape=shape,
                use_dask=use_dask,
                chunks=chunks,
            )
            sample_inputs = {
                **{f"value_{index}": array for index, array in enumerate(arrays.values())},
                **{f"grouper_{index}": grouper.values for index, grouper in enumerate(groupers.values())},
            }
            sampled, _ = _sample_grouped_values_dask_eager(
                sample_inputs,
                full_group_ids,
                subsample=subsample,
                subsample_per_group=False,
                random_state=random_state,
                subsampling_strategy=subsampling_strategy,
                use_dask=use_dask,
            )
            arrays = {name: sampled[f"value_{index}"] for index, name in enumerate(arrays)}
            groupers = {
                name: replace(grouper, values=sampled[f"grouper_{index}"])
                for index, (name, grouper) in enumerate(groupers.items())
            }
            mask = None
        return _calculate_grouped_stats_flox(
            arrays,
            groupers,
            statistics=statistics,
            mask=mask,
            full_group_ids=full_group_ids,
            shape=shape,
            support=support,
            observed=observed,
            subsample=subsample,
            subsampling_strategy=subsampling_strategy,
        )

    # 3/ Assign an integer ID to each group before sampling
    assignment_mp = None
    keep_mp_storage = False
    try:
        if use_reader_mp:
            assert mp_config is not None and tiles is not None
            assignment_mp = _assign_group_ids_mp(groupers, mask, shape, tiles, mp_config)
            full_group_ids = assignment_mp.group_ids
            levels = assignment_mp.levels
            total_groups = assignment_mp.total_groups
        else:
            full_group_ids, levels, total_groups = _assign_group_ids_dask_eager(
                groupers,
                mask=mask,
                shape=shape,
                use_dask=use_dask,
                chunks=chunks,
            )

        # 4/ Optionally subsample the common support, globally or per group
        group_ids = full_group_ids
        if subsample != 1:
            if mp_config is not None:
                arrays, group_ids = _sample_grouped_values_mp(
                    arrays,
                    full_group_ids,
                    shape=shape,
                    subsample=subsample,
                    subsample_per_group=subsample_per_group,
                    random_state=random_state,
                    subsampling_strategy=subsampling_strategy,
                    mp_config=mp_config,
                    assignment=assignment_mp,
                )
            else:
                arrays, group_ids = _sample_grouped_values_dask_eager(
                    arrays,
                    full_group_ids,
                    subsample=subsample,
                    subsample_per_group=subsample_per_group,
                    random_state=random_state,
                    subsampling_strategy=subsampling_strategy,
                    use_dask=use_dask,
                )

        # 5/ Reduce every grouped input for all the statistics defined by the user
        table, resolved_strategy = _reduce_values(
            list(arrays.values()),
            statistics,
            group_ids=group_ids,
            total_groups=total_groups,
            strategy=strategy,
            mp_config=mp_config if subsample == 1 else None,
        )
        result = _format_grouped_stats(
            table,
            statistics,
            resolved_strategy,
            value_names=list(arrays),
            by_names=list(by),
            levels=levels,
            total_groups=total_groups,
            full_group_ids=full_group_ids,
            shape=shape,
            support=support,
            observed=observed,
            subsample=subsample,
            subsample_per_group=subsample_per_group,
            subsampling_strategy=subsampling_strategy,
            return_masks=return_masks,
            group_numbers=(
                np.array(sorted(assignment_mp.observed_ids), dtype=np.int64)
                if observed and assignment_mp is not None
                else None
            ),
        )

        # Keep files available for masks returned after multiprocessing has finished
        if return_masks and assignment_mp is not None:
            weakref.finalize(result[1], assignment_mp.storage.close)
            keep_mp_storage = True
        return result
    finally:
        if assignment_mp is not None and not keep_mp_storage:
            assignment_mp.storage.close()


#########################################
# 8/ SELECT SPATIAL GROUPING VALUES
#########################################


def _vector_group_values(
    vector: VectorLike,
    selector: str | None,
    *,
    support: RasterBase | PointCloudBase,
    support_dataframe: gpd.GeoDataFrame | None,
    name: str,
    definition: _GroupDefinition | None,
    mp_config: MultiprocConfig | None,
) -> _PreparedGrouper:
    """
    Create grouping values on the common support from vector features or one selected attribute column.

    Without a selector (feature column), return a boolean inside/outside grouper.
    With a feature column, sample integer category numbers directly, allowing nonnumeric labels on raster and
    point support.

    The "support" argument follows _grouped_stats(), and "mp_config" follows stats().

    :param vector: Vector or GeoDataFrame providing feature geometries and optional category attributes.
    :param selector: Feature column containing category labels, or None to group by vector coverage.
    :param support_dataframe: Point coordinates and attributes on the selected support, or None for raster support.
    :param name: Name of this grouping variable in the result.
    :param definition: Optional validated bin or category declaration; otherwise infer labels from the selected feature
        column, excluding missing labels.
    :returns: Grouping values or category codes on the selected support, together with their group definition.
    """

    # 1/ Read the vector and distinguish boolean inside/outside grouping from a feature attribute ID grouping
    dataframe = _as_geodataframe(vector)
    encoded = False

    # Treat a vector without a selected column as a grouping variable for inside and outside
    if selector is None:
        if definition is not None and not isinstance(definition.groups, pd.CategoricalIndex):
            raise ValueError("Vector values require an explicit feature column.")
        feature_values = np.ones(len(dataframe))
        definition = definition or _resolve_group_definition(name, categories=(False, True))

    # 2/ Read feature labels once and keep the input order
    else:
        if selector not in dataframe.columns:
            raise ValueError(f"Vector column {selector!r} does not exist.")
        labels = dataframe[selector]
        if definition is None:
            if isinstance(labels.dtype, pd.CategoricalDtype):
                definition = _resolve_group_definition(name, values=labels)
            else:
                category_values = pd.unique(labels[labels.notna()])
                if not len(category_values):
                    raise ValueError(f"Vector column {selector!r} has no categories.")
                definition = _resolve_group_definition(name, categories=category_values)
        if isinstance(definition.groups, pd.CategoricalIndex):
            feature_values = pd.Categorical(labels, categories=definition.groups.categories, ordered=True).codes
            encoded = True
        else:
            feature_values = _prepare_grouper(labels, definition).values
            if not pd.api.types.is_numeric_dtype(feature_values.dtype):
                raise TypeError("Selected vector values must be numeric.")

    # 3/ Use the same vector sampling as cosample() and keep Dask results lazy
    if mp_config is not None:
        values = _reader_from_vector(dataframe, feature_values, support, mp_config, coverage=selector is None)
        if values is not None:
            return _PreparedGrouper(values, definition, encoded=encoded)
    if support_dataframe is None and _is_pointcloud(support):
        support_dataframe = cast("PointCloudBase", support).ds
    values = _sample_vector_values(dataframe, feature_values, support, support_dataframe, mp_config=mp_config)
    if selector is None:
        values = np.isfinite(values)
    return _PreparedGrouper(values, definition, encoded=encoded)


def _select_groupers_at_support(
    source: RasterLike | PointCloudLike | ArrayLike | Mapping[str, ArrayLike],
    by: Mapping[str, Any],
    *,
    support: RasterBase | PointCloudBase | None,
    definitions: Mapping[str, _GroupDefinition],
    interpolation: InterpolationMethod,
    align: Literal["raise", "reproject"],
    mp_config: MultiprocConfig | None,
    stack: ExitStack | None = None,
) -> dict[str, _PreparedGrouper]:
    """
    Reproject every grouping variable on the common spatial support.

    :param support: Raster or point cloud defining common support grid or point locations. None means the
        source and grouping inputs are already aligned arrays without spatial support.
    :param definitions: Validated bins or categories provided for grouping variables.
    :param stack: Optional ExitStack for aligned temporary files from Multiprocessing chunked execution that must
        remain available during reduction.

    :returns: Prepared grouping values on the selected support, in the input order.
    """

    # Use the locations already selected for values and masks by the shared statistics preparation
    support_dataframe = None
    groupers_at_support: dict[str, _PreparedGrouper] = {}
    source_support = _get_raster_interface(source)
    if source_support is None:
        source_support = _get_pointcloud_interface(source)

    # Align each grouper to the values common grid, point order, or array shape
    definition: _GroupDefinition | None
    for name, specification in by.items():
        # Keep ordinary array groupers unchanged when the source has no spatial support
        if support is None:
            definition = definitions.get(name) or _resolve_group_definition(name, values=specification)
            groupers_at_support[name] = _prepare_grouper(specification, definition)
            continue
        group_source, group_selector = _sampling_specification(source, specification)

        # Treat Xarrays without spatial coordinates as arrays
        group_source = _normalize_sampling_input(group_source)

        # Detect type of grouping variable: raster, point cloud, or vector
        is_raster = _is_raster(group_source)
        is_pointcloud = _is_pointcloud(group_source)
        is_vector = not is_raster and not is_pointcloud and _is_vector(group_source)

        # Native arrays already follow the source locations, so keep their masks and category metadata intact
        if not (is_raster or is_pointcloud or is_vector) and source_support is support:
            definition = definitions.get(name) or _resolve_group_definition(name, values=group_source)
            prepared = _prepare_grouper(group_source, definition)
            if _is_raster(support) and getattr(prepared.values, "ndim", 0) == 3 and prepared.values.shape[0] == 1:
                prepared = _PreparedGrouper(prepared.values[0], definition, prepared.encoded)
            groupers_at_support[name] = prepared
            continue

        # Keep native file columns and matching raster bands unloaded for worker processes
        if mp_config is not None:
            definition = definitions.get(name)
            if definition is None and is_raster and pd.api.types.is_bool_dtype(group_source.dtype):
                definition = _resolve_group_definition(name, values=group_source)
            categorical = definition is not None and isinstance(definition.groups, pd.CategoricalIndex)
            file_values = _reader_from_source(
                group_source,
                group_selector,
                support,
                mp_config,
                align=align,
                interpolation="nearest" if categorical else interpolation,
                stack=stack,
            )
            if file_values is not None:
                definition = definition or _resolve_group_definition(name, values=file_values)
                groupers_at_support[name] = _PreparedGrouper(file_values, definition)
                continue

        # Use feature labels for categorical zones
        if is_vector:
            if group_selector is not None and not isinstance(group_selector, str):
                raise TypeError(f"Vector selector for {name!r} must be a column name.")
            groupers_at_support[name] = _vector_group_values(
                group_source,
                group_selector,
                support=support,
                support_dataframe=support_dataframe,
                name=name,
                definition=definitions.get(name),
                mp_config=mp_config,
            )
            continue

        # Read point coordinates only when placing values actually requires them
        if support_dataframe is None and _is_pointcloud(support):
            support_dataframe = cast("PointCloudBase", support).ds

        # Preserve point cloud category labels and order before converting the column to an array
        metadata_values = group_source
        pointcloud = _get_pointcloud_interface(group_source)
        if pointcloud is not None:
            column = pointcloud.data_column if group_selector is None else group_selector
            if column is not None:
                if not isinstance(column, str) or column not in pointcloud.ds.columns:
                    raise ValueError(f"Point column {column!r} selected for {name!r} does not exist.")
                metadata_values = pointcloud.ds[column]
        definition = definitions.get(name) or _resolve_group_definition(name, values=metadata_values)

        if (
            pointcloud is not None
            and _is_pointcloud(support)
            and isinstance(getattr(metadata_values, "dtype", None), pd.CategoricalDtype)
        ):
            # Validate point locations before reading native category codes, avoiding conversion to object arrays
            with ExitStack() as temporary_files:
                intermediate = temporary_files.enter_context(mp_config.temporary()) if mp_config is not None else None
                aligned = _aligned_pointcloud(pointcloud, support, name, align, mp_config=intermediate)
                groupers_at_support[name] = _prepare_grouper(aligned.ds[column], definition)
            continue

        # Always use nearest neighbor for categorical raster labels, otherwise interpolation from the user
        grouped_values = _values_at_support(
            group_source,
            group_selector,
            input_support=source,
            support=support,
            support_dataframe=support_dataframe,
            name=name,
            interpolation="nearest" if isinstance(definition.groups, pd.CategoricalIndex) else interpolation,
            align=align,
            mp_config=mp_config,
        )
        groupers_at_support[name] = _prepare_grouper(grouped_values, definition)

    return groupers_at_support


#####################
# 9/ PARENT FUNCTION
#####################


def _grouped_stats(
    source: RasterLike | PointCloudLike | ArrayLike | Mapping[str, ArrayLike],
    by: Mapping[str, Any],
    *,
    values: ArrayLike | Mapping[str, ArrayLike],
    support: RasterBase | PointCloudBase | None,
    statistics: _Statistics,
    mask: Any | None,
    subsample: int | float,
    subsample_per_group: bool,
    random_state: int | np.random.Generator | None,
    strategy: Literal["auto", "dense", "sparse", "groupwise"],
    backend: Literal["geoutils", "flox"],
    subsampling_strategy: Literal["sequential", "topk"],
    interpolation: InterpolationMethod,
    align: Literal["raise", "reproject"],
    observed: bool,
    return_masks: bool,
    mp_config: MultiprocConfig | None,
    definitions: Mapping[str, _GroupDefinition],
    stack: ExitStack | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, Mapping[Hashable, RasterLike | PointCloudLike | ArrayLike]]:
    """
    Calculate grouped statistics after selecting values (source input) and groupers (grouping variables from ``by``) on
    one common spatial support.

    Logic:
    _select_groupers_at_support() places groupers on the same common support (grid or points), then
    _calculate_grouped_stats() derives missing bin edges, applies optional subsampling, calls the selected grouped
    reduction backend, and formats the result table.

    See stats() for all argument descriptions.

    :returns: Grouped dataframe, and optionally group masks.
    """

    # This is the geospatial-specific step, in addition to the _select step already done before in stats():
    # We reproject every grouper variable on the common support (raster or point)
    groupers_at_support = _select_groupers_at_support(
        source,
        by,
        support=support,
        definitions=definitions,
        interpolation=interpolation,
        align=align,
        mp_config=mp_config,
        stack=stack,
    )

    # Apply optional subsampling and calculate the result table with the selected grouped reduction backend
    return _calculate_grouped_stats(
        values,
        groupers_at_support,
        statistics=statistics,
        mask=mask,
        subsample=subsample,
        subsample_per_group=subsample_per_group,
        random_state=random_state,
        strategy=strategy,
        backend=backend,
        subsampling_strategy=subsampling_strategy,
        observed=observed,
        return_masks=return_masks,
        support=support,
        mp_config=mp_config,
    )


###########################
# 10/ GROUPED STAT PLOTTING
###########################


def _plot_axis(index: pd.Index) -> tuple[NDArray[Any], NDArray[Any], list[str] | None]:
    """
    Return plot edges, centers, and optional text labels for one grouping variable.

    Continuous binned intervals use their numeric bin bounds.
    Other intervals (categorical, zonal, or non-continuous bins) use equally spaced positions with labels so that
    gaps or nonnumeric values do not distort the plotted group order.

    :param index: Ordered labels for one grouping variable from the table passed to plot_grouped_stats().
    :returns: Bin edges, bin centers and optional tick labels, with one center per group.
    """

    # Draw continuous numeric intervals with their true widths
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
    """
    Plot grouped statistics for one or two grouping variables with their sample counts shown as side histograms.

    One grouping variable produces a curve below its histogram counts.
    Two grouping variables produce a colored grid, with count histograms above and to the right.

    Passing ``min_count`` hides estimates without removing observations from the count panels.

    :param table: Grouped dataframe returned by stats() or an object stats() method.
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

    # 1/ Validate the selected value, statistic, and group dimensions

    # Import Matplotlib only when the caller requests a plot
    matplotlib = import_optional("matplotlib")
    import matplotlib.pyplot as plt

    # Require the named column levels produced by stats() before looking up the selected values
    if not isinstance(table, pd.DataFrame) or not isinstance(table.columns, pd.MultiIndex):
        raise TypeError("Argument ``table`` must be a grouped statistics dataframe with MultiIndex columns.")
    if list(table.columns.names) != ["value", "statistic"]:
        raise ValueError("Argument ``table`` must have 'value' and 'statistic' column levels.")

    # Infer the value column only when exactly one is available
    available_values = list(dict.fromkeys(table.columns.get_level_values("value")))
    if value is None:
        if len(available_values) != 1:
            raise ValueError("Argument ``value`` must be selected when ``table`` contains multiple values.")
        value = available_values[0]

    # Every plot needs counts to show how many finite observations support each estimate
    if (value, statistic) not in table.columns or (value, "count") not in table.columns:
        raise ValueError(f"Value {value!r} must contain both {statistic!r} and 'count' statistics.")
    if table.index.nlevels not in {1, 2}:
        raise ValueError("plot_grouped_stats supports one or two group dimensions.")
    if min_count < 0:
        raise ValueError("Argument ``min_count`` cannot be negative.")

    # 2/ Prepare a common plotting area for the statistic and its count panels

    # Use the supplied axes as the plot area or create a new figure
    if ax is None:
        figure = plt.figure(figsize=(7, 6))
        frame = figure.add_axes((0.1, 0.1, 0.8, 0.8))
    elif isinstance(ax, matplotlib.axes.Axes):
        frame = ax
        figure = ax.figure
    else:
        raise TypeError("Argument ``ax`` must be a Matplotlib Axes or None.")
    frame.set_axis_off()

    # 3/ Draw one-dimensional curves or a two-dimensional grid with matching counts

    # Draw counts above the statistic for one grouping variable
    if table.index.nlevels == 1:
        count_axis = frame.inset_axes((0.0, 0.72, 1.0, 0.28))
        statistic_axis = frame.inset_axes((0.0, 0.0, 1.0, 0.64))
        edges, centers, labels = _plot_axis(table.index)

        # Hide unsupported estimates while keeping their counts visible in the panel above
        counts = table[(value, "count")].to_numpy(dtype=float)
        values = table[(value, statistic)].where(table[(value, "count")] >= min_count).to_numpy(dtype=float)

        # Match count-bar widths to the numeric intervals or category positions
        count_axis.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="0.7", edgecolor="white")
        count_axis.set_xlim(edges[0], edges[-1])
        count_axis.set_ylabel("Count")
        count_axis.tick_params(axis="x", labelbottom=False)

        # Align the statistic with the same group centers and display labels for nonnumeric positions
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

        # Give the statistic grid the same extent as both count panels
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

        # Expose all created axes so callers can adapt the finished figure
        colorbar = figure.colorbar(mesh, ax=statistic_axis, label=statistic)
        axes = {
            "count_x": count_x_axis,
            "count_y": count_y_axis,
            "statistic": statistic_axis,
            "colorbar": colorbar.ax,
        }

    # 4/ Save after every panel and label has been added
    if savefig_fname is not None:
        figure.savefig(savefig_fname, bbox_inches="tight")
    return axes
