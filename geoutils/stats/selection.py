# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Select reprojected values, masks and samples for statistics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, replace
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from geoutils._dispatch import (
    _get_pointcloud_interface,
    _get_raster_interface,
    _is_pointcloud,
    _is_raster,
    is_dask_array,
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
from geoutils.sampling.subsampling import _sample_valid_indices, _subsample_numpy
from geoutils.sampling.support import (
    _as_array,
    _mask_at_support,
    _normalize_mask_array,
    _normalize_sampling_input,
    _sampling_specification,
    _sampling_support,
    _values_at_support,
)

if TYPE_CHECKING:
    from geoutils.interface.interpolation import InterpolationMethod
    from geoutils.multiproc.mparray import MultiprocConfig
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterBase, RasterLike
    from geoutils.vector.base import VectorLike


@dataclass(frozen=True)
class _SelectionCounts:
    """Store value validity before masking and the number of locations selected by the mask."""

    valid_before_mask: Any
    selected_locations: Any


########################################
# 1/ SELECT A COMMON SPATIAL SUPPORT
########################################


def _select_values_and_mask_at_support(
    source: RasterLike | PointCloudLike | ArrayLike | Mapping[str, ArrayLike],
    *,
    by: Mapping[str, Any] | None,
    values: int | str | Iterable[int | str] | Mapping[str, Any] | None,
    at: Literal["self"] | RasterLike | PointCloudLike | None,
    mask: RasterLike | PointCloudLike | VectorLike | ArrayLike | None,
    mask_mode: str,
    interpolation: InterpolationMethod,
    align: Literal["raise", "reproject"],
    mp_config: MultiprocConfig | None,
    stack: ExitStack | None = None,
) -> tuple[Any, Any | None, RasterBase | PointCloudBase | None]:
    """
    Select values and an optional mask on the same support for summary and grouped statistics.

    _sampling_support() chooses the grid or point locations, and _values_at_support() aligns selected bands, columns,
    or named inputs to them. Keep native arrays unchanged when no alignment is needed so reductions retain their
    masked-array behavior and Dask laziness.

    All arguments follow stats().

    :returns: Selected values, a boolean eligibility mask or None, and the raster or point cloud defining their
        locations. Plain array inputs keep their input representation and have no spatial support.
    """

    # 1/ Check user input before reading any possibly lazy values
    if mask_mode not in {"inside", "outside"}:
        raise ValueError("Argument ``mask_mode`` must be 'inside' or 'outside'.")
    if align not in {"raise", "reproject"}:
        raise ValueError("Argument ``align`` must be 'raise' or 'reproject'.")
    if isinstance(at, str) and at != "self":
        raise ValueError("Argument ``at`` must be 'self' or a raster or point cloud support object.")

    # Use the shared dispatch checks, treating Xarrays without spatial coordinates as plain arrays
    spatial_source = _normalize_sampling_input(source)
    is_raster = _is_raster(spatial_source)
    is_pointcloud = _is_pointcloud(spatial_source)
    if not is_raster and not is_pointcloud:
        if values is not None or at is not None:
            raise ValueError("Arguments ``values`` and ``at`` require a raster or point cloud source.")
        return source, mask, None
    source = _get_raster_interface(spatial_source) if is_raster else _get_pointcloud_interface(spatial_source)
    raster = cast("RasterBase", source) if is_raster else None

    # Choose the same output grid or point locations for selected values and groupers
    inputs = [
        source,
        *(by.values() if by is not None else []),
        *(values.values() if isinstance(values, Mapping) else []),
    ]
    support = _sampling_support(inputs, source if isinstance(at, str) else at)
    support_dataframe = None

    # 2/ Name the requested values and read them on the selected grid or points
    # Preserve existing summary band labels and explicit names supplied in a mapping
    if isinstance(values, Mapping):
        value_specs = dict(values)
    elif raster is not None:
        # Interpret numeric selections as one-based bands; omitting values selects every band
        bands: Iterable[Any]
        if values is None:
            bands = range(1, raster.count + 1)
        elif isinstance(values, (int, np.integer)):
            bands = [values]
        else:
            bands = values
        if not isinstance(bands, Iterable) or isinstance(bands, (str, bytes)):
            raise TypeError("Raster ``values`` must select one or more integer band numbers.")
        value_specs = {}
        for selected_band in bands:
            if not isinstance(selected_band, (int, np.integer)) or not 1 <= selected_band <= raster.count:
                raise ValueError("Raster bands must be integers between one and the raster band count.")
            name = f"band_{selected_band}" if by is not None else f"band {selected_band}"
            value_specs[name] = int(selected_band)
    else:
        # Use the active point column by default, or geometry heights when no column is selected
        if values is None:
            pointcloud = cast("PointCloudBase", source)
            value_specs = {pointcloud.data_column or "z": pointcloud.data_column}
        else:
            columns = [values] if isinstance(values, str) else values
            if not isinstance(columns, Iterable):
                raise TypeError("Point cloud ``values`` must select one or more column names.")
            value_specs = {}
            for column in columns:
                if not isinstance(column, str):
                    raise TypeError("Point cloud ``values`` must select column names.")
                value_specs[column] = column
    if not value_specs or any(not isinstance(name, str) or not name for name in value_specs):
        raise ValueError("Selected value names must be non-empty strings.")

    # Read native raster bands directly; interpolate or align only when the selected support requires it
    selected_values: dict[str, Any] = {}
    for name, specification in value_specs.items():
        value_source, selector = _sampling_specification(source, specification)
        # Plain Xarrays follow the selected locations; their dimensions alone do not define a raster
        value_source = _normalize_sampling_input(value_source)
        file_values = _reader_from_source(
            value_source, selector, support, mp_config, align=align, interpolation=interpolation, stack=stack
        )
        if file_values is None and mp_config is not None and not support.is_loaded:
            from geoutils._dispatch import _is_vector

            if not _is_raster(value_source) and not _is_pointcloud(value_source) and _is_vector(value_source):
                from geoutils.sampling.support import _as_geodataframe

                dataframe = _as_geodataframe(value_source)
                if selector is None or selector not in dataframe.columns:
                    raise ValueError("Vector values require an explicit feature column.")
                if not pd.api.types.is_numeric_dtype(dataframe[selector]):
                    raise TypeError("Selected vector values must be numeric.")
                file_values = _reader_from_vector(
                    dataframe, np.asarray(dataframe[selector], dtype=float), support, mp_config
                )
        if file_values is not None:
            selected_values[name] = file_values
            continue
        if value_source is raster and support is raster:
            # Read a native band directly to retain its NumPy mask and avoid unnecessary interpolation
            selected_band = 1 if selector is None else selector
            if not isinstance(selected_band, (int, np.integer)) or not 1 <= selected_band <= raster.count:
                raise ValueError("Raster bands must be integers between one and the raster band count.")
            data = raster.data
            selected_values[name] = data[selected_band - 1] if data.ndim == 3 else data
        else:
            # Let the sampling helpers align, rasterize or interpolate spatial inputs as required
            if support_dataframe is None and not _is_raster(support):
                support_dataframe = cast("PointCloudBase", support).ds
            selected_values[name] = _values_at_support(
                value_source,
                selector,
                input_support=source,
                support=support,
                support_dataframe=support_dataframe,
                name=name,
                interpolation=interpolation,
                align=align,
                mp_config=mp_config,
            )

    # 3/ Place the optional mask through the same sampling helpers used by cosample()
    # Keep value validity separate because stats() reports finite counts independently for each selected value
    file_mask = (
        _reader_from_source(mask, None, support, mp_config, align=align, interpolation="nearest", stack=stack)
        if mask is not None
        else None
    )
    if file_mask is not None:
        return selected_values, file_mask, support
    if mask is not None:
        from geoutils._dispatch import _is_vector

        if not _is_raster(mask) and not _is_pointcloud(mask) and _is_vector(mask):
            from geoutils.sampling.support import _as_geodataframe

            dataframe = _as_geodataframe(mask)
            file_mask = _reader_from_vector(
                dataframe,
                np.ones(len(dataframe)),
                support,
                mp_config,
                mask_mode=cast(Literal["inside", "outside"], mask_mode),
            )
            if file_mask is not None:
                return selected_values, file_mask, support
        elif not _is_raster(support) and not _is_raster(mask) and not _is_pointcloud(mask):
            return selected_values, _normalize_mask_array(mask, (cast("PointCloudBase", support).point_count,)), support
    if mask is not None and support_dataframe is None and not _is_raster(support) and _is_pointcloud(mask):
        support_dataframe = cast("PointCloudBase", support).ds
    support_mask = _mask_at_support(
        mask,
        support,
        support_dataframe=support_dataframe,
        mask_mode=mask_mode,
        align=align,
        mp_config=mp_config,
    )
    return selected_values, support_mask, support


####################################
# 2/ SAMPLE ELIGIBLE LOCATIONS
####################################


def _sample_eligible_indices(
    eligible: Any,
    *,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    strategy: Literal["sequential", "topk"],
) -> NDArray[Any]:
    """
    Select flat eligible positions with the same sampling rules for summary and grouped statistics.

    Grid sampling delegates to _sample_valid_indices(): ``topk`` selects the same locations across chunk layouts,
    while ``sequential`` follows each backend's traversal order. Other array dimensions use the NumPy sampler after
    computing a lazy eligibility mask. Sampling finishes here before either reduction backend receives the values.

    Subsample and random_state follow stats().

    :param eligible: Boolean array marking locations available for sampling, independently of value validity.
    :param strategy: The subsampling_strategy option from stats(), distinct from its reduction strategy.
    :returns: Selected positions in the flattened input as an in-memory integer array.
    """

    # Delegate grid sampling to the shared sampler, including its Dask task handling
    if eligible.ndim == 2:
        rows, columns = _sample_valid_indices(
            eligible, subsample=subsample, random_state=random_state, strategy=strategy
        )
        return rows * eligible.shape[1] + columns

    # Keep the existing flat sampling order for point values and in-memory inputs
    if is_dask_array(eligible):
        eligible = eligible.compute()
    (flat_indices,) = _subsample_numpy(
        np.where(np.asarray(eligible).ravel(), 1.0, np.nan),
        subsample=subsample,
        return_indices=True,
        random_state=random_state,
        strategy=strategy,
    )
    return flat_indices


def _sample_reader_indices(
    shape: tuple[int, ...],
    mask: Any,
    subsample: int | float,
    random_state: Any,
    strategy: Literal["topk", "sequential"],
    mp_config: MultiprocConfig,
) -> Any:
    """Sample one global group while reading file masks and storing eligibility in bounded blocks."""

    if math.prod(shape) == 0:
        return np.empty(0, dtype=np.int64)

    # Sample an unmasked input as one group without allocating or writing a complete group array
    if mask is None:
        ids = np.broadcast_to(np.array(0, dtype=np.int8), shape)
        return _stratified_subsample_indices(
            ids, subsample, random_state=random_state, strategy=strategy, mp_config=mp_config
        )

    with ExitStack() as storage:
        directory = storage.enter_context(TemporaryDirectory(prefix="geoutils-stats-sample-"))
        ids = np.memmap(f"{directory}/eligible.dat", mode="w+", dtype=np.int8, shape=shape)
        storage.callback(ids._mmap.close)
        tiles = list(iter_chunk_slices(shape, mp_config.chunks))
        if isinstance(mask, _ValueReader):
            arguments = ((mask.block(tile),) for tile in tiles)
            for index, block_mask in _map_bounded(mp_config.cluster, _read_values, arguments):
                # Read only the mask: value validity does not determine the common sample
                ids[tiles[index]] = np.where(np.ma.filled(block_mask, False), 0, -1)
        else:
            for tile in tiles:
                ids[tile] = np.where(np.ma.filled(mask[tile], False), 0, -1)
        return _stratified_subsample_indices(
            ids, subsample, random_state=random_state, strategy=strategy, mp_config=mp_config
        )


###########################################
# 3/ APPLY MASKS AND KEEP SUMMARY COUNTS
###########################################


def _mask_global_values(
    array: Any,
    mask: Any | None,
    selected_locations: Any | None,
    mp_config: MultiprocConfig | None = None,
) -> tuple[Any, _SelectionCounts | None]:
    """
    Apply an eligible-location mask and retain counts before and after that selection.

    Boolean mask validation belongs to _normalize_mask_array(). Here, missing values remain excluded independently of
    the user mask. An eager array stays eager unless its mask is lazy; in that case, selection must wait for Dask too.
    The original array is never changed, and its dimensions are kept for user-defined statistics.

    :param array: One selected value array, already aligned to the mask.
    :param mask: Validated boolean eligibility array of the same shape, or None to keep the input unchanged.
    :returns: Values with excluded locations masked or set to NaN, plus their validity before masking and the common
        number of selected locations. Without a mask, the counts are None.
    """

    # Keep unmasked inputs exactly as supplied, including NumPy masks and lazy Dask graphs
    if mask is None:
        return array, None

    if isinstance(array, _ValueReader):
        assert mp_config is not None
        assert selected_locations is not None
        return _mask_reader_values(array, mask, selected_locations, mp_config)
    if isinstance(mask, _ValueReader):
        mask = _normalize_mask_array(mask.read(tuple(slice(0, length) for length in mask.shape)), array.shape)
    assert selected_locations is not None

    # Count finite input values before masking so summary validity still describes the full selected array
    valid = np.isfinite(array)
    if np.ma.isMaskedArray(valid):
        valid = valid.filled(False)
    counts = _SelectionCounts(valid_before_mask=valid.sum(), selected_locations=selected_locations)

    # Defer selection only when the array or its own mask is lazy; other selected arrays do not affect this choice
    if is_dask_array(array) or is_dask_array(mask):
        import_optional("dask")
        import dask.array as da

        raw_values = np.ma.getdata(array) if np.ma.isMaskedArray(array) else array
        selected = da.where(mask & valid, raw_values, np.nan)
    else:
        # Preserve the masked-array estimators and the original shape for in-memory data
        selected = np.ma.masked_where(~mask | ~valid, array)
    return selected, counts


def _count_reader_value_block(values: Any) -> int:
    """Count finite values in one raster or point cloud reader block."""

    data = _read_values(values)
    return int(np.count_nonzero(~get_mask_from_array(data)))


def _count_reader_mask_block(mask: Any) -> int:
    """Count selected locations in one raster or point cloud mask block."""

    selected = _read_values(mask)
    return int(np.count_nonzero(np.ma.filled(selected, False)))


def _count_selected_locations(mask: Any, shape: tuple[int, ...], mp_config: MultiprocConfig | None) -> Any:
    """Count one common mask once for all selected global values."""

    if not isinstance(mask, _ValueReader):
        return mask.sum()
    assert mp_config is not None
    arguments = ((mask.block(tile),) for tile in iter_chunk_slices(shape, mp_config.chunks))
    return sum(result for _, result in _map_bounded(mp_config.cluster, _count_reader_mask_block, arguments))


def _mask_reader_values(
    array: _ValueReader, mask: Any, selected_locations: int, mp_config: MultiprocConfig
) -> tuple[_ValueReader, _SelectionCounts]:
    """Attach a mask to a reader after counting its original finite values by blocks."""

    arguments = ((array.block(tile),) for tile in iter_chunk_slices(array.shape, mp_config.chunks))

    # Bound pending reads and keep only the accumulated validity count in the caller
    valid_before_mask = 0
    for _, count in _map_bounded(mp_config.cluster, _count_reader_value_block, arguments):
        valid_before_mask += count
    counts = _SelectionCounts(valid_before_mask=valid_before_mask, selected_locations=selected_locations)
    return replace(array, mask=mask), counts


################################
# 4/ GLOBAL SAMPLING AND MASKING
################################


def _sample_and_mask_global_values(
    values: ArrayLike | Mapping[str, ArrayLike],
    *,
    mask: Any | None,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    subsampling_strategy: Literal["sequential", "topk"],
    mp_config: MultiprocConfig | None,
) -> dict[str, tuple[Any, _SelectionCounts | None]]:
    """
    Name global values, sample common locations, and apply the common mask.

    Ordinary containers are converted to NumPy or Dask arrays, while raster and point cloud readers remain unloaded.
    Sampling first chooses the same eligible locations for every value. Masking then keeps each value's missing data
    independent and records the counts needed for global output.

    Values and selection options follow _global_stats().

    :returns: Selected values and their optional mask counts, keyed by output value name.
    """

    # 1/ Name values and check their common shape
    named_values = dict(values) if isinstance(values, Mapping) else {"value": values}
    if not named_values or any(not isinstance(name, str) or not name for name in named_values):
        raise ValueError("Argument ``values`` must contain at least one non-empty name.")
    arrays = {
        name: array if isinstance(array, _ValueReader) else _as_array(array) for name, array in named_values.items()
    }
    shape = next(iter(arrays.values())).shape
    if any(array.shape != shape for array in arrays.values()):
        raise ValueError("Selected values must have matching shapes.")

    # Keep readers unloaded and reject two task schedulers in the same calculation
    mask = _normalize_reader_mask(mask, shape)
    use_dask = is_dask_array(mask) or any(is_dask_array(array) for array in arrays.values())
    if use_dask and mp_config is not None:
        raise ValueError("Dask inputs cannot be combined with Multiprocessing statistics.")

    # 2/ Select the same eligible positions from every value when sampling is requested
    had_mask = mask is not None
    if subsample != 1:
        first = next(iter(arrays.values()))
        if isinstance(mask, _ValueReader) or any(isinstance(array, _ValueReader) for array in arrays.values()):
            assert mp_config is not None
            indexes = _sample_reader_indices(shape, mask, subsample, random_state, subsampling_strategy, mp_config)
        else:
            eligible = mask if mask is not None else np.ones_like(first, dtype=bool)
            indexes = _sample_eligible_indices(
                eligible, subsample=subsample, random_state=random_state, strategy=subsampling_strategy
            )
        arrays = {
            name: (
                _read_selected_values(array, indexes, mp_config)
                if isinstance(array, _ValueReader)
                else array.reshape(-1)[indexes]
            )
            for name, array in arrays.items()
        }
        mask = np.ones(len(indexes), dtype=bool) if had_mask else None

    # 3/ Read a mask once when in-memory values need it, then share its selected count across every value
    if isinstance(mask, _ValueReader) and any(not isinstance(array, _ValueReader) for array in arrays.values()):
        mask = _normalize_mask_array(mask.read(tuple(slice(0, length) for length in shape)), shape)
    selected_locations = None if mask is None else _count_selected_locations(mask, shape, mp_config)
    return {
        name: _mask_global_values(array, mask, selected_locations, mp_config=mp_config)
        for name, array in arrays.items()
    }
