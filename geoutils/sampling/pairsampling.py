# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Sample pairs in rasters or point clouds."""

from __future__ import annotations

import sqlite3
import warnings
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from math import isqrt
from pathlib import Path
from sys import modules
from tempfile import TemporaryDirectory
from threading import Lock
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from geoutils._dispatch import (
    _get_pointcloud_interface,
    _get_raster_interface,
    get_geo_attr,
    is_dask_array,
    is_dask_dataframe,
)
from geoutils._misc import import_optional
from geoutils._typing import ArrayLike, NDArrayBool, NDArrayNum
from geoutils.multiproc.chunked import iter_chunk_slices, normalize_chunks
from geoutils.multiproc.cluster import _map_bounded
from geoutils.multiproc.readers import (
    _BlockReader,
    _normalize_reader_mask,
    _read_selected_values,
    _read_values,
    _reader_from_source,
    _reader_from_vector,
    _ValueReader,
)
from geoutils.pointcloud.writing import _array_partitions, _stage_pointcloud_partition
from geoutils.raster.array import _selected_raster_data, get_mask_from_array
from geoutils.sampling.support import (
    _as_geodataframe,
    _is_pointcloud,
    _is_raster,
    _is_vector,
    _mask_at_support,
    _mask_on_raster,
    _normalize_mask_array,
    _normalize_sampling_input,
)

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterBase, RasterLike
    from geoutils.vector.base import VectorLike

###############################################
# 1/ SHARED INPUT CHECKS AND DUPLICATE REMOVAL
##############################################


def _uses_dask(value: Any) -> bool:
    """Return whether an array or spatial interface contains Dask-backed values without loading them."""

    if value is None:
        return False
    if is_dask_array(value) or is_dask_dataframe(value):
        return True
    raster = _get_raster_interface(value)
    if raster is not None:
        return raster._chunks is not None
    pointcloud = _get_pointcloud_interface(value)
    return pointcloud is not None and pointcloud._is_dask


def _check_input_pair_output(
    output_chunks: int,
    output_storage: str = "memory",
    output_directory: str | Path | None = None,
    force_output_to_memory: bool = False,
    mp_config: MultiprocConfig | None = None,
) -> None:
    """Check output options before pair selection reads any values."""

    if not isinstance(force_output_to_memory, (bool, np.bool_)):
        raise TypeError("Argument force_output_to_memory must be a boolean.")
    if not isinstance(output_chunks, (int, np.integer)) or isinstance(output_chunks, bool) or output_chunks < 1:
        raise ValueError("Argument output_chunks must be a positive integer.")
    if output_storage not in {"memory", "disk"}:
        raise ValueError("Argument output_storage must be 'memory' or 'disk'.")
    if output_directory is not None:
        if output_storage != "disk":
            raise ValueError("Argument output_directory requires output_storage='disk'.")
        if not Path(output_directory).is_dir():
            raise ValueError("Argument output_directory must be an existing directory.")
    if output_storage == "disk":
        if mp_config is None:
            raise ValueError("Argument output_storage='disk' requires mp_config. Save Dask output with Xarray instead.")
        import_optional("dask")


def _unique_pairs(
    first: NDArrayNum, second: NDArrayNum, *, n_observations: int
) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum]:
    """
    Find first occurrences of unordered pairs so indexes and cached values can use the same selection.

    Return endpoint indexes with the smaller index first, and their positions in the original arrays.
    """

    # Put the smaller row number first so A-B and B-A have the same key
    low = np.minimum(first, second).astype(np.int64, copy=False)
    high = np.maximum(first, second).astype(np.int64, copy=False)
    keys = low * np.int64(n_observations) + high

    # Restore input order after selecting the first occurrence of each key
    _, positions = np.unique(keys, return_index=True)
    positions.sort()
    return low[positions], high[positions], positions


def _deduplicate_pairs(first: NDArrayNum, second: NDArrayNum, *, n_observations: int) -> tuple[NDArrayNum, NDArrayNum]:
    """
    Remove repeated pairs while treating A-B and B-A as the same pair.

    Endpoint arrays follow the layout described by _pair_dataset().

    :param n_observations: Total number of source cells or rows, used to give each unordered pair a unique integer key.
    :returns: First and second endpoint indexes with the smaller index first, preserving first occurrence order.
    """

    first, second, _ = _unique_pairs(first, second, n_observations=n_observations)
    return first, second


#############################
# 2/ RASTER INPUT PREPARATION
#############################


def _count_finite_reader_block(block: Any) -> int:
    """Count finite and unmasked values in one multiprocessing reader block."""

    values = _read_values(block)
    return int(np.count_nonzero(~get_mask_from_array(values)))


def _pack_raster_validity(values: Any) -> tuple[NDArrayNum, int]:
    """Pack finite, unmasked cells in row order and count them before discarding a raster chunk."""

    if isinstance(values, _BlockReader):
        values = _read_values(values)
    valid = ~get_mask_from_array(values)
    return np.packbits(valid.ravel(), bitorder="little"), int(np.count_nonzero(valid))


def _iter_packed_dask_validity(array: Any) -> Iterator[tuple[int, tuple[NDArrayNum, int]]]:
    """
    Pack Dask chunks on the selected scheduler and gather at most 64 packed masks at once.

    An active distributed scheduler receives one scan graph. Workers retain only packed masks until their batch
    is gathered; source values are discarded after packing. Local schedulers compute one batch at a time, and
    explicit scheduler settings take precedence over an active distributed client.
    """

    dask = import_optional("dask")
    from dask.base import get_scheduler

    # Use Dask's scheduler selection so synchronous overrides never submit tasks to an existing client
    scheduler = get_scheduler(collections=[array])
    distributed = modules.get("distributed")
    client = getattr(scheduler, "__self__", None)
    if distributed is None or not isinstance(client, distributed.Client):
        client = None
    tasks = [dask.delayed(_pack_raster_validity)(block) for block in array.to_delayed().ravel()]
    futures = None if client is None else client.compute(tasks)

    # Submit the scan once on a cluster, avoiding repeated graph optimization and scheduling per result batch
    try:
        for start in range(0, len(tasks), 64):
            if futures is None:
                states = dask.compute(*tasks[start : start + 64])
            else:
                assert client is not None
                states = client.gather(futures[start : start + 64])
            yield from enumerate(states, start=start)
            del states
            if futures is not None:
                for future in futures[start : start + 64]:
                    future.release()
    finally:
        # Release completed or failed futures even when the scan or assembly stops early
        if futures is not None:
            for future in futures:
                future.release()


@dataclass(frozen=True)
class _PackedRasterValidity:
    """Store one validity bit per cell, with byte offsets for independently packed spatial chunks."""

    bits: NDArrayNum
    offsets: NDArrayNum
    count: int


@dataclass(frozen=True)
class _RasterPairSource:
    """
    Read raster pair values through NumPy, Dask, or multiprocessing without changing pair generation.

    The raster pair sampler uses shape and chunk_edges to generate endpoints. count_finite() and read_pairs()
    isolate the backend-specific value access needed to reject endpoints with nodata values.
    """

    values: Any
    mp_config: MultiprocConfig | None = None
    validity: _PackedRasterValidity | None = field(default=None, compare=False, repr=False)
    shape: tuple[int, int] = field(init=False)
    chunk_edges: tuple[NDArrayNum, NDArrayNum] = field(init=False)

    def __post_init__(self) -> None:
        """Check the two-dimensional source and record its actual processing chunk boundaries."""

        # Require a raster grid and a multiprocessing configuration for file readers
        shape = tuple(int(length) for length in self.values.shape)
        if len(shape) != 2:
            raise ValueError("Raster pair values must have two dimensions.")
        if isinstance(self.values, _ValueReader) and self.mp_config is None:
            raise ValueError("A multiprocessing value reader requires ``mp_config``.")

        # Match the chunks that determine anchor locality for Dask and multiprocessing
        if is_dask_array(self.values):
            chunks = self.values.chunks
        elif self.mp_config is not None:
            chunks = normalize_chunks(self.mp_config.chunks, shape)
        else:
            chunks = normalize_chunks(2048, shape)
        edges = tuple(np.asarray((0, *np.cumsum(axis_chunks)), dtype=np.int64) for axis_chunks in chunks)

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "chunk_edges", edges)

    def count_finite(self, min_count: int | None = None) -> int:
        """
        Count finite values without loading a complete Dask array or file-backed raster.

        With min_count, stop once enough finite cells have been found and return a lower bound on the total.
        A smaller result is an exact count because every source chunk has been read.
        """

        # Reuse the count calculated while packing validity, when explicitly requested
        if self.validity is not None:
            return self.validity.count

        # Read and reduce independent file windows while retaining only their small counts
        if isinstance(self.values, _ValueReader):
            assert self.mp_config is not None
            arguments = (
                (self.values.block(slices),) for slices in iter_chunk_slices(self.shape, self.mp_config.chunks)
            )
            total = 0
            for _, count in _map_bounded(self.mp_config.cluster, _count_finite_reader_block, arguments):
                total += count
                if min_count is not None and total >= min_count:
                    break
            return total

        # Stop a partial Dask scan once its lower bound is sufficient, with only eight source chunks per batch
        if is_dask_array(self.values) and min_count is not None:
            dask = import_optional("dask")
            blocks = self.values.to_delayed().ravel()
            total = 0
            for start in range(0, len(blocks), 8):
                tasks = [dask.delayed(_count_finite_reader_block)(block) for block in blocks[start : start + 8]]
                total += sum(dask.compute(*tasks))
                if total >= min_count:
                    break
            return total

        # Let Dask reduce each source chunk and count an in-memory raster directly
        if is_dask_array(self.values):
            import dask.array as da

            return int(da.count_nonzero(da.isfinite(self.values)).compute())
        return int(np.count_nonzero(~get_mask_from_array(self.values)))

    def with_packed_validity(self) -> _RasterPairSource:
        """
        Read each spatial chunk once and assemble its validity bits in bounded batches on the client.

        Each chunk starts at a byte boundary, including uneven interior chunks. The retained bitmap uses
        approximately one eighth of the cell count in bytes; at most 64 packed chunks are assembled at once.
        No raster values are retained, and the count follows the same mask rules as candidate value reads.
        """

        # Preallocate one packed bitmap, allowing padding bits at every chunk boundary
        row_lengths, column_lengths = (np.diff(edges) for edges in self.chunk_edges)
        lengths = ((row_lengths[:, None] * column_lengths[None, :] + 7) // 8).ravel()
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        bits = np.empty(int(offsets[-1]), dtype=np.uint8)
        count = 0

        # Reduce source chunks on their existing backend and transfer only packed masks and counts
        if isinstance(self.values, _ValueReader):
            assert self.mp_config is not None
            arguments = (
                (self.values.block(slices),) for slices in iter_chunk_slices(self.shape, self.mp_config.chunks)
            )
            states = _map_bounded(self.mp_config.cluster, _pack_raster_validity, arguments)
            for index, (packed, finite_count) in states:
                bits[offsets[index] : offsets[index + 1]] = packed
                count += finite_count
        elif is_dask_array(self.values):
            for index, (packed, finite_count) in _iter_packed_dask_validity(self.values):
                bits[offsets[index] : offsets[index + 1]] = packed
                count += finite_count
        else:
            # Slice in-memory values on the same boundaries used to choose local anchors
            row_edges, column_edges = self.chunk_edges
            for index in range(len(lengths)):
                row, column = divmod(index, len(column_lengths))
                slices = (
                    slice(row_edges[row], row_edges[row + 1]),
                    slice(column_edges[column], column_edges[column + 1]),
                )
                packed, finite_count = _pack_raster_validity(self.values[slices])
                bits[offsets[index] : offsets[index + 1]] = packed
                count += finite_count

        validity = _PackedRasterValidity(bits=bits, offsets=offsets, count=count)
        return replace(self, validity=validity)

    def finite_pairs(self, first: NDArrayNum, second: NDArrayNum) -> NDArrayNum:
        """Check both endpoints against cached validity bits or the original source values."""

        if self.validity is None:
            first_values, second_values = self.read_pairs(first, second)
            return np.isfinite(first_values) & np.isfinite(second_values)

        # Locate cells within their actual source chunks, preserving cross-chunk pairs and endpoint order
        indexes = np.concatenate((first, second)).astype(np.int64, copy=False)
        rows, columns = np.divmod(indexes, self.shape[1])
        row_edges, column_edges = self.chunk_edges
        chunk_rows = np.searchsorted(row_edges, rows, side="right") - 1
        chunk_columns = np.searchsorted(column_edges, columns, side="right") - 1
        chunks = chunk_rows * (len(column_edges) - 1) + chunk_columns
        widths = np.diff(column_edges)[chunk_columns]
        local = (rows - row_edges[chunk_rows]) * widths + columns - column_edges[chunk_columns]

        # Read only the byte containing each cell's validity bit
        positions = self.validity.offsets[chunks] + (local >> 3)
        valid = ((self.validity.bits[positions] >> (local & 7)) & 1).astype(bool)
        return valid[: len(first)] & valid[len(first) :]

    def read_pairs(
        self, first: NDArrayNum, second: NDArrayNum, *, preserve_mask: bool = False
    ) -> tuple[NDArrayNum, NDArrayNum]:
        """Read two endpoint vectors together so each required source chunk or tile is read once."""

        # Group both endpoint sets in one multiprocessing request, then restore their separate order
        if isinstance(self.values, _ValueReader):
            assert self.mp_config is not None
            combined = _read_selected_values(
                self.values,
                np.concatenate((first, second)).astype(np.int64, copy=False),
                self.mp_config,
            )
            selections = [combined[: len(first)], combined[len(first) :]]
        else:
            # Convert flat indexes to rows and columns, preserving Dask's joint graph computation
            use_dask = is_dask_array(self.values)
            selections = []
            for indexes in (first, second):
                rows, columns = np.divmod(np.asarray(indexes, dtype=np.int64), self.shape[1])
                selected = self.values.vindex[rows, columns] if use_dask else self.values[rows, columns]
                selections.append(selected)
            if use_dask:
                selections = list(import_optional("dask").compute(*selections))

        # Use masks while collecting accepted values so integer endpoints never pass through floating point
        if preserve_mask:
            return selections[0], selections[1]

        # Express masked values as NaN while preserving finite integer values where possible
        for index, values in enumerate(selections):
            if np.ma.isMaskedArray(values):
                if np.ma.is_masked(values):
                    values = values.astype(np.result_type(values.dtype, np.float32)).filled(np.nan)
                else:
                    values = np.ma.getdata(values)
            selections[index] = np.asarray(values)
        return selections[0], selections[1]


def _prepare_raster_pair_source(
    raster: RasterBase,
    band: int,
    mask: RasterLike | VectorLike | ArrayLike | None,
    mp_config: MultiprocConfig | None,
) -> _RasterPairSource:
    """
    Prepare raster values and their optional mask for in-memory, Dask, or multiprocessing pair reads.

    Unloaded rasters use _reader_from_source() so workers can read bounded windows. Other inputs keep their NumPy or
    Dask arrays, and every path returns a _RasterPairSource with the chunk boundaries used for local pair generation.
    """

    # Keep an unloaded raster band on disk when multiprocessing can read its windows
    reader = _reader_from_source(raster, band, raster, mp_config)
    if reader is not None:
        prepared_mask: Any | None = None
        if mask is not None:
            # Reuse aligned raster files and evaluate vector geometries within each source tile
            prepared_mask = _reader_from_source(mask, None, raster, mp_config)
            normalized_mask = _normalize_sampling_input(mask)
            if prepared_mask is None and not _is_raster(normalized_mask) and not _is_pointcloud(normalized_mask):
                if _is_vector(normalized_mask):
                    dataframe = _as_geodataframe(normalized_mask)
                    prepared_mask = _reader_from_vector(
                        dataframe,
                        np.ones(len(dataframe)),
                        raster,
                        mp_config,
                        mask_mode="inside",
                    )

            # Materialize only masks that cannot be represented by a worker reader
            if prepared_mask is None:
                prepared_mask = _mask_on_raster(mask, raster, "inside", "raise")
            prepared_mask = _normalize_reader_mask(prepared_mask, reader.shape)
        return _RasterPairSource(replace(reader, mask=prepared_mask), mp_config)

    # Loaded and accessor rasters use their native NumPy or Dask arrays
    array = _selected_raster_data(raster, band)
    if mask is not None:
        mask_array = _mask_on_raster(mask, raster, "inside", "raise")
        if is_dask_array(array):
            import dask.array as da

            array = da.where(mask_array, array, np.nan)
        else:
            array = np.where(mask_array, array, np.nan)
    return _RasterPairSource(array, mp_config)


############################
# 3/ POINT INPUT PREPARATION
############################


@dataclass(frozen=True)
class _PointPairData:
    """Store finite point rows used by pair sampling and their original table positions."""

    coordinates: NDArrayNum
    values: NDArrayNum
    original_indexes: NDArrayNum
    source_size: int


@dataclass(frozen=True)
class _PointPairMaskData:
    """Store a Dask point mask in row order for checks on each source partition."""

    coordinates: NDArrayNum
    values: NDArrayNum
    crs: Any
    source_size: int


def _close_memory_map(array: np.memmap[Any, Any]) -> None:
    """Close a temporary NumPy memory map before its containing directory is removed."""

    memory_map = getattr(array, "_mmap", None)
    if memory_map is not None:
        memory_map.close()


def _write_point_reader_partition(reader: _ValueReader, rows: slice, filename: Path) -> Path:
    """Read and save one point row partition in a worker for bounded processing in the parent."""

    return _stage_pointcloud_partition(reader.read_points(rows), filename)


def _point_pair_mask(
    mask: RasterLike | PointCloudLike | VectorLike | ArrayLike,
    pointcloud: PointCloudBase,
    dataframe: gpd.GeoDataFrame,
    mp_config: MultiprocConfig | None,
) -> Any:
    """Place a pair mask on prepared point rows while keeping a file-backed point mask unloaded."""

    # Represent the source with its collected rows so spatial placement never reads a lazy table again
    prepared_pointcloud = _get_pointcloud_interface(dataframe)
    assert prepared_pointcloud is not None

    # Read an ordered point mask in worker partitions, then compare its coordinates with the prepared source rows
    normalized_mask = _normalize_sampling_input(mask)
    mask_pointcloud = _get_pointcloud_interface(normalized_mask)
    if mask_pointcloud is not None and mp_config is not None and not mask_pointcloud._is_xr:
        if mask_pointcloud is pointcloud:
            mask_dataframe = dataframe
        else:
            mask_reader = _reader_from_source(mask_pointcloud, None, mask_pointcloud, mp_config)
            mask_dataframe = (
                mask_reader.read_points(slice(0, mask_reader.shape[0]))
                if mask_reader is not None
                else mask_pointcloud._dataset
            )

        # Require the same coordinate system and ordered X/Y locations as ordinary point masks
        if mask_pointcloud.crs != pointcloud.crs:
            raise ValueError("Point value 'mask' does not share the support CRS.")
        from geoutils.pointcloud.testing import _point_coords_equal

        if not _point_coords_equal(dataframe, mask_dataframe):
            raise ValueError("Point value 'mask' does not share the ordered support coordinates.")
        column = mask_pointcloud.data_name
        values = np.asarray(mask_dataframe.geometry.z if column is None else mask_dataframe[column])
        if not np.issubdtype(values.dtype, np.bool_):
            raise ValueError("A point support mask must contain boolean values.")
        return _normalize_mask_array(values, (len(dataframe),))

    # Reuse the shared spatial placement for arrays, rasters, vectors, and in-memory point masks
    return _mask_at_support(
        normalized_mask,
        prepared_pointcloud,
        support_dataframe=dataframe,
        mp_config=mp_config,
    )


def _point_partition_mask(
    mask: Any,
    pointcloud: PointCloudBase,
    dataframe: gpd.GeoDataFrame | xr.DataArray,
    rows: slice,
    mp_config: MultiprocConfig | None,
) -> NDArrayNum:
    """Evaluate a pair mask on one point partition."""

    # Read coordinates from the bounded partition once for either point representation
    if isinstance(dataframe, xr.DataArray):
        coordinates = np.column_stack(dataframe.pc.to_xyz()[:2])
    else:
        coordinates = np.column_stack((dataframe.geometry.x.to_numpy(), dataframe.geometry.y.to_numpy()))

    # Compare the matching rows of a Dask point mask with the source coordinates
    if isinstance(mask, _PointPairMaskData):
        if mask.crs != pointcloud.crs:
            raise ValueError("Point value 'mask' does not share the support CRS.")
        mask_coordinates = mask.coordinates[rows]
        if len(mask_coordinates) != len(dataframe) or not np.array_equal(coordinates, mask_coordinates):
            raise ValueError("Point value 'mask' does not share the ordered support coordinates.")
        return _normalize_mask_array(mask.values[rows], (len(dataframe),))

    # Read ordered point masks over the same row range and check their coordinates locally
    normalized_mask = _normalize_sampling_input(mask)
    mask_pointcloud = _get_pointcloud_interface(normalized_mask)
    if mask_pointcloud is not None:
        if mask_pointcloud.crs != pointcloud.crs:
            raise ValueError("Point value 'mask' does not share the support CRS.")

        if mask_pointcloud._is_xr:
            selected = mask_pointcloud._dataset.isel({mask_pointcloud._dataset.dims[0]: rows}).compute().pc
            if not np.array_equal(np.column_stack(selected.to_xyz()[:2]), coordinates):
                raise ValueError("Point value 'mask' does not share the ordered support coordinates.")
            return _normalize_mask_array(selected.data, (len(dataframe),))

        mask_reader = _reader_from_source(mask_pointcloud, None, pointcloud, mp_config)
        if mask_reader is None and not mask_pointcloud.is_loaded and not mask_pointcloud._is_dask:
            mask_reader = _ValueReader(mask_pointcloud)
        if mask_reader is not None:
            mask_dataframe = mask_reader.read_points(rows)
        else:
            mask_dataframe = mask_pointcloud._dataset.iloc[rows]

        mask_coordinates = np.column_stack((mask_dataframe.geometry.x, mask_dataframe.geometry.y))
        if not np.array_equal(coordinates, mask_coordinates):
            raise ValueError("Point value 'mask' does not share the ordered support coordinates.")
        column = mask_pointcloud.data_name
        values = np.asarray(mask_dataframe.geometry.z if column is None else mask_dataframe[column])
        if not np.issubdtype(values.dtype, np.bool_):
            raise ValueError("A point support mask must contain boolean values.")
        return _normalize_mask_array(values, (len(dataframe),))

    # Slice plain masks by original row, and place raster or vector masks only on this point partition
    if not _is_raster(normalized_mask) and not _is_vector(normalized_mask):
        if is_dask_array(normalized_mask):
            selected_mask = normalized_mask.reshape(-1)[rows].compute()
        else:
            selected_mask = np.asanyarray(normalized_mask).reshape(-1)[rows]
        return _normalize_mask_array(selected_mask, (len(dataframe),))
    partition = _get_pointcloud_interface(dataframe)
    assert partition is not None
    spatial_mask = _mask_at_support(normalized_mask, partition, support_dataframe=dataframe)
    return np.asarray(spatial_mask)


def _prepare_dask_point_mask(mask: PointCloudBase) -> _PointPairMaskData:
    """Read Dask point mask partitions into ordered coordinate and boolean arrays."""

    coordinates, values = [], []
    column = mask.data_name
    for partition in mask._dataset.to_delayed():
        dataframe = partition.compute()
        part_values = np.asarray(dataframe.geometry.z if column is None else dataframe[column])
        if not np.issubdtype(part_values.dtype, np.bool_):
            raise ValueError("A point support mask must contain boolean values.")
        coordinates.append(np.column_stack((dataframe.geometry.x, dataframe.geometry.y)))
        values.append(part_values)

    # Preserve all mask rows so each source partition can check its matching coordinates
    if not values:
        return _PointPairMaskData(np.empty((0, 2)), np.empty(0, dtype=bool), mask.crs, 0)
    mask_values = np.concatenate(values)
    return _PointPairMaskData(np.concatenate(coordinates), mask_values, mask.crs, len(mask_values))


def _write_finite_point_partition(
    directory: Path,
    part_index: int,
    dataframe: gpd.GeoDataFrame,
    rows: slice,
    pointcloud: PointCloudBase,
    mask: Any | None,
    mp_config: MultiprocConfig | None,
) -> tuple[Path, int]:
    """Filter one point partition and save its pair inputs without retaining the dataframe in memory."""

    # Extract the selected values and planar coordinates from this bounded partition
    column = pointcloud.data_name
    values = np.asarray(dataframe.geometry.z if column is None else dataframe[column])
    coordinates = np.column_stack((dataframe.geometry.x.to_numpy(), dataframe.geometry.y.to_numpy()))
    valid = np.isfinite(values) & np.all(np.isfinite(coordinates), axis=1)

    # Apply the matching part of a user mask before writing only eligible rows to temporary storage
    if mask is not None:
        valid &= _point_partition_mask(mask, pointcloud, dataframe, rows, mp_config)
    local_indexes = np.flatnonzero(valid)
    original_indexes = local_indexes.astype(np.int64, copy=False) + rows.start
    filename = directory / f"part-{part_index}.npz"
    np.savez(
        filename,
        coordinates=coordinates[local_indexes].astype(np.float64, copy=False),
        values=values[local_indexes],
        original_indexes=original_indexes,
    )
    return filename, len(local_indexes)


def _point_pair_partitions(
    reader: _ValueReader,
    mp_config: MultiprocConfig,
    directory: Path,
) -> Iterator[tuple[int, slice, Path]]:
    """Read and save multiprocessing point partitions in source row order."""

    from geoutils.pointcloud.las import _point_partition_size

    # Read unloaded files concurrently in bounded worker batches
    chunks = _point_partition_size(mp_config)
    slices = [slice(start, min(start + chunks, reader.shape[0])) for start in range(0, reader.shape[0], chunks)]
    arguments = ((reader, rows, directory / f"source-{part_index}.pkl") for part_index, rows in enumerate(slices))
    for part_index, filename in _map_bounded(mp_config.cluster, _write_point_reader_partition, arguments):
        yield part_index, slices[part_index], filename


def _prepare_dask_point_pair_data(pointcloud: PointCloudBase, mask: Any | None) -> _PointPairData:
    """
    Read finite Dask point rows into arrays for the spatial searches used by pair sampling.

    Source partitions are computed once, without changing the source object's loading state. Coordinates, values
    and original row indexes remain in memory for the existing spatial searches.
    """

    # Read a Dask point mask in its own partition order before matching it to source rows
    prepared_mask = _normalize_sampling_input(mask) if mask is not None else None
    mask_pointcloud = _get_pointcloud_interface(prepared_mask)
    if mask_pointcloud is not None and mask_pointcloud._is_dask and not mask_pointcloud._is_xr:
        prepared_mask = _prepare_dask_point_mask(mask_pointcloud)

    # Compute one source partition at a time, preserving its original row positions
    if pointcloud._is_xr:
        partitions = _array_partitions(pointcloud._dataset, None)
    else:
        partitions = (partition.compute() for partition in pointcloud._dataset.to_delayed())
    coordinates_parts, value_parts, index_parts = [], [], []
    source_size = 0
    for partition in partitions:
        if pointcloud._is_xr:
            x, y, values = partition.pc.to_xyz()
        else:
            x, y = partition.geometry.x.to_numpy(), partition.geometry.y.to_numpy()
            column = pointcloud.data_name
            values = np.asarray(partition.geometry.z if column is None else partition[column])
        coordinates = np.column_stack((x, y))
        rows = slice(source_size, source_size + len(values))
        valid = np.isfinite(values) & np.isfinite(coordinates).all(axis=1)

        # Apply the matching mask rows before collecting only finite sampling inputs
        if prepared_mask is not None:
            valid &= _point_partition_mask(prepared_mask, pointcloud, partition, rows, None)
        local_indexes = np.flatnonzero(valid)
        coordinates_parts.append(coordinates[local_indexes])
        value_parts.append(values[local_indexes])
        index_parts.append(source_size + local_indexes)
        source_size = rows.stop

    # Check mask lengths after observing every source partition, including unknown dataframe lengths
    if isinstance(prepared_mask, _PointPairMaskData):
        mask_size = prepared_mask.source_size
    elif mask_pointcloud is not None:
        mask_size = mask_pointcloud.point_count
    elif prepared_mask is not None and not _is_raster(prepared_mask) and not _is_vector(prepared_mask):
        mask_size = int(np.prod(np.shape(prepared_mask)))
    else:
        mask_size = source_size
    if mask_size != source_size:
        if mask_pointcloud is not None:
            raise ValueError("Point value 'mask' does not share the ordered support coordinates.")
        raise ValueError("Argument ``mask`` must be boolean and contain one value per input location.")

    # Join finite rows for the existing global point searches
    if not value_parts or sum(len(values) for values in value_parts) < 2:
        raise ValueError("At least two finite points are required to sample pairs.")
    return _PointPairData(
        np.concatenate(coordinates_parts), np.concatenate(value_parts), np.concatenate(index_parts), source_size
    )


@contextmanager
def _prepare_point_pair_data(
    pointcloud: PointCloudBase,
    mask: Any | None,
    mp_config: MultiprocConfig | None,
) -> Iterator[_PointPairData]:
    """
    Collect finite point inputs in memory, or read multiprocessing inputs through temporary arrays.

    Dask inputs use _prepare_dask_point_pair_data() without writing files. For multiprocessing file reads,
    _point_pair_partitions() and _write_finite_point_partition() save eligible rows in temporary arrays. The maps
    remain available until pair sampling and output construction finish without changing the source loading state.
    """

    if pointcloud._is_dask:
        yield _prepare_dask_point_pair_data(pointcloud, mask)
        return

    if pointcloud._is_xr:
        x, y, values = pointcloud.to_xyz()
        coordinates = np.column_stack((x, y))
        values = np.asarray(values)
        valid = np.isfinite(values) & np.isfinite(coordinates).all(axis=1)
        if mask is not None:
            eligible = _mask_at_support(mask, pointcloud)
            assert eligible is not None
            valid &= eligible.compute() if is_dask_array(eligible) else eligible
        yield _PointPairData(coordinates[valid], values[valid], np.flatnonzero(valid), len(values))
        return

    # Keep the direct in-memory path for ordinary point tables
    reader = _reader_from_source(pointcloud, None, pointcloud, mp_config)
    if reader is None:
        dataframe = pointcloud._dataset
        values = np.asarray(
            dataframe[pointcloud.data_name] if pointcloud.data_name is not None else dataframe.geometry.z
        )
        coordinates = np.column_stack((dataframe.geometry.x.to_numpy(), dataframe.geometry.y.to_numpy()))
        valid = np.isfinite(values) & np.all(np.isfinite(coordinates), axis=1)
        if mask is not None:
            mask_array = _point_pair_mask(mask, pointcloud, dataframe, mp_config)
            if is_dask_array(mask_array):
                mask_array = mask_array.compute()
            valid &= mask_array
        yield _PointPairData(coordinates[valid], values[valid], np.flatnonzero(valid), len(values))
        return

    # Filter each chunked partition and save it immediately so only bounded point rows reside in memory
    with ExitStack() as storage:
        directory = Path(storage.enter_context(TemporaryDirectory(prefix="geoutils-pairsample-")))
        prepared_mask = mask

        # We finish the writing before mask validation below can remove their temporary directory
        assert reader is not None and mp_config is not None
        source_parts = list(_point_pair_partitions(reader, mp_config, directory))
        parts: list[tuple[Path, int]] = []
        source_size = 0
        for part_index, rows, filename in source_parts:
            if rows.start != source_size:
                raise RuntimeError("Point partitions must cover consecutive source rows.")
            dataframe = pd.read_pickle(filename)
            parts.append(
                _write_finite_point_partition(
                    directory, part_index, dataframe, rows, pointcloud, prepared_mask, mp_config
                )
            )
            source_size = rows.stop

        # Check array mask length after reading every source row partition
        if prepared_mask is not None:
            normalized_mask = _normalize_sampling_input(prepared_mask)
            if (
                _get_pointcloud_interface(normalized_mask) is None
                and not _is_raster(normalized_mask)
                and not _is_vector(normalized_mask)
                and int(np.prod(np.shape(normalized_mask))) != source_size
            ):
                raise ValueError("Argument ``mask`` must be boolean and contain one value per input location.")

        # Allocate compact disk-backed arrays after partition counts reveal the finite point total
        valid_count = sum(count for _, count in parts)
        if valid_count < 2:
            raise ValueError("At least two finite points are required to sample pairs.")
        coordinates = np.memmap(directory / "coordinates.dat", mode="w+", dtype=np.float64, shape=(valid_count, 2))
        with np.load(parts[0][0]) as first_part:
            value_dtype = first_part["values"].dtype
        values = np.memmap(directory / "values.dat", mode="w+", dtype=value_dtype, shape=valid_count)
        original_indexes = np.memmap(directory / "indexes.dat", mode="w+", dtype=np.int64, shape=valid_count)
        for array in (coordinates, values, original_indexes):
            storage.callback(_close_memory_map, array)

        # Join the small temporary partitions in original row order without creating an in-memory concatenation
        start = 0
        for filename, count in parts:
            with np.load(filename) as part:
                stop = start + count
                coordinates[start:stop] = part["coordinates"]
                values[start:stop] = part["values"]
                original_indexes[start:stop] = part["original_indexes"]
                start = stop
        coordinates.flush()
        values.flush()
        original_indexes.flush()
        yield _PointPairData(coordinates, values, original_indexes, source_size)


#########################
# 4/ RASTER PAIR SAMPLING
#########################


class _RasterPairSampler:
    """
    Pair sampler for isotropic log-lag Monte Carlo sampling of a regular 2D grid.

    This method deals efficiently with large datasets by reading Dask arrays or file-backed rasters in chunks, and by
    anticipating the probability of nodata occurring in pairs (with iterative top-up). To sample short to long lags
    efficiently for variography, pairs are sampled by drawing separation vectors with log-uniform magnitude and
    uniformly distributed orientation, corresponding to an isotropic Monte Carlo sampling of logarithmic spatial lags.

    References
    ----------
    Sampling method is inspired from the log-lag sampling developed in Hugonnet et al. (2022), Section V-C and
    Supplementary Section II-C.

    Few literature references exist that describe this algorithm specifically. However, it was
    conceptualized fairly early, such as in Cressie (1993):
    "In isotropic settings, all directions are equivalent, and lag distances may be grouped on a
    logarithmic scale to ensure adequate sampling across short and long ranges."

    Or in fluid mechanics and turbulence, following Monin and Yaglom (1971):
    "For isotropic turbulence, ensemble averages over separation vectors are replaced by averages over
    uniformly distributed directions and logarithmically spaced magnitudes."

    Hugonnet et al. (2022): http://dx.doi.org/10.1109/JSTARS.2022.3188922
    Cressie (1993): http://dx.doi.org/10.1002/9781119115151
    Monin and Yaglom (1971). Statistical Fluid Mechanics: Mechanics of Turbulence (Vol. I). The MIT Press.

    The code expands an early version written in SciKit-GStat (as "RasterEquidistantMetricSpace").

    Use in GeoUtils
    ---------------

    sample() returns the first and second endpoint indices and their distances. _sample_raster_pairs() reads the
    selected raster values and builds the labelled Xarray result; all other methods are internal.

    Summary of algo
    ---------------
    We want to sample a large number of point pairs from a regular 2D grid such that separation distances cover
    short and long lags efficiently (approximately log-uniform in distance). We cannot enumerate all pairs due to
    the size of the grid, so we also subsample.

    The core log-lag sampling is to:
      1) Subsample a distance r ~ Uniform(log(min_distance), log(max_distance))  (log-uniform in r)
      2) Subsample an angle  θ ~ Uniform(0, 2π)                                  (isotropic)
      3) Convert (r, θ) to integer pixel offsets (ix, iy) by rounding
      4) Choose first endpoints and compute second endpoints using the offsets
      5) Reject out-of-bounds pairs and (optionally) exclude nonfinite endpoints
      6) Avoid pair duplication during sampling to circumvent costly duplicate removal

    Chunked value access
    --------------------
    - We never load the full array in memory.
    - We count finite cells once at the start to cap the request at the number of distinct pairs that can exist.
    - For each candidate batch, read only selected endpoint values and sample more pairs if needed.

    Strategies (strategy)
    ---------------------
    - "independent":
        Each pair is generated independently (first endpoint + offset).
        For 1M pairs, we have to sample 1M + 1M points.

    - "anchors":
        We reuse a set of random anchor points for one endpoint of each pair. Targets are generated relative to these
        anchors, so that anchor values (and their chunks) are reused across many pairs. For instance, for 1M pairs,
        we index 1000 anchors points that each match 1000 random points, so in the end we index 1k + 1M points
        (we thus use half the sample size of "independent").

    - "chunk_anchors":
        Like "anchors" but anchors are sampled from a few chunks per round to reduce source reads and scheduling.
        This method seems to perform the best overall in both speed and memory (default).

    - "anchor_batched":
        Structured generation: for each anchor sample multiple distances (log-uniform) and for each distance
        sample multiple angles. This produces blocks of pairs with shared anchors and controlled lag coverage.
        There might be some room for improvement in this method... which could make it more efficient to sample less
        chunks for one vindex (mostly affects speed, while batch size limits temporary memory).

    Hybrid local/global (hybrid_local_fraction)
    -------------------------------------------
    Optionally, one can require that a fraction (or all) of pairs remains within the first endpoint's chunk.
    This is situational (mostly for short-range variograms), but can massively reduce I/O overhead (both
    endpoints are always in the same chunk). The other pairs are sampled "globally" to preserve long-range lag coverage.

    NaN handling
    ------------
    To deal with NaNs without knowing their distribution ahead in a chunked source, the following steps are applied:
    1. We estimate the global finite fraction f_valid by a single reduction (counting chunk per chunk), and deduce the
       probability of a random pair containing at least 1 NaN: p_pair_valid ≈ f_valid^2. For instance, 10% of NaNs
       in the array gives us an 81% chance of selecting a valid pair at random.
    2. We oversample so that random pairs will roughly match requested samples, then filter out pairs where
       either endpoint is not finite (NaN/inf).
    3. We sample more pairs until the requested count of finite pairs is reached. This is typically not
       critical for variography (it rarely matters if sample count is slightly larger or smaller).

    Notes on scalability
    --------------------
    Storing more than 1e8 endpoint pairs and distances is RAM-heavy regardless of strategy.
    Pair output uses the requested index and distance dtypes to control memory use.
    """

    #################
    # CONFIGURATION
    #################

    def __init__(
        self,
        source: _RasterPairSource,
        *,
        # Raster geometry and target sample size
        dx: float,
        dy: float,
        n_pairs: int,
        # Distance range
        min_distance: float,
        max_distance: float,
        # Log-lag sampling strategies with various chunk-compatibility
        strategy: Literal["independent", "anchors", "chunk_anchors", "anchor_batched"],
        # Deduplication
        deduplicate: Literal["none", "per_anchor", "global"],
        # Random seed
        random_state: int | np.random.Generator | None,
        # Batching / Termination
        batch_pairs: int,
        max_rounds: int,
        max_oversample: float,
        # Chunk limits and nearby pairs
        chunks_per_round: int,
        anchors_per_round: int,
        # Parameters for anchor_batched
        distances_per_anchor: int,
        angles_per_distance: int,
        # Hybrid local/global
        hybrid_local_fraction: float,
        max_local_distance: float | None,
    ) -> None:
        """
        Pair sampling on a regular raster grid.

        :param source: Two-dimensional raster values and the backend used to read finite endpoint values.
        :param dx: Horizontal pixel spacing in coordinate units, such as meters.
        :param dy: Vertical pixel spacing in coordinate units, such as meters.
        :param n_pairs: Target number of valid pairs with two finite endpoints.
        :param min_distance: Smallest distance included in log-distance sampling.
        :param max_distance: Largest distance included in log-distance sampling.
        :param strategy: Pair generation strategy: "independent", "anchors", "chunk_anchors", or "anchor_batched".
            See the class docstring for details and performance trade-offs.
        :param deduplicate: Duplicate handling: "global" sorts pairs at the end, "per_anchor" removes repeated second
            endpoints for each anchor, and "none" skips removal. Duplicate pairs should be avoided for variography
            because they bias the distance distribution.
        :param random_state: Seed or NumPy Generator used for reproducible random sampling.
        :param batch_pairs: Maximum candidate pairs generated per round before checking finite values. Larger batches
            reduce Python and backend scheduling overhead but require more memory for temporary arrays.
        :param max_rounds: Maximum number of sampling rounds used to reach ``n_pairs`` valid pairs. Extra rounds help
            when NaNs are clustered or local constraints lower the acceptance rate.
        :param max_oversample: Maximum candidate multiplier relative to ``n_pairs``. This prevents very large temporary
            arrays when the finite fraction is small.
        :param chunks_per_round: Number of chunks selected when drawing anchors from a few chunks. Smaller values
            reduce source reads but cover less of the raster per round.
        :param anchors_per_round: Number of first endpoints drawn per round by anchor-based strategies.
        :param distances_per_anchor: Number of log-uniform radii drawn per anchor by "anchor_batched".
        :param angles_per_distance: Number of directions drawn for each radius by "anchor_batched".
        :param hybrid_local_fraction: Fraction of candidate pairs forced to remain in the first endpoint's chunk. Zero
            gives a fully global sample; one keeps every pair local.
        :param max_local_distance: Largest distance used for local pairs. Defaults to the chunk diagonal when omitted.
            Larger values allow longer local lags but increase rejection at chunk boundaries.
        """

        # Store the grid and sampling options with consistent numeric types
        self.source = source
        self.shape = source.shape
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

        # Use source chunk boundaries when drawing nearby pairs
        self.chunk_edges = source.chunk_edges
        chunk_rows, chunk_columns = (int(np.max(np.diff(edges))) for edges in self.chunk_edges)
        self.max_local_distance = (
            float(np.hypot((chunk_columns - 1) * self.dx, (chunk_rows - 1) * self.dy))
            if max_local_distance is None
            else float(max_local_distance)
        )

    ###################
    # CANDIDATE PAIRS
    ###################

    def _offsets(self, count: int, maximum: float) -> tuple[NDArrayNum, NDArrayNum]:
        """
        Draw random directions with distance ranges represented evenly on a log scale.

        :param count: Number of proposed offsets before rounding and distance checks.
        :param maximum: Requested upper distance limit in coordinate units, capped by max_distance. When it does not
            exceed min_distance, use the sampler's full distance interval instead.
        :returns: Matching row and column offsets, excluding zero offsets and distances outside the limits.
        """

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

    def _sample_anchors(self, count: int, *, limit_chunks: bool) -> NDArrayNum:
        """
        Draw first endpoints across the grid or from a small set of chunks.

        :param count: Number of first endpoints to draw, allowing repeated cells.
        :param limit_chunks: Whether to draw only from chunks_per_round randomly selected chunks.
        :returns: Flat raster cell indexes that can be reused as first endpoints.
        """

        # Draw directly from the full grid when pairs do not need to stay near selected chunks
        if not limit_chunks:
            return self.rng.integers(0, self.size, count, dtype=np.int64)

        # Select only a few source chunks before drawing cell numbers
        n_chunk_rows, n_chunk_columns = (len(edges) - 1 for edges in self.chunk_edges)
        chunk_count = min(self.chunks_per_round, n_chunk_rows * n_chunk_columns)
        chosen = self.rng.choice(n_chunk_rows * n_chunk_columns, chunk_count, replace=False)

        # Split first endpoints between those chunks to limit Dask reads
        anchors: list[NDArrayNum] = []
        per_chunk = int(np.ceil(count / chunk_count))
        remaining = count
        for flat_chunk in chosen:
            # Use each edge chunk's true size so every drawn cell exists
            chunk_row, chunk_column = divmod(int(flat_chunk), n_chunk_columns)
            row_start, row_stop = self.chunk_edges[0][chunk_row : chunk_row + 2]
            column_start, column_stop = self.chunk_edges[1][chunk_column : chunk_column + 2]
            take = min(per_chunk, remaining)
            rows = self.rng.integers(row_start, row_stop, take, dtype=np.int64)
            columns = self.rng.integers(column_start, column_stop, take, dtype=np.int64)
            anchors.append(rows * self.shape[1] + columns)
            remaining -= take
            if remaining == 0:
                break

        # Join the first endpoints after all selected chunks have contributed
        return np.concatenate(anchors) if anchors else np.empty(0, dtype=np.int64)

    def _same_chunk(
        self, rows: NDArrayNum, columns: NDArrayNum, second_rows: NDArrayNum, second_columns: NDArrayNum
    ) -> NDArrayBool:
        """
        Check whether each pair stays within its actual Dask chunk or in-memory sampling area.

        :param rows: Raster row indexes of the first endpoints.
        :param columns: Raster column indexes of the first endpoints.
        :param second_rows: Matching row indexes of the second endpoints.
        :param second_columns: Matching column indexes of the second endpoints.
        :returns: One boolean per pair, true when both endpoints belong to the same chunk.
        """

        # Locate both endpoints against the real boundaries, including uneven interior chunks
        row_edges, column_edges = self.chunk_edges
        same_row = np.searchsorted(row_edges, rows, side="right") == np.searchsorted(
            row_edges, second_rows, side="right"
        )
        same_column = np.searchsorted(column_edges, columns, side="right") == np.searchsorted(
            column_edges, second_columns, side="right"
        )
        return same_row & same_column

    def _pairs_from_offsets(
        self, first: NDArrayNum, row_offset: NDArrayNum, column_offset: NDArrayNum, *, local: bool
    ) -> tuple[NDArrayNum, NDArrayNum]:
        """Apply offsets to first endpoints and check boundaries and per-anchor duplicates."""

        # Locate second endpoints and exclude cells outside the raster
        rows, columns = np.divmod(first, self.shape[1])
        second_rows, second_columns = rows + row_offset, columns + column_offset
        inside = (
            (second_rows >= 0)
            & (second_rows < self.shape[0])
            & (second_columns >= 0)
            & (second_columns < self.shape[1])
        )
        if local:
            inside &= self._same_chunk(rows, columns, second_rows, second_columns)
        first = first[inside]
        second = second_rows[inside] * self.shape[1] + second_columns[inside]

        # Select each second endpoint once per first endpoint when requested
        if self.deduplicate == "per_anchor" and first.size:
            order = np.argsort(first, kind="stable")
            keys = first[order].astype(np.int64) * np.int64(self.size) + second[order]
            _, keep = np.unique(keys, return_index=True)
            positions = order[np.sort(keep)]
            first, second = first[positions], second[positions]
        return first, second

    def _from_anchors(self, anchors: NDArrayNum, count: int, *, local: bool) -> tuple[NDArrayNum, NDArrayNum]:
        """
        Reuse first endpoints and draw a separate offset for each pair.

        The candidate count is described by _candidates().

        :param anchors: Flat raster cell indexes to reuse as first endpoints.
        :param local: Whether to keep both endpoints in the same chunk and apply max_local_distance.
        :returns: First and second endpoint indexes after checking grid boundaries and configured duplicate handling.
        """

        # Return empty integer arrays before NumPy can try to repeat an empty input
        if anchors.size == 0 or count == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        # Repeat first endpoints and draw offsets before checking their grid positions
        repeated = np.resize(anchors, count)
        row_offset, column_offset = self._offsets(count, self.max_local_distance if local else self.max_distance)
        length = min(count, row_offset.size)
        return self._pairs_from_offsets(repeated[:length], row_offset[:length], column_offset[:length], local=local)

    def _anchor_batched(self, anchors: NDArrayNum, *, local: bool) -> tuple[NDArrayNum, NDArrayNum]:
        """
        Draw several distances and directions from every first endpoint.

        Anchors and local distance constraints follow _from_anchors(); the numbers of distances and directions
        are configured by _RasterPairSampler.__init__().
        """

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
        return self._pairs_from_offsets(repeated, row_offset, column_offset, local=local)

    def _independent(self, count: int) -> tuple[NDArrayNum, NDArrayNum]:
        """
        Draw each first endpoint and offset independently across the full grid.

        The candidate count and returned endpoint arrays are described by _candidates().
        """

        # Draw every first endpoint independently
        row_offset, column_offset = self._offsets(count, self.max_distance)
        rows = self.rng.integers(0, self.shape[0], row_offset.size, dtype=np.int64)
        columns = self.rng.integers(0, self.shape[1], column_offset.size, dtype=np.int64)
        second_rows, second_columns = rows + row_offset, columns + column_offset

        # Keep only second endpoints that remain inside the raster after applying offsets
        inside = (
            (second_rows >= 0)
            & (second_rows < self.shape[0])
            & (second_columns >= 0)
            & (second_columns < self.shape[1])
        )
        return rows[inside] * self.shape[1] + columns[inside], (
            second_rows[inside] * self.shape[1] + second_columns[inside]
        )

    ####################
    # STRATEGY CHOICE
    ####################

    def _candidates(self, count: int) -> tuple[NDArrayNum, NDArrayNum]:
        """
        Draw one limited batch of candidate pairs with the selected strategy.

        Sampling options are configured by _RasterPairSampler.__init__().

        :param count: Maximum number of candidate pairs to return before checking endpoint values.
        :returns: Matching flat cell indexes for the first and second endpoints, possibly fewer than count.
        """

        # Split the batch between nearby pairs (with local fraction)
        local_count = int(round(count * self.hybrid_local_fraction))
        global_count = count - local_count
        first_parts: list[NDArrayNum] = []
        second_parts: list[NDArrayNum] = []

        # Draw pairs independently (reuse first endpoints only for nearby pairs)
        if self.strategy == "independent":
            if local_count:
                anchors = self._sample_anchors(min(self.anchors_per_round, local_count), limit_chunks=True)
                first, second = self._from_anchors(anchors, local_count, local=True)
                first_parts.append(first)
                second_parts.append(second)
            if global_count:
                first, second = self._independent(global_count)
                first_parts.append(first)
                second_parts.append(second)

        # Reuse one set of first endpoints
        elif self.strategy in {"anchors", "chunk_anchors"}:
            anchors = self._sample_anchors(
                min(self.anchors_per_round, count), limit_chunks=self.strategy == "chunk_anchors"
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
                    anchors = self._sample_anchors(int(np.ceil(part_count / pairs_per_anchor)), limit_chunks=local)
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

    def sample(
        self, pair_values: list[NDArrayNum] | None = None, *, output: _DiskPairOutput | None = None
    ) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum] | None:
        """
        Collect the requested number of pairs whose two raster values are finite.

        Sampling options are configured by _RasterPairSampler.__init__().

        If pair_values is supplied, its contents are replaced by one (n_pairs, 2) array of accepted endpoint
        values. Only selected values are cached, and their order follows the returned indexes after deduplication.
        With output, accepted batches are written immediately and no complete pair arrays are collected.

        :returns: Three one-dimensional arrays containing first endpoint indexes, second endpoint indexes,
            and their distances in raster coordinate units, or None when batches are written to output.
            Indexes refer to flat raster cells in row order.
        """

        if pair_values is not None:
            pair_values.clear()

        # Count finite cells with Dask/MP
        # N cells can supply N * (N - 1) / 2 different pairs, so larger counts cannot change the sample target
        min_count = (1 + isqrt(1 + 8 * self.n_pairs)) // 2
        if min_count * (min_count - 1) // 2 < self.n_pairs:
            min_count += 1
        n_valid = self.source.count_finite(min_count=min_count)

        # Limit a unique sample to the number of different pairs that can exist
        maximum_unique = n_valid * (n_valid - 1) // 2
        target = min(self.n_pairs, maximum_unique)
        if target < self.n_pairs:
            warnings.warn(
                f"Argument ``n_pairs`` exceeds the {maximum_unique} possible finite pairs; using that maximum.",
                UserWarning,
            )
        if target == 0:
            raise ValueError("At least two finite raster cells are required to sample pairs.")

        # We loop on temporary arrays by batch limit/oversampling
        count = min(self.batch_pairs, max(1, int(np.ceil(target * self.max_oversample))))
        first_parts: list[NDArrayNum] = []
        second_parts: list[NDArrayNum] = []
        remaining, stalled = target, 0
        for _ in range(self.max_rounds):
            if remaining == 0:
                break

            # Draw another bounded batch when missing values or edge crossings leave too few pairs
            first, second = self._candidates(count)

            # Read candidate endpoints and select pairs with two finite values
            if first.size:
                read_values = pair_values is not None or (output is not None and self.source.validity is None)
                if not read_values:
                    finite = self.source.finite_pairs(first, second)
                else:
                    first_values, second_values = self.source.read_pairs(first, second, preserve_mask=True)
                    finite = np.atleast_1d(~get_mask_from_array(first_values) & ~get_mask_from_array(second_values))
                    first_values = np.ma.getdata(first_values)
                    second_values = np.ma.getdata(second_values)
                first, second = first[finite], second[finite]

            # Keep only the remaining number of pairs and detect rounds that find nothing
            if first.size:
                take = min(remaining, first.size)
                if output is not None:
                    values = None
                    if read_values:
                        values = np.column_stack((first_values[finite][:take], second_values[finite][:take]))
                    output.append(first[:take], second[:take], values=values)
                else:
                    first_parts.append(first[:take])
                    second_parts.append(second[:take])
                    if pair_values is not None:
                        pair_values.append(np.column_stack((first_values[finite][:take], second_values[finite][:take])))
                remaining -= take
                stalled = 0
            else:
                stalled += 1
            if stalled >= 5:
                break

        # Report when no round found a pair with two finite values
        if remaining == target:
            raise ValueError("No finite raster pairs could be sampled.")

        # Global duplicates are removed by the disk writer (if output is set)
        if output is not None:
            if output.count < target:
                warnings.warn(
                    f"Sampled {output.count} finite pairs out of {target} requested after {self.max_rounds} rounds.",
                    UserWarning,
                )
            return None

        # Remove duplicates
        first, second = np.concatenate(first_parts), np.concatenate(second_parts)
        values = np.concatenate(pair_values) if pair_values is not None else None
        if self.deduplicate == "global":
            if values is not None:
                # Reverse cached values when the smaller endpoint index is placed first
                unique_first, unique_second, positions = _unique_pairs(first, second, n_observations=self.size)
                values = np.where((first > second)[:, None], values[:, ::-1], values)[positions]
                first, second = unique_first, unique_second
            else:
                first, second = _deduplicate_pairs(first, second, n_observations=self.size)
        if pair_values is not None:
            assert values is not None
            pair_values[:] = [values]
        if first.size < target:
            warnings.warn(
                f"Sampled {first.size} finite pairs out of {target} requested after {self.max_rounds} rounds.",
                UserWarning,
            )

        # Calculate exact map distances for the selected pairs
        first_rows, first_columns = np.divmod(first.astype(np.int64), self.shape[1])
        second_rows, second_columns = np.divmod(second.astype(np.int64), self.shape[1])
        distances = np.hypot(
            (second_columns - first_columns) * self.dx,
            (second_rows - first_rows) * self.dy,
        )
        return first, second, distances


def _random_raster_pairs(
    source: _RasterPairSource,
    *,
    dx: float,
    dy: float,
    n_pairs: int,
    min_distance: float,
    max_distance: float,
    random_state: int | np.random.Generator | None,
    max_rounds: int,
    batch_pairs: int,
    pair_values: list[NDArrayNum] | None = None,
    output: _DiskPairOutput | None = None,
) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum] | None:
    """Sample raster pairs through two random endpoints."""

    # Start empty result arrays that each round extends after removing duplicates
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    size, n_columns = int(np.prod(source.shape)), source.shape[1]
    first: NDArrayNum = np.empty(0, dtype=np.int64)
    second: NDArrayNum = np.empty(0, dtype=np.int64)
    values: NDArrayNum | None = None
    if pair_values is not None:
        pair_values.clear()
    for _ in range(max_rounds):
        remaining = n_pairs - (first.size if output is None else output.count)
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
        keep = (first_candidate != second_candidate) & (distances >= min_distance) & (distances <= max_distance)
        # Read only pairs in range, sharing source reads between both endpoints
        first_candidate, second_candidate = first_candidate[keep], second_candidate[keep]
        read_values = pair_values is not None or (output is not None and source.validity is None)
        if not read_values:
            keep = source.finite_pairs(first_candidate, second_candidate)
        else:
            first_values, second_values = source.read_pairs(first_candidate, second_candidate, preserve_mask=True)
            keep = np.atleast_1d(~get_mask_from_array(first_values) & ~get_mask_from_array(second_values))
            first_values = np.ma.getdata(first_values)
            second_values = np.ma.getdata(second_values)
        if np.any(keep):
            if output is not None:
                candidate_values = None
                if read_values:
                    candidate_values = np.column_stack((first_values[keep], second_values[keep]))
                output.append(first_candidate[keep], second_candidate[keep], values=candidate_values, limit=remaining)
                continue
            # Remove duplicates across all rounds before keeping the requested count
            first = np.concatenate((first, first_candidate[keep]))
            second = np.concatenate((second, second_candidate[keep]))
            if pair_values is not None:
                candidates = np.column_stack((first_values[keep], second_values[keep]))
                values = candidates if values is None else np.concatenate((values, candidates))
                unique_first, unique_second, positions = _unique_pairs(first, second, n_observations=size)
                values = np.where((first > second)[:, None], values[:, ::-1], values)[positions[:n_pairs]]
                first, second = unique_first[:n_pairs], unique_second[:n_pairs]
            else:
                first, second = _deduplicate_pairs(first, second, n_observations=size)
            first, second = first[:n_pairs], second[:n_pairs]

    # Distinguish finding no pair from finding fewer unique pairs than requested
    count = first.size if output is None else output.count
    if count == 0:
        raise ValueError("No finite raster pairs could be sampled.")
    if count < n_pairs:
        warnings.warn(f"Sampled {count} unique raster pairs out of {n_pairs} requested.", UserWarning)
    if output is not None:
        return None
    if pair_values is not None:
        assert values is not None
        pair_values[:] = [values]

    # Calculate exact map distances after the final pair order is known
    first_rows, first_columns = np.divmod(first, n_columns)
    second_rows, second_columns = np.divmod(second, n_columns)
    distances = np.hypot((second_columns - first_columns) * dx, (second_rows - first_rows) * dy)
    return first, second, distances


########################
# 5/ POINT PAIR SAMPLING
########################


@dataclass(frozen=True)
class _GridSpec:
    """Grid cells used to find nearby point rows without comparing every point."""

    cell_size: float
    x_min: float
    y_min: float
    n_columns: int
    n_rows: int


class _PointPairSampler:
    """
    Irregular 2D coordinate sampler returning endpoint indices and distances.

    Unlike regular grid sampling, irregular coordinates have no fixed row and column offsets, which makes it less
    computationally efficient yet simplifies the approach a lot. The exact methods below ("kdtree" and "hashgrid") both
    choose logarithmically spaced distance rings and sample observed point pairs within them. The approximate
    "nn_logvector" method instead draws a log-uniform distance and random direction, then uses the observed point
    nearest the proposed endpoint.

    Strategies:
      - "kdtree"      : exact annulus sampling via KDTree query_ball_point(r_out) + annulus filter
      - "hashgrid"    : exact annulus sampling via hash-grid + AABB culling + annulus filter
      - "nn_logvector": approximate log-distance + random angle with vectorized KDTree NN queries
                        (no distance-bias correction; accepts that long distances may appear more often)
    """

    #################
    # CONFIGURATION
    #################

    def __init__(
        self,
        coordinates: NDArrayNum,
        *,
        # Target sample size and distance range
        n_pairs: int,
        min_distance: float,
        max_distance: float,
        n_bins: int,
        # Search strategy
        strategy: Literal["kdtree", "hashgrid", "nn_logvector"],
        # Distance ring sampling
        anchors_per_round: int,
        attempts_per_anchor: int,
        max_rounds: int,
        # Hash-grid tuning
        cell_size: float | None,
        # nn_logvector tuning (vectorized)
        nn_tolerance: float,
        nn_batch_size: int,
        nn_oversample: float,
        nn_max_batches: int,
        # Random seed
        random_state: int | np.random.Generator | None,
    ) -> None:
        """
        Pair sampling on irregular point coordinates.

        :param coordinates: Finite X/Y coordinates with shape (n_points, 2).
        :param n_pairs: Target number of point pairs.
        :param min_distance: Smallest allowed pair distance.
        :param max_distance: Largest allowed pair distance.
        :param n_bins: Number of logarithmically spaced distance rings used by the exact strategies.
        :param strategy: Pair search strategy: "kdtree", "hashgrid", or "nn_logvector". See the class docstring for
            details.
        :param anchors_per_round: Number of first points tested in each round by the exact strategies.
        :param attempts_per_anchor: Number of distance rings tested for each first point by the exact strategies.
        :param max_rounds: Maximum number of sampling rounds used by the exact strategies.
        :param cell_size: Square cell width used by "hashgrid". Defaults to one eighth of ``max_distance``.
        :param nn_tolerance: Largest nearest-point error accepted by "nn_logvector", as a fraction of the proposed
            distance.
        :param nn_batch_size: Maximum number of proposed endpoints checked together by "nn_logvector".
        :param nn_oversample: Number of endpoints proposed by "nn_logvector" relative to the pairs still needed.
        :param nn_max_batches: Maximum number of candidate batches used by "nn_logvector".
        :param random_state: Seed or NumPy Generator used for reproducible random sampling.
        """

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

        # Define distance rings and defer search indexes until needed
        self.edges = np.geomspace(self.min_distance, self.max_distance, self.n_bins + 1)
        self._tree: cKDTree | None = None
        self.grid: dict[tuple[int, int], NDArrayNum] | None = None
        self.grid_spec: _GridSpec | None = None

        # Check coordinates and batch options before starting repeated searches
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2 or self.size < 2:
            raise ValueError("Argument ``coordinates`` must contain at least two X/Y points.")

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

    def _grid_candidates(self, anchor: int, inner: float, outer: float) -> NDArrayNum:
        """
        Collect point rows from grid cells that may cross one distance ring.

        The anchor and distance boundaries are described by _one_in_ring(). This preliminary search can include
        points outside the ring; _one_in_ring() checks their exact distances before selecting an endpoint.
        """

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
                if nearest_x**2 + nearest_y**2 > outer**2 or farthest_x**2 + farthest_y**2 < inner**2:
                    continue
                indexes = self.grid.get((column, row))
                if indexes is not None:
                    parts.append(indexes)

        # Join point rows from matching cells before checking their exact distances
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)

    def _one_in_ring(self, anchor: int, inner: float, outer: float) -> tuple[int, float] | None:
        """
        Select one second point within an exact distance range of the first.

        :param anchor: Row index of the first endpoint in the sampler's finite coordinate array.
        :param inner: Inclusive lower distance boundary in coordinate units.
        :param outer: Exclusive upper distance boundary, except that max_distance is included.
        :returns: The selected second endpoint's row index and exact distance, or None when no point lies in the ring.
        """

        # Ask the selected search helper for points inside the outer distance
        candidates = (
            np.asarray(self.tree.query_ball_point(self.coordinates[anchor], outer), dtype=np.int64)
            if self.strategy == "kdtree"
            else self._grid_candidates(anchor, inner, outer).astype(np.int64, copy=False)
        )
        if candidates.size == 0:
            return None

        # Check exact distances because both helpers may also return points outside the inner distance
        differences = self.coordinates[candidates] - self.coordinates[anchor]
        squared = np.sum(differences**2, axis=1)
        # Include the final upper boundary so pairs exactly at max_distance remain available
        below_outer = squared <= outer**2 if outer == self.max_distance else squared < outer**2
        in_ring = (squared >= inner**2) & below_outer & (candidates != anchor)
        if not np.any(in_ring):
            return None

        # Choose one matching endpoint at random so stored row order does not bias the sample
        chosen = int(self.rng.choice(candidates[in_ring]))
        return chosen, float(np.linalg.norm(self.coordinates[chosen] - self.coordinates[anchor]))

    ###################
    # NEAREST POINT SAMPLING
    ###################

    def _sample_nearest_pairs(
        self, output: _DiskPairOutput | None = None
    ) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum] | None:
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
            candidate_coordinates = self.coordinates[anchors] + np.column_stack(
                (radii * np.cos(angles), radii * np.sin(angles))
            )

            # Match each candidate endpoint to the nearest observed point within the allowed error
            nearest_distance, neighbors = self.tree.query(candidate_coordinates, k=1)
            accepted = (neighbors != anchors) & (nearest_distance <= self.nn_tolerance * radii)
            first, second = anchors[accepted], neighbors[accepted].astype(np.int64, copy=False)
            distance = np.linalg.norm(self.coordinates[second] - self.coordinates[first], axis=1)
            in_range = (distance >= self.min_distance) & (distance <= self.max_distance)

            # Check exact distances and keep only the number of pairs still needed
            take = min(remaining, int(np.count_nonzero(in_range)))
            if take:
                if output is None:
                    first_parts.append(first[in_range][:take])
                    second_parts.append(second[in_range][:take])
                    distance_parts.append(distance[in_range][:take])
                else:
                    output.append(first[in_range][:take], second[in_range][:take], distance[in_range][:take])
                remaining -= take

        # Distinguish finding no pair from finding fewer pairs than requested
        if remaining == self.n_pairs:
            raise ValueError("No point pairs could be sampled within the requested distances.")
        if output is not None:
            return None
        return np.concatenate(first_parts), np.concatenate(second_parts), np.concatenate(distance_parts)

    #################
    # PAIR COLLECTION
    #################

    def sample(self, *, output: _DiskPairOutput | None = None) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum] | None:
        """
        Collect point pairs with the selected search strategy.

        Sampling options are configured by _PointPairSampler.__init__().

        If ``output`` is passed, pairs are written in batches.

        :returns: Three one-dimensional arrays containing first endpoint indexes, second endpoint indexes,
            and distances in coordinate units, or None when batches are written to output.
            Indexes refer to rows in the sampler's finite coordinate array.
        """

        # Use proposed offsets directly for the nearest-point strategy
        if self.strategy == "nn_logvector":
            result = self._sample_nearest_pairs(output)
            if output is not None:
                if output.count < self.n_pairs:
                    warnings.warn(f"Sampled {output.count} point pairs out of {self.n_pairs} requested.", UserWarning)
                return None
            assert result is not None
            first, second, distances = result
        else:
            # Search one first point and distance range at a time for the exact strategies
            first_values: list[int] = []
            second_values: list[int] = []
            distance_values: list[float] = []
            collected = 0
            for _ in range(self.max_rounds):
                if collected >= self.n_pairs:
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
                            collected += 1
                            if output is not None and len(first_values) == output.output_chunks:
                                output.append(
                                    np.asarray(first_values), np.asarray(second_values), np.asarray(distance_values)
                                )
                                first_values.clear()
                                second_values.clear()
                                distance_values.clear()
                        if collected == self.n_pairs:
                            break
                    if collected == self.n_pairs:
                        break

            # Convert the collected Python lists to numeric arrays
            if not collected:
                raise ValueError("No point pairs could be sampled within the requested distances.")
            if output is not None:
                if first_values:
                    output.append(np.asarray(first_values), np.asarray(second_values), np.asarray(distance_values))
                if output.count < self.n_pairs:
                    warnings.warn(f"Sampled {output.count} point pairs out of {self.n_pairs} requested.", UserWarning)
                return None
            first = np.asarray(first_values)
            second = np.asarray(second_values)
            distances = np.asarray(distance_values)

        # Warn when fewer pairs were found, but return every successful pair
        if first.size < self.n_pairs:
            warnings.warn(f"Sampled {first.size} point pairs out of {self.n_pairs} requested.", UserWarning)
        return first, second, distances


def _random_point_pairs(
    coordinates: NDArrayNum,
    *,
    n_pairs: int,
    min_distance: float,
    max_distance: float,
    random_state: int | np.random.Generator | None,
    max_rounds: int,
    batch_pairs: int,
    output: _DiskPairOutput | None = None,
) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum] | None:
    """Sample point pairs with two random endpoints."""

    # Draw independent endpoints in limited rounds for uniform random sampling
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    first: NDArrayNum = np.empty(0, dtype=np.int64)
    second: NDArrayNum = np.empty(0, dtype=np.int64)
    for _ in range(max_rounds):
        remaining = n_pairs - (first.size if output is None else output.count)
        if remaining <= 0:
            break

        # Draw extra candidate pairs before checking distances and removing duplicates
        count = min(batch_pairs, max(10_000, remaining * 3))
        first_candidate = rng.integers(0, len(coordinates), count, dtype=np.int64)
        second_candidate = rng.integers(0, len(coordinates), count, dtype=np.int64)
        candidate_distances = np.linalg.norm(coordinates[first_candidate] - coordinates[second_candidate], axis=1)
        keep = (
            (first_candidate != second_candidate)
            & (candidate_distances >= min_distance)
            & (candidate_distances <= max_distance)
        )
        if np.any(keep):
            # Remove duplicates across rounds before keeping the requested number
            if output is not None:
                output.append(first_candidate[keep], second_candidate[keep], limit=remaining)
            else:
                first, second = _deduplicate_pairs(
                    np.concatenate((first, first_candidate[keep])),
                    np.concatenate((second, second_candidate[keep])),
                    n_observations=len(coordinates),
                )
                first, second = first[:n_pairs], second[:n_pairs]

    # Return a smaller sample with a warning, but fail when no matching pair exists
    count = first.size if output is None else output.count
    if count == 0:
        raise ValueError("No point pairs could be sampled within the requested distances.")
    if count < n_pairs:
        warnings.warn(f"Sampled {count} unique point pairs out of {n_pairs} requested.", UserWarning)
    if output is not None:
        return None
    distances = np.linalg.norm(coordinates[first] - coordinates[second], axis=1)
    return first, second, distances


################
# 6/ PAIR OUTPUT
################


def _task_reads_pair_files(task: Any, read_keys: set[Any]) -> bool:
    """Find pair reads inside ordinary or fused Dask tasks."""

    from dask._task_spec import Task

    # Fused tasks store their original tasks in a nested graph
    pending = [task]
    while pending:
        part = pending.pop()
        if isinstance(part, Task):
            if part.key in read_keys:
                return True
            pending.extend(part.args)
            pending.extend(part.kwargs.values())
        elif isinstance(part, dict):
            if read_keys.intersection(part):
                return True
            pending.extend(part.values())
        elif isinstance(part, (list, tuple)):
            pending.extend(part)
    return False


def _pending_pair_reads(keys: Sequence[Any], read_keys: set[Any], dask_scheduler: Any) -> list[Any]:
    """Find submitted outputs that still depend on this dataset's pair files."""

    outputs = []
    for key in keys:
        # A newly submitted graph may still be waiting to reach the scheduler
        if key not in dask_scheduler.tasks:
            outputs.append(key)
            continue

        # Completed dependencies no longer need their source files
        pending = [dask_scheduler.tasks[key]]
        seen = set()
        while pending:
            task = pending.pop()
            if task.key in seen or task.state in {"memory", "erred", "forgotten"}:
                continue
            seen.add(task.key)
            if _task_reads_pair_files(task.run_spec, read_keys):
                outputs.append(key)
                break
            pending.extend(task.dependencies)
    return outputs


def _close_pair_storage(storage: ExitStack, read_keys: set[Any]) -> None:
    """Close temporary pair files after pending distributed reads finish."""

    # Persisted arrays contain futures and can release the original reader before workers read its files
    distributed = modules.get("distributed")
    if distributed is None:
        storage.close()
        return

    # Include non-default clients
    futures = []
    for client in list(distributed.Client._instances):
        if client.status != "running":
            continue
        keys = [key for key, state in list(client.futures.items()) if state.status == "pending"]
        if keys:
            keys = client.run_on_scheduler(_pending_pair_reads, keys, read_keys)
            futures.extend(distributed.Future(key, client=client) for key in keys)
    if not futures:
        storage.close()
        return

    # Retain storage until the submitted outputs finish; callbacks also release cancelled or failed reads
    remaining = set(futures)
    lock = Lock()

    def close_finished(future: Any) -> None:
        """Release storage when the last pending output completes."""

        with lock:
            remaining.discard(future)
            if not remaining:
                storage.close()

    for future in futures:
        future.add_done_callback(close_finished)


class _PairFileReader:
    """Read one column of a saved pair batch while its graph owns the temporary directory."""

    def __init__(self, storage: ExitStack) -> None:
        """Transfer file cleanup to the reader without serializing its cleanup callbacks to workers."""

        self.read_keys: set[Any] = set()
        weakref.finalize(self, _close_pair_storage, storage.pop_all(), self.read_keys)

    def __call__(self, filename: str, column: str, dtype: np.dtype[Any]) -> NDArrayNum:
        """Read only the requested column, with the common dtype chosen across all batches."""

        with np.load(filename, allow_pickle=False) as batch:
            return batch[column].astype(dtype, copy=False)


class _DiskPairOutput:
    """Append bounded accepted batches, optionally remove global duplicates on disk, and build a lazy Dataset."""

    def __init__(
        self,
        prepare_batch: Callable[
            [NDArrayNum, NDArrayNum, NDArrayNum | None, NDArrayNum | None], Mapping[str, NDArrayNum]
        ],
        *,
        output_chunks: int,
        directory: Path | None,
        deduplicate: bool,
    ) -> None:
        """Create isolated temporary storage and an optional pair index with an 8 MiB SQLite page cache."""

        self.storage = ExitStack()
        self.prepare_batch = prepare_batch
        self.output_chunks = output_chunks
        self.count = 0
        self.parts: list[tuple[str, int]] = []
        self.schema: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
        self.connection: sqlite3.Connection | None = None
        try:
            self.directory = Path(
                self.storage.enter_context(TemporaryDirectory(prefix="geoutils-pair-output-", dir=directory))
            ).resolve()
            if deduplicate:
                self.connection = sqlite3.connect(self.directory / "unique.sqlite", check_same_thread=False)
                self.storage.callback(self.connection.close)
                # Bound the index cache and put temporary SQL data on disk
                self.connection.execute("PRAGMA cache_size=-8192")
                self.connection.execute("PRAGMA temp_store=FILE")
                self.connection.execute("PRAGMA journal_mode=OFF")
                self.connection.execute("PRAGMA synchronous=OFF")
                self.connection.execute(
                    "CREATE TABLE pairs (first INTEGER, second INTEGER, PRIMARY KEY (first, second)) WITHOUT ROWID"
                )
        except BaseException:
            self.storage.close()
            raise

    def __enter__(self) -> _DiskPairOutput:
        """Return the writer whose context removes files if sampling fails."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Remove unfinished output; a returned Dataset has already transferred file ownership to its reader."""

        self.storage.close()

    def append(
        self,
        first: NDArrayNum,
        second: NDArrayNum,
        distances: NDArrayNum | None = None,
        values: NDArrayNum | None = None,
        *,
        limit: int | None = None,
    ) -> None:
        """
        Write accepted pairs in order, putting the smaller index first for unique pairs and limiting new rows.

        Log-lag raster sampling truncates before global deduplication; its caller supplies that truncated batch.
        Independent sampling instead supplies limit after deduplication, matching its existing refill behavior.
        Output buffers scale with output_chunks, alongside SQLite's bounded cache and the caller's candidate arrays.
        """

        written = 0
        for start in range(0, len(first), self.output_chunks):
            if limit is not None and written >= limit:
                break
            stop = start + self.output_chunks
            batch_first, batch_second = first[start:stop], second[start:stop]
            distance = None if distances is None else distances[start:stop]
            value = None if values is None else values[start:stop]

            # A disk index finds first occurrences across every batch without storing all pair keys in RAM
            if self.connection is not None:
                low, high = np.minimum(batch_first, batch_second), np.maximum(batch_first, batch_second)
                positions: list[int] = []
                with self.connection:
                    cursor = self.connection.cursor()
                    for position, (left_index, right_index) in enumerate(zip(low, high)):
                        cursor.execute("INSERT OR IGNORE INTO pairs VALUES (?, ?)", (int(left_index), int(right_index)))
                        if cursor.rowcount:
                            positions.append(position)
                selected = np.asarray(positions, dtype=np.int64)
                if value is not None:
                    value = np.where((batch_first > batch_second)[:, None], value[:, ::-1], value)[selected]
                batch_first, batch_second = low[selected], high[selected]
                if distance is not None:
                    distance = distance[selected]

            # Truncate unique random pairs after rejection, preserving the native sample order
            if limit is not None:
                take = limit - written
                batch_first, batch_second = batch_first[:take], batch_second[:take]
                distance = None if distance is None else distance[:take]
                value = None if value is None else value[:take]
            if not len(batch_first):
                continue
            batch = self.prepare_batch(batch_first, batch_second, distance, value)

            # Save uncompressed columns so reducers can read values and distances without loading coordinates
            filename = self.directory / f"batch-{len(self.parts)}.npz"
            np.savez(filename, **batch)
            for name, array in batch.items():
                shape, dtype = self.schema.get(name, (array.shape[1:], array.dtype))
                self.schema[name] = shape, np.result_type(dtype, array.dtype)
            self.parts.append((str(filename), len(batch_first)))
            self.count += len(batch_first)
            written += len(batch_first)

    def dataset(self, attrs: Mapping[str, Any], *, force_output_to_memory: bool = False) -> xr.Dataset:
        """
        Expose saved columns as aligned Dask chunks and give their graphs ownership of the temporary files.

        A compact pandas RangeIndex names pairs without allocating one coordinate value per pair. Graph size and
        file descriptors grow with the number of output chunks; neither the pair columns nor all pair keys are
        gathered. Workers must share the temporary directory with the client.
        """

        dask = import_optional("dask")
        import dask.array as da

        # The duplicate index is no longer needed once accepted batches are saved
        if self.connection is not None:
            self.connection.close()
            (self.directory / "unique.sqlite").unlink()
        reader = _PairFileReader(self.storage)
        columns = {}
        for name, (shape, dtype) in self.schema.items():
            chunks = []
            for filename, count in self.parts:
                task = dask.delayed(reader)(filename, name, dtype)
                reader.read_keys.add(task.key)
                chunks.append(da.from_delayed(task, shape=(count, *shape), dtype=dtype))
            columns[name] = da.concatenate(chunks)

        result = _pair_dataset(
            columns,
            {**attrs, "output_storage": "disk", "lazy_output": 1, "output_chunks": self.output_chunks},
        )
        # Compute all saved columns for memory output
        if force_output_to_memory:
            result = result.compute()
            result.attrs.pop("lazy_output")
            result.attrs.pop("output_chunks")
            result.attrs["output_storage"] = "memory"
        return result


def _raster_pair_columns(
    raster: RasterBase,
    source: _RasterPairSource,
    first: NDArrayNum,
    second: NDArrayNum,
    distances: NDArrayNum | None,
    values: Any | None,
    *,
    index_dtype: np.dtype[Any],
    distance_dtype: np.dtype[Any],
) -> dict[str, Any]:
    """Build raster pair columns from selected cell indexes and in-memory or lazy endpoint values."""

    # Convert cell indexes to raster and map coordinates
    first, second = first.astype(index_dtype, copy=False), second.astype(index_dtype, copy=False)
    first_rows, first_columns = np.divmod(first.astype(np.int64), source.shape[1])
    second_rows, second_columns = np.divmod(second.astype(np.int64), source.shape[1])
    first_x, first_y = raster.ij2xy(first_rows, first_columns)
    second_x, second_y = raster.ij2xy(second_rows, second_columns)

    # Read values only when sampling or a lazy read graph has not already supplied them
    if values is None:
        values = np.column_stack(source.read_pairs(first, second))
    if distances is None:
        dx, dy = (float(abs(value)) for value in get_geo_attr(raster, "res"))
        distances = np.hypot((second_columns - first_columns) * dx, (second_rows - first_rows) * dy)
    return {
        "index": np.column_stack((first, second)),
        "value": values,
        "distance": distances.astype(distance_dtype, copy=False),
        "row": np.column_stack((first_rows, second_rows)),
        "column": np.column_stack((first_columns, second_columns)),
        "x": np.column_stack((first_x, second_x)),
        "y": np.column_stack((first_y, second_y)),
    }


def _point_pair_columns(
    source: _PointPairData,
    first: NDArrayNum,
    second: NDArrayNum,
    distances: NDArrayNum | None,
    values: Any | None,
    *,
    index_dtype: np.dtype[Any],
    distance_dtype: np.dtype[Any],
) -> dict[str, Any]:
    """Restore original point rows and build pair columns with in-memory or lazy endpoint values."""

    # Select planar coordinates and recover the original source row numbers
    first_coordinates, second_coordinates = source.coordinates[first], source.coordinates[second]
    indexes = np.column_stack((source.original_indexes[first], source.original_indexes[second]))

    # Read values only when no lazy read graph was supplied and calculate missing distances
    if values is None:
        values = np.column_stack((source.values[first], source.values[second]))
    if distances is None:
        distances = np.linalg.norm(first_coordinates - second_coordinates, axis=1)
    return {
        "index": indexes.astype(index_dtype, copy=False),
        "value": values,
        "distance": distances.astype(distance_dtype, copy=False),
        "x": np.column_stack((first_coordinates[:, 0], second_coordinates[:, 0])),
        "y": np.column_stack((first_coordinates[:, 1], second_coordinates[:, 1])),
    }


def _pair_dataset(columns: Mapping[str, Any], attrs: Mapping[str, Any]) -> xr.Dataset:
    """Build the labelled pair Dataset from in-memory or lazy columns and sampling metadata."""

    # Share pair and endpoint dimensions across raster, point, memory and disk output
    data_vars = {}
    for name, values in columns.items():
        dimensions = ("pair",) if values.ndim == 1 else ("pair", "endpoint")
        data_vars[name] = (dimensions, values)

    # Name pairs without allocating one coordinate value per pair
    count = columns["index"].shape[0]
    pair_index = pd.RangeIndex(count, name="pair")
    result = xr.Dataset(data_vars, coords={"pair": pair_index, "endpoint": ["first", "second"]}, attrs=attrs)

    # Chunk every data column while preserving the deferred endpoint reads
    if attrs.get("lazy_output"):
        result = result.chunk({"pair": int(attrs["output_chunks"])})
    return result


###############################
# 7/ PAIRSAMPLE IMPLEMENTATIONS
###############################


def _sample_raster_pairs(
    raster: RasterBase,
    *,
    band: int,
    n_pairs: int,
    sampling: Literal["loglag", "random_xy"],
    min_distance: float | None,
    max_distance: float | None,
    random_state: int | np.random.Generator | None,
    mask: RasterLike | VectorLike | ArrayLike | None,
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
    mp_config: MultiprocConfig | None,
    force_output_to_memory: bool = False,
    output_chunks: int = 100_000,
    read_strategy: Literal["native", "reuse_values", "packed_validity"] = "native",
    output_storage: Literal["memory", "disk"] = "memory",
    output_directory: str | Path | None = None,
) -> xr.Dataset:
    """
    Sample pairs efficiently in a raster, with Dask/multiprocessing support.

    See pointcloud.pairsample() for the public API.

    Internal logic:
    - _prepare_raster_pair_source() prepares values and masks.
    - _RasterPairSampler.sample() draws pairs for "loglag", or _random_raster_pairs() for "random_xy".
    - _raster_pair_columns() builds the output columns and _pair_dataset() labels them.
    Optionally, _DiskPairOutput writes them.

    Strategy, duplicate, oversampling, anchor, and local distance controls apply only to ``"loglag"``.
    Both sampling schemes use ``batch_pairs`` and ``max_rounds``.

    :param raster: Raster to sample.
    :param band: Band to sample, with start index of 1.
    :param n_pairs: Requested number of pairs with two finite values, fewer may be returned if sampling stops early.
    :param sampling: Pairwise sampling method, ``"loglag"`` balances short and long distances on a log scale, while
        ``"random_xy"`` draws endpoints uniformly.
    :param min_distance: Smallest distance in CRS units (e.g. meters). Defaults to the smaller pixel spacing.
    :param max_distance: Largest distance in CRS units. Defaults to the diagonal between outermost cell centers.
    :param random_state: Seed for reproducible sampling (e.g. 42).
    :param mask: Eligible cells: True in a mask array or aligned mask raster, or inside vector geometries.
    :param strategy: GeoUtils log-lag strategy: ``"independent"`` draws each pair separately, ``"anchors"``
        reuses first endpoints, ``"chunk_anchors"`` also limits source chunks, and ``"anchor_batched"`` draws
        several distances and directions from each first endpoint.
    :param deduplicate: ``"none"`` keeps repeats, ``"per_anchor"`` removes repeated second endpoints within each anchor
        batch, and ``"global"`` removes repeated pairs across all batches. ``"random_xy"`` always removes repeats.
    :param batch_pairs: Maximum candidate pairs per batch; smaller batches use less temporary memory.
    :param max_rounds: Maximum attempts to fill the sample after rejecting missing values or out-of-range pairs.
    :param max_oversample: Maximum candidate count as a multiple of the target pair count (e.g. 8).
    :param chunks_per_round: Maximum source chunks used when drawing anchors from selected chunks.
    :param anchors_per_round: Maximum first endpoints reused per round by ``"anchors"``, ``"chunk_anchors"``,
        or local ``"independent"`` sampling.
    :param distances_per_anchor: Distances drawn per first endpoint with ``"anchor_batched"``.
    :param angles_per_distance: Directions drawn per distance with ``"anchor_batched"``.
    :param hybrid_local_fraction: Fraction of candidate pairs kept within the first endpoint's chunk (e.g. 0.5).
        Zero samples across the full raster; one keeps all pairs local.
    :param max_local_distance: Largest proposed local distance in CRS units. Defaults to the largest chunk diagonal.
    :param index_dtype: Integer NumPy dtype for returned cell indexes (e.g. ``"int64"`` for very large rasters).
    :param distance_dtype: Floating NumPy dtype for returned distances (e.g. ``"float32"`` to reduce memory).
    :param mp_config: Worker and tile settings for reading an unloaded raster without loading its complete band.
    :param force_output_to_memory: Return every output column in memory, including disk output.
    :param output_chunks: Maximum pairs in each returned Dask value chunk.
    :param read_strategy: Source reads independent of pair generation: native reads, cached accepted values,
        or a packed validity scan followed by final endpoint reads.
    :param output_storage: Store ordinary pair output in memory, or write bounded disk batches with mp_config.
    :param output_directory: Existing parent directory for temporary pair output, removed after its graphs are released.
    :returns: Xarray Dataset with pair and endpoint dimensions, containing cell indexes, values, coordinates,
        and distances.
    """

    # 1/ Check inputs and set distance limits
    _check_input_pair_output(output_chunks, output_storage, output_directory, force_output_to_memory, mp_config)
    lazy_output = _uses_dask(raster) and not force_output_to_memory
    if mp_config is not None and (_uses_dask(raster) or _uses_dask(mask)):
        raise ValueError("Cannot use Multiprocessing and Dask simultaneously in pairsample().")

    if read_strategy not in {"native", "reuse_values", "packed_validity"}:
        raise ValueError("Argument ``read_strategy`` must be 'native', 'reuse_values', or 'packed_validity'.")
    reused_values: list[NDArrayNum] | None = [] if read_strategy == "reuse_values" else None

    # Check input arguments
    if sampling not in {"loglag", "random_xy"}:
        raise ValueError("Argument ``sampling`` must be 'loglag' or 'random_xy'.")
    if band < 1 or band > raster.count:
        raise ValueError("Argument ``band`` must identify an existing raster band.")
    if int(n_pairs) < 1:
        raise ValueError("Argument ``n_pairs`` must be a positive integer.")
    if int(batch_pairs) < 1 or int(max_rounds) < 1:
        raise ValueError("Batch and round controls must be positive integers.")
    if sampling == "loglag":
        if strategy not in {"independent", "anchors", "chunk_anchors", "anchor_batched"}:
            raise ValueError("Unknown regular grid pair sampling ``strategy``.")
        if deduplicate not in {"none", "per_anchor", "global"}:
            raise ValueError("Argument ``deduplicate`` must be 'none', 'per_anchor' or 'global'.")
        if not 0 <= float(hybrid_local_fraction) <= 1:
            raise ValueError("Argument ``hybrid_local_fraction`` must be between 0 and 1.")
        if any(int(count) < 1 for count in (chunks_per_round, anchors_per_round)):
            raise ValueError("Chunk and anchor controls must be positive integers.")
        if any(int(count) < 1 for count in (distances_per_anchor, angles_per_distance)) or float(max_oversample) <= 0:
            raise ValueError("Distance, angle, and oversampling controls must be strictly positive.")
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    # Check output dtypes and index range
    index_type, distance_type = np.dtype(index_dtype), np.dtype(distance_dtype)
    if not np.issubdtype(index_type, np.integer) or not np.issubdtype(distance_type, np.floating):
        raise TypeError("Arguments ``index_dtype`` and ``distance_dtype`` must be integer and floating, respectively.")
    if int(np.prod(raster.shape)) - 1 > np.iinfo(index_type).max:
        raise ValueError("Argument ``index_dtype`` cannot represent every cell in this raster.")

    # Set default distances from raster geometry
    dx, dy = (float(abs(value)) for value in get_geo_attr(raster, "res"))
    diagonal = float(np.hypot((raster.shape[1] - 1) * dx, (raster.shape[0] - 1) * dy))
    minimum = min(dx, dy) if min_distance is None else float(min_distance)
    maximum = diagonal if max_distance is None else float(max_distance)
    if not 0 < minimum < maximum < np.inf:
        raise ValueError("Require 0 < ``min_distance`` < ``max_distance``.")

    # 2/ Select the raster band and apply its mask
    source = _prepare_raster_pair_source(raster, band, mask, mp_config)

    # 3/ Choose the pair sampler

    # For loglag
    if sampling == "loglag":
        sampler = _RasterPairSampler(
            source,
            dx=dx,
            dy=dy,
            n_pairs=n_pairs,
            min_distance=minimum,
            max_distance=maximum,
            strategy=strategy,
            deduplicate=deduplicate,
            random_state=rng,
            batch_pairs=batch_pairs,
            max_rounds=max_rounds,
            max_oversample=max_oversample,
            chunks_per_round=chunks_per_round,
            anchors_per_round=anchors_per_round,
            distances_per_anchor=distances_per_anchor,
            angles_per_distance=angles_per_distance,
            hybrid_local_fraction=hybrid_local_fraction,
            max_local_distance=max_local_distance,
        )
        if read_strategy == "packed_validity":
            source = source.with_packed_validity()
            sampler.source = source
        collect = partial(sampler.sample, pair_values=reused_values)

    # For "random_xy"
    else:
        if read_strategy == "packed_validity":
            source = source.with_packed_validity()
        collect = partial(
            _random_raster_pairs,
            source,
            dx=dx,
            dy=dy,
            n_pairs=n_pairs,
            min_distance=minimum,
            max_distance=maximum,
            random_state=rng,
            max_rounds=max_rounds,
            batch_pairs=batch_pairs,
            pair_values=reused_values,
        )

    # 4/ Define output columns and metadata
    attrs = {
        "source": "raster",
        "crs": str(get_geo_attr(raster, "crs")),
        "sampling": sampling,
        "strategy": strategy if sampling == "loglag" else "random_xy",
        "deduplicate": deduplicate,
        "requested_pairs": int(n_pairs),
        "min_distance": minimum,
        "max_distance": maximum,
        "band": int(band),
        **({"read_strategy": read_strategy} if read_strategy != "native" else {}),
    }
    prepare_batch = partial(_raster_pair_columns, raster, source, index_dtype=index_type, distance_dtype=distance_type)

    # 5/ Sample pairs
    if output_storage == "disk":
        directory = None if output_directory is None else Path(output_directory)
        with _DiskPairOutput(
            prepare_batch,
            output_chunks=output_chunks,
            directory=directory,
            deduplicate=deduplicate == "global" or sampling == "random_xy",
        ) as output:
            collect(output=output)
            return output.dataset(
                {**attrs, "accepted_pairs": output.count}, force_output_to_memory=force_output_to_memory
            )

    result = collect()
    assert result is not None
    first, second, distances = result

    # 6/ Reuse cached values or defer endpoint reads
    pair_values: Any = None
    if reused_values is not None:
        pair_values = reused_values[0]
        if lazy_output:
            import dask.array as da

            pair_values = da.from_array(pair_values, chunks=(output_chunks, 2))
    elif lazy_output:
        import_optional("dask")
        import dask.array as da

        array = source.values
        batches = []
        # Packed validity uses one final read graph with indexing bounded by source chunks
        batch_size = len(first) if read_strategy == "packed_validity" else output_chunks
        for start in range(0, len(first), batch_size):
            stop = start + batch_size
            selected = []
            for indexes in (first[start:stop], second[start:stop]):
                rows, columns = np.divmod(indexes.astype(np.int64), source.shape[1])
                endpoint = array.vindex[rows, columns]
                if np.ma.isMaskedArray(endpoint._meta):
                    data = da.ma.getdata(endpoint)
                    endpoint = da.where(da.ma.getmaskarray(endpoint), np.nan, data)
                selected.append(endpoint)
            batches.append(da.stack(selected, axis=1))
        pair_values = da.concatenate(batches).rechunk((output_chunks, 2))

    # 7/ Build pair dataset
    pair_columns = prepare_batch(first, second, distances, pair_values)
    return _pair_dataset(
        pair_columns,
        {
            **attrs,
            "accepted_pairs": int(len(first)),
            **({"lazy_output": 1, "output_chunks": int(output_chunks)} if lazy_output else {}),
        },
    )


def _sample_point_pairs(
    pointcloud: PointCloudBase,
    *,
    n_pairs: int,
    sampling: Literal["loglag", "random_xy"],
    min_distance: float | None,
    max_distance: float | None,
    random_state: int | np.random.Generator | None,
    mask: RasterLike | PointCloudLike | VectorLike | ArrayLike | None,
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
    mp_config: MultiprocConfig | None,
    force_output_to_memory: bool = False,
    output_chunks: int = 100_000,
    output_storage: Literal["memory", "disk"] = "memory",
    output_directory: str | Path | None = None,
) -> xr.Dataset:
    """
    Sample pairs efficiently in a point cloud, with Dask/multiprocessing support.

    See pointcloud.pairsample() for the public API.

    Internal logic:
    - _prepare_point_pair_data() prepares eligible coordinates and values.
    - _PointPairSampler.sample() searches for "loglag" pairs, or _random_point_pairs() for "random_xy".
    - _point_pair_columns() restores original row indexes and _pair_dataset() labels the columns.
    Optionally, _DiskPairOutput writes accepted batches for disk output.

    Strategies only apply to ``"loglag"``.
    ``"random_xy"`` uses ``max_rounds`` and ``nn_batch_size``.

    Dask point searches collect finite rows in memory. Multiprocessing file reads use temporary arrays.

    :param pointcloud: Point cloud to sample, using its main data attribute or geometry heights.
    :param n_pairs: Requested number of pairs with two finite values; fewer may be returned if sampling stops early.
    :param sampling: ``"loglag"`` balances short and long distances on a log scale; ``"random_xy"`` draws
        endpoints uniformly.
    :param min_distance: Smallest distance in CRS units (e.g. meters). Defaults to half the spacing estimated
        from the eligible point density.
    :param max_distance: Largest distance in CRS units. Defaults to the eligible point cloud's bounding box diagonal.
    :param random_state: Seed for reproducible sampling (e.g. 42).
    :param mask: Eligible points: True in a boolean array or spatial mask, or inside vector geometries.
        Point masks must follow the same ordered coordinates; raster masks use nearest interpolation.
        Missing mask entries are excluded.
    :param strategy: GeoUtils log-lag strategy: ``"kdtree"`` uses SciPy to search distance rings, ``"hashgrid"``
        searches rings using a spatial grid, and ``"nn_logvector"`` uses SciPy to match proposed endpoints
        to nearby points.
    :param n_bins: Log-spaced distance rings used by ``"kdtree"`` and ``"hashgrid"`` (e.g. 24).
    :param anchors_per_round: First endpoints tested per round by ``"kdtree"`` and ``"hashgrid"``.
    :param attempts_per_anchor: Distance rings tried per first endpoint by ``"kdtree"`` and ``"hashgrid"``.
    :param max_rounds: Maximum rounds to fill the sample with ``"kdtree"``, ``"hashgrid"``, or ``"random_xy"``.
    :param cell_size: Grid cell width in CRS units for ``"hashgrid"``. Defaults to one eighth of max_distance.
    :param nn_tolerance: Allowed endpoint snap distance for ``"nn_logvector"``, as a fraction of the proposed
        pair distance (e.g. 0.1 allows a 10% offset).
    :param nn_batch_size: Maximum candidate pairs per batch with ``"nn_logvector"`` or ``"random_xy"``;
        smaller batches use less temporary memory.
    :param nn_oversample: Candidate count as a multiple of the remaining pairs with ``"nn_logvector"`` (e.g. 2).
    :param nn_max_batches: Maximum batches to fill the sample with ``"nn_logvector"``.
    :param index_dtype: Integer NumPy dtype for returned row indexes (e.g. ``"int64"`` for very large point clouds).
    :param distance_dtype: Floating NumPy dtype for returned distances (e.g. ``"float64"`` for greater precision).
    :param mp_config: Worker and row partition settings for reading an unloaded point cloud into temporary
        disk-backed arrays before pair selection.
    :param force_output_to_memory: Return every output column in memory, including disk output.
    :param output_chunks: Maximum pairs in each deferred endpoint batch.
    :param output_storage: Store ordinary pair output in memory, or write bounded disk batches with mp_config.
    :param output_directory: Existing parent directory for temporary pair output, removed after its graphs are released.
    :returns: Xarray Dataset with pair and endpoint dimensions, containing original row indexes, values,
        coordinates, and distances.
    """

    # 1/ Check input options

    # Check Dask and Multiprocessing
    _check_input_pair_output(output_chunks, output_storage, output_directory, force_output_to_memory, mp_config)
    lazy_output = _uses_dask(pointcloud) and not force_output_to_memory

    if mp_config is not None and (_uses_dask(pointcloud) or _uses_dask(mask)):
        raise ValueError("Cannot use Multiprocessing and Dask simultaneously in pairsample().")
    if mp_config is not None:
        from geoutils.pointcloud.las import _point_partition_size

        _point_partition_size(mp_config)

    # Check output dtypes
    index_type, distance_type = np.dtype(index_dtype), np.dtype(distance_dtype)
    if not np.issubdtype(index_type, np.integer) or not np.issubdtype(distance_type, np.floating):
        raise TypeError("Arguments ``index_dtype`` and ``distance_dtype`` must be integer and floating, respectively.")

    # Check arguments
    if sampling not in {"loglag", "random_xy"}:
        raise ValueError("Argument ``sampling`` must be 'loglag' or 'random_xy'.")
    if n_pairs < 1 or max_rounds < 1:
        raise ValueError("Arguments ``n_pairs`` and ``max_rounds`` must be positive integers.")
    if nn_batch_size < 1:
        raise ValueError("Argument ``nn_batch_size`` must be a positive integer.")
    if min_distance is not None and not 0 < min_distance < np.inf:
        raise ValueError("Require 0 < ``min_distance`` < ``max_distance``.")
    if max_distance is not None and (
        not 0 < max_distance < np.inf or (min_distance is not None and min_distance >= max_distance)
    ):
        raise ValueError("Require 0 < ``min_distance`` < ``max_distance``.")
    if sampling == "loglag":
        if strategy not in {"kdtree", "hashgrid", "nn_logvector"}:
            raise ValueError("Unknown irregular point pair sampling ``strategy``.")
        if int(n_bins) < 1:
            raise ValueError("Argument ``n_bins`` must be a positive integer.")
        if int(anchors_per_round) < 1 or int(attempts_per_anchor) < 1:
            raise ValueError("Exact search batch controls must be positive integers.")
        if (cell_size is not None and cell_size <= 0) or nn_tolerance <= 0:
            raise ValueError("Arguments ``cell_size`` and ``nn_tolerance`` must be strictly positive.")
        if nn_oversample <= 0 or nn_max_batches < 1:
            raise ValueError("Nearest-neighbor batch controls must be strictly positive.")
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    # 2/ Read finite point values and infer distance limits
    with _prepare_point_pair_data(pointcloud, mask, mp_config) as pair_data:
        coordinates_valid = pair_data.coordinates
        values_valid = pair_data.values
        if len(values_valid) < 2:
            raise ValueError("At least two finite points are required to sample pairs.")

        # Check index range after discovering the original point count
        if pair_data.source_size - 1 > np.iinfo(index_type).max:
            raise ValueError("Argument ``index_dtype`` cannot represent every point in this point cloud.")

        # Choose default distances from the point extent and typical point spacing
        bounds = np.ptp(coordinates_valid, axis=0)
        diagonal = float(np.hypot(*bounds))
        density_spacing = float(np.sqrt(max(bounds[0] * bounds[1], 0) / len(values_valid)))
        minimum = (
            max(0.5 * density_spacing, float(np.finfo(float).eps)) if min_distance is None else float(min_distance)
        )
        maximum = diagonal if max_distance is None else float(max_distance)
        if not 0 < minimum < maximum:
            raise ValueError("Require 0 < ``min_distance`` < ``max_distance``.")

        # 3/ Choose the pair sampler

        # For "loglag"
        if sampling == "loglag":
            sampler = _PointPairSampler(
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
                random_state=rng,
            )
            collect = sampler.sample

        # For "random_xy"
        else:
            collect = partial(
                _random_point_pairs,
                coordinates_valid,
                n_pairs=n_pairs,
                min_distance=minimum,
                max_distance=maximum,
                random_state=rng,
                max_rounds=max_rounds,
                batch_pairs=nn_batch_size,
            )

        # 4/ Define output columns and metadata
        attrs = {
            "source": "pointcloud",
            "crs": str(pointcloud.crs),
            "sampling": sampling,
            "strategy": strategy if sampling == "loglag" else "random_xy",
            "requested_pairs": int(n_pairs),
            "min_distance": minimum,
            "max_distance": maximum,
        }
        prepare_batch = partial(_point_pair_columns, pair_data, index_dtype=index_type, distance_dtype=distance_type)

        # 5/ Sample pairs
        if output_storage == "disk":
            directory = None if output_directory is None else Path(output_directory)
            with _DiskPairOutput(
                prepare_batch,
                output_chunks=output_chunks,
                directory=directory,
                deduplicate=sampling == "random_xy",
            ) as output:
                collect(output=output)
                return output.dataset(
                    {**attrs, "accepted_pairs": output.count}, force_output_to_memory=force_output_to_memory
                )

        result = collect()
        assert result is not None
        first, second, distances = result

        # 6/ Build the pair dataset with original row indexes and selected values
        pair_columns = prepare_batch(first, second, distances, None)
        return _pair_dataset(
            pair_columns,
            {
                **attrs,
                "accepted_pairs": int(len(first)),
                **({"lazy_output": 1, "output_chunks": int(output_chunks)} if lazy_output else {}),
            },
        )
