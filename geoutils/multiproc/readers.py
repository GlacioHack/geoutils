# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Read raster or point cloud values in reusable blocks across one or more processing passes."""

from __future__ import annotations

import copy
import math
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pyogrio

from geoutils._dispatch import _get_pointcloud_interface, _get_raster_interface
from geoutils._misc import import_optional
from geoutils.raster.array import get_mask_from_array

if TYPE_CHECKING:
    import geopandas as gpd

    from geoutils.multiproc.mparray import MultiprocConfig
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.raster.base import RasterBase


########################################
# 1/ RASTER AND POINT CLOUD READERS
########################################


@dataclass(frozen=True)
class _ValueReader:
    """
    Describe raster or point cloud values that workers can read one block at a time.

    Values can come directly from a raster band or point cloud column. They can also be calculated from a raster or
    vector on the requested raster cells or point rows. block() limits the reader to one such part before it is sent
    to a worker.

    For rasters, ``selector`` is a band number. For point clouds, it is a column name, and None uses the main data
    column or geometry height. The reader gets the value shape and data type from the file.
    """

    source: Any
    selector: int | str | None = None
    mask: Any | None = None
    support: Any | None = None
    interpolation: str = "linear"
    chunks: int | tuple[int, int] = 512
    vector_values: Any | None = None
    mask_mode: Literal["inside", "outside"] | None = None
    coverage: bool = False
    shape: tuple[int, ...] = field(init=False)
    dtype: np.dtype[Any] = field(init=False)
    kind: Literal["raster", "point", "interpolated", "vector"] = field(init=False)

    def __post_init__(self) -> None:
        """Check the raster or point cloud source without loading its values."""

        # Use the support shape because vector values are calculated when each block is read
        if self.vector_values is not None:
            assert self.support is not None
            raster = _get_raster_interface(self.support)
            shape = tuple(raster.shape) if raster is not None else (int(self.support.point_count),)
            object.__setattr__(self, "shape", shape)
            object.__setattr__(self, "dtype", np.dtype(bool if self.mask_mode is not None or self.coverage else float))
            object.__setattr__(self, "kind", "vector")
            return

        # Copy the source information so worker reads cannot load or change the source object
        raster = _get_raster_interface(self.source)
        pointcloud = _get_pointcloud_interface(self.source) if raster is None else None
        source = raster if raster is not None else pointcloud
        if source is None or source.name is None or source.is_loaded:
            raise ValueError("Value readers require an unloaded raster or point cloud.")
        source = copy.copy(source)

        # Record the selected raster band and the shape of its output
        if raster is not None:
            selector = 1 if self.selector is None else self.selector
            if not isinstance(selector, (int, np.integer)) or not 1 <= selector <= source.count:
                raise ValueError("Raster bands must be integers between one and the raster band count.")
            selector = int(selector)
            shape = tuple(source.shape)
            dtype = np.dtype(bool if source.is_mask else source.dtype)
            kind = "raster"
            if self.support is not None:
                shape = (int(self.support.point_count),)
                dtype = np.dtype(bool if source.is_mask else float)
                kind = "interpolated"
        else:
            # Record the point count and selected column type from the file
            selector = source.data_column if self.selector is None else self.selector
            if selector is not None and (not isinstance(selector, str) or selector not in source.columns):
                raise ValueError(f"Point column {selector!r} does not exist.")
            from geoutils.pointcloud.las import _is_laspy_supported

            if _is_laspy_supported(source.name):
                laspy = import_optional("laspy")
                with laspy.open(source.name) as reader:
                    count = reader.header.point_count
                    dimension = reader.header.point_format.dimension_by_name(selector or "Z")
                    if dimension.num_elements != 1:
                        raise ValueError("Point values require one scalar value per selected column.")
                    if selector in (None, "Z") or dimension.scales is not None or dimension.offsets is not None:
                        dtype = np.dtype(float)
                    else:
                        dtype = np.dtype("uint8") if dimension.dtype is None else dimension.dtype
            else:
                info = pyogrio.read_info(source.name, force_feature_count=True)
                count = info["features"]
                dtypes = dict(zip(info["fields"], info["dtypes"]))
                dtype = np.dtype(float if selector is None else dtypes[selector])
            shape = (int(count),)
            kind = "point"

        # Store the checked source and its basic information on the frozen reader
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "selector", selector)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "kind", kind)

    @property
    def ndim(self) -> int:
        """Return the number of dimensions in the raster or point cloud layout."""

        return len(self.shape)

    @property
    def size(self) -> int:
        """Return the total number of raster cells or points."""

        return math.prod(self.shape)

    def block(self, slices: slice | tuple[slice, ...]) -> _BlockReader:
        """Create a reader for contiguous slices and the matching part of any in-memory mask."""

        # Make slice bounds explicit before they are sent to workers
        slices = (slices,) if isinstance(slices, slice) else slices
        if not isinstance(slices, tuple) or len(slices) != self.ndim or any(not isinstance(s, slice) for s in slices):
            raise TypeError("Argument ``slices`` must contain one slice per input dimension.")
        ranges = [part.indices(length) for part, length in zip(slices, self.shape)]
        if any(step != 1 for _, _, step in ranges):
            raise ValueError("Argument ``slices`` must select contiguous increasing slices.")
        slices = tuple(slice(start, max(start, stop)) for start, stop, _ in ranges)
        shape = tuple(part.stop - part.start for part in slices)

        # Send only the matching part of an in-memory mask to each worker
        mask = self.mask
        if isinstance(mask, _ValueReader):
            mask = mask.block(slices)
        elif mask is not None:
            mask = mask[slices]
        return _BlockReader(
            source=self.source,
            selector=self.selector,
            kind=self.kind,
            slices=slices,
            shape=shape,
            dtype=self.dtype,
            mask=mask,
            support=self.support,
            interpolation=self.interpolation,
            chunks=self.chunks,
            vector_values=self.vector_values,
            mask_mode=self.mask_mode,
            coverage=self.coverage,
        )

    def read(self, slices: slice | tuple[slice, ...]) -> Any:
        """Read only the requested raster window or point rows and apply the optional mask."""

        return self.block(slices).read()

    def read_points(self, rows: slice) -> gpd.GeoDataFrame:
        """Read point rows with their geometry and selected column, ignoring the value mask."""

        if self.kind != "point":
            raise TypeError("Point rows require a point cloud reader.")
        block = self.block(rows)
        return _read_point_rows(block.source, block.slices[0], block.selector)


def _normalize_reader_mask(mask: Any, shape: tuple[int, ...]) -> Any:
    """Validate a reader mask from file information and ordinary masks from their values."""

    if isinstance(mask, _ValueReader):
        if mask.shape != shape or not np.issubdtype(mask.dtype, np.bool_):
            raise ValueError("Argument ``mask`` must be boolean and contain one value per input location.")
        return mask

    from geoutils.sampling.support import _normalize_mask_array

    return _normalize_mask_array(mask, shape)


@dataclass(frozen=True)
class _BlockReader:
    """Store the source and slices needed to read one raster or point cloud block inside a worker."""

    source: Any
    selector: int | str | None
    kind: Literal["raster", "point", "interpolated", "vector"]
    slices: tuple[slice, ...]
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    mask: Any | None = None
    support: Any | None = None
    interpolation: str = "linear"
    chunks: int | tuple[int, int] = 512
    vector_values: Any | None = None
    mask_mode: Literal["inside", "outside"] | None = None
    coverage: bool = False

    def read(self) -> Any:
        """Read this raster window or set of point rows and apply its optional mask."""

        # Return an empty result without giving file readers an invalid zero-size request
        values: Any
        if math.prod(self.shape) == 0:
            values = np.empty(self.shape, dtype=self.dtype)
        elif self.kind == "raster":
            from geoutils.raster.transformation import _crop

            # Read the selected raster band and window
            assert isinstance(self.selector, (int, np.integer))
            raster = copy.copy(self.source)
            raster._bands = self.source.bands[int(self.selector) - 1]
            raster._out_count = 1
            rows, columns = self.slices
            bounds = (columns.start, rows.start, columns.stop, rows.stop)
            values, _ = _crop(raster, bounds, distance_unit="pixel")
        elif self.kind == "vector":
            from affine import Affine

            from geoutils.raster import Raster
            from geoutils.sampling.support import (
                _mask_at_support,
                _sample_vector_values,
            )

            # Build only the requested part of the raster or point support
            raster = _get_raster_interface(self.support)
            if raster is not None:
                rows, columns = self.slices
                transform = raster.transform * Affine.translation(columns.start, rows.start)
                support = Raster.from_array(
                    np.zeros(self.shape, dtype=bool), transform, raster.crs, area_or_point=raster.area_or_point
                )
                points = None
            else:
                support = self.support
                assert support is not None
                points = _read_point_rows(support, self.slices[0], None)

            # Calculate a vector mask or values on this part of the support
            if self.mask_mode is not None:
                values = _mask_at_support(self.source, support, support_dataframe=points, mask_mode=self.mask_mode)
            else:
                assert self.vector_values is not None
                values = _sample_vector_values(self.source, self.vector_values, support, points)
                if self.coverage:
                    values = np.isfinite(values)
        elif self.kind == "interpolated":
            from geoutils.multiproc.mparray import MultiprocConfig

            # Read raster values at the point locations in this block
            assert self.support is not None
            points = _read_point_rows(self.support, self.slices[0], None)
            coordinates = (points.geometry.x.to_numpy(), points.geometry.y.to_numpy())
            config = MultiprocConfig(chunks=self.chunks)
            values = self.source.interp_points(
                coordinates, method=self.interpolation, band=self.selector, as_array=True, mp_config=config
            )
            if self.source.is_mask:
                values = np.isfinite(values) & (values != 0)
        else:
            # Read point rows and check that a separate support file uses the same order
            dataframe = _read_point_rows(self.source, self.slices[0], self.selector)
            if self.support is not None:
                from geoutils.pointcloud.testing import _point_coords_equal_eager

                reference = _read_point_rows(self.support, self.slices[0], None)
                if not _point_coords_equal_eager(dataframe, reference):
                    raise ValueError("Point values do not share the ordered support coordinates.")
            values = np.asarray(dataframe.geometry.z if self.selector is None else dataframe[self.selector])

        # Apply the common mask while preserving integer values
        if self.mask is not None:
            mask = _read_values(self.mask)
            keep = np.ma.filled(mask, False)
            invalid = get_mask_from_array(values).reshape(values.shape)
            values = np.ma.masked_where(~keep | invalid, values)
        return values


########################################
# 2/ READ RASTER CELLS AND POINT ROWS
########################################


def _read_point_rows(source: PointCloudBase, rows: slice, selector: int | str | None) -> gpd.GeoDataFrame:
    """Read selected point rows and columns without loading the complete point cloud."""

    count = rows.stop - rows.start
    assert selector is None or isinstance(selector, str)
    columns = [] if selector is None else [selector]

    # Read in-memory rows directly and otherwise use the reader for the file format
    if source.is_loaded or source._is_pd:
        dataframe = source.ds.iloc[rows]
        return dataframe[[*columns, dataframe.geometry.name]]
    assert source.name is not None
    from geoutils.pointcloud.las import _is_laspy_supported, _load_laspy_data_slice

    if _is_laspy_supported(source.name):
        return _load_laspy_data_slice(source.name, columns=columns, start=rows.start, count=count)

    # A zero max_features means unlimited rows in Pyogrio, so empty requests read at most one row for their schema
    dataframe = pyogrio.read_dataframe(
        source.name, columns=columns, skip_features=rows.start, max_features=max(1, count)
    )
    return dataframe.iloc[:0] if count == 0 else dataframe


def _read_values(value: Any) -> Any:
    """Read a raster or point cloud block and pass ordinary in-memory values through unchanged."""

    return value.read() if isinstance(value, _BlockReader) else value


def _read_block_sample(block: Any, indexes: tuple[Any, ...]) -> Any:
    """Read one raster or point cloud block and return only its requested positions."""

    return _read_values(block)[indexes]


def _read_selected_values(reader: _ValueReader, indexes: Any, mp_config: MultiprocConfig | None) -> Any:
    """Read selected raster cells or points from only the blocks that contain them, preserving their order."""

    from geoutils.multiproc.chunked import iter_chunk_slices
    from geoutils.multiproc.cluster import _map_bounded

    assert mp_config is not None
    arguments = []
    positions_in_sample = []
    coordinates = np.unravel_index(indexes, reader.shape)

    # Find which requested positions belong to each block
    for slices in iter_chunk_slices(reader.shape, mp_config.chunks):
        selected = np.ones(len(indexes), dtype=bool)
        for positions, part in zip(coordinates, slices):
            selected &= (positions >= part.start) & (positions < part.stop)
        if not np.any(selected):
            continue

        # Record local positions and their place in the requested order
        local = tuple(positions[selected] - part.start for positions, part in zip(coordinates, slices))
        arguments.append((reader.block(slices), local))
        positions_in_sample.append(np.flatnonzero(selected))

    # Read only blocks that contain selected positions
    pieces = [result for _, result in _map_bounded(mp_config.cluster, _read_block_sample, arguments)]
    if not pieces:
        return np.empty(0, dtype=reader.dtype)

    # Restore the requested order and use the data type returned by the file reader
    concatenate = np.ma.concatenate if any(np.ma.isMaskedArray(piece) for piece in pieces) else np.concatenate
    sampled = concatenate(pieces)
    order = np.argsort(np.concatenate(positions_in_sample))
    return sampled[order]


########################################
# 3/ CREATE RASTER AND POINT CLOUD READERS
########################################


def _reader_from_vector(
    dataframe: Any,
    values: Any,
    support: Any,
    mp_config: MultiprocConfig | None,
    *,
    mask_mode: Literal["inside", "outside"] | None = None,
    coverage: bool = False,
) -> _ValueReader | None:
    """Create a reader that places vector values or a vector mask on an unloaded raster or point cloud support."""

    # Keep loaded and Dask supports on their existing processing paths
    if mp_config is None or support.is_loaded:
        return None
    raster = _get_raster_interface(support)
    if raster is not None and raster._is_xr:
        return None
    pointcloud = _get_pointcloud_interface(support)
    if pointcloud is not None and pointcloud._is_pd:
        return None
    return _ValueReader(dataframe, support=support, vector_values=values, mask_mode=mask_mode, coverage=coverage)


def _reader_from_source(
    value_source: Any,
    selector: int | str | None,
    support: RasterBase | PointCloudBase,
    mp_config: MultiprocConfig | None,
    *,
    align: str = "raise",
    interpolation: str = "linear",
    stack: ExitStack | None = None,
) -> _ValueReader | None:
    """Create a reader when an unloaded raster or point cloud can stay on disk."""

    if mp_config is None:
        return None
    raster = _get_raster_interface(value_source)
    if raster is not None:
        # Use a raster reader only when its values can stay on disk
        if raster._is_xr or raster.is_loaded or raster.name is None:
            return None
        support_raster = _get_raster_interface(support)
        if support_raster is not None and raster.georeferenced_grid_equal(support_raster):
            return _ValueReader(raster, selector)
        support_points = _get_pointcloud_interface(support)
        if support_points is not None and support_points._is_dask:
            raise ValueError("Dask inputs cannot be combined with multiprocessing reads from files.")
        needs_alignment = support_raster is not None or raster.crs != support.crs
        if needs_alignment:
            # Align the raster once because workers cannot read it directly on the requested support
            if align != "reproject" or stack is None:
                return None
            from geoutils.sampling.support import _aligned_raster

            config = stack.enter_context(mp_config.temporary())
            config.driver = "GTiff"
            raster = _aligned_raster(raster, raster, support, "values", align, mp_config=config)
        if support_raster is not None:
            return _ValueReader(raster, selector)
        return _ValueReader(
            raster, selector, support=support_points, interpolation=interpolation, chunks=mp_config.chunks
        )

    # Use a point reader only when values and support have the same ordered locations
    pointcloud = _get_pointcloud_interface(value_source)
    support_points = _get_pointcloud_interface(support)
    if pointcloud is not None and support_points is not None:
        if pointcloud._is_pd or pointcloud.is_loaded or pointcloud.name is None:
            return None
        if support_points._is_dask:
            raise ValueError("Dask inputs cannot be combined with multiprocessing reads from files.")
        if pointcloud.crs != support_points.crs:
            if align != "reproject" or stack is None:
                return None
            config = stack.enter_context(mp_config.temporary())
            config.driver = "GPKG"
            config.outfile += ".gpkg"
            if isinstance(config.chunks, tuple):
                config.chunks = math.prod(config.chunks)
            pointcloud = pointcloud.reproject(crs=support_points.crs, mp_config=config)
        if pointcloud.point_count != support_points.point_count:
            raise ValueError("Point values do not share the ordered support coordinates.")
        reference = None if pointcloud is support_points else support_points
        return _ValueReader(pointcloud, selector, support=reference)
    return None
