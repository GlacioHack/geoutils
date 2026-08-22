# Copyright (c) 2026 GeoUtils developers
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
"""LasPy-backed point-cloud reading, filtering and writing utilities."""

from __future__ import annotations

import os
import pathlib
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from rasterio.coords import BoundingBox

from geoutils._misc import import_optional


@dataclass(frozen=True)
class LasMetadata:
    """Metadata available from a LasPy-readable point-cloud file."""

    crs: CRS | None
    point_count: int
    bounds: BoundingBox
    columns: pd.Index


def is_laspy_supported(filename: str | pathlib.Path) -> bool:
    """Return whether a filename looks like a LasPy-supported file."""

    suffix = pathlib.Path(filename).suffix.lower()
    return suffix in [".las", ".laz"]


def bounds_from_tuple(bounds: BoundingBox | Sequence[float]) -> BoundingBox:
    """Convert a 4-value bounds-like object to a Rasterio BoundingBox."""

    if isinstance(bounds, BoundingBox):
        return bounds

    if len(bounds) != 4:
        raise ValueError("Bounds must contain four values: left, bottom, right and top.")

    left, bottom, right, top = (float(value) for value in bounds)
    if left > right:
        raise ValueError("Bounds left coordinate must be smaller than or equal to right " "coordinate.")
    if bottom > top:
        raise ValueError("Bounds bottom coordinate must be smaller than or equal to top " "coordinate.")
    return BoundingBox(left=left, bottom=bottom, right=right, top=top)


def spatial_bounds_grid(
    bounds: BoundingBox | Sequence[float],
    block_size: float | tuple[float, float],
) -> list[BoundingBox]:
    """
    Split bounds into X/Y coordinate blocks.

    :param bounds: Bounds to split as (left, bottom, right, top).
    :param block_size: Block size in coordinate units, either a scalar
        or ``(x_size, y_size)``.
    :returns: List of bounds in row-major order from top-left to bottom-right.
    """

    bbox = bounds_from_tuple(bounds)
    if isinstance(block_size, tuple):
        x_size, y_size = block_size
    else:
        x_size = y_size = block_size

    if x_size <= 0 or y_size <= 0:
        raise ValueError("Block sizes must be strictly positive.")

    x_edges = list(np.arange(bbox.left, bbox.right, x_size)) + [bbox.right]
    y_edges = list(np.arange(bbox.bottom, bbox.top, y_size)) + [bbox.top]
    x_edges = sorted({float(edge) for edge in x_edges})
    y_edges = sorted({float(edge) for edge in y_edges})

    blocks = []
    for y_min, y_max in zip(reversed(y_edges[:-1]), reversed(y_edges[1:])):
        bottom = min(y_min, y_max)
        top = max(y_min, y_max)
        for left, right in zip(x_edges[:-1], x_edges[1:]):
            blocks.append(BoundingBox(left=left, bottom=bottom, right=right, top=top))
    return blocks


def _is_dask_dataframe(obj: Any) -> bool:
    """Return True for Dask DataFrame-like objects without importing Dask."""

    return obj.__class__.__module__.startswith("dask.dataframe")


def _normalise_las_columns(columns: Literal["all", "main"] | Iterable[str], data_column: str | None) -> list[str]:
    """Resolve user-facing column choices to LAS dimensions to load."""

    if columns == "all":
        raise ValueError("Column value 'all' can only be resolved with file metadata.")
    if columns == "main":
        columns_to_load = [data_column] if data_column is not None else []
    else:
        columns_to_load = list(columns)

    if "Z" not in columns_to_load:
        columns_to_load = ["Z"] + columns_to_load

    return columns_to_load


def resolve_las_columns(
    columns: Literal["all", "main"] | list[str],
    data_column: str | None,
    available_columns: pd.Index,
) -> list[str]:
    """Resolve LAS columns to load from user input and file metadata."""

    if columns == "all":
        columns_to_load = list(available_columns)
    else:
        columns_to_load = _normalise_las_columns(columns=columns, data_column=data_column)

    missing = [column for column in columns_to_load if column not in available_columns]
    if missing:
        raise ValueError(
            f"Column(s) {', '.join(missing)} not found among LAS dimensions. "
            "Available columns are: "
            f"{', '.join(available_columns)}."
        )

    return columns_to_load


def _empty_las_geodataframe(columns: list[str], crs: CRS | None) -> gpd.GeoDataFrame:
    """Build an empty GeoDataFrame with the LAS chunk column order."""

    data = {column: pd.Series(dtype="float64") for column in columns}
    return gpd.GeoDataFrame(data=data, geometry=gpd.GeoSeries([], crs=crs), crs=crs)


def _concat_las_geodataframes(parts: list[gpd.GeoDataFrame], columns: list[str], crs: CRS | None) -> gpd.GeoDataFrame:
    """Concatenate LAS chunk GeoDataFrames, preserving geometry and CRS."""

    if len(parts) == 0:
        return _empty_las_geodataframe(columns=columns, crs=crs)

    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=crs)


def _laspy_points_to_geodataframe(points: Any, crs: CRS | None, columns: list[str]) -> gpd.GeoDataFrame:
    """Convert LasPy point records to a GeoDataFrame."""

    columns_no_z = [column for column in columns if column != "Z"]
    data = {"Z": np.asarray(points.z)}
    for column in columns_no_z:
        data[column] = np.asarray(points[column])

    return gpd.GeoDataFrame(
        data=data,
        geometry=gpd.points_from_xy(x=points.x, y=points.y, crs=crs),
        columns=["Z"] + columns_no_z + ["geometry"],
        crs=crs,
    )


def _point_bounds_mask(
    x: Any,
    y: Any,
    bounds: BoundingBox | Sequence[float],
    *,
    include_right: bool = True,
    include_top: bool = True,
) -> Any:
    """Build a boolean mask selecting points within X/Y bounds."""

    bbox = bounds_from_tuple(bounds)
    right_mask = x <= bbox.right if include_right else x < bbox.right
    top_mask = y <= bbox.top if include_top else y < bbox.top
    mask = (x >= bbox.left) & right_mask
    mask &= (y >= bbox.bottom) & top_mask
    return mask


def load_laspy_metadata(filename: str | pathlib.Path) -> LasMetadata:
    """Load metadata from a LAS/LAZ/COPC file without loading points."""

    laspy = import_optional("laspy")

    with laspy.open(filename) as f:
        crs = f.header.parse_crs(prefer_wkt=False)
        point_count = f.header.point_count
        bounds = BoundingBox(
            left=f.header.x_min,
            right=f.header.x_max,
            bottom=f.header.y_min,
            top=f.header.y_max,
        )

        columns_names = list(f.header.point_format.dimension_names)
        columns_names = [column for column in columns_names if column not in ["X", "Y"]]
        columns = pd.Index(columns_names)

    return LasMetadata(crs=crs, point_count=point_count, bounds=bounds, columns=columns)


def _load_laspy_metadata(
    filename: str | pathlib.Path,
) -> tuple[CRS | None, int, BoundingBox, pd.Index]:
    """Backward-compatible tuple form of :func:`load_laspy_metadata`."""

    metadata = load_laspy_metadata(filename)
    return (
        metadata.crs,
        metadata.point_count,
        metadata.bounds,
        metadata.columns,
    )


def load_laspy_data(
    filename: str | pathlib.Path,
    columns: Literal["all", "main"] | list[str],
    data_column: str | None = "Z",
) -> gpd.GeoDataFrame:
    """Load point-cloud data from a LAS/LAZ/COPC file as a GeoDataFrame."""

    laspy = import_optional("laspy")

    metadata = load_laspy_metadata(filename)
    columns_to_load = resolve_las_columns(
        columns=columns,
        data_column=data_column,
        available_columns=metadata.columns,
    )
    las = laspy.read(filename)
    return _laspy_points_to_geodataframe(points=las, crs=metadata.crs, columns=columns_to_load)


def _load_laspy_data(filename: str | pathlib.Path, columns: list[str]) -> gpd.GeoDataFrame:
    """Backward-compatible loader for callers that already resolved columns."""

    laspy = import_optional("laspy")

    with laspy.open(filename) as reader:
        crs = reader.header.parse_crs(prefer_wkt=False)
    las = laspy.read(filename)
    return _laspy_points_to_geodataframe(points=las, crs=crs, columns=columns)


def load_laspy_data_slice(filename: str | pathlib.Path, columns: list[str], start: int, count: int) -> gpd.GeoDataFrame:
    """Load a point-index slice of a LAS/LAZ/COPC file."""

    laspy = import_optional("laspy")

    if start < 0:
        raise ValueError("Slice start must be positive.")
    if count < 0:
        raise ValueError("Slice count must be positive.")

    with laspy.open(filename) as reader:
        crs = reader.header.parse_crs(prefer_wkt=False)
        reader.seek(start)
        points = reader.read_points(count)

    return _laspy_points_to_geodataframe(points=points, crs=crs, columns=columns)


def _load_laspy_data_slice(
    filename: str | pathlib.Path, columns: list[str], start: int, count: int
) -> gpd.GeoDataFrame:
    """Backward-compatible alias for :func:`load_laspy_data_slice`."""

    return load_laspy_data_slice(filename=filename, columns=columns, start=start, count=count)


def iter_laspy_data_chunks(
    filename: str | pathlib.Path,
    columns: Literal["all", "main"] | list[str],
    data_column: str | None = "Z",
    chunk_size: int = 1_000_000,
    bounds: BoundingBox | Sequence[float] | None = None,
) -> Iterator[gpd.GeoDataFrame]:
    """
    Iterate over LAS/LAZ points as GeoDataFrame chunks.

    Regular LAS/LAZ files do not contain a spatial index, so bounded selection
    streams through the file and filters points by coordinates. For COPC files,
    use :func:`load_laspy_data_bounds` with ``prefer_copc=True`` to use the
    LasPy COPC spatial index.
    """

    laspy = import_optional("laspy")

    if chunk_size <= 0:
        raise ValueError("Argument 'chunk_size' must be a strictly positive integer.")

    metadata = load_laspy_metadata(filename)
    columns_to_load = resolve_las_columns(
        columns=columns,
        data_column=data_column,
        available_columns=metadata.columns,
    )

    with laspy.open(filename) as reader:
        crs = reader.header.parse_crs(prefer_wkt=False)
        for points in reader.chunk_iterator(chunk_size):
            if bounds is not None:
                x = points.x.copy()
                y = points.y.copy()
                mask = _point_bounds_mask(x=x, y=y, bounds=bounds)
                if not np.any(mask):
                    continue
                points = points[mask]

            yield _laspy_points_to_geodataframe(points=points, crs=crs, columns=columns_to_load)


def _load_laspy_data_bounds_copc(
    filename: str | pathlib.Path,
    columns: list[str],
    bounds: BoundingBox | Sequence[float],
) -> gpd.GeoDataFrame:
    """Load points in X/Y bounds from a COPC file."""

    laspy = import_optional("laspy")

    bbox = bounds_from_tuple(bounds)
    with laspy.CopcReader.open(filename) as reader:
        crs = reader.header.parse_crs(prefer_wkt=False)
        query_bounds = laspy.Bounds(
            mins=np.array([bbox.left, bbox.bottom]),
            maxs=np.array([bbox.right, bbox.top]),
        )
        points = reader.spatial_query(query_bounds)

    return _laspy_points_to_geodataframe(points=points, crs=crs, columns=columns)


def load_laspy_data_bounds(
    filename: str | pathlib.Path,
    columns: Literal["all", "main"] | list[str],
    bounds: BoundingBox | Sequence[float],
    data_column: str | None = "Z",
    chunk_size: int = 1_000_000,
    prefer_copc: bool = True,
) -> gpd.GeoDataFrame:
    """
    Load points within X/Y bounds from a LasPy-readable file.

    COPC files are queried through LasPy's COPC spatial index when possible.
    Normal LAS/LAZ files are streamed by point chunks and filtered by
    coordinate masks.
    """

    metadata = load_laspy_metadata(filename)
    columns_to_load = resolve_las_columns(
        columns=columns,
        data_column=data_column,
        available_columns=metadata.columns,
    )

    if prefer_copc:
        try:
            return _load_laspy_data_bounds_copc(filename=filename, columns=columns_to_load, bounds=bounds)
        except Exception:
            pass

    parts = list(
        iter_laspy_data_chunks(
            filename=filename,
            columns=columns_to_load,
            data_column=data_column,
            chunk_size=chunk_size,
            bounds=bounds,
        )
    )
    return _concat_las_geodataframes(parts=parts, columns=columns_to_load, crs=metadata.crs)


def iter_laspy_spatial_chunks(
    filename: str | pathlib.Path,
    block_bounds: Iterable[BoundingBox | Sequence[float]],
    columns: Literal["all", "main"] | list[str],
    data_column: str | None = "Z",
    chunk_size: int = 1_000_000,
) -> Iterator[tuple[int, BoundingBox, gpd.GeoDataFrame]]:
    """
    Iterate over X/Y block chunks from a LAS/LAZ file.

    The input file is streamed once and every point chunk is routed to
    intersecting output blocks. Adjacent blocks are treated as left/bottom
    inclusive and right/top exclusive, except on the dataset outer edge,
    avoiding duplicates on tile boundaries for a non-overlapping block grid.
    """

    laspy = import_optional("laspy")

    if chunk_size <= 0:
        raise ValueError("Argument 'chunk_size' must be a strictly positive integer.")

    metadata = load_laspy_metadata(filename)
    columns_to_load = resolve_las_columns(
        columns=columns,
        data_column=data_column,
        available_columns=metadata.columns,
    )
    blocks = [bounds_from_tuple(bounds) for bounds in block_bounds]
    if len(blocks) == 0:
        return

    parts: list[list[gpd.GeoDataFrame]] = [[] for _ in blocks]
    full_bounds = metadata.bounds

    with laspy.open(filename) as reader:
        crs = reader.header.parse_crs(prefer_wkt=False)
        for points in reader.chunk_iterator(chunk_size):
            x = points.x.copy()
            y = points.y.copy()
            for index, bounds in enumerate(blocks):
                include_right = np.isclose(bounds.right, full_bounds.right)
                include_top = np.isclose(bounds.top, full_bounds.top)
                mask = _point_bounds_mask(
                    x=x,
                    y=y,
                    bounds=bounds,
                    include_right=include_right,
                    include_top=include_top,
                )
                if np.any(mask):
                    parts[index].append(
                        _laspy_points_to_geodataframe(
                            points=points[mask],
                            crs=crs,
                            columns=columns_to_load,
                        )
                    )

    for index, bounds in enumerate(blocks):
        yield index, bounds, _concat_las_geodataframes(parts=parts[index], columns=columns_to_load, crs=metadata.crs)


def _as_geodataframe(pc: gpd.GeoDataFrame | pd.DataFrame, crs: CRS | None = None) -> gpd.GeoDataFrame:
    """Ensure a dataframe has GeoPandas geometry and CRS."""

    if isinstance(pc, gpd.GeoDataFrame):
        out = pc
        if crs is not None and out.crs is None:
            out = out.set_crs(crs)
        return out

    attrs = getattr(pc, "attrs", {})
    pc_crs = crs if crs is not None else attrs.get("crs")
    return gpd.GeoDataFrame(
        pc,
        geometry="geometry",
        crs=pc_crs,
    )


def _iter_dataframe_chunks(pc: gpd.GeoDataFrame | pd.DataFrame, chunk_size: int | None) -> Iterator[gpd.GeoDataFrame]:
    """Iterate over a dataframe as GeoDataFrame row chunks."""

    if chunk_size is None:
        yield _as_geodataframe(pc)
        return

    if chunk_size <= 0:
        raise ValueError("Argument 'chunks' must be a strictly positive integer.")

    for start in range(0, len(pc), chunk_size):
        yield _as_geodataframe(pc.iloc[start : start + chunk_size])


def _header_native_dimensions(header: Any) -> set[str]:
    """Return dimensions already present in a LasPy point format."""

    return set(header.point_format.dimension_names)


def _non_geometry_columns(pc: gpd.GeoDataFrame | pd.DataFrame) -> list[str]:
    """Return dataframe columns except the geometry column."""

    return [column for column in pc.columns if column != "geometry"]


def _extra_las_columns(pc: gpd.GeoDataFrame | pd.DataFrame, data_column: str | None, header: Any) -> list[str]:
    """Return dataframe columns that need LAS extra dimensions."""

    native_dimensions = _header_native_dimensions(header)
    extra_columns = []
    for column in _non_geometry_columns(pc):
        if column == data_column:
            continue
        if column == "Z":
            raise ValueError("Column 'Z' is reserved for the LAS native Z dimension.")
        if column not in native_dimensions:
            extra_columns.append(column)
    return extra_columns


def build_laspy_header(
    pc: gpd.GeoDataFrame | pd.DataFrame,
    data_column: str | None,
    version: Any = None,
    point_format: Any = None,
    offsets: tuple[float, float, float] | None = None,
    scales: tuple[float, float, float] | None = None,
    crs: CRS | None = None,
    **kwargs: Any,
) -> Any:
    """Build a LasPy header for a GeoDataFrame point cloud."""

    laspy = import_optional("laspy")

    header = laspy.LasHeader(version=version, point_format=point_format)
    if scales is not None:
        header.scales = np.array(scales)
    if offsets is not None:
        header.offsets = np.array(offsets)
    for key, value in kwargs.items():
        setattr(header, key, value)

    pc = _as_geodataframe(pc, crs=crs)
    header_crs = crs if crs is not None else pc.crs
    if header_crs is not None:
        header.add_crs(CRS.from_user_input(header_crs))

    for column in _extra_las_columns(pc=pc, data_column=data_column, header=header):
        dtype = pc[column].dtype
        if not np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.bool_):
            raise TypeError(f"LAS extra dimension '{column}' must have a numeric or " "boolean dtype.")
        header.add_extra_dim(laspy.ExtraBytesParams(name=column, type=dtype))

    return header


def dataframe_to_lasdata(pc: gpd.GeoDataFrame | pd.DataFrame, data_column: str | None, header: Any) -> Any:
    """Convert a GeoDataFrame point-cloud chunk to a LasPy LasData object."""

    laspy = import_optional("laspy")

    pc = _as_geodataframe(pc)
    chunk_header = header.copy()
    chunk_header.partial_reset()
    las = laspy.LasData(chunk_header)

    las.x = pc.geometry.x.values
    las.y = pc.geometry.y.values
    if data_column is not None:
        las.z = pc[data_column].values
    else:
        las.z = pc.geometry.z.values

    native_dimensions = _header_native_dimensions(chunk_header)
    for column in _non_geometry_columns(pc):
        if column == data_column:
            continue
        if column == "Z":
            continue
        if column in native_dimensions:
            setattr(las, column, pc[column].values)
        else:
            setattr(las, column, pc[column].values)

    return las


def write_laspy_partitions(
    filename: str | pathlib.Path,
    partitions: Iterable[gpd.GeoDataFrame | pd.DataFrame],
    data_column: str | None,
    header: Any,
) -> None:
    """Write dataframe partitions to one LAS/LAZ file."""

    laspy = import_optional("laspy")

    write_header = header.copy()
    write_header.partial_reset()
    with laspy.open(filename, mode="w", header=write_header) as writer:
        for part in partitions:
            if len(part) == 0:
                continue
            las = dataframe_to_lasdata(pc=part, data_column=data_column, header=header)
            writer.write_points(las.points)


def _write_laspy_dataframe(
    filename: str | pathlib.Path,
    pc: gpd.GeoDataFrame | pd.DataFrame,
    data_column: str | None,
    header: Any,
    chunks: int | None = None,
) -> None:
    """Write an in-memory dataframe to LAS/LAZ, optionally by row chunks."""

    write_laspy_partitions(
        filename=filename,
        partitions=_iter_dataframe_chunks(pc=pc, chunk_size=chunks),
        data_column=data_column,
        header=header,
    )


def _write_laspy_dask_dataframe(
    filename: str | pathlib.Path,
    pc: Any,
    data_column: str | None,
    header: Any,
) -> None:
    """Write a Dask DataFrame to LAS/LAZ one partition at a time."""

    delayed_partitions = pc.to_delayed()

    def partitions() -> Iterator[gpd.GeoDataFrame]:
        for delayed_partition in delayed_partitions:
            part = delayed_partition.compute()
            yield _as_geodataframe(part, crs=getattr(pc, "_geoutils_attrs", {}).get("crs"))

    write_laspy_partitions(
        filename=filename,
        partitions=partitions(),
        data_column=data_column,
        header=header,
    )


def _write_laspy_temp_chunk(
    filename: str | pathlib.Path,
    pc: gpd.GeoDataFrame | pd.DataFrame,
    data_column: str | None,
    header: Any,
) -> str:
    """Write one dataframe chunk to a temporary LAS file."""

    _write_laspy_dataframe(
        filename=filename,
        pc=pc,
        data_column=data_column,
        header=header,
        chunks=None,
    )
    return os.fspath(filename)


def _stitch_laspy_files(
    filename: str | pathlib.Path,
    chunk_filenames: Iterable[str | pathlib.Path],
    header: Any,
    chunk_size: int,
) -> None:
    """Stitch temporary LAS chunk files into a final LAS/LAZ file."""

    laspy = import_optional("laspy")

    write_header = header.copy()
    write_header.partial_reset()
    with laspy.open(filename, mode="w", header=write_header) as writer:
        for chunk_filename in chunk_filenames:
            with laspy.open(chunk_filename) as reader:
                for points in reader.chunk_iterator(chunk_size):
                    writer.write_points(points)


def _write_laspy(
    filename: str | pathlib.Path,
    pc: gpd.GeoDataFrame | pd.DataFrame | Any,
    data_column: str | None,
    version: Any = None,
    point_format: Any = None,
    offsets: tuple[float, float, float] | None = None,
    scales: tuple[float, float, float] | None = None,
    chunks: int | None = None,
    mp_config: Any = None,
    **kwargs: Any,
) -> None:
    """Write a point-cloud dataframe to a LAS/LAZ/COPC file."""

    if chunks is not None and chunks <= 0:
        raise ValueError("Argument 'chunks' must be a strictly positive integer.")

    if mp_config is not None and _is_dask_dataframe(pc):
        raise ValueError("Multiprocessing LAS writing is not supported for Dask-backed " "point clouds.")

    header_pc = pc._meta if _is_dask_dataframe(pc) else pc
    crs = getattr(pc, "_geoutils_attrs", {}).get("crs") if _is_dask_dataframe(pc) else None
    header = build_laspy_header(
        pc=header_pc,
        data_column=data_column,
        version=version,
        point_format=point_format,
        offsets=offsets,
        scales=scales,
        crs=crs,
        **kwargs,
    )

    if _is_dask_dataframe(pc):
        _write_laspy_dask_dataframe(filename=filename, pc=pc, data_column=data_column, header=header)
        return

    if mp_config is not None:
        from geoutils.multiproc.pointcloud import _write_laspy_data_partitions

        _write_laspy_data_partitions(
            filename=filename,
            pc=pc,
            data_column=data_column,
            header=header,
            mp_config=mp_config,
        )
        return

    _write_laspy_dataframe(
        filename=filename,
        pc=pc,
        data_column=data_column,
        header=header,
        chunks=chunks,
    )


def write_laspy_spatial_chunks(
    filename: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    block_bounds: Iterable[BoundingBox | Sequence[float]],
    columns: Literal["all", "main"] | list[str],
    data_column: str | None = "Z",
    chunk_size: int = 1_000_000,
    prefix: str = "block",
) -> list[pathlib.Path]:
    """
    Split a LAS/LAZ file into X/Y block LAS files in one source pass.

    This is the preferred LAS/LAZ strategy when many spatial blocks are needed
    and the source is not COPC-indexed.
    """

    laspy = import_optional("laspy")

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = load_laspy_metadata(filename)
    resolve_las_columns(
        columns=columns,
        data_column=data_column,
        available_columns=metadata.columns,
    )
    blocks = [bounds_from_tuple(bounds) for bounds in block_bounds]
    if len(blocks) == 0:
        return []

    output_files = [output_path / f"{prefix}_{index}.las" for index in range(len(blocks))]
    writers: list[Any | None] = [None] * len(blocks)

    with laspy.open(filename) as reader:
        header = reader.header.copy()
        header.partial_reset()
        try:
            for points in reader.chunk_iterator(chunk_size):
                x = points.x.copy()
                y = points.y.copy()
                for index, bounds in enumerate(blocks):
                    include_right = np.isclose(bounds.right, metadata.bounds.right)
                    include_top = np.isclose(bounds.top, metadata.bounds.top)
                    mask = _point_bounds_mask(
                        x=x,
                        y=y,
                        bounds=bounds,
                        include_right=include_right,
                        include_top=include_top,
                    )
                    if not np.any(mask):
                        continue
                    writer = writers[index]
                    if writer is None:
                        writer = laspy.open(output_files[index], mode="w", header=header.copy())
                        writers[index] = writer
                    writer.write_points(points[mask])
        finally:
            for writer in writers:
                if writer is not None:
                    writer.close()

    # Ensure empty spatial chunks have valid LAS files too.
    for index, writer in enumerate(writers):
        if writer is None:
            write_laspy_partitions(
                filename=output_files[index],
                partitions=[],
                data_column=data_column,
                header=header,
            )

    return output_files


def write_laspy_multiproc_partitions(
    filename: str | pathlib.Path,
    pc: gpd.GeoDataFrame | pd.DataFrame,
    data_column: str | None,
    header: Any,
    chunks: int,
    cluster: Any,
) -> None:
    """Write dataframe partitions to temporary LAS files in workers."""

    if chunks <= 0:
        raise ValueError("Argument 'chunks' must be a strictly positive integer.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_paths = [pathlib.Path(tmp_dir) / f"chunk_{index}.las" for index, _ in enumerate(range(0, len(pc), chunks))]
        futures = []
        for tmp_path, part in zip(tmp_paths, _iter_dataframe_chunks(pc=pc, chunk_size=chunks)):
            futures.append(
                cluster.submit(_write_laspy_temp_chunk, tmp_path, part, data_column, header)
            )

        written_paths = cluster.gather(futures)
        _stitch_laspy_files(
            filename=filename,
            chunk_filenames=written_paths,
            header=header,
            chunk_size=chunks,
        )


def write_laspy_multiproc_temp_chunks(
    filename: str | pathlib.Path,
    pc: gpd.GeoDataFrame | pd.DataFrame,
    data_column: str | None,
    header: Any,
    chunks: int,
    cluster: Any,
) -> None:
    """Write dataframe chunks to temporary LAS files in workers."""

    write_laspy_multiproc_partitions(
        filename=filename,
        pc=pc,
        data_column=data_column,
        header=header,
        chunks=chunks,
        cluster=cluster,
    )
