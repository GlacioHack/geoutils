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
"""Multiprocessing orchestration for point-cloud operations."""

from __future__ import annotations

import pathlib
from typing import Any

import geopandas as gpd
import pandas as pd

from geoutils.multiproc.mparray import MultiprocConfig
from geoutils.pointcloud.las import (
    _concat_las_geodataframes,
    _load_laspy_data_slice,
    load_laspy_data_bounds,
    write_laspy_multiproc_partitions,
)


def _point_partition_size(mp_config: MultiprocConfig) -> int:
    """Return a scalar point partition size from a multiprocessing configuration."""

    if not isinstance(mp_config.chunks, int):
        raise ValueError("Point-cloud multiprocessing requires an integer chunk size.")
    return mp_config.chunks


def _load_laspy_data_partitions(
    filename: str | pathlib.Path,
    columns: list[str],
    point_count: int,
    partition_size: int,
    mp_config: MultiprocConfig,
) -> gpd.GeoDataFrame:
    """Load LAS/LAZ data by point partitions with multiprocessing."""

    if partition_size <= 0:
        raise ValueError("Argument 'partition_size' must be a strictly positive integer.")

    # Split point indexes into independent ranges that workers can seek to directly
    starts = list(range(0, point_count, partition_size))
    futures = []
    for start in starts:
        count = min(partition_size, point_count - start)
        futures.append(
            mp_config.cluster.submit(
                _load_laspy_data_slice,
                filename,
                columns,
                start,
                count,
            )
        )

    # Gather in submission order so the combined point cloud keeps source row order
    parts = mp_config.cluster.gather(futures)
    crs = parts[0].crs if len(parts) > 0 else None
    return _concat_las_geodataframes(parts=parts, columns=columns, crs=crs)


def _load_laspy_data_chunks(
    filename: str | pathlib.Path,
    columns: list[str],
    point_count: int,
    chunk_size: int,
    mp_config: MultiprocConfig,
) -> gpd.GeoDataFrame:
    """Load LAS/LAZ data by point chunks. Alias for :func:`_load_laspy_data_partitions`."""

    return _load_laspy_data_partitions(
        filename=filename,
        columns=columns,
        point_count=point_count,
        partition_size=chunk_size,
        mp_config=mp_config,
    )


def _load_laspy_data_spatial_partitions(
    filename: str | pathlib.Path,
    columns: list[str],
    block_bounds: list[tuple[float, float, float, float]],
    partition_size: int,
    mp_config: MultiprocConfig,
) -> list[gpd.GeoDataFrame]:
    """
    Load X/Y bounded LAS partitions through a multiprocessing cluster.

    This is efficient for COPC inputs because LasPy can use the COPC spatial
    index. For regular LAS/LAZ inputs each worker must stream-filter the source
    file for its bounds, so a one-pass splitter is generally preferable when
    many blocks are needed.
    """

    if partition_size <= 0:
        raise ValueError("Argument 'partition_size' must be a strictly positive integer.")

    # Each worker reads only the points intersecting one requested spatial block
    futures = [
        mp_config.cluster.submit(
            load_laspy_data_bounds,
            filename,
            columns,
            bounds,
            data_column="Z",
            chunk_size=partition_size,
        )
        for bounds in block_bounds
    ]
    # Preserve block order for callers that place results into a spatial grid
    return mp_config.cluster.gather(futures)


def _load_laspy_data_spatial_chunks(
    filename: str | pathlib.Path,
    columns: list[str],
    block_bounds: list[tuple[float, float, float, float]],
    chunk_size: int,
    mp_config: MultiprocConfig,
) -> list[gpd.GeoDataFrame]:
    """Load X/Y bounded LAS chunks. Alias for :func:`_load_laspy_data_spatial_partitions`."""

    return _load_laspy_data_spatial_partitions(
        filename=filename,
        columns=columns,
        block_bounds=block_bounds,
        partition_size=chunk_size,
        mp_config=mp_config,
    )


def _write_laspy_data_partitions(
    filename: str | pathlib.Path,
    pc: gpd.GeoDataFrame | pd.DataFrame,
    data_column: str | None,
    header: Any,
    mp_config: MultiprocConfig,
) -> None:
    """Write a dataframe to LAS/LAZ with multiprocessing partition files."""

    # Reuse the common temporary-file writer with the configuration's scalar chunk size
    write_laspy_multiproc_partitions(
        filename=filename,
        pc=pc,
        data_column=data_column,
        header=header,
        chunks=_point_partition_size(mp_config),
        cluster=mp_config.cluster,
    )


def _write_laspy_data_chunks(
    filename: str | pathlib.Path,
    pc: gpd.GeoDataFrame | pd.DataFrame,
    data_column: str | None,
    header: Any,
    mp_config: MultiprocConfig,
) -> None:
    """Write a dataframe to LAS/LAZ with multiprocessing chunk files."""

    _write_laspy_data_partitions(
        filename=filename,
        pc=pc,
        data_column=data_column,
        header=header,
        mp_config=mp_config,
    )
