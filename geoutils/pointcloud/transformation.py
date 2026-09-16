# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reproject point clouds in independent row partitions with file outputs."""

from __future__ import annotations

import pathlib
import tempfile
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import CRS

from geoutils._misc import import_optional
from geoutils._typing import NDArrayNum
from geoutils.pointcloud.las import (
    _as_geodataframe,
    _build_laspy_header,
    _is_laspy_supported,
    _load_laspy_data_slice,
    _point_partition_size,
)
from geoutils.pointcloud.writing import (
    _check_gpkg_attributes,
    _resolve_pointcloud_output,
    _stage_pointcloud_partition,
    _write_pointcloud_partitions,
)

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.pointcloud.pointcloud import PointCloud


############################################
# 1/ ROW PARTITION TRANSFORMATION
############################################


def _reproject_pointcloud_partition(
    source: gpd.GeoDataFrame | pathlib.Path,
    columns: list[str],
    start: int,
    count: int,
    crs: CRS,
    filename: pathlib.Path,
    las_output: bool,
) -> tuple[pathlib.Path, NDArrayNum | None]:
    """Read and reproject one row partition, saving its dataframe and returning its LAS coordinate bounds."""

    # Read independent row ranges so unloaded sources stay outside the parent process
    if isinstance(source, pathlib.Path):
        if _is_laspy_supported(source):
            dataframe = _load_laspy_data_slice(source, columns=columns, start=start, count=count)
        else:
            dataframe = pyogrio.read_dataframe(source, skip_features=start, max_features=count)
    else:
        dataframe = source
    projected = dataframe.to_crs(crs)

    # Use actual elevations for LAS, independently of the selected point value column
    bounds = None
    if las_output and len(projected) > 0:
        if projected.geometry.has_z.all():
            elevation = projected.geometry.z.to_numpy()
            if "Z" in projected.columns:
                if not np.array_equal(projected["Z"].to_numpy(), elevation):
                    raise ValueError("LAS output cannot store different values in geometry Z and column 'Z'.")
                projected = projected.drop(columns="Z")
        elif "Z" in projected.columns:
            elevation = projected["Z"].to_numpy()
        else:
            raise ValueError("LAS output requires 3D point geometry or a native elevation column named 'Z'.")

        # Derive one coordinate encoding from every projected partition before writing LAS records
        coordinates = np.column_stack((projected.geometry.x, projected.geometry.y, elevation))
        if not np.isfinite(coordinates).all():
            raise ValueError("LAS output requires finite X, Y and Z coordinates.")
        bounds = np.stack((coordinates.min(axis=0), coordinates.max(axis=0)))

    # Preserve all dataframe dtypes until the final format is selected
    return _stage_pointcloud_partition(projected, filename), bounds


############################################
# 2/ ORDERED OUTPUT CONSTRUCTION
############################################


def _reproject_las_header(
    dataframe: gpd.GeoDataFrame,
    bounds: list[NDArrayNum],
    crs: CRS,
    source_filename: pathlib.Path | None,
) -> Any:
    """Build a shared LAS schema and coordinate encoding from projected bounds and source dimensions."""

    # Keep native source dimensions and elevation precision when a LAS header is available
    source_header = None
    if source_filename is not None and _is_laspy_supported(source_filename) and source_filename.exists():
        laspy = import_optional("laspy")
        with laspy.open(source_filename) as reader:
            source_header = reader.header.copy()
    scales = np.array([1e-8, 1e-8, 1e-3] if crs.is_geographic else [1e-3, 1e-3, 1e-3])
    unchanged_axes = [2]
    if source_header is not None:
        scales[2] = source_header.scales[2]
        if source_header.parse_crs() == crs:
            scales = source_header.scales.copy()
            unchanged_axes = [0, 1, 2]

    # Center integer coordinates on the complete output extent and enlarge scales only to avoid overflow
    offsets = np.zeros(3)
    if bounds:
        minimum = np.min([part[0] for part in bounds], axis=0)
        maximum = np.max([part[1] for part in bounds], axis=0)
        offsets = minimum + (maximum - minimum) / 2
        required_scales = (maximum - minimum) / (2 * (np.iinfo(np.int32).max - 1))
        scales = np.maximum(scales, required_scales)
    if source_header is not None:
        # Keep unchanged coordinates on their original integer lattice whenever their current extent fits
        for axis in unchanged_axes:
            original_offset = source_header.offsets[axis]
            original_scale = source_header.scales[axis]
            if not bounds:
                offsets[axis] = original_offset
                scales[axis] = original_scale
                continue
            extrema = np.array([minimum[axis], maximum[axis]])
            encoded_extrema = np.rint((extrema - original_offset) / original_scale)
            if np.all(encoded_extrema >= np.iinfo(np.int32).min) and np.all(encoded_extrema <= np.iinfo(np.int32).max):
                offsets[axis] = original_offset
                scales[axis] = original_scale

    # Reuse the common writer schema, with native Z chosen from geometry or the LAS elevation column
    elevation_column = "Z" if "Z" in dataframe.columns else None
    return _build_laspy_header(
        dataframe,
        data_column=elevation_column,
        version=None if source_header is None else source_header.version,
        point_format=None if source_header is None else source_header.point_format,
        offsets=tuple(offsets),
        scales=tuple(scales),
        crs=crs,
    )


############################################
# 3/ MULTIPROCESSING REPROJECTION
############################################


def _reproject_pointcloud(source: PointCloudBase, crs: CRS, mp_config: MultiprocConfig) -> PointCloud:
    """
    Reproject independent row partitions and return an unopened point cloud at the configured output path.

    _reproject_pointcloud_partition() reads source slices or receives eager rows, applies GeoPandas to_crs(), and
    saves exact projected dataframes to temporary files.
    _reproject_las_header() chooses common scales and offsets, then _write_pointcloud_partitions() appends or
     encodes every format in source order.
    """

    # Validate configuration before inspecting point records or creating output files
    chunks = _point_partition_size(mp_config)
    if chunks <= 0:
        raise ValueError("Argument ``chunks`` must be a strictly positive integer.")
    if source._is_dask:
        raise ValueError("Argument ``mp_config`` cannot be combined with a Dask point cloud.")
    output_filename, driver = _resolve_pointcloud_output(
        mp_config.outfile,
        mp_config.driver,
        supported_drivers=("GPKG", "LAS", "LAZ"),
        operation_name="point cloud reprojection",
    )
    target_crs = CRS.from_user_input(crs)

    # Plan slices from source metadata without loading an unopened point cloud
    source_filename = pathlib.Path(source.name) if not source._is_pd and source.name is not None else None
    if not source.is_loaded:
        if source_filename is None or (
            not _is_laspy_supported(source_filename) and pyogrio.read_info(source_filename)["driver"] != "GPKG"
        ):
            raise ValueError("Unloaded point cloud reprojection supports LAS, LAZ and GPKG sources.")
        dataframe = None
    else:
        dataframe = _as_geodataframe(source.ds, crs=source.crs)
        if driver == "GPKG":
            _check_gpkg_attributes(dataframe)
    columns = list(source._nongeo_columns)
    point_count = source.point_count

    # Complete all source reads in temporary files before replacing any existing destination
    with tempfile.TemporaryDirectory(prefix=".geoutils-reproject-", dir=output_filename.parent) as directory:
        temporary_directory = pathlib.Path(directory)
        futures = []
        for index, start in enumerate(range(0, max(point_count, 1), chunks)):
            count = min(chunks, point_count - start)
            partition_source = source_filename if dataframe is None else dataframe.iloc[start : start + count]
            futures.append(
                mp_config.cluster.submit(
                    _reproject_pointcloud_partition,
                    partition_source,
                    columns,
                    start,
                    count,
                    target_crs,
                    temporary_directory / f"projected_{index}.pkl",
                    driver != "GPKG",
                )
            )
        projected_parts = mp_config.cluster.gather(futures)

        # Prepare one shared LAS coordinate encoding; GeoPackage needs no format-specific metadata
        las_header = None
        elevation_column = None
        if driver != "GPKG":
            projected = pd.read_pickle(projected_parts[0][0])
            bounds = [bounds for _, bounds in projected_parts if bounds is not None]
            las_header = _reproject_las_header(projected, bounds, target_crs, source_filename)
            elevation_column = "Z" if "Z" in projected.columns else None

        # Write the saved partitions to one point cloud in their existing row order
        # Build the result in a temporary file so readers never see a partly written output
        return _write_pointcloud_partitions(
            output_filename,
            [filename for filename, _ in projected_parts],
            driver=driver,
            data_column=source.data_column,
            geometry_type="Point Z" if source._has_z else "Point",
            mp_config=mp_config,
            las_header=las_header,
            las_elevation_column=elevation_column,
        )
