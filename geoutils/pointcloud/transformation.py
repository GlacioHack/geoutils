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

import os
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
    _dataframe_to_lasdata,
    _is_laspy_supported,
    _load_laspy_data_slice,
    _point_partition_size,
    _stitch_laspy_files,
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
    """Read and reproject one row partition, staging its exact dataframe and LAS coordinate bounds."""

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

    # Preserve all dataframe dtypes until the final format is selected; only paths return to the parent
    projected.to_pickle(filename)
    return filename, bounds


############################################
# 2/ ORDERED OUTPUT CONSTRUCTION
############################################


def _check_gpkg_attributes(dataframe: gpd.GeoDataFrame) -> None:
    """Reject attributes that GeoPackage storage or its dataframe reader would round."""

    for name in dataframe.columns:
        if name == dataframe.geometry.name:
            continue
        values = dataframe[name]

        # GeoPackage timestamps store milliseconds, so finer source times cannot round-trip exactly
        if pd.api.types.is_datetime64_any_dtype(values.dtype):
            submillisecond = (values.dt.microsecond % 1000 != 0) | (values.dt.nanosecond != 0)
            if (values.notna() & submillisecond).any():
                raise ValueError(f"GeoPackage cannot preserve submillisecond timestamps in column {name!r}.")

        # GeoPandas reads integer columns containing nulls as float64, even when nulls occur in another partition
        if pd.api.types.is_integer_dtype(values.dtype) and values.hasnans:
            valid = values.dropna()
            try:
                restored = valid.astype(np.float64).astype(valid.dtype)
                exact = restored.equals(valid)
            except (TypeError, ValueError, OverflowError):
                exact = False
            if not exact:
                raise ValueError(f"GeoPackage cannot preserve nullable integer values in column {name!r}.")


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


def _write_reprojected_las_partition(filename: pathlib.Path, output_filename: pathlib.Path, header: Any) -> str:
    """Encode a projected partition and reject attribute values changed by its LAS dimension types."""

    # Encode one partition with the common writer's conversion before publishing any point records
    dataframe = pd.read_pickle(filename)
    elevation_column = "Z" if "Z" in dataframe.columns else None
    try:
        encoded = _dataframe_to_lasdata(dataframe, data_column=elevation_column, header=header)
    except OverflowError as error:
        raise ValueError("LAS output cannot preserve point attributes with the selected dimension types.") from error

    # Check the encoded attributes for integer truncation, overflow and scaled dimension rounding
    columns = [column for column in dataframe.columns if column not in (dataframe.geometry.name, "Z")]
    for column in columns:
        expected_values = dataframe[column].to_numpy()
        encoded_values = np.asarray(encoded[column])

        # Compare Python scalars so NumPy cannot round large integer values while promoting mixed numeric dtypes
        equal_values = expected_values.astype(object) == encoded_values.astype(object)
        equal_values |= pd.isna(expected_values) & pd.isna(encoded_values)
        if not np.all(equal_values):
            raise ValueError(f"LAS output cannot preserve the values in column {column!r} with its dimension type.")
    encoded.write(output_filename)
    return os.fspath(output_filename)


############################################
# 3/ MULTIPROCESSING REPROJECTION
############################################


def _reproject_pointcloud(source: PointCloudBase, crs: CRS, mp_config: MultiprocConfig) -> PointCloud:
    """
    Reproject independent row partitions and return an unopened point cloud at the configured output path.

    _reproject_pointcloud_partition() reads source slices or receives eager rows, applies GeoPandas to_crs(), and
    stages exact projected dataframes. GPKG output appends these partitions in source order. LAS output first uses
    _reproject_las_header() to choose common scales and offsets, then _write_reprojected_las_partition() and
    _stitch_laspy_files() encode and stream the rows. Only paths and coordinate bounds are gathered in the parent.
    Output row order and attribute columns follow the source; reopened indices follow the destination format.
    """

    from geoutils.pointcloud.pointcloud import PointCloud

    # Validate configuration before inspecting point records or creating output files
    chunks = _point_partition_size(mp_config)
    if chunks <= 0:
        raise ValueError("Argument ``chunks`` must be a strictly positive integer.")
    if source._is_dask:
        raise ValueError("Argument ``mp_config`` cannot be combined with a Dask point cloud.")
    output_filename = pathlib.Path(mp_config.outfile)
    suffix = output_filename.suffix.lower()
    formats = {".las": "LAS", ".laz": "LAZ", ".gpkg": "GPKG"}
    driver = mp_config.driver.upper() if mp_config.driver is not None else formats.get(suffix, "GPKG")
    if driver not in formats.values():
        raise ValueError("Argument ``driver`` must be 'GPKG', 'LAS' or 'LAZ' for point cloud reprojection.")
    if (suffix and (suffix not in formats or formats[suffix] != driver)) or (not suffix and driver != "GPKG"):
        raise ValueError("Arguments ``driver`` and ``outfile`` must select the same supported point cloud format.")
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
        temporary_output = temporary_directory / f"output.{driver.lower()}"

        # Stream the exact GPKG geometry and attributes in the original point order
        if driver == "GPKG":
            # Reserve internal fields independently of user columns such as 'fid' and 'geom'
            column_names = {column.lower() for column in columns}
            layer_options = {"FID": "fid", "GEOMETRY_NAME": "geom"}
            for option, field_name in layer_options.items():
                while field_name.lower() in column_names:
                    field_name = "_" + field_name
                layer_options[option] = field_name
            for index, (filename, _) in enumerate(projected_parts):
                projected = pd.read_pickle(filename)
                geometry_type = None
                if len(projected) == 0:
                    geometry_type = "Point Z" if source._has_z else "Point"
                pyogrio.write_dataframe(
                    projected,
                    temporary_output,
                    layer="points",
                    driver="GPKG",
                    append=index > 0,
                    geometry_type=geometry_type,
                    layer_options=layer_options,
                )
        else:
            # Encode every LAS worker file using the same global bounds and native dimension schema
            projected = pd.read_pickle(projected_parts[0][0])
            bounds = [bounds for _, bounds in projected_parts if bounds is not None]
            header = _reproject_las_header(projected, bounds, target_crs, source_filename)
            del projected
            futures = []
            for index, (filename, _) in enumerate(projected_parts):
                futures.append(
                    mp_config.cluster.submit(
                        _write_reprojected_las_partition,
                        filename,
                        temporary_directory / f"encoded_{index}.las",
                        header,
                    )
                )
            written_paths = mp_config.cluster.gather(futures)
            _stitch_laspy_files(temporary_output, written_paths, header=header, chunk_size=chunks)

        # Publish only the completed file so a source and destination may safely refer to the same path
        os.replace(temporary_output, output_filename)

    return PointCloud(output_filename, data_column=source.data_column)
