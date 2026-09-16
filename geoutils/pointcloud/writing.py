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

"""Write saved dataframe partitions to point cloud files."""

from __future__ import annotations

import math
import os
import pathlib
import tempfile
from collections.abc import Iterable, Sequence
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

from geoutils.pointcloud.las import _write_laspy_multiproc_partitions

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.pointcloud.pointcloud import PointCloud


PointCloudDriver = Literal["GPKG", "LAS", "LAZ"]
_POINTCLOUD_FORMATS: dict[str, PointCloudDriver] = {".gpkg": "GPKG", ".las": "LAS", ".laz": "LAZ"}
_GPKG_WRITE_BATCH_ROWS = 65_536


def _stage_pointcloud_partition(dataframe: gpd.GeoDataFrame, filename: str | pathlib.Path) -> pathlib.Path:
    """Save one dataframe partition and return its normalized path for ordered file-based exchange."""

    output_filename = pathlib.Path(filename)
    dataframe.to_pickle(output_filename)
    return output_filename


def _resolve_pointcloud_output(
    filename: str | pathlib.Path,
    driver: str | None,
    *,
    supported_drivers: Sequence[PointCloudDriver],
    operation_name: str,
) -> tuple[pathlib.Path, PointCloudDriver]:
    """Resolve and validate a point cloud output path and driver for one operation."""

    # Infer the common GeoPackage default while recognizing formats selected by a filename suffix
    output_filename = pathlib.Path(filename)
    suffix = output_filename.suffix.lower()
    resolved_driver = driver.upper() if driver is not None else _POINTCLOUD_FORMATS.get(suffix, "GPKG")
    if resolved_driver not in supported_drivers:
        supported = ", ".join(repr(value) for value in supported_drivers)
        raise ValueError(f"Argument ``driver`` must be one of {supported} for {operation_name}.")

    # Require the suffix to identify the format because PointCloud uses it to choose specialized readers
    expected_driver = _POINTCLOUD_FORMATS.get(suffix)
    if suffix and expected_driver != resolved_driver:
        raise ValueError("Arguments ``driver`` and ``outfile`` must select the same supported point cloud format.")
    if not suffix and resolved_driver != "GPKG":
        raise ValueError("LAS and LAZ point cloud outputs require a matching filename suffix.")
    return output_filename, cast(PointCloudDriver, resolved_driver)


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


def _write_pointcloud_partitions(
    filename: str | pathlib.Path,
    partition_filenames: Iterable[str | pathlib.Path],
    *,
    driver: PointCloudDriver,
    data_column: str | None,
    geometry_type: Literal["Point", "Point Z"],
    mp_config: MultiprocConfig,
    las_header: Any | None = None,
    las_elevation_column: str | None = None,
) -> PointCloud:
    """
    Combine saved dataframe partitions into one file and return an unloaded PointCloud.

    The operation producing points saves each partition as a pickle so workers and the parent exchange only filenames.
    GeoPackage output combines small partitions before appending them. LAS/LAZ output asks the LAS writer to encode
    the partitions in workers and join their records into one file.
    """

    from geoutils.pointcloud.pointcloud import PointCloud

    partitions = iter(partition_filenames)
    try:
        first_filename = next(partitions)
    except StopIteration:
        raise ValueError("Point cloud output requires at least one saved partition.")
    output_filename = pathlib.Path(filename)
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    # Build the complete destination beside the requested path before replacing any existing file
    with tempfile.TemporaryDirectory(prefix=".geoutils-point-output-", dir=output_filename.parent) as directory:
        temporary_directory = pathlib.Path(directory)
        temporary_output = temporary_directory / f"output.{driver.lower()}"

        if driver == "GPKG":
            first = pd.read_pickle(first_filename)
            column_names = {str(column).lower() for column in first.columns}
            layer_options = {"FID": "fid", "GEOMETRY_NAME": "geom"}
            for option, field_name in layer_options.items():
                while field_name.lower() in column_names:
                    field_name = "_" + field_name
                layer_options[option] = field_name

            # Combine small partitions so scattered point samples do not require one file write per raster tile
            output_created = False
            pending_dataframes: list[gpd.GeoDataFrame] = []
            pending_rows = 0
            for index, partition_filename in enumerate(chain((first_filename,), partitions)):
                dataframe = first if index == 0 else pd.read_pickle(partition_filename)
                _check_gpkg_attributes(dataframe)
                pathlib.Path(partition_filename).unlink(missing_ok=True)

                # Write an empty first partition immediately so an all-empty result keeps its columns and geometry
                if len(dataframe) == 0:
                    if not output_created and not pending_dataframes:
                        pyogrio.write_dataframe(
                            dataframe,
                            temporary_output,
                            layer="points",
                            driver="GPKG",
                            append=False,
                            geometry_type=geometry_type,
                            layer_options=layer_options,
                        )
                        output_created = True
                    continue

                # Retain only a modest number of point rows before writing the next ordered file batch
                pending_dataframes.append(dataframe)
                pending_rows += len(dataframe)
                if pending_rows < _GPKG_WRITE_BATCH_ROWS:
                    continue

                combined = pending_dataframes[0] if len(pending_dataframes) == 1 else pd.concat(pending_dataframes)
                pyogrio.write_dataframe(
                    combined,
                    temporary_output,
                    layer="points",
                    driver="GPKG",
                    append=output_created,
                    layer_options=layer_options,
                )
                output_created = True
                pending_dataframes.clear()
                pending_rows = 0

            # Write the final short batch after every worker partition has been consumed
            if pending_dataframes:
                combined = pending_dataframes[0] if len(pending_dataframes) == 1 else pd.concat(pending_dataframes)
                pyogrio.write_dataframe(
                    combined,
                    temporary_output,
                    layer="points",
                    driver="GPKG",
                    append=output_created,
                    layer_options=layer_options,
                )
        else:
            if las_header is None:
                raise ValueError("LAS and LAZ point cloud output requires a shared LAS header.")

            # Let the LAS module encode and join the saved partitions without loading them in the parent
            chunk_size = mp_config.chunks if isinstance(mp_config.chunks, int) else math.prod(mp_config.chunks)
            _write_laspy_multiproc_partitions(
                filename=temporary_output,
                partitions=[first_filename, *partitions],
                data_column=las_elevation_column,
                header=las_header,
                chunk_size=chunk_size,
                cluster=mp_config.cluster,
                check_attributes=True,
            )

        os.replace(temporary_output, output_filename)

    return PointCloud(output_filename, data_column=data_column)
