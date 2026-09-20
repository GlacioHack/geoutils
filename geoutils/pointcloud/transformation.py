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

"""Transformations for point clouds."""

from __future__ import annotations

import pathlib
import tempfile
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from geoutils._dispatch import _clip_geometry, _get_reproject_crs
from geoutils._misc import import_optional
from geoutils._typing import NDArrayNum
from geoutils.pointcloud.dataframe import _build_pointcloud_output, _get_dataframe_attrs
from geoutils.pointcloud.las import (
    _as_geodataframe,
    _build_laspy_header,
    _is_laspy_supported,
    _load_laspy_data_slice,
    _load_laspy_metadata,
    _point_partition_size,
)
from geoutils.pointcloud.writing import (
    _check_gpkg_attributes,
    _resolve_pointcloud_output,
    _stage_pointcloud_partition,
    _write_pointcloud_partitions,
)
from geoutils.vector.transformation import (
    _apply_crop_filters,
    _clip,
    _clip_geodataframe,
    _reproject,
)

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.pointcloud.pointcloud import PointCloud
    from geoutils.raster.base import RasterLike
    from geoutils.vector.base import VectorLike


#####################
# 1/ SHARED HELPERS
#####################


def _prepare_las_partition(dataframe: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, NDArrayNum]:
    """Prepare one non-empty point partition for LAS output and return its XYZ bounds."""

    # Use geometry Z when present, and remove a duplicate Z column after checking that its values agree
    if dataframe.geometry.has_z.all():
        elevation = dataframe.geometry.z.to_numpy()
        if "Z" in dataframe.columns:
            if not np.array_equal(dataframe["Z"].to_numpy(), elevation):
                raise ValueError("LAS output cannot store different values in geometry Z and column 'Z'.")
            dataframe = dataframe.drop(columns="Z")
    elif "Z" in dataframe.columns:
        elevation = dataframe["Z"].to_numpy()
    else:
        raise ValueError("LAS output requires 3D point geometry or a native elevation column named 'Z'.")

    # Check coordinates before encoding them as scaled LAS integers, then summarize the range for the shared header
    coordinates = np.column_stack((dataframe.geometry.x, dataframe.geometry.y, elevation))
    if not np.isfinite(coordinates).all():
        raise ValueError("LAS output requires finite X, Y and Z coordinates.")
    bounds = np.stack((coordinates.min(axis=0), coordinates.max(axis=0)))
    return dataframe, bounds


def _cast_multiproc_output(source: PointCloudBase, output: PointCloud) -> PointCloud | gpd.GeoDataFrame:
    """Return an unloaded object or load every output column for a dataframe accessor."""

    if not source._is_pd:
        return output

    # Accessor results are dataframes, so read every stored attribute before discarding the file-backed wrapper
    output.load(columns="all")
    return _build_pointcloud_output(
        output.ds,
        data_column=output.data_column,
        as_dataframe=True,
        attrs=_get_dataframe_attrs(source.ds),
    )


##############
# 2/ REPROJECT
##############

# Eager execution


def _reproject_pointcloud_eager(
    source: PointCloudBase,
    ref: RasterLike | VectorLike | None,
    crs: CRS | str | int | None,
    inplace: bool,
) -> PointCloud | gpd.GeoDataFrame | None:
    """Reproject an eager point cloud and optionally update its dataframe in place."""

    projected = _reproject(source, ref=ref, crs=crs)
    if inplace:
        source.ds = projected
        return None
    return source._override_gdf_output(projected)


def _reproject_pointcloud_dask(
    source: PointCloudBase,
    ref: RasterLike | VectorLike | None,
    crs: CRS | str | int | None,
    inplace: bool,
) -> Any:
    """Build a lazy Dask reprojection graph without computing point partitions."""

    if inplace:
        raise ValueError("Dask-backed point clouds cannot be modified in place; use the returned dataframe instead.")
    projected = _reproject(source, ref=ref, crs=crs)
    return source._override_gdf_output(projected)


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
        projected, bounds = _prepare_las_partition(projected)

    # Preserve all dataframe dtypes until the final format is selected
    return _stage_pointcloud_partition(projected, filename), bounds


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


def _reproject_pointcloud_multiproc(
    source: PointCloudBase,
    ref: RasterLike | VectorLike | None,
    crs: CRS | str | int | None,
    inplace: bool,
    mp_config: MultiprocConfig,
) -> PointCloud | gpd.GeoDataFrame:
    """
    Reproject independent row partitions and return the matching object or dataframe interface.

    Internal logic for Multiproc here is the most complex:
    _reproject_pointcloud_partition() reads source slices or receives eager rows, applies GeoPandas to_crs(), and
    saves exact projected dataframes to temporary files.
    _reproject_las_header() chooses common scales and offsets, then _write_pointcloud_partitions() appends or
     encodes every format in source order.
    """

    # 1/ Validate inputs and prepare source partitions
    if source._is_dask:
        raise ValueError("Argument ``mp_config`` cannot be combined with a Dask point cloud.")
    if inplace:
        raise ValueError("Argument ``inplace`` is not supported with ``mp_config``; use the returned point cloud.")
    target_crs = _get_reproject_crs(ref=ref, crs=crs)
    chunks = _point_partition_size(mp_config)
    if chunks <= 0:
        raise ValueError("Argument ``chunks`` must be a strictly positive integer.")
    output_filename, driver = _resolve_pointcloud_output(
        mp_config.outfile,
        mp_config.driver,
        supported_drivers=("GPKG", "LAS", "LAZ"),
        operation_name="point cloud reprojection",
    )

    # Plan slices from source metadata without loading an unopened point cloud
    source_filename = pathlib.Path(source.name) if not source._is_pd and source.name is not None else None
    if not source.is_loaded:
        if getattr(source, "_downsample", 1) != 1:
            raise ValueError("Load a downsampled point cloud before using multiprocessing clip() to preserve its rows.")
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

    # 2/ Reproject and save every partition with the worker pool

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

        # 3/ Prepare shared output metadata and write the saved partitions

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
        return _cast_multiproc_output(
            source,
            _write_pointcloud_partitions(
                output_filename,
                [filename for filename, _ in projected_parts],
                driver=driver,
                data_column=source.data_column,
                geometry_type="Point Z" if source._has_z else "Point",
                las_header=las_header,
                las_elevation_column=elevation_column,
            ),
        )


# Backend dispatcher


def _reproject_pointcloud(
    source: PointCloudBase,
    ref: RasterLike | VectorLike | None,
    crs: CRS | str | int | None,
    inplace: bool,
    mp_config: MultiprocConfig | None,
) -> Any:
    """Dispatch point cloud reprojection to eager, Dask or multiprocessing execution."""

    if mp_config is not None:
        return _reproject_pointcloud_multiproc(source, ref, crs, inplace, mp_config)
    if source._is_dask:
        return _reproject_pointcloud_dask(source, ref, crs, inplace)
    return _reproject_pointcloud_eager(source, ref, crs, inplace)


#########
# 3/ CLIP
#########


def _clip_pointcloud_eager(
    source: PointCloudBase,
    mask: Any,
    keep_geom_type: bool,
    sort: bool,
) -> PointCloud | gpd.GeoDataFrame:
    """Clip an eager point cloud and return the matching object or dataframe interface."""

    clipped = _clip(source, mask=mask, keep_geom_type=keep_geom_type, sort=sort)
    return source._override_gdf_output(clipped)


def _clip_pointcloud_dask(
    source: PointCloudBase,
    mask: Any,
    keep_geom_type: bool,
    sort: bool,
) -> Any:
    """Build a lazy Dask clipping graph without computing point partitions."""

    clipped = _clip(source, mask=mask, keep_geom_type=keep_geom_type, sort=sort)
    return source._override_gdf_output(clipped)


def _clip_pointcloud_partition(
    source: gpd.GeoDataFrame | pathlib.Path,
    columns: list[str],
    start: int,
    count: int,
    crop_filters: list[tuple[tuple[float, float, float, float], Literal["intersects", "within"]]],
    geometry: BaseGeometry,
    keep_geom_type: bool,
    sort: bool,
    filename: pathlib.Path,
    las_output: bool,
) -> tuple[pathlib.Path, NDArrayNum | None]:
    """Read and clip one point row partition, saving its dataframe and LAS coordinate bounds."""

    # Read independent row ranges so unloaded LAS, LAZ and GeoPackage sources stay outside the parent process
    if isinstance(source, pathlib.Path):
        if _is_laspy_supported(source):
            dataframe = _load_laspy_data_slice(source, columns=columns, start=start, count=count)
        else:
            dataframe = pyogrio.read_dataframe(source, skip_features=start, max_features=count)
        dataframe = _apply_crop_filters(dataframe, crop_filters)
    else:
        dataframe = source

    # Apply the exact geometry after any earlier bounding box crops, without changing the selected point values
    clipped = _clip_geodataframe(
        dataframe,
        geometry=geometry,
        keep_geom_type=keep_geom_type,
        sort=sort,
    )

    # Prepare LAS coordinates and bounds now so writing needs no second scan of this partition later
    bounds = None
    if las_output and len(clipped) > 0:
        clipped, bounds = _prepare_las_partition(clipped)

    # Save each result separately so the parent can write partitions in source order
    return _stage_pointcloud_partition(clipped, filename), bounds


def _clip_pointcloud_multiproc(
    source: PointCloudBase,
    mask: Any,
    keep_geom_type: bool,
    sort: bool,
    mp_config: MultiprocConfig,
) -> PointCloud | gpd.GeoDataFrame:
    """
    Clip independent point row partitions and return the PointCloud object or dataframe interface.

    Internal logic for Multiproc here is the most complex:
    _clip_pointcloud_partition() reads and clips each source range, then stages exact dataframes and optional LAS
    bounds (i.e., writes them to tempfiles).
    _write_pointcloud_partitions() appends those results in source order without collecting all rows.
    """

    # 1/ Validate inputs and prepare source partitions
    if source._is_dask:
        raise ValueError("Argument ``mp_config`` cannot be combined with a Dask point cloud.")
    chunks = _point_partition_size(mp_config)
    if chunks <= 0:
        raise ValueError("Argument ``chunks`` must be a strictly positive integer.")
    output_filename, driver = _resolve_pointcloud_output(
        mp_config.outfile,
        mp_config.driver,
        supported_drivers=("GPKG", "LAS", "LAZ"),
        operation_name="point cloud clipping",
    )

    # Normalize the clipping geometry once before workers filter source partitions in the point cloud CRS
    target_crs = None if source.crs is None else CRS.from_user_input(source.crs)
    geometry = _clip_geometry(mask, target_crs=target_crs)

    # Plan slices from source metadata without loading an unopened point cloud
    source_filename = pathlib.Path(source.name) if not source._is_pd and source.name is not None else None
    if not source.is_loaded:
        if source_filename is None or (
            not _is_laspy_supported(source_filename) and pyogrio.read_info(source_filename)["driver"] != "GPKG"
        ):
            raise ValueError("Unloaded point cloud clipping supports LAS, LAZ and GPKG sources.")
        dataframe = None
    else:
        dataframe = _as_geodataframe(source.ds, crs=source.crs)
        if driver == "GPKG":
            _check_gpkg_attributes(dataframe)
    columns = list(source._nongeo_columns)
    if dataframe is not None:
        point_count = len(dataframe)
    elif source_filename is not None and _is_laspy_supported(source_filename):
        point_count = _load_laspy_metadata(source_filename).point_count
    else:
        assert source_filename is not None
        point_count = int(pyogrio.read_info(source_filename, force_feature_count=True)["features"])
        if point_count < 0:
            raise RuntimeError("Could not determine the number of points from the file metadata.")
    crop_filters = list(getattr(source, "_crop_filters", []))

    # 2/ Clip and save every partition with the worker pool

    # Complete all source reads in temporary files before replacing any existing destination
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".geoutils-clip-", dir=output_filename.parent) as directory:
        temporary_directory = pathlib.Path(directory)
        futures = []
        for index, start in enumerate(range(0, max(point_count, 1), chunks)):
            count = min(chunks, point_count - start)
            partition_source = source_filename if dataframe is None else dataframe.iloc[start : start + count]
            futures.append(
                mp_config.cluster.submit(
                    _clip_pointcloud_partition,
                    partition_source,
                    columns,
                    start,
                    count,
                    crop_filters,
                    geometry,
                    keep_geom_type,
                    sort,
                    temporary_directory / f"clipped_{index}.pkl",
                    driver != "GPKG",
                )
            )
        clipped_parts = mp_config.cluster.gather(futures)

        # 3/ Prepare shared output metadata and write the saved partitions

        # Reuse the original LAS schema when possible; otherwise derive one encoding from worker bounds
        las_header = None
        elevation_column = None
        partition_filenames = [filename for filename, _ in clipped_parts]
        if driver != "GPKG":
            first = pd.read_pickle(partition_filenames[0])
            elevation_column = "Z" if "Z" in first.columns else None
            if dataframe is None and source_filename is not None and _is_laspy_supported(source_filename):
                laspy = import_optional("laspy")
                with laspy.open(source_filename) as reader:
                    las_header = reader.header.copy()

        # Write each saved partition in source order, loading the result only for a dataframe accessor
        return _cast_multiproc_output(
            source,
            _write_pointcloud_partitions(
                output_filename,
                partition_filenames,
                driver=driver,
                data_column=source.data_column,
                geometry_type="Point Z" if source._has_z else "Point",
                las_header=las_header,
                las_elevation_column=elevation_column,
                las_bounds=[bounds for _, bounds in clipped_parts],
            ),
        )


# Backend dispatcher


def _clip_pointcloud(
    source: PointCloudBase,
    mask: Any,
    keep_geom_type: bool,
    sort: bool,
    mp_config: MultiprocConfig | None,
) -> Any:
    """Dispatch point cloud clipping to eager, Dask or multiprocessing execution."""

    if mp_config is not None:
        return _clip_pointcloud_multiproc(source, mask, keep_geom_type, sort, mp_config)
    if source._is_dask:
        return _clip_pointcloud_dask(source, mask, keep_geom_type, sort)
    return _clip_pointcloud_eager(source, mask, keep_geom_type, sort)
