# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read/write point cloud files, with optional downsampling, from local or remote files."""

from __future__ import annotations

import os
import pathlib
import tempfile
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio as rio
import xarray as xr
from pyproj import CRS
from rasterio.coords import BoundingBox
from xarray.backends import BackendArray
from xarray.core import indexing

from geoutils._config import config
from geoutils._dispatch import is_dask_dataframe
from geoutils._io import _is_remote_path, _path_suffix, _resolve_filesystem
from geoutils._misc import _deprecate_keyword, _validate_downsample
from geoutils._typing import Number
from geoutils.pointcloud.dataframe import _build_pointcloud_output, _get_dataframe_attrs, _set_dataframe_attrs
from geoutils.pointcloud.las import _build_laspy_header_from_partitions, _write_laspy_saved_partitions
from geoutils.pointcloud.parquet import _is_parquet_supported

if TYPE_CHECKING:
    from laspy.copc import Entry

    from geoutils.pointcloud.base import GeometryPointCloudBase
    from geoutils.pointcloud.pointcloud import PointCloud

##############
# 1/ OPENING
##############


class PointCloudDownsampleOptions(TypedDict, total=False):
    """Opening options specifying either a point count reduction factor or a maximum sample size."""

    factor: Number
    size: int


def _check_input_pointcloud_downsample(
    downsample: Number | PointCloudDownsampleOptions,
) -> float | PointCloudDownsampleOptions:
    """Check downsampling input for a point cloud."""

    # Either a downsampling factor, or an exact size
    if not isinstance(downsample, dict):
        return _validate_downsample(downsample)
    if set(downsample) not in ({"factor"}, {"size"}):
        raise ValueError("downsample must specify exactly one of factor or size.")
    if "factor" in downsample:
        return _validate_downsample(downsample["factor"])

    # Copy the size (so later edits to the input cannot change deferred I/O)
    size = downsample["size"]
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("downsample size must be a positive integer.")
    return {"size": int(size)}


def _pointcloud_downsample_count(point_count: int, downsample: float | PointCloudDownsampleOptions) -> int:
    """Calculate the downsampled point count, without exceeding the source point count."""

    if isinstance(downsample, dict):
        return min(point_count, downsample["size"])
    return int(np.ceil(point_count / downsample))


def _check_input_pointcloud_level(level: int | range | None, downsample: float | PointCloudDownsampleOptions) -> None:
    """Check COPC levels and their compatibility with downsampling."""

    if level is None:
        return

    # We match LasPy selection: single level or range
    if isinstance(level, range):
        valid_level = level.start >= 0 and level.step > 0 and len(level) > 0
    else:
        valid_level = not isinstance(level, (bool, np.bool_)) and isinstance(level, (int, np.integer)) and level >= 0
    if not valid_level:
        raise ValueError("Argument 'level' must be a non-negative integer or a non-empty increasing range of levels.")
    if downsample != 1:
        raise ValueError("downsample and level cannot be used together.")


@_deprecate_keyword("data_column", "data_name")
def open_pointcloud(
    filename: str | pathlib.Path,
    data_name: str | None = None,
    columns: Literal["all", "main"] | list[str] = "main",
    chunks: int | None = None,
    downsample: Number | PointCloudDownsampleOptions = 1,
    *,
    as_object_type: Literal["dataarray", "dataset", "geodataframe"] | None = None,
    level: int | range | None = None,
    load: bool = False,
    parse_sensor_metadata: bool = False,
    filesystem: Any = None,
    storage_options: dict[str, Any] | None = None,
) -> xr.DataArray | xr.Dataset | gpd.GeoDataFrame | Any:
    """
    Open a point cloud as a Xarray DataArray, Dataset or GeoDataFrame.

    LAS, LAZ and COPC files are read with LasPy, GeoParquet files and partition directories with PyArrow, and other
    point vector formats with GeoPandas.

    Passing ``chunks`` opens the point data lazily using Dask arrays (Xarray objects) or Dask-GeoPandas (Pandas
    object).

    Use ``downsample`` to select a random subsample, interpreted as a downsampling factor from total point size. For
    example, ``downsample=2`` returns half the points. Alternatively, select an exact size with ``downsample={"size":
    1000}``. With a COPC file, only necessary levels of details (overview-equivalent of point clouds) are loaded
    from coarse to fine until they contain enough points, then downsampled to the requested size.

    Use ``level`` to select COPC levels of detail explicitly. Each point belongs to one level:
    level 0 holds a coarse sample, with higher levels adding more and more points. Selected levels are combined into
    one point cloud. An integer selects only that level, ``level=range(3)`` combines levels 0 through 2. Thus,
    ``level`` cannot be combined with ``downsample``.

    See :ref:`scalability-downsampling` for examples on downsampling and overviews.

    To open a remote file, pass an HTTPS URL or S3 address, such as ``"https://example.org/data.copc.laz"`` or
    ``"s3://bucket/data.copc.laz"``.

    Use ``storage_options`` to pass a dictionary of connection settings, such as credentials or a custom
    server address. GeoUtils creates a Fsspec filesystem for you, which handles access to files on that service.
    For example, ``storage_options={"anon": True}`` reads a public S3 bucket without login credentials.

    Use ``filesystem`` if you already have a Fsspec filesystem object and want to reuse its connection
    settings across files. Both options are optional and cannot be combined.

    These settings are preserved for deferred and chunked reads.
    For vector formats, these options require the optional Fiona backend.
    See :ref:`scalability-formats` for examples on supported formats and reading files in the cloud.

    :param filename: Point file or directory of GeoParquet partitions.
    :param data_name: Attribute containing point values. Defaults to Z for LAS, the saved active attribute for
        GeoParquet, or geometry elevations when no attribute is selected in other vector formats.
    :param columns: Attributes to read from LAS or GeoParquet: main selects the active values, all selects every
        attribute, and a list selects named attributes as well as the active values. Ignored for other vector formats.
    :param chunks: Points per Dask chunk or partition. GeoParquet GeoDataFrame partitions follow the stored row
        groups. None uses deferred file reads unless load=True.
    :param downsample: A downsampling factor, or a dictionary containing exactly one of "factor" or "size". For example,
        passing 2 is equivalent to {"factor": 2} and selects ceil(N / 2) points. Passing {"size": 1000} reads
        exactly 1000 maximum points. Default 1 reads all points.
    :param as_object_type: Returned representation: dataarray, dataset, or geodataframe. None uses
        config["pointcloud_return_type"], which defaults to dataarray.
    :param level: COPC level of detail, starting at 0 for the coarsest, or a range of levels such as range(3). The
        default None uses all levels, loading the full point cloud.
    :param load: Whether to read values immediately or not. The default False defers reads but requires using Xarray
        as object type, a GeoDataFrame requires Dask-GeoPandas.
    :param parse_sensor_metadata: Parse sensor metadata from the filename and add it to the returned attributes.
    :param filesystem: An existing Fsspec filesystem object whose connection settings are used to open files.
        Cannot be combined with storage_options.
    :param storage_options: Dictionary of connection settings used to create a Fsspec filesystem from the file URL,
        such as credentials or a custom server address. Cannot be combined with filesystem.

    :returns: A DataArray with active values and auxiliary coordinates, a Dataset with separate value variables,
        or a GeoDataFrame.
    """
    # 1/ Input checks
    if as_object_type is None:
        as_object_type = config["pointcloud_return_type"]
    downsample = _check_input_pointcloud_downsample(downsample)
    if chunks is not None and (isinstance(chunks, bool) or not isinstance(chunks, int) or chunks <= 0):
        raise ValueError("Argument 'chunks' must be a strictly positive integer.")
    if as_object_type not in ("dataarray", "dataset", "geodataframe"):
        raise ValueError("Argument 'as_object_type' must be 'dataarray', 'dataset' or 'geodataframe'.")
    _check_input_pointcloud_level(level, downsample)

    filename, filesystem = _resolve_filesystem(filename, filesystem, storage_options, remote_default=False)

    # Plan COPC reads from hierarchy counts without loading point records
    from geoutils.pointcloud.las import (
        _is_laspy_supported,
        _load_laspy_metadata,
        _read_copc_hierarchy,
        _select_copc_nodes,
    )

    # Store levels, chunk sizes, and other relevant metadata in the object attributes
    copc_nodes = None
    file_metadata: dict[str, Any] = {}
    source_count = None
    if _is_laspy_supported(filename):
        metadata = _load_laspy_metadata(filename, filesystem=filesystem)
        file_metadata = {"point_record_size": metadata.point_record_size, "laz_chunk_size": metadata.laz_chunk_size}
        if metadata.is_copc:
            source_count = metadata.point_count
            target_count = _pointcloud_downsample_count(source_count, downsample) if downsample != 1 else None
            hierarchy = _read_copc_hierarchy(str(filename), filesystem=filesystem)
            copc_nodes = _select_copc_nodes(str(filename), level, target_count, nodes=hierarchy, filesystem=filesystem)
            file_metadata.update(
                {
                    "copc_levels": sorted({node.key.level for node in hierarchy}),
                    "copc_selected_levels": sorted({node.key.level for node in copc_nodes}),
                    "copc_chunk_sizes": [node.point_count for node in hierarchy],
                }
            )
    if level is not None and copc_nodes is None:
        raise ValueError("Argument 'level' is only supported for COPC files.")

    # 2/ Dispatch by output object type
    if as_object_type in ("dataarray", "dataset"):
        points = _open_pointcloud_array(
            str(filename), data_name, columns, chunks, copc_nodes, load=load, filesystem=filesystem
        )
    else:
        if not load and chunks is None:
            from geoutils.pointcloud.pointcloud import PointCloud

            chunks = max(1, PointCloud(filename, data_name=data_name, filesystem=filesystem).point_count)
        points = _open_pointcloud_geodataframe(
            str(filename), data_name, columns, chunks, copc_nodes, filesystem=filesystem
        )

    # 3/ Downsample, if requested
    partition_lengths = None
    if as_object_type == "geodataframe" and chunks is not None and copc_nodes is not None:
        # Count the dataframe partitions formed by splitting each compressed node
        lengths = []
        for node in copc_nodes:
            for offset in range(0, node.point_count, chunks):
                lengths.append(min(chunks, node.point_count - offset))
        partition_lengths = tuple(lengths) or (0,)
    elif as_object_type == "geodataframe" and chunks is not None:
        # Parquet partitions follow row groups; LAS and vector readers split consecutive rows by chunks
        attrs = _get_dataframe_attrs(points)
        if "row_group_sizes" in attrs:
            partition_lengths = tuple(attrs["row_group_sizes"])
        else:
            count = points.pc.point_count
            partition_lengths = tuple(min(chunks, count - start) for start in range(0, count, chunks)) or (0,)
    points = _downsample_open_pointcloud(points, downsample, source_count, partition_lengths)

    # Parse filenames independently of point records, including deferred and Dask reads
    if parse_sensor_metadata:
        from geoutils.raster.satimg import parse_and_convert_metadata_from_filename

        file_metadata.update(parse_and_convert_metadata_from_filename(str(filename), silent=True))

    # Cache hierarchy metadata on every representation without computing lazy point values
    if file_metadata:
        if isinstance(points, xr.DataArray):
            points.attrs.update(file_metadata)
        else:
            attrs = _get_dataframe_attrs(points).copy()
            attrs.update(file_metadata)
            _set_dataframe_attrs(points, attrs)

    # 4/ Convert to a Xarray Dataset, if requested
    if as_object_type == "dataset":
        from geoutils.pointcloud.xr_accessor import _pointcloud_array_to_dataset

        return _pointcloud_array_to_dataset(points)
    return points


def _open_pointcloud_array(
    filename: str,
    data_name: str | None,
    columns: Literal["all", "main"] | list[str],
    chunks: int | None,
    copc_nodes: list[Entry] | None = None,
    *,
    load: bool = True,
    filesystem: Any = None,
) -> xr.DataArray:
    """
    Open numeric point coordinates directly when the format supports them.

    Used by open_pointcloud() for the pc Xarray accessors and by PointCloud row reads.
    """
    from geoutils.pointcloud.las import _is_laspy_supported, _open_las_array
    from geoutils.pointcloud.parquet import _open_point_parquet

    # 1/ Open according to file format
    # LAS and GeoParquet readers can avoid creating point geometries (we read arrays directly)

    # 1.1/ Read LAS, LAZ or COPC coordinates
    if _is_laspy_supported(filename):
        points = _open_las_array(filename, data_name, columns, chunks, copc_nodes, load=load, filesystem=filesystem)

    # 1.2/ Read GeoParquet coordinates
    elif _is_parquet_supported(filename, filesystem=filesystem):
        points = _open_point_parquet(
            filename, data_name=data_name, columns=columns, chunks=chunks, load=load, filesystem=filesystem
        )

    # 1.3/ Defer other vector reads when no chunks are requested
    elif not load and chunks is None:
        points = _open_deferred_vector_points(filename, data_name, filesystem=filesystem)

    # 1.4/ Read other vector formats eagerly or in chunks
    else:
        frame = _open_pointcloud_geodataframe(filename, data_name, columns, chunks, filesystem=filesystem)

        # Derive partition lengths from metadata without loading points
        lengths = None
        if chunks is not None:
            count = frame.pc.point_count
            lengths = tuple(min(chunks, count - start) for start in range(0, count, chunks))
        points = frame.pc.to_xarray(partition_lengths=lengths)

        # Omit row labels to match LAS and Parquet arrays
        if points.dims[0] in points.coords:
            points = points.drop_vars(points.dims[0])

    # 2/ Store file access settings for later reads
    points.encoding["source"] = filename
    points.encoding["_filesystem"] = filesystem
    return points


def _open_deferred_vector_points(filename: str, data_name: str | None, *, filesystem: Any = None) -> xr.DataArray:
    """
    Describe vector point arrays from file metadata and defer geometry and attribute reads.

    Used by open_pointcloud() for the pc Xarray accessors and by PointCloud row reads.
    """
    from geoutils.vector.io import _read_vector_info

    # 1/ Read metadata and check inputs
    info = _read_vector_info(filename, filesystem=filesystem)
    geometry_type = info.get("geometry_type", "")
    if "Point" not in geometry_type:
        raise ValueError("This vector file contains non-point geometries, cannot be instantiated as a point cloud.")
    if data_name is not None and data_name not in info["fields"]:
        raise ValueError(f"Point data column {data_name!r} does not exist.")

    # 2/ Build an empty Xarray template

    # Use the same column types and attributes as an in-memory conversion
    columns = {name: pd.Series(dtype=dtype) for name, dtype in zip(info["fields"], info["dtypes"])}
    empty = gpd.GeoDataFrame(columns, geometry=gpd.GeoSeries([], crs=info["crs"]))
    empty.attrs["data_name"] = data_name
    template = empty.pc.to_xarray()

    # Empty geometries cannot describe Z coordinates, so we use the file geometry type
    if data_name is not None and geometry_type in ("Point Z", "3D Point"):
        template = template.assign_coords(_geometry_z=("point", np.empty(0)))

    # 3/ Create deferred readers for coordinates and attributes
    variables = {}
    for name, variable in template.to_dataset().variables.items():
        if name in template.dims:
            continue
        variables[name] = _deferred_point_variable(
            info["features"],
            variable.dtype,
            partial(_read_vector_point_attribute, filename, data_name, str(name), filesystem=filesystem),
        )

    # 4/ Return the deferred point array
    return xr.DataArray(variables.pop(template.name), coords=variables, name=template.name, attrs=template.attrs)


def _read_vector_point_attribute(
    filename: str, data_name: str | None, name: str, start: int, count: int, *, filesystem: Any = None
) -> Any:
    """
    Read a vector row range and expose the requested coordinate or point attribute.

    Used by open_pointcloud() for the pc Xarray accessors and by PointCloud row reads.
    """
    from geoutils.vector.io import _read_vector_file

    frame = _read_vector_file(filename, filesystem=filesystem, rows=slice(start, start + count))
    frame.attrs["data_name"] = data_name
    points = frame.pc.to_xarray()
    return points.data if name == points.name else points.coords[name].data


def _open_pointcloud_geodataframe(
    filename: str,
    data_name: str | None,
    columns: Literal["all", "main"] | list[str],
    chunks: int | None,
    copc_nodes: list[Entry] | None = None,
    *,
    filesystem: Any = None,
) -> gpd.GeoDataFrame | Any:
    """
    Open point geometry with the format reader and attach GeoUtils point metadata.

    Shared by PointCloud and open_pointcloud() for GeoDataFrame output.
    """
    from geoutils.pointcloud.las import _is_laspy_supported, _open_las_geodataframe
    from geoutils.pointcloud.parquet import _open_parquet_geodataframe
    from geoutils.pointcloud.pointcloud import PointCloud

    # 1/ Open according to file format

    # 1.1/ Read LAS, LAZ or COPC geometries
    if _is_laspy_supported(filename):
        return _open_las_geodataframe(filename, data_name, columns, chunks, copc_nodes, filesystem=filesystem)

    # 1.2/ Read GeoParquet geometries
    if _is_parquet_supported(filename, filesystem=filesystem):
        return _open_parquet_geodataframe(
            filename, data_name=data_name, columns=columns, chunks=chunks, filesystem=filesystem
        )

    # 1.3/ Read other vector formats without chunks
    if chunks is None:
        pc = PointCloud(filename, data_name=data_name, filesystem=filesystem)
        pc._dataset.attrs["data_name"] = pc.data_name
        return pc._dataset

    # 1.4/ Read other vector formats in chunks
    # Dask-GeoPandas partitions the file without reading point geometry at opening
    from geoutils.vector.io import open_vector

    frame = open_vector(filename, chunks=chunks, filesystem=filesystem)

    # 2/ Attach point cloud metadata to the chunked output
    _set_pointcloud_attrs_from_file(frame, filename=filename, data_name=data_name, filesystem=filesystem)
    return _build_pointcloud_output(
        frame,
        data_name=data_name,
        as_dataframe=True,
        attrs=_get_dataframe_attrs(frame),
        preserve_locations=True,
    )


def _set_pointcloud_attrs_from_file(ds: Any, filename: str, data_name: str | None, *, filesystem: Any = None) -> None:
    """
    Set point cloud metadata on a lazy dataframe opened from a vector file.

    Used by open_pointcloud() to prepare the pc GeoDataFrame accessor.
    """
    from geoutils.vector.io import _read_vector_info

    # Pyogrio exposes file metadata without asking Dask to compute feature partitions
    info = _read_vector_info(filename, filesystem=filesystem)
    geom_type = info.get("geometry_type")
    if geom_type is not None and "Point" not in geom_type:
        raise ValueError("This vector file contains non-point geometries, cannot be instantiated as a point cloud.")
    if data_name is not None and data_name not in info.get("fields", []):
        raise ValueError(
            f"Data column {data_name} not found among columns. Available columns "
            f"are: {', '.join(info.get('fields', []))}."
        )

    # Cache inexpensive spatial metadata for the ``pc`` accessor
    crs = CRS.from_user_input(info["crs"]) if info.get("crs") else getattr(ds, "crs", None)
    total_bounds = info.get("total_bounds")
    bounds = rio.coords.BoundingBox(*total_bounds) if total_bounds is not None else None
    _set_dataframe_attrs(
        ds,
        {
            "crs": crs,
            "bounds": bounds,
            "point_count": info.get("features"),
            "data_name": data_name,
            "geometry_type": geom_type,
        },
    )


def _downsample_open_pointcloud(
    pointcloud: Any,
    downsample: float | PointCloudDownsampleOptions,
    source_count: int | None = None,
    partition_lengths: tuple[int, ...] | None = None,
) -> Any:
    """
    Apply the same downsampling of point clouds to either input arrays or geodataframes.

    COPC reads provide the original count and known partition lengths so _subsample_numpy() can choose row
    positions without loading point values. _pointcloud_from_subsample_indices() selects those rows lazily.

    Shared by PointCloud and open_pointcloud() for the pc accessors.
    """

    # 1/ Determine the sample size
    if downsample == 1:
        return pointcloud

    # Base COPC downsampling on the total point count
    source = pointcloud.pc
    original_count = source.point_count if source_count is None else source_count
    if original_count == 0:
        return pointcloud
    target_count = _pointcloud_downsample_count(original_count, downsample)
    if target_count == source.point_count:
        return pointcloud

    # 2/ Select sampled rows

    # Use a fraction for one point because subsample(1) means all points
    request: int | float = target_count if target_count > 1 else 1 / source.point_count
    from geoutils.sampling.subsampling import _pointcloud_from_subsample_indices, _subsample_numpy

    # Choose reproducible rows without reading point values
    rows = _subsample_numpy(
        np.empty(source.point_count, dtype=np.uint8),
        request,
        return_indices=True,
        random_state=0,
        skip_nodata=False,
    )[0]
    sampled = _pointcloud_from_subsample_indices(source, rows, partition_lengths=partition_lengths)

    # 3/ Return the sample with its metadata
    if isinstance(sampled, xr.DataArray):
        return sampled

    # Preserve the source bounds and store the sampled point count
    attrs = _get_dataframe_attrs(sampled)
    attrs.update({"bounds": source.bounds, "point_count": target_count})
    _set_dataframe_attrs(sampled, attrs)
    return sampled


##############
# 2/ LOADING
##############

#########################
# Deferred Xarray reads
#########################


class _PointBackendArray(BackendArray):  # type: ignore[misc]
    """
    This class serves to add deferred I/O (load=False) to the point cloud Xarray accessor, with the possibility
    of later using .isel().

    We use helpers from Xarray.
    """

    def __init__(self, count: int, dtype: Any, read: Callable[[int, int], Any]) -> None:
        """Store row count, dtype and reader."""
        self.shape = (count,)
        self.dtype = np.dtype(dtype)
        self.read = read

    def __getitem__(self, key: indexing.ExplicitIndexer) -> Any:
        """Convert Xarray selections to consecutive file reads."""

        rows = key.tuple[0]
        if isinstance(rows, np.ndarray) and rows.size == 0:
            return np.empty(rows.shape, dtype=self.dtype)
        if isinstance(rows, slice) and not range(*rows.indices(self.shape[0])):
            return np.empty(0, dtype=self.dtype)
        return indexing.explicit_indexing_adapter(key, self.shape, indexing.IndexingSupport.BASIC, self._read)

    def _read(self, key: tuple[Any, ...]) -> Any:
        """Read smallest consecutive range containing a basic row selection."""
        selected = key[0]
        if isinstance(selected, slice):
            start, stop, step = selected.indices(self.shape[0])
            return self.read(start, stop - start)[::step]

        return self.read(int(selected), 1)[0]


def _deferred_point_variable(count: int, dtype: Any, read: Callable[[int, int], Any]) -> xr.Variable:
    """Wrap point reader for Xarray lazy indexing and in-memory cache."""
    backend = _PointBackendArray(count, dtype, read)
    data = indexing.MemoryCachedArray(indexing.LazilyIndexedArray(backend))
    return xr.Variable("point", data)


#################################
# Load bounds and row ranges
#################################


def _filter_points_by_bounds(pc: gpd.GeoDataFrame, bounds: BoundingBox) -> gpd.GeoDataFrame:
    """
    Filter point geometries by X/Y bounds.

    Shared by PointCloud and its pc accessors for filtering and gridding.
    """

    if len(pc) == 0 or not np.all(np.isfinite(tuple(bounds))):
        return pc

    # Apply the same inclusive bounds to in-memory and distributed point partitions
    if isinstance(pc, xr.DataArray):
        selected = pc.pc.crop(tuple(bounds))
        return selected.compute() if selected.pc._is_dask else selected
    mask = (pc.geometry.x >= bounds.left) & (pc.geometry.x <= bounds.right)
    mask &= (pc.geometry.y >= bounds.bottom) & (pc.geometry.y <= bounds.top)
    # Copy the selected geometries so Shapely can inspect them under Pandas 3
    return pc.loc[mask].copy()


def _filter_dask_points_by_bounds(ds: Any, bounds: BoundingBox) -> Any:
    """
    Filter a Dask-GeoPandas dataframe by bounds, using spatial partitions when available.

    Used by the pc GeoDataFrame accessor for Dask gridding.
    """

    if not np.all(np.isfinite(tuple(bounds))):
        return ds

    # Spatial partitions can discard unrelated partitions before any data is read
    try:
        if getattr(ds, "spatial_partitions", None) is not None:
            return ds.cx[bounds.left : bounds.right, bounds.bottom : bounds.top]
    except NotImplementedError:
        pass

    # Fall back to applying the same coordinate filter inside every partition
    meta = getattr(ds, "_meta", None)
    return ds.map_partitions(_filter_points_by_bounds, bounds, meta=meta)


def _concat_point_parts(parts: list[gpd.GeoDataFrame], crs: Any = None) -> gpd.GeoDataFrame:
    """
    Concatenate per-partition point-cloud subsets.

    Used by the pc GeoDataFrame accessor for Dask filtering and gridding.
    """

    if len(parts) == 0:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=crs), crs=crs)

    non_empty = [part for part in parts if len(part) > 0]
    if len(non_empty) == 0:
        return parts[0].iloc[0:0]

    # Pandas 3 can expose read-only geometry arrays from Dask partitions to GeoPandas
    independent_parts = [part.copy() for part in non_empty]
    return gpd.GeoDataFrame(
        pd.concat(independent_parts, ignore_index=False), geometry=non_empty[0].geometry.name, crs=crs
    )


def _source_dataframe(source_pointcloud: Any) -> gpd.GeoDataFrame | Any | None:
    """
    Return the backing dataframe if it is already available, without triggering file loading.

    Shared by PointCloud and its pc accessors for reads within bounds.
    """

    obj = getattr(source_pointcloud, "_obj", None)
    if obj is not None:
        return obj

    return getattr(source_pointcloud, "_ds", None)


def _load_pointcloud_bounds(
    source_pointcloud: Any,
    bounds: BoundingBox,
    data_name: str | None,
) -> gpd.GeoDataFrame:
    """
    Load or filter source points intersecting bounds.

    Shared by PointCloud and its pc accessors for filtering and gridding.
    """
    from geoutils.vector.io import _read_vector_file

    ds = _source_dataframe(source_pointcloud)
    if ds is not None:
        if is_dask_dataframe(ds):
            raise ValueError("Dask-backed point clouds must use a Dask execution backend.")
        return _filter_points_by_bounds(ds, bounds)

    filename = getattr(source_pointcloud, "name", None)
    filesystem = getattr(source_pointcloud, "_filesystem", None)
    if filename is None:
        return _filter_points_by_bounds(source_pointcloud._dataset, bounds)

    # Import the readers here to avoid circular imports with the point cloud classes
    from geoutils.pointcloud.las import _is_laspy_supported, _load_laspy_data_bounds

    if _is_laspy_supported(filename):
        return _load_laspy_data_bounds(
            filename=filename,
            columns="main",
            bounds=bounds,
            data_name=data_name or "Z",
            level=getattr(source_pointcloud, "_level", None),
            filesystem=filesystem,
        )

    if _is_parquet_supported(filename, filesystem=filesystem):
        from geoutils.pointcloud.parquet import _open_point_parquet

        points = _open_point_parquet(
            filename,
            data_name=data_name,
            columns="main",
            chunks=524_288,
            filesystem=filesystem,
        )
        return points.pc.crop(tuple(bounds)).pc.to_geoutils()._dataset
    if not np.all(np.isfinite(tuple(bounds))):
        return _read_vector_file(filename, filesystem=filesystem)
    return _read_vector_file(filename, filesystem=filesystem, bbox=tuple(bounds))


def _read_point_file_rows(
    filename: str | pathlib.Path,
    columns: list[str],
    start: int,
    count: int,
    *,
    level: int | range | None = None,
    filesystem: Any = None,
) -> gpd.GeoDataFrame:
    """
    Read a row range from a LAS or vector point file, including its schema when the range is empty.

    Shared by PointCloud and its pc accessors for multiprocessing reads.
    """
    from geoutils.pointcloud.las import (
        _is_laspy_supported,
        _load_laspy_data_slice,
        _select_copc_nodes,
        _select_copc_rows,
    )
    from geoutils.vector.io import _read_vector_file

    # LAS has its own row reader, while Pyogrio reads vector rows by offset
    if _is_laspy_supported(filename):
        if level is None:
            return _load_laspy_data_slice(filename, columns=columns, start=start, count=count, filesystem=filesystem)

        # Read intersecting COPC groups, then trim their rows to the requested slice
        nodes = _select_copc_nodes(str(filename), level, target_count=None, filesystem=filesystem)
        selected, local_start = _select_copc_rows(nodes, start, count)
        selected_count = sum(node.point_count for node in selected)
        frame = _load_laspy_data_slice(filename, columns, 0, selected_count, copc_nodes=selected, filesystem=filesystem)
        return frame.iloc[local_start : local_start + count]
    if _is_parquet_supported(filename, filesystem=filesystem):
        from geoutils.pointcloud.parquet import _open_point_parquet

        points = _open_point_parquet(filename, columns="all", chunks=max(count, 1), filesystem=filesystem)
        selected = points.isel({points.dims[0]: slice(start, start + count)}).compute()
        return selected.pc.to_geoutils()._dataset

    # A zero max_features means unlimited rows in Pyogrio, so empty requests read at most one row for their schema
    dataframe = _read_vector_file(filename, filesystem=filesystem, rows=slice(start, start + max(1, count)))
    return dataframe.iloc[:0] if count == 0 else dataframe


def _load_pointcloud_rows(
    source_pointcloud: Any, start: int, count: int, *, columns: list[str] | None = None
) -> gpd.GeoDataFrame:
    """
    Load a consecutive group of point cloud rows for one partition.

    Shared by PointCloud and its pc accessors for multiprocessing reads.
    """

    # Slice an existing dataframe directly when the point cloud is already loaded
    if source_pointcloud.is_loaded or source_pointcloud._is_pd:
        return source_pointcloud._dataset.iloc[start : start + count]
    assert source_pointcloud.name is not None

    if getattr(source_pointcloud, "_downsample", 1) != 1:
        # The array reader applies the opening sample before selecting worker rows
        points = open_pointcloud(
            source_pointcloud.name,
            data_name=source_pointcloud.data_name,
            columns="all" if columns is None else columns,
            downsample=source_pointcloud._downsample,
            filesystem=getattr(source_pointcloud, "_filesystem", None),
            as_object_type="dataarray",
        )
        selected = points.isel({points.dims[0]: slice(start, start + count)}).compute()
        return selected.pc.to_geoutils()._dataset

    # LAS needs its dimension names; vector files do not need columns for Pyogrio
    if columns is None:
        columns = list(source_pointcloud._nongeo_columns) if source_pointcloud._is_las else []
    return _read_point_file_rows(
        source_pointcloud.name,
        columns,
        start,
        count,
        level=getattr(source_pointcloud, "_level", None),
        filesystem=getattr(source_pointcloud, "_filesystem", None),
    )


##############
# 3/ WRITING
##############


PointCloudDriver = Literal["GPKG", "LAS", "LAZ", "PARQUET"]
_POINTCLOUD_FORMATS: dict[str, PointCloudDriver] = {
    ".gpkg": "GPKG",
    ".las": "LAS",
    ".laz": "LAZ",
    ".parquet": "PARQUET",
}
_GPKG_DIRECT_WRITE_ROWS = 524_288
_GPKG_WRITE_BATCH_ROWS = 524_288


def _array_partitions(points: xr.DataArray, chunks: int | None) -> Any:
    """
    Yield in-memory row partitions, computing values and attributes together within each partition.

    Shared by PointCloud and its pc accessors through Xarray writing and sampling.
    """

    # Use the requested row count or the largest existing chunk to bound each write
    if chunks is not None and (isinstance(chunks, bool) or not isinstance(chunks, int) or chunks <= 0):
        raise ValueError("Argument 'chunks' must be a strictly positive integer.")
    size = chunks or (max(points.chunks[0]) if points.chunks else 524_288)

    # Yield an empty partition too, so writers can save the schema of a zero-row result
    for start in range(0, max(points.size, 1), size):
        yield points.isel({points.dims[0]: slice(start, start + size)}).compute()


def _check_pointcloud_output_source(points: xr.DataArray, filename: str | pathlib.Path, filesystem: Any) -> None:
    """
    Prevent a direct write from truncating point data that still need to be read.

    Shared by PointCloud and its pc accessors through the Xarray file writers.
    """
    source = points.encoding.get("source")
    if filesystem is None or source is None or points.pc.is_loaded:
        return
    source_filesystem = points.encoding.get("_filesystem")
    if source_filesystem is None:
        source, source_filesystem = _resolve_filesystem(source, storage_options={})

    # Compare paths within the same filesystem before opening the destination
    if source_filesystem == filesystem:
        strip_protocol = getattr(filesystem, "_strip_protocol", os.fspath)
        if strip_protocol(source) == strip_protocol(str(filename)):
            raise ValueError("Cannot overwrite a lazy point cloud source. Load it first or choose another destination.")


def _stage_pointcloud_partition(dataframe: gpd.GeoDataFrame, filename: str | pathlib.Path) -> pathlib.Path:
    """
    Save one dataframe partition and return its normalized path for ordered file-based exchange.

    Shared by Raster, PointCloud and their accessors for multiprocessing point cloud output.
    """

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
    """
    Resolve and validate a point cloud output path and driver for one operation.

    Shared by Raster, PointCloud and their accessors for multiprocessing point cloud output.
    """

    if _is_remote_path(filename):
        raise ValueError("Multiprocessing point cloud output requires a local path.")
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
    """
    Reject attributes that GeoPackage storage or its dataframe reader would round.

    Shared by Raster, PointCloud and their accessors for multiprocessing point cloud output.
    """

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


def _write_gpkg_batch(
    dataframes: list[gpd.GeoDataFrame],
    filename: pathlib.Path,
    *,
    append: bool,
    layer_options: dict[str, str],
) -> None:
    """
    Combine one bounded dataframe batch and write its rows to a GeoPackage.

    Shared by Raster, PointCloud and their accessors for multiprocessing point cloud output.
    """

    combined = dataframes[0] if len(dataframes) == 1 else pd.concat(dataframes)
    pyogrio.write_dataframe(
        combined,
        filename,
        layer="points",
        driver="GPKG",
        append=append,
        layer_options=layer_options,
    )


def _write_pointcloud_partitions(
    filename: str | pathlib.Path,
    partition_filenames: Iterable[str | pathlib.Path],
    *,
    driver: PointCloudDriver,
    data_name: str | None,
    geometry_type: Literal["Point", "Point Z"],
    las_header: Any | None = None,
    las_elevation_column: str | None = None,
    las_bounds: Sequence[np.ndarray[Any, Any] | None] | None = None,
) -> PointCloud:
    """
    Combine saved dataframe partitions into one file and return an unloaded PointCloud.

    The operation producing points saves each partition as a pickle so workers and the parent exchange only filenames.
    GeoPackage output combines small partitions before appending them. LAS/LAZ output scans partition bounds for a
    shared header, then reads and appends one saved partition at a time.

    Shared by Raster, PointCloud and their accessors for multiprocessing point cloud output.
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

                # Flush small pending partitions before writing an already large partition directly
                if len(dataframe) >= _GPKG_DIRECT_WRITE_ROWS and pending_dataframes:
                    _write_gpkg_batch(
                        pending_dataframes,
                        temporary_output,
                        append=output_created,
                        layer_options=layer_options,
                    )
                    output_created = True
                    pending_dataframes.clear()
                    pending_rows = 0

                # Combine small partitions while avoiding copies of already large raster tiles
                pending_dataframes.append(dataframe)
                pending_rows += len(dataframe)
                if len(dataframe) < _GPKG_DIRECT_WRITE_ROWS and pending_rows < _GPKG_WRITE_BATCH_ROWS:
                    continue

                _write_gpkg_batch(
                    pending_dataframes,
                    temporary_output,
                    append=output_created,
                    layer_options=layer_options,
                )
                output_created = True
                pending_dataframes.clear()
                pending_rows = 0

            # Write the final short batch after every worker partition has been consumed
            if pending_dataframes:
                _write_gpkg_batch(
                    pending_dataframes,
                    temporary_output,
                    append=output_created,
                    layer_options=layer_options,
                )
        elif driver == "PARQUET":
            from geoutils.pointcloud.parquet import _write_parquet_row_groups

            # Read each staged dataframe only when the shared writer is ready for its next row group
            frames = (pd.read_pickle(filename) for filename in chain((first_filename,), partitions))
            _write_parquet_row_groups(frames, temporary_output, data_name=data_name)
        else:
            saved_partitions = [first_filename, *partitions]
            if las_header is None:
                las_header = _build_laspy_header_from_partitions(
                    saved_partitions, las_elevation_column, partition_bounds=las_bounds
                )

            # Encode each saved partition once into the final ordered LAS/LAZ stream
            _write_laspy_saved_partitions(
                filename=temporary_output,
                partition_filenames=saved_partitions,
                data_name=las_elevation_column,
                header=las_header,
                check_attributes=True,
            )

        os.replace(temporary_output, output_filename)

    return PointCloud(output_filename, data_name=data_name)


# Writing point cloud objects


def _write_geometry_pointcloud(
    source_pointcloud: GeometryPointCloudBase,
    filename: str,
    driver: Any = None,
    schema: Any = None,
    index: Any = None,
    **kwargs: Any,
) -> None:
    """
    Write geometry point clouds to GeoParquet or a vector format.

    Shared by PointCloud and the pc GeoDataFrame accessor.
    """
    filename, filesystem = _resolve_filesystem(
        filename, kwargs.pop("filesystem", None), kwargs.pop("storage_options", None), remote_default=False
    )
    if filesystem is not None:
        kwargs["filesystem"] = filesystem
    if _path_suffix(filename) == ".parquet" or driver == "PARQUET":
        source_pointcloud.to_parquet(filename, **kwargs)
    elif source_pointcloud._is_dask and (_path_suffix(filename) == ".gpkg" or driver == "GPKG"):
        # Write one partition at a time so the complete point cloud never occupies client memory
        options = dict(kwargs)
        for partition in source_pointcloud._dataset.to_delayed():
            dataframe = partition.compute()
            from geoutils.vector.io import _write_vector_file

            _write_vector_file(dataframe, filename, driver=driver, schema=schema, index=index, **options)
            options["mode"] = "a"
    else:
        # Use the shared vector writer for other geometry formats
        from geoutils.vector.io import _write_vector_file

        dataframe = source_pointcloud._dataset if source_pointcloud._is_pd else source_pointcloud.gdf
        _write_vector_file(dataframe, filename=filename, driver=driver, schema=schema, index=index, **kwargs)


def _write_pointcloud_array(array: xr.DataArray, filename: str, **kwargs: Any) -> None:
    """
    Write a point DataArray using the format selected by its filename.

    Used only by the pc Xarray accessors.
    """
    filename, filesystem = _resolve_filesystem(
        filename, kwargs.pop("filesystem", None), kwargs.pop("storage_options", None), remote_default=False
    )
    if filesystem is not None:
        kwargs["filesystem"] = filesystem
    suffix = _path_suffix(filename)
    if suffix in (".las", ".laz"):
        array.pc.to_las(filename, **kwargs)
    elif suffix == ".parquet":
        array.pc.to_parquet(filename, **kwargs)
    else:
        from geoutils.vector.io import _write_vector_file

        _write_vector_file(array.pc.to_geoutils()._dataset, filename, **kwargs)
