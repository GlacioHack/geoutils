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

"""Load the point cloud rows needed for each part of a calculation."""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

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
from geoutils._misc import _deprecate_keyword, _validate_downsample
from geoutils._typing import Number
from geoutils.pointcloud.dataframe import _build_pointcloud_output, _get_dataframe_attrs, _set_dataframe_attrs
from geoutils.pointcloud.parquet import _is_parquet_supported

if TYPE_CHECKING:
    from laspy.copc import Entry

#####################
# Open point cloud
#####################


@_deprecate_keyword("data_column", "data_name")
def open_pointcloud(
    filename: str | pathlib.Path,
    data_name: str | None = None,
    columns: Literal["all", "main"] | list[str] = "main",
    chunks: int | None = None,
    downsample: Number = 1,
    *,
    as_object_type: Literal["dataarray", "dataset", "geodataframe"] | None = None,
    level: int | range | None = None,
    load: bool = False,
    parse_sensor_metadata: bool = False,
) -> xr.DataArray | xr.Dataset | gpd.GeoDataFrame | Any:
    """
    Open a point cloud as a Xarray DataArray, Dataset or GeoDataFrame.

    LAS, LAZ and COPC files are read with LasPy, GeoParquet files and partition directories with PyArrow, and other
    point vector formats with GeoPandas.

    Passing ``chunks`` opens the point data lazily using Dask arrays (Xarray objects) or Dask-GeoPandas (Pandas
    object). Use ``downsample`` to  select a deterministic random sample. For example, ``downsample=2`` returns
    half the points, rounded up. For COPC, we select levels from coarse to fine until they contain enough points,
    then sample those points to the requested count. Other formats sample the full point cloud.

    Use ``level`` to select COPC levels of detail, following LasPy. Each point belongs to one level:
    level 0 holds a coarse sample, and higher levels add points. Selected levels are combined into one point cloud.
    An integer selects only that level; ``level=range(3)`` combines levels 0 through 2. Explicit levels cannot
    be combined with ``downsample > 1``.

    :param filename: Point file or directory of GeoParquet partitions.
    :param data_name: Attribute containing point values. Defaults to Z for LAS, the saved active attribute for
        GeoParquet, or geometry elevations when no attribute is selected in other vector formats.
    :param columns: Attributes to read from LAS or GeoParquet: main selects the active values, all selects every
        attribute, and a list selects named attributes as well as the active values. Ignored for other vector formats.
    :param chunks: Points per Dask chunk or partition. GeoParquet GeoDataFrame partitions follow the stored row
        groups. None uses deferred file reads unless load=True.
    :param downsample: Factor reducing the point count to ceil(N / downsample), with a deterministic random sample.
    :param as_object_type: Returned representation: dataarray, dataset, or geodataframe. None uses
        config["pointcloud_return_type"], which defaults to dataarray.
    :param level: COPC level of detail, or a range of levels. None combines all levels unless downsample is used.
    :param load: Whether to read values immediately or not. The default False defers reads but requires using Xarray
        as object type, a GeoDataFrame requires Dask-GeoPandas.
    :param parse_sensor_metadata: Parse sensor metadata from the filename and add it to the returned attributes.
    :returns: A DataArray with active values and auxiliary coordinates, a Dataset with separate value variables,
        or a GeoDataFrame.
    """
    # 1/ Input checks
    if as_object_type is None:
        as_object_type = config["pointcloud_return_type"]
    downsample = _validate_downsample(downsample)
    if chunks is not None and (isinstance(chunks, bool) or not isinstance(chunks, int) or chunks <= 0):
        raise ValueError("Argument 'chunks' must be a strictly positive integer.")
    if as_object_type not in ("dataarray", "dataset", "geodataframe"):
        raise ValueError("Argument 'as_object_type' must be 'dataarray', 'dataset' or 'geodataframe'.")
    if level is not None:
        if isinstance(level, range):
            valid_level = level.start >= 0 and level.step > 0 and len(level) > 0
        else:
            valid_level = (
                not isinstance(level, (bool, np.bool_)) and isinstance(level, (int, np.integer)) and level >= 0
            )
        if not valid_level:
            raise ValueError(
                "Argument 'level' must be a non-negative integer or a non-empty increasing range of levels."
            )
        if downsample > 1:
            raise ValueError("downsample and level cannot be used together.")

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
        metadata = _load_laspy_metadata(filename)
        file_metadata = {"point_record_size": metadata.point_record_size, "laz_chunk_size": metadata.laz_chunk_size}
        if metadata.is_copc:
            source_count = metadata.point_count
            target_count = int(np.ceil(source_count / downsample)) if downsample > 1 else None
            hierarchy = _read_copc_hierarchy(str(filename))
            copc_nodes = _select_copc_nodes(str(filename), level, target_count, nodes=hierarchy)
            file_metadata.update(
                {
                    "copc_levels": sorted({node.key.level for node in hierarchy}),
                    "copc_selected_levels": sorted({node.key.level for node in copc_nodes}),
                    "copc_chunk_sizes": [node.point_count for node in hierarchy],
                }
            )
    if level is not None and copc_nodes is None:
        raise ValueError("Argument 'level' is only supported for COPC files.")

    # 2/ Dispatch by request representation
    if as_object_type in ("dataarray", "dataset"):
        points = _open_pointcloud_array(str(filename), data_name, columns, chunks, copc_nodes, load=load)
    else:
        if not load and chunks is None:
            from geoutils.pointcloud.pointcloud import PointCloud

            chunks = max(1, PointCloud(filename, data_name=data_name).point_count)
        points = _open_pointcloud_geodataframe(str(filename), data_name, columns, chunks, copc_nodes)

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
) -> xr.DataArray:
    """Open numeric point coordinates directly when the format supports them."""
    from geoutils.pointcloud.las import _is_laspy_supported, _open_las_array
    from geoutils.pointcloud.parquet import _open_point_parquet

    # Native LAS and GeoParquet point fields avoid allocating Shapely objects
    if _is_laspy_supported(filename):
        points = _open_las_array(filename, data_name, columns, chunks, copc_nodes, load=load)
    elif _is_parquet_supported(filename):
        points = _open_point_parquet(filename, data_name=data_name, columns=columns, chunks=chunks, load=load)
    elif not load and chunks is None:
        points = _open_deferred_vector_points(filename, data_name)
    else:
        # Other vector formats expose their coordinates through the geometry reader
        frame = _open_pointcloud_geodataframe(filename, data_name, columns, chunks)
        # Vector file metadata gives the exact lengths used by Dask-GeoPandas read_file()
        lengths = None
        if chunks is not None:
            count = frame.pc.point_count
            lengths = tuple(min(chunks, count - start) for start in range(0, count, chunks))
        points = frame.pc.to_xarray(partition_lengths=lengths)
        # File rows use implicit positions, matching native LAS and Parquet arrays
        if points.dims[0] in points.coords:
            points = points.drop_vars(points.dims[0])
    points.encoding["source"] = filename
    return points


def _open_deferred_vector_points(filename: str, data_name: str | None) -> xr.DataArray:
    """Describe vector point arrays from file metadata and defer geometry and attribute reads."""

    # Use an empty GeoDataFrame to preserve the same attribute metadata as in-memory conversion
    info = pyogrio.read_info(filename)
    geometry_type = info.get("geometry_type", "")
    if "Point" not in geometry_type:
        raise ValueError("This vector file contains non-point geometries, cannot be instantiated as a point cloud.")
    if data_name is not None and data_name not in info["fields"]:
        raise ValueError(f"Point data column {data_name!r} does not exist.")
    columns = {name: pd.Series(dtype=dtype) for name, dtype in zip(info["fields"], info["dtypes"])}
    empty = gpd.GeoDataFrame(columns, geometry=gpd.GeoSeries([], crs=info["crs"]))
    empty.attrs["data_name"] = data_name
    template = empty.pc.to_xarray()

    # Empty geometries cannot describe Z coordinates, so use the file's geometry type
    if data_name is not None and geometry_type in ("Point Z", "3D Point"):
        template = template.assign_coords(_geometry_z=("point", np.empty(0)))

    # Every attribute uses its native dtype; point coordinates are floating arrays
    variables = {}
    for name, variable in template.to_dataset().variables.items():
        if name in template.dims:
            continue
        variables[name] = _deferred_point_variable(
            info["features"], variable.dtype, partial(_read_vector_point_attribute, filename, data_name, str(name))
        )
    return xr.DataArray(variables.pop(template.name), coords=variables, name=template.name, attrs=template.attrs)


def _read_vector_point_attribute(filename: str, data_name: str | None, name: str, start: int, count: int) -> Any:
    """Read a vector row range and expose the requested coordinate or point attribute."""
    frame = pyogrio.read_dataframe(filename, skip_features=start, max_features=count)
    frame.attrs["data_name"] = data_name
    points = frame.pc.to_xarray()
    return points.data if name == points.name else points.coords[name].data


def _open_pointcloud_geodataframe(
    filename: str,
    data_name: str | None,
    columns: Literal["all", "main"] | list[str],
    chunks: int | None,
    copc_nodes: list[Entry] | None = None,
) -> gpd.GeoDataFrame | Any:
    """Open point geometry with the format reader and attach GeoUtils point metadata."""
    from geoutils.pointcloud.las import _is_laspy_supported, _open_las_geodataframe
    from geoutils.pointcloud.parquet import _open_parquet_geodataframe
    from geoutils.pointcloud.pointcloud import PointCloud
    from geoutils.vector.pd_accessor import _import_dask_geopandas

    # Format-specific readers supply native attribute dtypes and geospatial metadata
    if _is_laspy_supported(filename):
        return _open_las_geodataframe(filename, data_name, columns, chunks, copc_nodes)
    if _is_parquet_supported(filename):
        return _open_parquet_geodataframe(filename, data_name=data_name, columns=columns, chunks=chunks)
    if chunks is None:
        # Preserve the established in-memory PointCloud loading and validation path
        pc = PointCloud(filename, data_name=data_name)
        pc._dataset.attrs["data_name"] = pc.data_name
        return pc._dataset

    # Dask-GeoPandas partitions the file without reading point geometry at opening
    dgpd = _import_dask_geopandas()
    frame = dgpd.read_file(filename, chunksize=chunks)
    _set_pointcloud_attrs_from_file(frame, filename=filename, data_name=data_name)
    return _build_pointcloud_output(
        frame,
        data_name=data_name,
        as_dataframe=True,
        attrs=_get_dataframe_attrs(frame),
        preserve_locations=True,
    )


def _set_pointcloud_attrs_from_file(ds: Any, filename: str, data_name: str | None) -> None:
    """Set point cloud metadata on a lazy dataframe opened from a vector file."""

    # Pyogrio exposes file metadata without asking Dask to compute feature partitions
    info = pyogrio.read_info(filename)
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
    downsample: float,
    source_count: int | None = None,
    partition_lengths: tuple[int, ...] | None = None,
) -> Any:
    """
    Apply the same deterministic opening sample to arrays and GeoDataFrames.

    COPC reads provide the original count and known partition lengths so _subsample_numpy() can choose row
    positions without loading point values. _pointcloud_from_subsample_indices() selects those rows lazily.
    """

    if downsample == 1:
        return pointcloud

    # Convert the factor to the count convention shared by in-memory and Dask point subsampling
    source = pointcloud.pc
    original_count = source.point_count if source_count is None else source_count
    if original_count == 0:
        return pointcloud
    target_count = max(1, int(np.ceil(original_count / downsample)))
    if target_count == source.point_count:
        return pointcloud

    # Request a fraction for one point because subsample(1) selects every row
    request: int | float = target_count if target_count > 1 else 1 / source.point_count
    from geoutils.sampling.subsampling import _pointcloud_from_subsample_indices, _subsample_numpy

    # Opening samples rows independently of measurement validity, without reading their values
    rows = _subsample_numpy(
        np.empty(source.point_count, dtype=np.uint8),
        request,
        return_indices=True,
        random_state=0,
        skip_nodata=False,
    )[0]
    sampled = _pointcloud_from_subsample_indices(source, rows, partition_lengths=partition_lengths)

    if isinstance(sampled, xr.DataArray):
        return sampled

    # Preserve the complete source extent while recording the exact deterministic sample size
    attrs = _get_dataframe_attrs(sampled)
    attrs.update({"bounds": source.bounds, "point_count": target_count})
    _set_dataframe_attrs(sampled, attrs)
    return sampled


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
    """Wrap a point reader in Xarray's lazy indexing and in-memory cache."""
    backend = _PointBackendArray(count, dtype, read)
    data = indexing.MemoryCachedArray(indexing.LazilyIndexedArray(backend))
    return xr.Variable("point", data)


#################################
# Load bounds and row ranges
#################################


def _filter_points_by_bounds(pc: gpd.GeoDataFrame, bounds: BoundingBox) -> gpd.GeoDataFrame:
    """Filter point geometries by X/Y bounds."""

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
    """Filter a Dask-GeoPandas dataframe by bounds, using spatial partitions when available."""

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
    """Concatenate per-partition point-cloud subsets."""

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
    """Return the backing dataframe if it is already available, without triggering file loading."""

    obj = getattr(source_pointcloud, "_obj", None)
    if obj is not None:
        return obj

    return getattr(source_pointcloud, "_ds", None)


def _load_pointcloud_bounds(
    source_pointcloud: Any,
    bounds: BoundingBox,
    data_name: str | None,
) -> gpd.GeoDataFrame:
    """Load or filter source points intersecting bounds."""

    ds = _source_dataframe(source_pointcloud)
    if ds is not None:
        if is_dask_dataframe(ds):
            raise ValueError("Dask-backed point clouds must use a Dask execution backend.")
        return _filter_points_by_bounds(ds, bounds)

    filename = getattr(source_pointcloud, "name", None)
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
        )

    if _is_parquet_supported(filename):
        from geoutils.pointcloud.parquet import _open_point_parquet

        points = _open_point_parquet(filename, data_name=data_name, columns="main", chunks=524_288)
        return points.pc.crop(tuple(bounds)).pc.to_geoutils()._dataset
    if not np.all(np.isfinite(tuple(bounds))):
        return gpd.read_file(filename)
    return gpd.read_file(filename, bbox=tuple(bounds))


def _read_point_file_rows(filename: str | pathlib.Path, columns: list[str], start: int, count: int) -> gpd.GeoDataFrame:
    """Read a row range from a LAS or vector point file, including its schema when the range is empty."""

    from geoutils.pointcloud.las import _is_laspy_supported, _load_laspy_data_slice

    # LAS has its own row reader, while Pyogrio reads vector rows by offset
    if _is_laspy_supported(filename):
        return _load_laspy_data_slice(
            filename,
            columns=columns,
            start=start,
            count=count,
        )
    if _is_parquet_supported(filename):
        from geoutils.pointcloud.parquet import _open_point_parquet

        points = _open_point_parquet(filename, columns="all", chunks=max(count, 1))
        selected = points.isel({points.dims[0]: slice(start, start + count)}).compute()
        return selected.pc.to_geoutils()._dataset

    dataframe = pyogrio.read_dataframe(
        filename,
        skip_features=start,
        max_features=max(1, count),
    )
    return dataframe.iloc[:0] if count == 0 else dataframe


def _load_pointcloud_rows(source_pointcloud: Any, start: int, count: int) -> gpd.GeoDataFrame:
    """Load a consecutive group of point cloud rows for one partition."""

    # Slice an existing dataframe directly when the point cloud is already loaded
    if source_pointcloud.is_loaded or source_pointcloud._is_pd:
        return source_pointcloud._dataset.iloc[start : start + count]
    assert source_pointcloud.name is not None

    # LAS needs its dimension names; vector files do not need columns for Pyogrio
    columns = list(source_pointcloud._nongeo_columns) if source_pointcloud._is_las else []
    return _read_point_file_rows(source_pointcloud.name, columns, start, count)
