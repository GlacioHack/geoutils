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
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio as rio
import xarray as xr
from pyproj import CRS
from rasterio.coords import BoundingBox

from geoutils._dispatch import is_dask_dataframe
from geoutils._misc import _deprecate_keyword, _validate_downsample
from geoutils._typing import Number
from geoutils.pointcloud.dataframe import _build_pointcloud_output, _get_dataframe_attrs, _set_dataframe_attrs

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
    as_type: Literal["dataarray", "dataset", "geodataframe"] = "dataarray",
) -> xr.DataArray | xr.Dataset | gpd.GeoDataFrame | Any:
    """
    Open a point cloud as a Xarray DataArray, Dataset or GeoDataFrame.

    LAS, LAZ and COPC files are read with LasPy, GeoParquet files and partition directories with PyArrow, and other
    point vector formats with GeoPandas.

    Passing ``chunks`` opens the point data lazily using Dask arrays (Xarray objects) or Dask-GeoPandas (Pandas
    object). Use ``downsample`` to  select a deterministic random sample. For example, ``downsample=2`` returns
    half the points, rounded up.

    :param filename: Point file or directory of GeoParquet partitions.
    :param data_name: Attribute containing point values. Defaults to Z for LAS, the saved active attribute for
        GeoParquet, or geometry elevations when no attribute is selected in other vector formats.
    :param columns: Attributes to read from LAS or GeoParquet: main selects the active values, all selects every
        attribute, and a list selects named attributes as well as the active values. Ignored for other vector formats.
    :param chunks: Points per Dask chunk or partition. GeoParquet GeoDataFrame partitions follow the stored row
        groups. None loads the point data eagerly.
    :param downsample: Factor reducing the point count to ceil(N / downsample), with a deterministic random sample.
    :param as_type: Returned representation: dataarray (the default), dataset, or geodataframe.
    :returns: A DataArray with active values and auxiliary coordinates, a Dataset with separate value variables,
        or a GeoDataFrame.
    """
    # 1/ Input checks
    downsample = _validate_downsample(downsample)
    if chunks is not None and (isinstance(chunks, bool) or not isinstance(chunks, int) or chunks <= 0):
        raise ValueError("Argument 'chunks' must be a strictly positive integer.")

    # 2/ Dispatch by request representation
    if as_type in ("dataarray", "dataset"):
        points = _open_pointcloud_array(str(filename), data_name, columns, chunks)
    elif as_type == "geodataframe":
        points = _open_pointcloud_geodataframe(str(filename), data_name, columns, chunks)
    else:
        raise ValueError("Argument 'as_type' must be 'dataarray', 'dataset' or 'geodataframe'.")

    # 3/ Downsample, if requested
    points = _downsample_open_pointcloud(points, downsample)

    # 4/ Convert to a Xarray Dataset, if requested
    if as_type == "dataset":
        # Value attributes become separate variables while coordinates and lazy graphs stay shared
        names = [name for name in points.pc.columns if name != points.name and name != "_geometry_z"]
        dataset = points.rename({"x": "x_point", "y": "y_point"}).to_dataset().reset_coords(names)
        dataset = dataset[[points.name, *names]]
        for name in dataset.data_vars:
            dataset[name].attrs["crs"] = points.attrs.get("crs")
        dataset.attrs = points.attrs.copy()
        dataset.encoding = points.encoding.copy()
        return dataset
    return points


def _open_pointcloud_array(
    filename: str,
    data_name: str | None,
    columns: Literal["all", "main"] | list[str],
    chunks: int | None,
) -> xr.DataArray:
    """Open numeric point coordinates directly when the format supports them."""
    from geoutils.pointcloud.las import _is_laspy_supported, _open_las_array
    from geoutils.pointcloud.parquet import _open_point_parquet

    # Native LAS and GeoParquet point fields avoid allocating Shapely objects
    if _is_laspy_supported(filename):
        points = _open_las_array(filename, data_name, columns, chunks)
    elif pathlib.Path(filename).suffix.lower() == ".parquet" or pathlib.Path(filename).is_dir():
        points = _open_point_parquet(filename, data_name=data_name, columns=columns, chunks=chunks)
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


def _open_pointcloud_geodataframe(
    filename: str,
    data_name: str | None,
    columns: Literal["all", "main"] | list[str],
    chunks: int | None,
) -> gpd.GeoDataFrame | Any:
    """Open point geometry with the format reader and attach GeoUtils point metadata."""
    from geoutils.pointcloud.las import _is_laspy_supported, _open_las_geodataframe
    from geoutils.pointcloud.parquet import _open_parquet_geodataframe
    from geoutils.pointcloud.pointcloud import PointCloud
    from geoutils.vector.pd_accessor import _import_dask_geopandas

    # Format-specific readers supply native attribute dtypes and geospatial metadata
    if _is_laspy_supported(filename):
        return _open_las_geodataframe(filename, data_name, columns, chunks)
    if pathlib.Path(filename).suffix.lower() == ".parquet" or pathlib.Path(filename).is_dir():
        return _open_parquet_geodataframe(filename, data_name=data_name, columns=columns, chunks=chunks)
    if chunks is None:
        # Preserve the established eager PointCloud loading and validation path
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


def _downsample_open_pointcloud(pointcloud: Any, downsample: float) -> Any:
    """Apply the same deterministic opening sample to arrays and GeoDataFrames."""

    if downsample == 1:
        return pointcloud

    # Convert the factor to the count convention shared by eager and Dask point subsampling
    source = pointcloud.pc
    source_count = source.point_count
    if source_count == 0:
        return pointcloud
    target_count = max(1, int(np.ceil(source_count / downsample)))
    request: int | float = target_count if target_count > 1 else 1 / source_count
    sampled = source.subsample(request, random_state=0)

    if isinstance(sampled, xr.DataArray):
        return sampled

    # Preserve the complete source extent while recording the exact deterministic sample size
    attrs = _get_dataframe_attrs(sampled)
    attrs.update({"bounds": source.bounds, "point_count": target_count})
    _set_dataframe_attrs(sampled, attrs)
    return sampled


#################################
# Load bounds and row ranges
#################################


def _filter_points_by_bounds(pc: gpd.GeoDataFrame, bounds: BoundingBox) -> gpd.GeoDataFrame:
    """Filter point geometries by X/Y bounds."""

    if len(pc) == 0 or not np.all(np.isfinite(tuple(bounds))):
        return pc

    # Apply the same inclusive bounds to eager and distributed point partitions
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

    if pathlib.Path(filename).suffix.lower() == ".parquet" or pathlib.Path(filename).is_dir():
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
    if pathlib.Path(filename).suffix.lower() == ".parquet" or pathlib.Path(filename).is_dir():
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
