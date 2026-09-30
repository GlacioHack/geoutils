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

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.coords import BoundingBox

from geoutils._dispatch import is_dask_dataframe


def _filter_points_by_bounds(pc: gpd.GeoDataFrame, bounds: BoundingBox) -> gpd.GeoDataFrame:
    """Filter point geometries by X/Y bounds."""

    if len(pc) == 0 or not np.all(np.isfinite(tuple(bounds))):
        return pc

    # Apply the same inclusive bounds to eager and distributed point partitions
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
    return gpd.GeoDataFrame(pd.concat(independent_parts, ignore_index=False), geometry="geometry", crs=crs)


def _source_dataframe(source_pointcloud: Any) -> gpd.GeoDataFrame | Any | None:
    """Return the backing dataframe if it is already available, without triggering file loading."""

    obj = getattr(source_pointcloud, "_obj", None)
    if obj is not None:
        return obj

    return getattr(source_pointcloud, "_ds", None)


def _load_pointcloud_bounds(
    source_pointcloud: Any,
    bounds: BoundingBox,
    data_column_name: str | None,
) -> gpd.GeoDataFrame:
    """Load or filter source points intersecting bounds."""

    ds = _source_dataframe(source_pointcloud)
    if ds is not None:
        if is_dask_dataframe(ds):
            raise ValueError("Dask-backed point clouds must use a Dask execution backend.")
        return _filter_points_by_bounds(ds, bounds)

    filename = getattr(source_pointcloud, "name", None)
    if filename is None:
        return _filter_points_by_bounds(source_pointcloud.ds, bounds)

    # Import the readers here to avoid circular imports with the point cloud classes
    from geoutils.pointcloud.las import _is_laspy_supported, _load_laspy_data_bounds

    if _is_laspy_supported(filename):
        return _load_laspy_data_bounds(
            filename=filename,
            columns="main",
            bounds=bounds,
            data_column=data_column_name or "Z",
        )

    if not np.all(np.isfinite(tuple(bounds))):
        return gpd.read_file(filename)
    return gpd.read_file(filename, bbox=tuple(bounds))


def _load_pointcloud_rows(source_pointcloud: Any, start: int, count: int) -> gpd.GeoDataFrame:
    """Load a consecutive group of point cloud rows for one partition."""

    # Slice an existing dataframe directly when the point cloud is already loaded
    if source_pointcloud.is_loaded or source_pointcloud._is_pd:
        return source_pointcloud.ds.iloc[start : start + count]
    assert source_pointcloud.name is not None
    from geoutils.pointcloud.las import _is_laspy_supported, _load_laspy_data_slice

    # LAS has its own row reader, while Pyogrio reads vector rows by offset
    if _is_laspy_supported(source_pointcloud.name):
        return _load_laspy_data_slice(
            source_pointcloud.name,
            columns=list(source_pointcloud._nongeo_columns),
            start=start,
            count=count,
        )
    import pyogrio

    dataframe = pyogrio.read_dataframe(
        source_pointcloud.name,
        skip_features=start,
        max_features=max(1, count),
    )
    return dataframe.iloc[:0] if count == 0 else dataframe
