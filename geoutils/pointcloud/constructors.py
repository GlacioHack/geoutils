"""Create point clouds from coordinate and value arrays."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import geopandas as gpd
import pandas as pd
import xarray as xr

from geoutils._config import config
from geoutils._dispatch import is_dask_array
from geoutils.pointcloud.dataframe import _build_pointcloud_output


def from_xyz(
    x: Any,
    y: Any,
    z: Any,
    crs: Any,
    data_name: str | None = None,
    use_z: bool = False,
    *,
    auxiliary: Mapping[str, Any] | None = None,
    as_object_type: Literal["dataarray", "dataset", "geodataframe"] | None = None,
) -> xr.DataArray | xr.Dataset | gpd.GeoDataFrame | Any:
    """
    Create a point cloud from separate X, Y and value arrays.

    Xarray outputs store coordinates directly. GeoDataFrame outputs construct point geometries, using
    Dask-GeoPandas when any input array is lazy. Auxiliary attributes preserve their individual dtypes.

    :param x: One-dimensional X coordinates.
    :param y: One-dimensional Y coordinates, with the same length as x.
    :param z: One-dimensional point values, with the same length as x.
    :param crs: Coordinate reference system, or None for unspecified coordinates.
    :param data_name: Name of the point values, defaulting to z.
    :param use_z: Also use the point values as geometric elevations. GeoDataFrames store them in 3D geometry.
    :param auxiliary: Additional named arrays with one value per point.
    :param as_object_type: Returned representation: dataarray, dataset, or geodataframe. None uses
        config["pointcloud_return_type"], which defaults to dataarray.
    :returns: A DataArray with auxiliary coordinates, a Dataset with separate value variables, or a GeoDataFrame.
        Dask inputs remain lazy in every representation.
    """
    from geoutils.pointcloud.xr_accessor import DataArrayPointCloudAccessor, _pointcloud_array_to_dataset

    # Validate the requested output before preparing its coordinates and attributes
    if as_object_type is None:
        as_object_type = config["pointcloud_return_type"]
    if as_object_type not in ("dataarray", "dataset", "geodataframe"):
        raise ValueError("Argument 'as_object_type' must be 'dataarray', 'dataset' or 'geodataframe'.")
    points = DataArrayPointCloudAccessor.from_xyz(x, y, z, crs, data_name, use_z, auxiliary=auxiliary)
    if as_object_type == "dataarray":
        return points
    if as_object_type == "dataset":
        return _pointcloud_array_to_dataset(points)

    # Construct geometries per partition so lazy coordinates and values do not need to be collected
    name = str(points.name)
    point_crs = points.pc.crs
    arrays = [points.data, *(coordinate.data for coordinate in points.coords.values())]
    if any(is_dask_array(array) for array in arrays):
        from geoutils.vector.pd_accessor import _import_dask_geopandas

        dgpd = _import_dask_geopandas()
        frame = points.to_dataset().unify_chunks().to_dask_dataframe().drop(columns="point")
        meta = _point_geometry_partition(frame._meta, name, point_crs, use_z)
        frame = frame.map_partitions(_point_geometry_partition, name, point_crs, use_z, meta=meta)
        frame = dgpd.from_dask_dataframe(frame, geometry="geometry")
    else:
        frame = _point_geometry_partition(points.to_dataset().to_dataframe(), name, point_crs, use_z)

    # Record the known count and geometry type without reading lazy partitions
    return _build_pointcloud_output(
        frame,
        data_name=None if use_z else name,
        as_dataframe=True,
        attrs={"point_count": points.sizes["point"], "geometry_type": "Point Z" if use_z else "Point"},
        preserve_locations=True,
    )


def _point_geometry_partition(frame: pd.DataFrame, data_name: str, crs: Any, use_z: bool) -> gpd.GeoDataFrame:
    """Construct point geometries and attach auxiliary values from one array partition."""
    from geoutils.pointcloud.pd_accessor import GeoDataFramePointCloudAccessor

    # Use the established geometry constructor for both eager data and Dask partitions
    points = GeoDataFramePointCloudAccessor.from_xyz(
        frame["x"].to_numpy(), frame["y"].to_numpy(), frame[data_name].to_numpy(), crs, data_name, use_z
    )
    for name in frame.columns:
        if name not in ("x", "y", data_name):
            points[name] = frame[name].to_numpy()
    points.index = frame.index.rename(None)
    return points
