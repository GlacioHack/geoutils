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

"""Module for the GeoPandas accessor ``pc`` mirroring the PointCloud API."""

from __future__ import annotations

import warnings
from typing import Any

import geopandas as gpd
import pandas as pd
import rasterio as rio
from pyproj import CRS

from geoutils._dispatch import is_dask_dataframe, is_dask_geodataframe
from geoutils._misc import import_optional
from geoutils.pointcloud.base import GeometryPointCloudBase
from geoutils.pointcloud.dataframe import (
    _get_dataframe_attrs,
    _set_dataframe_attrs,
)
from geoutils.pointcloud.las import _write_laspy
from geoutils.vector.pd_accessor import (
    VectorAccessor,
    _replace_geodataframe,
)

_DASK_ACCESSOR_REGISTERED = False


def _register_dask_pointcloud_accessor() -> None:
    """
    Add the ``.pc`` property to Dask DataFrames when lazy point cloud support is first needed.

    Pandas and Dask keep separate lists of dataframe accessors. The Pandas decorator on GeoPandasPointCloudAccessor
    therefore makes ``.pc`` available only on Pandas and GeoPandas objects. This function adds the same accessor to
    Dask objects without importing the optional Dask DataFrame package during ordinary GeoUtils imports.
    """

    global _DASK_ACCESSOR_REGISTERED

    # Register once because the accessor is added to the shared Dask DataFrame class for the rest of the process
    if _DASK_ACCESSOR_REGISTERED:
        return

    # Import Dask only when a lazy point cloud is requested
    # Attach GeoPandasPointCloudAccessor as its ``.pc`` property
    # https://docs.dask.org/en/stable/dataframe-extend.html#accessors
    import_optional("dask")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
        warnings.filterwarnings("ignore", message="registration of accessor.*", category=UserWarning)
        from dask.dataframe.accessor import register_dataframe_accessor

        register_dataframe_accessor("pc")(GeoPandasPointCloudAccessor)

    _DASK_ACCESSOR_REGISTERED = True


def _infer_data_name(ds: Any) -> str | None:
    """Infer a point cloud data attribute from dataframe metadata and columns."""

    attrs = _get_dataframe_attrs(ds)
    if "data_name" in attrs:
        # An explicit None selects elevation from 3D geometry, even when auxiliary columns exist
        return attrs["data_name"]

    nongeo_columns = [c for c in ds.columns if c != "geometry"]
    if "Z" in nongeo_columns:
        return "Z"
    if len(nongeo_columns) == 1:
        return nongeo_columns[0]
    return None


def _validate_point_partition(ds: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reject non-point geometries when a lazy point-cloud partition is computed."""

    if not isinstance(ds, gpd.GeoDataFrame) or not all(geometry_type == "Point" for geometry_type in ds.geom_type):
        raise ValueError("The 'pc' accessor is only available for GeoDataFrames with point geometries.")
    return ds


@pd.api.extensions.register_dataframe_accessor("pc")
class GeoPandasPointCloudAccessor(GeometryPointCloudBase, VectorAccessor):  # type: ignore[misc]
    """
    This class defines the GeoPandas dataframe accessor ``pc`` for point clouds.

    Most attributes and methods are inherited from PointCloudBase through GeometryPointCloudBase, also parent of
    PointCloud. Dataframe-specific methods handle initialization, metadata and file writing.

    The ``data_name`` property selects the active values: it can be None to use elevations stored in 3D geometries.
    """

    _ACCESSOR_OUTPUT = True
    _dataset = VectorAccessor._dataset

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        """Validate the dataframe and infer the point-cloud data attribute."""

        self._name = None

        # Validate the collection now and individual geometries only when unknown partitions are computed
        if is_dask_dataframe(pandas_obj):
            if not is_dask_geodataframe(pandas_obj):
                raise AttributeError("The 'pc' accessor is only available for Dask-GeoPandas GeoDataFrame objects.")
            attrs = _get_dataframe_attrs(pandas_obj)
            geometry_type = attrs.get("geometry_type")
            if geometry_type is not None and "Point" not in geometry_type:
                raise AttributeError("The 'pc' accessor is only available for GeoDataFrames with point geometries.")
            if geometry_type is None:
                pandas_obj = pandas_obj.map_partitions(_validate_point_partition, meta=pandas_obj._meta)
                _set_dataframe_attrs(pandas_obj, attrs)
            self._obj = pandas_obj
            self._data_name = _infer_data_name(pandas_obj)
            return

        # Normalize eager Pandas inputs to a GeoDataFrame with a named geometry column
        if isinstance(pandas_obj, gpd.GeoDataFrame):
            obj = pandas_obj
        elif isinstance(pandas_obj, pd.DataFrame) and "geometry" in pandas_obj.columns:
            obj = gpd.GeoDataFrame(pandas_obj, geometry="geometry", crs=pandas_obj.attrs.get("crs"))
            obj.attrs.update(getattr(pandas_obj, "attrs", {}))
        else:
            raise AttributeError("The 'pc' accessor is only available for point-cloud GeoDataFrame objects.")
        # Point-cloud operations require every eager geometry to be a point
        if not all(p == "Point" for p in obj.geom_type):
            raise AttributeError("The 'pc' accessor is only available for GeoDataFrames with point geometries.")

        # Store the selected data column on both the accessor and its dataframe
        self._obj: gpd.GeoDataFrame = obj
        self._data_name = _infer_data_name(obj)
        if self._data_name is not None:
            attrs = _get_dataframe_attrs(self._obj)
            attrs["data_name"] = self._data_name
            _set_dataframe_attrs(self._obj, attrs)

    @property
    def ds(self) -> gpd.GeoDataFrame | Any:
        """GeoDataFrame of the point cloud."""

        return self._obj

    @ds.setter
    def ds(self, new_ds: gpd.GeoDataFrame | gpd.GeoSeries | Any) -> None:
        """Set a new GeoDataFrame or lazy Dask DataFrame."""

        if is_dask_dataframe(new_ds):
            # Replacing a lazy collection is safe because it does not mutate partitions
            self._obj = new_ds
            return

        if isinstance(new_ds, gpd.GeoSeries):
            new_ds = gpd.GeoDataFrame(geometry=new_ds)
        if not isinstance(new_ds, gpd.GeoDataFrame):
            raise ValueError("The dataset of a point cloud must be set with a GeoSeries or a GeoDataFrame.")

        _replace_geodataframe(self._obj, new_ds)

    @property
    def crs(self) -> CRS:
        """Coordinate reference system of the point cloud."""

        if self._is_dask:
            # Direct Dask-GeoPandas construction retains CRS in geometry metadata without a GeoUtils cache
            return _get_dataframe_attrs(self.ds).get("crs", self.ds.crs)
        return self.ds.crs

    @property
    def bbox(self) -> rio.coords.BoundingBox:
        """Total bounding box of the point cloud."""

        if self._is_dask:
            return _get_dataframe_attrs(self.ds).get("bounds")
        return rio.coords.BoundingBox(*self.ds.total_bounds)

    @property
    def geometry(self) -> gpd.GeoSeries | Any:
        """Point geometry column as an eager or lazy GeoSeries."""

        if self._is_dask:
            return self.ds["geometry"]
        return self.ds.geometry

    def load(self) -> gpd.GeoDataFrame:
        """Compute and return a Dask-backed point cloud as an eager GeoDataFrame."""

        if not self._is_dask:
            raise ValueError("Data are already loaded.")

        # Dask collections are immutable, so return a replacement without changing the caller
        ds = self.ds.compute()
        attrs = _get_dataframe_attrs(self.ds)
        eager = gpd.GeoDataFrame(ds, geometry="geometry", crs=attrs.get("crs"))
        _set_dataframe_attrs(eager, attrs)
        return eager

    def to_las(
        self,
        filename: str,
        version: Any = None,
        point_format: Any = None,
        offsets: tuple[float, float, float] | None = None,
        scales: tuple[float, float, float] | None = None,
        chunks: int | None = None,
        mp_config: Any = None,
        **kwargs: Any,
    ) -> None:
        """
        Write the point cloud to a LAS, LAZ or COPC file.

        :param filename: Path to the output file.
        :param version: LAS file version.
        :param point_format: LAS point format identifier.
        :param offsets: Coordinate offsets for X, Y and Z.
        :param scales: Coordinate scales for X, Y and Z.
        :param chunks: Number of points per sequential write chunk. Dask inputs use their existing partitions.
        :param mp_config: Multiprocessing configuration for writing eager point-cloud chunks in workers. Not supported
            for Dask-backed point clouds.
        :param kwargs: Additional attributes to set on the LasPy header.
        """

        _write_laspy(
            filename=filename,
            pc=self.ds,
            data_name=self.data_name,
            version=version,
            point_format=point_format,
            offsets=offsets,
            scales=scales,
            chunks=chunks,
            mp_config=mp_config,
            **kwargs,
        )
