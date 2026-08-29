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

"""
Module for the Pandas accessor ``pc`` mirroring the PointCloud API.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import geopandas as gpd
import pandas as pd
import pyogrio
import rasterio as rio
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from geoutils._dispatch import is_dask_dataframe
from geoutils._misc import import_optional
from geoutils.pointcloud.base import (
    PointCloudBase,
    _get_dataframe_attrs,
    _set_dataframe_attrs,
)
from geoutils.pointcloud.las import (
    _empty_las_geodataframe,
    _write_laspy,
    is_laspy_supported,
    load_laspy_data_slice,
    load_laspy_metadata,
    resolve_las_columns,
)
from geoutils.vector.pd_accessor import (
    VectorAccessor,
    _import_dask_geopandas,
    _register_dask_vector_accessor,
    _replace_geodataframe,
)

_DASK_ACCESSOR_REGISTERED = False


def _import_dask_dataframe() -> Any:
    """Import Dask DataFrame while suppressing optional dask-expr warnings from older environments."""

    # Delay the optional import until lazy LAS partitions are requested
    import_optional("dask")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
        import dask.dataframe as dd

    return dd


def _register_dask_pointcloud_accessor() -> None:
    """Register the ``pc`` accessor on Dask DataFrames lazily."""

    global _DASK_ACCESSOR_REGISTERED

    # Dask warns if the same accessor is registered more than once
    if _DASK_ACCESSOR_REGISTERED:
        return

    # Register only after Dask is available so normal imports remain lightweight
    import_optional("dask")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
        warnings.filterwarnings("ignore", message="registration of accessor.*", category=UserWarning)
        from dask.dataframe.accessor import register_dataframe_accessor

        register_dataframe_accessor("pc")(PointCloudAccessor)

    _DASK_ACCESSOR_REGISTERED = True


def _infer_data_column(ds: Any) -> str | None:
    """Infer a point cloud data column from dataframe metadata and columns."""

    data_column = _get_dataframe_attrs(ds).get("data_column")
    if data_column is not None:
        return data_column

    nongeo_columns = [c for c in ds.columns if c != "geometry"]
    if "Z" in nongeo_columns:
        return "Z"
    if len(nongeo_columns) == 1:
        return nongeo_columns[0]
    return None


def _load_laspy_data_slice_dataframe(filename: str, columns: list[str], start: int, count: int) -> gpd.GeoDataFrame:
    """Adapt the common LAS point-slice reader into one indexed Dask partition."""

    # Give every partition its source row range so indexes stay unique after assembly
    ds = load_laspy_data_slice(filename, columns, start, count)
    ds.index = pd.RangeIndex(start, start + count)
    return ds


def _set_pointcloud_attrs_from_file(ds: Any, filename: str, data_column: str | None) -> None:
    """Set point-cloud metadata on a lazy dataframe opened from a vector file."""

    # Pyogrio exposes file metadata without asking Dask to compute feature partitions
    info = pyogrio.read_info(filename)
    geom_type = info.get("geometry_type")
    if geom_type is not None and "Point" not in geom_type:
        raise ValueError("This vector file contains non-point geometries, cannot be instantiated as a point cloud.")
    if data_column is not None and data_column not in info.get("fields", []):
        raise ValueError(
            f"Data column {data_column} not found among columns. Available columns "
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
            "data_column": data_column,
        },
    )


def open_pointcloud(
    filename: str,
    data_column: str | None = None,
    columns: Literal["all", "main"] | list[str] = "main",
    chunks: int | None = None,
) -> gpd.GeoDataFrame | Any:
    """
    Open a point cloud as a GeoDataFrame or a lazy Dask-GeoPandas GeoDataFrame if ``chunks`` is passed.

    LAS, LAZ and COPC files are read through LasPy.
    Other supported vector formats are read through PyOGRIO and GeoPandas.

    :param filename: Path to the point-cloud file to open.
    :param data_column: Column containing point values. For LAS, LAZ and COPC files, defaults to the native ``Z``
        dimension.
    :param columns: LAS dimensions to read. ``main`` reads the data column, ``all`` reads every dimension, and a list
        selects specific dimensions. Ignored for other vector formats.
    :param chunks: Number of points or features per Dask partition. If None, load eagerly into one GeoDataFrame.
    :returns: An eager GeoDataFrame, or a lazy Dask-GeoPandas GeoDataFrame when ``chunks`` is passed.
    """

    from geoutils.pointcloud.pointcloud import PointCloud

    if chunks is not None and chunks <= 0:
        raise ValueError("Argument 'chunks' must be a strictly positive integer.")

    # LAS needs its own slice reader while regular vector formats use GeoPandas
    is_las = is_laspy_supported(filename)

    if not is_las:
        if chunks is None:
            # Preserve the established eager PointCloud loading and validation path
            pc = PointCloud(filename, data_column=data_column)
            pc.ds.attrs["data_column"] = pc.data_column
            return pc.ds

        # Dask-GeoPandas creates file partitions without loading all features
        dgpd = _import_dask_geopandas()
        _register_dask_vector_accessor()
        _register_dask_pointcloud_accessor()
        dgdf = dgpd.read_file(filename, chunksize=chunks)
        _set_pointcloud_attrs_from_file(dgdf, filename=filename, data_column=data_column)
        return dgdf

    # Native LAS Z values are the default point-cloud data
    if data_column is None:
        data_column = "Z"

    # Resolve requested dimensions entirely from the LAS header
    metadata = load_laspy_metadata(filename)
    if data_column not in metadata.columns:
        raise ValueError(
            f"Data column {data_column} not found among columns. Available columns are: "
            f"{', '.join(metadata.columns)}."
        )
    columns_to_load = resolve_las_columns(
        columns=columns,
        data_column=data_column,
        available_columns=metadata.columns,
    )

    if chunks is None:
        # The eager path loads all requested LAS dimensions into one GeoDataFrame
        pc = PointCloud(filename, data_column=data_column)
        pc.load(columns=columns, mp_config=None)
        pc.ds.attrs["data_column"] = pc.data_column
        return pc.ds

    # Load optional Dask components only for partitioned LAS output
    dd = _import_dask_dataframe()
    dgpd = _import_dask_geopandas()
    _register_dask_vector_accessor()
    _register_dask_pointcloud_accessor()
    dask = import_optional("dask")
    delayed = dask.delayed

    # Represent every contiguous LAS row slice as one delayed partition
    starts = list(range(0, metadata.point_count, chunks))
    parts = [
        delayed(_load_laspy_data_slice_dataframe)(
            filename,
            columns_to_load,
            start,
            min(chunks, metadata.point_count - start),
        )
        for start in starts
    ]
    # Keep LAS numeric dimensions unchanged while assembling the lazy dataframe
    with dask.config.set({"dataframe.convert-string": False}):
        ddf = dd.from_delayed(parts, meta=_empty_las_geodataframe(columns_to_load, crs=metadata.crs))

    # Add geospatial behavior and cache header metadata for accessor properties
    ddf = dgpd.from_dask_dataframe(ddf, geometry="geometry")
    _set_dataframe_attrs(
        ddf,
        {
            "crs": metadata.crs,
            "bounds": metadata.bounds,
            "point_count": metadata.point_count,
            "data_column": data_column,
        },
    )
    return ddf


@pd.api.extensions.register_dataframe_accessor("pc")
class PointCloudAccessor(PointCloudBase, VectorAccessor):
    """
    Pandas accessor ``pc`` for point-cloud GeoDataFrames.
    """

    _ACCESSOR_OUTPUT = True

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        """Validate the dataframe and infer the point-cloud data column."""

        self._name = None

        # Dask validation relies on planned columns because partitions are still lazy
        if is_dask_dataframe(pandas_obj):
            self._obj = pandas_obj
            self._data_column = _infer_data_column(pandas_obj)
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
        self._data_column = _infer_data_column(obj)
        if self._data_column is not None:
            attrs = _get_dataframe_attrs(self._obj)
            attrs["data_column"] = self._data_column
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
            return _get_dataframe_attrs(self.ds).get("crs")
        return self.ds.crs

    @property
    def bounds(self) -> rio.coords.BoundingBox:
        """Total bounding box of the point cloud."""

        if self._is_dask:
            return _get_dataframe_attrs(self.ds).get("bounds")
        return rio.coords.BoundingBox(*self.ds.total_bounds)

    @property
    def columns(self) -> pd.Index:
        """Column names available on the point-cloud dataframe."""

        return self.ds.columns

    @property
    def geometry(self) -> gpd.GeoSeries | Any:
        """Point geometry column as an eager or lazy GeoSeries."""

        if self._is_dask:
            return self.ds["geometry"]
        return self.ds.geometry

    def _override_gdf_output(self, other: gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry | pd.Series | Any) -> Any:
        """Parse outputs of GeoPandas functions to facilitate object manipulation."""

        # Propagate cached metadata through lazy dataframe operations
        if is_dask_dataframe(other):
            attrs = _get_dataframe_attrs(self.ds)
            old_crs = attrs.get("crs")
            new_crs = getattr(other, "crs", None)
            if new_crs is not None:
                attrs["crs"] = new_crs
                if old_crs != new_crs:
                    attrs["bounds"] = None
            attrs["data_column"] = self.data_column
            _set_dataframe_attrs(other, attrs)
            return other
        # Normalize eager GeoPandas outputs to the public accessor representation
        if not isinstance(other, (gpd.GeoDataFrame, gpd.GeoSeries, pd.Series, BaseGeometry)):
            raise ValueError("Not implemented. This error should only be raised in tests.")

        if isinstance(other, gpd.GeoDataFrame):
            attrs = _get_dataframe_attrs(other)
            attrs["data_column"] = self.data_column
            _set_dataframe_attrs(other, attrs)
            return other
        if isinstance(other, gpd.GeoSeries):
            return gpd.GeoDataFrame(geometry=other)
        if isinstance(other, BaseGeometry):
            return gpd.GeoDataFrame({"geometry": [other]}, crs=self.crs)
        return other

    def load(self) -> None:
        """Compute a Dask-backed point cloud in-place."""

        if not self._is_dask:
            raise ValueError("Data are already loaded.")

        # Materialize all partitions once, then replace the accessor source in place
        ds = self.ds.compute()
        attrs = _get_dataframe_attrs(self.ds)
        self._obj = gpd.GeoDataFrame(ds, geometry="geometry", crs=attrs.get("crs"))
        _set_dataframe_attrs(self._obj, attrs)

    def to_geoutils(self) -> Any:
        """Convert to a GeoUtils PointCloud object."""

        from geoutils.pointcloud.pointcloud import PointCloud

        # A PointCloud is eager by design, so lazy partitions must be computed here
        if self._is_dask:
            ds = self.ds.compute()
            if not isinstance(ds, gpd.GeoDataFrame):
                ds = gpd.GeoDataFrame(ds, geometry="geometry", crs=self.crs)
        else:
            ds = self.ds
        return PointCloud(ds, data_column=self.data_column)

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

        # The common writer streams Dask partitions or eager chunks as appropriate
        _write_laspy(
            filename=filename,
            pc=self.ds,
            data_column=self.data_column,
            version=version,
            point_format=point_format,
            offsets=offsets,
            scales=scales,
            chunks=chunks,
            mp_config=mp_config,
            **kwargs,
        )
