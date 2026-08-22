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
Module for the Pandas accessor ``pc`` mirroring the GeoUtils-specific PointCloud API.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import geopandas as gpd
import pandas as pd
import rasterio as rio
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from geoutils.pointcloud.base import (
    PointCloudBase,
    _get_dataframe_attrs,
    _is_dask_dataframe,
    _set_dataframe_attrs,
)
from geoutils.pointcloud.las import (
    is_laspy_supported,
    _load_laspy_data_slice,
    _load_laspy_metadata,
    _write_laspy,
)
from geoutils.vector.pd_accessor import VectorAccessor, _replace_geodataframe

_DASK_ACCESSOR_REGISTERED = False


def _import_dask_dataframe() -> Any:
    """Import Dask DataFrame while suppressing optional dask-expr warnings from older environments."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
        import dask.dataframe as dd

    return dd


def _register_dask_pointcloud_accessor() -> None:
    """Register the ``pc`` accessor on Dask DataFrames lazily."""

    global _DASK_ACCESSOR_REGISTERED

    if _DASK_ACCESSOR_REGISTERED:
        return

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
        warnings.filterwarnings("ignore", message="registration of accessor.*", category=UserWarning)
        from dask.dataframe.accessor import register_dataframe_accessor

        register_dataframe_accessor("pc")(PointCloudAccessor)

    _DASK_ACCESSOR_REGISTERED = True


def _columns_to_load(
    columns: Literal["all", "main"] | list[str],
    data_column: str | None,
    available_columns: pd.Index,
) -> list[str]:
    """Resolve LAS columns to load from user input."""

    if columns == "all":
        columns_to_load = list(available_columns)
    elif columns == "main":
        columns_to_load = [data_column] if data_column is not None else []
    else:
        columns_to_load = columns

    if "Z" not in columns_to_load:
        columns_to_load = ["Z"] + columns_to_load
    return columns_to_load


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


def _empty_las_meta(columns: list[str], crs: CRS | None) -> pd.DataFrame:
    """Build an empty DataFrame used as Dask metadata for LAS chunks."""

    data = {column: pd.Series(dtype="float64") for column in columns}
    data["geometry"] = pd.Series(dtype="object")
    meta = pd.DataFrame(data=data)
    meta.attrs["crs"] = crs
    return meta


def _as_pandas_geometry_dataframe(ds: gpd.GeoDataFrame | pd.DataFrame) -> pd.DataFrame:
    """Convert a GeoDataFrame to a pandas DataFrame with object-typed shapely geometries."""

    df = pd.DataFrame(ds.copy())
    df["geometry"] = pd.Series(list(ds["geometry"]), index=ds.index, dtype="object")
    df.attrs.update(_get_dataframe_attrs(ds))
    if isinstance(ds, gpd.GeoDataFrame):
        df.attrs["crs"] = ds.crs
    return df


def _load_laspy_data_slice_dataframe(filename: str, columns: list[str], start: int, count: int) -> pd.DataFrame:
    """Load a LAS slice as a pandas DataFrame suitable for Dask."""

    return _as_pandas_geometry_dataframe(_load_laspy_data_slice(filename, columns, start, count))


def open_pointcloud(
    filename: str,
    data_column: str | None = None,
    columns: Literal["all", "main"] | list[str] = "main",
    chunks: int | None = None,
) -> gpd.GeoDataFrame | Any:
    """
    Open a point cloud as a GeoDataFrame or a lazy Dask DataFrame.

    LAS/LAZ files are opened through LasPy. Passing ``chunks`` returns a Dask DataFrame whose partitions are LAS point
    slices read lazily.
    """

    from geoutils.pointcloud.pointcloud import PointCloud

    if chunks is not None and chunks <= 0:
        raise ValueError("Argument 'chunks' must be a strictly positive integer.")

    is_las = is_laspy_supported(filename)

    if not is_las:
        pc = PointCloud(filename, data_column=data_column)
        pc.ds.attrs["data_column"] = pc.data_column
        if chunks is None:
            return pc.ds

        dd = _import_dask_dataframe()
        _register_dask_pointcloud_accessor()
        df = _as_pandas_geometry_dataframe(pc.ds)
        npartitions = max(1, int(len(pc.ds) / chunks) + int(len(pc.ds) % chunks != 0))
        import dask

        with dask.config.set({"dataframe.convert-string": False}):
            ddf = dd.from_pandas(df, npartitions=npartitions)
        _set_dataframe_attrs(
            ddf,
            {
                "crs": pc.crs,
                "bounds": pc.bounds,
                "point_count": pc.point_count,
                "data_column": pc.data_column,
            },
        )
        return ddf

    if data_column is None:
        data_column = "Z"

    crs, point_count, bounds, available_columns = _load_laspy_metadata(filename)
    if data_column not in available_columns:
        raise ValueError(
            f"Data column {data_column} not found among columns. Available columns are: "
            f"{', '.join(available_columns)}."
        )
    columns_to_load = _columns_to_load(columns=columns, data_column=data_column, available_columns=available_columns)

    if chunks is None:
        pc = PointCloud(filename, data_column=data_column)
        pc.load(columns=columns, mp_config=None)
        pc.ds.attrs["data_column"] = pc.data_column
        return pc.ds

    dd = _import_dask_dataframe()
    _register_dask_pointcloud_accessor()
    from dask import delayed

    starts = list(range(0, point_count, chunks))
    parts = [
        delayed(_load_laspy_data_slice_dataframe)(filename, columns_to_load, start, min(chunks, point_count - start))
        for start in starts
    ]
    import dask

    with dask.config.set({"dataframe.convert-string": False}):
        ddf = dd.from_delayed(parts, meta=_empty_las_meta(columns_to_load, crs=crs))
    _set_dataframe_attrs(
        ddf,
        {
            "crs": crs,
            "bounds": bounds,
            "point_count": point_count,
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
        self._name = None

        if _is_dask_dataframe(pandas_obj):
            self._obj = pandas_obj
            self._data_column = _infer_data_column(pandas_obj)
            return

        if isinstance(pandas_obj, gpd.GeoDataFrame):
            obj = pandas_obj
        elif isinstance(pandas_obj, pd.DataFrame) and "geometry" in pandas_obj.columns:
            obj = gpd.GeoDataFrame(pandas_obj, geometry="geometry", crs=pandas_obj.attrs.get("crs"))
            obj.attrs.update(getattr(pandas_obj, "attrs", {}))
        else:
            raise AttributeError("The 'pc' accessor is only available for point-cloud GeoDataFrame objects.")
        if not all(p == "Point" for p in obj.geom_type):
            raise AttributeError("The 'pc' accessor is only available for GeoDataFrames with point geometries.")

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

        if _is_dask_dataframe(new_ds):
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
        return self.ds.columns

    @property
    def geometry(self) -> gpd.GeoSeries | Any:
        if self._is_dask:
            return self.ds["geometry"]
        return self.ds.geometry

    def _override_gdf_output(self, other: gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry | pd.Series | Any) -> Any:
        """Parse outputs of GeoPandas functions to facilitate object manipulation."""

        if _is_dask_dataframe(other):
            attrs = _get_dataframe_attrs(self.ds)
            attrs["data_column"] = self.data_column
            _set_dataframe_attrs(other, attrs)
            return other
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

        ds = self.ds.compute()
        attrs = _get_dataframe_attrs(self.ds)
        self._obj = gpd.GeoDataFrame(ds, geometry="geometry", crs=attrs.get("crs"))
        _set_dataframe_attrs(self._obj, attrs)

    def to_geoutils(self) -> Any:
        """Convert to a GeoUtils PointCloud object."""

        from geoutils.pointcloud.pointcloud import PointCloud

        if self._is_dask:
            ds = self.ds.compute()
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
        """Write the point cloud to LAS/LAZ/COPC file."""

        if self._is_dask:
            ds = self.ds
        else:
            ds = self.ds
        _write_laspy(
            filename=filename,
            pc=ds,
            data_column=self.data_column,
            version=version,
            point_format=point_format,
            offsets=offsets,
            scales=scales,
            chunks=chunks,
            mp_config=mp_config,
            **kwargs,
        )
