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
Module for the Pandas accessor ``vct`` mirroring the Vector API.
"""

from __future__ import annotations

import pathlib
import warnings
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

from geoutils._dispatch import is_dask_dataframe, is_dask_geodataframe
from geoutils._misc import import_optional
from geoutils.vector.base import VectorBase

_DASK_ACCESSOR_REGISTERED = False


def _import_dask_geopandas() -> Any:
    """Import Dask-GeoPandas as an optional dependency."""

    # Delay the import until the caller explicitly requests chunks
    import_optional("dask_geopandas", package_name="dask-geopandas")
    import dask_geopandas as dgpd

    return dgpd


def _register_dask_vector_accessor() -> None:
    """Register the ``vct`` accessor on Dask DataFrames lazily."""

    global _DASK_ACCESSOR_REGISTERED

    # Dask warns if the same accessor is registered more than once
    if _DASK_ACCESSOR_REGISTERED:
        return

    # Register only after Dask is available so the dependency remains optional
    import_optional("dask")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
        warnings.filterwarnings("ignore", message="registration of accessor.*", category=UserWarning)
        from dask.dataframe.accessor import register_dataframe_accessor

        register_dataframe_accessor("vct")(VectorAccessor)

    _DASK_ACCESSOR_REGISTERED = True


def _replace_geodataframe(target: gpd.GeoDataFrame, source: gpd.GeoDataFrame) -> None:
    """Replace a GeoDataFrame's contents in-place while keeping the same Python object."""

    target.drop(index=target.index, inplace=True)
    target.drop(columns=list(target.columns), inplace=True)

    for col in source.columns:
        target[col] = source[col].copy()

    target.index = source.index.copy()
    target.set_geometry(source.geometry.name, inplace=True)
    if source.crs is not None:
        target.set_crs(source.crs, allow_override=True, inplace=True)


def open_vector(filename: str, chunks: int | None = None, **kwargs: Any) -> gpd.GeoDataFrame | Any:
    """
    Open a vector file using GeoPandas, or lazily with Dask-GeoPandas when passing ``chunks``.

    :param filename: Path to the vector file to open.
    :param chunks: Number of features per Dask partition. If None, load eagerly with GeoPandas.
    :param kwargs: Keyword arguments passed to :func:`geopandas.read_file`.
    """

    if chunks is not None:
        # A positive feature count defines the target size of each lazy partition
        if chunks <= 0:
            raise ValueError("Argument 'chunks' must be a strictly positive integer.")
        dgpd = _import_dask_geopandas()
        _register_dask_vector_accessor()
        return dgpd.read_file(filename, chunksize=chunks, **kwargs)

    # Without chunks, keep GeoPandas' regular eager behavior
    return gpd.read_file(filename, **kwargs)


@pd.api.extensions.register_dataframe_accessor("vct")
class VectorAccessor(VectorBase):
    """
    Pandas accessor ``vct`` for GeoPandas GeoDataFrames.

    GeoPandas' own attributes and methods remain available directly on the GeoDataFrame. The accessor exposes the
    GeoUtils-specific convenience methods shared with :class:`geoutils.Vector`.
    """

    _ACCESSOR_OUTPUT = True

    def __init__(self, pandas_obj: Any) -> None:
        """Validate and retain an eager or lazy geospatial dataframe."""

        super().__init__()

        if not isinstance(pandas_obj, gpd.GeoDataFrame) and not is_dask_geodataframe(pandas_obj):
            raise AttributeError(
                "The 'vct' accessor is only available for GeoPandas or Dask-GeoPandas GeoDataFrame objects."
            )

        # Accessor methods operate directly on this dataframe-like object
        self._obj: gpd.GeoDataFrame | Any = pandas_obj

    @property
    def ds(self) -> gpd.GeoDataFrame | Any:
        """GeoDataFrame of the vector."""

        return self._obj

    @ds.setter
    def ds(self, new_ds: gpd.GeoDataFrame | gpd.GeoSeries | Any) -> None:
        """Set a new GeoDataFrame."""

        if is_dask_geodataframe(new_ds):
            # Replacing a lazy collection does not evaluate or mutate its partitions
            self._obj = new_ds
            return

        if isinstance(new_ds, gpd.GeoSeries):
            new_ds = gpd.GeoDataFrame(geometry=new_ds)
        if not isinstance(new_ds, gpd.GeoDataFrame):
            raise ValueError("The dataset of a vector must be set with a GeoSeries or a GeoDataFrame.")

        _replace_geodataframe(self._obj, new_ds)

    def copy(self, deep: bool = True) -> gpd.GeoDataFrame:
        """Return a copy of the vector GeoDataFrame."""

        # Dask copies its task graph and does not expose Pandas' deep-copy option
        if is_dask_dataframe(self._obj):
            return self._obj.copy()
        return self._obj.copy(deep=deep)

    def _override_gdf_output(self, other: gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry | pd.Series | Any) -> Any:
        """Parse outputs of GeoPandas functions to facilitate object manipulation."""

        if is_dask_dataframe(other):
            return other

        if not isinstance(other, (gpd.GeoDataFrame, gpd.GeoSeries, pd.Series, BaseGeometry)):
            raise ValueError("Not implemented. This error should only be raised in tests.")

        if isinstance(other, gpd.GeoDataFrame):
            return other
        if isinstance(other, gpd.GeoSeries):
            return gpd.GeoDataFrame(geometry=other)
        if isinstance(other, BaseGeometry):
            return gpd.GeoDataFrame({"geometry": [other]}, crs=self.crs)
        return other

    def to_geoutils(self) -> Any:
        """Convert to a GeoUtils Vector object."""

        from geoutils.vector.vector import Vector

        # Vector is eager by design, so compute lazy partitions only at conversion time
        ds = self._obj.compute() if is_dask_dataframe(self._obj) else self._obj
        return Vector(ds)

    def to_file(self, filename: str, driver: Any = None, schema: Any = None, index: Any = None, **kwargs: Any) -> None:
        """Write the vector to file."""

        if is_dask_dataframe(self._obj):
            # GeoParquet supports direct partitioned output from Dask-GeoPandas
            suffix = pathlib.Path(filename).suffix.lower()
            driver_name = str(driver).lower() if driver is not None else None
            if suffix in [".parquet", ".pq"] or driver_name in ["parquet", "geoparquet"]:
                if schema is not None:
                    raise ValueError("Argument 'schema' is not supported for Dask GeoParquet output.")
                if driver is not None:
                    kwargs.pop("driver", None)
                if index is not None:
                    kwargs["write_index"] = index
                else:
                    kwargs.setdefault("write_index", False)
                self._obj.to_parquet(filename, **kwargs)
                return

            # Other vector formats require one eager GeoDataFrame for GeoPandas writing
            ds = self._obj.compute()
        else:
            ds = self._obj

        ds.to_file(filename=filename, driver=driver, schema=schema, index=index, **kwargs)
