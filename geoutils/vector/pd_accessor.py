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

import warnings
from typing import Any

import geopandas as gpd
import pandas as pd

from geoutils._dispatch import is_dask_dataframe, is_dask_geodataframe
from geoutils._misc import import_optional
from geoutils.vector.base import VectorBase
from geoutils.vector.io import _import_dask_geopandas as _import_dask_geopandas
from geoutils.vector.io import _write_vector_file
from geoutils.vector.io import open_vector as open_vector

_DASK_ACCESSOR_REGISTERED = False


def _register_dask_vector_accessor() -> None:
    """
    Add the ``.vct`` property to Dask DataFrames when lazy vector support is first needed.

    The Pandas decorator on GeoDataFrameVectorAccessor does not affect Dask DataFrames because Dask keeps its own
    accessor list. Registering here gives lazy GeoDataFrames the same GeoUtils vector API while keeping Dask DataFrame
    optional until a caller requests chunked vector data.
    """

    global _DASK_ACCESSOR_REGISTERED

    # Register once because the accessor is added to the shared Dask DataFrame class for the rest of the process
    if _DASK_ACCESSOR_REGISTERED:
        return

    # Import Dask only when a lazy vector is requested, then attach GeoDataFrameVectorAccessor as its ``.vct`` property
    import_optional("dask")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
        warnings.filterwarnings("ignore", message="registration of accessor.*", category=UserWarning)
        from dask.dataframe.accessor import register_dataframe_accessor

        register_dataframe_accessor("vct")(GeoDataFrameVectorAccessor)

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


@pd.api.extensions.register_dataframe_accessor("vct")
class GeoDataFrameVectorAccessor(VectorBase):
    """
    Pandas accessor ``vct`` for GeoPandas GeoDataFrames.

    GeoPandas' own attributes and methods remain available directly on the GeoDataFrame. The accessor exposes the
    GeoUtils-specific convenience methods shared with :class:`geoutils.Vector`.
    """

    _ACCESSOR_OUTPUT = True

    @property
    def _dataset(self) -> Any:
        return self.ds

    @_dataset.setter
    def _dataset(self, new_ds: Any) -> None:
        self.ds = new_ds

    def __init__(self, pandas_obj: Any) -> None:
        """Validate and retain an in-memory or lazy geospatial dataframe."""

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
        """
        Return a copy of the vector GeoDataFrame.

        :param deep: Whether to copy in-memory data deeply. Dask-backed dataframes always copy their task graph.
        :returns: A copy with the same in-memory or lazy dataframe type.
        """

        # Dask copies its task graph and does not expose Pandas' deep-copy option
        if is_dask_dataframe(self._obj):
            return self._obj.copy()
        return self._obj.copy(deep=deep)

    def to_geoutils(self) -> Any:
        """Convert to a GeoUtils Vector object."""

        from geoutils.vector.vector import Vector

        # Compute lazy partitions when converting to a Vector
        ds = self._obj.compute() if is_dask_dataframe(self._obj) else self._obj
        return Vector(ds)

    def to_file(
        self,
        filename: str,
        driver: Any = None,
        schema: Any = None,
        index: Any = None,
        *,
        filesystem: Any = None,
        storage_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Write the vector to a vector file or GeoParquet dataset.

        :param filename: Path to the output file or dataset.
        :param driver: Output format driver. By default, inferred from ``filename``.
        :param schema: Schema passed to GeoPandas for vector formats. Not supported for Dask GeoParquet.
        :param index: Whether to write the dataframe index.
        :param filesystem: An existing Fsspec filesystem object whose connection settings are used to write files.
            Cannot be combined with storage_options.
        :param storage_options: Dictionary of connection settings used to create a Fsspec filesystem from the
            destination URL, such as credentials or a custom server address. Cannot be combined with filesystem.
        :param kwargs: Additional keyword arguments passed to GeoPandas or Dask-GeoPandas.
        """

        _write_vector_file(
            self._obj,
            filename=filename,
            driver=driver,
            schema=schema,
            index=index,
            filesystem=filesystem,
            storage_options=storage_options,
            **kwargs,
        )
