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
Module for the Pandas accessor ``vct`` mirroring the GeoUtils-specific Vector API.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

from geoutils.vector.base import VectorBase


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


def open_vector(filename: str, **kwargs: Any) -> gpd.GeoDataFrame:
    """
    Open a vector file using GeoPandas.

    :param filename: Path to the vector file to open.
    :param kwargs: Keyword arguments passed to :func:`geopandas.read_file`.
    """

    return gpd.read_file(filename, **kwargs)


@pd.api.extensions.register_dataframe_accessor("vct")
class VectorAccessor(VectorBase):
    """
    Pandas accessor ``vct`` for GeoPandas GeoDataFrames.

    GeoPandas' own attributes and methods remain available directly on the GeoDataFrame. The accessor exposes the
    GeoUtils-specific convenience methods shared with :class:`geoutils.Vector`.
    """

    _ACCESSOR_OUTPUT = True

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        super().__init__()

        if not isinstance(pandas_obj, gpd.GeoDataFrame):
            raise AttributeError("The 'vct' accessor is only available for geopandas.GeoDataFrame objects.")

        self._obj: gpd.GeoDataFrame = pandas_obj

    @property
    def ds(self) -> gpd.GeoDataFrame:
        """GeoDataFrame of the vector."""

        return self._obj

    @ds.setter
    def ds(self, new_ds: gpd.GeoDataFrame | gpd.GeoSeries) -> None:
        """Set a new GeoDataFrame."""

        if isinstance(new_ds, gpd.GeoSeries):
            new_ds = gpd.GeoDataFrame(geometry=new_ds)
        if not isinstance(new_ds, gpd.GeoDataFrame):
            raise ValueError("The dataset of a vector must be set with a GeoSeries or a GeoDataFrame.")

        _replace_geodataframe(self._obj, new_ds)

    def copy(self, deep: bool = True) -> gpd.GeoDataFrame:
        """Return a copy of the vector GeoDataFrame."""

        return self._obj.copy(deep=deep)

    def _override_gdf_output(self, other: gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry | pd.Series | Any) -> Any:
        """Parse outputs of GeoPandas functions to facilitate object manipulation."""

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

        return Vector(self._obj)

    def to_file(self, filename: str, driver: Any = None, schema: Any = None, index: Any = None, **kwargs: Any) -> None:
        """Write the vector to file."""

        self._obj.to_file(filename=filename, driver=driver, schema=schema, index=index, **kwargs)
