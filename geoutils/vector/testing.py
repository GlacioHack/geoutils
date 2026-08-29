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

"""Testing helpers for eager and Dask-backed vector dataframes."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable
from functools import partial
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas.testing import assert_geodataframe_equal

from geoutils._dispatch import get_geo_attr, is_dask_dataframe
from geoutils._misc import import_optional


def _get_dataframe(obj: Any) -> Any:
    """Return an eager or Dask GeoDataFrame without computing it."""

    ds = obj if isinstance(obj, gpd.GeoDataFrame) or is_dask_dataframe(obj) else get_geo_attr(obj, "ds")
    if not isinstance(ds, gpd.GeoDataFrame) and not is_dask_dataframe(ds):
        raise TypeError(f"Expected a Vector or GeoDataFrame, received {type(obj).__name__}.")
    return ds


def _geometry_allclose(left: Any, right: Any, rtol: float, atol: float) -> bool:
    """Return whether two geometries have the same structure and numerically close coordinates."""

    from shapely import get_coordinates

    if left is None or right is None:
        return left is right
    if left.geom_type != right.geom_type:
        return False

    left_coordinates = get_coordinates(left, include_z=True)
    right_coordinates = get_coordinates(right, include_z=True)
    if left_coordinates.shape != right_coordinates.shape or not np.allclose(
        left_coordinates, right_coordinates, rtol=rtol, atol=atol, equal_nan=True
    ):
        return False

    # ``equals_exact`` additionally checks the geometry and ring/component structure
    coordinate_scale = max(
        float(np.nanmax(np.abs(left_coordinates), initial=0)),
        float(np.nanmax(np.abs(right_coordinates), initial=0)),
    )
    return bool(left.equals_exact(right, tolerance=atol + rtol * coordinate_scale))


def _vector_allclose_eager(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    rtol: float,
    atol: float,
    check_dtype: bool,
) -> bool:
    """Compare two eager vector partitions with numeric tolerances."""

    if (
        left.shape != right.shape
        or not left.columns.equals(right.columns)
        or not left.index.equals(right.index)
        or left.crs != right.crs
        or left.active_geometry_name != right.active_geometry_name
    ):
        return False

    geometry_name = left.active_geometry_name
    for column in left.columns:
        if check_dtype and left[column].dtype != right[column].dtype:
            return False
        if column == geometry_name:
            if not all(
                _geometry_allclose(left_geometry, right_geometry, rtol=rtol, atol=atol)
                for left_geometry, right_geometry in zip(left.geometry, right.geometry)
            ):
                return False
        elif pd.api.types.is_numeric_dtype(left[column].dtype) and pd.api.types.is_numeric_dtype(right[column].dtype):
            if not np.allclose(left[column], right[column], rtol=rtol, atol=atol, equal_nan=True):
                return False
        elif not left[column].equals(right[column]):
            return False

    return True


def _vector_equal_eager(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, kwargs: dict[str, Any]) -> bool:
    """Compare two eager vector partitions exactly."""

    try:
        assert_geodataframe_equal(left, right, **kwargs)
    except (AssertionError, AttributeError, TypeError):
        return False
    return True


def _dataframe_partition_lengths(ds: Any) -> list[int]:
    """Return row counts for eager or Dask dataframe partitions."""

    if not is_dask_dataframe(ds):
        return [len(ds)]
    lengths = ds.map_partitions(len, meta=("length", "int64")).compute()
    return [int(length) for length in lengths]


def _slice_dataframe_partition(partition: gpd.GeoDataFrame, start: int, stop: int) -> gpd.GeoDataFrame:
    """Slice one eager dataframe partition by row position."""

    return partition.iloc[start:stop]


def _compare_dataframes_partitionwise(
    left: Any,
    right: Any,
    comparator: Callable[[gpd.GeoDataFrame, gpd.GeoDataFrame], bool],
) -> bool:
    """Compare matching row ranges without collecting Dask dataframes on the client."""

    if not is_dask_dataframe(left) and not is_dask_dataframe(right):
        return comparator(left, right)

    # Partition lengths are small scalar results and let differently chunked inputs be aligned by row position
    left_lengths = _dataframe_partition_lengths(left)
    right_lengths = _dataframe_partition_lengths(right)
    if sum(left_lengths) != sum(right_lengths):
        return False

    dask = import_optional("dask")
    left_parts = list(left.to_delayed()) if is_dask_dataframe(left) else [dask.delayed(left)]
    right_parts = list(right.to_delayed()) if is_dask_dataframe(right) else [dask.delayed(right)]
    left_edges = np.cumsum([0, *left_lengths]).tolist()
    right_edges = np.cumsum([0, *right_lengths]).tolist()
    boundaries = sorted(set(left_edges + right_edges))

    if len(boundaries) == 1:
        left_empty = left._meta if is_dask_dataframe(left) else left
        right_empty = right._meta if is_dask_dataframe(right) else right
        return comparator(left_empty, right_empty)

    comparisons = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        left_index = bisect_right(left_edges, start) - 1
        right_index = bisect_right(right_edges, start) - 1
        left_slice = dask.delayed(_slice_dataframe_partition)(
            left_parts[left_index], start - left_edges[left_index], stop - left_edges[left_index]
        )
        right_slice = dask.delayed(_slice_dataframe_partition)(
            right_parts[right_index], start - right_edges[right_index], stop - right_edges[right_index]
        )
        comparisons.append(dask.delayed(comparator)(left_slice, right_slice))

    return all(dask.compute(*comparisons))


def _vector_equal(left_obj: Any, right_obj: Any, **kwargs: Any) -> bool:
    """Compare eager or Dask-backed vector-like objects exactly."""

    try:
        left = _get_dataframe(left_obj)
        right = _get_dataframe(right_obj)
    except (AttributeError, TypeError):
        return False
    comparator = partial(_vector_equal_eager, kwargs=kwargs)
    return _compare_dataframes_partitionwise(left, right, comparator=comparator)


def _vector_allclose(left_obj: Any, right_obj: Any, rtol: float, atol: float, **kwargs: Any) -> bool:
    """Compare eager or Dask-backed vector-like objects with numeric tolerances."""

    try:
        left = _get_dataframe(left_obj)
        right = _get_dataframe(right_obj)
    except (AttributeError, TypeError):
        return False

    # Schema and geospatial metadata can be checked without evaluating Dask partitions
    if not left.columns.equals(right.columns) or left.crs != right.crs or left.geometry.name != right.geometry.name:
        return False

    check_dtype = kwargs.get("check_dtype", True)
    comparator = partial(_vector_allclose_eager, rtol=rtol, atol=atol, check_dtype=check_dtype)
    return _compare_dataframes_partitionwise(left, right, comparator=comparator)
