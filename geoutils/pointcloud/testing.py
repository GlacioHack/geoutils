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

"""Testing helpers for eager and Dask-backed point-cloud dataframes."""

from __future__ import annotations

import warnings
from typing import Any

import geopandas as gpd
import numpy as np
from pyproj import CRS

from geoutils._dispatch import _get_pointcloud_interface, get_geo_attr
from geoutils.vector.testing import (
    _compare_dataframes_partitionwise,
    _get_dataframe,
)


def _point_coords_equal_eager(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame) -> bool:
    """Compare ordered X/Y coordinates in two eager point partitions."""

    if len(left) != len(right):
        return False
    return bool(
        np.array_equal(left.geometry.x.to_numpy(), right.geometry.x.to_numpy())
        and np.array_equal(left.geometry.y.to_numpy(), right.geometry.y.to_numpy())
    )


def _georeferenced_coords_equal(left_obj: Any, right_obj: Any, warn_3d_crs: bool = True) -> bool:
    """Compare point-cloud horizontal coordinates and CRS without collecting Dask dataframes."""

    try:
        left_crs = get_geo_attr(left_obj, "crs")
        right_crs = get_geo_attr(right_obj, "crs")
        left_interface = _get_pointcloud_interface(left_obj)
        right_interface = _get_pointcloud_interface(right_obj)
        array_points = getattr(left_interface, "_is_xr", False) or getattr(right_interface, "_is_xr", False)
        if not array_points:
            left = _get_dataframe(left_obj)
            right = _get_dataframe(right_obj)
    except (AttributeError, TypeError):
        return False

    # Compare horizontal CRS because vertical coordinates do not define point locations in the XY plane
    left_crs = CRS(left_crs) if left_crs is not None else None
    right_crs = CRS(right_crs) if right_crs is not None else None
    left_crs2d = left_crs.to_2d() if left_crs is not None else None
    right_crs2d = right_crs.to_2d() if right_crs is not None else None
    if left_crs2d != right_crs2d:
        return False

    if array_points:
        if left_interface is None or right_interface is None:
            return False
        pairs = zip(left_interface.to_xyz()[:2], right_interface.to_xyz()[:2])
        same_coordinates = True
        for left_axis, right_axis in pairs:
            if hasattr(left_axis, "to_dask_array"):
                left_axis = left_axis.to_dask_array(lengths=True)
            if hasattr(right_axis, "to_dask_array"):
                right_axis = right_axis.to_dask_array(lengths=True)
            if left_axis.shape != right_axis.shape:
                return False
            equal = np.all(left_axis == right_axis)
            same_coordinates &= bool(equal.compute() if hasattr(equal, "compute") else equal)
    else:
        same_coordinates = _compare_dataframes_partitionwise(left, right, comparator=_point_coords_equal_eager)

    # Report a vertical difference without treating it as different horizontal coordinates
    if same_coordinates and left_crs is not None and right_crs is not None and left_crs != right_crs and warn_3d_crs:
        warnings.warn(
            "The two point clouds have the same 2D CRS but a different vertical CRS: "
            f"{left_crs.name} and {right_crs.name}.",
            category=UserWarning,
        )

    return same_coordinates


def _compare_array_points(
    left_obj: Any, right_obj: Any, *, rtol: float | None = None, atol: float = 1e-8, check_dtype: bool = True
) -> bool:
    """Compare point arrays by row index, CRS, value names and attributes across storage representations."""
    left_interface = _get_pointcloud_interface(left_obj)
    right_interface = _get_pointcloud_interface(right_obj)
    if left_interface is None or right_interface is None or left_interface.crs != right_interface.crs:
        return False
    left, right = left_interface.to_xarray().compute(), right_interface.to_xarray().compute()

    # Dimension names and implicit range coordinates do not change ordered point locations
    left_index = left.coords[left.dims[0]].data if left.dims[0] in left.coords else left.get_index(left.dims[0])
    right_index = right.coords[right.dims[0]].data if right.dims[0] in right.coords else right.get_index(right.dims[0])
    if not np.array_equal(left_index, right_index):
        return False
    if left.pc.data_name != right.pc.data_name or not left.pc.columns.equals(right.pc.columns):
        return False
    if bool(left.attrs.get("geometry_z")) != bool(right.attrs.get("geometry_z")):
        return False

    # Geometry coordinates have a fixed float dtype; attribute dtypes remain part of their schema
    names = ["x", "y", *left.pc.columns]
    for name in names:
        left_values = np.asarray(left.data if name == left.name else left.coords[name].data)
        right_values = np.asarray(right.data if name == right.name else right.coords[name].data)
        if check_dtype and name not in ("x", "y") and left_values.dtype != right_values.dtype:
            return False
        try:
            numeric = np.issubdtype(left_values.dtype, np.number) and np.issubdtype(right_values.dtype, np.number)
            if rtol is not None and numeric:
                np.testing.assert_allclose(left_values, right_values, rtol=rtol, atol=atol)
            else:
                np.testing.assert_array_equal(left_values, right_values)
        except (AssertionError, TypeError, ValueError):
            return False
    return True
