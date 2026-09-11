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

from geoutils._dispatch import get_geo_attr
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

    same_coordinates = _compare_dataframes_partitionwise(left, right, comparator=_point_coords_equal_eager)

    # Report a vertical difference without treating it as different horizontal coordinates
    if same_coordinates and left_crs is not None and right_crs is not None and left_crs != right_crs and warn_3d_crs:
        warnings.warn(
            "The two point clouds have the same 2D CRS but a different vertical CRS: "
            f"{left_crs.name} and {right_crs.name}.",
            category=UserWarning,
        )

    return same_coordinates
