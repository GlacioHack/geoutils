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

from typing import Any

import geopandas as gpd
import numpy as np

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


def _georeferenced_coords_equal(left_obj: Any, right_obj: Any) -> bool:
    """Compare point-cloud coordinates and CRS without collecting Dask dataframes."""

    try:
        left_crs = get_geo_attr(left_obj, "crs")
        right_crs = get_geo_attr(right_obj, "crs")
        left = _get_dataframe(left_obj)
        right = _get_dataframe(right_obj)
    except (AttributeError, TypeError):
        return False

    if left_crs != right_crs:
        return False
    return _compare_dataframes_partitionwise(left, right, comparator=_point_coords_equal_eager)
