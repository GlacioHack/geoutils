# Copyright (c) 2025 GeoUtils developers
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

"""Functionalities for geotransformations of vectors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
from rasterio.crs import CRS

from geoutils._dispatch import get_geo_attr, has_geo_attr

if TYPE_CHECKING:
    from geoutils.raster.base import RasterLike
    from geoutils.vector.vector import VectorLike, Vector


def _reproject(
    source_vector: Vector,
    ref: RasterLike | VectorLike | None = None,
    crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Reproject a vector. See Vector.reproject() for more details."""

    # Check that either ref or crs is provided
    if (ref is not None and crs is not None) or (ref is None and crs is None):
        raise ValueError("Either of `ref` or `crs` must be set. Not both.")

    # Case a raster or vector is provided as reference
    if ref is not None:
        # Check that ref type is either str, Raster or rasterio data set
        if has_geo_attr(ref, "crs"):
            crs = get_geo_attr(ref, "crs")
        else:
            raise TypeError("Match-reference input must have a 'crs' attribute, such as a raster or vector.")
    else:
        # Determine user-input target CRS
        crs = CRS.from_user_input(crs)

    new_ds = source_vector.ds.to_crs(crs=crs)

    return new_ds
