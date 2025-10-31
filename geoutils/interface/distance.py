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

"""Functionalities related to distance operations."""

from __future__ import annotations

import warnings
from typing import Literal

import geopandas as gpd
import numpy as np
from scipy.ndimage import distance_transform_edt

import geoutils as gu
from geoutils._typing import NDArrayNum


def _proximity_from_vector_or_raster(
    raster: gu.Raster,
    vector: gu.Vector | None = None,
    target_values: list[float] | None = None,
    geometry_type: str = "boundary",
    in_or_out: Literal["in"] | Literal["out"] | Literal["both"] = "both",
    distance_unit: Literal["pixel"] | Literal["georeferenced"] = "georeferenced",
) -> NDArrayNum:
    """
    (This function is defined here as mostly raster-based, but used in a class method for both Raster and Vector)
    Proximity to a Raster's target values if no Vector is provided, otherwise to a Vector's geometry type
    rasterized on the Raster.

    :param raster: Raster to burn the proximity grid on.
    :param vector: Vector for which to compute the proximity to geometry,
        if not provided computed on the Raster target pixels.
    :param target_values: (Only with a Raster) List of target values to use for the proximity,
        defaults to all non-zero values.
    :param geometry_type: (Only with a Vector) Type of geometry to use for the proximity, defaults to 'boundary'.
    :param in_or_out: (Only with a Vector) Compute proximity only 'in' or 'out'-side the geometry, or 'both'.
    :param distance_unit: Distance unit, either 'georeferenced' or 'pixel'.
    """

    # 1/ First, if there is a vector input, we rasterize the geometry type
    # (works with .boundary that is a LineString (.exterior exists, but is a LinearRing)
    if vector is not None:

        # TODO: Only when using centroid... Maybe we should leave this operation to the user anyway?
        warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*")

        # We create a geodataframe with the geometry type
        boundary_shp = gpd.GeoDataFrame(geometry=vector.ds.__getattr__(geometry_type), crs=vector.crs)
        # We mask the pixels that make up the geometry type
        mask_boundary = gu.Vector(boundary_shp).create_mask(raster, as_array=True)

    else:
        # Get raster array
        raster_arr = raster.get_nanarray()

        # If input is a mask, target is implicit, and array needs to be converted to uint8
        if target_values is None and raster.is_mask:
            target_values = [1]
            raster_arr = raster_arr.astype("uint8")

        # We mask target pixels
        if target_values is not None:
            mask_boundary = np.logical_or.reduce([raster_arr == target_val for target_val in target_values])
        # Otherwise, all non-zero values are considered targets
        else:
            mask_boundary = raster_arr.astype(bool)

    # 2/ Now, we compute the distance matrix relative to the masked geometry type
    if distance_unit.lower() == "georeferenced":
        sampling: int | tuple[float | int, float | int] = raster.res
    elif distance_unit.lower() == "pixel":
        sampling = 1
    else:
        raise ValueError('Distance unit must be either "georeferenced" or "pixel".')

    # If not all pixels are targets, then we compute the distance
    non_targets = np.count_nonzero(mask_boundary)
    if non_targets > 0:
        proximity = distance_transform_edt(~mask_boundary, sampling=sampling)
    # Otherwise, pass an array full of nodata
    else:
        proximity = np.ones(np.shape(mask_boundary)) * np.nan

    # 3/ If there was a vector input, apply the in_and_out argument to optionally mask inside/outside
    if vector is not None:
        if in_or_out == "both":
            pass
        elif in_or_out in ["in", "out"]:
            mask_polygon = gu.Vector(vector.ds).create_mask(raster, as_array=True)
            if in_or_out == "in":
                proximity[~mask_polygon] = 0
            else:
                proximity[mask_polygon] = 0
        else:
            raise ValueError('The type of proximity must be one of "in", "out" or "both".')

    return proximity
