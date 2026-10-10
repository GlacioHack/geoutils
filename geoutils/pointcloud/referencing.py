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

"""Referencing functions for points: resolve dimension, X/Y coordinates and CRS."""

from __future__ import annotations

import numpy as np
import xarray as xr
from pyproj import CRS


def _point_coordinates(points: xr.DataArray | xr.Dataset) -> tuple[str, str, str]:
    """Find one common point dimension using X/Y names or CF coordinate attributes."""

    # Axis names and CF attributes distinguish point coordinates from independent raster dimensions
    axes: dict[str, dict[str, list[str]]] = {}
    candidates = points.variables if isinstance(points, xr.Dataset) else points.coords
    for name, coordinate in candidates.items():
        if coordinate.ndim != 1:
            continue
        dimension = coordinate.dims[0]
        axis = coordinate.attrs.get("axis", "").lower()
        standard = coordinate.attrs.get("standard_name", "")
        if name in ("x", "x_point", f"x_{dimension}") or standard in ("projection_x_coordinate", "longitude"):
            axis = "x"
        elif name in ("y", "y_point", f"y_{dimension}") or standard in ("projection_y_coordinate", "latitude"):
            axis = "y"
        if axis in ("x", "y"):
            axes.setdefault(dimension, {"x": [], "y": []})[axis].append(str(name))

    # Require a unique coordinate pair on the same dimension, rather than choosing an arbitrary support
    pairs = [
        (dimension, coordinates) for dimension, coordinates in axes.items() if coordinates["x"] and coordinates["y"]
    ]
    if not pairs:
        raise AttributeError("Point arrays require x/y coordinates along their point dimension.")
    if len(pairs) != 1 or any(len(names) != 1 for names in pairs[0][1].values()):
        raise ValueError("Point coordinates are ambiguous; use one X/Y pair on a common point dimension.")
    dimension, coordinates = pairs[0]
    x_name, y_name = coordinates["x"][0], coordinates["y"][0]
    if isinstance(points, xr.DataArray) and points.dims != (dimension,):
        raise AttributeError("The point accessor requires a one-dimensional DataArray.")
    if not all(np.issubdtype(candidates[name].dtype, np.number) for name in (x_name, y_name)):
        raise AttributeError("Point X/Y coordinates must be numeric.")
    return dimension, x_name, y_name


def _point_crs(points: xr.DataArray) -> CRS | None:
    """Read point CRS metadata without using an independent raster grid mapping."""

    _, x_name, y_name = _point_coordinates(points)
    values = [points.attrs.get("crs"), points.coords[x_name].attrs.get("crs"), points.coords[y_name].attrs.get("crs")]
    reference = None
    for value in values:
        if value is None:
            continue
        crs = CRS.from_user_input(value)
        # Point coordinates always store X/Y, including CF mappings without geographic axis order
        if reference is not None and not reference.equals(crs, ignore_axis_order=True):
            raise ValueError("Point values and X/Y coordinates must have the same CRS.")
        reference = crs

    # An explicitly referenced CF grid mapping also describes point support
    mapping = points.encoding.get("grid_mapping", points.attrs.get("grid_mapping"))
    if mapping in points.coords:
        metadata = points.coords[mapping].attrs
        value = metadata.get("crs_wkt", metadata.get("spatial_ref"))
        crs = CRS.from_user_input(value) if value is not None else CRS.from_cf(metadata)
        if reference is not None and not reference.equals(crs, ignore_axis_order=True):
            raise ValueError("Point values and their grid mapping must have the same CRS.")
        reference = crs
    return reference
