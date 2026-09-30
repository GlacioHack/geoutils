# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module that defines nodata propagation behavior at the package scale."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from geoutils._config import config
from geoutils._typing import NDArrayBool, NDArrayNum

# Internal handling for nodata: for an interpolator or reducer
# "ignore": exclude nodata before calculation
# "propagate": return nodata if any nodata contributes to the calculation
NodataHandling = Literal["ignore", "propagate"]

# Public options for nodata propagation
# "ignore" and "propagate": same as above
# "gdal": (for interpolators) sets nodata to targets whose nearest source is nodata
#         (for reducers) sets nodata to points with neighbourhood without sufficient source data
# integer of value X: output is set to nodata if within X pixels from a nodata
# "half_order_up/down": same as above, but distance depends on interpolation order (only for Interpolators)
NodataPropagation = Literal["gdal", "ignore", "propagate"]
NodataSpread = Literal["half_order_up", "half_order_down"] | int
NodataChoice = NodataPropagation | NodataSpread


##################################
# 1/ NODATA OPTIONS AND DISTANCES
##################################


def _resolve_nodata_handling(
    nodata_handling: NodataChoice | None, order: int | None
) -> tuple[NodataPropagation, int | None]:
    """Read and normalize a nodata user input."""

    # Raise appropriate errors
    choice = config["interpolation_nodata_handling"] if nodata_handling is None else nodata_handling
    if isinstance(choice, str):
        choice = choice.lower()
    if choice in ("gdal", "ignore", "propagate"):
        return cast(NodataPropagation, choice), None
    if isinstance(choice, str) and choice not in ("half_order_up", "half_order_down"):
        raise ValueError(
            "nodata_handling must be 'gdal', 'ignore', 'propagate', 'half_order_up', "
            "'half_order_down' or a non-negative integer."
        )
    if isinstance(choice, bool) or not isinstance(choice, (str, int)) or (isinstance(choice, int) and choice < 0):
        raise ValueError(
            "nodata_handling must be 'gdal', 'ignore', 'propagate', 'half_order_up', "
            "'half_order_down' or a non-negative integer."
        )
    return "ignore", _nodata_spread_distance(order=order, dist_nodata_spread=cast(NodataSpread, choice))


def _validate_nodata_propagation(nodata_propagation: str) -> NodataPropagation:
    """
    Validate and normalize a nodata propagation rule.

    :param nodata_propagation: Rule used to handle invalid source values.

    :return: Normalized propagation rule.
    """

    # Lowercase string values so public methods accept the same spelling variants
    normalized = nodata_propagation.lower()
    if normalized not in ("gdal", "ignore", "propagate"):
        raise ValueError("nodata_propagation must be one of 'gdal', 'ignore' or 'propagate'.")
    return cast(NodataPropagation, normalized)


def _nodata_spread_distance(order: int | None, dist_nodata_spread: NodataSpread) -> int:
    """
    Convert a nodata spread option to a non-negative number of pixels.

    :param order: Interpolation order, or None when the operation does not use interpolation weights.
    :param dist_nodata_spread: Fixed distance or half of the interpolation order rounded up or down.

    :returns: Mask distance in pixels (source pixels for raster interpolation, output pixels for point gridding).
    """

    # For example, cubic interpolation has order 3: half-order gives 1 pixel (down) or 2 pixels (up)
    if isinstance(dist_nodata_spread, str):
        if order is None:
            raise ValueError("Half-order nodata spreading requires an interpolation method with a known order.")
        if dist_nodata_spread == "half_order_up":
            return (order + 1) // 2
        if dist_nodata_spread == "half_order_down":
            return order // 2
        raise ValueError("dist_nodata_spread must be 'half_order_up', 'half_order_down' or a non-negative integer.")

    # Reject booleans and negative distances before expanding a raster mask or point search
    if isinstance(dist_nodata_spread, bool) or not isinstance(dist_nodata_spread, int) or dist_nodata_spread < 0:
        raise ValueError("dist_nodata_spread must be 'half_order_up', 'half_order_down' or a non-negative integer.")
    return dist_nodata_spread


def _nodata_mask_distance(
    order: int,
    nodata_propagation: NodataPropagation,
    dist_nodata_spread: NodataSpread | None,
) -> int | None:
    """Find how far to expand the nodata mask (None means no extra mask)."""

    # GDAL checks the nearest source cell; higher-order propagation uses half the order, rounded up
    if nodata_propagation == "ignore":
        base_distance = None
    elif nodata_propagation == "gdal":
        base_distance = 0
    else:
        base_distance = (order + 1) // 2

    # Zero is also an explicit distance: mask the original nodata cells without expanding around them
    if dist_nodata_spread is not None:
        return _nodata_spread_distance(order=order, dist_nodata_spread=dist_nodata_spread)
    return base_distance


#######################################
# 2/ MASK GRID VALUES FROM POINT INPUT
#######################################


def _nearest_source_validity(
    source_points: NDArrayNum,
    source_valid: NDArrayBool,
    queries: NDArrayNum,
) -> NDArrayBool:
    """Check each target's nearest source value, breaking distance ties like GDAL raster resampling."""

    # Two source points can be equally close to a target (e.g. halfway between raster cells)
    # We shift the target by the smallest possible X/Y increment to favour the right/lower cell, as GDAL does
    point_tree = cKDTree(source_points)
    shifted_queries = queries.copy()
    shifted_queries[:, 0] = np.nextafter(shifted_queries[:, 0], np.inf)
    shifted_queries[:, 1] = np.nextafter(shifted_queries[:, 1], -np.inf)
    _, selected_indexes = point_tree.query(shifted_queries, k=1)
    return source_valid[selected_indexes]


def _mask_grid_near_invalid_points(
    array: NDArrayNum,
    invalid_points: NDArrayNum,
    grid_coords: tuple[NDArrayNum, NDArrayNum],
    res_x: float,
    res_y: float,
    radius: float,
) -> None:
    """Set cells to NaN near a point with a nodata value, using a distance in output pixels."""

    from geoutils.operators.neighbours import _build_grid_queries, _build_scaled_point_tree

    if len(invalid_points) == 0:
        return

    # Scale both axes to output pixels before finding the nearest invalid point
    x_coords, y_coords = grid_coords
    x_start = float(np.min(x_coords))
    y_start = float(np.min(y_coords))
    invalid_tree = _build_scaled_point_tree(invalid_points, x_start=x_start, y_start=y_start, res_x=res_x, res_y=res_y)
    scaled_queries = _build_grid_queries((x_coords - x_start) / res_x, (y_coords - y_start) / res_y)
    distances, _ = invalid_tree.query(scaled_queries, k=1)
    array[distances.reshape(array.shape) <= radius] = np.nan


def _mask_grid_from_invalid_points(
    array: NDArrayNum,
    source_points: NDArrayNum,
    source_valid: NDArrayBool,
    grid_coords: tuple[NDArrayNum, NDArrayNum],
    res_x: float,
    res_y: float,
    radius: float,
    method: str | None,
    nodata_propagation: Literal["gdal", "propagate"],
) -> None:
    """
    Mask output cells affected by invalid source values, following the selected nodata rule.

    :param array: Gridded output before its Y axis is flipped.
    :param source_points: Coordinates of every positioned source observation in original row order.
    :param source_valid: Whether each positioned observation has a finite value.
    :param grid_coords: Output grid coordinates in X and Y.
    :param res_x: Positive output resolution along X.
    :param res_y: Positive output resolution along Y.
    :param radius: Maximum support distance expressed in output pixels.
    :param method: Built-in gridding method for ``"propagate"``. Custom Interpolators use None under ``"gdal"``.
    :param nodata_propagation: Whether to mask cells with an invalid nearest source or any invalid source used by
        the method.
    """

    from geoutils.operators.neighbours import _build_grid_queries

    # No additional mask is needed when every positioned source value is finite
    invalid_points = source_points[~source_valid]
    if len(invalid_points) == 0 or not np.any(source_valid):
        return

    x_coords, y_coords = grid_coords
    queries = _build_grid_queries(x_coords, y_coords)
    if nodata_propagation == "gdal":
        # Every Interpolator uses the original nearest point for this mask, regardless of its calculation method
        nearest_validity = _nearest_source_validity(source_points, source_valid, queries)
        array[~nearest_validity.reshape(array.shape)] = np.nan
        return

    # Propagation needs the built-in method to identify which missing points contribute
    assert method is not None
    if method in ("nearest", "linear", "cubic"):
        # Interpolate 1 for valid values and 0 for nodata (nearest, or linear for linear/cubic methods)
        # The linear check cannot identify every source point used by cubic interpolation
        # Values below 1 mark cells affected by nodata in this check; allow a small rounding error
        validity = source_valid.astype(np.float64)
        mask_method = "nearest" if method == "nearest" else "linear"
        interpolated_validity = griddata(
            points=source_points,
            values=validity,
            xi=queries,
            method=mask_method,
            fill_value=1,
            rescale=False,
        )
        array[interpolated_validity.reshape(array.shape) < 1 - np.finfo(np.float32).eps] = np.nan
        return

    # IDW gives points exactly at a target all the weight, so missing points farther away do not contribute
    if method == "idw":
        finite_exact = cKDTree(source_points[source_valid]).query(queries, k=1)[0] == 0
        invalid_exact = cKDTree(invalid_points).query(queries, k=1)[0] == 0
        unaffected_exact = (finite_exact & ~invalid_exact).reshape(array.shape)
        exact_values = array[unaffected_exact].copy()

    # Circular methods propagate every invalid point inside their requested neighborhood
    _mask_grid_near_invalid_points(
        array,
        invalid_points=invalid_points,
        grid_coords=grid_coords,
        res_x=res_x,
        res_y=res_y,
        radius=radius,
    )
    if method == "idw":
        # The nearby missing points had zero weight wherever a finite point lay exactly at the target
        array[unaffected_exact] = exact_values
