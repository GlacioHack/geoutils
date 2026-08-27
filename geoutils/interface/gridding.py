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

"""Grid point clouds eagerly or in bounded Dask and multiprocessing raster tiles."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

import affine
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.coords import BoundingBox
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

from geoutils._dispatch import (
    _check_match_grid,
    get_geo_attr,
    has_geo_attr,
    is_dask_dataframe,
)
from geoutils._misc import import_optional
from geoutils._typing import NDArrayNum
from geoutils.interface._nodata import NodataPropagation, _validate_nodata_propagation
from geoutils.multiproc.chunked import ChunkedGeoGrid, GeoGrid, normalize_chunks
from geoutils.multiproc.mparray import (
    MultiprocConfig,
    _split_chunk_size,
    _write_multiproc_result,
)
from geoutils.pointcloud.las import is_laspy_supported, load_laspy_data_bounds
from geoutils.raster.referencing import _coords

if TYPE_CHECKING:
    from geoutils.raster.base import RasterLike


GridPointCloudCallable = Callable[..., tuple[NDArrayNum, affine.Affine]]
GriddingEngine = Literal["scipy", "numba"]
CircularGriddingMethod = Literal[
    "idw",
    "mean",
    "minimum",
    "maximum",
    "range",
    "count",
    "stdev",
    "average_distance",
    "average_distance_pts",
]
GriddingMethod = Literal[
    "nearest",
    "linear",
    "cubic",
    "idw",
    "mean",
    "average",
    "minimum",
    "min",
    "maximum",
    "max",
    "range",
    "count",
    "stdev",
    "average_distance",
    "average_distance_pts",
]
_GRID_QUERY_ROWS = 128
_NUMBA_GRID_FUNCTIONS: dict[str, Callable[..., NDArrayNum]] = {}
_CIRCULAR_METHOD_ALIASES: dict[str, CircularGriddingMethod] = {
    "idw": "idw",
    "mean": "mean",
    "average": "mean",
    "minimum": "minimum",
    "min": "minimum",
    "maximum": "maximum",
    "max": "maximum",
    "range": "range",
    "count": "count",
    "stdev": "stdev",
    "average_distance": "average_distance",
    "average_distance_pts": "average_distance_pts",
}
_NUMBA_STATISTIC_CODES = {
    "mean": 0,
    "minimum": 1,
    "maximum": 2,
    "range": 3,
    "count": 4,
    "stdev": 5,
    "average_distance": 6,
}


def _normalize_gridding_method(method: GriddingMethod) -> str | CircularGriddingMethod:
    """Return the canonical name of an interpolation or circular gridding method."""

    return _CIRCULAR_METHOD_ALIASES.get(method, method)


def _grid_resolution(
    grid_coords: tuple[NDArrayNum, NDArrayNum],
    grid_res: tuple[float, float] | None,
) -> tuple[float, float]:
    """Return positive X/Y grid resolutions, including single-row or single-column grids."""

    if len(grid_coords[0]) > 1:
        res_x = float(np.abs(grid_coords[0][1] - grid_coords[0][0]))
    elif grid_res is not None:
        res_x = abs(grid_res[0])
    else:
        raise ValueError("At least two X coordinates or an explicit grid resolution are required.")

    if len(grid_coords[1]) > 1:
        res_y = float(np.abs(grid_coords[1][1] - grid_coords[1][0]))
    elif grid_res is not None:
        res_y = abs(grid_res[1])
    else:
        raise ValueError("At least two Y coordinates or an explicit grid resolution are required.")
    return res_x, res_y


def _grid_queries(x_coords: NDArrayNum, y_coords: NDArrayNum) -> NDArrayNum:
    """Return flattened coordinates for a bounded group of output rows."""

    # Filling two vectors avoids retaining complete X/Y meshgrids for a large raster
    queries = np.empty((len(x_coords) * len(y_coords), 2), dtype=np.float64)
    queries[:, 0] = np.tile(x_coords, len(y_coords))
    queries[:, 1] = np.repeat(y_coords, len(x_coords))
    return queries


def _point_coordinates_and_values(
    pc: gpd.GeoDataFrame,
    data_column_name: str | None,
) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum]:
    """
    Return finite values and the positions of invalid values as floating-point arrays.

    :param pc: Input point cloud.
    :param data_column_name: Name of the data column, or None to use geometry elevations.

    :return: Valid point coordinates, their values and coordinates whose values are invalid.
    """

    values = np.asarray(pc[data_column_name].values if data_column_name is not None else pc.geometry.z.values)
    points = np.column_stack((pc.geometry.x.values, pc.geometry.y.values))

    # Coordinates must be finite for either values or nodata positions to affect the output
    finite_coordinates = np.isfinite(points).all(axis=1)
    valid = finite_coordinates & np.isfinite(values)
    invalid_points = points[finite_coordinates & ~np.isfinite(values)]
    return (
        np.ascontiguousarray(points[valid], dtype=np.float64),
        np.ascontiguousarray(values[valid], dtype=np.float64),
        np.ascontiguousarray(invalid_points, dtype=np.float64),
    )


def _mask_grid_from_invalid_points(
    array: NDArrayNum,
    valid_points: NDArrayNum,
    invalid_points: NDArrayNum,
    grid_coords: tuple[NDArrayNum, NDArrayNum],
    res_x: float,
    res_y: float,
    radius: float,
    method: str | CircularGriddingMethod,
) -> None:
    """
    Propagate invalid point values through the support of a gridding method.

    :param array: Gridded output before its Y axis is flipped.
    :param valid_points: Coordinates whose values contributed to the output.
    :param invalid_points: Coordinates whose values are invalid.
    :param grid_coords: Output grid coordinates in X and Y.
    :param res_x: Positive output resolution along X.
    :param res_y: Positive output resolution along Y.
    :param radius: Maximum support distance expressed in output pixels.
    :param method: Normalized gridding method.
    """

    # No additional mask is needed when every positioned source value is finite
    if len(invalid_points) == 0 or len(valid_points) == 0:
        return

    x_coords, y_coords = grid_coords
    queries = _grid_queries(x_coords, y_coords)
    if method in ("nearest", "linear", "cubic"):
        # Interpolate a validity flag with the same point topology used for values
        all_points = np.concatenate((valid_points, invalid_points))
        validity = np.concatenate((np.ones(len(valid_points)), np.zeros(len(invalid_points))))
        mask_method = "nearest" if method == "nearest" else "linear"
        interpolated_validity = griddata(
            points=all_points,
            values=validity,
            xi=queries,
            method=mask_method,
            fill_value=1,
            rescale=False,
        )
        array[interpolated_validity.reshape(array.shape) < 1 - np.finfo(np.float32).eps] = np.nan
        return

    # Circular methods propagate every invalid point inside their requested neighborhood
    x_start = float(np.min(x_coords))
    y_start = float(np.min(y_coords))
    invalid_tree = _scaled_point_tree(invalid_points, x_start=x_start, y_start=y_start, res_x=res_x, res_y=res_y)
    scaled_queries = _grid_queries((x_coords - x_start) / res_x, (y_coords - y_start) / res_y)
    distances, _ = invalid_tree.query(scaled_queries, k=1)
    array[distances.reshape(array.shape) <= radius] = np.nan


def _scaled_point_tree(points: NDArrayNum, x_start: float, y_start: float, res_x: float, res_y: float) -> cKDTree:
    """Build a spatial tree whose distances are expressed in output pixels."""

    scaled_points = np.empty_like(points, dtype=np.float64)
    scaled_points[:, 0] = (points[:, 0] - x_start) / res_x
    scaled_points[:, 1] = (points[:, 1] - y_start) / res_y
    return cKDTree(scaled_points)


def _mask_grid_beyond_support(
    array: NDArrayNum,
    points: NDArrayNum,
    grid_coords: tuple[NDArrayNum, NDArrayNum],
    res_x: float,
    res_y: float,
    radius: float,
    n_threads: int,
) -> None:
    """Set cells farther than the support radius from every source point to NaN."""

    x_coords, y_coords = grid_coords
    x_start = float(np.min(x_coords))
    y_start = float(np.min(y_coords))
    point_tree = _scaled_point_tree(points, x_start=x_start, y_start=y_start, res_x=res_x, res_y=res_y)
    scaled_x = (x_coords - x_start) / res_x
    scaled_y = (y_coords - y_start) / res_y

    # Query only a bounded number of rows so the distance check remains memory efficient
    for row_start in range(0, len(y_coords), _GRID_QUERY_ROWS):
        row_stop = min(row_start + _GRID_QUERY_ROWS, len(y_coords))
        queries = _grid_queries(scaled_x, scaled_y[row_start:row_stop])
        distances, _ = point_tree.query(queries, k=1, workers=n_threads)
        block = array[row_start:row_stop]
        block[distances.reshape(block.shape) > radius] = np.nan


def _grid_nearest_scipy(
    points: NDArrayNum,
    values: NDArrayNum,
    grid_coords: tuple[NDArrayNum, NDArrayNum],
    res_x: float,
    res_y: float,
    radius: float,
    n_threads: int,
) -> NDArrayNum:
    """Interpolate nearest values in bounded row groups using a SciPy spatial tree."""

    x_coords, y_coords = grid_coords
    point_tree = cKDTree(points)
    output = np.empty((len(y_coords), len(x_coords)), dtype=np.float64)

    # Reuse one nearest-neighbor tree while limiting temporary query coordinates
    for row_start in range(0, len(y_coords), _GRID_QUERY_ROWS):
        row_stop = min(row_start + _GRID_QUERY_ROWS, len(y_coords))
        queries = _grid_queries(x_coords, y_coords[row_start:row_stop])
        _, point_indexes = point_tree.query(queries, k=1, workers=n_threads)
        output[row_start:row_stop] = values[point_indexes].reshape(row_stop - row_start, len(x_coords))

    if np.isfinite(radius):
        _mask_grid_beyond_support(
            output,
            points=points,
            grid_coords=grid_coords,
            res_x=res_x,
            res_y=res_y,
            radius=radius,
            n_threads=n_threads,
        )
    return output


def _grid_nearest_numba(
    points: NDArrayNum,
    values: NDArrayNum,
    x_coords: NDArrayNum,
    y_coords: NDArrayNum,
    res_x: float,
    res_y: float,
    radius: float,
) -> NDArrayNum:
    """Interpolate nearest values by comparing source distances in a compiled loop."""

    output = np.full((len(y_coords), len(x_coords)), np.nan, dtype=np.float64)
    radius_squared = radius * radius
    finite_radius = np.isfinite(radius)

    # A cell keeps the value of its closest point in unscaled source coordinates
    for row in range(len(y_coords)):
        for col in range(len(x_coords)):
            nearest_index = 0
            nearest_distance_squared = np.inf
            within_support = not finite_radius
            for point_index in range(len(points)):
                delta_x = x_coords[col] - points[point_index, 0]
                delta_y = y_coords[row] - points[point_index, 1]
                distance_squared = delta_x * delta_x + delta_y * delta_y
                if distance_squared < nearest_distance_squared:
                    nearest_index = point_index
                    nearest_distance_squared = distance_squared

                # The support distance is expressed in output pixels along each axis
                scaled_distance_squared = (delta_x / res_x) ** 2 + (delta_y / res_y) ** 2
                if scaled_distance_squared <= radius_squared:
                    within_support = True

            if within_support:
                output[row, col] = values[nearest_index]
    return output


def _numba_grid_function(name: str, function: Callable[..., NDArrayNum]) -> Callable[..., NDArrayNum]:
    """Return one lazily compiled Numba gridding function."""

    # Numba is imported only for an explicit Numba calculation engine
    if name not in _NUMBA_GRID_FUNCTIONS:
        numba = import_optional("numba")
        _NUMBA_GRID_FUNCTIONS[name] = numba.jit(nopython=True, cache=True)(function)
    return _NUMBA_GRID_FUNCTIONS[name]


def _grid_nearest(
    points: NDArrayNum,
    values: NDArrayNum,
    grid_coords: tuple[NDArrayNum, NDArrayNum],
    res_x: float,
    res_y: float,
    radius: float,
    n_threads: int,
    engine: GriddingEngine,
) -> NDArrayNum:
    """Dispatch nearest gridding to the selected SciPy or Numba engine."""

    if engine == "numba":
        function = _numba_grid_function("nearest", _grid_nearest_numba)
        return function(points, values, grid_coords[0], grid_coords[1], res_x, res_y, radius)
    return _grid_nearest_scipy(
        points,
        values,
        grid_coords=grid_coords,
        res_x=res_x,
        res_y=res_y,
        radius=radius,
        n_threads=n_threads,
    )


def _grid_radius_statistic_numba(
    points: NDArrayNum,
    values: NDArrayNum,
    x_start: float,
    y_start: float,
    res_x: float,
    res_y: float,
    width: int,
    height: int,
    radius: float,
    statistic_code: int,
    min_points: int,
) -> NDArrayNum:
    """Compute one circular statistic by visiting nearby cells from every source point."""

    output = np.zeros((height, width), dtype=np.float64)
    secondary = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.int32)
    radius_squared = radius * radius

    # Minimum and range start above every finite value while maxima start below them
    if statistic_code == 1 or statistic_code == 3:
        output[:, :] = np.inf
    elif statistic_code == 2:
        output[:, :] = -np.inf
    if statistic_code == 3:
        secondary[:, :] = -np.inf

    # Point-driven updates avoid searching empty cells in sparse point clouds
    for point_index in range(len(points)):
        point_x = (points[point_index, 0] - x_start) / res_x
        point_y = (points[point_index, 1] - y_start) / res_y
        col_start = max(0, int(np.ceil(point_x - radius)))
        col_stop = min(width - 1, int(np.floor(point_x + radius)))
        row_start = max(0, int(np.ceil(point_y - radius)))
        row_stop = min(height - 1, int(np.floor(point_y + radius)))
        for row in range(row_start, row_stop + 1):
            for col in range(col_start, col_stop + 1):
                distance_squared = (col - point_x) ** 2 + (row - point_y) ** 2
                if distance_squared <= radius_squared:
                    value = values[point_index]
                    if statistic_code == 0:
                        output[row, col] += value
                    elif statistic_code == 1:
                        output[row, col] = min(output[row, col], value)
                    elif statistic_code == 2:
                        output[row, col] = max(output[row, col], value)
                    elif statistic_code == 3:
                        output[row, col] = min(output[row, col], value)
                        secondary[row, col] = max(secondary[row, col], value)
                    elif statistic_code == 5:
                        output[row, col] += value
                        secondary[row, col] += value * value
                    elif statistic_code == 6:
                        output[row, col] += np.sqrt(((col - point_x) * res_x) ** 2 + ((row - point_y) * res_y) ** 2)
                    counts[row, col] += 1

    # Reuse the accumulation arrays for the final statistic
    required_points = max(1, min_points)
    for row in range(height):
        for col in range(width):
            count = counts[row, col]
            if count < required_points:
                output[row, col] = np.nan
            elif statistic_code == 0 or statistic_code == 6:
                output[row, col] /= count
            elif statistic_code == 3:
                output[row, col] = secondary[row, col] - output[row, col]
            elif statistic_code == 4:
                output[row, col] = count
            elif statistic_code == 5:
                mean = output[row, col] / count
                output[row, col] = np.sqrt(max(0.0, secondary[row, col] / count - mean * mean))
    return output


def _grid_radius_idw_numba(
    points: NDArrayNum,
    values: NDArrayNum,
    x_start: float,
    y_start: float,
    res_x: float,
    res_y: float,
    width: int,
    height: int,
    radius: float,
    power: float,
    min_points: int,
) -> NDArrayNum:
    """Compute IDW by visiting every output cell inside each point's support radius."""

    output = np.zeros((height, width), dtype=np.float64)
    weights = np.zeros((height, width), dtype=np.float64)
    exact_counts = np.zeros((height, width), dtype=np.int32)
    counts = np.zeros((height, width), dtype=np.int32)
    radius_squared = radius * radius

    # Negative weights mark cells containing exact source coordinates
    for point_index in range(len(points)):
        point_x = (points[point_index, 0] - x_start) / res_x
        point_y = (points[point_index, 1] - y_start) / res_y
        col_start = max(0, int(np.ceil(point_x - radius)))
        col_stop = min(width - 1, int(np.floor(point_x + radius)))
        row_start = max(0, int(np.ceil(point_y - radius)))
        row_stop = min(height - 1, int(np.floor(point_y + radius)))
        for row in range(row_start, row_stop + 1):
            for col in range(col_start, col_stop + 1):
                distance_squared = (col - point_x) ** 2 + (row - point_y) ** 2
                if distance_squared > radius_squared:
                    continue
                counts[row, col] += 1
                if distance_squared == 0:
                    if weights[row, col] >= 0:
                        output[row, col] = 0
                        weights[row, col] = -1
                    output[row, col] += values[point_index]
                    exact_counts[row, col] += 1
                elif weights[row, col] >= 0:
                    # GDAL selects an elliptical support but weights real coordinate distances
                    coordinate_distance_squared = ((col - point_x) * res_x) ** 2 + ((row - point_y) * res_y) ** 2
                    weight = coordinate_distance_squared ** (-power / 2)
                    output[row, col] += weight * values[point_index]
                    weights[row, col] += weight

    # Exact source values take precedence over surrounding weighted values
    required_points = max(1, min_points)
    for row in range(height):
        for col in range(width):
            if exact_counts[row, col] > 0:
                output[row, col] /= exact_counts[row, col]
            elif counts[row, col] < required_points:
                output[row, col] = np.nan
            elif weights[row, col] > 0:
                output[row, col] /= weights[row, col]
            else:
                output[row, col] = np.nan
    return output


def _grid_radius_scipy(
    points: NDArrayNum,
    values: NDArrayNum,
    grid_coords: tuple[NDArrayNum, NDArrayNum],
    res_x: float,
    res_y: float,
    radius: float,
    method: CircularGriddingMethod,
    distance_power: float,
    min_points: int,
) -> NDArrayNum:
    """Compute radius-based gridding with bounded SciPy sparse distance matrices."""

    x_coords, y_coords = grid_coords
    x_start = float(np.min(x_coords))
    y_start = float(np.min(y_coords))
    point_tree = _scaled_point_tree(points, x_start=x_start, y_start=y_start, res_x=res_x, res_y=res_y)
    scaled_x = (x_coords - x_start) / res_x
    scaled_y = (y_coords - y_start) / res_y
    output = np.full((len(y_coords), len(x_coords)), np.nan, dtype=np.float64)

    # Sparse pairs retain only point-cell distances inside the requested circular support
    for row_start in range(0, len(y_coords), _GRID_QUERY_ROWS):
        row_stop = min(row_start + _GRID_QUERY_ROWS, len(y_coords))
        queries = _grid_queries(scaled_x, scaled_y[row_start:row_stop])
        query_tree = cKDTree(queries)
        pairs = query_tree.sparse_distance_matrix(point_tree, radius, output_type="coo_matrix")
        block = np.full(len(queries), np.nan, dtype=np.float64)
        counts = np.bincount(pairs.row, minlength=len(queries))
        required_points = max(2 if method == "average_distance_pts" else 1, min_points)
        valid = counts >= required_points

        if method == "mean":
            sums = np.bincount(pairs.row, weights=values[pairs.col], minlength=len(queries))
            block[valid] = sums[valid] / counts[valid]
        elif method == "idw":
            exact = pairs.data == 0
            exact_counts = np.bincount(pairs.row[exact], minlength=len(queries))
            exact_sums = np.bincount(
                pairs.row[exact],
                weights=values[pairs.col[exact]],
                minlength=len(queries),
            )
            nonzero = ~exact
            # Sparse-tree distances select an output-pixel circle while weights use source coordinates
            delta_x = queries[pairs.row[nonzero], 0] - point_tree.data[pairs.col[nonzero], 0]
            delta_y = queries[pairs.row[nonzero], 1] - point_tree.data[pairs.col[nonzero], 1]
            coordinate_distances = np.sqrt((delta_x * res_x) ** 2 + (delta_y * res_y) ** 2)
            idw_weights = coordinate_distances**-distance_power
            weight_sums = np.bincount(pairs.row[nonzero], weights=idw_weights, minlength=len(queries))
            weighted_sums = np.bincount(
                pairs.row[nonzero],
                weights=idw_weights * values[pairs.col[nonzero]],
                minlength=len(queries),
            )
            # GDAL gives exact source coordinates precedence over a minimum point requirement
            exact_rows = exact_counts > 0
            weighted_rows = valid & (~exact_rows) & (weight_sums > 0)
            block[exact_rows] = exact_sums[exact_rows] / exact_counts[exact_rows]
            block[weighted_rows] = weighted_sums[weighted_rows] / weight_sums[weighted_rows]
        elif method in ("minimum", "maximum", "range"):
            minima = np.full(len(queries), np.inf)
            maxima = np.full(len(queries), -np.inf)
            np.minimum.at(minima, pairs.row, values[pairs.col])
            np.maximum.at(maxima, pairs.row, values[pairs.col])
            if method == "minimum":
                block[valid] = minima[valid]
            elif method == "maximum":
                block[valid] = maxima[valid]
            else:
                block[valid] = maxima[valid] - minima[valid]
        elif method == "count":
            block[valid] = counts[valid]
        elif method == "stdev":
            sums = np.bincount(pairs.row, weights=values[pairs.col], minlength=len(queries))
            squared_sums = np.bincount(pairs.row, weights=values[pairs.col] ** 2, minlength=len(queries))
            variance = np.zeros(len(queries), dtype=np.float64)
            variance[valid] = squared_sums[valid] / counts[valid] - (sums[valid] / counts[valid]) ** 2
            block[valid] = np.sqrt(np.maximum(variance[valid], 0))
        elif method == "average_distance":
            # Tree distances use output pixels, while GDAL reports distances in coordinate units
            dx = queries[pairs.row, 0] - point_tree.data[pairs.col, 0]
            dy = queries[pairs.row, 1] - point_tree.data[pairs.col, 1]
            distances = np.sqrt((dx * res_x) ** 2 + (dy * res_y) ** 2)
            sums = np.bincount(pairs.row, weights=distances, minlength=len(queries))
            block[valid] = sums[valid] / counts[valid]
        else:
            # Pairwise point distances depend on each complete local neighborhood
            for query_index in np.flatnonzero(valid):
                point_indexes = pairs.col[pairs.row == query_index]
                block[query_index] = float(np.mean(pdist(points[point_indexes])))
        output[row_start:row_stop] = block.reshape(row_stop - row_start, len(x_coords))
    return output


def _grid_radius(
    points: NDArrayNum,
    values: NDArrayNum,
    grid_coords: tuple[NDArrayNum, NDArrayNum],
    res_x: float,
    res_y: float,
    radius: float,
    method: CircularGriddingMethod,
    distance_power: float,
    min_points: int,
    engine: GriddingEngine,
) -> NDArrayNum:
    """Dispatch radius-based gridding to the selected SciPy or Numba engine."""

    if not np.isfinite(radius):
        raise ValueError("Circular gridding methods require a finite dist_nodata_pixel support radius.")
    if method == "idw" and (not np.isfinite(distance_power) or distance_power <= 0):
        raise ValueError("IDW distance_power must be finite and strictly positive.")

    if engine == "numba":
        # Average point spacing needs complete neighborhood membership retained by the SciPy engine
        if method == "average_distance_pts":
            raise ValueError("The Numba gridding engine does not support resampling='average_distance_pts'.")

        x_coords, y_coords = grid_coords
        x_start = float(np.min(x_coords))
        y_start = float(np.min(y_coords))
        function_name = "idw" if method == "idw" else "statistic"
        function = _grid_radius_idw_numba if method == "idw" else _grid_radius_statistic_numba
        numba_function = _numba_grid_function(function_name, function)
        if method != "idw":
            return numba_function(
                points,
                values,
                x_start,
                y_start,
                res_x,
                res_y,
                len(x_coords),
                len(y_coords),
                radius,
                _NUMBA_STATISTIC_CODES[method],
                min_points,
            )
        return numba_function(
            points,
            values,
            x_start,
            y_start,
            res_x,
            res_y,
            len(x_coords),
            len(y_coords),
            radius,
            distance_power,
            min_points,
        )

    return _grid_radius_scipy(
        points,
        values,
        grid_coords=grid_coords,
        res_x=res_x,
        res_y=res_y,
        radius=radius,
        method=method,
        distance_power=distance_power,
        min_points=min_points,
    )


def _grid_pointcloud(
    pc: gpd.GeoDataFrame,
    grid_coords: tuple[NDArrayNum, NDArrayNum],
    data_column_name: str | None = None,
    resampling: GriddingMethod = "linear",
    dist_nodata_pixel: float = 1.0,
    grid_res: tuple[float, float] | None = None,
    distance_power: float = 2.0,
    min_points: int = 1,
    n_threads: int = 1,
    engine: GriddingEngine = "scipy",
    nodata_propagation: NodataPropagation = "gdal",
) -> tuple[NDArrayNum, affine.Affine]:
    """
    Grid irregular points to a regular raster using interpolation or circular neighborhoods.

    :param pc: Point cloud.
    :param grid_coords: Regular raster grid coordinates in X and Y (i.e. equally spaced, independently for each axis).
    :param data_column_name: Name of data column for point cloud (if 2D point geometries are used).
    :param resampling: ``nearest``, ``linear`` or ``cubic`` interpolation, or a circular ``idw``, statistic or
        distance metric (defaults to linear). ``average``, ``min`` and ``max`` are aliases for ``mean``, ``minimum``
        and ``maximum``.
    :param dist_nodata_pixel: Maximum point distance or circular neighborhood radius, expressed in output pixels.
    :param grid_res: Grid resolution, used for chunks with a single row or column.
    :param distance_power: Distance exponent used for inverse-distance weighting (defaults to 2).
    :param min_points: Minimum number of finite points required inside a circular neighborhood (defaults to 1).
    :param engine: Calculation engine, either ``scipy`` (default) or ``numba``. Numba supports nearest and circular
        methods except ``average_distance_pts``.
    :param n_threads: Number of SciPy threads used for nearest-neighbor queries (defaults to 1).
    :param nodata_propagation: How invalid point values affect the output. ``gdal`` and ``ignore`` omit them, while
        ``propagate`` returns NaN where an invalid value participates in interpolation or falls inside a circular
        neighborhood.
    """

    if np.isnan(dist_nodata_pixel) or dist_nodata_pixel < 0:
        raise ValueError("Argument 'dist_nodata_pixel' must be non-negative.")
    if n_threads < 1:
        raise ValueError("Argument 'n_threads' must be a positive integer.")
    if engine not in ("scipy", "numba"):
        raise ValueError("Argument 'engine' must be either 'scipy' or 'numba'.")
    if resampling not in ("nearest", "linear", "cubic") and resampling not in _CIRCULAR_METHOD_ALIASES:
        raise ValueError(f"Unknown gridding resampling method: {resampling!r}.")
    normalized_method = _normalize_gridding_method(resampling)
    if engine == "numba" and normalized_method in ("linear", "cubic", "average_distance_pts"):
        raise ValueError(f"The Numba gridding engine does not support resampling={resampling!r}.")
    if engine == "numba":
        # Fail before building a lazy graph if the requested optional engine is unavailable
        import_optional("numba")
    if normalized_method in _CIRCULAR_METHOD_ALIASES.values() and not np.isfinite(dist_nodata_pixel):
        raise ValueError("Circular gridding methods require a finite dist_nodata_pixel support radius.")
    if normalized_method == "idw" and (not np.isfinite(distance_power) or distance_power <= 0):
        raise ValueError("IDW distance_power must be finite and strictly positive.")
    if isinstance(min_points, bool) or not isinstance(min_points, (int, np.integer)) or min_points < 0:
        raise ValueError("Argument 'min_points' must be a non-negative integer.")
    propagation = _validate_nodata_propagation(nodata_propagation)

    # Work with finite floating-point inputs and derive the distance scale from output pixels
    res_x, res_y = _grid_resolution(grid_coords=grid_coords, grid_res=grid_res)
    points, values, invalid_points = _point_coordinates_and_values(pc=pc, data_column_name=data_column_name)

    aligned_dem: NDArrayNum
    if len(points) == 0:
        aligned_dem = np.full((len(grid_coords[1]), len(grid_coords[0])), np.nan, dtype=np.float64)
    elif resampling == "nearest":
        aligned_dem = _grid_nearest(
            points,
            values,
            grid_coords=grid_coords,
            res_x=res_x,
            res_y=res_y,
            radius=dist_nodata_pixel,
            n_threads=n_threads,
            engine=engine,
        )
    elif normalized_method in _CIRCULAR_METHOD_ALIASES.values():
        aligned_dem = _grid_radius(
            points,
            values,
            grid_coords=grid_coords,
            res_x=res_x,
            res_y=res_y,
            radius=dist_nodata_pixel,
            method=cast(CircularGriddingMethod, normalized_method),
            distance_power=distance_power,
            min_points=int(min_points),
            engine=engine,
        )
    else:
        # SciPy's triangulation methods require complete query grids
        xx, yy = np.meshgrid(grid_coords[0], grid_coords[1])
        aligned_dem = griddata(
            points=points,
            values=values,
            xi=(xx, yy),
            method=resampling,
            rescale=False,
        )

        # Triangulation fills the convex hull, so remove cells beyond the requested local support
        if np.isfinite(dist_nodata_pixel):
            _mask_grid_beyond_support(
                aligned_dem,
                points=points,
                grid_coords=grid_coords,
                res_x=res_x,
                res_y=res_y,
                radius=dist_nodata_pixel,
                n_threads=n_threads,
            )

    # GDAL gridding ignores invalid point values, while explicit propagation retains their support
    if propagation == "propagate":
        _mask_grid_from_invalid_points(
            aligned_dem,
            valid_points=points,
            invalid_points=invalid_points,
            grid_coords=grid_coords,
            res_x=res_x,
            res_y=res_y,
            radius=dist_nodata_pixel,
            method=normalized_method,
        )

    # Flip Y axis of grid
    aligned_dem = np.flip(aligned_dem, axis=0)

    # Derive output transform from input grid
    transform_from_coords = rio.transform.from_origin(min(grid_coords[0]), max(grid_coords[1]), res_x, res_y)

    return aligned_dem, transform_from_coords


def _as_bounding_box(bounds: BoundingBox | tuple[float, float, float, float]) -> BoundingBox:
    """Convert a bounds tuple to a Rasterio bounding box."""

    if isinstance(bounds, BoundingBox):
        return bounds
    return BoundingBox(left=bounds[0], bottom=bounds[1], right=bounds[2], top=bounds[3])


def _buffer_bounds(bounds: BoundingBox, x_buffer: float, y_buffer: float) -> BoundingBox:
    """Expand bounds by X/Y buffers."""

    return BoundingBox(
        left=bounds.left - x_buffer,
        bottom=bounds.bottom - y_buffer,
        right=bounds.right + x_buffer,
        top=bounds.top + y_buffer,
    )


def _support_bounds(geogrid: GeoGrid, dist_nodata_pixel: float) -> BoundingBox:
    """Return block bounds expanded by the gridding local support radius."""

    if dist_nodata_pixel < 0:
        raise ValueError("Argument 'dist_nodata_pixel' must be non-negative.")
    return _buffer_bounds(
        bounds=geogrid.bounds,
        x_buffer=abs(geogrid.res[0]) * dist_nodata_pixel,
        y_buffer=abs(geogrid.res[1]) * dist_nodata_pixel,
    )


def _filter_points_by_bounds(pc: gpd.GeoDataFrame, bounds: BoundingBox) -> gpd.GeoDataFrame:
    """Filter point geometries by X/Y bounds."""

    bbox = _as_bounding_box(bounds)
    if len(pc) == 0:
        return pc

    mask = (pc.geometry.x >= bbox.left) & (pc.geometry.x <= bbox.right)
    mask &= (pc.geometry.y >= bbox.bottom) & (pc.geometry.y <= bbox.top)
    return pc.loc[mask]


def _filter_dask_points_by_bounds(ds: Any, bounds: BoundingBox) -> Any:
    """Filter a Dask-GeoPandas dataframe by bounds, using spatial partitions when available."""

    # Spatial partitions can discard unrelated partitions before any data is read
    bbox = _as_bounding_box(bounds)
    try:
        if getattr(ds, "spatial_partitions", None) is not None:
            return ds.cx[bbox.left : bbox.right, bbox.bottom : bbox.top]
    except NotImplementedError:
        pass

    # Fall back to applying the same coordinate filter inside every partition
    meta = getattr(ds, "_meta", None)
    return ds.map_partitions(_filter_points_by_bounds, bbox, meta=meta)


def _empty_points_like(pc: gpd.GeoDataFrame | None, crs: Any = None) -> gpd.GeoDataFrame:
    """Return an empty GeoDataFrame matching a point-cloud dataframe."""

    if pc is not None:
        return pc.iloc[0:0]
    return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=crs), crs=crs)


def _concat_point_parts(parts: list[gpd.GeoDataFrame], crs: Any = None) -> gpd.GeoDataFrame:
    """Concatenate per-partition point-cloud subsets."""

    if len(parts) == 0:
        return _empty_points_like(None, crs=crs)

    non_empty = [part for part in parts if len(part) > 0]
    if len(non_empty) == 0:
        return _empty_points_like(parts[0], crs=crs)

    return gpd.GeoDataFrame(pd.concat(non_empty, ignore_index=True), geometry="geometry", crs=crs)


def _source_is_dask(source_pointcloud: Any) -> bool:
    """Return whether a point-cloud source is backed by a Dask dataframe without loading it."""

    obj = getattr(source_pointcloud, "_obj", None)
    if obj is not None:
        return is_dask_dataframe(obj)

    ds = getattr(source_pointcloud, "_ds", None)
    return is_dask_dataframe(ds)


def _source_dataframe(source_pointcloud: Any) -> gpd.GeoDataFrame | Any | None:
    """Return the backing dataframe if it is already available, without triggering file loading."""

    obj = getattr(source_pointcloud, "_obj", None)
    if obj is not None:
        return obj

    return getattr(source_pointcloud, "_ds", None)


def _load_pointcloud_bounds(
    source_pointcloud: Any,
    bounds: BoundingBox,
    data_column_name: str | None,
) -> gpd.GeoDataFrame:
    """Load or filter source points intersecting bounds."""

    ds = _source_dataframe(source_pointcloud)
    if ds is not None:
        if is_dask_dataframe(ds):
            raise ValueError("Dask-backed point clouds must use the Dask gridding backend.")
        return _filter_points_by_bounds(ds, bounds)

    filename = getattr(source_pointcloud, "name", None)
    if filename is None:
        return _filter_points_by_bounds(source_pointcloud.ds, bounds)

    if is_laspy_supported(filename):
        return load_laspy_data_bounds(
            filename=filename,
            columns="main",
            bounds=bounds,
            data_column=data_column_name or "Z",
        )

    return gpd.read_file(filename, bbox=tuple(bounds))


def _grid_pointcloud_on_geogrid(
    pc: gpd.GeoDataFrame,
    geogrid: GeoGrid,
    data_column_name: str | None,
    gridding_func: GridPointCloudCallable = _grid_pointcloud,
    **kwargs: Any,
) -> NDArrayNum:
    """Grid a point-cloud subset on a single output geogrid."""

    if len(pc) == 0:
        return np.full(geogrid.shape, np.nan, dtype=np.float64)

    grid_coords = _coords(transform=geogrid.transform, shape=geogrid.shape, grid=False, area_or_point=None)
    gridding_kwargs = kwargs.copy()
    if gridding_func is _grid_pointcloud:
        gridding_kwargs["grid_res"] = geogrid.res

    array, _ = gridding_func(
        pc,
        grid_coords=grid_coords,
        data_column_name=data_column_name,
        **gridding_kwargs,
    )
    return array


def _grid_pointcloud_block_from_source(
    source_pointcloud: Any,
    geogrid: GeoGrid,
    data_column_name: str | None,
    gridding_func: GridPointCloudCallable = _grid_pointcloud,
    **kwargs: Any,
) -> NDArrayNum:
    """Load a point-cloud block subset and grid it."""

    pc = _load_pointcloud_bounds(
        source_pointcloud=source_pointcloud,
        bounds=_support_bounds(geogrid=geogrid, dist_nodata_pixel=kwargs["dist_nodata_pixel"]),
        data_column_name=data_column_name,
    )
    return _grid_pointcloud_on_geogrid(
        pc=pc,
        geogrid=geogrid,
        data_column_name=data_column_name,
        gridding_func=gridding_func,
        **kwargs,
    )


def _grid_pointcloud_block_from_dask_parts(
    parts: list[gpd.GeoDataFrame],
    geogrid: GeoGrid,
    data_column_name: str | None,
    crs: Any,
    gridding_func: GridPointCloudCallable = _grid_pointcloud,
    **kwargs: Any,
) -> NDArrayNum:
    """Concatenate Dask dataframe partitions for a block and grid them."""

    # Each delayed output block receives only the point subsets in its support area
    pc = _concat_point_parts(parts=parts, crs=crs)
    return _grid_pointcloud_on_geogrid(
        pc=pc,
        geogrid=geogrid,
        data_column_name=data_column_name,
        gridding_func=gridding_func,
        **kwargs,
    )


def _grid_pointcloud_multiproc_block(
    source_pointcloud: Any,
    geogrid: GeoGrid,
    dst_tile: tuple[int, int, int, int],
    data_column_name: str | None,
    gridding_func: GridPointCloudCallable = _grid_pointcloud,
    **kwargs: Any,
) -> tuple[NDArrayNum, tuple[int, int, int, int]]:
    """Grid one point-cloud block and return the output write window."""

    array = _grid_pointcloud_block_from_source(
        source_pointcloud=source_pointcloud,
        geogrid=geogrid,
        data_column_name=data_column_name,
        gridding_func=gridding_func,
        **kwargs,
    )
    return array, dst_tile


def _dask_grid_pointcloud(
    source_pointcloud: Any,
    dst_geotiling: ChunkedGeoGrid,
    dst_block_geogrids: list[GeoGrid],
    data_column_name: str | None,
    gridding_func: GridPointCloudCallable = _grid_pointcloud,
    **kwargs: Any,
) -> Any:
    """Grid a point cloud lazily into a Dask array."""

    # Delay each output tile independently and expose them as one Dask array
    dask = import_optional("dask")
    import dask.array as da

    delayed = dask.delayed
    source_ds = _source_dataframe(source_pointcloud)
    source_crs = get_geo_attr(source_pointcloud, "crs")

    # Build the nested block layout expected by ``dask.array.block``
    block_arrays = []
    for iy in range(dst_geotiling.numblocks[0]):
        row_arrays = []
        for ix in range(dst_geotiling.numblocks[1]):
            # Match this Dask block to its georeferenced output area
            block_index = dst_geotiling.ravel_block_index((iy, ix))
            geogrid = dst_block_geogrids[block_index]

            if is_dask_dataframe(source_ds):
                # Cull points outside the interpolation support before computing partitions
                source_ds_dask = cast(Any, source_ds)
                bounds = _support_bounds(geogrid=geogrid, dist_nodata_pixel=kwargs["dist_nodata_pixel"])
                filtered = _filter_dask_points_by_bounds(source_ds_dask, bounds)
                # One delayed task combines the filtered partitions and grids the tile
                tile = delayed(_grid_pointcloud_block_from_dask_parts)(
                    list(filtered.to_delayed()),
                    geogrid,
                    data_column_name,
                    source_crs,
                    gridding_func,
                    **kwargs,
                )
            else:
                # File-backed inputs load only the bounds needed by this output tile
                tile = delayed(_grid_pointcloud_block_from_source)(
                    source_pointcloud,
                    geogrid,
                    data_column_name,
                    gridding_func,
                    **kwargs,
                )

            # Declare the tile shape so Dask knows the final array layout before computing
            row_arrays.append(da.from_delayed(tile, shape=geogrid.shape, dtype=np.float64))
        block_arrays.append(row_arrays)

    # Join delayed tiles without evaluating any point-cloud data
    return da.block(block_arrays)


def _multiproc_grid_pointcloud(
    source_pointcloud: Any,
    dst_geotiling: ChunkedGeoGrid,
    dst_block_geogrids: list[GeoGrid],
    data_column_name: str | None,
    mp_config: MultiprocConfig,
    file_metadata: dict[str, Any],
    gridding_func: GridPointCloudCallable = _grid_pointcloud,
    **kwargs: Any,
) -> Any:
    """Grid a point cloud with multiprocessing and write tiles directly to disk."""

    # Submit one independent gridding task for each output file window
    block_ids = dst_geotiling.get_block_locations()
    tasks = []
    for index, geogrid in enumerate(dst_block_geogrids):
        dst_tile = (block_ids[index]["ys"], block_ids[index]["ye"], block_ids[index]["xs"], block_ids[index]["xe"])
        tasks.append(
            mp_config.cluster.submit(
                _grid_pointcloud_multiproc_block,
                source_pointcloud,
                geogrid,
                dst_tile,
                data_column_name,
                gridding_func,
                **kwargs,
            )
        )

    # Write tiles as workers finish instead of holding the full raster in memory
    return _write_multiproc_result(tasks=tasks, mp_config=mp_config, file_metadata=file_metadata)


def _grid_pointcloud_to_raster(
    source_pointcloud: Any,
    ref: RasterLike | None = None,
    grid_coords: tuple[NDArrayNum, NDArrayNum] | None = None,
    res: float | tuple[float, float] | None = None,
    shape: tuple[int, int] | None = None,
    bounds: tuple[float, float, float, float] | BoundingBox | None = None,
    resampling: GriddingMethod = "linear",
    dist_nodata_pixel: float = 1.0,
    nodata: int | float = -9999,
    *,
    distance_power: float = 2.0,
    min_points: int = 1,
    chunksizes: tuple[int, int] | None = None,
    mp_config: MultiprocConfig | None = None,
    dask: bool = False,
    n_threads: int = 0,
    engine: GriddingEngine = "scipy",
    nodata_propagation: NodataPropagation = "gdal",
    gridding_func: GridPointCloudCallable = _grid_pointcloud,
) -> Any:
    """Grid a point cloud to a raster with eager, Dask, or Multiprocessing backends."""

    # A single operation must have one owner for scheduling and memory management
    if dask and mp_config is not None:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove mp_config. "
            "To use Multiprocessing, use an eager PointCloud object."
        )

    if _source_is_dask(source_pointcloud) and mp_config is not None:
        raise ValueError("Multiprocessing gridding is only supported for eager or file-backed PointCloud objects.")

    # Resolve all supported grid definitions to one output shape and transform
    out_shape, out_transform, out_crs = _check_match_grid(
        source_pointcloud,
        ref=ref,
        coords=grid_coords,
        res=res,
        bounds=bounds,
        shape=shape,
        crs=None,
    )
    dst_geogrid = GeoGrid(transform=out_transform, shape=out_shape, crs=out_crs)

    if n_threads < 0:
        raise ValueError("Argument 'n_threads' must be non-negative.")

    # Eager calls can use SciPy threads while each parallel output task stays single-threaded
    is_parallel_backend = dask or mp_config is not None
    resolved_threads = n_threads if n_threads > 0 else (1 if is_parallel_backend else max(1, (os.cpu_count() or 2) - 1))
    kwargs: dict[str, Any] = {
        "resampling": resampling,
        "dist_nodata_pixel": dist_nodata_pixel,
    }
    if gridding_func is _grid_pointcloud:
        kwargs.update(
            distance_power=distance_power,
            min_points=min_points,
            n_threads=resolved_threads,
            engine=engine,
            nodata_propagation=nodata_propagation,
        )

    from geoutils.raster import Raster
    from geoutils.raster.xr_accessor import RasterAccessor

    # The eager path grids the complete source into an in-memory Raster
    if not dask and mp_config is None:
        array = _grid_pointcloud_block_from_source(
            source_pointcloud=source_pointcloud,
            geogrid=dst_geogrid,
            data_column_name=get_geo_attr(source_pointcloud, "data_column"),
            gridding_func=gridding_func,
            **kwargs,
        )
        return Raster.from_array(data=array, transform=out_transform, crs=out_crs, nodata=nodata)

    # Reuse explicit, multiprocessing, or reference chunks in that order
    if chunksizes is None:
        if mp_config is not None:
            chunksizes = _split_chunk_size(mp_config.chunks)
        else:
            ref_chunks = get_geo_attr(ref, "_chunks") if ref is not None and has_geo_attr(ref, "_chunks") else None
            chunksizes = ref_chunks if ref_chunks is not None else (1024, 1024)
    assert chunksizes is not None

    # Describe each output chunk as a georeferenced grid for local point selection
    dst_chunks = normalize_chunks(chunks=chunksizes, shape=out_shape)
    dst_geotiling = ChunkedGeoGrid(grid=dst_geogrid, chunks=dst_chunks)
    dst_block_geogrids = dst_geotiling.get_blocks_as_geogrids()

    # Return a lazy raster accessor whose chunks compute independently
    if dask:
        data = _dask_grid_pointcloud(
            source_pointcloud=source_pointcloud,
            dst_geotiling=dst_geotiling,
            dst_block_geogrids=dst_block_geogrids,
            data_column_name=get_geo_attr(source_pointcloud, "data_column"),
            gridding_func=gridding_func,
            **kwargs,
        )
        return RasterAccessor.from_array(data=data, transform=out_transform, crs=out_crs, nodata=nodata)

    # The remaining backend writes worker results directly to the configured file
    assert mp_config is not None
    file_metadata = {
        "height": out_shape[0],
        "width": out_shape[1],
        "count": 1,
        "dtype": np.dtype("float64"),
        "crs": out_crs,
        "transform": out_transform,
        "nodata": nodata,
    }
    return _multiproc_grid_pointcloud(
        source_pointcloud=source_pointcloud,
        dst_geotiling=dst_geotiling,
        dst_block_geogrids=dst_block_geogrids,
        data_column_name=get_geo_attr(source_pointcloud, "data_column"),
        mp_config=mp_config,
        file_metadata=file_metadata,
        gridding_func=gridding_func,
        **kwargs,
    )
