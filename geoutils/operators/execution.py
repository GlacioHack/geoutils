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

"""Map and run interpolation or reduction methods on source data."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates

from geoutils._misc import import_optional
from geoutils._typing import NDArrayNum, Number
from geoutils.operators.base import LocalData
from geoutils.operators.interpolator import (
    Interpolator,
    InverseDistance,
    IrregularInterpolationMethod,
    ScipyInterpolator,
    _interpolate_point_groups_numba,
)
from geoutils.operators.neighbours import (
    GridNeighbours,
    _grid_window_kernel,
    _prepare_fractional_window_data,
    _prepare_grid_neighbours_data,
    _prepare_point_neighbours_data,
)
from geoutils.operators.nodata import NodataHandling, NodataPropagation, NodataSpread, _nodata_spread_distance
from geoutils.operators.reducer import (
    IrregularReductionMethod,
    Mean,
    Reducer,
    _can_reduce_arrays,
    _reduce_point_groups_numba,
)
from geoutils.projtools import _affine_matmul

if TYPE_CHECKING:
    import geopandas as gpd
    import rasterio as rio

####################
# 1/ METHOD SELECTION
####################


def _get_builtin_gridding_method(
    operator: Interpolator | Reducer,
    *,
    default_neighborhood_only: bool = True,
) -> IrregularInterpolationMethod | IrregularReductionMethod | None:
    """Get gridding method name from an Interpolator or Reducer when available."""

    # An explicit neighbor choice must be applied before running the numerical method
    if default_neighborhood_only and (
        operator.default_neighborhood is not None
        or getattr(operator, "_requires_local_evaluation", False)
        or (operator.error_structure is not None and isinstance(operator, (Mean, InverseDistance)))
    ):
        return None
    # Check the class itself, not its parents: a custom subclass may change predict()/reduce()
    operator = getattr(operator, "_wrapped_operator", operator)
    method = operator.method if type(operator) is ScipyInterpolator else type(operator).__dict__.get("_grid_method")
    return cast(IrregularInterpolationMethod | IrregularReductionMethod | None, method)


############################
# 2/ APPLY THE NUMERICAL METHODS
############################


def _reduce_grid_queries(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    points: tuple[NDArrayNum, NDArrayNum],
    operator: Reducer,
    neighborhood: GridNeighbours | None,
    fractional_window: int | None,
    fractional_shape: Literal["square", "circular"] | None,
    handling: NodataHandling | None,
) -> NDArrayNum | None:
    """Filter once and sample the result when enough queries share the same window."""

    if not _can_reduce_arrays(operator):
        return None
    # Sparse queries cost less than calculating a result at every source cell
    if len(points[0]) < max(32, array.size / 4096):
        return None
    cols, rows = _affine_matmul(~transform, points)
    cells_row, cells_col = np.floor(rows).astype(np.int64), np.floor(cols).astype(np.int64)
    neighbors = neighborhood if neighborhood is not None else GridNeighbours(size=1)
    if fractional_window is not None:
        row_phase, col_phase = rows - cells_row, cols - cells_col
        if not (
            np.allclose(row_phase, row_phase[0], rtol=0, atol=1e-12)
            and np.allclose(col_phase, col_phase[0], rtol=0, atol=1e-12)
        ):
            return None
        neighbors = GridNeighbours(
            size=fractional_window,
            shape=fractional_shape or "square",
            coverage=neighborhood.coverage if neighborhood is not None else "fractional",
        )
        kernel = _grid_window_kernel(
            neighbors,
            phase=(row_phase[0], col_phase[0]),
            transform=transform,
            max_cells=max(4096, 4 * array.size),
        )
    else:
        kernel = _grid_window_kernel(neighbors, max_cells=max(4096, 4 * array.size))
    if kernel is None:
        return None

    # Direct convolution grows with the footprint (use a conservative crossover for sparse targets)
    # Sliding sums need only one pass per axis, regardless of rectangular window size
    if not np.all(kernel == 1) and len(rows) * 32768 < array.size * np.count_nonzero(kernel):
        return None
    from geoutils.filters.regular import _reduce_grid_array

    filtered = _reduce_grid_array(array, operator, kernel, nodata_propagation=handling)
    if filtered is None:
        return None
    inside = (cells_row >= 0) & (cells_row < array.shape[0]) & (cells_col >= 0) & (cells_col < array.shape[1])
    output = np.full(len(rows), np.nan)
    output[inside] = filtered[cells_row[inside], cells_col[inside]]
    return output


def _evaluate_operator_batch(
    operator: Interpolator | Reducer,
    data: Sequence[LocalData],
    *,
    nodata_propagation: NodataHandling | None,
) -> NDArrayNum:
    """Interpolate/reduce each group of source values and return the results in the same order."""

    # Custom batch implementations receive the same observation covariance as ordinary evaluate() calls
    if operator.error_structure is not None and operator._uses_error_covariance:
        from geoutils.operators.weighting import _local_error_data

        data = [_local_error_data(local, operator.error_structure, operator._error_predictors) for local in data]
    if isinstance(operator, Interpolator):
        return operator.predict_batch(data, nodata_propagation=nodata_propagation)
    return operator.reduce_batch(data, nodata_propagation=nodata_propagation)


def _evaluate_point_groups_numba(
    operator: Interpolator | Reducer, data: Sequence[LocalData], handling: NodataHandling | None
) -> NDArrayNum:
    """Flatten point neighborhoods and dispatch to the compiled interpolator or reducer."""

    import_optional("numba")

    # Offsets delimit each target's observations without padding shorter neighborhoods
    offsets = np.concatenate(([0], np.cumsum([len(local.values) for local in data]))).astype(np.int64)
    values = np.concatenate([local.values for local in data])
    valid = np.concatenate([local.valid for local in data]) & np.isfinite(values)
    distances = np.concatenate([cast(NDArrayNum, local.distances) for local in data])
    method = _get_builtin_gridding_method(operator, default_neighborhood_only=False)
    propagate_nodata = (operator.default_nodata_propagation if handling is None else handling) == "propagate"

    # Each operator module owns its compiled calculation
    if isinstance(operator, Interpolator):
        return _interpolate_point_groups_numba(
            values,
            valid,
            distances,
            offsets,
            method,
            getattr(operator, "power", 2.0),
            operator.minimum_inputs,
            propagate_nodata,
        )
    return _reduce_point_groups_numba(
        values, valid, distances, offsets, method, operator.minimum_inputs, propagate_nodata
    )


###################################
# 2.1/ RESAMPLE A RASTER AT POINTS
###################################


def _resample_at_points(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum],
    operator: Interpolator | Reducer,
    *,
    area_or_point: Literal["Area", "Point"] | None,
    shift_area_or_point: bool | None,
    nodata_propagation: NodataPropagation,
    dist_nodata_spread: NodataSpread | None,
    neighborhood: GridNeighbours | None = None,
    fractional_window: int | None = None,
    fractional_shape: Literal["square", "circular"] | None = None,
    band: int = 1,
    source_index_offset: tuple[int, int] = (0, 0),
    source_shape: tuple[int, int] | None = None,
) -> NDArrayNum:
    """
    Resample raster values at points with an Interpolator or Reducer.

    This function is the core in-memory function that englobes the public interp_points(),
    reduce_points(), and resampling within Raster.reproject() into one.

    The source array may either be a chunk, or the full array.
    The logic for chunked implementation lives in interface/resampling.py and raster/transformation.py.
    """

    # 1/ Check the raster band and select the cells used for each requested point
    if array.ndim != 2:
        raise ValueError("Resampling at points requires one raster band at a time.")

    # Dense built-in reductions reuse the raster filters when the array contains only source cells
    coordinates = (np.atleast_1d(points[0]), np.atleast_1d(points[1]))
    handling: NodataHandling | None = "ignore" if nodata_propagation == "nearest" else nodata_propagation
    full_shape = array.shape if source_shape is None else source_shape
    within_source = all(
        0 <= offset and offset + length <= full
        for offset, length, full in zip(source_index_offset, array.shape, full_shape)
    )
    if isinstance(operator, Reducer) and within_source and dist_nodata_spread is None:
        filtered = _reduce_grid_queries(
            array, transform, coordinates, operator, neighborhood, fractional_window, fractional_shape, handling
        )
        if filtered is not None:
            return filtered

    # Limit temporary groups for custom reducers and queries with different fractional alignments
    neighbors_per_query = len(neighborhood.offsets) if neighborhood is not None else (fractional_window or 3) ** 2
    batch_size = max(1, min(4096, 131072 // neighbors_per_query))
    if isinstance(operator, Reducer) and len(coordinates[0]) > batch_size:
        output = np.empty(len(coordinates[0]), dtype=float)
        for start in range(0, len(output), batch_size):
            stop = start + batch_size
            output[start:stop] = _resample_at_points(
                array,
                transform,
                (coordinates[0][start:stop], coordinates[1][start:stop]),
                operator,
                area_or_point=area_or_point,
                shift_area_or_point=shift_area_or_point,
                nodata_propagation=nodata_propagation,
                dist_nodata_spread=dist_nodata_spread,
                neighborhood=neighborhood,
                fractional_window=fractional_window,
                fractional_shape=fractional_shape,
                band=band,
                source_index_offset=source_index_offset,
                source_shape=source_shape,
            )
        return output

    # A fractional window follows the requested X/Y point, so edge cells can contribute partly
    if fractional_window is not None:
        local_inputs, source_rows, source_cols = _prepare_fractional_window_data(
            array,
            transform,
            points,
            area_or_point=area_or_point,
            fractional_window=fractional_window,
            fractional_shape=fractional_shape,
            coverage=neighborhood.coverage if neighborhood is not None else "fractional",
            band=band,
            source_index_offset=source_index_offset,
            source_shape=source_shape,
        )
        handling = "ignore" if nodata_propagation == "nearest" else nodata_propagation
    else:
        # Ordinary neighborhoods select complete source cells by row/column offset
        local_inputs, handling, source_rows, source_cols = _prepare_grid_neighbours_data(
            array,
            transform,
            points,
            operator,
            area_or_point=area_or_point,
            shift_area_or_point=shift_area_or_point,
            nodata_propagation=nodata_propagation,
            neighborhood=neighborhood,
            band=band,
            source_index_offset=source_index_offset,
            source_shape=source_shape,
        )

    # 2/ Calculate one result for each selected group of raster cells
    results = _evaluate_operator_batch(operator, local_inputs, nodata_propagation=handling)

    # 3/ Apply any nodata mask around missing cells
    # Interpolators check the source cell under the target, reducers use the finite values in their window
    invalid = ~np.isfinite(np.ma.getdata(array))
    if np.ma.isMaskedArray(array):
        invalid |= np.ma.getmaskarray(array)

    # GDAL nodata propagation uses the nearest cell under an interpolated target
    mask_distance = 0 if nodata_propagation == "nearest" and isinstance(operator, Interpolator) else None
    if dist_nodata_spread is not None:
        mask_distance = _nodata_spread_distance(
            order=operator.interpolation_order if isinstance(operator, Interpolator) else None,
            dist_nodata_spread=dist_nodata_spread,
        )
    # A custom distance extends the output nodata mask
    if mask_distance is not None:
        # Grow the mask in source pixels, then read it at the same positions as the interpolated values
        if mask_distance > 0:
            invalid = binary_dilation(invalid, iterations=mask_distance)
        # Read the mask at the cell containing each point, as we do when selecting nearby values
        target_invalid = map_coordinates(
            invalid.astype(np.uint8),
            (np.floor(source_rows), np.floor(source_cols)),
            order=0,
            mode="constant",
            cval=1,
            prefilter=False,
        )
        results[target_invalid.astype(bool)] = np.nan

    return results


########################
# 2.2/ GRID POINT CLOUD
########################


def _grid_from_points(
    pc: gpd.GeoDataFrame,
    grid_coords: tuple[NDArrayNum, NDArrayNum],
    data_name: str | None,
    operator: Interpolator | Reducer,
    *,
    res_x: float,
    res_y: float,
    radius: float,
    min_points: int,
    nodata_propagation: NodataPropagation,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayNum:
    """
    Grid onto raster from nearby points with a custom Interpolator/Reducer.

    This function is the core in-memory function used by the public grid().

    The source array may either be a chunk (point partition), or the full point array.
    The logic for the chunked implementations with Dask/MP lives in interface/gridding.py.
    """

    # 1/ Prepare point neighborhoods
    # Collect the nearby points once, with the output indexes needed to place their results
    local_inputs, output_indexes = _prepare_point_neighbours_data(
        pc,
        grid_coords,
        data_name,
        operator,
        res_x=res_x,
        res_y=res_y,
        radius=radius,
        min_points=min_points,
    )

    # 2/ Leave cells without selected points as NaN, then fill the cells we can calculate
    output = np.full((len(grid_coords[1]), len(grid_coords[0])), np.nan, dtype=np.float64)
    if local_inputs:
        # Nearest-point masking uses finite values for the calculation
        handling: NodataHandling | None
        if nodata_propagation == "nearest":
            handling = "ignore"
        else:
            handling = nodata_propagation

        # Place each batch result at the row and column returned with its nearby observations
        rows, columns = zip(*output_indexes)
        if engine == "numba" and operator.error_structure is None:
            results = _evaluate_point_groups_numba(operator, local_inputs, handling)
        else:
            results = _evaluate_operator_batch(operator, local_inputs, nodata_propagation=handling)
        output[rows, columns] = results
    return output
