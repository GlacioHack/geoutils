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

"""Interpolator classes to predict a continuous value at X/Y coordinates from nearby input values."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from math import pi, sin
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import numpy as np
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator, griddata
from scipy.ndimage import binary_dilation, distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree

from geoutils._config import config
from geoutils._misc import import_optional
from geoutils._typing import DTypeLike, NDArrayBool, NDArrayNum, Number
from geoutils.operators.base import LinearCoefficients, LocalData
from geoutils.operators.neighbours import (
    _GRID_QUERY_ROWS,
    GridNeighbours,
    PointNeighbours,
    _build_grid_queries,
    _check_regular_grid_neighbours,
    _grid_radius_scipy,
    _mask_grid_beyond_support,
)
from geoutils.operators.nodata import (
    NodataHandling,
    NodataPropagation,
    NodataSpread,
    _nodata_mask_distance,
    _nodata_spread_distance,
    _validate_nodata_propagation,
)
from geoutils.operators.weighting import _local_error_data

if TYPE_CHECKING:
    import rasterio as rio

    from geoutils.stats.variography import Variogram, VariogramModel
    from geoutils.uncertainty.error_structure import ErrorStructure

try:
    from numba import jit as _jit
except ImportError:

    def _jit(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a no-op decorator when Numba is not installed."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


# Regular interpolators (raster grids)
# Nearest, Linear (bilinear), Cubic; other SciPy methods through ScipyInterpolator
# InverseDistance, Kriging and custom Interpolator objects with raster neighborhoods
RegularInterpolationMethod = Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"]
ScipyInterpolationMethod = RegularInterpolationMethod
INTERPOLATION_ORDERS: dict[ScipyInterpolationMethod, int] = {
    "nearest": 0,
    "linear": 1,
    "cubic": 3,
    "quintic": 5,
    "slinear": 1,
    "pchip": 3,
    "splinef2d": 3,
}

# Irregular interpolators (point neighborhoods)
# Nearest, Linear (triangulation), Cubic, InverseDistance, Kriging and custom Interpolator objects
# String names for grid(); Kriging requires a configured object
IrregularInterpolationMethod = Literal["nearest", "linear", "cubic", "idw"]


########################
# 1/ INTERPOLATOR CLASSES
########################

# 1.1/ Base class for custom interpolators
########################################


class Interpolator:
    """
    Interpolator to continuously predict values at new coordinates from nearby observations.

    An Interpolator assumes a spatial field that can be predicted continuously at a point. It is tied to a neighborhood,
    a nodata propagation scheme, and has optional support for sample weights, when the method accepts them.
    Use a Reducer when the prediction should reduce observations across a neighborhood instead (e.g., average over an
    area). Fractional area support exists only for Reducers, not Interpolators.

    To define your own interpolator, implement predict(). It will receive a LocalData object with nearby values, their
    X/Y coordinates and the target X/Y coordinate. Implement coefficients() instead when the prediction is a weighted
    sum that can also be used for uncertainty propagation.

    For example, the custom interpolator below fits a plane to nearby raster cell values:

    .. code-block:: python

        import numpy as np
        import geoutils as gu

        class LocalPlane(gu.operators.Interpolator):
            minimum_inputs = 3

            def predict(self, data):
                # Calculate the X and Y differences between each source cell and the requested point
                x_from_target = data.coordinates[:, 0] - data.target[0]
                y_from_target = data.coordinates[:, 1] - data.target[1]

                # Fit value = value_at_target + x_slope * x_from_target + y_slope * y_from_target
                plane_terms = np.column_stack((np.ones(len(data.values)), x_from_target, y_from_target))
                value_at_target, x_slope, y_slope = np.linalg.lstsq(plane_terms, data.values, rcond=None)[0]
                return float(value_at_target)

        predicted = raster.interp_at_points((x, y), method=LocalPlane(), as_array=True)

    A linear interpolator should implement coefficients() to support weights and uncertainty propagation.
    As an example below, a simple inverse-distance interpolation can be naturally written as a weighted sum:

    .. code-block:: python

        class SimpleInverseDistance(gu.operators.Interpolator):
            '''Predict from nearby cells, giving closer cells more weight.'''

            def coefficients(self, data):
                '''Return normalized inverse-distance weights for each cell.'''

                # A cell at the target takes precedence and avoids division by zero
                exact = data.distances == 0
                if np.any(exact):
                    weights = exact.astype(float)
                else:
                    weights = 1 / data.distances
                return gu.operators.LinearCoefficients(weights / weights.sum())

        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 2.0)])
        summary = gu.uncertainty.propagate(
            raster.interp_at_points, error_structure=errors,
            operation_kwargs=dict(points=(x, y), method=SimpleInverseDistance(), as_array=True),
        )
        predicted = summary.estimate

    An Interpolator can also limit which nearby observations it uses. The following example uses at most eight
    points within 50 coordinate units of each output cell:

    .. code-block:: python

        nearby = gu.operators.PointNeighbours(k=8, radius=50)
        gridded = point_cloud.grid(
            ref=reference_raster, resampling=SimpleInverseDistance(neighborhood=nearby)
        )
    """

    minimum_inputs: ClassVar[int] = 1
    default_nodata_propagation: ClassVar[NodataHandling] = "ignore"
    accepts_sample_weights: ClassVar[bool] = False
    error_structure: ErrorStructure | None = None
    _error_predictors: Any = None
    _uses_error_covariance: bool = True
    default_neighborhood: PointNeighbours | GridNeighbours | None = None
    interpolation_order: int | None = None
    _grid_method: ClassVar[str | None] = None

    def __init__(self, neighborhood: PointNeighbours | GridNeighbours | None = None) -> None:
        """
        Choose how nearby points or raster cells are selected.

        :param neighborhood: PointNeighbours for a point cloud or GridNeighbours for a raster. By default, the
            input type determines which to use (eight nearest points or a 3 x 3 window for a custom interpolator).
        """

        if neighborhood is not None and not isinstance(neighborhood, (GridNeighbours, PointNeighbours)):
            raise TypeError("An Interpolator neighborhood must be a GridNeighbours or PointNeighbours object.")
        if neighborhood is not None:
            self.default_neighborhood = neighborhood

    def coefficients(self, data: LocalData) -> LinearCoefficients | None:
        """Return the weights and optional constant offset for a prediction.

        Override this method to define a prediction as a weighted sum of the supplied values. It applies the
        weights to the values and can reuse them to propagate source uncertainty, so no need to implement predict().

        :param data: Nearby values and their coordinates/distances, selected for one output coordinate.
        :returns: One coefficient per source value and an optional offset, or None when no coefficients are available.
        """

        return None

    def predict(self, data: LocalData) -> float:
        """
        Predict a value from the nearby source observations.

        Override this method to define a non-linear prediction (without support for weights or uncertainty).

        See the class description for an example on how to implement predict() for your subclass.

        :param data: Nearby source values and coordinates, plus the target coordinate.
        :returns: Predicted value at that coordinate.
        """

        affine = self.coefficients(data)
        if affine is None:
            raise NotImplementedError("An Interpolator subclass must implement predict() or coefficients().")
        if len(affine.weights) != len(data.values):
            raise ValueError("Linear coefficients must contain one weight per LocalData value.")
        return float(np.dot(affine.weights, data.values) + affine.offset)

    def evaluate(self, data: LocalData, *, nodata_propagation: NodataHandling | None = None) -> float:
        """
        Check nodata and weights before predicting a value.

        :param data: Source observations selected for one output coordinate.
        :param nodata_propagation: How to handle nodata: ``"ignore"`` excludes nodata values and ``"propagate"``
            returns NaN if any nodata value affects the result. Defaults to default_nodata_propagation.
        :returns: Predicted value, or NaN when too few valid observations remain or invalid values affect the result
            under the ``"propagate"`` rule.
        """

        handling = self.default_nodata_propagation if nodata_propagation is None else nodata_propagation
        if self._uses_error_covariance:
            data = _local_error_data(data, self.error_structure, self._error_predictors)
        return data._evaluate(
            calculation=self.predict,
            coefficients=self.coefficients,
            accepts_sample_weights=self.accepts_sample_weights,
            accepts_support_weights=False,
            minimum_inputs=self.minimum_inputs,
            nodata_handling=handling,
        )

    def predict_batch(
        self,
        data: Sequence[LocalData],
        *,
        nodata_propagation: NodataHandling | None = None,
    ) -> NDArrayNum:
        """Predict several values by calling evaluate() for each group of source observations.

        :param data: One LocalData object per output coordinate, in the desired output order.
        :param nodata_propagation: Override default_nodata_propagation with ``"ignore"`` or ``"propagate"``.
        :returns: One-dimensional array of predicted values in the same order as data.
        """

        return np.asarray(
            [self.evaluate(local, nodata_propagation=nodata_propagation) for local in data],
            dtype=np.float64,
        )

    def _grid_points(
        self,
        points: NDArrayNum,
        values: NDArrayNum,
        grid_coords: tuple[NDArrayNum, NDArrayNum],
        *,
        res_x: float,
        res_y: float,
        radius: float,
        min_points: int,
        n_threads: int,
        engine: Literal["scipy", "numba"],
    ) -> NDArrayNum:
        """
        Optionally calculate a whole point grid with a built-in SciPy or Numba method.

        Subclasses that use this shortcut override the method. Other interpolators use predict_batch() at each grid
        coordinate, so custom subclasses need only predict() or coefficients().
        """

        raise NotImplementedError(
            f"{type(self).__name__} must override _grid_points() to provide a whole-grid calculation."
        )


# 1.2/ Built-in interpolation methods
####################################


class Nearest(Interpolator):
    """
    The Nearest interpolator selects the value closest to each requested coordinate.

    An optional neighborhood limits the search to selected points or raster cells.
    """

    interpolation_order = 0
    _grid_method = "nearest"

    _uses_error_covariance = False

    def coefficients(self, data: LocalData) -> LinearCoefficients:
        """Give the closest value a weight of one (the first value if several are equally close)."""

        if data.distances is not None:
            distances = data.distances
        elif data.coordinates is not None and data.target is not None:
            distances = np.linalg.norm(data.coordinates - data.target, axis=1)
        else:
            raise ValueError("Nearest requires distances or coordinates and a target.")
        weights = np.zeros(len(data.values), dtype=np.float64)
        weights[int(np.argmin(distances))] = 1
        return LinearCoefficients(weights=weights)

    def _grid_points(
        self,
        points: NDArrayNum,
        values: NDArrayNum,
        grid_coords: tuple[NDArrayNum, NDArrayNum],
        *,
        res_x: float,
        res_y: float,
        radius: float,
        min_points: int,
        n_threads: int,
        engine: Literal["scipy", "numba"],
    ) -> NDArrayNum:
        """Dispatch nearest gridding to the selected SciPy or Numba engine."""

        if engine == "numba":
            return _grid_nearest_numba(points, values, grid_coords[0], grid_coords[1], res_x, res_y, radius)
        return _grid_nearest_scipy(
            points,
            values,
            grid_coords=grid_coords,
            res_x=res_x,
            res_y=res_y,
            radius=radius,
            n_threads=n_threads,
        )


class InverseDistance(Interpolator):
    """
    The InverseDistance interpolator uses nearby values with more weight on closer observations.

    Weights are ``1 / distance**power`` (then divided by their sum).
    """

    accepts_sample_weights = True
    _grid_method = "idw"

    def __init__(self, power: float = 2.0, neighborhood: PointNeighbours | GridNeighbours | None = None) -> None:
        """Set how quickly weights decrease with distance and which neighbours to use.

        :param power: Positive exponent in ``1 / distance**power`` (larger values give more weight to closer values).
        :param neighborhood: Optional point search or raster window. By default, select neighbours from the input type.
        """

        if not np.isfinite(power) or power <= 0:
            raise ValueError("InverseDistance power must be finite and strictly positive.")
        self.power = float(power)
        super().__init__(neighborhood=neighborhood)

    def coefficients(self, data: LocalData) -> LinearCoefficients | None:
        """Return normalized inverse-distance weights, averaging observations exactly at the target."""

        if data.distances is not None:
            distances = np.asarray(data.distances, dtype=np.float64)
        elif data.coordinates is not None and data.target is not None:
            distances = np.linalg.norm(data.coordinates - data.target, axis=1)
        else:
            raise ValueError("InverseDistance requires distances or coordinates and a target.")

        # If values lie exactly at the requested coordinate, we average those alone (as in point cloud gridding)
        exact = distances == 0
        if np.any(exact):
            weights = exact.astype(np.float64)
        else:
            weights = distances**-self.power
            if data.sample_weights is not None:
                weights *= data.sample_weights

        # Divide by the total weight so the prediction is an average, including any supplied sample weights
        total = float(np.sum(weights))
        if total == 0:
            return None
        return LinearCoefficients(weights=data.precision_weights(weights))

    def predict(self, data: LocalData) -> float:
        """Calculate the weighted average, or NaN if the weights sum to zero."""

        affine = self.coefficients(data)
        if affine is None:
            return float("nan")
        return float(np.dot(affine.weights, data.values))

    def _grid_points(
        self,
        points: NDArrayNum,
        values: NDArrayNum,
        grid_coords: tuple[NDArrayNum, NDArrayNum],
        *,
        res_x: float,
        res_y: float,
        radius: float,
        min_points: int,
        n_threads: int,
        engine: Literal["scipy", "numba"],
    ) -> NDArrayNum:
        """Dispatch radius-based gridding to the selected SciPy or Numba engine."""

        if not np.isfinite(radius):
            raise ValueError("Circular gridding methods require a finite dist_nodata_pixel support radius.")
        x_coords, y_coords = grid_coords
        if engine == "numba":
            return _grid_radius_idw_numba(
                points,
                values,
                float(np.min(x_coords)),
                float(np.min(y_coords)),
                res_x,
                res_y,
                len(x_coords),
                len(y_coords),
                radius,
                self.power,
                min_points,
            )

        def calculate(
            block: NDArrayNum,
            pairs: Any,
            queries: NDArrayNum,
            point_tree: cKDTree,
            counts: NDArrayNum,
            valid: NDArrayBool,
        ) -> None:
            """Apply inverse-distance weights to one group of grid cells."""

            _grid_sparse_idw(block, pairs, queries, point_tree, values, valid, res_x, res_y, self.power)

        return _grid_radius_scipy(
            points,
            values,
            grid_coords,
            res_x=res_x,
            res_y=res_y,
            radius=radius,
            min_points=min_points,
            minimum_inputs=1,
            calculate=calculate,
        )


class ScipyInterpolator(Interpolator):
    """
    The ScipyInterpolator class uses SciPy to predict values on a raster or from irregular points.

    The method selects nearest, linear, cubic or another SciPy interpolation method. Raster inputs use the regular
    grid directly. GridNeighbours must contain the natural stencil of nearest, linear, slinear or PCHIP; fitted
    cubic/quintic splines do not accept a raster window. For irregular points, the available methods are nearest,
    linear and cubic, with an optional PointNeighbours selection.
    """

    _uses_error_covariance = False

    def __init__(
        self,
        method: ScipyInterpolationMethod = "linear",
        neighborhood: PointNeighbours | GridNeighbours | None = None,
    ) -> None:
        """Select a SciPy method and, optionally, the points/cells used in each prediction.

        :param method: SciPy interpolation method.
        :param neighborhood: Optional point search for nearest/linear/cubic, or a raster window containing the
            natural nearest/linear/slinear/PCHIP stencil. Extra raster cells do not affect the result.
        """

        if method not in ("nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"):
            raise ValueError("Unknown SciPy interpolation method.")
        # A subclass can replace predict(), so only built-in classes impose SciPy's neighborhood restrictions
        is_builtin = type(self) in (ScipyInterpolator, Linear, Cubic)
        if is_builtin and isinstance(neighborhood, PointNeighbours) and method not in ("nearest", "linear", "cubic"):
            raise ValueError("PointNeighbours requires a nearest, linear, or cubic SciPy method.")
        self.method = method
        self.interpolation_order = INTERPOLATION_ORDERS[method]
        super().__init__(neighborhood=neighborhood)
        if is_builtin and isinstance(neighborhood, GridNeighbours):
            _check_regular_grid_neighbours(self, method)

    def predict(self, data: LocalData) -> float:
        """Predict a value at the requested coordinate with scipy.interpolate.griddata()."""

        if data.coordinates is None or data.target is None:
            raise ValueError("ScipyInterpolator requires coordinates and a target.")
        from scipy.interpolate import griddata

        result = griddata(data.coordinates, data.values, data.target[None, :], method=self.method)
        return float(result[0])

    def _grid_points(
        self,
        points: NDArrayNum,
        values: NDArrayNum,
        grid_coords: tuple[NDArrayNum, NDArrayNum],
        *,
        res_x: float,
        res_y: float,
        radius: float,
        min_points: int,
        n_threads: int,
        engine: Literal["scipy", "numba"],
    ) -> NDArrayNum:
        """Interpolate the output grid using nearest points or SciPy's linear or cubic triangulation."""

        if self.method == "nearest":
            return Nearest()._grid_points(
                points,
                values,
                grid_coords,
                res_x=res_x,
                res_y=res_y,
                radius=radius,
                min_points=min_points,
                n_threads=n_threads,
                engine=engine,
            )
        if engine == "numba":
            raise ValueError(f"The Numba gridding engine does not support resampling={self.method!r}.")
        return _grid_triangulation_scipy(
            points,
            values,
            grid_coords,
            method=cast(Literal["linear", "cubic"], self.method),
            radius=radius,
            res_x=res_x,
            res_y=res_y,
            n_threads=n_threads,
        )

    def _regular_grid_interpolator(
        self,
        points: tuple[NDArrayNum, NDArrayNum],
        values: NDArrayNum,
        *,
        fill_value: Number,
        bounds_error: bool,
        dist_nodata_spread: NodataSpread | None,
        nodata_propagation: NodataPropagation,
    ) -> Callable[[tuple[NDArrayNum, NDArrayNum]], NDArrayNum]:
        """Prepare SciPy interpolation on regular raster coordinates with this method."""

        return _interpn_interpolator(
            points=points,
            values=values,
            method=self.method,
            fill_value=fill_value,
            bounds_error=bounds_error,
            dist_nodata_spread=dist_nodata_spread,
            nodata_propagation=nodata_propagation,
        )


class Linear(ScipyInterpolator):
    """
    The Linear interpolator predicts values by linear interpolation between nearby observations.

    Raster.interp_points() uses bilinear interpolation on the regular grid, including with an explicit GridNeighbours
    window. PointCloud.grid() interpolates within triangles formed from the selected points.
    """

    _grid_method = "linear"

    def __init__(self, neighborhood: PointNeighbours | GridNeighbours | None = None) -> None:
        """Select an optional point search or raster window for linear interpolation.

        :param neighborhood: Neighbours to use, or None to select them from the input type.
        """

        super().__init__(method="linear", neighborhood=neighborhood)

    def coefficients(self, data: LocalData) -> LinearCoefficients | None:
        """Return linear weights from the surrounding triangle (or its equivalent in other dimensions)."""

        if data.coordinates is None or data.target is None:
            raise ValueError("Linear interpolation requires coordinates and a target.")
        coordinates = np.asarray(data.coordinates, dtype=np.float64)
        target = np.asarray(data.target, dtype=np.float64)
        dimensions = coordinates.shape[1]

        # In one dimension, we interpolate between the coordinates on either side of the requested point
        if dimensions == 1:
            order = np.argsort(coordinates[:, 0])
            sorted_coordinates = coordinates[order, 0]
            upper = int(np.searchsorted(sorted_coordinates, target[0], side="right"))
            if upper == 0 or upper == len(sorted_coordinates):
                return None
            lower = upper - 1
            width = sorted_coordinates[upper] - sorted_coordinates[lower]
            if width == 0:
                return None
            upper_weight = (target[0] - sorted_coordinates[lower]) / width
            weights = np.zeros(len(data.values), dtype=np.float64)
            weights[order[lower]] = 1 - upper_weight
            weights[order[upper]] = upper_weight
            return LinearCoefficients(weights=weights)

        # Find the triangle containing the requested point (a simplex in more than two dimensions)
        from scipy.spatial import Delaunay

        triangulation = Delaunay(coordinates)
        simplex = int(triangulation.find_simplex(target))
        if simplex < 0:
            return None

        # SciPy gives the weights for all but one vertex; the last completes their sum to one
        transform = triangulation.transform[simplex]
        leading_weights = transform[:dimensions] @ (target - transform[dimensions])
        simplex_weights = np.concatenate((leading_weights, [1 - np.sum(leading_weights)]))
        weights = np.zeros(len(data.values), dtype=np.float64)
        weights[triangulation.simplices[simplex]] = simplex_weights
        return LinearCoefficients(weights=weights)


class Cubic(ScipyInterpolator):
    """The Cubic interpolator uses SciPy's cubic interpolation for rasters or irregular points.

    Use it in Raster.interp_points() or PointCloud.grid(). Supply neighborhood to select the nearby values used
    for each prediction from irregular points. Raster cubic interpolation fits the full grid and does not accept
    GridNeighbours.
    """

    _grid_method = "cubic"

    def __init__(self, neighborhood: PointNeighbours | GridNeighbours | None = None) -> None:
        """Select an optional point search or raster window for cubic interpolation.

        :param neighborhood: Neighbours to use, or None to select them from the input type.
        """

        super().__init__(method="cubic", neighborhood=neighborhood)


@_jit(nopython=True, cache=True)
def _convolution_weight(distance: float, kernel: int) -> float:
    """Return a cubic convolution, cubic B-spline, or Lanczos kernel weight."""

    distance = abs(distance)
    if kernel == 0:
        if distance < 1:
            return 1 - 2.5 * distance**2 + 1.5 * distance**3
        if distance < 2:
            return 2 - 4 * distance + 2.5 * distance**2 - 0.5 * distance**3
        return 0.0
    if kernel == 1:
        if distance < 1:
            return (4 - 6 * distance**2 + 3 * distance**3) / 6
        if distance < 2:
            return (2 - distance) ** 3 / 6
        return 0.0
    if distance >= 3:
        return 0.0
    if distance < 1e-12:
        return 1.0
    return (sin(pi * distance) / (pi * distance)) * (sin(pi * distance / 3) / (pi * distance / 3))


@_jit(nopython=True, cache=True)
def _convolve_raster_points(
    values: NDArrayNum,
    source_rows: NDArrayNum,
    source_columns: NDArrayNum,
    kernel: int,
    propagate_nodata: bool,
    require_center: bool,
) -> NDArrayNum:
    """Apply one separable convolution kernel at each regular-grid coordinate."""

    height, width = values.shape
    output = np.full(len(source_rows), np.nan, dtype=np.float64)
    radius = 3 if kernel == 2 else 2
    for target in range(len(output)):
        row = source_rows[target]
        column = source_columns[target]
        if not np.isfinite(row) or not np.isfinite(column):
            continue
        nearest_row = int(np.floor(row + 0.5))
        nearest_column = int(np.floor(column + 0.5))
        if nearest_row < 0 or nearest_row >= height or nearest_column < 0 or nearest_column >= width:
            continue
        if require_center and not np.isfinite(values[nearest_row, nearest_column]):
            continue

        # Only valid cells contribute; divide by their total kernel weight near edges or nodata
        row_start = int(np.floor(row)) + 1 - radius
        column_start = int(np.floor(column)) + 1 - radius
        incomplete_stencil = (
            row_start < 0 or column_start < 0 or row_start + 2 * radius > height or column_start + 2 * radius > width
        )
        weighted_sum = 0.0
        weight_sum = 0.0
        invalid = False
        for source_row in range(row_start, row_start + 2 * radius):
            if source_row < 0 or source_row >= height:
                continue
            row_weight = _convolution_weight(row - source_row, kernel)
            for source_column in range(column_start, column_start + 2 * radius):
                if source_column < 0 or source_column >= width:
                    continue
                weight = row_weight * _convolution_weight(column - source_column, kernel)
                value = values[source_row, source_column]
                if np.isfinite(value):
                    weighted_sum += weight * value
                    weight_sum += weight
                else:
                    invalid = True
        if kernel == 0 and require_center and (invalid or incomplete_stencil):
            # GDAL uses bilinear values when missing cells interrupt the cubic stencil
            row_base = int(np.floor(row))
            column_base = int(np.floor(column))
            weighted_sum = 0.0
            weight_sum = 0.0
            for source_row in range(row_base, row_base + 2):
                if source_row < 0 or source_row >= height:
                    continue
                row_weight = 1 - abs(row - source_row)
                for source_column in range(column_base, column_base + 2):
                    if source_column < 0 or source_column >= width:
                        continue
                    value = values[source_row, source_column]
                    if np.isfinite(value):
                        weight = row_weight * (1 - abs(column - source_column))
                        weighted_sum += weight * value
                        weight_sum += weight
        if weight_sum != 0 and not (propagate_nodata and invalid):
            output[target] = weighted_sum / weight_sum
    return output


class RasterConvolution(Interpolator):
    """Interpolate a raster with GDAL-compatible cubic or Lanczos convolution.

    Use RasterConvolution("cubic"), RasterConvolution("cubic_spline"), or RasterConvolution("lanczos") in
    Raster.reproject() or Raster.interp_at_points(). The regular raster grid defines the kernel spacing; this
    interpolator does not apply to irregular point clouds. Cubic() remains SciPy's separate cubic spline method.
    """

    _uses_error_covariance = False

    def __init__(self, kernel: Literal["cubic", "cubic_spline", "lanczos"]) -> None:
        """Select the GDAL-style convolution kernel for a regular raster.

        :param kernel: Cubic convolution, cubic B-spline, or Lanczos windowed sinc.
        """

        if kernel not in ("cubic", "cubic_spline", "lanczos"):
            raise ValueError("Raster convolution kernel must be 'cubic', 'cubic_spline', or 'lanczos'.")
        super().__init__()
        self.kernel = kernel
        self.radius = 3 if kernel == "lanczos" else 2

    def predict(self, data: LocalData) -> float:
        """Require a regular raster grid for convolution instead of irregular observations."""

        raise TypeError("RasterConvolution requires a regular raster grid.")

    def _interpolate_grid(
        self,
        values: NDArrayNum,
        rows: NDArrayNum,
        columns: NDArrayNum,
        nodata_propagation: NodataPropagation,
    ) -> NDArrayNum:
        """Interpolate array values at source row/column coordinates measured from cell centers."""

        kernel_code = {"cubic": 0, "cubic_spline": 1, "lanczos": 2}[self.kernel]
        return _convolve_raster_points(
            values,
            np.asarray(rows, dtype=np.float64).reshape(-1),
            np.asarray(columns, dtype=np.float64).reshape(-1),
            kernel_code,
            nodata_propagation == "propagate",
            nodata_propagation == "nearest",
        )


# 1.3/ Select an interpolator from a method name
##############################################


def _resolve_interpolator(method: ScipyInterpolationMethod | Interpolator) -> Interpolator:
    """Turn a public method name into the corresponding Interpolator.

    :param method: Interpolation method name or an Interpolator supplied by the user.
    :returns: The supplied Interpolator or the one for the requested built-in method.
    """

    if isinstance(method, Interpolator):
        return method
    if method == "nearest":
        return Nearest()
    if method == "linear":
        return Linear()
    if method == "cubic":
        return Cubic()
    if method in INTERPOLATION_ORDERS:
        return ScipyInterpolator(method=cast(ScipyInterpolationMethod, method))
    raise ValueError(f"Unknown interpolation method: {method!r}.")


def _regular_interpolation_method(method: ScipyInterpolationMethod | Interpolator) -> ScipyInterpolationMethod | None:
    """Return a SciPy method that can use the raster grid directly, or None for a custom calculation."""

    operator = _resolve_interpolator(method)
    operator = getattr(operator, "_wrapped_operator", operator)
    if type(operator) is Nearest:
        _check_regular_grid_neighbours(operator, "nearest")
        return "nearest"
    if type(operator) in (ScipyInterpolator, Linear, Cubic):
        method_name = cast(ScipyInterpolator, operator).method
        _check_regular_grid_neighbours(operator, method_name)
        return method_name
    return None


def _resolve_irregular_interpolator(
    method: IrregularInterpolationMethod | Interpolator, *, distance_power: float
) -> Interpolator:
    """Select a point interpolator and validate methods that depend on the source geometry."""

    # We check that a variogram passed for kriging uses both X/Y dimensions
    if isinstance(method, Kriging):
        method.validate_geospatial_support(2)
    if isinstance(method, Interpolator):
        if type(method) is ScipyInterpolator and method.method not in ("nearest", "linear", "cubic"):
            raise ValueError(f"Point gridding does not support the regular-grid SciPy method {method.method!r}.")
        return method

    # IDW exponent from grid(); other names share the regular interpolation classes
    if method == "idw":
        return InverseDistance(power=distance_power)
    return _resolve_interpolator(method)


################################
# 2/ REGULAR RASTER INTERPOLATION
################################


class _RegularGridWeights(Interpolator):
    """Apply nearest or bilinear weights that were calculated from a regular grid."""

    _uses_error_covariance = False

    def coefficients(self, data: LocalData) -> LinearCoefficients:
        """Return the normalized weights for the finite source cells selected from the grid."""

        if data.interpolation_weights is None:
            raise ValueError("Regular-grid interpolation requires prepared interpolation_weights.")
        return LinearCoefficients(data.interpolation_weights)


class _RegularGridPrediction(Interpolator):
    """Repeat a fitted raster interpolation on perturbed values without changing its grid or algorithm."""

    _uses_error_covariance = False

    def __init__(
        self,
        shape: tuple[int, int],
        dtype: DTypeLike,
        transform: rio.transform.Affine,
        method: ScipyInterpolationMethod,
        area_or_point: Literal["Area", "Point"] | None,
        shift_area_or_point: bool | None,
        nodata_propagation: NodataPropagation,
    ) -> None:
        """Store the original raster layout and interpolation options for repeated error draws."""

        super().__init__()
        self.shape = shape
        self.dtype = _interp_output_dtype(dtype)
        self.transform = transform
        self.method = method
        self.area_or_point = area_or_point
        self.shift_area_or_point = shift_area_or_point
        self.nodata_propagation = nodata_propagation

    def predict(self, data: LocalData) -> float:
        """Interpolate one target from its original raster band with the supplied values."""

        return float(self.predict_batch([data])[0])

    def predict_batch(
        self, data: Sequence[LocalData], *, nodata_propagation: NodataHandling | None = None
    ) -> NDArrayNum:
        """Fit each perturbed band once, then evaluate all its requested target coordinates."""

        from geoutils.interface.resampling import _interp_points_base
        from geoutils.raster.referencing import _xy2ij

        output = np.full(len(data), np.nan)
        groups: dict[int, list[int]] = {}
        for index, local in enumerate(data):
            if len(local.values):
                band = int(local.source_ids[0])
                groups.setdefault(band, []).append(index)

        # Repeated source IDs receive identical perturbations, so all targets in a band share one source array
        for indexes in groups.values():
            local = data[indexes[0]]
            source = np.full(self.shape, np.nan, dtype=self.dtype)
            # IDs identify global observations; coordinates locate them within this loaded raster rectangle
            assert local.coordinates is not None
            rows, cols = _xy2ij(
                local.coordinates[:, 0],
                local.coordinates[:, 1],
                transform=self.transform,
                area_or_point=None,
                shift_area_or_point=False,
                op=np.float64,
            )
            source[np.floor(rows).astype(int), np.floor(cols).astype(int)] = local.values
            targets = np.asarray([data[index].target for index in indexes])
            values = _interp_points_base(
                source,
                self.transform,
                (targets[:, 0], targets[:, 1]),
                method=self.method,
                area_or_point=self.area_or_point,
                shift_area_or_point=self.shift_area_or_point,
                nodata_propagation=self.nodata_propagation,
            )
            output[indexes] = values
        return output


def _interp_output_dtype(dtype: DTypeLike, *, validity_only: bool = False) -> DTypeLike:
    """
    Return an interpolation dtype that can represent NaNs.

    Validity-only interpolation uses float32 regardless of the original values, following _resample_at_points().
    """

    # Promote booleans too so nodata interpolation results do not become True
    return np.float32 if validity_only or np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_) else dtype


def _interpolate_array_band(
    array: NDArrayNum,
    src_rows: NDArrayNum,
    src_cols: NDArrayNum,
    method: Literal["nearest", "linear"],
    nodata_propagation: NodataPropagation,
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int | None = None,
) -> NDArrayNum:
    """
    Interpolate one array band using normalized finite-value weights.

    :param array: Two-dimensional source values.
    :param src_rows: Source row indices of destination pixel centers.
    :param src_cols: Source column indices of destination pixel centers.
    :param method: Nearest-neighbor or linear interpolation.
    :param nodata_propagation: Rule used to handle invalid source values.
    :param dist_nodata_spread: Optional extra distance for spreading invalid cells.

    :return: Interpolated floating-point array.
    """

    # Convert masked and integer inputs to floating values where invalid cells are represented by NaN
    source = np.array(np.ma.getdata(array), dtype=_interp_output_dtype(array.dtype), copy=True)
    if np.ma.isMaskedArray(array):
        source[np.ma.getmaskarray(array)] = np.nan
    valid = np.isfinite(source)

    # Pixel areas extend half a cell beyond their centers but do not extend farther
    inside = (src_rows >= -0.5) & (src_rows < source.shape[0] - 0.5)
    inside &= (src_cols >= -0.5) & (src_cols < source.shape[1] - 0.5)

    # When every cell is finite, SciPy can interpolate values without a second pass over validity
    if np.all(valid):
        order = 0 if method == "nearest" else 1
        output = map_coordinates(source, (src_rows, src_cols), order=order, mode="nearest", prefilter=False)
        output[~inside] = np.nan
        return output

    # Interpolate values and validity separately so invalid neighbors never contribute numerically
    order = 0 if method == "nearest" else 1
    filled = np.where(valid, source, 0)
    numerator = map_coordinates(filled, (src_rows, src_cols), order=order, mode="nearest", prefilter=False)
    weights = map_coordinates(
        valid.astype(np.float32), (src_rows, src_cols), order=order, mode="nearest", prefilter=False
    )

    # Normalize the remaining finite weights as GDAL does for nearest and bilinear resampling
    output = np.full(numerator.shape, np.nan, dtype=_interp_output_dtype(source.dtype))
    np.divide(numerator, weights, out=output, where=weights > 0)

    output[~inside] = np.nan

    if nodata_propagation == "propagate":
        # A propagated output is invalid when any weighted source value is invalid
        output[weights < 1 - np.finfo(np.float32).eps] = np.nan
    elif nodata_propagation == "nearest":
        # GDAL invalidates an output when its nearest source cell is invalid
        invalid_center = map_coordinates(
            (~valid).astype(np.uint8),
            (src_rows, src_cols),
            order=0,
            mode="nearest",
            prefilter=False,
        )
        output[invalid_center.astype(bool)] = np.nan

    if dist_nodata_spread is not None:
        # A distance choice masks the original nodata cells and any cells within the requested distance
        distance = _get_dist_nodata_spread(
            order=order,
            dist_nodata_spread=dist_nodata_spread,
        )
        invalid = ~valid
        if distance > 0:
            invalid = binary_dilation(invalid, iterations=distance)
        spread_mask = map_coordinates(
            invalid.astype(np.uint8),
            (src_rows, src_cols),
            order=0,
            mode="nearest",
            prefilter=False,
        )
        output[spread_mask.astype(bool)] = np.nan
    return output


def _get_dist_nodata_spread(order: int, dist_nodata_spread: NodataSpread) -> int:
    """
    Derive distance of nodata spreading based on interpolation order.

    :param order: Interpolation order.
    :param dist_nodata_spread: Extra nodata spreading distance, either half-order rounded up or down, or a fixed
        integer.
    """

    return _nodata_spread_distance(order=order, dist_nodata_spread=dist_nodata_spread)


def _interpn_interpolator(
    points: tuple[NDArrayNum, NDArrayNum],
    values: NDArrayNum,
    fill_value: Number = np.nan,
    bounds_error: bool = False,
    dist_nodata_spread: NodataSpread | None = None,
    method: ScipyInterpolationMethod | None = None,
    nodata_propagation: NodataPropagation = "nearest",
) -> Callable[[tuple[NDArrayNum, NDArrayNum]], NDArrayNum]:
    """
    Create a SciPy interpolator and apply the chosen nodata rule.

    By default, the result is nodata when its nearest source cell is invalid (as in GDAL). A distance choice extends
    the mask around nodata cells, while "propagate" masks results that use a nodata source value. The method and
    default nodata choice can be configured with geoutils.config["interpolation_method"] and
    geoutils.config["interpolation_nodata_handling"].

    Gives the exact same result as scipy.interpolate.interpn, and allows interpolator to be re-used if required (
    for speed).
    In practice, returns either a NaN-modified RegularGridInterpolator or a NaN-modified RectBivariateSpline object,
    both expecting a tuple of X/Y coordinates to be evaluated.

    For input arguments, see scipy.interpolate.RegularGridInterpolator.
    The private dist_nodata_spread argument receives the distance resolved from Raster.interp_points() nodata_handling.

    Adapted from:
    https://github.com/scipy/scipy/blob/44e4ebaac992fde33f04638b99629d23973cb9b2/scipy/interpolate/_rgi.py#L743.
    """

    # If interpolation method undefined, default to the global system config
    if method is None:
        method = config["interpolation_method"]

    # Select the mask from the chosen nodata rule or distance
    order = INTERPOLATION_ORDERS[method]
    propagation = _validate_nodata_propagation(nodata_propagation)
    mask_distance = _nodata_mask_distance(
        order=order,
        nodata_propagation=propagation,
        dist_nodata_spread=dist_nodata_spread,
    )

    # We interpolate the mask separately so filling source gaps below does not hide which cells were nodata
    mask_nan = ~np.isfinite(values)
    has_missing = bool(np.any(mask_nan))
    fill_array = np.asarray(fill_value)
    fill_is_nan = np.issubdtype(fill_array.dtype, np.number) and bool(np.isnan(fill_array))
    interp_mask: RegularGridInterpolator | None = None
    if mask_distance is not None and (has_missing or not fill_is_nan):
        if mask_distance > 0:
            new_mask = binary_dilation(mask_nan, iterations=mask_distance).astype("uint8")
        else:
            new_mask = mask_nan.astype("uint8")
        interp_mask = RegularGridInterpolator(
            points,
            new_mask,
            method="nearest",
            bounds_error=bounds_error,
            fill_value=1,
        )

    def evaluate_nodata_mask(xi: tuple[NDArrayNum, NDArrayNum]) -> NDArrayNum:
        """Interpolate the mask, choosing the same cell as GDAL when a point is equally close to two cells."""

        assert interp_mask is not None
        if propagation != "nearest" or dist_nodata_spread is not None:
            return interp_mask(xi)

        # When a query lies exactly between cells, shift it toward the cell that GDAL selects
        shifted_queries = []
        for axis_points, axis_queries in zip(points, xi):
            tie_direction = np.inf if axis_points[-1] > axis_points[0] else -np.inf
            shifted_queries.append(np.nextafter(np.asarray(axis_queries), tie_direction))
        return interp_mask(tuple(shifted_queries))

    # Most methods (cubic, quintic, etc) do not support NaNs and require an array full of valid values
    # We replace thus replace all NaN values by nearest neighbours to give surrounding values of the same order of
    # magnitude and minimize interpolation errors near NaNs (errors of 10e-2/e-5 relative to the values)
    # Elegant solution from: https://stackoverflow.com/questions/5551286/filling-gaps-in-a-numpy-array for a fast
    # nearest neighbour fill
    if has_missing:
        indices = distance_transform_edt(mask_nan, return_distances=False, return_indices=True)
        values = values[tuple(indices)]

    # For the RegularGridInterpolator
    if method in RegularGridInterpolator._ALL_METHODS:
        # We create the classic interpolator
        interp = RegularGridInterpolator(
            points, values, method=method, bounds_error=bounds_error, fill_value=fill_value
        )

        # We create a new interpolator callable that propagates nodata as defined above
        def regulargrid_interpolator_with_nan(xi: tuple[NDArrayNum, NDArrayNum]) -> NDArrayNum:
            """Interpolate values and mask results affected by nodata input cells."""

            results = interp(xi)
            if interp_mask is not None:
                invalids = evaluate_nodata_mask(xi)
                results[invalids.astype(bool)] = np.nan

            return results

        return regulargrid_interpolator_with_nan

    # For the RectBivariateSpline
    else:
        # The coordinates must be in ascending order, which requires flipping the array too (more costly)
        interp = RectBivariateSpline(np.flip(points[0]), points[1], np.flip(values[:], axis=0))

        # We create a new interpolator callable that propagates nodata as defined above, and supports fill_value
        def rectbivariate_interpolator_with_fillvalue(xi: tuple[NDArrayNum, NDArrayNum]) -> NDArrayNum:
            """Interpolate spline values, fill outside targets and apply the nodata mask."""

            # RectBivariateSpline doesn't support fill_value, so we need to wrap here to add them
            xi_arr = np.array(xi).T
            xi_shape = xi_arr.shape
            xi_arr = xi_arr.reshape(-1, xi_arr.shape[-1])
            idx_valid = np.all(
                (
                    points[0][-1] <= xi_arr[:, 0],
                    xi_arr[:, 0] <= points[0][0],
                    points[1][0] <= xi_arr[:, 1],
                    xi_arr[:, 1] <= points[1][-1],
                ),
                axis=0,
            )
            # Make a copy of values for RectBivariateSpline
            result = np.empty_like(xi_arr[:, 0])
            result[idx_valid] = interp.ev(xi_arr[idx_valid, 0], xi_arr[idx_valid, 1])
            result[np.logical_not(idx_valid)] = fill_value

            # Mask the interpolated values selected by the nodata rule
            results = np.atleast_1d(result.reshape(xi_shape[:-1]))
            if interp_mask is not None:
                invalids = evaluate_nodata_mask(xi)
                results[invalids.astype(bool)] = np.nan

            return results

        return rectbivariate_interpolator_with_fillvalue


#################################
# 3/ POINT CLOUD GRID INTERPOLATION
#################################

# 3.1/ Selected point neighborhoods (Numba)
##########################################


@_jit(nopython=True, cache=True)
def _interpolate_point_groups_numba(
    values: NDArrayNum,
    valid: NDArrayBool,
    distances: NDArrayNum,
    offsets: NDArrayNum,
    method: str,
    power: float,
    minimum_inputs: int,
    propagate_nodata: bool,
) -> NDArrayNum:
    """Interpolate flattened point neighborhoods with nearest or inverse-distance weights.

    Offsets delimit the observations for each target. Return one value per target, with NaN for groups rejected
    by the minimum input count or nodata rule.
    """

    output = np.full(len(offsets) - 1, np.nan)
    for target in range(len(output)):
        start, stop = offsets[target], offsets[target + 1]
        count = 0
        missing = False
        nearest = -1
        nearest_distance = np.inf
        exact_sum, exact_count = 0.0, 0
        exact_missing = False

        # Count usable observations and identify the contributors for nearest and exact-location IDW
        for index in range(start, stop):
            if valid[index]:
                count += 1
            else:
                missing = True
            if (valid[index] or propagate_nodata) and distances[index] < nearest_distance:
                nearest, nearest_distance = index, distances[index]
            if distances[index] == 0:
                if valid[index]:
                    exact_sum += values[index]
                    exact_count += 1
                else:
                    exact_missing = True
        if count < minimum_inputs:
            continue
        if method == "nearest":
            if nearest >= 0 and valid[nearest]:
                output[target] = values[nearest]
            continue
        if exact_count > 0 or (propagate_nodata and exact_missing):
            if exact_count > 0 and not (propagate_nodata and exact_missing):
                output[target] = exact_sum / exact_count
            continue
        if propagate_nodata and missing:
            continue

        # All other contributions use the same finite observations selected by the Python operator path
        total, weight_sum = 0.0, 0.0
        for index in range(start, stop):
            if not valid[index]:
                continue
            weight = distances[index] ** (-power)
            total += weight * values[index]
            weight_sum += weight
        if weight_sum > 0:
            output[target] = total / weight_sum
    return output


# 3.2/ Nearest neighbour (SciPy spatial tree or Numba loop)
########################################################

# Nearest-neighbour using either SciPy spatial tree or a compiled Numba loop
############################################################################


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
    # Build one reusable index of source coordinates
    point_tree = cKDTree(points)
    output = np.empty((len(y_coords), len(x_coords)), dtype=np.float64)

    # Reuse one nearest-neighbor tree while limiting temporary query coordinates
    for row_start in range(0, len(y_coords), _GRID_QUERY_ROWS):
        row_stop = min(row_start + _GRID_QUERY_ROWS, len(y_coords))
        queries = _build_grid_queries(x_coords, y_coords[row_start:row_stop])
        # Query the closest source index for each cell and copy its value into the block
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


@_jit(nopython=True, cache=True)
def _grid_nearest_numba(
    points: NDArrayNum,
    values: NDArrayNum,
    x_coords: NDArrayNum,
    y_coords: NDArrayNum,
    res_x: float,
    res_y: float,
    radius: float,
) -> NDArrayNum:
    """Interpolate nearest values by comparing source distances in a loop."""

    output = np.full((len(y_coords), len(x_coords)), np.nan, dtype=np.float64)
    radius_squared = radius * radius
    finite_radius = np.isfinite(radius)

    # A cell keeps the value of its closest point in source coordinates
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


# 3.3/ Inverse distance weighting within a circular neighbourhood
###############################################################

# Circular-neighborhood engines using compiled accumulation or SciPy sparse neighborhoods
########################################################################################


@_jit(nopython=True, cache=True)
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
    """Compute inverse-distance weighting (IDW) in nearby cells inside each point's support radius."""

    output = np.zeros((height, width), dtype=np.float64)
    weights = np.zeros((height, width), dtype=np.float64)
    exact_counts = np.zeros((height, width), dtype=np.int32)
    counts = np.zeros((height, width), dtype=np.int32)
    radius_squared = radius * radius

    # Accumulate exact values or weighted values in cells inside each point's support
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
                    # Points at exact location discard earlier weighted contributions and are averaged together
                    if weights[row, col] >= 0:
                        output[row, col] = 0
                        weights[row, col] = -1
                    output[row, col] += values[point_index]
                    exact_counts[row, col] += 1
                elif weights[row, col] >= 0:
                    # Select neighbors by output-pixel radius, but calculate weights in source coordinate units
                    coordinate_distance_squared = ((col - point_x) * res_x) ** 2 + ((row - point_y) * res_y) ** 2
                    weight = coordinate_distance_squared ** (-power / 2)
                    # Accumulate the weighted value and weight for the final weighted mean
                    output[row, col] += weight * values[point_index]
                    weights[row, col] += weight

    # Finalize each cell, giving exact samples precedence over min_points and IDW
    required_points = max(1, min_points)
    for row in range(height):
        for col in range(width):
            if exact_counts[row, col] > 0:
                # Average source values located exactly at the cell center
                output[row, col] /= exact_counts[row, col]
            elif counts[row, col] < required_points:
                # Reject cells whose support contains too few source points
                output[row, col] = np.nan
            elif weights[row, col] > 0:
                # Normalize the accumulated weighted values by their total weight
                output[row, col] /= weights[row, col]
            else:
                # Leave cells without an exact or weighted source value empty
                output[row, col] = np.nan
    return output


def _grid_sparse_idw(
    block: NDArrayNum,
    pairs: Any,
    queries: NDArrayNum,
    point_tree: cKDTree,
    values: NDArrayNum,
    valid: NDArrayBool,
    res_x: float,
    res_y: float,
    distance_power: float,
) -> None:
    """Average nearby point values in each grid cell, weighted by inverse distance."""

    # Separate exact points from nonzero-distance points before weighting
    exact = pairs.data == 0
    exact_counts = np.bincount(pairs.row[exact], minlength=len(queries))
    exact_sums = np.bincount(
        pairs.row[exact],
        weights=values[pairs.col[exact]],
        minlength=len(queries),
    )
    nonzero = ~exact
    # The support radius uses output pixels, while IDW weights use source-coordinate distances
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


# 3.4/ Linear/cubic interpolation between irregular points
########################################################


def _grid_triangulation_scipy(
    points: NDArrayNum,
    values: NDArrayNum,
    grid_coords: tuple[NDArrayNum, NDArrayNum],
    method: Literal["linear", "cubic"],
    radius: float,
    res_x: float,
    res_y: float,
    n_threads: int,
) -> NDArrayNum:
    """Interpolate a complete grid by triangulation and apply the requested support radius."""

    # SciPy's triangulation methods require complete query grids
    xx, yy = np.meshgrid(grid_coords[0], grid_coords[1])
    aligned_dem = griddata(
        points=points,
        values=values,
        xi=(xx, yy),
        method=method,
        rescale=False,
    )

    # Triangulation fills the convex hull, so remove cells beyond the requested local support
    if np.isfinite(radius):
        _mask_grid_beyond_support(
            aligned_dem,
            points=points,
            grid_coords=grid_coords,
            res_x=res_x,
            res_y=res_y,
            radius=radius,
            n_threads=n_threads,
        )
    return aligned_dem


##########################
# 4/ KRIGING INTERPOLATION
##########################


KrigingBackend = Literal["gstools", "gpytorch"]


def _as_fitted_variogram(model: Variogram | VariogramModel) -> Variogram:
    """Use the fitted Variogram as supplied, or wrap a VariogramModel in one."""

    from geoutils.stats.variography import Variogram, VariogramModel

    if isinstance(model, Variogram):
        if model.model is None:
            raise ValueError("Kriging requires a fitted Variogram model.")
        return model
    if not isinstance(model, VariogramModel):
        raise TypeError("Kriging model must be a fitted Variogram or VariogramModel.")
    return Variogram(
        lags=np.empty(0),
        semivariance=np.empty(0),
        counts=np.empty(0, dtype=np.int64),
        model=model,
    )


def _model_ranges(model: VariogramModel) -> list[float]:
    """List the distances over which values are correlated for each component of the model."""

    if model.components:
        return [value for component in model.components for value in _model_ranges(component)]
    if model.effective_range is None:
        raise AssertionError("A validated base variogram model must define its effective range.")
    return [float(model.effective_range)]


def _active_dimensions(model: VariogramModel, coordinate_count: int) -> tuple[int, ...] | None:
    """Return the coordinate columns shared by every component of a variogram model."""

    components = model.components or (model,)
    selections = {component.active_dims for component in components if component.active_dims is not None}
    if len(selections) > 1:
        raise NotImplementedError("Kriging requires every variogram component to use the same active dimensions.")
    selected = next(iter(selections)) if selections else model.active_dims
    if selected is not None and (len(selected) == 0 or max(selected) >= coordinate_count):
        raise ValueError("Variogram active dimensions exceed the supplied coordinate columns.")
    return selected


class Kriging(Interpolator):
    """The Kriging interpolator predicts values from nearby observations and their spatial correlation.

    Pass a fitted variogram to describe how correlation changes with distance, then use this interpolator in
    Raster.interp_points() or PointCloud.grid(). The backend selects GSTools or GPyTorch for ordinary kriging;
    neighborhood and max_overlap control which nearby values are used.

    We reuse weights when points have the same positions relative to the requested coordinate (common in rasters).
    coefficients() returns those weights, and kriging_variance() gives the prediction variance from the model.

    :param model: Fitted GeoUtils variogram or its VariogramModel representation.
    :param backend: Library used to solve the kriging system, ``"gstools"`` or ``"gpytorch"``.
    :param neighborhood: Points or raster cells considered for each prediction. By default, select points/cells within
        the model's longest effective range (the distance over which its values are correlated).
    :param max_overlap: Optional distance limit for the effective range and the extra data read around chunk edges.
    :param exact: Include the nugget when predicting at a known source location.
    :param pseudo_inverse: Let GSTools use a pseudo inverse for redundant or coincident observations.
    :param cache_size: Maximum number of arrangements of nearby points for which calculated weights are stored.
    """

    accepts_sample_weights = False

    def __init__(
        self,
        model: Variogram | VariogramModel,
        *,
        backend: KrigingBackend = "gstools",
        neighborhood: PointNeighbours | GridNeighbours | None = None,
        max_overlap: float | None = None,
        exact: bool = True,
        pseudo_inverse: bool = True,
        cache_size: int = 256,
    ) -> None:
        """Check the covariance model, search radius, selected library, and cache size."""

        self.variogram = _as_fitted_variogram(model)
        assert self.variogram.model is not None
        if backend not in {"gstools", "gpytorch"}:
            raise ValueError("Kriging backend must be 'gstools' or 'gpytorch'.")
        if max_overlap is not None and (
            isinstance(max_overlap, (bool, np.bool_)) or not np.isfinite(max_overlap) or max_overlap <= 0
        ):
            raise ValueError("Kriging max_overlap must be finite and strictly positive.")
        if isinstance(cache_size, bool) or not isinstance(cache_size, (int, np.integer)) or cache_size < 0:
            raise ValueError("Kriging cache_size must be a non-negative integer.")

        # We search as far as the longest correlation range, unless max_overlap sets a shorter distance
        decorrelation_length = max(_model_ranges(self.variogram.model))
        self.decorrelation_length = decorrelation_length
        self.max_overlap = decorrelation_length if max_overlap is None else min(decorrelation_length, max_overlap)
        self.backend = backend
        self.exact = bool(exact)
        self.pseudo_inverse = bool(pseudo_inverse)
        self.cache_size = int(cache_size)
        super().__init__(neighborhood=neighborhood)
        self._coefficient_cache: OrderedDict[tuple[Any, ...], tuple[NDArrayNum, float]] = OrderedDict()

    def _selected_geometry(self, data: LocalData) -> tuple[NDArrayNum, NDArrayBool, tuple[int, ...] | None]:
        """Select the coordinate columns and source points inside the requested distance."""

        if data.coordinates is None or data.target is None:
            raise ValueError("Kriging requires source coordinates and a target coordinate.")
        coordinates = np.asarray(data.coordinates, dtype=np.float64)
        target = np.asarray(data.target, dtype=np.float64)
        assert self.variogram.model is not None
        active_dims = _active_dimensions(self.variogram.model, coordinates.shape[1])

        # Measure distances in the coordinates used by the model (for example X/Y, or X/Y/Z)
        selected_coordinates = coordinates if active_dims is None else coordinates[:, active_dims]
        selected_target = target if active_dims is None else target[np.asarray(active_dims)]
        relative = selected_coordinates - selected_target
        inside = np.linalg.norm(relative, axis=1) <= self.max_overlap * (1 + 1e-12)
        return relative, inside, active_dims

    def validate_geospatial_support(self, coordinate_count: int) -> None:
        """Check that the variogram uses all coordinates included in the neighbour search.

        :param coordinate_count: Number of spatial coordinates used by the operation (for example, two for X/Y).
        """

        model = self.variogram.model
        if model is None:
            raise AssertionError("A validated kriging operator must contain a fitted model.")
        active_dims = _active_dimensions(model, coordinate_count)
        if active_dims is not None and set(active_dims) != set(range(coordinate_count)):
            raise NotImplementedError(
                "Raster and point kriging require the variogram to use every spatial coordinate dimension."
            )

    def _cache_key(self, relative: NDArrayNum, active_dims: tuple[int, ...] | None) -> tuple[Any, ...]:
        """Identify the relative positions of nearby values so we can reuse weights after a translation."""

        # Round tiny coordinate differences to recognise the same raster neighbourhood at different locations
        rounded = np.round(np.asarray(relative, dtype=np.float64), decimals=12)
        return self.backend, self.exact, active_dims, rounded.shape, rounded.tobytes()

    @staticmethod
    def _ordered_selected(relative: NDArrayNum, inside: NDArrayBool) -> NDArrayNum:
        """Sort selected source positions by coordinate so equivalent neighborhoods have the same order."""

        selected = np.flatnonzero(inside)
        if len(selected) < 2:
            return selected
        keys = tuple(relative[selected, dimension] for dimension in reversed(range(relative.shape[1])))
        return selected[np.lexsort(keys)]

    def _gstools_coefficients(self, relative: NDArrayNum) -> tuple[NDArrayNum, float]:
        """Use GSTools to solve the ordinary kriging weights and variance."""

        gstools = import_optional("gstools", extra_name="geostat")
        converted = self.variogram.to_gstools(dim=relative.shape[1])
        positions = tuple(relative[:, dimension] for dimension in range(relative.shape[1]))

        # The weights depend on positions and the variogram, so zero values suffice to set up the calculation
        krige = gstools.krige.Ordinary(
            converted.model,
            cond_pos=positions,
            cond_val=np.zeros(len(relative), dtype=float),
            exact=self.exact,
            pseudo_inv=self.pseudo_inverse,
        )

        # Read the weights for a prediction at the origin (source positions are relative to that point)
        target = tuple(np.zeros(1, dtype=float) for _ in range(relative.shape[1]))
        prepared_target, _ = krige.pre_pos(target, mesh_type="unstructured")
        right_hand_side = krige._get_krige_vecs(prepared_target)  # noqa: SLF001
        solution = krige._krige_mat @ right_hand_side  # noqa: SLF001
        weights = np.asarray(solution[: len(relative), 0], dtype=np.float64)
        _, variance = krige(target, mesh_type="unstructured", return_var=True, store=False)
        return weights, max(float(np.asarray(variance).reshape(-1)[0]), 0.0)

    def _gpytorch_coefficients(self, relative: NDArrayNum) -> tuple[NDArrayNum, float]:
        """Use GPyTorch covariance values to solve the ordinary kriging weights and variance."""

        torch = import_optional("torch", extra_name="gp")
        import_optional("gpytorch", extra_name="gp")

        # Calculate covariance between source points and from each source point to the requested coordinate
        active_dims = tuple(range(relative.shape[1]))
        converted = self.variogram.to_gpytorch(active_dims=active_dims, trainable=False)
        coordinates = torch.as_tensor(relative, dtype=torch.float64)
        target = torch.zeros((1, relative.shape[1]), dtype=torch.float64)
        covariance = converted.kernel(coordinates, coordinates).to_dense()
        cross_covariance = converted.kernel(coordinates, target).to_dense().reshape(-1)

        # Add the nugget (uncorrelated variance), including it at coincident points if exact prediction is requested
        if converted.noise > 0:
            covariance = covariance + float(converted.noise) * torch.eye(len(relative), dtype=torch.float64)
            if self.exact:
                coincident = torch.all(coordinates == target, dim=1)
                cross_covariance = cross_covariance + float(converted.noise) * coincident

        # The extra row/column forces weights to sum to one, so a constant input gives the same constant prediction
        system = torch.zeros((len(relative) + 1, len(relative) + 1), dtype=torch.float64)
        system[:-1, :-1] = covariance
        system[:-1, -1] = 1
        system[-1, :-1] = 1
        right_hand_side = torch.cat((cross_covariance, torch.ones(1, dtype=torch.float64)))

        # Repeated coordinates can make the system singular, in which case we use a least-squares solution
        try:
            solution = torch.linalg.solve(system, right_hand_side)
        except RuntimeError:
            solution = torch.linalg.lstsq(system, right_hand_side.unsqueeze(1)).solution[:, 0]

        # Subtract the variance explained by the neighbours; the last solution value enforces the sum of weights
        target_variance = converted.kernel(target, target).to_dense().reshape(-1)[0] + float(converted.noise)
        variance = target_variance - torch.dot(solution[:-1], cross_covariance) - solution[-1]
        return solution[:-1].detach().cpu().numpy(), max(float(variance), 0.0)

    def _solve_geometry(self, relative: NDArrayNum, active_dims: tuple[int, ...] | None) -> tuple[NDArrayNum, float]:
        """Calculate weights and variance, or reuse them if these relative positions were already solved."""

        key = self._cache_key(relative, active_dims)
        cached = self._coefficient_cache.get(key)
        if cached is not None:
            self._coefficient_cache.move_to_end(key)
            return cached

        # Save the new result and remove the least recently used one when the cache is full
        solved = (
            self._gstools_coefficients(relative) if self.backend == "gstools" else self._gpytorch_coefficients(relative)
        )
        if self.cache_size > 0:
            self._coefficient_cache[key] = solved
            if len(self._coefficient_cache) > self.cache_size:
                self._coefficient_cache.popitem(last=False)
        return solved

    def _solve_with_observation_errors(
        self, relative: NDArrayNum, error_covariance: NDArrayNum
    ) -> tuple[NDArrayNum, float]:
        """Solve ordinary kriging after adding observation errors to the field covariance.

        The returned variance describes prediction of the field from noisy observations. Propagating the observation
        ErrorStructure separately reports only the contribution from those observation errors.
        """

        # The variogram describes the field; observation errors add covariance only among the measurements
        distances = np.linalg.norm(relative[:, None, :] - relative[None, :, :], axis=-1)
        covariance = self.variogram.covariance(distances) + error_covariance
        target_distances = np.linalg.norm(relative, axis=1)
        cross_covariance = np.asarray(self.variogram.covariance(target_distances), dtype=np.float64)

        # A nugget belongs to each observation separately, including observations at coincident coordinates
        assert self.variogram.model is not None
        nugget = self.variogram.model.nugget
        covariance -= nugget * ((distances == 0).astype(float) - np.eye(len(relative)))
        if not self.exact:
            cross_covariance -= nugget * (target_distances == 0)

        # Enforce weights summing to one, allowing a least-squares solution when observations are redundant
        system = np.ones((len(relative) + 1, len(relative) + 1))
        system[:-1, :-1] = covariance
        system[-1, -1] = 0
        rhs = np.append(cross_covariance, 1.0)
        try:
            solution = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(system, rhs, rcond=None)[0]

        # The constraint multiplier accounts for estimating the unknown constant mean
        variance = float(self.variogram.covariance(0)) - solution[:-1] @ cross_covariance - solution[-1]
        return solution[:-1], max(float(variance), 0.0)

    def coefficients(self, data: LocalData) -> LinearCoefficients | None:
        """Return ordinary kriging weights in the same order as the supplied source values."""

        relative, inside, active_dims = self._selected_geometry(data)
        selected = self._ordered_selected(relative, inside)
        if len(selected) == 0:
            return None

        # Values outside the selected neighbourhood receive zero weight, without changing the input order
        weights = np.zeros(len(data.values), dtype=np.float64)
        if len(selected) == 1:
            weights[selected[0]] = 1
            return LinearCoefficients(weights)
        if data.error_covariance is None or not np.any(data.error_covariance):
            solved, _ = self._solve_geometry(relative[selected], active_dims)
        else:
            solved, _ = self._solve_with_observation_errors(
                relative[selected], data.error_covariance[np.ix_(selected, selected)]
            )
        weights[selected] = solved
        return LinearCoefficients(weights)

    def kriging_variance(self, data: LocalData) -> float:
        """Return the model interpolation variance for the same source points used by coefficients().

        :param data: Nearby source values, their coordinates, and the coordinate at which to predict.
        :returns: Ordinary kriging variance, or NaN when there are no source points.
        """

        relative, inside, active_dims = self._selected_geometry(data)
        selected = self._ordered_selected(relative, inside)
        if len(selected) == 0:
            return float("nan")
        if data.error_covariance is not None and np.any(data.error_covariance):
            _, variance = self._solve_with_observation_errors(
                relative[selected], data.error_covariance[np.ix_(selected, selected)]
            )
            return variance
        if len(selected) == 1:
            # A single observation has weight one, but its uncertainty still depends on the distance from it
            _, variance = self._solve_geometry(relative[selected], active_dims)
            return variance
        _, variance = self._solve_geometry(relative[selected], active_dims)
        return variance
