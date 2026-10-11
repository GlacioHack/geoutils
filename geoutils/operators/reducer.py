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

"""Reducers classes to combine (e.g. average) several nearby input values into one reduced value."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

from geoutils._typing import NDArrayBool, NDArrayNum
from geoutils.operators.base import LinearCoefficients, LocalData
from geoutils.operators.neighbours import GridNeighbours, PointNeighbours, _grid_radius_scipy
from geoutils.operators.nodata import NodataHandling
from geoutils.operators.weighting import _local_error_data

if TYPE_CHECKING:
    from geoutils.operators.overlap import GridIntersection
    from geoutils.uncertainty.error_structure import ErrorStructure

# Regular reducers (raster windows or covered cells)
# Mean, Sum, Minimum, Maximum, Range, Count, Quantile, Median, Mode,
# RootMeanSquare, StandardDeviation, AverageDistance, AveragePairwiseDistance

# String names for reproject() or reduce_points() (other reducers supplied as objects)
RegularReductionMethod = Literal["average", "sum", "min", "max", "rms", "mode", "med", "q1", "q3"]

# Irregular reducers (point neighborhoods, same calculations through LocalData)
# Mean, Sum, Minimum, Maximum, Range, Count, Quantile, Median, Mode,
# RootMeanSquare, StandardDeviation, AverageDistance, AveragePairwiseDistance
# Custom Reducer objects supported for both; distances/weights supplied by each neighborhood

# String names for point gridding (other reducers supplied as objects)
IrregularReductionMethod = Literal[
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

try:
    from numba import jit as _jit
except ImportError:

    def _jit(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a no-op decorator when Numba is not installed."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


####################
# 1/ REDUCER CLASSES
####################

# 1.1/ Reducer base class
##########################


def _combined_weights(data: LocalData) -> NDArrayNum:
    """Multiply sample weights by area/length weights when supplied, or use equal weights otherwise."""

    weights = np.ones(len(data.values), dtype=np.float64)
    if data.sample_weights is not None:
        weights *= data.sample_weights
    if data.support_weights is not None:
        weights *= data.support_weights
    return weights


class Reducer:
    """
    Reducer of nearby input values into one result.

    A reducer uses all values selected in a neighborhood (rasters or points within a radius or window), potentially
    accounting for fractional overlap area on rasters, then applies a reducing function. Typical reducing functions
    are mean, median, etc.

    To define your own reducer, implement reduce(). It receives a LocalData object with the selected values and any
    available weights. Implement coefficients() instead when the result is a linear combination that can also be used
    for weighting or uncertainty propagation.

    Sample weights are used to describe the weight of each observation, while support weights are used to describe
    the fraction of a source raster cell covered by an output area (source points do not have fractional areas).

    The optional default_neighborhood selects raster cells for resample_at_points() or source points for grid().
    Without one, resample_at_points() uses a 3 x 3 window and grid() uses its dist_nodata_pixel radius.
    reproject() selects cells from each output pixel's footprint by default. With GridNeighbours, it selects source
    cells around the cell containing each transformed output pixel center, without fractional area weights.

    For example, this reducer averages the parts of source cells covered by each output cell:

    .. code-block:: python

        import numpy as np
        import geoutils as gu

        class CoveredMean(gu.operators.Reducer):
            accepts_support_weights = True

            def reduce(self, data):
                return float(np.average(data.values, weights=data.support_weights))

        reduced = raster.reproject(reference_raster, resampling=CoveredMean())

    A linear reducer should implement coefficients() instead to support weights or uncertainty propagation. The
    reducer example below sums all cells in the window, so its linear formulation is a weight of one to each,
    which can then be used with an error structure:

    .. code-block:: python

        class WindowSum(gu.operators.Reducer):
            '''Sum the raster cells in a window around each point.'''

            def coefficients(self, data):
                '''Give every selected cell a weight of one.'''

                return gu.operators.LinearCoefficients(np.ones(len(data.values)))

        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 2.0)])
        summary = gu.uncertainty.propagate(
            raster.resample_at_points, error_structure=errors,
            operation_kwargs=dict(points=(x, y), method=WindowSum(), as_array=True),
        )
        total = summary.estimate

    A Reducer receives a 3 x 3 raster window by default. Here GridNeighbours selects a 5 x 5 window instead:

    .. code-block:: python

        wider_window = gu.operators.GridNeighbours(size=5)
        total = raster.resample_at_points(
            (x, y), WindowSum(neighborhood=wider_window), as_array=True
        )

    PointNeighbours can instead select source observations when the Reducer grids a point cloud:

    .. code-block:: python

        from geoutils.operators import PointNeighbours
        from geoutils.operators.reducer import Mean

        nearby = PointNeighbours(k=8, radius=50)
        gridded = point_cloud.grid(ref=reference_raster, resampling=Mean(neighborhood=nearby))
    """

    minimum_inputs: ClassVar[int] = 1
    default_nodata_propagation: ClassVar[NodataHandling] = "ignore"
    accepts_sample_weights: bool = False
    accepts_support_weights: bool = False
    default_neighborhood: GridNeighbours | PointNeighbours | None = None
    error_structure: ErrorStructure | None = None
    _error_predictors: Any = None
    _uses_error_covariance: bool = True
    _grid_method: ClassVar[str | None] = None

    def __init__(self, neighborhood: GridNeighbours | PointNeighbours | None = None) -> None:
        """Choose the nearby raster cells or point observations to reduce.

        :param neighborhood: GridNeighbours for raster resample_at_points() or reproject(), or PointNeighbours
            for grid().
            Without one, resample_at_points() and grid() use their default windows or radii, and reproject() uses
            the output pixel's footprint.
        """

        if neighborhood is not None and not isinstance(neighborhood, (GridNeighbours, PointNeighbours)):
            raise TypeError("A Reducer neighborhood must be a GridNeighbours or PointNeighbours object.")
        self.default_neighborhood = neighborhood

    def coefficients(self, data: LocalData) -> LinearCoefficients | None:
        """Return weights and an optional constant offset for a weighted sum of values.

        Implement this method if your reducer can be written as a weighted sum.

        :param data: Nearby values, their optional weights and coordinates, grouped in a LocalData object.
        :returns: One weight per input value and an optional offset, or None if the method does not provide weights.
        """

        return None

    def reduce(self, data: LocalData) -> float:
        """Combine a group of values into one result.

        Implement this method for your own calculation, or define coefficients() to use a weighted sum. Call
        evaluate() to check nodata and weights before the calculation.

        :param data: Nearby values and their optional weights/coordinates.
        :returns: Reduced value for this group.
        """

        affine = self.coefficients(data)
        if affine is None:
            raise NotImplementedError("A Reducer subclass must implement reduce() or coefficients().")
        if len(affine.weights) != len(data.values):
            raise ValueError("Linear coefficients must contain one weight per LocalData value.")
        return float(np.dot(affine.weights, data.values) + affine.offset)

    def evaluate(self, data: LocalData, *, nodata_propagation: NodataHandling | None = None) -> float:
        """Check nodata and weights before calculating a result.

        :param data: Nearby values and their optional weights/coordinates.
        :param nodata_propagation: How to handle nodata: ``"ignore"`` excludes nodata values and ``"propagate"``
            returns NaN if any nodata value affects the result. Defaults to default_nodata_propagation.
        :returns: Reduced value, or NaN if too few valid values remain or the ``"propagate"`` rule rejects the group.
        """

        handling = self.default_nodata_propagation if nodata_propagation is None else nodata_propagation
        if self._uses_error_covariance:
            data = _local_error_data(data, self.error_structure, self._error_predictors)
        return data._evaluate(
            calculation=self.reduce,
            coefficients=self.coefficients,
            accepts_sample_weights=self.accepts_sample_weights,
            accepts_support_weights=self.accepts_support_weights,
            minimum_inputs=self.minimum_inputs,
            nodata_handling=handling,
        )

    def reduce_batch(
        self,
        data: Sequence[LocalData],
        *,
        nodata_propagation: NodataHandling | None = None,
    ) -> NDArrayNum:
        """Calculate a result for each group of values by calling evaluate().

        :param data: One LocalData object per group, in the desired output order.
        :param nodata_propagation: Optional nodata rule, as described in evaluate().
        :returns: One-dimensional array of reduced values in the same order as data.
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
        """Dispatch radius-based gridding to the selected SciPy or Numba engine."""

        method = self._grid_method
        if method is None:
            raise NotImplementedError(f"{type(self).__name__} has no built-in grid calculation.")
        if not np.isfinite(radius):
            raise ValueError("Circular gridding methods require a finite dist_nodata_pixel support radius.")
        if engine == "numba":
            # Average point spacing needs complete neighborhood membership retained by the SciPy engine
            if method == "average_distance_pts":
                raise ValueError("The Numba gridding engine does not support resampling='average_distance_pts'.")

            x_coords, y_coords = grid_coords
            return _grid_radius_statistic_numba(
                points,
                values,
                float(np.min(x_coords)),
                float(np.min(y_coords)),
                res_x,
                res_y,
                len(x_coords),
                len(y_coords),
                radius,
                _NUMBA_STATISTIC_CODES[method],
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
            """Apply this reducer to one group of grid cells."""

            _grid_sparse_reducer(block, pairs, queries, point_tree, points, values, counts, valid, method, res_x, res_y)

        return _grid_radius_scipy(
            points,
            values,
            grid_coords,
            res_x=res_x,
            res_y=res_y,
            radius=radius,
            min_points=min_points,
            minimum_inputs=max(1, self.minimum_inputs),
            calculate=calculate,
        )


class _CallableReducer(Reducer):
    """Apply a window callable with NaNs or a mask, preserving its handling of missing cells."""

    def __init__(self, function: Callable[[NDArrayNum], float], *, masked: bool) -> None:
        """Store the callable and its expected representation of missing cells."""

        super().__init__()
        self.function = function
        self.masked = masked

    def reduce(self, data: LocalData) -> float:
        """Pass all selected cells to the callable, including missing values."""

        if not len(data.values):
            return np.nan
        values = (
            np.ma.array(data.values, mask=~data.valid) if self.masked else np.where(data.valid, data.values, np.nan)
        )
        result = self.function(values)
        return np.nan if np.ma.is_masked(result) else float(result)

    def evaluate(self, data: LocalData, *, nodata_propagation: NodataHandling | None = None) -> float:
        """Let the callable handle missing values without filtering them beforehand."""

        return self.reduce(data)


# 1.2/ Built-in reduction methods
################################


class Mean(Reducer):
    """The Mean reducer averages values, optionally weighted by sample importance or covered area/length.

    Use it in a reduction such as PointCloud.grid(resampling=Mean()). With no supplied weights or observation
    covariance, each value contributes equally. Observation covariance adjusts the weights through
    LocalData.precision_weights().
    """

    accepts_sample_weights = True
    accepts_support_weights = True
    _grid_method = "mean"

    def coefficients(self, data: LocalData) -> LinearCoefficients | None:
        """Normalize the supplied weights, accounting for observation covariance when available."""

        weights = _combined_weights(data)
        total = float(np.sum(weights))
        if total == 0:
            return None
        return LinearCoefficients(weights=data.precision_weights(weights))

    def reduce(self, data: LocalData) -> float:
        """Calculate the weighted average, or NaN if the weights sum to zero."""

        affine = self.coefficients(data)
        if affine is None:
            return float("nan")
        return float(np.dot(affine.weights, data.values))


class Sum(Reducer):
    """The Sum reducer adds values, with optional sample weights and area/length weights."""

    accepts_sample_weights = True
    accepts_support_weights = True

    def coefficients(self, data: LocalData) -> LinearCoefficients:
        """Multiply the supplied weights, or use one for every value if no weights were supplied."""

        weights = _combined_weights(data)
        return LinearCoefficients(weights=weights)


class Quantile(Reducer):
    """The Quantile reducer selects a value at a chosen fraction of the sorted observations.

    For example, q=0.5 gives the median and q=0.9 the 90th percentile. By default, all observations count equally;
    weighted=True uses sample weights and covered area/length to determine their contributions.
    """

    def __init__(
        self,
        q: float,
        *,
        weighted: bool = False,
        method: str = "linear",
        neighborhood: GridNeighbours | PointNeighbours | None = None,
    ) -> None:
        """Choose the quantile and whether to use supplied weights.

        :param q: Fraction from zero to one (0.5 for the median).
        :param weighted: Whether to use supplied sample and area/length weights.
        :param method: NumPy quantile method for unweighted values. Weighted quantiles select the first sorted value
            whose cumulative weight reaches q times the total weight.
        :param neighborhood: Raster cells or point observations to select for spatial reduction.
        """

        super().__init__(neighborhood=neighborhood)
        if not np.isfinite(q) or q < 0 or q > 1:
            raise ValueError("Quantile q must be a finite number between zero and one.")
        self.q = float(q)
        self.weighted = bool(weighted)
        self.method = method
        self.accepts_sample_weights = self.weighted
        # Accept area/length weights with the values; weighted=True uses them in the quantile calculation
        self.accepts_support_weights = True

    def reduce(self, data: LocalData) -> float:
        """Find the requested quantile, optionally counting observations by their weights."""

        if not self.weighted:
            return float(np.quantile(data.values, self.q, method=cast(Any, self.method)))

        # Only positive weights can contribute to the quantile
        weights = _combined_weights(data)
        positive = np.isfinite(weights) & (weights > 0)
        if not np.any(positive):
            return float("nan")
        values = np.asarray(data.values[positive])
        weights = np.asarray(weights[positive], dtype=np.float64)

        # Sort the values, then find where their added weights reach the requested fraction of the total
        order = np.argsort(values, kind="stable")
        values = values[order]
        cumulative = np.cumsum(weights[order])
        threshold = self.q * cumulative[-1]
        index = int(np.searchsorted(cumulative, threshold, side="left"))
        return float(values[min(index, len(values) - 1)])


class Median(Quantile):
    """The Median reducer finds the middle value, optionally using sample and area/length weights."""

    def __init__(
        self,
        *,
        weighted: bool = False,
        method: str = "linear",
        neighborhood: GridNeighbours | PointNeighbours | None = None,
    ) -> None:
        """Use the 0.5 quantile, averaging the two middle values by default when their number is even.

        :param weighted: Whether to use sample and area/length weights, as described in Quantile.
        :param method: NumPy quantile method for unweighted values.
        :param neighborhood: Raster cells or point observations to select for spatial reduction.
        """

        super().__init__(0.5, weighted=weighted, method=method, neighborhood=neighborhood)


class Mode(Reducer):
    """The Mode reducer selects the most frequent value, optionally counting by sample/area weights.

    weighted controls whether supplied weights contribute to the count, and tie_break selects between values
    with the same count (the largest value by default).
    """

    def __init__(
        self,
        *,
        weighted: bool = True,
        tie_break: Literal["first", "first_to_mode", "smallest", "largest"] = "largest",
        neighborhood: GridNeighbours | PointNeighbours | None = None,
    ) -> None:
        """Choose whether to count with weights and how to resolve ties.

        :param weighted: Whether to add the supplied sample and area/length weights rather than count observations.
        :param tie_break: Which value to return if several have the same count: the first in the input order,
            the first to reach the winning count (GDAL's rule), the smallest, or the largest.
        :param neighborhood: Raster cells or point observations to select for spatial reduction.
        """

        super().__init__(neighborhood=neighborhood)
        if tie_break not in {"first", "first_to_mode", "smallest", "largest"}:
            raise ValueError("Mode tie_break must be 'first', 'first_to_mode', 'smallest' or 'largest'.")
        if weighted and tie_break == "first_to_mode":
            raise ValueError("Mode tie_break='first_to_mode' requires weighted=False.")
        self.weighted = bool(weighted)
        self.tie_break = tie_break
        self.accepts_sample_weights = self.weighted
        # Accept area/length weights with the values; weighted=True uses them when counting occurrences
        self.accepts_support_weights = True

    def reduce(self, data: LocalData) -> float:
        """Return the most frequent value, using the chosen weights and tie rule."""

        values, inverse = np.unique(data.values, return_inverse=True)
        weights = _combined_weights(data) if self.weighted else None
        frequencies = np.bincount(inverse, weights=weights, minlength=len(values))
        candidates = values[frequencies == np.max(frequencies)]
        if self.tie_break == "smallest":
            return float(candidates[0])
        if self.tie_break == "largest":
            return float(candidates[-1])

        if self.tie_break == "first_to_mode":
            # The earliest last occurrence reaches the winning count first
            last_indices = [np.flatnonzero(data.values == candidate)[-1] for candidate in candidates]
            return float(candidates[int(np.argmin(last_indices))])

        # Find the first source occurrence among tied values, matching GDAL's documented mode tie behavior
        candidate_indices = np.flatnonzero(np.isin(data.values, candidates))
        return float(data.values[candidate_indices[0]])


class Minimum(Reducer):
    """The Minimum reducer returns the smallest value in each group."""

    accepts_support_weights = True
    _grid_method = "minimum"

    def reduce(self, data: LocalData) -> float:
        """Return the smallest value."""

        return float(np.min(data.values))


class Maximum(Reducer):
    """The Maximum reducer returns the largest value in each group."""

    accepts_support_weights = True
    _grid_method = "maximum"

    def reduce(self, data: LocalData) -> float:
        """Return the largest value."""

        return float(np.max(data.values))


class Range(Reducer):
    """The Range reducer subtracts the smallest value from the largest in each group."""

    accepts_support_weights = True
    _grid_method = "range"

    def reduce(self, data: LocalData) -> float:
        """Return the difference between the largest and smallest values."""

        return float(np.max(data.values) - np.min(data.values))


class Count(Reducer):
    """The Count reducer counts observations, or adds their sample and area/length weights when supplied."""

    minimum_inputs = 0
    accepts_sample_weights = True
    accepts_support_weights = True
    _grid_method = "count"

    def reduce(self, data: LocalData) -> float:
        """Add the supplied weights, or return the number of values if no weights were supplied."""

        return float(np.sum(_combined_weights(data)))


class RootMeanSquare(Reducer):
    """The RootMeanSquare reducer averages squared values, then takes the square root.

    Supplied sample and area/length weights are used in the average.
    """

    accepts_sample_weights = True
    accepts_support_weights = True

    def reduce(self, data: LocalData) -> float:
        """Return the square root of the weighted mean squared value."""

        weights = _combined_weights(data)
        total = float(np.sum(weights))
        if total == 0:
            return float("nan")
        return float(np.sqrt(np.dot(weights, np.square(data.values)) / total))


class StandardDeviation(Reducer):
    """The StandardDeviation reducer measures the spread of values around their mean.

    It calculates the population standard deviation, with optional sample and area/length weights.
    """

    accepts_sample_weights = True
    accepts_support_weights = True
    _grid_method = "stdev"

    def reduce(self, data: LocalData) -> float:
        """Calculate the weighted population standard deviation."""

        weights = _combined_weights(data)
        total = float(np.sum(weights))
        if total == 0:
            return float("nan")

        # Average the squared differences from the mean, then take the square root to recover the original units
        mean = float(np.dot(weights, data.values) / total)
        return float(np.sqrt(np.dot(weights, np.square(data.values - mean)) / total))


class AverageDistance(Reducer):
    """The AverageDistance reducer averages distances from nearby observations to the output coordinate."""

    _grid_method = "average_distance"

    def reduce(self, data: LocalData) -> float:
        """Average the distances to the output coordinate, in the same units as the supplied distances."""

        if data.distances is None:
            raise ValueError("AverageDistance requires source-to-target distances.")
        return float(np.mean(data.distances))


class AveragePairwiseDistance(Reducer):
    """The AveragePairwiseDistance reducer averages distances between all pairs of nearby observations."""

    minimum_inputs = 2
    _grid_method = "average_distance_pts"

    def reduce(self, data: LocalData) -> float:
        """Average the distance between each pair of observations, in source coordinate units."""

        if data.coordinates is None:
            raise ValueError("AveragePairwiseDistance requires source coordinates.")
        return float(np.mean(pdist(data.coordinates)))


_IRREGULAR_REDUCER_TYPES: dict[IrregularReductionMethod, type[Reducer]] = {
    "mean": Mean,
    "average": Mean,
    "minimum": Minimum,
    "min": Minimum,
    "maximum": Maximum,
    "max": Maximum,
    "range": Range,
    "count": Count,
    "stdev": StandardDeviation,
    "average_distance": AverageDistance,
    "average_distance_pts": AveragePairwiseDistance,
}


################################
# 2/ REGULAR RASTER REDUCTION
################################


_ARRAY_REDUCER_TYPES = (
    Mean,
    Median,
    Minimum,
    Maximum,
    Range,
    Count,
    Sum,
    StandardDeviation,
    RootMeanSquare,
    AverageDistance,
)


def _can_reduce_arrays(reducer: Reducer) -> bool:
    """Check whether the reducer options match the shared array calculations."""

    if type(reducer) not in _ARRAY_REDUCER_TYPES or reducer.error_structure is not None:
        return False
    if getattr(reducer, "_requires_local_evaluation", False):
        return False
    if any(name in vars(reducer) for name in ("reduce", "coefficients", "evaluate", "reduce_batch")):
        return False
    if reducer.accepts_sample_weights != type(reducer).accepts_sample_weights:
        return False
    if reducer.accepts_support_weights != (type(reducer) is Median or type(reducer).accepts_support_weights):
        return False
    if type(reducer) is Median and (reducer.weighted or reducer.method != "linear" or reducer.q != 0.5):
        return False
    return type(reducer.minimum_inputs) is int and reducer.minimum_inputs >= type(reducer).minimum_inputs


def _reduce_overlap_batch(
    array: NDArrayNum,
    overlap: GridIntersection,
    operator: Reducer,
    *,
    nodata_propagation: NodataHandling | None,
) -> NDArrayNum:
    """
    Apply reduction to the cells overlapping each destination cell.

    The intersections group source cells by destination cell, so a source value can appear in more than one group.
    Return one value per destination cell, with NaN for groups that fail the minimum input count or nodata rules.
    """

    # Gather intersecting values and exclude both nonfinite values and explicitly masked cells
    source = np.asanyarray(np.ma.getdata(array))
    values = np.asarray(source[overlap.rows, overlap.columns], dtype=np.float64)
    valid = np.isfinite(values)
    if np.ma.isMaskedArray(array):
        valid &= ~np.ma.getmaskarray(array)[overlap.rows, overlap.columns]

    # Expand the group offsets into destination IDs so bincount() can reduce all destination cells together
    target_ids = np.repeat(np.arange(overlap.geometry_count, dtype=np.int64), np.diff(overlap.offsets))
    return _reduce_grouped_values(
        operator,
        np.where(valid, values, np.nan),
        target_ids,
        overlap.geometry_count,
        operator.default_nodata_propagation if nodata_propagation is None else nodata_propagation,
        weights=overlap.fractions,
    )


################################
# 3/ POINT CLOUD GRID REDUCTION
################################


_NUMBA_STATISTIC_CODES = {
    "mean": 0,
    "minimum": 1,
    "maximum": 2,
    "range": 3,
    "count": 4,
    "stdev": 5,
    "average_distance": 6,
}


# 3.1/ Selected point neighborhoods (Numba)
##########################################


@_jit(nopython=True, cache=True)
def _reduce_point_groups_numba(
    values: NDArrayNum,
    valid: NDArrayBool,
    distances: NDArrayNum,
    offsets: NDArrayNum,
    method: str,
    minimum_inputs: int,
    propagate_nodata: bool,
) -> NDArrayNum:
    """Reduce flattened point neighborhoods with the built-in statistics supported by Numba.

    Offsets delimit the observations for each target. Return one value per target, with NaN for groups rejected
    by the minimum input count or nodata rule.
    """

    output = np.full(len(offsets) - 1, np.nan)
    for target in range(len(output)):
        start, stop = offsets[target], offsets[target + 1]
        count = 0
        missing = False

        # Check finite count and nodata rule before reduction
        for index in range(start, stop):
            if valid[index]:
                count += 1
            else:
                missing = True
        if count < minimum_inputs or (propagate_nodata and missing):
            continue

        # Accumulate statistics from valid observations
        total, distance_sum = 0.0, 0.0
        smallest, largest = np.inf, -np.inf
        for index in range(start, stop):
            if not valid[index]:
                continue
            total += values[index]
            distance_sum += distances[index]
            smallest = min(smallest, values[index])
            largest = max(largest, values[index])

        if method == "count":
            output[target] = count
        elif method == "mean" and count > 0:
            output[target] = total / count
        elif method == "minimum":
            output[target] = smallest
        elif method == "maximum":
            output[target] = largest
        elif method == "range":
            output[target] = largest - smallest
        elif method == "average_distance" and count > 0:
            output[target] = distance_sum / count
        elif method == "stdev" and count > 0:
            # Center the values before squaring to avoid cancellation for large values with little variation
            mean = total / count
            squared_differences = 0.0
            for index in range(start, stop):
                if valid[index]:
                    squared_differences += (values[index] - mean) ** 2
            output[target] = np.sqrt(squared_differences / count)
    return output


# 3.2/ Numba calculation, adding each point to nearby grid cells
#############################################################


@_jit(nopython=True, cache=True)
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
    """Compute one circular statistic in nearby cells from every source point."""

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

    # Visit only the grid cells that can fall inside each point's support radius
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
                        # Mean: accumulate values, then divide by the point count further below
                        output[row, col] += value
                    elif statistic_code == 1:
                        # Minimum: retain the smallest value found for this cell
                        output[row, col] = min(output[row, col], value)
                    elif statistic_code == 2:
                        # Maximum: retain the largest value found for this cell
                        output[row, col] = max(output[row, col], value)
                    elif statistic_code == 3:
                        # Range: retain the minimum and maximum in separate arrays
                        output[row, col] = min(output[row, col], value)
                        secondary[row, col] = max(secondary[row, col], value)
                    elif statistic_code == 5:
                        # Standard deviation: accumulate values and squared values
                        output[row, col] += value
                        secondary[row, col] += value * value
                    elif statistic_code == 6:
                        # Average distance: accumulate point-to-cell distances
                        output[row, col] += np.sqrt(((col - point_x) * res_x) ** 2 + ((row - point_y) * res_y) ** 2)
                    counts[row, col] += 1

    # Reject cells with too few points, then convert accumulated values into final results
    required_points = max(1, min_points)
    for row in range(height):
        for col in range(width):
            count = counts[row, col]
            if count < required_points:
                output[row, col] = np.nan
            elif statistic_code == 0 or statistic_code == 6:
                # Mean and average distance are their accumulated sums divided by count
                output[row, col] /= count
            elif statistic_code == 3:
                # Range is the accumulated maximum minus minimum
                output[row, col] = secondary[row, col] - output[row, col]
            elif statistic_code == 4:
                # Count can be directly reuse
                output[row, col] = count
            elif statistic_code == 5:
                # Derive the population standard deviation from the two accumulated sums
                mean = output[row, col] / count
                output[row, col] = np.sqrt(max(0.0, secondary[row, col] / count - mean * mean))
    return output


# 3.3/ SciPy calculation, grouping nearby points by grid cell
##########################################################


def _grid_sparse_reducer(
    block: NDArrayNum,
    pairs: Any,
    queries: NDArrayNum,
    point_tree: Any,
    points: NDArrayNum,
    values: NDArrayNum,
    counts: NDArrayNum,
    valid: NDArrayBool,
    method: str,
    res_x: float,
    res_y: float,
) -> None:
    """Calculate each grid cell's statistic from points inside its search radius."""

    if method == "average_distance_pts":
        # Compute all point-to-point distances in each complete neighborhood, then average them
        for query_index in np.flatnonzero(valid):
            point_indexes = pairs.col[pairs.row == query_index]
            block[query_index] = float(np.mean(pdist(points[point_indexes])))
        return

    distances = None
    if method == "average_distance":
        # Convert tree distances back to coordinate units, then average them per cell
        dx = queries[pairs.row, 0] - point_tree.data[pairs.col, 0]
        dy = queries[pairs.row, 1] - point_tree.data[pairs.col, 1]
        distances = np.sqrt((dx * res_x) ** 2 + (dy * res_y) ** 2)
    reduced = _reduce_grouped_values(
        _IRREGULAR_REDUCER_TYPES[cast(IrregularReductionMethod, method)](),
        values[pairs.col],
        pairs.row,
        len(queries),
        "ignore",
        distances=distances,
    )
    block[valid] = reduced[valid]


# 3.4/ Shared grouped calculations
##################################


def _reduce_grouped_values(
    reducer: Reducer,
    values: NDArrayNum,
    target_indexes: NDArrayNum,
    target_count: int,
    nodata_handling: NodataHandling,
    min_points: int = 0,
    *,
    weights: NDArrayNum | None = None,
    distances: NDArrayNum | None = None,
) -> NDArrayNum:
    """Calculate a built-in statistic for groups of values, with optional covered-area weights."""

    # Count valid neighbours for each target, then apply the minimum count and missing value rule
    reducer_type = type(reducer)
    finite = np.isfinite(values)
    finite_counts = np.bincount(target_indexes[finite], minlength=target_count)
    required = max(min_points, reducer.minimum_inputs)
    eligible = finite_counts >= required
    if nodata_handling == "propagate":
        invalid_counts = np.bincount(target_indexes[~finite], minlength=target_count)
        eligible &= invalid_counts == 0

    # An empty neighborhood has a count of zero, while the other statistics have no result
    output = np.zeros(target_count, dtype=np.float64) if reducer_type is Count else np.full(target_count, np.nan)
    finite_target_indexes = target_indexes[finite]
    finite_values = values[finite]
    if len(finite_values) == 0:
        output[~eligible] = np.nan
        return output
    selected_weights = None if weights is None else weights[finite]
    weight_sums = (
        finite_counts
        if selected_weights is None
        else np.bincount(finite_target_indexes, weights=selected_weights, minlength=target_count)
    )

    # Unweighted groups reuse finite counts without allocating one weight per observation
    if reducer_type is Count:
        output[:] = weight_sums
    elif reducer_type in (Mean, Sum):
        weighted_values = finite_values if selected_weights is None else finite_values * selected_weights
        sums = np.bincount(finite_target_indexes, weights=weighted_values, minlength=target_count)
        output[eligible] = sums[eligible]
        if reducer_type is Mean:
            output[eligible] /= weight_sums[eligible]
    elif reducer_type in (Minimum, Maximum, Range):
        minima = np.full(target_count, np.inf)
        maxima = np.full(target_count, -np.inf)
        np.minimum.at(minima, finite_target_indexes, finite_values)
        np.maximum.at(maxima, finite_target_indexes, finite_values)
        if reducer_type is Minimum:
            output[eligible] = minima[eligible]
        elif reducer_type is Maximum:
            output[eligible] = maxima[eligible]
        else:
            output[eligible] = maxima[eligible] - minima[eligible]
    elif reducer_type in (StandardDeviation, RootMeanSquare):
        # We need the mean of squared values for both RMS and variance
        # Variance also subtracts the squared mean; rounding errors can make it slightly negative, so clip to zero
        squared_values = np.square(finite_values)
        if selected_weights is not None:
            squared_values *= selected_weights
        squared_sums = np.bincount(finite_target_indexes, weights=squared_values, minlength=target_count)
        if reducer_type is RootMeanSquare:
            output[eligible] = np.sqrt(squared_sums[eligible] / weight_sums[eligible])
        else:
            weighted_values = finite_values if selected_weights is None else finite_values * selected_weights
            sums = np.bincount(finite_target_indexes, weights=weighted_values, minlength=target_count)
            means = np.zeros(target_count, dtype=np.float64)
            means[eligible] = sums[eligible] / weight_sums[eligible]
            variance = squared_sums[eligible] / weight_sums[eligible] - np.square(means[eligible])
            output[eligible] = np.sqrt(np.maximum(variance, 0))
    elif reducer_type is AverageDistance:
        assert distances is not None
        distance_sums = np.bincount(
            finite_target_indexes,
            weights=distances[finite],
            minlength=target_count,
        )
        output[eligible] = distance_sums[eligible] / weight_sums[eligible]
    elif reducer_type in (Median, Quantile):
        # Sort each destination's valid values once, then select its requested order statistic
        order = np.lexsort((finite_values, finite_target_indexes))
        sorted_values = finite_values[order]
        counts = np.bincount(finite_target_indexes, minlength=target_count)
        starts = np.concatenate(([0], np.cumsum(counts[:-1])))
        selected = np.flatnonzero(eligible)
        if isinstance(reducer, Quantile) and reducer.method == "inverted_cdf":
            positions = np.maximum(np.ceil(reducer.q * counts[selected]).astype(int) - 1, 0)
            output[selected] = sorted_values[starts[selected] + positions]
        elif reducer_type is Median and reducer.method == "linear":
            # Even groups average the two central cells; odd groups select their common center
            lower = starts[selected] + (counts[selected] - 1) // 2
            upper = starts[selected] + counts[selected] // 2
            output[selected] = (sorted_values[lower] + sorted_values[upper]) / 2
        else:
            raise TypeError(f"Unsupported batch quantile method: {reducer.method}")
    elif reducer_type is Mode and not reducer.weighted:
        # Count runs of equal values within each destination, preserving the first source in ties
        order = np.lexsort((finite_values, finite_target_indexes))
        sorted_targets = finite_target_indexes[order]
        sorted_values = finite_values[order]
        run_start = np.r_[True, (sorted_targets[1:] != sorted_targets[:-1]) | (sorted_values[1:] != sorted_values[:-1])]
        run_first = np.flatnonzero(run_start)
        run_lengths = np.diff(np.r_[run_first, len(sorted_values)])
        run_targets = sorted_targets[run_first]
        largest_count = np.zeros(target_count, dtype=int)
        np.maximum.at(largest_count, run_targets, run_lengths)
        tied = run_lengths == largest_count[run_targets]
        if reducer.tie_break in ("first", "first_to_mode"):
            # Source positions identify the first value or the first value to reach the winning count
            source_order = np.full(len(run_first), -1, dtype=int)
            if reducer.tie_break == "first":
                source_order[:] = len(finite_values)
                np.minimum.at(source_order, np.cumsum(run_start) - 1, order)
            else:
                np.maximum.at(source_order, np.cumsum(run_start) - 1, order)
            winner_order = np.lexsort((source_order[tied], run_targets[tied]))
        elif reducer.tie_break == "smallest":
            winner_order = np.lexsort((sorted_values[run_first[tied]], run_targets[tied]))
        else:
            winner_order = np.lexsort((-sorted_values[run_first[tied]], run_targets[tied]))
        winners = np.flatnonzero(tied)[winner_order]
        first_per_target = np.r_[True, run_targets[winners][1:] != run_targets[winners][:-1]]
        selected_winners = winners[first_per_target]
        output[run_targets[selected_winners]] = sorted_values[run_first[selected_winners]]
    else:
        raise TypeError(f"Reducer {reducer_type.__name__} does not have a vectorized point-filter implementation.")

    # Set NaN where the missing value rule or minimum count prevents a result (this also applies to Count)
    output[~eligible] = np.nan
    return output
