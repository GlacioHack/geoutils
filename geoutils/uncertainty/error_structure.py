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

"""Module with the lightweight error structure classes to describe error magnitude and correlation."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.spatial import QhullError
from scipy.spatial.distance import cdist
from scipy.special import ndtri

from geoutils._misc import import_optional
from geoutils.stats.variography import Variogram, VariogramModel

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig

__all__ = ["ErrorComponent", "ErrorMagnitude", "ErrorStructure"]


############################################
# 1/ INTERPOLATION AND CORRELATION HELPERS
############################################


def _grouped_interpolator(
    table: pd.DataFrame,
    *,
    value_name: str,
    statistic: str,
    min_count: int,
) -> Callable[[Mapping[str, Any]], NDArray[np.float64]]:
    """
    Interpolate grouped error statistics at new predictor values (e.g. slope/elevation).

    This function is used to build an empirical model of heteroscedasticity from binned spread statistics,
    and accepts any number of predictors (so is equivalent to a N-dimension interpolator)

    Points inside the edge bins are interpolated linearly.
    Points outside the edge bins are interpolated by nearest neighbours (to avoid unrealistic linear extrapolation
    where few samples exist anyway).

    This function was simplified from an old ``interpolate_nd_binning`` in xDEM.
    """

    # Input checks
    predictor_names = tuple(table.index.names)
    if not predictor_names or any(not isinstance(name, str) or not name for name in predictor_names):
        raise ValueError("Grouped statistics must have named predictor index levels.")
    if table.empty or table.index.has_duplicates:
        raise ValueError("Grouped statistics must be non-empty and have unique group coordinates.")
    levels = [table.index] if table.index.nlevels == 1 else list(table.index.levels)
    coordinates: list[NDArray[np.float64]] = []
    for name, level in zip(predictor_names, levels):
        if isinstance(level, pd.IntervalIndex):
            coordinate = level.mid.to_numpy(dtype=float)
        elif np.issubdtype(level.dtype, np.number):
            coordinate = level.to_numpy(dtype=float)
        else:
            raise TypeError(f"Predictor {name!r} must use continuous numeric groups.")
        if coordinate.size == 0 or np.any(~np.isfinite(coordinate)) or np.any(np.diff(coordinate) <= 0):
            raise ValueError(f"Predictor {name!r} must have finite, increasing group coordinates.")
        coordinates.append(cast(NDArray[np.float64], coordinate))

    # Some combinations of groups may be missing from the table: we add them as NaN in the full grid
    # We also exclude estimates based on fewer than min_count observations
    statistic_column = (value_name, statistic)
    count_column = (value_name, "count")
    if statistic_column not in table.columns or count_column not in table.columns:
        raise ValueError(f"Grouped statistics must contain {statistic_column!r} and {count_column!r}.")
    full_index: pd.Index = levels[0] if len(levels) == 1 else pd.MultiIndex.from_product(levels, names=predictor_names)
    selected = table[statistic_column].where(table[count_column] >= min_count)
    shape = tuple(len(coordinate) for coordinate in coordinates)
    values = selected.reindex(full_index).to_numpy(dtype=float).reshape(shape)

    # Fill gaps between valid groups linearly, then use the nearest group for any remaining gaps
    coordinate_grid = np.meshgrid(*coordinates, indexing="ij")
    points = np.column_stack([coordinate.ravel() for coordinate in coordinate_grid])
    flat_values = values.ravel()
    valid = np.isfinite(flat_values)
    if not np.any(valid):
        raise ValueError(f"No finite {statistic!r} remains after applying min_count={min_count}.")
    if len(coordinates) == 1:
        filled = np.interp(coordinates[0], points[valid, 0], flat_values[valid])
    else:
        filled = np.full(len(points), np.nan, dtype=float)
        if np.count_nonzero(valid) >= len(coordinates) + 1:
            try:
                filled = np.asarray(griddata(points[valid], flat_values[valid], points, method="linear"), dtype=float)
            except QhullError:
                pass
        missing = ~np.isfinite(filled)
        if np.any(missing):
            filled[missing] = griddata(points[valid], flat_values[valid], points[missing], method="nearest")

    # Now that the group grid is complete, build an interpolator for new predictor values
    interpolator = RegularGridInterpolator(
        tuple(coordinates),
        np.asarray(filled).reshape(shape),
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    def evaluate(predictors: Mapping[str, Any]) -> NDArray[np.float64]:
        """Calculate error magnitudes at the supplied predictors, in their shared array shape."""

        # We use predictor names to avoid mixing up the order of dimensions
        missing_names = set(predictor_names).difference(predictors)
        if missing_names:
            raise ValueError(f"Missing predictors: {sorted(missing_names)!r}.")
        arrays = [np.asarray(predictors[name], dtype=float) for name in predictor_names]
        broadcast = np.broadcast_arrays(*arrays)
        prediction_points = np.column_stack([array.ravel() for array in broadcast])
        finite = np.all(np.isfinite(prediction_points), axis=1)
        result = np.full(len(prediction_points), np.nan, dtype=float)

        # Beyond the sampled range, we use the nearest outer group center (to not extrapolate the error magnitude)
        if np.any(finite):
            bounded = prediction_points[finite].copy()
            for index, coordinate in enumerate(coordinates):
                bounded[:, index] = np.clip(bounded[:, index], coordinate[0], coordinate[-1])
            result[finite] = interpolator(bounded)
        return result.reshape(broadcast[0].shape)

    return evaluate


def _normalize_correlation_model(model: VariogramModel) -> VariogramModel:
    """Scale a variogram to unit variance without changing its correlation shape."""

    if model.nugget != 0:
        raise ValueError("Component correlations cannot contain a nugget; use an independent component.")
    if model.sill <= 0:
        raise ValueError("Component correlations require a strictly positive sill.")
    if model.components:
        # For sums, divide each variance by the total; for products, distribute that division among the factors
        divisor = model.sill if model.model_name == "sum" else model.sill ** (1 / len(model.components))
        components = tuple(
            replace(component, partial_sill=float(component.partial_sill or 0.0) / divisor)
            for component in model.components
        )
        return replace(model, partial_sill=1.0, components=components)
    return replace(model, partial_sill=1.0)


############################################
# 2/ CONSTANT OR VARIABLE ERROR MAGNITUDES
############################################


@dataclass(frozen=True)
class ErrorMagnitude:
    """
    Error magnitude, either constant or variable (i.e. heteroscedastic) with named predictors.

    An ErrorMagnitude describes the statistical spread for an ErrorComponent, in the source units. Statistical spread
    can be derived from any spread estimator (such as STD, NMAD, percentiles...).

    It should generally be estimated jointly with error correlation through higher-level function such as
    ``estimate_error_structure()`` from a raster or point object, or ``ErrorStructure.estimate()``.
    Otherwise, to create it manually, use one of two methods:

    - For a constant error, use constant(value) with a single value,
    - For a variable error, use variable_from_grouped_stats(statistics) with a pandas DataFrame of spread estimates
      and counts, indexed by named predictors; the output of stats(by=) on an error proxy.

    Grouped tables can use plain statistic columns such as "nmad" and "count". Two-level (variable, statistic)
    columns from stats() are also accepted; use ``value_name`` to select the variable name.

    With statistic=None, "std" or "nmad" is selected automatically. If both are present, "std" is used with a warning.
    Pass statistic explicitly to select another spread estimate or override this choice.

    For example, a constant standard deviation of 2 m for elevation measurements:

    .. code-block:: python

        import geoutils as gu

        measurement = gu.ErrorMagnitude.constant(2.0)
        measurement.predict()  # 2.0

    Another example can use variable error magnitudes increasing from 1 to 3 m between slopes of 0 and 30 degrees.
    Then ``predicts`` interpolates linearly between groups, estimating an error of 2 m for a slope of 15 degrees:

    .. code-block:: python

        import pandas as pd

        slope_statistics = pd.DataFrame(
            {"nmad": [1.0, 3.0], "count": [20, 30]},
            index=pd.Index([0.0, 30.0], name="slope"),
        )
        terrain = gu.ErrorMagnitude.variable_from_grouped_stats(slope_statistics)
        terrain.predict({"slope": [0.0, 15.0, 30.0]})  # array([1., 2., 3.])

    Multiple predictors can be used at once, for instance slope and elevation, with a different
    dispersion column "std":

    .. code-block:: python

        terrain_statistics = pd.DataFrame(
            {"std": [1.0, 2.0, 3.0, 4.0], "count": [20, 20, 20, 20]},
            index=pd.MultiIndex.from_product(
                [[0.0, 30.0], [100.0, 1000.0]], names=["slope", "elevation"]
            ),
        )
        terrain = gu.ErrorMagnitude.variable_from_grouped_stats(terrain_statistics, statistic="std")
        terrain.predict({"slope": [0.0, 30.0], "elevation": [100.0, 1000.0]})  # array([1., 4.])

    :param kind: Kind of error magnitude, "constant" or "variable"; set automatically by constant() and
        variable_from_grouped_stats().
    :param value: Standard deviation for "constant" error magnitude, must be None for "variable".
    :param grouped_statistics: Statistics table as described above, required for "variable"; None for "constant".
    :param predictor_names: Predictor names in index order, required for direct "variable" construction.
        variable_from_grouped_stats() infers them from the table; leave empty for "constant".
    :param value_name: Variable to select from two-level columns; unneeded for plain statistic columns.
    :param statistic: Spread statistic selected from grouped statistics.
    :param min_count: Minimum number of values required to use a group's statistic.
    :param scale: Factor applied to error magnitudes from grouped statistics.
    :param variance_offset: Variance assigned to other error components, subtracted from the squared error magnitude.
    :param floor: Minimum error magnitude after subtracting the variance offset.
    """

    kind: Literal["constant", "variable"]
    value: float | None = None
    grouped_statistics: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    predictor_names: tuple[str, ...] = ()
    value_name: str = "error"
    statistic: str | None = None
    min_count: int = 0
    scale: float = 1.0
    variance_offset: float = 0.0
    floor: float = 0.0
    _interpolator: Callable[[Mapping[str, Any]], NDArray[np.float64]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Check the model and prepare interpolation between grouped statistics."""

        # Convert shared numeric options to floats and check their ranges
        object.__setattr__(self, "predictor_names", tuple(self.predictor_names))
        for name in ("scale", "variance_offset", "floor"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Magnitude scale must be finite and strictly positive.")
        if not np.isfinite(self.variance_offset) or self.variance_offset < 0:
            raise ValueError("Magnitude variance_offset must be finite and non-negative.")
        if not np.isfinite(self.floor) or self.floor < 0:
            raise ValueError("Magnitude floor must be finite and non-negative.")

        # Constant models need just one non-negative error magnitude
        if self.kind == "constant":
            if self.value is None or not np.isfinite(self.value) or self.value < 0:
                raise ValueError("A constant magnitude requires a finite, non-negative value.")
            if self.predictor_names or self.grouped_statistics is not None:
                raise ValueError("A constant magnitude cannot contain grouped predictor metadata.")
            object.__setattr__(self, "value", float(self.value))
        elif self.kind == "variable":
            # Copy the statistics so later edits to the input table cannot change this model
            if self.value is not None or self.grouped_statistics is None:
                raise ValueError("A variable magnitude requires grouped_statistics and no constant value.")
            if not self.predictor_names or len(set(self.predictor_names)) != len(self.predictor_names):
                raise ValueError("A variable magnitude requires unique predictor names.")
            table = self.grouped_statistics.copy(deep=True)
            if tuple(table.index.names) != self.predictor_names:
                raise ValueError("Grouped statistic index names must match predictor_names in the same order.")

            # Give plain statistic columns the same layout as stats() output for interpolation and plotting
            if not isinstance(table.columns, pd.MultiIndex):
                table.columns = pd.MultiIndex.from_product(
                    [[self.value_name], table.columns], names=["value", "statistic"]
                )

            # Infer a standard spread statistic only among the selected variable's columns
            statistic = self.statistic
            if statistic is None:
                available = [name for name in ("std", "nmad") if (self.value_name, name) in table.columns]
                if not available:
                    raise ValueError(
                        f"Grouped statistics must contain 'std' or 'nmad' for {self.value_name!r}; "
                        "pass statistic explicitly to select another spread column."
                    )
                statistic = available[0]
                if len(available) > 1:
                    warnings.warn(
                        "Both 'std' and 'nmad' are defined for the selected variable; using 'std'. "
                        "Pass statistic explicitly to choose the spread estimate.",
                        UserWarning,
                        stacklevel=3,
                    )
                object.__setattr__(self, "statistic", statistic)

            # Exclude negative spread estimates before filling gaps between valid groups
            column = (self.value_name, statistic)
            if column in table:
                table.loc[table[column] < 0, column] = np.nan
            object.__setattr__(self, "grouped_statistics", table)
            object.__setattr__(
                self,
                "_interpolator",
                _grouped_interpolator(table, value_name=self.value_name, statistic=statistic, min_count=self.min_count),
            )
        else:
            raise ValueError("Magnitude kind must be 'constant' or 'variable'.")

    @classmethod
    def constant(cls, value: float) -> ErrorMagnitude:
        """
        Create an error magnitude that is the same for every observation.

        :param value: Standard deviation, in the values units.

        :returns: An ErrorMagnitude with a constant value.
        """

        return cls(kind="constant", value=value)

    @classmethod
    def variable_from_grouped_stats(
        cls,
        statistics: pd.DataFrame,
        *,
        predictor_names: Sequence[str] | None = None,
        value_name: str = "error",
        statistic: str | None = None,
        min_count: int = 0,
        scale: float = 1.0,
        variance_offset: float = 0.0,
        floor: float = 0.0,
    ) -> ErrorMagnitude:
        """
        Create an error magnitude that varies with observations, as described by grouped stats with predictor values.

        With statistic=None, uses "std" if present, otherwise "nmad". Warns if both are present and use only "std".
        Other spread statistics require an explicit statistic name.

        To create a model from grouped statistics, values between groups are interpolated linearly between groups,
        and extrapolated with nearest outside groups. Groups with too few observations are excluded using ``min_count``.

        :param statistics: Grouped statistics indexed by named predictors, with statistic and "count" columns.
            Two-level (variable, statistic) columns from stats() are also accepted.
        :param predictor_names: Predictor names in index order. Defaults to the table's index names.
        :param value_name: Variable to select from two-level columns; unneeded for plain statistic columns.
        :param statistic: Statistic describing the error spread, such as "nmad" or "std".
        :param min_count: Minimum observations needed to use a group's statistic.
        :param scale: Factor multiplying the interpolated error magnitude.
        :param variance_offset: Variance to subtract after scaling (e.g. already assigned to another component).
        :param floor: Minimum error magnitude after subtracting that variance.

        :returns: An ErrorMagnitude evaluated with predict().
        """

        names = tuple(statistics.index.names if predictor_names is None else predictor_names)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Variable magnitude predictor names must be non-empty strings.")
        return cls(
            kind="variable",
            predictor_names=cast(tuple[str, ...], names),
            grouped_statistics=statistics,
            value_name=value_name,
            statistic=statistic,
            min_count=min_count,
            scale=scale,
            variance_offset=variance_offset,
            floor=floor,
        )

    def predict(self, predictors: Mapping[str, Any] | None = None) -> float | NDArray[np.float64]:
        """
        Calculate the error magnitude for the supplied predictor values.

        :param predictors: Named predictor values, as scalars or arrays with compatible shapes.
        :returns: A constant standard deviation, or an array with the predictors' shared shape.
        """

        if self.kind == "constant":
            if self.value is None:
                raise AssertionError("A validated constant magnitude must define value.")
            return self.value
        if predictors is None or self._interpolator is None:
            raise ValueError(f"Variable magnitude requires predictors {self.predictor_names!r}.")

        # Components add through their variances: subtract the variance assigned elsewhere, then take the square root
        total = self.scale * self._interpolator(predictors)
        return np.sqrt(np.maximum(total**2 - self.variance_offset, self.floor**2))

    @property
    def reference_value(self) -> float:
        """
        Return either the constant error magnitude, or the median of all valid groups for a variable error magnitude.

        This representative value weights each component when summarizing correlation for the whole error model.
        """

        if self.kind == "constant":
            if self.value is None:
                raise AssertionError("A validated constant magnitude must define value.")
            return self.value
        if self.grouped_statistics is None:
            raise AssertionError("A validated variable magnitude must contain statistics.")

        # Use the same count threshold, scaling and variance subtraction as predict()
        values = self.grouped_statistics[(self.value_name, self.statistic)].where(
            self.grouped_statistics[(self.value_name, "count")] >= self.min_count
        )
        scaled = self.scale * values.to_numpy(dtype=float)
        magnitude = np.sqrt(np.maximum(scaled**2 - self.variance_offset, self.floor**2))
        finite = magnitude[np.isfinite(magnitude)]
        if finite.size == 0:
            raise ValueError("Variable magnitude contains no finite reference values.")
        return float(np.median(finite))


############################################
# 3/ INDIVIDUAL ERROR COMPONENTS
############################################


@dataclass(frozen=True)
class ErrorComponent:
    """
    Error component describing both the error magnitude and its spatial autocorrelation.

    One or several ErrorComponent can be combined into an ErrorStructure. These combined error components are
    assumed independent of each other, but each component can be autocorrelated in space.

    An error component should generally be estimated jointly with error structure through higher-level function
    such as ``estimate_error_structure()`` from a raster or point object, or ``ErrorStructure.estimate()``.
    Otherwise, to create it manually, define the ``magnitude`` and ``correlation`` manually using an ErrorMagnitude
    and a Variogram object, as demonstrated below.

    As an example, defining independent measurement errors with a constant standard deviation of 2 m needs only a name
    and magnitude:

    .. code-block:: python

        import geoutils as gu

        measurement = gu.ErrorComponent("measurement", magnitude=2.0)
        measurement.predict_magnitude()  # 2.0

    Independent errors can also have magnitudes that vary with any predictor, for instance slope:

    .. code-block:: python

        import pandas as pd

        statistics = pd.DataFrame(
            {"std": [1.0, 3.0], "count": [20, 30]},
            index=pd.Index([0.0, 30.0], name="slope"),
        )
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        terrain = gu.ErrorComponent("terrain", magnitude)
        terrain.predict_magnitude({"slope": [0.0, 15.0, 30.0]})  # array([1., 2., 3.])

    Add a variogram model to describe spatial correlation. Here errors have a constant magnitude of 2 m and
    become uncorrelated at distances of 100 m or more, assuming coordinates are in metres:

    .. code-block:: python

        from geoutils.stats.variography import VariogramModel

        correlation = VariogramModel("spherical", effective_range=100, partial_sill=1)
        spatial = gu.ErrorComponent("spatial", magnitude=2.0, correlation=correlation)
        spatial.predict_correlation([0.0, 100.0])  # array([1., 0.])

    The same correlation model can also be combined with the slope-dependent error magnitude above:

    .. code-block:: python

        spatial_terrain = gu.ErrorComponent("spatial_terrain", magnitude, correlation)
        errors = gu.ErrorStructure([spatial_terrain])
        errors.predict_magnitude({"slope": [0.0, 30.0]})  # array([1., 3.])

    :param name: Unique component name.
    :param magnitude: Constant or variable error magnitude.
    :param correlation: Spatial variogram scaled to unit variance, or None for independent observations.
    :param metadata: Additional details about estimation or diagnostics.
    """

    name: str
    magnitude: ErrorMagnitude | float
    correlation: VariogramModel | Variogram | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        """Convert a numeric magnitude to ErrorMagnitude, check the correlation model, and copy metadata."""

        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Error component name must be a non-empty string.")
        if isinstance(self.magnitude, (int, float, np.integer, np.floating)):
            object.__setattr__(self, "magnitude", ErrorMagnitude.constant(float(self.magnitude)))
        elif not isinstance(self.magnitude, ErrorMagnitude):
            raise TypeError("Error component magnitude must be numeric or an ErrorMagnitude.")

        # A fitted Variogram and its VariogramModel describe the same correlation
        correlation = self.correlation
        if isinstance(correlation, Variogram):
            if correlation.model is None:
                raise ValueError("An error component requires a fitted variogram model.")
            correlation = correlation.model
        if correlation is not None:
            if not isinstance(correlation, VariogramModel):
                raise TypeError("Error component correlation must be a GeoUtils VariogramModel or Variogram.")
            # The error magnitude is defined separately, so the correlation model only needs its shape
            correlation = _normalize_correlation_model(correlation)
        object.__setattr__(self, "correlation", correlation)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def predict_magnitude(self, predictors: Mapping[str, Any] | None = None) -> float | NDArray[np.float64]:
        """
        Calculate error magnitude from this component.

        If the ErrorMagnitude is variable with predictors, predictor values need to be passed.

        :param predictors: Named predictor values required by the magnitude model.
        :returns: A standard deviation, or an array matching the predictors' shared shape.
        """

        if not isinstance(self.magnitude, ErrorMagnitude):
            raise AssertionError("A validated component must contain an ErrorMagnitude.")
        return self.magnitude.predict(predictors)

    def predict_correlation(self, distance: ArrayLike | float) -> float | NDArray[np.float64]:
        """
        Calculate correlation from this component at the given distances.

        :param distance: Pairwise distances, in the variogram model's distance units.
        :returns: Correlation values with the same shape as distance. An independent component gives 1 at
            distance zero and 0 elsewhere.
        """

        distances = np.asarray(distance, dtype=float)
        if self.correlation is None:
            result = np.where(distances == 0, 1.0, 0.0)
        else:
            result = np.asarray(self.correlation.correlation(distances), dtype=float)
        return float(result) if distances.ndim == 0 else cast(NDArray[np.float64], result)


############################################
# 4/ COMBINED ERROR MODELS
############################################


class ErrorStructure:
    """
    Error structure, a sum of independent error components of varying magnitude and autocorrelation.

    The ErrorStructure class combines ErrorComponent objects, each with an error magnitude and spatial
    correlation, stored in ``components``.
    Estimate it using ``estimate_error_structure()`` from a raster or point object, or
    ``ErrorStructure.estimate()``.

    Use ``predict_magnitude()`` and ``predict_covariance()`` to compute values of magnitude and
    covariance according to the error model.
    Pass this object to an interpolation/reduction with uncertainty, or use generate_random_field() to draw errors
    over a raster or point cloud.

    For example, to draw independent measurement errors with standard deviation 2 over a raster:

    .. code-block:: python

        errors = ErrorStructure([ErrorComponent("measurement", magnitude=2.0)])
        field = errors.generate_random_field(like=raster, random_state=42)

    Combine measurement noise of 3 m with a spatial component of 4 m. Their variances add (9 m² + 16 m²), giving a total
    magnitude of 5 m (i.e. 25 m²):

    .. code-block:: python

        import geoutils as gu
        from geoutils.stats.variography import VariogramModel

        measurement = gu.ErrorComponent("measurement", magnitude=3.0)
        correlation = VariogramModel("spherical", effective_range=100, partial_sill=1)
        spatial = gu.ErrorComponent("spatial", magnitude=4.0, correlation=correlation)
        errors = gu.ErrorStructure([measurement, spatial])
        errors.predict_magnitude()  # 5.0
        errors.predict_covariance([0.0, 100.0])  # array([25., 0.])
    """

    def __init__(
        self,
        components: Mapping[str, ErrorComponent] | Sequence[ErrorComponent],
        *,
        empirical_variogram: Variogram | None = None,
        fit_diagnostics: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Construct a named collection of mutually independent error components.

        :param components: Sequence of ErrorComponent objects, or a dictionary keyed by their names.
        :param empirical_variogram: Optional measured variogram used to estimate this model.
        :param fit_diagnostics: Optional details about how the model was fitted.
        :param metadata: Other information to store with the model.
        """

        # Copy components in their input order and check that their names are unique + match dictionary keys
        if isinstance(components, Mapping):
            normalized = dict(components)
            if any(name != component.name for name, component in normalized.items()):
                raise ValueError("Error component mapping keys must match component names.")
        else:
            normalized = {component.name: component for component in components}
            if len(normalized) != len(components):
                raise ValueError("Error component names must be unique.")
        if not normalized or any(not isinstance(component, ErrorComponent) for component in normalized.values()):
            raise ValueError("ErrorStructure requires at least one ErrorComponent.")

        # Store the error model without attaching it to a particular set of observations
        self._components = MappingProxyType(normalized)
        self.empirical_variogram = empirical_variogram
        self.fit_diagnostics = MappingProxyType(dict(fit_diagnostics or {}))
        self.metadata = MappingProxyType(dict(metadata or {}))

    def __repr__(self) -> str:
        """Show the component names."""

        return f"ErrorStructure(components={list(self.components)!r})"

    @property
    def components(self) -> Mapping[str, ErrorComponent]:
        """Return the components by name, as a mapping that cannot be modified."""

        return self._components

    @property
    def required_predictors(self) -> tuple[str, ...]:
        """List the predictors needed to calculate error magnitudes, in the order they first appear."""

        names: list[str] = []
        for component in self.components.values():
            magnitude = cast(ErrorMagnitude, component.magnitude)
            for name in magnitude.predictor_names:
                if name not in names:
                    names.append(name)
        return tuple(names)

    # 4.1/ Error magnitude, correlation and covariance at new locations

    def predict_magnitude(
        self,
        predictors: Mapping[str, Any] | None = None,
        *,
        component: str | None = None,
        like: Any | None = None,
    ) -> Any:
        """Calculate the error magnitude of one component or of all components together.

        :param predictors: Named values needed to calculate the component error magnitudes.
        :param component: Component name, or None to combine all components.
        :param like: Optional raster or point cloud whose type and support wrap the result.
        :returns: A standard deviation, an array, or a spatial object matching like.
        """

        from geoutils._dispatch import _is_pointcloud, _is_raster

        # Spatial predictors supply their numeric values while the optional output keeps the spatial support
        values = predictors
        if predictors is not None:
            values = {
                name: np.ma.asarray(value.data, dtype=float).filled(np.nan)
                if _is_raster(value) or _is_pointcloud(value)
                else value
                for name, value in predictors.items()
            }
        if component is not None:
            result = self.components[component].predict_magnitude(values)
        else:
            # Independent components add through their variances, so square first and take the root afterwards
            variance: Any = 0.0
            for item in self.components.values():
                variance = variance + np.asarray(item.predict_magnitude(values)) ** 2
            result = np.sqrt(variance)
        array = np.asarray(result)
        if like is None:
            return float(array) if array.ndim == 0 else cast(NDArray[np.float64], array)
        is_raster = _is_raster(like)
        if not is_raster and not _is_pointcloud(like):
            raise TypeError("like must be a GeoUtils raster or point cloud.")
        support_shape = like.shape if is_raster else (like.point_count,)
        array = np.broadcast_to(array, support_shape).astype(float, copy=True)
        if is_raster:
            support_values = np.ma.asarray(like.data, dtype=float)
            support_mask = np.ma.getmaskarray(support_values) | ~np.isfinite(support_values.filled(np.nan))
            array = np.ma.masked_array(array, mask=np.asarray(support_mask).squeeze() | ~np.isfinite(array))
        return like.copy(new_array=array)

    def predict_variance(
        self,
        predictors: Mapping[str, Any] | None = None,
        *,
        component: str | None = None,
    ) -> float | NDArray[np.float64]:
        """Calculate the variance of one component or of all components together.

        :param predictors: Named values needed to calculate the component error magnitudes.
        :param component: Component name, or None to combine all components.
        :returns: Squared error magnitudes, as a scalar or an array matching the predictors' shared shape.
        """

        magnitude = np.asarray(self.predict_magnitude(predictors, component=component), dtype=float)
        variance = magnitude**2
        return float(variance) if variance.ndim == 0 else cast(NDArray[np.float64], variance)

    def predict_correlation(self, distance: ArrayLike | float) -> float | NDArray[np.float64]:
        """Summarize correlation across components using their typical error magnitudes.

        For variable error magnitudes, each component uses its median group estimate. Use predict_covariance() with
        predictors when you need the error magnitudes at specific locations instead.

        :param distance: Pairwise distances, in the variogram models' distance units.
        :returns: Correlation values with the same shape as distance.
        """

        # Components with larger variance contribute more to the combined correlation
        total_variance = sum(
            cast(ErrorMagnitude, component.magnitude).reference_value ** 2 for component in self.components.values()
        )
        if total_variance <= 0:
            raise ValueError("An error structure with zero magnitude has no defined correlation.")
        weighted = np.zeros_like(np.asarray(distance, dtype=float))
        for component in self.components.values():
            magnitude = cast(ErrorMagnitude, component.magnitude).reference_value
            weighted += magnitude**2 * component.predict_correlation(distance)
        result = weighted / total_variance
        return float(result) if result.ndim == 0 else cast(NDArray[np.float64], result)

    def predict_covariance(
        self,
        distance: ArrayLike | float,
        *,
        predictors: Mapping[str, Any] | None = None,
        other_predictors: Mapping[str, Any] | None = None,
    ) -> float | NDArray[np.float64]:
        """Calculate covariance for pairs of locations.

        :param distance: Distance between each pair, in the variogram models' distance units.
        :param predictors: Named predictor values for the first location in each pair.
        :param other_predictors: Predictor values for the second locations. Defaults to predictors.
        :returns: Covariance values with the shared shape of distances and predictor arrays.
        """

        other = predictors if other_predictors is None else other_predictors
        covariance: Any = 0.0
        for component in self.components.values():
            # Multiply correlation by the error magnitude at each end of the pair, then add independent components
            first = component.predict_magnitude(predictors)
            second = component.predict_magnitude(other)
            covariance = covariance + np.asarray(first) * np.asarray(second) * component.predict_correlation(distance)
        result = np.asarray(covariance)
        return float(result) if result.ndim == 0 else cast(NDArray[np.float64], result)

    def to_covariance_matrix(
        self,
        coordinates: ArrayLike,
        *,
        predictors: Mapping[str, Any] | None = None,
        max_points: int = 2_000,
    ) -> NDArray[np.float64]:
        """Calculate covariance between every pair of supplied coordinates.

        :param coordinates: Finite array of shape (n_observations, n_dimensions), in the models' distance units.
        :param predictors: Named predictor values, each scalar or with one value per observation.
        :param max_points: Maximum observation count (the matrix uses memory proportional to its square).
        :returns: An array of shape (n_observations, n_observations), including all components' covariance.
        """

        coordinate_array = np.asarray(coordinates, dtype=float)
        if coordinate_array.ndim != 2 or len(coordinate_array) == 0 or np.any(~np.isfinite(coordinate_array)):
            raise ValueError("coordinates must be a finite two-dimensional observation array.")
        if len(coordinate_array) > max_points:
            raise ValueError(
                f"Complete covariance exceeds max_points={max_points}; match the model to source observations "
                "to calculate it in blocks."
            )

        # Observation indexes serve as IDs, so independent errors only contribute to the diagonal
        source_ids = np.arange(len(coordinate_array))
        bound = self.bind(source_ids, coordinates=coordinate_array, predictors=predictors)
        return bound.covariance_block(source_ids, source_ids)

    # 4.2/ Apply the model to observations and draw random errors

    def bind(
        self,
        source_ids: ArrayLike,
        *,
        coordinates: ArrayLike | None = None,
        predictors: Mapping[str, Any] | None = None,
    ) -> BoundErrorStructure:
        """Match this error model to a set of observations for repeated uncertainty calculations.

        :param source_ids: Unique observation IDs, in the order used for coordinates/predictors.
        :param coordinates: Array of shape (n_observations, n_dimensions), required for correlated components.
        :param predictors: Named predictor values, each scalar or with one value per observation.
        :returns: An object with covariance_block() and draw_error() methods for these observations.
        """

        return bind_error_structure(self, source_ids=source_ids, coordinates=coordinates, predictors=predictors)

    def iter_samples(
        self,
        source_ids: ArrayLike,
        *,
        coordinates: ArrayLike | None = None,
        predictors: Mapping[str, Any] | None = None,
        nominal: ArrayLike | None = None,
        n_samples: int = 1,
        kind: Literal["error", "value"] = "error",
        random_state: int | np.random.Generator | None = None,
    ) -> Iterator[NDArray[np.float64]]:
        """Yield random errors, or source values plus those errors, for the selected observations.

        :param source_ids: Unique observation IDs.
        :param coordinates: Array of shape (n_observations, n_dimensions), required for correlated components.
        :param predictors: Named values used to calculate each component's error magnitude.
        :param nominal: Original source values, added to the errors when kind="value". Defaults to zero.
        :param n_samples: Number of independent draws to yield.
        :param kind: "error" for errors alone or "value" for source values + errors.
        :param random_state: Seed or generator for repeatable draws.
        :returns: An iterator yielding arrays of shape (n_observations,), in source ID order.
        """

        if isinstance(n_samples, (bool, np.bool_)) or not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")
        if kind not in {"error", "value"}:
            raise ValueError("kind must be 'error' or 'value'.")

        bound = self.bind(source_ids, coordinates=coordinates, predictors=predictors)
        nominal_values = np.zeros(bound.size, dtype=float) if nominal is None else np.asarray(nominal, dtype=float)
        if nominal_values.shape != (bound.size,):
            raise ValueError("nominal must contain one value per source observation.")

        # Yield one complete draw at a time so we do not store every simulation in memory
        rng = np.random.default_rng(random_state)
        for _ in range(int(n_samples)):
            error = bound.draw_error(rng)
            yield error if kind == "error" else nominal_values + error

    def generate_random_field(
        self,
        like: Any | None = None,
        *,
        source_ids: ArrayLike | None = None,
        coordinates: ArrayLike | None = None,
        predictors: Mapping[str, Any] | None = None,
        n_fields: int = 1,
        random_state: int | np.random.Generator | None = None,
        chunksizes: tuple[int, int] | None = None,
        backend: Literal["gstools", "gpytorch"] = "gstools",
    ) -> Any:
        """Generate one or more error fields over a raster, point cloud, or set of coordinates.

        :param like: Optional raster or point cloud defining coordinates and the returned spatial object.
        :param source_ids: Unique observation IDs when like does not define the output locations.
        :param coordinates: Spatial coordinates aligned with source_ids.
        :param predictors: Named values used to calculate each component's error magnitude.
        :param n_fields: Number of independent fields.
        :param random_state: Seed or generator used for reproducible fields.
        :param chunksizes: Optional Dask raster chunk size (rows, columns) when using GSTools.
        :param backend: Library used to draw correlated components.
        :returns: One field when n_fields is one, otherwise a list or stacked array of fields.
        """

        from geoutils.uncertainty.random_field import random_field

        return random_field(
            self,
            source_ids=source_ids,
            like=like,
            coordinates=coordinates,
            predictors=predictors,
            n_fields=n_fields,
            random_state=random_state,
            chunksizes=chunksizes,
            backend=backend,
        )

    # 4.3/ Estimate a reusable spatial model from observed errors

    @classmethod
    def estimate(
        cls,
        error_proxy: Any,
        *,
        other: Any | None = None,
        other_precision: Literal["same", "negligible"] = "same",
        predictors: Mapping[str, Any] | None = None,
        components: Mapping[str, Mapping[str, Any]] | None = None,
        mask: Any | None = None,
        bins: Mapping[str, Any] | int | None = None,
        spread_estimator: Callable[[Any], Any] | None = None,
        min_count: int = 100,
        subsample_magnitude: int | float = 1_000_000,
        variogram_estimator: str | Callable[[Any], float] = "dowd",
        n_pairs: int = 1_000_000,
        pair_sampling: Literal["loglag", "random_xy"] = "loglag",
        n_lags: int = 24,
        min_lag: float | None = None,
        max_lag: float | None = None,
        n_runs: int = 1,
        fit_method: Literal["variogram"] = "variogram",
        fit_kwargs: Mapping[str, Any] | None = None,
        pair_sampling_kwargs: Mapping[str, Any] | None = None,
        mp_config: MultiprocConfig | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> ErrorStructure:
        """
        Estimate error structure from an error proxy variable.

        This function works on eager, Dask, or multiprocessing inputs.

        An error proxy consists of a variable that can be used to represent errors, for instance the difference of two
        coincident measurements that should normally have the same values, whether from the same sensor, or between a
        sensor and ground data. For instance, with elevation or velocity data, static surfaces (rock, grasslands, etc)
        do not move in time,
        and therefore the difference of elevation/velocity between two acquisitions (even not at the same time) can
        often be used as a decent error proxy to estimate the error structure.

        The error structure is estimated according to the composition chosen for error components, composed of
        error magnitude tied to an error autocorrelation.
        At most one component can have variable error magnitude, i.e. ``magnitude="heteroscedastic"``. Remaining
        components have constant magnitudes. Their variance fractions come from a nested standardized variogram.

        The error magnitude is estimated from the error proxy's statistical spread (e.g. STD, NMAD), optionally
        grouped by predictors to account for variability (e.g., with terrain slope, landcover type).
        The error correlation is estimated by variography, after optional standardization by the variable magnitude.

        When ``other`` is supplied, the proxy is the difference between the two measurements on their common finite
        support. ``other_precision="same"`` assumes independent errors with the same magnitude and correlation in
        both inputs, so the difference is divided by the square root of two. Use ``"negligible"`` when the other
        measurement's error is small enough to ignore.


        This error structure estimation was refactored from that of xDEM (which was method-based, and thus more
        volatile with inputs and outputs).

        For independent point errors, estimate a constant magnitude without fitting a variogram:

        .. code-block:: python

            import numpy as np

            import geoutils as gu

            points = gu.PointCloud.from_xyz(
                x=np.arange(4), y=np.zeros(4), z=np.array([-2.0, -1.0, 1.0, 2.0]), crs=32631
            )
            errors = gu.ErrorStructure.estimate(
                points,
                components={"measurement": {"magnitude": "constant", "correlation": None}},
                spread_estimator=np.std,
            )
            round(errors.predict_magnitude(), 2)  # 1.58

        With SciKit-GStat installed, a raster proxy can also fit a magnitude that varies with a predictor and a
        spatial correlation:

        .. code-block:: python

            import numpy as np
            from affine import Affine

            import geoutils as gu

            quality = np.broadcast_to(np.linspace(0, 1, 12), (12, 12)).copy()
            values = (1 + quality) * np.random.default_rng(31).normal(size=quality.shape)
            grid = Affine(10, 0, 0, 0, -10, 120)
            proxy = gu.Raster.from_array(values, grid, 32632, nodata=-9999)
            predictor = gu.Raster.from_array(quality, grid, 32632, nodata=-9999)
            errors = gu.ErrorStructure.estimate(
                proxy,
                predictors={"quality": predictor},
                components={"spatial": {"magnitude": "heteroscedastic", "correlation": "spherical"}},
                bins=3,
                min_count=10,
                spread_estimator=np.std,
                n_pairs=300,
                n_lags=5,
                pair_sampling="random_xy",
                random_state=4,
            )
            errors.predict_magnitude({"quality": np.array([0.2, 0.8])})
            errors.predict_correlation([0, 20, 80])

        :param error_proxy: Raster or point cloud whose values represent errors.
        :param other: Second raster or point cloud to compare with error_proxy, or None if it already contains errors.
        :param other_precision: Precision of the second input relative to the first, when other is supplied.
        :param predictors: Named continuous variables controlling a heteroscedastic magnitude.
        :param components: Ordered named component specifications with ``magnitude`` and ``correlation`` entries.
        :param mask: Spatial or Boolean mask identifying values used for estimation.
        :param bins: Group definitions by predictor, or one bin count applied to every predictor.
        :param spread_estimator: Spread estimator used for the error magnitude (default nmad).
        :param min_count: Smallest grouped sample retained in the magnitude model.
        :param subsample_magnitude: Maximum observations used to fit the magnitude.
        :param variogram_estimator: Empirical variogram estimator name or callable.
        :param n_pairs: Target spatial pairs per variogram run.
        :param pair_sampling: Pair sampling scheme exposed by GeoUtils spatial objects.
        :param n_lags: Number of empirical lag classes.
        :param min_lag: Smallest sampled spatial distance.
        :param max_lag: Largest sampled spatial distance.
        :param n_runs: Independent variogram samples used to estimate empirical sampling error.
        :param fit_method: Fitting backend, currently ``"variogram"``.
        :param fit_kwargs: Options passed to Variogram.fit().
        :param pair_sampling_kwargs: Advanced options passed to pairsample() and variogram().
        :param mp_config: Worker and tile settings for multiprocessing estimation.
        :param random_state: Random generator or seed used throughout estimation.
        :returns: Fitted error structure with compact diagnostics.
        """

        from geoutils import stats
        from geoutils.uncertainty.estimation import _estimate_error_structure

        return _estimate_error_structure(
            error_proxy,
            other=other,
            other_precision=other_precision,
            predictors=predictors,
            components=components,
            mask=mask,
            bins=bins,
            spread_estimator=stats.nmad if spread_estimator is None else spread_estimator,
            min_count=min_count,
            subsample_magnitude=subsample_magnitude,
            variogram_estimator=variogram_estimator,
            n_pairs=n_pairs,
            pair_sampling=pair_sampling,
            n_lags=n_lags,
            min_lag=min_lag,
            max_lag=max_lag,
            n_runs=n_runs,
            fit_method=fit_method,
            fit_kwargs=fit_kwargs,
            pair_sampling_kwargs=pair_sampling_kwargs,
            mp_config=mp_config,
            random_state=random_state,
        )

    def refit(
        self,
        correlation_models: str | Sequence[str] | None = None,
        *,
        fit_kwargs: Mapping[str, Any] | None = None,
    ) -> ErrorStructure:
        """Refit correlations and their component magnitude contributions.

        Refit uses the retained empirical variogram and variable magnitude model. It updates every dependent component
        together, avoiding an inconsistent magnitude fit paired with a new correlation fit.

        :param correlation_models: Ordered variogram models, defaulting to the current correlated components.
        :param fit_kwargs: Options passed to Variogram.fit().
        :returns: New error structure fitted from the retained compact diagnostics.
        """

        from geoutils.uncertainty.estimation import _refit_error_structure

        return _refit_error_structure(self, correlation_models=correlation_models, fit_kwargs=fit_kwargs)

    def plot_correlation(self, ax: Any | None = None, *, show_error: bool = True, **kwargs: Any) -> Any:
        """Plot the empirical and combined fitted variogram.

        :param ax: Existing Matplotlib axes, or None to create one.
        :param show_error: Whether to draw finite empirical sampling errors.
        :param kwargs: Keyword arguments passed to the empirical point plot.
        :returns: Axes containing the variogram diagnostics.
        """

        if self.empirical_variogram is None:
            raise ValueError("No empirical variogram is stored on this error structure.")
        return self.empirical_variogram.plot(ax=ax, show_error=show_error, **kwargs)

    def plot_magnitude(
        self,
        *,
        component: str | None = None,
        min_count: int | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        """Plot grouped magnitude statistics and sample counts.

        :param component: Variable component to plot, inferred when there is only one.
        :param min_count: Smallest count shown, defaulting to the fitted threshold.
        :param kwargs: Keyword arguments passed to plot_grouped_stats().
        :returns: Named plotting axes.
        """

        # Select the only variable component unless the user names another one
        variable_components = [
            item
            for item in self.components.values()
            if isinstance(item.magnitude, ErrorMagnitude) and item.magnitude.kind == "variable"
        ]
        if component is not None:
            selected = self.components[component]
        elif len(variable_components) == 1:
            selected = variable_components[0]
        else:
            selected = None
        if (
            selected is None
            or not isinstance(selected.magnitude, ErrorMagnitude)
            or selected.magnitude.kind != "variable"
        ):
            raise ValueError("Select one variable error component to plot.")

        # Apply the fitted component's scale and allocated variance to its observed bins
        magnitude = selected.magnitude
        if magnitude.grouped_statistics is None:
            raise AssertionError("A variable magnitude must contain statistics.")
        table = magnitude.grouped_statistics.copy(deep=True)
        column = (magnitude.value_name, magnitude.statistic)
        values = magnitude.scale * table[column].to_numpy(dtype=float)
        table[column] = np.sqrt(np.maximum(values**2 - magnitude.variance_offset, magnitude.floor**2))
        from geoutils import stats

        return stats.plot_grouped_stats(
            table,
            value=magnitude.value_name,
            statistic=cast(str, magnitude.statistic),
            min_count=magnitude.min_count if min_count is None else min_count,
            **kwargs,
        )

    def plot(self, **kwargs: Any) -> Mapping[str, Any]:
        """Plot every available magnitude and correlation diagnostic.

        :param kwargs: Keyword arguments passed to the correlation plot.
        :returns: Mapping containing the created plotting axes.
        """

        axes: dict[str, Any] = {}
        for component in self.components.values():
            if isinstance(component.magnitude, ErrorMagnitude) and component.magnitude.kind == "variable":
                axes[f"magnitude:{component.name}"] = self.plot_magnitude(component=component.name)
        if self.empirical_variogram is not None:
            axes["correlation"] = self.plot_correlation(**kwargs)
        if not axes:
            warnings.warn("This error structure contains no empirical diagnostics to plot.", UserWarning)
        return axes

    def info(self, *, verbose: bool = True) -> str | None:
        """Summarize the reusable model without displaying bound source observations.

        :param verbose: Whether to print the summary instead of returning it.
        :returns: Summary string when verbose is False.
        """

        lines = [f"ErrorStructure with {len(self.components)} independent component(s)"]
        for component in self.components.values():
            # ErrorComponent converts fitted Variogram results to their portable model at construction
            correlation = cast(VariogramModel | None, component.correlation)
            model = "independent" if correlation is None else correlation.model_name
            error_range = None if correlation is None else correlation.effective_range
            magnitude = cast(ErrorMagnitude, component.magnitude)
            size = (
                f"constant {magnitude.reference_value:.4g}"
                if magnitude.kind == "constant"
                else f"varies with {', '.join(magnitude.predictor_names)}"
            )
            range_text = "" if error_range is None else f", range {error_range:.4g}"
            lines.append(f"  {component.name}: {size}, {model}{range_text}")
        result = "\n".join(lines)
        if verbose:
            print(result)
            return None
        return result


# Match error models to the observations used in a calculation.


############################################
# 5/ INPUT CHECKS AND REPEATABLE RANDOM VALUES
############################################


def _normalize_source_ids(source_ids: ArrayLike) -> NDArray[Any]:
    """Check and copy the unique ID of each source observation."""

    if isinstance(source_ids, np.ndarray) and source_ids.ndim != 1:
        raise ValueError("source_ids must be a non-empty one-dimensional array of hashable IDs.")
    try:
        ids = list(cast(Iterable[Any], source_ids))
    except TypeError as exception:
        raise ValueError("source_ids must be a non-empty one-dimensional array of hashable IDs.") from exception
    if not ids:
        raise ValueError("source_ids must be a non-empty one-dimensional array of hashable IDs.")
    try:
        for source_id in ids:
            hash(source_id)
    except TypeError as exception:
        raise ValueError("source_ids must contain hashable IDs.") from exception
    labels = pd.Index(ids, tupleize_cols=False, name="source_id")
    if not labels.is_unique:
        raise ValueError("The selected source observations must contain each source_id once.")
    normalized = np.empty(len(labels), dtype=object)
    normalized[:] = labels.tolist()
    normalized.setflags(write=False)
    return normalized


def _normalize_coordinates(coordinates: ArrayLike | None, size: int) -> NDArray[np.float64] | None:
    """Copy finite coordinates with one row per source observation."""

    if coordinates is None:
        return None
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[0] != size or values.shape[1] == 0 or np.any(~np.isfinite(values)):
        raise ValueError("coordinates must be finite with shape (n_sources, n_dimensions).")
    values = values.copy()
    values.setflags(write=False)
    return values


def _normalize_predictors(
    predictors: Mapping[str, Any] | None,
    *,
    required: tuple[str, ...],
    size: int,
) -> Mapping[str, NDArray[np.float64]]:
    """Expand each predictor used to calculate error magnitudes to one value per observation."""

    supplied = dict(predictors or {})
    missing = set(required).difference(supplied)
    if missing:
        raise ValueError(f"Missing predictors required by the error structure: {sorted(missing)!r}.")
    normalized: dict[str, NDArray[np.float64]] = {}
    for name in required:
        # A scalar predictor applies to every observation; arrays must match the observation count
        values = np.asarray(supplied[name], dtype=float)
        try:
            aligned = np.broadcast_to(values, (size,)).astype(float, copy=True)
        except ValueError as exception:
            raise ValueError(f"Predictor {name!r} must be scalar or contain one value per source.") from exception
        aligned.setflags(write=False)
        normalized[name] = aligned
    return MappingProxyType(normalized)


def _indexed_standard_normal(seed: int, indexes: NDArray[np.int64]) -> NDArray[np.float64]:
    """Generate repeatable independent normal values from a seed and observation indexes."""

    # We use SplitMix64 to give each seed + index pair its own random bits
    # This way, the same pixel gets the same error even if it is calculated in a different chunk or order
    values = np.asarray(indexes, dtype=np.uint64) + np.uint64(seed)
    values += np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)

    # Convert 53 bits to a uniform probability, then to a standard normal value (mean 0, variance 1)
    uniform = ((values >> np.uint64(11)).astype(np.float64) + 0.5) / float(2**53)
    return np.asarray(ndtri(uniform), dtype=np.float64)


############################################
# 6/ COVARIANCE AND RANDOM ERRORS IN MEMORY
############################################


@dataclass(frozen=True)
class BoundErrorStructure:
    """
    Bind error structure calculations to a specific set of observations.

    ErrorStructure.bind() creates this object after checking source IDs/coordinates and calculating error magnitudes.
    covariance_block() returns covariance for selected observations; draw_error() samples their errors together.
    """

    source_ids: NDArray[Any]
    coordinates: NDArray[np.float64] | None
    _component_data: tuple[tuple[ErrorComponent, NDArray[np.float64]], ...] = ()
    predictors: Mapping[str, NDArray[np.float64]] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def size(self) -> int:
        """Return the number of distinct source observations."""

        return len(self.source_ids)

    def covariance_block(self, rows: ArrayLike, columns: ArrayLike) -> NDArray[np.float64]:
        """Calculate part of the covariance matrix without allocating the complete matrix."""

        # Select source rows/columns for this covariance block
        row_indexes = np.asarray(rows, dtype=np.int64).reshape(-1)
        column_indexes = np.asarray(columns, dtype=np.int64).reshape(-1)
        if (
            np.any(row_indexes < 0)
            or np.any(row_indexes >= self.size)
            or np.any(column_indexes < 0)
            or np.any(column_indexes >= self.size)
        ):
            raise ValueError("Covariance block indexes are outside the selected source observations.")
        # For independent errors, only the same source ID is correlated with itself
        # Two observations at identical coordinates still have independent errors if their IDs differ
        covariance = np.zeros((len(row_indexes), len(column_indexes)), dtype=float)
        distances_by_dimensions: dict[tuple[int, ...] | None, NDArray[np.float64]] = {}
        for component, magnitude in self._component_data:
            if component.correlation is None:
                correlation = self.source_ids[row_indexes, None] == self.source_ids[column_indexes][None, :]
            else:
                correlation_model = component.correlation
                if not isinstance(correlation_model, VariogramModel):
                    raise AssertionError("A validated error component must contain a VariogramModel.")
                if self.coordinates is None:
                    raise AssertionError("A bound correlated component must contain coordinates.")

                # Some models use only selected coordinates (e.g. horizontal distance without elevation)
                active_dims = correlation_model.active_dims
                row_coordinates = self.coordinates[row_indexes]
                column_coordinates = self.coordinates[column_indexes]
                if active_dims is not None:
                    row_coordinates = row_coordinates[:, active_dims]
                    column_coordinates = column_coordinates[:, active_dims]

                # Components using the same coordinates can reuse their pairwise distances
                distances = distances_by_dimensions.get(active_dims)
                if distances is None:
                    distances = cdist(row_coordinates, column_coordinates)
                    distances_by_dimensions[active_dims] = distances
                correlation = component.predict_correlation(distances)

            # Covariance = first error magnitude × second error magnitude × correlation; independent components add up
            covariance += magnitude[row_indexes, None] * magnitude[column_indexes][None, :] * correlation
        return covariance

    def draw_error(
        self,
        rng: np.random.Generator,
        *,
        backend: Literal["gstools", "gpytorch"] = "gstools",
        random_coordinates: Any | None = None,
        mesh_type: str = "unstructured",
        field_shape: tuple[int, ...] | None = None,
    ) -> NDArray[np.float64]:
        """Draw errors for all observations together, with the model's spatial correlation.

        GSTools can evaluate a regular raster faster from separate X/Y axes. These optional axes only affect the
        random field calculation; covariance_block() uses the original spatial coordinates.
        """

        combined = np.zeros(self.size, dtype=float)
        for component, magnitude in self._component_data:
            seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            if component.correlation is None:
                unit_field = _indexed_standard_normal(seed, np.arange(self.size, dtype=np.int64))
            else:
                correlation_model = component.correlation
                if not isinstance(correlation_model, VariogramModel):
                    raise AssertionError("A validated error component must contain a VariogramModel.")
                if self.coordinates is None:
                    raise AssertionError("A bound correlated component must contain coordinates.")

                # Draw the field using only the coordinates that this component depends on
                active_dims = correlation_model.active_dims
                coordinates = self.coordinates if active_dims is None else self.coordinates[:, active_dims]
                variogram = Variogram(
                    lags=np.empty(0),
                    semivariance=np.empty(0),
                    counts=np.empty(0, dtype=np.int64),
                    model=correlation_model,
                )
                if backend == "gstools":
                    gstools_variogram = variogram.to_gstools(dim=coordinates.shape[1])
                    gstools = import_optional("gstools", extra_name="geostat")

                    positions = (
                        tuple(coordinates[:, dimension] for dimension in range(coordinates.shape[1]))
                        if random_coordinates is None
                        else random_coordinates
                    )
                    unit_field = np.asarray(
                        gstools.SRF(gstools_variogram.model, seed=seed)(positions, mesh_type=mesh_type), dtype=float
                    )
                    if mesh_type == "structured":
                        if field_shape is None or len(field_shape) != 2:
                            raise ValueError("Structured random fields require a two-dimensional field_shape.")
                        # GSTools uses X/Y order, whereas raster arrays use rows/columns
                        unit_field = unit_field.T
                else:
                    torch = import_optional("torch", extra_name="gp")
                    gpytorch = import_optional("gpytorch", extra_name="gp")
                    gpytorch_variogram = variogram.to_gpytorch(
                        active_dims=tuple(range(coordinates.shape[1])),
                        trainable=False,
                    )
                    coordinate_tensor = torch.as_tensor(coordinates.copy(), dtype=torch.float64)
                    covariance = gpytorch_variogram.kernel(coordinate_tensor)

                    # Independent noise adds variance to each observation (the covariance diagonal)
                    if gpytorch_variogram.noise > 0:
                        diagonal = torch.full((self.size,), gpytorch_variogram.noise, dtype=torch.float64)
                        covariance = covariance.add_diagonal(diagonal)
                    distribution = gpytorch.distributions.MultivariateNormal(
                        torch.zeros(self.size, dtype=torch.float64), covariance
                    )

                    # Use a local generator so drawing a field does not change PyTorch's global random state
                    generator = torch.Generator(device=coordinate_tensor.device)
                    generator.manual_seed(seed)
                    base_samples = torch.randn(self.size, dtype=torch.float64, generator=generator)
                    with torch.no_grad(), gpytorch.settings.fast_computations(covar_root_decomposition=False):
                        unit_field = distribution.rsample(base_samples=base_samples).detach().cpu().numpy()
                unit_field = unit_field.reshape(-1)
                if unit_field.shape != (self.size,):
                    raise RuntimeError("The covariance library returned a field with the wrong number of values.")

            # Scale this unit-variance field by its error magnitude, then add it to the other components
            combined += magnitude * unit_field
        return combined


############################################
# 7/ MATCHING MODELS TO SOURCE OBSERVATIONS
############################################


def _bind_components(
    structure: ErrorStructure,
    *,
    source_ids: NDArray[Any],
    coordinates: NDArray[np.float64] | None,
    predictors: Mapping[str, NDArray[np.float64]],
) -> BoundErrorStructure:
    """Calculate each component's error magnitude and prepare its covariance calculation."""

    component_data: list[tuple[ErrorComponent, NDArray[np.float64]]] = []
    for component in structure.components.values():
        if component.correlation is not None and coordinates is None:
            raise ValueError(f"Coordinates are required for correlated component {component.name!r}.")
        predicted = np.asarray(component.predict_magnitude(predictors), dtype=float)

        # Constant magnitudes apply to every observation; variable magnitudes must supply one value for each
        try:
            magnitude = np.broadcast_to(predicted, (len(source_ids),)).astype(float, copy=True)
        except ValueError as exception:
            raise ValueError(
                f"Component {component.name!r} magnitude shape {predicted.shape} cannot match the source observations."
            ) from exception
        if np.any(~np.isfinite(magnitude)) or np.any(magnitude < 0):
            raise ValueError(f"Component {component.name!r} has non-finite or negative magnitudes.")
        magnitude.setflags(write=False)
        component_data.append((component, magnitude))

    return BoundErrorStructure(
        source_ids=source_ids,
        coordinates=coordinates,
        _component_data=tuple(component_data),
        predictors=predictors,
    )


############################################
# 8/ MATCH AN ERROR MODEL TO OBSERVATIONS
############################################


def bind_error_structure(
    structure: ErrorStructure,
    *,
    source_ids: ArrayLike,
    coordinates: ArrayLike | None = None,
    predictors: Mapping[str, Any] | None = None,
) -> BoundErrorStructure:
    """Match an ErrorStructure to unique source IDs and optional spatial coordinates.

    _normalize_source_ids(), _normalize_coordinates() and _normalize_predictors() check the inputs.
    _bind_components() then calculates each component's error magnitude at those locations.
    """

    if not isinstance(structure, ErrorStructure):
        raise TypeError("structure must be an ErrorStructure.")
    normalized_ids = _normalize_source_ids(source_ids)
    normalized_coordinates = _normalize_coordinates(coordinates, len(normalized_ids))
    normalized_predictors = _normalize_predictors(
        predictors, required=structure.required_predictors, size=len(normalized_ids)
    )
    return _bind_components(
        structure,
        source_ids=normalized_ids,
        coordinates=normalized_coordinates,
        predictors=normalized_predictors,
    )
