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
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist
from scipy.special import ndtri

from geoutils._misc import import_optional
from geoutils.stats.variography import Variogram, VariogramModel
from geoutils.uncertainty.predictors import _grouped_interpolator

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.uncertainty.random_field import GPyTorchInducingField

__all__ = ["ErrorComponent", "ErrorMagnitude", "ErrorStructure"]


############################################
# 1/ CORRELATION HELPERS
############################################


def _normalize_correlation_model(model: VariogramModel) -> VariogramModel:
    """Scale a variogram to unit variance without changing its correlation shape."""

    if model.nugget != 0:
        raise ValueError("Component correlations cannot contain a nugget; use a spatially uncorrelated component.")
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


def _correlation_info(model: VariogramModel, distance_units: str, indent: str = "") -> list[str]:
    """Describe the functional form and parameters of a correlation model, including sums and products."""

    # Describe base parameters in the same units as the empirical distances
    lines: list[str] = []
    if model.effective_range is not None:
        lines.append(f"{indent}Effective range: {model.effective_range:.4g} {distance_units}")
    lines.append(f"{indent}Functional form: {model.model_name}")
    if model.smoothness is not None:
        lines.append(f"{indent}Smoothness: {model.smoothness:.4g}")
    if model.shape is not None:
        lines.append(f"{indent}Shape: {model.shape:.4g}")
    if model.parameters:
        values = ", ".join(f"{value:.4g}" for value in model.parameters)
        lines.append(f"{indent}Additional parameters: {values}")
    if model.active_dims is not None:
        lines.append(f"{indent}Coordinate dimensions: {model.active_dims}")

    # Composite models need each constituent's parameters to describe their correlation
    for number, component in enumerate(model.components, 1):
        lines.append(f"{indent}Model {number}:")
        if model.model_name == "sum":
            lines.append(f"{indent}  Variance fraction: {component.sill / model.sill:.4g}")
        lines.extend(_correlation_info(component, distance_units, indent + "  "))
    return lines


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

        magnitude = gu.ErrorMagnitude.constant(2.0)
        magnitude.predict()  # 2.0

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
    :param model: "piecewise_linear" for interpolation, "linear" for an affine fit, or a function model(x, ...),
        whose remaining arguments are parameters to fit.
        The function receives a vector for one continuous predictor, or an array shaped (n_predictors, n_groups).
    :param fit_kwargs: Options passed to scipy.optimize.curve_fit(), such as p0 and bounds.
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
    model: Literal["piecewise_linear", "linear"] | Callable[..., Any] = "piecewise_linear"
    fit_kwargs: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)
    _fit_parameters: dict[tuple[Any, ...], NDArray[np.float64]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _interpolator: Callable[[Mapping[str, Any]], NDArray[np.float64]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Check the model and prepare interpolation between grouped statistics."""

        # Convert shared numeric options to floats and check their ranges
        object.__setattr__(self, "predictor_names", tuple(self.predictor_names))
        if self.fit_kwargs is not None:
            object.__setattr__(self, "fit_kwargs", dict(self.fit_kwargs))
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
                _grouped_interpolator(
                    table,
                    value_name=self.value_name,
                    statistic=statistic,
                    min_count=self.min_count,
                    model=self.model,
                    fit_kwargs=self.fit_kwargs,
                    fit_parameters=self._fit_parameters,
                ),
            )
        else:
            raise ValueError("Magnitude kind must be 'constant' or 'variable'.")

    def __getstate__(self) -> dict[str, Any]:
        """Serialize the inputs used to rebuild a variable magnitude's interpolator."""

        return {
            "kind": self.kind,
            "value": self.value,
            "grouped_statistics": self.grouped_statistics,
            "predictor_names": self.predictor_names,
            "value_name": self.value_name,
            "statistic": self.statistic,
            "min_count": self.min_count,
            "scale": self.scale,
            "variance_offset": self.variance_offset,
            "floor": self.floor,
            "model": self.model,
            "fit_kwargs": self.fit_kwargs,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuild the magnitude and its interpolator from saved constructor inputs."""

        ErrorMagnitude.__init__(self, **state)

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
        model: Literal["piecewise_linear", "linear"] | Callable[..., Any] = "piecewise_linear",
        fit_kwargs: Mapping[str, Any] | None = None,
    ) -> ErrorMagnitude:
        """
        Create an error magnitude that varies with observations, as described by grouped stats with predictor values.

        With statistic=None, uses "std" if present, otherwise "nmad". Warns if both are present and use only "std".
        Other spread statistics require an explicit statistic name.

        To create a model from grouped statistics, values between groups are interpolated linearly between groups,
        and extrapolated with nearest outside groups. Groups with too few observations are excluded using ``min_count``.
        Categorical index levels select separate models, with interpolation only along continuous predictors.

        The interpolation described above is the default. Set model="linear" for an affine fit, or pass a function
        model(x, ...) to fit another form with scipy.optimize.curve_fit(), with parameters as separate arguments.
        For one continuous predictor, x is a vector of group centers. For N predictors, x has shape (N, n_groups),
        in predictor_names order.
        Fits use finite groups meeting min_count, separately within each categorical combination. Fitted functions
        extrapolate outside the group centers. Negative or non-finite predictions give NaN.

        :param statistics: Grouped statistics indexed by named predictors, with statistic and "count" columns.
            Two-level (variable, statistic) columns from stats() are also accepted.
        :param predictor_names: Predictor names in index order. Defaults to the table's index names.
        :param value_name: Variable to select from two-level columns; unneeded for plain statistic columns.
        :param statistic: Statistic describing the error spread, such as "nmad" or "std".
        :param min_count: Minimum observations needed to use a group's statistic.
        :param scale: Factor multiplying the interpolated error magnitude.
        :param variance_offset: Variance to subtract after scaling (e.g. already assigned to another component).
        :param floor: Minimum error magnitude after subtracting that variance.
        :param model: "piecewise_linear", "linear", or a callable as described above.
        :param fit_kwargs: Options passed to curve_fit(), including initial parameters p0 and bounds.

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
            model=model,
            fit_kwargs=fit_kwargs,
        )

    @property
    def model_name(self) -> str:
        """Return the interpolation name or the fitted function's name."""

        if isinstance(self.model, str):
            return self.model
        return str(getattr(self.model, "__name__", type(self.model).__name__))

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


def _plot_label(name: str, attributes: Mapping[str, Any]) -> str:
    """Format a descriptive variable name and any supplied units for a plot label."""

    fallback = name.replace("_", " ")
    label = attributes.get("long_name") or fallback[:1].upper() + fallback[1:]
    unit = attributes.get("units")
    if unit:
        # Use familiar symbols for common unit names
        unit = {"degree": "°", "degrees": "°", "metre": "m", "meter": "m"}.get(unit, unit)
        label = f"{label} ({unit})"
    return str(label)


def _plot_magnitude_fit(
    magnitude: ErrorMagnitude, table: pd.DataFrame, axes: Mapping[str, Any], empirical_label: str, model_label: str
) -> None:
    """Overlay an interpolated magnitude model and label the empirical groups and fitted model."""

    from matplotlib.lines import Line2D

    from geoutils.stats.grouping import _plot_axis

    if magnitude.grouped_statistics is None:
        raise AssertionError("A variable magnitude must contain statistics.")

    # Match predictor values to the displayed bin centers, including numeric group labels
    levels = [table.index] if table.index.nlevels == 1 else list(table.index.levels)
    statistic_axis = axes["statistic"]
    if len(levels) == 2 and "colorbar" not in axes:
        # Evaluate a separate curve for each category or second predictor bin
        level_x, level_y = levels
        _, centers, _ = _plot_axis(level_x)
        predictor_values = level_x.mid if isinstance(level_x, pd.IntervalIndex) else level_x
        if isinstance(level_x, pd.CategoricalIndex):
            displayed, coordinates = centers, np.asarray(predictor_values)
        else:
            displayed = np.linspace(centers[0], centers[-1], 100)
            coordinates = np.interp(displayed, centers, np.asarray(predictor_values, dtype=float))
        empirical_lines = list(statistic_axis.lines)
        for category, empirical in zip(level_y, empirical_lines):
            fixed_value = category.mid if isinstance(category, pd.Interval) else category
            predictors = {table.index.names[0]: coordinates, table.index.names[1]: fixed_value}
            fitted = magnitude.predict(predictors)
            color = empirical.get_color()
            empirical.set(linestyle="none", markerfacecolor="white", markeredgecolor=color, zorder=3)
            statistic_axis.plot(displayed, fitted, color=color, linewidth=1.5)

        # Category colors are identified above, while line styles distinguish observations from the model
        empirical_handle = Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", color="0.35")
        model_handle = Line2D([], [], color="0.35")
        statistic_axis.legend(
            handles=[empirical_handle, model_handle], labels=[empirical_label, model_label], frameon=False
        )
        return
    if any(isinstance(level, pd.CategoricalIndex) for level in levels):
        # Show category estimates without connecting separate classes
        statistic_axis = axes["statistic"]
        if len(levels) == 1:
            if statistic_axis.containers:
                bars = statistic_axis.containers[0]
                bars.set_label(empirical_label)
                statistic_axis.legend(handles=[bars], framealpha=0.9)
            else:
                statistic_axis.lines[0].set(linestyle="none", label=empirical_label)
                statistic_axis.legend(framealpha=0.9)
        else:
            empirical = Line2D([], [], marker="s", linestyle="none", color="0.5", label=empirical_label)
            statistic_axis.legend(handles=[empirical], framealpha=0.8)
        return
    display_coordinates = []
    predictor_coordinates = []
    for level in levels:
        _, centers, _ = _plot_axis(level)
        predictor_values = level.mid if isinstance(level, pd.IntervalIndex) else level
        displayed = np.linspace(centers[0], centers[-1], 100)
        display_coordinates.append(displayed)
        predictor_coordinates.append(np.interp(displayed, centers, np.asarray(predictor_values, dtype=float)))

    # Evaluate the same interpolator used for magnitude predictions
    predictor_grids = np.meshgrid(*predictor_coordinates, indexing="xy")
    predictors = dict(zip(table.index.names, predictor_grids))
    fitted = magnitude.predict(predictors)
    statistic_axis = axes["statistic"]

    if len(levels) == 1:
        if statistic_axis.containers:
            empirical = statistic_axis.containers[0]
            empirical.set_label(empirical_label)
        else:
            empirical = statistic_axis.lines[0]
            empirical.set(linestyle="none", color="0.35", markerfacecolor="white", label=empirical_label, zorder=3)
        model_line = statistic_axis.plot(
            display_coordinates[0], fitted, color="0.15", linewidth=1.5, label=model_label
        )[0]
        statistic_axis.grid(axis="y", color="0.9", linewidth=0.6)
        statistic_axis.set_axisbelow(True)
        statistic_axis.legend(handles=[empirical, model_line], frameon=False)
    else:
        # Contours show the interpolated fit over the empirical magnitude grid
        finite = np.asarray(fitted)[np.isfinite(fitted)]
        if finite.size and np.ptp(finite) > 0 and all(len(np.unique(values)) > 1 for values in display_coordinates):
            statistic_axis.contour(*display_coordinates, fitted, levels=5, colors="0.15", linewidths=0.8)
        empirical_handle = Line2D([], [], marker="s", linestyle="none", color="0.5", label=empirical_label)
        model_handle = Line2D([], [], color="0.15", label=model_label)
        statistic_axis.legend(handles=[empirical_handle, model_handle], framealpha=0.8)


def _plot_constant_variances(structure: ErrorStructure, ax: Any) -> None:
    """Show how mutually independent components contribute to the combined error variance."""

    # Compare variances because independent components add in squared units
    names = list(structure.components)
    variances = [float(structure.predict_variance(component=name)) for name in names]
    if len(names) > 1:
        names.append("Combined")
        variances.append(sum(variances))
    positions = np.arange(len(names))
    bars = ax.bar(positions, variances, color="0.5")
    labels = [name[:1].upper() + name[1:] for name in names]
    ax.set_xticks(positions, labels)
    if len(structure.components) > 1:
        bars[-1].set_facecolor("0.2")
    ax.bar_label(bars, fmt="%.3g", padding=3)

    # Use squared measurement units when estimation recorded them
    units = structure.metadata.get("plot_attrs", {}).get("value", {}).get("units")
    label = "Error variance" if not units else f"Error variance ({units}²)"
    ax.set(title="Component variances", ylabel=label)
    ax.margins(y=0.15)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)


def _plot_model_correlation(
    structure: ErrorStructure,
    ax: Any,
    *,
    kind: str = "correlation",
    xscale: str = "log",
    title: str | None = None,
    show_counts: bool = True,
    show_error: bool = True,
    **kwargs: Any,
) -> Any:
    """Plot combined model correlation when no empirical variogram is available."""

    if kind not in {"semivariance", "correlation"}:
        raise ValueError("Argument 'kind' must be 'semivariance' or 'correlation'.")
    if xscale not in {"log", "linear"}:
        raise ValueError("Argument 'xscale' must be 'log' or 'linear'.")

    # Include every base range when a component combines several correlation models
    models = [
        cast(VariogramModel, component.correlation)
        for component in structure.components.values()
        if component.correlation is not None
    ]
    ranges = []
    while models:
        model = models.pop()
        if model.components:
            models.extend(model.components)
        elif model.effective_range is not None:
            ranges.append(model.effective_range)
    maximum_distance = 1.5 * max(ranges)
    if xscale == "log":
        # Show the approach to the shortest range without a long flat section near zero
        distances = np.geomspace(min(ranges) / 100, maximum_distance, 500)
    else:
        distances = np.linspace(maximum_distance / 500, maximum_distance, 500)

    # Start the curve above zero to separate uncorrelated variance from spatial covariance
    correlation = np.asarray(structure.predict_correlation(distances))
    values = correlation if kind == "correlation" else 1 - correlation
    options = {"color": "0.15", "label": "Combined model"}
    options.update(kwargs)
    ax.plot(distances, values, **options)
    if xscale == "linear":
        ax.scatter([0], [1 if kind == "correlation" else 0], color="0.15", s=20)

    # Match the labels and grid of the empirical correlation plot
    label = "Error correlation" if kind == "correlation" else "Error semivariance"
    ax.set(
        title="Spatial error autocorrelation" if title is None else title,
        xlabel="Spatial lag (coordinate units)",
        ylabel=label,
        xscale=xscale,
    )
    if xscale == "log" and ax.get_autoscalex_on():
        ax.set_xlim(left=distances[0])
    if kind == "correlation":
        ax.set_ylim(-0.05, 1)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    return ax


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

    As an example, defining spatially uncorrelated errors with a constant standard deviation of 2 m needs only a name
    and magnitude:

    .. code-block:: python

        import geoutils as gu

        uncorrelated = gu.ErrorComponent("uncorrelated", magnitude=2.0)
        uncorrelated.predict_magnitude()  # 2.0

    Spatially uncorrelated errors can also have magnitudes that vary with any predictor, for instance slope:

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
        correlated = gu.ErrorComponent("correlated", magnitude=2.0, correlation=correlation)
        correlated.predict_correlation([0.0, 100.0])  # array([1., 0.])

    The same correlation model can also be combined with the slope-dependent error magnitude above:

    .. code-block:: python

        spatial_terrain = gu.ErrorComponent("spatial_terrain", magnitude, correlation)
        errors = gu.ErrorStructure([spatial_terrain])
        errors.predict_magnitude({"slope": [0.0, 30.0]})  # array([1., 3.])

    :param name: Unique component name.
    :param magnitude: Constant or variable error magnitude.
    :param correlation: Spatial variogram scaled to unit variance, or None for spatially uncorrelated errors.
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

    def __getstate__(self) -> dict[str, Any]:
        """Serialize component metadata as a plain dictionary."""

        return {
            "name": self.name,
            "magnitude": self.magnitude,
            "correlation": self.correlation,
            "metadata": dict(self.metadata),
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the component and its read-only metadata."""

        ErrorComponent.__init__(self, **state)

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
        :returns: Correlation values with the same shape as distance. A spatially uncorrelated component gives 1 at
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
    Error structure, a sum of mutually independent components of varying magnitude and autocorrelation.

    The ErrorStructure class combines ErrorComponent objects, each with an error magnitude and spatial
    correlation, stored in ``components``.
    Estimate it using ``estimate_error_structure()`` from a raster or point object, or
    ``ErrorStructure.estimate()``.

    Use ``predict_magnitude()`` and ``predict_covariance()`` to compute values of magnitude and
    covariance according to the error model.
    Pass this object to an interpolation/reduction with uncertainty, or use generate_random_field() to draw errors
    over a raster or point cloud.

    For example, to draw spatially uncorrelated errors with standard deviation 2 over a raster:

    .. code-block:: python

        errors = ErrorStructure([ErrorComponent("uncorrelated", magnitude=2.0)])
        field = errors.generate_random_field(like=raster, random_state=42)

    Combine spatially uncorrelated errors of 3 m with spatially correlated errors of 4 m.
    Their variances add (9 m² + 16 m²), giving a total magnitude of 5 m (i.e. 25 m²):

    .. code-block:: python

        import geoutils as gu
        from geoutils.stats.variography import VariogramModel

        uncorrelated = gu.ErrorComponent("uncorrelated", magnitude=3.0)
        correlation = VariogramModel("spherical", effective_range=100, partial_sill=1)
        correlated = gu.ErrorComponent("correlated", magnitude=4.0, correlation=correlation)
        errors = gu.ErrorStructure([uncorrelated, correlated])
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

    def __getstate__(self) -> dict[str, Any]:
        """Serialize the components and diagnostics without their read-only wrappers."""

        return {
            "components": tuple(self.components.values()),
            "empirical_variogram": self.empirical_variogram,
            "fit_diagnostics": dict(self.fit_diagnostics),
            "metadata": dict(self.metadata),
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the complete model with read-only component and metadata mappings."""

        ErrorStructure.__init__(self, **state)

    def __repr__(self) -> str:
        """Show the component names."""

        return f"ErrorStructure(components={list(self.components)!r})"

    @property
    def components(self) -> Mapping[str, ErrorComponent]:
        """Return the components by name, as a mapping that cannot be modified."""

        return self._components

    def _drawing_model(self) -> ErrorStructure:
        """Copy the components needed for random fields without fit diagnostics or metadata."""

        return ErrorStructure(
            [ErrorComponent(item.name, item.magnitude, item.correlation) for item in self.components.values()]
        )

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
        chunksizes: int | tuple[int, int] | None = None,
        mp_config: MultiprocConfig | None = None,
    ) -> Any:
        """Calculate the error magnitude of one component or of all components together.

        :param predictors: Named values needed to calculate the component error magnitudes.
        :param component: Component name, or None to combine all components.
        :param like: Optional raster or point cloud whose type and support wrap the result.
        :param chunksizes: Dask raster tile size (rows, columns) or point rows per partition.
        :param mp_config: Worker and output file settings for multiprocessing error maps.
        :returns: A standard deviation, array, or spatial object matching like. Dask maps remain lazy and
            multiprocessing maps are file-backed.
        """

        from geoutils._dispatch import _is_pointcloud, _is_raster

        if like is not None or chunksizes is not None or mp_config is not None:
            from geoutils.uncertainty.predictors import predict_magnitude_map

            return predict_magnitude_map(
                self, predictors, component=component, like=like, chunksizes=chunksizes, mp_config=mp_config
            )

        # Spatial predictors supply their numeric values for direct array predictions
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
        return float(array) if array.ndim == 0 else cast(NDArray[np.float64], array)

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

        # Observation indexes serve as IDs, so uncorrelated errors only contribute to the diagonal
        source_ids = np.arange(len(coordinate_array))
        locations, component_data = _prepare_source_errors(self, len(source_ids), coordinate_array, predictors)
        return _source_covariance_matrix(source_ids, locations, component_data, source_ids)

    # 4.2/ Draw random errors at selected observations

    def iter_samples(
        self,
        source_ids: ArrayLike,
        *,
        coordinates: ArrayLike | None = None,
        predictors: Mapping[str, Any] | None = None,
        nominal: ArrayLike | None = None,
        n_simulations: int = 1,
        kind: Literal["error", "value"] = "error",
        random_state: int | np.random.Generator | None = None,
    ) -> Iterator[NDArray[np.float64]]:
        """Yield random errors, or source values plus those errors, for the selected observations.

        :param source_ids: Unique observation IDs.
        :param coordinates: Array of shape (n_observations, n_dimensions), required for correlated components.
        :param predictors: Named values used to calculate each component's error magnitude.
        :param nominal: Original source values, added to the errors when kind="value". Defaults to zero.
        :param n_simulations: Number of independent draws to yield.
        :param kind: "error" for errors alone or "value" for source values + errors.
        :param random_state: Seed or generator for repeatable draws.
        :returns: An iterator yielding arrays of shape (n_observations,), in source ID order.
        """

        if (
            isinstance(n_simulations, (bool, np.bool_))
            or not isinstance(n_simulations, (int, np.integer))
            or n_simulations < 1
        ):
            raise ValueError("n_simulations must be a positive integer.")
        if kind not in {"error", "value"}:
            raise ValueError("kind must be 'error' or 'value'.")

        ids = _normalize_source_ids(source_ids)
        locations, component_data = _prepare_source_errors(self, len(ids), coordinates, predictors)
        nominal_values = np.zeros(len(ids), dtype=float) if nominal is None else np.asarray(nominal, dtype=float)
        if nominal_values.shape != (len(ids),):
            raise ValueError("nominal must contain one value per source observation.")

        # Yield one complete draw at a time so we do not store every simulation in memory
        rng = np.random.default_rng(random_state)
        for _ in range(int(n_simulations)):
            error = _draw_source_errors(len(ids), locations, component_data, rng)
            yield error if kind == "error" else nominal_values + error

    def generate_random_field(
        self,
        like: Any | None = None,
        *,
        coordinates: ArrayLike | None = None,
        predictors: Mapping[str, Any] | None = None,
        n_fields: int = 1,
        random_state: int | np.random.Generator | None = None,
        chunksizes: int | tuple[int, int] | None = None,
        mp_config: MultiprocConfig | None = None,
        backend: Literal["gstools", "gpytorch"] = "gpytorch",
        gpytorch_inducing_points: int | Literal["auto"] | None = "auto",
    ) -> Any:
        """Generate one or more error fields over a raster, point cloud, or set of coordinates.

        :param like: Optional Raster, Xarray DataArray, PointCloud, or GeoDataFrame defining the result locations and
            type.
        :param coordinates: Spatial coordinates for an array result when like is not supplied.
        :param predictors: Named values used to calculate each component's error magnitude.
        :param n_fields: Number of independent fields.
        :param random_state: Seed or generator used for reproducible fields.
        :param chunksizes: Dask raster tile size (rows, columns) or point rows per partition.
        :param mp_config: Worker and output file settings for multiprocessing fields.
        :param backend: Library used to draw correlated components.
        :param gpytorch_inducing_points: Target grid size for approximate GPyTorch fields. "auto" uses 256 points
            for spatial inputs and an exact draw for plain coordinates; None requests an exact in-memory draw.
        :returns: One field when n_fields is one, otherwise a list or stacked array of fields.
        """

        from geoutils.uncertainty.random_field import random_field

        return random_field(
            self,
            like=like,
            coordinates=coordinates,
            predictors=predictors,
            n_fields=n_fields,
            random_state=random_state,
            chunksizes=chunksizes,
            mp_config=mp_config,
            backend=backend,
            gpytorch_inducing_points=gpytorch_inducing_points,
        )

    # 4.3/ Estimate a reusable spatial model from observed errors

    @classmethod
    def estimate(
        cls,
        source: Any,
        *,
        other: Any | None = None,
        other_precision: Literal["same", "negligible"] = "negligible",
        at: Literal["self", "other"] | None = "self",
        align: Literal["raise", "reproject"] = "raise",
        predictors: Mapping[str, Any] | None = None,
        components: Mapping[str, Mapping[str, Any]] | None = None,
        mask: Any | None = None,
        bins: Mapping[str, Any] | int | None = None,
        categories: Mapping[str, Sequence[Any]] | None = None,
        spread_estimator: Callable[[Any], Any] | None = None,
        min_count: int = 100,
        subsample_magnitude: int | float = 1_000_000,
        magnitude_model: Literal["piecewise_linear", "linear"] | Callable[..., Any] = "piecewise_linear",
        magnitude_fit_kwargs: Mapping[str, Any] | None = None,
        variogram_estimator: str | Callable[[Any], float] = "matheron",
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
        Estimate error structure from an error proxy or differences between two measurements.

        This function works on in-memory, Dask, or multiprocessing inputs.

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

        Range warnings consider components contributing at least 3% of the fitted variance.

        The error magnitude is estimated from the error proxy's statistical spread (e.g. STD, NMAD), optionally
        grouped by predictors to account for variability (e.g., with terrain slope, landcover type).
        The error correlation is estimated by variography, after optional standardization by the variable magnitude.

        With ``other=None``, source is treated as an error proxy. With two inputs, the proxy is their difference
        on their common finite support. ``other_precision="same"`` assumes independent errors with the same magnitude
        and correlation in both inputs, so the difference is divided by the square root of two. Use ``"negligible"``
        when the other measurement's error is small enough to ignore.


        This error structure estimation was refactored from that of xDEM (which was method-based, and thus more
        volatile with inputs and outputs).

        For spatially uncorrelated point errors, estimate a constant magnitude without fitting a variogram:

        .. code-block:: python

            import numpy as np

            import geoutils as gu

            points = gu.PointCloud.from_xyz(
                x=np.arange(4), y=np.zeros(4), z=np.array([-2.0, -1.0, 1.0, 2.0]), crs=32631
            )
            errors = gu.ErrorStructure.estimate(
                points,
                components={"uncorrelated": {"magnitude": "constant", "correlation": None}},
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
                components={"correlated": {"magnitude": "heteroscedastic", "correlation": "spherical"}},
                bins=3,
                min_count=10,
                n_pairs=3_000,
                n_lags=5,
                pair_sampling="random_xy",
                random_state=4,
            )
            errors.predict_magnitude({"quality": np.array([0.2, 0.8])})
            errors.predict_correlation([0, 20, 80])

        :param source: Raster or point cloud of measurements, or an error proxy when other is None.
        :param other: Second raster or point cloud to compare with source. None treats source as an error proxy.
        :param other_precision: Precision of the second input relative to the first (default "negligible").
            Only used with two inputs.
        :param at: Comparison locations: "self", "other", or None to prefer point locations when present.
        :param align: Whether mismatched grids or coordinate systems raise an error or are reprojected for comparison.
        :param predictors: Named variables controlling a heteroscedastic magnitude.
        :param components: Ordered named component specifications with ``magnitude`` and ``correlation`` entries.
            A correlation can be a model name, a custom variogram callable, or None.
        :param mask: Spatial or Boolean mask identifying values used for estimation.
        :param bins: Continuous group definitions, or one bin count applied to every continuous predictor.
        :param categories: Declared values for categorical predictors, using numeric codes for spatial inputs.
            Continuous predictors are interpolated separately within each category.
        :param spread_estimator: Spread estimator used for the error magnitude (default numpy.std()).
        :param min_count: Smallest grouped sample retained in the magnitude model.
        :param subsample_magnitude: Maximum observations used to fit the magnitude.
        :param magnitude_model: "piecewise_linear" for interpolation (default), "linear" for an affine fit,
            or a callable model(x, ...) with parameters as separate arguments, fitted to grouped spreads.
            With one continuous predictor x is a vector; with N predictors its shape is (N, n_groups), in predictor
            order. Fits are separate for each
            categorical combination. Fitted functions extrapolate and return NaN for negative or non-finite spreads.
        :param magnitude_fit_kwargs: Options passed to scipy.optimize.curve_fit(), including p0 and bounds.
        :param variogram_estimator: Empirical variogram estimator name or callable (default "matheron").
        :param n_pairs: Target spatial pairs per variogram run.
        :param pair_sampling: Pair sampling scheme exposed by GeoUtils spatial objects.
        :param n_lags: Number of empirical lag classes.
        :param min_lag: Smallest sampled spatial distance.
        :param max_lag: Largest sampled spatial distance. Defaults to half the raster diagonal or the point cloud
            diagonal, as in variogram().
        :param n_runs: Independent variogram samples used to estimate empirical sampling error.
        :param fit_method: Fitting backend, currently ``"variogram"``.
        :param fit_kwargs: Options passed to Variogram.fit().
            min_count defaults to 100 sampled pairs per lag bin for error structure fitting.
        :param pair_sampling_kwargs: Advanced options passed to pairsample() and variogram().
        :param mp_config: Worker and tile settings for multiprocessing estimation.
        :param random_state: Random generator or seed used throughout estimation.
        :returns: Fitted error structure with compact diagnostics.
        """

        from geoutils.uncertainty.estimation import _estimate_error_structure

        return _estimate_error_structure(
            source,
            other=other,
            other_precision=other_precision,
            at=at,
            align=align,
            predictors=predictors,
            components=components,
            mask=mask,
            bins=bins,
            categories=categories,
            spread_estimator=np.std if spread_estimator is None else spread_estimator,
            min_count=min_count,
            subsample_magnitude=subsample_magnitude,
            magnitude_model=magnitude_model,
            magnitude_fit_kwargs=magnitude_fit_kwargs,
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
        correlation_models: str | Callable[..., Any] | Sequence[str | Callable[..., Any]] | None = None,
        *,
        fit_kwargs: Mapping[str, Any] | None = None,
    ) -> ErrorStructure:
        """Refit correlations and their component magnitude contributions.

        Refit uses the retained empirical variogram and variable magnitude model. It updates every dependent component
        together, avoiding an inconsistent magnitude fit paired with a new correlation fit.

        :param correlation_models: Ordered variogram models, defaulting to the current correlated components.
        :param fit_kwargs: Options passed to Variogram.fit().
            min_count defaults to the minimum pair count recorded by the previous fit.
        :returns: New error structure fitted from the retained compact diagnostics.
        """

        return self.refit_variogram(correlation_models, fit_kwargs=fit_kwargs)

    def refit_magnitude(
        self,
        model: Literal["piecewise_linear", "linear"] | Callable[..., Any] | None = None,
        *,
        min_count: int | None = None,
        fit_kwargs: Mapping[str, Any] | None = None,
    ) -> ErrorStructure:
        """
        Fit a magnitude model to the stored grouped spread estimates without reading the source data again.

        Correlation shapes and their variance fractions are reused. The stored empirical variogram still describes
        the original standardization. Use estimate() again to change the groups or recompute spatial correlation
        after changing the magnitude model.

        :param model: "piecewise_linear", "linear", or a callable model(x, ...) with parameters as separate arguments.
            None refits the current model. See estimate() for predictor dimensions and categorical groups.
        :param min_count: Minimum observations per magnitude group, defaulting to the previous threshold.
        :param fit_kwargs: Options passed to scipy.optimize.curve_fit(). Previous options are reused when model
            and fit_kwargs are both None.
        :returns: New error structure fitted from the stored magnitude table.
        """

        from geoutils.uncertainty.estimation import _refit_error_structure_magnitude

        return _refit_error_structure_magnitude(self, model=model, min_count=min_count, fit_kwargs=fit_kwargs)

    def refit_variogram(
        self,
        correlation_models: str | Callable[..., Any] | Sequence[str | Callable[..., Any]] | None = None,
        *,
        fit_kwargs: Mapping[str, Any] | None = None,
    ) -> ErrorStructure:
        """
        Fit correlation models to the stored empirical variogram without drawing new pair samples.

        Component magnitudes are updated using the refitted variance fractions and the stored total magnitude model.

        :param correlation_models: Ordered model names or callables, defaulting to the current correlated components.
            Custom functions receive distance, effective range, partial sill and any additional fitted parameters.
        :param fit_kwargs: Options passed to Variogram.fit(). The previous minimum pair count is reused by default.
        :returns: New error structure fitted from the stored variogram statistics.
        """

        from geoutils.uncertainty.estimation import _refit_error_structure_correlation

        return _refit_error_structure_correlation(self, correlation_models=correlation_models, fit_kwargs=fit_kwargs)

    def plot_correlation(self, ax: Any | None = None, *, show_error: bool = True, **kwargs: Any) -> Any:
        """Plot the empirical and combined fitted variogram.

        :param ax: Existing Matplotlib axes, or None to create one.
        :param show_error: Whether to draw finite empirical sampling errors.
        :param kwargs: Keyword arguments passed to Variogram.plot().
        :returns: Axes containing the variogram diagnostics.
        """

        if self.empirical_variogram is None:
            raise ValueError("No empirical variogram is stored on this error structure.")

        # Name the error quantity separately from the estimator shown in the legend
        kwargs.setdefault("title", "Spatial error autocorrelation")
        axis = self.empirical_variogram.plot(ax=ax, show_error=show_error, **kwargs)
        label = "Error correlation" if kwargs.get("kind") == "correlation" else "Error semivariance"
        axis.set_ylabel(label)
        if kwargs.get("kind") == "correlation":
            axis.set_ylim(-0.05, 1)
        return axis

    def plot_magnitude(
        self,
        *,
        component: str | None = None,
        min_count: int | None = None,
        show_model: bool = True,
        plot_kind: Literal["auto", "line", "bar", "heatmap"] = "auto",
        title: str = "Error variability",
        xlabel: str | None = None,
        ylabel: str | None = None,
        cbar_title: str | None = None,
        count_label: str = "Count",
        empirical_label: str | None = None,
        model_label: str | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        """Plot grouped magnitude statistics and sample counts.

        Labels use the input long_name and units metadata saved during estimation.
        Pass label arguments to override them.

        :param component: Variable component to plot, inferred when there is only one.
        :param min_count: Smallest count shown, defaulting to the fitted threshold.
        :param show_model: Whether to overlay the fitted magnitude model on the empirical groups.
        :param plot_kind: "auto" draws bars for one categorical predictor, curves for continuous predictors or a
            categorical/continuous pair, and a heatmap for two continuous predictors. Use "line", "bar", or "heatmap"
            to select the layout explicitly. Bars require one predictor and heatmaps require two.
        :param title: Title above the plot and its count histograms.
        :param xlabel: Horizontal predictor label, inferred from its name and units when omitted.
        :param ylabel: Magnitude label for bars and curves, or the second predictor's label for a heatmap.
        :param cbar_title: Heatmap colorbar label, inferred from the value units when omitted.
        :param count_label: Sample count label on each histogram.
        :param empirical_label: Legend label for the empirical groups, defaulting to their spread estimator.
        :param model_label: Legend label for the fitted magnitude model.
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
        if model_label is None:
            name = "linear interpolation" if magnitude.model == "piecewise_linear" else magnitude.model_name
            model_label = f"Model ({name})"
        if magnitude.grouped_statistics is None:
            raise AssertionError("A variable magnitude must contain statistics.")
        table = magnitude.grouped_statistics.copy(deep=True)
        column = (magnitude.value_name, magnitude.statistic)
        values = magnitude.scale * table[column].to_numpy(dtype=float)
        table[column] = np.sqrt(np.maximum(values**2 - magnitude.variance_offset, magnitude.floor**2))
        from geoutils import stats
        from geoutils.stats.grouping import _prepare_grouped_plot

        # Apply the error labels before saving the completed figure
        table, layout = _prepare_grouped_plot(table, plot_kind)
        savefig_fname = kwargs.pop("savefig_fname", None)
        axes = stats.plot_grouped_stats(
            table,
            value=magnitude.value_name,
            statistic=cast(str, magnitude.statistic),
            min_count=magnitude.min_count if min_count is None else min_count,
            plot_kind=layout,
            **kwargs,
        )
        estimator = cast(str, magnitude.statistic)
        estimator_label = {"nmad": "NMAD", "std": "STD"}.get(estimator, estimator)

        # Label magnitudes and predictors from the original input metadata
        plot_attrs = self.metadata.get("plot_attrs", {})
        predictor_attrs = plot_attrs.get("predictors", {})
        statistic_axis = axes["statistic"]
        magnitude_label = _plot_label("Error magnitude", {"units": plot_attrs.get("value", {}).get("units")})
        if "colorbar" in axes:
            axes["colorbar"].set_xlabel(magnitude_label)
        else:
            statistic_axis.set_ylabel(magnitude_label)
        first_predictor = table.index.names[0]
        statistic_axis.set_xlabel(_plot_label(first_predictor, predictor_attrs.get(first_predictor, {})))
        if table.index.nlevels == 2:
            second_predictor = table.index.names[1]
            second_label = _plot_label(second_predictor, predictor_attrs.get(second_predictor, {}))
            if "colorbar" in axes:
                statistic_axis.set_ylabel(second_label)
            else:
                axes["count_x"].get_legend().set_title(second_label)

        # Apply supplied labels after deriving defaults from the input metadata
        if xlabel is not None:
            statistic_axis.set_xlabel(xlabel)
        if ylabel is not None:
            statistic_axis.set_ylabel(ylabel)
        if cbar_title is not None and "colorbar" in axes:
            axes["colorbar"].set_xlabel(cbar_title)

        # Label categorical raster codes with their recorded class names
        levels = [table.index] if table.index.nlevels == 1 else list(table.index.levels)
        for dimension, (name, level) in enumerate(zip(table.index.names, levels)):
            attributes = predictor_attrs.get(name, {})
            if not isinstance(level, pd.CategoricalIndex) or "flag_values" not in attributes:
                continue
            meanings = attributes.get("flag_meanings", "").split()
            labels = dict(zip(attributes["flag_values"], meanings))
            tick_labels = [labels.get(value, str(value)).replace("_", " ").capitalize() for value in level]
            positions = np.arange(len(level)) + 0.5
            if dimension == 0:
                statistic_axis.set_xticks(positions, tick_labels)
            elif "colorbar" in axes:
                statistic_axis.set_yticks(positions, tick_labels)
            else:
                axes["count_y"].set_yticks(positions, tick_labels)
                for text, label in zip(axes["count_x"].get_legend().get_texts(), tick_labels):
                    text.set_text(label)

        # Add the variability title and fitted magnitude
        count_axis = axes.get("count", axes.get("count_x"))
        title_x = 0.5
        if "count_y" in axes:
            count_bounds = count_axis.get_position()
            right_edge = axes["count_y"].get_position().x1
            title_x = (right_edge - count_bounds.x0) / (2 * count_bounds.width)
        count_axis.set_title(title, x=title_x)
        count_axis.set_ylabel(count_label)
        if "count_y" in axes:
            axes["count_y"].set_xlabel(count_label)
        if show_model:
            if empirical_label is None:
                empirical_label = f"Empirical ({estimator_label})"
            _plot_magnitude_fit(magnitude, table, axes, empirical_label, model_label)

        if savefig_fname is not None:
            statistic_axis.figure.savefig(savefig_fname, bbox_inches="tight")
        return axes

    @overload
    def plot(
        self,
        *,
        ax: Any | None = None,
        show_components: bool = False,
        plot_kind: Literal["auto", "line", "bar", "heatmap"] = "auto",
        return_axes: Literal[False] = False,
        **kwargs: Any,
    ) -> None: ...

    @overload
    def plot(
        self,
        *,
        ax: Any | None = None,
        show_components: bool = False,
        plot_kind: Literal["auto", "line", "bar", "heatmap"] = "auto",
        return_axes: Literal[True],
        **kwargs: Any,
    ) -> Mapping[str, Any]: ...

    @overload
    def plot(
        self,
        *,
        ax: Any | None = None,
        show_components: bool = False,
        plot_kind: Literal["auto", "line", "bar", "heatmap"] = "auto",
        return_axes: bool,
        **kwargs: Any,
    ) -> Mapping[str, Any] | None: ...

    def plot(
        self,
        *,
        ax: Any | None = None,
        show_components: bool = False,
        plot_kind: Literal["auto", "line", "bar", "heatmap"] = "auto",
        return_axes: bool = False,
        **kwargs: Any,
    ) -> Mapping[str, Any] | None:
        """
        Plot magnitude and correlation diagnostics side by side.

        Correlation is shown by default. Pass kind="semivariance" to show the variogram instead.
        Magnitude and predictor labels use the input long_name and units metadata saved during estimation.
        Structures with only constant magnitudes show component variances. If no empirical variogram is stored,
        a correlation model is plotted directly, with semivariance expressed relative to the combined variance.
        Its logarithmic distance axis starts at one hundredth of the shortest component range.

        :param ax: Optional Matplotlib axes whose area is divided between the diagnostics.
        :param show_components: Whether to add each component's variance-weighted contribution to the fitted curve.
        :param plot_kind: Magnitude layout: "auto", "line", "bar", or "heatmap", as in plot_magnitude().
        :param return_axes: Whether to return the named plotting axes instead of None.
        :param kwargs: Keyword arguments passed to plot_correlation(), or to the model curve when no empirical
            variogram is stored. Both accept kind, xscale and title; counts and sampling errors require empirical data.
        :returns: None, or a mapping naming the variance, magnitude and correlation axes when return_axes=True.
        """

        # Collect the available diagnostics before creating their common figure
        variable_components = [
            component
            for component in self.components.values()
            if isinstance(component.magnitude, ErrorMagnitude) and component.magnitude.kind == "variable"
        ]
        constant_magnitudes = not variable_components
        has_model = any(component.correlation is not None for component in self.components.values())
        has_variance = any(
            cast(ErrorMagnitude, component.magnitude).reference_value > 0 for component in self.components.values()
        )
        has_correlation = self.empirical_variogram is not None or (has_model and has_variance)
        panel_count = len(variable_components) + int(constant_magnitudes) + int(has_correlation)

        # Arrange one column per diagnostic, including within a supplied plotting area
        matplotlib = import_optional("matplotlib")
        import matplotlib.pyplot as plt

        if ax is None:
            figure, frames = plt.subplots(1, panel_count, figsize=(6 * panel_count, 4.5), squeeze=False)
            figure.subplots_adjust(wspace=0.4, bottom=0.15)
            panel_axes = list(frames[0])
        elif not isinstance(ax, matplotlib.axes.Axes):
            raise TypeError("Argument 'ax' must be a Matplotlib Axes or None.")
        elif panel_count == 1:
            panel_axes = [ax]
        else:
            ax.set_axis_off()
            width = 1 / (panel_count + 0.4 * (panel_count - 1))
            panel_axes = [ax.inset_axes((index * 1.4 * width, 0, width, 1)) for index in range(panel_count)]

        # Draw component magnitudes or variances beside the available correlation
        axes: dict[str, Any] = {}
        for component, panel in zip(variable_components, panel_axes):
            axes[f"magnitude:{component.name}"] = self.plot_magnitude(
                component=component.name, ax=panel, plot_kind=plot_kind
            )
        if constant_magnitudes:
            _plot_constant_variances(self, panel_axes[0])
            axes["variance"] = panel_axes[0]
        if has_correlation:
            kwargs.setdefault("kind", "correlation")
            if self.empirical_variogram is not None:
                axes["correlation"] = self.plot_correlation(ax=panel_axes[-1], **kwargs)
            else:
                axes["correlation"] = _plot_model_correlation(self, ax=panel_axes[-1], **kwargs)

            # Draw components on the same distance grid as the combined model
            if show_components and has_model and has_variance:
                correlation_axis = axes["correlation"]
                distances = np.asarray(correlation_axis.lines[-1].get_xdata())
                variances = {
                    name: cast(ErrorMagnitude, component.magnitude).reference_value ** 2
                    for name, component in self.components.items()
                }
                total_variance = sum(variances.values())
                fitted_model = self.empirical_variogram.model if self.empirical_variogram is not None else None

                # Use fitted sill shares for empirical diagnostics and magnitude shares for known models
                for name, component in self.components.items():
                    fraction = variances[name] / total_variance
                    if fitted_model is not None:
                        fraction = component.metadata.get("initial_variance_fraction", fraction)
                    correlation = component.predict_correlation(distances)
                    values = fraction * correlation
                    if kwargs["kind"] == "semivariance":
                        scale = fitted_model.sill if fitted_model is not None else 1.0
                        values = fraction * (1 - correlation) * scale
                    label = name.replace("_", " ").capitalize()
                    correlation_axis.plot(distances, values, linestyle="--", label=label)
                correlation_axis.legend(frameon=False)
        return axes if return_axes else None

    def info(self, *, verbose: bool = True) -> str | None:
        """
        Summarize component magnitudes, correlation models and estimation diagnostics.

        Variable magnitudes report the minimum, median and maximum across finite bins meeting min_count,
        after scaling and subtracting variance assigned to other components. Units and sample sizes are included
        when recorded. Pair counts include contributions from all variogram runs, including any repeated pairs.

        :param verbose: Whether to print the summary instead of returning it.
        :returns: Summary string when verbose is False.
        """

        # Read recorded units without accessing the original observations or predictors
        unit = self.metadata.get("plot_attrs", {}).get("value", {}).get("units", "")
        unit_suffix = f" {unit}" if unit else ""
        empirical = self.empirical_variogram
        distance_units = (
            "coordinate units" if empirical is None else empirical.attrs.get("distance_units", "coordinate units")
        )
        lines = [f"ErrorStructure with {len(self.components)} independent component(s)"]
        lines.extend(["", "Components:"])
        magnitude_details: list[str] = []
        spread_name = self.metadata.get("spread_estimator")

        # Describe each component's magnitude and the bins supporting a variable magnitude
        for number, component in enumerate(self.components.values(), 1):
            lines.extend(["", f"Component {number}: {component.name}"])
            magnitude = cast(ErrorMagnitude, component.magnitude)
            if magnitude.kind == "constant":
                lines.extend(
                    ["  Magnitude: Constant", f"    Error (±1σ): {magnitude.reference_value:.4g}{unit_suffix}"]
                )
            else:
                lines.append("  Magnitude: Variable")
                table = magnitude.grouped_statistics
                if table is None:
                    raise AssertionError("A variable magnitude must contain grouped statistics.")
                spread = table[(magnitude.value_name, magnitude.statistic)].to_numpy(dtype=float)
                counts = table[(magnitude.value_name, "count")].to_numpy(dtype=float)
                valid = np.isfinite(spread) & (counts >= magnitude.min_count)

                # Summarize the fitted error at observed bin centers, including each category
                predictors = {}
                for name in magnitude.predictor_names:
                    coordinates = table.index.get_level_values(name)[valid]
                    if isinstance(coordinates, pd.IntervalIndex):
                        coordinates = coordinates.mid
                    predictors[name] = coordinates.to_numpy()

                # Summarize component magnitudes after allocating variance to the other components
                values = np.asarray(magnitude.predict(predictors), dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    lines.append(
                        f"    Error range (±1σ): {np.min(values):.4g} to {np.max(values):.4g}{unit_suffix} "
                        f"(median {np.median(values):.4g}{unit_suffix})"
                    )
                else:
                    lines.append("    Error range (±1σ): No finite values at bin centers")

                # Describe the model after its error range
                lines.append(f"    Predictors: {', '.join(magnitude.predictor_names)}")
                lines.append(f"    Functional form: {magnitude.model_name}")
                for category, parameters in magnitude._fit_parameters.items():
                    label = f" {category}" if category else ""
                    values_text = ", ".join(f"{value:.4g}" for value in parameters)
                    lines.append(f"    Parameters{label}: {values_text}")
                if magnitude.scale != 1:
                    lines.append(f"    Scale: {magnitude.scale:.4g}")
                if magnitude.floor:
                    lines.append(f"    Floor: {magnitude.floor:.4g}{unit_suffix}")
                if spread_name is None:
                    spread_name = magnitude.statistic

                # Combine bin coverage and the sample counts behind the magnitude fit
                bin_text = (
                    f"  Magnitude bins: {np.count_nonzero(valid)}/{len(table)} valid "
                    f"(minimum {magnitude.min_count} samples)"
                )
                if np.any(valid):
                    bin_text += (
                        f", {np.min(counts[valid]):.4g} to {np.max(counts[valid]):.4g} samples per bin "
                        f"(median {np.median(counts[valid]):.4g})"
                    )
                magnitude_details.append(bin_text)

            # Show the full correlation form, including parameters of composed models
            correlation = cast(VariogramModel | None, component.correlation)
            if correlation is None:
                lines.append("  Correlation: Spatially uncorrelated")
            else:
                lines.append("  Correlation:")
                lines.extend("    " + line for line in _correlation_info(correlation, distance_units))

        # Distinguish the fitted spread sample from all available finite observations
        diagnostics = self.fit_diagnostics.get("magnitude", {})
        lines.extend(["", "Estimation:"])
        if not diagnostics and empirical is None and "spread_estimator" not in self.metadata and not magnitude_details:
            lines.append("  Not recorded")
        else:
            lines.append(f"  Magnitude estimator: {spread_name or 'Not recorded'}")
            sample_text = "  Magnitude sample size: "
            sample_count = diagnostics.get("sample_count")
            valid_count = diagnostics.get("valid_count")
            if sample_count is not None:
                if valid_count is not None and sample_count != valid_count:
                    sample_text += f"{sample_count} of {valid_count}"
                else:
                    sample_text += str(sample_count)
            else:
                sample_text += "Not recorded"
                if valid_count is not None:
                    sample_text += f" ({valid_count} available)"
            lines.append(sample_text)
            lines.extend(magnitude_details)
            estimator = self.metadata.get("variogram_estimator")
            if estimator is None:
                estimator = "Not estimated" if empirical is None else empirical.estimator or "Not recorded"
            lines.append(f"  Variogram estimator: {estimator}")

            # Counts sum the pairs contributing to the stored variogram, across all runs
            if empirical is not None:
                runs = empirical.attrs.get("n_runs", self.metadata.get("n_runs", 1))
                sample_text = f"  Variogram sample size: {int(np.sum(empirical.counts))} pairs"
                pair_details = []
                if "sampling" in empirical.attrs:
                    pair_details.append(str(empirical.attrs["sampling"]))
                if runs > 1:
                    pair_details.append(f"across {runs} runs")
                if pair_details:
                    sample_text += f" ({', '.join(pair_details)})"
                lines.append(sample_text)

            # Report valid lag bins with their sampled distance range
            if empirical is not None:
                min_pair_count = empirical.fit_result.get("min_count", 1)
                valid_lags = (
                    (empirical.counts >= min_pair_count)
                    & np.isfinite(empirical.lags)
                    & np.isfinite(empirical.semivariance)
                )
                pair_label = "pair" if min_pair_count == 1 else "pairs"
                lag_text = (
                    f"  Variogram lag bins: {np.count_nonzero(valid_lags)}/{len(empirical.lags)} valid "
                    f"(minimum {min_pair_count} {pair_label})"
                )
                if empirical.bin_lower_edges is not None and empirical.bin_edges is not None and empirical.lags.size:
                    lag_text += (
                        f", {empirical.bin_lower_edges[0]:.4g} to {empirical.bin_edges[-1]:.4g} {distance_units}"
                    )
                lines.append(lag_text)
            for message in self.fit_diagnostics.get("identifiability", []):
                lines.append(f"  Warning: {message}")

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


def _source_covariance_matrix(
    source_ids: NDArray[Any],
    source_coordinates: NDArray[np.float64] | None,
    component_data: tuple[tuple[ErrorComponent, NDArray[np.float64]], ...],
    indexes: ArrayLike,
    columns: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Calculate covariance among the observations used by one output.

    Supply columns to calculate a rectangular block between two observation groups.
    """

    size = len(source_ids)

    # Select source observations for this output
    selected = np.asarray(indexes, dtype=np.int64).reshape(-1)
    selected_columns = selected if columns is None else np.asarray(columns, dtype=np.int64).reshape(-1)
    if np.any(selected < 0) or np.any(selected >= size):
        raise ValueError("Covariance indexes are outside the selected source observations.")
    if np.any(selected_columns < 0) or np.any(selected_columns >= size):
        raise ValueError("Covariance indexes are outside the selected source observations.")
    # For uncorrelated errors, only the same source ID is correlated with itself
    # Two observations at identical coordinates still have uncorrelated errors if their IDs differ
    covariance = np.zeros((len(selected), len(selected_columns)), dtype=float)
    distances_by_dimensions: dict[tuple[int, ...] | None, NDArray[np.float64]] = {}
    for component, magnitude in component_data:
        if component.correlation is None:
            correlation = source_ids[selected, None] == source_ids[selected_columns][None, :]
        else:
            correlation_model = component.correlation
            if not isinstance(correlation_model, VariogramModel):
                raise AssertionError("A validated error component must contain a VariogramModel.")
            if source_coordinates is None:
                raise AssertionError("A correlated component requires source coordinates.")

            # Some models use only selected coordinates (e.g. horizontal distance without elevation)
            active_dims = correlation_model.active_dims
            selected_coordinates = source_coordinates[selected]
            column_coordinates = source_coordinates[selected_columns]
            if active_dims is not None:
                selected_coordinates = selected_coordinates[:, active_dims]
                column_coordinates = column_coordinates[:, active_dims]

            # Components using the same coordinates can reuse their pairwise distances
            distances = distances_by_dimensions.get(active_dims)
            if distances is None:
                distances = cdist(selected_coordinates, column_coordinates)
                distances_by_dimensions[active_dims] = distances
            correlation = component.predict_correlation(distances)

        # Covariance = first error magnitude × second error magnitude × correlation; independent components add up
        covariance += magnitude[selected, None] * magnitude[selected_columns][None, :] * correlation
    return covariance


def _draw_source_errors(
    size: int,
    source_coordinates: NDArray[np.float64] | None,
    component_data: tuple[tuple[ErrorComponent, NDArray[np.float64]], ...],
    rng: np.random.Generator,
    *,
    backend: Literal["gstools", "gpytorch"] = "gstools",
    random_coordinates: Any | None = None,
    mesh_type: str = "unstructured",
    field_shape: tuple[int, ...] | None = None,
    component_seeds: tuple[int, ...] | None = None,
    indexes: NDArray[np.int64] | None = None,
    inducing_fields: tuple[GPyTorchInducingField | None, ...] | None = None,
) -> NDArray[np.float64]:
    """Draw errors for all observations together, with the model's spatial correlation.

    GSTools can evaluate a regular raster faster from separate X/Y axes. These optional axes only affect the
    random field calculation; covariance uses the original spatial coordinates.
    Component seeds and global indexes let separate chunks draw the same field at their own locations.
    """

    if component_seeds is not None and len(component_seeds) != len(component_data):
        raise ValueError("component_seeds must contain one seed per error component.")
    if inducing_fields is not None and len(inducing_fields) != len(component_data):
        raise ValueError("inducing_fields must contain one entry per error component.")
    if indexes is None:
        indexes = np.arange(size, dtype=np.int64)
    elif np.shape(indexes) != (size,):
        raise ValueError("indexes must contain one position per observation.")

    combined = np.zeros(size, dtype=float)
    for component_index, (component, magnitude) in enumerate(component_data):
        seed = (
            int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            if component_seeds is None
            else component_seeds[component_index]
        )
        if component.correlation is None:
            unit_field = _indexed_standard_normal(seed, indexes)
        else:
            correlation_model = component.correlation
            if not isinstance(correlation_model, VariogramModel):
                raise AssertionError("A validated error component must contain a VariogramModel.")
            if source_coordinates is None:
                raise AssertionError("A correlated component requires source coordinates.")

            # Draw the field using only the coordinates that this component depends on
            active_dims = correlation_model.active_dims
            coordinates = source_coordinates if active_dims is None else source_coordinates[:, active_dims]
            variogram = Variogram(
                lags=np.empty(0),
                semivariance=np.empty(0),
                counts=np.empty(0, dtype=np.int64),
                model=correlation_model,
            )
            inducing_field = None if inducing_fields is None else inducing_fields[component_index]
            if backend == "gpytorch" and inducing_field is not None:
                from geoutils.uncertainty.random_field import _interpolate_inducing_field

                unit_field = _interpolate_inducing_field(inducing_field, coordinates)
            elif backend == "gstools":
                gstools_variogram = variogram.to_gstools(dim=coordinates.shape[1])
                gstools = import_optional("gstools")

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
                torch = import_optional("torch")
                gpytorch = import_optional("gpytorch")
                gpytorch_variogram = variogram.to_gpytorch(
                    active_dims=tuple(range(coordinates.shape[1])),
                    trainable=False,
                )
                coordinate_tensor = torch.as_tensor(coordinates.copy(), dtype=torch.float64)
                covariance = gpytorch_variogram.kernel(coordinate_tensor)

                # Uncorrelated errors add variance to each observation (the covariance diagonal)
                if gpytorch_variogram.noise > 0:
                    diagonal = torch.full((size,), gpytorch_variogram.noise, dtype=torch.float64)
                    covariance = covariance.add_diagonal(diagonal)
                distribution = gpytorch.distributions.MultivariateNormal(
                    torch.zeros(size, dtype=torch.float64), covariance
                )

                # Use a local generator so drawing a field does not change PyTorch's global random state
                generator = torch.Generator(device=coordinate_tensor.device)
                generator.manual_seed(seed)
                base_samples = torch.randn(size, dtype=torch.float64, generator=generator)
                with torch.no_grad(), gpytorch.settings.fast_computations(covar_root_decomposition=False):
                    unit_field = distribution.rsample(base_samples=base_samples).detach().cpu().numpy()
            unit_field = unit_field.reshape(-1)
            if unit_field.shape != (size,):
                raise RuntimeError("The covariance library returned a field with the wrong number of values.")

        # Scale this unit-variance field by its error magnitude, then add it to the other components
        combined += magnitude * unit_field
    return combined


############################################
# 7/ MATCHING MODELS TO SOURCE OBSERVATIONS
############################################


def _component_magnitudes(
    structure: ErrorStructure,
    *,
    size: int,
    coordinates: NDArray[np.float64] | None,
    predictors: Mapping[str, NDArray[np.float64]],
) -> tuple[tuple[ErrorComponent, NDArray[np.float64]], ...]:
    """Calculate each component's error magnitude at the selected observations."""

    component_data: list[tuple[ErrorComponent, NDArray[np.float64]]] = []
    for component in structure.components.values():
        if component.correlation is not None and coordinates is None:
            raise ValueError(f"Coordinates are required for correlated component {component.name!r}.")
        predicted = np.asarray(component.predict_magnitude(predictors), dtype=float)

        # Constant magnitudes apply to every observation; variable magnitudes must supply one value for each
        try:
            magnitude = np.broadcast_to(predicted, (size,)).astype(float, copy=True)
        except ValueError as exception:
            raise ValueError(
                f"Component {component.name!r} magnitude shape {predicted.shape} cannot match the source observations."
            ) from exception
        if np.any(~np.isfinite(magnitude)) or np.any(magnitude < 0):
            raise ValueError(f"Component {component.name!r} has non-finite or negative magnitudes.")
        magnitude.setflags(write=False)
        component_data.append((component, magnitude))

    return tuple(component_data)


############################################
# 8/ PREPARE ERRORS AT SOURCE OBSERVATIONS
############################################


def _prepare_source_errors(
    structure: ErrorStructure,
    size: int,
    coordinates: ArrayLike | None = None,
    predictors: Mapping[str, Any] | None = None,
) -> tuple[NDArray[np.float64] | None, tuple[tuple[ErrorComponent, NDArray[np.float64]], ...]]:
    """Check source coordinates and predictors, then calculate component error magnitudes.

    _normalize_coordinates() and _normalize_predictors() check the inputs.
    _component_magnitudes() then calculates each component's error magnitude at those locations.
    """

    if not isinstance(structure, ErrorStructure):
        raise TypeError("structure must be an ErrorStructure.")
    normalized_coordinates = _normalize_coordinates(coordinates, size)
    normalized_predictors = _normalize_predictors(predictors, required=structure.required_predictors, size=size)
    component_data = _component_magnitudes(
        structure,
        size=size,
        coordinates=normalized_coordinates,
        predictors=normalized_predictors,
    )
    return normalized_coordinates, component_data
