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

"""Module for propagation of uncertainty, with a PropagationSummary class generic to numerical/analytical methods."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from geoutils.operators.base import LocalData
from geoutils.operators.interpolator import Interpolator
from geoutils.operators.nodata import NodataHandling
from geoutils.operators.reducer import Reducer
from geoutils.pointcloud.base import PointCloudBase
from geoutils.raster.base import RasterBase
from geoutils.uncertainty.error_structure import (
    ErrorComponent,
    ErrorStructure,
    _draw_source_errors,
    _prepare_source_errors,
    _source_covariance_matrix,
)

############################################
# 1/ PROPAGATION RESULTS
############################################

@dataclass(frozen=True, kw_only=True)
class PropagationSummary:
    """
    Propagation summary that stores propagated uncertainty descriptors, whether obtained analytically or numerically.

    The PropagationSummary class is returned by propagate().
    estimate is the result from the original values; mean and std describe the result after adding source errors.
    With zero-mean errors, the expected result of a weighted sum equals estimate; a nonlinear calculation can shift it.
    samples optionally stores random draws used in numerical propagation. Use interval() for uncertainty bounds or
    quantile() for a percentile.

    :param estimate: Output calculated before adding source errors.
    :param mean: Expected output after adding source errors, estimated from draws for numerical propagation.
    :param std: Standard deviation of the results after including source uncertainty.
    :param error_structure: Error model used for the source observations.
    :param method: "analytical" for exact weighted sums, or "numerical" for repeated random draws (Monte Carlo).
    :param selection: Labels and original estimates for selected samples or quantiles.
    :param samples: Optional output values calculated from random draws of the source errors.
    :param n_samples: Number of random draws requested.
    :param n_valid: Number of finite draws at each output.
    :param metadata: Details about the calculation and any approximation it used.
    """

    estimate: Any
    mean: Any
    std: Any
    error_structure: ErrorStructure
    method: Literal["analytical", "numerical"]
    selection: pd.DataFrame
    output: Mapping[str, Any] = field(default_factory=dict)
    samples: pd.DataFrame | None = None
    quantiles: pd.DataFrame | None = None
    n_samples: int | None = None
    n_success: int | None = None
    n_valid: Any = None
    failures: Mapping[int, str] = field(default_factory=dict)
    resultant_length: Any = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy the input tables and check that they use the same output labels and order."""

        if not isinstance(self.selection, pd.DataFrame) or not self.selection.index.is_unique:
            raise ValueError("selection must be a DataFrame with unique output labels.")
        if {"flat_index", "estimate"}.difference(self.selection.columns):
            raise ValueError("selection must contain flat_index and estimate columns.")

        # Copy the tables so later edits to the input cannot change this summary
        selection = self.selection.copy(deep=True)
        selection.index = pd.Index(selection.index, tupleize_cols=False, name="output")
        object.__setattr__(self, "selection", selection)

        # Sample and quantile columns must refer to the selected outputs in the same order
        if self.samples is not None:
            if not self.samples.columns.equals(selection.index):
                raise ValueError("samples columns must exactly match selection.")
            object.__setattr__(self, "samples", self.samples.copy(deep=True))
        if self.quantiles is not None:
            if not self.quantiles.columns.equals(selection.index):
                raise ValueError("quantiles columns must exactly match selection.")
            object.__setattr__(self, "quantiles", self.quantiles.copy(deep=True))
        object.__setattr__(self, "failures", MappingProxyType(dict(self.failures)))
        object.__setattr__(self, "output", MappingProxyType(dict(self.output)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def bias(self) -> Any:
        """Return the difference between the mean and the result from the original source values."""

        estimate = _numeric_values(self.estimate)
        mean = _numeric_values(self.mean)
        difference = mean - estimate
        period = self.output.get("circular_period")
        if period is not None:
            difference = _wrap_difference(difference, float(period))
        return _wrap_like(self.estimate, difference)

    @property
    def variance(self) -> Any:
        """Return the variance of each output."""

        if self.output.get("circular_period") is not None:
            raise TypeError("Circular output has no ordinary variance; use std or resultant_length.")
        standard_deviation = _numeric_values(self.std)
        return _wrap_like(self.std, standard_deviation**2)

    def quantile(self, probability: float) -> pd.Series:
        """Return a quantile from saved draws or an explicitly known normal marginal.

        :param probability: Fraction below the requested value, from 0 to 1 (e.g. 0.5 for the median).
        :returns: One quantile per output, indexed by the output labels.
        """

        if isinstance(probability, (bool, np.bool_)) or not isinstance(
            probability, (int, float, np.integer, np.floating)
        ):
            raise TypeError("probability must be a scalar number.")
        q = float(probability)
        if not np.isfinite(q) or not 0 <= q <= 1:
            raise ValueError("probability must be finite and between zero and one.")

        # A stored quantile can be returned without keeping all the draws
        if self.quantiles is not None and q in self.quantiles.index:
            return self.quantiles.loc[q].copy()

        # Draws describe the observed output distribution, including nonlinear calculations
        if self.samples is not None:
            values = self.samples.to_numpy(dtype=float, copy=True)
            period = self.output.get("circular_period")
            if period is not None:
                nominal = self.selection["estimate"].to_numpy(dtype=float)
                values = nominal + _wrap_difference(values - nominal, float(period))
            values = np.nanquantile(values, q, axis=0, method="linear")
            return pd.Series(values, index=self.selection.index, dtype=float)

        # Moments alone do not establish a normal distribution
        means = _numeric_values(self.mean).reshape(-1)
        standard_deviations = _numeric_values(self.std).reshape(-1)
        if self.metadata.get("output_distribution") != "normal":
            raise ValueError("Quantiles require saved draws, retained quantiles or an explicit normal output law.")
        values = []
        for position in self.selection["flat_index"]:
            mean = float(means[position])
            std = float(standard_deviations[position])
            values.append(mean if std == 0 else float(norm.ppf(q, loc=mean, scale=std)))
        return pd.Series(values, index=self.selection.index, dtype=float)

    def interval(self, coverage: float = 0.95) -> pd.DataFrame:
        """Return uncertainty bounds with equal probability below and above them.

        :param coverage: Probability inside the interval (0.95 gives the 2.5th to 97.5th percentiles).
        :returns: A table with lower/upper bounds for each output.
        """

        if not isinstance(coverage, (int, float, np.integer, np.floating)) or isinstance(coverage, (bool, np.bool_)):
            raise TypeError("coverage must be a scalar number.")
        coverage_value = float(coverage)
        if not np.isfinite(coverage_value) or not 0 < coverage_value < 1:
            raise ValueError("coverage must be finite and strictly between zero and one.")
        tail = (1 - coverage_value) / 2
        result = pd.DataFrame({"lower": self.quantile(tail), "upper": self.quantile(1 - tail)})
        result.attrs["coverage"] = coverage_value
        result.attrs["kind"] = "equal_tail"
        result.attrs["coordinate_convention"] = (
            "unwrapped_about_estimate" if self.output.get("circular_period") is not None else "linear"
        )
        return result

    def marginal(self, at: Hashable) -> pd.Series | Mapping[str, Any]:
        """Return the uncertainty distribution for one output.

        :param at: Output label listed in selection.
        :returns: Saved draws or an explicit description of this output's distribution.
        """

        if at not in self.selection.index:
            raise ValueError(f"Output {at!r} is not present in selection.")
        if self.samples is not None:
            result = self.samples[at].copy()
            result.name = at
            return result
        if self.metadata.get("output_distribution") != "normal":
            raise ValueError("This output has no saved draws or known marginal distribution.")
        position = int(self.selection.loc[at, "flat_index"])
        mean = float(_numeric_values(self.mean).reshape(-1)[position])
        std = float(_numeric_values(self.std).reshape(-1)[position])
        return MappingProxyType({"family": "normal", "mean": mean, "std": std, "status": "exact"})


def _numeric_values(value: Any) -> NDArray[np.float64]:
    """Read numeric values from an output."""

    if isinstance(value, (RasterBase, PointCloudBase)):
        value = value.data
    if isinstance(value, (pd.Series, pd.DataFrame)):
        value = value.to_numpy()
    return np.ma.asarray(value, dtype=float).filled(np.nan)


def _wrap_like(template: Any, values: NDArray[Any]) -> Any:
    """Wrap calculated values into the same data structure as the template."""

    if isinstance(template, (RasterBase, PointCloudBase)):
        return template.copy(new_array=np.ma.masked_invalid(values))
    if isinstance(template, pd.Series):
        return pd.Series(values, index=template.index, name=template.name)
    if isinstance(template, pd.DataFrame):
        return pd.DataFrame(values, index=template.index, columns=template.columns)
    if isinstance(template, np.ma.MaskedArray):
        return np.ma.masked_array(values, mask=np.ma.getmaskarray(template) | ~np.isfinite(values))
    if isinstance(template, np.ndarray):
        return np.asarray(values)
    return float(np.asarray(values).reshape(()))


def _wrap_difference(difference: NDArray[np.float64], period: float) -> NDArray[np.float64]:
    """Measure an angular difference around zero using one chosen period."""

    return (difference + period / 2) % period - period / 2


############################################
# 2/ HELPERS FOR SOURCE DATA AND OUTPUT LABELS
############################################


def _source_sort_key(source_id: Hashable) -> tuple[str, str, str]:
    """Return a repeatable sort key for source IDs whose types cannot be compared directly."""

    identity_type = type(source_id)
    return identity_type.__module__, identity_type.__qualname__, repr(source_id)


def _hashable_source_id(value: Any) -> Hashable:
    """Convert a NumPy scalar to a Python value and check that it can be used as a dictionary key."""

    source_id = value.item() if isinstance(value, np.generic) else value
    try:
        hash(source_id)
    except TypeError as exception:
        raise TypeError("Every LocalData source_id must be hashable.") from exception
    return cast(Hashable, source_id)


def _evaluate_batch(
    operator: Interpolator | Reducer, data: Sequence[LocalData], handling: NodataHandling
) -> NDArray[np.float64]:
    """Call predict_batch() or reduce_batch() for several groups of source data."""

    if isinstance(operator, Interpolator):
        return np.asarray(operator.predict_batch(data, nodata_propagation=handling), dtype=float)
    return np.asarray(operator.reduce_batch(data, nodata_propagation=handling), dtype=float)


def _collect_source_support(
    data: Sequence[LocalData],
) -> tuple[NDArray[Any], NDArray[np.float64] | None, dict[Hashable, int]]:
    """Collect each valid observation once and check repeated IDs for matching values and coordinates."""

    # Several output locations can use the same source observation; identify it by its source ID
    records: dict[Hashable, tuple[float, NDArray[np.float64] | None]] = {}
    any_coordinates = False
    missing_coordinates = False
    for local in data:
        effective = local._valid_values()
        for position in np.flatnonzero(effective):
            source_id = _hashable_source_id(local.source_ids[position])
            coordinate = None if local.coordinates is None else np.asarray(local.coordinates[position], dtype=float)
            any_coordinates |= coordinate is not None
            missing_coordinates |= coordinate is None
            value = float(local.values[position])
            previous = records.get(source_id)
            if previous is None:
                records[source_id] = value, None if coordinate is None else coordinate.copy()
                continue

            # One source may be used by many targets, but its value and coordinates must agree each time
            previous_value, previous_coordinate = previous
            if not np.isclose(previous_value, value, rtol=0, atol=0, equal_nan=True):
                raise ValueError(f"Repeated source_id {source_id!r} has inconsistent source values.")
            if (previous_coordinate is None) != (coordinate is None):
                raise ValueError(f"Repeated source_id {source_id!r} has inconsistent coordinate availability.")
            if coordinate is not None:
                if previous_coordinate is None:
                    raise AssertionError("Coordinate availability was validated immediately above.")
                coordinates_equal = np.array_equal(previous_coordinate, coordinate)
            else:
                coordinates_equal = True
            if not coordinates_equal:
                raise ValueError(f"Repeated source_id {source_id!r} has inconsistent coordinates.")

    # Sort comparable IDs naturally; mixed or custom IDs use a repeatable type/representation order
    try:
        ordered_ids = sorted(records, key=lambda source_id: cast(Any, source_id))
    except TypeError:
        ordered_ids = sorted(records, key=_source_sort_key)
    source_ids = np.empty(len(ordered_ids), dtype=object)
    source_ids[:] = ordered_ids
    positions = {source_id: index for index, source_id in enumerate(ordered_ids)}
    coordinates: NDArray[np.float64] | None = None

    # Spatial correlation requires coordinates for every source observation
    if any_coordinates and not missing_coordinates:
        coordinates = np.vstack([cast(NDArray[np.float64], records[source_id][1]) for source_id in ordered_ids])
    return source_ids, coordinates, positions


def _align_predictors(
    predictors: Mapping[str, Any] | None,
    *,
    source_ids: NDArray[Any],
) -> Mapping[str, Any] | None:
    """Arrange predictors in source ID order, accepting scalars, dictionaries or already ordered arrays."""

    if predictors is None:
        return None
    aligned: dict[str, Any] = {}
    for name, predictor in predictors.items():
        # A dictionary uses source IDs as keys, so its insertion order does not matter
        if isinstance(predictor, Mapping):
            missing = [source_id for source_id in source_ids if source_id not in predictor]
            if missing:
                raise ValueError(f"Predictor {name!r} has no values for source_ids {missing!r}.")
            aligned[name] = np.asarray([predictor[source_id] for source_id in source_ids], dtype=float)
            continue

        # Arrays must already follow the sorted source IDs; a scalar applies to every observation
        values = np.asarray(predictor, dtype=float)
        if values.ndim > 0 and values.shape != (len(source_ids),):
            raise ValueError(
                f"Predictor {name!r} must be scalar, keyed by source_id, or aligned with sorted source_ids."
            )
        aligned[name] = values
    return aligned


def _output_selection(labels: pd.Index, estimate: NDArray[np.float64]) -> pd.DataFrame:
    """Pair each output label with its array position and calculated value."""

    return pd.DataFrame(
        {
            "flat_index": pd.Series(np.arange(len(labels), dtype=np.int64), index=labels),
            "estimate": pd.Series(estimate, index=labels, dtype=float),
        },
        index=labels,
    )


def _normalize_output_labels(labels: Sequence[Hashable] | None, count: int, *, scalar: bool) -> pd.Index:
    """Create unique output labels for stored samples and quantiles."""

    if labels is None:
        values: list[Hashable] = ["value"] if scalar else list(range(count))
    else:
        values = list(labels)
    index = pd.Index(values, tupleize_cols=False, name="output")
    if len(index) != count or not index.is_unique:
        raise ValueError("output_labels must contain one unique label per target.")
    return index


############################################
# 3/ EXACT PROPAGATION FOR LOCAL OPERATORS
############################################


def _affine_rows(
    operator: Interpolator | Reducer,
    data: Sequence[LocalData],
    *,
    handling: NodataHandling,
    source_positions: Mapping[Hashable, int],
    estimate: NDArray[np.float64],
) -> list[tuple[NDArray[np.int64], NDArray[np.float64]] | None] | None:
    """Collect source indexes and weights for each output, or return None when the method is not a weighted sum."""

    rows: list[tuple[NDArray[np.int64], NDArray[np.float64]] | None] = []
    for local, nominal in zip(data, estimate):
        # Missing values must affect the weights in the same way as the original interpolation/reduction
        prepared, prepared_valid = local._prepare(
            accepts_sample_weights=operator.accepts_sample_weights,
            accepts_support_weights=isinstance(operator, Reducer) and operator.accepts_support_weights,
            minimum_inputs=operator.minimum_inputs,
            nodata_handling=handling,
        )
        if prepared is None or not np.isfinite(nominal):
            rows.append(None)
            continue

        # Without coefficients, we need random draws to find how source errors affect the result
        coefficients = operator.coefficients(prepared)
        if coefficients is None:
            return None
        if len(coefficients.weights) != len(prepared.values):
            raise ValueError("Linear coefficients must contain one weight per prepared LocalData value.")
        if handling == "propagate" and np.any((~prepared_valid) & (coefficients.weights != 0)):
            rows.append(None)
            continue

        # Add weights when one source appears more than once, since every occurrence has the same error
        accumulated: dict[int, float] = {}
        for source_id_value, weight, valid in zip(prepared.source_ids, coefficients.weights, prepared_valid):
            if not valid or weight == 0:
                continue
            source_id = _hashable_source_id(source_id_value)
            position = source_positions[source_id]
            accumulated[position] = accumulated.get(position, 0.0) + float(weight)
        indexes = np.fromiter(accumulated.keys(), dtype=np.int64)
        weights = np.fromiter(accumulated.values(), dtype=float)
        rows.append((indexes, weights))
    return rows


def _propagate_local_operator_analytically(
    *,
    rows: Sequence[tuple[NDArray[np.int64], NDArray[np.float64]] | None],
    estimate: NDArray[np.float64],
    source_ids: NDArray[Any],
    coordinates: NDArray[np.float64] | None,
    component_data: tuple[tuple[ErrorComponent, NDArray[np.float64]], ...],
    selection: pd.DataFrame,
    error_structure: ErrorStructure,
    scalar: bool,
) -> PropagationSummary:
    """Calculate each output error from its source weights and observation covariance."""

    output_count = len(rows)
    mean = estimate.copy()
    variance = np.full(output_count, np.nan, dtype=float)
    for position, row in enumerate(rows):
        if row is None:
            continue
        indexes, weights = row

        # For a weighted sum, each pair contributes weight_i × covariance_ij × weight_j to the variance
        covariance = _source_covariance_matrix(source_ids, coordinates, component_data, indexes)
        variance[position] = max(float(weights @ covariance @ weights), 0.0)

    # Standard deviation has the output values units
    standard_deviation = np.sqrt(variance)
    return PropagationSummary(
        estimate=float(estimate[0]) if scalar else estimate,
        mean=float(mean[0]) if scalar else mean,
        std=float(standard_deviation[0]) if scalar else standard_deviation,
        error_structure=error_structure,
        method="analytical",
        selection=selection,
        metadata={"approximation": "exact", "output_distribution": "normal"},
    )


def _empty_local_operator_summary(
    *,
    estimate: NDArray[np.float64],
    selection: pd.DataFrame,
    error_structure: ErrorStructure,
    scalar: bool,
) -> PropagationSummary:
    """Return undefined uncertainty for targets with no finite source observations."""

    undefined = np.full(len(estimate), np.nan, dtype=float)
    return PropagationSummary(
        estimate=float(estimate[0]) if scalar else estimate,
        mean=float(undefined[0]) if scalar else undefined,
        std=float(undefined[0]) if scalar else undefined,
        error_structure=error_structure,
        method="analytical",
        selection=selection,
        metadata={"approximation": "exact", "output_distribution": "normal"},
    )


############################################
# 4/ NUMERICAL PROPAGATION FOR LOCAL OPERATORS
############################################


def _perturb_local_data(
    data: Sequence[LocalData],
    error: NDArray[np.float64],
    source_positions: Mapping[Hashable, int],
) -> list[LocalData]:
    """Add one random error to each valid source value without changing its other information."""

    perturbed: list[LocalData] = []
    for local in data:
        values = np.asanyarray(local.values).astype(float, copy=True)
        effective = local._valid_values()
        for position in np.flatnonzero(effective):
            # Repeated source IDs receive the same error, even when they appear in different neighbourhoods
            source_id = _hashable_source_id(local.source_ids[position])
            values[position] += error[source_positions[source_id]]
        perturbed.append(replace(local, values=values))
    return perturbed


def _propagate_local_operator_numerically(
    operator: Interpolator | Reducer,
    data: Sequence[LocalData],
    *,
    estimate: NDArray[np.float64],
    source_ids: NDArray[Any],
    coordinates: NDArray[np.float64] | None,
    component_data: tuple[tuple[ErrorComponent, NDArray[np.float64]], ...],
    source_positions: Mapping[Hashable, int],
    selection: pd.DataFrame,
    error_structure: ErrorStructure,
    scalar: bool,
    handling: NodataHandling,
    n_samples: int,
    random_state: int | np.random.Generator | None,
    return_samples: bool,
    max_sample_bytes: int,
    quantiles: Sequence[float],
    on_error: Literal["raise", "warn"],
) -> PropagationSummary:
    """Repeat the interpolation/reduction with random source errors and summarize the resulting spread."""

    def calculate(error: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply one shared source draw to every local target before evaluating the operator."""

        perturbed = _perturb_local_data(data, error, source_positions)
        values = _evaluate_batch(operator, perturbed, handling)
        values[~np.isfinite(estimate)] = np.nan
        return values[0] if scalar else values

    def draw_error(generator: np.random.Generator) -> NDArray[np.float64]:
        """Draw one error for each distinct source observation."""

        return _draw_source_errors(len(source_ids), coordinates, component_data, generator)

    # One source draw is used by all targets, including targets with overlapping observations
    nominal = float(estimate[0]) if scalar else estimate
    return simulate(
        estimate=nominal,
        draw_error=draw_error,
        calculate=calculate,
        error_structure=error_structure,
        selection=selection,
        n_samples=int(n_samples),
        random_state=random_state,
        return_samples=return_samples,
        max_sample_bytes=max_sample_bytes,
        quantiles=quantiles,
        on_error=on_error,
    )


############################################
# 5/ SPATIAL METHOD PROPAGATION
############################################


class _SpatialMoment:
    """Evaluate one uncertainty quantity using the same selected observations as the original operator."""

    _requires_local_evaluation = True
    _uses_error_covariance = False

    def __init__(self, operator: Interpolator | Reducer, quantity: str, options: dict[str, Any]) -> None:
        """Copy neighborhood and missing-data options while saving the estimator and requested quantity."""

        self._wrapped_operator = operator
        self.quantity = quantity
        self.options = options
        self.error_structure = operator.error_structure
        self.default_neighborhood = operator.default_neighborhood
        self.default_nodata_propagation = operator.default_nodata_propagation
        self.minimum_inputs = operator.minimum_inputs
        self.accepts_sample_weights = operator.accepts_sample_weights
        if isinstance(operator, Interpolator):
            self.interpolation_order = operator.interpolation_order
        else:
            self.accepts_support_weights = operator.accepts_support_weights

    def evaluate(self, data: LocalData, *, nodata_propagation: Any = None) -> float:
        """Calculate a local moment without returning or accumulating source observations."""

        summary = propagate(
            self._wrapped_operator,
            data,
            self.error_structure,
            nodata_propagation=nodata_propagation,
            **self.options,
        )
        value = getattr(summary, self.quantity)
        return float(value) if value is not None else float("nan")


class _MomentInterpolator(_SpatialMoment, Interpolator):
    """Interpolate output uncertainty, including precomputed regular-grid coefficients."""

    def _with_regular_coefficients(self, operator: Interpolator) -> _MomentInterpolator:
        """Apply this moment to the actual nearest, bilinear, or spline calculation on a raster."""

        from geoutils.operators.weighting import _with_error_structure

        return type(self)(_with_error_structure(operator, self.error_structure), self.quantity, self.options)


class _MomentReducer(_SpatialMoment, Reducer):
    """Reduce output uncertainty over the original point neighborhood or cell footprint."""


class _SpatialInputs(_SpatialMoment):
    """Record selected observations for saved samples, quantiles, or selected outputs."""

    def evaluate(self, data: LocalData, *, nodata_propagation: Any = None) -> float:
        """Return a record index so output construction also arranges records in destination order."""

        records = self.options["records"]
        records.append((self._wrapped_operator, data, nodata_propagation))
        return float(len(records) - 1)


class _InputInterpolator(_SpatialInputs, _MomentInterpolator):
    """Record interpolation inputs with the same coefficients and output positions as their values."""


class _InputReducer(_SpatialInputs, _MomentReducer):
    """Record reduction inputs with the same neighborhoods and output positions as their values."""


############################################
# RUN THE SPATIAL METHOD WITH THE SAME INPUT SELECTION
############################################


def _propagate_spatial_method(
    operation: Callable[..., Any],
    error_structure: ErrorStructure,
    operation_kwargs: dict[str, Any],
    options: dict[str, Any],
) -> PropagationSummary:
    """Run a spatial estimator and its local moments through the same neighborhood calculation.

    Each output depends only on its selected source observations. Moment operators calculate the marginal
    uncertainty inside each chunk, so they do not retain LocalData for the complete raster or point cloud.
    _MomentInterpolator and _MomentReducer call propagate() on each selected neighborhood. Requested samples or
    quantiles use _InputInterpolator or _InputReducer to collect groups in destination order.
    """

    from geoutils._config import config
    from geoutils.interface.gridding import _resolve_gridding_operator
    from geoutils.operators.interpolator import Kriging, _regular_interpolation_method, _resolve_interpolator
    from geoutils.operators.neighbours import _build_kriging_grid_neighbours
    from geoutils.operators.reducer import Mean
    from geoutils.operators.weighting import _with_error_structure
    from geoutils.raster.transformation import _resolve_reprojection_operator

    # 1/ Resolve aliases before replacing the estimator; the public method still owns validation and output layout
    source = operation.__self__
    name = operation.__name__
    kwargs = dict(operation_kwargs)
    if name == "krige":
        # The convenience methods choose a Kriging operator before delegating to reproject() or grid()
        variogram = kwargs.pop("variogram")
        kriging_options = {
            key: kwargs.pop(key) for key in ("backend", "max_overlap", "exact", "pseudo_inverse") if key in kwargs
        }
        kriging = Kriging(variogram, **kriging_options)
        kwargs["resampling"] = kriging
        if isinstance(source, RasterBase):
            target = kwargs.setdefault("ref", source)
            kriging.default_neighborhood = _build_kriging_grid_neighbours(
                kriging,
                source.transform,
                aligned_targets=target.transform == source.transform,
            )
            kwargs.update(dtype=np.float64, nodata_propagation="ignore")
            operation, name = source.reproject, "reproject"
        else:
            kwargs.update(dist_nodata_pixel=0, nodata_handling="ignore")
            operation, name = source.grid, "grid"
    if kwargs.get("inplace") or kwargs.get("return_interpolator"):
        raise ValueError("Spatial propagation requires returned values, without inplace or returned interpolators.")
    if kwargs.get("mp_config") is not None:
        raise ValueError("Use Dask for chunked spatial propagation; multiprocessing needs separate output files.")
    if name in {"interp_points", "interp_at_points", "resample_at_points"}:
        operation = source.resample_at_points
        parameter = "method"
        selected_method = kwargs.get(parameter, config["interpolation_method"])
        operator = selected_method if isinstance(selected_method, Reducer) else _resolve_interpolator(selected_method)
    elif name in {"reduce_points", "reduce_at_points"}:
        operation = source.reduce_at_points
        parameter = "reducer_function"
        operator = kwargs.get(parameter, Mean())
        if not isinstance(operator, Reducer):
            raise TypeError("Spatial propagation requires a Reducer for reduce_at_points().")
        if kwargs.get("window") is None:
            kwargs["window"] = 1
    elif name == "grid":
        parameter = "resampling"
        operator = _resolve_gridding_operator(
            kwargs.get(parameter, "nearest"), distance_power=kwargs.get("distance_power", 2)
        )
    elif name == "reproject":
        parameter = "resampling"
        operator = _resolve_reprojection_operator(kwargs.get(parameter, config["reprojection_method"]))
    else:
        raise TypeError("Spatial propagation supports grid(), reproject(), and the point resampling methods.")

    # 2/ Give all three calculations the same model, including any covariance used by fitting
    operator = _with_error_structure(operator, error_structure)
    operator._error_predictors = options.get("predictors")
    kwargs.pop("error_structure", None)
    kwargs[parameter] = operator

    # 3/ Saved draws collect source groups; ordinary marginal propagation stays local to each chunk
    collect_inputs = options.get("return_samples") or options.get("quantiles") or options.get("at") is not None
    if collect_inputs:
        if getattr(source, "_chunks", None) is not None or getattr(source, "_is_dask", False):
            raise ValueError(
                "Spatial samples and quantiles require eager inputs; marginal propagation supports chunks."
            )
        records: list[Any] = []
        record_type = _InputInterpolator if isinstance(operator, Interpolator) else _InputReducer
        kwargs[parameter] = record_type(operator, "inputs", {"records": records})
        if name in {"grid", "reproject"}:
            kwargs["nodata"] = np.nan
        if name == "reproject":
            kwargs["dtype"] = np.float64
        layout = operation(**kwargs)
        indexes = _numeric_values(layout)
        empty = LocalData(np.empty(0), np.empty(0, dtype=bool), np.empty(0, dtype=int))
        inputs = [records[int(index)][1] if np.isfinite(index) else empty for index in indexes.ravel()]
        prepared_operator, _, handling = records[0] if records else (operator, empty, None)
        summary = propagate(prepared_operator, inputs, error_structure, nodata_propagation=handling, **options)
        moments = {
            key: _wrap_like(layout, np.asarray(getattr(summary, key)).reshape(indexes.shape))
            for key in ("estimate", "mean", "std")
        }
        return replace(summary, **moments)

    # 4/ Calculate the estimate and marginal moments on the same spatial layout
    estimate = operation(**kwargs)
    # Error magnitudes and means need floating-point output even when the original raster stores integers
    if name in {"grid", "reproject"}:
        kwargs["nodata"] = np.nan
    if name == "reproject":
        kwargs["dtype"] = np.float64
    moment_type = _MomentInterpolator if isinstance(operator, Interpolator) else _MomentReducer
    kwargs[parameter] = moment_type(operator, "mean", options)
    mean = operation(**kwargs)
    kwargs[parameter] = moment_type(operator, "std", options)
    std = operation(**kwargs)

    # 5/ Record the propagation method and valid draw counts
    analytical = type(operator).coefficients not in (Interpolator.coefficients, Reducer.coefficients)
    if isinstance(source, RasterBase) and isinstance(operator, Interpolator):
        analytical |= _regular_interpolation_method(operator) in {"nearest", "linear"}
    method: Literal["analytical", "numerical"] = (
        "numerical" if options["method"] == "numerical" or not analytical else "analytical"
    )
    n_valid = None
    if method == "numerical":
        kwargs[parameter] = moment_type(operator, "n_valid", options)
        n_valid = operation(**kwargs)

    # 6/ Small eager outputs can expose labelled intervals without forcing any lazy result to compute
    selection = pd.DataFrame(columns=["flat_index", "estimate"])
    raw_estimate = getattr(estimate, "data", estimate)
    if not hasattr(raw_estimate, "compute"):
        values = _numeric_values(estimate).reshape(-1)
        if len(values) <= 256:
            selection = pd.DataFrame({"flat_index": np.arange(len(values)), "estimate": values})
    return PropagationSummary(
        estimate=estimate,
        mean=mean,
        std=std,
        error_structure=error_structure,
        method=method,
        selection=selection,
        metadata={
            "operation": name,
            "local_marginals": True,
            "output_distribution": "normal" if method == "analytical" else "unknown",
        },
        n_samples=options["n_samples"] if method == "numerical" else None,
        n_valid=n_valid,
    )


#######################
# 6/ PARENT FUNCTIONS
#######################


def simulate(
    *,
    estimate: Any,
    draw_error: Callable[[np.random.Generator], Any],
    calculate: Callable[[Any], Any],
    error_structure: ErrorStructure,
    selection: pd.DataFrame,
    n_samples: int,
    random_state: int | np.random.Generator | None = None,
    output: Mapping[str, Any] | None = None,
    return_samples: bool = False,
    quantiles: Sequence[float] = (),
    max_sample_bytes: int = 268_435_456,
    on_error: Literal["raise", "warn"] = "raise",
    metadata: Mapping[str, object] | None = None,
) -> PropagationSummary:
    """
    Simulate errors numerically for uncertainty propagation into marginal statistics.

    This function estimates mean/variance by accumulating statistics per batch of simulations, without storing
    individual results at once, except if quantiles are requested.

    :param estimate: Calculation performed on the original data.
    :param draw_error: Return one complete draw of errors from the initial source.
    :param calculate: Apply one complete calculation to that draw.
    :param error_structure: Initial error model used for the draws.
    :param selection: Output keys and flat positions selected for samples or quantiles.
    :param n_samples: Number of requested realizations, at least two.
    :param random_state: Seed or generator controlling source draws.
    :param output: Optional circular period for angular outputs.
    :param return_samples: Whether to save every selected realization.
    :param quantiles: Probabilities of exact empirical quantiles to save.
    :param max_sample_bytes: Maximum temporary memory used for selected realizations.
    :param on_error: Raise a calculation error or warn and count its draw as failed.
    :param metadata: Description of the calculation and its assumptions.
    :returns: Nominal and propagated results with requested selected information.
    """

    # 1/ Check user input
    if isinstance(n_samples, (bool, np.bool_)) or not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("n_samples must be an integer of at least two.")
    if on_error not in {"raise", "warn"}:
        raise ValueError("on_error must be 'raise' or 'warn'.")
    if max_sample_bytes < 1:
        raise ValueError("max_sample_bytes must be positive.")
    if not isinstance(selection, pd.DataFrame) or not {"flat_index", "estimate"}.issubset(selection.columns):
        raise ValueError("selection must contain flat_index and estimate columns.")
    if not selection.index.is_unique:
        raise ValueError("selection keys must be unique.")

    # 2/ Validate selected positions and RAM budget before drawing a random field
    nominal = np.asarray(_numeric_values(estimate), dtype=float)
    shape = nominal.shape
    positions = selection["flat_index"].to_numpy(dtype=np.int64, copy=True)
    if np.any(positions < 0) or np.any(positions >= nominal.size):
        raise ValueError("selection contains a position outside the output.")
    probabilities = tuple(float(value) for value in quantiles)
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("quantiles must be finite probabilities between zero and one.")
    if len(set(probabilities)) != len(probabilities):
        raise ValueError("quantiles must be distinct.")
    keep_buffer = return_samples or bool(probabilities)
    required_bytes = int(n_samples) * len(positions) * np.dtype(np.float64).itemsize
    if keep_buffer and required_bytes > max_sample_bytes:
        raise ValueError(f"Selected sample buffer needs {required_bytes} bytes, exceeding max_sample_bytes.")
    samples_buffer = np.full((int(n_samples), len(positions)), np.nan) if keep_buffer else None
    output_info = dict(output or {})
    period = output_info.get("circular_period")
    if period is not None and (not np.isfinite(period) or period <= 0):
        raise ValueError("A circular output requires a finite positive period.")

    # 3/ We allocate running moments without storing complete draws
    count = np.zeros(shape, dtype=np.int64)
    mean = np.zeros(shape, dtype=float) if period is None else None
    moment = np.zeros(shape, dtype=float) if period is None else None
    sine = np.zeros(shape, dtype=float) if period is not None else None
    cosine = np.zeros(shape, dtype=float) if period is not None else None
    failures: dict[int, str] = {}
    n_success = 0
    generator = np.random.default_rng(random_state)

    # 4/ We draw random source errors and update each output by accumulation
    for number in range(1, int(n_samples) + 1):
        try:
            values = np.asarray(_numeric_values(calculate(draw_error(generator))), dtype=float)
            if values.shape != shape:
                raise ValueError("A simulation changed the output shape.")
            if not np.any(np.isfinite(values)):
                raise ValueError("The calculation returned no finite values.")
        except Exception as exception:
            if on_error == "raise":
                raise
            failures[number] = str(exception)
            warnings.warn(f"Simulation {number} of {n_samples} failed and was skipped: {exception}", stacklevel=2)
            continue

        valid = np.isfinite(values)
        count[valid] += 1
        if period is None:
            assert mean is not None and moment is not None

            # Update the mean and sum of squared deviations without storing full draws
            difference = values[valid] - mean[valid]
            mean[valid] += difference / count[valid]
            moment[valid] += difference * (values[valid] - mean[valid])
        else:
            assert sine is not None and cosine is not None

            # Sum unit-circle components so values near zero and one period average together
            angles = values[valid] * (2 * np.pi / period)
            sine[valid] += np.sin(angles)
            cosine[valid] += np.cos(angles)
        n_success += 1

        if samples_buffer is not None:
            samples_buffer[number - 1] = values.reshape(-1)[positions]

    if n_success < 2:
        raise RuntimeError(f"Only {n_success} of {n_samples} simulations succeeded; at least two are required.")

    # 5/ Angular means and spreads use the circular resultant; ordinary outputs use sample variance
    std = np.full(shape, np.nan, dtype=float)
    resultant: Any = None
    if period is None:
        assert mean is not None and moment is not None
        mean[count == 0] = np.nan
        std[count > 1] = np.sqrt(np.maximum(moment[count > 1], 0) / (count[count > 1] - 1))
    else:
        assert sine is not None and cosine is not None
        length = np.divide(np.hypot(sine, cosine), count, out=np.full(shape, np.nan), where=count > 0)
        length = np.minimum(length, 1.0)
        mean = np.asarray(np.mod(np.arctan2(sine, cosine), 2 * np.pi) * period / (2 * np.pi))
        undefined = (count > 0) & (length <= 32 * np.finfo(float).eps)
        mean[(count == 0) | undefined] = np.nan
        regular = (count > 1) & ~undefined
        std[regular] = period / (2 * np.pi) * np.sqrt(-2 * np.log(length[regular]))
        std[(count > 1) & undefined] = np.inf
        resultant = _wrap_like(estimate, length)

    # 6/ We use one selected buffer for requested draws and empirical quantiles, then release it
    samples: pd.DataFrame | None = None
    quantile_table: pd.DataFrame | None = None
    if samples_buffer is not None:
        if return_samples:
            samples = pd.DataFrame(
                samples_buffer, index=pd.RangeIndex(1, int(n_samples) + 1, name="simulation"), columns=selection.index
            )
        if probabilities:
            values = samples_buffer
            if period is not None:
                # Measure angular quantiles around the original estimate to avoid a wraparound jump
                nominal_selected = selection["estimate"].to_numpy(dtype=float, copy=True)
                values = nominal_selected + _wrap_difference(values - nominal_selected, period)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
                calculated = np.nanquantile(values, probabilities, axis=0, method="linear")
            quantile_table = pd.DataFrame(
                calculated, index=pd.Index(probabilities, name="probability"), columns=selection.index
            )

    # 7/ Return moments in the original output layout with draw counts and selected results
    details = dict(metadata or {})
    details.setdefault("approximation", "monte_carlo")
    n_valid = int(count.reshape(-1)[0]) if nominal.ndim == 0 else _wrap_like(estimate, count)
    return PropagationSummary(
        estimate=estimate,
        mean=_wrap_like(estimate, mean),
        std=_wrap_like(estimate, std),
        error_structure=error_structure,
        method="numerical",
        selection=selection,
        output=output_info,
        samples=samples,
        quantiles=quantile_table,
        n_samples=int(n_samples),
        n_success=n_success,
        n_valid=n_valid,
        failures=failures,
        resultant_length=resultant,
        metadata=details,
    )


def propagate(
    operator: Interpolator | Reducer | Callable[..., Any],
    data: LocalData | Sequence[LocalData] | None = None,
    error_structure: ErrorStructure | None = None,
    *,
    operation_kwargs: Mapping[str, Any] | None = None,
    predictors: Mapping[str, Any] | None = None,
    method: Literal["auto", "analytical", "numerical"] = "auto",
    nodata_propagation: NodataHandling | None = None,
    nominal_estimate: ArrayLike | float | None = None,
    output_labels: Sequence[Hashable] | None = None,
    at: Literal["all"] | Sequence[Hashable] | None = None,
    n_samples: int = 1_000,
    random_state: int | np.random.Generator | None = None,
    return_samples: bool = False,
    max_sample_bytes: int = 268_435_456,
    on_error: Literal["raise", "warn"] = "raise",
    quantiles: Sequence[float] = (),
) -> PropagationSummary:
    """
    Use observation errors for fitting and propagate them through an operator or spatial operation.

    For a weighted sum, _affine_rows() collects the weights and _propagate_local_operator_analytically() calculates
    the resulting mean/variance directly. Other methods use _propagate_local_operator_numerically(): we draw source
    errors, add them to the source values and repeat the interpolation/reduction to measure its spread.

    A repeated source ID receives the same error everywhere it is used. Different IDs represent separate
    observations, even at identical coordinates.

    For a spatial operation, pass a bound method and its arguments as operation_kwargs, for example
    propagate(raster.reproject, error_structure=errors, operation_kwargs={"ref": reference, "resampling": Mean()}).
    The estimate, mean and std then have the operation's raster or point layout. Marginal uncertainty can remain
    lazy with Dask; selected samples currently require eager inputs. Calling the spatial method directly
    with error_structure uses errors only for fitting and returns its ordinary result.

    :param operator: Interpolator, Reducer, or supported bound spatial method.
    :param data: Source data for one output or an ordered sequence of outputs. Omit for a bound spatial method.
    :param error_structure: Source uncertainty model, before it is matched to these observations.
    :param operation_kwargs: Arguments to a bound grid(), reproject(), krige(), or point resampling method.
    :param predictors: Values used to calculate error magnitudes, given as scalars, dictionaries keyed by source ID,
        or arrays in sorted source ID order.
    :param method: Choose automatically between an exact weighted calculation and repeated random draws, or request
        one explicitly.
    :param nodata_propagation: Optional replacement for the Interpolator or Reducer's rule for missing values.
    :param nominal_estimate: Optional results already calculated from the original source values.
    :param output_labels: Unique labels for stored samples and quantiles.
    :param at: Output labels selected for samples or quantiles. Small results are selected by default.
    :param n_samples: Number of random draws for numerical propagation.
    :param random_state: Seed or generator for numerical propagation.
    :param return_samples: Whether to return the result of each random draw. This selects numerical propagation in
        automatic mode and cannot be combined with method="analytical".
    :param max_sample_bytes: Largest selected draw buffer for exact samples or empirical quantiles.
    :returns: A PropagationSummary with the original results, their mean and standard deviation after adding source
        errors, and any requested samples. One LocalData input produces scalars; a sequence produces
        arrays.
    """

    # 1/ Check user inputs
    if not isinstance(error_structure, ErrorStructure):
        raise TypeError("error_structure must be an ErrorStructure.")
    if method not in {"auto", "analytical", "numerical"}:
        raise ValueError("method must be 'auto', 'analytical' or 'numerical'.")
    if (
        isinstance(max_sample_bytes, (bool, np.bool_))
        or not isinstance(max_sample_bytes, (int, np.integer))
        or max_sample_bytes < 1
    ):
        raise ValueError("max_sample_bytes must be a positive integer.")
    if method == "analytical" and return_samples:
        raise ValueError("return_samples cannot be used with analytical propagation.")

    # Run bound spatial methods with their own source and output layouts
    if operation_kwargs is not None:
        if data is not None or not callable(operator) or getattr(operator, "__self__", None) is None:
            raise TypeError("operation_kwargs requires a bound spatial method and no separate data argument.")
        if nominal_estimate is not None or output_labels is not None:
            raise ValueError("nominal_estimate and output_labels apply only to local operators.")
        return _propagate_spatial_method(
            operator,
            error_structure,
            dict(operation_kwargs),
            {
                "predictors": predictors,
                "method": method,
                "n_samples": n_samples,
                "random_state": random_state,
                "max_sample_bytes": max_sample_bytes,
                "on_error": on_error,
                "at": at,
                "return_samples": return_samples,
                "quantiles": quantiles,
            },
        )

    # Prepare LocalData targets for an interpolator or reducer
    if not isinstance(operator, (Interpolator, Reducer)):
        raise TypeError("operator must be an Interpolator, Reducer or supported bound spatial method.")
    scalar = isinstance(data, LocalData)
    targets: list[LocalData]
    if scalar:
        targets = [cast(LocalData, data)]
    else:
        targets = list(cast(Sequence[LocalData], data))
    if not targets or any(not isinstance(local, LocalData) for local in targets):
        raise ValueError("data must contain at least one LocalData target.")

    # 2/ We use each distinct source observation once when calculating errors for the selected neighborhoods
    source_ids, coordinates, source_positions = _collect_source_support(targets)
    component_data = None
    if len(source_ids):
        aligned_predictors = _align_predictors(predictors, source_ids=source_ids)
        coordinates, component_data = _prepare_source_errors(
            error_structure, len(source_ids), coordinates, aligned_predictors
        )
        if operator._uses_error_covariance:
            weighted_targets = []
            for local in targets:
                valid = local._valid_values()
                positions = np.asarray(
                    [source_positions[_hashable_source_id(i)] for i in local.source_ids[valid]], dtype=int
                )
                covariance = np.zeros((len(local.values), len(local.values)))
                covariance[np.ix_(valid, valid)] = _source_covariance_matrix(
                    source_ids, coordinates, component_data, positions
                )
                weighted_targets.append(replace(local, error_covariance=covariance))
            targets = weighted_targets

    # 3/ Calculate or validate the result from the original values
    handling = operator.default_nodata_propagation if nodata_propagation is None else nodata_propagation
    if nominal_estimate is None:
        estimate = _evaluate_batch(operator, targets, handling)
    else:
        estimate = np.atleast_1d(np.asarray(nominal_estimate, dtype=float))
        if estimate.shape != (len(targets),):
            raise ValueError("nominal_estimate must contain one value per LocalData target.")
    labels = _normalize_output_labels(output_labels, len(targets), scalar=scalar)

    # 4/ Select outputs for optional samples and quantiles
    all_outputs = _output_selection(labels, estimate)
    if at is None:
        selection = all_outputs if len(targets) <= 256 else all_outputs.iloc[:0]
    elif isinstance(at, str) and at == "all":
        selection = all_outputs
    else:
        if isinstance(at, (str, bytes)):
            raise TypeError("at must be 'all' or a sequence of output labels.")
        chosen = list(at)
        if len(chosen) != len(set(chosen)) or any(label not in labels for label in chosen):
            raise ValueError("at must name distinct existing output labels.")
        selection = all_outputs.iloc[[int(labels.get_loc(label)) for label in chosen]]
    if (return_samples or quantiles) and selection.empty:
        raise ValueError("Select outputs with at when requesting samples or quantiles.")

    # 5/ Return an empty summary when no output can be calculated
    if len(source_ids) == 0 or not np.any(np.isfinite(estimate)):
        if return_samples:
            raise ValueError("return_samples requires at least one valid source observation.")
        return _empty_local_operator_summary(
            estimate=estimate,
            selection=selection,
            error_structure=error_structure,
            scalar=scalar,
        )
    assert component_data is not None

    # 6/ Calculate exact uncertainty for weighted sums when possible
    if method != "numerical" and not return_samples:
        rows = _affine_rows(
            operator,
            targets,
            handling=handling,
            source_positions=source_positions,
            estimate=estimate,
        )
        if method == "analytical" and rows is None:
            raise TypeError("Analytical propagation requires coefficients() for every finite operator target.")
        if rows is not None:
            summary = _propagate_local_operator_analytically(
                rows=rows,
                estimate=estimate,
                source_ids=source_ids,
                coordinates=coordinates,
                component_data=component_data,
                selection=selection,
                error_structure=error_structure,
                scalar=scalar,
            )
            if quantiles:
                probabilities = tuple(float(value) for value in quantiles)
                table = pd.DataFrame(
                    [summary.quantile(probability) for probability in probabilities],
                    index=pd.Index(probabilities, name="probability"),
                    columns=summary.selection.index,
                )
                return replace(summary, quantiles=table)
            return summary

    # 7/ Estimate uncertainty by drawing source errors
    return _propagate_local_operator_numerically(
        operator,
        targets,
        estimate=estimate,
        source_ids=source_ids,
        coordinates=coordinates,
        component_data=component_data,
        source_positions=source_positions,
        selection=selection,
        error_structure=error_structure,
        scalar=scalar,
        handling=handling,
        n_samples=n_samples,
        random_state=random_state,
        return_samples=return_samples,
        max_sample_bytes=max_sample_bytes,
        quantiles=quantiles,
        on_error=on_error,
    )
