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

"""Carry source uncertainty through interpolation and reduction calculations."""

from __future__ import annotations

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
from geoutils.uncertainty.error_structure import BoundErrorStructure, ErrorStructure

Operator = Interpolator | Reducer


############################################
# 1/ SOURCE DATA AND OUTPUT LABELS
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


def _prepare_for_operator(
    operator: Operator,
    data: LocalData,
    handling: NodataHandling,
) -> tuple[LocalData | None, NDArray[np.bool_]]:
    """Apply the same rules for missing values and weights as in the original calculation."""

    return data._prepare(
        accepts_sample_weights=operator.accepts_sample_weights,
        accepts_support_weights=isinstance(operator, Reducer) and operator.accepts_support_weights,
        minimum_inputs=operator.minimum_inputs,
        nodata_handling=handling,
    )


def _evaluate_batch(operator: Operator, data: Sequence[LocalData], handling: NodataHandling) -> NDArray[np.float64]:
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
    """Create unique output labels for stored covariance and samples."""

    if labels is None:
        values: list[Hashable] = ["value"] if scalar else list(range(count))
    else:
        values = list(labels)
    index = pd.Index(values, tupleize_cols=False, name="output")
    if len(index) != count or not index.is_unique:
        raise ValueError("output_labels must contain one unique label per target.")
    return index


############################################
# 2/ EXACT PROPAGATION FOR WEIGHTED SUMS
############################################


def _affine_rows(
    operator: Operator,
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
        prepared, prepared_valid = _prepare_for_operator(operator, local, handling)
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


def _analytical_propagation(
    *,
    rows: Sequence[tuple[NDArray[np.int64], NDArray[np.float64]] | None],
    estimate: NDArray[np.float64],
    bound: BoundErrorStructure,
    labels: pd.Index,
    selection: pd.DataFrame,
    error_structure: ErrorStructure,
    scalar: bool,
    return_covariance: bool,
) -> PropagationSummary:
    """Calculate output means and covariance directly from source weights and covariance."""

    output_count = len(rows)
    mean = estimate.copy()
    variance = np.full(output_count, np.nan, dtype=float)
    for position, row in enumerate(rows):
        if row is None:
            continue
        indexes, weights = row
        mean[position] += float(weights @ bound.error_mean[indexes])

        # For a weighted sum, each pair contributes weight_i × covariance_ij × weight_j to the variance
        covariance = bound.covariance_block(indexes, indexes)
        variance[position] = max(float(weights @ covariance @ weights), 0.0)

    # We only need source pairs used by the two outputs being compared, not the full source covariance matrix
    covariance_table: pd.DataFrame | None = None
    if return_covariance:
        chosen = selection["flat_index"].to_numpy(dtype=np.int64)
        output_covariance = np.full((len(chosen), len(chosen)), np.nan, dtype=float)
        for selected_row, row_position in enumerate(chosen):
            first = rows[row_position]
            if first is None:
                continue
            first_indexes, first_weights = first
            for selected_column in range(selected_row, len(chosen)):
                column_position = chosen[selected_column]
                second = rows[column_position]
                if second is None:
                    continue
                second_indexes, second_weights = second
                block = bound.covariance_block(first_indexes, second_indexes)
                value = float(first_weights @ block @ second_weights)
                output_covariance[selected_row, selected_column] = value
                # Covariance is symmetric, so the opposite half of the matrix has the same values
                output_covariance[selected_column, selected_row] = value
        covariance_table = pd.DataFrame(output_covariance, index=selection.index, columns=selection.index)

    # Standard deviation has the output values' units; variance/covariance use squared units
    standard_deviation = np.sqrt(variance)
    complete = bool(
        return_covariance
        and len(selection) == len(rows)
        and np.all(np.isfinite(estimate))
        and np.all(np.isfinite(variance))
    )
    return PropagationSummary(
        estimate=float(estimate[0]) if scalar else estimate,
        mean=float(mean[0]) if scalar else mean,
        std=float(standard_deviation[0]) if scalar else standard_deviation,
        error_structure=error_structure,
        method="analytical",
        selection=selection,
        covariance=covariance_table,
        covariance_mean=(
            pd.Series(mean[selection["flat_index"].to_numpy(dtype=np.int64)], index=selection.index)
            if return_covariance
            else None
        ),
        metadata={
            "approximation": "exact",
            "covariance_is_complete": complete,
            "output_distribution": "joint_gaussian",
        },
    )


def _empty_propagation(
    *,
    estimate: NDArray[np.float64],
    labels: pd.Index,
    selection: pd.DataFrame,
    error_structure: ErrorStructure,
    scalar: bool,
    return_covariance: bool,
) -> PropagationSummary:
    """Return undefined uncertainty for targets with no finite source observations."""

    undefined = np.full(len(estimate), np.nan, dtype=float)
    covariance = None
    if return_covariance:
        covariance = pd.DataFrame(
            np.full((len(selection), len(selection)), np.nan), index=selection.index, columns=selection.index
        )
    return PropagationSummary(
        estimate=float(estimate[0]) if scalar else estimate,
        mean=float(undefined[0]) if scalar else undefined,
        std=float(undefined[0]) if scalar else undefined,
        error_structure=error_structure,
        method="analytical",
        selection=selection,
        covariance=covariance,
        metadata={"approximation": "exact", "covariance_is_complete": False, "output_distribution": "joint_gaussian"},
    )


############################################
# 3/ NUMERICAL PROPAGATION WITH RANDOM DRAWS
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


def _numerical_propagation(
    operator: Operator,
    data: Sequence[LocalData],
    *,
    estimate: NDArray[np.float64],
    bound: BoundErrorStructure,
    source_positions: Mapping[Hashable, int],
    labels: pd.Index,
    selection: pd.DataFrame,
    error_structure: ErrorStructure,
    scalar: bool,
    handling: NodataHandling,
    n_samples: int,
    random_state: int | np.random.Generator | None,
    return_covariance: bool,
    return_samples: bool,
    max_sample_bytes: int,
    quantiles: Sequence[float],
    on_error: Literal["raise", "warn"],
) -> PropagationSummary:
    """Repeat the interpolation/reduction with random source errors and summarize the resulting spread."""

    from geoutils.uncertainty.simulation import simulate

    def calculate(error: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply one shared source draw to every local target before evaluating the operator."""

        perturbed = _perturb_local_data(data, error, source_positions)
        values = _evaluate_batch(operator, perturbed, handling)
        values[~np.isfinite(estimate)] = np.nan
        return values[0] if scalar else values

    # One source draw is used by all targets, including targets with overlapping observations
    nominal = float(estimate[0]) if scalar else estimate
    return simulate(
        estimate=nominal,
        draw_error=bound.draw_error,
        calculate=calculate,
        error_structure=error_structure,
        selection=selection,
        n_samples=int(n_samples),
        random_state=random_state,
        return_covariance=return_covariance,
        return_samples=return_samples,
        max_sample_bytes=max_sample_bytes,
        quantiles=quantiles,
        on_error=on_error,
    )


############################################
# 4/ COMPLETE USER CALCULATIONS
############################################


def _callable_output_selection(
    estimate: Any,
    at: Literal["all"] | Sequence[Hashable] | None,
    *,
    default_limit: int,
) -> pd.DataFrame:
    """Select a small set of final outputs without constructing labels for an entire large raster."""

    values = _numeric_values(estimate)
    shape = values.shape

    # Only a request for every output needs a key for every cell
    if at is None and values.size > default_limit:
        chosen: list[Hashable] = []
    elif at is None or (isinstance(at, str) and at == "all"):
        if isinstance(estimate, pd.Series):
            chosen = estimate.index.tolist()
        elif isinstance(estimate, pd.DataFrame):
            chosen = [(row, column) for row in estimate.index for column in estimate.columns]
        elif values.ndim == 0:
            chosen = ["value"]
        else:
            chosen = list(np.ndindex(shape)) if values.ndim > 1 else list(range(values.size))
    elif isinstance(at, str):
        raise ValueError("at must be 'all' or a sequence of output labels.")
    else:
        chosen = list(at)

    # Resolve each requested key directly so a few raster cells do not build a full-grid index
    positions: list[int] = []
    try:
        if len(set(chosen)) != len(chosen):
            raise KeyError("duplicate key")
        for key in chosen:
            if isinstance(estimate, pd.Series):
                position = estimate.index.get_loc(key)
            elif isinstance(estimate, pd.DataFrame):
                if not isinstance(key, tuple) or len(key) != 2:
                    raise KeyError(key)
                row, column = key
                position = np.ravel_multi_index((estimate.index.get_loc(row), estimate.columns.get_loc(column)), shape)
            elif values.ndim == 0:
                if key != "value":
                    raise KeyError(key)
                position = 0
            else:
                coordinates: tuple[int, ...]
                if values.ndim == 1:
                    if not isinstance(key, (int, np.integer)):
                        raise KeyError(key)
                    coordinates = (int(key),)
                else:
                    if not isinstance(key, tuple) or len(key) != values.ndim:
                        raise KeyError(key)
                    if any(not isinstance(number, (int, np.integer)) for number in key):
                        raise KeyError(key)
                    coordinates = tuple(int(number) for number in key)
                if any(number < 0 or number >= size for number, size in zip(coordinates, shape)):
                    raise KeyError(key)
                position = np.ravel_multi_index(coordinates, shape)
            if not isinstance(position, (int, np.integer)):
                raise KeyError(key)
            positions.append(int(position))
    except (KeyError, TypeError, ValueError) as exception:
        raise ValueError("at must name distinct existing output labels.") from exception
    index = pd.Index(chosen, tupleize_cols=False, name="output")
    return pd.DataFrame({"flat_index": positions, "estimate": values.reshape(-1)[positions]}, index=index)


def _propagate_callable(
    operation: Callable[[Any], Any],
    data: Any,
    error_structure: ErrorStructure,
    *,
    predictors: Mapping[str, Any] | None,
    method: Literal["auto", "analytical", "numerical"],
    at: Literal["all"] | Sequence[Hashable] | None,
    n_samples: int,
    random_state: int | np.random.Generator | None,
    return_covariance: bool,
    return_samples: bool,
    max_covariance_size: int,
    max_sample_bytes: int,
    circular_period: float | None,
    on_error: Literal["raise", "warn"],
    quantiles: Sequence[float],
) -> PropagationSummary:
    """Run an entire callable calculation for every draw of the initial error model."""

    from geoutils.uncertainty.simulation import simulate

    if method == "analytical":
        raise NotImplementedError("An arbitrary callable needs explicit derivatives for analytical propagation.")
    estimate = operation(data)
    selection = _callable_output_selection(estimate, at, default_limit=max_covariance_size)
    if return_covariance and len(selection) > max_covariance_size:
        raise ValueError("Selected covariance exceeds max_covariance_size.")
    if (return_covariance or return_samples or quantiles) and selection.empty:
        raise ValueError("Select outputs with at when requesting covariance, samples or quantiles.")

    # A spatial model draws one complete field; finite Gaussian arrays use their stored observation labels
    if isinstance(data, (RasterBase, PointCloudBase)):

        def draw_error(generator: np.random.Generator) -> Any:
            """Draw one field on the original spatial support."""

            return error_structure.generate_random_field(like=data, predictors=predictors, random_state=generator)

        def calculate(error: Any) -> Any:
            """Run the full user calculation on one perturbed spatial input."""

            values = np.ma.asarray(data.data, dtype=float) + np.ma.asarray(error.data, dtype=float)
            return operation(data.copy(new_array=values))

        draw_error_callback = draw_error
        calculate_callback = calculate
    else:
        if predictors:
            raise ValueError("Predictors require a spatial component input.")
        nominal = _numeric_values(data)
        source_ids = data.index if isinstance(data, pd.Series) else np.arange(nominal.size)
        bound = error_structure.bind(source_ids)

        def draw_vector_error(generator: np.random.Generator) -> NDArray[np.float64]:
            """Draw one aligned finite error vector."""

            return bound.draw_error(generator).reshape(nominal.shape)

        def calculate_vector(error: NDArray[np.float64]) -> Any:
            """Run the full user calculation on one perturbed array or series."""

            return operation(_wrap_like(data, nominal + error))

        draw_error_callback = draw_vector_error
        calculate_callback = calculate_vector

    return simulate(
        estimate=estimate,
        draw_error=draw_error_callback,
        calculate=calculate_callback,
        error_structure=error_structure,
        selection=selection,
        n_samples=n_samples,
        random_state=random_state,
        output={"circular_period": circular_period},
        return_covariance=return_covariance,
        return_samples=return_samples,
        max_sample_bytes=max_sample_bytes,
        quantiles=quantiles,
        on_error=on_error,
        metadata={"operation": "callable", "output_distribution": "unknown"},
    )


############################################
# 5/ PROPAGATION THROUGH AN INTERPOLATOR OR REDUCER
############################################


def propagate(
    operator: Operator | Callable[[Any], Any],
    data: Any = None,
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
    return_covariance: bool = False,
    return_samples: bool = False,
    max_covariance_size: int = 256,
    max_sample_bytes: int = 268_435_456,
    circular_period: float | None = None,
    on_error: Literal["raise", "warn"] = "raise",
    quantiles: Sequence[float] = (),
) -> PropagationSummary:
    """Use observation errors for fitting and propagate them through an operator or spatial operation.

    For a weighted sum, _affine_rows() collects the weights and _analytical_propagation() calculates the resulting
    mean/variance directly. Other methods use _numerical_propagation(): we draw source errors, add them to the
    source values and repeat the interpolation/reduction to measure its spread.

    A repeated source ID receives the same error everywhere it is used. Different IDs represent separate
    observations, even at identical coordinates.

    For a spatial operation, pass a bound method and its arguments as operation_kwargs, for example
    propagate(raster.reproject, error_structure=errors, operation_kwargs={"ref": reference, "resampling": Mean()}).
    The estimate, mean and std then have the operation's raster or point layout. Marginal uncertainty can remain
    lazy with Dask; joint covariance or samples currently requires eager inputs. Calling the spatial method directly
    with error_structure uses errors only for fitting and returns its ordinary result.

    :param operator: Interpolator, Reducer, bound spatial method, or complete callable calculation.
    :param data: Source data for one output or an ordered sequence of outputs. Omit for a bound spatial method.
    :param error_structure: Source uncertainty model, before it is matched to these observations.
    :param operation_kwargs: Arguments to a bound grid(), reproject(), krige(), or point resampling method.
    :param predictors: Values used to calculate error magnitudes, given as scalars, dictionaries keyed by source ID,
        or arrays in sorted source ID order.
    :param method: Choose automatically between an exact weighted calculation and repeated random draws, or request
        one explicitly.
    :param nodata_propagation: Optional replacement for the Interpolator or Reducer's rule for missing values.
    :param nominal_estimate: Optional results already calculated from the original source values.
    :param output_labels: Unique labels for output covariance and stored samples.
    :param at: Output labels selected for covariance or samples. Small results are selected by default.
    :param n_samples: Number of random draws for numerical propagation.
    :param random_state: Seed or generator for numerical propagation.
    :param return_covariance: Whether to return covariance between outputs.
    :param return_samples: Whether to return the result of each random draw. This selects numerical propagation in
        automatic mode and cannot be combined with method="analytical".
    :param max_covariance_size: Largest covariance dimension allowed in one result.
    :param max_sample_bytes: Largest selected draw buffer for exact samples or empirical quantiles.
    :returns: A PropagationSummary with the original results, their mean and standard deviation after adding source
        errors, and any requested covariance or samples. One LocalData input produces scalars; a sequence produces
        arrays.
    """

    # Check the operator and error model, then use a scalar result for one target or an array for several
    if not isinstance(error_structure, ErrorStructure):
        raise TypeError("error_structure must be an ErrorStructure.")
    if method not in {"auto", "analytical", "numerical"}:
        raise ValueError("method must be 'auto', 'analytical' or 'numerical'.")
    if (
        isinstance(max_covariance_size, (bool, np.bool_))
        or not isinstance(max_covariance_size, (int, np.integer))
        or max_covariance_size < 1
    ):
        raise ValueError("max_covariance_size must be a positive integer.")
    if (
        isinstance(max_sample_bytes, (bool, np.bool_))
        or not isinstance(max_sample_bytes, (int, np.integer))
        or max_sample_bytes < 1
    ):
        raise ValueError("max_sample_bytes must be a positive integer.")
    if method == "analytical" and return_samples:
        raise ValueError("return_samples cannot be used with analytical propagation.")
    if operation_kwargs is not None:
        from geoutils.uncertainty.spatial import _propagate_spatial

        if data is not None or not callable(operator) or getattr(operator, "__self__", None) is None:
            raise TypeError("operation_kwargs requires a bound spatial method and no separate data argument.")
        if nominal_estimate is not None or output_labels is not None or circular_period is not None:
            raise ValueError("These output options apply to LocalData groups or complete callable calculations.")
        return _propagate_spatial(
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
                "return_covariance": return_covariance,
                "return_samples": return_samples,
                "max_covariance_size": max_covariance_size,
                "quantiles": quantiles,
            },
        )
    if callable(operator) and not isinstance(operator, (Interpolator, Reducer)):
        if nodata_propagation is not None or nominal_estimate is not None or output_labels is not None:
            raise ValueError("nodata_propagation, nominal_estimate and output_labels apply only to local operators.")
        return _propagate_callable(
            operator,
            data,
            error_structure,
            predictors=predictors,
            method=method,
            at=at,
            n_samples=n_samples,
            random_state=random_state,
            return_covariance=return_covariance,
            return_samples=return_samples,
            max_covariance_size=max_covariance_size,
            max_sample_bytes=max_sample_bytes,
            circular_period=circular_period,
            on_error=on_error,
            quantiles=quantiles,
        )
    if not isinstance(operator, (Interpolator, Reducer)):
        raise TypeError("operator must be an Interpolator, Reducer or callable.")
    if circular_period is not None:
        raise ValueError("Circular outputs require a complete callable calculation.")
    scalar = isinstance(data, LocalData)
    targets: list[LocalData]
    if scalar:
        targets = [cast(LocalData, data)]
    else:
        targets = list(cast(Sequence[LocalData], data))
    if not targets or any(not isinstance(local, LocalData) for local in targets):
        raise ValueError("data must contain at least one LocalData target.")

    # Fit and propagate with the same observation covariance, matched once across overlapping neighborhoods
    source_ids, coordinates, source_positions = _collect_source_support(targets)
    bound = None
    if len(source_ids):
        aligned_predictors = _align_predictors(predictors, source_ids=source_ids)
        bound = error_structure.bind(source_ids, coordinates=coordinates, predictors=aligned_predictors)
        if operator._uses_error_covariance:
            weighted_targets = []
            for local in targets:
                valid = local._valid_values()
                positions = np.asarray(
                    [source_positions[_hashable_source_id(i)] for i in local.source_ids[valid]], dtype=int
                )
                covariance = np.zeros((len(local.values), len(local.values)))
                covariance[np.ix_(valid, valid)] = bound.covariance_block(positions, positions)
                weighted_targets.append(replace(local, error_covariance=covariance))
            targets = weighted_targets

    # Calculate the original result, or use the supplied estimate after checking its shape
    handling = operator.default_nodata_propagation if nodata_propagation is None else nodata_propagation
    if nominal_estimate is None:
        estimate = _evaluate_batch(operator, targets, handling)
    else:
        estimate = np.atleast_1d(np.asarray(nominal_estimate, dtype=float))
        if estimate.shape != (len(targets),):
            raise ValueError("nominal_estimate must contain one value per LocalData target.")
    labels = _normalize_output_labels(output_labels, len(targets), scalar=scalar)

    # Full output moments can be large; select only the joint outputs the user can afford to store
    all_outputs = _output_selection(labels, estimate)
    if at is None:
        selection = all_outputs if len(targets) <= max_covariance_size else all_outputs.iloc[:0]
    elif isinstance(at, str) and at == "all":
        selection = all_outputs
    else:
        if isinstance(at, (str, bytes)):
            raise TypeError("at must be 'all' or a sequence of output labels.")
        chosen = list(at)
        if len(chosen) != len(set(chosen)) or any(label not in labels for label in chosen):
            raise ValueError("at must name distinct existing output labels.")
        selection = all_outputs.iloc[[int(labels.get_loc(label)) for label in chosen]]
    if return_covariance and len(selection) > max_covariance_size:
        raise ValueError("Selected covariance exceeds max_covariance_size.")
    if (return_covariance or return_samples or quantiles) and selection.empty:
        raise ValueError("Select outputs with at when requesting covariance, samples or quantiles.")

    # Match the error model to all source observations once so every output and draw uses the same source errors
    if len(source_ids) == 0 or not np.any(np.isfinite(estimate)):
        if return_samples:
            raise ValueError("return_samples requires at least one valid source observation.")
        return _empty_propagation(
            estimate=estimate,
            labels=labels,
            selection=selection,
            error_structure=error_structure,
            scalar=scalar,
            return_covariance=return_covariance,
        )
    assert bound is not None

    # Use exact propagation for weighted sums, unless the caller requested draws or numerical propagation
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
            summary = _analytical_propagation(
                rows=rows,
                estimate=estimate,
                bound=bound,
                labels=labels,
                selection=selection,
                error_structure=error_structure,
                scalar=scalar,
                return_covariance=return_covariance,
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
    return _numerical_propagation(
        operator,
        targets,
        estimate=estimate,
        bound=bound,
        source_positions=source_positions,
        labels=labels,
        selection=selection,
        error_structure=error_structure,
        scalar=scalar,
        handling=handling,
        n_samples=n_samples,
        random_state=random_state,
        return_covariance=return_covariance,
        return_samples=return_samples,
        max_sample_bytes=max_sample_bytes,
        quantiles=quantiles,
        on_error=on_error,
    )


############################################
# 6/ PROPAGATION RESULTS
############################################

# Results from carrying source uncertainty through a calculation.


def _as_output(values: NDArray[np.float64], *, scalar: bool) -> float | NDArray[np.float64]:
    """Return a scalar when the uncertainty calculation received one LocalData object."""

    return float(values[0]) if scalar else values.copy()


def _numeric_values(value: Any) -> NDArray[np.float64]:
    """Read numeric values from an output without changing its labels or spatial support."""

    if isinstance(value, (RasterBase, PointCloudBase)):
        value = value.data
    if isinstance(value, (pd.Series, pd.DataFrame)):
        value = value.to_numpy()
    return np.ma.asarray(value, dtype=float).filled(np.nan)


def _wrap_like(template: Any, values: NDArray[Any]) -> Any:
    """Give calculated values the same spatial or labelled layout as the original output."""

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


@dataclass(frozen=True, kw_only=True)
class PropagationSummary:
    """
    Propagation summary that stores propagated uncertainty descriptors, whether analytically or numerically.

    The PropagationSummary class is returned by propagate().
    estimate is the result from the original values; mean and std describe the result after adding source errors.
    covariance describes how errors in different outputs vary together, and samples optionally stores the random
    draws used in numerical propagation. Use interval() for uncertainty bounds or quantile() for a percentile.

    :param estimate: Results calculated from the original source values.
    :param mean: Mean results after including source uncertainty.
    :param std: Standard deviation of the results after including source uncertainty.
    :param error_structure: Error model used for the source observations.
    :param method: "analytical" for exact weighted sums, or "numerical" for repeated random draws (Monte Carlo).
    :param selection: Labels and original estimates for the outputs described by covariance or samples.
    :param covariance: Covariance between all selected outputs.
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
    covariance: pd.DataFrame | None = None
    covariance_mean: pd.Series | None = None
    covariance_n: int | None = None
    samples: pd.DataFrame | None = None
    quantiles: pd.DataFrame | None = None
    marginals: Mapping[Hashable, Mapping[str, Any]] = field(default_factory=dict)
    n_samples: int | None = None
    n_success: int | None = None
    n_valid: Any = None
    failures: Mapping[int, str] = field(default_factory=dict)
    resultant_length: Any = None
    linearization_valid: Any = None
    context: Mapping[str, Any] | None = None
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
        if "unit" not in selection:
            selection["unit"] = None
        if "period" not in selection:
            selection["period"] = self.output.get("circular_period")
        object.__setattr__(self, "selection", selection)

        # Covariance rows/columns and sample columns must all refer to the same outputs, in the same order
        if self.covariance is not None:
            if not self.covariance.index.equals(selection.index) or not self.covariance.columns.equals(selection.index):
                raise ValueError("covariance axes must exactly match selection.")
            object.__setattr__(self, "covariance", self.covariance.copy(deep=True))
        if self.covariance_mean is not None:
            if not self.covariance_mean.index.equals(selection.index):
                raise ValueError("covariance_mean index must exactly match selection.")
            object.__setattr__(self, "covariance_mean", self.covariance_mean.copy(deep=True))
        if self.samples is not None:
            if not self.samples.columns.equals(selection.index):
                raise ValueError("samples columns must exactly match selection.")
            object.__setattr__(self, "samples", self.samples.copy(deep=True))
        if self.quantiles is not None:
            if not self.quantiles.columns.equals(selection.index):
                raise ValueError("quantiles columns must exactly match selection.")
            object.__setattr__(self, "quantiles", self.quantiles.copy(deep=True))
        if set(self.marginals).difference(selection.index):
            raise ValueError("marginals must refer to selected outputs.")
        copied_marginals: dict[Hashable, Mapping[str, Any]] = {}
        for key, marginal in self.marginals.items():
            copied: dict[str, Any] = {}
            for name, value in marginal.items():
                if isinstance(value, np.ndarray):
                    value = np.array(value, copy=True)
                    value.setflags(write=False)
                copied[name] = value
            copied_marginals[key] = MappingProxyType(copied)
        object.__setattr__(self, "marginals", MappingProxyType(copied_marginals))
        object.__setattr__(self, "failures", MappingProxyType(dict(self.failures)))
        object.__setattr__(self, "output", MappingProxyType(dict(self.output)))
        if self.context is not None:
            copied_context: dict[str, Any] = {}
            for name, value in self.context.items():
                if isinstance(value, np.ndarray):
                    value = np.array(value, copy=True)
                    value.setflags(write=False)
                copied_context[name] = value
            object.__setattr__(self, "context", MappingProxyType(copied_context))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def nsim(self) -> int | None:
        """Return the number of requested simulations."""

        return self.n_samples

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

    @property
    def correlation(self) -> pd.DataFrame | None:
        """Return correlations between outputs, or None if covariance was not requested.

        Correlation is undefined for outputs with zero variance; their rows/columns contain NaN.
        """

        if self.covariance is None:
            return None
        values = self.covariance.to_numpy(dtype=float, copy=True)
        diagonal = np.diag(values)
        standard_deviation = np.full(len(diagonal), np.nan)
        positive = np.isfinite(diagonal) & (diagonal > 0)
        standard_deviation[positive] = np.sqrt(diagonal[positive])

        # Dividing covariance by both outputs' standard deviations removes their scale/units
        divisor = np.outer(standard_deviation, standard_deviation)
        correlation = np.divide(
            values,
            divisor,
            out=np.full_like(values, np.nan),
            where=np.isfinite(divisor) & (divisor > 0),
        )
        return pd.DataFrame(correlation, index=self.selection.index, columns=self.selection.index)

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
        known_normal = self.metadata.get("output_distribution") == "joint_gaussian"
        if not known_normal and not all(
            self.marginals.get(key, {}).get("family") == "normal" for key in self.selection.index
        ):
            raise ValueError("Quantiles require saved draws, retained quantiles or an explicit normal output law.")
        values = []
        for key, position in zip(self.selection.index, self.selection["flat_index"]):
            marginal = self.marginals.get(key)
            mean = float(marginal["mean"]) if marginal is not None else float(means[position])
            std = float(marginal["std"]) if marginal is not None else float(standard_deviations[position])
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
        if at in self.marginals:
            return self.marginals[at]
        if self.samples is not None:
            result = self.samples[at].copy()
            result.name = at
            return result
        if self.metadata.get("output_distribution") != "joint_gaussian":
            raise ValueError("This output has no saved draws or known marginal distribution.")
        position = int(self.selection.loc[at, "flat_index"])
        mean = float(_numeric_values(self.mean).reshape(-1)[position])
        std = float(_numeric_values(self.std).reshape(-1)[position])
        return MappingProxyType({"family": "normal", "mean": mean, "std": std, "status": "exact"})
