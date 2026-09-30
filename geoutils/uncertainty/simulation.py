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

"""Summarize complete calculations made from repeated draws of one source error model."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from geoutils.uncertainty.error_structure import ErrorStructure
from geoutils.uncertainty.propagation import (
    PropagationSummary,
    _numeric_values,
    _wrap_difference,
    _wrap_like,
)


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
    return_covariance: bool = False,
    return_samples: bool = False,
    quantiles: Sequence[float] = (),
    max_sample_bytes: int = 268_435_456,
    on_error: Literal["raise", "warn"] = "raise",
    fatal_exceptions: tuple[type[Exception], ...] = (),
    marginals: Mapping[Any, Mapping[str, Any]] | None = None,
    context: Mapping[str, Any] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> PropagationSummary:
    """Draw initial errors and stream the final calculation's marginal and selected joint statistics.

    calculate() receives one draw from draw_error(). It must execute the whole calculation on that draw, so
    repeated uses of the same observation share one error realization. Selected samples are stored only when
    requested for output or exact empirical quantiles. Covariance uses complete selected realizations and reports
    the mean of those same realizations.

    :param estimate: Calculation performed on the original data.
    :param draw_error: Return one complete draw of errors from the initial source.
    :param calculate: Apply one complete calculation to that draw.
    :param error_structure: Initial error model used for the draws.
    :param selection: Output keys and flat positions retained for joint statistics.
    :param n_samples: Number of requested realizations, at least two.
    :param random_state: Seed or generator controlling source draws.
    :param output: Output units, support and optional circular period.
    :param return_covariance: Whether to calculate selected joint covariance.
    :param return_samples: Whether to save every selected realization.
    :param quantiles: Probabilities of exact empirical quantiles to save.
    :param max_sample_bytes: Maximum temporary memory used for selected realizations.
    :param on_error: Raise a calculation error or warn and count its draw as failed.
    :param fatal_exceptions: Errors that always indicate an invalid calculation contract.
    :param marginals: Optional known distributions for selected outputs.
    :param context: Additional information needed to interpret the output.
    :param metadata: Description of the calculation and its assumptions.
    :returns: Nominal and propagated results with requested selected information.
    """

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

    # Validate the selected positions and storage budget before drawing a source field
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

    # Update each output's moments without retaining complete raster or point-cloud realizations
    count = np.zeros(shape, dtype=np.int64)
    mean = np.zeros(shape, dtype=float)
    moment = np.zeros(shape, dtype=float)
    sine = np.zeros(shape, dtype=float)
    cosine = np.zeros(shape, dtype=float)
    covariance_mean = np.zeros(len(positions), dtype=float)
    covariance_moment = np.zeros((len(positions), len(positions)), dtype=float) if return_covariance else None
    covariance_n = 0
    failures: dict[int, str] = {}
    n_success = 0
    generator = np.random.default_rng(random_state)
    nominal_selected = selection["estimate"].to_numpy(dtype=float, copy=True)

    for number in range(1, int(n_samples) + 1):
        try:
            values = np.asarray(_numeric_values(calculate(draw_error(generator))), dtype=float)
            if values.shape != shape:
                raise ValueError("A simulation changed the output shape.")
            if not np.any(np.isfinite(values)):
                raise ValueError("The calculation returned no finite values.")
        except Exception as exception:
            if on_error == "raise" or isinstance(exception, fatal_exceptions):
                raise
            failures[number] = str(exception)
            warnings.warn(f"Simulation {number} of {n_samples} failed and was skipped: {exception}", stacklevel=2)
            continue

        # Partial results contribute to their own marginal statistics but not to complete-case covariance
        valid = np.isfinite(values)
        count[valid] += 1
        if period is None:
            difference = values[valid] - mean[valid]
            mean[valid] += difference / count[valid]
            moment[valid] += difference * (values[valid] - mean[valid])
        else:
            angles = values[valid] * (2 * np.pi / period)
            sine[valid] += np.sin(angles)
            cosine[valid] += np.cos(angles)
        n_success += 1

        selected = values.reshape(-1)[positions]
        if samples_buffer is not None:
            samples_buffer[number - 1] = selected
        if covariance_moment is not None and np.all(np.isfinite(selected)):
            if period is not None:
                selected = nominal_selected + _wrap_difference(selected - nominal_selected, period)
            covariance_n += 1
            difference = selected - covariance_mean
            covariance_mean += difference / covariance_n
            covariance_moment += np.outer(difference, selected - covariance_mean)

    if n_success < 2:
        raise RuntimeError(f"Only {n_success} of {n_samples} simulations succeeded; at least two are required.")

    # Angular means and spreads use the circular resultant; ordinary outputs use sample variance
    std = np.full(shape, np.nan, dtype=float)
    resultant: Any = None
    if period is None:
        mean[count == 0] = np.nan
        std[count > 1] = np.sqrt(np.maximum(moment[count > 1], 0) / (count[count > 1] - 1))
    else:
        length = np.divide(np.hypot(sine, cosine), count, out=np.full(shape, np.nan), where=count > 0)
        length = np.minimum(length, 1.0)
        mean = np.mod(np.arctan2(sine, cosine), 2 * np.pi) * period / (2 * np.pi)
        undefined = (count > 0) & (length <= 32 * np.finfo(float).eps)
        mean[(count == 0) | undefined] = np.nan
        regular = (count > 1) & ~undefined
        std[regular] = period / (2 * np.pi) * np.sqrt(-2 * np.log(length[regular]))
        std[(count > 1) & undefined] = np.inf
        resultant = _wrap_like(estimate, length)

    # Keep the covariance's own mean and count when partial output failures change its population
    covariance: pd.DataFrame | None = None
    covariance_mean_series: pd.Series | None = None
    retained_covariance_n: int | None = None
    if covariance_moment is not None:
        values = np.full_like(covariance_moment, np.nan)
        if covariance_n >= 2:
            values = covariance_moment / (covariance_n - 1)
        covariance = pd.DataFrame(values, index=selection.index, columns=selection.index)
        if covariance_n == 0:
            covariance_mean[:] = np.nan
        covariance_mean_series = pd.Series(covariance_mean, index=selection.index, dtype=float)
        retained_covariance_n = covariance_n

    # Use one selected buffer for requested draws and empirical quantiles, then release it
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
                values = nominal_selected + _wrap_difference(values - nominal_selected, period)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
                calculated = np.nanquantile(values, probabilities, axis=0, method="linear")
            quantile_table = pd.DataFrame(
                calculated, index=pd.Index(probabilities, name="probability"), columns=selection.index
            )

    details = dict(metadata or {})
    details.setdefault("approximation", "monte_carlo")
    details["covariance_is_complete"] = bool(
        covariance is not None
        and len(selection) == int(np.count_nonzero(np.isfinite(nominal)))
        and np.all(np.isfinite(covariance.to_numpy(dtype=float)))
    )
    n_valid = int(count.reshape(-1)[0]) if nominal.ndim == 0 else _wrap_like(estimate, count)
    return PropagationSummary(
        estimate=estimate,
        mean=_wrap_like(estimate, mean),
        std=_wrap_like(estimate, std),
        error_structure=error_structure,
        method="numerical",
        selection=selection,
        output=output_info,
        covariance=covariance,
        covariance_mean=covariance_mean_series,
        covariance_n=retained_covariance_n,
        samples=samples,
        quantiles=quantile_table,
        marginals=dict(marginals or {}),
        n_samples=int(n_samples),
        n_success=n_success,
        n_valid=n_valid,
        failures=failures,
        resultant_length=resultant,
        context=context,
        metadata=details,
    )
