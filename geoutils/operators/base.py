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

"""Base module to define local data and weights passed to interpolators and reducers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from geoutils._typing import NDArrayBool, NDArrayNum
from geoutils.operators.nodata import NodataHandling

################
# 1/ LOCAL DATA
################


@dataclass(frozen=True)
class LocalData:
    """
    This data class stores the nearby source values used to calculate one interpolator/reducer output.

    We convert arrays with np.asanyarray() to avoid copies where possible. This object is only used temporarily for
    each neighbourhood.

    :param values: One-dimensional source values.
    :param valid: Boolean validity of every source value.
    :param source_ids: ID of every original observation.
    :param coordinates: Optional source coordinates with observations along the first dimension.
    :param target: Optional target coordinate.
    :param distances: Optional source-to-target distances.
    :param sample_weights: Optional weights attached to source observations.
    :param support_weights: Optional geometric weights, such as covered cell fractions.
    :param interpolation_weights: Optional precomputed weights for regular-grid interpolation.
    :param error_covariance: Optional covariance of observation errors, with one row and column per value.
    """

    values: NDArrayNum
    valid: NDArrayBool
    source_ids: NDArray[Any]
    coordinates: NDArrayNum | None = None
    target: NDArrayNum | None = None
    distances: NDArrayNum | None = None
    sample_weights: NDArrayNum | None = None
    support_weights: NDArrayNum | None = None
    interpolation_weights: NDArrayNum | None = None
    error_covariance: NDArrayNum | None = None

    def __post_init__(self) -> None:
        """Convert the source arrays and check that they line up."""

        # We check that values, validity flags and IDs have the same dimension/length
        values = np.asanyarray(self.values)
        valid = np.asarray(self.valid, dtype=bool)
        source_ids = np.asarray(self.source_ids)
        if values.ndim != 1 or valid.ndim != 1 or source_ids.ndim != 1:
            raise ValueError("LocalData values, valid and source_ids must be one-dimensional.")
        if len(valid) != len(values) or len(source_ids) != len(values):
            raise ValueError("LocalData values, valid and source_ids must have the same length.")

        # We check optional arrays too (one row per observation, one column per coordinate dimension)
        coordinates = None if self.coordinates is None else np.asarray(self.coordinates)
        if coordinates is not None and (coordinates.ndim != 2 or len(coordinates) != len(values)):
            raise ValueError("LocalData coordinates must have shape (n_observations, n_dimensions).")
        for name in ("distances", "sample_weights", "support_weights", "interpolation_weights"):
            array = getattr(self, name)
            if array is not None and (np.asarray(array).ndim != 1 or len(array) != len(values)):
                raise ValueError(f"LocalData {name} must be one-dimensional with one value per observation.")

        # Finally, we store arrays so the calculation can index them directly
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "source_ids", source_ids)
        target = None if self.target is None else np.asarray(self.target)
        if target is not None and target.ndim != 1:
            raise ValueError("LocalData target must be a one-dimensional coordinate.")
        if coordinates is not None and target is not None and coordinates.shape[1] != len(target):
            raise ValueError("LocalData source and target coordinates must have the same number of dimensions.")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "target", target)
        for name in ("distances", "sample_weights", "support_weights", "interpolation_weights"):
            array = getattr(self, name)
            object.__setattr__(self, name, None if array is None else np.asarray(array))

        # Covariance describes pairs of observations, so both axes must follow the value order
        if self.error_covariance is not None:
            covariance = np.asarray(self.error_covariance, dtype=float)
            if covariance.shape != (len(values), len(values)):
                raise ValueError("LocalData error_covariance must have one row and column per observation.")
            if not np.all(np.isfinite(covariance)) or not np.allclose(covariance, covariance.T):
                raise ValueError("LocalData error_covariance must be finite and symmetric.")
            object.__setattr__(self, "error_covariance", covariance)

    def select(self, selection: NDArrayBool) -> LocalData:
        """
        Select observations from this LocalData object.

        :param selection: Boolean array with one entry per observation (True selects that observation).
        :returns: A new LocalData object with the selected values, IDs and optional arrays.
        """

        selected = np.asarray(selection, dtype=bool)
        if selected.shape != self.values.shape:
            raise ValueError("LocalData selection must have one boolean per observation.")

        # Select the same observations from every array (the target coordinate is shared by all of them)
        optional = {}
        for name in ("coordinates", "distances", "sample_weights", "support_weights", "interpolation_weights"):
            array = getattr(self, name)
            optional[name] = None if array is None else array[selected]
        if self.error_covariance is not None:
            optional["error_covariance"] = self.error_covariance[np.ix_(selected, selected)]
        return replace(
            self,
            values=self.values[selected],
            valid=self.valid[selected],
            source_ids=self.source_ids[selected],
            **optional,
        )

    def precision_weights(self, weights: NDArrayNum) -> NDArrayNum:
        """Adjust nonnegative observation weights using their error covariance.

        Geometric or sample weights scale precision: a weight q divides the error variance by q.
        Correlated errors use the corresponding scaled covariance in a generalized least-squares mean.
        Zero weights exclude observations, and zero error variance gives exact observations precedence.
        The result is normalized; negative coefficients are possible for correlated observations.
        """

        weights = np.asarray(weights, dtype=float)
        if weights.shape != self.values.shape or np.any(weights < 0):
            raise ValueError("Precision weighting requires one nonnegative weight per observation.")
        selected = weights > 0
        result = np.zeros(len(weights), dtype=float)
        if not np.any(selected):
            return result
        if self.error_covariance is None:
            return weights / np.sum(weights)

        # Error-free observations alone determine the estimate when any are available
        exact = selected & (np.diag(self.error_covariance) == 0)
        if np.any(exact):
            result[exact] = weights[exact] / np.sum(weights[exact])
            return result

        # Scale both covariance axes by the square root of the requested observation importance
        scale = np.sqrt(weights[selected])
        covariance = self.error_covariance[np.ix_(selected, selected)] / np.outer(scale, scale)
        ones = np.ones(np.count_nonzero(selected))
        try:
            coefficients = np.linalg.solve(covariance, ones)
        except np.linalg.LinAlgError:
            # Singular errors may cancel exactly; enforce an unbiased mean even in the covariance null space
            system = np.ones((len(ones) + 1, len(ones) + 1))
            system[:-1, :-1] = covariance
            system[-1, -1] = 0
            right_hand_side = np.zeros(len(ones) + 1)
            right_hand_side[-1] = 1
            coefficients = np.linalg.lstsq(system, right_hand_side, rcond=None)[0][:-1]
        total = float(np.sum(coefficients))
        if not np.isfinite(total) or total <= 0:
            raise ValueError("The error covariance does not define a finite precision-weighted mean.")
        result[selected] = coefficients / total
        return result

    def _valid_values(self) -> NDArrayBool:
        """Identify source values that are flagged valid, finite, and unmasked."""

        finite = np.isfinite(np.ma.getdata(self.values))
        unmasked = ~np.ma.getmaskarray(self.values)
        return self.valid & finite & unmasked

    def _prepare(
        self,
        *,
        accepts_sample_weights: bool,
        accepts_support_weights: bool,
        minimum_inputs: int,
        nodata_handling: NodataHandling,
    ) -> tuple[LocalData | None, NDArrayBool]:
        """Prepare data passed to an interpolator/reducer, following its nodata and weight options."""

        # Check that the operator supports the supplied weights
        if self.sample_weights is not None and not accepts_sample_weights:
            raise ValueError("This operator does not accept sample_weights.")
        if self.support_weights is not None and not accepts_support_weights:
            raise ValueError("This operator does not accept support_weights.")

        # Check operator options to see if enough valid values are available
        if isinstance(minimum_inputs, bool) or not isinstance(minimum_inputs, int) or minimum_inputs < 0:
            raise ValueError("minimum_inputs must be a non-negative integer.")
        if nodata_handling not in ("ignore", "propagate"):
            raise ValueError("nodata handling must be 'ignore' or 'propagate'.")

        # Masked and non-finite values are nodata even when their validity flag says otherwise
        valid = self._valid_values()
        if np.count_nonzero(valid) < minimum_inputs:
            return None, np.empty(0, dtype=bool)

        # "ignore" removes nodata; "propagate" checks whether it affects the result
        if nodata_handling == "ignore":
            prepared = self.select(valid)
            prepared_valid = np.ones(len(prepared.values), dtype=bool)
        else:
            prepared = self
            prepared_valid = valid
        return prepared, prepared_valid

    def _evaluate(
        self,
        *,
        calculation: Callable[[LocalData], float],
        coefficients: Callable[[LocalData], LinearCoefficients | None],
        accepts_sample_weights: bool,
        accepts_support_weights: bool,
        minimum_inputs: int,
        nodata_handling: NodataHandling,
    ) -> float:
        """
        Calculate one result after checking the source values and the operator options.

        _prepare() applies the nodata and weight rules shared with uncertainty propagation.
        If coefficients() supplies weights, we calculate the weighted sum here, otherwise calculation() calls the
        operator predict() or reduce() method.
        """

        # Apply the same selection for the value calculation and its uncertainty
        prepared, prepared_valid = self._prepare(
            accepts_sample_weights=accepts_sample_weights,
            accepts_support_weights=accepts_support_weights,
            minimum_inputs=minimum_inputs,
            nodata_handling=nodata_handling,
        )
        if prepared is None:
            return float("nan")
        affine = coefficients(prepared)
        if affine is not None:
            if len(affine.weights) != len(prepared.values):
                raise ValueError("Linear coefficients must contain one weight per LocalData value.")

            # A nodata value with zero weight should not make the result NaN (0 * NaN would still be NaN in NumPy)
            if nodata_handling == "propagate" and np.any((~prepared_valid) & (affine.weights != 0)):
                return float("nan")
            values = np.where(prepared_valid, prepared.values, 0) if nodata_handling == "propagate" else prepared.values
            return float(np.dot(affine.weights, values) + affine.offset)

        # Without weights, we cannot tell whether a nodata value affects the result, so "propagate" rejects it
        if nodata_handling == "propagate" and not np.all(prepared_valid):
            return float("nan")
        result = calculation(prepared)
        if np.ndim(result) != 0:
            raise ValueError("An interpolation or reduction method must return a single value.")
        return float(result)


###########################
# 2/ LINEAR COEFFICIENTS
###########################


@dataclass(frozen=True)
class LinearCoefficients:
    """
    This class allows to define linear operators (interpolator or reducer) that can be optionally used with weights,
    or with uncertainty propagation.

    Such an Interpolator or Reducer can be expressed as a weighted sum ``weights @ values + offset``, and needs to
    implement a coefficients() method that returns this LinearCoefficients class.

    :param weights: One coefficient for every value in LocalData.
    :param offset: Constant added after the weighted sum.
    """

    weights: NDArrayNum
    offset: float = 0.0

    def __post_init__(self) -> None:
        """Convert the weights and check that all coefficients are finite."""

        weights = np.asarray(self.weights)
        if weights.ndim != 1:
            raise ValueError("Linear coefficient weights must be one-dimensional.")
        if not np.all(np.isfinite(weights)) or not np.isfinite(self.offset):
            raise ValueError("Linear coefficient weights and offset must be finite.")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "offset", float(self.offset))
