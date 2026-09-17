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

"""Estimate, fit, and convert variograms across Python packages for interoperability."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit

from geoutils._misc import import_optional
from geoutils._typing import NDArrayNum

if TYPE_CHECKING:
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.raster.base import RasterBase

__all__ = ["Variogram"]


#############################
# 1/ VARIOGRAM MODEL METADATA
#############################

_BASE_MODELS = {"spherical", "exponential", "gaussian", "cubic", "stable", "matern"}
_COMPOSITE_MODELS = {"sum", "product"}


@dataclass(frozen=True)
class VariogramModel:
    """
    Parameters of a fitted theoretical variogram in a form shared by all supported packages: GSTools, GPyTorch, and
    SciKit-GStat.

    The ``effective_range`` follows SciKit-GStat's convention (as numerical ranges are defined differently between
    packages), and ``partial_sill`` excludes the nugget, making the conversion to covariance kernels unambiguous.
    A composite model holds its independent structures in ``components`` and keeps their shared nugget on the parent
    model.

    :param model_name: Base model name or ``"sum"``/``"product"`` for a composition.
    :param effective_range: Distance at which a base model effectively reaches its sill.
    :param partial_sill: Structured variance excluding the nugget.
    :param nugget: Uncorrelated variance added at positive distances.
    :param smoothness: Matérn smoothness parameter.
    :param shape: Stable model shape parameter.
    :param active_dims: Feature columns used by a converted covariance kernel.
    :param components: Base structures contained by a composite model.
    """

    model_name: str
    effective_range: float | None = None
    partial_sill: float | None = None
    nugget: float = 0.0
    smoothness: float | None = None
    shape: float | None = None
    active_dims: tuple[int, ...] | None = None
    components: tuple[VariogramModel, ...] = ()

    #####################
    # MODEL VALIDATION
    #####################

    def __post_init__(self) -> None:
        # Replace accepted short names with the one standard name used by every conversion
        aliases = {
            "cub": "cubic",
            "exp": "exponential",
            "gau": "gaussian",
            "mat": "matern",
            "rbf": "gaussian",
            "sph": "spherical",
            "sta": "stable",
        }
        model_name = aliases.get(self.model_name.lower(), self.model_name.lower())
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "components", tuple(self.components))

        # Convert numeric fields once so loaded and newly fitted models behave alike
        for name in ("effective_range", "partial_sill", "nugget", "smoothness", "shape"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))
        if self.active_dims is not None:
            object.__setattr__(self, "active_dims", tuple(int(value) for value in self.active_dims))
        if self.partial_sill is not None and (not np.isfinite(self.partial_sill) or self.partial_sill < 0):
            raise ValueError("Variogram partial sill must be a finite, non-negative number.")

        # Check combined models and calculate their sill from their component models
        if model_name in _COMPOSITE_MODELS:
            if len(self.components) < 2:
                raise ValueError("A composite variogram requires at least two components.")
            if any(component.model_name == model_name for component in self.components):
                raise ValueError(f"Nested {model_name} variograms are not supported; flatten the components first.")
            if any(component.nugget != 0 for component in self.components):
                raise ValueError("Composite components must omit nuggets; set one shared nugget on the parent model.")
            if self.partial_sill is None:
                partial_sills = [component.partial_sill or 0.0 for component in self.components]
                combined_sill = sum(partial_sills) if model_name == "sum" else np.prod(partial_sills)
                object.__setattr__(self, "partial_sill", float(combined_sill))
        elif model_name not in _BASE_MODELS:
            raise ValueError(f"Unsupported variogram model {model_name!r}.")
        else:
            # Require the range, variance, and optional shape values needed by this model type
            if self.components:
                raise ValueError("Only a composite variogram can contain components.")
            if self.effective_range is None or not np.isfinite(self.effective_range) or self.effective_range <= 0:
                raise ValueError("Variogram effective range must be a finite, strictly positive number.")
            if self.partial_sill is None:
                raise ValueError("A base variogram requires a partial sill.")
            if model_name == "matern" and self.smoothness is None:
                raise ValueError("A Matérn variogram requires a smoothness parameter.")
            if model_name == "stable" and self.shape is None:
                raise ValueError("A stable variogram requires a shape parameter.")

        # Check the shared numeric fields after model-specific defaults are applied
        if not np.isfinite(self.nugget) or self.nugget < 0:
            raise ValueError("Variogram nugget must be a finite, non-negative number.")
        if self.smoothness is not None and (not np.isfinite(self.smoothness) or self.smoothness <= 0):
            raise ValueError("Variogram smoothness must be a finite, strictly positive number.")
        if self.shape is not None and (not np.isfinite(self.shape) or self.shape <= 0):
            raise ValueError("Variogram shape must be a finite, strictly positive number.")
        if self.active_dims is not None and (
            len(self.active_dims) == 0
            or len(set(self.active_dims)) != len(self.active_dims)
            or min(self.active_dims) < 0
        ):
            raise ValueError("Argument ``active_dims`` must contain unique, non-negative dimensions.")

    ####################
    # MODEL EVALUATION
    ####################

    @property
    def sill(self) -> float:
        """Total sill, including the nugget."""

        return float((self.partial_sill or 0.0) + self.nugget)

    def variogram(self, distance: NDArrayNum | float) -> NDArrayNum:
        """
        Evaluate the theoretical variogram at one or more distances.

        :param distance: Spatial distance or array of distances.
        :returns: Semivariance at each distance.
        """

        # Use one array calculation, then return a scalar when the input was scalar
        scalar_input = np.ndim(distance) == 0
        distances = np.atleast_1d(np.asarray(distance, dtype=float))

        # Add component semivariances, then add the shared nugget once
        if self.model_name == "sum":
            values = sum((component.variogram(distances) for component in self.components), np.zeros_like(distances))
            output = values + np.where(distances > 0, self.nugget, 0.0)
            return output[0] if scalar_input else output

        # Multiply component covariances, then convert the result back to semivariance
        if self.model_name == "product":
            if self.partial_sill is None:
                raise AssertionError("A product variogram model must define its partial sill.")
            covariance = np.ones_like(distances)
            for component in self.components:
                covariance *= component.covariance(distances)
            output = (float(self.partial_sill) - covariance) + np.where(distances > 0, self.nugget, 0.0)
            return output[0] if scalar_input else output

        # Confirm that validation supplied the numeric fields required by SciKit-GStat
        if self.effective_range is None or self.partial_sill is None:
            raise AssertionError("A base variogram model must define its range and partial sill.")

        # Pass parameters in the order expected by the selected SciKit-GStat model function
        skgstat = import_optional("skgstat", package_name="scikit-gstat", extra_name="geostat")
        model_function = getattr(skgstat.models, self.model_name)
        arguments: list[float] = [self.effective_range, self.partial_sill]
        if self.model_name == "matern":
            if self.smoothness is None:
                raise AssertionError("A Matérn variogram model must define its smoothness.")
            arguments.append(float(self.smoothness))
        elif self.model_name == "stable":
            if self.shape is None:
                raise AssertionError("A stable variogram model must define its shape.")
            arguments.append(float(self.shape))

        # Calculate the distance-dependent part and add the nugget above zero distance
        values = np.asarray(model_function(distances.ravel(), *arguments), dtype=float).reshape(distances.shape)
        output = values + np.where(distances > 0, self.nugget, 0.0)
        return output[0] if scalar_input else output

    def covariance(self, distance: NDArrayNum | float) -> NDArrayNum:
        """
        Evaluate covariance implied by this variogram.

        :param distance: Spatial distance or array of distances.
        :returns: Covariance at each distance.
        """

        return self.sill - self.variogram(distance)

    def correlation(self, distance: NDArrayNum | float) -> NDArrayNum:
        """
        Evaluate correlation implied by this variogram.

        :param distance: Spatial distance or array of distances.
        :returns: Correlation at each distance.
        """

        if self.sill == 0:
            raise ValueError("A variogram with zero sill does not define correlation.")
        return self.covariance(distance) / self.sill

    ###################
    # MODEL COMPOSITION
    ###################

    @classmethod
    def sum(cls, components: Sequence[VariogramModel], nugget: float = 0.0) -> VariogramModel:
        """
        Combine independent nested structures into a summed variogram model.

        :param components: Fitted structures to add.
        :param nugget: Shared uncorrelated variance.
        :returns: Summed model with normalized components.
        """

        return cls.combine(components, combination="sum", nugget=nugget)

    @classmethod
    def combine(
        cls,
        components: Sequence[VariogramModel],
        *,
        combination: str,
        nugget: float = 0.0,
    ) -> VariogramModel:
        """
        Combine independently parameterized components by addition or multiplication.

        :param components: Fitted structures to combine.
        :param combination: Either ``"sum"`` or ``"product"``.
        :param nugget: Shared uncorrelated variance.
        :returns: Composite model with normalized components.
        """

        if combination not in _COMPOSITE_MODELS:
            raise ValueError("Argument ``combination`` must be 'sum' or 'product'.")

        # Flatten nested combinations of the same kind and store all nuggets once on the parent
        flattened: list[VariogramModel] = []
        for component in components:
            if component.model_name == combination:
                flattened.extend(component.components)
            else:
                flattened.append(replace(component, nugget=0.0))
        return cls(model_name=combination, nugget=nugget, components=tuple(flattened))

    #################
    # SERIALIZATION
    #################

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible model description."""

        # Convert nested components too so the complete model contains only plain Python data
        return {
            "model_name": self.model_name,
            "effective_range": self.effective_range,
            "partial_sill": self.partial_sill,
            "nugget": self.nugget,
            "smoothness": self.smoothness,
            "shape": self.shape,
            "active_dims": self.active_dims,
            "components": [component.to_dict() for component in self.components],
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> VariogramModel:
        """
        Restore a model from VariogramModel.to_dict() output.

        :param values: Serialized model fields.
        :returns: Restored fitted model.
        """

        # Restore nested components before the dataclass checks the complete parent model
        components = tuple(cls.from_dict(component) for component in values.get("components", ()))
        return cls(
            model_name=str(values["model_name"]),
            effective_range=values.get("effective_range"),
            partial_sill=values.get("partial_sill"),
            nugget=float(values.get("nugget", 0.0)),
            smoothness=values.get("smoothness"),
            shape=values.get("shape"),
            active_dims=(
                None if values.get("active_dims") is None else tuple(int(value) for value in values["active_dims"])
            ),
            components=components,
        )


################################
# 2/ RESULTS FROM OTHER PACKAGES
################################


@dataclass(frozen=True)
class GPyTorchVariogram:
    """
    A GPyTorch covariance kernel and its separate observation noise nugget.

    :param kernel: Converted GPyTorch covariance kernel.
    :param noise: Nugget variance for the observation likelihood.
    """

    kernel: Any
    noise: float


@dataclass(frozen=True)
class GSToolsVariogram:
    """
    A GSTools covariance model and the source coordinate columns it uses.

    :param model: Converted GSTools covariance model.
    :param active_dims: Feature columns to select before passing coordinates to GSTools.
    """

    model: Any
    active_dims: tuple[int, ...] | None


############################
# 3/ MEASURED VARIOGRAM RESULT
############################


@dataclass(frozen=True)
class Variogram:
    """
    Measured and fitted variogram values that can be saved without a fitting package object.

    The arrays have one value per distance bin. ``backend_object`` is absent by default because a SciKit-GStat Variogram
    retains sampled coordinates, pairwise distances and pairwise differences. Pass ``keep_backend=True`` during
    estimation or conversion only when its extra methods are worth the additional memory.

    :param lags: Mean sampled distance in each distance bin.
    :param semivariance: Measured semivariance in each distance bin.
    :param counts: Number of sampled pairs in each distance bin.
    :param semivariance_error: Sampling error estimated across independent runs.
    :param bin_lower_edges: Inclusive lower boundaries of distance bins.
    :param bin_edges: Upper boundaries of distance bins.
    :param fitted_semivariance: Fitted values evaluated at ``lags``.
    :param model: Portable fitted model parameters.
    :param estimator: Name of the formula used to estimate semivariance.
    :param distance: Distance measure name.
    :param binning: Method used to build distance bins.
    :param backend: Package used for fitting or import.
    :param backend_object: Optional retained object from that package.
    :param fit_result: Small details returned by the fit.
    :param attrs: Additional serializable metadata.
    """

    lags: NDArrayNum
    semivariance: NDArrayNum
    counts: NDArrayNum
    semivariance_error: NDArrayNum | None = None
    bin_lower_edges: NDArrayNum | None = None
    bin_edges: NDArrayNum | None = None
    fitted_semivariance: NDArrayNum | None = None
    model: VariogramModel | None = None
    estimator: str | None = None
    distance: str | None = None
    binning: str | None = None
    backend: str | None = None
    backend_object: Any = field(default=None, repr=False, compare=False)
    fit_result: Mapping[str, Any] = field(default_factory=dict, compare=False)
    attrs: Mapping[str, Any] = field(default_factory=dict, compare=False)

    ###################
    # RESULT VALIDATION
    ###################

    def __post_init__(self) -> None:
        # Copy required distance bin arrays so the result owns its small stored data
        lags = np.asarray(self.lags, dtype=float).copy()
        semivariance = np.asarray(self.semivariance, dtype=float).copy()
        counts = np.asarray(self.counts, dtype=np.int64).copy()
        if lags.ndim != 1 or semivariance.ndim != 1 or counts.ndim != 1:
            raise ValueError("Variogram lag statistics must be one-dimensional.")
        if not (len(lags) == len(semivariance) == len(counts)):
            raise ValueError("Variogram lags, semivariances and counts must have equal lengths.")
        if np.any(counts < 0):
            raise ValueError("Variogram pair counts cannot be negative.")

        # Check optional arrays against the same number of distance bins
        optional_arrays: dict[str, NDArrayNum | None] = {
            "semivariance_error": self.semivariance_error,
            "bin_lower_edges": self.bin_lower_edges,
            "bin_edges": self.bin_edges,
            "fitted_semivariance": self.fitted_semivariance,
        }
        normalized: dict[str, NDArrayNum | None] = {}
        for name, values in optional_arrays.items():
            if values is None:
                normalized[name] = None
                continue
            array = np.asarray(values, dtype=float).copy()
            if array.ndim != 1 or len(array) != len(lags):
                raise ValueError(f"Argument ``{name}`` must be one-dimensional and aligned with ``lags``.")
            normalized[name] = array

        # Make result arrays read-only so later edits cannot disagree with the fitted model
        for array in (lags, semivariance, counts, *[value for value in normalized.values() if value is not None]):
            array.setflags(write=False)
        object.__setattr__(self, "lags", lags)
        object.__setattr__(self, "semivariance", semivariance)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "semivariance_error", normalized["semivariance_error"])
        object.__setattr__(self, "bin_lower_edges", normalized["bin_lower_edges"])
        object.__setattr__(self, "bin_edges", normalized["bin_edges"])
        object.__setattr__(self, "fitted_semivariance", normalized["fitted_semivariance"])
        object.__setattr__(self, "fit_result", dict(self.fit_result))
        object.__setattr__(self, "attrs", dict(self.attrs))

    ############################
    # CONSTRUCTION AND FITTING
    ############################

    @classmethod
    def estimate(
        cls,
        coordinates: NDArrayNum,
        values: NDArrayNum,
        *,
        model: str = "spherical",
        active_dims: tuple[int, ...] | None = None,
        keep_backend: bool = False,
        **kwargs: Any,
    ) -> Variogram:
        """
        Estimate a variogram from available point coordinates using SciKit-GStat.

        :param coordinates: Observation coordinates arranged by row.
        :param values: One value per observation.
        :param model: SciKit-GStat theoretical model to fit.
        :param active_dims: Feature columns used by later covariance conversion.
        :param keep_backend: Whether to retain the SciKit-GStat object, which can use substantial memory.
        :param kwargs: Additional SciKit-GStat Variogram options.
        :returns: Per-distance measurements and fitted parameters without the source object by default.
        """

        # Import SciKit-GStat only for this direct coordinate-based estimate
        skgstat = import_optional("skgstat", package_name="scikit-gstat", extra_name="geostat")

        # Put coordinates into one row per observation and check value length
        coordinates_array = np.asarray(coordinates, dtype=float)
        values_array = np.asarray(values, dtype=float).squeeze()
        if coordinates_array.ndim == 1:
            coordinates_array = coordinates_array[:, np.newaxis]
        if coordinates_array.ndim != 2 or values_array.ndim != 1:
            raise ValueError(
                "Argument ``coordinates`` must be (observation, feature) and ``values`` must be one-dimensional."
            )
        if len(coordinates_array) != len(values_array):
            raise ValueError("Arguments ``coordinates`` and ``values`` must contain the same number of observations.")

        # Remove rows with missing coordinates or values before creating the SciKit-GStat object
        valid = np.isfinite(values_array) & np.all(np.isfinite(coordinates_array), axis=1)
        if np.count_nonzero(valid) < 2:
            raise ValueError("At least two finite observations are required to estimate a variogram.")
        backend_variogram = skgstat.Variogram(coordinates_array[valid], values_array[valid], model=model, **kwargs)

        # Copy the small distance bin result and discard the larger SciKit-GStat object by default
        return cls.from_skgstat(
            backend_variogram,
            active_dims=active_dims,
            keep_backend=keep_backend,
        )

    @classmethod
    def from_model(
        cls,
        model_name: str,
        effective_range: float,
        partial_sill: float,
        *,
        nugget: float = 0.0,
        smoothness: float | None = None,
        shape: float | None = None,
        active_dims: tuple[int, ...] | None = None,
    ) -> Variogram:
        """
        Create a variogram from known fitted parameters without measured values.

        :param model_name: Supported theoretical model name.
        :param effective_range: Distance at which the model effectively reaches its sill.
        :param partial_sill: Structured variance excluding the nugget.
        :param nugget: Uncorrelated variance.
        :param smoothness: Matérn smoothness parameter.
        :param shape: Stable model shape parameter.
        :param active_dims: Feature columns used by later covariance conversion.
        :returns: Variogram containing only fitted parameters.
        """

        # Use the standard Matérn smoothness when the caller supplies no value
        if model_name.lower() == "matern" and smoothness is None:
            smoothness = 1.5

        # Use empty measured arrays because this result contains only a supplied model
        return cls(
            lags=np.empty(0),
            semivariance=np.empty(0),
            counts=np.empty(0, dtype=np.int64),
            model=VariogramModel(
                model_name=model_name,
                effective_range=effective_range,
                partial_sill=partial_sill,
                nugget=nugget,
                smoothness=smoothness,
                shape=shape,
                active_dims=active_dims,
            ),
        )

    @classmethod
    def combine(
        cls,
        *variograms: Variogram,
        combination: str = "sum",
        nugget: float = 0.0,
    ) -> Variogram:
        """
        Combine fitted structures for covariance conversion.

        :param variograms: Two or more lightweight variograms with fitted models.
        :param combination: Either ``"sum"`` or ``"product"``.
        :param nugget: Shared uncorrelated variance.
        :returns: Lightweight variogram containing the composite model.
        """

        # Require fitted models because measured distance bins alone do not define covariance at every distance
        if len(variograms) < 2 or any(variogram.model is None for variogram in variograms):
            raise ValueError("At least two variograms with fitted models are required.")

        # Combine only fitted model parameters and leave out unrelated measured bins
        models = tuple(variogram.model for variogram in variograms if variogram.model is not None)
        return cls(
            lags=np.empty(0),
            semivariance=np.empty(0),
            counts=np.empty(0, dtype=np.int64),
            model=VariogramModel.combine(models, combination=combination, nugget=nugget),
        )

    @classmethod
    def from_pairs(
        cls,
        pairs: xr.Dataset,
        *,
        estimator: str | Callable[[NDArrayNum], float] = "dowd",
        bins: Literal["log", "uniform"] | Iterable[float] = "log",
        n_lags: int = 24,
        min_lag: float | None = None,
        max_lag: float | None = None,
    ) -> Variogram:
        """
        Calculate measured variogram values from a pair dataset.

        This method only reads the pair distances and endpoint values. The pair dataset can therefore be discarded as
        soon as the per-distance statistics have been computed.

        :param pairs: Dataset returned by Raster.pairsample() or PointCloud.pairsample().
        :param estimator: SciKit-GStat estimator name or a function accepting absolute pair differences.
        :param bins: ``"log"``, ``"uniform"`` or explicit distance boundaries.
        :param n_lags: Number of distance bins used for named binning.
        :param min_lag: Lower distance boundary. Defaults to the smallest positive pair distance.
        :param max_lag: Upper distance boundary. Defaults to the largest pair distance.
        :returns: Per-distance variogram values with no retained pair data.
        """

        # Check the expected Xarray pair layout before reading endpoint values
        required = {"distance", "value"}
        if not isinstance(pairs, xr.Dataset) or not required.issubset(pairs.data_vars):
            raise TypeError("Argument ``pairs`` must be an Xarray Dataset containing 'distance' and 'value'.")
        if pairs["value"].dims != ("pair", "endpoint") or pairs.sizes.get("endpoint") != 2:
            raise ValueError(
                "Variable 'value' in argument ``pairs`` must have dimensions ('pair', 'endpoint') of length two."
            )
        if pairs["distance"].dims != ("pair",):
            raise ValueError("Variable 'distance' in argument ``pairs`` must have dimensions ('pair',).")

        # Calculate the absolute value difference in each pair and remove missing pairs
        distances = np.asarray(pairs["distance"], dtype=float)
        endpoint_values = np.asarray(pairs["value"], dtype=float)
        differences = np.abs(endpoint_values[:, 0] - endpoint_values[:, 1])
        valid = np.isfinite(distances) & np.isfinite(differences) & (distances > 0)
        distances, differences = distances[valid], differences[valid]
        if distances.size == 0:
            raise ValueError("Argument ``pairs`` contains no finite observations with positive distance.")

        # Build log-spaced or equal-width bins, or check the caller's exact bin edges
        binning: str
        if isinstance(bins, str):
            # Use the requested sampling limits when available so repeated runs share the same bins
            minimum = float(pairs.attrs.get("min_distance", np.min(distances))) if min_lag is None else float(min_lag)
            maximum = float(pairs.attrs.get("max_distance", np.max(distances))) if max_lag is None else float(max_lag)
            if not 0 < minimum < maximum:
                raise ValueError("Require 0 < ``min_lag`` < ``max_lag``.")
            if n_lags < 1 or bins not in {"log", "uniform"}:
                raise ValueError("Argument ``bins`` must be 'log' or 'uniform', with ``n_lags`` at least one.")
            edges = (
                np.geomspace(minimum, maximum, n_lags + 1)
                if bins == "log"
                else np.linspace(minimum, maximum, n_lags + 1)
            )
            binning = bins
        else:
            edges = np.asarray(tuple(bins), dtype=float)
            if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
                raise ValueError("Argument ``bins`` must contain at least two increasing lag boundaries.")
            binning = "explicit"

        # Choose the named semivariance formula or use the caller's function
        if callable(estimator):
            estimator_function = estimator
            estimator_name = getattr(estimator, "__name__", "callable")
        else:
            skgstat = import_optional("skgstat", package_name="scikit-gstat", extra_name="geostat")
            if not hasattr(skgstat.estimators, estimator):
                raise ValueError(f"Unknown SciKit-GStat ``estimator`` {estimator!r}.")
            estimator_function = getattr(skgstat.estimators, estimator)
            estimator_name = estimator

        # Assign each pair to one distance bin, including both outer edges
        membership = np.digitize(distances, edges, right=True) - 1
        membership[distances == edges[0]] = 0
        experimental = np.full(len(edges) - 1, np.nan, dtype=float)
        counts = np.zeros(len(edges) - 1, dtype=np.int64)
        lag_centers = np.full(len(edges) - 1, np.nan, dtype=float)

        # Sort once so each estimator receives a contiguous bin without scanning all pairs again
        order = np.argsort(membership, kind="stable")
        sorted_membership = membership[order]
        boundaries = np.searchsorted(sorted_membership, np.arange(len(edges)))
        sorted_distances, sorted_differences = distances[order], differences[order]
        for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            counts[index] = stop - start
            if stop > start:
                # Keep the original order within each bin, including for user supplied estimators
                experimental[index] = float(estimator_function(sorted_differences[start:stop]))
                lag_centers[index] = float(np.mean(sorted_distances[start:stop]))

        # Return only per-bin arrays and plain source details that can be saved
        return cls(
            lags=lag_centers,
            semivariance=experimental,
            counts=counts,
            semivariance_error=np.full(len(experimental), np.nan),
            bin_lower_edges=edges[:-1],
            bin_edges=edges[1:],
            estimator=estimator_name,
            distance="euclidean",
            binning=binning,
            attrs={**pairs.attrs, "pair_count": int(np.sum(counts))},
        )

    def fit(
        self,
        models: str | Callable[..., Any] | Sequence[str | Callable[..., Any]] = "spherical",
        *,
        use_nugget: bool = False,
        bounds: Sequence[tuple[float, float]] | None = None,
        p0: Sequence[float] | None = None,
        maxfev: int | None = None,
    ) -> Variogram:
        """
        Fit one or more summed theoretical models to the measured bins.

        Finite, positive sampling errors are used as weights. The returned copy retains only fitted parameters and
        the small covariance matrix produced by the optimizer.

        :param models: Model name, SciKit-GStat model function or sequence ordered from short to long range.
        :param use_nugget: Whether to fit a shared non-negative nugget.
        :param bounds: Lower and upper bound for every fitted parameter.
        :param p0: Initial parameter values in range/sill order, followed by optional model shape and nugget.
        :param maxfev: Maximum number of model evaluations.
        :returns: New variogram containing fitted model parameters in the shared form.
        """

        # Turn short names, model functions, or a sequence into standard model names
        requested_models: list[str | Callable[..., Any]] = []
        if isinstance(models, str):
            requested_models.extend(models.split("+"))
        elif callable(models):
            requested_models = [models]
        else:
            requested_models = list(models)
        aliases = {
            "cub": "cubic",
            "exp": "exponential",
            "gau": "gaussian",
            "mat": "matern",
            "sph": "spherical",
            "sta": "stable",
        }
        model_names = []
        for requested in requested_models:
            name = requested.strip().lower() if isinstance(requested, str) else getattr(requested, "__name__", "")
            model_names.append(aliases.get(name, name))
        if not model_names or any(name not in _BASE_MODELS for name in model_names):
            raise ValueError(f"Argument ``models`` must contain names from {sorted(_BASE_MODELS)}.")

        # Fit only bins with measured values and require more bins than fitted parameters
        valid = np.isfinite(self.lags) & np.isfinite(self.semivariance)
        if np.count_nonzero(valid) < 2:
            raise ValueError("At least two finite empirical lag classes are required for fitting.")

        skgstat = import_optional("skgstat", package_name="scikit-gstat", extra_name="geostat")
        parameter_counts = [3 if name in {"stable", "matern"} else 2 for name in model_names]

        def summed_model(distance: NDArrayNum, *parameters: float) -> NDArrayNum:
            # Add each component's parameters in the order expected by SciPy's curve fit
            values = np.zeros_like(np.asarray(distance, dtype=float))
            position = 0
            for name, count in zip(model_names, parameter_counts):
                values += getattr(skgstat.models, name)(distance, *parameters[position : position + count])
                position += count
            if use_nugget:
                values += np.where(np.asarray(distance) > 0, parameters[position], 0.0)
            return values

        # Choose starting range and variance values from the measured distances and semivariances
        maximum_lag = float(np.nanmax(self.lags[valid]))
        maximum_variance = float(np.nanmax(self.semivariance[valid]))
        if maximum_variance <= 0:
            maximum_variance = 1.0
        if p0 is None:
            guesses: list[float] = []
            for index, count in enumerate(parameter_counts, start=1):
                guesses.extend((index * maximum_lag / len(model_names), maximum_variance / len(model_names)))
                if count == 3:
                    guesses.append(1.0)
            if use_nugget:
                guesses.append(maximum_variance * 0.05)
            p0 = guesses

        # Check optional starting values and limits against the number of fitted parameters
        expected = sum(parameter_counts) + int(use_nugget)
        if len(p0) != expected:
            raise ValueError(f"Argument ``p0`` must contain {expected} parameters for the selected models.")

        # Keep fitted parameters nonnegative unless the caller supplies other limits
        if bounds is None:
            model_bounds: list[tuple[float, float]] = []
            for count in parameter_counts:
                model_bounds.extend(((float(np.finfo(float).eps), maximum_lag), (0.0, np.inf)))
                if count == 3:
                    model_bounds.append((float(np.finfo(float).eps), np.inf))
            if use_nugget:
                model_bounds.append((0.0, np.inf))
            bounds = model_bounds
        if len(bounds) != expected:
            raise ValueError(f"Argument ``bounds`` must contain {expected} lower/upper pairs for the selected models.")
        lower, upper = np.asarray(bounds, dtype=float).T

        # Give bins with smaller measured errors more influence when usable errors exist
        errors = None
        if self.semivariance_error is not None:
            candidate_errors = self.semivariance_error[valid]
            if np.any(np.isfinite(candidate_errors) & (candidate_errors > 0)):
                positive = np.isfinite(candidate_errors) & (candidate_errors > 0)
                replacement = float(np.nanmedian(candidate_errors[positive]))
                errors = np.where(positive, candidate_errors, replacement)

        # Fit all summed components together so their parameters can adjust to one another
        coefficients, covariance = curve_fit(
            summed_model,
            self.lags[valid],
            self.semivariance[valid],
            p0=np.asarray(p0, dtype=float),
            bounds=(lower, upper),
            sigma=errors,
            absolute_sigma=errors is not None,
            method="trf",
            maxfev=maxfev,
        )

        # Split SciPy's fitted numbers back into the shared component model objects
        components: list[VariogramModel] = []
        position = 0
        for name, count in zip(model_names, parameter_counts):
            parameters = coefficients[position : position + count]
            position += count
            components.append(
                VariogramModel(
                    model_name=name,
                    effective_range=float(parameters[0]),
                    partial_sill=float(parameters[1]),
                    smoothness=float(parameters[2]) if name == "matern" else None,
                    shape=float(parameters[2]) if name == "stable" else None,
                )
            )

        # Store the shared nugget on the single model or the combined parent model
        nugget = float(coefficients[position]) if use_nugget else 0.0
        fitted_model = (
            replace(components[0], nugget=nugget)
            if len(components) == 1
            else VariogramModel.sum(components, nugget=nugget)
        )

        # Return a new read-only result with fitted values and small fit details
        return replace(
            self,
            fitted_semivariance=fitted_model.variogram(self.lags),
            model=fitted_model,
            backend="scikit-gstat models/scipy fit",
            fit_result={"coefficients": coefficients.tolist(), "covariance": covariance.tolist()},
        )

    @classmethod
    def from_skgstat(
        cls,
        variogram: Any,
        *,
        active_dims: tuple[int, ...] | None = None,
        keep_backend: bool = False,
    ) -> Variogram:
        """
        Copy a small result from a fitted SciKit-GStat Variogram.

        :param variogram: Fitted SciKit-GStat Variogram object.
        :param active_dims: Feature columns used by later covariance conversion.
        :param keep_backend: Whether to retain the input object.
        :returns: Per-distance measurements and fitted parameters.
        """

        # Read SciKit-GStat's description once to get its standard model names and settings
        description = variogram.describe()
        configured_model = str(description.get("params", {}).get("model", description["model"])).lower()

        # Copy fitted numbers into the shared `VariogramModel` form
        model = _model_from_skgstat(
            variogram,
            configured_model,
            description,
            active_dims=active_dims,
        )

        # Copy per-bin measurements and fitted values from SciKit-GStat
        lag_centers, experimental = variogram.get_empirical(bin_center=True)
        lag_centers = np.asarray(lag_centers, dtype=float)
        fitted = np.asarray(variogram.fitted_model(lag_centers), dtype=float)
        fit_result = {
            key: description.get(key)
            for key in ("normalized_effective_range", "normalized_sill", "normalized_nugget")
            if key in description
        }

        # Keep the full SciKit-GStat object only when the caller requests it
        return cls(
            lags=lag_centers,
            semivariance=np.asarray(experimental, dtype=float),
            counts=np.asarray(variogram.bin_count, dtype=np.int64),
            semivariance_error=np.full(len(lag_centers), np.nan),
            bin_lower_edges=np.r_[0.0, np.asarray(variogram.bins, dtype=float)[:-1]],
            bin_edges=np.asarray(variogram.bins, dtype=float),
            fitted_semivariance=fitted,
            model=model,
            estimator=str(description["estimator"]),
            distance=str(description["dist_func"]),
            binning=str(description.get("params", {}).get("bin_func", "unknown")),
            backend="skgstat",
            backend_object=variogram if keep_backend else None,
            fit_result=fit_result,
        )

    ################################
    # REPRESENTATION AND STORAGE
    ################################

    def without_backend(self) -> Variogram:
        """Return a copy that releases any retained fitting package object."""

        return replace(self, backend_object=None)

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per distance bin in a Pandas DataFrame."""

        # Add required distance bin columns first, then optional error, edge, and fitted columns
        data: dict[str, Any] = {
            "lag": self.lags,
            "semivariance": self.semivariance,
            "count": self.counts,
        }
        if self.semivariance_error is not None:
            data["semivariance_error"] = self.semivariance_error
        if self.bin_lower_edges is not None:
            data["bin_lower_edge"] = self.bin_lower_edges
        if self.bin_edges is not None:
            data["bin_edge"] = self.bin_edges
        if self.fitted_semivariance is not None:
            data["fitted_semivariance"] = self.fitted_semivariance
        return pd.DataFrame(data)

    def to_xarray(self) -> xr.Dataset:
        """Return labelled lag statistics in an Xarray Dataset."""

        # Store all measured arrays on one labelled distance bin dimension
        data_vars: dict[str, Any] = {
            "semivariance": ("lag", self.semivariance),
            "count": ("lag", self.counts),
        }
        if self.semivariance_error is not None:
            data_vars["semivariance_error"] = ("lag", self.semivariance_error)
        if self.bin_lower_edges is not None:
            data_vars["bin_lower_edge"] = ("lag", self.bin_lower_edges)
        if self.bin_edges is not None:
            data_vars["bin_edge"] = ("lag", self.bin_edges)
        if self.fitted_semivariance is not None:
            data_vars["fitted_semivariance"] = ("lag", self.fitted_semivariance)

        # Store model and method descriptions as plain attributes that survive Xarray save and load
        attrs = {
            **self.attrs,
            "estimator": self.estimator or "",
            "distance": self.distance or "",
            "binning": self.binning or "",
            "backend": self.backend or "",
        }
        if self.model is not None:
            attrs["model"] = json.dumps(self.model.to_dict())
        return xr.Dataset(data_vars=data_vars, coords={"lag": self.lags}, attrs=attrs)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation without backend state."""

        # Convert arrays and nested models to lists and dictionaries accepted by JSON
        return {
            "lags": self.lags.tolist(),
            "semivariance": self.semivariance.tolist(),
            "counts": self.counts.tolist(),
            "semivariance_error": (None if self.semivariance_error is None else self.semivariance_error.tolist()),
            "bin_lower_edges": None if self.bin_lower_edges is None else self.bin_lower_edges.tolist(),
            "bin_edges": None if self.bin_edges is None else self.bin_edges.tolist(),
            "fitted_semivariance": None if self.fitted_semivariance is None else self.fitted_semivariance.tolist(),
            "model": None if self.model is None else self.model.to_dict(),
            "estimator": self.estimator,
            "distance": self.distance,
            "binning": self.binning,
            "backend": self.backend,
            "fit_result": dict(self.fit_result),
            "attrs": dict(self.attrs),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Variogram:
        """
        Restore a lightweight variogram from Variogram.to_dict() output.

        :param values: Serialized per-distance measurements, optional fitted model, and metadata.
        :returns: Variogram containing the restored measurements and model, without an external backend object.
        """

        # Restore the optional model before the dataclass checks all per-bin arrays
        model_values = values.get("model")
        return cls(
            lags=np.asarray(values["lags"], dtype=float),
            semivariance=np.asarray(values["semivariance"], dtype=float),
            counts=np.asarray(values["counts"], dtype=np.int64),
            semivariance_error=(
                None
                if values.get("semivariance_error") is None
                else np.asarray(values["semivariance_error"], dtype=float)
            ),
            bin_lower_edges=(
                None if values.get("bin_lower_edges") is None else np.asarray(values["bin_lower_edges"], dtype=float)
            ),
            bin_edges=None if values.get("bin_edges") is None else np.asarray(values["bin_edges"], dtype=float),
            fitted_semivariance=(
                None
                if values.get("fitted_semivariance") is None
                else np.asarray(values["fitted_semivariance"], dtype=float)
            ),
            model=None if model_values is None else VariogramModel.from_dict(model_values),
            estimator=values.get("estimator"),
            distance=values.get("distance"),
            binning=values.get("binning"),
            backend=values.get("backend"),
            fit_result=values.get("fit_result", {}),
            attrs=values.get("attrs", {}),
        )

    ########################
    # EVALUATION AND PLOTTING
    ########################

    def variogram(self, distance: NDArrayNum | float) -> NDArrayNum:
        """
        Evaluate the fitted theoretical variogram.

        :param distance: Spatial distance or array of distances.
        :returns: Semivariance at each distance.
        """

        if self.model is None:
            raise ValueError("A fitted variogram model is required for evaluation.")
        return self.model.variogram(distance)

    __call__ = variogram

    def covariance(self, distance: NDArrayNum | float) -> NDArrayNum:
        """
        Evaluate covariance implied by the fitted model.

        :param distance: Spatial distance or array of distances.
        :returns: Covariance at each distance.
        """

        if self.model is None:
            raise ValueError("A fitted variogram model is required for evaluation.")
        return self.model.covariance(distance)

    def correlation(self, distance: NDArrayNum | float) -> NDArrayNum:
        """
        Evaluate correlation implied by the fitted model.

        :param distance: Spatial distance or array of distances.
        :returns: Correlation at each distance.
        """

        if self.model is None:
            raise ValueError("A fitted variogram model is required for evaluation.")
        return self.model.correlation(distance)

    def plot(self, ax: Any | None = None, *, show_error: bool = True, **kwargs: Any) -> Any:
        """
        Plot measured bins and the fitted model when present.

        :param ax: Existing Matplotlib axes. A new figure and axes are created by default.
        :param show_error: Whether to draw available sampling errors.
        :param kwargs: Keyword arguments passed to the measured point plot.
        :returns: Matplotlib axes containing the variogram.
        """

        # Import Matplotlib only when the caller requests a plot
        pyplot = import_optional("matplotlib.pyplot", package_name="matplotlib")
        if ax is None:
            _, ax = pyplot.subplots()

        # Draw measured bins and optional error bars, then add the fitted curve
        error = self.semivariance_error if show_error else None
        ax.errorbar(self.lags, self.semivariance, yerr=error, fmt="o", **kwargs)
        if self.model is not None and np.any(np.isfinite(self.lags)):
            distances = np.linspace(0, float(np.nanmax(self.lags)), 500)
            ax.plot(distances, self.variogram(distances))
        ax.set(xlabel="Lag distance", ylabel="Semivariance")
        return ax

    ############################
    # CONVERSIONS TO OTHER PACKAGES
    ############################

    def to_gstools(self, *, dim: int = 2) -> GSToolsVariogram:
        """
        Convert the fitted model to GSTools with its source feature dimensions.

        :param dim: Number of dimensions passed to the GSTools covariance model.
        :returns: Native covariance model and dimensions selected by the source model.
        """

        # Import GSTools only for this conversion and require a fitted model
        gstools = import_optional("gstools", extra_name="geostat")
        if self.model is None:
            raise ValueError("A fitted variogram model is required for conversion.")
        if dim <= 0:
            raise ValueError("GSTools model dimension must be strictly positive.")

        # Require all components to use the same coordinate columns expected by one GSTools model
        active_dims = {component.active_dims for component in self.model.components} if self.model.components else set()
        if len(active_dims) > 1:
            raise NotImplementedError("GSTools cannot combine variogram components that select different dimensions.")

        # Return coordinate-column choices separately because GSTools does not store them
        selected_dims = next(iter(active_dims)) if active_dims else self.model.active_dims
        return GSToolsVariogram(
            model=_model_to_gstools(self.model, gstools=gstools, dim=dim),
            active_dims=selected_dims,
        )

    def gpytorch_parameters(self) -> dict[str, Any]:
        """Describe the fitted model with plain parameters used by the GPyTorch conversion."""

        if self.model is None:
            raise ValueError("A fitted variogram model is required for conversion.")
        return _model_to_gpytorch_parameters(self.model)

    def to_gpytorch(self, *, active_dims: tuple[int, ...] | None = None, trainable: bool = True) -> GPyTorchVariogram:
        """
        Convert supported fitted structures to a GPyTorch covariance kernel.

        :param active_dims: Optional feature column override applied to every structure.
        :param trainable: Whether converted kernel parameters may be optimized.
        :returns: Native covariance kernel and separate likelihood noise.
        """

        # Import GPyTorch only for this conversion and require a fitted model
        gpytorch = import_optional("gpytorch", extra_name="gp")
        if self.model is None:
            raise ValueError("A fitted variogram model is required for conversion.")
        kernel, noise = _model_to_gpytorch(
            self.model,
            gpytorch=gpytorch,
            active_dims=active_dims,
        )

        # Make kernel parameters fixed when the caller will use the model only for prediction
        if not trainable:
            for parameter in kernel.parameters():
                parameter.requires_grad_(False)
        return GPyTorchVariogram(kernel=kernel, noise=noise)


############################
# 4/ REPEATED PAIR ESTIMATION
############################


def _estimate_variogram(
    source: RasterBase | PointCloudBase,
    *,
    n_runs: int,
    estimator: str | Callable[[NDArrayNum], float],
    bins: Literal["log", "uniform"] | Iterable[float],
    n_lags: int,
    min_lag: float | None,
    max_lag: float | None,
    models: str | Callable[..., Any] | Sequence[str | Callable[..., Any]] | None,
    fit_kwargs: Mapping[str, Any] | None,
    random_state: int | np.random.Generator | None,
    pair_kwargs: Mapping[str, Any],
) -> Variogram:
    """
    Sample one or more pair sets and combine their per-distance variogram values.

    Each run calls the source pairsample() method, then Variogram.from_pairs() groups value differences by distance.
    Repeated runs share the first run's bins and are combined before Variogram.fit() fits any requested model.

    Source, estimation, binning, repetition, fitting, and random-state options are documented by
    geoutils.stats.variogram().

    :param models: Public model argument forwarded to Variogram.fit(): one model or a sequence of models to sum,
        or None to keep only the measured variogram.
    :param pair_kwargs: Complete options for source.pairsample(), including sample size, mask, distance limits,
        and any source-specific controls. This function supplies a separate random_state for each run.
    :returns: Variogram with mean measurements and total pair counts across runs, sampling errors when repeated,
        and any requested fitted model.
    """

    # Check the repeat count before creating one random seed per run
    if not isinstance(n_runs, (int, np.integer)) or isinstance(n_runs, bool) or n_runs < 1:
        raise ValueError("Argument ``n_runs`` must be a positive integer.")
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    seeds = rng.integers(0, np.iinfo(np.int32).max, n_runs)

    # Materialize explicit boundaries once because a generator cannot be read by every repeated run
    if not isinstance(bins, str):
        bins = tuple(bins)

    def run(seed: np.integer[Any], bin_spec: Literal["log", "uniform"] | Iterable[float]) -> Variogram:
        """
        Calculate one run with its own seed and either the initial or shared distance bins.

        :param seed: Integer seed for this run's pair sample.
        :param bin_spec: Initial binning choice, or fixed boundaries copied from the first run.
        """

        # Convert one pair sample to per-distance values and then release the pairs
        pairs = source.pairsample(random_state=int(seed), **pair_kwargs)
        return Variogram.from_pairs(
            pairs,
            estimator=estimator,
            bins=bin_spec,
            n_lags=n_lags,
            min_lag=min_lag,
            max_lag=max_lag,
        )

    # Build distance bins from the first run so all later runs use the same edges
    first = run(seeds[0], bins)
    if n_runs == 1:
        result = replace(first, attrs={**first.attrs, "n_runs": 1})
        return result if models is None else result.fit(models, **dict(fit_kwargs or {}))

    # Draw each remaining sample through the object's own raster or point implementation
    shared_bins: Literal["log", "uniform"] | Iterable[float] = bins
    if isinstance(bins, str):
        assert first.bin_lower_edges is not None and first.bin_edges is not None
        shared_bins = np.r_[first.bin_lower_edges[0], first.bin_edges]

    runs = [first, *[run(seed, shared_bins) for seed in seeds[1:]]]

    # Stack the small per-distance arrays from all runs
    semivariances = np.vstack([run_result.semivariance for run_result in runs])
    lag_centers = np.vstack([run_result.lags for run_result in runs])
    counts = np.sum(np.vstack([run_result.counts for run_result in runs]), axis=0)
    mean_semivariance = np.full(semivariances.shape[1], np.nan)
    mean_lags = np.full(semivariances.shape[1], np.nan)
    errors = np.full(semivariances.shape[1], np.nan)

    # Average each distance bin and calculate sampling error only when runs are repeated
    for index in range(semivariances.shape[1]):
        finite_semivariance = semivariances[np.isfinite(semivariances[:, index]), index]
        finite_lags = lag_centers[np.isfinite(lag_centers[:, index]), index]
        if finite_semivariance.size:
            mean_semivariance[index] = np.mean(finite_semivariance)
        if finite_semivariance.size > 1:
            errors[index] = np.std(finite_semivariance, ddof=1) / np.sqrt(finite_semivariance.size)
        if finite_lags.size:
            mean_lags[index] = np.mean(finite_lags)

    # Reuse the first run's bin details with the combined values and counts
    result = replace(
        first,
        lags=mean_lags,
        semivariance=mean_semivariance,
        semivariance_error=errors,
        counts=counts,
        attrs={**first.attrs, "n_runs": n_runs, "pair_count": int(np.sum(counts))},
    )
    return result if models is None else result.fit(models, **dict(fit_kwargs or {}))


################################
# 5/ HELPERS FOR OTHER PACKAGES
################################


def _model_from_skgstat(
    variogram: Any,
    configured_model: str,
    description: Mapping[str, Any],
    *,
    active_dims: tuple[int, ...] | None,
) -> VariogramModel:
    """
    Copy SciKit-GStat fitted numbers into the shared `VariogramModel` form.

    The fitted variogram and active_dims arguments are described by Variogram.from_skgstat().

    :param configured_model: Lowercase SciKit-GStat model name, with summed components separated by plus signs.
    :param description: Model parameters and settings returned by the fitted SciKit-GStat object's describe().
    :returns: Portable fitted model, preserving component order and the shared nugget for summed structures.
    """

    # Read a single model's range, sill, shape, and nugget from SciKit-GStat's description
    if "+" not in configured_model:
        return VariogramModel(
            model_name=configured_model,
            effective_range=float(description["effective_range"]),
            partial_sill=float(description["sill"]),
            nugget=float(description["nugget"]),
            smoothness=None if description.get("smoothness") is None else float(description["smoothness"]),
            shape=None if description.get("shape") is None else float(description["shape"]),
            active_dims=active_dims,
        )

    # Read one shared nugget after all summed component parameters
    names = [name.strip() for name in configured_model.split("+")]
    coefficients = list(np.asarray(variogram.cof, dtype=float))
    use_nugget = bool(description.get("params", {}).get("use_nugget", False))
    nugget = float(coefficients.pop()) if use_nugget else 0.0
    components: list[VariogramModel] = []
    position = 0
    for name in names:
        # Read each component's range and sill before its optional shape value
        effective_range, partial_sill = coefficients[position : position + 2]
        position += 2
        smoothness = shape = None
        if name == "matern":
            smoothness = float(coefficients[position])
            position += 1
        elif name == "stable":
            shape = float(coefficients[position])
            position += 1
        components.append(
            VariogramModel(
                model_name=name,
                effective_range=float(effective_range),
                partial_sill=float(partial_sill),
                smoothness=smoothness,
                shape=shape,
                active_dims=active_dims,
            )
        )

    # Build the shared summed model with one nugget on its parent
    return VariogramModel.sum(components, nugget=nugget)


def _model_to_gstools(model: VariogramModel, *, gstools: Any, dim: int) -> Any:
    """
    Build the equivalent GSTools model from one shared `VariogramModel`.

    The dimension argument is described by Variogram.to_gstools().

    :param model: Portable fitted model to convert, including any summed components.
    :param gstools: GSTools module already imported by Variogram.to_gstools().
    :returns: Native GSTools covariance model with the fitted range, variance, and nugget.
    """

    # Convert each summed component, then apply the shared nugget once
    if model.model_name == "sum":
        components = [_model_to_gstools(component, gstools=gstools, dim=dim) for component in model.components]
        return gstools.SumModel(*components, nugget=model.nugget)
    if model.model_name == "product":
        raise NotImplementedError("Product covariance conversion is not supported by the GSTools adapter.")
    if model.effective_range is None or model.partial_sill is None:
        raise AssertionError("A base variogram model must define its range and partial sill.")

    # Pass the shared model's variance and effective range to the matching GSTools class
    common = {"dim": dim, "var": model.partial_sill, "nugget": model.nugget, "len_scale": model.effective_range}
    if model.model_name == "spherical":
        return gstools.Spherical(**common)
    if model.model_name == "exponential":
        return gstools.Exponential(rescale=3.0, **common)
    if model.model_name == "gaussian":
        return gstools.Gaussian(rescale=2.0, **common)
    if model.model_name == "cubic":
        return gstools.Cubic(**common)
    if model.model_name == "stable":
        if model.shape is None:
            raise AssertionError("A stable variogram model must define its shape.")
        return gstools.Stable(alpha=model.shape, rescale=float(3 ** (1 / model.shape)), **common)

    # Pass Matérn smoothness through GSTools' matching parameter name
    if model.model_name == "matern":
        return gstools.Matern(nu=model.smoothness, rescale=4.0, **common)
    raise NotImplementedError(f"Variogram model {model.model_name!r} has no GSTools adapter.")


def _model_to_gpytorch(
    model: VariogramModel, *, gpytorch: Any, active_dims: tuple[int, ...] | None
) -> tuple[Any, float]:
    """
    Build an equivalent GPyTorch kernel for models both packages represent exactly.

    The active_dims override is described by Variogram.to_gpytorch().

    :param model: Portable fitted model to convert, including any summed or multiplied components.
    :param gpytorch: GPyTorch module already imported by Variogram.to_gpytorch().
    :returns: Native GPyTorch covariance kernel and separate nugget variance for the observation likelihood.
    """

    # Convert nested components and join them by the model's sum or product rule
    if model.model_name == "sum":
        converted = [
            _model_to_gpytorch(component, gpytorch=gpytorch, active_dims=active_dims) for component in model.components
        ]
        kernel = converted[0][0]
        for component_kernel, _ in converted[1:]:
            kernel = kernel + component_kernel
        return kernel, model.nugget

    if model.model_name == "product":
        converted = [
            _model_to_gpytorch(component, gpytorch=gpytorch, active_dims=active_dims) for component in model.components
        ]
        kernel = converted[0][0]
        for component_kernel, _ in converted[1:]:
            kernel = kernel * component_kernel
        return kernel, model.nugget

    # Convert effective range to the matching GPyTorch kernel length scale
    parameters = _model_to_gpytorch_parameters(model, active_dims=active_dims)
    resolved_active_dims = parameters["active_dims"]
    if model.partial_sill is None:
        raise AssertionError("A base variogram model must define its partial sill.")

    if model.model_name == "gaussian":
        base_kernel = gpytorch.kernels.RBFKernel(active_dims=resolved_active_dims)
    elif model.model_name == "exponential":
        base_kernel = gpytorch.kernels.MaternKernel(nu=0.5, active_dims=resolved_active_dims)
    elif model.model_name == "matern":
        base_kernel = gpytorch.kernels.MaternKernel(nu=model.smoothness, active_dims=resolved_active_dims)
    else:
        raise NotImplementedError(f"Variogram model {model.model_name!r} has no exact GPyTorch adapter.")

    # Apply the model variance around GPyTorch's base correlation kernel
    kernel = gpytorch.kernels.ScaleKernel(base_kernel)
    kernel.base_kernel.lengthscale = parameters["lengthscale"]
    kernel.outputscale = float(model.partial_sill)
    return kernel, model.nugget


def _model_to_gpytorch_parameters(
    model: VariogramModel,
    *,
    active_dims: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """
    Describe the GPyTorch conversion with plain data without importing GPyTorch.

    Model and active_dims inputs follow _model_to_gpytorch().

    :returns: Kernel name, length scale, variance, selected dimensions, and nugget; nested component dictionaries
        preserve summed or multiplied structures.
    """

    # Describe nested models recursively so callers can inspect them without GPyTorch
    if model.model_name in _COMPOSITE_MODELS:
        return {
            "combination": model.model_name,
            "components": [
                _model_to_gpytorch_parameters(component, active_dims=active_dims) for component in model.components
            ],
            "noise": model.nugget,
        }
    if model.effective_range is None or model.partial_sill is None:
        raise AssertionError("A base variogram model must define its range and partial sill.")

    # Convert effective range to the matching GPyTorch kernel length scale
    if model.model_name == "gaussian":
        kernel_name = "RBF"
        lengthscale = float(model.effective_range) / (2 * np.sqrt(2))
        smoothness = None
    elif model.model_name == "exponential":
        kernel_name = "Matern"
        lengthscale = float(model.effective_range) / 3
        smoothness = 0.5
    elif model.model_name == "matern":
        if model.smoothness not in (0.5, 1.5, 2.5):
            raise NotImplementedError("GPyTorch Matérn kernels support smoothness values 0.5, 1.5 and 2.5.")
        kernel_name = "Matern"
        lengthscale = float(model.effective_range) / (2 * np.sqrt(2))
        smoothness = model.smoothness
    else:
        raise NotImplementedError(f"Variogram model {model.model_name!r} has no exact GPyTorch adapter.")

    # Keep observation noise separate because GPyTorch applies it in the likelihood
    return {
        "kernel_name": kernel_name,
        "lengthscale": lengthscale,
        "outputscale": float(model.partial_sill),
        "smoothness": smoothness,
        "active_dims": model.active_dims if active_dims is None else active_dims,
        "noise": model.nugget,
    }
