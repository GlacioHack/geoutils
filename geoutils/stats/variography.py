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

"""Variography module to manipulate a lightweight variogram object across backends (SciKit-GStat, GSTools, GPyTorch)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit

from geoutils._misc import import_optional
from geoutils._typing import NDArrayNum

__all__ = ["GPyTorchVariogram", "GSToolsVariogram", "Variogram", "VariogramModel"]


###########################
# 1/ PORTABLE MODEL METADATA
###########################

_BASE_MODELS = {"spherical", "exponential", "gaussian", "cubic", "stable", "matern"}
_COMPOSITE_MODELS = {"sum", "product"}


@dataclass(frozen=True)
class VariogramModel:
    """Parameters of a fitted theoretical variogram independent of backend.

    ``effective_range`` follows SciKit-GStat's convention. ``partial_sill`` excludes the nugget, making the conversion
    to covariance kernels unambiguous. A composite model holds its independent structures in ``components`` and
    keeps their shared nugget on the parent model.

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
        # Normalize accepted aliases so every conversion uses canonical model names
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

        # Convert numeric metadata once so serialized and fitted models behave alike
        for name in ("effective_range", "partial_sill", "nugget", "smoothness", "shape"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))
        if self.active_dims is not None:
            object.__setattr__(self, "active_dims", tuple(int(value) for value in self.active_dims))
        if self.partial_sill is not None and (not np.isfinite(self.partial_sill) or self.partial_sill < 0):
            raise ValueError("Variogram partial sill must be a finite, non-negative number.")

        # Validate composite structures and derive their combined structured sill
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
            # Require the parameters used by each supported base model
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

        # Validate shared scalar metadata after model-specific normalization
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
            raise ValueError("active_dims must contain unique, non-negative dimensions.")

    ####################
    # MODEL EVALUATION
    ####################

    @property
    def sill(self) -> float:
        """Total sill, including the nugget."""

        return float((self.partial_sill or 0.0) + self.nugget)

    def variogram(self, distance: NDArrayNum | float) -> NDArrayNum:
        """Evaluate the theoretical variogram at one or more distances.

        :param distance: Spatial distance or array of distances.
        :returns: Semivariance at each distance.
        """

        # Preserve scalar return behavior while evaluating through one array path
        scalar_input = np.ndim(distance) == 0
        distances = np.atleast_1d(np.asarray(distance, dtype=float))

        # Add independent semivariances before applying the shared nugget
        if self.model_name == "sum":
            values = sum((component.variogram(distances) for component in self.components), np.zeros_like(distances))
            output = values + np.where(distances > 0, self.nugget, 0.0)
            return output[0] if scalar_input else output

        # Multiply component covariances to define product model semivariance
        if self.model_name == "product":
            if self.partial_sill is None:
                raise AssertionError("A product variogram model must define its partial sill.")
            covariance = np.ones_like(distances)
            for component in self.components:
                covariance *= component.covariance(distances)
            output = (float(self.partial_sill) - covariance) + np.where(distances > 0, self.nugget, 0.0)
            return output[0] if scalar_input else output

        # Guard normalized base parameters before calling backend model functions
        if self.effective_range is None or self.partial_sill is None:
            raise AssertionError("A base variogram model must define its range and partial sill.")

        # Build arguments in SciKit-GStat coefficient order for the selected model
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

        # Evaluate structured semivariance and add nugget only at positive distances
        values = np.asarray(model_function(distances.ravel(), *arguments), dtype=float).reshape(distances.shape)
        output = values + np.where(distances > 0, self.nugget, 0.0)
        return output[0] if scalar_input else output

    def covariance(self, distance: NDArrayNum | float) -> NDArrayNum:
        """Evaluate covariance implied by this variogram.

        :param distance: Spatial distance or array of distances.
        :returns: Covariance at each distance.
        """

        return self.sill - self.variogram(distance)

    def correlation(self, distance: NDArrayNum | float) -> NDArrayNum:
        """Evaluate correlation implied by this variogram.

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
        """Combine independent nested structures into a summed variogram model.

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
        """Combine independently parameterized components by addition or multiplication.

        :param components: Fitted structures to combine.
        :param combination: Either ``"sum"`` or ``"product"``.
        :param nugget: Shared uncorrelated variance.
        :returns: Composite model with normalized components.
        """

        if combination not in _COMPOSITE_MODELS:
            raise ValueError("combination must be 'sum' or 'product'.")

        # Flatten like compositions and move every component nugget to the parent
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

        # Serialize components recursively so compositions remain portable
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
        """Restore a model from :meth:`to_dict` output.

        :param values: Serialized model fields.
        :returns: Restored fitted model.
        """

        # Restore nested components before validating the parent model fields
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


###############################
# 2/ BACKEND RESULT CONTAINERS
###############################


@dataclass(frozen=True)
class GPyTorchVariogram:
    """A native GPyTorch covariance kernel and its separate observation noise nugget.

    :param kernel: Converted GPyTorch covariance kernel.
    :param noise: Nugget variance for the observation likelihood.
    """

    kernel: Any
    noise: float


@dataclass(frozen=True)
class GSToolsVariogram:
    """A native GSTools covariance model and its source feature dimensions.

    :param model: Converted GSTools covariance model.
    :param active_dims: Feature columns to select before passing coordinates to GSTools.
    """

    model: Any
    active_dims: tuple[int, ...] | None


##############################
# 3/ EMPIRICAL VARIOGRAM RESULT
##############################


@dataclass(frozen=True)
class Variogram:
    """Small, serializable empirical and fitted variogram result.

    The arrays have one value per lag class. ``backend_object`` is absent by default because a SciKit-GStat Variogram
    retains sampled coordinates, pairwise distances and pairwise differences. Pass ``keep_backend=True`` during
    estimation or conversion only when those advanced backend operations are worth the additional memory.

    :param lags: Mean sampled distance in each lag class.
    :param semivariance: Empirical semivariance in each lag class.
    :param counts: Number of sampled pairs in each lag class.
    :param semivariance_error: Sampling error estimated across independent runs.
    :param bin_lower_edges: Inclusive lower boundaries of lag classes.
    :param bin_edges: Upper boundaries of lag classes.
    :param fitted_semivariance: Fitted values evaluated at ``lags``.
    :param model: Portable fitted model parameters.
    :param estimator: Empirical estimator name.
    :param distance: Distance measure name.
    :param binning: Lag binning method.
    :param backend: Backend used for fitting or import.
    :param backend_object: Optional retained backend object.
    :param fit_result: Small optimizer diagnostics.
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
        # Copy required lag arrays so the result owns compact independent storage
        lags = np.asarray(self.lags, dtype=float).copy()
        semivariance = np.asarray(self.semivariance, dtype=float).copy()
        counts = np.asarray(self.counts, dtype=np.int64).copy()
        if lags.ndim != 1 or semivariance.ndim != 1 or counts.ndim != 1:
            raise ValueError("Variogram lag statistics must be one-dimensional.")
        if not (len(lags) == len(semivariance) == len(counts)):
            raise ValueError("Variogram lags, semivariances and counts must have equal lengths.")
        if np.any(counts < 0):
            raise ValueError("Variogram pair counts cannot be negative.")

        # Normalize optional arrays through the same lag alignment checks
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
                raise ValueError(f"Variogram {name} must be one-dimensional and aligned with lags.")
            normalized[name] = array

        # Freeze result arrays to prevent mutation from invalidating fitted metadata
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
        """Estimate a variogram from finite point coordinates using SciKit-GStat.

        :param coordinates: Observation coordinates arranged by row.
        :param values: One value per observation.
        :param model: SciKit-GStat theoretical model to fit.
        :param active_dims: Feature columns used by later covariance conversion.
        :param keep_backend: Whether to retain the SciKit-GStat object, which can use substantial memory.
        :param kwargs: Additional SciKit-GStat Variogram options.
        :returns: Lightweight empirical statistics and fitted parameters.
        """

        # Import SciKit-GStat only when direct coordinate fitting is requested
        skgstat = import_optional("skgstat", package_name="scikit-gstat", extra_name="geostat")

        # Normalize one-dimensional coordinates and reject incompatible shapes
        coordinates_array = np.asarray(coordinates, dtype=float)
        values_array = np.asarray(values, dtype=float).squeeze()
        if coordinates_array.ndim == 1:
            coordinates_array = coordinates_array[:, np.newaxis]
        if coordinates_array.ndim != 2 or values_array.ndim != 1:
            raise ValueError("Coordinates must be (observation, feature) and values must be one-dimensional.")
        if len(coordinates_array) != len(values_array):
            raise ValueError("Coordinates and values must contain the same number of observations.")

        # Remove invalid observations before constructing the temporary backend object
        valid = np.isfinite(values_array) & np.all(np.isfinite(coordinates_array), axis=1)
        if np.count_nonzero(valid) < 2:
            raise ValueError("At least two finite observations are required to estimate a variogram.")
        backend_variogram = skgstat.Variogram(coordinates_array[valid], values_array[valid], model=model, **kwargs)

        # Extract compact lag and fit metadata while discarding backend state by default
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
        """Create a lightweight variogram from known fitted parameters.

        :param model_name: Supported theoretical model name.
        :param effective_range: Distance at which the model effectively reaches its sill.
        :param partial_sill: Structured variance excluding the nugget.
        :param nugget: Uncorrelated variance.
        :param smoothness: Matérn smoothness parameter.
        :param shape: Stable model shape parameter.
        :param active_dims: Feature columns used by later covariance conversion.
        :returns: Lightweight variogram containing only fitted parameters.
        """

        # Supply the conventional Matérn default when only portable parameters are given
        if model_name.lower() == "matern" and smoothness is None:
            smoothness = 1.5

        # Represent model-only results with empty empirical arrays
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
        """Combine fitted structures for covariance conversion.

        :param variograms: Two or more lightweight variograms with fitted models.
        :param combination: Either ``"sum"`` or ``"product"``.
        :param nugget: Shared uncorrelated variance.
        :returns: Lightweight variogram containing the composite model.
        """

        # Require fitted inputs because empirical bins alone cannot define covariance
        if len(variograms) < 2 or any(variogram.model is None for variogram in variograms):
            raise ValueError("At least two variograms with fitted models are required.")

        # Combine only portable model metadata and omit unrelated empirical bins
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
        """Calculate an empirical variogram from a pair dataset.

        This method only reads the pair distances and endpoint values. The pair dataset can therefore be discarded as
        soon as the empirical lag statistics have been computed.

        :param pairs: Dataset returned by ``Raster.sample_pairs()`` or ``PointCloud.sample_pairs()``.
        :param estimator: SciKit-GStat estimator name or a function accepting absolute pair differences.
        :param bins: ``"log"``, ``"uniform"`` or explicit lag boundaries.
        :param n_lags: Number of lag classes used for named binning.
        :param min_lag: Lower lag boundary. Defaults to the smallest positive pair distance.
        :param max_lag: Upper lag boundary. Defaults to the largest pair distance.
        :returns: Lightweight empirical variogram with no retained pair data.
        """

        # Validate the labelled pair schema before reading endpoint arrays
        required = {"distance", "value"}
        if not isinstance(pairs, xr.Dataset) or not required.issubset(pairs.data_vars):
            raise TypeError("pairs must be an Xarray Dataset containing 'distance' and 'value'.")
        if pairs["value"].dims != ("pair", "endpoint") or pairs.sizes.get("endpoint") != 2:
            raise ValueError("pairs['value'] must have dimensions ('pair', 'endpoint') of length two.")

        # Compute absolute endpoint differences and discard invalid pair observations
        distances = np.asarray(pairs["distance"], dtype=float)
        endpoint_values = np.asarray(pairs["value"], dtype=float)
        differences = np.abs(endpoint_values[:, 0] - endpoint_values[:, 1])
        valid = np.isfinite(distances) & np.isfinite(differences) & (distances > 0)
        distances, differences = distances[valid], differences[valid]
        if distances.size == 0:
            raise ValueError("pairs contains no finite observations with positive distance.")

        # Prefer sampling bounds from metadata so repeated runs share stable bin limits
        minimum = float(pairs.attrs.get("min_distance", np.min(distances))) if min_lag is None else float(min_lag)
        maximum = float(pairs.attrs.get("max_distance", np.max(distances))) if max_lag is None else float(max_lag)
        if not 0 < minimum < maximum:
            raise ValueError("Require 0 < min_lag < max_lag.")

        # Build named bins or validate explicit boundaries supplied by the caller
        binning: str
        if isinstance(bins, str):
            if n_lags < 1 or bins not in {"log", "uniform"}:
                raise ValueError("Named bins must be 'log' or 'uniform', with n_lags at least one.")
            edges = (
                np.geomspace(minimum, maximum, n_lags + 1)
                if bins == "log"
                else np.linspace(minimum, maximum, n_lags + 1)
            )
            binning = bins
        else:
            edges = np.asarray(tuple(bins), dtype=float)
            if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
                raise ValueError("Explicit bins must contain at least two increasing lag boundaries.")
            binning = "explicit"

        # Resolve named estimators lazily while accepting custom reduction functions
        if callable(estimator):
            estimator_function = estimator
            estimator_name = getattr(estimator, "__name__", "callable")
        else:
            skgstat = import_optional("skgstat", package_name="scikit-gstat", extra_name="geostat")
            if not hasattr(skgstat.estimators, estimator):
                raise ValueError(f"Unknown SciKit-GStat estimator {estimator!r}.")
            estimator_function = getattr(skgstat.estimators, estimator)
            estimator_name = estimator

        # Digitize once to reduce every lag class without retaining pair membership
        membership = np.digitize(distances, edges, right=True) - 1
        membership[distances == edges[0]] = 0
        experimental = np.full(len(edges) - 1, np.nan, dtype=float)
        counts = np.zeros(len(edges) - 1, dtype=np.int64)
        lag_centers = np.full(len(edges) - 1, np.nan, dtype=float)
        for index in range(len(experimental)):
            selected = membership == index
            counts[index] = np.count_nonzero(selected)
            if counts[index]:
                experimental[index] = float(estimator_function(differences[selected]))
                lag_centers[index] = float(np.mean(distances[selected]))

        # Return only compact aggregate arrays and serializable source metadata
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
        """Fit one or more summed theoretical models to the empirical bins.

        Finite, positive sampling errors are used as weights. The returned copy retains only fitted parameters and
        the small covariance matrix produced by the optimizer.

        :param models: Model name, SciKit-GStat model function or sequence ordered from short to long range.
        :param use_nugget: Whether to fit a shared non-negative nugget.
        :param bounds: Lower and upper bound for every fitted parameter.
        :param p0: Initial parameter values in range/sill order, followed by optional model shape and nugget.
        :param maxfev: Maximum number of model evaluations.
        :returns: New variogram containing normalized fitted model metadata.
        """

        # Normalize strings, callables and sequences to canonical model names
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
            raise ValueError(f"models must contain names from {sorted(_BASE_MODELS)}.")

        # Fit only finite empirical lag classes and require enough points to constrain a curve
        valid = np.isfinite(self.lags) & np.isfinite(self.semivariance)
        if np.count_nonzero(valid) < 2:
            raise ValueError("At least two finite empirical lag classes are required for fitting.")

        skgstat = import_optional("skgstat", package_name="scikit-gstat", extra_name="geostat")
        parameter_counts = [3 if name in {"stable", "matern"} else 2 for name in model_names]

        def summed_model(distance: NDArrayNum, *parameters: float) -> NDArrayNum:
            # Accumulate each component in the same parameter order used by the optimizer
            values = np.zeros_like(np.asarray(distance, dtype=float))
            position = 0
            for name, count in zip(model_names, parameter_counts):
                values += getattr(skgstat.models, name)(distance, *parameters[position : position + count])
                position += count
            if use_nugget:
                values += np.where(np.asarray(distance) > 0, parameters[position], 0.0)
            return values

        # Derive scale aware initial values from observed lags and semivariance
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

        # Check parameter counts before sending arrays to SciPy
        expected = sum(parameter_counts) + int(use_nugget)
        if len(p0) != expected:
            raise ValueError(f"p0 must contain {expected} parameters for the selected models.")

        # Build positive default bounds while honoring explicitly supplied limits
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
            raise ValueError(f"bounds must contain {expected} lower/upper pairs for the selected models.")
        lower, upper = np.asarray(bounds, dtype=float).T

        # Use empirical sampling errors as weights when at least one is informative
        errors = None
        if self.semivariance_error is not None:
            candidate_errors = self.semivariance_error[valid]
            if np.any(np.isfinite(candidate_errors) & (candidate_errors > 0)):
                positive = np.isfinite(candidate_errors) & (candidate_errors > 0)
                replacement = float(np.nanmedian(candidate_errors[positive]))
                errors = np.where(positive, candidate_errors, replacement)

        # Fit the complete summed model once to preserve covariance among parameters
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

        # Split fitted coefficients back into portable component models
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

        # Store a shared nugget on either the base model or composite parent
        nugget = float(coefficients[position]) if use_nugget else 0.0
        fitted_model = (
            replace(components[0], nugget=nugget)
            if len(components) == 1
            else VariogramModel.sum(components, nugget=nugget)
        )

        # Return a new immutable result with predictions and small fit diagnostics
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
        """Extract a lightweight result from a fitted SciKit-GStat Variogram.

        :param variogram: Fitted SciKit-GStat Variogram object.
        :param active_dims: Feature columns used by later covariance conversion.
        :param keep_backend: Whether to retain the input object.
        :returns: Lightweight empirical statistics and fitted parameters.
        """

        # Read the backend description once because it normalizes fitted model metadata
        description = variogram.describe()
        configured_model = str(description.get("params", {}).get("model", description["model"])).lower()

        # Convert fitted coefficients to the portable model representation
        model = _model_from_skgstat(
            variogram,
            configured_model,
            description,
            active_dims=active_dims,
        )

        # Extract only aggregate lag arrays and fitted predictions from the backend
        lag_centers, experimental = variogram.get_empirical(bin_center=True)
        lag_centers = np.asarray(lag_centers, dtype=float)
        fitted = np.asarray(variogram.fitted_model(lag_centers), dtype=float)
        fit_result = {
            key: description.get(key)
            for key in ("normalized_effective_range", "normalized_sill", "normalized_nugget")
            if key in description
        }

        # Retain the heavy backend only when explicitly requested by the caller
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
        """Return a copy that releases any retained backend object."""

        return replace(self, backend_object=None)

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per lag class in a Pandas DataFrame."""

        # Start with required lag statistics before adding available optional columns
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

        # Store empirical arrays on one shared labelled lag dimension
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

        # Keep scalar descriptions as attributes for portable Xarray round trips
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

        # Convert arrays and nested models to standard Python containers
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
        """Restore a lightweight variogram from :meth:`to_dict` output."""

        # Restore the optional model before normalizing all aggregate arrays
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
        """Evaluate the fitted theoretical variogram.

        :param distance: Spatial distance or array of distances.
        :returns: Semivariance at each distance.
        """

        if self.model is None:
            raise ValueError("A fitted variogram model is required for evaluation.")
        return self.model.variogram(distance)

    __call__ = variogram

    def covariance(self, distance: NDArrayNum | float) -> NDArrayNum:
        """Evaluate covariance implied by the fitted model.

        :param distance: Spatial distance or array of distances.
        :returns: Covariance at each distance.
        """

        if self.model is None:
            raise ValueError("A fitted variogram model is required for evaluation.")
        return self.model.covariance(distance)

    def correlation(self, distance: NDArrayNum | float) -> NDArrayNum:
        """Evaluate correlation implied by the fitted model.

        :param distance: Spatial distance or array of distances.
        :returns: Correlation at each distance.
        """

        if self.model is None:
            raise ValueError("A fitted variogram model is required for evaluation.")
        return self.model.correlation(distance)

    def plot(self, ax: Any | None = None, *, show_error: bool = True, **kwargs: Any) -> Any:
        """Plot empirical bins and the fitted model when present.

        :param ax: Existing Matplotlib axes. A new figure and axes are created by default.
        :param show_error: Whether to draw finite empirical sampling errors.
        :param kwargs: Keyword arguments passed to the empirical point plot.
        :returns: Matplotlib axes containing the variogram.
        """

        # Import plotting only for callers that request visualization
        pyplot = import_optional("matplotlib.pyplot", package_name="matplotlib")
        if ax is None:
            _, ax = pyplot.subplots()

        # Draw empirical bins with optional uncertainty before the fitted curve
        error = self.semivariance_error if show_error else None
        ax.errorbar(self.lags, self.semivariance, yerr=error, fmt="o", **kwargs)
        if self.model is not None and np.any(np.isfinite(self.lags)):
            distances = np.linspace(0, float(np.nanmax(self.lags)), 500)
            ax.plot(distances, self.variogram(distances))
        ax.set(xlabel="Lag distance", ylabel="Semivariance")
        return ax

    #######################
    # BACKEND CONVERSIONS
    #######################

    def to_gstools(self, *, dim: int = 2) -> GSToolsVariogram:
        """Convert the fitted model to GSTools with its source feature dimensions.

        :param dim: Number of dimensions passed to the GSTools covariance model.
        :returns: Native covariance model and dimensions selected by the source model.
        """

        # Import GSTools lazily and require fitted metadata before conversion
        gstools = import_optional("gstools", extra_name="geostat")
        if self.model is None:
            raise ValueError("A fitted variogram model is required for conversion.")
        if dim <= 0:
            raise ValueError("GSTools model dimension must be strictly positive.")

        # Require shared dimensions because GSTools receives preselected coordinates
        active_dims = {component.active_dims for component in self.model.components} if self.model.components else set()
        if len(active_dims) > 1:
            raise NotImplementedError("GSTools cannot combine variogram components that select different dimensions.")

        # Return selected dimensions separately because GSTools does not store them
        selected_dims = next(iter(active_dims)) if active_dims else self.model.active_dims
        return GSToolsVariogram(
            model=_model_to_gstools(self.model, gstools=gstools, dim=dim),
            active_dims=selected_dims,
        )

    def gpytorch_parameters(self) -> dict[str, Any]:
        """Return parameters independent of backend for the GPyTorch adapter."""

        if self.model is None:
            raise ValueError("A fitted variogram model is required for conversion.")
        return _model_to_gpytorch_parameters(self.model)

    def to_gpytorch(self, *, active_dims: tuple[int, ...] | None = None, trainable: bool = True) -> GPyTorchVariogram:
        """Convert supported fitted structures to a GPyTorch covariance kernel.

        :param active_dims: Optional feature column override applied to every structure.
        :param trainable: Whether converted kernel parameters may be optimized.
        :returns: Native covariance kernel and separate likelihood noise.
        """

        # Import GPyTorch lazily and require fitted metadata before conversion
        gpytorch = import_optional("gpytorch", extra_name="gp")
        if self.model is None:
            raise ValueError("A fitted variogram model is required for conversion.")
        kernel, noise = _model_to_gpytorch(
            self.model,
            gpytorch=gpytorch,
            active_dims=active_dims,
        )

        # Freeze kernel parameters when the conversion is intended only for prediction
        if not trainable:
            for parameter in kernel.parameters():
                parameter.requires_grad_(False)
        return GPyTorchVariogram(kernel=kernel, noise=noise)


############################
# 4/ REPEATED PAIR ESTIMATION
############################


def _estimate_variogram(
    source: Any,
    *,
    n_runs: int,
    n_jobs: int,
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
    """Sample independent pair sets and reduce them to one lightweight result."""

    # Validate execution controls before deriving independent random seeds
    if n_runs < 1 or n_jobs < 1:
        raise ValueError("n_runs and n_jobs must be positive integers.")
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    seeds = rng.integers(0, np.iinfo(np.int32).max, n_runs)

    def run(seed: np.integer[Any], bin_spec: Literal["log", "uniform"] | Iterable[float]) -> Variogram:
        # Discard each pair dataset immediately after reducing it to lag statistics
        pairs = source.sample_pairs(random_state=int(seed), **pair_kwargs)
        return Variogram.from_pairs(
            pairs,
            estimator=estimator,
            bins=bin_spec,
            n_lags=n_lags,
            min_lag=min_lag,
            max_lag=max_lag,
        )

    # Establish bins once so every independent run describes the same lag classes
    first = run(seeds[0], bins)
    shared_bins: Literal["log", "uniform"] | Iterable[float] = bins
    if isinstance(bins, str):
        assert first.bin_lower_edges is not None and first.bin_edges is not None
        shared_bins = np.r_[first.bin_lower_edges[0], first.bin_edges]

    # Evaluate remaining runs sequentially or with bounded worker threads
    if n_runs == 1:
        runs = [first]
    elif n_jobs == 1:
        runs = [first, *[run(seed, shared_bins) for seed in seeds[1:]]]
    else:
        with ThreadPoolExecutor(max_workers=min(n_jobs, n_runs - 1)) as executor:
            remaining = list(executor.map(lambda seed: run(seed, shared_bins), seeds[1:]))
        runs = [first, *remaining]

    # Return the single run directly while preserving execution metadata
    if n_runs == 1:
        result = replace(first, attrs={**first.attrs, "n_runs": 1, "n_jobs": 1})
        return result if models is None else result.fit(models, **dict(fit_kwargs or {}))

    # Stack compact lag arrays to aggregate independent estimates by class
    semivariances = np.vstack([run_result.semivariance for run_result in runs])
    lag_centers = np.vstack([run_result.lags for run_result in runs])
    counts = np.sum(np.vstack([run_result.counts for run_result in runs]), axis=0)
    mean_semivariance = np.full(semivariances.shape[1], np.nan)
    mean_lags = np.full(semivariances.shape[1], np.nan)
    errors = np.full(semivariances.shape[1], np.nan)

    # Average available bins and estimate standard errors only from repeated values
    for index in range(semivariances.shape[1]):
        finite_semivariance = semivariances[np.isfinite(semivariances[:, index]), index]
        finite_lags = lag_centers[np.isfinite(lag_centers[:, index]), index]
        if finite_semivariance.size:
            mean_semivariance[index] = np.mean(finite_semivariance)
        if finite_semivariance.size > 1:
            errors[index] = np.std(finite_semivariance, ddof=1) / np.sqrt(finite_semivariance.size)
        if finite_lags.size:
            mean_lags[index] = np.mean(finite_lags)

    # Replace the first result arrays while retaining shared bin metadata
    result = replace(
        first,
        lags=mean_lags,
        semivariance=mean_semivariance,
        semivariance_error=errors,
        counts=counts,
        attrs={**first.attrs, "n_runs": n_runs, "n_jobs": n_jobs},
    )
    return result if models is None else result.fit(models, **dict(fit_kwargs or {}))


########################
# 5/ BACKEND MODEL ADAPTERS
########################


def _model_from_skgstat(
    variogram: Any,
    configured_model: str,
    description: Mapping[str, Any],
    *,
    active_dims: tuple[int, ...] | None,
) -> VariogramModel:
    """Normalize SciKit-GStat's fitted coefficient ordering."""

    # Read simple model fields directly from the normalized backend description
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

    # Separate a shared nugget from coefficients of summed model components
    names = [name.strip() for name in configured_model.split("+")]
    coefficients = list(np.asarray(variogram.cof, dtype=float))
    use_nugget = bool(description.get("params", {}).get("use_nugget", False))
    nugget = float(coefficients.pop()) if use_nugget else 0.0
    components: list[VariogramModel] = []
    position = 0
    for name in names:
        # Consume common range and sill parameters before optional shape values
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

    # Rebuild the summed portable model with one parent nugget
    return VariogramModel.sum(components, nugget=nugget)


def _model_to_gstools(model: VariogramModel, *, gstools: Any, dim: int) -> Any:
    """Convert one normalized model, including summed structures, to GSTools."""

    # Convert summed components recursively before applying their shared nugget
    if model.model_name == "sum":
        components = [_model_to_gstools(component, gstools=gstools, dim=dim) for component in model.components]
        return gstools.SumModel(*components, nugget=model.nugget)
    if model.model_name == "product":
        raise NotImplementedError("Product covariance conversion is not supported by the GSTools adapter.")
    if model.effective_range is None or model.partial_sill is None:
        raise AssertionError("A base variogram model must define its range and partial sill.")

    # Share normalized variance and range parameters across GSTools constructors
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

    # Apply the fitted Matérn smoothness through the corresponding GSTools parameter
    if model.model_name == "matern":
        return gstools.Matern(nu=model.smoothness, rescale=4.0, **common)
    raise NotImplementedError(f"Variogram model {model.model_name!r} has no GSTools adapter.")


def _model_to_gpytorch(
    model: VariogramModel, *, gpytorch: Any, active_dims: tuple[int, ...] | None
) -> tuple[Any, float]:
    """Convert exact common covariance structures to native GPyTorch kernels."""

    # Combine recursively converted kernels according to the portable composition
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

    # Derive exact kernel parameters before constructing a native base kernel
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

    # Apply structured variance outside the normalized base correlation kernel
    kernel = gpytorch.kernels.ScaleKernel(base_kernel)
    kernel.base_kernel.lengthscale = parameters["lengthscale"]
    kernel.outputscale = float(model.partial_sill)
    return kernel, model.nugget


def _model_to_gpytorch_parameters(
    model: VariogramModel,
    *,
    active_dims: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Describe an exact conversion without importing GPyTorch."""

    # Describe composite models recursively so inspection needs no optional backend
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

    # Translate effective range conventions to native kernel length scales
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

    # Return observation noise separately because it belongs to the likelihood
    return {
        "kernel_name": kernel_name,
        "lengthscale": lengthscale,
        "outputscale": float(model.partial_sill),
        "smoothness": smoothness,
        "active_dims": model.active_dims if active_dims is None else active_dims,
        "noise": model.nugget,
    }
