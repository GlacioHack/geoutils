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

"""Resolve resampling names and calculation backends for the public spatial methods."""

from __future__ import annotations

from collections import OrderedDict
from copy import copy
from typing import Any, Callable, Literal, cast, get_args

from geoutils.operators.interpolator import (
    Interpolator,
    InverseDistance,
    Kriging,
    RasterConvolution,
    ScipyInterpolationMethod,
    ScipyInterpolator,
    _resolve_interpolator,
)
from geoutils.operators.reducer import (
    _IRREGULAR_REDUCER_TYPES,
    Median,
    Mode,
    Quantile,
    Reducer,
    RootMeanSquare,
    Sum,
)

ResamplingMethod = Literal[
    "nearest",
    "linear",
    "bilinear",
    "cubic",
    "cubic_convolution",
    "cubic_spline",
    "lanczos",
    "quintic",
    "slinear",
    "pchip",
    "splinef2d",
    "idw",
    "inv_dist",
    "kriging",
    "mean",
    "average",
    "sum",
    "median",
    "med",
    "quantile",
    "q1",
    "q3",
    "mode",
    "minimum",
    "min",
    "maximum",
    "max",
    "range",
    "count",
    "stdev",
    "std",
    "standard_deviation",
    "rms",
    "rmse",
    "average_distance",
    "average_distance_pts",
]
ResamplingBackend = Literal["auto", "gdal", "scipy", "numba", "gstools", "gpytorch"]
FilterMethod = ResamplingMethod | Literal["gaussian", "distance"]

_METHOD_ALIASES = {
    "bilinear": "linear",
    "inv_dist": "idw",
    "med": "median",
    "std": "stdev",
    "standard_deviation": "stdev",
    "rmse": "rms",
}
_REDUCER_TYPES: dict[str, type[Reducer]] = dict(_IRREGULAR_REDUCER_TYPES.items())
_REDUCER_TYPES.update(
    {
        "sum": Sum,
        "median": Median,
        "mode": Mode,
        "rms": RootMeanSquare,
        "quantile": Quantile,
        "q1": Quantile,
        "q3": Quantile,
    }
)


def _resampling_keyword(method: Any, resampling: Any, *, default: Any = None) -> Any:
    """Select resampling or its legacy method argument, checking conflicting choices."""

    if method is not None and resampling is not None:
        if method is not resampling and method != resampling:
            raise ValueError("Pass only one of 'resampling' and the legacy method argument.")
    selected = resampling if resampling is not None else method
    return default if selected is None else selected


def _resampling_backend(
    backend: ResamplingBackend,
    *,
    default: ResamplingBackend,
    allowed: tuple[ResamplingBackend, ...],
    engine: str | None = None,
) -> ResamplingBackend:
    """Select an available backend and check its legacy engine alias before calculating values."""

    if backend not in get_args(ResamplingBackend):
        raise ValueError(f"Unknown resampling backend: {backend!r}.")
    if engine is not None and backend != "auto" and backend != engine:
        raise ValueError("Pass only one of 'backend' and the legacy 'engine' argument.")
    if engine is not None:
        selected = engine
    elif backend == "auto":
        selected = default
    else:
        selected = backend
    if selected not in allowed:
        choices = ", ".join(repr(name) for name in allowed)
        raise ValueError(f"Backend {selected!r} is unavailable for this operation. Choose {choices}.")
    return cast(ResamplingBackend, selected)


def _resampling_operator(
    method: str | Interpolator | Reducer,
    *,
    backend: ResamplingBackend = "auto",
    options: dict[str, Any] | None = None,
    distance_power: float = 2.0,
) -> Interpolator | Reducer:
    """
    Construct the named interpolator or reducer without requiring public callers to instantiate a class.

    Existing operator objects are used directly. A requested kriging backend is applied to a copy so the supplied
    object and its cached coefficients remain unchanged. Constructor options configure named methods only.
    """

    # Kriging has its own solvers, including when an advanced object was supplied
    if (isinstance(method, Kriging) or method == "kriging") and backend not in ("auto", "gstools", "gpytorch"):
        raise ValueError("Kriging requires backend='gstools', 'gpytorch' or 'auto'.")

    # Preserve advanced operators and reject constructor options that would change their configuration
    if isinstance(method, (Interpolator, Reducer)):
        if options:
            raise ValueError("Method settings apply to a string method; configure an operator object directly.")
        if isinstance(method, Kriging) and backend in ("gstools", "gpytorch") and backend != method.backend:
            configured = copy(method)
            configured.backend = cast(Literal["gstools", "gpytorch"], backend)
            configured._coefficient_cache = OrderedDict()
            return configured
        if backend in ("gstools", "gpytorch") and not isinstance(method, Kriging):
            raise ValueError(f"Backend {backend!r} requires resampling='kriging'.")
        return method

    # Normalize equivalent names before selecting an interpolation or reduction calculation
    if method not in get_args(ResamplingMethod):
        raise ValueError(f"Unknown resampling method: {method!r}.")
    name = _METHOD_ALIASES.get(method, method)
    parameters = {} if options is None else options.copy()
    if backend in ("gstools", "gpytorch") and name != "kriging":
        raise ValueError(f"Backend {backend!r} requires resampling='kriging'.")

    # Kriging needs an explicit model; IDW and quantiles use their existing constructor options
    if name == "kriging":
        if "model" not in parameters:
            raise ValueError("resampling='kriging' requires a fitted model in resampling_kwargs['model'].")
        if backend in ("gstools", "gpytorch"):
            if "backend" in parameters and parameters["backend"] != backend:
                raise ValueError("The kriging backend conflicts with resampling_kwargs['backend'].")
            parameters["backend"] = backend
        return Kriging(**parameters)
    if name == "idw":
        parameters.setdefault("power", distance_power)
        return InverseDistance(**parameters)
    if name in _REDUCER_TYPES:
        if name in ("q1", "q3"):
            parameters.setdefault("q", 0.25 if name == "q1" else 0.75)
        if name == "quantile" and "q" not in parameters:
            raise ValueError("The quantile method requires q in its method settings.")
        constructor: Callable[..., Reducer] = _REDUCER_TYPES[name]
        return constructor(**parameters)

    # Raster convolution names and SciPy spline names describe distinct calculations
    if name in ("cubic_convolution", "cubic_spline", "lanczos"):
        if parameters:
            raise ValueError("Raster convolution does not accept resampling_kwargs.")
        kernel = "cubic" if name == "cubic_convolution" else name
        return RasterConvolution(cast(Literal["cubic", "cubic_spline", "lanczos"], kernel))
    operator = _resolve_interpolator(cast(ScipyInterpolationMethod, name))
    if not parameters:
        return operator
    if type(operator) is ScipyInterpolator:
        return ScipyInterpolator(method=cast(ScipyInterpolationMethod, name), **parameters)
    return type(operator)(**parameters)
