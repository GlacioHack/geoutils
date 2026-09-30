"""Propagate local observation errors through public spatial operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd

from geoutils._config import config
from geoutils.interface.gridding import _resolve_gridding_operator
from geoutils.operators.base import LocalData
from geoutils.operators.interpolator import Interpolator, Kriging, _regular_interpolation_method, _resolve_interpolator
from geoutils.operators.neighbours import _build_kriging_grid_neighbours
from geoutils.operators.reducer import Mean, Reducer
from geoutils.operators.weighting import _with_error_structure
from geoutils.uncertainty.error_structure import ErrorStructure
from geoutils.uncertainty.propagation import PropagationSummary, _numeric_values, _wrap_like

############################################
# 1/ OPERATORS FOR LOCAL MOMENTS AND JOINT INPUTS
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

        from geoutils.uncertainty.propagation import propagate

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

    _requires_local_evaluation = True

    def _with_regular_coefficients(self, operator: Interpolator) -> _MomentInterpolator:
        """Apply this moment to the actual nearest, bilinear, or spline calculation on a raster."""

        return type(self)(_with_error_structure(operator, self.error_structure), self.quantity, self.options)


class _MomentReducer(_SpatialMoment, Reducer):
    """Reduce output uncertainty over the original point neighborhood or cell footprint."""


class _SpatialInputs(_SpatialMoment):
    """Record selected observations for explicitly requested covariance or joint simulations."""

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
# 2/ PUBLIC SPATIAL OPERATIONS AND OUTPUT LAYOUT
############################################


def _propagate_spatial(
    operation: Callable[..., Any],
    error_structure: ErrorStructure,
    operation_kwargs: dict[str, Any],
    options: dict[str, Any],
) -> PropagationSummary:
    """Run a spatial estimator and its local moments through the same neighborhood calculation.

    Each output depends only on its selected source observations. Moment operators calculate the marginal
    uncertainty inside each chunk, so they do not retain LocalData for the complete raster or point cloud.
    _MomentInterpolator and _MomentReducer call propagate() on each selected neighborhood. Joint calculations
    instead use _InputInterpolator or _InputReducer to collect the groups and preserve shared source identities.
    """

    from geoutils.raster.base import RasterBase
    from geoutils.raster.transformation import _resolve_reprojection_operator

    # Resolve aliases before replacing the estimator; the public method still owns validation and output layout
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

    # Give all three calculations the same model, including any covariance used by fitting
    operator = _with_error_structure(operator, error_structure)
    operator._error_predictors = options.get("predictors")
    kwargs.pop("error_structure", None)
    kwargs[parameter] = operator

    # Joint results deliberately collect source groups; ordinary marginal propagation stays local to each chunk
    joint = (
        options.get("return_covariance")
        or options.get("return_samples")
        or options.get("quantiles")
        or options.get("at") is not None
    )
    if joint:
        from geoutils.uncertainty.propagation import propagate

        if getattr(source, "_chunks", None) is not None or getattr(source, "_is_dask", False):
            raise ValueError("Joint spatial propagation requires eager inputs; marginal propagation supports chunks.")
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
        from dataclasses import replace

        return replace(summary, **moments)

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

    # Marginal results have the same spatial layout and laziness as the ordinary public call
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

    # Small eager outputs can expose labelled intervals without forcing any lazy result to compute
    selection = pd.DataFrame(columns=["flat_index", "estimate"])
    raw_estimate = getattr(estimate, "data", estimate)
    if not hasattr(raw_estimate, "compute"):
        values = _numeric_values(estimate).reshape(-1)
        if len(values) <= options["max_covariance_size"]:
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
            "output_distribution": "joint_gaussian" if method == "analytical" else "unknown",
        },
        n_samples=options["n_samples"] if method == "numerical" else None,
        n_valid=n_valid,
    )
