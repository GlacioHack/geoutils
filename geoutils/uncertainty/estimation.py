# Copyright (c) 2026 xDEM developers
#
# This file is part of the xDEM project:
# https://github.com/glaciohack/xdem
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

"""Estimate error component magnitudes and correlations from an error proxy, with out-of-memory support on Dask/MP."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import xarray as xr

from geoutils._dispatch import (
    _get_pointcloud_interface,
    _get_raster_interface,
    _is_pointcloud,
    _is_raster,
    is_dask_array,
)
from geoutils._typing import NDArrayNum
from geoutils.stats.variography import Variogram, VariogramModel
from geoutils.uncertainty.error_structure import (
    ErrorComponent,
    ErrorMagnitude,
    ErrorStructure,
)

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.raster.base import RasterBase
    from geoutils.raster.raster import Raster

############################
# 1/ INPUT AND MASK ALIGNMENT
############################


def _prepare_proxy_inputs(
    error_proxy: RasterBase | PointCloudBase,
    predictors: Mapping[str, Any],
    mask: Any | None,
    mp_config: MultiprocConfig | None,
    other: Any | None,
    other_precision: Literal["same", "negligible"],
    temporary_files: ExitStack,
) -> tuple[Any, dict[str, str]]:
    """
    Prepare error proxy and predictors on their common finite support using ``cosample()``, which supports
    Dask/MP itself, so does all the heavy lifting.
    """

    # Input checks
    raster_proxy = _get_raster_interface(error_proxy)
    pointcloud_proxy = _get_pointcloud_interface(error_proxy) if raster_proxy is None else None
    proxy = raster_proxy if raster_proxy is not None else pointcloud_proxy
    if proxy is None:
        raise TypeError("Argument 'error_proxy' must be a raster or point cloud.")

    # We give predictors private names to avoid conflict with cosample() columns
    auxiliary: dict[str, Any] = {}
    auxiliary_at: dict[str, Literal["self"]] = {}
    output_names: dict[str, str] = {}
    for index, (name, predictor) in enumerate(predictors.items()):
        output_name = f"predictor_{index}"
        output_names[name] = output_name
        if isinstance(predictor, str):
            if pointcloud_proxy is None:
                raise TypeError("Raster magnitude predictors cannot be column names.")
            auxiliary[output_name] = (error_proxy, predictor)
        else:
            predictor_is_spatial = _is_raster(predictor) or _is_pointcloud(predictor)
            if predictor_is_spatial:
                auxiliary[output_name] = predictor
            else:
                auxiliary[output_name] = predictor.to_numpy() if hasattr(predictor, "to_numpy") else predictor
                auxiliary_at[output_name] = "self"

    # We sample the two datasets and predictors at valid locations inside the mask
    sampled = proxy.cosample(
        error_proxy if other is None else other,
        auxiliary=auxiliary or None,
        auxiliary_at=auxiliary_at or None,
        at="self",
        mask=mask,
        mp_config=mp_config,
    )
    if other is not None:
        scale = np.sqrt(2.0) if other_precision == "same" else 1.0
        sampled = _difference_proxy(sampled, scale, mp_config, temporary_files)
    return sampled, output_names


def _difference_raster_values(data: Any, scale: float) -> Any:
    """Replace the first raster band with the scaled difference of the first two bands."""

    if is_dask_array(data):
        import dask.array as da

        difference = (data[0:1] - data[1:2]) / scale
        return da.concatenate((difference, data[1:]), axis=0)

    values = np.ma.asarray(data, dtype=float).copy()
    values[0] = (values[0] - values[1]) / scale
    return values


def _wrapper_difference_raster_tile_multiproc(tile: RasterBase, scale: float) -> Raster:
    """Calculate the difference band in one multiprocessing raster tile."""

    from geoutils.raster import Raster

    values = _difference_raster_values(tile.data, scale)
    return Raster.from_array(values, tile.transform, tile.crs, nodata=tile.nodata, area_or_point=tile.area_or_point)


def _difference_point_partition(dataframe: Any, scale: float) -> Any:
    """Calculate the scaled difference within one point partition."""

    result = dataframe.copy()
    result["self"] = (dataframe["self"] - dataframe["other"]) / scale
    return result


def _difference_proxy(sampled: Any, scale: float, mp_config: MultiprocConfig | None, files: ExitStack) -> Any:
    """Build a difference proxy while preserving eager, Dask, or multiprocessing execution."""

    raster = _get_raster_interface(sampled)
    points = _get_pointcloud_interface(sampled) if raster is None else None

    if points is not None and points._is_xr:
        difference = (points.data - points._get_column_values("other")) / scale
        return points.copy(new_array=difference)

    # Multiprocessing: raster tiles or ordered point chunks
    if mp_config is not None:
        if raster is not None:
            from geoutils.multiproc import map_overlap

            output_config = files.enter_context(mp_config.temporary())
            return map_overlap(_wrapper_difference_raster_tile_multiproc, sampled, output_config, scale)

        import pandas as pd

        from geoutils.multiproc.cluster import _map_bounded

        assert points is not None
        dataframe = points._dataset
        chunks = mp_config.chunks
        if not isinstance(chunks, int):
            raise ValueError("Point cloud multiprocessing requires an integer chunk size.")

        # Difference point chunks in workers and restore their original row order
        arguments = ((dataframe.iloc[start : start + chunks], scale) for start in range(0, len(dataframe), chunks))
        result = pd.concat(part for _, part in _map_bounded(mp_config.cluster, _difference_point_partition, arguments))
        return points._cast_pointcloud_output(result)

    # Dask: preserve lazy raster blocks or point partitions
    if raster is not None and is_dask_array(sampled.data):
        values = _difference_raster_values(sampled.data, scale)
        if isinstance(sampled, xr.DataArray):
            return sampled.copy(data=values, deep=False)
        return sampled.copy(new_array=values)
    if points is not None and hasattr(points._dataset, "map_partitions"):
        dataframe = points._dataset
        result = dataframe.map_partitions(_difference_point_partition, scale, meta=dataframe._meta)
        return points._cast_pointcloud_output(result)

    # Eager: update a copy of the sampled raster or point cloud
    if raster is not None:
        values = _difference_raster_values(sampled.data, scale)
        if isinstance(sampled, xr.DataArray):
            return sampled.copy(data=values, deep=False)
        return sampled.copy(new_array=values)

    assert points is not None
    dataframe = points._dataset
    return points._cast_pointcloud_output(_difference_point_partition(dataframe, scale))


################################
# 2/ TOTAL MAGNITUDE ESTIMATION
################################


@dataclass(frozen=True)
class _FiniteSpread:
    """Ensure the user statistical spread estimator respects nodata/NaNs."""

    estimator: Callable[[Any], Any]

    @property
    def __name__(self) -> str:
        return getattr(self.estimator, "__name__", "spread")

    def __call__(self, values: Any) -> float:
        finite = np.ma.asarray(values, dtype=float).filled(np.nan)
        return float(self.estimator(finite[np.isfinite(finite)]))


def _estimate_total_magnitude(
    sampled: Any,
    output_names: Mapping[str, str],
    *,
    bins: Mapping[str, Any] | int | None,
    spread_estimator: Callable[[Any], Any],
    min_count: int,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    mp_config: MultiprocConfig | None,
) -> tuple[ErrorMagnitude, int]:
    """
    Estimate error magnitude from the statistical spread of error proxy array, optionally grouped by predictors.

    This step supports Dask/MP through ``stats()``.
    """

    # Check inputs
    if min_count < 1:
        raise ValueError("min_count must be a positive integer.")

    # Select the same finite errors for every statistics backend
    finite_spread = _FiniteSpread(spread_estimator)
    statistic = finite_spread.__name__
    raster_interface = _get_raster_interface(sampled)
    raster = raster_interface is not None
    sampled_interface = raster_interface if raster else _get_pointcloud_interface(sampled)
    value_selector = 1 if raster else "self"
    valid_count = int(sampled_interface.stats("validcount", values=value_selector, mp_config=mp_config))
    if valid_count < 2:
        raise ValueError("At least two finite error proxy observations are required.")
    grouped_subsample = 1 if isinstance(subsample, (int, np.integer)) and subsample >= valid_count else subsample

    # Use the direct spread when no variable magnitude was requested
    if not output_names:
        constant_spread = float(
            sampled_interface.stats(
                finite_spread,
                values=value_selector,
                subsample=grouped_subsample,
                random_state=random_state,
                mp_config=mp_config,
            )
        )
        if not np.isfinite(constant_spread) or constant_spread <= 0:
            raise ValueError("spread_estimator returned a non-positive or non-finite magnitude.")
        return ErrorMagnitude.constant(constant_spread), valid_count

    # Expand one shared bin count or the default across all named predictors
    if bins is None:
        grouped_bins = dict.fromkeys(output_names, 10)
    elif isinstance(bins, (int, np.integer)):
        grouped_bins = {name: int(bins) for name in output_names}
    else:
        grouped_bins = dict(bins)
    if set(grouped_bins) != set(output_names):
        raise ValueError("bins must define every magnitude predictor and no unknown names.")

    # Keep unobserved predictor combinations so the fitted magnitude grid can fill their gaps
    predictor_specs = {
        name: (sampled, index + 3 if raster else output_name)
        for index, (name, output_name) in enumerate(output_names.items())
    }
    table = sampled_interface.stats(
        finite_spread,
        values={"error": value_selector},
        by=predictor_specs,
        bins=grouped_bins,
        subsample=grouped_subsample,
        random_state=random_state,
        observed=False,
        mp_config=mp_config,
    )

    # Construct variable ErrorMagnitude from grouped statistics
    magnitude = ErrorMagnitude(
        kind="variable",
        predictor_names=tuple(output_names),
        grouped_statistics=table,
        statistic=statistic,
        min_count=min_count,
    )

    return magnitude, valid_count


#################################
# 3/ CORRELATION STANDARDIZATION
#################################


def _standardized_raster_values(
    data: Any,
    output_names: Mapping[str, str],
    magnitude: ErrorMagnitude,
) -> NDArrayNum:
    """Divide all values in an error proxy raster tile by the estimated error magnitude."""

    bands = np.ma.asarray(data, dtype=float).filled(np.nan)
    values = np.asarray(bands[0], dtype=float)
    predictors = {name: np.asarray(bands[index + 2], dtype=float) for index, name in enumerate(output_names)}
    local_magnitude = np.asarray(magnitude.predict(predictors), dtype=float)
    return np.divide(
        values,
        local_magnitude,
        out=np.full(values.shape, np.nan, dtype=float),
        where=np.isfinite(local_magnitude) & (local_magnitude > 0),
    )


def _wrapper_standardize_raster_multiproc(
    tile: RasterBase,
    output_names: Mapping[str, str],
    magnitude: ErrorMagnitude,
) -> Raster:
    """Wrapper for standardizing a raster tile for MP."""

    from geoutils.raster import Raster

    values = _standardized_raster_values(tile.data, output_names, magnitude)
    return Raster.from_array(values, tile.transform, tile.crs, nodata=np.nan, area_or_point=tile.area_or_point)


def _standardized_point_values(
    dataframe: Any,
    output_names: Mapping[str, str],
    magnitude: ErrorMagnitude,
) -> Any:
    """Divide all values in an error proxy point cloud by the estimated error magnitude."""

    values = dataframe["self"].to_numpy(dtype=float)
    predictors = {name: dataframe[column].to_numpy(dtype=float) for name, column in output_names.items()}
    local_magnitude = np.asarray(magnitude.predict(predictors), dtype=float)
    values = np.divide(
        values,
        local_magnitude,
        out=np.full(values.shape, np.nan, dtype=float),
        where=np.isfinite(local_magnitude) & (local_magnitude > 0),
    )
    result = dataframe["self"].copy()
    result.iloc[:] = values
    return result


def _standardized_point_partition(
    dataframe: Any,
    output_names: Mapping[str, str],
    magnitude: ErrorMagnitude,
) -> Any:
    """Replace the proxy column within one lazy point partition."""

    result = dataframe.copy()
    result["self"] = _standardized_point_values(dataframe, output_names, magnitude)
    return result


def _standardized_array_point_values(
    values: Any, *predictors: Any, names: tuple[str, ...], magnitude: ErrorMagnitude
) -> Any:
    """Divide point errors by positive finite magnitudes within one array block."""
    local_magnitude = np.asarray(magnitude.predict(dict(zip(names, predictors))), dtype=float)
    return np.divide(
        values,
        local_magnitude,
        out=np.full(values.shape, np.nan, dtype=float),
        where=np.isfinite(local_magnitude) & (local_magnitude > 0),
    )


def _standardize_proxy(
    sampled: Any,
    output_names: Mapping[str, str],
    magnitude: ErrorMagnitude,
    mp_config: MultiprocConfig | None,
    temporary_files: ExitStack,
) -> Any:
    """
    Standardize the error proxy used to estimate spatial correlation by its error magnitude (i.e. divide by it).

    This function calls some helpers above to support chunked behaviour with MP.
    Multiprocessing writes raster tiles to a temporary file, while Dask preserves lazy raster blocks or point
    partitions.
    """

    raster = _is_raster(sampled)
    point_sample: Any = None if raster else _get_pointcloud_interface(sampled)

    if point_sample is not None and point_sample._is_xr:
        # Predict and divide within array blocks, leaving the sampled support lazy
        predictors = [point_sample._dataset.coords[column] for column in output_names.values()]
        inputs = xr.unify_chunks(point_sample._dataset, *predictors)
        values = xr.apply_ufunc(
            _standardized_array_point_values,
            *inputs,
            kwargs={"names": tuple(output_names), "magnitude": magnitude},
            dask="parallelized",
            output_dtypes=[float],
        )
        return point_sample.copy(new_array=values.data)

    # Multiprocessing: write temporary raster or point file with standardized error proxy
    if mp_config is not None:
        if raster:
            from geoutils.multiproc import map_overlap

            output_config = temporary_files.enter_context(mp_config.temporary())
            return map_overlap(_wrapper_standardize_raster_multiproc, sampled, output_config, output_names, magnitude)

        # Point cosampling already selected eager rows; a Dask point input stays lazy
        dataframe = point_sample._dataset
        if not hasattr(dataframe, "map_partitions"):
            standardized_values = _standardized_point_values(dataframe, output_names, magnitude)
            return point_sample.copy(new_array=standardized_values)

    # Dask: lazy division of the raster blocks or point partitions until the variogram runs
    if raster:
        data = sampled.data
        if is_dask_array(data):
            import dask.array as da

            # Each raster block needs all bands to predict its local magnitude
            data = data.rechunk({0: data.shape[0]})
            standardized = da.map_blocks(
                _standardized_raster_values,
                data,
                output_names,
                magnitude,
                dtype=float,
                chunks=data.chunks[1:],
                drop_axis=0,
            )
            return sampled.isel(band=0).copy(data=standardized, deep=False)
    else:
        dataframe = point_sample._dataset
        if hasattr(dataframe, "map_partitions"):
            standardized = dataframe.map_partitions(
                _standardized_point_partition,
                output_names,
                magnitude,
                meta=dataframe._meta,
            )
            return point_sample._cast_pointcloud_output(standardized)

    # Eager: return a copy with standardized raster cells or point values
    if raster:
        values = _standardized_raster_values(data, output_names, magnitude)
        if isinstance(sampled, xr.DataArray):
            return sampled.isel(band=0).copy(data=values, deep=False)
        return sampled.copy(new_array=np.ma.masked_invalid(values))

    standardized_values = _standardized_point_values(dataframe, output_names, magnitude)
    return point_sample.copy(new_array=standardized_values)


#################################
# 4/ COMPONENT CONFIGURATION
#################################


def _normalize_component_configuration(
    components: Mapping[str, Mapping[str, Any]] | None,
    *,
    has_predictors: bool,
) -> list[dict[str, Any]]:
    """Normalized the component specifications for the variogram estimator."""

    # We use the common local heterosc. + long homosc. component model when callers provide predictors
    if components is None:
        if has_predictors:
            components = {
                "short_range": {"magnitude": "heteroscedastic", "correlation": "gaussian"},
                "long_range": {"magnitude": "constant", "correlation": "spherical"},
            }
        else:
            components = {"short_range": {"magnitude": "constant", "correlation": "gaussian"}}
    if not components:
        raise ValueError("components must define at least one named error contribution.")

    # We normalize possible aliases, and ensure correlation ranges are always ordered from short to long
    normalized: list[dict[str, Any]] = []
    variable_count = 0
    for name, configuration in components.items():
        if not isinstance(name, str) or not name or not isinstance(configuration, Mapping):
            raise TypeError("components must map non-empty names to configuration mappings.")
        unknown = set(configuration).difference({"magnitude", "correlation"})
        if unknown:
            raise ValueError(f"Unknown configuration for component {name!r}: {sorted(unknown)!r}.")

        # We check for different magnitude aliases and require predictors for a variable magnitude
        magnitude = configuration.get("magnitude", "constant")
        if magnitude == "variable":
            magnitude = "heteroscedastic"
        if magnitude not in {"constant", "heteroscedastic"}:
            raise ValueError("Estimated component magnitude must be 'constant' or 'heteroscedastic'.")
        if magnitude == "heteroscedastic":
            variable_count += 1
            if not has_predictors:
                raise ValueError("A heteroscedastic component requires at least one magnitude predictor.")

        # We keep "independent" (no correlation) errors distinct from components with a variogram model name
        correlation = configuration.get("correlation")
        if correlation is not None and not isinstance(correlation, str):
            raise TypeError("Estimated component correlation must be a variogram model name or None.")
        normalized.append({"name": name, "magnitude": magnitude, "correlation": correlation})

    # For now, we only support one heteroscedastic component
    if variable_count > 1:
        raise NotImplementedError(
            "Variogram estimation currently supports at most one heteroscedastic component; "
            "manually constructed ErrorStructure objects can contain more."
        )
    return normalized


#################################
# 5/ FITTED ERROR COMPONENTS
#################################


def _build_error_components(
    configuration: list[dict[str, Any]],
    total_magnitude: ErrorMagnitude,
    fitted_variogram: Variogram,
) -> list[ErrorComponent]:
    """Build error components from a fitted variogram and the total error magnitude."""

    # Match fitted correlation terms to the requested components
    if fitted_variogram.model is None:
        raise ValueError("A fitted variogram is required to build error components.")
    fitted_model = fitted_variogram.model
    structured_models = list(fitted_model.components) if fitted_model.model_name == "sum" else [fitted_model]
    correlated = [item for item in configuration if item["correlation"] is not None]
    independent = [item for item in configuration if item["correlation"] is None]
    if len(structured_models) != len(correlated):
        raise RuntimeError("The fitted variogram does not match the requested correlated components.")
    if len(independent) > 1:
        raise ValueError("Only one independent component can be identified from a shared variogram nugget.")

    # Sort ranges with their sills; requested model forms follow short-to-long order
    if any(model.effective_range is None for model in structured_models):
        raise ValueError("Each fitted correlation model needs a finite effective range.")
    fitted_ranges = np.array([float(cast(float, model.effective_range)) for model in structured_models])
    fitted_order = np.argsort(fitted_ranges)
    if not np.array_equal(fitted_order, np.arange(len(structured_models))):
        warnings.warn(
            "Fitted nested variogram ranges crossed; assigning their contributions from short to long range.",
            UserWarning,
        )
    ordered_ranges = fitted_ranges[fitted_order]
    ordered_sills = np.array([float(model.partial_sill or 0.0) for model in structured_models])[fitted_order]

    # Assign each ordered term to a correlated component and the nugget to independent noise
    model_by_name: dict[str, VariogramModel | None] = {}
    sill_by_name: dict[str, float] = {}
    model_position = 0
    for item in configuration:
        if item["correlation"] is None:
            model_by_name[item["name"]] = None
            sill_by_name[item["name"]] = float(fitted_model.nugget)
        else:
            model = replace(
                structured_models[model_position],
                effective_range=float(ordered_ranges[model_position]),
                partial_sill=float(ordered_sills[model_position]),
            )
            model_position += 1

            # Correlated components exclude the shared independent nugget
            model_by_name[item["name"]] = replace(model, nugget=0.0)
            sill_by_name[item["name"]] = float(model.partial_sill or 0.0)

    # Sum fitted variance assigned to the requested components
    total_sill = sum(sill_by_name.values())
    if total_sill <= 0:
        raise ValueError("The fitted variogram has no positive component variance.")

    # Scale the standardized sill shares by total error variance in the source units
    reference_variance = total_magnitude.reference_value**2
    initial_variance = {
        item["name"]: reference_variance * sill_by_name[item["name"]] / total_sill for item in configuration
    }
    variable = next((item for item in configuration if item["magnitude"] == "heteroscedastic"), None)
    if variable is not None:
        # Constant components contribute the same variance at every predictor value
        fixed_variance = sum(
            initial_variance[item["name"]] for item in configuration if item["magnitude"] == "constant"
        )

        # Cap fixed variance so the variable component has at least five percent at the reference magnitude
        maximum_fixed = 0.95 * reference_variance
        if fixed_variance > maximum_fixed:
            factor = maximum_fixed / fixed_variance
            for item in configuration:
                if item["magnitude"] == "constant":
                    initial_variance[item["name"]] *= factor
            fixed_variance = maximum_fixed
    else:
        fixed_variance = 0.0

    # Build components and record their fitted sill shares before any fixed-variance cap
    output: list[ErrorComponent] = []
    for item in configuration:
        # The variable model subtracts fixed variance locally; constant magnitudes use square roots
        magnitude = (
            replace(total_magnitude, variance_offset=fixed_variance)
            if item["magnitude"] == "heteroscedastic"
            else ErrorMagnitude.constant(np.sqrt(initial_variance[item["name"]]))
        )
        output.append(
            ErrorComponent(
                name=item["name"],
                magnitude=magnitude,
                correlation=model_by_name[item["name"]],
                metadata={
                    "magnitude_kind": item["magnitude"],
                    "initial_variance_fraction": sill_by_name[item["name"]] / total_sill,
                },
            )
        )
    return output


################################
# 6/ CORRELATION ESTIMATION
################################


def _representative_variogram(
    empirical: Variogram,
    components: list[ErrorComponent],
) -> Variogram:
    """Attach the representative normalized component model to empirical bins."""

    # Normalize reference component variances so the combined model retains unit sill
    reference_variances = np.array(
        [cast(ErrorMagnitude, component.magnitude).reference_value ** 2 for component in components]
    )
    fractions = reference_variances / np.sum(reference_variances)
    structured: list[VariogramModel] = []
    nugget = 0.0

    # Represent independent variance as a nugget and retain normalized structured contributions
    for component, fraction in zip(components, fractions):
        if component.correlation is None:
            nugget += float(fraction)
        else:
            structured.append(replace(cast(VariogramModel, component.correlation), partial_sill=float(fraction)))
    if not structured:
        raise ValueError("At least one spatially correlated component is required for variogram fitting.")

    # Attach the combined model and its predictions at the original empirical lag centers
    model = (
        replace(structured[0], nugget=nugget) if len(structured) == 1 else VariogramModel.sum(structured, nugget=nugget)
    )
    return replace(empirical, model=model, fitted_semivariance=model.variogram(empirical.lags))


def _estimate_correlation(
    standardized_proxy: Any,
    configuration: list[dict[str, Any]],
    total_magnitude: ErrorMagnitude,
    valid_count: int,
    *,
    correlated_models: list[str],
    variogram_estimator: str | Callable[[Any], float],
    n_pairs: int,
    pair_sampling: Literal["loglag", "random_xy"],
    n_lags: int,
    min_lag: float | None,
    max_lag: float | None,
    n_runs: int,
    fit_kwargs: Mapping[str, Any] | None,
    pair_options: Mapping[str, Any],
    mp_config: MultiprocConfig | None,
    random_state: int,
) -> tuple[list[ErrorComponent], Variogram, list[str]]:
    """Fit spatial correlation on error proxy using ``variogram()``."""

    # We fit the requested (potentially nested) models through GeoUtils lightweight variography
    fit_options = dict(fit_kwargs or {})
    independent_count = sum(item["correlation"] is None for item in configuration)
    if independent_count:
        fit_options.setdefault("use_nugget", True)

    # We know valid count ahead, so we can limit it to avoid raising the underlying warning
    available_pairs = valid_count * (valid_count - 1) // 2
    effective_n_pairs = min(n_pairs, available_pairs)

    # Estimate the standardized variogram with internal Dask/MP support
    standardized_interface = _get_raster_interface(standardized_proxy)
    if standardized_interface is None:
        standardized_interface = _get_pointcloud_interface(standardized_proxy)
    empirical = standardized_interface.variogram(
        n_pairs=effective_n_pairs,
        sampling=pair_sampling,
        estimator=variogram_estimator,
        n_lags=n_lags,
        min_lag=min_lag,
        max_lag=max_lag,
        n_runs=n_runs,
        model=correlated_models,
        fit_kwargs=fit_options,
        random_state=random_state,
        mask=None,
        mp_config=mp_config,
        **pair_options,
    )

    # Now, we inspect output components to warn if something looks fishy
    fitted_components = _build_error_components(configuration, total_magnitude, empirical)

    # Update variogram predictions to match the final component decomposition
    empirical = _representative_variogram(empirical, fitted_components)

    # We report weak range separation and domain limited long range estimates explicitly
    messages: list[str] = []
    ranges = [
        float(cast(float, cast(VariogramModel, component.correlation).effective_range))
        for component in fitted_components
        if component.correlation is not None
    ]
    if len(ranges) > 1 and any(second / first < 1.5 for first, second in zip(ranges[:-1], ranges[1:])):
        messages.append("Some fitted correlation ranges overlap and their contributions may be weakly identified.")

    # Finally, we flag a long correlation range near the sampled extent, where it may be difficult to distinguish from
    # an actual trend in the data
    sampled_maximum = float(empirical.attrs.get("max_distance", np.nanmax(empirical.lags)))
    if ranges and np.isfinite(sampled_maximum) and ranges[-1] >= 0.9 * sampled_maximum:
        messages.append("The longest correlation range approaches the sampled extent and may represent a trend.")
    for message in messages:
        warnings.warn(message, UserWarning)

    return fitted_components, empirical, messages


################################
# 7/ COMPLETE ESTIMATION WORKFLOW
################################


def _estimate_error_structure(
    error_proxy: RasterBase | PointCloudBase,
    *,
    other: Any | None,
    other_precision: Literal["same", "negligible"],
    predictors: Mapping[str, Any] | None,
    components: Mapping[str, Mapping[str, Any]] | None,
    mask: Any | None,
    bins: Mapping[str, Any] | int | None,
    spread_estimator: Callable[[Any], Any],
    min_count: int,
    subsample_magnitude: int | float,
    variogram_estimator: str | Callable[[Any], float],
    n_pairs: int,
    pair_sampling: Literal["loglag", "random_xy"],
    n_lags: int,
    min_lag: float | None,
    max_lag: float | None,
    n_runs: int,
    fit_method: Literal["variogram"],
    fit_kwargs: Mapping[str, Any] | None,
    pair_sampling_kwargs: Mapping[str, Any] | None,
    mp_config: MultiprocConfig | None,
    random_state: int | np.random.Generator | None,
) -> ErrorStructure:
    """
    Parent function to estimate error structure.

    See ErrorStructure.estimate() for parameter descriptions.

    This function supports in-memory, Dask and MP execution through support in ``cosample()``, ``stats()`` and
    ``variogram()``.

    This error structure estimation was refactored from that of xDEM (which was method-based, and thus more
    volatile with inputs and outputs).

    Internal logic, in order:
    - _prepare_proxy_inputs() aligns the data using ``cosample()``,
    - _estimate_total_magnitude() estimates the variable magnitude using ``stats()``,
    - _standardize_proxy() performs the standardization of variable errors using a short Dask/MP implementation,
    - _estimate_correlation() estimates the correlation using ``variogram`` on the standardized error proxy.
    """

    # 1/ Check inputs
    # For now, we only support fitting error covariance with a variogram
    if fit_method != "variogram":
        raise NotImplementedError("Only fit_method='variogram' is currently implemented.")
    if other_precision not in ("same", "negligible"):
        raise ValueError("other_precision must be 'same' or 'negligible'.")

    # Normalize component configuration and run ``cosample`` on error proxy
    predictor_mapping = {} if predictors is None else dict(predictors)
    configuration = _normalize_component_configuration(components, has_predictors=bool(predictor_mapping))

    # Draw separate RNG seeds so magnitude estimation and variography are both reproducible
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    magnitude_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    variogram_seed = int(rng.integers(0, np.iinfo(np.int32).max))

    pair_options = dict(pair_sampling_kwargs or {})
    if "mp_config" in pair_options:
        if mp_config is not None:
            raise ValueError("Pass mp_config either directly or in pair_sampling_kwargs, not both.")
        mp_config = pair_options.pop("mp_config")

    # 2/ Run cosampling, grouped stats and variogram estimation (with direct Dask/MP support for large datasets)

    # We keep MP intermediate files available until statistics and variogram are done
    with ExitStack() as temporary_files:
        cosample_config = temporary_files.enter_context(mp_config.temporary()) if mp_config is not None else None
        sampled, output_names = _prepare_proxy_inputs(
            error_proxy, predictor_mapping, mask, cosample_config, other, other_precision, temporary_files
        )

        # 2.1/ Estimate error magnitude from the error proxy
        total_magnitude, valid_count = _estimate_total_magnitude(
            sampled,
            output_names,
            bins=bins,
            spread_estimator=spread_estimator,
            min_count=min_count,
            subsample=subsample_magnitude,
            random_state=magnitude_seed,
            mp_config=mp_config,
        )

        # If no correlation analysis was requested (assumed independent), return error structure immediately
        # Otherwise, we continue with correlation estimation below
        correlated_models = [item["correlation"] for item in configuration if item["correlation"] is not None]
        if not correlated_models:
            if len(configuration) != 1:
                raise ValueError("Only one independent component can be estimated without a spatial correlation model.")
            independent_component = ErrorComponent(configuration[0]["name"], total_magnitude, None)
            return ErrorStructure(
                [independent_component],
                fit_diagnostics={
                    "magnitude": {"valid_count": valid_count},
                    "identifiability": [],
                },
                metadata={
                    "fit_method": fit_method,
                    "spread_estimator": getattr(spread_estimator, "__name__", "callable"),
                    "variogram_estimator": None,
                    "component_configuration": configuration,
                    "total_magnitude": total_magnitude,
                    "n_runs": 0,
                },
            )

        # # 2.2/ We standardize variable errors before estimating spatial correlation
        standardized_proxy = _standardize_proxy(
            sampled,
            output_names,
            total_magnitude,
            mp_config,
            temporary_files,
        )

        # 2.3/ Fit the requested nested models through GeoUtils lightweight variography
        fitted_components, empirical, messages = _estimate_correlation(
            standardized_proxy,
            configuration,
            total_magnitude,
            valid_count,
            correlated_models=correlated_models,
            variogram_estimator=variogram_estimator,
            n_pairs=n_pairs,
            pair_sampling=pair_sampling,
            n_lags=n_lags,
            min_lag=min_lag,
            max_lag=max_lag,
            n_runs=n_runs,
            fit_kwargs=fit_kwargs,
            pair_options=pair_options,
            mp_config=mp_config,
            random_state=variogram_seed,
        )

    # 3/ Construct final error structure
    return ErrorStructure(
        fitted_components,
        empirical_variogram=empirical,
        fit_diagnostics={
            "magnitude": {"valid_count": valid_count},
            "identifiability": messages,
        },
        metadata={
            "fit_method": fit_method,
            "spread_estimator": getattr(spread_estimator, "__name__", "callable"),
            "variogram_estimator": (
                getattr(variogram_estimator, "__name__", "callable")
                if callable(variogram_estimator)
                else variogram_estimator
            ),
            "component_configuration": configuration,
            "total_magnitude": total_magnitude,
            "n_runs": n_runs,
        },
    )


def _refit_error_structure(
    structure: ErrorStructure,
    *,
    correlation_models: str | Sequence[str] | None,
    fit_kwargs: Mapping[str, Any] | None,
) -> ErrorStructure:
    """Refit variogram model only for the ErrorStructure."""

    # Check we still have the empirical variogram + component metadata required to refit
    if structure.empirical_variogram is None:
        raise ValueError("Refitting requires a retained empirical variogram.")
    configuration_value = structure.metadata.get("component_configuration")
    total_magnitude = structure.metadata.get("total_magnitude")
    if not isinstance(configuration_value, list) or not isinstance(total_magnitude, ErrorMagnitude):
        raise ValueError("This error structure does not retain the estimation metadata required for refitting.")
    configuration = [dict(item) for item in configuration_value]

    # We default to current models, and ensure short to long correlation order
    correlated = [component for component in structure.components.values() if component.correlation is not None]
    models: str | list[str]
    if correlation_models is None:
        models = [cast(VariogramModel, component.correlation).model_name for component in correlated]
    elif isinstance(correlation_models, str):
        models = correlation_models
    else:
        models = list(correlation_models)

    # The replacement model should describe the same number of correlated components
    expected = sum(item["correlation"] is not None for item in configuration)
    model_count = len(models.split("+")) if isinstance(models, str) else len(models)
    if model_count != expected:
        raise ValueError(f"correlation_models must contain {expected} model(s).")

    # Preserve nugget fitting when the structure contains an independent component
    options = dict(fit_kwargs or {})
    if any(item["correlation"] is None for item in configuration):
        options.setdefault("use_nugget", True)
    empirical = structure.empirical_variogram.fit(models, **options)

    # We finally update component definitions before reallocating their magnitudes from the refitted variogram
    for item, model_name in zip(
        [item for item in configuration if item["correlation"] is not None],
        models.split("+") if isinstance(models, str) else models,
    ):
        item["correlation"] = model_name
    components = _build_error_components(configuration, total_magnitude, empirical)
    empirical = _representative_variogram(empirical, components)

    return ErrorStructure(
        components,
        empirical_variogram=empirical,
        fit_diagnostics=dict(structure.fit_diagnostics),
        metadata={**structure.metadata, "component_configuration": configuration},
    )
