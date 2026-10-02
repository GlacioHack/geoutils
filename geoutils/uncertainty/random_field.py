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

"""Module to generate random correlated fields from spatial error structure, using either GSTools or GPyTorch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from geoutils._dispatch import _get_pointcloud_interface, _get_raster_interface
from geoutils._misc import import_optional
from geoutils.uncertainty.error_structure import ErrorStructure
from geoutils.uncertainty.predictors import (
    _point_predictor_columns,
    _raster_chunk_predictors,
    _raster_chunk_source,
    _raster_values_mask,
    _spatial_predictors,
)

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.raster.base import RasterBase

#######################
# 1/ SHARED HELPERS
#######################


@dataclass(frozen=True)
class GPyTorchInducingField:
    """A shared GPyTorch draw on a small spatial grid, with its local covariance."""

    axes: tuple[NDArray[np.float64], ...]
    values: NDArray[np.float64]
    local_covariance: NDArray[np.float64]


def _interpolate_inducing_field(field: GPyTorchInducingField, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
    """Interpolate a shared GP draw and restore unit variance at each location."""

    if coordinates.shape[1] != len(field.axes):
        raise ValueError("Inducing grid dimensions must match the field coordinates.")
    shape = tuple(len(axis) for axis in field.axes)
    lower_indexes = []
    fractions = []
    for dimension, axis in enumerate(field.axes):
        if len(axis) == 1:
            lower_indexes.append(np.zeros(len(coordinates), dtype=np.int64))
            fractions.append(np.zeros(len(coordinates), dtype=float))
            continue
        lower = np.clip(np.searchsorted(axis, coordinates[:, dimension], side="right") - 1, 0, len(axis) - 2)
        fraction = np.clip((coordinates[:, dimension] - axis[lower]) / (axis[lower + 1] - axis[lower]), 0, 1)
        lower_indexes.append(lower)
        fractions.append(fraction)

    # The same grid corners meet on either side of a chunk boundary
    corner_indexes = []
    corner_weights = []
    choices = product(*((0, 1) if length > 1 else (0,) for length in shape))
    for corner in choices:
        corner_coordinates = tuple(lower_indexes[dimension] + offset for dimension, offset in enumerate(corner))
        weights = np.ones(len(coordinates), dtype=float)
        for dimension, offset in enumerate(corner):
            weights *= fractions[dimension] if offset else 1 - fractions[dimension]
        corner_indexes.append(np.ravel_multi_index(corner_coordinates, shape))
        corner_weights.append(weights)
    indexes = np.column_stack(corner_indexes)
    weights = np.column_stack(corner_weights)

    # Bilinear interpolation changes variance, so normalize by the corners' covariance
    variance = np.einsum("ni,ij,nj->n", weights, field.local_covariance, weights)
    interpolated = np.sum(weights * field.values[indexes], axis=1)
    return interpolated / np.sqrt(np.maximum(variance, np.finfo(float).eps))


def _raster_coordinates(
    transform: Any, full_shape: tuple[int, int], row_start: int, column_start: int, shape: tuple[int, int]
) -> tuple[NDArray[np.int64], NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Calculate global pixel positions and map coordinates for a raster window."""

    row_indexes = np.arange(row_start, row_start + shape[0], dtype=np.int64)
    column_indexes = np.arange(column_start, column_start + shape[1], dtype=np.int64)
    columns, rows = np.meshgrid(column_indexes, row_indexes)
    indexes = (rows * full_shape[1] + columns).reshape(-1)

    # Use pixel centers (including rotation, if present in the transform)
    x = transform.a * (columns + 0.5) + transform.b * (rows + 0.5) + transform.c
    y = transform.d * (columns + 0.5) + transform.e * (rows + 0.5) + transform.f
    coordinates = np.column_stack((x.reshape(-1), y.reshape(-1)))
    axes = (column_indexes.astype(float) * abs(float(transform.a)), row_indexes.astype(float) * abs(float(transform.e)))
    return indexes, coordinates, axes


def _spatial_support(
    like: RasterBase | PointCloudBase,
) -> tuple[NDArray[Any], NDArray[np.float64], tuple[int, ...], Any, str]:
    """Read row positions, coordinates, and output shape from a raster or point cloud."""

    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.raster.base import RasterBase

    if isinstance(like, RasterBase):
        source_ids, coordinates, axes = _raster_coordinates(like.transform, like.shape, 0, 0, like.shape)

        # For an unrotated grid, GSTools only needs the X/Y axes and pixel spacing
        # The full coordinates above are still used to calculate covariance
        if like.transform.b == 0 and like.transform.d == 0:
            random_coordinates: Any = axes
            mesh_type = "structured"
        else:
            random_coordinates = (coordinates[:, 0], coordinates[:, 1])
            mesh_type = "unstructured"
        return source_ids, coordinates, like.shape, random_coordinates, mesh_type
    if isinstance(like, PointCloudBase):
        # Load point locations and use row positions instead of table labels
        x, y, _ = like.to_xyz()
        x = x.compute() if hasattr(x, "compute") else x
        y = y.compute() if hasattr(y, "compute") else y
        coordinates = np.column_stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
        return (
            np.arange(len(coordinates)),
            coordinates,
            (len(coordinates),),
            (coordinates[:, 0], coordinates[:, 1]),
            "unstructured",
        )
    raise TypeError("like must be a raster or point cloud.")


def _raster_spatial_mask(like: RasterBase) -> NDArray[np.bool_]:
    """Return a 2D mask of missing pixels for Raster and Xarray inputs."""

    if hasattr(like, "get_mask"):
        mask = np.asarray(like.get_mask(), dtype=np.bool_)
    else:
        values: Any = like.data
        values = values.compute() if hasattr(values, "compute") else values
        mask = _raster_values_mask(values, like.nodata)

    # A random field has one value per pixel, so mask a pixel only when every source band is invalid
    if mask.ndim == 3:
        mask = np.all(mask, axis=0)
    if mask.shape != like.shape:
        raise ValueError("Raster mask must match its two-dimensional spatial shape.")
    return np.asarray(mask, dtype=np.bool_)


def _wrap_spatial_field(like: RasterBase | PointCloudBase, values: NDArray[np.float64]) -> Any:
    """Return random field values as the same type of raster or point cloud as the source."""

    from geoutils.raster.base import RasterBase

    if isinstance(like, RasterBase):
        array = np.ma.masked_array(values.reshape(like.shape), mask=_raster_spatial_mask(like))
        return like.from_array(
            data=array,
            transform=like.transform,
            crs=like.crs,
            nodata=like.nodata,
            area_or_point=like.area_or_point,
            tags=like.tags,
        )
    return like.copy(new_array=values)


def _component_seeds(
    error_structure: ErrorStructure, n_fields: int, random_state: int | np.random.Generator | None
) -> list[tuple[int, ...]]:
    """Choose one seed per component and field before scheduling any chunks."""

    rng = np.random.default_rng(random_state)
    component_count = len(error_structure.components)
    return [
        tuple(int(seed) for seed in rng.integers(0, np.iinfo(np.uint32).max, component_count, dtype=np.uint32))
        for _ in range(n_fields)
    ]


def _spatial_bounds(like: RasterBase | PointCloudBase) -> tuple[tuple[float, float], tuple[float, float]]:
    """Find X/Y limits without reading a raster or point cloud's full values."""

    from geoutils.raster.base import RasterBase

    if isinstance(like, RasterBase):
        rows = np.array([0, like.shape[0] - 1], dtype=float) + 0.5
        columns = np.array([0, like.shape[1] - 1], dtype=float) + 0.5
        column_grid, row_grid = np.meshgrid(columns, rows)
        x = like.transform.a * column_grid + like.transform.b * row_grid + like.transform.c
        y = like.transform.d * column_grid + like.transform.e * row_grid + like.transform.f
        return (float(x.min()), float(x.max())), (float(y.min()), float(y.max()))
    bounds = like.bounds
    if bounds is None:
        extent = like.ds.total_bounds
        extent = extent.compute() if hasattr(extent, "compute") else extent
        return (float(extent[0]), float(extent[2])), (float(extent[1]), float(extent[3]))
    return (float(bounds.left), float(bounds.right)), (float(bounds.bottom), float(bounds.top))


def _gpytorch_inducing_fields(
    error_structure: ErrorStructure,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    component_seeds: tuple[int, ...],
    n_inducing: int,
) -> tuple[GPyTorchInducingField | None, ...]:
    """Draw one global GP grid per correlated component for interpolation by every chunk."""

    from geoutils.stats.variography import Variogram, VariogramModel

    fields: list[GPyTorchInducingField | None] = []
    for component, seed in zip(error_structure.components.values(), component_seeds):
        correlation = component.correlation
        if correlation is None:
            fields.append(None)
            continue
        assert isinstance(correlation, VariogramModel)
        torch = import_optional("torch", extra_name="gp")
        gpytorch = import_optional("gpytorch", extra_name="gp")

        # A regular inducing grid covers the complete output extent, including rotated raster corners
        selected_bounds = (
            bounds if correlation.active_dims is None else tuple(bounds[d] for d in correlation.active_dims)
        )
        varying_dimensions = sum(upper > lower for lower, upper in selected_bounds)
        points_per_axis = max(2, int(round(n_inducing ** (1 / varying_dimensions)))) if varying_dimensions else 1
        axes = tuple(
            np.linspace(lower, upper, points_per_axis) if upper > lower else np.array([lower], dtype=float)
            for lower, upper in selected_bounds
        )
        shape = tuple(len(axis) for axis in axes)
        grids = np.meshgrid(*axes, indexing="ij")
        inducing_coordinates = np.column_stack([grid.reshape(-1) for grid in grids])
        variogram = Variogram(
            lags=np.empty(0), semivariance=np.empty(0), counts=np.empty(0, dtype=np.int64), model=correlation
        )
        converted = variogram.to_gpytorch(active_dims=tuple(range(len(axes))), trainable=False)

        # Sample the same latent GP values once; workers receive only NumPy arrays
        positions = torch.as_tensor(inducing_coordinates.copy(), dtype=torch.float64)
        covariance = converted.kernel(positions).to_dense()
        if converted.noise > 0:
            covariance = covariance + torch.eye(len(positions), dtype=torch.float64) * converted.noise
        # Closely spaced inducing points can make a smooth kernel numerically singular
        covariance = covariance + torch.eye(len(positions), dtype=torch.float64) * 1e-6
        distribution = gpytorch.distributions.MultivariateNormal(
            torch.zeros(len(positions), dtype=torch.float64), covariance
        )
        generator = torch.Generator(device=positions.device)
        generator.manual_seed(seed)
        base_samples = torch.randn(len(positions), dtype=torch.float64, generator=generator)
        with torch.no_grad(), gpytorch.settings.fast_computations(covar_root_decomposition=False):
            values = distribution.rsample(base_samples=base_samples).detach().cpu().numpy()

        # Stationary kernels give every grid cell the same covariance among its corners
        corners = product(*((0, 1) if length > 1 else (0,) for length in shape))
        corner_indexes = [np.ravel_multi_index(corner, shape) for corner in corners]
        local_covariance = covariance.detach().cpu().numpy()[np.ix_(corner_indexes, corner_indexes)]
        fields.append(GPyTorchInducingField(axes, values, local_covariance))
    return tuple(fields)


def _field_draws(
    error_structure: ErrorStructure,
    like: RasterBase | PointCloudBase,
    n_fields: int,
    random_state: int | np.random.Generator | None,
    backend: Literal["gstools", "gpytorch"],
    gpytorch_inducing_points: int | None,
) -> list[tuple[tuple[int, ...], tuple[GPyTorchInducingField | None, ...] | None]]:
    """Choose field seeds and prepare any shared GP grids before drawing fields."""

    seeds = _component_seeds(error_structure, n_fields, random_state)
    if backend == "gstools":
        return [(field_seeds, None) for field_seeds in seeds]
    bounds = _spatial_bounds(like)
    count = 256 if gpytorch_inducing_points is None else gpytorch_inducing_points
    return [
        (field_seeds, _gpytorch_inducing_fields(error_structure, bounds, field_seeds, count)) for field_seeds in seeds
    ]


def _draw_error_chunk(
    model: ErrorStructure,
    indexes: NDArray[np.int64],
    coordinates: NDArray[np.float64],
    predictors: Mapping[str, Any] | None,
    component_seeds: tuple[int, ...],
    *,
    random_coordinates: Any | None = None,
    mesh_type: str = "unstructured",
    field_shape: tuple[int, ...] | None = None,
    backend: Literal["gstools", "gpytorch"] = "gstools",
    inducing_fields: tuple[GPyTorchInducingField | None, ...] | None = None,
) -> NDArray[np.float64]:
    """Draw one chunk from shared GSTools or GPyTorch random values."""

    if backend == "gpytorch" and inducing_fields is None:
        raise ValueError("Chunked GPyTorch fields require a shared inducing grid.")
    from geoutils.uncertainty.error_structure import _draw_source_errors, _prepare_source_errors

    locations, component_data = _prepare_source_errors(model, len(indexes), coordinates, predictors)
    return _draw_source_errors(
        len(indexes),
        locations,
        component_data,
        np.random.default_rng(0),
        backend=backend,
        component_seeds=component_seeds,
        indexes=indexes,
        inducing_fields=inducing_fields,
        random_coordinates=random_coordinates,
        mesh_type=mesh_type,
        field_shape=field_shape,
    )


############################################
# 2/ EAGER RANDOM FIELD
############################################


def _eager_random_fields(
    error_structure: ErrorStructure,
    coordinates: ArrayLike | None,
    predictors: Mapping[str, Any] | None,
    *,
    like: RasterBase | PointCloudBase | None,
    n_fields: int,
    random_state: int | np.random.Generator | None,
    backend: Literal["gstools", "gpytorch"],
    gpytorch_inducing_points: int | None,
) -> Any:
    """Derive random fields in-memory."""

    # The ``like`` raster or point cloud defines output support, and returned object type (Xarray/Pandas or GeoUtils)
    if like is not None:
        source_ids, coordinates, spatial_shape, random_coordinates, mesh_type = _spatial_support(like)
        predictors = _spatial_predictors(like, predictors, size=int(np.prod(spatial_shape)))
        from geoutils.stats.variography import VariogramModel

        component_dims = []
        for component in error_structure.components.values():
            correlation = component.correlation
            if correlation is not None:
                assert isinstance(correlation, VariogramModel)
                component_dims.append(correlation.active_dims)
        if any(dimensions not in (None, (0, 1)) for dimensions in component_dims):
            # Let each component select its own coordinate dimensions before drawing
            random_coordinates = None
            mesh_type = "unstructured"
    else:
        assert coordinates is not None
        source_ids = np.arange(np.asarray(coordinates).shape[0])
        spatial_shape = None
        random_coordinates = None
        mesh_type = "unstructured"

    from geoutils.uncertainty.error_structure import _draw_source_errors, _prepare_source_errors

    locations, component_data = _prepare_source_errors(error_structure, len(source_ids), coordinates, predictors)

    # When using GPyTorch, we need to induce a grid for the in-memory field to match chunked output
    if backend == "gpytorch" and gpytorch_inducing_points is not None:
        assert like is not None
        indexes = np.arange(len(source_ids), dtype=np.int64)
        fields = np.vstack(
            [
                _draw_source_errors(
                    len(source_ids),
                    locations,
                    component_data,
                    np.random.default_rng(0),
                    backend=backend,
                    component_seeds=seeds,
                    indexes=indexes,
                    inducing_fields=inducing_fields,
                )
                for seeds, inducing_fields in _field_draws(
                    error_structure, like, n_fields, random_state, backend, gpytorch_inducing_points
                )
            ]
        )
    else:
        # Without chunks, draw each complete field in memory
        rng = np.random.default_rng(random_state)
        fields = np.vstack(
            [
                _draw_source_errors(
                    len(source_ids),
                    locations,
                    component_data,
                    rng,
                    backend=backend,
                    random_coordinates=random_coordinates,
                    mesh_type=mesh_type,
                    field_shape=spatial_shape,
                )
                for _ in range(n_fields)
            ]
        )

    # Return spatial objects when given a raster/point cloud, otherwise plain arrays
    if like is not None:
        wrapped = [_wrap_spatial_field(like, field) for field in fields]
        return wrapped[0] if n_fields == 1 else wrapped
    return fields[0] if n_fields == 1 else fields


############################################
# 3/ CHUNKED EXECUTION FOR DASK AND MP
############################################


def _wrapper_draw_raster_tile(
    source_values: Any,
    predictor_values: Mapping[str, Any],
    model: ErrorStructure,
    component_seeds: tuple[int, ...],
    *,
    row_start: int,
    column_start: int,
    transform: Any,
    full_shape: tuple[int, int],
    nodata: float | int | None,
    backend: Literal["gstools", "gpytorch"],
    inducing_fields: tuple[GPyTorchInducingField | None, ...] | None,
) -> NDArray[np.float64]:
    """Wrapper for Dask backend to draw a random field over a raster tile."""

    from geoutils.stats.variography import VariogramModel

    source_shape = np.shape(source_values)
    tile_shape = (int(source_shape[-2]), int(source_shape[-1]))
    indexes, coordinates, axes = _raster_coordinates(transform, full_shape, row_start, column_start, tile_shape)
    # Bind each pixel to the predictor values from this tile
    aligned_predictors = {
        name: value if np.isscalar(value) else np.ma.asarray(value, dtype=float).filled(np.nan).reshape(-1)
        for name, value in predictor_values.items()
    }

    # For an unrotated grid, GSTools only needs the X/Y axes and pixel spacing
    structured = (
        backend == "gstools"
        and transform.b == 0
        and transform.d == 0
        and all(
            component.correlation is None
            or (
                isinstance(component.correlation, VariogramModel)
                and component.correlation.active_dims in (None, (0, 1))
            )
            for component in model.components.values()
        )
    )
    field = _draw_error_chunk(
        model,
        indexes,
        coordinates,
        aligned_predictors,
        component_seeds,
        random_coordinates=axes if structured else None,
        mesh_type="structured" if structured else "unstructured",
        field_shape=tile_shape,
        backend=backend,
        inducing_fields=inducing_fields,
    ).reshape(tile_shape)
    return np.where(_raster_values_mask(source_values, nodata), np.nan, field)


def _chunked_raster_fields_dask(
    like: RasterBase,
    error_structure: ErrorStructure,
    *,
    predictors: Mapping[str, Any] | None,
    n_fields: int,
    random_state: int | np.random.Generator | None,
    chunksizes: tuple[int, int] | None,
    backend: Literal["gstools", "gpytorch"],
    gpytorch_inducing_points: int | None,
) -> Any:
    """Derive random fields per rastr tile lazily with Dask backend."""

    import_optional("dask")
    import dask
    import dask.array as da

    from geoutils.multiproc.chunked import normalize_chunks
    from geoutils.raster.xr_accessor import RasterAccessor

    if chunksizes is not None:
        chunks = normalize_chunks(chunks=chunksizes, shape=like.shape)
    source = _raster_chunk_source(like, chunksizes)
    if chunksizes is None:
        chunks = source.chunks[-2:]
    predictor_chunksize = (max(chunks[0]), max(chunks[1]))
    predictor_data = _raster_chunk_predictors(like, predictors, predictor_chunksize)
    model = error_structure._drawing_model()
    row_starts = np.cumsum((0, *chunks[0][:-1]))
    column_starts = np.cumsum((0, *chunks[1][:-1]))
    fields = []

    draws = _field_draws(error_structure, like, n_fields, random_state, backend, gpytorch_inducing_points)
    for seeds, inducing_fields in draws:
        # We choose seeds once per field, so all chunks belong to the same spatial pattern
        rows = []
        for row_start, row_size in zip(row_starts, chunks[0]):
            row_blocks = []
            for column_start, column_size in zip(column_starts, chunks[1]):
                row_slice = slice(int(row_start), int(row_start + row_size))
                column_slice = slice(int(column_start), int(column_start + column_size))
                tile_predictors = {
                    name: value if np.isscalar(value) else value[row_slice, column_slice]
                    for name, value in predictor_data.items()
                }
                delayed = dask.delayed(_wrapper_draw_raster_tile)(
                    source[..., row_slice, column_slice],
                    tile_predictors,
                    model,
                    seeds,
                    row_start=int(row_start),
                    column_start=int(column_start),
                    transform=like.transform,
                    full_shape=like.shape,
                    nodata=like.nodata,
                    backend=backend,
                    inducing_fields=inducing_fields,
                )
                row_blocks.append(da.from_delayed(delayed, shape=(row_size, column_size), dtype=np.float64))
            rows.append(row_blocks)

        # Assemble the chunks without calculating them, then attach the raster georeferencing
        fields.append(
            RasterAccessor.from_array(
                data=da.block(rows),
                transform=like.transform,
                crs=like.crs,
                nodata=like.nodata,
                area_or_point=like.area_or_point,
            )
        )
    return fields[0] if n_fields == 1 else fields


def _draw_point_rows(
    dataframe: Any,
    model: ErrorStructure,
    predictor_columns: Mapping[str, str | float],
    component_seeds: tuple[int, ...],
    start: int,
    data_column: str | None,
    backend: Literal["gstools", "gpytorch"],
    inducing_fields: tuple[GPyTorchInducingField | None, ...] | None,
) -> Any:
    """Derive random field for point rows: This helper is shared between MP and Dask."""

    if len(dataframe) == 0:
        return dataframe.copy()
    indexes = np.arange(start, start + len(dataframe), dtype=np.int64)
    coordinates = np.column_stack((dataframe.geometry.x.to_numpy(), dataframe.geometry.y.to_numpy()))
    # Named predictors come from the current point partition
    predictors = {
        name: dataframe[value].to_numpy() if isinstance(value, str) else value
        for name, value in predictor_columns.items()
    }
    values = _draw_error_chunk(
        model, indexes, coordinates, predictors, component_seeds, backend=backend, inducing_fields=inducing_fields
    )
    output = dataframe.copy()
    if data_column is None:
        import geopandas as gpd

        output.geometry = gpd.points_from_xy(coordinates[:, 0], coordinates[:, 1], values, crs=dataframe.crs)
    else:
        output[data_column] = values
    return output


def _wrapper_draw_point_partition_dask(
    dataframe: Any,
    model: ErrorStructure,
    predictor_columns: Mapping[str, str | float],
    component_seeds: tuple[int, ...],
    starts: NDArray[np.int64],
    data_column: str | None,
    original_columns: list[str],
    backend: Literal["gstools", "gpytorch"],
    inducing_fields: tuple[GPyTorchInducingField | None, ...] | None,
    partition_info: dict[str, Any] | None = None,
) -> Any:
    """Wrapper for Dask backend to draw a random field over a point partition."""

    if partition_info is None:
        raise RuntimeError("Dask did not provide the point partition number.")
    start = int(starts[partition_info["number"]])
    result = _draw_point_rows(
        dataframe, model, predictor_columns, component_seeds, start, data_column, backend, inducing_fields
    )
    return result[original_columns]


def _chunked_point_fields_dask(
    like: PointCloudBase,
    error_structure: ErrorStructure,
    *,
    predictors: Mapping[str, Any] | None,
    n_fields: int,
    random_state: int | np.random.Generator | None,
    chunksize: int | None,
    backend: Literal["gstools", "gpytorch"],
    gpytorch_inducing_points: int | None,
) -> Any:
    """Derive random fields per point partition lazily with Dask backend."""

    import_optional("dask")
    import pandas as pd

    from geoutils.pointcloud.dataframe import _assign_point_values, _point_partition_lengths

    if chunksize is None:
        dataframe = like.ds
    elif like._is_dask:
        dataframe = like.ds.repartition(npartitions=max(1, int(np.ceil(like.point_count / chunksize))))
    elif not like.is_loaded:
        from geoutils.pointcloud.pd_accessor import open_pointcloud

        assert like.name is not None
        dataframe = open_pointcloud(
            like.name,
            data_column=like.data_column,
            columns=list(like._nongeo_columns),
            chunks=chunksize,
            downsample=getattr(like, "_downsample", 1),
        )
    else:
        dask_geopandas = import_optional("dask_geopandas")
        from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

        _register_dask_pointcloud_accessor()
        dataframe = dask_geopandas.from_geopandas(
            like.ds, npartitions=max(1, int(np.ceil(like.point_count / chunksize))), sort=False
        )
    from geoutils.pointcloud.dataframe import _get_dataframe_attrs

    source_attrs = _get_dataframe_attrs(dataframe)
    original_columns = list(dataframe.columns)
    lengths = _point_partition_lengths(dataframe)
    starts = np.cumsum((0, *lengths))
    predictor_columns, arrays = _point_predictor_columns(like, predictors, size=int(starts[-1]))
    if arrays:
        dataframe = _assign_point_values(dataframe, arrays, partition_lengths=lengths)
    meta = dataframe._meta[original_columns].copy()
    if like.data_column is not None:
        meta[like.data_column] = pd.Series([], dtype=np.float64)

    fields = []
    model = error_structure._drawing_model()
    draws = _field_draws(error_structure, like, n_fields, random_state, backend, gpytorch_inducing_points)
    for seeds, inducing_fields in draws:
        result = dataframe.map_partitions(
            _wrapper_draw_point_partition_dask,
            model,
            predictor_columns,
            seeds,
            starts,
            like.data_column,
            original_columns,
            backend,
            inducing_fields,
            meta=meta,
        )
        from geoutils.pointcloud.dataframe import _build_pointcloud_output

        fields.append(
            _build_pointcloud_output(
                result,
                data_column=like.data_column,
                as_dataframe=True,
                attrs=source_attrs,
                preserve_locations=True,
            )
        )
    return fields[0] if n_fields == 1 else fields


def _field_multiproc_config(mp_config: MultiprocConfig, field_index: int, n_fields: int) -> MultiprocConfig:
    """We give each multiprocessing field its own output file."""

    from geoutils.multiproc import MultiprocConfig

    if n_fields == 1:
        return mp_config
    path = Path(mp_config.outfile)
    output = path.with_name(f"{path.stem}_field_{field_index + 1}{path.suffix}")
    return MultiprocConfig(
        chunks=mp_config.chunks, outfile=str(output), driver=mp_config.driver, cluster=mp_config.cluster
    )


def _wrapper_draw_raster_tile_multiproc(
    tile: RasterBase,
    model: ErrorStructure,
    predictors: Mapping[str, Any] | None,
    component_seeds: tuple[int, ...],
    transform: Any,
    full_shape: tuple[int, int],
    backend: Literal["gstools", "gpytorch"],
    inducing_fields: tuple[GPyTorchInducingField | None, ...] | None,
) -> Any:
    """Wrapper for MP backend to draw a random field over a raster tile."""

    from geoutils.multiproc.mparray import _load_raster_tile
    from geoutils.raster import Raster
    from geoutils.raster.base import RasterBase

    column_start, row_start = ~(transform) * (tile.transform.c, tile.transform.f)
    row_start, column_start = int(round(row_start)), int(round(column_start))
    bounds = np.array([row_start, row_start + tile.shape[0], column_start, column_start + tile.shape[1]])
    tile_predictors: dict[str, Any] = {}
    for name, predictor in (predictors or {}).items():
        # Read only the matching predictor tile, leaving source rasters unopened
        if np.isscalar(predictor) and not isinstance(predictor, str):
            tile_predictors[name] = predictor
        elif isinstance(predictor, RasterBase):
            tile_predictors[name] = _load_raster_tile(cast(Raster, predictor), bounds).data
        else:
            value = (
                predictor.data if hasattr(predictor, "data") and not isinstance(predictor, np.ndarray) else predictor
            )
            tile_predictors[name] = value[..., bounds[0] : bounds[1], bounds[2] : bounds[3]]
    values = _wrapper_draw_raster_tile(
        tile.data,
        tile_predictors,
        model,
        component_seeds,
        row_start=row_start,
        column_start=column_start,
        transform=transform,
        full_shape=full_shape,
        nodata=tile.nodata,
        backend=backend,
        inducing_fields=inducing_fields,
    )
    return Raster.from_array(
        np.ma.masked_invalid(values),
        transform=tile.transform,
        crs=tile.crs,
        nodata=tile.nodata,
        area_or_point=tile.area_or_point,
        tags=dict(tile.tags),
    )


def _chunked_raster_fields_multiproc(
    like: RasterBase,
    error_structure: ErrorStructure,
    *,
    predictors: Mapping[str, Any] | None,
    n_fields: int,
    random_state: int | np.random.Generator | None,
    mp_config: MultiprocConfig,
    backend: Literal["gstools", "gpytorch"],
    gpytorch_inducing_points: int | None,
) -> Any:
    """Derive random fields per raster tile and write to file with MP backend."""

    from geoutils.multiproc import map_overlap

    predictor_data = _raster_chunk_predictors(like, predictors, None)
    fields = []
    model = error_structure._drawing_model()
    draws = _field_draws(error_structure, like, n_fields, random_state, backend, gpytorch_inducing_points)
    for index, (seeds, inducing_fields) in enumerate(draws):
        output_config = _field_multiproc_config(mp_config, index, n_fields)
        fields.append(
            map_overlap(
                _wrapper_draw_raster_tile_multiproc,
                like,
                output_config,
                model,
                predictor_data,
                seeds,
                like.transform,
                like.shape,
                backend,
                inducing_fields,
            )
        )
    return fields[0] if n_fields == 1 else fields


def _wrapper_draw_point_partition_multiproc(
    dataframe: Any,
    model: ErrorStructure,
    predictor_columns: Mapping[str, str | float],
    predictor_arrays: Mapping[str, Any],
    component_seeds: tuple[int, ...],
    start: int,
    data_column: str | None,
    original_columns: list[str],
    backend: Literal["gstools", "gpytorch"],
    inducing_fields: tuple[GPyTorchInducingField | None, ...] | None,
) -> Any:
    """Wrapper for MP backend to draw a random field over a point partition."""

    source = dataframe.copy()
    for name, values in predictor_arrays.items():
        source[name] = values
    result = _draw_point_rows(
        source, model, predictor_columns, component_seeds, start, data_column, backend, inducing_fields
    )
    return result[original_columns]


def _chunked_point_fields_multiproc(
    like: PointCloudBase,
    error_structure: ErrorStructure,
    *,
    predictors: Mapping[str, Any] | None,
    n_fields: int,
    random_state: int | np.random.Generator | None,
    mp_config: MultiprocConfig,
    backend: Literal["gstools", "gpytorch"],
    gpytorch_inducing_points: int | None,
) -> Any:
    """Derive random fields per point partition and write to file with MP backend."""

    from geoutils.multiproc.cluster import _map_bounded
    from geoutils.pointcloud.loading import _load_pointcloud_rows
    from geoutils.pointcloud.writing import (
        _resolve_pointcloud_output,
        _stage_pointcloud_partition,
        _write_pointcloud_partitions,
    )

    chunk_size = mp_config.chunks
    if not isinstance(chunk_size, int):
        raise ValueError("Point cloud multiprocessing requires an integer chunk size.")
    point_count = like.point_count
    predictor_columns, arrays = _point_predictor_columns(like, predictors, size=point_count)
    original_columns = list(like.columns)
    fields = []
    model = error_structure._drawing_model()

    draws = _field_draws(error_structure, like, n_fields, random_state, backend, gpytorch_inducing_points)
    for field_index, (seeds, inducing_fields) in enumerate(draws):
        output_config = _field_multiproc_config(mp_config, field_index, n_fields)
        output_path, driver = _resolve_pointcloud_output(
            output_config.outfile,
            output_config.driver,
            supported_drivers=("GPKG",),
            operation_name="random field generation",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".geoutils-random-field-", dir=output_path.parent) as directory:

            def arguments(
                field_seeds: tuple[int, ...], field_inducing: tuple[GPyTorchInducingField | None, ...] | None
            ) -> Any:
                """Yield bounded point rows and aligned predictors in source order."""

                for start in range(0, max(point_count, 1), chunk_size):
                    count = min(chunk_size, point_count - start)
                    partition = _load_pointcloud_rows(like, start, count)
                    tile_arrays = {name: np.asarray(value)[start : start + count] for name, value in arrays.items()}
                    yield (
                        partition,
                        model,
                        predictor_columns,
                        tile_arrays,
                        field_seeds,
                        start,
                        like.data_column,
                        original_columns,
                        backend,
                        field_inducing,
                    )

            saved = []
            tasks = arguments(seeds, inducing_fields)
            for index, result in _map_bounded(output_config.cluster, _wrapper_draw_point_partition_multiproc, tasks):
                saved.append(_stage_pointcloud_partition(result, Path(directory) / f"field_{index}.pkl"))
            output = _write_pointcloud_partitions(
                output_path,
                saved,
                driver=driver,
                data_column=like.data_column,
                geometry_type="Point Z" if like._has_z else "Point",
            )
        if like._is_pd:
            from geoutils.pointcloud.transformation import _cast_multiproc_output

            fields.append(_cast_multiproc_output(like, output))
        else:
            fields.append(output)
    return fields[0] if n_fields == 1 else fields


############################################
# 4/ PARENT RANDOM FIELD GENERATION
############################################


def random_field(
    error_structure: ErrorStructure,
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
    """
    Generate random correlated error fields.

    This function supports chunked execution with Dask/Multiprocessing, with spatial correlation that continues across
    chunk boundaries by sharing component seeds across the field.

    The random field can be generated on a raster or point cloud support passed to ``like``, and can account for
    multiple independent error component and heteroscedasticity as described in ``ErrorStructure``.

    Two backends are supported, with a slightly different approach:
    - GSTools, which reuses a global coordinate seed for chunk invariance,
    - GPyTorch, which reuses a shared inducing grid for chunk invariance.

    :param error_structure: Error model defined by magnitude and correlation components.
    :param like: Optional Raster, Xarray DataArray, PointCloud, or GeoDataFrame defining output locations and type.
    :param coordinates: Array of shape (n_observations, n_dimensions), in the correlation model's distance units.
    :param predictors: Named values used to calculate each component's error magnitude (e.g. slope or elevation).
    :param n_fields: Number of independent fields.
    :param random_state: Seed or generator used for reproducible fields.
    :param chunksizes: Dask raster chunk size as (rows, columns), or target point rows per Dask partition.
        Requires an Xarray raster or Dask GeoDataFrame so the result has the same type as the input.
    :param mp_config: Worker and output file settings for multiprocessing fields. Xarray rasters and Dask
        GeoDataFrames cannot use multiprocessing because its output has a different type.
    :param backend: Library used to draw correlated errors; GPyTorch is the default.
    :param gpytorch_inducing_points: Target grid size for approximate GPyTorch fields. The default, "auto", uses
        256 points for spatial inputs and an exact draw for plain coordinates. Pass None for an exact eager draw.
    :returns: With like, a result of the same spatial object type, or a list when n_fields > 1. Dask results remain
        lazy. Multiprocessing Raster and PointCloud results are file-backed. Without like, an array of shape
        (n_observations,) or (n_fields, n_observations).
    """

    # 1/ Input checks
    if not isinstance(error_structure, ErrorStructure):
        raise TypeError("error_structure must be an ErrorStructure.")
    if backend not in {"gstools", "gpytorch"}:
        raise ValueError("Random-field backend must be 'gstools' or 'gpytorch'.")
    if gpytorch_inducing_points == "auto":
        gpytorch_inducing_points = 256 if backend == "gpytorch" and like is not None else None
    if gpytorch_inducing_points is not None:
        if backend != "gpytorch":
            raise ValueError("gpytorch_inducing_points requires the GPyTorch backend.")
        if (
            isinstance(gpytorch_inducing_points, bool)
            or not isinstance(gpytorch_inducing_points, (int, np.integer))
            or gpytorch_inducing_points < 4
        ):
            raise ValueError("gpytorch_inducing_points must be an integer of at least 4.")
        if like is None:
            raise ValueError("An inducing grid requires a raster or point cloud defining spatial bounds.")
    if isinstance(n_fields, (bool, np.bool_)) or not isinstance(n_fields, (int, np.integer)) or n_fields < 1:
        raise ValueError("n_fields must be a positive integer.")

    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.raster.base import RasterBase

    if like is not None and not isinstance(like, (RasterBase, PointCloudBase)):
        raster = _get_raster_interface(like)
        point = _get_pointcloud_interface(like) if raster is None else None
        like = raster if raster is not None else point
        if like is None or not isinstance(like, (RasterBase, PointCloudBase)):
            raise TypeError("like must be a GeoUtils raster or point cloud.")
    if like is not None and coordinates is not None:
        raise ValueError("coordinates must be omitted when like defines the output locations.")

    if like is not None and not isinstance(like, (RasterBase, PointCloudBase)):
        raise TypeError("like must be a GeoUtils raster or point cloud.")
    if chunksizes is not None and mp_config is not None:
        raise ValueError("Choose chunksizes for Dask or mp_config for multiprocessing.")
    if chunksizes is not None:
        if isinstance(like, RasterBase) and (not isinstance(chunksizes, tuple) or len(chunksizes) != 2):
            raise ValueError("Raster chunk sizes must be a (rows, columns) tuple.")
        if isinstance(like, PointCloudBase) and (
            isinstance(chunksizes, bool) or not isinstance(chunksizes, int) or chunksizes <= 0
        ):
            raise ValueError("Point cloud chunk size must be a positive integer.")
        if isinstance(like, RasterBase) and not like._is_xr:
            raise ValueError("Dask raster chunks require an Xarray input.")
        if isinstance(like, PointCloudBase) and not like._is_dask:
            raise ValueError("Dask point chunks require a Dask GeoDataFrame input.")
    if mp_config is not None:
        if isinstance(like, RasterBase) and like._is_xr:
            raise ValueError("Multiprocessing raster fields require a Raster input.")
        if isinstance(like, PointCloudBase) and like._is_dask:
            raise ValueError("Multiprocessing point fields require a PointCloud or GeoDataFrame input.")
    dask_raster = isinstance(like, RasterBase) and like._is_xr and hasattr(like.data, "compute")
    dask_points = isinstance(like, PointCloudBase) and like._is_dask
    chunked = chunksizes is not None or mp_config is not None or dask_raster or dask_points
    if mp_config is not None and (dask_raster or dask_points):
        raise ValueError("Multiprocessing cannot be combined with a Dask raster or point cloud.")
    if chunked and like is None:
        raise ValueError("Chunked random fields require a raster or point cloud.")
    if like is None:
        if coordinates is None:
            raise ValueError("Provide like or coordinates to define the output locations.")
        coordinate_array = np.asarray(coordinates)
        if coordinate_array.ndim != 2 or 0 in coordinate_array.shape:
            raise ValueError(
                "coordinates must have shape (n_observations, n_dimensions) with at least one observation."
            )
        coordinates = coordinate_array

    # 2/ Dispatch to Dask/multiprocessing/eager backends, differentiating point/raster output

    # Raster
    if isinstance(like, RasterBase):
        # Multiproc
        if mp_config is not None:
            return _chunked_raster_fields_multiproc(
                like,
                error_structure,
                predictors=predictors,
                n_fields=n_fields,
                random_state=random_state,
                mp_config=mp_config,
                backend=backend,
                gpytorch_inducing_points=gpytorch_inducing_points,
            )
        # Dask
        if chunksizes is not None or dask_raster:
            return _chunked_raster_fields_dask(
                like,
                error_structure,
                predictors=predictors,
                n_fields=n_fields,
                random_state=random_state,
                chunksizes=chunksizes if isinstance(chunksizes, tuple) else None,
                backend=backend,
                gpytorch_inducing_points=gpytorch_inducing_points,
            )

    # Point
    if isinstance(like, PointCloudBase):
        # Multiproc
        if mp_config is not None:
            return _chunked_point_fields_multiproc(
                like,
                error_structure,
                predictors=predictors,
                n_fields=n_fields,
                random_state=random_state,
                mp_config=mp_config,
                backend=backend,
                gpytorch_inducing_points=gpytorch_inducing_points,
            )
        # Dask
        if chunksizes is not None or dask_points:
            return _chunked_point_fields_dask(
                like,
                error_structure,
                predictors=predictors,
                n_fields=n_fields,
                random_state=random_state,
                chunksize=chunksizes if isinstance(chunksizes, int) else None,
                backend=backend,
                gpytorch_inducing_points=gpytorch_inducing_points,
            )

    # Eager for both point/raster
    return _eager_random_fields(
        error_structure,
        coordinates,
        predictors,
        like=like,
        n_fields=n_fields,
        random_state=random_state,
        backend=backend,
        gpytorch_inducing_points=gpytorch_inducing_points,
    )
