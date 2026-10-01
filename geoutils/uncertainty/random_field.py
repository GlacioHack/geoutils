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
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from geoutils._misc import import_optional
from geoutils.uncertainty.error_structure import BoundErrorStructure, ErrorStructure, _indexed_standard_normal

if TYPE_CHECKING:
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.raster.base import RasterBase

RandomFieldBackend = Literal["gstools", "gpytorch"]


############################################
# 1/ INPUT COORDINATES, PREDICTORS AND MASKS
############################################


def _spatial_support(
    like: RasterBase | PointCloudBase,
) -> tuple[NDArray[Any], NDArray[np.float64], tuple[int, ...], Any, str]:
    """Read IDs, coordinates, and output shape from a raster or point cloud."""

    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.raster.base import RasterBase

    if isinstance(like, RasterBase):
        # Use pixel centers (including rotation, if present in the transform)
        rows, columns = np.indices(like.shape, dtype=float)
        centered_columns = columns + 0.5
        centered_rows = rows + 0.5
        x_grid = like.transform.a * centered_columns + like.transform.b * centered_rows + like.transform.c
        y_grid = like.transform.d * centered_columns + like.transform.e * centered_rows + like.transform.f
        coordinates = np.column_stack((x_grid.reshape(-1), y_grid.reshape(-1)))

        # For an unrotated grid, GSTools only needs the X/Y axes and pixel spacing
        # The full coordinates above are still used to calculate covariance
        if like.transform.b == 0 and like.transform.d == 0:
            x = np.arange(like.shape[1], dtype=float) * abs(float(like.res[0]))
            y = np.arange(like.shape[0], dtype=float) * abs(float(like.res[1]))
            random_coordinates: Any = (x, y)
            mesh_type = "structured"
        else:
            random_coordinates = (coordinates[:, 0], coordinates[:, 1])
            mesh_type = "unstructured"
        return np.arange(coordinates.shape[0]), coordinates, like.shape, random_coordinates, mesh_type
    if isinstance(like, PointCloudBase):
        # Load point locations and IDs before drawing their error field
        x, y, _ = like.to_xyz()
        x = x.compute() if hasattr(x, "compute") else x
        y = y.compute() if hasattr(y, "compute") else y
        coordinates = np.column_stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
        point_index = like.ds.index
        point_index = point_index.compute() if hasattr(point_index, "compute") else point_index
        source_ids = np.asarray(point_index)

        # Duplicate table labels still represent separate observations, so use row numbers in that case
        if not point_index.is_unique:
            source_ids = np.arange(len(coordinates))
        return (
            source_ids,
            coordinates,
            (len(coordinates),),
            (coordinates[:, 0], coordinates[:, 1]),
            "unstructured",
        )
    raise TypeError("like must be a GeoUtils raster or point cloud.")


def _spatial_predictors(
    like: RasterBase | PointCloudBase,
    predictors: Mapping[str, Any] | None,
    *,
    size: int,
) -> Mapping[str, Any] | None:
    """Arrange predictor arrays or point cloud columns as one value per output location."""

    if predictors is None:
        return None
    from geoutils.pointcloud.base import PointCloudBase

    aligned: dict[str, Any] = {}
    for name, predictor in predictors.items():
        # A point cloud column name is a convenient alternative to supplying its values
        if isinstance(like, PointCloudBase) and isinstance(predictor, str):
            if predictor not in like.ds.columns:
                raise ValueError(f"Point cloud predictor column {predictor!r} does not exist.")
            value: Any = like.ds[predictor]
        else:
            value = (
                predictor.data if hasattr(predictor, "data") and not isinstance(predictor, np.ndarray) else predictor
            )
        value = value.compute() if hasattr(value, "compute") else value
        array = np.ma.asarray(value, dtype=float).filled(np.nan)

        # A single value applies everywhere; arrays must describe every pixel/point
        if array.ndim == 0:
            aligned[name] = float(array)
        elif array.size == size:
            aligned[name] = np.asarray(array).reshape(-1)
        else:
            raise ValueError(f"Spatial predictor {name!r} must be scalar or match every output location.")
    return aligned


def _raster_spatial_mask(like: RasterBase) -> NDArray[np.bool_]:
    """Return a 2D mask of missing pixels for Raster and Xarray inputs."""

    if hasattr(like, "get_mask"):
        mask = np.asarray(like.get_mask(), dtype=np.bool_)
    else:
        values: Any = like.data
        values = values.compute() if hasattr(values, "compute") else values
        array = np.asanyarray(values)
        mask = np.ma.getmaskarray(array) | ~np.isfinite(np.ma.getdata(array))
        if like.nodata is not None:
            mask |= np.ma.getdata(array) == like.nodata

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


############################################
# 2/ IN-MEMORY CALCULATION FOR A RASTER TILE
############################################


def _draw_raster_tile(
    component_data: tuple[tuple[Any, NDArray[np.float64]], ...],
    component_seeds: tuple[int, ...],
    *,
    row_start: int,
    row_stop: int,
    column_start: int,
    column_stop: int,
    transform: Any,
    full_shape: tuple[int, int],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Draw errors for one raster tile, using the same seed for each component across all tiles."""

    # Number pixels in the full raster, so a different tile size cannot change their random errors
    tile_shape = (row_stop - row_start, column_stop - column_start)
    row_indexes = np.arange(row_start, row_stop, dtype=np.int64)
    column_indexes = np.arange(column_start, column_stop, dtype=np.int64)
    columns, rows = np.meshgrid(column_indexes, row_indexes)
    flattened_indexes = rows * full_shape[1] + columns
    output = np.zeros(tile_shape, dtype=np.float64)

    for (component, magnitude), seed in zip(component_data, component_seeds):
        tile_magnitude = magnitude.reshape(full_shape)[row_start:row_stop, column_start:column_stop]
        if component.correlation is None:
            unit_field = _indexed_standard_normal(seed, flattened_indexes)
        else:
            from geoutils.stats.variography import Variogram

            gstools = import_optional("gstools", extra_name="geostat")
            correlation = component.correlation
            variogram = Variogram(
                lags=np.empty(0),
                semivariance=np.empty(0),
                counts=np.empty(0, dtype=np.int64),
                model=correlation,
            )
            structured = transform.b == 0 and transform.d == 0 and correlation.active_dims in (None, (0, 1))
            if structured:
                # GSTools returns values in X/Y order; transpose them to raster rows/columns
                x = column_indexes.astype(float) * abs(float(transform.a))
                y = row_indexes.astype(float) * abs(float(transform.e))
                converted = variogram.to_gstools(dim=2)
                unit_field = np.asarray(
                    gstools.SRF(converted.model, seed=seed)((x, y), mesh_type="structured"), dtype=float
                ).T
            else:
                # Rotated grids or models using only some coordinates need explicit point locations
                x = transform.a * (columns + 0.5) + transform.b * (rows + 0.5) + transform.c
                y = transform.d * (columns + 0.5) + transform.e * (rows + 0.5) + transform.f
                coordinates = np.column_stack((x.reshape(-1), y.reshape(-1)))
                active_dims = correlation.active_dims
                if active_dims is not None:
                    coordinates = coordinates[:, active_dims]
                converted = variogram.to_gstools(dim=coordinates.shape[1])
                positions = tuple(coordinates[:, dimension] for dimension in range(coordinates.shape[1]))
                unit_field = np.asarray(
                    gstools.SRF(converted.model, seed=seed)(positions, mesh_type="unstructured"), dtype=float
                ).reshape(tile_shape)

        # Each field has unit variance; scale it to this component's error magnitude before adding the components
        output += tile_magnitude * unit_field
    return np.where(mask[row_start:row_stop, column_start:column_stop], np.nan, output)


############################################
# 3/ DASK CALCULATION OVER RASTER CHUNKS
############################################


def _chunked_raster_fields(
    like: RasterBase,
    bound: BoundErrorStructure,
    *,
    n_fields: int,
    random_state: int | np.random.Generator | None,
    chunksizes: tuple[int, int],
) -> Any:
    """Build Dask rasters with one delayed call to _draw_raster_tile() per chunk."""

    import_optional("dask")
    import dask
    import dask.array as da

    from geoutils.multiproc.chunked import normalize_chunks
    from geoutils.raster.xr_accessor import RasterAccessor

    # Work out where each chunk falls in the full raster (the last row/column may be shorter)
    chunks = normalize_chunks(chunks=chunksizes, shape=like.shape)
    row_starts = np.cumsum((0, *chunks[0][:-1]))
    column_starts = np.cumsum((0, *chunks[1][:-1]))
    mask = _raster_spatial_mask(like)
    rng = np.random.default_rng(random_state)
    component_count = len(bound._component_data)
    fields = []
    for _ in range(n_fields):
        # We choose seeds once per field, so all chunks belong to the same spatial pattern
        seeds = tuple(
            int(value) for value in rng.integers(0, np.iinfo(np.uint32).max, size=component_count, dtype=np.uint32)
        )
        rows = []
        for row_start, row_size in zip(row_starts, chunks[0]):
            row_blocks = []
            for column_start, column_size in zip(column_starts, chunks[1]):
                delayed = dask.delayed(_draw_raster_tile)(
                    bound._component_data,
                    seeds,
                    row_start=int(row_start),
                    row_stop=int(row_start + row_size),
                    column_start=int(column_start),
                    column_stop=int(column_start + column_size),
                    transform=like.transform,
                    full_shape=like.shape,
                    mask=mask,
                )
                row_blocks.append(da.from_delayed(delayed, shape=(row_size, column_size), dtype=np.float64))
            rows.append(row_blocks)

        # Assemble the chunks without calculating them, then attach the raster georeferencing
        data = da.block(rows)
        fields.append(
            RasterAccessor.from_array(
                data=data,
                transform=like.transform,
                crs=like.crs,
                nodata=like.nodata,
                area_or_point=like.area_or_point,
            )
        )
    return fields[0] if n_fields == 1 else fields


############################################
# 4/ GENERATE FIELDS FROM AN ERROR MODEL
############################################


def random_field(
    error_structure: ErrorStructure,
    source_ids: ArrayLike | None = None,
    *,
    like: RasterBase | PointCloudBase | None = None,
    coordinates: ArrayLike | None = None,
    predictors: Mapping[str, Any] | None = None,
    n_fields: int = 1,
    random_state: int | np.random.Generator | None = None,
    chunksizes: tuple[int, int] | None = None,
    backend: RandomFieldBackend = "gstools",
) -> Any:
    """Generate error fields whose spatial correlation continues across chunk boundaries.

    _spatial_support() reads locations from a raster/point cloud, and ErrorStructure.bind() calculates the error
    magnitudes there. We then draw all errors together, or use _chunked_raster_fields() to build a Dask raster. Raster
    chunks share their random seeds, so changing their size does not create breaks in the spatial pattern.

    :param error_structure: Error model defined by magnitude and correlation components.
    :param source_ids: Unique ID of every observation when like does not define the output locations.
    :param like: Optional raster or point cloud defining coordinates, output locations, and result type.
    :param coordinates: Array of shape (n_observations, n_dimensions), in the correlation model's distance units.
    :param predictors: Named values used to calculate each component's error magnitude (e.g. slope or elevation).
    :param n_fields: Number of independent fields.
    :param random_state: Seed or generator used for reproducible fields.
    :param chunksizes: Optional Dask raster chunk size, as (rows, columns).
    :param backend: Library used to draw correlated errors. GSTools supports regular grids and chunked calculation;
        GPyTorch draws all observation errors together from its covariance model.
    :returns: With like, a raster/point cloud or a list of them when n_fields > 1. Without like, an array of shape
        (n_observations,) or (n_fields, n_observations).
    """

    if not isinstance(error_structure, ErrorStructure):
        raise TypeError("error_structure must be an ErrorStructure.")
    if backend not in {"gstools", "gpytorch"}:
        raise ValueError("Random-field backend must be 'gstools' or 'gpytorch'.")
    if isinstance(n_fields, (bool, np.bool_)) or not isinstance(n_fields, (int, np.integer)) or n_fields < 1:
        raise ValueError("n_fields must be a positive integer.")

    # A raster or point cloud supplies all output locations and determines the returned object type
    if like is not None:
        if source_ids is not None or coordinates is not None:
            raise ValueError("source_ids and coordinates must be omitted when like defines the output locations.")
        source_ids, coordinates, spatial_shape, random_coordinates, mesh_type = _spatial_support(like)
        predictors = _spatial_predictors(like, predictors, size=int(np.prod(spatial_shape)))
        from geoutils.stats.variography import VariogramModel

        component_dims: set[tuple[int, ...] | None] = set()
        for component in error_structure.components.values():
            correlation = component.correlation
            if correlation is None:
                continue
            if not isinstance(correlation, VariogramModel):
                raise AssertionError("A validated error component must contain a VariogramModel.")
            component_dims.add(correlation.active_dims)
        if any(dimensions not in (None, (0, 1)) for dimensions in component_dims):
            # Let each component select its own coordinate dimensions before drawing
            random_coordinates = None
            mesh_type = "unstructured"
    else:
        spatial_shape = None
        random_coordinates = None
        mesh_type = "unstructured"

    if source_ids is None:
        raise ValueError("source_ids are required when like is not supplied.")
    bound = error_structure.bind(source_ids, coordinates=coordinates, predictors=predictors)

    # Chunked fields are available for rasters through GSTools
    if chunksizes is not None:
        from geoutils.raster.base import RasterBase

        if like is None or not isinstance(like, RasterBase):
            raise ValueError("Chunked random fields currently require a raster.")
        if backend != "gstools":
            raise ValueError("Chunked random fields require the GSTools backend.")
        return _chunked_raster_fields(
            like,
            bound,
            n_fields=int(n_fields),
            random_state=random_state,
            chunksizes=chunksizes,
        )

    # Without chunks, draw each complete field in memory
    rng = np.random.default_rng(random_state)
    fields = np.vstack(
        [
            bound.draw_error(
                rng,
                backend=backend,
                random_coordinates=random_coordinates,
                mesh_type=mesh_type,
                field_shape=spatial_shape,
            )
            for _ in range(int(n_fields))
        ]
    )

    # Return spatial objects when given a raster/point cloud, otherwise plain arrays
    if like is not None:
        wrapped = [_wrap_spatial_field(like, field) for field in fields]
        return wrapped[0] if n_fields == 1 else wrapped
    return fields[0] if n_fields == 1 else fields
