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

"""Prepare error predictors and predict variable magnitude from them with Dask/MP support."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.spatial import QhullError

from geoutils._dispatch import _get_pointcloud_interface, _get_raster_interface, is_dask_array, is_dask_dataframe
from geoutils._misc import import_optional

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.raster.base import RasterBase
    from geoutils.uncertainty.error_structure import ErrorStructure


############################################
# 1/ PREPARING SPATIAL PREDICTORS
############################################


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
    from geoutils.raster.base import RasterBase

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
        # Eager fields read predictor values before binding the complete support
        value = value.compute() if hasattr(value, "compute") else value

        # Represent masked predictor values as NaN before prediction
        array = np.ma.asarray(value, dtype=float).filled(np.nan)

        # A single value applies everywhere; arrays must describe every pixel/point
        if array.ndim == 0:
            aligned[name] = float(array)
        elif array.size == size:
            aligned[name] = np.asarray(array).reshape(-1)
        elif isinstance(like, RasterBase) and array.size == np.prod(like.shape):
            # A single predictor grid applies to every band of a multiband magnitude map
            aligned[name] = np.broadcast_to(array.reshape(like.shape), (like.count, *like.shape)).reshape(-1)
        else:
            raise ValueError(f"Spatial predictor {name!r} must be scalar or match every output location.")
    return aligned


def _raster_values_mask(values: Any, nodata: float | int | None) -> NDArray[np.bool_]:
    """Mark pixels with no finite value in any source band."""

    array = np.asanyarray(values)
    data = np.ma.getdata(array)
    mask = np.ma.getmaskarray(array) | ~np.isfinite(data)
    if nodata is not None:
        mask |= data == nodata
    if mask.ndim == 3:
        # A pixel remains usable when at least one band has data
        mask = np.all(mask, axis=0)
    return np.asarray(mask, dtype=np.bool_)


def _raster_chunk_source(like: RasterBase, chunksizes: tuple[int, int] | None) -> Any:
    """Expose raster values as tile-readable data without loading an unopened file."""

    # Open file data as Dask tiles so the complete raster stays unloaded
    if not like._is_xr and not like.is_loaded:
        from geoutils.raster.xr_accessor import open_raster

        if like.name is None:
            raise ValueError("An unloaded raster must have a source filename.")
        assert chunksizes is not None
        return open_raster(like.name, chunks={"y": chunksizes[0], "x": chunksizes[1]}).data
    return like.data


def _raster_chunk_predictors(
    like: RasterBase, predictors: Mapping[str, Any] | None, chunksizes: tuple[int, int] | None
) -> dict[str, Any]:
    """Check predictor shapes and expose raster values for spatial slicing."""

    from geoutils.raster.base import RasterBase

    aligned: dict[str, Any] = {}
    for name, predictor in (predictors or {}).items():
        # Scalars apply to every tile without reading a raster
        if np.isscalar(predictor) and not isinstance(predictor, str):
            aligned[name] = float(cast(Any, predictor))
            continue
        # Open raster files in Dask chunks; multiprocessing workers read their own tile
        if isinstance(predictor, RasterBase):
            value = predictor if chunksizes is None else _raster_chunk_source(predictor, chunksizes)
        else:
            value = (
                predictor.data if hasattr(predictor, "data") and not isinstance(predictor, np.ndarray) else predictor
            )
        if not hasattr(value, "shape"):
            value = np.asarray(value)
        if len(value.shape) == 0:
            aligned[name] = float(np.asarray(value).item())
            continue
        # Reshape flat arrays to the raster grid before slicing tiles
        spatial_size = int(np.prod(like.shape))
        full_shape = (like.count, *like.shape) if like.count > 1 else like.shape

        # Accept one shared grid or separate values for every band
        if np.prod(value.shape) not in (spatial_size, np.prod(full_shape)):
            raise ValueError(f"Spatial predictor {name!r} must be scalar or match every output location.")
        shape = full_shape if np.prod(value.shape) == np.prod(full_shape) else like.shape
        aligned[name] = value if isinstance(value, RasterBase) else value.reshape(shape)
    return aligned


def _point_predictor_columns(
    like: PointCloudBase, predictors: Mapping[str, Any] | None, *, size: int
) -> tuple[dict[str, str | float], dict[str, Any]]:
    """Resolve point column names and separate arrays that need row alignment."""

    columns: dict[str, str | float] = {}
    arrays: dict[str, Any] = {}
    for name, predictor in (predictors or {}).items():
        # Column names are resolved separately by each point partition
        if isinstance(predictor, str):
            if predictor not in like.columns:
                raise ValueError(f"Point cloud predictor column {predictor!r} does not exist.")
            columns[name] = predictor
        elif np.isscalar(predictor):
            columns[name] = float(cast(Any, predictor))
        else:
            # Use array data from spatial objects, but leave indexed Series intact
            value = (
                predictor.data
                if hasattr(predictor, "data")
                and not hasattr(predictor, "index")
                and not isinstance(predictor, np.ndarray)
                else predictor
            )
            if not hasattr(value, "shape"):
                value = np.asarray(value)
            if len(value.shape) == 0:
                columns[name] = float(value)
                continue

            # Dask Series already follows row partitions; arrays need one value per point
            if not is_dask_dataframe(value) and np.prod(value.shape) != size:
                raise ValueError(f"Spatial predictor {name!r} must be scalar or match every output location.")
            # Temporary columns align supplied arrays with point rows in Dask/MP
            temporary_name = f"__uncertainty_predictor_{len(arrays)}"
            while temporary_name in like.columns:
                temporary_name = "_" + temporary_name
            columns[name] = temporary_name
            arrays[temporary_name] = value if is_dask_dataframe(value) else value.reshape(-1)
    return columns, arrays


############################################
# 2/ INTERPOLATING GROUPED STATISTICS
############################################


def _grouped_interpolator(
    table: pd.DataFrame,
    *,
    value_name: str,
    statistic: str,
    min_count: int,
) -> Callable[[Mapping[str, Any]], NDArray[np.float64]]:
    """
    Interpolate grouped error statistics at new predictor values (e.g. slope/elevation).

    This function is used to build an empirical model of heteroscedasticity from binned spread statistics,
    and accepts any number of predictors (so is equivalent to a N-dimension interpolator)

    Points inside the edge bins are interpolated linearly.
    Points outside the edge bins are interpolated by nearest neighbours (to avoid unrealistic linear extrapolation
    where few samples exist anyway).

    This function was simplified from an old ``interpolate_nd_binning`` in xDEM.
    """

    # Input checks
    predictor_names = tuple(table.index.names)
    if not predictor_names or any(not isinstance(name, str) or not name for name in predictor_names):
        raise ValueError("Grouped statistics must have named predictor index levels.")
    if table.empty or table.index.has_duplicates:
        raise ValueError("Grouped statistics must be non-empty and have unique group coordinates.")
    levels = [table.index] if table.index.nlevels == 1 else list(table.index.levels)
    coordinates: list[NDArray[np.float64]] = []
    for name, level in zip(predictor_names, levels):
        if isinstance(level, pd.IntervalIndex):
            coordinate = level.mid.to_numpy(dtype=float)
        elif np.issubdtype(level.dtype, np.number):
            coordinate = level.to_numpy(dtype=float)
        else:
            raise TypeError(f"Predictor {name!r} must use continuous numeric groups.")
        if coordinate.size == 0 or np.any(~np.isfinite(coordinate)) or np.any(np.diff(coordinate) <= 0):
            raise ValueError(f"Predictor {name!r} must have finite, increasing group coordinates.")
        coordinates.append(cast(NDArray[np.float64], coordinate))

    # Some combinations of groups may be missing from the table: we add them as NaN in the full grid
    # We also exclude estimates based on fewer than min_count observations
    statistic_column = (value_name, statistic)
    count_column = (value_name, "count")
    if statistic_column not in table.columns or count_column not in table.columns:
        raise ValueError(f"Grouped statistics must contain {statistic_column!r} and {count_column!r}.")
    full_index: pd.Index = levels[0] if len(levels) == 1 else pd.MultiIndex.from_product(levels, names=predictor_names)
    selected = table[statistic_column].where(table[count_column] >= min_count)
    shape = tuple(len(coordinate) for coordinate in coordinates)
    values = selected.reindex(full_index).to_numpy(dtype=float).reshape(shape)

    # Fill gaps between valid groups linearly, then use the nearest group for any remaining gaps
    coordinate_grid = np.meshgrid(*coordinates, indexing="ij")
    points = np.column_stack([coordinate.ravel() for coordinate in coordinate_grid])
    flat_values = values.ravel()
    valid = np.isfinite(flat_values)
    if not np.any(valid):
        raise ValueError(f"No finite {statistic!r} remains after applying min_count={min_count}.")
    if len(coordinates) == 1:
        # One predictor needs only interpolation along its sorted group centers
        filled = np.interp(coordinates[0], points[valid, 0], flat_values[valid])
    else:
        filled = np.full(len(points), np.nan, dtype=float)
        if np.count_nonzero(valid) >= len(coordinates) + 1:
            try:
                filled = np.asarray(griddata(points[valid], flat_values[valid], points, method="linear"), dtype=float)
            except QhullError:
                # Some group layouts cannot support linear interpolation
                pass
        missing = ~np.isfinite(filled)
        if np.any(missing):
            filled[missing] = griddata(points[valid], flat_values[valid], points[missing], method="nearest")

    # Now that the group grid is complete, build an interpolator for new predictor values
    interpolator = RegularGridInterpolator(
        tuple(coordinates),
        np.asarray(filled).reshape(shape),
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    def evaluate(predictors: Mapping[str, Any]) -> NDArray[np.float64]:
        """Calculate error magnitudes at the supplied predictors, in their shared array shape."""

        # We use predictor names to avoid mixing up the order of dimensions
        missing_names = set(predictor_names).difference(predictors)
        if missing_names:
            raise ValueError(f"Missing predictors: {sorted(missing_names)!r}.")

        # Give scalar and array predictors the same location shape
        arrays = [np.asarray(predictors[name], dtype=float) for name in predictor_names]
        broadcast = np.broadcast_arrays(*arrays)
        prediction_points = np.column_stack([array.ravel() for array in broadcast])

        # Leave locations with missing predictor values as NaN
        finite = np.all(np.isfinite(prediction_points), axis=1)
        result = np.full(len(prediction_points), np.nan, dtype=float)

        # Beyond the sampled range, we use the nearest outer group center (to not extrapolate the error magnitude)
        if np.any(finite):
            bounded = prediction_points[finite].copy()
            for index, coordinate in enumerate(coordinates):
                bounded[:, index] = np.clip(bounded[:, index], coordinate[0], coordinate[-1])
            result[finite] = interpolator(bounded)
        return result.reshape(broadcast[0].shape)

    return evaluate


############################################
# 3/ MAGNITUDE VALUES FOR ONE SPATIAL BLOCK
############################################


def _raster_magnitude_values(
    source_values: Any,
    predictors: Mapping[str, Any] | None,
    model: ErrorStructure,
    component: str | None,
    nodata: float | int | None,
) -> NDArray[np.float64]:
    """Predict one raster tile's magnitudes and apply its source mask."""

    # Match flattened predictors to the raster's pixel and band order
    shape = np.shape(source_values)
    flattened: dict[str, Any] = {}
    for name, value in (predictors or {}).items():
        if np.isscalar(value):
            flattened[name] = value
            continue
        array = np.ma.asarray(value, dtype=float).filled(np.nan)
        if len(shape) == 3 and array.size == np.prod(shape[-2:]):
            # A single predictor grid supplies the same values to every band
            array = np.broadcast_to(array.reshape(shape[-2:]), shape)
        flattened[name] = array.reshape(-1)

    # A constant magnitude broadcasts over the tile; a variable one has one value per pixel
    predicted = np.asarray(model.predict_magnitude(flattened, component=component), dtype=float)
    if predicted.ndim > 0:
        predicted = predicted.reshape(shape)
    values = np.broadcast_to(predicted, shape).astype(float, copy=True)

    # Apply missing data separately for each band
    source = np.asanyarray(source_values)
    source_data = np.ma.getdata(source)
    source_mask = np.ma.getmaskarray(source) | ~np.isfinite(source_data)
    if nodata is not None:
        source_mask |= source_data == nodata
    return np.where(source_mask, np.nan, values)


def _point_magnitude_rows(
    dataframe: Any,
    model: ErrorStructure,
    columns: Mapping[str, Any],
    component: str | None,
    data_column: str | None,
) -> Any:
    """Replace a point partition's values with its predicted error magnitudes."""

    if len(dataframe) == 0:
        return dataframe.copy()

    # Read named columns alongside supplied scalars or row-aligned arrays
    predictors = {
        name: dataframe[value].to_numpy() if isinstance(value, str) else value for name, value in columns.items()
    }
    predicted = np.asarray(model.predict_magnitude(predictors, component=component), dtype=float)

    # A fixed magnitude still needs one output value per point
    values = np.broadcast_to(predicted, (len(dataframe),)).astype(float, copy=True)

    # Write magnitudes to the selected data column or to geometry Z
    result = dataframe.copy()
    if data_column is None:
        import geopandas as gpd

        result.geometry = gpd.points_from_xy(
            dataframe.geometry.x.to_numpy(), dataframe.geometry.y.to_numpy(), values, crs=dataframe.crs
        )
    else:
        result[data_column] = values
    return result


########################
# 4/ EAGER RASTER/POINT
########################


def _eager_raster_magnitude(
    like: RasterBase, model: ErrorStructure, predictors: Mapping[str, Any] | None, component: str | None
) -> Any:
    """Predict raster error magnitudes with the entire array in-memory."""

    from geoutils.raster.xr_accessor import RasterAccessor

    # Predict every band before rebuilding the raster on its original grid
    aligned = _spatial_predictors(like, predictors, size=int(like.count * np.prod(like.shape)))
    values = _raster_magnitude_values(like.data, aligned, model, component, like.nodata)
    if like._is_xr:
        return RasterAccessor.from_array(
            values, transform=like.transform, crs=like.crs, nodata=like.nodata, area_or_point=like.area_or_point
        )
    return like.copy(new_array=np.ma.masked_invalid(values))


def _eager_point_magnitude(
    like: PointCloudBase, model: ErrorStructure, predictors: Mapping[str, Any] | None, component: str | None
) -> Any:
    """Predict point cloud error magnitudes with the entire array in-memory."""

    # Share row prediction with chunked point clouds, then restore the source type
    aligned = _spatial_predictors(like, predictors, size=like.point_count)
    result = _point_magnitude_rows(like.ds, model, aligned or {}, component, like.data_column)
    return like._cast_pointcloud_output(result)


###########################
# 5/ DASK FOR RASTER/ POINT
###########################


def _raster_dask_chunks(
    like: RasterBase, predictors: Mapping[str, Any] | None, chunksizes: tuple[int, int] | None
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Choose spatial output chunks from a request or an existing Dask array."""

    from geoutils.multiproc.chunked import normalize_chunks
    from geoutils.raster.base import RasterBase

    if chunksizes is not None:
        return normalize_chunks(chunks=chunksizes, shape=like.shape[-2:])
    if like._is_xr and is_dask_array(like.data):
        return cast(tuple[tuple[int, ...], tuple[int, ...]], cast(Any, like.data).chunks[-2:])
    for predictor in (predictors or {}).values():
        # Inspect existing chunks without loading unopened raster predictors
        if isinstance(predictor, RasterBase) and not predictor.is_loaded:
            continue
        data = predictor.data if hasattr(predictor, "data") and not isinstance(predictor, np.ndarray) else predictor
        if is_dask_array(data):
            return cast(tuple[tuple[int, ...], tuple[int, ...]], data.chunks[-2:])
    raise ValueError("Dask magnitude output requires chunksizes or a Dask raster or predictor.")


def _raster_magnitude_block(
    source_values: Any,
    *predictor_values: Any,
    predictor_names: tuple[str, ...],
    scalar_predictors: Mapping[str, float],
    model: ErrorStructure,
    component: str | None,
    nodata: float | int | None,
) -> NDArray[np.float64]:
    """Predict one aligned Dask raster block from its source and predictor values."""

    predictors = dict(zip(predictor_names, predictor_values))
    predictors.update(scalar_predictors)
    return _raster_magnitude_values(source_values, predictors, model, component, nodata)


def _dask_raster_magnitude(
    like: RasterBase,
    model: ErrorStructure,
    predictors: Mapping[str, Any] | None,
    component: str | None,
    chunksizes: tuple[int, int] | None,
) -> Any:
    """Build lazy raster tiles with local predictor values and the source mask."""

    import_optional("dask")
    import dask.array as da

    from geoutils.raster.xr_accessor import RasterAccessor

    chunks = _raster_dask_chunks(like, predictors, chunksizes)
    tile_size = (max(chunks[0]), max(chunks[1]))
    source = _raster_chunk_source(like, tile_size)

    # Split only rows and columns; each block receives all bands
    source_chunks = tuple((int(length),) for length in source.shape[:-2]) + chunks
    source = source.rechunk(source_chunks) if is_dask_array(source) else da.from_array(source, chunks=source_chunks)
    predictor_data = _raster_chunk_predictors(like, predictors, tile_size)

    # Match predictor blocks to source blocks, including shorter edge chunks and full-band blocks
    predictor_names: list[str] = []
    predictor_blocks: list[Any] = []
    scalar_predictors: dict[str, float] = {}
    for name, value in predictor_data.items():
        if np.isscalar(value):
            scalar_predictors[name] = float(cast(Any, value))
        else:
            array = da.asarray(value)
            array_chunks = tuple((int(length),) for length in array.shape[:-2]) + chunks
            predictor_names.append(name)
            predictor_blocks.append(array.rechunk(array_chunks))

    # Pass scalar predictors once and matching array blocks to each map task
    values = da.map_blocks(
        _raster_magnitude_block,
        source,
        *predictor_blocks,
        predictor_names=tuple(predictor_names),
        scalar_predictors=scalar_predictors,
        model=model,
        component=component,
        nodata=like.nodata,
        dtype=np.float64,
        meta=np.empty((0,) * source.ndim, dtype=np.float64),
    )

    # Attach the original grid without computing any mapped block
    return RasterAccessor.from_array(
        values, transform=like.transform, crs=like.crs, nodata=like.nodata, area_or_point=like.area_or_point
    )


def _dask_point_magnitude(
    like: PointCloudBase,
    model: ErrorStructure,
    predictors: Mapping[str, Any] | None,
    component: str | None,
    chunksize: int | None,
) -> Any:
    """Build lazy point partitions with the same row order and location metadata."""

    import_optional("dask")
    import pandas as pd

    from geoutils.pointcloud.dataframe import (
        _assign_point_values,
        _build_pointcloud_output,
        _get_dataframe_attrs,
        _point_partition_lengths,
    )

    # 1/ Split source points into lazy row partitions
    if chunksize is None and not like._is_dask:
        chunksize = max(1, min(like.point_count, 100_000))

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

    # Attach supplied arrays by row position before running each point partition
    source_attrs = _get_dataframe_attrs(dataframe)
    original_columns = list(dataframe.columns)

    # Row counts also account for a shorter final partition
    lengths = _point_partition_lengths(dataframe)
    columns, arrays = _point_predictor_columns(like, predictors, size=sum(lengths))
    if arrays:
        dataframe = _assign_point_values(dataframe, arrays, partition_lengths=lengths)
    meta = dataframe._meta.copy()
    if like.data_column is not None:
        meta[like.data_column] = pd.Series([], dtype=np.float64)

    # 2/ Predict each partition and rebuild the lazy point cloud
    result = dataframe.map_partitions(
        _point_magnitude_rows,
        model,
        columns,
        component,
        like.data_column,
        meta=meta,
    )
    # Remove columns added only to carry predictor values
    result = result[original_columns]
    return _build_pointcloud_output(
        result, data_column=like.data_column, as_dataframe=True, attrs=source_attrs, preserve_locations=True
    )


############################################
# 6/ MULTIPROCESSING FOR POINT/RASTER
############################################


def _wrapper_raster_magnitude_tile_multiproc(
    tile: RasterBase,
    model: ErrorStructure,
    predictors: Mapping[str, Any],
    component: str | None,
    transform: Any,
) -> Any:
    """Wrapper to predict error magnitude within a raster chunk, for MP backend."""

    from geoutils.multiproc.mparray import _load_raster_tile
    from geoutils.raster import Raster
    from geoutils.raster.base import RasterBase

    # Locate tile on the full raster grid
    column_start, row_start = ~(transform) * (tile.transform.c, tile.transform.f)
    row_start, column_start = int(round(row_start)), int(round(column_start))
    bounds = np.array([row_start, row_start + tile.shape[0], column_start, column_start + tile.shape[1]])
    tile_predictors: dict[str, Any] = {}
    for name, predictor in predictors.items():
        # Load matching predictor tiles
        if np.isscalar(predictor) and not isinstance(predictor, str):
            tile_predictors[name] = predictor
        elif isinstance(predictor, RasterBase):
            tile_predictors[name] = _load_raster_tile(cast(Raster, predictor), bounds).data
        else:
            data = predictor.data if hasattr(predictor, "data") and not isinstance(predictor, np.ndarray) else predictor
            tile_predictors[name] = data[..., bounds[0] : bounds[1], bounds[2] : bounds[3]]

    # We keep original raster NaNs in the result
    values = _raster_magnitude_values(tile.data, tile_predictors, model, component, tile.nodata)
    return Raster.from_array(
        np.ma.masked_invalid(values),
        transform=tile.transform,
        crs=tile.crs,
        nodata=tile.nodata,
        area_or_point=tile.area_or_point,
        tags=dict(tile.tags),
    )


def _multiproc_raster_magnitude(
    like: RasterBase,
    model: ErrorStructure,
    predictors: Mapping[str, Any] | None,
    component: str | None,
    mp_config: MultiprocConfig,
) -> Any:
    """Predict error magnitude to a file-backed raster per chunk with MP."""

    from geoutils.multiproc import map_overlap

    predictor_data = _raster_chunk_predictors(like, predictors, None)
    return map_overlap(
        _wrapper_raster_magnitude_tile_multiproc,
        like,
        mp_config,
        model,
        predictor_data,
        component,
        like.transform,
    )


def _wrapper_point_magnitude_partition_multiproc(
    dataframe: Any,
    model: ErrorStructure,
    columns: Mapping[str, str | float],
    arrays: Mapping[str, Any],
    component: str | None,
    data_column: str | None,
    original_columns: list[str],
    filename: Path,
) -> Any:
    """Wrapper to predict error magnitude within a point chunk, for MP backend."""

    from geoutils.pointcloud.writing import _stage_pointcloud_partition

    source = dataframe.copy()
    for name, values in arrays.items():
        source[name] = values
    result = _point_magnitude_rows(source, model, columns, component, data_column)[original_columns]
    return _stage_pointcloud_partition(result, filename)


def _multiproc_point_magnitude(
    like: PointCloudBase,
    model: ErrorStructure,
    predictors: Mapping[str, Any] | None,
    component: str | None,
    mp_config: MultiprocConfig,
) -> Any:
    """Write predicted point magnitudes with MP into a file-backed point cloud."""

    from geoutils.multiproc.cluster import _map_bounded
    from geoutils.pointcloud.las import _point_partition_size
    from geoutils.pointcloud.loading import _load_pointcloud_rows
    from geoutils.pointcloud.writing import (
        _resolve_pointcloud_output,
        _write_pointcloud_partitions,
    )

    # 1/ Check row chunks and prepare output file
    chunk_size = _point_partition_size(mp_config)
    point_count = like.point_count
    columns, arrays = _point_predictor_columns(like, predictors, size=point_count)
    original_columns = list(like.columns)
    output_path, driver = _resolve_pointcloud_output(
        mp_config.outfile, mp_config.driver, supported_drivers=("GPKG",), operation_name="error magnitude prediction"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 2/ Predict row groups and assemble output file

    # We send one row group and its matching predictor values to each worker
    with TemporaryDirectory(prefix=".geoutils-error-magnitude-", dir=output_path.parent) as directory:

        def arguments() -> Any:
            for start in range(0, max(point_count, 1), chunk_size):
                count = min(chunk_size, point_count - start)
                partition = _load_pointcloud_rows(like, start, count)

                # Slice predictor arrays by the same row positions
                tile_arrays = {name: np.asarray(value)[start : start + count] for name, value in arrays.items()}
                filename = Path(directory) / f"magnitude_{start}.pkl"
                yield partition, model, columns, tile_arrays, component, like.data_column, original_columns, filename

        saved = []
        for _, filename in _map_bounded(mp_config.cluster, _wrapper_point_magnitude_partition_multiproc, arguments()):
            saved.append(filename)

        # Write staged row groups in source order
        output = _write_pointcloud_partitions(
            output_path,
            saved,
            driver=driver,
            data_column=like.data_column,
            geometry_type="Point Z" if like._has_z or like.data_column is None else "Point",
        )

    # 3/ Return depending on input point cloud type
    if like._is_pd:
        from geoutils.pointcloud.transformation import _cast_multiproc_output

        return _cast_multiproc_output(like, output)
    return output


############################################
# 7/ PARENT FUNCTION FOR PREDICT
############################################


def predict_magnitude_map(
    model: ErrorStructure,
    predictors: Mapping[str, Any] | None,
    *,
    component: str | None,
    like: Any,
    chunksizes: int | tuple[int, int] | None,
    mp_config: MultiprocConfig | None,
) -> Any:
    """
    Predict error magnitudes on a raster or point cloud support using its eager, Dask, or MP backend.
    """

    # 1/ Input checks
    if like is None:
        raise ValueError("like is required for a spatial error magnitude map.")
    if component is not None and component not in model.components:
        raise KeyError(component)
    if chunksizes is not None and mp_config is not None:
        raise ValueError("Choose chunksizes for Dask or mp_config for multiprocessing.")

    raster = _get_raster_interface(like)
    point = _get_pointcloud_interface(like) if raster is None else None
    if raster is None and point is None:
        raise TypeError("like must be a raster or point cloud.")
    has_dask_predictor = False
    for predictor in (predictors or {}).values():
        # Unopened spatial files have no Dask data to inspect, and reading data here would load them
        if getattr(predictor, "is_loaded", True) is False:
            continue
        data = getattr(predictor, "data", predictor)
        if is_dask_array(data) or is_dask_dataframe(predictor):
            has_dask_predictor = True
            break

    # 2/ If input is a raster, dispatch to MP, Dask or eager
    if raster is not None:
        if chunksizes is not None and (not isinstance(chunksizes, tuple) or len(chunksizes) != 2):
            raise ValueError("Raster chunk sizes must be a (rows, columns) tuple.")
        dask_source = raster._is_xr and is_dask_array(raster.data)
        if mp_config is not None:
            if dask_source or has_dask_predictor:
                raise ValueError("Multiprocessing cannot be combined with Dask raster inputs.")
            return _multiproc_raster_magnitude(raster, model, predictors, component, mp_config)
        if chunksizes is not None or dask_source or has_dask_predictor:
            return _dask_raster_magnitude(raster, model, predictors, component, chunksizes)
        return _eager_raster_magnitude(raster, model, predictors, component)

    # 3/ If input is point, dispatch the same way
    assert point is not None
    if chunksizes is not None and (isinstance(chunksizes, bool) or not isinstance(chunksizes, int) or chunksizes < 1):
        raise ValueError("Point cloud chunk size must be a positive integer.")
    if mp_config is not None:
        if point._is_dask or has_dask_predictor:
            raise ValueError("Multiprocessing cannot be combined with Dask point inputs.")
        return _multiproc_point_magnitude(point, model, predictors, component, mp_config)
    if chunksizes is not None or point._is_dask or has_dask_predictor:
        return _dask_point_magnitude(point, model, predictors, component, chunksizes)
    return _eager_point_magnitude(point, model, predictors, component)
