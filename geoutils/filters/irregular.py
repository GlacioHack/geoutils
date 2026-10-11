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

"""Filter point values."""

from __future__ import annotations

import os
import pathlib
import tempfile
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from geoutils._dispatch import is_dask_array
from geoutils._misc import import_optional
from geoutils._typing import NDArrayNum
from geoutils.multiproc import MultiprocConfig
from geoutils.operators.base import LocalData
from geoutils.operators.neighbours import PointNeighbours, _DefaultNeighbour, _query_point_neighbours
from geoutils.operators.nodata import NodataHandling
from geoutils.operators.reducer import (
    Count,
    Maximum,
    Mean,
    Median,
    Minimum,
    Range,
    Reducer,
    RootMeanSquare,
    StandardDeviation,
    Sum,
    _can_reduce_arrays,
    _reduce_grouped_values,
)
from geoutils.pointcloud.loading import _concat_point_parts, _load_pointcloud_bounds, _load_pointcloud_rows

if TYPE_CHECKING:
    import geopandas as gpd

    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.pointcloud.pointcloud import PointCloudLike


PointFilterMethod = (
    Literal[
        "mean",
        "median",
        "minimum",
        "min",
        "maximum",
        "max",
        "range",
        "count",
        "sum",
        "stdev",
        "std",
        "rms",
        "rmse",
    ]
    | Reducer
)

_POINT_FILTER_ID_COLUMN_BASE = "_geoutils_filter_id"
_POINT_FILTER_QUERY_BATCH_SIZE = 65_536
_POINT_FILTER_REDUCER_TYPES: dict[str, type[Reducer]] = {
    "mean": Mean,
    "median": Median,
    "minimum": Minimum,
    "min": Minimum,
    "maximum": Maximum,
    "max": Maximum,
    "range": Range,
    "count": Count,
    "sum": Sum,
    "stdev": StandardDeviation,
    "std": StandardDeviation,
    "rms": RootMeanSquare,
    "rmse": RootMeanSquare,
}


######################################
# 1/ FILTER OPTIONS + POINT DATA HELPERS
######################################


def _resolve_point_filter(method: PointFilterMethod) -> Reducer:
    """Convert a filter name to a Reducer, or return the Reducer supplied by the user."""

    if isinstance(method, Reducer):
        return method
    if method not in _POINT_FILTER_REDUCER_TYPES:
        available = ", ".join(repr(name) for name in _POINT_FILTER_REDUCER_TYPES)
        raise ValueError(f"Unknown point filter method {method!r}. Available names are: {available}.")
    return _POINT_FILTER_REDUCER_TYPES[method]()


def _point_filter_nodata_handling(
    reducer: Reducer,
    nodata_propagation: Literal["ignore", "propagate"] | None,
) -> NodataHandling:
    """Use the Reducer's nodata rule unless point filtering requests ignore or propagate."""

    if nodata_propagation is None:
        return reducer.default_nodata_propagation
    if nodata_propagation not in ("ignore", "propagate"):
        raise ValueError("Argument ``nodata_propagation`` must be 'ignore' or 'propagate'.")
    return nodata_propagation


def _point_filter_id_column(dataframe: gpd.GeoDataFrame) -> str:
    """Find an unused column name for temporary row IDs."""

    column_name = _POINT_FILTER_ID_COLUMN_BASE
    while column_name in dataframe.columns:
        column_name = "_" + column_name
    return column_name


def _point_frame_arrays(
    dataframe: gpd.GeoDataFrame,
    data_name: str | None,
    id_column: str | None,
) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum | None]:
    """Extract X/Y coordinates, values, and optional row IDs from a point dataframe."""

    coordinates = np.column_stack((dataframe.geometry.x.to_numpy(), dataframe.geometry.y.to_numpy()))
    values = np.asarray(dataframe.geometry.z if data_name is None else dataframe[data_name])
    identifiers = None if id_column is None else np.asarray(dataframe[id_column])
    return coordinates, values, identifiers


def _assign_filtered_point_values(
    dataframe: gpd.GeoDataFrame,
    values: NDArrayNum,
    data_name: str | None,
    id_column: str | None,
) -> gpd.GeoDataFrame:
    """Copy the point rows and write filtered values to their data attribute or elevation."""

    import geopandas as gpd

    output = dataframe.copy()
    if id_column is not None:
        output.drop(columns=id_column, inplace=True)

    # Write to the named data column, or rebuild the 3D geometry when the values are stored as elevation
    if data_name is not None:
        output[data_name] = values
    else:
        output.geometry = gpd.points_from_xy(
            output.geometry.x.to_numpy(),
            output.geometry.y.to_numpy(),
            z=values,
            crs=output.crs,
        )
    return output


###################################################
# 2/ IN-MEMORY FILTERING: NEIGHBOURS + REDUCED VALUES
###################################################


def _reduce_point_pairs_generic(
    reducer: Reducer,
    source_coordinates: NDArrayNum,
    source_values: NDArrayNum,
    source_ids: NDArrayNum,
    target_coordinates: NDArrayNum,
    target_indexes: NDArrayNum,
    source_indexes: NDArrayNum,
    distances: NDArrayNum,
    nodata_handling: NodataHandling,
    min_points: int,
) -> NDArrayNum:
    """Pass each target neighbours to a custom Reducer as a LocalData object."""

    # Group neighbours by target, nearest first (source IDs resolve equal distances consistently)
    order = np.lexsort((source_ids[source_indexes], distances, target_indexes))
    ordered_targets = target_indexes[order]
    ordered_sources = source_indexes[order]
    ordered_distances = distances[order]

    # Find the slice of the sorted arrays that belongs to each target, including empty slices
    starts = np.searchsorted(ordered_targets, np.arange(len(target_coordinates)), side="left")
    stops = np.searchsorted(ordered_targets, np.arange(len(target_coordinates)), side="right")
    output = np.full(len(target_coordinates), np.nan, dtype=np.float64)

    # Collect values + coordinates for each target, excluding targets with too few valid neighbours
    # We pass the list to reduce_batch() so a custom Reducer can calculate several results at once
    local_inputs: list[LocalData] = []
    output_positions: list[int] = []
    for target_index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        neighbor_indexes = ordered_sources[start:stop]
        valid = np.isfinite(source_values[neighbor_indexes])
        if np.count_nonzero(valid) < min_points:
            continue
        local_inputs.append(
            LocalData(
                values=source_values[neighbor_indexes],
                valid=valid,
                source_ids=source_ids[neighbor_indexes],
                coordinates=source_coordinates[neighbor_indexes],
                target=target_coordinates[target_index],
                distances=ordered_distances[start:stop],
            )
        )
        output_positions.append(target_index)
    if local_inputs:
        output[output_positions] = reducer.reduce_batch(local_inputs, nodata_propagation=nodata_handling)
    return output


def _reduce_point_pairs(
    reducer: Reducer,
    source_coordinates: NDArrayNum,
    source_values: NDArrayNum,
    source_ids: NDArrayNum,
    target_coordinates: NDArrayNum,
    target_indexes: NDArrayNum,
    source_indexes: NDArrayNum,
    distances: NDArrayNum,
    nodata_handling: NodataHandling,
    min_points: int,
) -> NDArrayNum:
    """Use NumPy for built-in statistics, or call reduce_batch() for a custom Reducer."""

    if nodata_handling in ("ignore", "propagate") and _can_reduce_arrays(reducer):
        return _reduce_grouped_values(
            reducer,
            source_values[source_indexes],
            target_indexes,
            len(target_coordinates),
            nodata_handling,
            min_points,
            distances=distances,
        )
    return _reduce_point_pairs_generic(
        reducer,
        source_coordinates,
        source_values,
        source_ids,
        target_coordinates,
        target_indexes,
        source_indexes,
        distances,
        nodata_handling,
        min_points,
    )


def _filter_point_values(
    source_coordinates: NDArrayNum,
    source_values: NDArrayNum,
    target_coordinates: NDArrayNum,
    target_values: NDArrayNum,
    reducer: Reducer,
    neighborhood: PointNeighbours,
    *,
    source_ids: NDArrayNum | None,
    target_ids: NDArrayNum | None,
    include_self: bool,
    nodata_handling: NodataHandling,
    min_points: int,
    n_threads: int,
    batch_size: int,
) -> NDArrayNum:
    """Find neighbours and calculate filtered values in batches to limit temporary memory use."""

    from scipy.spatial import cKDTree

    # We exclude invalid X/Y coordinates before building the source search tree
    output_shape = (len(target_coordinates), *target_values.shape[1:])
    output = np.full(output_shape, np.nan, dtype=np.float64)
    finite_source_coordinates = np.isfinite(source_coordinates).all(axis=1)
    finite_target_coordinates = np.isfinite(target_coordinates).all(axis=1)
    source_coordinates = np.ascontiguousarray(source_coordinates[finite_source_coordinates], dtype=np.float64)
    source_values = np.asarray(source_values[finite_source_coordinates], dtype=np.float64)
    source_ids = None if source_ids is None else np.asarray(source_ids[finite_source_coordinates])
    if len(source_coordinates) == 0 or not np.any(finite_target_coordinates):
        if type(reducer) is Count and max(min_points, reducer.minimum_inputs) == 0:
            output[finite_target_coordinates] = 0
        return output

    # Build the source search tree once, then query a limited number of targets at a time
    source_tree = cKDTree(source_coordinates)

    # Custom Reducers need a unique ID for each source point, even without Dask partitions
    stable_source_ids = np.arange(len(source_coordinates)) if source_ids is None else source_ids
    target_positions = np.flatnonzero(finite_target_coordinates)
    for start in range(0, len(target_positions), batch_size):
        positions = target_positions[start : start + batch_size]
        batch_coordinates = np.ascontiguousarray(target_coordinates[positions], dtype=np.float64)
        batch_values = np.asarray(target_values[positions], dtype=np.float64)
        batch_ids = None if target_ids is None else np.asarray(target_ids[positions])
        target_indexes, source_indexes, distances = _query_point_neighbours(
            source_tree,
            source_coordinates,
            source_values if source_values.ndim == 1 else source_values[:, 0],
            source_ids,
            batch_coordinates,
            batch_values if batch_values.ndim == 1 else batch_values[:, 0],
            batch_ids,
            neighborhood,
            include_self,
            n_threads,
        )
        # Reuse the neighbor pairs for each value variable, handling its missing values independently
        columns = [source_values] if source_values.ndim == 1 else source_values.T
        reduced = [
            _reduce_point_pairs(
                reducer,
                source_coordinates,
                values,
                stable_source_ids,
                batch_coordinates,
                target_indexes,
                source_indexes,
                distances,
                nodata_handling,
                min_points,
            )
            for values in columns
        ]
        output[positions] = reduced[0] if source_values.ndim == 1 else np.stack(reduced, axis=1)
    return output


def _filter_point_dataframe(
    targets: gpd.GeoDataFrame,
    sources: gpd.GeoDataFrame,
    data_name: str | None,
    reducer: Reducer,
    neighborhood: PointNeighbours,
    *,
    id_column: str | None,
    include_self: bool,
    nodata_handling: NodataHandling,
    min_points: int,
    n_threads: int,
    batch_size: int,
) -> gpd.GeoDataFrame:
    """Filter one dataframe using nearby source points, then write the values to a copy of its rows."""

    # Read matching arrays of X/Y coordinates, values and IDs for the source/target points
    source_coordinates, source_values, source_ids = _point_frame_arrays(sources, data_name, id_column)
    target_coordinates, target_values, target_ids = _point_frame_arrays(targets, data_name, id_column)
    values = _filter_point_values(
        source_coordinates,
        source_values,
        target_coordinates,
        target_values,
        reducer,
        neighborhood,
        source_ids=source_ids,
        target_ids=target_ids,
        include_self=include_self,
        nodata_handling=nodata_handling,
        min_points=min_points,
        n_threads=n_threads,
        batch_size=batch_size,
    )
    return _assign_filtered_point_values(targets, values, data_name, id_column)


def _array_filter_halo(source: tuple[Any, ...], target: tuple[Any, ...], radius: float) -> tuple[Any, ...]:
    """Select source arrays inside a target chunk's bounds expanded by the filter radius."""
    x, y, values, identifiers = source
    target_x, target_y = target[:2]
    finite = np.isfinite(target_x) & np.isfinite(target_y)
    if not finite.any():
        return tuple(value[:0] for value in source)
    selected = (x >= target_x[finite].min() - radius) & (x <= target_x[finite].max() + radius)
    selected &= (y >= target_y[finite].min() - radius) & (y <= target_y[finite].max() + radius)
    return x[selected], y[selected], values[selected], identifiers[selected]


def _filter_array_partition(
    target: tuple[Any, ...],
    sources: list[tuple[Any, ...]],
    reducer: Reducer,
    neighborhood: PointNeighbours,
    **options: Any,
) -> NDArrayNum:
    """Calculate a chunk point values from source arrays and row IDs."""

    # Source chunks arrive in row order so ties and custom reducers agree with eager calculations
    source_arrays = [np.concatenate([part[index] for part in sources]) for index in range(4)]
    source_coordinates = np.column_stack(source_arrays[:2])
    target_coordinates = np.column_stack(target[:2])
    return _filter_point_values(
        source_coordinates,
        source_arrays[2],
        target_coordinates,
        target[2],
        reducer,
        neighborhood,
        source_ids=source_arrays[3],
        target_ids=target[3],
        **options,
    )


def _filter_array_pointcloud(
    point_arrays: tuple[Any, Any, Any],
    reducer: Reducer,
    neighborhood: PointNeighbours,
    mp_config: Any,
    *,
    is_dask: bool,
    **options: Any,
) -> Any:
    """Filter array point values, in-memory or with Dask/MP."""

    if mp_config is not None:
        raise ValueError("Array point clouds use Dask chunks rather than mp_config for filtering.")
    x, y, values = point_arrays
    if not is_dask:
        rows = (x, y, values, np.arange(len(x), dtype=np.uint64))
        return _filter_array_partition(rows, [rows], reducer, neighborhood, **options)

    # Align coordinates and attributes with value chunks without loading their arrays
    dask = import_optional("dask")
    import dask.array as da

    values = da.asarray(values)
    if values.ndim == 2:
        values = values.rechunk({1: -1})
    lengths = values.chunks[0]
    arrays = [da.asarray(axis).rechunk({0: lengths}) for axis in (x, y, values)]
    blocks = [axis.to_delayed().ravel() for axis in arrays]
    starts = np.cumsum((0, *lengths))
    parts = [
        (
            blocks[0][index],
            blocks[1][index],
            blocks[2][index],
            np.arange(starts[index], starts[index + 1], dtype=np.uint64),
        )
        for index in range(len(lengths))
    ]

    # Filter each source chunk to the required surrounding area before joining a target's neighborhood
    outputs = []
    assert neighborhood.radius is not None
    for target, length in zip(parts, lengths):
        halos = [dask.delayed(_array_filter_halo)(part, target, neighborhood.radius) for part in parts]
        output = dask.delayed(_filter_array_partition)(target, halos, reducer, neighborhood, **options)
        outputs.append(da.from_delayed(output, shape=(length, *values.shape[1:]), dtype=float))
    return da.concatenate(outputs)


###################
# 3/ DASK FILTERING
###################


def _add_point_filter_ids(
    dataframe: gpd.GeoDataFrame,
    id_column: str,
    partition_info: dict[str, Any] | None = None,
) -> gpd.GeoDataFrame:
    """Add row IDs based on Dask partition number and position within the partition."""

    partition_number = 0 if partition_info is None else partition_info["number"]
    if partition_number >= 2**32 or len(dataframe) >= 2**32:
        raise ValueError("Point filter partitions exceed the supported internal row identity range.")

    # Store the partition number in the upper 32 bits and the row position in the lower 32 bits
    # This gives each point a unique ID without reading other partitions
    output = dataframe.copy()
    local_rows = np.arange(len(output), dtype=np.uint64)
    output[id_column] = (np.uint64(partition_number) << np.uint64(32)) | local_rows
    return output


def _point_filter_source_subset(
    source: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    radius: float,
) -> gpd.GeoDataFrame:
    """Select source points within a target partition's bounds + the search radius."""

    if len(source) == 0 or len(targets) == 0:
        return source.iloc[:0]

    # Ignore invalid X/Y coordinates when finding the target area
    bounds = targets.total_bounds
    if not np.isfinite(bounds).all():
        finite = np.isfinite(targets.geometry.x) & np.isfinite(targets.geometry.y)
        if not np.any(finite):
            return source.iloc[:0]
        bounds = targets.loc[finite].total_bounds

    # Expand the bounds by the search radius so points near partition edges have all their neighbours
    keep = (source.geometry.x >= bounds[0] - radius) & (source.geometry.x <= bounds[2] + radius)
    keep &= (source.geometry.y >= bounds[1] - radius) & (source.geometry.y <= bounds[3] + radius)
    return source.loc[keep]


def _filter_dask_point_partition(
    targets: gpd.GeoDataFrame,
    source_parts: list[gpd.GeoDataFrame],
    data_name: str | None,
    id_column: str,
    reducer: Reducer,
    neighborhood: PointNeighbours,
    include_self: bool,
    nodata_handling: NodataHandling,
    min_points: int,
    batch_size: int,
) -> gpd.GeoDataFrame:
    """Combine nearby source points from several partitions and filter one target partition."""

    # Combine only the source rows inside each partition's expanded target bounds
    nonempty_parts = [part for part in source_parts if len(part) > 0]
    if nonempty_parts:
        sources = _concat_point_parts(nonempty_parts, crs=targets.crs)
    else:
        sources = targets.iloc[:0]

    # After the rectangular selection, apply the exact circular distance limit
    return _filter_point_dataframe(
        targets,
        sources,
        data_name,
        reducer,
        neighborhood,
        id_column=id_column,
        include_self=include_self,
        nodata_handling=nodata_handling,
        min_points=min_points,
        n_threads=1,
        batch_size=batch_size,
    )


def _dask_filter_pointcloud(
    source_pointcloud: PointCloudBase,
    reducer: Reducer,
    neighborhood: PointNeighbours,
    *,
    include_self: bool,
    nodata_handling: NodataHandling,
    min_points: int,
    batch_size: int,
) -> Any:
    """Create a Dask task for each point partition, including nearby source points from other partitions."""

    import_optional("dask")
    import dask
    import dask.dataframe as dd

    dask_geopandas = import_optional("dask_geopandas")
    assert neighborhood.radius is not None
    dataframe = source_pointcloud._dataset
    data_name = source_pointcloud.data_name

    # Assign IDs without loading the points, so include_self=False can distinguish otherwise identical rows
    id_column = _point_filter_id_column(dataframe._meta)
    meta_with_ids = dataframe._meta.copy()
    meta_with_ids[id_column] = np.asarray([], dtype=np.uint64)
    identified = dataframe.map_partitions(_add_point_filter_ids, id_column, meta=meta_with_ids)
    partitions = list(identified.to_delayed())
    spatial_partitions = dataframe.spatial_partitions
    output_parts: list[Any] = []
    for target_index, targets in enumerate(partitions):
        # Spatial partition bounds let us skip source partitions that are too far from this target partition
        source_indexes: list[int]
        if spatial_partitions is None:
            source_indexes = list(range(len(partitions)))
        else:
            target_support = spatial_partitions.iloc[target_index].buffer(neighborhood.radius)
            source_indexes = [
                int(index) for index in np.flatnonzero(spatial_partitions.intersects(target_support).to_numpy())
            ]

        # Each source partition discards distant rows before the target task combines the results
        source_subsets = [
            dask.delayed(_point_filter_source_subset)(partitions[index], targets, neighborhood.radius)
            for index in source_indexes
        ]
        output_parts.append(
            dask.delayed(_filter_dask_point_partition)(
                targets,
                source_subsets,
                data_name,
                id_column,
                reducer,
                neighborhood,
                include_self,
                nodata_handling,
                min_points,
                batch_size,
            )
        )

    # Record the output column and spatial bounds without reading any point rows
    output_meta = dataframe._meta.copy()
    if data_name is not None:
        output_meta[data_name] = np.asarray([], dtype=np.float64)
    with dask.config.set({"dataframe.convert-string": False}):
        output = dd.from_delayed(output_parts, meta=output_meta)
    filtered = dask_geopandas.from_dask_dataframe(output, geometry=dataframe.geometry.name)
    filtered.spatial_partitions = spatial_partitions
    return filtered


##############################
# 4/ MULTIPROCESSING FILTERING
##############################


def _stage_filtered_point_partition(
    source_pointcloud: PointCloudBase,
    start: int,
    count: int,
    filename: pathlib.Path,
    reducer: Reducer,
    neighborhood: PointNeighbours,
    include_self: bool,
    nodata_handling: NodataHandling,
    min_points: int,
    batch_size: int,
    needs_las_bounds: bool,
) -> tuple[pathlib.Path, NDArrayNum | None]:
    """Read and filter one group of point rows, then save it to a temporary file."""

    from rasterio.coords import BoundingBox

    from geoutils.pointcloud.writing import _stage_pointcloud_partition

    # Read one group of target rows and the source points inside their expanded bounds
    assert neighborhood.radius is not None
    targets = _load_pointcloud_rows(source_pointcloud, start=start, count=count)
    if len(targets) == 0:
        sources = targets
    else:
        left, bottom, right, top = targets.total_bounds
        radius = neighborhood.radius
        support_bounds = BoundingBox(left - radius, bottom - radius, right + radius, top + radius)
        sources = _load_pointcloud_bounds(source_pointcloud, support_bounds, source_pointcloud.data_name)

    # Each worker uses one thread and saves its result instead of sending a dataframe back to the parent process
    filtered = _filter_point_dataframe(
        targets,
        sources,
        source_pointcloud.data_name,
        reducer,
        neighborhood,
        id_column=None,
        include_self=include_self,
        nodata_handling=nodata_handling,
        min_points=min_points,
        n_threads=1,
        batch_size=batch_size,
    )
    bounds = None
    if needs_las_bounds:
        from geoutils.pointcloud.las import _las_coordinate_bounds

        # LAS/LAZ stores coordinates as scaled integers; all partitions must use the same scale and offset
        # The combined bounds let the writer choose these for the complete output
        bounds = _las_coordinate_bounds(filtered, source_pointcloud.data_name)
    return _stage_pointcloud_partition(filtered, filename), bounds


def _multiproc_filter_pointcloud(
    source_pointcloud: PointCloudBase,
    reducer: Reducer,
    neighborhood: PointNeighbours,
    mp_config: MultiprocConfig,
    *,
    include_self: bool,
    nodata_handling: NodataHandling,
    min_points: int,
    batch_size: int,
) -> PointCloudLike:
    """Filter groups of point rows in worker processes, then write them to one output file."""

    from geoutils.multiproc.cluster import _map_bounded
    from geoutils.pointcloud.las import _point_partition_size
    from geoutils.pointcloud.writing import _resolve_pointcloud_output, _write_pointcloud_partitions

    if source_pointcloud._is_dask:
        raise ValueError("Cannot use Multiprocessing and Dask simultaneously. Remove ``mp_config`` or use eager data.")
    assert neighborhood.radius is not None

    # Choose the row partition size and file format before starting the worker processes
    partition_size = _point_partition_size(mp_config)
    output_filename, driver = _resolve_pointcloud_output(
        mp_config.outfile,
        mp_config.driver,
        supported_drivers=("GPKG", "LAS", "LAZ", "PARQUET"),
        operation_name="point cloud filtering",
    )
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    # Workers return filenames, which avoids collecting completed point dataframes in the parent process
    with tempfile.TemporaryDirectory(prefix=".geoutils-point-filter-", dir=output_filename.parent) as directory:
        temporary_directory = pathlib.Path(directory)
        point_count = source_pointcloud.point_count
        partition_arguments = []
        for start in range(0, max(point_count, 1), partition_size):
            count = min(partition_size, point_count - start)
            partition_arguments.append(
                (
                    source_pointcloud,
                    start,
                    count,
                    temporary_directory / f"partition_{start}.pkl",
                    reducer,
                    neighborhood,
                    include_self,
                    nodata_handling,
                    min_points,
                    batch_size,
                    driver in ("LAS", "LAZ"),
                )
            )
        partition_results = [
            result
            for _, result in _map_bounded(
                mp_config.cluster,
                _stage_filtered_point_partition,
                partition_arguments,
            )
        ]

        # Append the saved partitions in their original order, including the points' other attributes
        pointcloud = _write_pointcloud_partitions(
            output_filename,
            [filename for filename, _ in partition_results],
            driver=driver,
            data_name=source_pointcloud.data_name if driver in ("GPKG", "PARQUET") else None,
            geometry_type="Point Z" if source_pointcloud._has_z else "Point",
            las_elevation_column=source_pointcloud.data_name,
            las_bounds=[bounds for _, bounds in partition_results] if driver in ("LAS", "LAZ") else None,
        )

    if source_pointcloud._is_pd:
        pointcloud.load(columns="all")
        return source_pointcloud._cast_pointcloud_output(pointcloud._dataset)
    return pointcloud


###########################
# 5/ PARENT FILTER FUNCTION
###########################


def _filter_pointcloud(
    source_pointcloud: PointCloudBase | tuple[Any, Any, Any],
    method: PointFilterMethod = "median",
    radius: float | None | _DefaultNeighbour = _DefaultNeighbour.VALUE,
    *,
    k: int | None | _DefaultNeighbour = _DefaultNeighbour.VALUE,
    include_self: bool = True,
    min_points: int = 0,
    nodata_propagation: Literal["ignore", "propagate"] | None = None,
    n_threads: int = 0,
    batch_size: int = _POINT_FILTER_QUERY_BATCH_SIZE,
    mp_config: MultiprocConfig | None = None,
) -> Any:
    """
    Filter point values using nearby points, in memory or with Dask/multiprocessing.

    :param source_pointcloud: Point cloud or dataframe accessor to filter (its active data attribute or elevations).
        An X/Y/value array tuple returns filtered values, with one row per point and an optional column per variable.
    :param method: Built-in reducer name or a Reducer instance.
    :param radius: Maximum X/Y neighbor distance in CRS units. Required for Dask and multiprocessing execution.
    :param k: Optional maximum number of nearest neighbors inside radius. Omit to use every point inside radius.
    :param include_self: Whether to include a point's own value when calculating its filtered value.
    :param min_points: Minimum number of finite neighbors, combined with the Reducer's own minimum requirement.
    :param nodata_propagation: Ignore missing neighbour values (``"ignore"``), return NaN when missing values
        contribute to the calculation (``"propagate"``), or use the Reducer's default (None).
    :param n_threads: Threads used by SciPy to find neighbors in memory. Zero uses the CPU count minus one.
    :param batch_size: Maximum target points queried together. Smaller batches use less temporary memory when
        neighborhoods are dense.
    :param mp_config: Worker count, row partition size and output file for multiprocessing execution.
    :returns: Point cloud with unchanged rows, X/Y coordinates and attributes and filtered active values.
        For an array tuple, returns filtered values with the same shape as the input values.
    """

    # 1/ Resolve the filter options and check inputs
    reducer = _resolve_point_filter(method)
    defaults = reducer.default_neighborhood
    if defaults is None:
        defaults = PointNeighbours(radius=1.0)
    if not isinstance(defaults, PointNeighbours):
        raise TypeError("Point filtering requires PointNeighbours.")
    neighborhood = PointNeighbours(
        k=defaults.k if isinstance(k, _DefaultNeighbour) else k,
        radius=defaults.radius if isinstance(radius, _DefaultNeighbour) else radius,
    )
    nodata_handling = _point_filter_nodata_handling(reducer, nodata_propagation)

    # Check options before choosing in-memory, Dask or MP filtering
    if not isinstance(include_self, (bool, np.bool_)):
        raise TypeError("Argument ``include_self`` must be a boolean.")
    if isinstance(min_points, bool) or not isinstance(min_points, (int, np.integer)) or min_points < 0:
        raise ValueError("Argument ``min_points`` must be a non-negative integer.")
    if isinstance(n_threads, bool) or not isinstance(n_threads, (int, np.integer)) or n_threads < 0:
        raise ValueError("Argument ``n_threads`` must be a non-negative integer.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("Argument ``batch_size`` must be a positive integer.")

    # To use Dask and MP, a radius needs to be set!
    if isinstance(source_pointcloud, tuple):
        is_dask = any(is_dask_array(array) for array in source_pointcloud)
    else:
        is_dask = source_pointcloud._is_dask
    if (is_dask or mp_config is not None) and neighborhood.radius is None:
        raise ValueError("Dask and multiprocessing point filtering require a finite ``radius``.")
    if is_dask and mp_config is not None:
        raise ValueError("Cannot use Multiprocessing and Dask simultaneously. Remove ``mp_config`` or use eager data.")

    if isinstance(source_pointcloud, tuple) or source_pointcloud._is_xr:
        arrays = source_pointcloud if isinstance(source_pointcloud, tuple) else source_pointcloud.to_xyz()
        resolved_threads = max(1, (os.cpu_count() or 2) - 1) if n_threads == 0 else int(n_threads)
        filtered = _filter_array_pointcloud(
            arrays,
            reducer,
            neighborhood,
            mp_config,
            is_dask=is_dask,
            include_self=bool(include_self),
            nodata_handling=nodata_handling,
            min_points=int(min_points),
            n_threads=1 if is_dask else resolved_threads,
            batch_size=int(batch_size),
        )
        if isinstance(source_pointcloud, tuple):
            return filtered
        return source_pointcloud.copy(new_array=filtered)

    # 2/ Dispatch to Dask/MP/in-memory

    # For Dask
    if source_pointcloud._is_dask:
        filtered = _dask_filter_pointcloud(
            source_pointcloud,
            reducer,
            neighborhood,
            include_self=bool(include_self),
            nodata_handling=nodata_handling,
            min_points=int(min_points),
            batch_size=int(batch_size),
        )
        output = source_pointcloud._cast_pointcloud_output(filtered)

        # As filtering keeps the same point coords as input, the saved point count and bounds are still valid
        from geoutils.pointcloud.dataframe import _get_dataframe_attrs, _set_dataframe_attrs

        source_attrs = _get_dataframe_attrs(source_pointcloud._dataset)
        output_attrs = _get_dataframe_attrs(output).copy()
        output_attrs.update(
            point_count=source_attrs.get("point_count"),
            bounds=source_attrs.get("bounds"),
            geometry_type=source_attrs.get("geometry_type", "Point"),
        )
        _set_dataframe_attrs(output, output_attrs)
        return output

    # For MP
    if mp_config is not None:
        return _multiproc_filter_pointcloud(
            source_pointcloud,
            reducer,
            neighborhood,
            mp_config,
            include_self=bool(include_self),
            nodata_handling=nodata_handling,
            min_points=int(min_points),
            batch_size=int(batch_size),
        )

    # Otherwise, in memory
    # IDs are for include_self=False to distinguish points at the same coordinates
    dataframe = source_pointcloud._dataset
    id_column = None
    identified = dataframe
    if not include_self:
        id_column = _point_filter_id_column(dataframe)
        identified = dataframe.copy()
        identified[id_column] = np.arange(len(identified), dtype=np.uint64)

    # SciPy can use several threads for the neighbor search
    resolved_threads = max(1, (os.cpu_count() or 2) - 1) if n_threads == 0 else int(n_threads)
    filtered = _filter_point_dataframe(
        identified,
        identified,
        source_pointcloud.data_name,
        reducer,
        neighborhood,
        id_column=id_column,
        include_self=bool(include_self),
        nodata_handling=nodata_handling,
        min_points=int(min_points),
        n_threads=resolved_threads,
        batch_size=int(batch_size),
    )
    return source_pointcloud._cast_pointcloud_output(filtered)
