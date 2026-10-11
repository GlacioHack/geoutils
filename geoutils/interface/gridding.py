# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grid point clouds through eager, Dask or multiprocessing execution (uses interpolator/reducers in operators)."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

import affine
import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.coords import BoundingBox

from geoutils._config import config
from geoutils._dispatch import (
    _check_match_grid,
    get_geo_attr,
    has_geo_attr,
    is_dask_dataframe,
)
from geoutils._misc import import_optional
from geoutils._typing import NDArrayNum
from geoutils.multiproc.chunked import ChunkedGeoGrid, GeoGrid, normalize_chunks
from geoutils.multiproc.mparray import (
    MultiprocConfig,
    _split_chunk_size,
    _write_multiproc_result,
)
from geoutils.operators.execution import _get_builtin_gridding_method, _grid_from_points
from geoutils.operators.interpolator import Interpolator, IrregularInterpolationMethod, _resolve_irregular_interpolator
from geoutils.operators.neighbours import (
    _prepare_point_gridding_data,
    _resolve_point_neighbours_for_interpolator,
    _resolve_point_neighbours_for_reducer,
)
from geoutils.operators.nodata import (
    NodataChoice,
    _mask_grid_from_invalid_points,
    _mask_grid_near_invalid_points,
    _resolve_nodata_handling,
)
from geoutils.operators.reducer import _IRREGULAR_REDUCER_TYPES, IrregularReductionMethod, Reducer
from geoutils.pointcloud.loading import (
    _concat_point_parts,
    _filter_dask_points_by_bounds,
    _load_pointcloud_bounds,
    _source_dataframe,
)
from geoutils.raster.referencing import _coords

if TYPE_CHECKING:
    from geoutils.raster.base import RasterLike

################################
# 1/ SHARED GRIDDING DEFINITIONS
################################

# Methods, coordinate preparation and support/nodata handling shared by all calculation engines
##############################################################################################


GriddingEngine = Literal["scipy", "numba"]
GriddingMethod = IrregularInterpolationMethod | IrregularReductionMethod | Interpolator | Reducer
GridPointCloudCallable = Callable[..., tuple[NDArrayNum, affine.Affine]]


def _resolve_gridding_operator(
    method: GriddingMethod,
    *,
    distance_power: float,
) -> Interpolator | Reducer:
    """Convert a gridding method name to an Interpolator/Reducer, or use the object directly."""

    # Resolve names and geometry-specific options in the operator modules
    if isinstance(method, Reducer):
        return method
    if isinstance(method, Interpolator) or method in get_args(IrregularInterpolationMethod):
        return _resolve_irregular_interpolator(
            cast(IrregularInterpolationMethod | Interpolator, method), distance_power=distance_power
        )
    if method in _IRREGULAR_REDUCER_TYPES:
        return _IRREGULAR_REDUCER_TYPES[cast(IrregularReductionMethod, method)]()
    raise ValueError(f"Unknown gridding resampling method: {method!r}.")


def _gridding_nodata_spread(method: GriddingMethod, nodata_handling: NodataChoice | None) -> int | None:
    """Return the selected distance around missing point values, when requested."""

    # Convert a method name first, since an Interpolator records the order needed by half-order options
    operator = (
        method
        if isinstance(method, (Interpolator, Reducer))
        else _resolve_gridding_operator(method, distance_power=2.0)
    )
    if isinstance(operator, Reducer):
        if nodata_handling is None:
            nodata_handling = "ignore"
        elif isinstance(nodata_handling, str) and nodata_handling.lower() == "nearest":
            raise ValueError("Nearest-source nodata handling requires an Interpolator.")
    order = operator.interpolation_order if isinstance(operator, Interpolator) else None
    return _resolve_nodata_handling(nodata_handling=nodata_handling, order=order)[1]


def _grid_resolution(
    grid_coords: tuple[NDArrayNum, NDArrayNum],
    grid_res: tuple[float, float] | None,
) -> tuple[float, float]:
    """Return positive X/Y grid resolutions, including single-row or single-column grids."""

    if len(grid_coords[0]) > 1:
        res_x = float(np.abs(grid_coords[0][1] - grid_coords[0][0]))
    elif grid_res is not None:
        res_x = abs(grid_res[0])
    else:
        raise ValueError("At least two X coordinates or an explicit grid resolution are required.")

    if len(grid_coords[1]) > 1:
        res_y = float(np.abs(grid_coords[1][1] - grid_coords[1][0]))
    elif grid_res is not None:
        res_y = abs(grid_res[1])
    else:
        raise ValueError("At least two Y coordinates or an explicit grid resolution are required.")
    return res_x, res_y


################################
# 2/ EAGER CALCULATION ENGINES
################################

# Common eager dispatcher for interpolation and circular-neighborhood methods
############################################################################


def _grid_pointcloud(
    pc: gpd.GeoDataFrame,
    grid_coords: tuple[NDArrayNum, NDArrayNum],
    data_name: str | None = None,
    resampling: GriddingMethod = "linear",
    dist_nodata_pixel: float = 1.0,
    nodata_handling: NodataChoice | None = None,
    grid_res: tuple[float, float] | None = None,
    distance_power: float = 2.0,
    min_points: int = 1,
    n_threads: int = 1,
    engine: GriddingEngine = "scipy",
) -> tuple[NDArrayNum, affine.Affine]:
    """
    Grid irregular points to a regular raster using interpolation or circular neighborhoods.

    :param pc: Point cloud.
    :param grid_coords: Regular raster grid coordinates in X and Y (i.e. equally spaced, independently for each axis).
    :param data_name: Name of data attribute for point cloud (if 2D point geometries are used).
    :param resampling: Interpolator, Reducer, or an existing interpolation/statistic name. ``average``, ``min`` and
        ``max`` are aliases for ``mean``, ``minimum`` and ``maximum``.
    :param dist_nodata_pixel: Maximum point distance or circular neighborhood radius, expressed in output pixels.
        A Reducer with PointNeighbours uses its configured point count or radius instead.
    :param nodata_handling: ``"nearest"`` calculates from finite points, then masks an Interpolator's result when the
        nearest source point is missing. ``"ignore"`` uses available finite values; ``"propagate"`` masks cells
        using a missing source. Reducers default to ``"ignore"`` and cannot use ``"nearest"``. A non-negative
        integer or half-order choice masks cells within that distance of missing source values.
    :param grid_res: Grid resolution, used for chunks with a single row or column.
    :param distance_power: Distance exponent used for inverse-distance weighting (defaults to 2).
    :param min_points: Minimum number of finite points required inside a circular neighborhood (defaults to 1).
    :param engine: Calculation engine, either ``scipy`` (default) or ``numba``. Numba supports nearest and circular
        methods except ``average_distance_pts``.
    :param n_threads: Number of SciPy threads used for nearest-neighbor queries (defaults to 1).
    """

    if np.isnan(dist_nodata_pixel) or dist_nodata_pixel < 0:
        raise ValueError("Argument 'dist_nodata_pixel' must be non-negative.")
    if n_threads < 1:
        raise ValueError("Argument 'n_threads' must be a positive integer.")
    if engine not in ("scipy", "numba"):
        raise ValueError("Argument 'engine' must be either 'scipy' or 'numba'.")
    if resampling == "idw" and (not np.isfinite(distance_power) or distance_power <= 0):
        raise ValueError("IDW distance_power must be finite and strictly positive.")
    operator = _resolve_gridding_operator(resampling, distance_power=distance_power)
    optimized_kernel = _get_builtin_gridding_method(operator)
    engine_method = _get_builtin_gridding_method(operator, default_neighborhood_only=False)
    reducer_neighborhood = _resolve_point_neighbours_for_reducer(operator) if isinstance(operator, Reducer) else None
    if engine == "numba" and engine_method is None:
        raise ValueError("The Numba gridding engine only supports built-in operators.")
    if engine == "numba" and engine_method in ("linear", "cubic", "average_distance_pts"):
        raise ValueError(f"The Numba gridding engine does not support resampling={resampling!r}.")
    if engine == "numba":
        # Fail before building a lazy graph if the requested optional engine is unavailable
        import_optional("numba")
    needs_pixel_radius = (
        (engine_method == "idw" and operator.default_neighborhood is None)
        or optimized_kernel in _IRREGULAR_REDUCER_TYPES
        or (isinstance(operator, Reducer) and reducer_neighborhood is None)
    )
    if needs_pixel_radius and not np.isfinite(dist_nodata_pixel):
        raise ValueError("Circular gridding methods require a finite dist_nodata_pixel support radius.")
    if isinstance(min_points, bool) or not isinstance(min_points, (int, np.integer)) or min_points < 0:
        raise ValueError("Argument 'min_points' must be a non-negative integer.")
    if isinstance(operator, Reducer):
        if nodata_handling is None:
            nodata_handling = "ignore"
        elif isinstance(nodata_handling, str) and nodata_handling.lower() == "nearest":
            raise ValueError("Nearest-source nodata handling requires an Interpolator.")
    order = operator.interpolation_order if isinstance(operator, Interpolator) else None
    propagation, spread_distance = _resolve_nodata_handling(nodata_handling=nodata_handling, order=order)

    # Work with finite floating-point inputs and derive the distance scale from output pixels
    res_x, res_y = _grid_resolution(grid_coords=grid_coords, grid_res=grid_res)
    points, values, source_points, source_valid = _prepare_point_gridding_data(
        pc=pc,
        data_name=data_name,
    )
    invalid_points = source_points[~source_valid]

    aligned_dem: NDArrayNum
    uses_local_method = optimized_kernel is None
    if uses_local_method:
        aligned_dem = _grid_from_points(
            pc,
            grid_coords=grid_coords,
            data_name=data_name,
            operator=operator,
            res_x=res_x,
            res_y=res_y,
            radius=dist_nodata_pixel,
            min_points=int(min_points),
            nodata_propagation=propagation,
            engine=engine,
        )
    elif len(points) == 0:
        aligned_dem = np.full((len(grid_coords[1]), len(grid_coords[0])), np.nan, dtype=np.float64)
    else:
        aligned_dem = operator._grid_points(
            points,
            values,
            grid_coords,
            res_x=res_x,
            res_y=res_y,
            radius=dist_nodata_pixel,
            min_points=int(min_points),
            n_threads=n_threads,
            engine=engine,
        )

    # Interpolators check the nearest original point; reducers use their finite result
    needs_nodata_mask = (propagation == "nearest" and isinstance(operator, Interpolator)) or (
        propagation == "propagate" and (not uses_local_method or operator.default_neighborhood is None)
    )
    if len(invalid_points) > 0 and needs_nodata_mask:
        mask_propagation = cast(Literal["nearest", "propagate"], propagation)
        _mask_grid_from_invalid_points(
            aligned_dem,
            source_points=source_points,
            source_valid=source_valid,
            grid_coords=grid_coords,
            res_x=res_x,
            res_y=res_y,
            radius=dist_nodata_pixel,
            method=engine_method,
            nodata_propagation=mask_propagation,
        )

    # A distance choice masks nearby missing points instead of applying one of the named rules
    if spread_distance is not None and len(invalid_points) > 0:
        _mask_grid_near_invalid_points(
            aligned_dem,
            invalid_points=invalid_points,
            grid_coords=grid_coords,
            res_x=res_x,
            res_y=res_y,
            radius=spread_distance,
        )

    # Flip Y axis of grid
    aligned_dem = np.flip(aligned_dem, axis=0)

    # Derive output transform from input grid
    transform_from_coords = rio.transform.from_origin(min(grid_coords[0]), max(grid_coords[1]), res_x, res_y)

    return aligned_dem, transform_from_coords


################################
# 3/ CHUNKED POINT SELECTION
################################

# Bounds filtering and per-block calculation shared by Dask and multiprocessing
###############################################################################


def _support_bounds(geogrid: GeoGrid, dist_nodata_pixel: float) -> BoundingBox:
    """Return block bounds expanded by the gridding local support radius."""

    if dist_nodata_pixel < 0:
        raise ValueError("Argument 'dist_nodata_pixel' must be non-negative.")
    x_buffer = abs(geogrid.res[0]) * dist_nodata_pixel
    y_buffer = abs(geogrid.res[1]) * dist_nodata_pixel
    return BoundingBox(
        left=geogrid.bounds.left - x_buffer,
        bottom=geogrid.bounds.bottom - y_buffer,
        right=geogrid.bounds.right + x_buffer,
        top=geogrid.bounds.top + y_buffer,
    )


def _source_support_pixels(
    geogrid: GeoGrid,
    resampling: GriddingMethod,
    dist_nodata_pixel: float,
    nodata_handling: NodataChoice | None = None,
) -> float:
    """Return the extra output pixels that must be read around each grid tile."""

    spread_distance = _gridding_nodata_spread(resampling, nodata_handling=nodata_handling) or 0
    operator = _resolve_gridding_operator(resampling, distance_power=2.0)
    builtin = _get_builtin_gridding_method(operator, default_neighborhood_only=False)
    if builtin == "idw" and operator.default_neighborhood is None:
        return max(dist_nodata_pixel, spread_distance)
    if isinstance(operator, Reducer):
        neighborhood = _resolve_point_neighbours_for_reducer(operator)
        if neighborhood is None:
            return max(dist_nodata_pixel, spread_distance)
    elif isinstance(operator, Interpolator) and _get_builtin_gridding_method(operator) is None:
        neighborhood = _resolve_point_neighbours_for_interpolator(operator)
    else:
        return max(dist_nodata_pixel, spread_distance)
    if neighborhood.radius is None:
        return float("inf")
    neighborhood_pixels = max(
        neighborhood.radius / abs(geogrid.res[0]),
        neighborhood.radius / abs(geogrid.res[1]),
    )
    return max(neighborhood_pixels, spread_distance)


def _grid_pointcloud_on_geogrid(
    pc: gpd.GeoDataFrame,
    geogrid: GeoGrid,
    data_name: str | None,
    gridding_func: GridPointCloudCallable = _grid_pointcloud,
    **kwargs: Any,
) -> NDArrayNum:
    """Grid a point-cloud subset on a single output geogrid."""

    if len(pc) == 0:
        return np.full(geogrid.shape, np.nan, dtype=np.float64)

    grid_coords = _coords(transform=geogrid.transform, shape=geogrid.shape, grid=False, area_or_point=None)
    gridding_kwargs = kwargs.copy()
    if gridding_func is _grid_pointcloud:
        gridding_kwargs["grid_res"] = geogrid.res

    array, _ = gridding_func(
        pc,
        grid_coords=grid_coords,
        data_name=data_name,
        **gridding_kwargs,
    )
    return array


def _load_pointcloud_for_geogrid(
    source_pointcloud: Any,
    geogrid: GeoGrid,
    data_name: str | None,
    gridding_options: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Load the source points needed for one output grid and its surrounding support."""

    source_support = _source_support_pixels(
        geogrid,
        resampling=gridding_options["resampling"],
        dist_nodata_pixel=gridding_options["dist_nodata_pixel"],
        nodata_handling=gridding_options.get("nodata_handling"),
    )
    return _load_pointcloud_bounds(
        source_pointcloud=source_pointcloud,
        bounds=_support_bounds(geogrid=geogrid, dist_nodata_pixel=source_support),
        data_name=data_name,
    )


def _grid_pointcloud_block_from_source(
    source_pointcloud: Any,
    geogrid: GeoGrid,
    data_name: str | None,
    gridding_func: GridPointCloudCallable = _grid_pointcloud,
    **kwargs: Any,
) -> NDArrayNum:
    """Load a point-cloud block subset and grid it."""

    pc = _load_pointcloud_for_geogrid(
        source_pointcloud=source_pointcloud,
        geogrid=geogrid,
        data_name=data_name,
        gridding_options=kwargs,
    )
    return _grid_pointcloud_on_geogrid(
        pc=pc,
        geogrid=geogrid,
        data_name=data_name,
        gridding_func=gridding_func,
        **kwargs,
    )


def _grid_pointcloud_block_from_dask_parts(
    parts: list[gpd.GeoDataFrame],
    geogrid: GeoGrid,
    data_name: str | None,
    crs: Any,
    gridding_func: GridPointCloudCallable = _grid_pointcloud,
    **kwargs: Any,
) -> NDArrayNum:
    """Concatenate Dask dataframe partitions for a block and grid them."""

    # Each delayed output block receives only the point subsets in its support area
    pc = _concat_point_parts(parts=parts, crs=crs)
    return _grid_pointcloud_on_geogrid(
        pc=pc,
        geogrid=geogrid,
        data_name=data_name,
        gridding_func=gridding_func,
        **kwargs,
    )


def _grid_pointcloud_multiproc_block(
    source_pointcloud: Any,
    geogrid: GeoGrid,
    dst_tile: tuple[int, int, int, int],
    data_name: str | None,
    gridding_func: GridPointCloudCallable = _grid_pointcloud,
    **kwargs: Any,
) -> tuple[NDArrayNum, tuple[int, int, int, int]]:
    """Grid one point-cloud block and return the output write window."""

    array = _grid_pointcloud_block_from_source(
        source_pointcloud=source_pointcloud,
        geogrid=geogrid,
        data_name=data_name,
        gridding_func=gridding_func,
        **kwargs,
    )
    return array, dst_tile


############################################
# 4/ DASK AND MULTIPROCESSING EXECUTION
############################################

# Assemble lazy Dask tiles or submit file-backed multiprocessing tiles
#######################################################################


def _dask_grid_pointcloud(
    source_pointcloud: Any,
    dst_geotiling: ChunkedGeoGrid,
    dst_block_geogrids: list[GeoGrid],
    data_name: str | None,
    gridding_func: GridPointCloudCallable = _grid_pointcloud,
    **kwargs: Any,
) -> Any:
    """Grid a point cloud lazily into a Dask array."""

    # Delay each output tile independently and expose them as one Dask array
    dask = import_optional("dask")
    import dask.array as da

    delayed = dask.delayed
    source_ds = _source_dataframe(source_pointcloud)
    source_crs = get_geo_attr(source_pointcloud, "crs")

    # Build the nested block layout expected by ``dask.array.block``
    block_arrays = []
    for iy in range(dst_geotiling.num_chunks[0]):
        row_arrays = []
        for ix in range(dst_geotiling.num_chunks[1]):
            # Match this Dask block to its georeferenced output area
            block_index = dst_geotiling.flat_block_index((iy, ix))
            geogrid = dst_block_geogrids[block_index]

            if is_dask_dataframe(source_ds):
                # Select only points inside the interpolation support before computing partitions
                source_ds_dask = cast(Any, source_ds)
                bounds = _support_bounds(
                    geogrid=geogrid,
                    dist_nodata_pixel=_source_support_pixels(
                        geogrid,
                        resampling=kwargs["resampling"],
                        dist_nodata_pixel=kwargs["dist_nodata_pixel"],
                        nodata_handling=kwargs.get("nodata_handling"),
                    ),
                )
                filtered = _filter_dask_points_by_bounds(source_ds_dask, bounds)
                # One delayed task combines the filtered partitions and grids the tile
                tile = delayed(_grid_pointcloud_block_from_dask_parts)(
                    list(filtered.to_delayed()),
                    geogrid,
                    data_name,
                    source_crs,
                    gridding_func,
                    **kwargs,
                )
            else:
                # File-backed inputs load only the bounds needed by this output tile
                tile = delayed(_grid_pointcloud_block_from_source)(
                    source_pointcloud,
                    geogrid,
                    data_name,
                    gridding_func,
                    **kwargs,
                )

            # Declare the tile shape so Dask knows the final array layout before computing
            row_arrays.append(da.from_delayed(tile, shape=geogrid.shape, dtype=np.float64))
        block_arrays.append(row_arrays)

    # Join delayed tiles without evaluating any point-cloud data
    return da.block(block_arrays)


def _multiproc_grid_pointcloud(
    source_pointcloud: Any,
    dst_geotiling: ChunkedGeoGrid,
    dst_block_geogrids: list[GeoGrid],
    data_name: str | None,
    mp_config: MultiprocConfig,
    file_metadata: dict[str, Any],
    gridding_func: GridPointCloudCallable = _grid_pointcloud,
    **kwargs: Any,
) -> Any:
    """Grid a point cloud with multiprocessing and write tiles directly to disk."""

    # Submit one independent gridding task for each output file window
    block_ids = dst_geotiling.get_block_locations()
    tasks = []
    for index, geogrid in enumerate(dst_block_geogrids):
        dst_tile = (block_ids[index]["ys"], block_ids[index]["ye"], block_ids[index]["xs"], block_ids[index]["xe"])
        tasks.append(
            mp_config.cluster.submit(
                _grid_pointcloud_multiproc_block,
                source_pointcloud,
                geogrid,
                dst_tile,
                data_name,
                gridding_func,
                **kwargs,
            )
        )

    # Write tiles as workers finish instead of holding the full raster in memory
    return _write_multiproc_result(tasks=tasks, mp_config=mp_config, file_metadata=file_metadata)


######################
# 5/ BACKEND DISPATCH
######################

# Resolve the output grid once, then select eager, Dask or multiprocessing execution
####################################################################################


def _grid_pointcloud_to_raster(
    source_pointcloud: Any,
    ref: RasterLike | None = None,
    grid_coords: tuple[NDArrayNum, NDArrayNum] | None = None,
    res: float | tuple[float, float] | None = None,
    shape: tuple[int, int] | None = None,
    bounds: tuple[float, float, float, float] | BoundingBox | None = None,
    resampling: GriddingMethod = "linear",
    dist_nodata_pixel: float = 1.0,
    nodata: int | float = -9999,
    *,
    data_name: str | None = None,
    nodata_handling: NodataChoice | None = None,
    distance_power: float = 2.0,
    min_points: int = 1,
    chunksizes: tuple[int, int] | None = None,
    mp_config: MultiprocConfig | None = None,
    dask: bool = False,
    n_threads: int = 0,
    engine: GriddingEngine = "scipy",
    gridding_func: GridPointCloudCallable = _grid_pointcloud,
) -> Any:
    """
    Grid a point cloud to a raster with eager, Dask, or Multiprocessing backends.

    A Dask reference selects lazy output even when the point source is eager. Its spatial chunks are reused unless
    chunksizes is supplied, following the same reference-grid behavior as rasterization. An explicit data_name
    selects values without copying the source or changing its active column.
    """

    # Resolve the value column from metadata so file-backed sources stay available for bounded worker reads
    if data_name is not None and (
        not isinstance(data_name, str) or data_name not in get_geo_attr(source_pointcloud, "columns")
    ):
        raise ValueError("Argument ``data_name`` must name an existing point column.")
    data_name = get_geo_attr(source_pointcloud, "data_name") if data_name is None else data_name

    # Follow a lazy output reference without converting an eager point cloud to a Dask dataframe
    ref_chunks = get_geo_attr(ref, "_chunks") if ref is not None and has_geo_attr(ref, "_chunks") else None
    if ref_chunks is not None:
        ref_chunks = ref_chunks[-2:]
        dask = True

    # A single operation must have one owner for scheduling and memory management
    if dask and mp_config is not None:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove mp_config. "
            "To use Multiprocessing, use an eager PointCloud and an unchunked raster reference."
        )

    if is_dask_dataframe(_source_dataframe(source_pointcloud)) and mp_config is not None:
        raise ValueError("Multiprocessing gridding is only supported for eager or file-backed PointCloud objects.")

    # Resolve all supported grid definitions to one output shape and transform
    out_shape, out_transform, out_crs = _check_match_grid(
        source_pointcloud,
        ref=ref,
        coords=grid_coords,
        res=res,
        bounds=bounds,
        shape=shape,
        crs=None,
    )
    dst_geogrid = GeoGrid(transform=out_transform, shape=out_shape, crs=out_crs)

    if n_threads < 0:
        raise ValueError("Argument 'n_threads' must be non-negative.")
    if gridding_func is _grid_pointcloud:
        # Convert the method name once before eager or chunked execution decides which source points to read
        resampling = _resolve_gridding_operator(resampling, distance_power=distance_power)
        if isinstance(resampling, Reducer):
            _resolve_point_neighbours_for_reducer(resampling)
        if nodata_handling is None:
            nodata_handling = "ignore" if isinstance(resampling, Reducer) else config["interpolation_nodata_handling"]
        if (
            isinstance(resampling, Reducer)
            and isinstance(nodata_handling, str)
            and nodata_handling.lower() == "nearest"
        ):
            raise ValueError("Nearest-source nodata handling requires an Interpolator.")
        _gridding_nodata_spread(resampling, nodata_handling=nodata_handling)

    # Eager calls can use SciPy threads while each parallel output task stays single-threaded
    is_parallel_backend = dask or mp_config is not None
    resolved_threads = n_threads if n_threads > 0 else (1 if is_parallel_backend else max(1, (os.cpu_count() or 2) - 1))
    kwargs: dict[str, Any] = {
        "resampling": resampling,
        "dist_nodata_pixel": dist_nodata_pixel,
    }
    if gridding_func is _grid_pointcloud:
        kwargs.update(
            nodata_handling=nodata_handling,
            distance_power=distance_power,
            min_points=min_points,
            n_threads=resolved_threads,
            engine=engine,
        )

    from geoutils.raster import Raster
    from geoutils.raster.xr_accessor import DataArrayRasterAccessor

    # The eager path grids the complete source into an in-memory Raster
    if not dask and mp_config is None:
        array = _grid_pointcloud_block_from_source(
            source_pointcloud=source_pointcloud,
            geogrid=dst_geogrid,
            data_name=data_name,
            gridding_func=gridding_func,
            **kwargs,
        )
        return Raster.from_array(data=array, transform=out_transform, crs=out_crs, nodata=nodata)

    # Reuse explicit, multiprocessing, or reference chunks in that order
    if chunksizes is None:
        if mp_config is not None:
            chunksizes = _split_chunk_size(mp_config.chunks)
        else:
            chunksizes = ref_chunks if ref_chunks is not None else (1024, 1024)
    assert chunksizes is not None

    # Describe each output chunk as a georeferenced grid for local point selection
    dst_chunks = normalize_chunks(chunks=chunksizes, shape=out_shape)
    dst_geotiling = ChunkedGeoGrid(grid=dst_geogrid, chunks=dst_chunks)
    dst_block_geogrids = dst_geotiling.get_blocks_as_geogrids()

    # Return a lazy raster accessor whose chunks compute independently
    if dask:
        data = _dask_grid_pointcloud(
            source_pointcloud=source_pointcloud,
            dst_geotiling=dst_geotiling,
            dst_block_geogrids=dst_block_geogrids,
            data_name=data_name,
            gridding_func=gridding_func,
            **kwargs,
        )
        return DataArrayRasterAccessor.from_array(data=data, transform=out_transform, crs=out_crs, nodata=nodata)

    # The remaining backend writes worker results directly to the configured file
    assert mp_config is not None
    file_metadata = {
        "height": out_shape[0],
        "width": out_shape[1],
        "count": 1,
        "dtype": np.dtype("float64"),
        "crs": out_crs,
        "transform": out_transform,
        "nodata": nodata,
    }
    return _multiproc_grid_pointcloud(
        source_pointcloud=source_pointcloud,
        dst_geotiling=dst_geotiling,
        dst_block_geogrids=dst_block_geogrids,
        data_name=data_name,
        mp_config=mp_config,
        file_metadata=file_metadata,
        gridding_func=gridding_func,
        **kwargs,
    )
