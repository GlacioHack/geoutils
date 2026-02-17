# Copyright (c) 2025 GeoUtils developers
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

"""Functionalities at the interface of rasters and vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
from rasterio import features
from rasterio.crs import CRS
from shapely.strtree import STRtree
from shapely.geometry import box as shapely_box

from geoutils._dispatch import _check_match_grid, _check_match_points
from geoutils._misc import silence_rasterio_message, import_optional
from geoutils._typing import NDArrayBool, NDArrayNum, Number, DTypeLike
from geoutils.multiproc.chunked import GeoGrid, ChunkedGeoGrid, _chunks2d_from_chunksizes_shape
from geoutils.multiproc.mparray import MultiprocConfig, _write_multiproc_result

from typing import Any, Iterable, Sequence

if TYPE_CHECKING:
    from geoutils.pointcloud.pointcloud import PointCloud, PointCloudLike
    from geoutils.raster.base import Raster, RasterLike, RasterType
    from geoutils.vector.vector import Vector

try:
    import dask.array as da
except ImportError:
    da = None


##################
# 1/ RASTERIZATION
##################

# Common helpers

@dataclass(frozen=True)
class _VectorBurnSpec:
    """
    Normalized rasterization inputs.

    :param geoms: Vector geometries in output CRS.
    :param values: Per-geometry burn values, or None for scalar burn.
    :param default_value: Scalar burn value if values is None.
    """
    geoms: NDArrayNum
    values: NDArrayNum | None
    default_value: int | float | None


def _normalize_burn_values(vect_geoms: Sequence[Any], in_value: int | float | Iterable[int | float] | None) -> _VectorBurnSpec:
    """
    Normalize burn values into either per-geometry values or a scalar default value.

    :param vect_geoms: Geometry sequence (length N).
    :param in_value: None, scalar, or iterable length N.
    """
    geoms = np.asarray(vect_geoms, dtype=object)

    # Default burn value, index from 1..N
    if in_value is None:
        values = np.arange(1, len(geoms) + 1, dtype=np.int64)
        return _VectorBurnSpec(geoms=geoms, values=values, default_value=None)

    # Per-geometry values
    if isinstance(in_value, Iterable) and not isinstance(in_value, (str, bytes)):
        vals = np.asarray(list(in_value))
        if len(vals) != len(geoms):
            raise ValueError(f"in_value must match geometry length, currently {len(vals)} != {len(geoms)}.")
        return _VectorBurnSpec(geoms=geoms, values=vals, default_value=None)

    # Scalar burn
    if isinstance(in_value, (int, float, np.integer, np.floating)):
        return _VectorBurnSpec(geoms=geoms, values=None, default_value=float(in_value))

    raise ValueError("in_value must be a single number or an iterable with same length as geometry.")


def _make_dtype(out_value: int | float, burn: _VectorBurnSpec, out_dtype: DTypeLike | None = None) -> np.dtype:
    """
    Determine output dtype from fill and burn values.

    :param out_value: Fill value for background.
    :param burn: Normalized burn values.
    """
    if out_dtype is not None:
        return np.dtype(out_dtype)
    dts = [np.asarray(out_value).dtype]
    if burn.values is not None:
        dts.append(np.asarray(burn.values).dtype)
    if burn.default_value is not None:
        dts.append(np.asarray(burn.default_value).dtype)
    return np.result_type(*dts)


def _get_strtree_and_lut(geoms: NDArrayNum, cache: dict[str, Any]):
    """
    Build and cache STRtree + id to idx LUT for Shapely 1.8 compatibility.

    :param geoms: Geometry array.
    :param cache: Mutable dict used as per-worker cache.
    """

    tree = cache.get("tree", None)
    if tree is None:
        tree = STRtree(list(geoms))
        cache["tree"] = tree
        cache["id_to_idx"] = {id(g): i for i, g in enumerate(geoms)}
    return tree, cache["id_to_idx"]


def _query_indices(tree, id_to_idx: dict[int, int], query_geom) -> NDArrayNum:
    """
    Query indices of geometries intersecting query geometry.

    :param tree: STRtree instance.
    :param id_to_idx: id(geom)->index LUT (Shapely 1.8 fallback).
    :param query_geom: Shapely geometry.
    """

    # Shapely 2: query(..., predicate="intersects") returns indices
    try:
        idx = tree.query(query_geom, predicate="intersects")
        return np.asarray(idx, dtype=np.int64)
    # Shapely 1.8: query returns geometries
    except TypeError:
        hits = tree.query(query_geom)
        if not hits:
            return np.empty((0,), dtype=np.int64)
        return np.asarray([id_to_idx[id(g)] for g in hits if query_geom.intersects(g)], dtype=np.int64)

def _rasterio_rasterize_burn(
    geoms: NDArrayNum,
    values: NDArrayNum | None,
    default_value: int | float | None,
    out_shape: tuple[int, int],
    transform: Any,
    fill: int | float,
    dtype: np.dtype,
    all_touched: bool = False,
) -> NDArrayNum:
    """
    Call rasterio.features.rasterize with either per-geometry values or scalar default_value.

    :param geoms: Geometry array (dtype=object).
    :param values: Per-geometry burn values, or None for scalar burn.
    :param default_value: Scalar burn value if values is None.
    :param out_shape: Output shape (rows, cols).
    :param transform: Affine transform for the output grid.
    :param fill: Fill value for background.
    :param dtype: Output dtype.
    :param all_touched: Rasterio rasterize option.
    """
    if values is None:
        return features.rasterize(
            shapes=geoms,
            out_shape=out_shape,
            transform=transform,
            fill=fill,
            default_value=default_value,
            all_touched=all_touched,
            dtype=dtype,
        )

    shapes = ((geoms[i], values[i]) for i in range(len(geoms)))
    return features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=fill,
        all_touched=all_touched,
        dtype=dtype,
    )

def _rasterize_on_geogrid(
    gg: GeoGrid,
    burn: _VectorBurnSpec,
    out_value: int | float,
    out_dtype: DTypeLike | None = None,
    *,
    all_touched: bool = False,
    cache: dict[str, Any] | None = None,
) -> NDArrayNum:
    """
    Rasterize vector features onto a single block GeoGrid.

    :param gg: Output block geogrid (shape/transform/bounds in output CRS).
    :param burn: Normalized burn values and geometries.
    :param out_value: Background fill value.
    :param all_touched: Rasterio rasterize option.
    :param cache: Optional per-worker cache for spatial index.
    """

    dtype = _make_dtype(out_value=out_value, burn=burn, out_dtype=out_dtype)

    # Build spatial index (cached per worker if provided)
    if cache is None:
        cache = {}
    tree, id_to_idx = _get_strtree_and_lut(burn.geoms, cache=cache)

    # Conservative bbox selection for candidate features
    bb = gg.bounds
    qbox = shapely_box(bb.left, bb.bottom, bb.right, bb.top)
    idx = _query_indices(tree, id_to_idx=id_to_idx, query_geom=qbox)

    # Early exit if no candidates
    if idx.size == 0:
        return np.full(gg.shape, out_value, dtype=dtype)

    # Subset geometries/values for this tile
    geoms = burn.geoms[idx]
    vals = None if burn.values is None else burn.values[idx]

    # Rasterize only candidates into the tile
    return _rasterio_rasterize_burn(
        geoms=geoms,
        values=vals,
        default_value=burn.default_value,
        out_shape=gg.shape,
        transform=gg.transform,
        fill=out_value,
        dtype=dtype,
        all_touched=all_touched,
    )

def _rasterize_base(
    vect_geoms: Sequence[Any],
    out_shape: tuple[int, int],
    out_transform: Any,
    in_value: int | float | Iterable[int | float] | None,
    out_value: int | float = 0,
    out_dtype: DTypeLike | None = None,
    all_touched: bool = False,
) -> NDArrayNum:
    """
    Rasterize geometry into a NumPy array.

    :param vect_geoms: Geometry sequence (already in output CRS).
    :param out_shape: Output array shape (rows, cols).
    :param out_transform: Output affine transform.
    :param in_value: Burn value(s) (scalar, iterable, or None for 1..N).
    :param out_value: Background fill value.
    :param all_touched: Rasterio rasterize option.
    """
    # Process inputs
    burn = _normalize_burn_values(vect_geoms=vect_geoms, in_value=in_value)
    dtype = _make_dtype(out_value=out_value, burn=burn, out_dtype=out_dtype)

    # Rasterize
    return _rasterio_rasterize_burn(
        geoms=burn.geoms,
        values=burn.values,
        default_value=burn.default_value,
        out_shape=out_shape,
        transform=out_transform,
        fill=out_value,
        dtype=dtype,
        all_touched=all_touched,
    )

def _dask_rasterize(
    burn: _VectorBurnSpec,
    dst_geotiling: ChunkedGeoGrid,
    dst_block_geogrids: list[GeoGrid],
    out_value: int | float = 0,
    out_dtype: DTypeLike | None = None,
    all_touched: bool = False,
) -> da.Array:
    """
    Rasterize lazily into a Dask array.

    :param burn: Normalized burn values and geometries (in output CRS).
    :param dst_geotiling: Chunked geogrid for the output.
    :param dst_block_geogrids: List of per-chunk GeoGrids.
    :param out_value: Background fill value.
    :param all_touched: Rasterio rasterize option.
    """
    import_optional("dask")
    import dask.array as da

    dtype = _make_dtype(out_value=out_value, burn=burn, out_dtype=out_dtype)
    template = da.empty(dst_geotiling.grid.shape, chunks=dst_geotiling.chunks, dtype=dtype)

    # Per-worker cache (tree + LUT)
    _CACHE: dict[str, Any] = {}

    def _flat_block_index(chunk_location: tuple[int, int]) -> int:
        """Convert Dask chunk location (iy, ix) into flat index matching get_blocks_as_geogrids()."""
        iy, ix = chunk_location
        nx = len(dst_geotiling.chunks[1])
        return iy * nx + ix

    def _block_func(block: np.ndarray, block_info=None) -> np.ndarray:
        """Block function for Dask."""
        info = block_info[None]
        bidx = _flat_block_index(info["chunk-location"])
        gg = dst_block_geogrids[bidx]
        return _rasterize_on_geogrid(gg, burn, out_value, dtype, all_touched=all_touched, cache=_CACHE)

    return template.map_blocks(_block_func, dtype=dtype)

def _multiproc_rasterize(
    burn: _VectorBurnSpec,
    dst_geotiling: ChunkedGeoGrid,
    dst_block_geogrids: list[GeoGrid],
    mp_config: MultiprocConfig,
    file_metadata: dict[str, Any],
    out_value: int | float = 0,
    out_dtype: DTypeLike | None = None,
    all_touched: bool = False,
) -> Raster:
    """
    Rasterize using multiprocessing and write results lazily to file.

    :param burn: Normalized burn values and geometries (in output CRS).
    :param dst_geotiling: Chunked geogrid for the output.
    :param dst_block_geogrids: List of per-chunk GeoGrids.
    :param mp_config: Multiprocessing configuration (includes cluster/outfile/driver).
    :param file_metadata: Rasterio metadata for output file.
    :param out_value: Background fill value.
    :param all_touched: Rasterio rasterize option.
    """
    block_ids = dst_geotiling.get_block_locations()

    def rasterize_block(task: tuple[int, dict[str, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """
        Block function for multiprocessing.

        Returns result_tile (2D array for the tile) and dst_tile (ys, ye, xs, xe) destination indexes.
        """
        bidx, bid = task

        gg = dst_block_geogrids[bidx]
        tile = _rasterize_on_geogrid(gg, burn, out_value, out_dtype, all_touched=all_touched, cache={})

        dst_tile = (bid["ys"], bid["ye"], bid["xs"], bid["xe"])
        return tile, dst_tile

    # Submit tasks to cluster interface
    tasks = [mp_config.cluster.launch_task(rasterize_block, [i, block_ids[i]]) for i in range(len(block_ids))]

    # Write tiles as they complete
    return _write_multiproc_result(tasks=tasks, mp_config=mp_config, file_metadata=file_metadata)

def _rasterize(
    source_vector: Vector,
    ref: RasterType | None = None,
    in_value: int | float | Iterable[int | float] | None = None,
    out_value: int | float = 0,
    all_touched: bool = False,
    out_dtype: DTypeLike | None = None,
    res: tuple[Number, Number] | Number | None = None,
    shape: tuple[int, int] | None = None,
    grid_coords: tuple[NDArrayNum, NDArrayNum] | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    crs: CRS | int | None = None,
    *,
    chunksizes: tuple[int, int] | None = None,
    mp_config: MultiprocConfig | None = None,
    dask: bool = False,
) -> Raster:
    """
    Rasterize vector to raster, with optional Dask or Multiprocessing backends.

    :param source_vector: Input vector.
    :param ref: Reference raster to match grid.
    :param in_value: Burn values (scalar, iterable, or None for 1..N).
    :param out_value: Background fill value.
    :param out_dtype: Output dtype for the raster.
    :param all_touched: Whether to rasterize all touched geometries or not.
    :param res: Output resolution.
    :param shape: Output shape.
    :param grid_coords: Output coordinates.
    :param bounds: Output bounds.
    :param crs: Output CRS.
    :param chunksizes: Chunk size (rows, cols) for Dask/Multiproc (if no reference raster is passed, or not chunked).
    :param mp_config: Multiprocessing config.
    :param dask: If True, return a Dask-backed Raster.
    """
    # Compute output grid
    out_shape, out_transform, out_crs = _check_match_grid(
        src=source_vector, ref=ref, res=res, shape=shape, bounds=bounds, crs=crs, coords=grid_coords
    )

    # Reproject vector into output CRS if needed
    if out_crs is not None:
        source_vector = source_vector.to_crs(out_crs)
    vect = source_vector.ds

    # Cannot use Multiprocessing backend and Dask backend simultaneously
    mp_backend = mp_config is not None
    # If input reference raster is Dask-based, create Dask output by default
    dask_backend = (da is not None and ref is not None and ref._chunks is not None) or bool(dask)

    if mp_backend and dask_backend:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove mp_config. "
            "To use Multiprocessing, set dask=False."
        )

    # Normalize burn once
    burn = _normalize_burn_values(vect_geoms=vect.geometry.values, in_value=in_value)

    # Runtime import to avoid circular import
    from geoutils.raster import Raster

    # Base backend (eager)
    if not mp_backend and not dask_backend:
        mask = _rasterize_base(
            vect_geoms=burn.geoms,
            out_shape=out_shape,
            out_transform=out_transform,
            in_value=in_value,
            out_value=out_value,
            out_dtype=out_dtype,
            all_touched=all_touched,
        )
        return Raster.from_array(data=mask, transform=out_transform, crs=out_crs, nodata=None)

    # Build chunked geogrid (shared for Dask and multiproc)
    if chunksizes is None:
        if ref is not None and ref._chunks is not None:
            chunksizes = ref._chunks
        else:
            chunksizes = (1024, 1024)

    dst_geogrid = GeoGrid(transform=out_transform, shape=out_shape, crs=out_crs)
    dst_chunks = _chunks2d_from_chunksizes_shape(chunksizes=chunksizes, shape=out_shape)
    dst_geotiling = ChunkedGeoGrid(grid=dst_geogrid, chunks=dst_chunks)
    dst_block_geogrids = dst_geotiling.get_blocks_as_geogrids()

    # Dask backend (lazy)
    if dask_backend:
        data = _dask_rasterize(
            burn=burn,
            dst_geotiling=dst_geotiling,
            dst_block_geogrids=dst_block_geogrids,
            out_value=out_value,
            out_dtype=out_dtype,
            all_touched=all_touched,
        )
        return Raster.from_array(data=data, transform=out_transform, crs=out_crs, nodata=None)

    # Multiprocessing backend (lazy and writes to file)

    # Build minimal output metadata for file writer
    dtype = _make_dtype(out_value=out_value, burn=burn, out_dtype=out_dtype)
    file_metadata = {
        "height": out_shape[0],
        "width": out_shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": out_crs,
        "transform": out_transform,
        "nodata": None,
    }
    return _multiproc_rasterize(
        burn=burn,
        dst_geotiling=dst_geotiling,
        dst_block_geogrids=dst_block_geogrids,
        mp_config=mp_config,
        file_metadata=file_metadata,
        out_value=out_value,
        out_dtype=dtype,
        all_touched=all_touched,
    )

#######################################################################
# 2/ GEOMETRY MASKING (BOOLEAN RASTERIZE OR BOOLEAN POINT WITHIN CHECK)
#######################################################################

def _create_mask_pointcloud(source_vector: Vector,
                            points: tuple[NDArrayNum, NDArrayNum] | PointCloudLike,
                            as_array: bool = False) -> NDArrayBool:
    """Subfunction to create a point cloud mask using geopandas."""

    # Normalize input
    points = _check_match_points(src=source_vector, points=points)
    points_gs = gpd.points_from_xy(x=points[0], y=points[1])

    # Project to same CRS if required
    points_gs = points_gs.reproject(crs=source_vector.crs)

    # Check that points are contained no matter alignment
    contained = points_gs.within(source_vector.ds, align=False)

    if as_array:
        # Extract resulting boolean array
        return contained.values
    else:

        # Runtime import to avoid circularity issues
        from geoutils.pointcloud import PointCloud

        # Return PointCloud with z = mask
        return PointCloud.from_xyz(
            x=points[0],
            y=points[1],
            z=contained,
            crs=source_vector.crs,
        )


def _create_mask_raster(
    source_vector: Vector,
    ref: RasterLike | None,
    all_touched: bool,
    crs: CRS | None,
    res: float | tuple[float, float] | None,
    bounds: tuple[float, float, float, float] | None,
    shape: tuple[int, int] | None,
    grid_coords: tuple[NDArrayNum, NDArrayNum] | None,
    *,
    chunksizes: tuple[int, int] | None = None,
    mp_config: MultiprocConfig | None = None,
    dask: bool = False,
    as_array: bool = False,
) -> Any:
    """
    Subfunction to create a raster mask using rasterization.

    Burns 1 inside geometries and 0 outside, then returns a boolean array/dask array
    """
    rst01 = _rasterize(
        source_vector=source_vector,
        ref=ref,
        in_value=1,
        out_value=0,
        all_touched=all_touched,
        crs=crs,
        res=res,
        shape=shape,
        grid_coords=grid_coords,
        bounds=bounds,
        chunksizes=chunksizes,
        mp_config=mp_config,
        dask=dask,
        out_dtype=np.uint8,  # avoid large dtype + keep rasterize fast/safe
    )

    # Convert to boolean (lazy if dask-backed)
    # TODO: Add logic to load as bool in delayed manner in Raster class?
    rst_bool = rst01.copy(new_array=rst01.data.astype(bool))

    if as_array:
        return rst_bool.data
    return rst_bool


def _create_mask(
    source_vector: Vector,
    ref: RasterLike | PointCloudLike | None = None,
    all_touched: bool = False,
    crs: CRS | None = None,
    res: float | tuple[float, float] | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    shape: tuple[int, int] | None = None,
    grid_coords: tuple[NDArrayNum, NDArrayNum] | None = None,
    points: tuple[NDArrayNum, NDArrayNum] | None = None,
    as_array: bool = False,
    *,
    chunksizes: tuple[int, int] | None = None,
    mp_config: MultiprocConfig | None = None,
    dask: bool = False,
) -> Raster | PointCloud | NDArrayBool:
    """
    Create a mask from a vector.

    If a raster reference or raster definition is provided, returns a raster mask.
    If a point cloud reference or points are provided, returns a point cloud mask.
    """

    # Check raster definition
    err_rast = None
    try:
        _check_match_grid(
            src=source_vector,
            ref=ref,
            res=res,
            shape=shape,
            bounds=bounds,
            coords=grid_coords,
            crs=crs,
        )
        is_ref_raster = True
    except ValueError as e:
        is_ref_raster = False
        err_rast = e

    # Check point definition
    err_points = None
    try:
        _check_match_points(
            src=source_vector,
            points=ref if ref is not None else points,
        )
        is_ref_points = True
    except ValueError as e:
        is_ref_points = False
        err_points = e

    # Validate
    if not (is_ref_raster or is_ref_points):
        # Prefer to chain the raster error (or the points one if raster not triggered)
        cause = err_rast or err_points
        raise ValueError(
            "Input arguments must define a valid raster or point cloud."
        ) from cause

    # For raster input
    if is_ref_raster:
        # Compute raster mask as 0/1 with dtype uint8, then convert to bool
        return _create_mask_raster(
            source_vector=source_vector,
            ref=ref,
            all_touched=all_touched,
            crs=crs,
            res=res,
            shape=shape,
            grid_coords=grid_coords,
            bounds=bounds,
            chunksizes=chunksizes,
            mp_config=mp_config,
            dask=dask,
            as_array=as_array
        )
    # Point cloud mask path: point cloud ref OR points provided
    else:
        # Create boolean mask for points
        return _create_mask_pointcloud(source_vector=source_vector, points=points if points is not None else ref,
                                       as_array=as_array)
