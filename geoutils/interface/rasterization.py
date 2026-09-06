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

"""Rasterize vector geometries and create raster or point cloud masks."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
from rasterio import features
from rasterio.crs import CRS
from shapely.geometry import box as shapely_box
from shapely.strtree import STRtree

from geoutils._dispatch import (
    _check_match_grid,
    _check_match_points,
    get_geo_attr,
    has_geo_attr,
    is_dask_geodataframe,
)
from geoutils._misc import import_optional
from geoutils._typing import DTypeLike, NDArrayBool, NDArrayNum, Number
from geoutils.multiproc.chunked import (
    ChunkedGeoGrid,
    GeoGrid,
    normalize_chunks,
)
from geoutils.multiproc.mparray import (
    MultiprocConfig,
    _split_chunk_size,
    _write_multiproc_result,
)

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


def _normalize_burn_values(
    vect_geoms: Sequence[Any], in_value: int | float | Iterable[int | float] | None
) -> _VectorBurnSpec:
    """
    Normalize burn values into either per-geometry values or a scalar default value.

    :param vect_geoms: Geometry sequence (length N).
    :param in_value: None, scalar, or iterable length N.
    """
    geoms = np.asarray(vect_geoms, dtype=object)

    # Default burn value, index from 1 to N
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


def _make_dtype(out_value: int | float, burn: _VectorBurnSpec, out_dtype: DTypeLike | None = None) -> DTypeLike:
    """
    Determine output dtype from fill and burn values.

    :param out_value: Fill value for background.
    :param burn: Normalized burn values.
    :param out_dtype: User-defined output data type, if provided.

    :return: Data type used for the rasterized output.
    """
    if out_dtype is not None:
        return np.dtype(out_dtype)
    dts = [np.asarray(out_value).dtype]
    if burn.values is not None:
        dts.append(np.asarray(burn.values).dtype)
    if burn.default_value is not None:
        dts.append(np.asarray(burn.default_value).dtype)
    return np.result_type(*dts)


def _build_spatial_index(geoms: NDArrayNum) -> tuple[Any, dict[int, int]]:
    """
    Build the spatial index used to find features in each raster block.

    :param geoms: Geometry array.

    :return: Spatial index and mapping from geometry identities to their original positions.
    """
    # Build the index once from all input geometries
    tree = STRtree(list(geoms))

    # Older Shapely versions return geometries instead of their original positions
    geometry_positions = {id(geometry): index for index, geometry in enumerate(geoms)}
    return tree, geometry_positions


def _query_indices(tree: Any, geometry_positions: dict[int, int], query_geom: Any) -> NDArrayNum:
    """
    Return the positions of geometries intersecting a geographic area.

    :param tree: Spatial index containing all input geometries.
    :param geometry_positions: Mapping from geometry identities to their original positions.
    :param query_geom: Geographic area to query.

    :return: Positions of intersecting geometries in the original geometry array.
    """

    # Current Shapely versions return the original array positions directly
    try:
        indices = tree.query(query_geom, predicate="intersects")
        return np.asarray(indices, dtype=np.int64)

    # Older Shapely versions return geometry objects that need mapping back to positions
    except TypeError:
        intersecting_geometries = tree.query(query_geom)
        if not intersecting_geometries:
            return np.empty((0,), dtype=np.int64)
        return np.asarray(
            [
                geometry_positions[id(geometry)]
                for geometry in intersecting_geometries
                if query_geom.intersects(geometry)
            ],
            dtype=np.int64,
        )


def _subset_burn(burn: _VectorBurnSpec, indices: NDArrayNum) -> _VectorBurnSpec:
    """
    Select the geometries and optional values needed by one output block.

    :param burn: Complete vector rasterization inputs.
    :param indices: Positions of features intersecting the output block.

    :return: Rasterization inputs limited to the selected features.
    """

    # Keep geometry values aligned when an individual value is used for each feature
    values = None if burn.values is None else burn.values[indices]
    return _VectorBurnSpec(geoms=burn.geoms[indices], values=values, default_value=burn.default_value)


def _partition_burn_by_geogrids(burn: _VectorBurnSpec, geogrids: list[GeoGrid]) -> list[_VectorBurnSpec]:
    """
    Select the relevant vector features once for every output block.

    :param burn: Complete vector rasterization inputs.
    :param geogrids: Georeferenced output blocks.

    :return: Rasterization inputs limited to each output block.
    """

    # Build one spatial index for the complete vector input
    tree, geometry_positions = _build_spatial_index(burn.geoms)

    # Query every block while the complete index remains available in this process
    block_burns = []
    for geogrid in geogrids:
        # Query the geographic area covered by this output block
        bounds = geogrid.bounds
        query_box = shapely_box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        indices = _query_indices(tree, geometry_positions=geometry_positions, query_geom=query_box)
        block_burns.append(_subset_burn(burn, indices))
    return block_burns


def _rasterio_rasterize_burn(
    geoms: NDArrayNum,
    values: NDArrayNum | None,
    default_value: int | float | None,
    out_shape: tuple[int, int],
    transform: Any,
    fill: int | float,
    dtype: DTypeLike,
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
    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
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


def _rasterize_selected_on_geogrid(
    geogrid: GeoGrid,
    burn: _VectorBurnSpec,
    out_value: int | float,
    out_dtype: DTypeLike | None = None,
    *,
    all_touched: bool = False,
) -> NDArrayNum:
    """
    Rasterize features already selected for one output block.

    :param geogrid: Georeferencing and shape of the output block.
    :param burn: Rasterization inputs limited to features intersecting the block.
    :param out_value: Background fill value.
    :param out_dtype: Output array data type.
    :param all_touched: Whether to include every pixel touched by a geometry.

    :return: Rasterized output block.
    """

    # Use one consistent data type for empty and rasterized blocks
    dtype = _make_dtype(out_value=out_value, burn=burn, out_dtype=out_dtype)

    # Empty blocks can be filled without calling Rasterio
    if len(burn.geoms) == 0:
        return np.full(geogrid.shape, out_value, dtype=dtype)

    # Burn only the features intersecting this block
    return _rasterio_rasterize_burn(
        geoms=burn.geoms,
        values=burn.values,
        default_value=burn.default_value,
        out_shape=geogrid.shape,
        transform=geogrid.transform,
        fill=out_value,
        dtype=dtype,
        all_touched=all_touched,
    )


def _rasterize_base(
    burn: _VectorBurnSpec,
    out_shape: tuple[int, int],
    out_transform: Any,
    out_value: int | float = 0,
    out_dtype: DTypeLike | None = None,
    all_touched: bool = False,
) -> NDArrayNum:
    """
    Rasterize geometry into a NumPy array.

    :param burn: Normalized geometries and burn values in the output CRS.
    :param out_shape: Output array shape (rows, cols).
    :param out_transform: Output affine transform.
    :param out_value: Background fill value.
    :param out_dtype: Output array data type.
    :param all_touched: Whether to include every pixel touched by a geometry.

    :return: Rasterized NumPy array.
    """
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
    :param out_dtype: Output array data type.
    :param all_touched: Whether to include every pixel touched by a geometry.

    :return: Lazy rasterized Dask array.
    """
    dask = import_optional("dask")
    import dask.array as da

    dtype = _make_dtype(out_value=out_value, burn=burn, out_dtype=out_dtype)

    # Select each block's geometries before building the lazy computation
    block_burns = _partition_burn_by_geogrids(burn, dst_block_geogrids)

    # Build a two-dimensional layout of independently computed blocks
    block_arrays = []
    for iy in range(dst_geotiling.num_chunks[0]):
        row_arrays = []
        for ix in range(dst_geotiling.num_chunks[1]):
            block_index = dst_geotiling.flat_block_index((iy, ix))
            geogrid = dst_block_geogrids[block_index]

            # Give each block computation only the geometries it needs
            tile = dask.delayed(_rasterize_selected_on_geogrid)(
                geogrid,
                block_burns[block_index],
                out_value,
                dtype,
                all_touched=all_touched,
            )
            row_arrays.append(da.from_delayed(tile, shape=geogrid.shape, dtype=dtype))
        block_arrays.append(row_arrays)

    # Join the blocks lazily while preserving their requested chunk sizes
    return da.block(block_arrays)


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
    :param out_dtype: Output array data type.
    :param all_touched: Whether to include every pixel touched by a geometry.

    :return: File-backed raster containing the completed blocks.
    """
    block_ids = dst_geotiling.get_block_locations()

    # Build the spatial index once and send each task only its relevant features
    block_burns = _partition_burn_by_geogrids(burn, dst_block_geogrids)

    # Send one independent output block to the available worker pool
    tasks = [
        mp_config.cluster.submit(
            _multiproc_rasterize_block,
            block_ids[i],
            dst_block_geogrids[i],
            block_burns[i],
            out_value,
            out_dtype,
            all_touched,
        )
        for i in range(len(block_ids))
    ]

    # Write tiles as they complete
    return _write_multiproc_result(tasks=tasks, mp_config=mp_config, file_metadata=file_metadata)


def _multiproc_rasterize_block(
    block_id: dict[str, Any],
    block_geogrid: GeoGrid,
    burn: _VectorBurnSpec,
    out_value: int | float,
    out_dtype: DTypeLike | None,
    all_touched: bool,
) -> tuple[NDArrayNum, tuple[int, int, int, int]]:
    """
    Rasterize one output block in a multiprocessing worker.

    :param block_id: Pixel positions of the output block in the complete raster.
    :param block_geogrid: Georeferencing and shape of the output block.
    :param burn: Rasterization inputs limited to features intersecting the block.
    :param out_value: Background fill value.
    :param out_dtype: Output array data type.
    :param all_touched: Whether to include every pixel touched by a geometry.

    :return: Rasterized block and its pixel positions in the complete raster.
    """

    # The parent process already selected this block's vector inputs
    tile = _rasterize_selected_on_geogrid(
        block_geogrid,
        burn,
        out_value,
        out_dtype,
        all_touched=all_touched,
    )

    # Return destination indexes together with the array for the shared writer
    dst_tile = (block_id["ys"], block_id["ye"], block_id["xs"], block_id["xe"])
    return tile, dst_tile


def _rasterize(
    source_vector: Any,
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
    mask_output: bool = False,
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
    :param dask: If True, return a Dask-backed Raster. A Dask-backed reference raster also selects this backend.
    :param mask_output: Return boolean values for an in-memory or Dask mask.
    """
    # Compute output grid
    out_shape, out_transform, out_crs = _check_match_grid(
        src=source_vector, ref=ref, res=res, shape=shape, bounds=bounds, crs=crs, coords=grid_coords
    )

    # Reproject only when the source and destination reference systems differ
    if out_crs is not None and source_vector.crs != out_crs:
        source_vector = source_vector.to_crs(out_crs)
    vect = source_vector.ds

    # Cannot use Multiprocessing backend and Dask backend simultaneously
    mp_backend = mp_config is not None
    # A Dask reference keeps its chunked representation unless Multiprocessing is requested
    ref_chunks = get_geo_attr(ref, "_chunks") if ref is not None and has_geo_attr(ref, "_chunks") else None
    dask_backend = bool(dask) or (da is not None and ref_chunks is not None)

    if mp_backend and dask_backend:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove mp_config. "
            "To use Multiprocessing, set dask=False."
        )

    # Normalize burn once
    burn = _normalize_burn_values(vect_geoms=vect.geometry.values, in_value=in_value)

    # Runtime import to avoid circular import
    from geoutils.raster import Raster
    from geoutils.raster.xr_accessor import RasterAccessor

    # Base backend (eager)
    if not mp_backend and not dask_backend:
        data = _rasterize_base(
            burn=burn,
            out_shape=out_shape,
            out_transform=out_transform,
            out_value=out_value,
            out_dtype=out_dtype,
            all_touched=all_touched,
        )

        # Byte rasterization is supported by Rasterio and has a zero-copy boolean view
        if mask_output:
            data = data.view(np.bool_)
        return Raster.from_array(data=data, transform=out_transform, crs=out_crs, nodata=None)

    # Build chunked geogrid (shared for Dask and multiproc)
    if chunksizes is None:
        if mp_backend:
            assert mp_config is not None
            chunksizes = _split_chunk_size(mp_config.chunks)
        else:
            if ref_chunks is not None:
                chunksizes = ref_chunks
            else:
                chunksizes = (1024, 1024)
    assert chunksizes is not None

    dst_geogrid = GeoGrid(transform=out_transform, shape=out_shape, crs=out_crs)
    dst_chunks = normalize_chunks(chunks=chunksizes, shape=out_shape)
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

        # Convert each completed byte block to a boolean view without another array allocation
        if mask_output:
            data = data.view(np.bool_)
        return RasterAccessor.from_array(data=data, transform=out_transform, crs=out_crs, nodata=None)

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
    assert mp_config is not None
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


def _create_mask_pointcloud(
    source_vector: Vector, points: tuple[NDArrayNum, NDArrayNum] | PointCloudLike, as_array: bool = False
) -> NDArrayBool | PointCloud:
    """Subfunction to create a point cloud mask using geopandas."""

    # Normalize input
    points, _ = _check_match_points(src=source_vector, points=points)
    points_gs = gpd.GeoSeries(gpd.points_from_xy(x=points[0], y=points[1]), crs=source_vector.crs)

    # Project to same CRS if required
    points_gs = points_gs.to_crs(crs=source_vector.crs)

    # Check whether points are contained in any source geometry
    contained = points_gs.within(source_vector.ds.geometry.union_all())

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


def _empty_point_mask_meta(crs: CRS | None) -> gpd.GeoDataFrame:
    """Build an empty point-cloud mask GeoDataFrame for Dask outputs."""

    # Dask uses this empty object to plan partition output without reading points
    return gpd.GeoDataFrame(
        data={"z": pd.Series(dtype="bool")},
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )


def _mask_pointcloud_partition(part: gpd.GeoDataFrame, source_geom: Any, crs: CRS | None) -> gpd.GeoDataFrame:
    """Create a point-cloud mask for one GeoDataFrame partition."""

    # Preserve the declared Dask metadata when a partition contains no points
    if len(part) == 0:
        return _empty_point_mask_meta(crs)

    # Test all points in this partition against the combined source geometry
    contained = part.geometry.within(source_geom)
    return gpd.GeoDataFrame(
        data={"z": contained.to_numpy(dtype=bool)},
        geometry=part.geometry,
        crs=crs,
        index=part.index,
    )


def _create_mask_pointcloud_dask(source_vector: Vector, points: Any, as_array: bool = False) -> Any:
    """Create a point-cloud mask lazily from a Dask-GeoPandas point cloud."""

    # Keep Dask-GeoPandas optional until a lazy point cloud reaches this path
    import_optional("dask_geopandas", package_name="dask-geopandas")

    # Reproject lazily and prepare one geometry shared by all partition tasks
    points_in_crs = points if points.crs == source_vector.crs else points.to_crs(source_vector.crs)
    source_geom = source_vector.ds.geometry.union_all()
    meta = _empty_point_mask_meta(source_vector.crs)
    # Each point partition becomes an equally partitioned boolean point cloud
    out = points_in_crs.map_partitions(_mask_pointcloud_partition, source_geom, source_vector.crs, meta=meta)

    # Import at runtime because the point-cloud base also uses rasterization through its vector parent
    from geoutils.pointcloud.base import _set_dataframe_attrs

    # Restore the metadata expected by the GeoUtils ``pc`` accessor
    _set_dataframe_attrs(
        out,
        {
            "crs": source_vector.crs,
            "bounds": None,
            "point_count": None,
            "data_column": "z",
            "geometry_type": "Point",
        },
    )

    if as_array:
        # Return a lazy value array while keeping point partitions uncomputed
        return out["z"].to_dask_array(lengths=True)
    return out


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
        mask_output=True,
    )

    # File-backed masks remain bytes on disk and are converted when loaded
    if not isinstance(rst01, xr.DataArray):
        rst01._is_mask = True

    if as_array:
        return rst01.data
    return rst01


def _create_mask(
    source_vector: Any,
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
    point_ref = ref if ref is not None else points
    try:
        if is_dask_geodataframe(point_ref):
            is_ref_points = True
        else:
            _check_match_points(
                src=source_vector,
                points=point_ref,
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
            "Input arguments must define a valid raster or point cloud. Pass either a raster or point cloud "
            "reference 'ref', or at least 'res'/'shape' for a raster mask, or 'points' for a point cloud mask."
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
            as_array=as_array,
        )
    # Point cloud mask path: point cloud ref OR points provided
    else:
        if is_dask_geodataframe(point_ref):
            return _create_mask_pointcloud_dask(source_vector=source_vector, points=point_ref, as_array=as_array)

        # Create boolean mask for points
        return _create_mask_pointcloud(
            source_vector=source_vector, points=points if points is not None else ref, as_array=as_array
        )
