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

"""Exact conversions between rasters and regular point clouds, without gridding or interpolation."""

from __future__ import annotations

import pathlib
import tempfile
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Iterable, Literal, cast

import affine
import geopandas as gpd
import numpy as np
import rasterio as rio
from numpy.typing import NDArray
from rasterio.crs import CRS

from geoutils._dispatch import get_geo_attr, has_geo_attr, is_dask_array
from geoutils._misc import import_optional
from geoutils._typing import DTypeLike, NDArrayNum
from geoutils.raster.referencing import _default_nodata, _ij2xy, _xy2ij

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterType


##################################
# 1/ REGULAR POINT CLOUD TO RASTER
##################################


def _regular_pointcloud_to_raster(
    pointcloud: PointCloudLike,
    grid_coords: tuple[NDArrayNum, NDArrayNum] = None,
    transform: rio.transform.Affine = None,
    shape: tuple[int, int] = None,
    nodata: int | float | None = None,
    data_column_name: str | None = "b1",
    area_or_point: Literal["Area", "Point"] = "Point",
) -> tuple[NDArrayNum, affine.Affine, CRS, int | float | None, Literal["Area", "Point"]]:
    """
    Convert a regular point cloud to a raster. See Raster.from_pointcloud_regular() for details.
    """

    # Extract geodataframe and data column name depending on input
    if has_geo_attr(pointcloud, "ds", accessors=("pc",)):
        gdf_pc = get_geo_attr(pointcloud, "ds", accessors=("pc",))
        pc_data_column_name = get_geo_attr(pointcloud, "data_column", accessors=("pc",))
        if pc_data_column_name is not None:
            data_column_name = pc_data_column_name
    else:
        gdf_pc = pointcloud

    # Get transform and shape from input
    if grid_coords is not None:

        # Input checks
        if (
            not isinstance(grid_coords, tuple)
            or len(grid_coords) != 2
            or not (isinstance(grid_coords[0], np.ndarray) and grid_coords[0].ndim == 1)
            or not (isinstance(grid_coords[1], np.ndarray) and grid_coords[1].ndim == 1)
        ):
            raise TypeError("Input grid coordinates must be 1D arrays.")
        if len(grid_coords[0]) < 2 or len(grid_coords[1]) < 2:
            raise ValueError("Grid coordinates must contain at least two values along X and Y.")

        diff_x = np.diff(grid_coords[0])
        diff_y = np.diff(grid_coords[1])

        if not np.allclose(diff_x, diff_x[0]) or not np.allclose(diff_y, diff_y[0]):
            raise ValueError("Grid coordinates must be regular (equally spaced, independently along X and Y).")
        if diff_x[0] <= 0 or diff_y[0] <= 0:
            raise ValueError("Grid coordinates must increase along X and Y.")

        # Build transform from min X, max Y and step in both
        out_transform = rio.transform.from_origin(np.min(grid_coords[0]), np.max(grid_coords[1]), diff_x[0], diff_y[0])
        # Y is first axis, X is second axis
        out_shape = (len(grid_coords[1]), len(grid_coords[0]))

    elif transform is not None and shape is not None:

        out_transform = transform
        out_shape = shape

    else:
        raise ValueError("Either grid coordinates or both geotransform and shape must be provided.")

    # Create raster from inputs, with placeholder data for now
    dtype = gdf_pc[data_column_name].dtype
    out_nodata = nodata if nodata is not None else _default_nodata(dtype)
    arr = np.ones(out_shape, dtype=dtype)

    # Get indexes of point cloud coordinates in the raster, forcing no shift
    i, j = _xy2ij(
        x=gdf_pc.geometry.x.values,
        y=gdf_pc.geometry.y.values,
        shift_area_or_point=False,
        transform=out_transform,
        area_or_point=area_or_point,
    )

    # If coordinates are not integer type (forced in xy2ij), then some points are not falling on exact coordinates
    if not np.issubdtype(i.dtype, np.integer) or not np.issubdtype(j.dtype, np.integer):
        raise ValueError("Some point cloud coordinates differ from the grid coordinates.")

    # Reject positions outside the requested grid before NumPy can wrap negative indexes around an array edge
    if np.any(i < 0) or np.any(i >= out_shape[0]) or np.any(j < 0) or np.any(j >= out_shape[1]):
        raise ValueError("Some point cloud coordinates fall outside the grid.")

    # Set values
    mask = np.ones(np.shape(arr), dtype=bool)
    mask[i, j] = False
    arr[i, j] = gdf_pc[data_column_name].values

    # Set output values
    raster_arr = np.ma.masked_array(data=arr, mask=mask)

    return raster_arr, out_transform, gdf_pc.crs, out_nodata, area_or_point


###################################
# 2/ RASTER TO REGULAR POINT CLOUD
###################################


########################
# 2A/ SHARED HELPERS
########################


def _sample_raster_cell_indices(
    source_raster: RasterType,
    data_band: int,
    subsample: float | int,
    skip_nodata: bool,
    random_state: int | np.random.Generator | None,
    mp_config: MultiprocConfig | None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Get row and column indexes of the requested subsample.

    See description of _raster_to_pointcloud_from_indices() for details on the overall logic.
    """

    # We use subsample() only with "topk" to keep full-grid order, nodata handling, and partial sampling consistent
    indices = source_raster.subsample(
        subsample=subsample,
        band=data_band,
        return_indices=True,
        random_state=random_state,
        strategy="topk",
        mp_config=mp_config,
        skip_nodata=skip_nodata,
    )
    if any(is_dask_array(index) for index in indices):
        indices = import_optional("dask").compute(*indices)
    rows, columns = indices
    return np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)


def _extract_raster_cell_values(
    source_raster: RasterType,
    bands: list[int],
    rows: NDArray[np.int64],
    columns: NDArray[np.int64],
    mp_config: MultiprocConfig | None,
) -> Any:
    """
    Read every band at the raster row/columns of the subsample.

    See description of _raster_to_pointcloud_from_indices() for details on the overall logic.
    """

    # Choose how to read the requested subsample from a lazy array, memory, or the raster file
    data = source_raster.data if source_raster.is_loaded or source_raster._is_xr else None

    # Dask
    if data is not None and is_dask_array(data):
        import dask.array as da

        # Select each requested band lazily at the row/column of the subsample
        dask_data: Any = data
        band_values = [
            dask_data.vindex[rows, columns] if dask_data.ndim == 2 else dask_data[band - 1].vindex[rows, columns]
            for band in bands
        ]

        # Join the selected bands while keeping their values lazy
        pixel_data = da.stack(band_values, axis=0)

    # Eager
    elif data is not None:
        # Select each requested band directly from the raster array in memory
        band_values = [data[rows, columns] if data.ndim == 2 else data[band - 1, rows, columns] for band in bands]

        # Respect nodata masks when a band has one
        pixel_data = (
            np.ma.stack(band_values, axis=0)
            if any(np.ma.isMaskedArray(values) for values in band_values)
            else np.stack(band_values, axis=0)
        )

    # Multiproc
    else:
        from geoutils.multiproc import MultiprocConfig
        from geoutils.multiproc.cluster import _map_bounded
        from geoutils.multiproc.readers import _read_selected_raster_bands

        # Prepare small raster tiles for reading the requested cells from the file
        read_config = mp_config if mp_config is not None else MultiprocConfig(chunks=512)

        # Convert row/column to a flat index so that every band reads the same raster cells
        flat_indices = rows * source_raster.shape[1] + columns
        chunk_rows, chunk_columns = (
            (read_config.chunks, read_config.chunks) if isinstance(read_config.chunks, int) else read_config.chunks
        )
        tile_columns = (source_raster.shape[1] + chunk_columns - 1) // chunk_columns
        tile_ids = (rows // chunk_rows) * tile_columns + columns // chunk_columns

        # Group the sample positions once so each selected tile reads every requested band in one worker call
        order = np.argsort(tile_ids, kind="stable")
        boundaries = np.flatnonzero(np.diff(tile_ids[order])) + 1
        positions_by_tile = np.split(order, boundaries) if len(order) > 0 else []
        arguments = (
            (source_raster, flat_indices[positions], bands, read_config.chunks) for positions in positions_by_tile
        )
        dtype = np.dtype(bool if source_raster.is_mask else source_raster.dtype)
        pixel_data = np.ma.masked_all((len(bands), len(flat_indices)), dtype=dtype)
        for positions, (_, values) in zip(
            positions_by_tile,
            _map_bounded(read_config.cluster, _read_selected_raster_bands, arguments),
        ):
            pixel_data.data[:, positions] = np.ma.getdata(values)
            pixel_data.mask[:, positions] = np.ma.getmaskarray(values)

    return pixel_data


def _raster_to_pointcloud_from_indices(
    source_raster: RasterType,
    bands: list[int],
    column_names: list[str],
    data_column_name: str,
    subsample: float | int,
    skip_nodata: bool,
    random_state: int | np.random.Generator | None,
    force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"],
    as_array: bool,
    read_config: MultiprocConfig | None,
) -> Any:
    """
    Build the complete point cloud or X/Y/data array for the subsample.

    All subsample indices and values will at some point concatenate in memory at once.

    Thus, this function is used by:
    - Eager,
    - Dask only for sample sizes smaller than raster chunk size, otherwise it requires different logic explained in
        chunked helpers below),
    - Multiproc for small sample sizes too, but also when as_array=True, because Multiproc cannot stream chunk-by-chunk
        to an array like Dask (only to a point cloud file).
    """

    # 1/ Select indices through subsample()
    rows, columns = _sample_raster_cell_indices(
        source_raster=source_raster,
        data_band=bands[0],  # We can use the same indices for all bands
        subsample=subsample,
        skip_nodata=skip_nodata,
        random_state=random_state,
        mp_config=read_config,
    )

    # 2/ Read every output band at the positions selected from the main band
    pixel_data = _extract_raster_cell_values(
        source_raster=source_raster,
        bands=bands,
        rows=rows,
        columns=columns,
        mp_config=read_config,
    )

    # 3/ Normalize nodata values after cell selection
    if is_dask_array(pixel_data) and skip_nodata:
        import dask.array as da

        pixel_data = da.ma.getdata(pixel_data)
    elif np.ma.isMaskedArray(pixel_data):
        pixel_data = pixel_data.data

    # Convert retained nodata values to NaN in a floating output array
    if not skip_nodata:
        pixel_data = pixel_data.astype(np.result_type(pixel_data.dtype, np.float32))
        if is_dask_array(pixel_data):
            import dask.array as da

            pixel_data = da.ma.filled(pixel_data, np.nan)
            if source_raster.nodata is not None:
                pixel_data = da.where(pixel_data == source_raster.nodata, np.nan, pixel_data)
        elif source_raster.nodata is not None:
            pixel_data[pixel_data == source_raster.nodata] = np.nan

    # 4/ Calculate coordinates from affine transform and pixel offset
    x_coords, y_coords = _ij2xy(
        i=rows,
        j=columns,
        transform=source_raster.transform,
        area_or_point=source_raster.area_or_point,
        shift_area_or_point=False,
        force_offset=force_pixel_offset,
    )

    # 5/ Build output, lazy for Dask, otherwise eager
    if is_dask_array(pixel_data):
        import dask.array as da

        if not is_dask_array(x_coords):
            x_coords = da.from_array(np.asarray(x_coords), chunks=pixel_data.chunks[1])
            y_coords = da.from_array(np.asarray(y_coords), chunks=pixel_data.chunks[1])
        if as_array:
            return da.stack((x_coords, y_coords, *[pixel_data[index] for index in range(len(bands))]), axis=1)

        from geoutils.pointcloud.dataframe import (
            _build_pointcloud_output,
            _import_dask_dataframe,
        )
        from geoutils.vector.pd_accessor import _import_dask_geopandas

        # Build matching Dask series so dataframe values keep their raster dtype and continuous point index
        point_chunks = pixel_data.chunks[1]
        x_coords = x_coords.rechunk(point_chunks)
        y_coords = y_coords.rechunk(point_chunks)
        dask_dataframe = _import_dask_dataframe()
        dask_geopandas = _import_dask_geopandas()
        dataframe = dask_dataframe.from_dask_array(pixel_data.T, columns=column_names)
        coordinate_frame = dask_dataframe.concat(
            [
                dask_dataframe.from_dask_array(x_coords, columns="x"),
                dask_dataframe.from_dask_array(y_coords, columns="y"),
            ],
            axis=1,
        )
        geometry = dask_geopandas.points_from_xy(coordinate_frame, x="x", y="y", crs=source_raster.crs)
        dataframe = dataframe.assign(geometry=geometry)
        dataframe = dask_geopandas.from_dask_dataframe(dataframe, geometry="geometry")

        # Finalize point metadata; the shared builder also adds ``.pc`` and ``.vct`` to this Dask frame
        return _build_pointcloud_output(dataframe, data_column=data_column_name, as_dataframe=True)

    # Build an eager array or PointCloud result
    if as_array:
        return np.vstack((np.asarray(x_coords), np.asarray(y_coords), pixel_data)).T

    from geoutils.pointcloud import PointCloud

    dataframe = gpd.GeoDataFrame(
        pixel_data.T,
        columns=column_names,
        geometry=gpd.points_from_xy(np.asarray(x_coords), np.asarray(y_coords)),
        crs=source_raster.crs,
    )
    return PointCloud(dataframe, data_column=data_column_name)


#################################
# 2B/ CHUNKED-ONLY HELPERS
#################################


def _subsample_exceeds_largest_chunk(subsample: float | int, raster_shape: tuple[int, int], largest_chunk: int) -> bool:
    """
    Check whether the requested subsample contains more points than the largest raster chunk.

    If the subsample size exceed the size of a raster chunk, it cannot concatenate all keys in memory to find the
    subsampled point indices directly, and instead uses the algorithm described in _iterative_topk_cutoff().
    """

    return (0 < subsample <= 1 and subsample * int(np.prod(raster_shape)) > largest_chunk) or subsample > largest_chunk


def _raster_values_to_point_partition(
    band_values: Any,
    selected_indices: NDArray[np.int64],
    raster_shape: tuple[int, int],
    transform: affine.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    crs: CRS | None,
    nodata: int | float | None,
    column_names: list[str],
    skip_nodata: bool,
    force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"],
    topk_selection: tuple[int, np.uint64] | None,
    as_array: bool,
) -> NDArrayNum | gpd.GeoDataFrame:
    """
    Convert the values and global indexes of cells selected in one chunk into point rows.

    Used only when the requested subsample is larger than one chunk.

    The objective is eventually to write each raster chunk subsamples out-of-memory into a point partition of the point
    file output.
    """

    band_values = band_values.reshape((len(column_names), -1))

    # Keep cells with random key at or below the cutoff
    from geoutils.sampling.subsampling import _splitmix64, _valid_subsample_mask

    if topk_selection is not None:
        seed, cutoff = topk_selection
        if skip_nodata:
            eligible = _valid_subsample_mask(band_values[0], skip_nodata=True).reshape(-1)
            selected_indices = selected_indices[eligible]
            band_values = band_values[:, eligible]
        keys = _splitmix64(np.uint64(seed) ^ selected_indices.astype(np.uint64))
        selected = keys <= cutoff
        selected_indices = selected_indices[selected]
        band_values = band_values[:, selected]

    # Remove cells with nodata in the main band, or replace nodata values with NaN
    elif skip_nodata:
        keep = _valid_subsample_mask(band_values[0], skip_nodata=True).reshape(-1)
        selected_indices = selected_indices[keep]
        band_values = band_values[:, keep]

    if skip_nodata:
        band_values = np.ma.getdata(band_values)
    else:
        band_values = np.ma.filled(band_values.astype(np.result_type(band_values.dtype, np.float32)), np.nan)
        if nodata is not None:
            band_values[band_values == nodata] = np.nan

    # Derive point coordinates from the geotransform and pixel interpretation
    rows, columns = np.unravel_index(selected_indices, raster_shape)
    x_coords, y_coords = _ij2xy(
        i=rows,
        j=columns,
        transform=transform,
        area_or_point=area_or_point,
        shift_area_or_point=False,
        force_offset=force_pixel_offset,
    )

    # Return array rows for Dask or a point dataframe for Dask/multiprocessing
    if as_array:
        return np.column_stack((x_coords, y_coords, *band_values))
    return gpd.GeoDataFrame(
        {name: band_values[index] for index, name in enumerate(column_names)},
        geometry=gpd.points_from_xy(x_coords, y_coords),
        crs=crs,
    )


def _raster_chunk_topk_keys(
    array: Any | None,
    tile_idx: NDArrayNum,
    raster_width: int,
    seed: int,
    skip_nodata: bool,
) -> NDArray[np.uint64]:
    """
    Calculate a deterministic random key for every valid cell in a raster chunk, re-using the same logic as in the
    subsampling module "topk" method.

    Used only when the requested subsample is larger than one chunk.
    """

    # Build full raster indexes directly from the chunk row and column offsets
    row_start, row_stop, column_start, column_stop = (int(value) for value in tile_idx)
    tile_width = column_stop - column_start
    row_offsets = np.arange(row_start, row_stop, dtype=np.int64) * np.int64(raster_width) + column_start
    global_indices = (row_offsets[:, None] + np.arange(tile_width, dtype=np.int64)).reshape(-1)

    # Keep only cells allowed by the main raster band
    if skip_nodata:
        if array is None:
            raise RuntimeError("Raster values are required when skip_nodata=True.")
        from geoutils.sampling.subsampling import _valid_subsample_mask

        valid = _valid_subsample_mask(array, skip_nodata=True)
        global_indices = global_indices[valid.reshape(-1)]

    # Calculate repeatable random keys from full raster indexes
    from geoutils.sampling.subsampling import _splitmix64

    return np.asarray(_splitmix64(np.uint64(seed) ^ global_indices.astype(np.uint64)), dtype=np.uint64)


def _topk_key_histogram(keys: NDArray[np.uint64], prefix: int, prefix_bits: int, digit_bits: int) -> NDArray[np.int64]:
    """
    Count keys in the next set of ranges for one chunk of a large requested subsample.

    See description of _iterative_topk_cutoff() for details on the implementation logic.
    """

    # Keep only keys in the range chosen by earlier passes
    if prefix_bits:
        prefix_matches = keys >> np.uint64(64 - prefix_bits) == np.uint64(prefix)
        keys = keys[prefix_matches]

    # Split the remaining keys into smaller ranges and count each range
    shift = 64 - prefix_bits - digit_bits
    digit_mask = np.uint64((1 << digit_bits) - 1)
    digits = ((keys >> np.uint64(shift)) & digit_mask).astype(np.intp)
    return np.bincount(digits, minlength=1 << digit_bits).astype(np.int64, copy=False)


def _topk_prefix_keys(keys: NDArray[np.uint64], prefix: int, prefix_bits: int) -> NDArray[np.uint64]:
    """
    Return keys from one chunk in the range chosen for the requested subsample.

    See description of _iterative_topk_cutoff() for details on the implementation logic.
    """

    prefix_matches = keys >> np.uint64(64 - prefix_bits) == np.uint64(prefix)
    return keys[prefix_matches]


def _topk_histogram_candidates(
    keys: NDArray[np.uint64], digit_bits: int, candidate_prefixes: tuple[int, int]
) -> tuple[NDArray[np.int64], NDArray[np.uint64]]:
    """Count the first key ranges and keep keys from an interval likely to contain the cutoff."""

    histogram = _topk_key_histogram(keys, 0, 0, digit_bits)
    prefix_start, prefix_stop = candidate_prefixes
    prefixes = keys >> np.uint64(64 - digit_bits)
    candidate_mask = (prefixes >= prefix_start) & (prefixes < prefix_stop)
    return histogram, keys[candidate_mask]


def _merge_topk_histogram_candidates(
    parts: list[tuple[NDArray[np.int64], NDArray[np.uint64] | None]], candidate_limit: int
) -> tuple[NDArray[np.int64], NDArray[np.uint64] | None]:
    """Add histograms while keeping no more than one raster chunk of possible cutoff keys."""

    histogram = np.sum([part[0] for part in parts], axis=0, dtype=np.int64)
    candidate_parts = [part[1] for part in parts]
    if any(part is None for part in candidate_parts):
        return histogram, None

    candidates = cast(list[NDArray[np.uint64]], candidate_parts)
    if sum(len(part) for part in candidates) > candidate_limit:
        return histogram, None
    return histogram, np.concatenate(candidates)


def _topk_candidate_prefixes(
    subsample: float | int, total_cells: int, largest_chunk: int, digit_bits: int
) -> tuple[int, int]:
    """Choose a bounded key interval likely to contain the requested sample cutoff."""

    prefix_count = 1 << digit_bits

    # Cache ranges expected to hold three quarters of one chunk, leaving room for uneven key counts
    cached_prefix_count = max(1, min(prefix_count, largest_chunk * prefix_count * 3 // (total_cells * 4)))
    if 0 < subsample <= 1:
        # Center fractional samples on their expected key quantile
        predicted_prefix = min(prefix_count - 1, int(subsample * prefix_count))
        lower_prefixes = cached_prefix_count // 2
    else:
        # Reserve most prefixes above the all-valid estimate because missing cells can move the cutoff upward
        requested_size = max(1, min(int(subsample), total_cells))
        predicted_prefix = (requested_size - 1) * prefix_count // total_cells
        lower_prefixes = max(1, cached_prefix_count // 8)

    prefix_start = max(0, predicted_prefix - lower_prefixes)
    prefix_stop = min(prefix_count, prefix_start + cached_prefix_count)
    return max(0, prefix_stop - cached_prefix_count), prefix_stop


def _iterative_topk_cutoff(
    subsample: float | int,
    random_state: int | np.random.Generator | None,
    largest_chunk: int,
    total_cells: int,
    number_chunks: int,
    histogram_for_prefix: Callable[
        [int, int, int, int, tuple[int, int] | None],
        tuple[NDArray[np.int64], NDArray[np.uint64] | None],
    ],
    keys_for_prefix: Callable[[int, int, int, int], NDArray[np.uint64]],
) -> tuple[int, int, np.uint64 | None]:
    """
    Find the cutoff separating cells in/out of the subsample without loading all indexes in memory.

    Dask and multiprocessing use this method only when the sample size exceeds the largest raster chunk.

    This function finds the "k" value that separates the "topk" samples kept for the subsample,
    but in a chunk-by-chunk manner for cases where the subsample itself is very large (e.g. 80% of the raster).
    This requires several iterations to converge towards the right value.

    The cutoff algorithm follows these steps:
    1. Start with the full unsigned 64-bit key interval (0 through 2**64 - 1), split it into subranges, and count,
       across all chunks, how many keys fall in each subrange.
    2. Use the cumulative counts and requested sample size to identify the subrange containing the cutoff, and discard
       the other subranges.
    3. Repeat the count within that range until it contains no more keys than the largest raster chunk.
    4. Collect the remaining keys and select the exact cutoff value.

    References
    ----------
    - NIST Dictionary of Algorithms and Data Structures, "Selection problem"
      https://xlinux.nist.gov/dads/HTML/selectkth.html
    - Alabi et al., "Fast k-selection algorithms for graphics processing units", Journal of Experimental
      Algorithmics 17, 2012. https://doi.org/10.1145/2133803.2345676
    """

    from geoutils.sampling.subsampling import _get_subsample_size_from_user_input

    # Convert the random state once so every pass calculates the same key for each raster cell
    if isinstance(random_state, np.random.Generator):
        seed = int(random_state.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    elif random_state is None:
        seed = 0
    else:
        seed = int(random_state)

    # Keep count arrays small while using more ranges when the raster contains many chunks
    digit_bits = min(12, max(8, (number_chunks - 1).bit_length()))
    prefix = 0
    prefix_bits = 0
    rank = -1
    sample_size = 0
    candidate_cache: tuple[int, int, int, NDArray[np.uint64]] | None = None
    while prefix_bits < 64:
        current_digit_bits = min(digit_bits, 64 - prefix_bits)
        candidate_prefixes = None
        if prefix_bits == 0:
            candidate_prefixes = _topk_candidate_prefixes(subsample, total_cells, largest_chunk, current_digit_bits)
        histogram, candidate_keys = histogram_for_prefix(
            seed, prefix, prefix_bits, current_digit_bits, candidate_prefixes
        )
        if candidate_prefixes is not None and candidate_keys is not None:
            candidate_cache = (*candidate_prefixes, current_digit_bits, candidate_keys)

        # Use the first set of counts to calculate the requested number of points
        if prefix_bits == 0:
            sample_size = _get_subsample_size_from_user_input(subsample, int(histogram.sum()))
            if sample_size == 0:
                return 0, seed, None
            rank = sample_size - 1

        # Find the key range containing the last selected point (subsample size)
        cumulative_counts = np.cumsum(histogram)
        digit = int(np.searchsorted(cumulative_counts, rank, side="right"))
        preceding_count = 0 if digit == 0 else int(cumulative_counts[digit - 1])
        rank -= preceding_count
        prefix = (prefix << current_digit_bits) | digit
        prefix_bits += current_digit_bits
        group_size = int(histogram[digit])

        # Find the exact cutoff from no more than one chunk of keys
        if group_size <= largest_chunk:
            if prefix_bits == 64:
                return sample_size, seed, np.uint64(prefix)
            group_keys = None
            if candidate_cache is not None:
                cached_prefix_start, cached_prefix_stop, cached_prefix_bits, cached_keys = candidate_cache
                if cached_prefix_bits == prefix_bits and cached_prefix_start <= prefix < cached_prefix_stop:
                    cached_group_keys = _topk_prefix_keys(cached_keys, prefix, prefix_bits)
                    if len(cached_group_keys) == group_size:
                        group_keys = cached_group_keys
            if group_keys is None:
                group_keys = keys_for_prefix(seed, prefix, prefix_bits, group_size)
            if len(group_keys) != group_size:
                raise RuntimeError("The number of random keys changed while finding the subsample cutoff.")
            group_keys.partition(rank)
            return sample_size, seed, group_keys[rank]

    raise RuntimeError("Could not find the subsample cutoff.")


#########################################
# 2C/ EAGER RASTER TO POINT CLOUD
#########################################


def _eager_raster_to_pointcloud(
    source_raster: RasterType,
    bands: list[int],
    column_names: list[str],
    data_column_name: str,
    subsample: float | int,
    skip_nodata: bool,
    random_state: int | np.random.Generator | None,
    force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"],
    as_array: bool,
) -> Any:
    """
    Build an eager point output, reading an unloaded partial raster in bounded tiles.

    If the entire raster should be converted (subsample == 1), it is loaded.
    For a smaller requested subsample, values are read in small parts without loading the source raster.
     (mirroring Xarray.isel() default behaviour)
    """

    read_config = None
    if not source_raster.is_loaded and not source_raster._is_xr:
        if subsample == 1:
            source_raster.load()
        else:
            from geoutils.multiproc import MultiprocConfig

            read_config = MultiprocConfig(chunks=512)

    return _raster_to_pointcloud_from_indices(
        source_raster,
        bands,
        column_names,
        data_column_name,
        subsample,
        skip_nodata,
        random_state,
        force_pixel_offset,
        as_array,
        read_config,
    )


########################################
# 2D/ DASK RASTER TO POINT CLOUD
########################################


def _wrapper_raster_topk_histogram_dask(
    array: Any | None,
    tile_idx: NDArrayNum,
    raster_width: int,
    seed: int,
    prefix: int,
    prefix_bits: int,
    digit_bits: int,
    skip_nodata: bool,
) -> NDArray[np.int64]:
    """Count random keys in one Dask chunk during one pass over the requested subsample."""

    keys = _raster_chunk_topk_keys(array, tile_idx, raster_width, seed, skip_nodata)
    return _topk_key_histogram(keys, prefix, prefix_bits, digit_bits)


def _wrapper_raster_topk_histogram_candidates_dask(
    array: Any | None,
    tile_idx: NDArrayNum,
    raster_width: int,
    seed: int,
    digit_bits: int,
    candidate_prefixes: tuple[int, int],
    skip_nodata: bool,
) -> tuple[NDArray[np.int64], NDArray[np.uint64]]:
    """Count first-pass key ranges and return possible cutoff keys from one Dask chunk."""

    keys = _raster_chunk_topk_keys(array, tile_idx, raster_width, seed, skip_nodata)
    return _topk_histogram_candidates(keys, digit_bits, candidate_prefixes)


def _wrapper_raster_topk_prefix_keys_dask(
    array: Any | None,
    tile_idx: NDArrayNum,
    raster_width: int,
    seed: int,
    prefix: int,
    prefix_bits: int,
    skip_nodata: bool,
) -> NDArray[np.uint64]:
    """Return keys in the range chosen for the requested subsample from one Dask chunk."""

    keys = _raster_chunk_topk_keys(array, tile_idx, raster_width, seed, skip_nodata)
    return _topk_prefix_keys(keys, prefix, prefix_bits)


def _sum_topk_histograms(histograms: list[NDArray[np.int64]]) -> NDArray[np.int64]:
    """Add key counts for the requested subsample from a group of up to eight Dask chunks."""

    return np.sum(histograms, axis=0, dtype=np.int64)


def _dask_raster_topk_cutoff(
    main_blocks: list[Any],
    tiles: NDArrayNum,
    raster_shape: tuple[int, int],
    largest_chunk: int,
    subsample: float | int,
    skip_nodata: bool,
    random_state: int | np.random.Generator | None,
) -> tuple[int, int, np.uint64 | None]:
    """
    Find the key cutoff for the subsample, with Dask delayed operations.

    Only used for a subsample size larger than a raster chunk.
    """

    dask = import_optional("dask")
    delayed = dask.delayed

    def histogram_for_prefix(
        seed: int,
        prefix: int,
        prefix_bits: int,
        digit_bits: int,
        candidate_prefixes: tuple[int, int] | None,
    ) -> tuple[NDArray[np.int64], NDArray[np.uint64] | None]:
        """Count keys for the requested subsample in every Dask chunk and add the results."""

        # Keep a bounded interval around the expected cutoff during the first raster pass
        if candidate_prefixes is not None:
            histogram_candidates = [
                delayed(_wrapper_raster_topk_histogram_candidates_dask)(
                    block if skip_nodata else None,
                    tile,
                    raster_shape[1],
                    seed,
                    digit_bits,
                    candidate_prefixes,
                    skip_nodata,
                )
                for block, tile in zip(main_blocks, tiles)
            ]
            while len(histogram_candidates) > 1:
                histogram_candidates = [
                    delayed(_merge_topk_histogram_candidates)(histogram_candidates[start : start + 2], largest_chunk)
                    for start in range(0, len(histogram_candidates), 2)
                ]
            return cast(
                tuple[NDArray[np.int64], NDArray[np.uint64] | None],
                dask.compute(histogram_candidates[0])[0],
            )

        histograms = [
            delayed(_wrapper_raster_topk_histogram_dask)(
                block if skip_nodata else None,
                tile,
                raster_shape[1],
                seed,
                prefix,
                prefix_bits,
                digit_bits,
                skip_nodata,
            )
            for block, tile in zip(main_blocks, tiles)
        ]
        while len(histograms) > 1:
            histograms = [
                delayed(_sum_topk_histograms)(histograms[start : start + 8]) for start in range(0, len(histograms), 8)
            ]
        return cast(NDArray[np.int64], dask.compute(histograms[0])[0]), None

    def keys_for_prefix(seed: int, prefix: int, prefix_bits: int, group_size: int) -> NDArray[np.uint64]:
        """Collect keys in the last range chosen for the requested subsample from every Dask chunk."""

        key_parts = [
            delayed(_wrapper_raster_topk_prefix_keys_dask)(
                block if skip_nodata else None,
                tile,
                raster_shape[1],
                seed,
                prefix,
                prefix_bits,
                skip_nodata,
            )
            for block, tile in zip(main_blocks, tiles)
        ]
        group_keys = np.empty(group_size, dtype=np.uint64)
        offset = 0
        for chunk_keys in dask.compute(*key_parts):
            stop = offset + len(chunk_keys)
            group_keys[offset:stop] = chunk_keys
            offset = stop
        return group_keys[:offset]

    return _iterative_topk_cutoff(
        subsample,
        random_state,
        largest_chunk,
        int(np.prod(raster_shape)),
        len(tiles),
        histogram_for_prefix,
        keys_for_prefix,
    )


def _wrapper_raster_to_pointcloud_partition_dask(
    band_values: Any,
    tile_idx: NDArrayNum,
    raster_shape: tuple[int, int],
    transform: affine.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    crs: CRS | None,
    nodata: int | float | None,
    column_names: list[str],
    skip_nodata: bool,
    force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"],
    topk_selection: tuple[int, np.uint64] | None,
    as_array: bool,
) -> NDArrayNum | gpd.GeoDataFrame:
    """
    Convert one Dask raster chunk into one lazy part of the point result.

    Only used for a subsample size larger than a raster chunk.
    """

    # Build the global cell indexes covered by this raster chunk
    row_slice = slice(int(tile_idx[0]), int(tile_idx[1]))
    column_slice = slice(int(tile_idx[2]), int(tile_idx[3]))
    rows = np.arange(row_slice.start, row_slice.stop, dtype=np.int64)[:, None]
    columns = np.arange(column_slice.start, column_slice.stop, dtype=np.int64)[None, :]
    selected_indices = (rows * raster_shape[1] + columns).ravel()

    return _raster_values_to_point_partition(
        band_values,
        selected_indices,
        raster_shape,
        transform,
        area_or_point,
        crs,
        nodata,
        column_names,
        skip_nodata,
        force_pixel_offset,
        topk_selection,
        as_array,
    )


def _build_dask_pointcloud_partitions(
    parts: list[Any],
    column_names: list[str],
    column_dtype: DTypeLike,
    crs: CRS | None,
    data_column_name: str,
) -> Any:
    """
    Build one lazy point dataframe from the point rows produced for each Dask chunk.

    Only used for a subsample size larger than a raster chunk.
    """

    from geoutils.pointcloud.dataframe import (
        _build_pointcloud_output,
        _import_dask_dataframe,
    )
    from geoutils.vector.pd_accessor import _import_dask_geopandas

    empty_frame = gpd.GeoDataFrame(
        {name: np.empty(0, dtype=column_dtype) for name in column_names},
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )
    dask_dataframe = _import_dask_dataframe()
    dataframe = (
        dask_dataframe.from_delayed(parts, meta=empty_frame)
        if parts
        else dask_dataframe.from_pandas(empty_frame, npartitions=1)
    )
    dataframe = _import_dask_geopandas().from_dask_dataframe(dataframe, geometry="geometry")
    return _build_pointcloud_output(dataframe, data_column=data_column_name, as_dataframe=True)


def _dask_raster_to_pointcloud(
    source_raster: RasterType,
    bands: list[int],
    column_names: list[str],
    data_column_name: str,
    subsample: float | int,
    skip_nodata: bool,
    random_state: int | np.random.Generator | None,
    force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"],
    as_array: bool,
) -> Any:
    """
    Build a lazy Dask point output from raster chunks.

    If subsample size is smaller than a raster chunk, we use subsample() directly through
    _raster_to_pointcloud_from_indices().

    If subsample size is larger than a raster chunk, we use _dask_raster_topk_cutoff() to find the subsample
    indices without loading more than a single raster chunk, then build the output with
    _build_dask_pointcloud_partitions().
    """

    dask = import_optional("dask")
    import dask.array as da

    data: Any = source_raster.data
    main_data = data if data.ndim == 2 else data[bands[0] - 1]
    row_chunks, column_chunks = main_data.chunks
    row_starts = np.cumsum((0, *row_chunks))
    column_starts = np.cumsum((0, *column_chunks))
    tiles = np.array(
        [
            (row_starts[row], row_starts[row + 1], column_starts[column], column_starts[column + 1])
            for row in range(len(row_chunks))
            for column in range(len(column_chunks))
        ],
        dtype=np.int64,
    )
    largest_chunk = max(int(rows * columns) for rows in row_chunks for columns in column_chunks)

    # Collect all indexes in memory when the subsample has no more points than the largest raster chunk
    if subsample != 1 and not _subsample_exceeds_largest_chunk(subsample, source_raster.shape, largest_chunk):
        return _raster_to_pointcloud_from_indices(
            source_raster,
            bands,
            column_names,
            data_column_name,
            subsample,
            skip_nodata,
            random_state,
            force_pixel_offset,
            as_array,
            read_config=None,
        )

    # Otherwise, find the key cutoff for the subsample, and write partition by partition to a point cloud file
    topk_selection = None
    if subsample != 1:
        main_blocks = main_data.to_delayed().ravel().tolist()
        sample_size, seed, cutoff = _dask_raster_topk_cutoff(
            main_blocks,
            tiles,
            source_raster.shape,
            largest_chunk,
            subsample,
            skip_nodata,
            random_state,
        )
        if sample_size == 0:
            empty = np.empty((0, 2 + len(bands)), dtype=np.result_type(np.float64, source_raster.dtype))
            if as_array:
                return da.from_array(empty, chunks=empty.shape)
            return _build_dask_pointcloud_partitions(
                [],
                column_names,
                source_raster.dtype,
                source_raster.crs,
                data_column_name,
            )
        assert cutoff is not None
        topk_selection = (seed, cutoff)

    # Select input bands in one block per raster chunk
    band_data = data[None, ...] if data.ndim == 2 else data[[band - 1 for band in bands]]
    band_data = band_data.rechunk({0: len(bands)})
    band_blocks = band_data.to_delayed()[0].ravel().tolist()
    parts = [
        dask.delayed(_wrapper_raster_to_pointcloud_partition_dask)(
            block,
            tile,
            source_raster.shape,
            source_raster.transform,
            source_raster.area_or_point,
            source_raster.crs,
            source_raster.nodata,
            column_names,
            skip_nodata,
            force_pixel_offset,
            topk_selection,
            as_array,
        )
        for block, tile in zip(band_blocks, tiles)
    ]

    # Assemble rows lazily
    if as_array:
        dtype = np.result_type(np.float64, source_raster.dtype)
        arrays = [da.from_delayed(part, shape=(np.nan, 2 + len(bands)), dtype=dtype) for part in parts]
        return da.concatenate(arrays, axis=0)

    column_dtype = np.result_type(source_raster.dtype, np.float32) if not skip_nodata else source_raster.dtype
    return _build_dask_pointcloud_partitions(
        parts,
        column_names,
        column_dtype,
        source_raster.crs,
        data_column_name,
    )


###################################################
# 2E/ MULTIPROCESSING RASTER TO POINT CLOUD
###################################################


def _raster_tile_topk_keys(
    source_raster: RasterType,
    tile_idx: NDArrayNum,
    seed: int,
    band: int,
    skip_nodata: bool,
) -> NDArray[np.uint64]:
    """Calculate random keys for usable cells in one multiprocessing tile of a large requested subsample."""

    # Read the main band only when its values determine which cells may be selected
    array = None
    if skip_nodata:
        from geoutils.sampling.subsampling import _read_subsample_raster_block

        array, _ = _read_subsample_raster_block(source_raster, tile_idx, band=band, skip_nodata=True)

    return _raster_chunk_topk_keys(array, tile_idx, source_raster.shape[1], seed, skip_nodata)


def _wrapper_raster_topk_histogram_mp(
    source_raster: RasterType,
    tile_idx: NDArrayNum,
    seed: int,
    prefix: int,
    prefix_bits: int,
    digit_bits: int,
    band: int,
    skip_nodata: bool,
) -> NDArray[np.int64]:
    """Count random keys in one tile during one pass over the requested subsample."""

    # Calculate keys only for cells that may appear in the output
    keys = _raster_tile_topk_keys(source_raster, tile_idx, seed, band, skip_nodata)
    return _topk_key_histogram(keys, prefix, prefix_bits, digit_bits)


def _wrapper_raster_topk_histogram_candidates_mp(
    source_raster: RasterType,
    tile_idx: NDArrayNum,
    seed: int,
    digit_bits: int,
    candidate_prefixes: tuple[int, int],
    band: int,
    skip_nodata: bool,
) -> tuple[NDArray[np.int64], NDArray[np.uint64]]:
    """Count first-pass key ranges and return possible cutoff keys from one multiprocessing tile."""

    keys = _raster_tile_topk_keys(source_raster, tile_idx, seed, band, skip_nodata)
    return _topk_histogram_candidates(keys, digit_bits, candidate_prefixes)


def _wrapper_raster_topk_prefix_keys_mp(
    source_raster: RasterType,
    tile_idx: NDArrayNum,
    seed: int,
    prefix: int,
    prefix_bits: int,
    band: int,
    skip_nodata: bool,
) -> NDArray[np.uint64]:
    """Return keys in the range chosen for the requested subsample from one multiprocessing tile."""

    # Calculate keys only for cells that may appear in the output
    keys = _raster_tile_topk_keys(source_raster, tile_idx, seed, band, skip_nodata)
    return _topk_prefix_keys(keys, prefix, prefix_bits)


def _multiproc_raster_topk_cutoff(
    source_raster: RasterType,
    tiles: NDArrayNum,
    largest_chunk: int,
    subsample: float | int,
    band: int,
    skip_nodata: bool,
    random_state: int | np.random.Generator | None,
    mp_config: MultiprocConfig,
) -> tuple[int, int, np.uint64 | None]:
    """
    Find the key cutoff for the subsample, with Multiproc per-chunk operations.

    Only used for a subsample size larger than a raster chunk.
    """

    from geoutils.multiproc.cluster import _map_bounded

    def histogram_for_prefix(
        seed: int,
        prefix: int,
        prefix_bits: int,
        digit_bits: int,
        candidate_prefixes: tuple[int, int] | None,
    ) -> tuple[NDArray[np.int64], NDArray[np.uint64] | None]:
        """Count keys for the requested subsample in every multiprocessing tile and add the results."""

        # Collect the bounded first-pass cutoff candidates selected by the shared algorithm
        if candidate_prefixes is not None:
            candidate_arguments = (
                (
                    source_raster,
                    tile,
                    seed,
                    digit_bits,
                    candidate_prefixes,
                    band,
                    skip_nodata,
                )
                for tile in tiles
            )
            histogram_candidates = [
                result
                for _, result in _map_bounded(
                    mp_config.cluster, _wrapper_raster_topk_histogram_candidates_mp, candidate_arguments
                )
            ]
            while len(histogram_candidates) > 1:
                histogram_candidates = [
                    _merge_topk_histogram_candidates(histogram_candidates[start : start + 2], largest_chunk)
                    for start in range(0, len(histogram_candidates), 2)
                ]
            return histogram_candidates[0]

        histogram_arguments = (
            (
                source_raster,
                tile,
                seed,
                prefix,
                prefix_bits,
                digit_bits,
                band,
                skip_nodata,
            )
            for tile in tiles
        )
        histogram = np.zeros(1 << digit_bits, dtype=np.int64)
        for _, tile_histogram in _map_bounded(
            mp_config.cluster, _wrapper_raster_topk_histogram_mp, histogram_arguments
        ):
            histogram += tile_histogram
        return histogram, None

    def keys_for_prefix(seed: int, prefix: int, prefix_bits: int, group_size: int) -> NDArray[np.uint64]:
        """Collect keys in the last range chosen for the requested subsample from every multiprocessing tile."""

        arguments = ((source_raster, tile, seed, prefix, prefix_bits, band, skip_nodata) for tile in tiles)
        group_keys = np.empty(group_size, dtype=np.uint64)
        offset = 0
        for _, tile_keys in _map_bounded(mp_config.cluster, _wrapper_raster_topk_prefix_keys_mp, arguments):
            stop = offset + len(tile_keys)
            group_keys[offset:stop] = tile_keys
            offset = stop
        return group_keys[:offset]

    return _iterative_topk_cutoff(
        subsample,
        random_state,
        largest_chunk,
        int(np.prod(source_raster.shape)),
        len(tiles),
        histogram_for_prefix,
        keys_for_prefix,
    )


def _wrapper_raster_to_pointcloud_partition_mp(
    source_raster: RasterType,
    flat_indices: NDArray[np.int64] | tuple[slice, slice],
    bands: list[int],
    column_names: list[str],
    skip_nodata: bool,
    force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"],
    chunks: int | tuple[int, int],
    filename: pathlib.Path,
    topk_selection: tuple[int, np.uint64] | None = None,
) -> pathlib.Path:
    """
    Convert cells from one raster tile in a multiprocessing worker and save its point rows.

    Every multiprocessing point cloud saved to a file uses this function.

    A complete conversion passes a tile slice, a subsample no larger than the biggest tile passes the indexes of its selected raster cells, and a larger
    subsample passes a tile slice with the key separating selected and unselected cells. Processing one tile per call
    limits raster reads and point construction to that tile. Multiprocessing array output collects the indexes of all
    raster cells selected for the requested subsample instead.
    """

    # 1/ Build indexes for the complete tile, or keep its sampled cell indexes
    if isinstance(flat_indices, tuple):
        row_slice, column_slice = flat_indices
        rows = np.arange(row_slice.start, row_slice.stop, dtype=np.int64)[:, None]
        columns = np.arange(column_slice.start, column_slice.stop, dtype=np.int64)[None, :]
        selected_indices = (rows * source_raster.shape[1] + columns).ravel()
    else:
        selected_indices = np.asarray(flat_indices, dtype=np.int64)

    from geoutils.multiproc.readers import _read_selected_raster_bands

    selected_main_values = None
    if topk_selection is not None:
        seed, cutoff = topk_selection
        if skip_nodata:
            from geoutils.sampling.subsampling import _valid_subsample_mask

            selected_main_values = _read_selected_raster_bands(source_raster, selected_indices, bands[:1], chunks)
            eligible = _valid_subsample_mask(selected_main_values[0], skip_nodata=True).reshape(-1)
            selected_indices = selected_indices[eligible]
            selected_main_values = selected_main_values[:, eligible]

        from geoutils.sampling.subsampling import _splitmix64

        keys = _splitmix64(np.uint64(seed) ^ selected_indices.astype(np.uint64))
        selected = keys <= cutoff
        selected_indices = selected_indices[selected]
        if selected_main_values is not None:
            selected_main_values = selected_main_values[:, selected]
        topk_selection = None

    # 2/ Read every requested band from small raster tiles
    if selected_main_values is None:
        band_values = _read_selected_raster_bands(source_raster, selected_indices, bands, chunks)
    elif len(bands) == 1:
        band_values = selected_main_values
    else:
        auxiliary_values = _read_selected_raster_bands(source_raster, selected_indices, bands[1:], chunks)
        band_values = np.ma.concatenate((selected_main_values, auxiliary_values), axis=0)

    # 3/ Convert the selected raster values and positions to one point partition
    dataframe = cast(
        gpd.GeoDataFrame,
        _raster_values_to_point_partition(
            band_values,
            selected_indices,
            source_raster.shape,
            source_raster.transform,
            source_raster.area_or_point,
            source_raster.crs,
            source_raster.nodata,
            column_names,
            skip_nodata,
            force_pixel_offset,
            topk_selection,
            as_array=False,
        ),
    )

    # 4/ Save the point partition to a temporary file
    from geoutils.pointcloud.writing import _stage_pointcloud_partition

    return _stage_pointcloud_partition(dataframe, filename)


def _multiproc_raster_to_pointcloud(
    source_raster: RasterType,
    bands: list[int],
    column_names: list[str],
    data_column_name: str,
    subsample: float | int,
    skip_nodata: bool,
    random_state: int | np.random.Generator | None,
    force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"],
    mp_config: MultiprocConfig,
    as_array: bool,
) -> Any:
    """
    Build a Multiproc point output from raster chunks.

    If subsample size is smaller than a raster chunk or as_array=True, we use subsample() directly through
    _raster_to_pointcloud_from_indices().

    If subsample size is larger than a raster chunk, we use _multiproc_raster_topk_cutoff() to find the subsample
    indices without loading more than a single raster chunk, then build the output chunk by chunk with
    _write_pointcloud_partitions().
    """

    if as_array:
        # Use loaded values directly instead of sending the complete in-memory raster to worker tasks
        read_config = None if source_raster.is_loaded else mp_config
        return _raster_to_pointcloud_from_indices(
            source_raster,
            bands,
            column_names,
            data_column_name,
            subsample,
            skip_nodata,
            random_state,
            force_pixel_offset,
            as_array=True,
            read_config=read_config,
        )

    from geoutils.multiproc.cluster import _map_bounded
    from geoutils.pointcloud.writing import (
        _resolve_pointcloud_output,
        _stage_pointcloud_partition,
        _write_pointcloud_partitions,
    )

    # Raster conversion currently preserves its two-dimensional geometry and named band columns through GeoPackage
    output_filename, driver = _resolve_pointcloud_output(
        mp_config.outfile,
        mp_config.driver,
        supported_drivers=("GPKG",),
        operation_name="raster to point cloud conversion",
    )
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".geoutils-raster-points-", dir=output_filename.parent) as directory:
        temporary_directory = pathlib.Path(directory)

        # Give workers an unloaded file even when the caller supplied an in-memory raster or Xarray accessor
        worker_source = source_raster
        if source_raster.is_loaded or source_raster.name is None:
            from geoutils.raster import Raster

            source_filename = temporary_directory / "source.tif"
            source_raster.to_file(source_filename)
            worker_source = Raster(source_filename, load_data=False)

        from geoutils.multiproc import compute_tiling

        tiling = compute_tiling(mp_config.chunks, worker_source.shape, overlap=0)
        tiles = tiling.reshape((-1, 4))
        largest_chunk = max(int((tile[1] - tile[0]) * (tile[3] - tile[2])) for tile in tiles)
        sample_exceeds_largest_chunk = _subsample_exceeds_largest_chunk(subsample, worker_source.shape, largest_chunk)

        # Materialize a bounded sample, save it once, and return an unloaded wrapper for the requested output
        if not sample_exceeds_largest_chunk:
            pointcloud = _raster_to_pointcloud_from_indices(
                worker_source,
                bands,
                column_names,
                data_column_name,
                subsample,
                skip_nodata,
                random_state,
                force_pixel_offset,
                as_array=False,
                read_config=mp_config,
            )
            partition_filenames: Iterable[pathlib.Path] = (
                _stage_pointcloud_partition(pointcloud.ds, temporary_directory / "partition.pkl"),
            )

        # Split complete conversions by raster tile
        elif subsample == 1:
            selected_parts: Iterable[
                tuple[int, NDArray[np.int64] | tuple[slice, slice], tuple[int, np.uint64] | None]
            ] = (
                (
                    tile_id,
                    (slice(int(tile[0]), int(tile[1])), slice(int(tile[2]), int(tile[3]))),
                    None,
                )
                for tile_id, tile in enumerate(tiles)
            )

        # Find cutoff key out-of-memory when subsample size exceeds one raster chunk size
        else:
            sample_size, seed, cutoff = _multiproc_raster_topk_cutoff(
                source_raster=worker_source,
                tiles=tiles,
                largest_chunk=largest_chunk,
                subsample=subsample,
                band=bands[0],
                skip_nodata=skip_nodata,
                random_state=random_state,
                mp_config=mp_config,
            )
            if sample_size == 0:
                selected_parts = ((0, np.empty(0, dtype=np.int64), None),)
            else:
                assert cutoff is not None
                selected_parts = (
                    (
                        tile_id,
                        (slice(int(tile[0]), int(tile[1])), slice(int(tile[2]), int(tile[3]))),
                        (seed, cutoff),
                    )
                    for tile_id, tile in enumerate(tiles)
                )

        if sample_exceeds_largest_chunk:
            arguments = (
                (
                    worker_source,
                    selected,
                    bands,
                    column_names,
                    skip_nodata,
                    force_pixel_offset,
                    mp_config.chunks,
                    temporary_directory / f"partition_{tile_id}.pkl",
                    topk_selection,
                )
                for tile_id, selected, topk_selection in selected_parts
            )
            partition_filenames = (
                filename
                for _, filename in _map_bounded(
                    mp_config.cluster, _wrapper_raster_to_pointcloud_partition_mp, arguments
                )
            )

        # Finally, we assemble the final file and return it as an unloaded PointCloud!
        return _write_pointcloud_partitions(
            output_filename,
            partition_filenames,
            driver=driver,
            data_column=data_column_name,
            geometry_type="Point",
            mp_config=mp_config,
        )


##########################################
# 2F/ RASTER TO POINT CLOUD PARENT
##########################################


def _raster_to_pointcloud(
    source_raster: RasterType,
    data_column_name: str = "b1",
    data_band: int = 1,
    auxiliary_data_bands: Iterable[int] | None = None,
    auxiliary_column_names: Iterable[str] | None = None,
    subsample: float | int = 1,
    skip_nodata: bool = True,
    as_array: bool = False,
    random_state: int | np.random.Generator | None = None,
    force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"] = "ul",
    mp_config: MultiprocConfig | None = None,
) -> Any:
    """
    Convert raster to a point cloud with eager, Dask, or multiprocessing implementation.

    See RasterBase.to_pointcloud() for details on the arguments.

    Internally, this function checks user inputs, then passes on to:
    - _eager_raster_to_pointcloud() for a in-memory input,
    - _dask_raster_to_pointcloud() for a Dask input, reading chunk by chunk and returning a lazy Dask array or
        GeoDataFrame, optionally written chunk-by-chunk as well,
    - _multiproc_raster_to_pointcloud() for a Multiproc input, reading the raster chunk by chunk and writing to a point
    cloud file, optionally chunk-by-chunk too.

    Without subsampling, the array is reshaped chunk-by-chunk to a lazy Dask object or point cloud file.

    With subsampling, the "topk" subsampling method is used as it is deterministic, and we read the raster
    chunk-by-chunk as in raster.subsample().
    Then, depending on subsampling size, two scenarios are triggered:
    - For a subsample size smaller than one raster chunk, the input reading happens chunk-by-chunk, but the output
        subsample is computed all once in memory (whether eagerly for MP, or computed lazily for Dask).
    - For a subsample size larger than one raster chunk, both Dask/Multiproc implementations call an
        iterative algorithm to find the k cutoff of the "topk" algorithm without loading the equivalent of the subsample
        size in memory. Then, the output subsampled points are written chunk-by-chunk to file/lazy Dask objects.

    Altogether, this ensures that never more than a multiple of raster input chunksize is loaded or returned at once!
    """

    # 1/ Input checks

    # Main data column checks
    if not isinstance(data_column_name, str):
        raise ValueError("Data column name must be a string.")
    if not (isinstance(data_band, int) and data_band >= 1 and data_band <= source_raster.count):
        raise ValueError(
            f"Data band number must be an integer between 1 and the total number of bands ({source_raster.count})."
        )

    # Rename data column if a different band is selected but the name is still default
    if data_band != 1 and data_column_name == "b1":
        data_column_name = "b" + str(data_band)

    # Auxiliary data columns checks
    if auxiliary_column_names is not None and auxiliary_data_bands is None:
        raise ValueError("Passing auxiliary column names requires passing auxiliary data band numbers as well.")
    if auxiliary_data_bands is not None:
        if not isinstance(auxiliary_data_bands, Iterable):
            raise ValueError("Auxiliary data band number must be an iterable containing only integers.")
        auxiliary_data_bands = list(auxiliary_data_bands)
        if not all(isinstance(b, int) for b in auxiliary_data_bands):
            raise ValueError("Auxiliary data band number must be an iterable containing only integers.")
        if any((1 > b or source_raster.count < b) for b in auxiliary_data_bands):
            raise ValueError(
                f"Auxiliary data band numbers must be between 1 and the total number of bands ({source_raster.count})."
            )
        if data_band in auxiliary_data_bands:
            raise ValueError(
                f"Main data band {data_band} should not be listed in auxiliary data bands {auxiliary_data_bands}."
            )

        # Ensure auxiliary column name is defined if auxiliary data bands is not None
        if auxiliary_column_names is not None:
            if not isinstance(auxiliary_column_names, Iterable) or isinstance(auxiliary_column_names, (str, bytes)):
                raise ValueError("Auxiliary column names must be an iterable containing only strings.")
            auxiliary_column_names = list(auxiliary_column_names)
            if not all(isinstance(b, str) for b in auxiliary_column_names):
                raise ValueError("Auxiliary column names must be an iterable containing only strings.")
            if not len(auxiliary_column_names) == len(auxiliary_data_bands):
                raise ValueError(
                    f"Length of auxiliary column name and data band numbers should be the same, "
                    f"found {len(auxiliary_column_names)} and {len(auxiliary_data_bands)} respectively."
                )

        else:
            auxiliary_column_names = [f"b{i}" for i in auxiliary_data_bands]

        # Define bigger list with all bands and names
        all_bands = [data_band] + auxiliary_data_bands
        all_column_names = [data_column_name] + auxiliary_column_names

    else:
        all_bands = [data_band]
        all_column_names = [data_column_name]

    if len(set(all_column_names)) != len(all_column_names) or "geometry" in all_column_names:
        raise ValueError("Point cloud data column names must be unique and cannot be 'geometry'.")

    # 2/ Validate execution backend and point coordinate convention

    # One operation cannot be scheduled by Dask and multiprocessing at the same time
    dask_backend = source_raster._chunks is not None
    if dask_backend and mp_config is not None:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove ``mp_config`` from "
            "to_pointcloud(). To use Multiprocessing, open the raster without ``chunks``."
        )
    if source_raster._is_xr and mp_config is not None:
        raise ValueError("Argument ``mp_config`` requires a Raster input rather than an Xarray accessor.")

    # Validate the coordinate convention before launching lazy or multiprocessing work
    if force_pixel_offset not in ("center", "ul", "ur", "ll", "lr"):
        raise ValueError(f"Unknown pixel offset {force_pixel_offset!r}.")

    # Use one seed so every tile participates in the same random subsample
    if random_state is None and subsample != 1:
        random_state = int(np.random.default_rng().integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))

    # 3/ Call the relevant backend depending on input type (eager, Dask, Multiproc)
    if dask_backend:
        return _dask_raster_to_pointcloud(
            source_raster=source_raster,
            bands=all_bands,
            column_names=all_column_names,
            data_column_name=data_column_name,
            subsample=subsample,
            skip_nodata=skip_nodata,
            random_state=random_state,
            force_pixel_offset=force_pixel_offset,
            as_array=as_array,
        )

    if mp_config is not None:
        return _multiproc_raster_to_pointcloud(
            source_raster=source_raster,
            bands=all_bands,
            column_names=all_column_names,
            data_column_name=data_column_name,
            subsample=subsample,
            skip_nodata=skip_nodata,
            random_state=random_state,
            force_pixel_offset=force_pixel_offset,
            mp_config=mp_config,
            as_array=as_array,
        )

    return _eager_raster_to_pointcloud(
        source_raster=source_raster,
        bands=all_bands,
        column_names=all_column_names,
        data_column_name=data_column_name,
        subsample=subsample,
        skip_nodata=skip_nodata,
        random_state=random_state,
        force_pixel_offset=force_pixel_offset,
        as_array=as_array,
    )
