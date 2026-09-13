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

import operator
import pathlib
import tempfile
from typing import TYPE_CHECKING, Any, Iterable, Literal

import affine
import geopandas as gpd
import numpy as np
import rasterio as rio
from numpy.typing import NDArray
from rasterio.crs import CRS

from geoutils._dispatch import get_geo_attr, has_geo_attr, is_dask_array
from geoutils._misc import import_optional
from geoutils._typing import NDArrayNum
from geoutils.raster.array import get_mask_from_array
from geoutils.raster.referencing import _default_nodata, _xy2ij

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
            or not (isinstance(grid_coords[0], np.ndarray) and grid_coords[0].ndim == 1)
            or not (isinstance(grid_coords[1], np.ndarray) and grid_coords[1].ndim == 1)
        ):
            raise TypeError("Input grid coordinates must be 1D arrays.")

        diff_x = np.diff(grid_coords[0])
        diff_y = np.diff(grid_coords[1])

        if not all(diff_x == diff_x[0]) and all(diff_y == diff_y[0]):
            raise ValueError("Grid coordinates must be regular (equally spaced, independently along X and Y).")

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
    if not np.issubdtype(i.dtype, np.integer) or not np.issubdtype(i.dtype, np.integer):
        raise ValueError("Some point cloud coordinates differ from the grid coordinates.")

    # Set values
    mask = np.ones(np.shape(arr), dtype=bool)
    mask[i, j] = False
    arr[i, j] = gdf_pc[data_column_name].values

    # Set output values
    raster_arr = np.ma.masked_array(data=arr, mask=mask)

    return raster_arr, out_transform, gdf_pc.crs, out_nodata, area_or_point


#########################
# 2/ RASTER TO POINT CLOUD
#########################


def _raster_to_pointcloud_partition(
    source_raster: RasterType,
    flat_indices: slice | NDArray[np.int64],
    bands: list[int],
    column_names: list[str],
    skip_nodata: bool,
    row_offset: float,
    column_offset: float,
    chunks: int | tuple[int, int],
    filename: pathlib.Path,
) -> pathlib.Path:
    """Read selected raster cells into one ordered point partition and stage it for the common file writer."""

    # Expand compact full-conversion ranges while preserving the supplied order of sampled cell indexes
    total_pixels = int(np.prod(source_raster.shape))
    if isinstance(flat_indices, slice):
        start, stop, step = flat_indices.indices(total_pixels)
        selected_indices = np.arange(start, stop, step, dtype=np.int64)
    else:
        selected_indices = np.asarray(flat_indices, dtype=np.int64)
    rows, columns = np.unravel_index(selected_indices, source_raster.shape)

    # Group requested cells by raster tile so each worker reads bounded windows instead of individual samples
    chunk_rows, chunk_columns = (chunks, chunks) if isinstance(chunks, int) else chunks
    tile_columns = (source_raster.shape[1] + chunk_columns - 1) // chunk_columns
    tile_ids = (rows // chunk_rows) * tile_columns + columns // chunk_columns
    dtype = np.dtype(bool if source_raster.is_mask else source_raster.dtype)
    band_values = np.ma.masked_all((len(bands), len(selected_indices)), dtype=dtype)
    native_bands = [source_raster.bands[band - 1] for band in bands]

    assert source_raster.name is not None
    with rio.open(source_raster.name) as dataset:
        for tile_id in np.unique(tile_ids):
            selected = np.flatnonzero(tile_ids == tile_id)
            tile_row, tile_column = divmod(int(tile_id), tile_columns)
            row_start = tile_row * chunk_rows
            column_start = tile_column * chunk_columns
            row_stop = min(row_start + chunk_rows, source_raster.shape[0])
            column_stop = min(column_start + chunk_columns, source_raster.shape[1])

            # Read every requested band once for this tile and restore values to their output positions
            block = dataset.read(
                native_bands,
                window=((row_start, row_stop), (column_start, column_stop)),
                masked=True,
            )
            local_rows = rows[selected] - row_start
            local_columns = columns[selected] - column_start
            selected_values = block[:, local_rows, local_columns]
            band_values.data[:, selected] = np.ma.getdata(selected_values)
            band_values.mask[:, selected] = np.ma.getmaskarray(selected_values)

    # Use main-band validity for every output column, matching eager and Dask selection behavior
    if skip_nodata:
        keep = ~get_mask_from_array(band_values[0]).reshape(-1)
        selected_indices = selected_indices[keep]
        band_values = band_values[:, keep].data
    else:
        band_values = np.ma.filled(band_values.astype("float32"), np.nan)
        if source_raster.nodata is not None:
            band_values[band_values == source_raster.nodata] = np.nan

    # Convert row/column positions with the full affine transform and requested pixel offset
    rows, columns = np.unravel_index(selected_indices, source_raster.shape)
    transform = source_raster.transform
    x_coords = transform.c + transform.a * (columns + column_offset) + transform.b * (rows + row_offset)
    y_coords = transform.f + transform.d * (columns + column_offset) + transform.e * (rows + row_offset)
    dataframe = gpd.GeoDataFrame(
        {name: band_values[index] for index, name in enumerate(column_names)},
        geometry=gpd.points_from_xy(x_coords, y_coords),
        crs=source_raster.crs,
    )
    dataframe.to_pickle(filename)
    return filename


def _multiproc_raster_to_pointcloud(
    source_raster: RasterType,
    bands: list[int],
    column_names: list[str],
    data_column_name: str,
    subsample: float | int,
    skip_nodata: bool,
    random_state: int | np.random.Generator | None,
    row_offset: float,
    column_offset: float,
    mp_config: MultiprocConfig,
) -> Any:
    """
    Convert raster cells in bounded workers and return an unloaded point cloud at the configured output path.

    _raster_to_pointcloud_partition() reads raster windows and saves each group of output points as a GeoDataFrame.
    _write_pointcloud_partitions() writes those saved groups to the requested file one at a time, so the parent process
    never holds all point values at once.
    """

    from geoutils.multiproc.cluster import _map_bounded
    from geoutils.pointcloud.writing import (
        _resolve_pointcloud_output,
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
        if source_raster.is_loaded or source_raster._is_xr or source_raster.name is None:
            from geoutils.raster import Raster

            source_filename = temporary_directory / "source.tif"
            source_raster.to_file(source_filename)
            worker_source = Raster(source_filename, load_data=False)

        total_pixels = int(np.prod(worker_source.shape))
        chunk_shape = (mp_config.chunks, mp_config.chunks) if isinstance(mp_config.chunks, int) else mp_config.chunks
        output_partition_size = int(np.prod(chunk_shape))

        # Keep the full path compact; sampled paths preserve their deterministic selection order across partitions
        if subsample == 1:
            selected_parts: list[slice | NDArray[np.int64]] = [
                slice(start, min(start + output_partition_size, total_pixels))
                for start in range(0, total_pixels, output_partition_size)
            ]
        elif not skip_nodata:
            from geoutils.sampling.subsampling import (
                _get_subsample_size_from_user_input,
            )

            sample_size = _get_subsample_size_from_user_input(subsample, total_pixels)
            flat_indices = np.random.default_rng(random_state).choice(total_pixels, sample_size, replace=False)
            selected_parts = [
                flat_indices[start : start + output_partition_size]
                for start in range(0, len(flat_indices), output_partition_size)
            ]
        else:
            from geoutils.sampling.subsampling import _subsample

            stable_random_state = random_state
            if stable_random_state is None:
                stable_random_state = int(np.random.default_rng().integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            row_indices, column_indices = _subsample(
                source_raster=worker_source,
                subsample=subsample,
                band=bands[0],
                return_indices=True,
                random_state=stable_random_state,
                strategy="topk",
                mp_config=mp_config,
            )
            flat_indices = np.asarray(row_indices, dtype=np.int64) * worker_source.shape[1] + np.asarray(
                column_indices, dtype=np.int64
            )
            selected_parts = [
                flat_indices[start : start + output_partition_size]
                for start in range(0, len(flat_indices), output_partition_size)
            ]

        # Preserve a valid empty output schema when no eligible cells were selected
        if not selected_parts:
            selected_parts = [np.empty(0, dtype=np.int64)]
        arguments = (
            (
                worker_source,
                selected,
                bands,
                column_names,
                skip_nodata,
                row_offset,
                column_offset,
                mp_config.chunks,
                temporary_directory / f"partition_{index}.pkl",
            )
            for index, selected in enumerate(selected_parts)
        )
        partition_filenames = [
            filename for _, filename in _map_bounded(mp_config.cluster, _raster_to_pointcloud_partition, arguments)
        ]

        # Assemble one ordered file and return its metadata-only PointCloud wrapper
        return _write_pointcloud_partitions(
            output_filename,
            partition_filenames,
            driver=driver,
            data_column=data_column_name,
            geometry_type="Point",
            mp_config=mp_config,
        )


def _raster_to_pointcloud(
    source_raster: RasterType,
    data_column_name: str = "b1",
    data_band: int = 1,
    auxiliary_data_bands: list[int] | None = None,
    auxiliary_column_names: list[str] | None = None,
    subsample: float | int = 1,
    skip_nodata: bool = True,
    as_array: bool = False,
    random_state: int | np.random.Generator | None = None,
    force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"] = "ul",
    mp_config: MultiprocConfig | None = None,
) -> Any:
    """
    Convert a raster to a point cloud. See Raster.to_pointcloud() for details.
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
        if not (isinstance(auxiliary_data_bands, Iterable) and all(isinstance(b, int) for b in auxiliary_data_bands)):
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
            if not (
                isinstance(auxiliary_column_names, Iterable) and all(isinstance(b, str) for b in auxiliary_column_names)
            ):
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

    # One operation cannot be scheduled by Dask and multiprocessing at the same time
    dask_backend = source_raster._chunks is not None
    if dask_backend and mp_config is not None:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove ``mp_config`` from "
            "to_pointcloud(). To use Multiprocessing, open the raster without ``chunks``."
        )

    # Validate the coordinate convention before launching lazy or multiprocessing work
    offsets = {"center": (0.5, 0.5), "ul": (0.0, 0.0), "ur": (0.0, 1.0), "ll": (1.0, 0.0), "lr": (1.0, 1.0)}
    try:
        row_offset, column_offset = offsets[force_pixel_offset]
    except KeyError as exception:
        raise ValueError(f"Unknown pixel offset {force_pixel_offset!r}.") from exception

    # Multiprocessing point output is assembled on disk and returned without loading its values in the parent
    if mp_config is not None and not as_array:
        return _multiproc_raster_to_pointcloud(
            source_raster=source_raster,
            bands=all_bands,
            column_names=all_column_names,
            data_column_name=data_column_name,
            subsample=subsample,
            skip_nodata=skip_nodata,
            random_state=random_state,
            row_offset=row_offset,
            column_offset=column_offset,
            mp_config=mp_config,
        )

    # Preserve eager full-conversion loading, while file-backed chunked conversions use window reads
    sampling_config = mp_config
    if not dask_backend and not source_raster.is_loaded and sampling_config is None:
        if subsample == 1:
            source_raster.load()
        else:
            from geoutils.multiproc import MultiprocConfig

            sampling_config = MultiprocConfig(chunks=512)

    data = source_raster.data if source_raster.is_loaded or source_raster._is_xr else None
    total_pixels = int(np.prod(source_raster.shape))

    # Flatten complete rasters directly, reading file-backed inputs in whole tiles without random sampling
    if subsample == 1:
        if data is not None:
            band_values = [data.reshape(-1) if data.ndim == 2 else data[band - 1].reshape(-1) for band in all_bands]
            main_values = data.reshape(-1) if data.ndim == 2 else data[data_band - 1].reshape(-1)
        else:
            from geoutils.multiproc import MultiprocConfig
            from geoutils.multiproc.chunked import iter_chunk_slices
            from geoutils.multiproc.cluster import _map_bounded
            from geoutils.multiproc.readers import _read_values, _ValueReader

            read_config = sampling_config if sampling_config is not None else MultiprocConfig(chunks=512)
            tile_slices = list(iter_chunk_slices(source_raster.shape, read_config.chunks))
            band_values = []
            for band in all_bands:
                reader = _ValueReader(source_raster, band)
                arguments = [(reader.block(slices),) for slices in tile_slices]
                pieces = [piece for _, piece in _map_bounded(read_config.cluster, _read_values, arguments)]

                # Place rectangular tiles on the raster grid before taking a row-major flattened view
                raster_values = np.ma.masked_all(source_raster.shape, dtype=reader.dtype)
                for slices, piece in zip(tile_slices, pieces):
                    raster_values.data[slices] = np.ma.getdata(piece)
                    raster_values.mask[slices] = np.ma.getmaskarray(piece)
                band_values.append(raster_values.reshape(-1))
            main_values = band_values[0]

        if data is not None and is_dask_array(data):
            import dask.array as da

            dask = import_optional("dask")
            flat_indices: Any = da.arange(total_pixels, chunks=main_values.chunks)

            # Find the output size of each lazy block before constructing arrays with a ragged validity filter
            if skip_nodata:
                valid = da.isfinite(main_values)
                if source_raster.nodata is not None:
                    valid &= main_values != source_raster.nodata
                valid_blocks = valid.to_delayed().ravel()
                valid_counts = dask.compute(*[dask.delayed(np.count_nonzero)(block) for block in valid_blocks])

                # Apply each validity block to the flat indices and bands while keeping their values lazy
                if sum(valid_counts) != total_pixels:
                    selected_values = []
                    for values in [flat_indices, *band_values]:
                        value_blocks = values.rechunk(valid.chunks).to_delayed().ravel()
                        selected_blocks = [
                            da.from_delayed(
                                dask.delayed(operator.getitem)(value_block, valid_block),
                                shape=(int(count),),
                                dtype=values.dtype,
                            )
                            for value_block, valid_block, count in zip(value_blocks, valid_blocks, valid_counts)
                        ]
                        selected_values.append(da.concatenate(selected_blocks))
                    flat_indices, *band_values = selected_values

            pixel_data = da.stack(band_values, axis=0)
        else:
            # Keep the common all-finite case as reshaped views, filtering only when the main band has missing cells
            flat_indices = np.arange(total_pixels, dtype=np.int64)
            if skip_nodata:
                invalid = get_mask_from_array(main_values).reshape(-1)
                if np.any(invalid):
                    flat_indices = np.flatnonzero(~invalid)
                    band_values = [values[flat_indices] for values in band_values]
            if len(band_values) == 1:
                pixel_data = band_values[0].reshape(1, -1)
            elif any(np.ma.isMaskedArray(values) for values in band_values):
                pixel_data = np.ma.stack(band_values, axis=0)
            else:
                pixel_data = np.stack(band_values, axis=0)

        rows = flat_indices // source_raster.shape[1]
        columns = flat_indices % source_raster.shape[1]

    else:
        from geoutils.sampling.subsampling import (
            _get_subsample_size_from_user_input,
            _subsample,
        )

        # Select missing cells when requested, otherwise use stable keys for identical results from every backend
        if not skip_nodata:
            sample_size = _get_subsample_size_from_user_input(subsample, total_pixels)
            flat_indices = np.random.default_rng(random_state).choice(total_pixels, sample_size, replace=False)
            rows, columns = np.unravel_index(flat_indices, source_raster.shape)
        else:
            stable_random_state = random_state
            if stable_random_state is None:
                stable_random_state = int(np.random.default_rng().integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            indices = _subsample(
                source_raster=source_raster,
                subsample=subsample,
                band=data_band,
                return_indices=True,
                random_state=stable_random_state,
                strategy="topk",
                mp_config=sampling_config,
            )
            if any(is_dask_array(index) for index in indices):
                indices = import_optional("dask").compute(*indices)
            rows, columns = (np.asarray(index, dtype=np.int64) for index in indices)

        # Gather the selected values lazily, from memory, or through file windows
        if data is not None and is_dask_array(data):
            import dask.array as da

            dask_data: Any = data
            band_values = [
                dask_data.vindex[rows, columns] if dask_data.ndim == 2 else dask_data[band - 1].vindex[rows, columns]
                for band in all_bands
            ]
            pixel_data = da.stack(band_values, axis=0)
        elif data is not None:
            band_values = [
                data[rows, columns] if data.ndim == 2 else data[band - 1, rows, columns] for band in all_bands
            ]
            pixel_data = (
                np.ma.stack(band_values, axis=0)
                if any(np.ma.isMaskedArray(values) for values in band_values)
                else np.stack(band_values, axis=0)
            )
        else:
            from geoutils.multiproc import MultiprocConfig
            from geoutils.multiproc.readers import _read_selected_values, _ValueReader

            read_config = sampling_config if sampling_config is not None else MultiprocConfig(chunks=512)
            flat_indices = rows * source_raster.shape[1] + columns
            band_values = [
                _read_selected_values(_ValueReader(source_raster, band), flat_indices, read_config)
                for band in all_bands
            ]
            pixel_data = (
                np.ma.stack(band_values, axis=0)
                if any(np.ma.isMaskedArray(values) for values in band_values)
                else np.stack(band_values, axis=0)
            )

    # Remove mask metadata after valid cells are selected; kept nodata cells are converted to NaN below
    if is_dask_array(pixel_data) and skip_nodata:
        import dask.array as da

        pixel_data = da.ma.getdata(pixel_data)
    elif np.ma.isMaskedArray(pixel_data):
        pixel_data = pixel_data.data

    # If nodata values were not skipped, convert them to NaNs and change data type
    if not skip_nodata:
        pixel_data = pixel_data.astype("float32")
        if is_dask_array(pixel_data):
            import dask.array as da

            pixel_data = da.ma.filled(pixel_data, np.nan)
            if source_raster.nodata is not None:
                pixel_data = da.where(pixel_data == source_raster.nodata, np.nan, pixel_data)
        elif source_raster.nodata is not None:
            pixel_data[pixel_data == source_raster.nodata] = np.nan

    # Calculate coordinates with the complete affine transform, including rotation and the requested pixel offset
    transform = source_raster.transform
    x_coords = transform.c + transform.a * (columns + column_offset) + transform.b * (rows + row_offset)
    y_coords = transform.f + transform.d * (columns + column_offset) + transform.e * (rows + row_offset)

    # Preserve Dask execution in both array and point dataframe outputs
    if is_dask_array(pixel_data):
        import dask.array as da

        if not is_dask_array(x_coords):
            x_coords = da.from_array(np.asarray(x_coords), chunks=pixel_data.chunks[1])
            y_coords = da.from_array(np.asarray(y_coords), chunks=pixel_data.chunks[1])
        if as_array:
            return da.stack((x_coords, y_coords, *[pixel_data[index] for index in range(len(all_bands))]), axis=1)

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
        dataframe = dask_dataframe.from_dask_array(pixel_data.T, columns=all_column_names)
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

    # Build the established eager array or PointCloud result
    if as_array:
        return np.vstack((np.asarray(x_coords), np.asarray(y_coords), pixel_data)).T

    from geoutils.pointcloud import PointCloud

    dataframe = gpd.GeoDataFrame(
        pixel_data.T,
        columns=all_column_names,
        geometry=gpd.points_from_xy(np.asarray(x_coords), np.asarray(y_coords)),
        crs=source_raster.crs,
    )
    return PointCloud(dataframe, data_column=data_column_name)
