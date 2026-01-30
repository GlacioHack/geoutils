# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES)
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

"""Chunked calculations with Multiprocessing."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio as rio
from rasterio._io import Resampling

from geoutils._typing import DTypeLike, NDArrayNum
from geoutils.multiproc.mparray import MultiprocConfig, _write_multiproc_result
from geoutils.raster.distributed_computing.chunked import (
    _build_geotiling_and_meta,
    _chunks2d_from_chunksizes_shape,
    _reproject_per_block,
)


def _wrapper_multiproc_reproject_per_block(
    rst: gu.Raster,
    src_block_ids: list[dict[str, int]],
    dst_block_id: dict[str, int],
    idx_d2s: list[int],
    block_ids: list[dict[str, int]],
    combined_meta: dict[str, Any],
    **kwargs: Any,
) -> tuple[NDArrayNum, tuple[int, int, int, int]]:
    """Wrapper to use Delayed reprojection per destination block
    (also rebuilds a square array combined from intersecting source blocks)."""

    # Get source array block for each destination block
    s = src_block_ids
    src_arrs = (rst.icrop(bbox=(s[idx]["xs"], s[idx]["ys"], s[idx]["xe"], s[idx]["ye"])).data for idx in idx_d2s)

    # Call reproject per block
    dst_block_arr = _reproject_per_block(*src_arrs, block_ids=block_ids, combined_meta=combined_meta, **kwargs)

    return dst_block_arr, (dst_block_id["ys"], dst_block_id["ye"], dst_block_id["xs"], dst_block_id["xe"])


def _multiproc_reproject(
    rst: gu.Raster,
    config: MultiprocConfig,
    src_crs: rio.CRS,
    src_nodata: int | float | None,
    dst_shape: tuple[int, int],
    dst_transform: rio.Affine,
    dst_crs: rio.CRS,
    dst_nodata: int | float | None,
    dtype: DTypeLike,
    resampling: Resampling,
    **kwargs: Any,
) -> None:
    """
    Reproject georeferenced raster on out-of-memory chunks with multiprocessing.
    See Raster.reproject() for details.
    """

    # Prepare geotiling and reprojection metadata for source and destination grids
    src_chunks = _chunks2d_from_chunksizes_shape(chunksizes=(config.chunk_size, config.chunk_size), shape=rst.shape)
    src_geotiling, dst_geotiling, dst_chunks, dest2source, src_block_ids, meta_params, dst_block_geogrids = (
        _build_geotiling_and_meta(
            src_shape=rst.shape,
            src_transform=rst.transform,
            src_crs=rst.crs,
            dst_shape=dst_shape,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_chunks=src_chunks,
            dst_chunksizes=(config.chunk_size, config.chunk_size),
        )
    )

    # 4/ Call a delayed function that uses rio.warp to reproject the combined source block(s) to each destination block
    kwargs.update(
        {
            "src_nodata": src_nodata,
            "dst_nodata": dst_nodata,
            "resampling": resampling,
            "src_crs": src_crs,
            "dst_crs": dst_crs,
        }
    )
    # Get location of destination blocks to write file
    dst_block_ids = np.array(dst_geotiling.get_block_locations())

    # Create tasks for multiprocessing
    tasks = []
    for i in range(len(dest2source)):
        tasks.append(
            config.cluster.launch_task(
                fun=_wrapper_multiproc_reproject_per_block,
                args=[
                    rst,
                    src_block_ids,
                    dst_block_ids[i],
                    dest2source[i],
                    meta_params[i][1],
                    meta_params[i][0],
                ],
                kwargs=kwargs,
            )
        )

    # Retrieve metadata for saving file
    file_metadata = {
        "width": dst_shape[1],
        "height": dst_shape[0],
        "count": rst.count,
        "crs": dst_crs,
        "transform": dst_transform,
        "dtype": dtype,
        "nodata": dst_nodata,
    }

    # Create a new raster file to save the processed results
    _write_multiproc_result(tasks, config, file_metadata)
