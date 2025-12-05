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

"""Module implementing operations on raster chunks, defines subfunctions called by "dask" and "multiproc" modules."""

from __future__ import annotations

from typing import Any, Literal, TypeVar

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio import CRS

from geoutils._typing import NDArrayNum
from geoutils.projtools import _get_bounds_projected, _get_footprint_projected
from geoutils.raster._geotransformations import _rio_reproject

# REPROJECT (subfunctions called both in "dask" or "multiproc" module)

# At the date of April 2024: not supported by Rioxarray
# Part of the code (defining a GeoGrid and GeoTiling classes) is inspired by
# https://github.com/opendatacube/odc-geo/pull/88, modified to be concise, stand-alone and rely only on
# Rasterio/GeoPandas

# We define a GeoGrid and GeoTiling class (which composes GeoGrid) to consistently deal with georeferenced footprints
# of chunked grids
GeoGridType = TypeVar("GeoGridType", bound="GeoGrid")


class GeoGrid:
    """
    Georeferenced grid class.

    Describes a georeferenced grid through a geotransform (one-sided bounds and resolution), shape and CRS.
    """

    def __init__(self, transform: rio.transform.Affine, shape: tuple[int, int], crs: rio.crs.CRS | None):

        self._transform = transform
        self._shape = shape
        self._crs = crs

    @property
    def transform(self) -> rio.transform.Affine:
        return self._transform

    @property
    def crs(self) -> rio.crs.CRS:
        return self._crs

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def res(self) -> tuple[int, int]:
        return self.transform[0], abs(self.transform[4])

    def bounds_projected(self, crs: rio.crs.CRS = None) -> rio.coords.BoundingBox:
        if crs is None:
            crs = self.crs
        bounds = rio.coords.BoundingBox(*rio.transform.array_bounds(self.height, self.width, self.transform))
        return _get_bounds_projected(bounds=bounds, in_crs=self.crs, out_crs=crs)

    @property
    def bounds(self) -> rio.coords.BoundingBox:
        return self.bounds_projected()

    def footprint_projected(self, crs: rio.crs.CRS = None) -> gpd.GeoDataFrame:
        if crs is None:
            crs = self.crs
        return _get_footprint_projected(self.bounds, in_crs=self.crs, out_crs=crs, densify_points=100)

    @property
    def footprint(self) -> gpd.GeoDataFrame:
        return self.footprint_projected()

    @classmethod
    def from_dict(cls: type[GeoGridType], dict_meta: dict[str, Any]) -> GeoGridType:
        """Create a GeoGrid from a dictionary containing transform, shape and CRS."""
        return cls(**dict_meta)

    def translate(
        self: GeoGridType,
        xoff: float,
        yoff: float,
        distance_unit: Literal["georeferenced"] | Literal["pixel"] = "pixel",
    ) -> GeoGridType:
        """Translate into a new geogrid (not inplace)."""

        if distance_unit not in ["georeferenced", "pixel"]:
            raise ValueError("Argument 'distance_unit' should be either 'pixel' or 'georeferenced'.")

        # Get transform
        dx, b, xmin, d, dy, ymax = list(self.transform)[:6]

        # Convert pixel offsets to georeferenced units
        if distance_unit == "pixel":
            # Can either multiply the offset by the resolution
            # xoff *= self.res[0]
            # yoff *= self.res[1]

            # Or use the boundaries instead! (maybe less floating point issues? doesn't seem to matter in tests)
            xoff = xoff / self.shape[1] * (self.bounds.right - self.bounds.left)
            yoff = yoff / self.shape[0] * (self.bounds.top - self.bounds.bottom)

        shifted_transform = rio.transform.Affine(dx, b, xmin + xoff, d, dy, ymax + yoff)

        return self.from_dict({"transform": shifted_transform, "crs": self.crs, "shape": self.shape})


def _get_block_ids_per_chunk(chunks: tuple[tuple[int, ...], tuple[int, ...]]) -> list[dict[str, int]]:
    """Get location of chunks based on array shape and list of chunk sizes."""

    # Get number of chunks
    num_chunks = (len(chunks[0]), len(chunks[1]))

    # Get robust list of chunk locations (using what is done in block_id of dask.array.map_blocks)
    # https://github.com/dask/dask/blob/24493f58660cb933855ba7629848881a6e2458c1/dask/array/core.py#L908
    from dask.utils import cached_cumsum

    starts = [cached_cumsum(c, initial_zero=True) for c in chunks]
    nb_blocks = num_chunks[0] * num_chunks[1]
    ixi, iyi = np.unravel_index(np.arange(nb_blocks), shape=(num_chunks[0], num_chunks[1]))
    # Starting and ending indexes "s" and "e" for both X/Y, to place the chunk in the full array
    block_ids = [
        {
            "num_block": i,
            "ys": starts[0][ixi[i]],
            "xs": starts[1][iyi[i]],
            "ye": starts[0][ixi[i] + 1],
            "xe": starts[1][iyi[i] + 1],
        }
        for i in range(nb_blocks)
    ]

    return block_ids


class ChunkedGeoGrid:
    """
    Chunked georeferenced grid class.

    Associates a georeferenced grid to chunks (possibly of varying sizes).
    """

    def __init__(self, grid: GeoGrid, chunks: tuple[tuple[int, ...], tuple[int, ...]]):

        self._grid = grid
        self._chunks = chunks

    @property
    def grid(self) -> GeoGrid:
        return self._grid

    @property
    def chunks(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self._chunks

    def get_block_locations(self) -> list[dict[str, int]]:
        """Get block locations in 2D: xstart, xend, ystart, yend."""
        return _get_block_ids_per_chunk(self._chunks)

    def get_blocks_as_geogrids(self) -> list[GeoGrid]:
        """Get blocks as geogrids with updated transform/shape."""

        block_ids = self.get_block_locations()

        list_geogrids = []
        for bid in block_ids:
            # We get the block size
            block_shape = (bid["ye"] - bid["ys"], bid["xe"] - bid["xs"])
            # Build a temporary geogrid with the same transform as the full grid, but with the chunk shape
            geogrid_tmp = GeoGrid(transform=self.grid.transform, crs=self.grid.crs, shape=block_shape)
            # And shift it to the right location (X is positive in index direction, Y is negative)
            geogrid_block = geogrid_tmp.translate(xoff=bid["xs"], yoff=-bid["ys"])
            list_geogrids.append(geogrid_block)

        return list_geogrids

    def get_block_footprints(self, crs: rio.crs.CRS = None) -> gpd.GeoDataFrame:
        """Get block projected footprints as a single geodataframe."""

        geogrids = self.get_blocks_as_geogrids()
        footprints = [gg.footprint_projected(crs=crs) if crs is not None else gg.footprint for gg in geogrids]

        return pd.concat(footprints)


def _chunks2d_from_chunksizes_shape(
    chunksizes: tuple[int, int], shape: tuple[int, int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Get tuples of chunk sizes for X/Y dimensions based on chunksizes and array shape."""

    # Chunksize is fixed, except for the last chunk depending on the shape
    chunks_y = tuple(
        min(
            chunksizes[0],
            shape[0] - i * chunksizes[0],
        )
        for i in range(int(np.ceil(shape[0] / chunksizes[0])))
    )
    chunks_x = tuple(
        min(
            chunksizes[1],
            shape[1] - i * chunksizes[1],
        )
        for i in range(int(np.ceil(shape[1] / chunksizes[1])))
    )

    return chunks_y, chunks_x


def _combined_blocks_shape_transform(
    sub_block_ids: list[dict[str, int]], src_geogrid: GeoGrid
) -> tuple[dict[str, Any], list[dict[str, int]]]:
    """Derive combined shape and transform from a subset of several blocks (for source input during reprojection)."""

    # Get combined shape by taking min of X/Y starting indices, max of X/Y ending indices
    all_xs, all_ys, all_xe, all_ye = ([b[s] for b in sub_block_ids] for s in ["xs", "ys", "xe", "ye"])
    minmaxs = {"min_xs": np.min(all_xs), "max_xe": np.max(all_xe), "min_ys": np.min(all_ys), "max_ye": np.max(all_ye)}
    combined_shape = (minmaxs["max_ye"] - minmaxs["min_ys"], minmaxs["max_xe"] - minmaxs["min_xs"])

    # Shift source transform with start indexes to get the one for combined block location
    combined_transform = src_geogrid.translate(xoff=minmaxs["min_xs"], yoff=-minmaxs["min_ys"]).transform

    # Compute relative block indexes that will be needed to reconstruct a square array in the delayed function,
    # by subtracting the minimum starting indices in X/Y
    relative_block_indexes = [
        {"r" + s1 + s2: b[s1 + s2] - minmaxs["min_" + s1 + "s"] for s1 in ["x", "y"] for s2 in ["s", "e"]}
        for b in sub_block_ids
    ]

    combined_meta = {"src_shape": combined_shape, "src_transform": tuple(combined_transform)}

    return combined_meta, relative_block_indexes


def _build_geotiling_and_meta(
    src_shape: tuple[int, int],
    src_transform: rio.transform.Affine,
    src_crs: CRS,
    dst_shape: tuple[int, int],
    dst_transform: rio.transform.Affine,
    dst_crs: CRS,
    src_chunks: tuple[tuple[int, ...], tuple[int, ...]],
    dst_chunksizes: tuple[int, int],
) -> tuple[
    ChunkedGeoGrid,
    ChunkedGeoGrid,
    tuple[tuple[int, ...], tuple[int, ...]],
    list[list[int]],
    list[dict[str, int]],
    list[tuple[dict[str, Any], list[dict[str, int]]]],
    list[GeoGrid],
]:
    """
    Constructs georeferenced tiling information and reprojection metadata for both source and destination grids,
    used to support block-wise reprojection operations (e.g. with multiprocessing or dask).

    This function performs the following:
    1. Constructs `GeoGrid` and `ChunkedGeoGrid` objects for source and destination rasters,
       based on provided shape, transform, CRS, and chunk sizes.
    2. Computes spatial footprints for each chunk in both grids, and determines which
       source chunks intersect each destination chunk (with a buffer to ensure overlap).
    3. For each destination chunk, calculates metadata required for reprojection, including:
       - The combined shape and transform of all intersecting source chunks.
       - The specific shape and transform of the destination block.

    :return: A tuple containing:
        - Source `ChunkedGeoGrid`
        - Destination `ChunkedGeoGrid`
        - Destination chunks
        - Mapping from destination to intersecting source block indices
        - Array of source block locations
        - List of metadata dictionaries per destination block
        - List of destination `GeoGrid` blocks
    """

    # 1/ Define source and destination chunked georeferenced grid through simple classes storing CRS/transform/shape,
    # which allow to consistently derive shape/transform for each block and their CRS-projected footprints

    # Define GeoGrids for source/destination array
    src_geogrid = GeoGrid(transform=src_transform, shape=src_shape, crs=src_crs)
    dst_geogrid = GeoGrid(transform=dst_transform, shape=dst_shape, crs=dst_crs)

    # Create tilings
    src_geotiling = ChunkedGeoGrid(grid=src_geogrid, chunks=src_chunks)
    dst_chunks = _chunks2d_from_chunksizes_shape(chunksizes=dst_chunksizes, shape=dst_shape)
    dst_geotiling = ChunkedGeoGrid(grid=dst_geogrid, chunks=dst_chunks)

    # 2/ Get footprints of tiles in CRS of destination array, with a buffer of 2 pixels for destination ones to ensure
    # overlap, then map indexes of source blocks that intersect a given destination block
    src_footprints = src_geotiling.get_block_footprints(crs=dst_crs)
    dst_footprints = dst_geotiling.get_block_footprints().buffer(2 * max(dst_geogrid.res))
    dest2source = [list(np.where(dst.intersects(src_footprints).values)[0]) for dst in dst_footprints]

    # 3/ To reconstruct a square source array during chunked reprojection, we need to derive the combined shape and
    # transform of each tuples of source blocks
    src_block_ids = src_geotiling.get_block_locations()
    meta_params = [
        (
            _combined_blocks_shape_transform(sub_block_ids=[src_block_ids[i] for i in sbid], src_geogrid=src_geogrid)
            if len(sbid) > 0
            else ({}, [])
        )
        for sbid in dest2source
    ]

    # Append dst shape/transform to metadata
    dst_block_geogrids = dst_geotiling.get_blocks_as_geogrids()
    for i, (c, _) in enumerate(meta_params):
        c.update({"dst_shape": dst_block_geogrids[i].shape, "dst_transform": tuple(dst_block_geogrids[i].transform)})

    return src_geotiling, dst_geotiling, dst_chunks, dest2source, src_block_ids, meta_params, dst_block_geogrids


def _reproject_per_block(
    *src_arrs: tuple[NDArrayNum], block_ids: list[dict[str, int]], combined_meta: dict[str, Any], **kwargs: Any
) -> NDArrayNum:
    """
    Reprojection per destination block (also rebuilds a square array combined from intersecting source blocks).
    """

    # If no source chunk intersects, we return a chunk of destination nodata values
    if len(src_arrs) == 0:
        # We can use float32 to return NaN, will be cast to other floating type later if that's not source array dtype
        dst_arr = np.zeros(combined_meta["dst_shape"], dtype=np.dtype("float32"))
        dst_arr[:] = kwargs["dst_nodata"]
        return dst_arr

    # First, we build an empty array with the combined shape, only with nodata values
    is_multiband = len(src_arrs[0].shape) > 2
    shape = (src_arrs[0].shape[0], *combined_meta["src_shape"]) if is_multiband else combined_meta["src_shape"]

    comb_src_arr = np.full(shape, kwargs["src_nodata"], dtype=src_arrs[0].dtype)

    # Then fill it with the source chunks values
    for arr, bid in zip(src_arrs, block_ids):
        comb_src_arr[..., bid["rys"] : bid["rye"], bid["rxs"] : bid["rxe"]] = arr

    # Now, we can simply call Rasterio!

    # We build the combined transform from tuple
    src_transform = rio.transform.Affine(*combined_meta["src_transform"])
    dst_transform = rio.transform.Affine(*combined_meta["dst_transform"])

    # Reproject wrapper

    # Force the number of threads to 1 to avoid Dask/Rasterio conflicting on multi-threading
    kwargs.update(
        {
            "dst_shape": combined_meta["dst_shape"],
            "src_transform": src_transform,
            "dst_transform": dst_transform,
            "n_threads": 1,
        }
    )
    # Define dtype if undefined
    if "dtype" not in kwargs:
        kwargs.update({"dtype": comb_src_arr.dtype})

    src_mask = comb_src_arr == kwargs["src_nodata"]
    dst_arr, dst_mask = _rio_reproject(src_arr=comb_src_arr, src_mask=src_mask, reproj_kwargs=kwargs)  # type: ignore
    if np.issubdtype(dst_arr.dtype, np.floating):
        dst_arr[dst_mask] = np.nan

    return dst_arr
