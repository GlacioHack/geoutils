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
"""Module defining array and configuration routines for chunked operations with Multiprocessing."""

from typing import TypeVar, Any, Literal

import numpy as np
import rasterio as rio
import geopandas as gpd
import pandas as pd

from geoutils.projtools import _get_footprint_projected, _get_bounds_projected

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
        return _get_bounds_projected(bounds=bounds, in_crs=self.crs, out_crs=crs, densify_points=5)

    @property
    def bounds(self) -> rio.coords.BoundingBox:
        return self.bounds_projected()

    def footprint_projected(self, crs: rio.crs.CRS = None, buffer_px: int = 0) -> gpd.GeoDataFrame:
        if crs is None:
            crs = self.crs
        return _get_footprint_projected(self.bounds, in_crs=self.crs, out_crs=crs, densify_points=5)

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

    @property
    def num_chunks(self) -> tuple[int, int]:
        return len(self.chunks[0]), len(self.chunks[1])

    def flat_block_index(self, chunk_location: tuple[int, int]) -> int:
        """"""
        iy, ix = chunk_location
        ny, nx = self.num_chunks
        if not (0 <= iy < ny and 0 <= ix < nx):
            raise IndexError(chunk_location)
        return iy * nx + ix

    def iter_block_geogrids(self):
        """Yield (flat_index, GeoGrid) in same order as get_blocks_as_geogrids()"""
        for i, gg in enumerate(self.get_blocks_as_geogrids()):
            yield i, gg

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

