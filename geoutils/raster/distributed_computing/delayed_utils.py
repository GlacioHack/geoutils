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

"""
Common module with functions for managing chunks, indices, and grid-based operations for raster data.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, TypeVar

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio

from geoutils._typing import NDArrayNum
from geoutils.projtools import _get_bounds_projected, _get_footprint_projected

# 1/ SUBSAMPLING


def _get_subsample_size_from_user_input(
    subsample: int | float, total_nb_valids: int, silence_max_subsample: bool
) -> int:
    """Get subsample size based on a user input of either integer size or fraction of the number of valid points."""

    # If value is between 0 and 1, use a fraction
    if (subsample <= 1) & (subsample > 0):
        npoints = int(subsample * total_nb_valids)
    # Otherwise use the value directly
    elif subsample > 1:
        # Use the number of valid points if larger than subsample asked by user
        npoints = min(int(subsample), total_nb_valids)
        if subsample > total_nb_valids:
            if not silence_max_subsample:
                warnings.warn(
                    f"Subsample value of {subsample} is larger than the number of valid pixels of {total_nb_valids},"
                    f"using all valid pixels as a subsample.",
                    category=UserWarning,
                )
    else:
        raise ValueError("Subsample must be > 0.")

    return npoints


def _get_indices_block_per_subsample(
    indices_1d: NDArrayNum, num_chunks: tuple[int, int], nb_valids_per_block: list[int]
) -> list[list[int]]:
    """
    Get list of 1D valid subsample indices relative to the block for each block.

    The 1D valid subsample indices correspond to the subsample index to apply for a flattened array of valid values.
    Relative to the block means converted so that the block indexes for valid values starts at 0 up to the number of
    valid values in that block (while the input indices go from zero to the total number of valid values in the full
    array).

    :param indices_1d: Subsample 1D indexes among a total number of valid values.
    :param num_chunks: Number of chunks in X and Y.
    :param nb_valids_per_block: Number of valid pixels per block.

    :returns: Relative 1D valid subsample index per block.
    """

    # Apply a cumulative sum to get the first 1D total index of each block
    valids_cumsum = np.cumsum(nb_valids_per_block)

    # We can write a faster algorithm by sorting
    indices_1d = np.sort(indices_1d)

    # TODO: Write nested lists into array format to further save RAM?
    # We define a list of indices per block
    relative_index_per_block = [[] for _ in range(num_chunks[0] * num_chunks[1])]
    k = 0  # K is the block number
    for i in indices_1d:

        # Move to the next block K where current 1D subsample index is, if not in this one
        while i >= valids_cumsum[k]:
            k += 1

        # Add 1D subsample index  relative to first subsample index of this block
        first_index_block = valids_cumsum[k - 1] if k >= 1 else 0  # The first 1D valid subsample index of the block
        relative_index = i - first_index_block
        relative_index_per_block[k].append(relative_index)

    return relative_index_per_block


# 2/ POINT INTERPOLATION ON REGULAR OR EQUAL GRID


def _get_interp_indices_per_block(
    interp_x: NDArrayNum,
    interp_y: NDArrayNum,
    starts: list[tuple[int, ...]],
    num_chunks: tuple[int, int],
    chunksize: tuple[int, int],
    xres: float,
    yres: float,
) -> list[list[int]]:
    """Map blocks where each pair of interpolation coordinates will have to be computed."""

    # TODO 1: Check the robustness for chunksize different and X and Y

    # TODO 2: Check if computing block_i_id matricially + using an == comparison (possibly delayed) to get index
    #  per block is not more computationally efficient?
    #  (as it uses array instead of nested lists, and nested lists grow in RAM very fast)

    # The argument "starts" contains the list of chunk first X/Y index for the full array, plus the last index

    # We use one bucket per block, assuming a flattened blocks shape
    ind_per_block = [[] for _ in range(num_chunks[0] * num_chunks[1])]
    for i, (x, y) in enumerate(zip(interp_x, interp_y)):
        # Because it is a regular grid, we know exactly in which block ID the coordinate will fall
        block_i_1d = int((x - starts[0][0]) / (xres * chunksize[0])) * num_chunks[1] + int(
            (y - starts[1][0]) / (yres * chunksize[1])
        )
        ind_per_block[block_i_1d].append(i)

    return ind_per_block


# 3/ REPROJECT
# The following GeoGrid and GeoTiling classes assist in managing georeferenced grids and performing reprojection
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


def _get_block_ids_per_chunk(chunks: tuple[tuple[int, ...], tuple[int, ...]]) -> list[dict[str, int]]:
    """Get location of chunks based on array shape and list of chunk sizes."""

    # Get number of chunks
    num_chunks = (len(chunks[0]), len(chunks[1]))

    # Calculate the cumulative sum of the chunk sizes to determine block start/end indices
    starts = [np.cumsum([0] + list(c)) for c in chunks]  # Add initial zero for the start

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
