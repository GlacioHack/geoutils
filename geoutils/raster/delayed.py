"""
Module for dask-delayed functions for out-of-memory raster operations.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, TypeVar

import dask.array as da
import dask.delayed
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from dask.utils import cached_cumsum
from scipy.interpolate import interpn

from geoutils._typing import NDArrayBool, NDArrayNum
from geoutils.projtools import _get_bounds_projected, _get_footprint_projected

# 1/ SUBSAMPLING
# At the date of April 2024:
# Getting an exact subsample size out-of-memory only for valid values is not supported directly by Dask/Xarray

# It is not trivial because we don't know where valid values will be in advance, and because of ragged output (varying
# output length considerations), which prevents from using high-level functions with good efficiency
# We thus follow https://blog.dask.org/2021/07/02/ragged-output (the dask.array.map_blocks solution has a larger RAM
# usage by having to drop an axis and re-chunk along 1D of the 2D array, so we use the dask.delayed solution instead)


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


@dask.delayed  # type: ignore
def _delayed_nb_valids(arr_chunk: NDArrayNum | NDArrayBool) -> NDArrayNum:
    """Count number of valid values per block."""
    if arr_chunk.dtype == "bool":
        return np.array([np.count_nonzero(arr_chunk)]).reshape((1, 1))
    return np.array([np.count_nonzero(np.isfinite(arr_chunk))]).reshape((1, 1))


@dask.delayed  # type: ignore
def _delayed_subsample_block(
    arr_chunk: NDArrayNum | NDArrayBool, subsample_indices: NDArrayNum
) -> NDArrayNum | NDArrayBool:
    """Subsample the valid values at the corresponding 1D valid indices per block."""

    if arr_chunk.dtype == "bool":
        return arr_chunk[arr_chunk][subsample_indices]
    return arr_chunk[np.isfinite(arr_chunk)][subsample_indices]


@dask.delayed  # type: ignore
def _delayed_subsample_indices_block(
    arr_chunk: NDArrayNum | NDArrayBool, subsample_indices: NDArrayNum, block_id: dict[str, Any]
) -> NDArrayNum:
    """Return 2D indices from the subsampled 1D valid indices per block."""

    if arr_chunk.dtype == "bool":
        ix, iy = np.unravel_index(np.argwhere(arr_chunk.flatten())[subsample_indices], shape=arr_chunk.shape)
    else:
        #  Unravel indices of valid data to the shape of the block
        ix, iy = np.unravel_index(
            np.argwhere(np.isfinite(arr_chunk.flatten()))[subsample_indices], shape=arr_chunk.shape
        )

    # Convert to full-array indexes by adding the row and column starting indexes for this block
    ix += block_id["xstart"]
    iy += block_id["ystart"]

    return np.hstack((ix, iy))


def delayed_subsample(
    darr: da.Array,
    subsample: int | float = 1,
    return_indices: bool = False,
    random_state: int | np.random.Generator | None = None,
    silence_max_subsample: bool = False,
) -> NDArrayNum | tuple[NDArrayNum, NDArrayNum]:
    """
    Subsample a raster at valid values on out-of-memory chunks.

    Optionally, this function can return the 2D indices of the subsample of valid values instead.

    The random subsample is distributed evenly across valid values, no matter which chunk they belong to.
    First, the number of valid values in each chunk are computed out-of-memory. Then, a subsample is defined among
    the total number of valid values, which are then indexed sequentially along the chunk valid values out-of-memory.

    A random state will give a fixed subsample for a delayed array with a fixed chunksize. However, the subsample
    will vary with changing chunksize because the 1D delayed indexing depends on it (indexing per valid value per
    flattened chunk). For this reason, a loaded array will also have a different subsample due to its direct 1D
    indexing (per valid value for the entire flattened array).

    To ensure you reuse a similar subsample of valid values for several arrays, call this function with
    return_indices=True, then sample your arrays out-of-memory with .vindex[indices[0], indices[1]]
    (this assumes that these arrays have valid values at the same locations).

    Only valid values are sampled. If passing a numerical array, then only finite values are considered valid values.
    If passing a boolean array, then only True values are considered valid values.

    :param darr: Input dask array. This can be a boolean or a numerical array.
    :param subsample: Subsample size. If <= 1, will be considered a fraction of valid pixels to extract.
        If > 1 will be considered the number of valid pixels to extract.
    :param return_indices: If set to True, will return the extracted indices only.
    :param random_state: Random state, or seed number to use for random calculations.
    :param silence_max_subsample: Whether to silence the warning for the subsample size being larger than the total
        number of valid points (warns by default).

    :return: Subsample of values from the array (optionally, their indexes).
    """

    # Get random state
    rng = np.random.default_rng(random_state)

    # Compute number of valid points for each block out-of-memory
    blocks = darr.to_delayed().ravel()
    list_delayed_valids = [
        da.from_delayed(_delayed_nb_valids(b), shape=(1, 1), dtype=np.dtype("int32")) for b in blocks
    ]
    nb_valids_per_block = np.concatenate([dask.compute(*list_delayed_valids)])

    # Sum to get total number of valid points
    total_nb_valids = np.sum(nb_valids_per_block)

    # Get subsample size (depending on user input)
    subsample_size = _get_subsample_size_from_user_input(
        subsample=subsample, total_nb_valids=total_nb_valids, silence_max_subsample=silence_max_subsample
    )

    # Get random 1D indexes for the subsample size
    indices_1d = rng.choice(total_nb_valids, subsample_size, replace=False)

    # Sort which indexes belong to which chunk
    ind_per_block = _get_indices_block_per_subsample(
        indices_1d, num_chunks=darr.numblocks, nb_valids_per_block=nb_valids_per_block
    )

    # To just get the subsample without indices
    if not return_indices:
        # Task a delayed subsample to be computed for each block, skipping blocks with no values to sample
        list_subsamples = [
            _delayed_subsample_block(b, ind)
            for i, (b, ind) in enumerate(zip(blocks, ind_per_block))
            if len(ind_per_block[i]) > 0
        ]
        # Cast output to the right expected dtype and length, then compute and concatenate
        list_subsamples_delayed = [
            da.from_delayed(s, shape=(nb_valids_per_block[i]), dtype=darr.dtype) for i, s in enumerate(list_subsamples)
        ]
        subsamples = np.concatenate(dask.compute(*list_subsamples_delayed), axis=0)

        return subsamples

    # To return indices
    else:
        # Get starting 2D index for each chunk of the full array
        # (mirroring what is done in block_id of dask.array.map_blocks)
        # https://github.com/dask/dask/blob/24493f58660cb933855ba7629848881a6e2458c1/dask/array/core.py#L908
        # This list also includes the last index as well (not used here)
        starts = [cached_cumsum(c, initial_zero=True) for c in darr.chunks]
        num_chunks = darr.numblocks
        # Get the starts per 1D block ID by unravelling starting indexes for each block
        indexes_xi, indexes_yi = np.unravel_index(np.arange(len(blocks)), shape=(num_chunks[0], num_chunks[1]))
        block_ids = [
            {"xstart": starts[0][indexes_xi[i]], "ystart": starts[1][indexes_yi[i]]} for i in range(len(blocks))
        ]

        # Task delayed subsample indices to be computed for each block, skipping blocks with no values to sample
        list_subsample_indices = [
            _delayed_subsample_indices_block(b, ind, block_id=block_ids[i])
            for i, (b, ind) in enumerate(zip(blocks, ind_per_block))
            if len(ind_per_block[i]) > 0
        ]
        # Cast output to the right expected dtype and length, then compute and concatenate
        list_subsamples_indices_delayed = [
            da.from_delayed(s, shape=(2, len(ind_per_block[i])), dtype=np.dtype("int32"))
            for i, s in enumerate(list_subsample_indices)
        ]
        indices = np.concatenate(dask.compute(*list_subsamples_indices_delayed), axis=0)

        return indices[:, 0], indices[:, 1]


# 2/ POINT INTERPOLATION ON REGULAR OR EQUAL GRID
# At the date of April 2024:
# This functionality is not covered efficiently by Dask/Xarray, because they need to support rectilinear grids, which
# is difficult when interpolating in the chunked dimensions, and loads nearly all array memory when using .interp().

# Here we harness the fact that rasters are always on regular (or sometimes equal) grids to efficiently map
# the location of the blocks required for interpolation, which requires little memory usage.

# Code structure inspired by https://blog.dask.org/2021/07/02/ragged-output and the "block_id" in map_blocks


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


@dask.delayed  # type: ignore
def _delayed_interp_points_block(
    arr_chunk: NDArrayNum, block_id: dict[str, Any], interp_coords: NDArrayNum
) -> NDArrayNum:
    """
    Interpolate block in 2D out-of-memory for a regular or equal grid.
    """

    # Extract information out of block_id dictionary
    xs, ys, xres, yres = (block_id["xstart"], block_id["ystart"], block_id["xres"], block_id["yres"])

    # Reconstruct the coordinates from xi/yi/xres/yres (as it has to be a regular grid)
    x_coords = np.arange(xs, xs + xres * arr_chunk.shape[0], xres)
    y_coords = np.arange(ys, ys + yres * arr_chunk.shape[1], yres)

    # TODO: Use scipy.map_coordinates for an equal grid as in Raster.interp_points?

    # Interpolate to points
    interp_chunk = interpn(points=(x_coords, y_coords), values=arr_chunk, xi=(interp_coords[0, :], interp_coords[1, :]))

    # And return the interpolated array
    return interp_chunk


def delayed_interp_points(
    darr: da.Array,
    points: tuple[list[float], list[float]],
    resolution: tuple[float, float],
    method: Literal["nearest", "linear", "cubic", "quintic"] = "linear",
) -> NDArrayNum:
    """
    Interpolate raster at point coordinates on out-of-memory chunks.

    This function harnesses the fact that a raster is defined on a regular (or equal) grid, and it is therefore
    faster than Xarray.interpn (especially for small sample sizes) and uses only a fraction of the memory usage.

    :param darr: Input dask array.
    :param points: Point(s) at which to interpolate raster value. If points fall outside of image, value
            returned is nan. Shape should be (N,2).
    :param resolution: Resolution of the raster (xres, yres).
    :param method: Interpolation method, one of 'nearest', 'linear', 'cubic', or 'quintic'. For more information,
            see scipy.ndimage.map_coordinates and scipy.interpolate.interpn. Default is linear.

    :return: Array of raster value(s) for the given points.
    """

    # TODO: Replace by a generic 2D point casting function accepting multiple inputs (living outside this function)
    # Convert input to 2D array
    points_arr = np.vstack((points[0], points[1]))

    # Map depth of overlap required for each interpolation method
    # TODO: Double-check this window somewhere in SciPy's documentation
    map_depth = {"nearest": 1, "linear": 2, "cubic": 3, "quintic": 5}

    # Expand dask array for overlapping computations
    chunksize = darr.chunksize
    expanded = da.overlap.overlap(darr, depth=map_depth[method], boundary=np.nan)

    # Get starting 2D index for each chunk of the full array
    # (mirroring what is done in block_id of dask.array.map_blocks)
    starts = [cached_cumsum(c, initial_zero=True) for c in darr.chunks]
    num_chunks = expanded.numblocks

    # Get samples indices per blocks
    ind_per_block = _get_interp_indices_per_block(
        points_arr[0, :], points_arr[1, :], starts, num_chunks, chunksize, resolution[0], resolution[1]
    )

    # Create a delayed object for each block, and flatten the blocks into a 1d shape
    blocks = expanded.to_delayed().ravel()

    # Build the block IDs by unravelling starting indexes for each block
    indexes_xi, indexes_yi = np.unravel_index(np.arange(len(blocks)), shape=(num_chunks[0], num_chunks[1]))
    block_ids = [
        {
            "xstart": (starts[0][indexes_xi[i]] - map_depth[method]) * resolution[0],
            "ystart": (starts[1][indexes_yi[i]] - map_depth[method]) * resolution[1],
            "xres": resolution[0],
            "yres": resolution[1],
        }
        for i in range(len(blocks))
    ]

    # Compute values delayed
    list_interp = [
        _delayed_interp_points_block(data_chunk, block_ids[i], points_arr[:, ind_per_block[i]])
        for i, data_chunk in enumerate(blocks)
        if len(ind_per_block[i]) > 0
    ]

    # We define the expected output shape and dtype to simplify things for Dask
    list_interp_delayed = [
        da.from_delayed(p, shape=(1, len(ind_per_block[i])), dtype=darr.dtype) for i, p in enumerate(list_interp)
    ]
    interp_points = np.concatenate(dask.compute(*list_interp_delayed), axis=0)

    # Re-order per-block output points to match their original indices
    indices = np.concatenate(ind_per_block).astype(int)
    argsort = np.argsort(indices)
    interp_points = np.array(interp_points)[argsort]

    return interp_points


# 3/ REPROJECT
# At the date of April 2024: not supported by Rioxarray
# Part of the code (defining a GeoGrid and GeoTiling classes) is inspired by
# https://github.com/opendatacube/odc-geo/pull/88, modified to be concise, stand-alone and rely only on
# Rasterio/GeoPandas
# Could be submitted as a PR to Rioxarray? (but not sure the dependency to GeoPandas would work there)

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


@dask.delayed  # type: ignore
def _delayed_reproject_per_block(
    *src_arrs: tuple[NDArrayNum], block_ids: list[dict[str, int]], combined_meta: dict[str, Any], **kwargs: Any
) -> NDArrayNum:
    """
    Delayed reprojection per destination block (also rebuilds a square array combined from intersecting source blocks).
    """

    # If no source chunk intersects, we return a chunk of destination nodata values
    if len(src_arrs) == 0:
        # We can use float32 to return NaN, will be cast to other floating type later if that's not source array dtype
        dst_arr = np.zeros(combined_meta["dst_shape"], dtype=np.dtype("float32"))
        dst_arr[:] = kwargs["dst_nodata"]
        return dst_arr

    # First, we build an empty array with the combined shape, only with nodata values
    comb_src_arr = np.ones((combined_meta["src_shape"]), dtype=src_arrs[0].dtype)
    comb_src_arr[:] = kwargs["src_nodata"]

    # Then fill it with the source chunks values
    for i, arr in enumerate(src_arrs):
        bid = block_ids[i]
        comb_src_arr[bid["rys"] : bid["rye"], bid["rxs"] : bid["rxe"]] = arr

    # Now, we can simply call Rasterio!

    # We build the combined transform from tuple
    src_transform = rio.transform.Affine(*combined_meta["src_transform"])
    dst_transform = rio.transform.Affine(*combined_meta["dst_transform"])

    # Reproject
    dst_arr = np.zeros(combined_meta["dst_shape"], dtype=comb_src_arr.dtype)

    _ = rio.warp.reproject(
        comb_src_arr,
        dst_arr,
        src_transform=src_transform,
        src_crs=kwargs["src_crs"],
        dst_transform=dst_transform,
        dst_crs=kwargs["dst_crs"],
        resampling=kwargs["resampling"],
        src_nodata=kwargs["src_nodata"],
        dst_nodata=kwargs["dst_nodata"],
        num_threads=1,  # Force the number of threads to 1 to avoid Dask/Rasterio conflicting on multi-threading
    )

    return dst_arr


def delayed_reproject(
    darr: da.Array,
    src_transform: rio.transform.Affine,
    src_crs: rio.crs.CRS,
    dst_transform: rio.transform.Affine,
    dst_shape: tuple[int, int],
    dst_crs: rio.crs.CRS,
    resampling: rio.enums.Resampling,
    src_nodata: int | float | None = None,
    dst_nodata: int | float | None = None,
    dst_chunksizes: tuple[int, int] | None = None,
    **kwargs: Any,
) -> da.Array:
    """
    Reproject georeferenced raster on out-of-memory chunks.

    Each chunk of the destination array is mapped to one or several intersecting chunks of the source array, and
    reprojection is performed using rio.warp.reproject for each mapping.

    Part of the code is inspired by https://github.com/opendatacube/odc-geo/pull/88.

    :param darr: Input dask array for source raster.
    :param src_transform: Geotransform of source raster.
    :param src_crs: Coordinate reference system of source raster.
    :param dst_transform: Geotransform of destination raster.
    :param dst_shape: Shape of destination raster.
    :param dst_crs: Coordinate reference system of destination raster.
    :param resampling: Resampling method.
    :param src_nodata: Nodata value of source raster.
    :param dst_nodata: Nodata value of destination raster.
    :param dst_chunksizes: Chunksizes for destination raster.
    :param kwargs: Other arguments to pass to rio.warp.reproject().

    :return: Dask array of reprojected raster.
    """

    # 1/ Define source and destination chunked georeferenced grid through simple classes storing CRS/transform/shape,
    # which allow to consistently derive shape/transform for each block and their CRS-projected footprints

    # Define georeferenced grids for source/destination array
    src_geogrid = GeoGrid(transform=src_transform, shape=darr.shape, crs=src_crs)
    dst_geogrid = GeoGrid(transform=dst_transform, shape=dst_shape, crs=dst_crs)

    # Add the chunking
    # For source, we can use the .chunks attribute
    src_chunks = darr.chunks
    src_geotiling = ChunkedGeoGrid(grid=src_geogrid, chunks=src_chunks)

    # For destination, we need to create the chunks based on destination chunksizes
    if dst_chunksizes is None:
        dst_chunksizes = darr.chunksize
    dst_chunks = _chunks2d_from_chunksizes_shape(chunksizes=dst_chunksizes, shape=dst_shape)
    dst_geotiling = ChunkedGeoGrid(grid=dst_geogrid, chunks=dst_chunks)

    # 2/ Get footprints of tiles in CRS of destination array, with a buffer of 2 pixels for destination ones to ensure
    # overlap, then map indexes of source blocks that intersect a given destination block
    src_footprints = src_geotiling.get_block_footprints(crs=dst_crs)
    dst_footprints = dst_geotiling.get_block_footprints().buffer(2 * max(dst_geogrid.res))
    dest2source = [np.where(dst.intersects(src_footprints).values)[0] for dst in dst_footprints]

    # 3/ To reconstruct a square source array during chunked reprojection, we need to derive the combined shape and
    # transform of each tuples of source blocks
    src_block_ids = np.array(src_geotiling.get_block_locations())
    meta_params = [
        (
            _combined_blocks_shape_transform(sub_block_ids=src_block_ids[sbid], src_geogrid=src_geogrid)  # type: ignore
            if len(sbid) > 0
            else ({}, [])
        )
        for sbid in dest2source
    ]
    # We also add the output transform/shape for this destination chunk in the combined meta
    # (those are the only two that are chunk-specific)
    dst_block_geogrids = dst_geotiling.get_blocks_as_geogrids()
    for i, (c, _) in enumerate(meta_params):
        c.update({"dst_shape": dst_block_geogrids[i].shape, "dst_transform": tuple(dst_block_geogrids[i].transform)})

    # 4/ Call a delayed function that uses rio.warp to reproject the combined source block(s) to each destination block

    # Add fixed arguments to keywords
    kwargs.update(
        {
            "src_nodata": src_nodata,
            "dst_nodata": dst_nodata,
            "resampling": resampling,
            "src_crs": src_crs,
            "dst_crs": dst_crs,
        }
    )

    # Create a delayed object for each block, and flatten the blocks into a 1d shape
    blocks = darr.to_delayed().ravel()
    # Run the delayed reprojection, looping for each destination block
    list_reproj = [
        _delayed_reproject_per_block(
            *blocks[dest2source[i]], block_ids=meta_params[i][1], combined_meta=meta_params[i][0], **kwargs
        )
        for i in range(len(dest2source))
    ]

    # We define the expected output shape and dtype to simplify things for Dask
    list_reproj_delayed = [
        da.from_delayed(r, shape=dst_block_geogrids[i].shape, dtype=darr.dtype) for i, r in enumerate(list_reproj)
    ]

    # Array comes out as flat blocks x chunksize0 (varying) x chunksize1 (varying), so we can't reshape directly
    # We need to unravel the flattened blocks indices to align X/Y, then concatenate all columns, then rows
    indexes_xi, indexes_yi = np.unravel_index(
        np.arange(len(dest2source)), shape=(len(dst_chunks[0]), len(dst_chunks[1]))
    )

    lists_columns = [
        [l for i, l in enumerate(list_reproj_delayed) if j == indexes_xi[i]] for j in range(len(dst_chunks[0]))
    ]
    concat_columns = [da.concatenate(c, axis=1) for c in lists_columns]
    concat_all = da.concatenate(concat_columns, axis=0)

    return concat_all
